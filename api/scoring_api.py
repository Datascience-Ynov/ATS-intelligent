"""
API REST de scoring — ATS Intelligent
Auteur : Mourad Amoussa | Lyon Ynov M2 IA & Data | RNCP39586

Compétences couvertes :
  C5.3.2 — Déploiement du modèle via API REST (FastAPI)
  C5.3.3 — Monitoring de la performance en production (drift alert, logs)
  C5.3.4 — Maintenance et retraining (endpoint de déclenchement)

Usage :
    uvicorn api.scoring_api:app --reload --port 8000

Endpoints :
    POST /score                  → score de matching CV ↔ Offre
    POST /score/batch            → scoring en lot
    GET  /health                 → état du modèle
    GET  /monitoring             → métriques de production
    POST /admin/retrain          → déclenche le re-entraînement
    GET  /docs                   → documentation interactive Swagger
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, Field, field_validator

log = logging.getLogger("api.scoring")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Chemins ──────────────────────────────────────────────────────────────────
MODEL_DIR     = Path(__file__).resolve().parent.parent / "data_lake" / "models" / "latest"
LOGS_DIR      = Path(__file__).resolve().parent.parent / "data_lake" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"


# ══════════════════════════════════════════════════════════════════════════════
# Chargement du modèle au démarrage de l'API
# ══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """Gère le chargement et l'accès au modèle ML."""

    def __init__(self) -> None:
        self.model           = None
        self.feature_names:  list[str] = []
        self.metadata:       dict      = {}
        self.loaded_at:      str       = ""
        self.is_ready:       bool      = False
        self._load()

    def _load(self) -> None:
        model_path    = MODEL_DIR / "model.joblib"
        features_path = MODEL_DIR / "feature_names.json"
        meta_path     = MODEL_DIR / "metadata.json"

        if not model_path.exists():
            log.warning(
                "Modèle non trouvé dans %s. "
                "Lancez d'abord : python -m ml.train", MODEL_DIR
            )
            self.is_ready = False
            return

        try:
            self.model         = joblib.load(model_path)
            self.feature_names = json.loads(features_path.read_text())
            self.metadata      = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            self.loaded_at     = datetime.now(timezone.utc).isoformat()
            self.is_ready      = True
            log.info("Modèle chargé : %s - %d features",
                     self.metadata.get("modele_nom", "?"), len(self.feature_names))
        except Exception as e:
            log.error("Erreur chargement modèle : %s", e)
            self.is_ready = False

    def reload(self) -> None:
        """Recharge le modèle (après re-entraînement)."""
        log.info("Rechargement du modèle...")
        self._load()


registry = ModelRegistry()

# ── Compteurs de monitoring (C5.3.3) ────────────────────────────────────────
_stats: dict[str, Any] = {
    "total_predictions": 0,
    "total_errors":      0,
    "scores":            [],    # derniers 1000 scores
    "start_time":        time.time(),
}


# ══════════════════════════════════════════════════════════════════════════════
# Schémas Pydantic — Validation des données d'entrée
# ══════════════════════════════════════════════════════════════════════════════

ALL_SKILLS = [
    "Python", "SQL", "Machine Learning", "Deep Learning", "NLP",
    "Scikit-learn", "TensorFlow", "PyTorch", "Pandas", "NumPy",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP",
    "Spark", "Hadoop", "FastAPI", "Django", "Git",
    "Power BI", "Tableau", "Streamlit", "MLflow", "Airflow",
    "dbt", "Snowflake", "MongoDB", "Redis", "R",
    "Scala", "Java", "JavaScript",
]


class CVInput(BaseModel):
    """Données du CV candidat."""
    competences:   list[str] = Field(
        ..., validation_alias=AliasChoices("competences", "competences_tech"),
        description="Liste des compétences du CV",
    )
    experience:    float     = Field(
        0.0, ge=0, le=50, validation_alias=AliasChoices("experience", "experience_annees"),
        description="Années d'expérience",
    )
    localisation:  str       = Field("Non précisé", description="Ville du candidat")
    nb_skills:     int | None = Field(None, description="Nombre de compétences (auto-calculé si absent)")

    @field_validator("competences")
    @classmethod
    def validate_competences(cls, v):
        return [str(c)[:100] for c in v[:50]]  # max 50, 100 chars chacune


class OffreInput(BaseModel):
    """Données de l'offre d'emploi."""
    competences_requises: list[str] = Field(
        ..., validation_alias=AliasChoices("competences_requises", "competences_tech"),
        description="Compétences requises dans l'offre",
    )
    exp_min:              int       = Field(
        0, ge=0, le=30, validation_alias=AliasChoices("exp_min", "experience_requise"),
        description="Expérience minimale requise",
    )
    exp_max:              int       = Field(10,  ge=0, le=50, description="Expérience maximale requise")
    localisation:         str       = Field("Non précisé", description="Localisation de l'offre")
    type_contrat:         str       = Field("CDI", description="Type de contrat")

    @field_validator("competences_requises")
    @classmethod
    def validate_offre_competences(cls, v):
        return [str(c)[:100] for c in v[:30]]


class CVOffreInput(BaseModel):
    """Requête de scoring : paire CV + Offre."""
    cv:     CVInput
    offre:  OffreInput
    cv_id:  int | None = Field(None, description="ID du CV en base (optionnel)")
    offre_id: int | None = Field(None, description="ID de l'offre en base (optionnel)")


class BatchInput(BaseModel):
    """Scoring en lot : une liste de paires CV-Offre."""
    pairs: list[CVOffreInput] = Field(..., max_length=50, description="Max 50 paires")


class ScoreResponse(BaseModel):
    """Réponse du scoring."""
    score:           float
    score_percent:   int
    confidence:      float
    interpretation:  str
    details:         dict
    model_version:   str
    timestamp:       str


# ══════════════════════════════════════════════════════════════════════════════
# Logique de scoring
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(cv: CVInput, offre: OffreInput) -> pd.DataFrame:
    """
    Construit le vecteur de features depuis les données CV + Offre.
    Doit correspondre exactement aux features utilisées à l'entraînement.
    """
    cv_set    = set(cv.competences)
    offre_set = set(offre.competences_requises)

    # Jaccard
    intersection = cv_set & offre_set
    union        = cv_set | offre_set
    jaccard      = len(intersection) / len(union) if union else 0.0
    skill_ratio  = len(intersection) / len(offre_set) if offre_set else 0.0

    # Localisation
    cv_loc    = cv.localisation
    offre_loc = offre.localisation
    if cv_loc == offre_loc:
        loc_score = 1.0
    elif "remote" in offre_loc.lower() or "remote" in cv_loc.lower():
        loc_score = 0.8
    else:
        loc_score = 0.3

    # Expérience
    if offre.exp_min <= cv.experience <= offre.exp_max:
        exp_match = 1.0
    elif cv.experience < offre.exp_min:
        gap = offre.exp_min - cv.experience
        exp_match = max(0.0, 1.0 - gap * 0.3)
    else:
        gap = cv.experience - offre.exp_max
        exp_match = max(0.5, 1.0 - gap * 0.1)

    cv_nb_skills    = cv.nb_skills if cv.nb_skills else len(cv.competences)
    offre_nb_skills = len(offre.competences_requises)

    # Features composites
    exp_gap          = cv.experience - offre.exp_min
    skills_density   = cv_nb_skills / (offre_nb_skills + 1)
    overlap_x_loc    = jaccard * loc_score
    match_difficulty = offre.exp_min / (cv_nb_skills + 1)
    exp_match_sq     = exp_match ** 2
    skills_sq        = jaccard ** 2

    # Features binaires
    skill_binary = {
        f"skill_{s.lower().replace(' ', '_').replace('-', '_').replace('/', '_')}":
        int(s in cv_set)
        for s in ALL_SKILLS
    }

    row = {
        "skill_overlap":   jaccard,
        "skill_ratio":     skill_ratio,
        "loc_score":       loc_score,
        "exp_match":       exp_match,
        "cv_nb_skills":    cv_nb_skills,
        "offre_nb_skills": offre_nb_skills,
        "cv_experience":   cv.experience,
        "exp_min_requis":  offre.exp_min,
        "global_score":    0.5 * jaccard + 0.25 * exp_match + 0.15 * loc_score + 0.1 * skill_ratio,
        "exp_gap":         exp_gap,
        "skills_density":  skills_density,
        "overlap_x_loc":   overlap_x_loc,
        "match_difficulty": match_difficulty,
        "exp_match_sq":    exp_match_sq,
        "skills_sq":       skills_sq,
    }
    row.update(skill_binary)

    # Aligner avec les features du modèle
    feature_names = registry.feature_names
    df = pd.DataFrame([row])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names].fillna(0)

    return df


def interpret_score(score: float, skills_overlap: int, total_offre_skills: int) -> str:
    """Génère une interprétation textuelle du score."""
    pct = int(score * 100)
    if pct >= 75:
        base = "Excellente compatibilité"
    elif pct >= 55:
        base = "Bonne compatibilité"
    elif pct >= 35:
        base = "Compatibilité partielle"
    else:
        base = "Compatibilité faible"

    if total_offre_skills > 0:
        skill_pct = int(skills_overlap / total_offre_skills * 100)
        return f"{base} ({pct}%) — {skill_pct}% des compétences requises couvertes."
    return f"{base} ({pct}%)."


def log_prediction(cv_id: int | None, offre_id: int | None,
                   score: float, confidence: float) -> None:
    """Log chaque prédiction pour le monitoring (C5.3.3)."""
    entry = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "cv_id":      cv_id,
        "offre_id":   offre_id,
        "score":      round(score, 4),
        "confidence": round(confidence, 4),
    }
    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # Mettre à jour les stats en mémoire
    _stats["total_predictions"] += 1
    _stats["scores"].append(score)
    if len(_stats["scores"]) > 1000:  # garder seulement les 1000 derniers
        _stats["scores"] = _stats["scores"][-1000:]


# ══════════════════════════════════════════════════════════════════════════════
# Application FastAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="ATS Scoring API",
    description=(
        "API de matching CV ↔ Offre d'emploi par ML.\n\n"
        "Projet RNCP39586 — Mourad Amoussa — Lyon Ynov M2 IA & Data\n\n"
        "**Compétences RNCP** : C5.3.2 (déploiement), C5.3.3 (monitoring), C5.3.4 (maintenance)"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # En production : restreindre à votre domaine
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "api":     "ATS Scoring API",
        "version": "1.0.0",
        "status":  "ready" if registry.is_ready else "model_not_loaded",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Monitoring"])
def health():
    """Vérifie que l'API et le modèle sont opérationnels (C5.3.3)."""
    if not registry.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status":  "unhealthy",
                "reason":  "Modèle non chargé. Lancez d'abord : python -m ml.train",
                "model_dir": str(MODEL_DIR),
            },
        )
    return {
        "status":        "healthy",
        "model":         registry.metadata.get("modele_nom", "?"),
        "model_version": registry.metadata.get("version", "?"),
        "n_features":    len(registry.feature_names),
        "loaded_at":     registry.loaded_at,
        "performances":  registry.metadata.get("performances", {}),
    }


@app.get("/monitoring", tags=["Monitoring"])
def monitoring():
    """
    Retourne les métriques de production en temps réel (C5.3.3).

    Inclut :
      - Volume de prédictions
      - Distribution des scores (mean, std, p25, p75)
      - Alerte de drift si score moyen anormalement bas
      - Uptime de l'API
    """
    scores = _stats["scores"]
    uptime = round(time.time() - _stats["start_time"], 0)

    stats_scores: dict = {}
    drift_alert = False

    if scores:
        arr = np.array(scores)
        stats_scores = {
            "mean":   round(float(arr.mean()), 4),
            "std":    round(float(arr.std()),  4),
            "min":    round(float(arr.min()),  4),
            "max":    round(float(arr.max()),  4),
            "p25":    round(float(np.percentile(arr, 25)), 4),
            "p75":    round(float(np.percentile(arr, 75)), 4),
            "n_last": len(scores),
        }
        # Alerte drift : si score moyen < 30% (données entrantes anormales)
        drift_alert = float(arr.mean()) < 0.30

    return {
        "api_uptime_s":       uptime,
        "total_predictions":  _stats["total_predictions"],
        "total_errors":       _stats["total_errors"],
        "score_distribution": stats_scores,
        "drift_alert":        drift_alert,
        "drift_alert_message": (
            "⚠️ Score moyen anormalement bas — possible data drift. "
            "Envisager un re-entraînement via POST /admin/retrain"
        ) if drift_alert else None,
        "model_loaded":       registry.is_ready,
        "model_version":      registry.metadata.get("version", "?"),
        "log_file":           str(PREDICTIONS_LOG),
    }


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_cv_offre(payload: CVOffreInput):
    """
    Calcule le score de matching entre un CV et une offre d'emploi.

    Retourne :
      - score (0-1) et score_percent (0-100)
      - confidence : probabilité retournée par le modèle
      - interpretation : texte explicatif
      - details : features principales utilisées
    """
    if not registry.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible. Lancez d'abord : python -m ml.train",
        )

    t0 = time.time()
    try:
        # Construction du vecteur de features
        X = build_feature_vector(payload.cv, payload.offre)

        # Prédiction
        pred_proba = registry.model.predict_proba(X)[0]
        confidence = float(pred_proba[1])   # probabilité de la classe 1 (match)
        score      = confidence

        # Compétences communes pour l'interprétation
        cv_set    = set(payload.cv.competences)
        offre_set = set(payload.offre.competences_requises)
        overlap   = cv_set & offre_set
        manquantes = offre_set - cv_set

        interpretation = interpret_score(score, len(overlap), len(offre_set))
        latency = round((time.time() - t0) * 1000, 1)

        # Log de la prédiction (C5.3.3)
        log_prediction(payload.cv_id, payload.offre_id, score, confidence)

        return ScoreResponse(
            score=round(score, 4),
            score_percent=int(score * 100),
            confidence=round(confidence, 4),
            interpretation=interpretation,
            details={
                "competences_communes":     sorted(overlap),
                "competences_manquantes":   sorted(manquantes),
                "n_competences_cv":         len(cv_set),
                "n_competences_offre":      len(offre_set),
                "n_competences_communes":   len(overlap),
                "taux_couverture_pct":      int(len(overlap) / len(offre_set) * 100) if offre_set else 0,
                "experience_candidat":      payload.cv.experience,
                "experience_requise":       f"{payload.offre.exp_min}-{payload.offre.exp_max} ans",
                "latency_ms":               latency,
            },
            model_version=registry.metadata.get("version", "v1"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        _stats["total_errors"] += 1
        log.error("Erreur scoring : %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du calcul du score : {str(e)}",
        )


@app.post("/score/batch", tags=["Scoring"])
def score_batch(payload: BatchInput):
    """
    Scoring en lot : évalue une liste de paires CV-Offre.
    Maximum 50 paires par requête.
    """
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    results = []
    for pair in payload.pairs:
        try:
            response = score_cv_offre(pair)
            results.append({"status": "ok", "result": response.model_dump()})
        except Exception as e:
            results.append({"status": "error", "error": str(e)})

    return {
        "n_pairs":    len(payload.pairs),
        "n_success":  sum(1 for r in results if r["status"] == "ok"),
        "n_errors":   sum(1 for r in results if r["status"] == "error"),
        "results":    results,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }


@app.post("/admin/retrain", tags=["Administration"])
def trigger_retrain(confirm: bool = False):
    """
    Déclenche un re-entraînement du modèle (C5.3.4).

    Paramètres :
      - confirm=true : requis pour lancer le retraining (sécurité)

    Le re-entraînement :
      1. Génère un nouveau dataset
      2. Lance le pipeline ML complet
      3. Recharge le modèle dans l'API
    """
    if not confirm:
        return {
            "message": "Ajoutez ?confirm=true pour lancer le re-entraînement",
            "warning": "Le re-entraînement peut prendre quelques minutes",
        }

    log.info("Re-entraînement déclenché via API")
    try:
        from ml.train import run_full_pipeline
        results = run_full_pipeline(skip_grid_search=False, version="v2")
        registry.reload()
        log.info("Re-entraînement terminé. Modèle rechargé.")
        return {
            "status":   "success",
            "message":  "Re-entraînement terminé, modèle rechargé",
            "results":  results,
        }
    except Exception as e:
        log.error("Erreur re-entraînement : %s", e)
        raise HTTPException(status_code=500, detail=f"Re-entraînement échoué : {str(e)}")


@app.get("/admin/features", tags=["Administration"])
def get_features():
    """Retourne la liste des features utilisées par le modèle."""
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    return {
        "feature_names":  registry.feature_names,
        "n_features":     len(registry.feature_names),
        "model_metadata": registry.metadata,
    }


# ── Point d'entrée direct ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8000"))
    log.info("Démarrage de l'API sur http://localhost:%d", port)
    log.info("Documentation : http://localhost:%d/docs", port)
    uvicorn.run(
        "api.scoring_api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
