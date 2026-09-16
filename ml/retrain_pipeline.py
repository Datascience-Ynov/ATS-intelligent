"""
Pipeline MLOps — Détection de drift & Re-entraînement automatique
Auteur : Mourad Amoussa | Lyon Ynov M2 IA & Data | RNCP39586

Compétences couvertes :
  C5.3.3 — Monitoring de la performance en production
  C5.3.4 — Maintenance : détection de drift, retraining automatisé,
            historisation des prédictions

Usage :
    python -m ml.retrain_pipeline                    # check drift + retrain si nécessaire
    python -m ml.retrain_pipeline --force            # forcer le retrain sans check
    python -m ml.retrain_pipeline --check-only       # juste vérifier le drift
    python -m ml.retrain_pipeline --report           # afficher le rapport de drift
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("ml.retrain")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent.parent
DATA_LAKE_PROC    = BASE_DIR / "data_lake" / "processed"
DATA_LAKE_MODELS  = BASE_DIR / "data_lake" / "models"
LOGS_DIR          = BASE_DIR / "data_lake" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_LOG   = LOGS_DIR / "predictions.jsonl"
DRIFT_REPORT_PATH = LOGS_DIR / "drift_report.json"
RETRAIN_LOG_PATH  = LOGS_DIR / "retrain_history.jsonl"

# ── Seuils de drift (ajustables selon le domaine) ────────────────────────────
DRIFT_THRESHOLDS = {
    "score_mean_min":    0.30,   # Score moyen < 30% → data drift
    "score_mean_max":    0.90,   # Score moyen > 90% → possiblement biais
    "score_std_max":     0.35,   # Variance excessive
    "n_min_predictions": 50,     # Minimum de prédictions pour déclencher l'analyse
    "low_score_ratio":   0.70,   # Plus de 70% de scores < 0.3 → problème
}


# ══════════════════════════════════════════════════════════════════════════════
# C5.3.3 — Monitoring : Historisation des prédictions
# ══════════════════════════════════════════════════════════════════════════════

def log_prediction(cv_id: int | None, offre_id: int | None,
                   score: float, confidence: float,
                   features: dict | None = None) -> None:
    """
    Enregistre une prédiction dans le log JSONL (C5.3.4).

    Le format JSONL (JSON Lines) permet :
      - Lecture ligne par ligne sans charger tout le fichier
      - Appends atomiques (pas de corruption)
      - Compatible avec pandas, Spark, Dask

    Max 10 000 prédictions gardées (rotation automatique).
    """
    entry = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "cv_id":      cv_id,
        "offre_id":   offre_id,
        "score":      round(float(score), 4),
        "confidence": round(float(confidence), 4),
    }
    if features:
        # Ne garder que les features numériques principales
        entry["key_features"] = {
            k: round(float(v), 4)
            for k, v in features.items()
            if k in ["skill_overlap", "exp_match", "loc_score", "cv_nb_skills"]
        }

    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # Rotation : garder max 10 000 lignes
    _rotate_log_if_needed(PREDICTIONS_LOG, max_lines=10_000)


def _rotate_log_if_needed(path: Path, max_lines: int) -> None:
    """Coupe le fichier de log à max_lines en conservant les plus récentes."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_recent_predictions(n: int = 500) -> pd.DataFrame | None:
    """
    Charge les N prédictions les plus récentes pour l'analyse de drift.

    Returns:
        DataFrame ou None si pas assez de données
    """
    if not PREDICTIONS_LOG.exists():
        log.info("Aucun log de prédictions trouvé : %s", PREDICTIONS_LOG)
        return None

    lines = PREDICTIONS_LOG.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]

    if len(lines) < DRIFT_THRESHOLDS["n_min_predictions"]:
        log.info("Pas assez de prédictions (%d/%d) pour l'analyse de drift",
                 len(lines), DRIFT_THRESHOLDS["n_min_predictions"])
        return None

    # Charger les N plus récentes
    recent = lines[-n:]
    rows   = []
    for line in recent:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["score"]     = pd.to_numeric(df["score"], errors="coerce")
    return df.dropna(subset=["score"])


# ══════════════════════════════════════════════════════════════════════════════
# C5.3.3 — Détection de data drift
# ══════════════════════════════════════════════════════════════════════════════

def detect_data_drift(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyse la distribution des prédictions récentes pour détecter un drift.

    Méthodes utilisées :
      1. Statistiques descriptives (mean, std, percentiles)
      2. Ratio de scores extrêmes (< 0.3 ou > 0.9)
      3. Comparaison avec les stats de référence du modèle

    Returns:
        Rapport de drift avec flag drift_detected et détails
    """
    scores = df["score"].values
    n      = len(scores)

    mean_score   = float(np.mean(scores))
    std_score    = float(np.std(scores))
    low_ratio    = float(np.mean(scores < 0.3))
    high_ratio   = float(np.mean(scores > 0.9))

    # Charge les stats de référence du modèle (si disponibles)
    meta_path = DATA_LAKE_MODELS / "latest" / "metadata.json"
    ref_auc   = None
    if meta_path.exists():
        meta    = json.loads(meta_path.read_text())
        ref_auc = meta.get("performances", {}).get("test_roc_auc")

    # Évaluation du drift
    drift_reasons = []

    if mean_score < DRIFT_THRESHOLDS["score_mean_min"]:
        drift_reasons.append(
            f"Score moyen trop bas ({mean_score:.3f} < {DRIFT_THRESHOLDS['score_mean_min']})"
        )
    if mean_score > DRIFT_THRESHOLDS["score_mean_max"]:
        drift_reasons.append(
            f"Score moyen suspect ({mean_score:.3f} > {DRIFT_THRESHOLDS['score_mean_max']})"
        )
    if std_score > DRIFT_THRESHOLDS["score_std_max"]:
        drift_reasons.append(
            f"Variance excessive (std={std_score:.3f} > {DRIFT_THRESHOLDS['score_std_max']})"
        )
    if low_ratio > DRIFT_THRESHOLDS["low_score_ratio"]:
        drift_reasons.append(
            f"{low_ratio:.0%} des scores sont < 30% (données entrantes peut-être hors distribution)"
        )

    drift_detected = len(drift_reasons) > 0

    # Tendance temporelle (si assez de données)
    trend = None
    if n >= 100 and "timestamp" in df.columns:
        df_sorted = df.sort_values("timestamp")
        first_half = df_sorted.head(n // 2)["score"].mean()
        last_half  = df_sorted.tail(n // 2)["score"].mean()
        delta      = last_half - first_half
        if abs(delta) > 0.05:
            trend = f"Score {'en hausse' if delta > 0 else 'en baisse'} ({delta:+.3f} entre 1ère et 2ème moitié)"
            if delta < -0.10:
                drift_detected = True
                drift_reasons.append(f"Tendance négative significative : {trend}")

    report = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "n_predictions":   n,
        "drift_detected":  drift_detected,
        "drift_reasons":   drift_reasons,
        "stats": {
            "mean":      round(mean_score, 4),
            "std":       round(std_score,  4),
            "min":       round(float(np.min(scores)), 4),
            "max":       round(float(np.max(scores)), 4),
            "p25":       round(float(np.percentile(scores, 25)), 4),
            "p75":       round(float(np.percentile(scores, 75)), 4),
            "low_ratio": round(low_ratio, 4),
            "high_ratio": round(high_ratio, 4),
        },
        "reference_auc": ref_auc,
        "trend":         trend,
        "thresholds":    DRIFT_THRESHOLDS,
    }

    # Sauvegarde du rapport
    DRIFT_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    if drift_detected:
        log.warning("⚠️  DRIFT DÉTECTÉ !")
        for reason in drift_reasons:
            log.warning("    → %s", reason)
    else:
        log.info("✅ Pas de drift détecté (mean=%.3f, std=%.3f, n=%d)",
                 mean_score, std_score, n)

    return report


# ══════════════════════════════════════════════════════════════════════════════
# C5.3.4 — Re-entraînement automatique
# ══════════════════════════════════════════════════════════════════════════════

def auto_retrain_if_needed(force: bool = False) -> dict[str, Any]:
    """
    Vérifie le drift et re-entraîne le modèle si nécessaire.

    Workflow :
      1. Charger les prédictions récentes
      2. Analyser le drift
      3. Si drift détecté (ou force=True) → lancer run_full_pipeline()
      4. Logger le résultat dans retrain_history.jsonl

    Returns:
        Rapport complet du processus
    """
    t0 = time.time()
    log.info("=" * 55)
    log.info("Vérification drift + re-entraînement conditionnel")
    log.info("=" * 55)

    result: dict[str, Any] = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "drift_checked":  False,
        "drift_detected": False,
        "retrained":      False,
        "error":          None,
    }

    # ── 1. Charger les prédictions récentes ──────────────────────────────────
    df_preds = load_recent_predictions(n=500)
    if df_preds is None and not force:
        log.info("Pas assez de prédictions — re-entraînement ignoré")
        result["reason"] = "Pas assez de prédictions en production"
        return result

    # ── 2. Détection du drift ─────────────────────────────────────────────────
    if df_preds is not None:
        drift_report       = detect_data_drift(df_preds)
        result["drift_checked"]  = True
        result["drift_detected"] = drift_report["drift_detected"]
        result["drift_report"]   = drift_report
    else:
        result["drift_report"] = {"note": "Aucune donnée de production disponible"}

    # ── 3. Décision de re-entraînement ───────────────────────────────────────
    should_retrain = force or result["drift_detected"]

    if not should_retrain:
        log.info("Re-entraînement non nécessaire — modèle en bonne santé")
        result["reason"] = "Pas de drift détecté"
        _log_retrain_history(result)
        return result

    reason = "Forcé par l'utilisateur" if force else "Drift détecté"
    log.info("Re-entraînement lancé — raison : %s", reason)
    result["retrain_reason"] = reason

    # ── 4. Re-entraînement ────────────────────────────────────────────────────
    try:
        from ml.train import run_full_pipeline

        # Générer une nouvelle version
        existing_versions = [
            d.name for d in DATA_LAKE_MODELS.iterdir()
            if d.is_dir() and d.name.startswith("v") and d.name != "latest"
        ]
        version_nums = []
        for v in existing_versions:
            try:
                version_nums.append(int(v[1:]))
            except ValueError:
                pass
        next_version = f"v{max(version_nums, default=0) + 1}"
        log.info("Nouvelle version : %s", next_version)

        pipeline_results = run_full_pipeline(
            skip_grid_search=False,
            version=next_version,
        )

        result["retrained"]        = True
        result["new_version"]      = next_version
        result["pipeline_results"] = pipeline_results
        log.info("✅ Re-entraînement terminé — version %s", next_version)

    except Exception as e:
        log.error("❌ Erreur lors du re-entraînement : %s", e)
        result["error"] = str(e)
        result["retrained"] = False

    result["duration_s"] = round(time.time() - t0, 1)
    _log_retrain_history(result)

    return result


def _log_retrain_history(entry: dict) -> None:
    """Appende une entrée dans l'historique des re-entraînements."""
    serializable = {
        k: v for k, v in entry.items()
        if k not in ("drift_report",)  # trop verbeux pour l'historique
    }
    with open(RETRAIN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def get_retrain_history(n: int = 10) -> list[dict]:
    """Retourne les N derniers re-entraînements."""
    if not RETRAIN_LOG_PATH.exists():
        return []
    lines = RETRAIN_LOG_PATH.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()][-n:]
    return [json.loads(l) for l in lines]


def simulate_production_predictions(n: int = 100, seed: int = 99) -> None:
    """
    Simule des prédictions de production pour tester le monitoring.

    Utile pour la démo au jury : génère du trafic réaliste.
    """
    rng = np.random.default_rng(seed)
    log.info("Simulation de %d prédictions de production...", n)

    for i in range(n):
        score      = float(np.clip(rng.normal(0.55, 0.20), 0, 1))
        confidence = float(np.clip(rng.normal(0.60, 0.15), 0, 1))
        log_prediction(
            cv_id=int(rng.integers(1, 50)),
            offre_id=int(rng.integers(1, 200)),
            score=score,
            confidence=confidence,
        )

    log.info("Simulation terminée — %d prédictions loguées dans %s", n, PREDICTIONS_LOG)


# ── Point d'entrée CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline — Détection drift & Re-entraînement (RNCP39586)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--force",       action="store_true",
                       help="Forcer le re-entraînement sans vérifier le drift")
    group.add_argument("--check-only",  action="store_true",
                       help="Analyser le drift sans re-entraîner")
    group.add_argument("--report",      action="store_true",
                       help="Afficher le dernier rapport de drift")
    group.add_argument("--simulate",    type=int, metavar="N",
                       help="Simuler N prédictions de production")
    group.add_argument("--history",     action="store_true",
                       help="Afficher l'historique des re-entraînements")

    args = parser.parse_args()

    if args.simulate:
        simulate_production_predictions(n=args.simulate)

    elif args.report:
        if DRIFT_REPORT_PATH.exists():
            report = json.loads(DRIFT_REPORT_PATH.read_text())
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print("Aucun rapport de drift disponible. Lancez d'abord sans --report.")

    elif args.history:
        history = get_retrain_history(n=10)
        if not history:
            print("Aucun historique de re-entraînement trouvé.")
        for entry in history:
            print(json.dumps(entry, indent=2, ensure_ascii=False))
            print("---")

    elif args.check_only:
        df = load_recent_predictions()
        if df is not None:
            report = detect_data_drift(df)
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print("Pas assez de prédictions pour l'analyse.")

    else:
        # Comportement par défaut : check + retrain si nécessaire
        result = auto_retrain_if_needed(force=args.force)
        print("\n📊 Résultat :")
        print(f"  Drift détecté  : {result.get('drift_detected', 'N/A')}")
        print(f"  Re-entraîné    : {result.get('retrained', False)}")
        if result.get("new_version"):
            print(f"  Nouvelle version : {result['new_version']}")
        if result.get("error"):
            print(f"  ❌ Erreur : {result['error']}")
        print(f"  Durée : {result.get('duration_s', 0):.1f} s")
