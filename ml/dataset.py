"""
Générateur de dataset synthétique — ATS Intelligent
Auteur : Mourad Amoussa | Lyon Ynov M2 IA & Data | RNCP39586

Génère 500 paires (CV, Offre) synthétiques avec des features réalistes
pour entraîner les modèles de matching.

Compétences couvertes :
  C5.2.1 — Ingénierie des features (feature engineering)
  C5.1.3 — Justification des choix méthodologiques (dataset synthétique documenté)

Usage :
    python -m ml.dataset           # génère et sauvegarde le dataset
    python -m ml.dataset --samples 1000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("ml.dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_LAKE_PROC = Path(__file__).resolve().parent.parent / "data_lake" / "processed"
DATA_LAKE_PROC.mkdir(parents=True, exist_ok=True)

# ── Référentiel compétences (33 skills binaires = features) ──────────────────
ALL_SKILLS = [
    "Python", "SQL", "Machine Learning", "Deep Learning", "NLP",
    "Scikit-learn", "TensorFlow", "PyTorch", "Pandas", "NumPy",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP",
    "Spark", "Hadoop", "FastAPI", "Django", "Git",
    "Power BI", "Tableau", "Streamlit", "MLflow", "Airflow",
    "dbt", "Snowflake", "MongoDB", "Redis", "R",
    "Scala", "Java", "JavaScript",
]

# Profils de CVs typiques (distribution réaliste)
CV_PROFILES = {
    "data_scientist_junior": {
        "skills":   ["Python", "SQL", "Pandas", "NumPy", "Scikit-learn", "Git"],
        "exp_mean": 1.0, "exp_std": 0.8,
    },
    "data_scientist_senior": {
        "skills":   ["Python", "SQL", "Machine Learning", "Deep Learning", "Scikit-learn",
                     "TensorFlow", "Docker", "AWS", "MLflow", "Git", "Pandas", "NumPy"],
        "exp_mean": 6.0, "exp_std": 2.0,
    },
    "data_engineer": {
        "skills":   ["Python", "SQL", "Spark", "Airflow", "Docker", "AWS", "dbt",
                     "Snowflake", "Git", "Pandas"],
        "exp_mean": 4.0, "exp_std": 2.5,
    },
    "ml_engineer": {
        "skills":   ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
                     "Docker", "Kubernetes", "FastAPI", "MLflow", "Git", "AWS"],
        "exp_mean": 3.5, "exp_std": 2.0,
    },
    "data_analyst": {
        "skills":   ["SQL", "Python", "Power BI", "Tableau", "Pandas", "Excel",
                     "Git", "NumPy"],
        "exp_mean": 2.5, "exp_std": 1.5,
    },
    "nlp_specialist": {
        "skills":   ["Python", "NLP", "Deep Learning", "TensorFlow", "PyTorch",
                     "Hugging Face", "Pandas", "Git", "Docker"],
        "exp_mean": 3.0, "exp_std": 1.5,
    },
}

# Profils d'offres typiques
OFFRE_PROFILES = {
    "data_scientist": {
        "skills":     ["Python", "Machine Learning", "SQL", "Scikit-learn", "Pandas", "Git"],
        "exp_min":    2, "exp_max": 5,
        "contrats":   ["CDI", "CDI", "CDI", "CDD"],
        "villes":     ["Paris", "Lyon", "Bordeaux", "Toulouse", "Nantes"],
    },
    "data_engineer": {
        "skills":     ["Python", "SQL", "Spark", "Airflow", "Docker", "dbt", "Cloud"],
        "exp_min":    3, "exp_max": 7,
        "contrats":   ["CDI", "CDI", "Freelance"],
        "villes":     ["Paris", "Lyon", "Bordeaux", "Remote"],
    },
    "ml_engineer": {
        "skills":     ["Python", "Machine Learning", "Deep Learning", "Docker",
                       "Kubernetes", "FastAPI", "MLflow"],
        "exp_min":    3, "exp_max": 8,
        "contrats":   ["CDI", "CDI", "Freelance"],
        "villes":     ["Paris", "Sophia-Antipolis", "Remote"],
    },
    "data_analyst": {
        "skills":     ["SQL", "Python", "Power BI", "Excel", "Tableau"],
        "exp_min":    1, "exp_max": 4,
        "contrats":   ["CDI", "CDD", "Stage"],
        "villes":     ["Paris", "Lyon", "Marseille", "Lille"],
    },
    "alternance_data": {
        "skills":     ["Python", "SQL", "Machine Learning"],
        "exp_min":    0, "exp_max": 1,
        "contrats":   ["Alternance", "Stage"],
        "villes":     ["Paris", "Lyon", "Toulouse", "Bordeaux"],
    },
}


def generate_cv(rng: np.random.Generator, profile_name: str) -> dict:
    """Génère un CV synthétique à partir d'un profil."""
    profile = CV_PROFILES[profile_name]

    # Compétences du profil + quelques aléatoires
    n_extra  = rng.integers(0, 4)
    extra    = list(rng.choice(ALL_SKILLS, size=n_extra, replace=False))
    skills   = list(set(profile["skills"]) | set(extra))

    exp_years = max(0.0, rng.normal(profile["exp_mean"], profile["exp_std"]))

    localisation = rng.choice(["Paris", "Lyon", "Bordeaux", "Toulouse",
                                "Nantes", "Marseille", "Lille", "Remote"])

    return {
        "cv_profile":    profile_name,
        "cv_skills":     skills,
        "cv_nb_skills":  len(skills),
        "cv_experience": round(float(exp_years), 1),
        "cv_localisation": localisation,
    }


def generate_offre(rng: np.random.Generator, profile_name: str) -> dict:
    """Génère une offre d'emploi synthétique à partir d'un profil."""
    profile = OFFRE_PROFILES[profile_name]

    n_skills  = rng.integers(3, len(profile["skills"]) + 1)
    skills    = list(rng.choice(profile["skills"], size=n_skills, replace=False))

    # Ajouter parfois des compétences bonus
    if rng.random() < 0.3:
        bonus = list(rng.choice(ALL_SKILLS, size=2, replace=False))
        skills = list(set(skills) | set(bonus))

    exp_min  = int(profile["exp_min"])
    exp_max  = int(profile["exp_max"])
    contrat  = str(rng.choice(profile["contrats"]))
    ville    = str(rng.choice(profile["villes"]))

    return {
        "offre_profile":     profile_name,
        "offre_skills":      skills,
        "offre_nb_skills":   len(skills),
        "exp_min_requis":    exp_min,
        "exp_max_requis":    exp_max,
        "type_contrat":      contrat,
        "offre_localisation": ville,
    }


def compute_features(cv: dict, offre: dict, rng: np.random.Generator) -> dict:
    """
    Calcule les features de matching entre un CV et une offre.

    Features principales (C5.2.1) :
      - skill_overlap     : taux de recouvrement des compétences (Jaccard)
      - loc_score         : score de localisation (1=même ville, 0.5=remote, 0=autre)
      - exp_match         : expérience dans la fourchette requise
      - cv_nb_skills      : nombre de compétences dans le CV
      - offre_nb_skills   : nombre de compétences requises dans l'offre
      - cv_experience     : années d'expérience du candidat
      - exp_min_requis    : expérience minimale requise
      - [skills binaires] : 1 si le CV possède chaque compétence (33 features)
    """
    cv_set    = set(cv["cv_skills"])
    offre_set = set(offre["offre_skills"])

    # Jaccard similarity sur les compétences
    intersection = cv_set & offre_set
    union        = cv_set | offre_set
    jaccard      = len(intersection) / len(union) if union else 0.0

    # Score de localisation
    cv_loc    = cv["cv_localisation"]
    offre_loc = offre["offre_localisation"]
    if cv_loc == offre_loc:
        loc_score = 1.0
    elif offre_loc == "Remote" or cv_loc == "Remote":
        loc_score = 0.8
    elif cv_loc == "Paris" and offre_loc in ["Paris", "Île-de-France"]:
        loc_score = 0.9
    else:
        loc_score = rng.uniform(0.1, 0.5)

    # Match d'expérience
    cv_exp  = cv["cv_experience"]
    exp_min = offre["exp_min_requis"]
    exp_max = offre["exp_max_requis"]
    if exp_min <= cv_exp <= exp_max:
        exp_match = 1.0
    elif cv_exp < exp_min:
        gap = exp_min - cv_exp
        exp_match = max(0.0, 1.0 - gap * 0.3)   # pénalité progressive
    else:
        # Surqualifié → légère pénalité
        gap = cv_exp - exp_max
        exp_match = max(0.5, 1.0 - gap * 0.1)

    # Features composites (C5.2.1 — feature engineering avancé)
    skill_ratio   = len(intersection) / len(offre_set) if offre_set else 0.0
    global_score  = 0.5 * jaccard + 0.25 * exp_match + 0.15 * loc_score + 0.1 * skill_ratio

    # Features binaires : 1 si le CV possède chaque skill du référentiel
    skill_binary = {
        f"skill_{s.lower().replace(' ', '_').replace('-', '_').replace('/', '_')}":
        int(s in cv_set)
        for s in ALL_SKILLS
    }

    features = {
        # Features numériques core
        "skill_overlap":   round(jaccard, 4),
        "skill_ratio":     round(skill_ratio, 4),
        "loc_score":       round(loc_score, 4),
        "exp_match":       round(exp_match, 4),
        "cv_nb_skills":    cv["cv_nb_skills"],
        "offre_nb_skills": offre["offre_nb_skills"],
        "cv_experience":   round(cv["cv_experience"], 1),
        "exp_min_requis":  offre["exp_min_requis"],
        "global_score":    round(global_score, 4),

        # Métadonnées (non utilisées comme features ML, mais utiles pour l'audit)
        "cv_profile":      cv["cv_profile"],
        "offre_profile":   offre["offre_profile"],
        "type_contrat":    offre["type_contrat"],
        "n_skills_overlap": len(intersection),
    }
    features.update(skill_binary)

    # Label : matching positif si score global > 0.45
    # Ajout de bruit réaliste (un humain ne décide pas parfaitement)
    noise = rng.normal(0, 0.05)
    score_reel = np.clip(global_score + noise, 0, 1)
    features["score_reel"] = round(float(score_reel), 4)
    label = int(score_reel > 0.45)
    features["label"] = label
    features["match"] = label

    return features


def generate_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Génère n_samples paires (CV, Offre) synthétiques.

    La distribution des profils reflète le marché IT français :
      - Data Scientist (~30%)
      - Data Engineer (~20%)
      - Data Analyst (~20%)
      - ML Engineer (~15%)
      - Alternance/Stage (~15%)
    """
    rng = np.random.default_rng(seed)

    cv_profiles    = list(CV_PROFILES.keys())
    offre_profiles = list(OFFRE_PROFILES.keys())

    # Pondérations réalistes
    cv_weights    = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
    offre_weights = [0.30, 0.20, 0.15, 0.20, 0.15]

    rows = []
    for i in range(n_samples):
        cv_profile    = rng.choice(cv_profiles,    p=cv_weights)
        offre_profile = rng.choice(offre_profiles, p=offre_weights)

        cv    = generate_cv(rng, cv_profile)
        offre = generate_offre(rng, offre_profile)
        feat  = compute_features(cv, offre, rng)
        feat["sample_id"] = i
        rows.append(feat)

    df = pd.DataFrame(rows)

    log.info("Dataset généré : %d échantillons", len(df))
    log.info("Distribution des labels : %s", df["label"].value_counts().to_dict())
    log.info("Score réel moyen : %.3f", df["score_reel"].mean())

    # Vérification du déséquilibre de classes
    ratio = df["label"].mean()
    if ratio < 0.3 or ratio > 0.7:
        log.warning(
            "Déséquilibre de classes détecté (%.1f%% positifs). "
            "Envisager SMOTE ou class_weight='balanced'.", ratio * 100
        )

    return df


def save_dataset(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Sauvegarde le dataset en Parquet dans le data lake."""
    if path is None:
        path = DATA_LAKE_PROC / "dataset_matching.parquet"

    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    log.info("Dataset sauvegardé → %s (%.1f Ko, %d lignes, %d colonnes)",
             path, path.stat().st_size / 1024, len(df), len(df.columns))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de dataset synthétique ATS")
    parser.add_argument("--samples", type=int, default=500,
                        help="Nombre de paires CV-Offre à générer")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Graine aléatoire (reproductibilité)")
    args = parser.parse_args()

    df   = generate_dataset(n_samples=args.samples, seed=args.seed)
    path = save_dataset(df)

    print(f"\n✅ Dataset prêt : {path}")
    print(f"   Colonnes     : {len(df.columns)}")
    print(f"   Lignes       : {len(df)}")
    print(f"   Label=1 (match) : {df['label'].sum()} ({df['label'].mean():.1%})")
    print(f"   Label=0 (non-match) : {(~df['label'].astype(bool)).sum()}")
    print("\nAperçu des features :")
    print(df[["skill_overlap", "exp_match", "loc_score", "cv_experience",
              "offre_nb_skills", "score_reel", "label"]].describe().round(3).to_string())
