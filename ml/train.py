"""
Pipeline ML complet — ATS Intelligent
Auteur : Mourad Amoussa | Lyon Ynov M2 IA & Data | RNCP39586

Couvre TOUTES les compétences éliminatoires du Bloc 5 :
  C5.2.1 — Ingénierie des features (feature engineering documenté)
  C5.2.2 — Sélection de features (3 méthodes comparées → consensus)
  C5.2.3 — Entraînement + évaluation de 3 modèles avec cross-validation
  C5.2.4 — Optimisation des hyperparamètres (GridSearchCV)
  C5.3.1 — Sauvegarde du modèle (joblib + metadata.json)

Usage :
    python -m ml.train                    # pipeline complet
    python -m ml.train --skip-grid-search # rapide (sans GridSearchCV)
    python -m ml.train --samples 1000     # avec plus de données
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import (SelectKBest, f_classif,
                                       mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn


warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger("ml.train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_LAKE_PROC    = Path(__file__).resolve().parent.parent / "data_lake" / "processed"
DATA_LAKE_MODELS  = Path(__file__).resolve().parent.parent / "data_lake" / "models"

DATA_LAKE_PROC.mkdir(parents=True, exist_ok=True)
DATA_LAKE_MODELS.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# C5.2.1 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(dataset_path: Path | None = None) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Charge le dataset et réalise l'ingénierie des features.

    Features construites :
      CORE (numériques) :
        - skill_overlap   : recouvrement Jaccard des compétences
        - skill_ratio     : proportion des compétences requises couvertes
        - loc_score       : score de localisation géographique
        - exp_match       : adéquation de l'expérience à la fourchette requise
        - cv_nb_skills    : richesse du profil candidat
        - offre_nb_skills : complexité de l'offre
        - cv_experience   : années d'expérience
        - exp_min_requis  : seuil d'expérience demandé

      COMPOSITES (ingénierie non-linéaire) :
        - exp_gap          : cv_experience - exp_min_requis (positif = surqualifié)
        - skills_density   : cv_nb_skills / (offre_nb_skills + 1)
        - overlap_x_loc    : interaction skill_overlap × loc_score
        - match_difficulty : exp_min_requis / (cv_nb_skills + 1)

      BINAIRES (33 skills) :
        - skill_python, skill_sql, skill_machine_learning, ...

    Returns:
        X : DataFrame des features
        y : Série des labels (0/1)
        feature_names : liste des noms de features
    """
    if dataset_path is None:
        dataset_path = DATA_LAKE_PROC / "dataset_matching.parquet"

    if not dataset_path.exists():
        log.info("Dataset non trouvé — génération automatique...")
        from ml.dataset import generate_dataset, save_dataset
        df_raw = generate_dataset(n_samples=500)
        save_dataset(df_raw, dataset_path)

    df = pd.read_parquet(dataset_path)
    log.info("Dataset chargé : %d lignes, %d colonnes", len(df), len(df.columns))

    if "label" not in df.columns and "match" in df.columns:
        df = df.rename(columns={"match": "label"})

    # Colonnes à exclure (métadonnées, target, identifiants)
    exclude_cols = {
        "label", "match", "score_reel", "sample_id",
        "cv_profile", "offre_profile", "type_contrat",
        "n_skills_overlap",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df["label"]

    # ── Features composites (C5.2.1 — non-linéarité) ─────────────────────────
    X["exp_gap"]          = X["cv_experience"] - X["exp_min_requis"]
    X["skills_density"]   = X["cv_nb_skills"]  / (X["offre_nb_skills"] + 1)
    X["overlap_x_loc"]    = X["skill_overlap"] * X["loc_score"]
    X["match_difficulty"] = X["exp_min_requis"] / (X["cv_nb_skills"] + 1)
    X["exp_match_sq"]     = X["exp_match"] ** 2                     # non-linéarité
    X["skills_sq"]        = X["skill_overlap"] ** 2                 # carré overlap

    # Remplir les éventuels NaN
    X = X.fillna(0)

    feature_names = list(X.columns)
    log.info("Features construites : %d (core + composites + binaires)", len(feature_names))
    log.info("Distribution labels  : %s", y.value_counts().to_dict())

    return X, y, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# C5.2.2 — Sélection de features (3 méthodes + consensus)
# ══════════════════════════════════════════════════════════════════════════════

def select_features(X: pd.DataFrame, y: pd.Series,
                    k: int = 15) -> tuple[list[str], dict]:
    """
    Compare 3 méthodes de sélection de features et produit un consensus.

    Méthodes comparées (C5.2.2) :
      1. ANOVA F-score  : test statistique univarié (hypothèse linéaire)
      2. Mutual Info    : mesure d'information mutuelle (non-linéaire)
      3. Random Forest  : importance par réduction d'impureté (modèle-agnostique)

    Stratégie de consensus :
      Une feature est retenue si elle apparaît dans ≥2 méthodes parmi 3.

    Returns:
        selected_features : liste des features sélectionnées
        report            : dictionnaire détaillé des scores pour le jury
    """
    log.info("Sélection de features — comparaison de 3 méthodes (k=%d)", k)

    # ── Méthode 1 : ANOVA F-score ────────────────────────────────────────────
    selector_anova = SelectKBest(score_func=f_classif, k=k)
    selector_anova.fit(X, y)
    features_anova = set(X.columns[selector_anova.get_support()])

    scores_anova = dict(zip(X.columns, selector_anova.scores_))
    top_anova    = sorted(scores_anova, key=scores_anova.get, reverse=True)[:k]

    # ── Méthode 2 : Mutual Information ──────────────────────────────────────
    mi_scores     = mutual_info_classif(X, y, random_state=42)
    scores_mi     = dict(zip(X.columns, mi_scores))
    features_mi   = set(sorted(scores_mi, key=scores_mi.get, reverse=True)[:k])
    top_mi        = sorted(scores_mi, key=scores_mi.get, reverse=True)[:k]

    # ── Méthode 3 : Random Forest importance ─────────────────────────────────
    rf_selector   = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X, y)
    scores_rf     = dict(zip(X.columns, rf_selector.feature_importances_))
    features_rf   = set(sorted(scores_rf, key=scores_rf.get, reverse=True)[:k])
    top_rf        = sorted(scores_rf, key=scores_rf.get, reverse=True)[:k]

    # ── Consensus : apparaît dans ≥2 méthodes ────────────────────────────────
    all_candidates = features_anova | features_mi | features_rf
    consensus = sorted(
        f for f in all_candidates
        if sum([f in features_anova, f in features_mi, f in features_rf]) >= 2
    )

    # Si moins de 10 features dans le consensus, on complète avec le top ANOVA
    if len(consensus) < 10:
        for f in top_anova:
            if f not in consensus:
                consensus.append(f)
            if len(consensus) >= k:
                break

    log.info("Features sélectionnées (%d) :", len(consensus))
    for f in consensus[:10]:
        log.info("  %-30s | ANOVA: %6.2f | MI: %.4f | RF: %.4f",
                 f,
                 scores_anova.get(f, 0),
                 scores_mi.get(f, 0),
                 scores_rf.get(f, 0))
    if len(consensus) > 10:
        log.info("  ... (%d features au total)", len(consensus))

    report = {
        "methode_anova":         {"top_features": top_anova, "scores": {k: round(v, 4) for k, v in sorted(scores_anova.items(), key=lambda x: -x[1])[:20]}},
        "methode_mutual_info":   {"top_features": top_mi,    "scores": {k: round(v, 4) for k, v in sorted(scores_mi.items(),    key=lambda x: -x[1])[:20]}},
        "methode_random_forest": {"top_features": top_rf,    "scores": {k: round(v, 4) for k, v in sorted(scores_rf.items(),    key=lambda x: -x[1])[:20]}},
        "consensus":             consensus,
        "n_features_selectionnees": len(consensus),
    }

    return consensus, report


# ══════════════════════════════════════════════════════════════════════════════
# C5.2.3 — Entraînement et évaluation de 3 modèles
# ══════════════════════════════════════════════════════════════════════════════

def train_and_compare(X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series,  y_test: pd.Series,
                      feature_names: list[str]) -> dict:
    """
    Entraîne et compare 3 modèles avec cross-validation + métriques complètes.

    Modèles comparés (C5.2.3) :
      1. Régression Logistique — modèle linéaire, interprétable, baseline solide
      2. Random Forest         — ensemble bagging, robuste au bruit
      3. Gradient Boosting     — ensemble boosting, souvent le plus performant

    Métriques calculées :
      - Accuracy, Precision, Recall, F1-score (classification binaire)
      - ROC-AUC (discrimination globale)
      - Cross-validation 5 plis stratifiée (variance du score)

    Returns:
        dict avec résultats détaillés pour chaque modèle
    """
    log.info("═" * 55)
    log.info("Entraînement et comparaison de 3 modèles")
    log.info("  Train : %d | Test : %d | Features : %d",
             len(X_train), len(X_test), len(feature_names))
    log.info("═" * 55)

    models = {
        "Régression Logistique": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000,
                class_weight="balanced",   # gère le déséquilibre de classes
                random_state=42,
                C=1.0,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        log.info("\n── %s ──", name)
        t0 = time.time()

        # ── Cross-validation (C5.2.3) ────────────────────────────────────────
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring="roc_auc", n_jobs=-1)
        log.info("  CV ROC-AUC : %.4f ± %.4f (5 plis)", cv_scores.mean(), cv_scores.std())

        # ── Entraînement final ────────────────────────────────────────────────
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)
        y_proba   = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # ── Métriques sur test ────────────────────────────────────────────────
        acc       = accuracy_score(y_test, y_pred)
        prec      = precision_score(y_test, y_pred, zero_division=0)
        rec       = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        auc       = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        duree     = time.time() - t0

        log.info("  Accuracy   : %.4f", acc)
        log.info("  Precision  : %.4f", prec)
        log.info("  Recall     : %.4f", rec)
        log.info("  F1-score   : %.4f", f1)
        if auc:
            log.info("  ROC-AUC    : %.4f (test)", auc)
        log.info("  Durée      : %.2f s", duree)

        # Rapport de classification complet
        clf_report = classification_report(y_test, y_pred,
                                           target_names=["Non-match", "Match"],
                                           output_dict=True)

        results[name] = {
            "model":          model,
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std":  round(float(cv_scores.std()),  4),
            "cv_scores":       cv_scores.tolist(),
            "test_accuracy":   round(acc,  4),
            "test_precision":  round(prec, 4),
            "test_recall":     round(rec,  4),
            "test_f1":         round(f1,   4),
            "test_roc_auc":    round(float(auc), 4) if auc else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": clf_report,
            "training_time_s": round(duree, 3),
        }

    # ── Comparaison et sélection du meilleur modèle ──────────────────────────
    log.info("\n═" * 55)
    log.info("COMPARAISON FINALE")
    log.info("%-28s %-10s %-10s %-10s", "Modèle", "CV AUC", "Test F1", "Test AUC")
    log.info("-" * 60)
    for name, r in results.items():
        log.info("%-28s %-10.4f %-10.4f %-10s",
                 name,
                 r["cv_roc_auc_mean"],
                 r["test_f1"],
                 str(r["test_roc_auc"]) if r["test_roc_auc"] else "N/A")

    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    log.info("\n🏆 Meilleur modèle : %s (CV AUC = %.4f)", best_name, results[best_name]["cv_roc_auc_mean"])
    results["_best_model_name"] = best_name

    return results


# ══════════════════════════════════════════════════════════════════════════════
# C5.2.4 — Optimisation des hyperparamètres (GridSearchCV)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame,  y_test: pd.Series,
                              skip: bool = False) -> dict:
    """
    Optimise les hyperparamètres du Random Forest par GridSearchCV.

    Grille de recherche (27 combinaisons) :
      - n_estimators : [50, 100, 200]
      - max_depth    : [5, 10, None]
      - min_samples_split : [2, 5, 10]

    Validation croisée : 5 plis stratifiés, scoring = ROC-AUC.

    Returns:
        dict avec best_params, best_score, model optimisé, et rapport
    """
    if skip:
        log.info("GridSearchCV ignoré (--skip-grid-search)")
        return {"skipped": True}

    log.info("GridSearchCV — Random Forest (27 combinaisons × 5 plis = 135 fits)")
    t0 = time.time()

    param_grid = {
        "n_estimators":     [50, 100, 200],
        "max_depth":        [5, 10, None],
        "min_samples_split": [2, 5, 10],
    }

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    duree = time.time() - t0

    log.info("GridSearchCV terminé en %.1f s", duree)
    log.info("Meilleurs hyperparamètres : %s", grid_search.best_params_)
    log.info("Meilleur score CV (AUC)   : %.4f", grid_search.best_score_)

    # Évaluation du modèle optimisé sur le test set
    best_model  = grid_search.best_estimator_
    y_pred_opt  = best_model.predict(X_test)
    y_proba_opt = best_model.predict_proba(X_test)[:, 1]
    auc_opt     = roc_auc_score(y_test, y_proba_opt)
    f1_opt      = f1_score(y_test, y_pred_opt, zero_division=0)

    log.info("Modèle optimisé — Test AUC : %.4f | F1 : %.4f", auc_opt, f1_opt)

    # Top 10 des combinaisons
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    top10 = (
        cv_results_df[["params", "mean_test_score", "std_test_score"]]
        .sort_values("mean_test_score", ascending=False)
        .head(10)
        .to_dict("records")
    )

    return {
        "skipped":           False,
        "best_params":       grid_search.best_params_,
        "best_cv_score":     round(grid_search.best_score_, 4),
        "test_roc_auc":      round(auc_opt, 4),
        "test_f1":           round(f1_opt, 4),
        "n_combinations":    len(grid_search.cv_results_["params"]),
        "duration_s":        round(duree, 1),
        "top10_combinations": top10,
        "model":             best_model,
    }


# ══════════════════════════════════════════════════════════════════════════════
# C5.3.1 — Sauvegarde et versionning du modèle
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, feature_names: list[str],
               comparison_results: dict, grid_results: dict,
               selection_report: dict,
               version: str = "v1") -> Path:
    """
    Sauvegarde le modèle et ses métadonnées pour le déploiement.

    Structure créée :
      data_lake/models/v1/
        ├── model.joblib           ← modèle sérialisé
        ├── feature_names.json     ← liste ordonnée des features
        ├── metadata.json          ← métriques, hyperparamètres, contexte
        ├── feature_selection.json ← rapport de sélection (C5.2.2)
        └── comparison.json        ← comparaison des 3 modèles (C5.2.3)
      data_lake/models/latest/     ← copie du meilleur modèle (symlink logique)
    """
    model_dir = DATA_LAKE_MODELS / version
    model_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Sauvegarder le modèle
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    log.info("Modèle sauvegardé → %s (%.1f Ko)", model_path, model_path.stat().st_size / 1024)

    # 2. Features
    features_path = model_dir / "feature_names.json"
    features_path.write_text(json.dumps(feature_names, ensure_ascii=False, indent=2))

    # 3. Métadonnées complètes
    best_name = comparison_results.get("_best_model_name", "Random Forest")
    #best_name = "Random Forest (optimisé GridSearchCV)"
    best_res  = comparison_results.get(best_name, {})

    metadata = {
        "version":          version,
        "date_creation":    timestamp,
        "modele_type":      type(model).__name__,
        "modele_nom":       best_name,
        "n_features":       len(feature_names),
        "performances": {
            "cv_roc_auc_mean":  best_res.get("cv_roc_auc_mean"),
            "cv_roc_auc_std":   best_res.get("cv_roc_auc_std"),
            "test_accuracy":    best_res.get("test_accuracy"),
            "test_precision":   best_res.get("test_precision"),
            "test_recall":      best_res.get("test_recall"),
            "test_f1":          best_res.get("test_f1"),
            "test_roc_auc":     best_res.get("test_roc_auc"),
        },
        "hyperparametres_optimises": grid_results.get("best_params") if not grid_results.get("skipped") else "non effectué",
        "gridsearch_score":         grid_results.get("best_cv_score"),
        "contexte": {
            "projet":         "ATS Intelligent — RNCP39586",
            "auteur":         "Mourad Amoussa",
            "formation":      "M2 IA & Data — Lyon Ynov Campus",
            "description":    "Modèle de matching CV ↔ Offre d'emploi IT",
            "dataset":        "Synthétique 500 échantillons (reproductible seed=42)",
            "competences_rncp": ["C5.2.1", "C5.2.2", "C5.2.3", "C5.2.4", "C5.3.1"],
        },
    }
    meta_path = model_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    # 4. Rapport sélection features
    sel_path = model_dir / "feature_selection.json"
    sel_path.write_text(json.dumps(selection_report, ensure_ascii=False, indent=2))

    # 5. Comparaison modèles (sans les objets Python non-sérialisables)
    comp_serializable = {}
    for k, v in comparison_results.items():
        if k == "_best_model_name":
            comp_serializable[k] = v
        elif isinstance(v, dict) and "model" in v:
            comp_serializable[k] = {kk: vv for kk, vv in v.items() if kk != "model"}
        else:
            comp_serializable[k] = v

    comp_path = model_dir / "comparison.json"
    comp_path.write_text(json.dumps(comp_serializable, ensure_ascii=False, indent=2))

    # 6. Copie dans "latest" (déploiement)
    latest_dir = DATA_LAKE_MODELS / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["model.joblib", "feature_names.json", "metadata.json"]:
        src = model_dir / fname
        dst = latest_dir / fname
        dst.write_bytes(src.read_bytes())

    log.info("Copie dans latest/ → %s", latest_dir)
    log.info("Modèle prêt pour déploiement ✅")

    return model_dir


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(skip_grid_search: bool = False,
                      n_samples: int = 500,
                      test_size: float = 0.2,
                      version: str = "v1") -> dict:
    """
    Exécute le pipeline ML complet de A à Z.

    Étapes :
      1. Chargement + feature engineering (C5.2.1)
      2. Split train/test (stratifié)
      3. Sélection de features — 3 méthodes (C5.2.2)
      4. Entraînement + évaluation — 3 modèles (C5.2.3)
      5. Optimisation hyperparamètres — GridSearchCV (C5.2.4)
      6. Sauvegarde (C5.3.1)

    Returns:
        Dictionnaire complet des résultats
    """
    t_global = time.time()

    log.info("╔" + "═" * 58 + "╗")
    log.info("║  Pipeline ML — ATS Intelligent %-26s║", "")
    log.info("║  RNCP39586 — C5.2.1 / C5.2.2 / C5.2.3 / C5.2.4 / C5.3.1  ║")
    log.info("╚" + "═" * 58 + "╝")

    # ── 1. Chargement + Feature Engineering (C5.2.1) ─────────────────────────
    log.info("\n[1/5] Feature Engineering...")
    X, y, all_features = load_and_prepare()

    # ── 2. Split train/test ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    log.info("Split : %d train / %d test (stratifié)", len(X_train), len(X_test))

    # ── 3. Sélection de features (C5.2.2) ────────────────────────────────────
    log.info("\n[2/5] Sélection de features...")
    selected_features, selection_report = select_features(X_train, y_train, k=15)

    X_train_sel = X_train[selected_features]
    X_test_sel  = X_test[selected_features]
    log.info("%d features retenues par consensus", len(selected_features))

    # ── 4. Entraînement + comparaison (C5.2.3) ───────────────────────────────
    log.info("\n[3/5] Entraînement des modèles...")
    comparison_results = train_and_compare(
        X_train_sel, X_test_sel, y_train, y_test, selected_features
    )

    # ── 5. GridSearchCV (C5.2.4) ─────────────────────────────────────────────
    log.info("\n[4/5] Optimisation hyperparamètres (GridSearchCV)...")
    grid_results = optimize_hyperparameters(
        X_train_sel, y_train, X_test_sel, y_test, skip=skip_grid_search
    )

    # ── Sélection du modèle final ─────────────────────────────────────────────
    if not grid_results.get("skipped") and grid_results.get("model"):
        final_model = grid_results["model"]
        log.info("Modèle final : Random Forest (optimisé par GridSearchCV)")
    else:
        best_name   = comparison_results["_best_model_name"]
        final_model = comparison_results[best_name]["model"]
        log.info("Modèle final : %s (sans GridSearchCV)", best_name)

    # ── 6. Sauvegarde (C5.3.1) ───────────────────────────────────────────────
    log.info("\n[5/5] Sauvegarde du modèle...")
    model_dir = save_model(
        model=final_model,
        feature_names=selected_features,
        comparison_results=comparison_results,
        grid_results=grid_results,
        selection_report=selection_report,
        version=version,
    )
    
    # ── MLflow Tracking & Model Registry (C5.3.1) ─────────────────────────────
    try:
        # URI vers le serveur MLflow Docker (localhost:5001)
        # Fallback sur SQLite local si le serveur est inaccessible
        import os as _os
        _mlflow_uri = _os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
        try:
            import requests as _req
            _req.get(_mlflow_uri, timeout=2)
        except Exception:
            _mlflow_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(_mlflow_uri)
        mlflow.set_experiment("ATS_Matching_Model")
        with mlflow.start_run(run_name=f"RF_GridSearchCV_{version}"):
            # Hyperparamètres GridSearchCV
            if not grid_results.get("skipped"):
                mlflow.log_params(grid_results["best_params"])
                mlflow.log_metric("gridsearch_cv_auc", grid_results["best_cv_score"])
            # Métriques du modèle final
            _best_name = comparison_results.get("_best_model_name", "")
            _best_res  = comparison_results.get(_best_name, {})
            mlflow.log_metric("cv_roc_auc_mean", _best_res.get("cv_roc_auc_mean", 0))
            mlflow.log_metric("test_accuracy",   _best_res.get("test_accuracy",   0))
            mlflow.log_metric("test_precision",  _best_res.get("test_precision",  0))
            mlflow.log_metric("test_recall",     _best_res.get("test_recall",     0))
            mlflow.log_metric("test_f1",         _best_res.get("test_f1",         0))
            mlflow.log_metric("test_roc_auc",    _best_res.get("test_roc_auc",    0) or 0)
            # Tags contexte
            mlflow.set_tag("n_features", len(selected_features))
            mlflow.set_tag("version",    version)
            mlflow.set_tag("auteur",     "Mourad Amoussa")
            mlflow.set_tag("dataset",    "Synthétique 500 échantillons (seed=42)")
            # Enregistrement dans le Model Registry
            mlflow.sklearn.log_model(
                final_model,
                artifact_path="model",
                registered_model_name="ATS_Matching_Model",
            )
            log.info("MLflow ✅ run enregistré (experiment=ATS_Matching_Model)")
    except Exception as mlflow_err:
        log.warning("MLflow tracking ignoré : %s", mlflow_err)
    

    
    duree_total = time.time() - t_global
    log.info("\n✅ Pipeline terminé en %.1f s", duree_total)
    log.info("   Modèle disponible : %s", model_dir)

    return {
        "model_dir":         str(model_dir),
        "selected_features": selected_features,
        "comparison":        {k: {kk: vv for kk, vv in v.items() if kk != "model"}
                              for k, v in comparison_results.items()
                              if k != "_best_model_name"},
        "best_model_name":   comparison_results.get("_best_model_name"),
        "grid_search":       {k: v for k, v in grid_results.items() if k != "model"},
        "duration_s":        round(duree_total, 1),
    }


# ── Point d'entrée CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML — ATS Intelligent (RNCP39586)")
    parser.add_argument("--skip-grid-search", action="store_true",
                        help="Sauter le GridSearchCV (entraînement rapide)")
    parser.add_argument("--samples",  type=int,   default=500,
                        help="Nombre d'échantillons à générer si dataset manquant")
    parser.add_argument("--version",  type=str,   default="v1",
                        help="Version du modèle à sauvegarder")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion du jeu de test (défaut: 0.2)")
    args = parser.parse_args()

    results = run_full_pipeline(
        #skip_grid_search=args.skip_grid_search,
        n_samples=args.samples,
        test_size=args.test_size,
        version=args.version,
    )

    print("\n📊 Résumé :")
    print(f"  Meilleur modèle   : {results['best_model_name']}")
    print(f"  Features retenues : {len(results['selected_features'])}")
    print(f"  Durée totale      : {results['duration_s']} s")
    print(f"  Modèle sauvegardé : {results['model_dir']}")
