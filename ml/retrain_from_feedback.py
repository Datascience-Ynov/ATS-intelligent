"""
ml/retrain_from_feedback.py — Réentraînement du modèle à partir des feedbacks réels.
Auteur : Mourad Amoussa | RNCP39586 — C2.1.4, C3.x

Principe (Reinforcement Learning supervisé) :
  1. Récupère les candidatures avec feedback_resultat renseigné (entretien_obtenu / refus)
  2. Extrait les features depuis scores_matching.features_json
  3. Combine avec le dataset synthétique existant
  4. Réentraîne le RandomForestClassifier
  5. Logue dans MLflow avec tag source=feedback_reel
  6. Sauvegarde le nouveau modèle dans data_lake/models/latest/

Usage :
    python ml/retrain_from_feedback.py
    python ml/retrain_from_feedback.py --min-samples 5
"""

from __future__ import annotations
import argparse
import json
import os
import smtplib
import sys
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# ── Alerting email ────────────────────────────────────────────────────────────
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "251102mourad@gmail.com")
ALERT_EMAIL_TO   = os.getenv("ALERT_EMAIL_TO",   "251102mourad@gmail.com")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")   # App Password Gmail (pas le mot de passe Google)

def send_alert_email(subject: str, body: str) -> None:
    """Envoie un email d'alerte via Gmail SMTP.

    Prérequis : créer un App Password Gmail (compte Google → Sécurité → Mots de passe des applications)
    et le mettre dans la variable d'environnement GMAIL_APP_PASSWORD.
    """
    if not GMAIL_APP_PASSWORD:
        print(f"[alert] ⚠️ GMAIL_APP_PASSWORD non défini — email non envoyé. Sujet : {subject}")
        return
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = f"[ATS Intelligent] {subject}"
        msg["From"]    = ALERT_EMAIL_FROM
        msg["To"]      = ALERT_EMAIL_TO

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(ALERT_EMAIL_FROM, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print(f"[alert] ✅ Email envoyé à {ALERT_EMAIL_TO} : {subject}")
    except Exception as e:
        print(f"[alert] ❌ Échec envoi email : {e}")
MODELS_DIR = ROOT / "data_lake" / "models" / "latest"
DATASET_PATH = ROOT / "data_lake" / "processed" / "dataset_matching.parquet"

FEATURE_COLUMNS = [
    "skill_overlap", "soft_overlap", "exp_gap",
    "cv_meme_ville", "salaire_compatible",
    "cv_teletravail", "offre_teletravail",
]


def get_feedback_data(min_samples: int = 5) -> pd.DataFrame | None:
    """Récupère les candidatures avec feedback depuis PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", 15432)),
            dbname=os.getenv("DB_NAME", "ats_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
        )
        query = """
            SELECT
                sm.features_json,
                CASE
                    WHEN c.statut IN ('entretien', 'test_technique', 'offre', 'accepte') THEN 1
                    WHEN c.statut = 'refus' THEN 0
                    ELSE NULL
                END AS label_reel
            FROM candidatures c
            JOIN scores_matching sm
                ON sm.cv_id = c.cv_id AND sm.offre_id = c.offre_id
            WHERE c.statut IN ('entretien', 'test_technique', 'offre', 'accepte', 'refus')
              AND sm.features_json IS NOT NULL
        """
        df = pd.read_sql(query, conn)
        conn.close()

        df = df.dropna(subset=["label_reel"])
        if len(df) < min_samples:
            print(f"[retrain] Seulement {len(df)} feedbacks — minimum {min_samples} requis. Abandon.")
            return None

        # Décompresser features_json → colonnes
        features = pd.json_normalize(df["features_json"].apply(
            lambda x: x if isinstance(x, dict) else json.loads(x)
        ))
        df = pd.concat([features, df[["label_reel"]].reset_index(drop=True)], axis=1)
        print(f"[retrain] {len(df)} feedbacks réels récupérés (1={df['label_reel'].sum():.0f} succès, 0={len(df)-df['label_reel'].sum():.0f} refus)")
        return df
    except Exception as e:
        print(f"[retrain] Erreur DB : {e}")
        return None


def main(min_samples: int = 5) -> None:
    # ── 1. Feedbacks réels ────────────────────────────────────────────────
    df_feedback = get_feedback_data(min_samples=min_samples)

    # ── 2. Dataset synthétique de base ───────────────────────────────────
    if not DATASET_PATH.exists():
        print(f"[retrain] Dataset synthétique introuvable : {DATASET_PATH}")
        sys.exit(1)

    df_synth = pd.read_parquet(DATASET_PATH)

    # Harmoniser les colonnes
    for col in FEATURE_COLUMNS:
        if col not in df_synth.columns:
            df_synth[col] = 0
    df_synth = df_synth[FEATURE_COLUMNS + ["label"]].rename(columns={"label": "label_reel"})

    # ── 3. Fusionner les deux sources ─────────────────────────────────────
    if df_feedback is not None:
        for col in FEATURE_COLUMNS:
            if col not in df_feedback.columns:
                df_feedback[col] = 0
        df_feedback = df_feedback[FEATURE_COLUMNS + ["label_reel"]]
        df_combined = pd.concat([df_synth, df_feedback], ignore_index=True)
        source_tag = "feedback_reel+synthetique"
        n_feedback = len(df_feedback)
    else:
        df_combined = df_synth
        source_tag = "synthetique_only"
        n_feedback = 0

    X = df_combined[FEATURE_COLUMNS].fillna(0)
    y = df_combined["label_reel"].astype(int)

    print(f"[retrain] Dataset total : {len(X)} lignes ({n_feedback} feedbacks réels + {len(df_synth)} synthétiques)")

    # ── 4. Split train/test ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 5. Entraînement ───────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=3,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"[retrain] F1={f1:.4f}  AUC={auc:.4f}")

    # ── 6. MLflow logging ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("ATS_Retrain_Feedback")

    with mlflow.start_run(tags={"source": source_tag}):
        mlflow.log_param("n_total",       len(X))
        mlflow.log_param("n_feedback",    n_feedback)
        mlflow.log_param("n_synthetique", len(df_synth))
        mlflow.log_param("n_estimators",  300)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc",  auc)
        mlflow.sklearn.log_model(model, "model_retrain")
        run_id = mlflow.active_run().info.run_id

    print(f"[retrain] Run MLflow : {run_id}")

    # ── 7. Lire le F1 du modèle précédent (pour comparaison alerting) ────────
    previous_f1: float | None = None
    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        try:
            prev_meta = json.loads(meta_path.read_text())
            previous_f1 = float(prev_meta.get("performances", {}).get("test_f1", 0.0))
        except Exception:
            previous_f1 = None

    # ── 8. Sauvegarder le modèle si F1 ≥ 0.75 ───────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    if f1 >= 0.75:
        joblib.dump(model, MODELS_DIR / "model.pkl")

        metadata = {
            "version":      "v_retrain",
            "modele_nom":   "Random Forest (réentraîné feedback réel)",
            "modele_type":  "RandomForestClassifier",
            "n_features":   len(FEATURE_COLUMNS),
            "features":     FEATURE_COLUMNS,
            "performances": {
                "test_f1":      round(f1,  4),
                "test_roc_auc": round(auc, 4),
            },
            "source":       source_tag,
            "n_feedback":   n_feedback,
            "mlflow_run_id": run_id,
        }
        (MODELS_DIR / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        print(f"[retrain] ✅ Modèle sauvegardé (F1={f1:.4f} ≥ 0.75)")

        # Alerte si dégradation significative par rapport au modèle précédent
        if previous_f1 is not None and f1 < previous_f1 - 0.05:
            send_alert_email(
                subject=f"⚠️ Dégradation F1 détectée ({ts})",
                body=(
                    f"Le modèle ATS vient d'être réentraîné mais ses performances ont baissé.\n\n"
                    f"  F1 précédent : {previous_f1:.4f}\n"
                    f"  F1 nouveau   : {f1:.4f}\n"
                    f"  AUC-ROC      : {auc:.4f}\n"
                    f"  Source       : {source_tag}\n"
                    f"  Feedbacks    : {n_feedback}\n"
                    f"  Run MLflow   : {run_id}\n\n"
                    f"Le nouveau modèle A quand même été déployé (F1 ≥ 0.75).\n"
                    f"Vérifiez la qualité des feedbacks dans MLflow : http://localhost:5001"
                ),
            )
        elif previous_f1 is not None and f1 >= previous_f1:
            print(f"[retrain] 📈 F1 amélioré : {previous_f1:.4f} → {f1:.4f}")

    else:
        print(f"[retrain] ⚠️ F1={f1:.4f} < 0.75 — modèle non remplacé (ancien conservé)")
        # Alerte systématique si le F1 est sous le seuil minimal acceptable
        send_alert_email(
            subject=f"🚨 Réentraînement échoué — F1 insuffisant ({ts})",
            body=(
                f"Le réentraînement du modèle ATS a produit un F1 trop faible.\n\n"
                f"  F1 obtenu  : {f1:.4f}  (seuil minimum : 0.75)\n"
                f"  AUC-ROC    : {auc:.4f}\n"
                f"  Source     : {source_tag}\n"
                f"  Feedbacks  : {n_feedback}\n"
                f"  Run MLflow : {run_id}\n\n"
                f"L'ancien modèle est conservé. Vérifiez la qualité des feedbacks.\n"
                f"MLflow : http://localhost:5001"
            ),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Réentraîne le modèle avec feedbacks réels")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Nombre minimum de feedbacks pour réentraîner (défaut: 5)")
    args = parser.parse_args()
    main(min_samples=args.min_samples)
