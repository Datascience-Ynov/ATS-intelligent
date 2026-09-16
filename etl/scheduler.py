"""
ETL Scheduler — ATS Intelligent
Auteur : Mourad Amoussa | Lyon Ynov M2 IA & Data | RNCP39586

Compétences couvertes :
  C1.1.2 — Automatisation de la collecte (CronTrigger APScheduler)
  C1.3.2 — Pipeline ETL planifié, traçabilité des exécutions

Lance le pipeline ETL automatiquement toutes les 2 heures.
Peut aussi être déclenché par n8n (via un nœud HTTP qui appelle ce script).

Usage :
    python -m etl.scheduler              # démarre le scheduler (tourne en continu)
    python -m etl.scheduler --run-now    # exécute le pipeline une fois et quitte
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
import random
    

import time

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Allow both `python -m etl.scheduler` and `python etl/scheduler.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from etl.pipeline import ETLPipeline

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("etl.scheduler")


import os as _os
# Nombre max d'offres par run — configurable via variable d'environnement ETL_MAX_OFFRES
# Valeur par défaut : 20 (rapide). Mettre 50-100 pour enrichir la base en production.
ETL_MAX_OFFRES = int(_os.getenv("ETL_MAX_OFFRES", "20"))


def run_pipeline() -> None:
    """Fonction appelée par le scheduler à chaque déclenchement."""
    log.info("=" * 60)
    log.info("[SCHEDULER] Déclenchement ETL — %s (max %d offres/run)",
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ETL_MAX_OFFRES)
    log.info("=" * 60)

    try:
        pipeline = ETLPipeline()
        results = pipeline.run(source="all", query="Data Scientist Python",
                               max_offres=ETL_MAX_OFFRES)
        log.info(
            "[SCHEDULER] ETL run 1 terminé — %d extraites, %d insérées, %d mises à jour, %d erreurs",
            results.get("nb_extraites", 0),
            results.get("nb_insérées", 0),
            results.get("nb_màj", 0),
            len(results.get("erreurs", [])),
        )

        # Pause avant le second run pour éviter le rate-limit 429 de France Travail
        log.info("[SCHEDULER] Pause 60 s avant run 2 (anti-rate-limit France Travail)…")
        time.sleep(60)

        # Second run pour diversifier les requêtes
        KEYWORDS_LIST = [
            "data scientist", "machine learning", "data engineer",
            "data analyst", "MLops", "NLP", "intelligence artificielle",
            "deep learning", "Python data", "LLM",

            # Data / IA / ML
            "machine learning engineer", "ML engineer", "AI engineer",
            "AI developer", "AI researcher", "deep learning engineer",
            "computer vision", "data architect", "data platform",
            "data manager", "business intelligence", "BI analyst",
            "BI developer", "statistician", "quantitative analyst",
            "data consultant", "RAG engineer", "genAI",

            # Dev / Software
            "backend developer", "frontend developer", "fullstack developer",
            "software engineer", "python developer", "java developer",
            "javascript developer", "typescript developer", "go developer",
            "rust developer", "c++ developer", "mobile developer",
            "android developer", "ios developer", "dev lead", "tech lead",

            # Cloud / DevOps / Infra
            "devops engineer", "cloud engineer", "cloud architect",
            "azure engineer", "aws engineer", "gcp engineer",
            "site reliability engineer", "SRE", "platform engineer",
            "system administrator", "network engineer", "security engineer",
            "cybersecurity analyst", "pentester", "cloud ops",

            # Produit / QA / Management
            "product owner", "product manager", "scrum master",
            "project manager", "IT project manager", "QA engineer",
            "test engineer", "automation tester", "quality assurance",

            # No-code / Low-code / Automation
            "no-code developer", "low-code developer", "n8n developer",
            "make.com developer", "zapier expert", "bubble developer",
            "airtable expert", "automation engineer", "workflow engineer",

            # Big Data / ETL / DB
            "data governance", "data quality", "data steward",
            "ETL developer", "big data engineer", "spark developer",
            "hadoop engineer", "database administrator", "DBA",
            "SQL developer", "API developer", "microservices engineer"
        ]

        keyword = random.choice(KEYWORDS_LIST)
        
        results2 = pipeline.run(source="france_travail", query=keyword,
                               max_offres=ETL_MAX_OFFRES)
        log.info(
            "[SCHEDULER] ETL run 2 terminé — %d extraites, %d insérées ('%s')",
            results2.get("nb_extraites", 0),
            results2.get("nb_insérées", 0),
            keyword,
        )

    except Exception as exc:
        log.error("[SCHEDULER] Erreur lors de l'exécution ETL : %s", exc, exc_info=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduler ETL — ATS Intelligent")
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Exécuter le pipeline immédiatement puis quitter",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=None,
        help="Heure de déclenchement quotidien (0-23). Par défaut : toutes les 2h.",
    )
    args = parser.parse_args()

    if args.run_now:
        log.info("[SCHEDULER] Mode --run-now : exécution immédiate")
        run_pipeline()
        log.info("[SCHEDULER] Terminé. Au revoir.")
        return

    # ── Configuration du scheduler ────────────────────────────────────────────
    scheduler = BlockingScheduler(timezone="Europe/Paris")

    if args.hour is not None:
        # Déclenchement une fois par jour à l'heure spécifiée
        trigger = CronTrigger(hour=args.hour, minute=0, timezone="Europe/Paris")
        log.info("[SCHEDULER] Déclenchement quotidien à %02dh00", args.hour)
    else:
        # Déclenchement toutes les 2 heures (00h00, 02h00, 04h00, ..., 22h00)
        # CronTrigger heure paire : hour="*/2" = toutes les 2h
        trigger = CronTrigger(hour="*/2", minute=0, timezone="Europe/Paris")
        log.info("[SCHEDULER] Déclenchement toutes les 2 heures (CronTrigger hour='*/2')")

    scheduler.add_job(
        func=run_pipeline,
        trigger=trigger,
        id="etl_job",
        name="ETL ATS Intelligent",
        replace_existing=True,
        misfire_grace_time=300,   # 5 min de grâce si le scheduler était éteint
        coalesce=True,            # une seule exécution si plusieurs déclenchements manqués
    )

    # ── Arrêt propre sur Ctrl+C / SIGTERM ────────────────────────────────────
    def shutdown(signum, frame):
        log.info("[SCHEDULER] Signal %s reçu — arrêt propre", signum)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    job = scheduler.get_jobs()[0]
    next_run = getattr(job, "next_run_time", None)
    if next_run is None:
        next_run = getattr(job, "next_fire_time", "non disponible")
    log.info("[SCHEDULER] Démarré. Prochain run : %s", next_run)
    log.info("[SCHEDULER] Ctrl+C ou SIGTERM pour arrêter.")
    scheduler.start()


if __name__ == "__main__":
    main()
