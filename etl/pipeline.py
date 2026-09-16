"""
ETL Pipeline — ATS Intelligent
Auteur : Mourad Amoussa | Lyon Ynov M2 IA & Data | RNCP39586

Compétences couvertes :
  C1.1.2 — Collecte de données multi-sources (France Travail API + python-jobspy)
  C1.2.2 — Solution Big Data : data lake Parquet (rôle de substitut Hadoop)
  C1.3.2 — Sécurité : credentials en variables d'env, pas en dur dans le code
  C1.3.3 — Intégrité et traçabilité : logs ETL en base, checksums, idempotence

Usage :
    python -m etl.pipeline                        # exécution complète
    python -m etl.pipeline --source france_travail --query "Data Scientist"
    python -m etl.pipeline --source jobspy        --query "Machine Learning Lyon"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv

import math

def _safe_int(v):
    """Convertit une valeur pandas (float/NaN/int/None) en int Python ou None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None

# ── Jobspy (optionnel — ne plante pas si absent)
try:
    from jobspy import scrape_jobs
    JOBSPY_AVAILABLE = True
except ImportError:
    JOBSPY_AVAILABLE = False

load_dotenv()

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("etl.pipeline")

# ── Constantes ───────────────────────────────────────────────────────────────────
DATA_LAKE_ROOT = Path(__file__).resolve().parent.parent / "data_lake"
DATA_LAKE_RAW  = DATA_LAKE_ROOT / "raw" / "offres_json"
DATA_LAKE_PROC = DATA_LAKE_ROOT / "processed"

DATA_LAKE_RAW.mkdir(parents=True, exist_ok=True)
DATA_LAKE_PROC.mkdir(parents=True, exist_ok=True)

# Ontologie des compétences IT — 40+ termes pour l'extraction NLP (C1.1.2)
COMPETENCES_IT: dict[str, str] = {
    "python":             "Python",
    "sql":                "SQL",
    "machine learning":   "Machine Learning",
    "deep learning":      "Deep Learning",
    "nlp":                "NLP",
    "traitement du langage": "NLP",
    "natural language":   "NLP",
    "scikit-learn":       "Scikit-learn",
    "sklearn":            "Scikit-learn",
    "tensorflow":         "TensorFlow",
    "pytorch":            "PyTorch",
    "keras":              "TensorFlow",
    "pandas":             "Pandas",
    "numpy":              "NumPy",
    "docker":             "Docker",
    "kubernetes":         "Kubernetes",
    "k8s":                "Kubernetes",
    "aws":                "AWS",
    "amazon web services": "AWS",
    "azure":              "Azure",
    "gcp":                "GCP",
    "google cloud":       "GCP",
    "bigquery":           "GCP",
    "spark":              "Spark",
    "pyspark":            "Spark",
    "apache spark":       "Spark",
    "hadoop":             "Hadoop",
    "fastapi":            "FastAPI",
    "fast api":           "FastAPI",
    "django":             "Django",
    "flask":              "Flask",
    "git":                "Git",
    "github":             "Git",
    "gitlab":             "Git",
    "power bi":           "Power BI",
    "powerbi":            "Power BI",
    "tableau":            "Tableau",
    "streamlit":          "Streamlit",
    "mlflow":             "MLflow",
    "airflow":            "Airflow",
    "dbt":                "dbt",
    "snowflake":          "Snowflake",
    "mongodb":            "MongoDB",
    "redis":              "Redis",
    "postgresql":         "PostgreSQL",
    "mysql":              "SQL",
    "r language":         "R",
    "rstudio":            "R",
    "scala":              "Scala",
    "java":               "Java",
    "javascript":         "JavaScript",
    "node.js":            "JavaScript",
    "nodejs":             "JavaScript",
    "react":              "JavaScript",
    "vue":                "JavaScript",
    "angular":            "JavaScript",
    "linux":              "Linux",
    "bash":               "Linux",
    "shell":              "Linux",
    "rest":               "API REST",
    "api rest":           "API REST",
    "microservices":      "Microservices",
    "agile":              "Agile",
    "scrum":              "Agile",
    "ci/cd":              "CI/CD",
    "devops":             "DevOps",
    "llm":                "LLM",
    "langchain":          "LangChain",
    "rag":                "RAG",
    "openai":             "LLM",
    "hugging face":       "Hugging Face",
    "transformers":       "Deep Learning",
}

# ── Fonctions de connexion DB ────────────────────────────────────────────────────

def get_db_connection() -> psycopg2.extensions.connection:
    """Ouvre une connexion PostgreSQL depuis les variables d'environnement (C1.3.2)."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "ats_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        connect_timeout=10,
    )


def log_etl_step(conn, etape: str, source: str, nb_in: int, nb_out: int,
                 nb_err: int = 0, statut: str = "success",
                 message: str = "", duree: float = 0.0) -> None:
    """Écrit un log ETL en base pour la traçabilité (C1.3.3)."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO etl_logs (etape, source, nb_lignes_in, nb_lignes_out,
                                  nb_erreurs, statut, message, duree_secondes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (etape, source, nb_in, nb_out, nb_err, statut, message, duree))
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT — Collecte multi-sources (C1.1.2)
# ══════════════════════════════════════════════════════════════════════════════

class FranceTravailExtractor:
    """
    Extrait les offres depuis l'API France Travail (Pôle Emploi).

    Authentification OAuth2 client_credentials — jamais de credentials en dur.
    Endpoint : https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search
    Max      : 150 offres par requête (limite API).
    """

    TOKEN_URL  = "https://entreprise.francetravail.fr/connexion/oauth2/access_token"
    SEARCH_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"

    def __init__(self) -> None:
        self.client_id     = os.getenv("FT_CLIENT_ID", "")
        self.client_secret = os.getenv("FT_CLIENT_SECRET", "")
        self.search_url    = os.getenv("FT_SEARCH_URL", self.SEARCH_URL)
        self._token: str   = ""
        self._token_expiry = 0.0

    def _get_token(self) -> str:
        """Récupère ou renouvelle le token OAuth2."""
        if self._token and time.time() < self._token_expiry - 60:
            return self._token

        if not self.client_id or not self.client_secret:
            raise EnvironmentError(
                "Variables FT_CLIENT_ID et FT_CLIENT_SECRET manquantes dans .env\n"
                "→ Créer un compte sur https://francetravail.io/data/api/"
            )

        resp = requests.post(
            self.TOKEN_URL,
            params={"realm": "/partenaire"},
            data={
                "grant_type":    "client_credentials",
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
                "scope":         "api_offresdemploiv2 o2dsoffre",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token        = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 1200)
        log.info("Token France Travail obtenu (expire dans %d s)", data.get("expires_in", 1200))
        return self._token

    def extract(self, query: str = "Data Scientist",
                departement: str = "", max_offres: int = 150) -> list[dict]:
        """
        Collecte jusqu'à max_offres offres pour une requête donnée.

        Args:
            query       : mots-clés de recherche (ex: "Machine Learning Lyon")
            departement : code département (ex: "69" pour Rhône)
            max_offres  : 150 max par appel API

        Returns:
            Liste de dicts bruts (format API France Travail)
        """
        token  = self._get_token()
        params = {
            "motsCles":  query,
            "range":     f"0-{min(max_offres, 149)}",
            "sort":      "1",            # tri par date
        }
        if departement:
            params["departement"] = departement

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept":        "application/json",
        }

        log.info("France Travail API — requête : '%s' (max %d)", query, max_offres)
        resp = requests.get(self.search_url, params=params, headers=headers, timeout=30)

        if resp.status_code == 206:
            # 206 = résultats partiels (normal si < max disponible)
            pass
        elif resp.status_code != 200:
            detail = resp.text[:300].replace("\n", " ")
            log.error(
                "France Travail API — erreur %d sur %s : %s",
                resp.status_code, self.search_url, detail,
            )
            raise RuntimeError(
                f"API France Travail indisponible ({resp.status_code}) sur "
                f"{self.search_url}. Vérifiez FT_SEARCH_URL ou l'état du service."
            )

        offres = resp.json().get("resultats", [])
        log.info("France Travail API — %d offres récupérées", len(offres))

        # Sauvegarde brute dans le data lake (C1.2.2)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        raw_path = DATA_LAKE_RAW / f"ft_{query.replace(' ', '_')}_{ts}.json"
        raw_path.write_text(json.dumps(offres, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Data lake RAW → %s", raw_path)

        return offres


class JobSpyExtractor:
    """
    Extrait les offres via python-jobspy (LinkedIn, Indeed, Glassdoor).
    Ne nécessite pas de clé API — scraping éthique avec délais.
    """

    def extract(self, query: str = "Data Scientist",
                location: str = "Lyon, France", max_results: int = 100) -> list[dict]:
        if not JOBSPY_AVAILABLE:
            log.warning("python-jobspy non installé — pip install python-jobspy")
            return []

        log.info("JobSpy — recherche '%s' à '%s'", query, location)
        try:
            df = scrape_jobs(
                site_name=["linkedin", "indeed"],  # Glassdoor retiré (400 sur localisation FR)
                search_term=query,
                location=location,
                results_wanted=max_results,
                hours_old=72,
                country_indeed="France",
            )
        except Exception as e:
            log.error("JobSpy — erreur scraping : %s", e)
            return []

        if df is None or df.empty:
            log.warning("JobSpy — aucune offre trouvée")
            return []

        log.info("JobSpy — %d offres récupérées", len(df))

        # Normalisation vers le format interne
        offres = []
        for _, row in df.iterrows():
            offres.append({
                #"id":          f"jobspy_{row.get('job_url_direct', '')[:50]}",
                "id":          f"jobspy_{str(row.get('job_url_direct', ''))[:100]}",
                "intitule":    str(row.get("title", "")),
                "description": {"libelle": str(row.get("description", ""))},
                "entreprise":  {"nom": str(row.get("company", ""))},
                "lieuTravail": {"libelle": str(row.get("location", ""))},
                "typeContrat": str(row.get("job_type", "")),
                "experienceLibelle": "",
                "salaire":     {"libelle": str(row.get("compensation", ""))},
                "origineOffre": {"urlOrigineOffre": str(row.get("job_url", ""))},
                "_source":     str(row.get("site", "jobspy")),
            })

        # Sauvegarde brute
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        raw_path = DATA_LAKE_RAW / f"jobspy_{query.replace(' ', '_')}_{ts}.json"
        raw_path.write_text(json.dumps(offres, ensure_ascii=False, indent=2), encoding="utf-8")

        return offres


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM — Nettoyage, extraction NLP, normalisation (C1.1.2)
# ══════════════════════════════════════════════════════════════════════════════

class DataTransformer:
    """
    Transforme les données brutes en données structurées exploitables.

    Opérations :
      - Nettoyage du texte (HTML, caractères spéciaux)
      - Extraction des compétences par matching ontologique
      - Normalisation du type de contrat
      - Détection de la séniorité
      - Parsing du salaire
    """

    # Mapping normalisation type de contrat
    CONTRAT_MAP = {
        "cdi":         "CDI",
        "cdd":         "CDD",
        "stage":       "Stage",
        "alternance":  "Alternance",
        "apprentissage": "Alternance",
        "interim":     "Intérim",
        "freelance":   "Freelance",
        "independant": "Freelance",
        "full-time":   "CDI",
        "permanent":   "CDI",
        "contract":    "CDD",
        "internship":  "Stage",
    }

    SENIORITE_PATTERNS = {
        "Junior":    [r"\bjunior\b", r"\bdébutant\b", r"\b0[-–]?[123]\s*ans?\b"],
        "Mid-level": [r"\bconfirmé\b", r"\b[23][-–]?5\s*ans?\b"],
        "Senior":    [r"\bsenior\b", r"\bexpert\b", r"\b5\s*ans?\s*et\s*plus\b", r"\b[5-9]\+?\s*ans?\b"],
        "Lead":      [r"\blead\b", r"\bmanager\b", r"\bdirecteur\b", r"\barchitecte\b"],
    }

    @staticmethod
    def clean_text(text: str) -> str:
        """Supprime le HTML, nettoie les espaces et caractères parasites."""
        if not text:
            return ""
        # Suppression balises HTML
        text = re.sub(r"<[^>]+>", " ", text)
        # Entités HTML courantes
        text = text.replace("&nbsp;",  " ").replace("&amp;",   "&") \
                   .replace("&lt;",    "<").replace("&gt;",    ">") \
                   .replace("&eacute;","é").replace("&egrave;","è") \
                   .replace("&agrave;","à").replace("&#39;",   "'")
        # Espaces multiples
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def extract_competences(text: str) -> list[str]:
        """
        Extrait les compétences IT par matching sur l'ontologie COMPETENCES_IT.
        Retourne une liste dédupliquée et triée.
        """
        text_lower = text.lower()
        found: set[str] = set()
        for keyword, canonical in COMPETENCES_IT.items():
            # Recherche avec délimiteurs de mots pour éviter les faux positifs
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, text_lower):
                found.add(canonical)
        return sorted(found)

    @staticmethod
    def normalize_contrat(raw: str) -> str:
        """Normalise le type de contrat vers une valeur canonique."""
        if not raw:
            return "Non précisé"
        raw_lower = raw.lower().strip()
        for key, val in DataTransformer.CONTRAT_MAP.items():
            if key in raw_lower:
                return val
        return raw.upper() if len(raw) <= 5 else raw.title()

    @staticmethod
    def detect_seniority(text: str) -> str:
        """Détecte le niveau de séniorité depuis le texte de l'offre."""
        text_lower = text.lower()
        for level, patterns in DataTransformer.SENIORITE_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    return level
        return "Non précisé"

    @staticmethod
    def parse_salary(salaire_libelle: str) -> tuple[int | None, int | None]:
        """
        Extrait salaire min/max depuis une chaîne libre.
        Ex: "35-45K€" → (35000, 45000)
            "À partir de 40K" → (40000, None)
        """
        if not salaire_libelle:
            return None, None
        text = salaire_libelle.replace(" ", "").replace(" ", "")
        # Fourchette "35-45K"
        match = re.search(r"(\d+)[-–à](\d+)\s*[kK]", text)
        if match:
            return int(match.group(1)) * 1000, int(match.group(2)) * 1000
        # Valeur unique "40K"
        match = re.search(r"(\d+)\s*[kK]", text)
        if match:
            val = int(match.group(1)) * 1000
            return val, None
        # Valeur brute en euros "45000"
        match = re.search(r"(\d{4,6})", text)
        if match:
            val = int(match.group(1))
            if 15000 <= val <= 200000:
                return val, None
        return None, None

    def transform_ft_offre(self, raw: dict) -> dict | None:
        """Transforme une offre brute France Travail en dict normalisé."""
        try:
            desc_brut = raw.get("description", {})
            if isinstance(desc_brut, dict):
                desc_text = desc_brut.get("libelle", "")
            else:
                desc_text = str(desc_brut)

            description = self.clean_text(desc_text)
            titre        = self.clean_text(raw.get("intitule", ""))
            texte_complet = f"{titre} {description}"

            salaire_raw  = raw.get("salaire", {})
            salaire_lib  = salaire_raw.get("libelle", "") if isinstance(salaire_raw, dict) else ""
            sal_min, sal_max = self.parse_salary(salaire_lib)

            lieu = raw.get("lieuTravail", {})
            lieu_lib = lieu.get("libelle", "") if isinstance(lieu, dict) else str(lieu)
            cp = lieu.get("codePostal", "") if isinstance(lieu, dict) else ""

            return {
                "id_offre":       raw.get("id", f"ft_{hashlib.md5(titre.encode()).hexdigest()[:8]}"),
                "titre":          titre,
                "description":    description,
                "entreprise":     raw.get("entreprise", {}).get("nom", "") if isinstance(raw.get("entreprise"), dict) else "",
                "localisation":   lieu_lib,
                "code_postal":    cp,
                "type_contrat":   self.normalize_contrat(raw.get("typeContrat", "")),
                "experience":     raw.get("experienceLibelle", ""),
                "competences":    self.extract_competences(texte_complet),
                "salaire_libelle": salaire_lib,
                "salaire_min":    sal_min,
                "salaire_max":    sal_max,
                "secteur":        raw.get("secteurActiviteLibelle", ""),
                "source":         raw.get("_source", "france_travail"),
                "url":            raw.get("origineOffre", {}).get("urlOrigineOffre", "") if isinstance(raw.get("origineOffre"), dict) else "",
                "date_publication": raw.get("dateCreation"),
                "_seniority":     self.detect_seniority(texte_complet),
            }
        except Exception as e:
            log.warning("Transform — erreur sur une offre : %s", e)
            return None

    def transform_batch(self, raw_offres: list[dict]) -> pd.DataFrame:
        """Transforme une liste d'offres brutes en DataFrame propre."""
        t0 = time.time()
        results = [self.transform_ft_offre(o) for o in raw_offres]
        results = [r for r in results if r is not None]

        df = pd.DataFrame(results)
        if df.empty:
            log.warning("Transform — aucune offre valide après transformation")
            return df

        # Déduplication sur id_offre
        before = len(df)
        df = df.drop_duplicates(subset=["id_offre"])
        after  = len(df)
        if before != after:
            log.info("Déduplication : %d → %d lignes (%d doublons supprimés)",
                     before, after, before - after)

        log.info("Transform terminé en %.2f s — %d offres valides", time.time() - t0, len(df))
        return df


# ══════════════════════════════════════════════════════════════════════════════
# LOAD — Chargement PostgreSQL + data lake Parquet (C1.2.2)
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Charge les données transformées vers :
      1. PostgreSQL (base relationnelle — tables offres)
      2. Data lake Parquet (C1.2.2 — solution Big Data académique)
    """

    def __init__(self, conn: psycopg2.extensions.connection) -> None:
        self.conn = conn

    def load_to_postgres(self, df: pd.DataFrame) -> tuple[int, int]:
        """
        Insère les offres en base avec stratégie UPSERT (INSERT OR UPDATE).
        Retourne (nb_insérées, nb_mises_à_jour).
        """
        if df.empty:
            return 0, 0

        inserted = updated = 0
        with self.conn.cursor() as cur:
            for _, row in df.iterrows():
                competences = row.get("competences", [])
                if isinstance(competences, str):
                    competences = []

                cur.execute("""
                    INSERT INTO offres (
                        id_offre, titre, description, entreprise,
                        localisation, code_postal, type_contrat, experience,
                        competences, salaire_libelle, salaire_min, salaire_max,
                        secteur, source, url, date_publication
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (id_offre) DO UPDATE SET
                        competences    = EXCLUDED.competences,
                        description    = EXCLUDED.description,
                        salaire_min    = EXCLUDED.salaire_min,
                        salaire_max    = EXCLUDED.salaire_max,
                        is_processed   = FALSE
                    RETURNING (xmax = 0) AS is_insert
                """, (
                    str(row.get("id_offre", ""))[:100],
                    str(row.get("titre", ""))[:500],
                    str(row.get("description", "")),
                    str(row.get("entreprise", ""))[:500],
                    str(row.get("localisation", ""))[:300],
                    str(row.get("code_postal", ""))[:20],
                    str(row.get("type_contrat", ""))[:50],
                    str(row.get("experience", ""))[:100],
                    competences,
                    str(row.get("salaire_libelle", ""))[:200],
                    _safe_int(row.get("salaire_min")),
                    _safe_int(row.get("salaire_max")),
                    #row.get("salaire_min"),
                    #row.get("salaire_max"),
                    str(row.get("secteur", ""))[:200],
                    str(row.get("source", "france_travail"))[:50],
                    str(row.get("url", "")),
                    row.get("date_publication"),
                ))
                result = cur.fetchone()
                if result and result[0]:
                    inserted += 1
                else:
                    updated += 1

        self.conn.commit()
        log.info("PostgreSQL — %d insérées, %d mises à jour", inserted, updated)
        return inserted, updated

    def load_to_data_lake(self, df: pd.DataFrame, query: str = "") -> Path:
        """
        Sauvegarde en Parquet dans le data lake (C1.2.2).
        Le format Parquet = compression columnar, efficient pour l'analyse.
        """
        if df.empty:
            log.warning("Data lake — DataFrame vide, pas de sauvegarde")
            return DATA_LAKE_PROC

        # Conversion des listes en strings pour Parquet (interop avec certains readers)
        df_save = df.copy()
        if "competences" in df_save.columns:
            df_save["competences_str"] = df_save["competences"].apply(
                lambda x: "|".join(x) if isinstance(x, list) else ""
            )

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = query.replace(" ", "_")[:30] if query else "batch"
        path = DATA_LAKE_PROC / f"offres_{slug}_{ts}.parquet"

        df_save.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        log.info("Data lake PROCESSED → %s (%.1f Ko)", path, path.stat().st_size / 1024)

        # Consolidation — merge avec le fichier maître si existe
        master = DATA_LAKE_PROC / "offres_clean.parquet"
        if master.exists():
            df_master = pd.read_parquet(master)
            df_merged = pd.concat([df_master, df_save]).drop_duplicates(subset=["id_offre"])
            df_merged.to_parquet(master, index=False, engine="pyarrow", compression="snappy")
            log.info("Fichier maître mis à jour : %d lignes total", len(df_merged))
        else:
            df_save.to_parquet(master, index=False, engine="pyarrow", compression="snappy")
            log.info("Fichier maître créé : %d lignes", len(df_save))

        return path

    def mark_as_processed(self, ids: list[str]) -> None:
        """Marque les offres comme traitées par l'ETL."""
        if not ids:
            return
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE offres SET is_processed = TRUE WHERE id_offre = ANY(%s)",
                (ids,)
            )
        self.conn.commit()

    def save_aggregated_stats(self, df: pd.DataFrame) -> None:
        """Calcule et sauvegarde les stats agrégées pour le dashboard (C2.1.2)."""
        if df.empty:
            return

        stats = {
            "date_calcul":        datetime.now(timezone.utc).isoformat(),
            "nb_offres_total":    len(df),
            "nb_sources":         df.get("source", pd.Series()).nunique() if "source" in df.columns else 0,
            "top_competences":    self._top_competences(df),
            "repartition_contrats": df["type_contrat"].value_counts().to_dict() if "type_contrat" in df.columns else {},
            "repartition_villes": df["localisation"].value_counts().head(10).to_dict() if "localisation" in df.columns else {},
        }

        stats_path = DATA_LAKE_PROC / "stats_aggregees.json"
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Stats agrégées sauvegardées → %s", stats_path)

    @staticmethod
    def _top_competences(df: pd.DataFrame, top_n: int = 20) -> dict:
        """Extrait les N compétences les plus fréquentes."""
        from collections import Counter
        counter: Counter = Counter()
        for comps in df.get("competences", []):
            if isinstance(comps, list):
                counter.update(comps)
        return dict(counter.most_common(top_n))


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATEUR
# ══════════════════════════════════════════════════════════════════════════════

class ETLPipeline:
    """
    Orchestre le pipeline Extract → Transform → Load.

    Peut être appelé depuis :
      - la ligne de commande (voir __main__)
      - n8n (via un nœud Code qui importe ce module)
      - un scheduler (cron, Airflow, etc.)
    """

    def __init__(self) -> None:
        self.transformer = DataTransformer()
        self.ft_extractor = FranceTravailExtractor()
        self.js_extractor = JobSpyExtractor()

    def run(self, query: str = "Data Scientist Lyon",
            source: str = "all",
            max_offres: int = 100,
            departement: str = "69") -> dict[str, Any]:
        """
        Exécute le pipeline complet.

        Args:
            query      : requête de recherche
            source     : 'france_travail', 'jobspy', ou 'all'
            max_offres : nombre max d'offres à collecter
            departement: filtre département France Travail (ex: "69")

        Returns:
            Dictionnaire de résultats (stats, chemins, erreurs)
        """
        t_start = time.time()
        log.info("═" * 60)
        log.info("ETL Pipeline — démarrage")
        log.info("  Requête   : %s", query)
        log.info("  Source    : %s", source)
        log.info("  Max offres: %d", max_offres)
        log.info("═" * 60)

        results: dict[str, Any] = {
            "query":       query,
            "source":      source,
            "nb_extraites": 0,
            "nb_insérées": 0,
            "nb_màj":      0,
            "erreurs":     [],
            "parquet_path": None,
        }

        # ── EXTRACT ──────────────────────────────────────────────────────────
        raw_offres: list[dict] = []

        if source in ("france_travail", "all"):
            try:
                t0 = time.time()
                ft_offres = self.ft_extractor.extract(query, departement, max_offres)
                raw_offres.extend(ft_offres)
                log.info("[EXTRACT] France Travail : %d offres en %.2f s",
                         len(ft_offres), time.time() - t0)
            except Exception as e:
                log.error("[EXTRACT] France Travail — erreur : %s", e)
                results["erreurs"].append(f"France Travail: {e}")

        if source in ("jobspy", "all"):
            try:
                t0 = time.time()
                js_offres = self.js_extractor.extract(query, max_results=min(50, max_offres))
                raw_offres.extend(js_offres)
                log.info("[EXTRACT] JobSpy : %d offres en %.2f s",
                         len(js_offres), time.time() - t0)
            except Exception as e:
                log.error("[EXTRACT] JobSpy — erreur : %s", e)
                results["erreurs"].append(f"JobSpy: {e}")

        results["nb_extraites"] = len(raw_offres)
        log.info("[EXTRACT] Total : %d offres brutes", len(raw_offres))

        if not raw_offres:
            log.warning("Aucune donnée extraite — pipeline interrompu")
            return results

        # ── TRANSFORM ────────────────────────────────────────────────────────
        log.info("[TRANSFORM] Début du nettoyage et de l'enrichissement...")
        t0 = time.time()
        df = self.transformer.transform_batch(raw_offres)
        log.info("[TRANSFORM] %d offres transformées en %.2f s", len(df), time.time() - t0)

        # ── LOAD ─────────────────────────────────────────────────────────────
        log.info("[LOAD] Chargement PostgreSQL + data lake...")

        
        
        # ── Parquet + Stats (sans PostgreSQL requis) ──────────────────────────
        log.info("[LOAD] Sauvegarde data lake Parquet...")
        try:
            _dl = DataLoader.__new__(DataLoader)
            _dl.conn = None
            parquet_path = _dl.load_to_data_lake(df, query)
            results["parquet_path"] = str(parquet_path)
            _dl.save_aggregated_stats(df)
            log.info("[LOAD] Data lake Parquet + stats OK")
        except Exception as _e:
            log.warning("[LOAD] Data lake — erreur : %s", _e)
        
       
       
        conn = None
        try:
            conn = get_db_connection()

            loader = DataLoader(conn)

            # 1. Data lake Parquet
            # t0 = time.time()
            # parquet_path = loader.load_to_data_lake(df, query)
            # results["parquet_path"] = str(parquet_path)
            # log.info("[LOAD] Data lake : %.2f s", time.time() - t0)

            # 2. PostgreSQL
            t0 = time.time()
            inserted, updated = loader.load_to_postgres(df)
            results["nb_insérées"] = inserted
            results["nb_màj"]      = updated
            log.info("[LOAD] PostgreSQL : %.2f s", time.time() - t0)

            # 3. Stats agrégées
            #loader.save_aggregated_stats(df)

            # 4. Marquer comme traité
            if "id_offre" in df.columns:
                loader.mark_as_processed(df["id_offre"].tolist())

            # 5. Log ETL en base (C1.3.3)
            log_etl_step(
                conn,
                etape="full_pipeline",
                source=source,
                nb_in=len(raw_offres),
                nb_out=inserted + updated,
                statut="success",
                message=f"query='{query}' insérées={inserted} màj={updated}",
                duree=time.time() - t_start,
            )

        except psycopg2.OperationalError as e:
            log.error("[LOAD] Impossible de se connecter à PostgreSQL : %s", e)
            log.error("Vérifiez que PostgreSQL est lancé et que .env est configuré")
            results["erreurs"].append(f"PostgreSQL connexion: {e}")
        except Exception as e:
            log.error("[LOAD] Erreur inattendue : %s", e)
            results["erreurs"].append(str(e))
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

        duree_total = time.time() - t_start
        log.info("═" * 60)
        log.info("ETL terminé en %.2f s", duree_total)
        log.info("  Extraites : %d", results["nb_extraites"])
        log.info("  Insérées  : %d", results["nb_insérées"])
        log.info("  Màj       : %d", results["nb_màj"])
        if results["erreurs"]:
            log.warning("  Erreurs   : %s", results["erreurs"])
        log.info("═" * 60)

        return results


# ── Point d'entrée CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Pipeline — ATS Intelligent")
    parser.add_argument("--query",       default="Data Scientist Lyon",
                        help="Mots-clés de recherche")
    parser.add_argument("--source",      default="all",
                        choices=["france_travail", "jobspy", "all"],
                        help="Source de données")
    parser.add_argument("--max-offres",  type=int, default=100,
                        help="Nombre max d'offres à collecter")
    parser.add_argument("--departement", default="69",
                        help="Code département (France Travail uniquement)")
    args = parser.parse_args()

    pipeline = ETLPipeline()
    result   = pipeline.run(
        query=args.query,
        source=args.source,
        max_offres=args.max_offres,
        departement=args.departement,
    )

    print("\n✅ Résultat :")
    for k, v in result.items():
        print(f"  {k} : {v}")
