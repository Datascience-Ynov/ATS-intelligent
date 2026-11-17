"""
ATS Intelligent — Interface Streamlit principale
Auteur : Mourad Amoussa | Lyon Ynov M2 IA & Data | RNCP39586

Intègre :
  - Chatbot IA Générative (Groq)
  - Recherche d'offres + analyse CV
  - Dashboard analytique (Bloc 2 — C2.1.2, C2.1.3, C2.1.4, C2.2.1)
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import hashlib
import psycopg2
from openai import OpenAI

from core import CSS, UI, Message
from core import LLMService, CVParser, JobSearchService, CompatibilityAnalyzer, MLScorer

# Chemins data lake
DATA_LAKE_PROC = Path(__file__).resolve().parent / "data_lake" / "processed"
DATA_LAKE_MODELS = Path(__file__).resolve().parent / "data_lake" / "models" / "latest"

load_dotenv()


def _get_db():
    """Connexion PostgreSQL — supporte DATABASE_URL (Supabase) ou paramètres séparés (local).
    Si DATABASE_URL contient un @ dans le mot de passe, préférer DB_HOST/DB_PASSWORD séparés.
    """
    # Essai 1 : paramètres séparés (prioritaires si DB_HOST est défini — évite pb d'encodage URL)
    try:
        db_host = st.secrets.get("DB_HOST", "")
    except Exception:
        db_host = ""
    if not db_host:
        db_host = os.getenv("DB_HOST", "")
    if db_host:
        try:
            pw = st.secrets.get("DB_PASSWORD", "")
        except Exception:
            pw = ""
        if not pw:
            pw = os.getenv("DB_PASSWORD", "postgres")
        try:
            port_val = st.secrets.get("DB_PORT", "")
        except Exception:
            port_val = ""
        if not port_val:
            port_val = os.getenv("DB_PORT", "5432")
        try:
            dbname = st.secrets.get("DB_NAME", "")
        except Exception:
            dbname = ""
        if not dbname:
            dbname = os.getenv("DB_NAME", "postgres")
        try:
            user = st.secrets.get("DB_USER", "")
        except Exception:
            user = ""
        if not user:
            user = os.getenv("DB_USER", "postgres")
        return psycopg2.connect(
            host=db_host,
            port=int(port_val),
            dbname=dbname,
            user=user,
            password=pw,
        )

    # Essai 2 : DATABASE_URL (attention aux @ dans le mot de passe → encoder %40)
    try:
        url = st.secrets.get("DATABASE_URL", "")
    except Exception:
        url = ""
    if not url:
        url = os.getenv("DATABASE_URL", "")
    if url:
        return psycopg2.connect(url)

    # Fallback local Docker
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", 15432)),
        dbname=os.getenv("DB_NAME", "ats_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )


def _extract_cv_metadata(texte: str) -> dict:
    """Extrait les métadonnées structurées d'un CV à partir du texte brut.
    Retourne un dict avec : experience_annees, formations, langages, localisation.
    """
    import re

    # ── Années d'expérience ───────────────────────────────────────────────────
    exp_annees = None
    exp_match = re.search(
        r'(\d+)\s*(?:ans?|années?|years?)\s+(?:d\'?expérience|of\s+experience)',
        texte, re.IGNORECASE
    )
    if exp_match:
        exp_annees = int(exp_match.group(1))
    else:
        # Heuristique : nombre d'emplois/stages × ~1.5 an
        nb_postes = len(re.findall(
            r'\b(?:20\d{2}|19\d{2})\s*[-–]\s*(?:20\d{2}|présent|present|aujourd|actuel)',
            texte, re.IGNORECASE
        ))
        if nb_postes > 0:
            exp_annees = nb_postes  # approximation 1 an par poste

    # ── Formations ────────────────────────────────────────────────────────────
    formations: list[str] = []
    for pat in [
        r'\b(Master\s*\d?|M\s*2|M\s*1|MBA|MSc|MS\b)',
        r'\b(Licence\s*\d?|Bachelor|BSc|L\s*3)',
        r'\b(Bac\s*\+\s*\d)',
        r'\b(Ingénieur|Grande\s+École|RNCP)',
        r'\b(BTS|DUT|BUT)',
        r'\b(Doctorat|PhD|Thèse)',
    ]:
        found = re.findall(pat, texte, re.IGNORECASE)
        formations.extend([f.strip() for f in found])
    formations = list(dict.fromkeys(formations))[:5]  # dédoublonnage, max 5

    # ── Langues ───────────────────────────────────────────────────────────────
    LANGUES_FR = {
        "Français": r'\bfrançais\b', "Anglais": r'\banglais\b',
        "Espagnol": r'\bespagnol\b', "Allemand": r'\ballemand\b',
        "Arabe": r'\barabe\b', "Mandarin": r'\bmandarin\b',
        "Italien": r'\bitalien\b', "Portugais": r'\bportugais\b',
        "English": r'\benglish\b', "French": r'\bfrench\b',
    }
    langages = [lang for lang, pat in LANGUES_FR.items() if re.search(pat, texte, re.IGNORECASE)]

    # ── Localisation ──────────────────────────────────────────────────────────
    localisation = None
    VILLES = [
        "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse", "Nantes", "Lille",
        "Strasbourg", "Nice", "Montpellier", "Rennes", "Grenoble", "Rouen",
        "Toulon", "Metz", "Nancy", "Limoges", "Brest", "Tours", "Saint-Étienne",
        "Villeurbanne", "Aix-en-Provence", "Caen", "Clermont-Ferrand",
    ]
    for ville in VILLES:
        if re.search(r'\b' + re.escape(ville) + r'\b', texte, re.IGNORECASE):
            localisation = ville
            break

    return {
        "experience_annees": exp_annees,
        "formations": formations,
        "langages":   langages,
        "localisation": localisation,
    }


def _save_cv_to_db(nom_fichier: str, texte_brut: str, competences: list) -> int | None:
    """
    Persiste le CV uploadé dans la table PostgreSQL cvs (C1.2.2, C2.3.2).
    Extrait automatiquement : experience_annees, formations, langages, localisation.
    Retourne l'id de la ligne insérée, ou None si la connexion échoue.
    """
    try:
        conn = _get_db()
        hash_cv = hashlib.sha256(texte_brut.encode()).hexdigest()
        meta = _extract_cv_metadata(texte_brut)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO cvs (nom_fichier, texte_brut, competences, hash_fichier,
                                 experience_annees, formations, langages, localisation)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (hash_fichier) DO UPDATE
                    SET date_upload       = NOW(),
                        experience_annees = EXCLUDED.experience_annees,
                        formations        = EXCLUDED.formations,
                        langages          = EXCLUDED.langages,
                        localisation      = EXCLUDED.localisation
                RETURNING id
            """, (
                nom_fichier[:500],
                texte_brut,
                competences,
                hash_cv,
                meta["experience_annees"],
                meta["formations"] or None,
                meta["langages"]   or None,
                meta["localisation"],
            ))
            cv_id = cur.fetchone()[0]
        conn.commit()
        conn.close()
        return cv_id
    except Exception as e:
        return None

def _save_scores_to_db(cv_nom: str, cv_id: int | None, scored_offers: list, features_map: dict) -> int:
    """
    Persiste TOUS les scores RF en base dans scores_matching.
    Enrichit la table avec cv_nom et offre_titre pour la lisibilité.
    features_map : dict {offer_url → features_raw dict}
    """
    try:
        conn = _get_db()
        # S'assurer que les colonnes cv_nom et offre_titre existent
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE scores_matching
                    ADD COLUMN IF NOT EXISTS cv_nom VARCHAR(255),
                    ADD COLUMN IF NOT EXISTS offre_titre VARCHAR(255);
            """)
        conn.commit()

        inserted = 0
        with conn.cursor() as cur:
            for offer in scored_offers:
                feats = features_map.get(offer.url, {})
                score_ml = offer.score / 100.0  # score normalisé 30-90 → 0.30-0.90
                # Récupérer l'id entier de l'offre si elle existe (peut être NULL)
                cur.execute(
                    "SELECT id FROM offres WHERE url_offre = %s LIMIT 1",
                    (offer.url[:500],)
                )
                row = cur.fetchone()
                offre_db_id = row[0] if row else None
                cur.execute("""
                    INSERT INTO scores_matching
                        (cv_id, offre_id, cv_nom, offre_titre, score_ml, score_percent,
                         confidence, features_json, modele_version, interpretation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                """, (
                    cv_id or 0,
                    offre_db_id,
                    cv_nom[:255],
                    offer.title[:255],
                    round(score_ml, 4),
                    offer.score,
                    round(score_ml, 4),
                    json.dumps(feats) if feats else None,
                    "v1",
                    offer.analysis[:500] if offer.analysis else None,
                ))
                inserted += 1
        conn.commit()
        conn.close()
        return inserted
    except Exception:
        return 0


def _save_offers_to_db(offers) -> int:
    """
    Persiste toutes les offres scrapées dans la table PostgreSQL `offres`.
    Utilise ON CONFLICT DO NOTHING pour éviter les doublons (idempotent).
    Retourne le nombre d'offres réellement insérées.
    """
    import hashlib
    try:
        conn = _get_db()
        inserted = 0
        with conn.cursor() as cur:
            for offer in offers:
                # id_offre = hash URL → stable, court
                id_offre = hashlib.md5(offer.url.encode()).hexdigest()[:32]
                try:
                    cur.execute(
                        """
                        INSERT INTO offres
                            (id_offre, titre, entreprise, localisation, type_contrat,
                             date_publication, url_offre, description, source_scraping)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id_offre) DO UPDATE
                            SET description = EXCLUDED.description
                            WHERE (offres.description IS NULL OR offres.description = '')
                              AND EXCLUDED.description IS NOT NULL
                              AND EXCLUDED.description <> ''
                        """,
                        (
                            id_offre,
                            offer.title[:255],
                            (offer.company or "")[:255],
                            (offer.location or "")[:255],
                            (offer.contract or "")[:100],
                            offer.date_posted or None,
                            offer.url[:500],
                            offer.body[:5000],
                            "jobspy",
                        ),
                    )
                    if cur.rowcount:
                        inserted += 1
                except Exception:
                    conn.rollback()   # ligne unique échouée → on passe
        conn.commit()
        conn.close()
        return inserted
    except Exception:
        return 0


GROQ_MODELS = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
_configured_model = os.getenv("GROQ_MODEL", "").strip()
if _configured_model and _configured_model not in GROQ_MODELS:
    GROQ_MODELS.insert(0, _configured_model)


def get_available_models(api_key: str) -> list[str]:
    """Retourne la liste des modèles Groq disponibles."""
    return list(GROQ_MODELS)


DEFAULT_SYSTEM_PROMPT = (
    "Tu es un assistant IA intelligent, utile et bienveillant. "
    "Réponds toujours en français de manière claire et structurée. "
    "Si tu ne sais pas quelque chose, dis-le franchement."
)


class App:
    """Orchestre l'application Streamlit."""

    def __init__(self) -> None:
        st.set_page_config(page_title="ATS Intelligent · IA", page_icon="✦", layout="wide")
        st.markdown(CSS, unsafe_allow_html=True)

        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            api_key = ""
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            st.error("❌ Clé API Groq manquante. Ajoutez `GROQ_API_KEY` dans les secrets Streamlit (`.streamlit/secrets.toml`) ou dans `.env`.")
            st.stop()

        self._llm        = LLMService(api_key)
        self._cv_parser  = CVParser()
        self._job_search = JobSearchService(max_results=50)
        self.available_models = get_available_models(api_key)
        self.model = os.getenv("GROQ_MODEL", self.available_models[0])
        if self.model not in self.available_models:
            self.model = self.available_models[0]

        self._init_state()
        self._render_sidebar()
        self._render_header()
        self._render_tabs()

    # ── Session state ──────────────────────────────────────────────────────
    def _init_state(self) -> None:
        for key, default in [
            ("messages", []),
            ("cv_text", ""),
            ("job_offers", []),
            ("matched_offers", []),
            ("offre_cible_chat", None),
        ]:
            st.session_state.setdefault(key, default)

    # ── Sidebar ────────────────────────────────────────────────────────────
    def _render_sidebar(self) -> None:
        with st.sidebar:
            st.markdown('<span class="sidebar-title">⚙️ Configuration</span>', unsafe_allow_html=True)
            self.model = st.selectbox(
                "Modèle",
                self.available_models,
                index=self.available_models.index(self.model) if self.model in self.available_models else 0,
            )
            self.system_prompt = st.text_area("System Prompt", value=DEFAULT_SYSTEM_PROMPT, height=130)
            st.markdown("<br>**Paramètres**", unsafe_allow_html=True)
            self.temperature = st.slider("Température", 0.0, 1.0, 0.5, 0.05)
            self.max_tokens  = st.slider("Tokens max",  256, 2048, 1024, 128)
            st.divider()
            if st.button("🗑️ Effacer la conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Propulsé par [Groq](https://console.groq.com) · Llama 3.1")

    # ── Header ─────────────────────────────────────────────────────────────
    def _render_header(self) -> None:
        model_short = self.model.split("/")[-1]
        st.markdown(f"""
        <div class="chat-header">
          <div class="logo-wrap">
            <svg width="44" height="44" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="lg1" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stop-color="#00e5c3"/>
                  <stop offset="100%" stop-color="#6366f1"/>
                </linearGradient>
                <filter id="glow"><feGaussianBlur stdDeviation="2" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
              </defs>
              <rect width="44" height="44" rx="12" fill="url(#lg1)" fill-opacity="0.15"/>
              <rect width="44" height="44" rx="12" stroke="url(#lg1)" stroke-width="1.5" fill="none"/>
              <!-- Nodes -->
              <circle cx="22" cy="10" r="3" fill="url(#lg1)" filter="url(#glow)"/>
              <circle cx="10" cy="30" r="3" fill="url(#lg1)" filter="url(#glow)"/>
              <circle cx="34" cy="30" r="3" fill="url(#lg1)" filter="url(#glow)"/>
              <circle cx="22" cy="22" r="4" fill="url(#lg1)" filter="url(#glow)"/>
              <!-- Lignes -->
              <line x1="22" y1="10" x2="10" y2="30" stroke="url(#lg1)" stroke-width="1" stroke-opacity="0.5"/>
              <line x1="22" y1="10" x2="34" y2="30" stroke="url(#lg1)" stroke-width="1" stroke-opacity="0.5"/>
              <line x1="10" y1="30" x2="34" y2="30" stroke="url(#lg1)" stroke-width="1" stroke-opacity="0.5"/>
              <line x1="22" y1="10" x2="22" y2="22" stroke="url(#lg1)" stroke-width="1.5"/>
              <line x1="22" y1="22" x2="10" y2="30" stroke="url(#lg1)" stroke-width="1.5"/>
              <line x1="22" y1="22" x2="34" y2="30" stroke="url(#lg1)" stroke-width="1.5"/>
            </svg>
            <h1>ATS Intelligent</h1>
          </div>
          <p class="caption">
              Modèle actif&nbsp;<code>{model_short}</code>
              &nbsp;·&nbsp;Température&nbsp;<code>{self.temperature}</code>
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────
    def _render_tabs(self) -> None:
        # Le chat_input DOIT être déclaré hors des tabs pour se fixer en bas
        user_input = st.chat_input("Écrivez votre message ici...")

        tab_chat, tab_jobs, tab_dashboard, tab_historique, tab_candidatures = st.tabs([
            "💬  Chat",
            "🔍  Recherche d'offres & CV",
            "📊  Dashboard & Analyse",
            "🗂️  Historique",
            "📁  Mes Candidatures",
        ])
        with tab_chat:
            self._tab_chat(user_input)
        with tab_jobs:
            self._tab_jobs()
        with tab_dashboard:
            self._tab_dashboard()
        with tab_historique:
            self._tab_historique()
        with tab_candidatures:
            self._tab_candidatures()

    # ── Tab Chat ───────────────────────────────────────────────────────────
    def _tab_chat(self, user_input: str | None) -> None:
        # ── Sélecteur d'offre cible ────────────────────────────────────────
        offres_matchees = st.session_state.get("matched_offers", [])
        if offres_matchees:
            options = [None] + offres_matchees
            idx_actuel = 0
            offre_actuelle = st.session_state.get("offre_cible_chat")
            if offre_actuelle in offres_matchees:
                idx_actuel = offres_matchees.index(offre_actuelle) + 1

            offre_choisie = st.selectbox(
                "💼 Sur quelle offre veux-tu que je t'aide ?",
                options=options,
                index=idx_actuel,
                format_func=lambda o: (
                    "— Aucune sélection —" if o is None
                    else f"{o.get('titre','?')} — {o.get('entreprise','?')} ({o.get('score_percent', 0)}%)"
                ),
                key="chat_offre_selector",
            )
            if offre_choisie is not None:
                st.session_state["offre_cible_chat"] = offre_choisie
                st.caption(
                    f"🎯 Offre sélectionnée : **{offre_choisie.get('titre')}** "
                    f"chez {offre_choisie.get('entreprise')} | Score actuel : {offre_choisie.get('score_percent', 0)}%"
                )

            # ── Upload CV amélioré + re-scoring ───────────────────────────
            offre_cible = st.session_state.get("offre_cible_chat")
            if offre_cible:
                cv_ameliore = st.file_uploader(
                    "📄 Uploader un CV amélioré pour re-scorer sur cette offre",
                    type=["pdf"],
                    key="cv_ameliore_chat",
                )
                if cv_ameliore:
                    import io
                    with st.spinner("Analyse du CV amélioré…"):
                        nouveau_texte = self._cv_parser.extract(cv_ameliore)
                        offre_temp = type("O", (), {
                            "title": offre_cible.get("titre", ""),
                            "url": offre_cible.get("url", "#"),
                            "body": offre_cible.get("description", "")[:2000],
                            "source": offre_cible.get("entreprise", ""),
                            "analysis": "",
                            "score": 0,
                            "company": offre_cible.get("entreprise", ""),
                            "location": offre_cible.get("localisation", ""),
                            "contract": offre_cible.get("type_contrat", ""),
                            "date_posted": "",
                        })()
                        analyzer = CompatibilityAnalyzer(self._llm, self.model)
                        from core.models import JobOffer as JO
                        offre_obj = JO(
                            title=offre_cible.get("titre", ""),
                            url=offre_cible.get("url", "#"),
                            body=offre_cible.get("description", "")[:2000],
                            source=offre_cible.get("entreprise", ""),
                        )
                        offre_obj = analyzer.analyze(nouveau_texte, offre_obj)
                    ancien_score = offre_cible.get("score_percent", 0)
                    nouveau_score = offre_obj.score
                    delta = nouveau_score - ancien_score
                    col_s1, col_s2 = st.columns(2)
                    col_s1.metric("Score CV original", f"{ancien_score}%")
                    col_s2.metric("Score CV amélioré", f"{nouveau_score}%", delta=f"{delta:+d}%")
                    if offre_obj.analysis:
                        st.info(offre_obj.analysis)

        st.divider()

        if not st.session_state.messages:
            UI.render_welcome()

        for msg in st.session_state.messages:
            UI.render_message(msg["role"], msg["content"])

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            UI.render_message("user", user_input)
            self._stream_response()

    def _stream_response(self) -> None:
        contexte_cv = ""
        if st.session_state.get("cv_text"):
            contexte_cv = f"\n\nCV DE L'UTILISATEUR :\n{st.session_state['cv_text'][:1500]}"

        offres = st.session_state.get("matched_offers", [])
        contexte_offres = ""
        if offres:
            offres_str = "\n".join(
                f"- {o.get('titre','?')} chez {o.get('entreprise','?')} (score: {o.get('score_percent',0)}%)"
                for o in offres[:5]
            )
            contexte_offres = f"\n\nOFFRES MATCHÉES POUR CE CV (top 5) :\n{offres_str}"

        # Contexte de l'offre spécifiquement sélectionnée dans le chat
        offre_cible = st.session_state.get("offre_cible_chat")
        contexte_offre_cible = ""
        if offre_cible:
            contexte_offre_cible = (
                f"\n\nOFFRE CIBLE SÉLECTIONNÉE PAR L'UTILISATEUR :\n"
                f"Poste : {offre_cible.get('titre', '?')}\n"
                f"Entreprise : {offre_cible.get('entreprise', '?')}\n"
                f"Localisation : {offre_cible.get('localisation', '?')}\n"
                f"Contrat : {offre_cible.get('type_contrat', '?')}\n"
                f"Score de compatibilité actuel : {offre_cible.get('score_percent', 0)}%\n"
                f"Analyse existante : {offre_cible.get('analyse', '')}\n"
                f"Description (extrait) : {str(offre_cible.get('description', ''))[:800]}"
            )

        system_prompt = (
            "Tu es un assistant ATS (Applicant Tracking System) intelligent. "
            "Tu aides les candidats à optimiser leur CV, préparer leurs entretiens, "
            "et analyser leur compatibilité avec les offres d'emploi IT. "
            "Réponds toujours en français de manière concrète et actionnable.\n"
            + self.system_prompt
            + contexte_cv
            + contexte_offres
            + contexte_offre_cible
        )
        messages = [Message("system", system_prompt)] + [
            Message(m["role"], m["content"]) for m in st.session_state.messages
        ]
        placeholder   = st.empty()
        full_response = ""
        try:
            for chunk in self._llm.stream(messages, self.model, self.temperature, self.max_tokens):
                full_response += chunk
                UI.render_streaming(placeholder, full_response, cursor=True)
            UI.render_streaming(placeholder, full_response, cursor=False)
        except Exception as e:
            full_response = f"❌ Erreur API : {e}"
            placeholder.error(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # ── Tab Offres ─────────────────────────────────────────────────────────
    def _tab_jobs(self) -> None:
        col_cv, col_search = st.columns([1, 1], gap="large")

        with col_cv:
            st.markdown("### 📄 Votre CV")
            uploaded = st.file_uploader("Uploader votre CV (PDF)", type=["pdf"])
            if uploaded:
                with st.spinner("Extraction du texte…"):
                    st.session_state.cv_text = self._cv_parser.extract(uploaded)
                st.session_state["cv_filename"] = uploaded.name  # pour scores_matching
                st.success(f"✅ CV chargé — {len(st.session_state.cv_text)} caractères extraits")
                with st.expander("Aperçu du texte extrait"):
                    st.text(st.session_state.cv_text[:1200] + "…")
                # ── Persistance PostgreSQL (C1.2.2) ─────────────────────────────
                from etl.pipeline import DataTransformer
                cv_competences = DataTransformer.extract_competences(st.session_state.cv_text)
                cv_id = _save_cv_to_db(uploaded.name, st.session_state.cv_text, cv_competences)
                if cv_id:
                    st.session_state["cv_db_id"] = cv_id
                    st.caption(f"💾 CV sauvegardé en base PostgreSQL (id={cv_id})")
            # if uploaded:
            #     with st.spinner("Extraction du texte…"):
            #         st.session_state.cv_text = self._cv_parser.extract(uploaded)
            #     st.success(f"✅ CV chargé — {len(st.session_state.cv_text)} caractères extraits")
            #     with st.expander("Aperçu du texte extrait"):
            #         st.text(st.session_state.cv_text[:1200] + "…")
            elif st.session_state.cv_text:
                st.info("✅ CV déjà chargé en session.")

        with col_search:
            st.markdown("### 🔍 Recherche d'offres")
            query    = st.text_input(
                "Intitulé du poste / mots-clés",
                placeholder="ex: Data Scientist",
                help="Mets uniquement le titre du poste ou des compétences clés. Pas besoin d'y mettre la ville ou le type de contrat.",
            )
            col_loc, col_contract, col_days = st.columns([2, 1, 1])
            with col_loc:
                location = st.text_input(
                    "Localisation",
                    value="France",
                    placeholder="ex: Paris, Lyon, Bordeaux…",
                )
            with col_contract:
                contract_filter = st.selectbox(
                    "Contrat",
                    options=["Tous", "CDI", "Stage", "Alternance", "CDD", "Freelance"],
                )
            with col_days:
                days_old = st.selectbox(
                    "Ancienneté max",
                    options=[7, 14, 30],
                    format_func=lambda d: f"{d} jours",
                    index=1,
                )
            max_results_ui = st.slider(
                "Nombre d'offres à scraper",
                min_value=5, max_value=100, value=20, step=5,
                help="Plus d'offres = analyse plus riche, mais plus lente (~2s par tranche de 5)",
            )
            analyze = st.checkbox(
                "Analyser la compatibilité avec mon CV",
                disabled=not st.session_state.cv_text,
                help="Uploadez d'abord votre CV pour activer cette option",
            )
            search_btn = st.button("🔎 Rechercher", use_container_width=True)

        st.divider()

        if search_btn and query:
            # Enrichir le terme de recherche avec le type de contrat si sélectionné
            search_term = query.strip()
            if contract_filter != "Tous":
                search_term = f"{search_term} {contract_filter}"
            hours_old = days_old * 24
            with st.spinner(f"Recherche de '{search_term}' à {location} (≤{days_old}j, max {max_results_ui} offres)…"):
                try:
                    offers = self._job_search.search(search_term, location=location, hours_old=hours_old, max_results=max_results_ui)
                except Exception as e:
                    st.error(f"⚠️ Erreur lors de la recherche : {e}")
                    return

            if not offers:
                st.warning("Aucune offre trouvée. Essaie d'autres mots-clés ou élargis la période.")
                return

            # ── 1. Sauvegarde TOUTES les offres en PostgreSQL ───────────────────
            nb_inserted = _save_offers_to_db(offers)
            if nb_inserted > 0:
                st.caption(f"💾 {nb_inserted} nouvelles offres sauvegardées en base.")

            # ── 2. Scoring RandomForest pour toutes les offres ──────────────────
            if st.session_state.cv_text:
                from etl.pipeline import DataTransformer
                scorer = MLScorer(DATA_LAKE_MODELS)

                # Extraire compétences + métadonnées CV
                cv_skills   = DataTransformer.extract_competences(st.session_state.cv_text)
                cv_meta     = _extract_cv_metadata(st.session_state.cv_text)
                cv_exp      = float(cv_meta.get("experience_annees") or 0)
                cv_loc      = cv_meta.get("localisation") or ""

                if scorer.available:
                    score_src = "🤖 RandomForest"
                else:
                    score_src = "📐 Heuristique"

                # features_map : url → dict de features (pour scores_matching)
                features_map: dict = {}

                rf_bar = st.progress(0, text=f"Scoring {score_src} ({len(offers)} offres)…")
                for i, offer in enumerate(offers):
                    offre_skills = DataTransformer.extract_competences(offer.body)
                    cv_set   = {s.lower() for s in cv_skills if s}
                    off_set  = {s.lower() for s in offre_skills if s}
                    inter    = cv_set & off_set
                    union    = cv_set | off_set
                    jaccard  = len(inter) / len(union) if union else 0.0
                    sk_ratio = len(inter) / len(off_set) if off_set else 0.0
                    rf_score = scorer.score_offer(
                        cv_skills=cv_skills,
                        cv_exp=cv_exp,
                        cv_loc=cv_loc,
                        offre_skills=offre_skills,
                        offre_loc=offer.location or "",
                        offre_exp_min=0.0,
                    )
                    offer.score = int(round(rf_score * 100))
                    # Sauvegarder les features brutes pour scores_matching
                    features_map[offer.url] = {
                        "skill_overlap":  round(jaccard, 4),
                        "skill_ratio":    round(sk_ratio, 4),
                        "cv_nb_skills":   len(cv_set),
                        "offre_nb_skills": len(off_set),
                        "skills_communs": sorted(list(inter))[:10],
                        "cv_localisation": cv_loc,
                        "offre_localisation": offer.location or "",
                        "cv_experience":  cv_exp,
                        "source": score_src,
                    }
                    rf_bar.progress((i + 1) / len(offers),
                                    text=f"Scoring {score_src} {i+1}/{len(offers)}…")
                rf_bar.empty()

                # ── Normalisation des scores RF → plage lisible [30–90] ─────────
                raw_scores = [o.score for o in offers]
                min_s, max_s = min(raw_scores), max(raw_scores)
                if max_s > min_s:
                    for o in offers:
                        o.score = int(30 + (o.score - min_s) / (max_s - min_s) * 60)
                else:
                    for o in offers:
                        o.score = 55

                # Trier par score normalisé décroissant
                offers.sort(key=lambda o: o.score, reverse=True)

                # ── 3. Enrichissement qualitatif IA — top 10 par score RF ──────
                if analyze:
                    # Le RF score TOUTES les offres ; le LLM enrichit les top 10
                    # (limite VRAM : le modèle local ne peut traiter trop de requêtes)
                    MAX_LLM = 10
                    top_ia = offers[:MAX_LLM]
                    n_ia = len(top_ia)
                    analyzer = CompatibilityAnalyzer(self._llm, self.model)
                    st.caption(
                        f"✅ Scoring RF terminé sur {len(offers)} offres — "
                        f"enrichissement qualitatif IA sur le top {n_ia}…"
                    )
                    progress = st.progress(0, text=f"Analyse qualitative 0/{n_ia}…")
                    n_ok = 0
                    for i, offer in enumerate(top_ia):
                        progress.progress((i + 1) / n_ia,
                                          text=f"Analyse qualitative {i+1}/{n_ia}…")
                        try:
                            enriched = analyzer.analyze(st.session_state.cv_text, offer)
                            offer.analysis = enriched.analysis
                            n_ok += 1
                        except Exception as _llm_err:
                            # OOM ou erreur réseau : on continue sans crasher
                            offer.analysis = None
                            st.warning(
                                f"⚠️ Analyse IA ignorée pour « {offer.title[:60]} » "
                                f"(VRAM insuffisante ou erreur réseau). "
                                f"Le score RF reste valide.",
                                icon="⚠️",
                            )
                    progress.empty()
                    if n_ok < n_ia:
                        st.caption(
                            f"💡 {n_ok}/{n_ia} analyses qualitatives réussies. "
                            f"Vérifiez votre quota Groq ou réessayez."
                        )

                # ── 4. Sauvegarder TOUS les scores dans scores_matching ───────────
                cv_nom = st.session_state.get("cv_filename", "cv_inconnu")
                cv_id  = st.session_state.get("cv_db_id")
                nb_scores = _save_scores_to_db(cv_nom, cv_id, offers, features_map)
                if nb_scores > 0:
                    st.caption(f"💾 {nb_scores} scores sauvegardés dans scores_matching.")

            st.session_state.job_offers = offers
            # Stocker les offres matchées sous forme de dicts pour candidatures + chat
            st.session_state.matched_offers = [
                {
                    "titre": o.title,
                    "entreprise": o.company or o.source.split(" — ")[0],
                    "localisation": o.location,
                    "type_contrat": o.contract,
                    "url": o.url,
                    "description": o.body,
                    "score_percent": o.score,
                    "analyse": o.analysis,
                    "date_posted": o.date_posted,
                }
                for o in offers
            ]

        if st.session_state.job_offers:
            n = len(st.session_state.job_offers)
            scored_rf  = sum(1 for o in st.session_state.job_offers if o.score > 0)
            scored_ia  = sum(1 for o in st.session_state.job_offers if o.analysis)
            if scored_rf:
                label = f"— {scored_rf} scorées par notre modèle ML"
                if scored_ia:
                    label += f", dont {scored_ia} avec analyse qualitative (top {scored_ia})"
            else:
                label = ""
            st.markdown(f"**{n} offre(s) trouvée(s)** {label}")
            for offer in sorted(st.session_state.job_offers, key=lambda o: o.score, reverse=True):
                UI.render_job_card(offer)


    # ── Tab Dashboard & Analyse ────────────────────────────────────────────
    def _tab_dashboard(self) -> None:
        """
        Dashboard candidat — Couvre C2.1.2, C2.1.3, C2.1.4, C2.2.1
        Orienté utilisateur : analyse du marché + gap compétences + recommandations.
        """
        import altair as alt

        st.markdown("## 📊 Mon Tableau de Bord Candidat")
        st.caption("Analyse des offres collectées · Recommandations personnalisées | RNCP39586 — Bloc 2")

        parquet_path = DATA_LAKE_PROC / "offres_clean.parquet"

        # ── Bouton ETL manuel ─────────────────────────────────────────────────
        with st.expander("🔄 Lancer / rafraîchir le pipeline ETL", expanded=not parquet_path.exists()):
            col_q, col_src, col_n = st.columns([3, 1, 1])
            with col_q:
                etl_query = st.text_input("Requête", value="Data Scientist", key="etl_query")
            with col_src:
                etl_source = st.selectbox("Source", ["jobspy", "france_travail", "all"], key="etl_source")
            with col_n:
                etl_max = st.number_input("Max offres", min_value=10, max_value=200, value=50, step=10, key="etl_max")
            if st.button("▶️ Lancer le pipeline ETL", use_container_width=True, key="etl_run_btn"):
                with st.spinner("Pipeline ETL en cours — collecte et transformation des offres…"):
                    try:
                        from etl.pipeline import ETLPipeline
                        result = ETLPipeline().run(query=etl_query, source=etl_source, max_offres=int(etl_max))
                        st.success(
                            f"✅ ETL terminé — {result.get('nb_extraites', 0)} offres extraites, "
                            f"{result.get('nb_insérées', 0)} insérées, {result.get('nb_màj', 0)} mises à jour."
                        )
                        if result.get("erreurs"):
                            st.warning("Avertissements : " + " | ".join(str(e) for e in result["erreurs"][:3]))
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Erreur ETL : {exc}")

        if not parquet_path.exists():
            st.info("📂 Aucune donnée disponible — lancez le pipeline ETL ci-dessus.")
            self._show_demo_dashboard()
            return

        df = pd.read_parquet(parquet_path)
        if df.empty:
            st.warning("Le fichier Parquet est vide — relancez le pipeline ETL.")
            return

        # Nettoyage compétences
        if "competences" not in df.columns and "competences_str" in df.columns:
            df["competences"] = df["competences_str"].apply(
                lambda x: x.split("|") if isinstance(x, str) and x else []
            )
        if "competences" not in df.columns:
            df["competences"] = [[] for _ in range(len(df))]

        # Toutes les compétences — sans vides ni doublons parasites
        all_comps = []
        for comps in df["competences"]:
            if isinstance(comps, list):
                all_comps.extend([c.strip() for c in comps if isinstance(c, str) and c.strip()])

        comp_series = pd.Series(all_comps).value_counts() if all_comps else pd.Series(dtype=int)

        # Fallback 1 : re-extraire les compétences depuis les colonnes texte du parquet
        if comp_series.empty and not df.empty:
            from etl.pipeline import DataTransformer
            _desc_col = next((c for c in ("description", "texte_complet", "titre") if c in df.columns), None)
            if _desc_col:
                _fallback_comps: list[str] = []
                for _txt in df[_desc_col].dropna():
                    _comps = DataTransformer.extract_competences(str(_txt))
                    _fallback_comps.extend([c.strip() for c in _comps if isinstance(c, str) and c.strip()])
                if _fallback_comps:
                    comp_series = pd.Series(_fallback_comps).value_counts()
                    all_comps = _fallback_comps
                    st.caption("ℹ️ Compétences re-extraites depuis les descriptions (colonne `competences` du parquet vide)")

        # Fallback 2 : si parquet n'a pas de compétences extraites, on utilise les offres de la session
        if comp_series.empty and st.session_state.get("job_offers"):
            from etl.pipeline import DataTransformer
            _fallback_comps2: list[str] = []
            for _offer in st.session_state.job_offers:
                _comps = DataTransformer.extract_competences(_offer.body)
                _fallback_comps2.extend([c.strip() for c in _comps if isinstance(c, str) and c.strip()])
            if _fallback_comps2:
                comp_series = pd.Series(_fallback_comps2).value_counts()
                all_comps = _fallback_comps2
                st.caption("ℹ️ Compétences issues de la session de recherche (parquet non peuplé par ETL)")

        # ── KPIs (C2.1.2) ────────────────────────────────────────────────────
        st.markdown("### 🔢 Indicateurs clés")
        col1, col2, col3, col4 = st.columns(4)
        total_offres  = len(df)
        n_entreprises = df["entreprise"].nunique() if "entreprise" in df.columns else 0
        n_villes      = df["localisation"].nunique() if "localisation" in df.columns else 0
        top_comp      = comp_series.index[0] if len(comp_series) > 0 else "N/A"

        with col1: st.metric("📋 Offres collectées", f"{total_offres:,}")
        with col2: st.metric("🏢 Entreprises uniques", f"{n_entreprises:,}")
        with col3: st.metric("📍 Villes représentées", f"{n_villes:,}")
        with col4: st.metric("🏆 Compétence #1", top_comp)

        st.divider()

        # ── Visualisations (C2.1.3) ───────────────────────────────────────────
        st.markdown("### 📈 Analyse du marché")
        col_left, col_right = st.columns(2)

        # Répartition contrats — filtrage des valeurs non pertinentes (NAN, NONE, LIB…)
        with col_left:
            st.markdown("#### Répartition des contrats")
            if "type_contrat" in df.columns:
                LABEL_MAP = {
                    "fulltime": "CDI", "full_time": "CDI", "Full-time": "CDI",
                    "contract": "CDD", "part_time": "Temps partiel",
                    "internship": "Stage", "temporary": "CDD",
                }
                VALID_CONTRATS = {"CDI", "CDD", "Stage", "Alternance", "Freelance", "Temps partiel"}
                mapped = df["type_contrat"].map(
                    lambda x: LABEL_MAP.get(str(x), str(x)) if pd.notna(x) else None
                )
                mapped_valid = mapped[mapped.isin(VALID_CONTRATS)]
                if not mapped_valid.empty:
                    counts = mapped_valid.value_counts().reset_index()
                    counts.columns = ["Contrat", "Offres"]
                    chart = (
                        alt.Chart(counts)
                        .mark_bar(color="#4ef2d2")
                        .encode(
                            x=alt.X("Contrat:N", title="", axis=alt.Axis(labelAngle=0, labelColor="#d1dce8", titleColor="#d1dce8")),
                            y=alt.Y("Offres:Q", title="Nombre d'offres", axis=alt.Axis(labelColor="#d1dce8", titleColor="#d1dce8")),
                            tooltip=["Contrat", "Offres"],
                        )
                        .properties(height=260, background="transparent")
                        .configure_view(strokeWidth=0)
                        .configure_axis(gridColor="#2a3a4a")
                    )
                    st.altair_chart(chart, use_container_width=True)
                    with st.expander("📋 Requête SQL équivalente"):
                        st.code("""
SELECT type_contrat, COUNT(*) AS nb_offres
FROM offres
WHERE type_contrat IN ('CDI','CDD','Stage','Alternance','Freelance')
GROUP BY type_contrat ORDER BY nb_offres DESC;
""", language="sql")
                else:
                    st.info("Aucune donnée de contrat exploitable pour le moment.")

        # Top 10 compétences — barres horizontales pour une meilleure lisibilité
        with col_right:
            st.markdown("#### Top 10 compétences demandées")
            if len(comp_series) > 0:
                top10 = comp_series.head(10).reset_index()
                top10.columns = ["Compétence", "Occurrences"]
                chart = (
                    alt.Chart(top10)
                    .mark_bar(color="#4ef2d2")
                    .encode(
                        x=alt.X("Occurrences:Q", title="Nombre d'offres", axis=alt.Axis(labelColor="#d1dce8", titleColor="#d1dce8")),
                        y=alt.Y("Compétence:N", sort="-x", title="", axis=alt.Axis(labelColor="#d1dce8")),
                        tooltip=["Compétence", "Occurrences"],
                    )
                    .properties(height=260, background="transparent")
                    .configure_view(strokeWidth=0)
                    .configure_axis(gridColor="#2a3a4a")
                )
                st.altair_chart(chart, use_container_width=True)
                with st.expander("📋 Requête SQL équivalente"):
                    st.code("""
SELECT comp AS competence, COUNT(*) AS frequence
FROM offres, UNNEST(competences) AS comp
WHERE comp != ''
GROUP BY comp ORDER BY frequence DESC LIMIT 10;
""", language="sql")
            else:
                st.info("Aucune compétence extraite. Relancez le pipeline ETL.")

        st.divider()

        # ── Test statistique Mann-Whitney (C2.1.4) — uniquement si assez de données ──
        if "type_contrat" in df.columns and "competences" in df.columns:
            df["n_competences"] = df["competences"].apply(
                lambda x: len([c for c in x if isinstance(c, str) and c.strip()]) if isinstance(x, list) else 0
            )
            cdi_scores = df[df["type_contrat"] == "CDI"]["n_competences"].dropna()
            alt_scores = df[df["type_contrat"].isin(["Alternance", "Stage"])]["n_competences"].dropna()

            if len(cdi_scores) >= 5 and len(alt_scores) >= 5:
                st.markdown("### 🔬 Test statistique — Mann-Whitney (C2.1.4)")
                st.caption("H₀ : La distribution des compétences requises est identique entre CDI et Alternance")
                try:
                    from scipy import stats as scipy_stats
                    stat, p_value = scipy_stats.mannwhitneyu(cdi_scores, alt_scores, alternative="two-sided")
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1: st.metric("Statistic U", f"{stat:.1f}")
                    with col_s2: st.metric("p-value", f"{p_value:.4f}")
                    with col_s3:
                        st.metric("Conclusion (α=0.05)", "H₀ rejetée ✅" if p_value < 0.05 else "H₀ non rejetée")
                    if p_value < 0.05:
                        st.success(
                            f"**Résultat significatif** (p={p_value:.4f}) : "
                            f"Les offres CDI requièrent en moyenne {cdi_scores.mean():.1f} compétences "
                            f"vs {alt_scores.mean():.1f} pour les Alternances."
                        )
                    else:
                        st.info(f"**Résultat non significatif** (p={p_value:.4f} ≥ 0.05).")
                    with st.expander("📋 Code Python du test"):
                        st.code(
                            "from scipy import stats\n"
                            "stat, p = stats.mannwhitneyu(cdi, alt, alternative='two-sided')",
                            language="python"
                        )
                except ImportError:
                    st.warning("scipy non installé.")
                st.divider()

        # ── Analyse personnalisée : Gap de compétences (C2.2.1) ──────────────
        st.markdown("### 🎯 Vos compétences face au marché")
        cv_text = st.session_state.get("cv_text", "")
        if not cv_text:
            st.info(
                "📄 **Uploadez votre CV** dans l'onglet *Recherche d'offres & CV* "
                "pour obtenir une analyse personnalisée de vos compétences."
            )
        elif len(comp_series) > 0:
            from etl.pipeline import DataTransformer
            cv_comps_raw = DataTransformer.extract_competences(cv_text)
            cv_comps_lower = {c.strip().lower() for c in cv_comps_raw if c.strip()}
            top20 = comp_series.head(20)
            market_lower_map = {c.lower(): c for c in top20.index}

            present = [(market_lower_map[c], top20[market_lower_map[c]])
                       for c in cv_comps_lower if c in market_lower_map]
            missing = [(comp, top20[comp])
                       for comp in top20.index if comp.lower() not in cv_comps_lower]

            RESSOURCES = {
                "Python": ("Python.org", "https://www.learnpython.org/fr/"),
                "SQL": ("W3Schools SQL", "https://www.w3schools.com/sql/"),
                "Machine Learning": ("Coursera ML", "https://www.coursera.org/learn/machine-learning"),
                "Docker": ("Docs officielles", "https://docs.docker.com/get-started/"),
                "AWS": ("AWS Training", "https://aws.amazon.com/fr/training/"),
                "Azure": ("Microsoft Learn", "https://learn.microsoft.com/fr-fr/training/azure/"),
                "GCP": ("Google Cloud", "https://cloud.google.com/training"),
                "Git": ("Git Book FR", "https://git-scm.com/book/fr/v2"),
                "Pandas": ("Pandas Guide", "https://pandas.pydata.org/docs/getting_started/"),
                "FastAPI": ("FastAPI Tutorial", "https://fastapi.tiangolo.com/tutorial/"),
                "Deep Learning": ("DeepLearning.AI", "https://www.deeplearning.ai/"),
                "Spark": ("Apache Spark", "https://spark.apache.org/docs/latest/quick-start.html"),
                "TensorFlow": ("TensorFlow Learn", "https://www.tensorflow.org/learn"),
                "PyTorch": ("PyTorch Tutorials", "https://pytorch.org/tutorials/"),
                "Scikit-learn": ("Sklearn docs", "https://scikit-learn.org/stable/tutorial/"),
                "Kubernetes": ("K8s tutorial", "https://kubernetes.io/fr/docs/tutorials/"),
                "Airflow": ("Airflow docs", "https://airflow.apache.org/docs/apache-airflow/stable/tutorial/"),
                "Power BI": ("Microsoft Learn", "https://learn.microsoft.com/fr-fr/power-bi/"),
                "Tableau": ("Tableau Training", "https://www.tableau.com/learn/training"),
                "NLP": ("NLTK Book", "https://www.nltk.org/book/"),
                "Java": ("dev.java", "https://dev.java/learn/"),
                "JavaScript": ("javascript.info", "https://fr.javascript.info/"),
                "TypeScript": ("TS Handbook", "https://www.typescriptlang.org/docs/handbook/"),
                "React": ("React docs", "https://react.dev/learn"),
                "Node.js": ("Node.js learn", "https://nodejs.dev/en/learn/"),
                "MLflow": ("MLflow docs", "https://mlflow.org/docs/latest/tutorials-and-examples/"),
                "PostgreSQL": ("PostgreSQL tuto", "https://www.postgresql.org/docs/current/tutorial.html"),
                "MongoDB": ("MongoDB Learn", "https://learn.mongodb.com/"),
                "Elasticsearch": ("Elastic docs", "https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started.html"),
                "Scala": ("Scala docs", "https://docs.scala-lang.org/getting-started/"),
                "R": ("R intro", "https://cran.r-project.org/doc/manuals/r-release/R-intro.html"),
            }

            col_pres, col_miss = st.columns(2)
            with col_pres:
                st.markdown(f"**✅ Maîtrisées** ({len(present)} / {len(top20)} top marché)")
                if present:
                    for comp, freq in sorted(present, key=lambda x: -x[1]):
                        st.markdown(f"- **{comp}** — dans {freq} offres")
                else:
                    st.info("Aucune compétence reconnue dans le top marché. Pensez à détailler les technologies dans votre CV.")
            with col_miss:
                st.markdown(f"**📚 À acquérir** (top priorités marché)")
                for comp, freq in missing[:8]:
                    res = RESSOURCES.get(comp)
                    if res:
                        platform, url = res
                        st.markdown(f"- **{comp}** ({freq} offres) → [Apprendre : {platform}]({url})")
                    else:
                        q = comp.replace(" ", "+")
                        st.markdown(f"- **{comp}** ({freq} offres) → [Chercher sur YouTube](https://www.youtube.com/results?search_query=apprendre+{q})")
        else:
            st.info("Relancez le pipeline ETL pour collecter des offres et générer votre analyse.")

        st.divider()

        # ── Insights Marché IT (C2.2.1) ──────────────────────────────────────
        st.markdown("### 💡 Insights Marché IT")
        if all_comps:
            total = len(df)
            insights = []
            if len(comp_series) >= 3:
                t = comp_series.head(3)
                insights.append(
                    f"🏆 **Trio dominant** : **{t.index[0]}** ({t.iloc[0]} offres), "
                    f"**{t.index[1]}** ({t.iloc[1]}) et **{t.index[2]}** ({t.iloc[2]})"
                )
            python_count = sum(1 for c in df["competences"] if isinstance(c, list) and any(x.strip() == "Python" for x in c))
            if python_count > 0:
                insights.append(f"🐍 **Python** est demandé dans **{python_count/total:.0%} des offres** — incontournable")
            cloud_count = sum(1 for c in df["competences"] if isinstance(c, list) and any(x.strip() in ("AWS", "Azure", "GCP") for x in c))
            if cloud_count > 0:
                insights.append(f"☁️ **Cloud** (AWS/Azure/GCP) présent dans **{cloud_count/total:.0%} des offres** — compétence différenciante")
            if "type_contrat" in df.columns:
                cdi_pct = (df["type_contrat"] == "CDI").mean()
                if cdi_pct > 0:
                    insights.append(f"📋 **{cdi_pct:.0%} des offres en CDI** — marché de l'emploi IT stable")
            if "source" in df.columns:
                n_src = df["source"].nunique()
                if n_src > 1:
                    insights.append(f"🌐 Données issues de **{n_src} sources** (LinkedIn, Indeed, France Travail…)")
            for ins in insights:
                st.markdown(f"- {ins}")
            if not insights:
                st.info("Collectez plus d'offres pour générer des insights significatifs.")
        else:
            st.info("Pas encore de données marché. Lancez le pipeline ETL.")

        st.divider()

        # ── Modèle ML (info) ──────────────────────────────────────────────────
        st.markdown("### 🤖 Modèle ML de matching")
        meta_path = DATA_LAKE_MODELS / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            perfs = meta.get("performances", {})
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1: st.metric("Modèle", meta.get("modele_nom", "?"))
            with col_m2:
                auc = perfs.get("test_roc_auc")
                st.metric("ROC-AUC (test)", f"{auc:.3f}" if auc else "N/A")
            with col_m3:
                f1 = perfs.get("test_f1")
                st.metric("F1-score", f"{f1:.3f}" if f1 else "N/A")
            with col_m4: st.metric("Features", meta.get("n_features", "?"))
            st.caption(
                f"Version : {meta.get('version','?')} | "
                f"Créé le : {meta.get('date_creation','?')[:10]} | "
                f"Projet : {meta.get('contexte',{}).get('projet','')}"
            )
        else:
            st.info("Modèle non encore entraîné.\n\nLancez : `python -m ml.train`")

    def _tab_candidatures(self) -> None:
        st.markdown("## 📁 Mes Candidatures")

        cv_id = st.session_state.get("cv_db_id")

        # Connexion DB (nécessaire même pour récupérer cv_id après redémarrage)
        try:
            conn = _get_db()
        except Exception as e:
            st.error(f"Connexion DB impossible : {e}")
            st.info("Vérifiez que DATABASE_URL est configuré dans les secrets Streamlit ou dans .env")
            return

        # Après un redémarrage Streamlit, la session_state est vide — on récupère le CV le plus récent
        if not cv_id:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, nom_fichier FROM cvs ORDER BY date_upload DESC LIMIT 1")
                    row = cur.fetchone()
                    if row:
                        cv_id = row[0]
                        st.session_state["cv_db_id"] = cv_id
                        st.caption(f"ℹ️ Session restaurée — CV détecté en base : **{row[1]}** (id={cv_id})")
            except Exception:
                pass

        if not cv_id:
            st.info("⬆️ Commence par uploader ton CV dans l'onglet **Recherche & Offres** pour activer le suivi de candidatures.")
            conn.close()
            return

        # ── Ajouter une candidature ───────────────────────────────────────
        st.markdown("### ➕ Postuler à une offre")

        # Priorité : offres de la session de recherche (avec scores LLM)
        offres_session = st.session_state.get("matched_offers", [])

        if offres_session:
            st.info(f"💡 {len(offres_session)} offres issues de ta dernière recherche disponibles.")
            offre_labels_sess = {
                i: f"{o.get('titre','?')} — {o.get('entreprise','?')} ({o.get('localisation','?')}) | {o.get('type_contrat','?')} | Score: {o.get('score_percent',0)}%"
                for i, o in enumerate(offres_session)
            }
            selected_idx = st.selectbox(
                "Choisir une offre (depuis ta recherche)",
                options=list(offre_labels_sess.keys()),
                format_func=lambda x: offre_labels_sess[x],
                key="cand_offre_select",
            )
            offre_selectionnee = offres_session[selected_idx]
            conseils_auto = offre_selectionnee.get("analyse", "")

            lettre = st.text_area(
                "✉️ Lettre de motivation",
                placeholder="Rédigez ou collez votre lettre de motivation ici…",
                height=150,
            )
            notes = st.text_area(
                "📝 Notes personnelles",
                placeholder="Points à préparer, questions à poser, contacts…",
                height=80,
            )
            if conseils_auto:
                with st.expander("💡 Conseils IA générés pour cette offre"):
                    st.write(conseils_auto)

            if st.button("📩 Enregistrer la candidature"):
                try:
                    # Essayer d'insérer l'offre dans la DB si pas encore présente
                    offre_db_id = None
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT id FROM offres WHERE url = %s LIMIT 1",
                            (offre_selectionnee.get("url", ""),)
                        )
                        row_offre = cur.fetchone()
                        if row_offre:
                            offre_db_id = row_offre[0]
                        else:
                            cur.execute("""
                                INSERT INTO offres (id_offre, titre, entreprise, localisation,
                                    type_contrat, url, description, source)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, 'jobspy')
                                ON CONFLICT (id_offre) DO NOTHING RETURNING id
                            """, (
                                f"sess_{hash(offre_selectionnee.get('url',''))}",
                                offre_selectionnee.get("titre", "")[:500],
                                offre_selectionnee.get("entreprise", "")[:500],
                                offre_selectionnee.get("localisation", "")[:300],
                                offre_selectionnee.get("type_contrat", "")[:50],
                                offre_selectionnee.get("url", ""),
                                offre_selectionnee.get("description", "")[:3000],
                            ))
                            row_new = cur.fetchone()
                            if row_new:
                                offre_db_id = row_new[0]
                    conn.commit()

                    if offre_db_id:
                        score_val = offre_selectionnee.get("score_percent", 0) / 100.0

                        # ── Score ML + features (pour scores_matching + retrain) ──────
                        ml_score    = None
                        ml_conf     = None
                        features_js = None
                        try:
                            from ml.dataset import compute_features
                            from etl.pipeline import DataTransformer
                            cv_txt_feat = st.session_state.get("cv_text", "")
                            cv_comps    = DataTransformer.extract_competences(cv_txt_feat)
                            offre_comps = DataTransformer.extract_competences(
                                offre_selectionnee.get("description", "")
                            )
                            cv_dict  = {
                                "competences": cv_comps,
                                "localisation": "",
                                "teletravail":  False,
                                "salaire_min":  0,
                                "experience_annees": 0,
                            }
                            offre_dict = {
                                "competences": offre_comps,
                                "localisation": offre_selectionnee.get("localisation", ""),
                                "teletravail":  False,
                                "salaire_max":  0,
                                "exp_min":      0,
                            }
                            import numpy as _np
                            feats = compute_features(cv_dict, offre_dict, _np.random.default_rng(42))
                            model_path = DATA_LAKE_MODELS / "model.joblib"
                            if model_path.exists():
                                import joblib as _jl
                                _clf = _jl.load(model_path)
                                _feat_names = feats.get("feature_names", list(feats.keys()))
                                _X = [[feats.get(f, 0) for f in _feat_names]]
                                ml_score = float(_clf.predict_proba(_X)[0][1])
                                ml_conf  = round(ml_score, 4)
                            features_js = json.dumps({k: v for k, v in feats.items()
                                                       if isinstance(v, (int, float, bool))})
                        except Exception:
                            pass  # ML facultatif

                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO candidatures
                                    (cv_id, offre_id, statut, notes, score_matching,
                                     lettre_motivation, nom_cv, conseils_ia)
                                VALUES (%s, %s, 'nouveau', %s, %s, %s, %s, %s)
                                RETURNING id
                            """, (
                                cv_id,
                                offre_db_id,
                                notes or None,
                                score_val,
                                lettre or None,
                                st.session_state.get("cv_db_id", ""),
                                conseils_auto or None,
                            ))
                            cand_new_id = cur.fetchone()

                            # Peuplement scores_matching avec features ML réelles
                            if cand_new_id and features_js is not None:
                                effective_score = ml_score if ml_score is not None else score_val
                                cur.execute("""
                                    INSERT INTO scores_matching
                                        (cv_id, offre_id, score_ml, score_percent,
                                         confidence, features_json, modele_version)
                                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, 'v1')
                                    ON CONFLICT DO NOTHING
                                """, (
                                    cv_id,
                                    offre_db_id,
                                    effective_score,
                                    int(effective_score * 100),
                                    ml_conf or round(score_val, 4),
                                    features_js,
                                ))
                        conn.commit()
                        st.success(f"✅ Candidature enregistrée — {offre_selectionnee.get('titre')} chez {offre_selectionnee.get('entreprise')} !")
                    else:
                        st.warning("L'offre n'a pas pu être enregistrée en base.")
                except Exception as e:
                    st.error(f"Erreur : {e}")
        else:
            st.info("🔍 Lance d'abord une recherche dans l'onglet **Recherche & Offres** pour voir les offres ici.")

        st.divider()

        # ── Pipeline de candidatures ──────────────────────────────────────
        st.markdown("### 📋 Suivi de mes candidatures")
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.id, o.titre, o.entreprise, o.type_contrat,
                           c.statut,
                           COALESCE(ROUND(c.score_matching * 100)::int, 0) AS score_pct,
                           c.notes, c.date_candidature::date,
                           c.lettre_motivation, c.conseils_ia
                    FROM candidatures c
                    JOIN offres o ON c.offre_id = o.id
                    WHERE c.cv_id = %s
                    ORDER BY c.date_candidature DESC
                """, (cv_id,))
                cands = cur.fetchall()
        except Exception:
            # Fallback si colonnes lettre_motivation / conseils_ia pas encore créées
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.id, o.titre, o.entreprise, o.type_contrat,
                           c.statut,
                           COALESCE(ROUND(c.score_matching * 100)::int, 0) AS score_pct,
                           c.notes, c.date_candidature::date,
                           NULL, NULL
                    FROM candidatures c
                    JOIN offres o ON c.offre_id = o.id
                    WHERE c.cv_id = %s
                    ORDER BY c.date_candidature DESC
                """, (cv_id,))
                cands = cur.fetchall()

        if not cands:
            st.info("Aucune candidature enregistrée pour ce CV.")
        else:
            STATUTS = ['nouveau', 'en_cours', 'entretien', 'test_technique', 'offre', 'refus', 'accepte']
            STATUT_EMOJI = {
                'nouveau': '🆕', 'en_cours': '⏳', 'entretien': '🤝',
                'test_technique': '💻', 'offre': '🎉', 'refus': '❌', 'accepte': '✅'
            }
            st.caption("💡 Vue rapide — pour consulter la lettre de motivation et les conseils IA, ouvrez l'onglet **🗂️ Historique**.")
            for row in cands:
                cand_id, titre, entreprise, contrat, statut, score_pct, notes, date, _lettre, _conseils = row
                col1, col2, col3 = st.columns([4, 1, 2])
                with col1:
                    st.markdown(f"**{titre}** — {entreprise}")
                    st.caption(f"`{contrat or '—'}` · 📅 {date}{' · 📝 ' + notes if notes else ''}")
                with col2:
                    score_color = "🟢" if (score_pct or 0) >= 70 else "🟡" if (score_pct or 0) >= 40 else "🔴"
                    st.markdown(f"**{score_color} {score_pct or 0}%**")
                with col3:
                    idx_statut = STATUTS.index(statut) if statut in STATUTS else 0
                    nouveau_statut = st.selectbox(
                        "Statut",
                        STATUTS,
                        index=idx_statut,
                        key=f"statut_{cand_id}",
                        format_func=lambda s: f"{STATUT_EMOJI.get(s, '')} {s}",
                        label_visibility="collapsed",
                    )
                    if nouveau_statut != statut:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE candidatures SET statut=%s WHERE id=%s",
                                (nouveau_statut, cand_id)
                            )
                        conn.commit()
                        st.rerun()
                st.divider()

        conn.close()

    

    def _show_demo_dashboard(self) -> None:
        """Affiche un dashboard de démo avec des données simulées."""
        st.markdown("---")
        st.markdown("#### 🎯 Aperçu avec données de démonstration")

        # Données simulées réalistes
        demo_contrats = pd.Series({
            "CDI": 145, "Alternance": 67, "Stage": 43,
            "CDD": 28, "Freelance": 17
        })
        demo_comps = pd.Series({
            "Python": 198, "SQL": 167, "Machine Learning": 134,
            "Docker": 112, "AWS": 98, "Git": 95,
            "Pandas": 87, "FastAPI": 72, "Deep Learning": 65, "Spark": 48
        })

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Répartition des contrats (démo)**")
            st.bar_chart(demo_contrats)
        with col2:
            st.markdown("**Top compétences (démo)**")
            st.bar_chart(demo_comps)

        st.caption("*Données simulées à titre illustratif*")

    # ── Tab Historique ─────────────────────────────────────────────────────────
    def _tab_historique(self) -> None:
        """Dossiers de candidature : offre + CV + lettre de motivation + conseils IA."""
        st.markdown("## 🗂️ Historique de mes dossiers")
        st.caption("Consultez et gérez vos candidatures avec leur dossier complet.")

        try:
            conn = _get_db()
        except Exception as e:
            st.error(f"Connexion DB impossible : {e}")
            st.info("Vérifiez que DATABASE_URL est configuré dans les secrets Streamlit ou dans .env")
            return

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.id, o.titre, o.entreprise, o.localisation, o.type_contrat,
                           c.statut,
                           COALESCE(ROUND(c.score_matching * 100)::int, 0) AS score_pct,
                           c.notes, c.date_candidature::date,
                           c.lettre_motivation, c.conseils_ia,
                           cv.nom_fichier, o.description
                    FROM candidatures c
                    JOIN offres o ON c.offre_id = o.id
                    JOIN cvs cv ON c.cv_id = cv.id
                    ORDER BY c.date_candidature DESC
                    LIMIT 50
                """)
                rows = cur.fetchall()
        except Exception:
            # Fallback si colonnes lettre_motivation / conseils_ia absentes
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT c.id, o.titre, o.entreprise, o.localisation, o.type_contrat,
                               c.statut,
                               COALESCE(ROUND(c.score_matching * 100)::int, 0) AS score_pct,
                               c.notes, c.date_candidature::date,
                               NULL, NULL,
                               cv.nom_fichier, o.description
                        FROM candidatures c
                        JOIN offres o ON c.offre_id = o.id
                        JOIN cvs cv ON c.cv_id = cv.id
                        ORDER BY c.date_candidature DESC
                        LIMIT 50
                    """)
                    rows = cur.fetchall()
            except Exception as e2:
                st.error(f"Erreur de lecture : {e2}")
                conn.close()
                return

        if not rows:
            st.info(
                "Aucun dossier enregistré pour l'instant.\n\n"
                "Postulez à des offres dans l'onglet **📁 Mes Candidatures** pour voir vos dossiers ici."
            )
            conn.close()
            return

        STATUT_EMOJI = {
            'nouveau': '🆕', 'en_cours': '⏳', 'entretien': '🤝',
            'test_technique': '💻', 'offre': '🎉', 'refus': '❌', 'accepte': '✅',
        }

        # Sélecteur de dossier
        labels = {
            i: f"{STATUT_EMOJI.get(row[5], '•')}  {row[1]} — {row[2]}  ({row[8]})"
            for i, row in enumerate(rows)
        }
        selected = st.selectbox(
            "Choisir un dossier de candidature",
            list(labels.keys()),
            format_func=lambda i: labels[i],
        )

        if selected is not None:
            row = rows[selected]
            cand_id, titre, entreprise, localisation, contrat, statut, score_pct, notes, date, lettre, conseils, nom_cv, description = row

            st.divider()
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"## 💼 {titre}")
                st.markdown(f"**{entreprise}** · {localisation or '—'} · `{contrat or '—'}`")
                st.caption(f"Candidature du {date} · CV : {nom_cv or '—'}")
            with col_b:
                st.metric("Score matching", f"{score_pct}%")
                STATUTS_HIST = ['nouveau', 'en_cours', 'entretien', 'test_technique', 'offre', 'refus', 'accepte']
                idx_s = STATUTS_HIST.index(statut) if statut in STATUTS_HIST else 0
                nouveau_statut_h = st.selectbox(
                    "Statut",
                    STATUTS_HIST,
                    index=idx_s,
                    key=f"hist_statut_{cand_id}",
                    format_func=lambda s: f"{STATUT_EMOJI.get(s, '')} {s}",
                )
                if nouveau_statut_h != statut:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE candidatures SET statut=%s WHERE id=%s", (nouveau_statut_h, cand_id))
                    conn.commit()
                    st.rerun()

            # Sous-onglets du dossier
            sub1, sub2, sub3, sub4 = st.tabs([
                "📋 Offre", "✉️ Lettre de motivation", "💡 Conseils IA", "📝 Notes",
            ])

            with sub1:
                st.markdown("### Description du poste")

                # ── Infos structurées ───────────────────────────────────────────
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.markdown(f"🏢 **Entreprise :** {entreprise or '—'}")
                    st.markdown(f"📍 **Localisation :** {localisation or '—'}")
                with col_i2:
                    st.markdown(f"📋 **Contrat :** {contrat or '—'}")
                    st.markdown(f"📊 **Score de matching :** {score_pct} %")

                st.divider()

                # ── Compétences extraites automatiquement ───────────────────────
                from etl.pipeline import DataTransformer
                # Texte à analyser : body DB + titre comme fallback
                _desc_for_skills = (description or "") + " " + (titre or "")
                comp_extraites = DataTransformer.extract_competences(_desc_for_skills) if _desc_for_skills.strip() else []
                if comp_extraites:
                    st.markdown("#### 🔧 Compétences détectées")
                    badges = "  ".join(f"`{c}`" for c in sorted(comp_extraites)[:20])
                    st.markdown(badges)
                    st.divider()

                # ── Texte complet de l'offre ────────────────────────────────────
                st.markdown("#### 📄 Description complète")
                # Chercher dans session_state si la DB n'a pas le body
                _body_live = ""
                for _o in st.session_state.get("job_offers", []):
                    if hasattr(_o, "title") and _o.title == titre:
                        _body_live = getattr(_o, "body", "") or ""
                        break
                desc_display = (description or "").strip() or _body_live.strip()
                if desc_display:
                    st.text_area(
                        "", value=desc_display[:4000], height=280,
                        disabled=True, label_visibility="collapsed",
                    )
                else:
                    st.info("Description complète non disponible — seules les informations structurées sont enregistrées pour cette offre.")

            with sub2:
                st.markdown("### Lettre de motivation")
                if lettre:
                    st.text_area(
                        "", value=lettre, height=300,
                        disabled=True, label_visibility="collapsed",
                    )
                    if st.button("🔄 Régénérer avec l'IA", key=f"regen_{cand_id}"):
                        cv_txt = st.session_state.get("cv_text", "")
                        with st.spinner("Génération en cours…"):
                            from core.models import Message as Msg
                            prompt = (
                                f"Rédige une lettre de motivation concise et professionnelle (3 paragraphes) "
                                f"pour le poste de **{titre}** chez **{entreprise}**.\n\n"
                                f"CV (extrait) :\n{cv_txt[:1500]}\n\n"
                                f"Offre (extrait) :\n{(description or '')[:800]}"
                            )
                            result = self._llm.complete(
                                [Msg("user", prompt)], self.model,
                                temperature=0.5, max_tokens=600,
                            )
                        st.text_area("Lettre régénérée", value=result, height=300)
                else:
                    st.info("Aucune lettre enregistrée pour ce dossier.")
                    cv_txt = st.session_state.get("cv_text", "")
                    if cv_txt and description and st.button("✍️ Générer une lettre avec l'IA", key=f"gen_{cand_id}"):
                        with st.spinner("Génération en cours…"):
                            from core.models import Message as Msg
                            prompt = (
                                f"Rédige une lettre de motivation concise et professionnelle (3 paragraphes) "
                                f"pour le poste de **{titre}** chez **{entreprise}**.\n\n"
                                f"CV (extrait) :\n{cv_txt[:1500]}\n\n"
                                f"Offre (extrait) :\n{(description or '')[:800]}"
                            )
                            result = self._llm.complete(
                                [Msg("user", prompt)], self.model,
                                temperature=0.5, max_tokens=600,
                            )
                        st.text_area("Lettre générée", value=result, height=300)
                    elif not cv_txt:
                        st.caption("Uploadez votre CV dans l'onglet *Recherche d'offres & CV* pour générer une lettre.")

            with sub3:
                st.markdown("### Conseils IA pour ce poste")
                if conseils:
                    st.write(conseils)
                else:
                    st.info("Aucun conseil IA enregistré. Lors de votre prochaine recherche, activez *Analyser la compatibilité avec mon CV* pour obtenir des conseils automatiques.")

            with sub4:
                st.markdown("### Vos notes personnelles")
                if notes:
                    st.write(notes)
                else:
                    st.info("Aucune note pour ce dossier. Ajoutez des notes depuis l'onglet *Mes Candidatures*.")

        conn.close()

    # ── Tab Glossaire (commenté — remplacé par Historique) ────────────────────
    # def _tab_glossaire(self) -> None:
    #     st.markdown("## 📖 Glossaire des termes techniques")
    #     ... (contenu déplacé — disponible dans la documentation du projet)


# ── Entry point ────────────────────────────────────────────────────────────────
App()
