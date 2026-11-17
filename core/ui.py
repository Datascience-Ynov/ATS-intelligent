"""
Composants UI — helpers de rendu HTML statiques pour Streamlit.
"""

from __future__ import annotations
import html as _html
import re as _re

import streamlit as st

from .models import JobOffer


def _clean_url(raw: str) -> str:
    """Extrait l'URL propre depuis une chaîne qui pourrait contenir du HTML brut.
    Ex: '<a class="job-link" href="https://...">' → 'https://...'
    """
    if raw and raw.strip().startswith("<"):
        m = _re.search(r'href=["\']([^"\']+)["\']', raw)
        return m.group(1) if m else "#"
    return raw or "#"


class UI:
    """Méthodes statiques de rendu des éléments visuels."""

    # ── Bulles de chat ─────────────────────────────────────────────────────

    @staticmethod
    def render_message(role: str, content: str) -> None:
        """Affiche une bulle utilisateur ou assistant."""
        safe = _html.escape(content).replace("\n", "<br>")
        if role == "assistant":
            st.markdown(f"""
            <div class="msg-row assistant">
                <div class="avatar">🤖</div>
                <div class="bubble-assistant">{safe}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row user">
                <div class="avatar">👤</div>
                <div class="bubble-user">{safe}</div>
            </div>""", unsafe_allow_html=True)

    @staticmethod
    def render_streaming(placeholder, text: str, cursor: bool = True) -> None:
        """Met à jour le placeholder avec le texte streamé."""
        safe = _html.escape(text).replace("\n", "<br>")
        cur  = "▌" if cursor else ""
        placeholder.markdown(f"""
        <div class="msg-row assistant">
            <div class="avatar">🤖</div>
            <div class="bubble-assistant">{safe}{cur}</div>
        </div>""", unsafe_allow_html=True)

    @staticmethod
    def render_welcome() -> None:
        """Affiche le message d'accueil (conversation vide)."""
        st.markdown("""
        <div class="msg-row assistant">
            <div class="avatar">🤖</div>
            <div class="bubble-welcome">
                👋 Bonjour ! Je suis votre assistant IA propulsé par
                <strong>Llama 3.1</strong> via <strong>Groq</strong>.
                Posez-moi n'importe quelle question, je suis là pour vous aider !
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Cartes offres d'emploi ─────────────────────────────────────────────

    @staticmethod
    def render_job_card(offer: JobOffer) -> None:
        """Affiche une carte offre avec score et analyse si disponibles."""
        score_class = (
            "score-high" if offer.score >= 70
            else "score-mid" if offer.score >= 40
            else "score-low"
        )
        score_html = (
            f'<span class="score-badge {score_class}">Compatibilité : {offer.score} %</span>'
            if offer.score else ""
        )
        analysis_html = ""
        if offer.analysis:
            safe_analysis = (
                _html.escape(offer.analysis)
                .replace("\n", "<br>")
                .replace("**Points forts :**", "<strong>Points forts :</strong>")
                .replace("**Points faibles :**", "<strong>Points faibles :</strong>")
                .replace("**Conseil :**", "<strong>Conseil :</strong>")
                .replace("**", "")   # neutralise tout ** résiduel (LLM markdown → casse st.markdown)
                .replace("*", "")    # idem pour * simple
            )
            analysis_html = f'<div class="analysis-box">{safe_analysis}</div>'

        # Séparer entreprise et localisation si encodés en "Entreprise — Ville"
        comp  = offer.company  or (offer.source.split(" — ", 1)[0] if offer.source else "")
        loc   = offer.location or (offer.source.split(" — ", 1)[1] if " — " in offer.source else "")
        company_html  = f'<span class="job-company">🏢 {_html.escape(comp)}</span>'  if comp else ""
        location_html = f'<span class="job-location">📍 {_html.escape(loc)}</span>'  if loc  else ""
        contract_html = f'<span class="job-location">📋 {_html.escape(offer.contract)}</span>' if offer.contract else ""
        date_html     = f'<span class="job-location">📅 Publié le {_html.escape(offer.date_posted)}</span>' if offer.date_posted else ""

        # Snippet : ignorer les lignes "Poste :" / "Entreprise :" / "Localisation :" déjà affichées
        body_lines = [l for l in offer.body.split("\n") if not l.startswith(("Poste :", "Entreprise :", "Localisation :", "Contrat :", "Rémunération :"))]
        snippet = " ".join(body_lines)[:280]

        st.markdown(f"""
        <div class="job-card">
            <div class="job-title">{_html.escape(offer.title)}</div>
            <div class="job-meta">{company_html} {location_html} {contract_html} {date_html}</div>
            <div class="job-snippet">{_html.escape(snippet)}{"…" if len(snippet) >= 280 else ""}</div>
            {score_html}
            {analysis_html}
            <a class="job-link" href="{_clean_url(offer.url)}" target="_blank">→ Voir l'offre complète</a>
        </div>
        """, unsafe_allow_html=True)
