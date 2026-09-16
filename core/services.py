"""
Services métier : LLM, parsing CV, recherche d'offres, analyse de compatibilité.
"""

from __future__ import annotations
import json
import re
from typing import Generator

import pdfplumber
from openai import OpenAI


# ── RGPD : anonymisation des données personnelles ────────────────────────────

def anonymize_cv(text: str) -> str:
    """Supprime les informations personnelles identifiables d'un CV avant stockage.

    Patterns supprimés :
      - Adresses email  → [EMAIL]
      - Numéros de téléphone français (07, 06, +33…) → [TELEPHONE]
      - Numéros de sécurité sociale (15 chiffres) → [NIR]

    Le nom/prénom ne peut pas être supprimé de façon fiable sans NER ;
    cette fonction couvre les données les plus sensibles (contact direct).

    Usage :
        raw_text = CVParser.extract(uploaded_file)
        clean_text = anonymize_cv(raw_text)   ← avant INSERT en base ou envoi au LLM
    """
    # Emails
    text = re.sub(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
        '[EMAIL]',
        text,
    )
    # Téléphones FR : 06/07 xx xx xx xx, +33 6 xx, 0033 6 xx, numéros fixes
    text = re.sub(
        r'(?:\+33|0033|0)[1-9](?:[\s.\-]?\d{2}){4}',
        '[TELEPHONE]',
        text,
    )
    # Numéro de sécurité sociale (15 chiffres consécutifs, parfois espacés par tirets)
    text = re.sub(
        r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b',
        '[NIR]',
        text,
    )
    return text

try:
    import jobspy
except ImportError:  # pragma: no cover - dépendance optionnelle
    jobspy = None

from .models import Message, JobOffer


class LLMService:
    """Wraps le client NVIDIA NIM (OpenAI-compatible)."""

    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )

    def stream(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        """Génère les tokens un par un (streaming)."""
        payload = [{"role": m.role, "content": m.content} for m in messages]
        response = self._client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature,
            top_p=0.7,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Retourne la réponse complète (non-streaming)."""
        return "".join(self.stream(messages, model, temperature, max_tokens))


class CVParser:
    """Extrait le texte brut d'un fichier PDF uploadé, puis l'anonymise (RGPD)."""

    @staticmethod
    def extract(uploaded_file) -> str:
        """Extrait le texte du PDF et supprime les PII avant tout stockage ou traitement."""
        with pdfplumber.open(uploaded_file) as pdf:
            raw = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
        # Anonymisation RGPD : email, téléphone, NIR supprimés avant stockage BDD / LLM
        return anonymize_cv(raw)


def _clean(val, fallback: str = "") -> str:
    """Convertit une valeur pandas en str propre (gère NaN, None, float)."""
    if val is None:
        return fallback
    s = str(val).strip()
    return fallback if s.lower() in ("nan", "none", "") else s


class JobSearchService:
    """Recherche des offres d'emploi via JobSpy (LinkedIn + Indeed + Glassdoor).
    Gratuit, sans clé API, sans inscription.
    """

    def __init__(self, max_results: int = 50) -> None:
        self._max = max_results

    def search(self, query: str, location: str = "France", hours_old: int = 720,
               max_results: int | None = None) -> list[JobOffer]:
        if jobspy is None:
            return []

        total_wanted = max_results if max_results is not None else self._max
        # jobspy interprète results_wanted PAR SITE — on divise pour respecter le total souhaité
        _SITES = ["linkedin", "indeed"]
        per_site = max(1, total_wanted // len(_SITES))
        df = jobspy.scrape_jobs(
            site_name=_SITES,  # Glassdoor retiré (400 sur localisation FR)
            search_term=query,
            location=location,
            results_wanted=per_site,
            hours_old=hours_old,
            country_indeed="France",
        )
        # Limiter au total voulu (les deux sites peuvent donner légèrement plus)
        if len(df) > total_wanted:
            df = df.head(total_wanted)

        results: list[JobOffer] = []
        for _, row in df.iterrows():
            title    = _clean(row.get("title"), "Sans titre")
            desc     = _clean(row.get("description"), "")
            url      = _clean(row.get("job_url"), "#")
            comp     = _clean(row.get("company"), "")
            loc      = _clean(row.get("location"), "")
            src      = f"{comp} — {loc}" if comp and loc else comp or loc
            contract = _clean(row.get("job_type") or row.get("employment_type"), "")
            salary   = _clean(row.get("min_amount") or row.get("salary"), "")

            # Date de publication
            raw_date = row.get("date_posted") or row.get("date") or ""
            date_posted = str(raw_date)[:10] if raw_date else ""

            # Enrichir le body avec titre + localisation pour le contexte LLM
            body_parts = []
            if title:
                body_parts.append(f"Poste : {title}")
            if comp:
                body_parts.append(f"Entreprise : {comp}")
            if loc:
                body_parts.append(f"Localisation : {loc}")
            if contract:
                body_parts.append(f"Contrat : {contract}")
            if salary:
                body_parts.append(f"Rémunération : {salary}")
            # Nettoyage HTML (certaines offres jobspy contiennent des balises brutes)
            desc = re.sub(r"<[^>]+>", "", desc).strip()
            if desc:
                body_parts.append(desc)
            body = "\n".join(body_parts)

            results.append(JobOffer(
                title=title,
                url=url,
                body=body[:2000],
                source=src,
                company=comp,
                location=loc,
                contract=contract,
                date_posted=date_posted,
            ))
        return results


def _strip_html(text: str) -> str:
    """Supprime les balises HTML d'un texte (fallback basique)."""
    return re.sub(r"<[^>]+>", "", text).strip()


class CompatibilityAnalyzer:
    """Analyse la compatibilité entre un CV et une offre d'emploi via LLM."""

    # Prompt très strict : on demande du JSON pur, on interdit tout le reste
    _PROMPT = """Tu es un expert RH spécialisé dans les profils tech et data.

CV du candidat :
---
{cv}
---

Offre d'emploi :
---
{job}
---

INSTRUCTIONS :
- Réponds UNIQUEMENT avec un objet JSON valide.
- N'ajoute aucun texte avant ou après le JSON.
- N'utilise pas de balises HTML, markdown, ou code block.
- Ne mentionne jamais le nom ou prénom du candidat dans ta réponse. Réfère-toi à lui uniquement par "le candidat".
- Respecte exactement ce schéma :

{{"score": <entier 0 à 100>, "points_forts": "<phrase courte>", "points_faibles": "<phrase courte>", "conseil": "<action concrète>"}}"""

    def __init__(self, llm: LLMService, model: str) -> None:
        self._llm   = llm
        self._model = model

    def analyze(self, cv_text: str, offer: JobOffer) -> JobOffer:
        prompt = self._PROMPT.format(cv=cv_text[:3000], job=offer.body[:1500])
        raw = self._llm.complete(
            [Message("user", prompt)],
            model=self._model,
            temperature=0.1,   # très bas pour forcer la rigueur du format
            max_tokens=400,
        )

        # Chercher un bloc JSON dans la réponse, même si le LLM a mis du texte autour
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                score = int(data.get("score", 0))
                offer.score    = max(0, min(100, score))  # clamp 0–100
                offer.analysis = (
                    f"**Points forts :** {data.get('points_forts', '—')}\n\n"
                    f"**Points faibles :** {data.get('points_faibles', '—')}\n\n"
                    f"**Conseil :** {data.get('conseil', '—')}"
                )
                return offer
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback : afficher le texte brut sans HTML
        clean = _strip_html(raw).strip()
        offer.score    = 0
        offer.analysis = clean if clean else "Analyse indisponible pour cette offre."
        return offer
