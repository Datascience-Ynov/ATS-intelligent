"""
Tests unitaires — ETL Pipeline
RNCP39586 — Bloc 1 : C1.3.2 (transformation), C1.3.3 (ETL)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.pipeline import DataTransformer


def test_clean_text_supprime_html():
    """C1.3.2 — Le nettoyage supprime les balises HTML."""
    result = DataTransformer.clean_text("<p>Développeur <b>Python</b></p>")
    assert "<" not in result
    assert "Python" in result


def test_clean_text_chaine_vide():
    """C1.3.2 — Texte vide retourne chaîne vide."""
    assert DataTransformer.clean_text("") == ""
    assert DataTransformer.clean_text(None) == ""


def test_extract_competences_detecte_python():
    """C1.3.2 — L'ontologie détecte Python dans une description."""
    comps = DataTransformer.extract_competences("Nous recherchons un expert Python et SQL.")
    assert "Python" in comps
    assert "SQL" in comps


def test_extract_competences_pas_de_faux_positif():
    """C1.3.2 — Pas de compétence inventée sur un texte vide."""
    comps = DataTransformer.extract_competences("")
    assert comps == []


def test_parse_salary_fourchette_k():
    """C1.3.2 — Parse 'Annuel de 35-45K€' → (35000, 45000)."""
    sal_min, sal_max = DataTransformer.parse_salary("35-45K€")
    assert sal_min == 35000
    assert sal_max == 45000


def test_parse_salary_aucune_info():
    """C1.3.2 — Salaire absent → (None, None)."""
    sal_min, sal_max = DataTransformer.parse_salary("")
    assert sal_min is None
    assert sal_max is None


def test_normalize_contrat_cdi():
    """C1.3.2 — 'CDI' normalisé en 'CDI'."""
    assert DataTransformer.normalize_contrat("CDI") == "CDI"


def test_normalize_contrat_alternance():
    """C1.3.2 — 'alternance' normalisé en 'Alternance'."""
    assert DataTransformer.normalize_contrat("alternance") == "Alternance"