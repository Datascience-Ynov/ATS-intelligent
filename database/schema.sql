-- =============================================================================
-- ATS Intelligent — Schéma PostgreSQL complet
-- Projet RNCP39586 — Mourad Amoussa — Lyon Ynov M2 IA & Data
-- Couvre : C1.1.2 (modélisation), C1.3.2 (sécurité / contraintes), C1.3.3 (intégrité)
-- =============================================================================

-- Extension pour UUID (optionnel mais pro)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- TABLE : offres
-- Stocke les offres d'emploi collectées (France Travail + jobspy)
-- =============================================================================
CREATE TABLE IF NOT EXISTS offres (
    id               SERIAL PRIMARY KEY,
    id_offre         VARCHAR(100) UNIQUE NOT NULL,          -- identifiant source (ex: FT-123456)
    titre            VARCHAR(500),
    description      TEXT,
    entreprise       VARCHAR(500),
    localisation     VARCHAR(300),
    code_postal      VARCHAR(20),
    type_contrat     VARCHAR(50),                           -- CDI, CDD, Stage, Alternance...
    experience       VARCHAR(100),                          -- "2 ans", "Débutant accepté"...
    competences      TEXT[],                                -- ARRAY PostgreSQL : ['Python','SQL',...]
    salaire_libelle  VARCHAR(200),                          -- "35-45K€ selon profil"
    salaire_min      INTEGER,                               -- en € annuel brut (si parsé)
    salaire_max      INTEGER,
    secteur          VARCHAR(200),
    source           VARCHAR(50) DEFAULT 'france_travail',  -- 'france_travail','linkedin','indeed'...
    url              TEXT,
    date_publication TIMESTAMP,
    date_collecte    TIMESTAMP DEFAULT NOW(),
    is_processed     BOOLEAN DEFAULT FALSE,                 -- ETL traité ?
    nb_candidatures  INTEGER DEFAULT 0
);

-- Index pour les requêtes fréquentes
CREATE INDEX IF NOT EXISTS idx_offres_type_contrat  ON offres(type_contrat);
CREATE INDEX IF NOT EXISTS idx_offres_localisation  ON offres(localisation);
CREATE INDEX IF NOT EXISTS idx_offres_source        ON offres(source);
CREATE INDEX IF NOT EXISTS idx_offres_is_processed  ON offres(is_processed);
CREATE INDEX IF NOT EXISTS idx_offres_date_collecte ON offres(date_collecte);
-- Index GIN pour la recherche dans les tableaux de compétences
CREATE INDEX IF NOT EXISTS idx_offres_competences   ON offres USING GIN(competences);

COMMENT ON TABLE offres IS 'Offres d''emploi collectées via France Travail API et python-jobspy';
COMMENT ON COLUMN offres.competences IS 'Tableau de compétences extraites par NLP depuis la description';

-- =============================================================================
-- TABLE : cvs
-- Stocke les CVs uploadés et leur texte extrait
-- =============================================================================
CREATE TABLE IF NOT EXISTS cvs (
    id               SERIAL PRIMARY KEY,
    nom_fichier      VARCHAR(500) NOT NULL,
    texte_brut       TEXT,                                  -- texte extrait par pdfplumber
    competences      TEXT[],                                -- compétences détectées dans le CV
    experience_annees INTEGER,                              -- années d'expérience estimées
    formations       TEXT[],                                -- diplômes détectés
    langages         TEXT[],                                -- langages de programmation
    localisation     VARCHAR(300),
    date_upload      TIMESTAMP DEFAULT NOW(),
    hash_fichier     VARCHAR(64) UNIQUE                     -- SHA256 pour éviter les doublons
);

CREATE INDEX IF NOT EXISTS idx_cvs_competences ON cvs USING GIN(competences);

COMMENT ON TABLE cvs IS 'CVs uploadés et analysés par pdfplumber + NLP';

-- =============================================================================
-- TABLE : scores_matching
-- Résultats du modèle ML de matching CV ↔ Offre
-- =============================================================================
CREATE TABLE IF NOT EXISTS scores_matching (
    id               SERIAL PRIMARY KEY,
    cv_id            INTEGER REFERENCES cvs(id) ON DELETE CASCADE,
    offre_id         INTEGER REFERENCES offres(id) ON DELETE CASCADE,
    score_ml         FLOAT NOT NULL CHECK (score_ml >= 0 AND score_ml <= 1),
    score_percent    INTEGER GENERATED ALWAYS AS (ROUND(score_ml * 100)::INTEGER) STORED,
    confidence       FLOAT,                                 -- probabilité retournée par le modèle
    features_json    JSONB,                                 -- features utilisées (pour audit)
    modele_version   VARCHAR(50),                           -- ex: 'v1', 'v2'
    interpretation   TEXT,                                  -- texte d'interprétation (LLM)
    date_calcul      TIMESTAMP DEFAULT NOW(),
    UNIQUE(cv_id, offre_id)                                 -- un score unique par paire
);

CREATE INDEX IF NOT EXISTS idx_scores_cv_id    ON scores_matching(cv_id);
CREATE INDEX IF NOT EXISTS idx_scores_offre_id ON scores_matching(offre_id);
CREATE INDEX IF NOT EXISTS idx_scores_score    ON scores_matching(score_ml DESC);

COMMENT ON TABLE scores_matching IS 'Scores de matching ML entre CVs et offres (C5.3.2)';

-- =============================================================================
-- TABLE : candidatures
-- Suivi des candidatures (fonctionnalité ATS)
-- =============================================================================
CREATE TABLE IF NOT EXISTS candidatures (
    id               SERIAL PRIMARY KEY,
    cv_id            INTEGER REFERENCES cvs(id) ON DELETE SET NULL,
    offre_id         INTEGER REFERENCES offres(id) ON DELETE CASCADE,
    statut           VARCHAR(50) DEFAULT 'nouveau'          -- nouveau, en_cours, entretien, refus, accepte
                     CHECK (statut IN ('nouveau','en_cours','entretien','test_technique','offre','refus','accepte')),
    score_matching   FLOAT,
    notes            TEXT,
    date_candidature TIMESTAMP DEFAULT NOW(),
    date_maj         TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_candidatures_statut   ON candidatures(statut);
CREATE INDEX IF NOT EXISTS idx_candidatures_offre_id ON candidatures(offre_id);

COMMENT ON TABLE candidatures IS 'Pipeline de suivi des candidatures — fonctionnalité ATS core';

-- =============================================================================
-- TABLE : competences_ref
-- Référentiel des compétences IT (ontologie métier)
-- =============================================================================
CREATE TABLE IF NOT EXISTS competences_ref (
    id               SERIAL PRIMARY KEY,
    nom              VARCHAR(200) UNIQUE NOT NULL,
    categorie        VARCHAR(100),                          -- 'langage', 'framework', 'cloud', 'ml', 'data'...
    synonymes        TEXT[],                                -- ['py', 'python3', 'Python 3']
    popularite       INTEGER DEFAULT 0                      -- nombre d'offres contenant cette compétence
);

-- Données initiales — référentiel compétences IT
INSERT INTO competences_ref (nom, categorie, synonymes) VALUES
    ('Python',           'langage',    ARRAY['py', 'python3', 'Python 3']),
    ('SQL',              'langage',    ARRAY['MySQL', 'PostgreSQL', 'SQLite', 'T-SQL']),
    ('Machine Learning', 'domaine_ml', ARRAY['ML', 'apprentissage automatique']),
    ('Deep Learning',    'domaine_ml', ARRAY['DL', 'réseau de neurones', 'neural network']),
    ('NLP',              'domaine_ml', ARRAY['traitement du langage', 'Natural Language Processing']),
    ('Scikit-learn',     'framework',  ARRAY['sklearn']),
    ('TensorFlow',       'framework',  ARRAY['TF', 'tensorflow2']),
    ('PyTorch',          'framework',  ARRAY['torch']),
    ('Pandas',           'librairie',  ARRAY['pandas']),
    ('NumPy',            'librairie',  ARRAY['numpy']),
    ('Docker',           'devops',     ARRAY['conteneur', 'container']),
    ('Kubernetes',       'devops',     ARRAY['K8s', 'k8s']),
    ('AWS',              'cloud',      ARRAY['Amazon Web Services', 'EC2', 'S3']),
    ('Azure',            'cloud',      ARRAY['Microsoft Azure', 'Azure ML']),
    ('GCP',              'cloud',      ARRAY['Google Cloud', 'BigQuery']),
    ('Spark',            'bigdata',    ARRAY['Apache Spark', 'PySpark']),
    ('Hadoop',           'bigdata',    ARRAY['HDFS', 'MapReduce']),
    ('FastAPI',          'framework',  ARRAY['fast api']),
    ('Django',           'framework',  ARRAY['django rest']),
    ('Git',              'outil',      ARRAY['GitHub', 'GitLab', 'version control']),
    ('Power BI',         'bi',         ARRAY['PowerBI', 'PBIX']),
    ('Tableau',          'bi',         ARRAY['Tableau Desktop', 'Tableau Server']),
    ('Streamlit',        'framework',  ARRAY['streamlit']),
    ('R',                'langage',    ARRAY['R language', 'Rstudio']),
    ('Scala',            'langage',    ARRAY['scala']),
    ('Java',             'langage',    ARRAY['Java 8', 'Java 11', 'Spring']),
    ('JavaScript',       'langage',    ARRAY['JS', 'Node.js', 'NodeJS']),
    ('MLflow',           'mlops',      ARRAY['mlflow']),
    ('Airflow',          'mlops',      ARRAY['Apache Airflow']),
    ('dbt',              'data',       ARRAY['data build tool']),
    ('Snowflake',        'data',       ARRAY['snowflake']),
    ('MongoDB',          'database',   ARRAY['Mongo']),
    ('Redis',            'database',   ARRAY['redis'])
ON CONFLICT (nom) DO NOTHING;

-- =============================================================================
-- TABLE : etl_logs
-- Traçabilité des exécutions ETL (C1.3.3)
-- =============================================================================
CREATE TABLE IF NOT EXISTS etl_logs (
    id               SERIAL PRIMARY KEY,
    run_id           UUID DEFAULT uuid_generate_v4(),
    etape            VARCHAR(100),                          -- 'extract', 'transform', 'load'
    source           VARCHAR(100),
    nb_lignes_in     INTEGER,
    nb_lignes_out    INTEGER,
    nb_erreurs       INTEGER DEFAULT 0,
    statut           VARCHAR(20) DEFAULT 'success'          -- 'success', 'error', 'warning'
                     CHECK (statut IN ('success', 'error', 'warning')),
    message          TEXT,
    duree_secondes   FLOAT,
    date_execution   TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE etl_logs IS 'Logs de traçabilité ETL — prouve la gouvernance des données (C1.3.3)';

-- =============================================================================
-- VUE : stats_offres
-- Statistiques agrégées pour le dashboard (C2.1.2, C2.1.3)
-- =============================================================================
CREATE OR REPLACE VIEW stats_offres AS
SELECT
    type_contrat,
    COUNT(*)                                    AS nb_offres,
    COUNT(DISTINCT entreprise)                  AS nb_entreprises,
    AVG(CARDINALITY(competences))               AS avg_nb_competences,
    AVG(salaire_min)                            AS avg_salaire_min,
    AVG(salaire_max)                            AS avg_salaire_max,
    MIN(date_collecte)                          AS premiere_collecte,
    MAX(date_collecte)                          AS derniere_collecte
FROM offres
GROUP BY type_contrat
ORDER BY nb_offres DESC;

-- =============================================================================
-- VUE : top_competences
-- Compétences les plus demandées (C2.2.1)
-- =============================================================================
CREATE OR REPLACE VIEW top_competences AS
SELECT
    comp                         AS competence,
    COUNT(*)                     AS frequence,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM offres), 2) AS pct_offres
FROM offres, UNNEST(competences) AS comp
GROUP BY comp
ORDER BY frequence DESC;

-- =============================================================================
-- VUE : pipeline_matching
-- Vue métier : offres + meilleur score de matching
-- =============================================================================
CREATE OR REPLACE VIEW pipeline_matching AS
SELECT
    o.id,
    o.titre,
    o.entreprise,
    o.type_contrat,
    o.localisation,
    o.competences,
    COALESCE(MAX(sm.score_percent), 0) AS meilleur_score,
    COUNT(DISTINCT sm.cv_id)           AS nb_cvs_evalues,
    COUNT(DISTINCT c.id)               AS nb_candidatures
FROM offres o
LEFT JOIN scores_matching sm ON sm.offre_id = o.id
LEFT JOIN candidatures c     ON c.offre_id  = o.id
GROUP BY o.id, o.titre, o.entreprise, o.type_contrat, o.localisation, o.competences
ORDER BY meilleur_score DESC;

-- =============================================================================
-- Trigger : mise à jour automatique date_maj dans candidatures
-- =============================================================================
CREATE OR REPLACE FUNCTION update_date_maj()
RETURNS TRIGGER AS $$
BEGIN
    NEW.date_maj = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_candidatures_date_maj
    BEFORE UPDATE ON candidatures
    FOR EACH ROW EXECUTE FUNCTION update_date_maj();

-- =============================================================================
-- Permissions minimales (sécurité — C1.3.2)
-- Décommenter si vous créez un user dédié "ats_user"
-- =============================================================================
-- CREATE USER ats_user WITH PASSWORD 'votre_mdp_ici';
-- GRANT CONNECT ON DATABASE ats_db TO ats_user;
-- GRANT USAGE ON SCHEMA public TO ats_user;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO ats_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ats_user;

-- =============================================================================
-- Message de confirmation
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE '✅ Schéma ATS créé avec succès !';
    RAISE NOTICE '   Tables  : offres, cvs, scores_matching, candidatures, competences_ref, etl_logs';
    RAISE NOTICE '   Vues    : stats_offres, top_competences, pipeline_matching';
    RAISE NOTICE '   Trigger : update_date_maj sur candidatures';
END $$;
