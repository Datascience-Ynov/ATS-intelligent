# Guide d'installation — ATS Intelligent
### Mourad Amoussa | Lyon Ynov M2 IA & Data | RNCP39586

---

## C'est quoi un "cluster" PostgreSQL ?

Quand vous lisez "cluster PostgreSQL" dans la documentation, **ne vous inquiétez pas** — ce n'est PAS un cluster Hadoop/Spark (qui nécessite 10 serveurs et une équipe DevOps).

En PostgreSQL, "cluster" = **l'instance du serveur PostgreSQL** sur votre machine. C'est juste le nom technique de "PostgreSQL installé et qui tourne chez vous". Vous n'avez besoin que de votre PC.

| Terme             | Ce que ça signifie en pratique        |
|-------------------|---------------------------------------|
| Cluster PostgreSQL | PostgreSQL installé sur votre PC      |
| Base de données   | Un "dossier logique" dans PostgreSQL  |
| Table             | Comme une feuille Excel dans la base  |
| Serveur           | Le programme PostgreSQL qui tourne    |

---

## Étape 1 — Installer PostgreSQL sur Windows

### 1.1 Télécharger l'installeur

Allez sur : **https://www.postgresql.org/download/windows/**

Cliquez sur **"Download the installer"** → choisissez la version **16.x** (la plus récente stable).

### 1.2 Lancer l'installation

Double-cliquez sur le fichier `.exe` téléchargé. L'assistant vous demandera :

- **Installation Directory** : laisser la valeur par défaut (`C:\Program Files\PostgreSQL\16`)
- **Components** : cocher **tout** (PostgreSQL Server ✅, pgAdmin ✅, Command Line Tools ✅, Stack Builder ✅)
- **Data Directory** : laisser par défaut
- **Password** : **IMPORTANT** — entrez un mot de passe et **notez-le**. Exemple : `monmotdepasse123`
- **Port** : laisser `5432` (valeur par défaut)
- **Locale** : laisser par défaut
Cliquez "Next" jusqu'à la fin. L'installation prend 2-3 minutes.

### 1.3 Vérifier que PostgreSQL tourne

Ouvrez le menu Démarrer → cherchez **"pgAdmin 4"** et lancez-le.

Si pgAdmin s'ouvre dans votre navigateur → ✅ PostgreSQL est installé et actif.

---

## Étape 2 — Créer la base de données `ats_db`

### Option A : Via pgAdmin 4 (recommandé pour débutants)

1. Dans pgAdmin, à gauche, cliquez sur **Servers** → **PostgreSQL 16**
2. Il vous demande le mot de passe → entrez celui choisi à l'installation
3. Clic droit sur **Databases** → **Create** → **Database...**
4. Dans le champ **Database** : tapez `ats_db`
5. Cliquez **Save**

Votre base `ats_db` apparaît maintenant dans la liste.

### Option B : Via la ligne de commande (psql)

Ouvrez le menu Démarrer → cherchez **"SQL Shell (psql)"** :

```
Server [localhost]:          ← appuyez Entrée
Database [postgres]:         ← appuyez Entrée
Port [5432]:                 ← appuyez Entrée
Username [postgres]:         ← appuyez Entrée
Password for user postgres:  ← entrez votre mot de passe
```

Puis tapez :
```sql
CREATE DATABASE ats_db;
\q
```

---

## Étape 3 — Créer les tables (exécuter le schéma SQL)

### Via pgAdmin 4

1. Dans pgAdmin, cliquez sur `ats_db` (dans la liste des bases)
2. Cliquez sur **Tools** (menu du haut) → **Query Tool**
3. Une zone de texte s'ouvre à droite
4. Ouvrez le fichier `database/schema.sql` de votre projet dans un éditeur de texte (Notepad, VS Code...)
5. Copiez **tout le contenu**
6. Collez-le dans la Query Tool de pgAdmin
7. Cliquez sur le bouton **▶ (Run/Execute)** ou appuyez sur `F5`

Vous devriez voir en bas : `✅ Schéma ATS créé avec succès !`

---

## Étape 4 — Configurer le fichier `.env`

Dans votre dossier projet `chatbot_ia_generative/` :

1. Copiez `.env.example` et renommez-le `.env`
2. Ouvrez `.env` avec VS Code ou Notepad
3. Remplissez les valeurs :

```env
NVIDIA_API_KEY=nvapi-votre-cle    ← depuis https://build.nvidia.com
DB_HOST=localhost
DB_PORT=15432
DB_NAME=ats_db
DB_USER=postgres
DB_PASSWORD=monmotdepasse123      ← le mot de passe que vous avez choisi
```

---

## Étape 5 — Installer les dépendances Python

Dans un terminal (Anaconda Prompt, PowerShell, ou terminal VS Code) :

```bash
# Naviguez vers le dossier projet
cd "C:\Users\HP\Data Science\Perso\Deep Learning\Projet\chatbot_ia_generative"

# Installez tout
pip install -r requirements.txt
```

Si vous avez une erreur `psycopg2`, essayez :
```bash
pip install psycopg2-binary
```

---

## Étape 6 — Lancer le projet (ordre recommandé)

### 6.1 Générer le dataset ML et entraîner le modèle
```bash
# Générer 500 échantillons synthétiques
python -m ml.dataset

# Entraîner les 3 modèles + GridSearchCV (5-10 min)
python -m ml.train

# Version rapide sans GridSearchCV (1-2 min)
python -m ml.train --skip-grid-search
```

### 6.2 Collecter des offres (ETL)
```bash
# Collecte via python-jobspy (LinkedIn/Indeed/Glassdoor) - ne nécessite pas de clé API
python -m etl.pipeline --source jobspy --query "Data Scientist Lyon"

# Collecte via France Travail (nécessite FT_CLIENT_ID et FT_CLIENT_SECRET dans .env)
python -m etl.pipeline --source france_travail --query "Machine Learning" --departement 69

# Si France Travail fournit une URL de remplacement pendant une maintenance :
# FT_SEARCH_URL=https://.../partenaire/offresdemploi/v2/offres/search
```

### 6.3 Lancer l'interface Streamlit
```bash
streamlit run app.py
```
→ S'ouvre automatiquement sur http://localhost:8501

### 6.4 Lancer l'API FastAPI (optionnel)
```bash
uvicorn api.scoring_api:app --reload --port 8000
```
→ Documentation interactive : http://localhost:8000/docs

---

## Structure finale du projet

```
chatbot_ia_generative/
├── .env                          ← vos credentials (NE PAS COMMITTER)
├── .env.example                  ← template des variables
├── requirements.txt              ← toutes les dépendances
├── app.py                        ← interface Streamlit principale
├── core/                         ← modules chatbot existants
│   ├── services.py               (LLMService, CVParser, JobSearchService...)
│   ├── models.py
│   ├── ui.py
│   └── styles.py
├── etl/
│   └── pipeline.py               ← ETL France Travail + JobSpy (C1.1.2)
├── ml/
│   ├── dataset.py                ← génération dataset synthétique (C5.2.1)
│   ├── train.py                  ← pipeline ML complet (C5.2.1→C5.3.1)
│   └── retrain_pipeline.py       ← MLOps drift + retrain (C5.3.3→C5.3.4)
├── api/
│   └── scoring_api.py            ← API FastAPI (C5.3.2)
├── database/
│   └── schema.sql                ← schéma PostgreSQL complet
└── data_lake/
    ├── raw/offres_json/          ← données brutes JSON
    ├── raw/cvs_pdf/              ← CVs uploadés
    ├── processed/
    │   ├── offres_clean.parquet  ← données nettoyées
    │   ├── dataset_matching.parquet ← dataset ML
    │   └── stats_aggregees.json  ← stats pour le dashboard
    └── models/
        ├── v1/                   ← modèle version 1
        │   ├── model.joblib
        │   ├── metadata.json
        │   ├── feature_names.json
        │   ├── feature_selection.json
        │   └── comparison.json
        └── latest/               ← modèle actif (copie de la dernière version)
```

---

## Correspondance compétences RNCP39586

| Fichier / Action               | Compétences validées                          |
|-------------------------------|-----------------------------------------------|
| `database/schema.sql`          | C1.1.2, C1.3.2, C1.3.3                       |
| `etl/pipeline.py`              | C1.1.2, C1.2.2, C1.3.2, C1.3.3              |
| `ml/dataset.py`                | C5.1.3, C5.2.1                               |
| `ml/train.py`                  | C5.2.1, C5.2.2, C5.2.3, C5.2.4, C5.3.1     |
| `api/scoring_api.py`           | C5.3.2, C5.3.3                               |
| `ml/retrain_pipeline.py`       | C5.3.3, C5.3.4                               |
| `app.py` onglet Dashboard      | C2.1.2, C2.1.3, C2.1.4, C2.2.1              |

---

## Dépannage courant

**`psycopg2.OperationalError: could not connect to server`**
→ PostgreSQL n'est pas démarré. Ouvrez les Services Windows (Win+R → `services.msc`) et démarrez "postgresql-x64-16".

**`ModuleNotFoundError: No module named 'xxx'`**
→ Lancez `pip install -r requirements.txt` depuis le bon dossier.

**`FileNotFoundError: data_lake/models/latest/model.joblib`**
→ Vous n'avez pas encore entraîné le modèle. Lancez `python -m ml.train`.

**L'API renvoie 503**
→ Le modèle n'est pas chargé. Vérifiez que `data_lake/models/latest/model.joblib` existe.
