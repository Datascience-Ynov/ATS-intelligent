# Scénario de soutenance — ATS Intelligent
**RNCP39586 — Spécialité Data Scientist — Bloc 5**
Durée cible : 30 minutes d'oral + 15 minutes de questions

---

## SLIDE 1 — Titre

Bonjour, et merci de m'accueillir pour cette soutenance. Je m'appelle Mourad Amoussa, étudiant en Master 2 IA & Data à Lyon Ynov, et je vais vous présenter aujourd'hui mon projet ATS Intelligent.

L'idée de base est simple : j'ai voulu créer un outil qui aide les étudiants à évaluer leur CV par rapport à des offres d'emploi réelles, grâce à du machine learning et à une intelligence artificielle conversationnelle.

Ce projet couvre l'ensemble du cycle de vie d'un modèle ML, du recueil des besoins jusqu'au déploiement et à l'automatisation — ce qui correspond précisément aux compétences du Bloc 5, Spécialité Data Scientist.

---

## SLIDE 2 — Contexte & Problématique

Pourquoi ce projet ? Parce que j'ai constaté un paradoxe : les étudiants en fin de formation ont les compétences, mais ne savent pas comment les présenter dans leur CV pour passer les filtres automatiques des ATS recruteurs.

Côté gauche : le problème. 70% des CVs sont éliminés sans être lus. Les étudiants ne savent pas quels mots-clés mettre. Il n'existe aucun outil gratuit qui permette de tester son CV contre une vraie offre.

Côté droit : la solution que j'ai construite. L'étudiant uploade son CV, l'outil le compare à des offres réelles récupérées via API, calcule un score de compatibilité ML, et propose un chat avec une IA qui donne des conseils personnalisés.

---

## SLIDE 3 — Vue d'ensemble du projet

Le projet est structuré en 5 blocs fonctionnels :
- La **collecte** : France Travail API et JobSpy pour récupérer des offres réelles
- Le **pipeline ETL** : une tâche automatisée toutes les 2 heures qui extrait, transforme et charge les données
- Le **stockage** : PostgreSQL avec 6 tables bien définies
- Le **modèle ML** : un RandomForestClassifier avec GridSearchCV, F1 à 0.857
- Et l'**interface** : une application Streamlit avec 5 onglets et un chat LLM

La boucle que je vais développer en détail aujourd'hui, c'est celle du ML — du feature engineering au déploiement.

---

## SLIDE 4 — Plan

Voici le plan de ma présentation. Je vais commencer par un survol rapide des Blocs 1 à 3 — collecte, analyse, management — qui seront détaillés dans mes rapports écrits. Puis je vais me concentrer sur le Bloc 5 dans ses trois activités : analyser et concevoir, développer le modèle, et déployer. Je ferai ensuite une démonstration live, et je terminerai par le bilan.

---

## SLIDE 5 — Blocs 1–3 (survol)

Avant d'entrer dans le détail du Bloc 5, je vais vous donner une vue rapide sur les trois premiers blocs, qui correspondent à la partie "tronc commun" du RNCP.

**Bloc 1 — Collecter et sécuriser** : J'ai mis en place un pipeline ETL avec APScheduler, un Data Lake en Parquet, une base PostgreSQL, et une gestion des accès avec Docker. Tout est documenté dans mon rapport Bloc 1.

**Bloc 2 — Analyser et valoriser** : Le modèle RandomForest est au cœur, avec 5 tests statistiques formalisés, un dashboard Streamlit, et l'intégration d'un LLM NVIDIA NIM pour le chat.

**Bloc 3 — Manager le projet** : Cahier des charges, planning Gantt, SWOT, veille techno — tout ce qu'il faut pour gérer un projet data seul.

---

## SLIDE 6 — Transition Bloc 5

Maintenant, le cœur de cette soutenance : le Bloc 5, Spécialité Data Scientist.

La question que je me suis posée : le modèle est entraîné, les features sont choisies — mais comment passer de l'idée à une application ML vraiment déployée, qui tourne, se surveille elle-même, et s'améliore avec le temps ?

C'est exactement ce que couvrent les trois activités A5.1, A5.2 et A5.3.

---

## SLIDE 7 — C5.1.1 Analyser la problématique

**Compétence C5.1.1 : analyser la problématique et le contexte.**

Avant d'écrire une seule ligne de code, j'ai conduit une analyse du besoin. Les questions que j'ai posées au marché : quels outils les étudiants utilisent-ils ? Quels critères retiennent les ATS recruteurs ? Est-ce qu'il existe une solution gratuite ?

La réponse est claire : aucun outil gratuit d'auto-évaluation n'existe. L'API France Travail donne accès à des données officielles gratuitement. Et la contrainte RGPD m'a imposé de travailler avec des CVs synthétiques.

**Conclusion faisabilité** : projet réalisable en solo, avec un stack open-source, en respectant les contraintes légales.

---

## SLIDE 8 — C5.1.2 Stratégie de résolution

**Compétence C5.1.2 : cadrer la stratégie de résolution.**

Une fois le problème posé, j'ai choisi le type de problème ML. Quatre options possibles :
- La régression : non, je n'ai pas de score continu à prédire
- Le clustering : non, je n'ai pas de labels, difficile à interpréter
- La **classification binaire** : oui ! Compatible (1) ou non compatible (0). C'est le choix retenu.
- La recommandation : intéressant pour la phase 2, mais pas réalisable sans historique de recrutement

Pourquoi la classification ? Parce que je peux générer un label synthétique à partir de règles métier simples — par exemple, si le skill_overlap est supérieur à 0.5, le candidat est compatible — sans avoir besoin de vrais CVs ou de vrais recruteurs.

---

## SLIDE 9 — C5.1.3 Sélection des technologies

**Compétence C5.1.3 : sélectionner les technologies et comparer les alternatives.**

Pour chaque besoin, j'ai fait un choix argumenté. Voici les principaux :
- **Modèle ML** : scikit-learn RandomForest, plutôt que PyTorch. Choix délibéré pour la lisibilité, l'interprétabilité, et parce que je n'ai pas besoin de GPU.
- **Tracking ML** : MLflow open-source, plutôt que Neptune ou W&B. Auto-hébergé, intégré nativement à scikit-learn.
- **API** : FastAPI plutôt que Flask, pour l'async, la validation Pydantic automatique, et la documentation Swagger auto-générée.
- **LLM** : NVIDIA NIM avec llama-3.2-11b, plutôt qu'OpenAI GPT-4. Gratuit, compatible on-premise, sans envoyer de données à un tiers.

---

## SLIDE 10 — A5.2 Vue d'ensemble ML

**Activité A5.2 : développer le modèle.**

La pipeline ML se déroule en 4 étapes enchaînées :

1. **C5.2.1** : construire les variables — le feature engineering
2. **C5.2.2** : sélectionner les variables pertinentes
3. **C5.2.3** : entraîner le modèle RandomForest
4. **C5.2.4** : optimiser les hyperparamètres avec GridSearchCV

Le résultat final : F1-Score de 0.857, AUC-ROC de 0.941, score GridSearchCV de 94.4%.

---

## SLIDE 11 — C5.2.1 Feature Engineering

**Compétence C5.2.1 : construire les variables.**

La question centrale était : qu'est-ce qui fait qu'un CV est compatible avec une offre ?

J'ai identifié 7 features :
- **skill_overlap** : l'indice de Jaccard entre les compétences du CV et de l'offre. C'est la feature la plus importante.
- **exp_gap** : l'écart en années d'expérience. Négatif si le candidat est sur-qualifié.
- **cv_meme_ville** : le candidat est-il dans la même ville que l'offre ? (booléen)
- **salaire_compatible** : le salaire attendu est-il dans la fourchette de l'offre ? (booléen)
- **soft_overlap** : mêmes soft skills, même méthode Jaccard
- **cv_teletravail** : le candidat préfère-t-il le télétravail ? (extrait du CV)
- **offre_teletravail** : l'offre propose-t-elle du télétravail ? (extrait de la description)

---

## SLIDE 12 — C5.2.2 Sélection des variables

**Compétence C5.2.2 : sélectionner les variables.**

Pour choisir mes features, j'ai utilisé 4 méthodes :
1. La **feature importance** du RandomForest lui-même — Gini impurity reduction
2. La **matrice de corrélation** pour détecter les redondances
3. Des **tests statistiques Mann-Whitney** pour mesurer l'impact de chaque feature sur le label
4. L'**élimination récursive RFE** avec validation croisée 5-fold

**Résultat** : les 7 features ont toutes été validées. skill_overlap domine avec une importance de 0.38, suivi de exp_gap à 0.22, et salaire_compatible à 0.15.

---

## SLIDE 13 — C5.2.3 Entraînement du modèle

**Compétence C5.2.3 : entraîner le modèle.**

Les données : 500 paires CV–Offre synthétiques, générées par dataset.py. La variable cible est un label binaire calculé par règle métier : si skill_overlap > 0.5 OU exp_gap ≤ 0, le label vaut 1 (compatible).

Le split : 80% entraînement / 20% test, stratifié pour conserver la distribution des labels.

Le code est simple, délibérément :
```python
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
```

J'utilise `class_weight='balanced'` pour gérer un léger déséquilibre entre classes compatibles et non compatibles.

---

## SLIDE 14 — C5.2.4 Optimisation hyperparamètres

**Compétence C5.2.4 : optimiser les hyperparamètres.**

J'ai utilisé GridSearchCV avec 4 hyperparamètres :
- `n_estimators` : [100, 200, 300]
- `max_depth` : [None, 5, 10, 15]
- `min_samples_split` : [2, 5, 10]
- `class_weight` : ['balanced', None]

**Meilleurs hyperparamètres trouvés** : n_estimators=300, max_depth=None, min_samples_split=2, class_weight=balanced.

Ce sont ces paramètres qui donnent les métriques finales : F1=0.857, AUC=0.941, score CV=94.4%.

Les trois leviers pour améliorer un modèle ML : les données, les hyperparamètres, et le pré-traitement — j'ai utilisé les trois.

---

## SLIDE 15 — Résultats du modèle

Les résultats dépassent les seuils objectifs du RNCP.

- **F1-Score = 0.857** (objectif ≥ 0.75 ✓)
- **AUC-ROC = 0.941** (objectif ≥ 0.80 ✓)

La matrice de confusion sur 100 paires test : 42 vrais positifs, 48 vrais négatifs, seulement 5 faux positifs et 5 faux négatifs.

Pourquoi RandomForest ? Il est résistant au surapprentissage (ensemble de 300 arbres), il gère naturellement un mix de features binaires et continues, et il fournit directement la feature importance.

La boucle s'améliore avec le temps : quand les étudiants donnent un feedback sur leurs candidatures (entretien obtenu, refus), le script `retrain_from_feedback.py` réentraîne le modèle sur ces données réelles.

---

## SLIDE 16 — Transition A5.3

Le modèle est entraîné et validé. La vrai question du déploiement arrive maintenant.

Un bon modèle qui ne tourne pas en production, ça ne sert à rien. L'activité A5.3 couvre tout : comment sauvegarder le modèle, comment le rendre accessible via une API, comment surveiller ses performances, et comment automatiser sa mise à jour.

---

## SLIDE 17 — C5.3.1 Sauvegarder le modèle

**Compétence C5.3.1 : sauvegarder le modèle.**

Trois niveaux de sauvegarde :

**Sérialisation avec joblib** : `joblib.dump(model, 'model.pkl')`. Joblib est optimisé pour les numpy arrays — bien plus rapide que pickle pour un RandomForest. Le fichier est sauvegardé dans `data_lake/models/latest/`.

**Versioning avec MLflow Model Registry** : chaque run est horodaté, avec ses paramètres, ses métriques et ses artefacts. Les versions passent par trois états : Staging → Production. v1 = entraînement initial, v2 = après réentraînement RL.

**Docker** : PostgreSQL tourne dans un conteneur isolé. FastAPI dans son propre processus. L'architecture est prête pour une dockerisation complète.

---

## SLIDE 18 — C5.3.2 Déployer via FastAPI

**Compétence C5.3.2 : déployer le modèle.**

L'API de scoring tourne sur le port 8001, avec deux endpoints principaux :

- `GET /health` : vérifie que le modèle est bien chargé, retourne la version et la liste des features attendues
- `POST /score` : reçoit un JSON avec les features, retourne le score ML, le score en pourcentage, le label (compatible/non), et la confiance

La documentation Swagger est auto-générée à `/docs` — je pourrai vous la montrer en démo.

Le processus de déploiement : on entraîne, on logue dans MLflow, on lance uvicorn, on teste les endpoints, et Streamlit appelle `/score` en temps réel.

---

## SLIDE 19 — C5.3.3 Superviser le système ML

**Compétence C5.3.3 : superviser le système ML.**

Trois outils de monitoring en place :

**MLflow Tracking** à localhost:5001 : chaque run est visible, les métriques sont comparables entre versions. Si le F1 d'un réentraînement est inférieur à 0.75, on alerte.

**Table etl_logs dans PostgreSQL** : chaque run de l'ETL loggue son statut, le nombre de nouvelles offres insérées, les doublons, la durée. Si l'API France Travail tombe, on le voit immédiatement.

**Dashboard Streamlit** : évolution des scores dans le temps, distribution des scores par type de contrat, top compétences du marché.

---

## SLIDE 20 — C5.3.4 Automatiser le cycle de vie ML

**Compétence C5.3.4 : automatiser.**

Deux pipelines automatisés :

**ETL avec APScheduler** : toutes les 2 heures, une tâche en CronTrigger extrait les nouvelles offres depuis France Travail et JobSpy, les transforme en features, les charge dans PostgreSQL, et logue l'opération dans etl_logs. C'est le carburant du système.

**Réentraînement RL supervisé** : quand un étudiant met à jour le statut d'une candidature — entretien obtenu ou refus — ce feedback devient une donnée d'entraînement. Le script `retrain_from_feedback.py` s'active si on a au moins 5 feedbacks, réentraîne sur ces données réelles combinées aux 500 paires synthétiques, et remplace le model.pkl si le nouveau F1 est supérieur à 0.75. Tout est tracé dans MLflow.

---

## SLIDE 21 — Récapitulatif Bloc 5

Voici le bilan des 11 compétences du Bloc 5. Toutes sont couvertes et démontrables :

- C5.1.1 à C5.1.3 : analyse, stratégie, choix technologiques — dans les slides précédents
- C5.2.1 à C5.2.4 : feature engineering, sélection, entraînement, GridSearchCV — dans le code et les métriques
- C5.3.1 à C5.3.4 : joblib, FastAPI, MLflow monitoring, APScheduler + RL — dans l'architecture déployée

---

## SLIDE 22 — Introduction démo

*(Cette partie est live — adaptez le discours selon ce qui se passe à l'écran.)*

Je vais maintenant vous montrer le système en action, de bout en bout. Le parcours complet d'un étudiant : de l'upload de son CV jusqu'à la candidature, en passant par le scoring ML et le chat IA.

7 étapes : démarrage, upload CV, recherche d'offres, scoring LLM, chat IA, candidature, et MLflow.

---

## SLIDE 23 — Démo 1 : Démarrage

Je lance `python start.py`.

Ce script orchestre le démarrage de toute l'infrastructure :
- Docker lance PostgreSQL sur le port 15432
- MLflow démarre sur localhost:5001
- FastAPI se lance sur localhost:8001
- Streamlit ouvre sur localhost:8501

*(Montrez les 4 terminaux ou les 4 services démarrés.)*

---

## SLIDE 24 — Démo 2 : Upload CV

Je vais dans l'onglet "Upload CV" de Streamlit et j'uploade un CV synthétique en PDF.

Voici ce qui se passe sous le capot en quelques secondes :
1. **pdfplumber** extrait le texte brut — environ 1200 caractères
2. Le NLP détecte les compétences : Python, SQL, FastAPI, Machine Learning...
3. Un hash SHA-256 est calculé pour garantir qu'on n'insère pas deux fois le même CV
4. Le tout est inséré dans la table `cvs` de PostgreSQL
5. Et l'identifiant du CV est stocké dans la session Streamlit pour la suite

---

## SLIDE 25 — Démo 3 : Recherche d'offres

*(Dans l'onglet "Recherche")*

Je saisis "Data Scientist" à Lyon, avec un filtre CDI sur les 14 derniers jours.

Le système interroge en parallèle JobSpy — qui touche LinkedIn, Indeed et Glassdoor — et l'API France Travail avec OAuth2.

On obtient jusqu'à 50 offres dédupliquées, triées par pertinence, avec le titre, l'entreprise, la localisation, le type de contrat et la date de publication. L'étudiant peut filtrer par type de contrat.

---

## SLIDE 26 — Démo 4 : Scoring LLM

*(Je coche "Analyser la compatibilité")*

Pour chaque offre sélectionnée, le LLM NVIDIA NIM — llama-3.2-11b — reçoit le CV complet et la description de l'offre, et retourne un JSON strict :
- un score de 0 à 100
- des points forts (ce que le CV a)
- des points faibles (ce qui manque)
- un conseil personnalisé

Les offres sont ensuite triées par score. Un badge vert au-dessus de 70%, orange entre 40 et 70%, rouge en dessous.

*(Montrez un exemple concret avec une offre et son analyse.)*

---

## SLIDE 27 — Démo 5 : Chat IA

*(Dans l'onglet "Chat IA")*

L'assistant est contextuel : il connaît le CV de l'étudiant, les 5 meilleures offres matchées, et l'offre cible si elle a été sélectionnée.

Je lui pose la question : *"Pour l'offre GenAI Engineer chez Younup, que dois-je ajouter dans mon CV pour passer les ATS ?"*

La réponse est personnalisée — pas générique. Le LLM sait ce que le CV contient, et ce que l'offre demande.

On peut ensuite re-uploader un CV amélioré et voir l'évolution du score en temps réel.

---

## SLIDE 28 — Démo 6 : Candidatures

*(Dans l'onglet "Mes candidatures")*

Je sélectionne une offre intéressante, je rédige une lettre de motivation avec l'aide de l'IA, et j'enregistre la candidature.

Elle apparaît dans le tableau de suivi avec le statut "nouveau". Au fil du temps, l'étudiant la fait évoluer vers "entretien" ou "refus".

Et c'est là que le RL entre en jeu : ce feedback réel est utilisé pour améliorer le modèle ML à chaque réentraînement.

---

## SLIDE 29 — Démo 7 : MLflow & FastAPI

*(Ouverture de localhost:5001 dans le navigateur)*

Voici le dashboard MLflow. On voit les deux expériences : l'entraînement initial et les runs de réentraînement RL. Chaque run est horodaté, avec ses paramètres et ses métriques.

Le Model Registry montre la v1 en Production.

*(Passage à localhost:8001/docs)*

Et voici la Swagger UI de FastAPI. Je peux tester en live le endpoint `/score` avec un vrai JSON de features — la réponse arrive en moins de 50ms.

---

## SLIDE 30 — Résultats & métriques

Pour récapituler les métriques clés du projet :

- **F1-Score = 0.857** — au-delà de l'objectif de 0.75
- **AUC-ROC = 0.941** — au-delà de l'objectif de 0.80
- **11 compétences Bloc 5** sur 11 couvertes
- **50 offres** récupérées par recherche, en temps réel
- 6 tables PostgreSQL, 5 tests statistiques formalisés, pipeline ETL toutes les 2 heures

Stack : Python 3.10, scikit-learn, MLflow 3.15, FastAPI, Streamlit, PostgreSQL 15, Docker, APScheduler, pdfplumber, NVIDIA NIM, python-jobspy.

---

## SLIDE 31 — Perspectives

Si j'avais eu plus de temps ou si je continuais ce projet, voici ce que j'aurais voulu ajouter :

- **N8n + FastAPI** : automatiser les alertes quand une offre bien scorée est publiée
- **Streamlit Cloud** : déployer l'application publiquement en quelques minutes
- **RL en production complet** : déclencher le réentraînement automatiquement si le F1 décline
- **HTTPS et sécurité** : Nginx + Let's Encrypt pour un vrai déploiement production
- **Multi-utilisateurs** : authentification + isolation des CVs par utilisateur
- **Data Drift detection** : détecter les dérives de distribution entre les features d'entraînement et les données réelles

---

## SLIDE 32 — Réflexivité et apprentissages

Ce projet m'a appris quatre choses importantes :

**La donnée, c'est la vraie contrainte.** J'avais prévu de travailler avec de vrais CVs. Le RGPD a tout changé. Le dataset synthétique était la solution — mais aussi une limite réelle que j'assume.

**Un modèle simple bien déployé vaut mieux qu'un modèle complexe non accessible.** RandomForest avec F1=0.857 et une API qui répond en 50ms, c'est infiniment plus utile qu'un transformer non déployé.

**Le MLOps représente 50% du travail.** MLflow, FastAPI, APScheduler, le réentraînement RL — j'y ai passé autant de temps qu'au modèle lui-même. C'est ce qui rend le ML réel.

**L'UX compte même en data science.** Streamlit m'a forcé à penser à l'utilisateur final à chaque itération. Ce n'est pas juste un bonus — c'est ce qui donne du sens aux métriques.

---

## SLIDE 33 — Conclusion

Pour conclure :

Le projet ATS Intelligent couvre les 11 compétences du Bloc 5 Spécialité Data Scientist, de l'analyse du besoin jusqu'à l'automatisation du cycle de vie ML.

Le modèle tourne en production — FastAPI le sert en moins de 50ms, MLflow le versionne, APScheduler alimente son dataset, et les feedbacks candidatures l'améliorent en continu.

Ce n'est pas un projet de labo. C'est une application qui peut vraiment aider un étudiant à préparer sa recherche d'emploi.

Je vous remercie pour votre attention.

Je suis disponible pour répondre à vos questions.

---

## Notes pour la démo live

**Avant la soutenance :**
- Lancer `python start.py` — attendre que les 4 services soient up
- Avoir un PDF de CV synthétique prêt à uploader
- Vérifier que FastAPI répond sur localhost:8001/health
- Vérifier que MLflow est accessible sur localhost:5001
- Avoir les onglets navigateur ouverts à l'avance

**Timing suggéré (30 min) :**
- Slides 1–5 : 4 min (intro + contexte + blocs 1–3)
- Slides 6–21 : 14 min (Bloc 5 complet, toutes compétences)
- Slides 22–29 : 8 min (démo live)
- Slides 30–33 : 4 min (bilan + perspectives + conclusion)

**Questions fréquentes du jury :**
- *"Pourquoi pas un deep learning ?"* → Interprétabilité, pas de GPU, feature importance directe
- *"Comment gérez-vous le RGPD ?"* → CVs 100% synthétiques, pas de données personnelles réelles
- *"Le modèle est-il vraiment en production ?"* → Oui, FastAPI tourne, on peut le tester en live
- *"Qu'est-ce qui améliore le modèle ?"* → Les feedbacks candidatures via retrain_from_feedback.py
- *"Pourquoi MLflow ?"* → Open-source, self-hosted, intégration native scikit-learn, Model Registry gratuit
