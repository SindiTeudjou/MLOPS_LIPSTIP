# Text Similarity Scoring — Pipeline MLOps

Pipeline MLOps complet pour la prédiction de similarité sémantique entre deux phrases.
Orchestré par **Airflow**, tracké par **MLflow**, versionné par **DVC**, servi par **FastAPI**, le tout conteneurisé avec **Docker Compose**.

---

## Sommaire

- [Architecture](#architecture)
- [Stack technique](#stack-technique)
- [Prérequis](#prérequis)
- [Installation et démarrage](#installation-et-démarrage)
- [Lancer le pipeline d'entraînement](#lancer-le-pipeline-dentraînement)
- [Suivre les expériences dans MLflow](#suivre-les-expériences-dans-mlflow)
- [Utiliser l'API de prédiction](#utiliser-lapi-de-prédiction)
- [Surveillance et rollback](#surveillance-et-rollback)
- [Versioning des données avec DVC](#versioning-des-données-avec-dvc)
- [Boucle de feedback](#boucle-de-feedback)
- [Choisir le modèle](#choisir-le-modèle)
- [Structure du projet](#structure-du-projet)
- [Architecture cible complète](#architecture-cible-complète)
- [Étapes suivantes](#étapes-suivantes)

---

## Architecture

```
[Airflow DAG — text_similarity_pipeline]
    │
    ├── 0. incorporate_feedback   ← Intègre les prédictions API dans le dataset + DVC version
    ├── 1. ingest_data            ← Validation CSV, lecture hash DVC → tag MLflow
    ├── 2. clean_and_preprocess   ← Nettoyage, déduplication, split 80/20
    ├── 3. train_model            ← Entraîne TF-IDF ET SBERT → 2 versions MLflow (stage: None)
    ├── 4. evaluate_model         ← Évalue les deux → MAE, MSE, Pearson r → MLflow (stage: Staging)
    └── 5. promote_if_better      ← Compare TF-IDF vs SBERT vs Production → meilleur promu

[FastAPI]  ──── charge models:/text-similarity/Production depuis MLflow
           ──── détecte le type de modèle (SBERT ou TF-IDF) automatiquement
           ──── stocke chaque prédiction dans /data/predictions.csv

[DVC]      ──── versionne /data/dataset.csv dans /data/dvc-store/
           ──── hash loggé comme tag MLflow sur chaque run
```

### Cycle de vie des modèles MLflow

| Stage        | Signification                                                     |
|--------------|-------------------------------------------------------------------|
| `None`       | Modèle enregistré juste après l'entraînement                      |
| `Staging`    | Évalué — métriques loggées, en attente de comparaison             |
| `Production` | Meilleur modèle actuel, utilisé par l'API FastAPI                 |
| `Archived`   | Ancien modèle Production ou nouveau modèle moins performant       |

### Boucle de réentraînement automatique

```
Utilisateur appelle /predict
    → score stocké dans predictions.csv
    → DAG hebdomadaire : incorporate_feedback ajoute les paires au dataset
    → DVC versionne le nouveau dataset
    → Modèle réentraîné sur données enrichies
    → Si meilleur Pearson r → promu en Production
```

---

## Stack technique

| Outil | Rôle | Justification |
|-------|------|---------------|
| **Apache Airflow 2.8** | Orchestration du pipeline | Standard industrie, DAG = étapes claires avec retry |
| **MLflow 2.10** | Tracking expériences + Model Registry | Open-source, suivi complet, gestion des stages |
| **DVC 3.49** | Versioning des données | Hash MD5 par version, store local, traçabilité dataset ↔ modèle |
| **FastAPI** | API de prédiction + monitoring | Async Python moderne, docs OpenAPI auto-générées |
| **SentenceTransformer** (`all-MiniLM-L6-v2`) | Modèle principal | Embeddings sémantiques 384d, CPU, Pearson r ~0.80 |
| **TF-IDF + cosine similarity** | Modèle baseline alternatif | Rapide, interprétable, sans dépendance lourde |
| **Docker Compose** | Conteneurisation | Tous les services démarrent en une commande |

**Métrique de promotion :** Pearson r (standard pour les benchmarks STS comme STS-B).

**Modèle actif par défaut :** `all-MiniLM-L6-v2` (SBERT). Configurable via `MODEL_TYPE` dans `docker-compose.yml`.

---

## Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et démarré
- Docker Compose v2+

---

## Installation et démarrage

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd MLOPS_LIPSTIP
```

### 2. Placer le dataset

```bash
cp /chemin/vers/dataset.csv data/dataset.csv
```

Le fichier CSV doit contenir les colonnes suivantes (séparateur `;`) :

| Colonne | Type | Description |
|---------|------|-------------|
| `sentence_1` | string | Première phrase |
| `sentence_2` | string | Deuxième phrase |
| `similarity_score` | float [0, 1] | Score de similarité réel |

### 3. Démarrer tous les services

```bash
docker compose up --build
```

> Le premier démarrage prend environ 5–10 minutes (build des images + installation de PyTorch/sentence-transformers + initialisation DVC).

Au démarrage, Airflow exécute automatiquement :
- Initialisation de DVC dans `/data/` avec un store local
- Versionnage initial du dataset (`dataset.csv.dvc` créé)
- Migration de la base de données Airflow
- Création de l'utilisateur admin

### 4. Vérifier que tout est en ligne

| Service | URL | Identifiants |
|---------|-----|--------------|
| Airflow (orchestration) | http://localhost:8080 | `admin` / `admin` |
| MLflow (tracking) | http://localhost:5000 | — |
| FastAPI (prédiction) | http://localhost:8000 | — |
| FastAPI Docs (Swagger) | http://localhost:8000/docs | — |

---

## Lancer le pipeline d'entraînement

### Via l'interface Airflow — http://localhost:8080

1. Ouvrir http://localhost:8080 → identifiants : `admin` / `admin`
2. Localiser le DAG **`text_similarity_pipeline`**
3. Cliquer sur le bouton **▶ (Trigger DAG)** à droite
4. Confirmer dans le popup → **Trigger DAG**

### Via le terminal

```bash
docker compose exec airflow airflow dags trigger text_similarity_pipeline
```

### Ce que fait le pipeline (6 tâches)

| # | Tâche | Action |
|---|-------|--------|
| 0 | `incorporate_feedback` | Incorpore les prédictions API dans le dataset + `dvc add` + `dvc push` |
| 1 | `ingest_data` | Charge le CSV, valide les colonnes/scores, lit le hash DVC → tag MLflow |
| 2 | `clean_and_preprocess` | Supprime les nulls, normalise le texte, split 80/20 |
| 3 | `train_model` | Entraîne **TF-IDF ET SBERT** (`all-MiniLM-L6-v2`), enregistre 2 versions dans MLflow |
| 4 | `evaluate_model` | Calcule MAE, MSE, Pearson r, Spearman r → passe en `Staging` |
| 5 | `promote_if_better` | Compare Pearson r avec `Production` → promeut ou archive |

> **Note :** lors du premier run SBERT, le modèle `all-MiniLM-L6-v2` (~90 Mo) est téléchargé depuis HuggingFace. Les runs suivants utilisent le cache.

---

## Suivre les expériences dans MLflow

Ouvrir **http://localhost:5000**

### Expériences

- Cliquer sur **`text-similarity`** dans la colonne de gauche
- Chaque run affiche :
  - **Params :** `model_type`, `sbert_model` (ou `ngram_range`, `max_features` pour TF-IDF), `train_rows`
  - **Metrics :** `mae`, `mse`, `pearson_r`, `spearman_r`
  - **Tags :** `dataset_hash` (version DVC du dataset utilisé)

### Modèles enregistrés

- Onglet **Models** → `text-similarity`
- Chaque version affiche son stage : `Staging`, `Production`, `Archived`
- Le tag `dataset_hash` relie chaque modèle à la version exacte du dataset qui l'a produit

---

## Utiliser l'API de prédiction

### Documentation interactive

Ouvrir **http://localhost:8000/docs** → bouton **"Try it out"**

### Endpoints

#### `GET /health` — Santé du service

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_loaded": true, "model_type": "sbert_cosine_similarity"}
```

#### `GET /model/info` — Informations sur le modèle en production

```bash
curl http://localhost:8000/model/info
```
```json
{
  "model_name": "text-similarity",
  "version": "10",
  "stage": "Production",
  "model_type": "sbert_cosine_similarity",
  "run_id": "abc123...",
  "metrics": {"mae": 0.14, "mse": 0.03, "pearson_r": 0.81, "spearman_r": 0.79}
}
```

#### `POST /predict` — Prédire un score de similarité

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sentence_1": "machine learning is great", "sentence_2": "deep learning is powerful"}'
```
```json
{
  "sentence_1": "machine learning is great",
  "sentence_2": "deep learning is powerful",
  "similarity_score": 0.78
}
```

> **Important :** Le modèle est entraîné sur des données en **anglais**. Chaque prédiction est automatiquement stockée dans `data/predictions.csv` pour alimenter le prochain cycle d'entraînement.

### Exemples de prédictions

| sentence_1 | sentence_2 | Score attendu (SBERT) |
|-----------|-----------|---------------|
| `"I love cats"` | `"I like cats"` | Élevé (~0.95) |
| `"machine learning is great"` | `"deep learning is powerful"` | Élevé (~0.78) |
| `"I love cats"` | `"The car is red"` | Faible (~0.05) |

---

## Surveillance et rollback

#### `GET /metrics` — Statistiques des prédictions + détection de drift

```bash
curl http://localhost:8000/metrics
```
```json
{
  "total_predictions": 42,
  "mean_score": 0.61,
  "min_score": 0.02,
  "max_score": 0.97,
  "std_score": 0.28,
  "drift_warning": false,
  "drift_message": null
}
```

**Alerte drift** : `drift_warning: true` si le score moyen sort de la plage `[0.05, 0.80]`. Action recommandée : déclencher manuellement le DAG.

#### `POST /rollback` — Revenir au modèle précédent

```bash
curl -X POST http://localhost:8000/rollback
```
```json
{
  "message": "Rollback effectué vers la version 9",
  "new_production_version": "9",
  "previous_production_version": "10"
}
```

Le rollback promeut le dernier modèle `Archived` en `Production` et recharge le modèle en mémoire **sans redémarrage de l'API**.

---

## Versioning des données avec DVC

DVC est initialisé dans `/data/` avec un store local (`/data/dvc-store/`).

### Fichiers créés

```
data/
├── dataset.csv         ← dataset actuel
├── dataset.csv.dvc     ← métadonnées DVC (hash MD5 + taille)
├── dvc-store/          ← copies versionnées du dataset
└── .dvc/               ← configuration DVC
```

### Traçabilité dataset ↔ modèle

Chaque run MLflow contient le tag `dataset_hash` — le hash MD5 du dataset utilisé pour l'entraînement. Cela permet de savoir exactement quelle version des données a produit quel modèle.

### Restaurer une version

```bash
# Depuis le container Airflow
docker compose exec airflow bash -c "cd /data && dvc checkout"
```

---

## Boucle de feedback

À chaque appel à `/predict`, la paire de phrases + le score sont stockés dans `data/predictions.csv`.

Au prochain déclenchement du DAG, la tâche `incorporate_feedback` :
1. Lit `predictions.csv`
2. Ajoute les nouvelles paires au `dataset.csv` (silver labels)
3. Appelle `dvc add dataset.csv` + `dvc push` pour versionner le dataset enrichi
4. Vide `predictions.csv`

Le modèle suivant est ainsi entraîné sur des données plus représentatives de l'usage réel.

> **Limite :** Les scores viennent du modèle lui-même (silver labels). Risque de biais cumulatif si le modèle fait des erreurs systématiques. En production, prévoir une validation humaine sur un échantillon avant incorporation.

---

## Comparaison automatique des modèles

À chaque run du DAG, les **deux modèles sont entraînés et comparés** :

| Modèle | Pearson r attendu | Temps d'entraînement |
|--------|-------------------|----------------------|
| TF-IDF ngram (1,2) + cosine | ~0.33 | ~10 sec |
| SBERT `all-MiniLM-L6-v2` + cosine | ~0.80 | ~2–5 min (CPU) |

**Logique de promotion dans `promote_if_better` :**
1. TF-IDF et SBERT sont comparés entre eux → le perdant passe en `Archived`
2. Le gagnant est comparé au modèle en `Production` → promu si meilleur Pearson r

Dans MLflow, chaque run affiche **deux nouvelles versions** avec leurs métriques respectives, ce qui rend la comparaison et la décision de promotion entièrement traçables.

---

## Structure du projet

```
MLOPS_LIPSTIP/
├── dags/
│   └── training_pipeline.py    # DAG Airflow (6 tâches séquentielles)
├── src/
│   └── train.py                # Logique ML : SBERT + TF-IDF + intégration DVC
├── api/
│   └── main.py                 # FastAPI : predict, metrics, rollback
├── data/
│   ├── dataset.csv             # Dataset (à placer ici)
│   ├── dataset.csv.dvc         # Métadonnées DVC (généré au 1er démarrage)
│   ├── dvc-store/              # Store DVC local (généré au 1er démarrage)
│   └── predictions.csv         # Prédictions API (feedback loop)
├── entrypoint.sh               # Script de démarrage Airflow (DVC init + db migrate)
├── Dockerfile.airflow           # Image Airflow + ML dependencies + DVC + sentence-transformers
├── Dockerfile.api               # Image FastAPI + sentence-transformers
├── docker-compose.yml           # Orchestration des 3 services
├── requirements.train.txt       # pandas, scikit-learn, scipy, mlflow, dvc, sentence-transformers
└── requirements.api.txt         # fastapi, uvicorn, scikit-learn, mlflow, sentence-transformers
```

---

## Architecture cible complète

Le pipeline actuel couvre les 10 composantes d'un système MLOps complet :

| # | Composante | Implémentation |
|---|-----------|----------------|
| 1 | Acquisition / ingestion de texte brut | `ingest_data` — charge et valide le CSV |
| 2 | Génération de paires candidates | Dataset pré-construit + paires issues des appels API |
| 3 | Création / validation des scores | Validation plage [0, 1] + silver labels via `/predict` |
| 4 | Stockage et versioning des données | **DVC** — hash par version, store local, tag MLflow |
| 5 | Entraînement et évaluation | `train_model` + `evaluate_model` (MAE, MSE, Pearson r) |
| 6 | Comparaison et décision de mise en prod | `promote_if_better` — compare Pearson r |
| 7 | Déploiement en production | FastAPI sert le modèle `Production` depuis MLflow |
| 8 | Surveillance + rollback + feedback | `/metrics` (drift), `/rollback`, stockage prédictions |
| 9 | Réutilisation des données notées | `incorporate_feedback` — boucle de réentraînement |
| 10 | Extraction d'informations en aval | Paires avec score > 0.8 → base pour graphe de connaissances |

---

## Étapes suivantes

1. **DVC remote distant** — Migrer le store DVC vers S3/GCS pour un accès multi-nœuds
2. **Validation humaine** — Ajouter une interface d'annotation (Label Studio) pour valider les silver labels avant incorporation
3. **Infrastructure production** — LocalExecutor + PostgreSQL pour Airflow, S3 pour les artefacts MLflow
4. **Monitoring avancé** — Intégrer Grafana/Prometheus pour des alertes automatiques sur le drift
5. **Cross-encoder** — Remplacer SBERT bi-encoder par `cross-encoder/stsb-roberta-base` pour un Pearson r > 0.85
