"""
Core ML logic for text similarity scoring.
Each DAG run trains TWO models (SBERT + TF-IDF), evaluates both, and promotes the best.
This makes the MLflow comparison and promotion logic visible and meaningful.

Each function is called as a separate Airflow task via PythonOperator.
"""

import hashlib
import os
import subprocess

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DATA_PATH = os.getenv("DATA_PATH", "/data/dataset.csv")
PREDICTIONS_PATH = os.getenv("PREDICTIONS_PATH", "/data/predictions.csv")
EXPERIMENT_NAME = "text-similarity"
MODEL_NAME = "text-similarity"

TRAIN_PATH = "/tmp/train.csv"
TEST_PATH = "/tmp/test.csv"
DVC_ROOT = "/data"


# ── DVC helpers ───────────────────────────────────────────────────────────────

def _dvc_version_dataset():
    try:
        r = subprocess.run(
            ["dvc", "add", "dataset.csv"],
            cwd=DVC_ROOT, capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"[dvc] dvc add échoué : {r.stderr.strip()}")
            return
        subprocess.run(["dvc", "push"], cwd=DVC_ROOT, capture_output=True, text=True)
        print("[dvc] Dataset versionné et poussé dans le store local.")
    except FileNotFoundError:
        print("[dvc] DVC non disponible, versioning ignoré.")


def _get_dvc_hash():
    import yaml
    dvc_file = os.path.join(DVC_ROOT, "dataset.csv.dvc")
    if not os.path.exists(dvc_file):
        return None
    with open(dvc_file) as f:
        meta = yaml.safe_load(f)
    outs = meta.get("outs", [])
    return outs[0].get("md5") if outs else None


# ── SBERT pyfunc wrapper ──────────────────────────────────────────────────────

class SBertSimilarityWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for SentenceTransformer.
    Input : DataFrame avec colonnes 'sentence_1' et 'sentence_2'.
    Output : liste de scores cosine similarity en [0, 1].
    """

    def load_context(self, context):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(context.artifacts["sbert_model"])

    def predict(self, context, model_input):
        s1 = model_input["sentence_1"].tolist()
        s2 = model_input["sentence_2"].tolist()
        emb1 = self.model.encode(s1, convert_to_numpy=True)
        emb2 = self.model.encode(s2, convert_to_numpy=True)
        return [float(cosine_similarity([e1], [e2])[0][0]) for e1, e2 in zip(emb1, emb2)]


# ── Tâche 0 : Feedback loop ───────────────────────────────────────────────────

def incorporate_feedback(**context):
    """Append API predictions to the training dataset (silver labels feedback loop)."""
    if not os.path.exists(PREDICTIONS_PATH):
        print("[feedback] No predictions file found. Skipping.")
        return

    preds_df = pd.read_csv(PREDICTIONS_PATH)
    preds_df = preds_df[["sentence_1", "sentence_2", "similarity_score"]].dropna()

    if preds_df.empty:
        print("[feedback] No predictions to incorporate.")
        return

    main_df = pd.read_csv(DATA_PATH, sep=";")
    combined = pd.concat([main_df, preds_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sentence_1", "sentence_2"])

    added = len(combined) - len(main_df)
    combined.to_csv(DATA_PATH, sep=";", index=False)

    pd.DataFrame(columns=["sentence_1", "sentence_2", "similarity_score", "timestamp"]).to_csv(
        PREDICTIONS_PATH, index=False
    )
    print(f"[feedback] {added} nouvelles paires incorporées dans le dataset.")
    _dvc_version_dataset()


# ── Tâche 1 : Ingestion ───────────────────────────────────────────────────────

def ingest_data(**context):
    """Load raw CSV, validate structure + score range, push dataset hash via XCom."""
    df = pd.read_csv(DATA_PATH, sep=";")

    required_cols = {"sentence_1", "sentence_2", "similarity_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    out_of_range = df["similarity_score"].between(0, 1, inclusive="both").eq(False).sum()
    if out_of_range > 0:
        print(f"Warning: {out_of_range} rows have scores outside [0, 1] — will be clipped.")

    print(f"[ingest] Loaded {len(df)} rows")
    print(f"[ingest] Score stats: min={df['similarity_score'].min():.3f}, "
          f"max={df['similarity_score'].max():.3f}, "
          f"mean={df['similarity_score'].mean():.3f}")

    dvc_hash = _get_dvc_hash()
    if dvc_hash:
        data_hash = dvc_hash[:8]
        print(f"[ingest] DVC hash : {data_hash}")
    else:
        with open(DATA_PATH, "rb") as f:
            data_hash = hashlib.md5(f.read()).hexdigest()[:8]
        print(f"[ingest] MD5 hash : {data_hash} (DVC non initialisé)")

    context["ti"].xcom_push(key="raw_row_count", value=len(df))
    context["ti"].xcom_push(key="data_hash", value=data_hash)


# ── Tâche 2 : Cleaning & Preprocessing ───────────────────────────────────────

def clean_and_preprocess(**context):
    """Clean data, deduplicate, split into train/test and persist to /tmp."""
    df = pd.read_csv(DATA_PATH, sep=";")
    n_before = len(df)

    df = df.dropna(subset=["sentence_1", "sentence_2", "similarity_score"])
    df["sentence_1"] = df["sentence_1"].str.strip()
    df["sentence_2"] = df["sentence_2"].str.strip()
    df["similarity_score"] = df["similarity_score"].clip(0.0, 1.0)
    df = df.drop_duplicates(subset=["sentence_1", "sentence_2"])

    print(f"[preprocess] Rows before: {n_before} → after: {len(df)}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"[preprocess] Train: {len(train_df)} rows | Test: {len(test_df)} rows")


# ── Tâche 3 : Training — entraîne TF-IDF ET SBERT ────────────────────────────

def train_model(**context):
    """
    Entraîne les deux modèles (TF-IDF et SBERT) sur le même dataset.
    Chaque modèle est enregistré comme une version séparée dans MLflow.
    Les run_ids ET les versions MLflow sont poussés via XCom.
    """
    train_df = pd.read_csv(TRAIN_PATH)
    data_hash = context["ti"].xcom_pull(task_ids="ingest_data", key="data_hash")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("[train] Entraînement TF-IDF...")
    run_id_tfidf, version_tfidf = _train_tfidf(train_df, data_hash)

    print("[train] Entraînement SBERT (all-MiniLM-L6-v2)...")
    run_id_sbert, version_sbert = _train_sbert(train_df, data_hash)

    context["ti"].xcom_push(key="run_id_tfidf", value=run_id_tfidf)
    context["ti"].xcom_push(key="run_id_sbert", value=run_id_sbert)
    context["ti"].xcom_push(key="version_tfidf", value=version_tfidf)
    context["ti"].xcom_push(key="version_sbert", value=version_sbert)


def _train_tfidf(train_df: pd.DataFrame, data_hash: str) -> tuple:
    params = {
        "model_type": "tfidf_cosine_similarity",
        "ngram_range": "(1, 2)",
        "max_features": 50000,
        "test_size": 0.2,
        "random_state": 42,
        "train_rows": len(train_df),
    }
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    train_texts = list(train_df["sentence_1"]) + list(train_df["sentence_2"])
    vectorizer.fit(train_texts)

    with mlflow.start_run(run_name="tfidf") as run:
        for k, v in params.items():
            mlflow.log_param(k, v)
        if data_hash:
            mlflow.set_tag("dataset_hash", data_hash)
        mlflow.sklearn.log_model(
            vectorizer, artifact_path="model", registered_model_name=MODEL_NAME
        )
        run_id = run.info.run_id

    # Récupère la version par run_id (plus fiable que model_info.registered_model_version)
    client = MlflowClient()
    versions = client.search_model_versions(f"run_id='{run_id}'")
    version = versions[0].version

    print(f"[train] TF-IDF run_id: {run_id} → version MLflow: {version}")
    return run_id, version


def _train_sbert(train_df: pd.DataFrame, data_hash: str) -> tuple:
    from sentence_transformers import SentenceTransformer

    params = {
        "model_type": "sbert_cosine_similarity",
        "sbert_model": "all-MiniLM-L6-v2",
        "test_size": 0.2,
        "random_state": 42,
        "train_rows": len(train_df),
    }
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    model_dir = "/tmp/sbert_model"
    sbert.save(model_dir)

    with mlflow.start_run(run_name="sbert") as run:
        for k, v in params.items():
            mlflow.log_param(k, v)
        if data_hash:
            mlflow.set_tag("dataset_hash", data_hash)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SBertSimilarityWrapper(),
            artifacts={"sbert_model": model_dir},
            registered_model_name=MODEL_NAME,
            pip_requirements=["sentence-transformers==2.7.0", "scikit-learn==1.4.0"],
        )
        run_id = run.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(f"run_id='{run_id}'")
    version = versions[0].version

    print(f"[train] SBERT run_id: {run_id} → version MLflow: {version}")
    return run_id, version


# ── Tâche 4 : Evaluation ──────────────────────────────────────────────────────

def evaluate_model(**context):
    """
    Évalue les deux modèles (TF-IDF et SBERT) sur le test set.
    Logue les métriques dans MLflow et passe les deux en Staging.
    Les versions MLflow sont récupérées depuis XCom (capturées lors du training).
    """
    run_id_tfidf = context["ti"].xcom_pull(task_ids="train_model", key="run_id_tfidf")
    run_id_sbert = context["ti"].xcom_pull(task_ids="train_model", key="run_id_sbert")
    version_tfidf = context["ti"].xcom_pull(task_ids="train_model", key="version_tfidf")
    version_sbert = context["ti"].xcom_pull(task_ids="train_model", key="version_sbert")
    test_df = pd.read_csv(TEST_PATH)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"[evaluate] TF-IDF : run={run_id_tfidf}, version={version_tfidf}")
    pearson_tfidf = _evaluate_single(run_id_tfidf, "tfidf_cosine_similarity", version_tfidf, test_df)

    print(f"[evaluate] SBERT : run={run_id_sbert}, version={version_sbert}")
    pearson_sbert = _evaluate_single(run_id_sbert, "sbert_cosine_similarity", version_sbert, test_df)

    print(f"[evaluate] TF-IDF Pearson r = {pearson_tfidf:.4f}  |  SBERT Pearson r = {pearson_sbert:.4f}")

    context["ti"].xcom_push(key="version_tfidf", value=version_tfidf)
    context["ti"].xcom_push(key="version_sbert", value=version_sbert)
    context["ti"].xcom_push(key="pearson_tfidf", value=pearson_tfidf)
    context["ti"].xcom_push(key="pearson_sbert", value=pearson_sbert)


def _evaluate_single(run_id: str, model_type: str, version: str, test_df: pd.DataFrame) -> float:
    """Évalue un modèle, logue ses métriques, le passe en Staging. Retourne pearson_r."""
    client = MlflowClient()

    preds = _predict_batch(run_id, model_type, test_df)
    true = test_df["similarity_score"].values

    metrics = {
        "mae": float(np.mean(np.abs(preds - true))),
        "mse": float(np.mean((preds - true) ** 2)),
        "pearson_r": float(pearsonr(preds, true)[0]),
        "spearman_r": float(spearmanr(preds, true)[0]),
    }

    with mlflow.start_run(run_id=run_id):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    print(f"[evaluate] {model_type} v{version} → {metrics}")

    # Passe la bonne version en Staging (version capturée à l'entraînement, pas de recherche)
    client.transition_model_version_stage(name=MODEL_NAME, version=version, stage="Staging")
    print(f"[evaluate] v{version} ({model_type}) → Staging")

    return metrics["pearson_r"]


def _predict_batch(run_id: str, model_type: str, test_df: pd.DataFrame) -> np.ndarray:
    """Charge le modèle et prédit la similarité pour toutes les lignes du test set."""
    if "sbert" in model_type:
        loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        scores = loaded.predict(pd.DataFrame({
            "sentence_1": test_df["sentence_1"].tolist(),
            "sentence_2": test_df["sentence_2"].tolist(),
        }))
        return np.array(scores, dtype=float)
    else:
        vectorizer = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        preds = []
        for _, row in test_df.iterrows():
            v1 = vectorizer.transform([row["sentence_1"]])
            v2 = vectorizer.transform([row["sentence_2"]])
            preds.append(float(cosine_similarity(v1, v2)[0][0]))
        return np.array(preds)


# ── Tâche 5 : Promotion ───────────────────────────────────────────────────────

def promote_if_better(**context):
    """
    Compare TF-IDF vs SBERT (nouveaux) et le modèle Production existant via Pearson r.
    - Le perdant des deux nouveaux → Archived
    - Le gagnant des deux nouveaux est comparé au Production actuel :
        - Si meilleur → Production (nouveau), Archived (ancien)
        - Si moins bon → Archived (nouveau), Production inchangé
    """
    version_tfidf = context["ti"].xcom_pull(task_ids="evaluate_model", key="version_tfidf")
    version_sbert = context["ti"].xcom_pull(task_ids="evaluate_model", key="version_sbert")
    pearson_tfidf = context["ti"].xcom_pull(task_ids="evaluate_model", key="pearson_tfidf")
    pearson_sbert = context["ti"].xcom_pull(task_ids="evaluate_model", key="pearson_sbert")

    client = MlflowClient()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Étape 1 : comparaison entre les deux nouveaux modèles
    if pearson_sbert >= pearson_tfidf:
        best_version, best_pearson = version_sbert, pearson_sbert
        worst_version, worst_type = version_tfidf, "tfidf"
        best_type = "sbert"
    else:
        best_version, best_pearson = version_tfidf, pearson_tfidf
        worst_version, worst_type = version_sbert, "sbert"
        best_type = "tfidf"

    print(f"[promote] Comparaison : SBERT Pearson={pearson_sbert:.4f}  vs  TF-IDF Pearson={pearson_tfidf:.4f}")
    print(f"[promote] Meilleur nouveau modèle : {best_type} (v{best_version})")

    # Archive le perdant des deux nouveaux
    client.transition_model_version_stage(name=MODEL_NAME, version=worst_version, stage="Archived")
    print(f"[promote] v{worst_version} ({worst_type}) → Archived (perdant de la comparaison)")

    # Étape 2 : compare le gagnant avec le Production actuel
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

    if not prod_versions:
        client.transition_model_version_stage(name=MODEL_NAME, version=best_version, stage="Production")
        print(f"[promote] Aucun modèle en Production. v{best_version} ({best_type}) → Production.")
        return

    prod_version = prod_versions[0]
    prod_run = client.get_run(prod_version.run_id)
    prod_pearson = prod_run.data.metrics.get("pearson_r", -1.0)
    prod_type = prod_run.data.params.get("model_type", "unknown")

    print(f"[promote] Production actuel : v{prod_version.version} ({prod_type}) Pearson={prod_pearson:.4f}")

    if best_pearson > prod_pearson:
        client.transition_model_version_stage(name=MODEL_NAME, version=prod_version.version, stage="Archived")
        client.transition_model_version_stage(name=MODEL_NAME, version=best_version, stage="Production")
        print(
            f"[promote] NOUVEAU modèle v{best_version} ({best_type}) → Production "
            f"(Pearson {best_pearson:.4f} > {prod_pearson:.4f}). "
            f"Ancien v{prod_version.version} → Archived."
        )
    else:
        client.transition_model_version_stage(name=MODEL_NAME, version=best_version, stage="Archived")
        print(
            f"[promote] Nouveau modèle v{best_version} NON promu "
            f"(Pearson {best_pearson:.4f} ≤ {prod_pearson:.4f}). "
            f"Production reste v{prod_version.version} ({prod_type})."
        )
