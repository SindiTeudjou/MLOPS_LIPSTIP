"""
FastAPI — Text Similarity Prediction Service

Loads the Production model from MLflow Model Registry at startup.
Supports both model types: SBERT (pyfunc) and TF-IDF (sklearn).
Exposes:
    GET  /health        → service + model status
    GET  /model/info    → current production model metadata
    POST /predict       → similarity score for a sentence pair
    GET  /metrics       → prediction stats + drift detection
    POST /rollback      → promote latest Archived model to Production
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "text-similarity")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
PREDICTIONS_PATH = os.getenv("PREDICTIONS_PATH", "/data/predictions.csv")

model = None
model_type: str = "unknown"
model_metadata: dict = {}


def _load_production_model():
    """Load model from MLflow, detect its type (sbert or tfidf), return (model, model_type)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise RuntimeError(f"No model found in stage '{MODEL_STAGE}'")

    v = versions[0]
    run = client.get_run(v.run_id)
    mtype = run.data.params.get("model_type", "tfidf_cosine_similarity")

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    if "sbert" in mtype:
        loaded = mlflow.pyfunc.load_model(model_uri)
    else:
        loaded = mlflow.sklearn.load_model(model_uri)

    metadata = {
        "model_name": MODEL_NAME,
        "version": v.version,
        "stage": MODEL_STAGE,
        "model_type": mtype,
        "run_id": v.run_id,
        "metrics": run.data.metrics,
    }
    return loaded, mtype, metadata


def _compute_score(s1: str, s2: str) -> float:
    """Compute similarity score using the currently loaded model."""
    if "sbert" in model_type:
        result = model.predict(pd.DataFrame({"sentence_1": [s1], "sentence_2": [s2]}))
        return float(result[0])
    else:
        v1 = model.transform([s1])
        v2 = model.transform([s2])
        return float(cosine_similarity(v1, v2)[0][0])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_type, model_metadata
    try:
        model, model_type, model_metadata = _load_production_model()
        print(f"[startup] Model loaded: {model_metadata.get('model_type')} v{model_metadata.get('version')}")
    except Exception as e:
        print(f"[startup] Model not available — {e}. Run the Airflow DAG first.")

    yield

    model = None
    model_type = "unknown"
    model_metadata = {}


app = FastAPI(
    title="Text Similarity API",
    description="Predicts semantic similarity score (0–1) between two sentences",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    sentence_1: str
    sentence_2: str


class PredictResponse(BaseModel):
    sentence_1: str
    sentence_2: str
    similarity_score: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": model_type,
    }


@app.get("/model/info")
def model_info():
    if not model_metadata:
        raise HTTPException(status_code=404, detail="No production model available yet.")
    return model_metadata


def _store_prediction(sentence_1: str, sentence_2: str, score: float) -> None:
    row = pd.DataFrame([{
        "sentence_1": sentence_1,
        "sentence_2": sentence_2,
        "similarity_score": score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }])
    write_header = not os.path.exists(PREDICTIONS_PATH) or os.path.getsize(PREDICTIONS_PATH) == 0
    row.to_csv(PREDICTIONS_PATH, mode="a", header=write_header, index=False)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Trigger the Airflow DAG to train and promote a model.",
        )

    score = _compute_score(request.sentence_1, request.sentence_2)
    _store_prediction(request.sentence_1, request.sentence_2, score)

    return PredictResponse(
        sentence_1=request.sentence_1,
        sentence_2=request.sentence_2,
        similarity_score=score,
    )


@app.get("/metrics")
def metrics():
    """Statistiques des prédictions en production + détection de drift."""
    if not os.path.exists(PREDICTIONS_PATH):
        return {"total_predictions": 0, "drift_warning": False}

    df = pd.read_csv(PREDICTIONS_PATH)
    if df.empty:
        return {"total_predictions": 0, "drift_warning": False}

    scores = df["similarity_score"].dropna()
    mean_score = float(scores.mean())
    std_score = float(scores.std()) if len(scores) > 1 else 0.0
    drift_warning = mean_score < 0.05 or mean_score > 0.80

    return {
        "total_predictions": len(df),
        "mean_score": round(mean_score, 4),
        "min_score": round(float(scores.min()), 4),
        "max_score": round(float(scores.max()), 4),
        "std_score": round(std_score, 4),
        "drift_warning": drift_warning,
        "drift_message": (
            "Score moyen hors plage normale [0.05, 0.80] — possible drift de distribution"
            if drift_warning else None
        ),
    }


@app.post("/rollback")
def rollback():
    """Promeut le dernier modèle Archived en Production et recharge le modèle en mémoire."""
    global model, model_type, model_metadata

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    archived_versions = client.get_latest_versions(MODEL_NAME, stages=["Archived"])

    if not archived_versions:
        raise HTTPException(status_code=404, detail="Aucun modèle archivé disponible pour le rollback.")

    latest_archived = max(archived_versions, key=lambda v: int(v.version))

    if prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=prod_versions[0].version, stage="Archived"
        )

    client.transition_model_version_stage(
        name=MODEL_NAME, version=latest_archived.version, stage="Production"
    )

    model, model_type, model_metadata = _load_production_model()

    return {
        "message": f"Rollback effectué vers la version {latest_archived.version}",
        "new_production_version": latest_archived.version,
        "previous_production_version": prod_versions[0].version if prod_versions else None,
    }
