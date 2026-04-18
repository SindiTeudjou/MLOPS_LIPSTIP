"""
Airflow DAG — Text Similarity Training Pipeline

Flow:
    incorporate_feedback → ingest_data → clean_and_preprocess
    → train_model (TF-IDF + SBERT)
    → evaluate_model (les deux évalués, les deux passent en Staging)
    → promote_if_better (compare les deux + Production actuel → meilleur promu)

MLflow model lifecycle stages used:
    None       → modèle enregistré juste après l'entraînement
    Staging    → évalué, métriques loggées, en attente de comparaison
    Production → meilleur modèle actuel, servi par l'API
    Archived   → perdant de la comparaison ou ancien Production
"""

import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/src")

from train import (
    clean_and_preprocess,
    evaluate_model,
    incorporate_feedback,
    ingest_data,
    promote_if_better,
    train_model,
)

default_args = {
    "owner": "mlops",
    "retries": 1,
}

with DAG(
    dag_id="text_similarity_pipeline",
    description="End-to-end training pipeline for text pair similarity scoring",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",   # Re-train weekly when new data arrives
    catchup=False,
    tags=["mlops", "text-similarity", "baseline"],
) as dag:

    t0 = PythonOperator(
        task_id="incorporate_feedback",
        python_callable=incorporate_feedback,
    )

    t1 = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    t2 = PythonOperator(
        task_id="clean_and_preprocess",
        python_callable=clean_and_preprocess,
    )

    t3 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t4 = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    t5 = PythonOperator(
        task_id="promote_if_better",
        python_callable=promote_if_better,
    )

    t0 >> t1 >> t2 >> t3 >> t4 >> t5
