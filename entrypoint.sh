#!/bin/bash
set -e

chmod -R 777 /mlflow 2>/dev/null || true
chmod -R 777 /data 2>/dev/null || true

# Initialise DVC dans /data si pas encore fait
if [ ! -d "/data/.dvc" ]; then
  cd /data && dvc init --no-scm && dvc remote add -d localstore /data/dvc-store
  cd /opt/airflow
fi

# Versionne le dataset initial avec DVC
if [ -f "/data/dataset.csv" ] && [ ! -f "/data/dataset.csv.dvc" ]; then
  cd /data && dvc add dataset.csv && dvc push || true
  cd /opt/airflow
fi

airflow db migrate

airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname Admin \
  --role Admin --email admin@example.com 2>/dev/null || true

airflow webserver &
airflow scheduler
