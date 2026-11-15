# The shared MLflow tracking server (KAN-107). Runbook: docs/runbooks/mlflow-server.md.
# The official image has no Postgres or GCS driver; the backend store and artifact store need both.
FROM ghcr.io/mlflow/mlflow:v2.17.2
RUN pip install --no-cache-dir psycopg2-binary google-cloud-storage
