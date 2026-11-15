# The shared MLflow tracking server (KAN-107). Runbook: docs/runbooks/mlflow-server.md.
# The official image has no Postgres or GCS driver; the backend store and artifact store need both.
# Pinned so a rebuild on a fresh VM gives the server it had. These are the latest releases on PyPI
# on 2026-11-15, the day the server was first built, so they should match what runs now; confirm
# with `sudo docker-compose exec mlflow pip freeze | grep -E "psycopg2|google-cloud-storage"`.
FROM ghcr.io/mlflow/mlflow:v2.17.2
RUN pip install --no-cache-dir psycopg2-binary==2.9.13 google-cloud-storage==3.17.1
