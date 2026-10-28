# Airflow with the primer package installed. Built by docker-compose.yaml at the repo root.
FROM apache/airflow:3.3.2-python3.12

WORKDIR /opt/primer
COPY --chown=airflow:root pyproject.toml README.md ./
COPY --chown=airflow:root src ./src

# Editable install: docker-compose.yaml mounts ./src over /opt/primer/src, so code changes apply
# without a rebuild. Pinning apache-airflow stops our dependencies from changing the Airflow version.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -e .
