# Airflow DAGs

The data pipeline runs on Apache Airflow 3.3 in Docker Compose. The DAGs only wire tasks together;
every task calls a function in [`src/primer/data/orchestration.py`](../src/primer/data/orchestration.py),
which is tested without Airflow (`tests/data/test_orchestration.py`).

## DAGs

### `primer_data_pipeline`

```
tinystories        ingest → validate → preprocess ─┐
quickdraw          ingest → validate → preprocess ─┤
clear_corpus       ingest → validate → preprocess ─┤
sight_words        ingest → validate → preprocess ─┼──→ summarize
kuperman_aoa       ingest → validate → preprocess ─┤
storyweaver        ingest → validate → preprocess ─┤
african_storybook  ingest → validate → preprocess ─┘
```

- The seven source groups run **in parallel**; inside a group the stages run in order.
- **ingest** fetches the next batch into `raw/`. When a source has nothing new (an unchanged
  snapshot, or an incremental source that is fully ingested) the task is **skipped**, and so are the
  rest of that source's tasks. Downloads retry 3 times with exponential backoff.
- **validate** runs the source's checks. A failed check **fails** the task (no retry: the data won't
  change), so bad data never reaches `processed/`.
- **preprocess** writes the cleaned batch to `processed/` in the dataset's schema.
- **summarize** always runs once every source has finished. It records each source's outcome
  (`success`, `skipped`, or `failed` with the stage) and row counts per dataset and source in
  `metadata/reports/<run_id>.json`, then **fails the run** if any source failed, naming which ones.
  So a partial run is red but still leaves a report of what landed.
- `max_active_runs=1`: incremental sources keep a position, so two runs at once would race on it.

Trigger parameters (the **Trigger** form in the UI):

| Param | Default | Meaning |
|---|---|---|
| `sources` | all seven | Run only these; the rest are skipped |
| `tinystories_batch_size` | 2000 | Stories fetched per run |
| `quickdraw_categories` | 5 | Quick, Draw! categories fetched per run (of 345, up to 2,000 drawings each) |

### `primer_freeze_version`

Freezes the current processed data as a named version (`tag`, e.g. `v1`) in
`metadata/versions/<tag>.json`. Airflow has no git clone, so afterwards a person refreshes the DVC
pointer files and tags the commit:

```bash
primer-data dvc-pointers
git add data/*.dvc && git commit -m "chore(data): data version v1 (KAN-<n>)"
git tag data-v1
```

## Run it

Needs [Docker Desktop](https://docs.docker.com/get-docker/) (or Docker Engine with Compose v2.24+)
and about 4 GB of memory for Docker. From the repo root:

```bash
cp .env.example .env                 # once; on Linux set AIRFLOW_UID to the output of `id -u`
docker compose up airflow-init       # once: builds the image, creates the database and admin user
docker compose up -d                 # start Airflow
```

Open http://localhost:8080 and log in as `airflow` / `airflow`. DAGs start **paused**: unpause
`primer_data_pipeline`, then **Trigger**. Each task's log is in the UI and in `logs/`.

Data goes to **`.local-data/`** in the repo (git-ignored), so no cloud account is needed. The
first full run downloads the two picture-book image banks from GitHub (about 750 MB to stream),
so for a quick check trigger with `sources` set to e.g. `["clear_corpus", "sight_words"]`.

```bash
docker compose down                  # stop
docker compose down -v               # stop and delete the Airflow database
```

### Against the shared bucket

Team members with access to the `young-illustrators-primer` GCP project can write to
`gs://young-illustrators-primer-data` instead. Run `gcloud auth application-default login` on the
host, set `GCLOUD_CONFIG_DIR` in `.env` (see `.env.example`), then:

```bash
docker compose -f docker-compose.yaml -f docker-compose.gcs.yaml up -d
```

## How it's put together

| File | Role |
|---|---|
| `docker-compose.yaml` | Postgres, API server (UI), scheduler (runs the tasks: LocalExecutor), DAG processor, one-off init |
| `docker-compose.gcs.yaml` | Override: point the scheduler at the bucket and mount your credentials |
| `infra/airflow/Dockerfile` | `apache/airflow:3.3.2-python3.12` plus the `primer` package (editable) |
| `.env.example` | Container user and login; copy to `.env` |

`dags/`, `src/` and `configs/` are mounted into the containers, so edits apply without rebuilding.
Rebuild (`docker compose build`) only after changing dependencies in `pyproject.toml`.

## Tests

`tests/dags/test_dags.py` checks the DAGs load and are wired as above. Airflow doesn't run on
Windows, so the test skips there; run it inside the containers:

```bash
docker compose run --rm airflow-cli bash -c "pip install -q pytest && cd /opt/primer && python -m pytest -q tests/dags"
```
