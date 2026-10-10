# Project tracker dashboard

A Streamlit app with two tabs:

- **Project:** current week, days to MVP and code freeze, workstream progress, tasks, the MVP checklist, stretch goals, open decisions, and the timeline. Data comes from [project.yaml](project.yaml).
- **Data pipeline:** each data source's license, how far it has gone through the pipeline, record counts, recent runs, failures, and validation checks. Data comes from [configs/sources.yaml](../../configs/sources.yaml) and the run manifests the pipeline writes.

## Run it

From the repo root:

```bash
pip install -r dashboards/tracker/requirements.txt
streamlit run dashboards/tracker/app.py
```

Before the pipeline has written anything, turn on **Show sample pipeline data** in the sidebar to preview the Data pipeline tab.

## Update it

- **Tasks, decisions, milestones:** edit [project.yaml](project.yaml) and open a PR. Task status is `todo`, `in_progress`, `done`, or `blocked`.
- **Data sources:** edit [configs/sources.yaml](../../configs/sources.yaml). Set `license_verified: true` only after reading the license on the source's own site.
- **Pipeline status:** updates itself. Each pipeline run writes one JSON manifest to `<metadata location>/runs/`, and the dashboard reads them every 60 seconds.

## Where pipeline metadata lives

The default location is `data/metadata/` in the repo, which is git-ignored. To read from GCS once the bucket exists:

```bash
TRACKER_METADATA_URI=gs://<bucket>/metadata streamlit run dashboards/tracker/app.py
```

Reading from GCS needs Google Cloud credentials (`gcloud auth application-default login`). You can also paste the location into the sidebar.

## Run manifest format

Every pipeline stage (`ingest`, `validate`, `preprocess`, `split`, `version`) writes one file per run. The full contract is in the docstring of [loaders.py](loaders.py), and [examples/metadata/runs/](examples/metadata/runs/) has samples.

```json
{
  "run_id": "20261014T030000Z-tinystories-ingest",
  "source": "tinystories",
  "stage": "ingest",
  "status": "success",
  "started_at": "2026-10-14T03:00:00Z",
  "finished_at": "2026-10-14T03:04:12Z",
  "records_in": 0,
  "records_out": 50000,
  "records_total": 50000,
  "dataset_version": null,
  "output_uri": "gs://<bucket>/raw/tinystories/20261014/",
  "validation": {"checks": [{"name": "no_empty_text", "passed": true, "detail": ""}]},
  "error": null
}
```
