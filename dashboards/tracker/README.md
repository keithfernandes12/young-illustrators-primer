# Project tracker

A one-page Streamlit dashboard showing what has happened so far and what's next:

- **Header:** current week, days to the MVP and code freeze.
- **Numbers:** tasks done, overdue tasks, data sources in the pipeline, records processed.
- **What's happened:** newest-first feed of pipeline batches (from the run manifests in GCS), completed tasks, and decisions.
- **Data:** each source's status, record count, and last update.
- **Open tasks:** sorted by due date with overdue ones flagged, filterable by workstream, plus open decisions.

## Run it

From the repo root:

```bash
pip install -e ".[dashboard]"
gcloud auth application-default login
streamlit run dashboards/tracker/app.py
```

Reading the pipeline runs needs Google Cloud credentials with access to the `young-illustrators-primer` project. Set `TRACKER_METADATA_URI` to read runs from somewhere else, such as a local folder.

## Update it

- **Tasks and decisions:** edit [project.yaml](project.yaml) and open a PR. When you finish a task, set `status: done` and add `done: <date>` so it appears in the feed. Decisions get `decided: <date>`.
- **Data sources:** edit [configs/sources.yaml](../../configs/sources.yaml).
- **Pipeline status:** updates itself from the manifests each pipeline run writes to `gs://young-illustrators-primer-data/metadata/runs/`. The manifest format is in the docstring of [loaders.py](loaders.py).
