# Data pipeline

Ingests, validates, and cleans the training datasets into the project bucket, `gs://young-illustrators-primer-data`. Each run writes a manifest that the [tracker dashboard](../../../dashboards/tracker/) shows.

## Setup

```bash
pip install -e ".[dev]"
gcloud auth application-default login
```

You need access to the `young-illustrators-primer` GCP project. Nothing is stored on your machine; everything goes to the bucket.

## Commands

```bash
primer-data sources                       # registered sources and which have a pipeline
primer-data run tinystories               # ingest + validate + preprocess one new batch
primer-data ingest tinystories            # fetch the next batch into raw/
primer-data validate tinystories          # check a raw batch (default: latest)
primer-data preprocess tinystories        # clean a raw batch into processed/
```

- `--batch-size N` sets records per batch (default from `configs/sources.yaml`).
- `--strict-license` refuses to run for a source whose `license_verified` is false. Use it when building a release dataset.
- `--data-uri <folder>` writes somewhere else, such as a local folder for testing.

Each `run` continues where the last one stopped. The position is saved in `metadata/state/<source>.json`.

## Bucket layout

```
raw/<source>/<batch_id>/           original data, never modified (gzipped JSON lines + _batch.json)
processed/<dataset>/source=<source>/batch=<batch_id>.parquet
                                   cleaned data in a standard schema (src/primer/data/schemas.py)
metadata/runs/<run_id>.json        one manifest per stage run (read by the dashboard)
metadata/state/<source>.json       where the next ingest continues
```

Every processed record keeps its `license` and `attribution`.

## Stages

| Stage | What it does |
|---|---|
| `ingest` | Copies the next batch, unchanged, into `raw/` |
| `validate` | Checks the raw batch (record count, required fields, unique IDs, empty text, length range). Any failed check stops the pipeline |
| `preprocess` | Normalizes text, filters by length, removes exact duplicates (within the batch and against earlier batches), and writes Parquet |

## Adding a source

1. Add an `ingest:` block for the source in `configs/sources.yaml`.
2. Create `sources/<source>.py` with a `Source` subclass implementing `ingest`, `validate`, and `preprocess`. See `sources/tinystories.py`.
3. Register it in `sources/__init__.py`.
4. Add tests under `tests/data/` that use a fake fetcher and a temporary local store. Tests never touch GCS.
