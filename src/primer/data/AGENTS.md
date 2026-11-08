# src/primer/data

The data pipeline. `README.md` in this folder is the full reference (datasets, commands, bucket
layout, versions, source notes); read it before changing anything here. These are the rules that
are easiest to break, and why:

- **`raw/` is never modified.** Picture-book `image_uri`s in every dataset version point into `raw/`;
  changing it breaks old versions.
- **Versions are permanent.** `metadata/versions/<tag>.json` records exact GCS object generations;
  a tag can never be overwritten. Use the `freeze-dataset-version` skill — a person runs the
  `version` command.
- **Every processed record carries `license` and `attribution`**, set by each source from
  `configs/sources.yaml`, so datasets can be credited. They are metadata, not a gate: this is a
  non-commercial course project and no check blocks ingestion on licence. Keep both columns filled
  when adding a source.
- **Tests never touch GCS.** Use a fake fetcher and the `store` fixture from `tests/data/conftest.py`
  (enforced by `tests/conftest.py`).
- **Incremental sources keep their position** in `metadata/state/<source>.json`. Any run against the
  bucket moves it, so agents run with `--data-uri .local-data` (the flag goes before the command).
- **Adding a source** is the four steps in `README.md` → "Adding a source", with tests. Copy
  `sources/tinystories.py` for an incremental source, `sources/clear_corpus.py` for a snapshot.
