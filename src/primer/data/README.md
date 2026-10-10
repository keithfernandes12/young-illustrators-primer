# Data pipeline

Ingests, validates, and cleans the training datasets into the project bucket, `gs://young-illustrators-primer-data`. Each run writes a manifest that the [tracker dashboard](../../../dashboards/tracker/) shows.

## Setup

```bash
pip install -e ".[dev]"
gcloud auth application-default login
```

You need access to the `young-illustrators-primer` GCP project. Nothing is stored on your machine; everything goes to the bucket.

## Using the data

```python
from primer.data.catalog import read_dataset

stories = read_dataset("stories_text")                     # TinyStories
drawings = read_dataset("drawings")                        # Quick, Draw!
readability = read_dataset("readability")                  # CLEAR
words = read_dataset("word_lists", sources=["sight_words"])
```

| Dataset | Sources | Schema in `schemas.py` |
|---|---|---|
| `stories_text` | TinyStories | `STORIES_TEXT` |
| `drawings` | Quick, Draw! | `DRAWINGS`: strokes as JSON on a 0–255 grid, label, recognized |
| `readability` | CLEAR | `READABILITY`: text, `score` (BT_easiness, higher = easier), Flesch-Kincaid grade baseline |
| `word_lists` | Dolch, Fry, Kuperman | `WORD_LISTS`: `list_name`, word, level/rank, age of acquisition |

## Commands

```bash
primer-data sources                 # registered sources and which have a pipeline
primer-data run <source>            # ingest + validate + preprocess one new batch
primer-data ingest <source>         # fetch the next batch into raw/
primer-data validate <source>       # check a raw batch (default: latest)
primer-data preprocess <source>     # clean a raw batch into processed/
```

Sources with a pipeline: `tinystories`, `quickdraw`, `clear_corpus`, `sight_words`, `kuperman_aoa`.

- `--batch-size N` sets the batch size: records for TinyStories, categories for Quick, Draw! (defaults in `configs/sources.yaml`).
- `--data-uri <folder>` writes somewhere else, such as a local folder for testing.

## Two kinds of source

- **Incremental** (TinyStories, Quick, Draw!): each `run` fetches the next batch and continues where the last one stopped (position in `metadata/state/<source>.json`). The dataset is all batches together.
- **Snapshot** (CLEAR, sight words, Kuperman): the source is a fixed file. `run` downloads it and makes a new batch only if the file changed, otherwise it reports "nothing new". The latest batch is the whole dataset.

`read_dataset` handles both.

## Bucket layout

```
raw/<source>/<batch_id>/           original data, never modified, plus _batch.json
processed/<dataset>/source=<source>/batch=<batch_id>.parquet
                                   cleaned data in a standard schema (schemas.py)
metadata/runs/<run_id>.json        one manifest per stage run (read by the dashboard)
metadata/state/<source>.json       where the next ingest continues / last snapshot hash
```

Every processed record keeps its `license` and `attribution`.

## Stages

| Stage | What it does |
|---|---|
| `ingest` | Copies the next batch, unchanged, into `raw/` |
| `validate` | Source-specific checks: record counts, required fields, unique IDs, empty or malformed records, value ranges. Any failed check stops the pipeline |
| `preprocess` | Cleans the batch, removes duplicates, and writes Parquet in the dataset's schema |

## Source notes

- **Quick, Draw!** keeps the first 2,000 drawings of each of the 345 categories (50 categories per run), and only drawings the game's model recognized. Rendering strokes to images and augmenting them comes in a later step.
- **CLEAR** keeps all excerpts with a score. One exact duplicate excerpt is dropped.
- **Sight words:** Fry's list contains "hot" twice (#30 and #487). That's in the source, so it's listed under `allowed_duplicates` and the first position is kept. Any other duplicate fails validation.
- **Kuperman** comes from Brysbaert's 51,715-word file on OSF; only the ~31,000 words with a Kuperman rating are kept.

## Adding a source

1. Add an `ingest:` block for the source in `configs/sources.yaml`.
2. Create `sources/<source>.py` with a `Source` subclass implementing `ingest`, `validate`, and `preprocess`. Use `tinystories.py` as the incremental template, or `clear_corpus.py` for a snapshot file.
3. Register it in `sources/__init__.py`.
4. Add tests under `tests/data/` that use a fake fetcher and a temporary local store. Tests never touch GCS.
