# Data pipeline

Ingests, validates, and cleans the training datasets into the project bucket, `gs://young-illustrators-primer-data`. Each run writes a manifest that the [tracker dashboard](../../../dashboards/tracker/) shows.

## Setup

```bash
pip install -e ".[dev,dvc]"
gcloud auth application-default login
```

With access to the `young-illustrators-primer` GCP project, the CLI writes to the bucket by default. Without it (or to experiment safely), pass `--data-uri .local-data` to keep everything in a local folder.

## Using the data

```python
from primer.data.catalog import read_dataset

stories = read_dataset("stories_text")                     # TinyStories + picture books
pages = read_dataset("storybook_pages")                    # picture-book pages with illustrations
drawings = read_dataset("drawings")                        # Quick, Draw!
readability = read_dataset("readability")                  # CLEAR
words = read_dataset("word_lists", sources=["sight_words"])
drawings_v01 = read_dataset("drawings", version="v0.1")   # exactly as frozen in a version
```

Train on a named version (see [Versions](#versions)) so results are reproducible.

| Dataset | Sources | Schema in `schemas.py` |
|---|---|---|
| `stories_text` | TinyStories, StoryWeaver, African Storybook | `STORIES_TEXT`: one row per story |
| `storybook_pages` | StoryWeaver, African Storybook | `STORYBOOK_PAGES`: page text, `image_uri` (GCS), caption, alignment |
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
primer-data version <tag> -m "..."  # freeze all processed datasets as a version
primer-data versions                # list versions
primer-data dvc-pointers            # refresh data/*.dvc to the current object versions
primer-data profile                 # statistics + schema validation + drift for every dataset (reads only)
primer-data profile --write-schemas # re-infer configs/schemas/*.json from the current data
primer-data test-alert              # send a test alert email
```

The same stages run on a schedule in **Airflow**, one parallel task group per source: see
[`dags/README.md`](../../../dags/README.md). The DAG tasks call `orchestration.py`.

Sources with a pipeline: `tinystories`, `quickdraw`, `clear_corpus`, `sight_words`, `kuperman_aoa`, `storyweaver`, `african_storybook`.

- `--batch-size N` sets the batch size: records for TinyStories, categories for Quick, Draw! (defaults in `configs/sources.yaml`).
- `--data-uri <folder>` writes somewhere else, such as a local folder for testing.

## Two kinds of source

- **Incremental** (TinyStories, Quick, Draw!): each `run` fetches the next batch and continues where the last one stopped (position in `metadata/state/<source>.json`). The dataset is all batches together.
- **Snapshot** (CLEAR, sight words, Kuperman, StoryWeaver, African Storybook): the source is a fixed file or repo. `run` downloads it and makes a new batch only if the file changed, otherwise it reports "nothing new". The latest batch is the whole dataset.

`read_dataset` handles both.

## Versions

`primer-data version v1 -m "what changed"` freezes everything in `processed/` as a named version:

1. Every file in each dataset is recorded with its exact GCS object generation in `metadata/versions/v1.json`. The bucket has object versioning on, so a version reads back identically even if files are later replaced or re-processed. `read_dataset(..., version="v1")` reads those exact objects.
2. A `version` run manifest is written per source, so the dashboard shows which version each source is in.
3. DVC pointer files `data/<dataset>.dvc` are refreshed with `dvc import-url --version-aware --no-download`. They record the same object versions **without downloading or copying any data**. Commit them and tag the commit:

```bash
git add data/*.dvc
git commit -m "chore(data): data version v1 (KAN-<n>)"
git tag data-v1
```

To get a version's files onto a machine that needs local copies (e.g. a training VM): `git checkout data-v1 && dvc pull data/drawings.dvc`.

Versions can't be overwritten. Picture-book `image_uri`s point into `raw/`, which is never modified, so they stay valid for every version. Old copies of `metadata/` files are deleted after 30 days; old copies of datasets are kept.

## Bucket layout

```
raw/<source>/<batch_id>/           original data, never modified, plus _batch.json
processed/<dataset>/source=<source>/batch=<batch_id>.parquet
                                   cleaned data in a standard schema (schemas.py)
metadata/runs/<run_id>.json        one manifest per stage run (read by the dashboard)
metadata/state/<source>.json       where the next ingest continues / last snapshot hash
metadata/versions/<tag>.json       a frozen dataset version: every file and its GCS generation
metadata/profiles/<dataset>/<run_id>.json   statistics per column, one per data check
metadata/anomalies/<run_id>.json   schema errors and drift warnings from a data check
dvc/                               DVC remote, for anything DVC needs to push (e.g. models)
```

Every processed record keeps its `license` and `attribution`.

## Statistics, schema and anomalies

`profiling.py` checks each processed dataset as a whole (the stage checks above look at one batch):

1. **Statistics.** Per column: null fraction; min, max, mean, std and percentiles for numbers;
   length, distinct count and category shares for text. Saved to `metadata/profiles/`.
2. **Schema.** `configs/schemas/<dataset>.json` is the dataset's contract: exact column set,
   minimum row count, maximum null fraction (or "not entirely null" for mostly-null columns),
   numeric ranges, non-empty text and allowed categories. It is inferred from known-good data (`primer-data profile --write-schemas`),
   reviewed, and committed, so a change to the contract shows up in a pull request.
   **Great Expectations** validates the data against it; any failure is an **error**. So is a
   dataset that had rows and now has none, a dataset with no schema, or a dataset that could not be
   read. The one exception is the top of a numeric range: growing sources (more Quick, Draw!
   categories, a longer book upstream) legitimately exceed what was seen, so that is a warning.
3. **Drift.** The new profile is compared with the previous one. Rows falling by over 5%, nulls up
   by over 5 points, a numeric mean moving by over half a standard deviation (or at all, for a
   column that was constant), mean text length changing by over 30%, category shares moving by
   over 0.15 (total variation), or a column crossing between categorical and open-ended
   (25 distinct values) is a **warning**.

In Airflow, `check_data` runs this after every source: errors fail the run, warnings don't. Both are
emailed (see `alerts.py` and `PRIMER_ALERT_EMAILS` in `.env.example`), as is any failed task.

When a source legitimately changes (a new column, a new source value), re-run
`primer-data profile --write-schemas`, review the diff, and commit it with the change.

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
- **StoryWeaver and African Storybook** come from the Global Storybooks GitHub mirrors ([pb-source](https://github.com/global-asp/pb-source), [asp-source](https://github.com/global-asp/asp-source) and their image banks): English Markdown stories plus medium-size illustrations, streamed from GitHub into the bucket. A new batch is made only when either repo's latest commit changes. StoryWeaver here means Pratham Books' StoryWeaver titles as mirrored there (a 2016 snapshot).
  - Pages are matched to images by count: one extra image is the cover; two extra are cover and back cover; equal counts map one to one. Anything else is marked `unaligned` and gets no images rather than wrong ones.
  - Pratham's image bank describes every illustration; those descriptions become `image_caption` (useful for the illustration style model).
  - Neither mirror has reading levels, so `reading_level` is empty until the readability scorer fills it.
- **Kuperman** comes from Brysbaert's 51,715-word file on OSF; only the ~31,000 words with a Kuperman rating are kept.

## Adding a source

1. Add an `ingest:` block for the source in `configs/sources.yaml`.
2. Create `sources/<source>.py` with a `Source` subclass implementing `ingest`, `validate`, and `preprocess`. Use `tinystories.py` as the incremental template, or `clear_corpus.py` for a snapshot file.
3. Register it in `sources/__init__.py`.
4. Add tests under `tests/data/` that use a fake fetcher and a temporary local store. Tests never touch GCS.
