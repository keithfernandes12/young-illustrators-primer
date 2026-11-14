---
title: Character consistency (stage 6) with a pretrained CLIP score
type: plan
created: 2026-11-14
status: active
author: rutvijsurti
jira: [KAN-48, KAN-103, KAN-104, KAN-105, KAN-42]
tags: [models, illustration, consistency, clip, evaluation]
related: [model-development, model-stages-and-data-coverage, mvp-cloud-deploy, where-model-training-happens]
---

# Character consistency (stage 6) with a pretrained CLIP score

## Where things stand

Stage 6 of the model stage map: a pretrained CLIP similarity score, run once per book, deciding whether
a book's pages still look like the character the child drew. It feeds the gated `clip_consistency` key,
whose threshold in `.claude/skills/golden-gate/thresholds.json` is still `null`. The stage lives in
`src/primer/illustration/` because `Illustrator.consistency()` and `contract.METRIC_OWNERS` put it
there. It sits in its own subpackage, `consistency/`, so the stage 6 owner and the KAN-42 illustration
owner never edit the same file. Jira: KAN-48 (the consistency benchmark) and KAN-42 (illustration,
where the score is called).

## Sources read

- `AGENTS.md`, `docs/plans/AGENTS.md`, `docs/adrs/where-model-training-happens.md`.
- KAN-42, KAN-38, KAN-48 and KAN-94 in Jira (read 2026-11-14).
- On `main` (KAN-89 merged 2026-11-14): `src/primer/illustration/*`, `src/primer/drawing/interface.py`,
  `src/primer/evaluation/contract.py`, `configs/golden/README.md`.
- On `docs/model-development-blueprint` (KAN-93, unmerged): `docs/engineering/model-development.md`,
  §2 (contract), §3 (steps and the illustration numbers) and §4 (ownership).
- The 2026-11-13 stand-up transcript and the model stage map (stage 06 row).
- `src/primer/data/schemas.py` (`STORYBOOK_PAGES`), `catalog.py`, `bias.py`.
- **Assumed**: an illustrated page's `image_uri` is readable with `fsspec` by whoever runs the
  experiment. **Not read**: `src/primer/serving/book.py` in depth, and the Dockerfiles.

## Contract

- **Target behaviour**:
  - `ConsistencyScorer.score(references, pages)` returns a 0..1 book score. The primary part is
    identity (each page against the closest reference); the secondary part is cohesion (the pages
    against each other). It also returns a per-page breakdown, the weakest page, and a verdict once a
    threshold is set.
  - `python -m primer.illustration.consistency.experiment` compares embedders on same-book against
    other-book page pairs, split by book. It picks the embedder and threshold on `val`, reports `test`
    metrics per slice plus sensitivity, writes `consistency_report.json` and can log to MLflow.
- **Not doing**:
  - Editing `illustration/interface.py`, `model.py` or `evaluate.py` (the KAN-42 owner's).
  - Setting the threshold in `thresholds.json`, which needs team approval.
  - Generating images.
  - Running the experiment from an agent session.
- **Patterns to follow**: the area layout in `model-development.md` §2; Protocol plus Fake, as in
  `illustration/interface.py`; lazy weights, as in `illustration/model.py`; `primer.data.bias.SMALL_SLICE`
  for judging a slice.
- **Boundaries**: `consistency.scorer` imports neither torch nor `primer.data`, because it runs at
  request time. Only `experiment` imports `primer.data`. Weights load on first `embed()`.
- **Test plan**: `tests/models/test_consistency.py`, on `FakeEmbedder` and in-memory bytes. No weights,
  network or bucket.

## Slices

### S1 · Scorer and threshold experiment

*size: medium*

`consistency/scorer.py` (score, result, `summarise`, `FakeEmbedder`), `embedders.py` (CLIP B/32, CLIP
L/14 and DINOv2 from Hugging Face, loaded lazily), `experiment.py`, `configs/consistency.yaml`,
`CARD.md`, and the `consistency` extra in `pyproject.toml`.

- **Why separate**: everything after this needs the score and the report; none of it needs images.
- **Touches**: `src/primer/illustration/consistency/`, `tests/models/test_consistency.py`, `pyproject.toml`.
- **Acceptance**:
  - Six identical pages of the wrong character score below 0.5 while their cohesion is 1.0.
  - A book is embedded in one call.
  - The experiment selects a candidate only when it beats the baseline on `val`.
  - Importing the scorer loads neither torch nor `primer.data`.
- **Testing**: each acceptance line has a test. Weighting cohesion alone (`identity_weight=0`) fails the
  wrong-character test. Selecting on `test`, or letting a tie replace the baseline, fails the selection
  tests.
- **Depends on**: KAN-89 (the illustration area and the contract), merged 2026-11-14.
- **Jira**: KAN-103 (sub-task of KAN-48).

### S2 · Run the experiment and propose the threshold

*size: small*

A person runs:

```
pip install -e ".[consistency]"
python -m primer.illustration.consistency.experiment --out .local-data/consistency --mlflow
```

Then copies the selected embedder, its weights revision, the test metrics, the slice gaps and the
sensitivity results into `CARD.md`, pins `revision` in the config, and proposes the `clip_consistency`
threshold in `thresholds.json` for the team to approve.

- **Acceptance**: `CARD.md` has no _pending_ metric rows; the run ids are in it.
- **Depends on**: S1, read access to the bucket, and the shared MLflow server.
- **Jira**: KAN-104 (sub-task of KAN-48).

### S3 · Score real books in serving and on the golden set

*size: medium*

With the KAN-42 owner: `IllustratorModel.consistency()` calls `ConsistencyScorer.score()`, using the
approved reference images (`reference_image_ref`, KAN-38) and the generated pages. Illustration's
`evaluate.py` reports `clip_consistency = summarise(...)` over the golden books. The threshold is then
re-set from the spread of real generated books, not picture books.

- **Depends on**: S1, KAN-42 generating pages, and KAN-38's `reference_image_ref`.
- **Jira**: KAN-105 (sub-task of KAN-48), with KAN-42.

## Status

| Slice | State | PR | Notes |
|---|---|---|---|
| S1 | in review | [#24](https://github.com/keithfernandes12/young-illustrators-primer/pull/24) | KAN-103; draft, needs a reviewer |
| S2 | not started | | KAN-104; a person runs it |
| S3 | not started | | KAN-105; blocked on KAN-42 pages and KAN-38 references |

## Decisions

**2026-11-14**: Stage 6 code goes in `src/primer/illustration/consistency/` rather than a new area.
`clip_consistency` is owned by `illustration` in the contract, and `model-development.md` §4 says stage
6 cannot leave `illustration/`. The subpackage gives the stage its own files, so the meeting's split of
stages 5 and 6 between two people causes no shared edits.

**2026-11-14**: The threshold is calibrated on picture-book page pairs (same book against another book
from the same source), because no generated books exist yet. It is a starting value, re-set in S3.

**2026-11-14**: The book score is identity alone (`identity_weight=1.0`); cohesion is reported next to
it. Page-to-page similarity scores six identical pages of the wrong character at 1.0
(`model-development.md`). The experiment calibrates on page pairs, which is what identity measures. A
0.75/0.25 blend was tried first and dropped, because of the smoke-test result below.

## Verified facts

**2026-11-14**: The golden set (`configs/golden/scenarios.jsonl`, KAN-89) holds 100 text scenarios and
no images, so it cannot calibrate an image threshold by itself. Checked by reading the file's fields.

**2026-11-14**: Real `clip-vit-b32` runs through the scorer and `experiment.run()` on CPU (transformers
5.23, torch 2.15). It loads in about 3 s and scores a 4-page book in 0.07 s after that. On synthetic
shapes, the same character scored 0.99, while a different shape and colour still scored 0.77
(identity). The blueprint's proposed ≥ 0.80 would pass the wrong character, so the threshold has to
come from S2. transformers 5 returns an output object from `get_image_features`, which `embedders.py`
handles. Checked with a scratch script; synthetic images, not the dataset.

**2026-11-14**: `storybook_pages.image_uri` is a full URI (`Store.url(...)` in
`src/primer/data/sources/storybooks.py`), and `book_id` is only unique within a `source`.

**2026-11-14**: The reference question is settled on `main`. The `Illustrator` docstring in
`src/primer/illustration/interface.py` now says to condition IP-Adapter on `reference_image_ref` (the
parent-approved storybook render), after checking `has_approved_reference`. It says not to use
`crop_ref` (raw crayon) or `embedding` (a cache key in a different space from this score). The same
image is the scorer's reference. `Illustration.character_embedding` became `identity` (an opaque key);
this package never read it. Checked by reading the diff when `main` was merged in (`8e49be5`).

## Open questions

- Does `Illustration` carry page image bytes or only `image_ref`? The scorer needs bytes, so serving
  resolves the ref. Recommended: the illustrator resolves it, since it wrote the image. The KAN-42
  owner decides.

## Out of scope

- The DINOv2 half of KAN-48 as a separate metric. It is a candidate embedder here.
- The LLM-as-judge and human rubric parts of KAN-48.
- Image safety (KAN-94).
