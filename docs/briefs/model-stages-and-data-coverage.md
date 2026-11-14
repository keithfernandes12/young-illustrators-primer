---
title: Which pipeline stages need a model, and whether the data supports them
type: brief
created: 2026-11-12
updated: 2026-11-14
status: draft
author: rishiparekh
jira: [KAN-93]
tags: [models, data, mvp, scoping]
related: [mvp-cloud-deploy]
---

# Which pipeline stages need a model, and whether the data supports them

## Problem

The scoping doc names the models by component (§5.3) and `mvp-cloud-deploy` slices them by service
(S3–S8), but no one has checked each stage against the datasets that are actually in the bucket. The
two lists are organised differently, so a stage can appear in both and still have no usable training
data.

Three stages do have that gap today, and one has an ordering dependency that neither document states.
The cost of not checking is a slice that a person or a `loop-worker` iteration picks up, gets halfway
into, and then stalls on because a label column is empty or the training distribution does not match
what production sends.

Solved when every stage that needs a model has either a named dataset in `processed/`, a written-down
gap with an owner, or a decision to stay pretrained.

## Sources

Read:

- `docs/project_scoping.md` §2 (data card), §3 (loading, preprocessing, splits), §5.3 (proposed
  solutions and the fine-tuning table), §7 (metrics and targets), §10 (monitoring).
- `configs/sources.yaml` — the source registry and which sources have `ingest` options.
- `src/primer/data/schemas.py` — the six processed schemas and their columns.
- `src/primer/data/README.md` — which sources have a pipeline, the bucket layout, versioning.
- `src/primer/data/sources/storybooks.py` — how picture-book pages and captions are built.
- `src/primer/data/sources/amateur_drawings.py` and `docs/runbooks/full-ingest.md` — added by PR #10
  (KAN-24), which this brief was first written without.
- `docs/plans/mvp-cloud-deploy.md` — slices S1–S14 and their dependencies.

Not read: Jira issue contents; the bucket itself. Every row count referred to here is the configured
minimum in `configs/sources.yaml` (`min_stories`, `min_records`, `max_per_category`), not a measured
count — nobody ran `primer-data profile` for this brief.

Inferred, not measured: that Quick, Draw! strokes differ enough from a photographed crayon drawing to
cost real accuracy. It is a reasonable expectation from how the two datasets were collected, but it is
an assumption until someone measures it.

## Vocabulary

- **Stage** — one point in the request path or the data pipeline where a model runs. Deliberately not
  "model": one model can serve two stages. The readability scorer is one model that runs as a request-path
  gate *and* as an offline labelling stage.
- **Request path** / **offline** / **eval** — when a stage runs: per book, in the data pipeline before
  training, or only when measuring a model version.
- **Owned** / **pretrained** — as §5.3 uses them: a model the team fine-tunes versus one used as
  published. A pretrained model in a stage still makes that stage a model stage.
- **Back-fill** — writing a model's output into a processed column the source did not provide. Today
  that is `storybook_pages.reading_level`, which the ingest writes as null.
- Words to avoid: "inference stage" (ambiguous between the request path and the batch jobs) and
  bare "pipeline" (the data pipeline and the book-generation path are both pipelines — say which).

### The stages

| # | Stage | When | Model | Data in `processed/` |
|---|---|---|---|---|
| 1 | Drawing recognition and attribute extraction | request | owned (SigLIP/CLIP/ConvNeXt) | `drawings` (labelled strokes); `character_drawings` (child-style raster, **no label**) |
| 2 | Story generation | request | owned (1–3B instruct + LoRA) | `stories_text`, `storybook_pages` |
| 3 | Readability scoring | request + offline | owned (DistilBERT/DeBERTa regression) | `readability` (CLEAR, with the Flesch-Kincaid baseline column) |
| 4 | Safety / moderation — request prompt, story text, **and the generated image** | request + offline | pretrained → owned | **none** |
| 5 | Illustration | request | pretrained SD + ControlNet; style LoRA is a stretch goal | `storybook_pages.image_uri` |
| 6 | Character-consistency check | request + eval | pretrained CLIP | golden set |
| 7 | Narration with word timings | request | pretrained TTS; an aligner too if the TTS emits no word timestamps | none needed |
| 8 | Read-aloud ASR and misread-word alignment | request (stretch) | pretrained Whisper | **none** (MyST is registered without a pipeline) |
| 9 | Learner adaptation | request (stretch) | owned, knowledge-tracing shaped | **none until the pilot** |
| 10 | Toxicity filter over training text (§3.2) | offline | same model as stage 4 | operates on `stories_text` |
| 11 | Auto-captioning illustrations with a VLM (§3.2) | offline | pretrained VLM | `storybook_pages` rows whose `image_caption` is null |
| 12 | Reading-level back-fill | offline | same model as stage 3 | writes `storybook_pages.reading_level` |
| 13 | Drift embeddings and the LLM-as-judge story score | eval | pretrained | production events; golden set |

Target-word *selection* is not on this list. §5.3 calls it "word-selection logic", and rules over
`word_lists` (Dolch, Fry, Kuperman age-of-acquisition) are the right implementation. It becomes stage 9
only when adaptation starts.

Two notes on what the table shows:

- Stage 3 is the only stage whose training data is ready as ingested: CLEAR arrives with both the
  target (`score`, BT easiness) and the baseline the scorer has to beat (`flesch_kincaid_grade`).
- §3.1 and §3.2 still describe a PyMuPDF page splitter and an OCR fallback for text inside images.
  The pipelines ingest Markdown from the Global Storybooks mirrors instead, so that stage, and the OCR
  model it implied, no longer exists.

## Options

**Option A — close the data gaps before the model slices start.** Block S5b (story LoRA) until levels
exist and S6 (drawing recognizer) until a labelled child-style split exists. Costs the most schedule:
both slices sit behind new ingestion work with the MVP due 2026-11-15, and it rules out having a
measured end-to-end path to show if the new ingestion itself slips.

**Option B — ship every stage on a baseline first, close a gap only where a gate fails.** What
`mvp-cloud-deploy` already implies ("the bar is end-to-end and measured, not best model"). Costs a
round of rework when a gate does fail, and risks a target being reported against a test split that
does not represent production. Rules out nothing permanently.

**Option C — cut stages out of the MVP.** Drop the owned drawing recognizer to a zero-shot CLIP
baseline and count it as pretrained. Cheapest, but the scoping doc's §11 requires three models trained by the team, and
the recognizer is one of the three — so this costs an acceptance criterion, not just scope.

Recommended: **Option B**, with two exceptions where the gap is not about model quality but about
whether a number means anything (questions 2 and 3 below).

## Questions to decide

- **Where do picture-book reading levels come from?** Recommended: the readability scorer back-fills
  them (stage 12), and S5b's instruction pairs are built only after that back-fill runs. `storybook_pages`
  carries a `reading_level` column, but `storybooks.py:297` writes it as null with the comment that the
  scorer fills it later, so the story LoRA's `level` constraint has no source until S3 lands. Training on
  pairs with a guessed level is §8's "LLM ignores level" failure, created on purpose. — *Language lead*
- **Where do labels for child-style drawings come from?** Recommended: hand-label a few hundred
  `character_drawings` rows as a held-out evaluation split, and do not try to train on them. PR #10
  ingests Amateur Drawings, so the imagery gap is closed — but `CHARACTER_DRAWINGS` carries a bounding
  box, 17 COCO joints and a segmentation polygon and **no category at all**. It says where a character
  is and how its limbs are arranged, never what it is. The only labelled drawing data is still Quick,
  Draw! strokes, so top-5 accuracy on a Quick, Draw! split still says nothing about a photographed
  crayon drawing. — *Vision lead with the Data lead*
- ~~Does the golden set carry safety cases?~~ **Settled for text.** `configs/golden/scenarios.jsonl`
  holds 100 scenarios, 80 `book` and 20 `safety`, so "zero safety violations on the golden set" now
  measures something rather than restating an empty set. All 20 are text cases, so the question
  reopens for images — see the next line.
- **Do we fine-tune a safety classifier, or stay pretrained plus a blocklist?** Recommended: stay
  pretrained for the MVP, as §5.3 already allows. Worth being explicit that an off-the-shelf toxicity
  model does not cover "unsafe for a four-year-old": scary, violent or sad-ending content scores clean.
  The blocklist and the golden-set cases carry that, not the classifier. — *Language lead*
- **What checks the generated image?** Recommended: an image entry point in the safety area, called
  from serving after `illustrate()`, counted by `safety_violations_golden`. Today `build_book` checks
  the request prompt and the story text and returns illustrations that nothing has looked at, while
  the scoping doc requires a filter at the image stage and an alert on the image flag rate.
  `SafetyClassifier.check` takes only `text`. The golden set needs safety scenarios whose *image* is
  the problem; the 20 it has are all text. — *Safety owner, with the serving owner wiring it*
- **Has MyST access been requested?** Recommended: request it now or drop the ASR word-error-rate
  target from §7 until it arrives. The corpus is access-by-request and has no pipeline, so the stretch
  goal currently cannot even be evaluated, let alone fine-tuned. — *Speech lead*
- **Do we correct §3.1/§3.2 about the PDF splitter and OCR?** Recommended: leave the submitted scoping
  doc as submitted and let this brief record the difference. — *Data lead*

## Candidate slices

Each needs a Jira key before anyone starts it; `loop-worker` skips slices without one.

- **B1 · A labelled child-style evaluation split.** Hand-label a few hundred `character_drawings` rows
  with a category each, as a held-out set. The ingestion half is done (PR #10); what is missing is the
  label, which `CHARACTER_DRAWINGS` does not carry. Usable on its own: it is the first honest number for
  how any recogniser behaves on real child-style art.
- **B2 · Safety scenarios in the golden set.** *Text half done* — 20 of the 100 scenarios in
  `configs/golden/scenarios.jsonl` are `kind: safety`, which turned `safety_violations_golden == 0`
  from a tautology into a measurement. The image half is not: all 20 are text cases, so an unsafe
  *picture* from a harmless prompt still passes. Goes with the image check in the stage 4 question.
- **B3 · Stroke-to-raster rendering.** One helper used by both the recogniser's training and its request
  path, so the two cannot drift apart. Goes with S6; cannot be retrofitted cheaply once a model has
  trained on one rendering and serving uses another.
- **B4 · Reading-level back-fill.** Runs the readability scorer over `storybook_pages` and writes
  `reading_level`. Depends on S3. Usable before any LoRA, because the level distribution of the
  picture-book corpus becomes visible in `primer-data profile`.
- **B5 · VLM auto-captioning for uncaptioned illustrations.** Only the illustration style LoRA needs it,
  which is stretch goal 3, so this goes last.

## Recommendation

Take Option B. B2 is done. Start B1 now — it does not depend on a model slice, and like B2 it is about
whether a §7 number means anything rather than about model quality. Sequence B4 after S3, and B3 with
S6. Leave B5 until the style LoRA starts.

Before any of this starts: each slice needs a Jira key, and someone should run `primer-data profile`
so the row counts in this brief come from the bucket instead of from the configured minimums. That run
also answers a question this brief cannot: `docs/runbooks/full-ingest.md` is person-only and moves a
48 GB archive, so the Amateur Drawings *pipeline* existing does not mean its 178,166 rows are in the
bucket yet.
