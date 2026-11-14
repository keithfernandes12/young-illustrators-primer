---
title: MVP on Google Cloud — models v0, serving, deploy, gates and monitoring
type: plan
created: 2026-11-07
updated: 2026-11-14
status: active
author: rishiparekh
tags: [mvp, gcp, serving, models, deploy, loops]
related: [ai-coding-framework, ci-hardening, autonomous-loops]
---

# MVP on Google Cloud — models v0, serving, deploy, gates and monitoring

## Where things stand

On 2026-11-07 `main` has the data pipeline (`src/primer/data/`, seven sources, Airflow 3.3 DAGs in
Docker Compose, DVC pointers) and the AI coding framework. Every model area
(`src/primer/{drawing,story,readability,illustration,speech,safety,serving}/`) and `app/` is empty.
The MVP (`docs/project_scoping.md` §10, due 2026-11-15) needs an end-to-end cloud deployment, the
safety and readability gate, parent settings, a CI eval gate and monitoring. This plan is the
backlog that people and `loop-worker` iterations pull from, one slice per PR. Every slice is a
deliberately small v0: the bar is "end-to-end and measured", not "best model".

## Sources read

- `docs/project_scoping.md` §3, §7 (targets), §8 (failure analysis), §9 (deployment), §10 (timeline).
- `AGENTS.md`, `docs/plans/ai-coding-framework.md`, `docs/plans/ci-hardening.md`, `dags/README.md`.
- Open PRs #7 (bias slicing, KAN-76) and #8 (schema validation and drift, KAN-36).
- Not read: Jira issue contents. **Each slice needs a Jira key before anyone starts it.** A person
  creates the issue and writes the key into the slice; `loop-worker` skips slices without one.

## Contract

- **Target behaviour** — a parent sets reading level and target words, uploads a drawing to
  `yip-api` on Cloud Run (staging), and receives a narrated, illustrated book whose text passed the
  readability and safety gate; the golden-gate metrics for every component are logged in MLflow;
  alerts email the team; a canary can be promoted or rolled back with one command.
- **Not doing** — Whisper read-aloud, IP-Adapter, style LoRA (stretch goals, §5.4). The three
  team-trained models the MVP requires (drawing recognizer, story LoRA, readability scorer) each get a
  pretrained baseline slice first, so the end-to-end path works even if a training run slips.
- **Patterns to follow** — `src/primer/data/` (pure functions, Airflow only wires them; fakes in tests).
  New code under `src/primer/<area>/`. Model weights load lazily behind an interface so tests use a fake.
- **Boundaries** — `serving` may import every model area; model areas never import `serving`;
  nothing imports `data` at request time. Tests never load real weights, never touch GCS or the network.
- **Test plan** — unit tests per area with fake models; `golden-gate` on the golden split for model
  quality; `deploy-cloud-run` smoke test for the deployed path.

## Slices

### S1 · GCP foundation in Terraform

*size: medium · person-only for apply*

`infra/` Terraform for the topology this project actually uses: **one shared team project**
(`$YIP_SHARED_PROJECT`) holding every durable thing -- APIs, Artifact Registry `yip`, models
bucket (separate from `gs://young-illustrators-primer-data`), Firestore, Pub/Sub topics, Secret
Manager, service accounts per service, Workload Identity Federation for GitHub Actions, MLflow,
and a budget with 50/90/100% email alerts -- plus **one compute project per member**
(`$YIP_COMPUTE_PROJECT`) that runs Vertex AI jobs and nothing else.

There is no staging/prod pair: `docs/engineering/model-development.md` section 6 and
`.claude/hooks/guard_cloud.py` both assume shared-plus-compute, and the live/not-live boundary
is traffic rather than a project. Credits attach to a billing account, so member projects stay
on personal billing until their credits are spent.

- **Acceptance** — `terraform -chdir=infra validate` and `plan` are clean; a person runs `apply` for staging.
- **Depends on** — nothing. **Jira** — to create.

### S2 · Golden set and gate

*size: medium*

`configs/golden/scenarios.jsonl` (about 100 synthetic parent-setting scenarios: mode, level, target
words, themes, a QuickDraw drawing reference) and `src/primer/evaluation/golden.py`, which runs the
pipeline with whatever components exist and writes the metrics JSON read by `golden-gate`.

- **Acceptance** — with fake components, `golden.py` writes every key in `thresholds.json`; the gate
  passes on a perfect fake and fails on a fake that drops a target word.
- **Depends on** — nothing. **Jira** — to create.

### S3 · Readability scorer v0

*size: medium*

`src/primer/readability/`: Flesch-Kincaid baseline plus a small regressor (DistilBERT or a gradient-boosted
model on handcrafted features) trained on the CLEAR corpus version already in the pipeline, on Vertex AI.

- **Acceptance** — golden metrics `readability_rmse` < `fk_baseline_rmse` and `reading_level_within1` reported.
- **Depends on** — S2. **Jira** — to create.

### S4 · Safety gate v0

*size: small*

`src/primer/safety/`: blocklist plus an off-the-shelf toxicity classifier, applied to story text and to
image prompts; a book that fails is regenerated, never shown.

- **Acceptance** — `safety_violations_golden == 0`; an injected unsafe page is caught in a unit test.
- **Depends on** — S2. **Jira** — to create.

### S5 · Story generation v0

*size: medium*

`src/primer/story/`: an instruct model (Llama 3.x 1–3B, Qwen 2.5 or Phi) with a prompt per age mode,
then check, and regenerate on failure, for target-word inclusion, readability (S3) and safety (S4).

- **Acceptance** — golden `target_word_inclusion` and `story_quality_mean` reported; gate group `story`
  passes or the gap is recorded under Verified facts.
- **Depends on** — S3, S4. **Jira** — to create.

### S5b · Story LoRA fine-tune

*size: medium*

LoRA on the chosen 1–3B base with the instruction dataset (TinyStories plus the picture-book corpora),
trained on Vertex AI with `train-vertex`; replaces the S5 base model only if the `story` gate group improves.

- **Acceptance** — MLflow run with data version; golden `story` metrics compared with S5 in Verified facts.
- **Depends on** — S1, S5. **Jira** — to create.

### S6a · Drawing recogniser, pretrained baseline

*size: small*

`src/primer/drawing/model.py`: `DrawingRecognizerModel` behind the `DrawingRecognizer` contract.
Weights load lazily, never at import. SigLIP 2's image tower scores the QuickDraw categories
zero-shot and fills one `Character` — `label`, `confidence`, the ranked `candidates` the metric is
measured from, and an `embedding` from the same tower. One character per drawing for now: the whole
image is the subject. Tests use a recording fake encoder, so CI installs no weights and reaches no
network; `python -m primer.drawing.smoke` produces one analysis from a synthetic drawing with
cached weights. (An earlier draft of this line said it mirrors `primer.illustration.smoke`; no such
module exists — S6a's is the first smoke module in the repository, and the pattern is its to set.)

- **Why separate** — it needs no dataset. Recognition takes image bytes, so the baseline can be
  built and proved before S6b renders anything, and it is the slice everything else in S6 builds on.
- **Acceptance** — `tests/drawing/` passes with no weights and no network; the smoke module returns
  a `DrawingAnalysis` whose `candidates` are ranked and whose `top_labels(5)` has five entries.
- **Not in this slice** — the per-drawing `description` (a separate VLM), multi-character
  detection, `body_type`, `crop_ref`/`mask_ref`, `keypoints` (all S6e), any fine-tune (S6d), and
  any measured `drawing_top5_acc` (S6c, which needs S6b's dataset).
- **Depends on** — nothing. **Jira** — KAN-38.

### S6b · `render_strokes`, the one QuickDraw renderer

*size: small*

The shared function KAN-87 asks for: strokes on the 0–255 grid in, a deterministic image out, with
size, line width and padding from config rather than from a call site. **One** implementation,
imported by both the training loader and inference, because two renderers that drift is the exact
failure this ticket exists to prevent. `license` and `attribution` stay on every rendered record.

- **Why separate** — it is a pure function with no model and no frozen data, testable on fixed
  strokes, and three later slices import it. It is also the one piece of S6 that another area may
  want, so it should land on its own and early.
- **Acceptance** — a known stroke list renders to the expected pixels in a test with no bucket
  access; the same strokes and settings give identical output across runs; training and inference
  import the same function, asserted by a test rather than by convention.
- **Decide while building** — render on the fly in the loader, or cache per frozen version. If
  cached, write under `processed/` with a record of the settings, never into `raw/`. Record the
  choice as a dated Decision.
- **Depends on** — nothing. **Jira** — KAN-87.

### S6b1 · The shared preprocessing module

*size: medium*

`src/primer/drawing/preprocess.py`: **one** module holding both directions of KAN-31 — the
inference path that turns an arbitrary uploaded image into the form the recogniser sees, and the
training-time augmentation that makes a rendered Quick, Draw! canvas look like such an upload.
Settings in config, not in call sites.

Numbered between S6b and S6b2 because that is where it belongs in build order: S6b2 and S6c2 both
need it and neither can start without it. Nothing exists today — there is no `preprocess.py`, and
`analyse(image: bytes)` hands raw upload bytes straight to the processor.

- **Why separate** — it is KAN-87's argument a second time. If training degrades images one way and
  inference normalises them another, a measured number is a comparison between a model and two
  preprocessing paths, and nobody can tell which moved. One module, both directions, so they cannot
  drift. It is also the only thing standing between S6b2 and S6c2 and the "ready" column: KAN-31
  gates both and has had no slice, which by this plan's own rule — `loop-worker` builds slices, not
  tickets — meant neither could ever be scheduled.
- **Inference direction** — decode, optionally convert to greyscale, optionally crop and binarise to
  the renderer's black-on-white convention, and land on the same canvas geometry
  `drawing_render.yaml` defines. This half needs nothing from the other and should land first: it is
  what S6c2's variants call, and S6c2 is what would tell us whether the augmentation half is any
  good.
- **Training direction** — paper texture, line wobble, partial erasure, colour fill and lighting over
  a rendered canvas, applied at training time only and seeded so a run reproduces. What it produces
  has to be something the inference direction could also produce; that constraint is the point of
  one module.
- **Not in this slice** — KAN-31 also covers Amateur Drawings derived sets: character crops, masks,
  normalised keypoints, OpenPose renders and auto-captions. Those are detection and segmentation
  rather than preprocessing, and **S6e** already owns that work. Say so when closing KAN-31 so the
  ticket is not read as done.
- **Acceptance** — a test asserts a single entry point per direction, as KAN-87's single-renderer
  guard does, so a second preprocessing path cannot appear later under another name. Augmentation is
  deterministic given a seed. The whole module runs on `numpy` alone, so CI covers it without the
  `drawing` extra — the same reason S6b is testable.
- **Where the risk is** — the augmentation is synthetic and nothing validates that it resembles a
  real upload. S6c2 is what would say, and S6c2 needs only the inference half, so build that half
  first and treat the augmentation as unproven until there are numbers.
- **Depends on** — S6b for the canvas convention. **Jira** — KAN-31.

### S6b2 · Rendered QuickDraw split for the recogniser

*size: medium*

`src/primer/drawing/data.py`: train/val/test over the frozen QuickDraw version through S6b's
renderer, with S6b1's paper/crayon augmentation applied at training time only.

- **Why separate** — it is the first slice that needs frozen data on disk, which S6b does not.
- **Acceptance** — the loader returns the three splits for a pinned version, no record appears in
  two splits, and augmentation is off for val and test.
- **Depends on** — S6b, and **S6b1** for the augmentation. **Jira** — KAN-38.

### S6c · `drawing_top5_acc` from the golden split

*size: small*

`src/primer/drawing/evaluate.py` and `slices.py`: run S6a's baseline over S6b's test split, write
the `primer.evaluation.contract` shape, and report the gated key per slice as well as overall.

- **Acceptance** — `check_gate.py --only drawing` returns a verdict rather than `metric missing`,
  and the baseline number is recorded as a dated Verified fact whether or not it clears 0.85. The
  fact has to say what the number is over: once this slice lands it will be top-5 on **rendered
  Quick, Draw!**, which is the gate's number and not the product's. (Today it is neither — the
  metric is scored over `FakeEvaluationComponents`, which is handed the answer. This slice is what
  makes it mean anything.) KAN-64 delivers a child's own drawing, uploaded; S6c2 measures how far
  the preprocessing moves the answer, and until it has, "clears 0.85" means the gate passed, not
  that the stage works.
- **Check KAN-116 first.** Ten of the golden set's eighty book scenarios ask for `boat` or `kite`,
  neither of which is in the 345 categories, so this slice's number is capped at 0.875 against a
  0.85 threshold. Settle that before recording a baseline, or the first real number will be read
  as a model result when it is partly an arithmetic one.
- **Depends on** — S6a, S6b2. **Jira** — KAN-38.

### S6c2 · Preprocessing stability on real uploaded drawings

*size: medium*

`src/primer/drawing/evaluate.py`: measure how far the preprocessing path moves the recogniser's
answer on real child-drawing uploads. **No labels, and none are needed** — what is measured is how
often the prediction changes between variants, not whether it is right.

- **Why separate** — S6c measures rendered Quick, Draw!, while every input the product receives is
  a child's own drawing, uploaded or photographed by a parent (KAN-64, "Upload or photograph
  drawings"). The two differ in colour, lighting, paper texture and framing, and nothing has
  measured what any of that costs.
- **Why not accuracy** — Amateur Drawings is "children's drawings of **human figures**"
  (`src/primer/data/sources/amateur_drawings.py`), and the 345 categories name no whole person:
  `person`, `human`, `man`, `woman`, `boy`, `girl`, `child`, `people`, `figure` and `body` are all
  absent. What is present is body parts (`arm`, `leg`, `hand`, `foot`, `face`, `smiley face`) and
  figure-shaped objects (`angel`, `yoga`, `snowman`, `teddy-bear`), so a prediction lands on one of
  those. There is no correct label to hand-label towards, `drawing_top5_acc` is undefined on this
  corpus, and a hand-labelled set would record the labeller's forced mapping rather than the
  model's behaviour. **KAN-111** owns that decision. An accuracy number needs a corpus whose
  subjects the vocabulary can express; KAN-110's kiddraw is the only candidate found, and it drops
  `person` too, so it answers the photographed-corpus problem and not this one. Do not reach for a
  captioning VLM to label the gap away either — Florence-2, Qwen3-VL, Moondream and SmolVLM2 are
  photo-trained like the recogniser, so labeller and subject fail on the same drawings in the same
  direction, and the set would score the model correct exactly where both are confused. S6e brings
  a captioning VLM into this very module, so that shortcut is close at hand.
- **What it measures** — S6a's baseline over the same drawings under the variants in the table
  below, reported pairwise: how often top-1 changes, how far the top-5 ranking moves, **and the
  top-1 confidence and top-1/top-2 margin within each variant**. The margins are not decoration.
  On a corpus the vocabulary cannot name, every prediction is a forced near-miss, so a flip may
  mean the model was perturbed or merely that it never had a preference to perturb — the margin is
  what separates those. A change rate reported without it is unattributable. (S6a's smoke run
  scored `house` at 0.1255 with `barn` second at 0.0284 on a drawing the vocabulary *can* express;
  expect thinner margins here, which is the point.)
- **Decide while building** — which rank statistic "how far the top-5 ranking moves" is: Kendall's
  tau, top-5 set overlap, or mean rank displacement. Record the choice and the reason as a Decision.
- **Acceptance** — the pairwise stability figures, the margin figures, the sample size and its data
  version are reported from one MLflow run, or in Verified facts with a note that no host exists
  yet (KAN-107). Every derived record carries `license` and `attribution` from Amateur Drawings,
  which is **CC-BY-NC-4.0** (`configs/sources.yaml`, flagged there as metadata not checked) — a
  non-commercial corpus behind a release-gate number is worth seeing without digging. A Decision
  records whether the 0.85 gate keeps reading the rendered number.
- **What it cannot say** — which variant is *right*. Stability bounds how much preprocessing
  perturbs the model **plus** the tie-breaking noise of a corpus the vocabulary cannot name, and
  the margin figures are the only thing separating the two. If the margins come back uniformly
  thin, the honest result of this slice is that this corpus cannot support the measurement — a
  finding rather than a failure, and one KAN-111 needs. It is not accuracy either way, so it does
  not close the distance between the gate's number and the product's. That distance also spans a
  change of corpus and subject matter, which none of these variants decomposes.
- **Depends on** — S6a; S6c, which establishes the `primer.evaluation.contract` shape in the same
  file; and **S6b1** for the binarise variant **only** — the first two variants need nothing from it
  and land without it. **Jira** — KAN-109.

| Variant | Isolates | Crop |
|---|---|---|
| As uploaded | the path the product takes today | none |
| Converted to greyscale | colour | none |
| Cropped to the annotated box, then binarised | photography, lighting, paper texture **and framing** | the dataset's `bbox_*` |

The crop is Amateur Drawings' own user-confirmed `bbox_x/y/width/height` (`CHARACTER_DRAWINGS`),
which the product does not have at upload time — nothing in the inference path crops, and
`crop_ref`/`mask_ref` are S6e. That variant is therefore an **upper bound the product cannot reach
today**, not the real inference path, and its number has to be quoted that way.

### S6d · Fine-tune the recogniser

*size: medium*

`src/primer/drawing/train.py`: fine-tune the SigLIP 2 tower on S6b's split, logging one MLflow run
with the tags `model-development.md` lists. `--max-steps` and a wall-clock bound are part of this
slice, not an afterthought: `docs/adrs/bounded-local-training-for-agents.md` clause 2 requires both
before an agent may run training at all, and no area has them yet.

- **Why separate** — it is the first slice that trains, and the first that can move the gated
  number. A capped local run proves the wiring; the run that gets registered is a person's, on
  Vertex AI, from the area's container against the frozen data version.
- **Acceptance** — a capped local run completes and logs an MLflow run; `drawing_top5_acc` from
  S6c is compared with the S6a baseline in Verified facts, against the 0.85 target. S6c2's
  stability figures are re-measured on the fine-tuned tower and compared with the baseline's.
  **A gain on rendered Quick, Draw! together with a drop in stability on photographs is the outcome
  this stage most needs to catch** — it is what fine-tuning on clean renders is most likely to do —
  and recording only the gated number makes it invisible. If S6c2 has not run yet, this slice may
  still proceed, but its Verified fact has to say the result is unvalidated against photographs
  rather than leaving a reader to assume otherwise. Stability is not accuracy: it can show the
  model became more brittle, not that it became wrong.
- **Depends on** — S6b2, S6c; S6c2 for the second half of its acceptance. **Jira** — KAN-38.

### S6e · The rest of the character sheet

*size: medium*

Fill the fields KAN-42 reads and S6a leaves empty: the per-drawing `description` from a captioning
VLM, `body_type`, `crop_ref`/`mask_ref` from a detector plus segmentation, and `keypoints` for
humanoids. `docs/briefs/drawing-recognition-model-candidates.md` lists the candidates per field and
the licence constraint on each.

- **Why separate** — it is four more models behind one method, and none of them moves
  `drawing_top5_acc`. It is also what unblocks KAN-42's reposing branch and IP-Adapter
  conditioning, so it is worth keeping independent of the gate work in S6c and S6d.
- **Acceptance** — a multi-character drawing returns one sheet per character with every field
  populated, and `test_every_character_sheet_field_reaches_the_illustrator` passes against the
  real recogniser rather than the Fake.
- **Not in this slice** — `reference_image_ref` and `approved`, which the stylise-and-approve step
  sets; who owns that step is still open between KAN-38 and KAN-42.
- **Depends on** — S6a. **Jira** — KAN-38.

### S7 · Illustration worker v0

*size: medium*

`src/primer/illustration/`: Stable Diffusion + ControlNet-Scribble conditioned on the child's drawing,
one image per page, written to GCS; container for Cloud Run with an L4 GPU.

- **Acceptance** — `clip_consistency` reported; p95 time per page measured on staging.
- **Depends on** — S1. **Jira** — to create.

### S8 · Narration v0

*size: small*

`src/primer/speech/`: Piper or Kokoro on CPU with word timestamps for read-along highlighting.

- **Acceptance** — a page of text returns audio plus word timings; runs on a CPU Cloud Run instance.
- **Depends on** — nothing. **Jira** — to create.

### S9 · Serving API

*size: large (split if review needs it)*

`src/primer/serving/`: FastAPI `yip-api` (parent settings, book request, status, `/healthz`,
`/v1/books:dryRun`), Pub/Sub fan-out to the workers, Firestore for books and settings. Adds
`src/primer/serving/AGENTS.md` (what must never be logged), the later slice listed in
`ai-coding-framework`.

- **Acceptance** — the dry-run endpoint returns a complete fake book in tests; `privacy-guard` passes.
- **Depends on** — S5, S8 (fakes are fine for S6, S7). **Jira** — to create.

### S10 · Build and canary deploy

*size: medium*

Dockerfiles, Cloud Build, and the `deploy-cloud-run` scripts wired for `yip-api`, `yip-story`,
`yip-illustrate`, `yip-tts` on staging.

- **Acceptance** — a canary at 10%, a passing smoke test, promotion, and a rollback have each been run
  once on staging and recorded under Verified facts.
- **Depends on** — S1, S9. **Jira** — to create.

### S11 · CI eval gate (own PR, `.github/` rules apply)

*size: small*

A workflow that runs the golden gate with fakes on every PR and with real models nightly; warn-only
first, as `docs/engineering/ci.md` requires for new gates.

- **Depends on** — S2. **Jira** — to create.

### S12 · Monitoring and alerts

*size: small*

Cloud Monitoring alert policies and an SLO in Terraform: 5xx rate, p95 latency, GPU instance count,
budget, safety-flag log metric. Email routing, not Slack.

- **Depends on** — S1, S10. **Jira** — to create.

### S13 · Parent settings and reader web app v0

*size: medium*

`app/`: parent settings form and a page-turning reader with read-along highlighting.

- **Depends on** — S9. **Jira** — to create.

### S14 · Data pipeline in the cloud (decision first)

*size: small*

Choose: Managed Airflow (Gen 3, which now runs Airflow 3) for the existing DAGs, or Cloud Run Jobs plus
Cloud Scheduler calling the same `orchestration.py` functions. Record the choice as an ADR.

- **Depends on** — S1. **Jira** — to create.

## Status

| Slice | State | PR | Notes |
|---|---|---|---|
| S1 | not started | | person-only apply |
| S2 | merged | [#11](https://github.com/keithfernandes12/young-illustrators-primer/pull/11) | golden set + runner + gate done on fakes; real-model evaluation still open. Unblocks S3–S6, S11 |
| S3 | not started | | |
| S4 | not started | | |
| S5 | not started | | |
| S5b | not started | | Vertex GPU job |
| S6a | merged | [#18](https://github.com/keithfernandes12/young-illustrators-primer/pull/18) | zero-shot SigLIP 2 baseline; no measured number until S6c |
| S6b | merged | [#19](https://github.com/keithfernandes12/young-illustrators-primer/pull/19) | `render_strokes` + config + PNG; numpy only, so CI covers it. KAN-87 |
| S6b1 | not started | | the shared preprocessing module, both directions. Unblocks S6b2 and S6c2. KAN-31 |
| S6b2 | not started | | S6b merged; waits on S6b1's augmentation |
| S6c | not started | | the gate's number, over rendered Quick, Draw! — not the product's. Capped at 0.875 until KAN-116 |
| S6c2 | not started | | preprocessing stability on uploads; no labels needed. Needs S6b1's inference half. KAN-109 |
| S6d | not started | | first slice that trains; builds its own `--max-steps` cap. Reads S6c2's stability figures |
| S6e | not started | | unblocks KAN-42's reposing and IP-Adapter branches |
| S7 | not started | | needs GPU quota in region |
| S8 | not started | | |
| S9 | not started | | |
| S10 | not started | | |
| S11 | not started | | |
| S12 | not started | | |
| S13 | not started | | |
| S14 | not started | | |

## Decisions

- **2026-11-14** — S6a takes **`google/siglip2-base-patch16-224`**, pinned at revision
  `75de2d55…`, as the recogniser's tower. The brief rates SigLIP 2 highest of the Apache-2.0
  candidates for this output; the **base** variant rather than a larger one is `AGENTS.md`'s local
  GPU rule, and it binds here rather than at S6d because S6d fine-tunes whatever this slice picks.
  DINOv3 is not taken: its custom licence is a product risk the brief flags, and nothing it offers
  is worth that for a course MVP. The revision is pinned so a rerun measures the weights a number
  was taken on.
- **2026-11-14** — The 345 Quick, Draw! category names are **checked in** as
  `src/primer/drawing/configs/quickdraw_categories.txt` rather than listed from the bucket at run
  time. Listing needs the network, which `tests/drawing/` must not touch, and a vocabulary that can
  change underneath a measured `drawing_top5_acc` makes two runs incomparable. The file is written
  in the bucket's own listing order, which is the order
  `primer.data.sources.quickdraw` counts category indexes in.
- **2026-11-14** — `primer.drawing.smoke` **takes a drawing as a path** instead of generating one.
  Turning strokes into pixels is S6b's `render_strokes`, and KAN-87 exists precisely so there is
  one renderer; an `ImageDraw` loop in `smoke.py` would have been the second one before the first
  had landed.

- **2026-11-14** — **Render on the fly in the loader; do not cache rendered images.** S6b's
  contract left this open. A simplified Quick, Draw! drawing is a few hundred bytes of coordinates
  and rasterises in 0.36 ms measured, so a cache under `processed/` would buy a few minutes per
  epoch across the whole split while costing a bucket copy of every image and a freeze step to keep
  it honest. It would also buy
  the one thing KAN-87 exists to prevent: a second copy of the pixels, which can drift from the
  function that made them. KAN-31's paper-and-crayon augmentation is applied at training time only
  (S6b2), so a cache would hold the cheap half of the work anyway. Reversible: if profiling in S6d
  shows rendering dominates a step, caching can be added behind the same function without changing
  its signature.
- **2026-11-14** — **The render defaults are 224px, 3px strokes, 8px padding, black ink on white.**
  The other point S6b left to decide. `size: 224` is the SigLIP 2 checkpoint's own input
  resolution, so the image processor's resize is a no-op rather than a second resample of an
  already-rasterised sketch; that couples the default to S6a's checkpoint on purpose, and moving
  resolution in S6d means re-rendering the split. A 1px stroke at the 0.81 scale factor leaves a
  sketch that is mostly background, and padding keeps a stroke off the border where a convolution's
  receptive field is truncated. The whole 0–255 grid is mapped with one scale factor for both axes
  rather than each drawing's bounding box, because the simplified format has already been aligned
  and scaled with aspect ratio preserved, and re-fitting per drawing would stretch each one
  differently. All of it lives in `src/primer/drawing/configs/drawing_render.yaml` — the area's own
  config folder, not the repository's top-level `configs/`: these settings travel with a
  measured `drawing_top5_acc`, since a number taken at one stroke weight is not comparable with one
  taken at another. **Decided by the loop, not with the vision teammate KAN-87 names** — none of it
  is expensive to reverse before S6b2 renders a split, so push back on the PR.
- **2026-11-14** — **The renderer uses no image library.** Geometry is integer Bresenham plus a
  disc stamp over a numpy array, and PNG encoding is `zlib` and `struct` directly. Two reasons:
  the pixels are then a function of the strokes and the config alone rather than of whichever
  Pillow is installed, and `numpy` is a declared dependency while Pillow is declared nowhere in
  this project at all — so the whole strokes-to-bytes path is covered by CI instead of being
  skipped there. Pillow is still used where it happens to be present, to assert the encoder emits
  a PNG a real decoder reads; that one test is gated on an incidental install rather than on an
  extra anybody can ask for, which is worth fixing once S6a lands the `drawing` extra.
- **2026-11-14** — The drawing stage takes over **KAN-87** (`render_strokes`), which was assigned
  elsewhere and unstarted since 2026-11-08 — no branch, no PR, no `render_strokes` in the tree. It
  is the hard dependency of every measured slice in S6, so leaving it where it was left S6c and S6d
  blocked behind work nobody had begun. Split out as S6b, a pure function with no frozen data, and
  deliberately kept as **one** renderer imported by training and inference alike: KAN-87's whole
  point is that the two cannot drift, and that reason survives the change of owner. The dataset
  split that was S6b is now S6b2. **Tell the previous assignee before starting S6b.** Done on
  2026-11-14 by Rishi, in person rather than on the ticket — so KAN-87 itself carries no comment,
  and nobody needs to raise this again.
- **2026-11-14** — S6 split into S6a–S6e so a loop can take one at a time, and each slice given the
  Jira key that already covers it (KAN-38, or KAN-87 for the render dependency) rather than new
  issues: `loop-worker` skips a slice with no key, and creating issues is an `ask` rule that would
  stall an unattended run. S6a is deliberately first because it needs no dataset — recognition
  takes image bytes — so the stage is not blocked behind another member's ticket.
- **2026-11-14** — The drawing stage keeps its slices here rather than in a plan of its own.
  `docs/plans/AGENTS.md`: "Never keep a second plan elsewhere." The research that chose the
  candidate models lives in `docs/briefs/drawing-recognition-model-candidates.md`, which is a brief,
  not a second plan.
- **2026-11-07** — Pretrained baseline first, then the team-trained model, for each of the three models
  §5.4 requires; both measured by the golden gate. Eight days remain before the MVP and every model
  area is empty, so the baseline keeps the end-to-end path alive if a training run slips.

## Verified facts

- **2026-11-14 — S6a: the zero-shot baseline loads and ranks sensibly.** *Hypothesis:* SigLIP 2's
  towers can score the 345 Quick, Draw! categories for one drawing and return a ranked character
  sheet, with no training and no dataset. *Result:* yes.
  `python -m primer.drawing.smoke --drawing <a synthetic house>` on
  `google/siglip2-base-patch16-224` @ `75de2d55…` returned `house` at 0.1255, with `barn` 0.0284,
  `square` 0.0028, `church` 0.0027, `tent` 0.0023 behind it; 345 labels scored, a 768-dimension
  embedding, 9.9 s to load on **CPU**, then 10.5 s for the first analyse and **0.12 s for the
  second**. Almost all of that first figure is encoding the 345 prompts once; the warm number is
  what a drawing costs and what KAN-71's budget is about. An earlier version of this fact said
  "11.3 s to analyse", which was the cold number under a name everyone reads as per-drawing.
  The input was a crude house drawn with
  the standard library, so the ranking is evidence that the label/score pairing and the prompt
  template are right — a shifted pairing would not have put four buildings in the top five.
  **This is evidence about wiring, not about model quality:** it is one synthetic image, and no
  `drawing_top5_acc` exists until S6c runs the real split.
- **2026-11-14 — S6a: the confidences are sigmoid scores, not a distribution.** The winning label
  scored 0.1255 and the five shown sum to ~0.16. That is expected — SigLIP's loss is pairwise, so
  each score is "does this label fit" independently — but it means a confidence threshold tuned as
  if it were a softmax probability would reject nearly everything. Ranking, which is all
  `drawing_top5_acc` reads, is unaffected.
- **2026-11-14 — S6a: SigLIP 2 base's parameter count and weight footprint, measured.** (The
  heading used to say "fits the 8 GB rule, measured", which a reader skimming headings would take
  as a VRAM measurement. It is not one — see the end of this bullet.) Loaded and counted rather than
  estimated: 375,187,970 parameters total, **1.398 GB in fp32 / 0.699 GB in bf16**, split as a
  92,884,224-parameter vision tower (0.346 GB fp32) and a 282,303,744-parameter text tower
  (1.052 GB fp32). S6d fine-tunes the **vision** tower, so the figure that matters there is 93 M
  parameters — with AdamW states and gradients that is single-digit hundreds of MB before
  activations, well inside 8 GB. **VRAM itself was not measured:** PyPI's Windows `torch` wheel is
  CPU-only (`torch 2.15.0+cpu`, `cuda False`), so this run never touched the GPU. The measured VRAM
  figure AGENTS.md asks for belongs to S6d, which needs a CUDA build anyway.
- **2026-11-14 — S6a: the smoke module earned its place on its first run.**
  `transformers` 5.x returns a `BaseModelOutputWithPooling` from `get_text_features`, not the bare
  tensor the model card's example shows, so normalising it raised `AttributeError` on the first
  real call. The fake-encoder tests could not have caught it — they inject an `Encoder` and never
  execute `SiglipEncoder`. Fixed by pulling the one version-dependent assumption into
  `model.pooled`, which *is* unit-tested, and pinning the extra to `transformers>=5,<6` because
  that is the only API actually exercised.
- **2026-11-14 — S6a: the CUDA path was broken and is fixed but still unrun.** *Hypothesis:* a
  CPU smoke run proves the encoder works. *Result:* no. A self-review found that the processor
  returns float32 `pixel_values` and `BatchFeature.to(device)` moves without casting, so on CUDA —
  where `load()` puts the tower in fp16 — the first matmul raises `mat1 and mat2 must have the
  same dtype`. Fixed by passing the dtype as well (`.to(device, dtype)`, which `BatchFeature`
  forwards to `Tensor.to`, verified against the installed source). **Not executed on a GPU:** this
  machine has `torch 2.15.0+cpu`. The first GPU run of `smoke.py` is still an untested path, and
  whoever makes it should expect to find something.
- **2026-11-14 — S6a: a NaN confidence survived every guard.** With a NaN among the scores,
  `sorted` produced an arbitrary order and `Character.__post_init__` accepted it — its ranking
  check compares two lists, and a NaN element compares equal to itself by identity, so an unsorted
  list whose `label` was not the top-scoring one passed validation. A mislabelled character would
  have reached `drawing_top5_acc` as a number rather than an error. Closed at both ends: a floor on
  the norm in `l2_normalised` (the 0/0 that produces it) and an explicit non-finite check in
  `analyse`. The contract-level hole in `interface.py` is left alone — see Open questions.
- **2026-11-14 — S6a: that NaN floor was a no-op in the dtype it was written for.** An independent
  review found it: `1e-12` is below float16's smallest subnormal (~6e-8), so on CUDA
  `clamp_min(1e-12)` clamps to exactly `0.0` and the 0/0 happens anyway — in the fp16 path the
  comment named as the reason for the floor. The floor is now the caller's
  `torch.finfo(features.dtype).tiny`. Mutation-checked: putting `1e-12` back, and separately
  reverting the `.to(device, dtype)` cast, each now fails exactly one test, and neither failed
  anything before this round. Both are fp16-only, and **still nothing has run on a GPU** — these
  are stub towers in float16 on a CPU.
- **2026-11-14 — S6a: the pinned checkpoint revision exists and is Apache-2.0.**
  `https://huggingface.co/api/models/google/siglip2-base-patch16-224/revision/75de2d55…` returns
  200 and reports `license:apache-2.0`. Every measured figure in `CARD.md` is anchored to that
  sha, so it is worth knowing it resolves rather than assuming it.
- **2026-11-14 — S6a: three assertions could not fail, and two inputs were accepted that should
  not be.** The stub tower used `logit_scale = exp(0)` and `logit_bias = 0`, both identities, so
  multiplying by the scale was indistinguishable from dividing and adding the bias from
  subtracting; the dtype assertion guarding the CUDA fix passed with the fix reverted. `top_k: 0`
  passed construction and died on `candidates[0]`. Nothing checked that a confidence is in 0–1, so
  deleting `torch.sigmoid` from the encoder left every ranking test green while an unbounded logit
  flowed out under a name three documents call a probability. All closed.
- **2026-11-14 — S6a: the text tower's normalisation was the last unguarded line in the scoring
  path.** A second review round found that replacing it with the raw features left all 38 tests
  green, because the stub's text rows were already unit length. Against the real checkpoint the
  text norms are 18–26, so skipping it scales every logit by about twenty, all 345 sigmoids
  saturate to 1.0, and the winner becomes whichever label sorts first — `The Eiffel Tower`, for
  every drawing. Neither guard catches it: the 0–1 check passes because a saturated sigmoid is in
  range, and the ranking check passes because all-equal is sorted. The stub's rows are now five
  units long, so that mutation fails, and a six-prompt case exercises a zero text row.
- **2026-11-14 — S6a: the CUDA path now picks bf16, not fp16.** `AGENTS.md` names bf16 for local
  GPU work and S6d will fine-tune in it, while `CARD.md` and these facts already said bf16 and the
  code said fp16 — one of the two was wrong. bf16 also keeps fp32's exponent range, which is the
  half of the precision question that produces a NaN rather than a drift. **Still unrun on a GPU.**
- **2026-11-14 — S6a: that dtype switch was unasserted, and it silently decoupled three tests
  from the code.** A third review round found that `load()`'s dtype choice could be set to fp16,
  or to plain fp32 giving up reduced precision on CUDA entirely, with the whole suite green —
  while the three tests added to close the CUDA gap still said "what `load()` picks on CUDA" and
  ran float16. `load()` now has a test of its own, the first it has ever had, covering the dtype,
  the device and the `from_pretrained` keywords against a stubbed factory; and the
  reduced-precision fixture runs both fp16 and bf16, because the norm-floor regression only shows
  in fp16 while the production path is bf16. Mutation-checked: setting the dtype to fp16 now fails
  one test. The same round found the **text**-side norm floor unpinned — only the image side's was
  — which is the side where one NaN ruins every confidence rather than one encoding.
- **2026-11-14 — S6a: comments and card carried fp16 numbers after the switch.** The cross-dtype
  cache-key warning said one step is "about thirty times" the rounding step, true of fp16; in
  bf16 it is 2.441e-4 at the component the sentence names, **244 times** (a fourth round caught a
  first correction that overshot it by 15%, by computing the binade rather than scaling the value). `CARD.md` contradicted itself on one page. The
  justification for making the norm floor a parameter also claimed clamping to 1e-12 fails "on the
  CUDA path", which is false for bf16 — it keeps fp32's exponent range and represents 1e-12
  exactly. Corrected in all four places; the parameter stays, because the point is not to depend
  on which dtype someone picks.
- **2026-11-14 — S6a: a transparent drawing reached the tower as a solid black square.**
  *Hypothesis:* `Image.open(...).convert("RGB")` normalises any upload to three channels.
  *Method:* built a four-pixel PNG with one black stroke on a fully transparent canvas, in each of
  the three shapes transparency takes — `RGBA`, `LA`, and a palette image carrying it in `info` —
  and converted each. *Result:* all three came out a uniform `(0, 0, 0)`, ink and canvas alike:
  `convert("RGB")` does not composite alpha, it drops it, and a sketch is black ink on
  transparent **black**. Nothing raised; the tower scored a black square and would have returned
  a confident label for it. Now composited onto `alpha_background` (255, white) before conversion,
  because `drawing_render.yaml` rasterises the Quick, Draw! split as dark ink on white and a
  photograph of a paper drawing is too — compositing onto anything else would invert an upload
  relative to the split S6c scores it against. The mode alone does not decide it: the palette case
  reports `"P"`. Found by the automated reviewer on the pull request.
- **2026-11-14 — S6a: two more races and 1113 device syncs per drawing, from the same review.**
  The lazy `_model is None` guard in `load()` let two threadpool requests on a cold start load two
  checkpoints onto one device; locking only `load()` would not have fixed it, because two threads
  racing `DrawingRecognizerModel.encoder` build two *different* encoders that each load past their
  own lock. Both are locked now, and `load()` publishes `_model` last, after the move and `eval()`.
  Separately, `float(value)` per element is a device-to-host copy that synchronises the stream —
  345 + 768 = **1113** of them per drawing for the shipped checkpoint, on the warm path KAN-71
  budgets — replaced with one `tolist()` per vector, with a tensor subclass counting the crossings
  so the test pins the cost rather than only the values.

- **2026-11-14** — `render_strokes` renders the Quick, Draw! 0–255 grid onto the canvas with the
  endpoints pinned: at the shipped settings coordinate 0 lands on column 8 and 255 on column 215,
  the first and last drawable pixels of a 224px canvas with 8px padding. Pinned by
  `tests/drawing/test_render.py`, which asserts the exact inked pixels of horizontal, vertical,
  45°, shallow, steep and downhill diagonal, tie-breaking and single-point strokes on a 10×10
  canvas rather than only their shape — the interior of a segment, not only its endpoints. 74
  tests, all but one running on `numpy` alone, so CI covers the whole path.
- **2026-11-14** — Four shapes that are not drawings were rendering as plausible ones, found by
  the automated reviewer on PR #19 and reproduced before fixing: a run of numeric strings
  (`float("12")` is 12.0), a run of booleans (`bool` is a subclass of `int`), a **mapping** run
  with numeric keys — iterating one yields its keys, so `{"0": 9, "1": 9}` drew at 0 and 1 and
  discarded the 9s, while a word-keyed mapping raised, which is the tell — and a `set` run, which
  renders in sorted order and so is a different drawing. Each is now refused by type and by name.
  numpy arrays and numpy scalars still render, since that is what a loader holds; `np.bool_` is
  refused like a plain `bool`. This is the same standard `RenderConfig` already applied to its own
  settings, now applied to coordinates.
- **2026-11-14** — `render_strokes` accepts numpy arrays in its type signature, not only at run
  time. `ndarray` is not a `Sequence`, so the original alias made the loader S6b2 is going to
  build — which holds strokes as arrays, and is the function's primary caller — a mypy error at
  its one honest call site. Verified both ways: before the widening a list of arrays failed
  `mypy` with `incompatible type … expected Sequence[Sequence[Sequence[float]]]`; after it, a
  caller passing a list of arrays and one passing a single array both type-check. The usual
  answer to that error is a `# type: ignore`, which `AGENTS.md` treats as weakening CI.
- **2026-11-14** — The geometry tests are checked against mutations rather than assumed to bite.
  Three edits that move pixels without looking like they would have been found so far: loosening
  `_line`'s first tie-break from `>=` to `>`, swapping half-up rounding for round-half-to-even,
  and loosening the *second* tie-break from `<=` to `<`. The first two were caught in one round
  and the third survived it — a tie in that branch needs an even `dx`, and every stroke pinned at
  the time had `dx` of 7 or 0. Each now fails exactly one test. **Not a proof that no fourth
  exists**, which is the honest status of any mutation check.
- **2026-11-14** — The single-renderer guard scanned `src/primer` while claiming the repository,
  and widening it introduced a second bug: a `.claude/worktrees` checkout made it report
  `render.py` twice, so it failed on a developer's machine and passed in CI, which has no
  worktrees. It now prunes any directory with its own `.git` and asserts repository-relative
  paths, so a genuine second renderer is located rather than merely counted. Verified by dropping
  a copy of the branch under `.claude/worktrees` and re-running.
- **2026-11-14** — The same strokes and config give byte-identical PNGs across calls, and the
  hand-written encoder's output is a real PNG: Pillow opens it as mode `L` at the right size and
  the array it decodes equals the array that was encoded. That one check is skipped wherever
  Pillow is absent, CI included; the chunk, length and CRC checks beside it are not, and one test
  corrupts a CRC to prove they fail when they should.
- **2026-11-14** — Eyeballed, not measured: **real** Quick, Draw! drawings render legibly at the
  shipped settings. The first three `house` records of
  `gs://quickdraw_dataset/full/simplified/house.ndjson` (public bucket, anonymous read, nothing
  written) have 3–8 strokes, 24–50 points, coordinates spanning the full 0–255 range, and come out
  with 2,383–3,387 inked pixels of 50,176; printed as ASCII, the first is a house with a pitched
  roof, a panelled window and a door, all strokes continuous. This confirms the stroke format the
  renderer assumes and says the settings produce a drawing rather than noise. It says nothing about
  whether a model recognises it; there is no `drawing_top5_acc` until S6c.
- **2026-11-14** — Measured on 50 real `house` drawings at the shipped 224px settings: 0.356 ms to
  render one to an array, 0.454 ms more to encode it as PNG. At the 500k-record scale S6b2 is
  heading for that is roughly 3 minutes of rendering single-threaded, or 7 with PNG — which is why
  rendering on the fly beats caching, and the figure to check against if a loader ever looks slow.
  **One category, and a sparse one.** A denser drawing of comparable point count measures around
  1.1 ms, so take 3× as the working upper bound until S6b2 re-measures over the real split.
- **2026-11-14** — `zlib` compression level 9 cost 2.647 ms per image against the default's
  0.312 ms, for 875 bytes against 1,134 — 8.5× the time for 23% fewer bytes. A sketch is long runs
  of one byte, deflate's worst case for the effort. `to_png` uses the default; the level was the
  single largest cost in the whole strokes-to-bytes path before this was measured.
- **2026-11-14** — The disc stamp does **not** equalise stroke weight at the shipped
  `line_width: 3`, contrary to the obvious argument for using one. At radius 1 the "disc" is a
  five-cell plus: a horizontal stroke lays down 3.0 pixels of ink per pixel of length, a diagonal
  2.1, and a 3×3 square stamp would give 3.5 — 29% thin against 18% thick. The disc is kept
  because the gap closes as the radius grows and a square's does not, but it is a choice, not a
  result; S6c is the first slice that can say whether it costs accuracy.
- **2026-11-14** — A malformed drawing is refused rather than rendered as something plausible:
  empty stroke lists, mismatched x/y runs, empty strokes, coordinates off the grid and NaN all
  raise, and JSON text raises a `TypeError` naming `json.loads` — the `strokes` column is stored as
  JSON, so passing it unparsed is the likely call-site mistake. Config errors raise at construction
  rather than at the first drawing, including an even `line_width`, ink equal to background, and
  padding too small for the stroke width to stay on the canvas.

- **2026-11-14 — S6a's `package-data` dependency on S6b is closed, by merging in that order.**
  A wheel built from the S6a branch alone contained no `configs/` YAML, so `DrawingRecognizerModel()`
  would raise `FileNotFoundError` from a wheel install while editable installs, the area
  `Dockerfile`, `verify` and the smoke run all passed. S6b carried the
  `[tool.setuptools.package-data]` block that fixes it for every area; #19 merged first, as
  recommended, and this branch picks it up in the merge rather than duplicating it. Nothing builds
  a wheel today, so it was latent either way.

- **2026-11-14 — Colour reaches the tower at inference and is absent from training.** `model.py`'s
  `flattened()` composites an image with transparency onto an opaque canvas of the baseline
  config's `alpha_background` grey (`drawing_baseline.yaml`; not the renderer's `background`, which
  is a different field in a different file) and converts that to RGB, falling back to a bare
  `.convert("RGB")` only when
  there is no transparency — the bare call on an alpha image would *drop* the channel and leave a
  black square, which is what the helper exists to avoid. Either way a colour photograph reaches
  the tower with its colour. Training has none: `render_strokes` writes a 2-D array holding two
  grey levels, and its config validates `background`/`ink` as single 0–255 integers, so a colour
  cannot be expressed there at all. Nothing stores one either — `Character` has no colour field and
  neither do the data schemas, although KAN-38 lists "colors and features (k-means + Florence-2)"
  as a character-sheet output. What the asymmetry costs is unmeasured; S6c2's greyscale variant is
  what would measure it.
- **2026-11-14 — The 345-category vocabulary cannot name a human figure, and Amateur Drawings is
  nothing else.** `src/primer/data/sources/amateur_drawings.py` describes the corpus as "children's
  drawings of human figures", `CHARACTER_DRAWINGS` as "drawings of characters (human figures)" with
  17 COCO joints on every row. Probed against
  `src/primer/drawing/configs/quickdraw_categories.txt`: `person`, `human`, `man`, `woman`, `boy`,
  `girl`, `child`, `people`, `figure` and `body` are all absent — no category names a *whole
  person*. Present instead are body parts (`arm`, `leg`, `hand`, `foot`, `knee`, `elbow`, `finger`,
  `eye`, `ear`, `nose`, `mouth`, `face`, `smiley face`) and figure-shaped objects (`angel`, `yoga`,
  `mermaid`, `snowman`, `teddy-bear`), and a prediction lands on one of those — far more often on
  `arm` or `smiley face` than on `angel`, which is what a flip between variants will look like. So
  `drawing_top5_acc` is undefined over this corpus and hand-labelling it would record the
  labeller's forced mapping, not the model. This is why S6c2 measures stability rather than
  accuracy. Tracked as KAN-111.
- **2026-11-14 — The golden set asks for two labels the recogniser cannot emit, capping
  `drawing_top5_acc` at 0.875 against a 0.85 gate.** Counted, not estimated:
  `configs/golden/scenarios.jsonl` holds 80 `book` scenarios over 16 distinct `drawing_label`
  values, and two of them are absent from `quickdraw_categories.txt` — `boat` (5 scenarios; the
  vocabulary has `sailboat` and `speedboat` but no plain `boat`) and `kite` (5; absent entirely).
  Ten of eighty can therefore never score, so a flawless recogniser reaches **0.8750** while
  `.claude/skills/golden-gate/thresholds.json` asks for **0.85** — 0.025 of headroom, and 97.1% of
  the achievable scenarios required. It is invisible today because `golden.py:114` scores against
  `FakeEvaluationComponents`, whose `top5(true_label)` is handed the answer and echoes it back
  (`fakes.py:27-31`), so the vocabulary is never consulted. **S6c is where it becomes live**, and
  it would present as an unexplained plateau just below the gate. Tracked as KAN-116.
- **2026-11-14 — kiddraw is black-stroke only, so it does not answer the colour question.**
  Measured from the repository rather than the paper: `experiments/museumstation/trial-sequence.js`
  sets `path.strokeColor = 'black'`, `strokeCap = 'round'`, `strokeWidth = 10`, and the analysis
  code measures ink from the alpha channel (`img[:,:,3]`), which only works when every stroke is one
  colour on a transparent ground. It is still the only source found with drawings by real children
  **and** category labels: 37,770 drawings, ages 2–10, 48 categories of which 44 map onto our 345
  (39 exact, 5 by clear synonym), checked against the same categories file. It carries no licence
  file and so cannot be ingested as it stands — KAN-110. No PNG was opened; the colour finding is
  from source code.

## Open questions

- **Should CI run any line of `SiglipEncoder`?** **Tracked as KAN-106.** Today it runs none: CI installs `.[dev]`, `torch`
  is only in the `drawing` extra, and `tests/drawing/test_siglip_encoder.py` skips wholesale. Both
  bugs this slice found, and both the review found, lived in exactly that region. *Recommended:* a
  separate, non-blocking CI job that installs the extra and runs that file on CPU — a few minutes
  of wall clock for the only coverage of the code that produces the numbers. *Decides:* the KAN-38
  owner; it changes `.github/`, which AGENTS.md keeps out of a feature PR.
- **Should the `drawing` extra be pinned like the rest?** `tests/test_requirements.py` covers
  `dependencies`, `dev` and `dvc` only, so `transformers` floats anywhere in `>=5,<6`. The
  checkpoint sha is pinned precisely so a rerun reproduces a measured number, and `model.pooled`
  exists because the return shape changed between majors — a minor is the same class of risk
  against the same numbers. *Recommended:* add the extra to the pinned set when CI installs it.
- **Where does a zero-shot number get logged?** **Tracked as KAN-107.** `AGENTS.md` requires every model comparison to go
  to MLflow, and nothing from S6a is in one: the baseline's figures live in these Verified facts
  only. `docs/engineering/model-development.md` still has an open question about who hosts the
  shared MLflow server, so there is nowhere to log to yet. *Recommended:* settle the host before
  S6c produces the first `drawing_top5_acc`, which is a number that genuinely needs comparing.
- **Does the recogniser see colour, or greyscale?** The pipeline answers this inconsistently today
  — see the Verified fact above — and nothing has measured what the inconsistency costs. Zero-shot
  the asymmetry is probably mild, since SigLIP 2 was trained on photographs; the concern is after
  S6d, where a tower fine-tuned only on greyscale renders can regress on colour drawings while the
  gate, scored on those same renders, stays green. That is reasoning, not measurement.
  *Recommended:* give the recogniser greyscale, since S6b1's preprocessing module has to produce
  it anyway, and keep colour for the character sheet and KAN-42, where it matters far more than it
  does for naming an object. That needs a colour field on `Character`, which does not exist. S6c2's
  greyscale variant measures the cost either way. *Decides:* the KAN-38 owner with the KAN-31 owner,
  before S6d — and before S6b1 is built, since it decides whether greyscale is a variant or the
  default path.
- **Does the 345-category vocabulary fit what children actually draw?** **Tracked as KAN-111.** It
  cannot name a whole person
  (see the Verified fact above), and KAN-64's flow is "name each character" — children draw
  people constantly. The recogniser will answer `angel` or `snowman`, confidently, and no amount of
  fine-tuning adds a word the label set does not contain. This is a product question before it is a
  model one, and it also decides what an accuracy number on real child drawings could ever mean:

  | Corpus | Real children | How it was captured | Subjects the 345 can name | Gives an accuracy number? |
  |---|---|---|---|---|
  | Quick, Draw!, rendered | no — adults, 20s timer | drawn in a browser | all 345 | yes, and it is the gate's |
  | Amateur Drawings | yes | **uploads, mix unknown** | **none** — human figures only | no |
  | kiddraw (KAN-110) | yes, ages 2–10 | natively digital, museum kiosk | 44 of its 48, but **not** `person` | yes, if the licence clears |

  "Photographed" is not established for Amateur Drawings and this plan should stop implying it: the
  source docstring says Meta resized "the uploaded drawings" to 512 px, and `configs/sources.yaml`
  says only "PNG images". Uploads to a web demo plausibly mix photographs, scans and natively
  digital drawings. The mix is checkable — the corpus is frozen locally — and it is a confound for
  S6c2's "as uploaded" variant, so it is worth reporting alongside those figures. Note too that
  kiddraw drops `person`, so it answers the corpus problem and not the vocabulary one.

  No public dataset of labelled **colour** children's drawings was found on 2026-11-14, searching
  for labelled child-drawing corpora and following the leads to kiddraw and ChildlikeSHAPES; that
  is an absence from one afternoon's search, not a proof. *Recommended:* settle what the metric is
  for before S6d optimises against it. *Decides:* the KAN-38 owner with the product owner.

- **Should `Character.__post_init__` reject non-finite confidences?** Its ranking check passes on a
  list containing NaN, because CPython's list comparison short-circuits on identity, so the
  invariant `drawing_top5_acc` is scored from is not actually enforced. S6a guards its own side in
  `drawing/model.py`, but every other producer of a `Character` — the Fake, S6d, anything KAN-42
  builds — has the same hole. *Recommended:* add the check in `interface.py` where the invariant is
  declared, rather than in each caller. *Decides:* the KAN-38 owner; it touches a shared contract,
  so it should not ride along inside a model slice.
- **`DrawingRecognizer.analyse`'s docstring says it fills every field except `reference_image_ref`
  and `approved`, which the real model does not do.** S6a deliberately leaves `description`,
  `body_type`, `crop_ref`, `mask_ref` and `keypoints` at their defaults for S6e, and `build_book`
  accepts `description=""` without complaint — so the first person to see a book built on an empty
  description will see it in serving. *Recommended:* say in the Protocol which fields may be unset
  before S6e lands, rather than discovering it downstream. *Decides:* KAN-38 with KAN-42.


- Should `tests/drawing/test_render.py`'s Pillow check be gated on a real extra rather than on
  whether Pillow happens to be installed? Recommended: once S6a lands the `drawing` extra, add
  `pillow` to it and point the skip at that, so the check can be opted into rather than only
  arriving by accident through `transformers`. Whoever merges S6a.
- **Half of S6b's acceptance is deferred to S6b2, and the acceptance line does not say so.** "Training
  and inference import the same function, asserted by a test" is asserted today as "defined exactly
  once and nothing else draws", because nothing imports it yet; and "`license` and `attribution`
  stay on every rendered record" belongs to the slice that handles records. Both are the right
  calls, but S6b2 inherits them. *Recommended:* S6b2 adds the import assertion and the column
  check, and this line moves with them.
- **What should the S6b2 loader do when `render_strokes` raises mid-epoch?** The renderer refuses a
  malformed drawing rather than rendering something plausible, and it will be called once per
  record across hundreds of thousands. `preprocess` filters on `valid_strokes` before writing, so
  processed records should never trip it — but "should never" is not a policy. Related: `valid_strokes`
  accepts three runs (the raw format's timings) where the renderer requires two, so the two
  definitions of a valid stroke have already diverged. *Recommended:* S6b2 decides, and says so.
- Should the renderer's per-record cost be re-measured over the real S6b2 split? Recommended: yes,
  as part of S6b2 — the 0.356 ms figure is 50 drawings of one sparse category, and the obvious
  optimisation (indexing the canvas from the point list instead of building a list of tuples) has
  not been tried because nothing has needed it yet.

- Which region has L4 GPU quota for the team's projects? Recommended: request quota in `us-central1`
  now; the team lead decides.
- Does the course's GCP credit cover Claude models on Vertex AI? Recommended: check the billing account's
  credit scope before routing Claude Code through Vertex. Rishi checks.

## Out of scope

- Whisper read-aloud scoring and IP-Adapter character consistency: after the MVP.
- Bias slicing and drift alerts: PRs #7 and #8.
