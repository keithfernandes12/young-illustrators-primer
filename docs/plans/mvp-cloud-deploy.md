---
title: MVP on Google Cloud — models v0, serving, deploy, gates and monitoring
type: plan
created: 2026-11-07
updated: 2026-11-14
status: draft
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
cached weights, mirroring `primer.illustration.smoke`.

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

### S6b2 · Rendered QuickDraw split for the recogniser

*size: medium*

`src/primer/drawing/data.py`: train/val/test over the frozen QuickDraw version through S6b's
renderer, with KAN-31's paper/crayon augmentation applied at training time only.

- **Why separate** — it is the first slice that needs frozen data on disk, which S6b does not.
- **Acceptance** — the loader returns the three splits for a pinned version, no record appears in
  two splits, and augmentation is off for val and test.
- **Depends on** — S6b, and KAN-31 for the augmentation set. **Jira** — KAN-38.

### S6c · `drawing_top5_acc` from the golden split

*size: small*

`src/primer/drawing/evaluate.py` and `slices.py`: run S6a's baseline over S6b's test split, write
the `primer.evaluation.contract` shape, and report the gated key per slice as well as overall.

- **Acceptance** — `check_gate.py --only drawing` returns a verdict rather than `metric missing`,
  and the baseline number is recorded as a dated Verified fact whether or not it clears 0.85.
- **Depends on** — S6a, S6b2. **Jira** — KAN-38.

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
  S6c is compared with the S6a baseline in Verified facts, against the 0.85 target.
- **Depends on** — S6b2, S6c. **Jira** — KAN-38.

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
| S6a | not started | | ready: needs no dataset |
| S6b | not started | | ready: pure function, no data. KAN-87, taken over by this stage |
| S6b2 | not started | | |
| S6c | not started | | |
| S6d | not started | | first slice that trains; builds its own `--max-steps` cap |
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

- **2026-11-14** — The drawing stage takes over **KAN-87** (`render_strokes`), which was assigned
  elsewhere and unstarted since 2026-11-08 — no branch, no PR, no `render_strokes` in the tree. It
  is the hard dependency of every measured slice in S6, so leaving it where it was left S6c and S6d
  blocked behind work nobody had begun. Split out as S6b, a pure function with no frozen data, and
  deliberately kept as **one** renderer imported by training and inference alike: KAN-87's whole
  point is that the two cannot drift, and that reason survives the change of owner. The dataset
  split that was S6b is now S6b2. **Tell the previous assignee before starting S6b.**
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

## Open questions

- Which region has L4 GPU quota for the team's projects? Recommended: request quota in `us-central1`
  now; the team lead decides.
- Does the course's GCP credit cover Claude models on Vertex AI? Recommended: check the billing account's
  credit scope before routing Claude Code through Vertex. Rishi checks.

## Out of scope

- Whisper read-aloud scoring and IP-Adapter character consistency: after the MVP.
- Bias slicing and drift alerts: PRs #7 and #8.
