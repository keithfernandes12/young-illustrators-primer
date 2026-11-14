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

### S6 · Drawing recognizer

*size: medium*

`src/primer/drawing/`: zero-shot SigLIP/CLIP baseline over the QuickDraw categories, then a fine-tune on
the frozen QuickDraw version on Vertex AI (`train-vertex`), plus the parent-label correction path.

- **Acceptance** — golden `drawing_top5_acc` for baseline and fine-tune, against the 0.85 target.
- **Depends on** — S2. **Jira** — to create.

### S7a · Illustration model, pretrained baseline

*size: small*

`src/primer/illustration/model.py` and `sketch.py`: `IllustratorModel` behind the merged
`Illustrator` contract. Each page loads the child's sketch (the in-memory bytes, or re-read from a
`gs://` or `file://` `drawing_ref`), turns it into a white-on-black square control image, and runs
SD 1.5 + ControlNet-Scribble v1.1 at the pinned revisions and settings in
`configs/illustration_baseline.yaml`. The page is written to a local folder under a random name and
returned as a `file://` `image_ref`. The diffusion stack is the `illustration` extra; CI does not
install it, and the tests use a recording fake pipeline.

- **Why separate** — the model can be built and proved locally now; S7b needs S1's bucket and GPU
  quota, which are person-only and not in place.
- **Acceptance** — the tests in `tests/illustration/` pass with no weights or network; `python -m
  primer.illustration.smoke` produces one 512×512 page from a synthetic sketch with the cached weights.
- **Not in this slice** — IP-Adapter on `characters` (pages return `identity=None`), the style LoRA,
  reposing, the consistency score (`consistency()` raises), GCS output, the GPU container, the first
  storybook reference image (open with KAN-38), and the generated-image safety check (KAN-94).
- **Depends on** — nothing. **Jira** — KAN-42.

### S7b · Illustration worker on Cloud Run

*size: medium*

S7a's model behind a GPU worker: pages written to GCS, a container for Cloud Run with an L4 GPU.

**Before `build_book` can use `IllustratorModel`, `consistency()` must work.** `build_book` calls it on
every book, and in S7a it raises `NotImplementedError`, so wiring the model in now would fail every
book even though `isinstance(model, Illustrator)` passes. The real score is stage 6. If S7b lands first,
it needs an agreed interim score instead, and that score must be labelled so the gate cannot mistake it
for a measurement.

- **Acceptance** — `clip_consistency` reported; p95 time per page measured on staging.
- **Depends on** — S1, S7a, a working `consistency()` (above), and KAN-94 before any real child's
  drawing is illustrated. **Jira** — KAN-42.

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
- **Depends on** — S5, S8 (fakes are fine for S6, S7a, S7b). **Jira** — to create.

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
| S2 | in review | [#11](https://github.com/keithfernandes12/young-illustrators-primer/pull/11) | golden set + runner + gate done on fakes; real-model evaluation still open. Unblocks S3–S6, S11 |
| S3 | not started | | |
| S4 | not started | | |
| S5 | not started | | |
| S5b | not started | | Vertex GPU job |
| S6 | not started | | |
| S7a | in progress | branch `feat/KAN-42-controlnet-baseline` | pretrained SD 1.5 + ControlNet-Scribble, one page, local output; synthetic inputs only until KAN-94 |
| S7b | not started | | needs S1, GPU quota in region, a working `consistency()`, and KAN-94 |
| S8 | not started | | |
| S9 | not started | | |
| S10 | not started | | |
| S11 | not started | | |
| S12 | not started | | |
| S13 | not started | | |
| S14 | not started | | |

## Decisions

- **2026-11-07** — Pretrained baseline first, then the team-trained model, for each of the three models
  §5.4 requires; both measured by the golden gate. Eight days remain before the MVP and every model
  area is empty, so the baseline keeps the end-to-end path alive if a training run slips.
- **2026-11-14** — S7 is split into S7a (the model, provable locally) and S7b (the Cloud Run worker),
  because S7b waits on S1 and GPU quota while S7a needs neither.
- **2026-11-14** — S7a uses SD 1.5, not SDXL: it runs in fp16 on a 16 GB laptop GPU, and the scoping
  doc's §5.3 leaves "SDXL / 1.5" open. SDXL is a sensitivity question for later, not a baseline.
- **2026-11-14** — The generated-image safety check lives in serving, after `illustrate()`, not in the
  model (KAN-94). Diffusers' built-in checker is off because it returns a black image and reports
  nothing, which would reach a `complete` book as a blank page. Until KAN-94 lands, S7a runs only on
  synthetic sketches.
- **2026-11-14** — The sketch is read from the in-memory bytes when serving passes them, and from
  `drawing_ref` only otherwise; `http(s)://` and `fixture://` refs are refused rather than fetched.
- **2026-11-14** — From an independent review of S7a:
  - The stroke threshold is relative to the paper, not a fixed 128. At 128, a transparent canvas
    export, grey pencil, yellow crayon and a dim phone photo all produced empty control images.
  - A drawing with no strokes is refused rather than generated from nothing.
  - Every device loads the pinned fp16 files, upcast to fp32 on CPU, because the fp32 files are not
    cached and an offline CPU run could not fetch them.
  - Read errors drop the path, which can be the child's file name.

## Verified facts

- **2026-11-13** — Local baseline on an Apple M4 (16 GB, MPS, fp16), SD 1.5 rev `451f4fe` +
  `control_v11p_sd15_scribble` rev `3564ec7`, UniPC, 25 steps, guidance 7.5, ControlNet scale 1.0,
  seed 42, 512×512: 44.2 s to generate one page, 3.4 s to load from cache. Download is 2.86 GB of fp16
  weights. Edge recall against the sketch was 0.993, against 0.619 for the same sketch rotated 180°.
  Checked by running the standalone baseline script (Adit's local copy, not in the repo).
- **2026-11-14** — `python -m primer.illustration.smoke` on the same synthetic sketch, through a
  `file://` `drawing_ref`, on the same machine: 39.8 s to generate, 5.7 s to load on the
  committed code (39.3–42.5 s and 5.1–5.9 s on earlier runs). The house, roof, door, round window, cloud ellipses and sun kept their places. At that speed a 6-page book takes
  about 4–5 minutes locally, well over the 120 s `p95_full_book_s` target, so S7b needs the L4 or a
  few-step sampler (LCM/Turbo).

## Open questions

- Which region has L4 GPU quota for the team's projects? Recommended: request quota in `us-central1`
  now; the team lead decides.
- Does the course's GCP credit cover Claude models on Vertex AI? Recommended: check the billing account's
  credit scope before routing Claude Code through Vertex. Rishi checks.

## Out of scope

- Whisper read-aloud scoring and IP-Adapter character consistency: after the MVP.
- Bias slicing and drift alerts: PRs #7 and #8.
