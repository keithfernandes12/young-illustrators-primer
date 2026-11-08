---
title: MVP on Google Cloud — models v0, serving, deploy, gates and monitoring
type: plan
created: 2026-11-07
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

`infra/` Terraform for two projects (staging, prod): APIs, Artifact Registry `yip`, models bucket
(separate from `gs://young-illustrators-primer-data`), Firestore, Pub/Sub topics, Secret Manager,
service accounts per service, Workload Identity Federation for GitHub Actions, and a budget with
50/90/100% email alerts.

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
| S2 | not started | | unblocks S3–S6, S11 |
| S3 | not started | | |
| S4 | not started | | |
| S5 | not started | | |
| S5b | not started | | Vertex GPU job |
| S6 | not started | | |
| S7 | not started | | needs GPU quota in region |
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

## Verified facts

## Open questions

- Which region has L4 GPU quota for the team's projects? Recommended: request quota in `us-central1`
  now; the team lead decides.
- Does the course's GCP credit cover Claude models on Vertex AI? Recommended: check the billing account's
  credit scope before routing Claude Code through Vertex. Rishi checks.

## Out of scope

- Whisper read-aloud scoring and IP-Adapter character consistency: after the MVP.
- Bias slicing and drift alerts: PRs #7 and #8.
