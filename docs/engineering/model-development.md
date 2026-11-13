---
title: How six people build six models at once
type: engineering
created: 2026-11-12
updated: 2026-11-13
status: active
author: rishiparekh
tags: [models, mlops, gcp, parallel, process]
related: [engineering-loop, ci, mvp-cloud-deploy, model-stages-and-data-coverage]
---

# How six people build six models at once

Every model area in this project follows the same lifecycle, writes the same metrics file, logs to the
same tracking server and registers the same way. The point is not uniformity for its own sake: it is
that six people can work in six directories at the same time without blocking each other, and that one
evaluation gate can read all six.

**Scope.** The six stages in focus, numbered as in `model-stages-and-data-coverage`:

| Stage | Area | Trains? |
|---|---|---|
| 1 · Drawing recognition | `src/primer/drawing/` | fine-tune |
| 2 · Story generation | `src/primer/story/` | LoRA |
| 4 · Safety gate | `src/primer/safety/` | threshold tuning, optional fine-tune |
| 5 · Illustration | `src/primer/illustration/` | pretrained (style LoRA is a stretch goal) |
| 6 · Character consistency | `src/primer/illustration/consistency.py` | pretrained |
| 8 · Read-aloud ASR | `src/primer/speech/` | pretrained, optional fine-tune |

Three of the six do not train anything. The course guidelines are explicit that this does not excuse
them: a pretrained model still needs validation, bias detection and experiment tracking. Those three
areas therefore do the same steps 0–4 below, and skip only step 5.

**Stage 3 is in, as a seventh.** The readability scorer was not in the original six, which left only
stages 1 and 2 training anything — and the scoping doc's §11 requires three models trained by the team.
Stage 2's gate also needs a reading-level number. It belongs to member 3 alongside safety: the same
corpus, the same gate, the same CPU-only profile. `src/primer/readability/` already exists with a
`ReadabilityScorer` protocol and a fake, and two of its keys are gated.

---

## 1. What must exist before anyone starts

Six people writing model code against six different conventions produces six models that cannot be
compared, gated or served. Three of the four prerequisites are done; the fourth is not, and it is the
one agents cannot do.

1. **`src/primer/evaluation/`** — *done.* The metrics contract, the golden-set runner and
   `configs/golden/scenarios.jsonl`. `check_gate.py` now returns PASS or FAIL, so
   `GATE_RESULT: SKIP` means exactly one thing: the metrics file is missing.
2. **The area skeleton** — *done.* All six areas exist with the layout in section 2 (stage 6 lives
   inside `illustration/`, which is why seven stages need six directories).
3. **The frozen data version** — everyone pins **`v1.1`**, the tag `main` carries today (`data-v1.1`,
   from KAN-24). Bumping it is a team decision, never an individual one.
4. **The shared GCP services** — section 6, and still outstanding. Person-only, and the GPU quota
   requests want to be in flight now rather than on the day someone first needs a GPU.

Until the fourth lands, nobody can run a real training job, however finished their area's code is.

## 2. The contract every area implements

This layout exists in every area already: `interface.py` is real, the rest are stubs that raise
`NotImplementedError`, and `primer.serving.book.build_book` composes all six on fakes.
`src/primer/readability/` is the worked example to copy from — not a layout to create.

```
src/primer/<area>/
  interface.py      # the Protocol serving imports, plus Fake<Thing> for tests
  data.py           # read_dataset(..., version=PINNED) -> train / val / test
  model.py          # the real implementation; weights load lazily
  train.py          # CLI: --config --data-version --out; logs one MLflow run
  evaluate.py       # CLI: --model --split --out metrics.json
  slices.py         # this stage's slice keys, reusing primer.data.bias
  sensitivity.py    # SHAP for classifiers, ablation for generative stages
  register.py       # MLflow registry + Artifact Registry push
  configs/*.yaml    # the hyperparameter search space, versioned with the code
  Dockerfile        # the same image runs locally and on Vertex AI
  CARD.md           # model card: data version, metrics, slices, mitigations, trade-offs
```

**Boundaries** (from `mvp-cloud-deploy`, unchanged): `serving` may import any model area; a model area
never imports `serving`; nothing imports `primer.data` at request time. Tests never load real weights,
never touch GCS and never reach the network.

### The metrics file

`evaluate.py` in every area writes the same shape. This is the integration point: the golden runner
merges these into one file, and `golden-gate` reads it.

```json
{
  "stage": "drawing",
  "model_uri": "models:/primer-drawing/7",
  "data_version": "v1.1",
  "git_sha": "afa1e9a",
  "overall":  { "drawing_top5_acc": 0.87 },
  "baseline": { "drawing_top5_acc": 0.71 },
  "slices": [ { "key": "country_code", "value": "US", "n": 1240,
                "metrics": { "drawing_top5_acc": 0.89 } } ]
}
```

Keys inside `overall` must be the keys in `.claude/skills/golden-gate/thresholds.json` — `drawing_top5_acc`,
`target_word_inclusion`, `story_quality_mean`, `reading_level_within1`, `readability_rmse`,
`safety_violations_golden`, `clip_consistency`. A missing key is a gate failure, not a gap in the report.
`baseline` is what the model must beat and is never omitted; for a pretrained stage it is the
off-the-shelf model's own score, which is what makes step 1 below worth doing.

### Tracking conventions

Every run, local or cloud, logs to the shared MLflow server. The experiment name is fixed by
`train-vertex`: **`MLFLOW_EXPERIMENT_NAME=yip-<component>`**. On top of that, these tags, without which
runs from six machines cannot be compared:

`stage` · `owner` · `data_version` · `git_sha` · `base_model` · `gate_group` · `where` (`local` | `vertex`)

Params are the whole config file. Metrics are everything in `overall`, plus each slice as
`slice/<key>=<value>/<metric>`. Artifacts are `metrics.json`, the bias report, the comparison plots the
guidelines ask for (bar plot across candidates, confusion matrix where the task has classes) and
`CARD.md`.

### Registry and rollback

- **Images** → `$YIP_REGION-docker.pkg.dev/$YIP_SHARED_PROJECT/yip/<service>:<git-sha>`, which is the
  Artifact Registry push the guidelines require.
- **Checkpoints and adapters** → `gs://$YIP_ARTIFACT_BUCKET/models/<component>/<run>/`, never into
  `gs://young-illustrators-primer-data`. The job's service account writes them; the session does not.
- **Weights** → MLflow Model Registry, with aliases `production` and `rollback`.
- **Only a `GATE_RESULT: PASS` run may take the `candidate` alias.** Registration is gated on the
  golden gate, not on training finishing. A candidate that scores worse than the current `production`
  on its gate group never gets promoted.
- **Rollback** is the `rollback` alias plus the previous Cloud Run revision; the `deploy-cloud-run`
  skill already does the revision half.
- **Concurrency** is capped by `YIP_MAX_TRAIN_JOBS` (default 4 launches per goal), with at most one
  GPU job per component at a time. Never cancel another member's job.

## 3. The seven steps, identical for every stage

One PR per step. Steps 0 and 1 land in the first week, for all six areas at once.

| Step | What | Done means |
|---|---|---|
| 0 | Interface + Fake | Another member can import your area and write a test against it |
| 1 | Baseline, pretrained or zero-shot, measured on the hold-out | A `baseline` number exists in `metrics.json`; nothing later ships without beating it |
| 2 | Validation harness | `evaluate.py` writes the contract file; the golden runner picks up your keys |
| 3 | Bias slices + report | Per-slice metrics logged; disparities named; a mitigation proposed or ruled out in `CARD.md` |
| 4 | Sensitivity analysis | Which inputs and which hyperparameters actually move the metric |
| 5 | Train and tune (pretrained stages skip) | An MLflow run with the search space, beating step 1 |
| 6 | Register + containerize | Image in Artifact Registry, model in the registry, `CARD.md` complete |
| 7 | Deploy to Cloud Run | The smoke test in `deploy-cloud-run` passes against the real service. **The first real deploy of each service is watched by a person, never run from a loop** — nothing in `deploy-cloud-run` or `train-vertex` has yet made a real Google Cloud call |

Step 1 before step 5 is the rule that keeps the project honest: a fine-tune that cannot beat the
zero-shot baseline is a finding worth reporting, not a failure worth hiding, and we only know which it
is because the baseline was measured first.

### What each step means per stage

These differ enough that a shared definition would be useless. Slice keys reuse `src/primer/data/bias.py`,
which already implements country, gender-swap, era, genre and picture-book origin slices with a
100-row floor on judging a slice.

| Stage | Hold-out metric | Slices | Sensitivity | Compute |
|---|---|---|---|---|
| 1 Drawing | top-5 accuracy on strokes; the same on the labelled child-style split | `country_code`, category | stroke count, line thickness, partial erasure | T4/L4, hours |
| 2 Story | target-word inclusion, judge score, level within ±1 | gender-swapped prompts, origin, age mode | drop each prompt constraint in turn; temperature sweep | L4, hours |
| 4 Safety | violations on golden, false-positive rate on benign | age mode, theme, gender-swap | threshold sweep — the real tuning here; word-level LIME | CPU |
| 5 Illustration | CLIP consistency, p95 seconds per page | subject category, skin-tone and origin prompts | diffusion steps, guidance scale, ControlNet weight | L4, the largest GPU cost |
| 6 Consistency | agreement with human ranking on the golden set | subject category, dominant colour | choice of embedding model; threshold sweep | CPU |
| 8 ASR | word error rate | speaker age, accent | added noise, speed perturbation | CPU/T4 |

Stage 8 cannot complete steps 1–4 today: MyST is registered in `configs/sources.yaml` with no pipeline
and access is by request. Treat that as the first thing that member does.

Stages 1 and 5 now share a dataset, which changes who talks to whom. `character_drawings` (PR #10)
holds child-style raster art with a bounding box, 17 COCO joints and a segmentation polygon per
character, and no category. For stage 1 that makes it an evaluation set once someone labels it, never
a training set. For stage 5 the segmentation and joints are conditioning signal — the character's
outline and pose, which is what ControlNet wants — so members 1 and 4 should agree on how it is read
before either writes a loader.

For stages 2, 5 and 8, SHAP and LIME do not apply — there is no feature vector to attribute. Ablation
is the honest substitute: remove one conditioning input at a time and measure the metric. Say so in
`CARD.md` rather than producing a SHAP plot that means nothing.

## 4. Who owns what

Each member owns one directory outright. Nobody edits another member's area; the three members whose
stages are pretrained carry a shared component as a second hat, because steps 0–4 on a pretrained model
is roughly half the work of a fine-tune.

| Member | Primary stage | Second hat |
|---|---|---|
| 1 | Stage 1 — drawing recognition | the labelled child-style eval split, and reconciling two drawing datasets |
| 2 | Stage 2 — story generation | the instruction-pair builder |
| 3 | Stage 4 — safety, **and stage 3 — readability** | the readability back-fill that stage 2 and the scoping doc's §11 both depend on |
| 4 | Stage 5 — illustration | the GPU container and Cloud Run GPU deploy |
| 5 | Stage 6 — consistency | `src/primer/evaluation/` and the golden set — the shared contract |
| 6 | Stage 8 — ASR (data-blocked) | `src/primer/serving/` and the CI eval gate |

Shared files have one named owner and change only by PR with that owner reviewing:
`src/primer/evaluation/` and `configs/golden/` (member 5), `src/primer/serving/` (member 6),
`.github/` (member 6), `infra/` (member 4), `src/primer/data/` (whoever owns data engineering).

### Why this parallelises

- **One directory per member** removes almost every merge conflict by construction.
- **Fakes land first**, so nobody waits for anyone's weights.
- **Interfaces land before implementations**, so the serving path can be built against six areas that
  are all still empty.
- **Everyone pins the same data version**, so two members' numbers are comparable.
- **The metrics contract is agreed before any model exists**, so no one rewrites their evaluation code
  to fit someone else's.

## 5. Mapping the course guidelines to this repo

| Guideline | Where it lives | Tool |
|---|---|---|
| Load data from the versioned pipeline | `<area>/data.py` → `read_dataset(..., version=)` | existing, done |
| Train and select the best model | `<area>/train.py`, candidates compared in MLflow | MLflow |
| Model validation on a hold-out | `<area>/evaluate.py`, book-level and speaker-level splits | existing split rules |
| Bias detection by slicing | `<area>/slices.py` + `primer.data.bias` | Fairlearn |
| Code that checks bias and reports | `primer-data bias-report` extended per model | Fairlearn |
| Push to Artifact / Model Registry | `<area>/register.py` | Artifact Registry + MLflow |
| Hyperparameter tuning | `<area>/configs/*.yaml`, search space in git | Optuna |
| Experiment tracking | every run, tagged as in section 2 | MLflow |
| Result visualisations | logged as run artifacts | matplotlib |
| Sensitivity analysis | `<area>/sensitivity.py` | SHAP / LIME / ablation |
| CI/CD triggers training and validation | `.github/workflows/` | GitHub Actions |
| Automated bias detection that can block | the gate's bias group | `golden-gate` |
| Notifications | email, per the existing decision | Actions + Cloud Monitoring |
| Rollback | registry alias + previous revision | `deploy-cloud-run` |
| Docker format | one Dockerfile per area | the same image local and on Vertex |

## 6. GCP layout: what is shared, what is personal

The rule that makes consolidation painless: **no durable state lives in a personal project.** Personal
projects run compute and nothing else. The two variables are set in `.claude/settings.local.json`, and
`guard_cloud.py` accepts either and refuses a third; `YIP_PROJECT` alone still works for a
single-project checkout. There is no production project — a guard that keyed on one refused six of six
everyday commands, because the project id is also the repository name.

**`YIP_SHARED_PROJECT` — the team project, from day one:**

- `gs://young-illustrators-primer-data` — already exists. All six accounts need
  `roles/storage.objectViewer` on it. One copy of the data, not six.
- **MLflow tracking server** — one URL everyone logs to, local runs included. Host it on an
  `e2-micro` in `us-central1`, `us-west1` or `us-east1`, which is free-tier eligible, running MLflow and
  Postgres under Docker Compose with artifacts in GCS. Confirm the free-tier terms before relying on them.
- **Artifact Registry `yip`** — one repository all six push to. Six copies of a multi-gigabyte CUDA image
  is a real line item.
- `gs://yip-models` (`$YIP_ARTIFACT_BUCKET`) — checkpoints and adapters, separate from the data bucket.
- The served Cloud Run revision.

**`YIP_COMPUTE_PROJECT` — each member's own project:**

- Vertex AI custom training jobs, or Cloud Run GPU jobs, burning that member's own credits.
- Nothing else. No second copy of the data, no second registry, no second tracking server.

**Every grant above is person-only.** Agents cannot make them: `gcloud projects
add-iam-policy-binding` and service-account key creation are denied in `.claude/settings.json`,
`guard_cloud.py` refuses them in both projects, and `AGENTS.md` forbids JSON keys outright. Do not file
them as agent work. The same goes for the GPU quota: a new project starts at **zero**, and an increase
can take days or be refused outright on credit billing, so those requests want to be in flight now.

**Writes to the data bucket are person-only too.** Agent sessions get `PRIMER_DATA_URI=.local-data`
from `.claude/settings.json` and write to that git-ignored folder; writes to
`gs://young-illustrators-primer-data` go through `freeze-dataset-version`, run by a person. An agent
reading this document should not assume the bucket is writable.

**Consolidation** is then changing `YIP_COMPUTE_PROJECT` and stopping. Nothing migrates.

One caveat that decides the order: **credits attach to a billing account, not to a project.** Moving a
personal project onto the team billing account abandons that project's credits. So keep personal
projects on personal billing until their credits are spent, and put the shared services on the team
billing account from the start — not the other way round.

## 7. Spending $100 a head

Figures below are order-of-magnitude and need confirming in the pricing calculator before anyone
commits to them; Vertex AI custom training bills Compute Engine rates plus a management fee, and Spot
VMs are supported and billed at Compute Engine Spot rates.

**Do not use:**

| Service | Why not | Instead |
|---|---|---|
| Cloud Composer | managed Airflow alone would eat the entire budget | the existing Airflow in `docker-compose.yaml` |
| Vertex AI Endpoints | the node bills whether or not anyone is reading a book | Cloud Run, which scales to zero |
| A100 / H100 | several dollars per GPU-hour; nothing here needs one | L4, or T4 for the CPU-ish stages |
| Per-member data copies | six times the storage and six divergent datasets | read access to the one bucket |

**Do use:** Spot GPUs for every training job, with checkpoints to GCS so a preemption costs minutes
rather than the run. Cloud Run with `min-instances=0`. Cloud Build's free daily minutes. The free-tier
`e2-micro` for the tracking server.

**The largest saving is the one the training ADR allows:** iterate locally. Agent sessions never train;
a member may train on their own machine; any run that is *registered* must be reproducible from the
area's container on Vertex AI, with the same frozen data version and the run logged to the shared
MLflow. So the thing credits pay for is the registration, not the laptop experiment that motivated it —
and each member spends their own credits in their own `YIP_COMPUTE_PROJECT`. Local GPU work runs from
Git Bash as well as WSL; `model-containers-on-windows.md` has the recipe.

A rough split per member: about $40 held back for the shared deploy and demo window, leaving about $60
of training. On Spot L4 or T4 that is a few hundred GPU-hours, which is far more than a LoRA on a 1–3B
model or a ConvNeXt fine-tune consumes. Remember that credits attach to a billing account, not a
project: moving a personal project onto team billing abandons its credits.

The one stage that can break this budget is stage 5: illustration is GPU-bound at inference, not at
training, and every golden-set run generates images for about 100 scenarios. Cap the golden set's image
count, use a Turbo or LCM variant with few steps during development, and measure `cost_per_book_usd`
early — it is already a gate metric with a $0.25 threshold.

## 8. Open source instead of a GCP service

| Need | The GCP service | What we use | Why |
|---|---|---|---|
| Experiment tracking | Vertex AI Experiments | self-hosted MLflow | free, and §9 of the scoping doc already chose it |
| Hyperparameter tuning | Vertex AI Vizier | Optuna | free, and runs unchanged inside the training container |
| Bias slicing | TensorFlow Model Analysis | Fairlearn | already a dependency; `bias.py` already uses it |
| Model registry | Vertex Model Registry | MLflow Registry, images in Artifact Registry | the guidelines ask for Artifact Registry; MLflow adds aliases and stages |
| Orchestration | Cloud Composer | Airflow in Docker Compose | cost |
| Monitoring and drift | Vertex Model Monitoring | Evidently + Cloud Monitoring | free tier covers our volume |
| Explainability | Vertex Explainable AI | SHAP, LIME, ablation | works on the three pretrained stages too |

GCP is still doing the work that only it can do: GPU training jobs, Artifact Registry, Cloud Run,
Cloud Storage, Cloud Build, Cloud Monitoring and the budget alerts.

## 9. What CI does, and when

`docs/engineering/ci.md` governs changes to CI; new gates start warn-only.

- **Every PR** — ruff, mypy, pytest, the privacy lint, and the golden gate against *fakes*. Fast, no
  GPU, no credentials. This is what proves the contract holds, not what proves the model is good.
- **On demand, manually dispatched** — the golden gate against the registered models, and, on a model
  PR, that area's training job on the member's compute project, then validation, then bias checks, then
  the gate. Training does not run on every push: it would exhaust the credits in a week.
- **Nightly — parked, deliberately.** The nightly gate workflow was left out because its metrics step
  is not `continue-on-error`, so one bad night fails the whole job instead of warning. The unpark
  condition is recorded in `docs/plans/autonomous-loops.md`.
- **On a gate failure** — email, and the registry alias does not move.

## 10. Decisions still needed

- **Who hosts the shared MLflow, and on whose billing?** Recommended: the infra owner, on the team
  billing account, so it outlives the personal projects. — *member 4*
- **Spot or on-demand for training?** Recommended: Spot everywhere, with checkpointing, since no
  training run here is latency-sensitive. — *member 4*
- **Does the golden set cap generated images?** Recommended: yes, a sampled subset for development runs
  and the full set nightly. — *member 5*
- **Has MyST access been requested?** Until it has, stage 8 cannot be validated at all. — *member 6*

Record the consolidation plan as an ADR once section 6 is agreed; it is a decision others must follow,
not a process.
