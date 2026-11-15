# Model card: story (stage 2)

Writes a short picture-book story for a child aged 3-5, as JSON pages, using every target word, at a
young reading level, and free of the topics on the avoid-list. Plan:
`docs/plans/story-generation-llm.md` (Decisions 2026-11-15 for this submission's scope).

| Field | Value |
|---|---|
| Stage | story (2) |
| Purpose | write an age-appropriate story for ages 3-5 (Learn to Read only in this submission) |
| Candidates | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775`; `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (both Apache-2.0, pinned in `configs/story_grid.yaml`) |
| Training | none in this submission: zero-shot pretrained instruct models. The 4B candidates and QLoRA are later work |
| Baseline (frozen configuration) | Qwen2.5-0.5B-Instruct, temperature 0.7, max_new_tokens 512, top_p 0.9, seed 1234, 3 attempts |
| Selected configuration | _not yet measured_: the grid has not run on real models (needs bucket access for the requests) |
| Data version | `v1.1` (validation requests from the **val** split; `test` and the golden set untouched) |
| Level | `fk_formula@1` (`levels.fk_grade`: whole story, clamped at 0, rounded to 0.5) |
| Hardware | the owner's MacBook Air (Apple M4, 16 GB), MPS, fp16 |
| Owner | DiyaKaswa11 (member 2) |

## How it works

`LLMStoryWriter` builds a prompt (`prompts.py`: the 3-5 rules, the avoid-list **topics**, the JSON
shape), asks the model, parses the reply (`schema.py`), and **retries** -- with a new seed -- on exactly
three failures: invalid JSON, a missing target word, or a failed safety check. After 3 attempts it
raises `StoryGenerationError` rather than return a story that failed one of them (serving's handling
is KAN-121). Page and sentence limits and the reading level are measured, not retried on.

## Safety guardrails (owned by this stage)

- **Avoid-list** (`configs/avoid_list.yaml`, `avoid_list@1`): violence, weapons, death, fear, unkind
  words, adult topics. The prompt names the topics; the output is checked word by word (whole words,
  case-insensitive), and a flagged story is regenerated, never returned.
- **It is a floor, not a guarantee.** A blocklist misses anything phrased around it. Words that 3-5
  stories use innocently (kick, beat, arrow, shut) are left out on purpose, so they never flag a good
  story.
- **An independent second check** stays in serving (`build_book` runs member 3's `SafetyClassifier`
  after `write`). `returned_safety_violation_rate` is zero by construction under the writer's own
  guardrail, so it only means something when measured with a *different* classifier; the evaluation
  takes one (`judge=`) for that reason.
- **Privacy.** The character's name goes into the prompt only. Attempt logs, results and MLflow
  artifacts hold outcomes and counts, never story text or names (tested).

## Metrics

Definitions are in `evaluate.py`. Gated key this stage owns and measures: `target_word_inclusion`
(threshold 0.95). Also reported (not gated): first-attempt valid-JSON rate, success rate, page and
sentence limit rates, FK mean and within-±1 rate, raw safety-flag rate, returned-violation rate, mean
attempts, p95 latency.

| Metric | Selected | Baseline | Run id |
|---|---|---|---|
| target_word_inclusion | _not yet measured_ | _not yet measured_ | |

## Slices

`source` (picture-book origin or TinyStories) and `age` (3, 4, 5). With ~30 requests every slice is
below the 100-row floor `primer.data.bias` uses, so slices are reported with `judged: false`; no
disparity is claimed from them.

## Course Model Development guidelines: what is done, and what is not

| Guideline | Status | Where, or why not |
|---|---|---|
| Load data from the versioned pipeline | **Done (code); data pending** | `data.py` reads `v1.1` via `read_dataset(version=)`: one pair per book, cross-source exact dedupe, pure-hash split. The ~30 real requests wait on the owner's bucket read access |
| Train and select the best model | **Select: done. Train: deferred** | `grid.py` compares two pretrained models over a grid and applies a rule fixed in config before any run. No training: QLoRA on 4B candidates needs a CUDA GPU and is later work (plan Decisions 2026-11-15) |
| Validation on a hold-out | **Done** | Requests come only from the book-level **val** split; `test` is never read; the golden set is not used to choose |
| Bias detection by slicing | **Partial** | Per-slice metrics by source and age in `metrics.json` and MLflow. Gender-swapped prompt pairs (the slice `model-development` names) are deferred: they need paired requests |
| Code that checks bias and reports | **Partial** | `slices.py` reports every slice with its size and `judged`. Not Fairlearn `MetricFrame`: with all slices under the floor there is nothing to judge yet |
| Bias mitigation | **Not applicable yet** | Nothing is trained, so there is no training data to re-weight; `balance_by_swap` applies to training pairs (plan S2) |
| Hyperparameter tuning | **Done (grid)** | temperature × max_new_tokens per model, exhaustive. Optuna is unnecessary for 12 points; it arrives with QLoRA |
| Experiment tracking | **Done** | One MLflow run per grid point, local git-ignored SQLite store (`.local-data/mlflow.db`, artifacts in `.local-data/mlartifacts`), with the section 2 tags plus `level_source`; re-logged to the shared server when it exists |
| Result visualisations | **Done** | `charts.py`: comparison bars and the temperature sweep (needs the `story` extra) |
| Sensitivity analysis | **Partial** | Hyperparameter spreads (`sensitivity.py`). Prompt-constraint ablation is deferred (plan S8a). SHAP/LIME do not apply to a generative stage |
| `story_quality_mean` (gated) | **Deferred** | Needs an LLM-as-judge (plan open question 3). Listed in `metrics.json` under `not_measured`, never faked |
| Push to Artifact / Model Registry | **Deferred** | Registration requires a golden-gate PASS and a run reproducible on Vertex (ADR `bounded-local-training-for-agents`); weights are pinned by Hugging Face revision meanwhile |
| Docker | **Deferred** | `Dockerfile` is the area stub; a container cannot use the Mac's MPS, so this submission runs natively |
| CI/CD that triggers training and validation | **Partial** | CI runs every unit test here with a stub model. The real-model grid is manual on the Mac; a workflow change goes in its own `.github/` PR |
| Automated bias detection that can block | **Not applicable yet** | No gated bias key exists in `contract.py`; adding one is a team change, not this stage's |
| Notifications, rollback | **Deferred** | Both hang off registration and deployment (registry alias, Cloud Run revision), which come later |

## Limitations and trade-offs

- Zero-shot 0.5B/1.5B models will miss target words and break the JSON more often than a fine-tuned
  4B model; the retries trade latency for constraint following, and the grid measures the cost.
- FK on 1-3 sentence texts is noisy and pulls young stories toward 0 (Keith's decision).
- The avoid-list is English, word-level and small; it cannot judge tone or a frightening plot.
- Only ages 3-5 are built.
