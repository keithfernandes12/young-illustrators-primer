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
| Selected configuration | **None.** No configuration passed `selection@1` (grid of 2026-11-15 on the post-fix requests, commit `f07d2a8`, summary run `69d093b6915745ea82c2a9785faa7c97`). Best was 1.5B at temperature 0.7: a story for 20% of requests (the filter needs 80%), target-word inclusion 0.20. Best available for a demo only; it passes neither the selection rule nor the release gate |
| Selection rule | `selection@1` in `configs/story_grid.yaml`, frozen as of commit `bac7bca` (`813c6c2` before the rebase onto main). Its safety filter (`max_raw_safety_flag_rate: 0.2`) was added after a first grid had started and was stopped. Those runs' flag rates were not seen before 0.2 was chosen. Any later change bumps the version and needs a full re-run |
| Data version | `v1.1`: 30 validation requests from the **val** split, built from all `storybook_pages` plus the smallest TinyStories file (2,000 stories, pinned by generation); `test` and the golden set untouched |
| Decoding | ordinary sampling, then parse-and-retry. **Grammar-constrained decoding** (which KAN-39 asks for) is **deferred** in M1: the JSON validity rate measures how much it is needed first |
| Level | `fk_formula@1` (`levels.fk_grade`: whole story, clamped at 0, rounded to 0.5) |
| Hardware | the owner's MacBook Air (Apple M4, 16 GB), MPS, fp16 |
| Grid (hardware constraint) | **Reduced to 4 configurations on 20 requests**: both models × temperature {0.3, 0.7} × max_new_tokens {512}. The Air is fanless, so it heats and throttles over a long MPS run, and the full 2 × 3 × 2 grid on all 30 requests took ~75 min. Temperature 1.0 and max_new_tokens 256 are not measured in this submission. 512 was kept over 256 because a 4–6 page JSON story can be cut off at 256, which would score the token limit, not the model. The full grid runs where there is a GPU |
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
| target_word_inclusion (gate ≥ 0.95) | none selected; best 0.20 (1.5B, t 0.7) | 0.08 (0.5B, t 0.7) | best `7b5c2e52…`, baseline `312123fc…` |

All four configurations on the first 20 validation requests, built after PR #36's fixes: target words
come only from the Learn to Read lists (requests sha256 `23c55df0c3d2268d…`; grid of 2026-11-15 from
commit `f07d2a8`). The full table and run ids are in the plan's Verified facts.

| Configuration | success | inclusion | first-try valid JSON | page limit | sentence limit | FK mean | raw safety flags |
|---|---|---|---|---|---|---|---|
| 0.5B, t 0.3 | 0.05 | 0.04 | 0.70 | 0.00 | 0.00 | 2.5 | 0.0 |
| 0.5B, t 0.7 (baseline) | 0.10 | 0.08 | 0.60 | 0.00 | 0.00 | 2.75 | 0.0 |
| 1.5B, t 0.3 | 0.05 | 0.06 | 0.85 | 1.00 | 0.00 | 4.5 | 0.0 |
| 1.5B, t 0.7 | 0.20 | 0.20 | 0.85 | 0.25 | 0.00 | 4.6 | 0.0 |

An earlier run on the pre-fix requests (Dolch to grade 3 and Fry to 1,000, with a fallback to book
words) scored higher: best inclusion 0.36, and success 0.40 for the same 1.5B configuration. Those
results are kept in the plan's Verified facts as a record. They are not this model's result.

**What the numbers say.**
- Zero-shot 0.5B–1.5B models do not meet the 3–5 constraints.
- Most failures are missing target words. The safety guardrail never fired, and JSON is usually
  valid.
- Learn to Read target words did *not* make inclusion easier, against expectation. A hand check of 8
  requests (1.5B, t 0.7, one attempt each) found 2 invalid JSON, 1 with every word, and 5 that each
  missed one word. Of those 5 words, 1 was inflected (a requested "jump" written as "jumped"; it counts
  as a miss, because matching stays exact for ages 3–5, as decided with Keith in the plan's Decisions) and 4
  were absent. With 20 requests, the drop from 0.36 to 0.20 is about 3 requests.
- No returned story kept every sentence to 10 words or fewer. FK means are 2.5–4.6 against targets of
  0–1.
- The larger model and the higher temperature each helped a little: inclusion +0.07 and +0.09.
- These results are the case for a larger candidate, constrained decoding with word forcing, and the
  QLoRA fine-tune (plan S4/S6/S7), not for loosening the rule.

## Slices

`source` (picture-book origin or TinyStories) and `age` (3, 4, 5). With 20 requests every slice is
below the 100-row floor `primer.data.bias` uses, so slices are reported with `judged: false`; no
disparity is claimed from them.

## Course Model Development guidelines: what is done, and what is not

| Guideline | Status | Where, or why not |
|---|---|---|
| Load data from the versioned pipeline | **Done** | `data.py` reads `v1.1` at exact GCS generations: one pair per book, cross-source exact dedupe, pure-hash split. 30 requests built 2026-11-15 (manifest beside them in `.local-data/story/`); every MLflow run carries `dvc/<dataset>` tags |
| Train and select the best model | **Select: done. Train: deferred** | `grid.py` compares two pretrained models over a grid and applies a rule fixed in config before any run. No training: QLoRA on 4B candidates needs a CUDA GPU and is later work (plan Decisions 2026-11-15) |
| Validation on a hold-out | **Done** | Requests come only from the book-level **val** split; `test` is never read; the golden set is not used to choose |
| Bias detection by slicing | **Partial** | Per-slice metrics by source and age in `metrics.json` and MLflow. Gender-swapped prompt pairs (the slice `model-development` names) are deferred: they need paired requests |
| Code that checks bias and reports | **Partial** | `slices.py` reports every slice with its size and `judged`. Not Fairlearn `MetricFrame`: with all slices under the floor there is nothing to judge yet |
| Bias mitigation | **Not applicable yet** | Nothing is trained, so there is no training data to re-weight; `balance_by_swap` applies to training pairs (plan S2) |
| Hyperparameter tuning | **Partial (reduced grid)** | temperature × max_new_tokens per model, exhaustive over a 4-point grid cut down for the Mac (see "Grid" above); max_new_tokens has one value, so its sensitivity is not measured here. Optuna arrives with QLoRA |
| Experiment tracking | **Done** | One MLflow run per grid point, local git-ignored SQLite store (`.local-data/mlflow.db`, artifacts in `.local-data/mlartifacts`), with the section 2 tags plus `level_source`; re-logged to the shared server when it exists |
| Result visualisations | **Done** | `charts.py`: comparison bars and the temperature sweep (needs the `story` extra) |
| Sensitivity analysis | **Partial** | Hyperparameter spreads (`sensitivity.py`). Prompt-constraint ablation is deferred (plan S8a). SHAP/LIME do not apply to a generative stage |
| Constrained decoding (KAN-39) | **Deferred** | M1 samples freely and retries on invalid JSON. Grammar-constrained decoding (`outlines` / `xgrammar`, plan S4) comes once the grid shows how often small models break the JSON |
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
