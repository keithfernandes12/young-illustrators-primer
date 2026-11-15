---
title: Story generation LLM — candidate bake-off, QLoRA fine-tune and constrained output
type: plan
created: 2026-11-14
status: draft
author: DiyaKaswa11
jira: [KAN-39, KAN-92]
tags: [story, llm, qlora, models, evaluation]
related: [mvp-cloud-deploy, model-stages-and-data-coverage, bounded-local-training-for-agents, model-development]
---

# Story generation LLM — candidate bake-off, QLoRA fine-tune and constrained output

## Where things stand

`src/primer/story/` is the stage 2 area skeleton (`model-development` §2). `interface.py` is real:
`StoryRequest`, `Story`, the `StoryWriter` protocol and `FakeStoryWriter`, which `serving/book.py` and
`tests/models/test_area_interfaces.py` already use. Every other file raises `NotImplementedError`.
KAN-39 asks for three things: compare Qwen3-4B, Phi-4-mini and Llama 3.2 3B on constraint following,
QLoRA fine-tune the winner on the instruction dataset, and produce structured JSON with constrained
decoding and vocabulary-aware prompts. KAN-92 is the deterministic template generator that "defines the
interface the LLM generator should implement". This plan expands `mvp-cloud-deploy` S5 (story v0) and
S5b (story LoRA) into PR-sized slices; those two slices there now point here, so there is never a
second plan. Scope sources: `docs/project_scoping.md` §3.2, §3.3, §5.3 and §7. Epic:
KAN-7.

## Sources read

- Jira **KAN-39**: description, and one comment saying the first PR is this plan, including the
  `level` field. Parent epic KAN-7, "04 · Model Development". **KAN-92**: description and acceptance.
  It is unassigned and in *Idea*.
- `AGENTS.md`, `src/primer/data/AGENTS.md`, `src/primer/data/README.md`. There is no other `AGENTS.md`
  under `src/primer/`.
- `docs/engineering/engineering-loop.md` and `docs/engineering/model-development.md`, in full.
- `docs/project_scoping.md` §3.2, §3.3, §5.3 and §7.
- `docs/adrs/bounded-local-training-for-agents.md` and the ADR it supersedes.
- `docs/briefs/model-stages-and-data-coverage.md`. It already asks where picture-book levels come
  from and proposes B4, a reading-level back-fill.
- `docs/plans/mvp-cloud-deploy.md` S5, S5b and the status table. `docs/plans/_template.md` and
  `docs/plans/AGENTS.md`.
- Every file in `src/primer/story/`, plus `src/primer/serving/book.py`, `src/primer/evaluation/`
  (`contract.py`, `golden.py`, the `fakes.py` story section), `src/primer/readability/interface.py`,
  `src/primer/data/schemas.py` and the `stories_text` path in `src/primer/data/sources/storybooks.py`.
- `configs/golden/scenarios.jsonl` (counted, see Verified facts) and
  `.claude/skills/golden-gate/thresholds.json`.
- `src/primer/illustration/consistency/experiment.py`, as the pattern for MLflow logging with one
  run per candidate.

**Not read:** `docs/plans/story-demo.md`, which KAN-92 cites. It is not on `main` or on any branch
fetched on 2026-11-14. Also not read: the bucket itself (no row counts), the `train-vertex` and
`golden-gate` skill scripts beyond `thresholds.json`, and `src/primer/data/sources/tinystories.py`.

**Assumed, not checked:**
- The candidate facts below (licences, sizes) come from the model cards as I remember them. S6
  re-checks them when it pins revisions.
- `bitsandbytes` 4-bit QLoRA needs an NVIDIA GPU, so it will not run on a MacBook Air.
- The shared MLflow server and GPU quota are still outstanding, as `model-development` §1.4 said on
  2026-11-14.

## Contract

- **Target behaviour.** `StoryWriterModel` implements `StoryWriter`. Given a `StoryRequest`, it returns
  a `Story` parsed from schema-valid JSON. Every target word is present, or the writer raises after
  N regeneration attempts, and the story is aimed at the requested reading level. `evaluate.py` writes
  the contract `metrics.json` with `target_word_inclusion` and `story_quality_mean`, plus a measured
  `baseline`. The base model was picked by a logged three-way bake-off, and the QLoRA adapter replaces
  the zero-shot model only if it beats it on the `story` gate group.
- **Not doing.**
  - Serving, the API and Cloud Run deploy: `mvp-cloud-deploy` S9 and S10.
  - The reader UI.
  - The readability scorer and the safety classifier, which are member 3's areas. This plan only
    calls their interfaces.
  - Writing anything to the data bucket.
  - Learner adaptation.
  - Illustration prompts per page. They could be an optional JSON field later; the illustration area
    does not ask for one today.
- **Patterns to follow.**
  - `src/primer/readability/` is the worked example of the area layout.
  - `src/primer/drawing/configs/drawing_baseline.yaml` shows a config that pins a revision and records
    why each setting is a setting.
  - `src/primer/illustration/consistency/experiment.py` shows one MLflow run per candidate with the
    standard tags.
  - Pure functions with fakes in tests, as in `src/primer/data/`.
- **Boundaries.**
  - `story` never imports `serving`.
  - Nothing imports `primer.data` at request time. So the vocabulary lists the prompt needs are built
    offline into a file in `story/configs/`, and `model.py` reads that file. Only `data.py` and the
    build scripts read `primer.data`.
  - Weights load lazily. Importing `primer.story.*` stays cheap.
  - `src/primer/evaluation/` and `configs/golden/` belong to member 5. Any change there (S5) is
    reviewed by them.
  - The existing `StoryRequest` and `Story` fields keep working for `serving/book.py`. Changes are
    additive (see Open questions 1).
- **Test plan.**
  - Hermetic unit tests under `tests/story/` for: request validation and age-mode presets, the JSON
    schema and parser, the instruction-pair builder and splits (on small in-memory frames), the prompt
    builder, the constraint checker, and the regenerate loop.
  - The model backend is replaced by a scripted fake that returns chosen JSON strings, including
    invalid JSON and stories that drop a word. No test loads weights, touches GCS or reaches the
    network.
  - `train.py --smoke` runs one step on a tiny randomly initialised causal LM built in the test.
    Nothing is downloaded.
  - Model quality is proved by `golden-gate` and the MLflow runs, not by `pytest`.
- **Proof table** — one per slice, under each slice's *Testing*.

## The instruction pair and the JSON output

Scoping §3.2 defines the pair as `{age_group, level, target_words, pages, characters, setting} → story`.
What the data can actually supply on `v1.1`:

| Field | Source | Status |
|---|---|---|
| `age_group` | derived from `level` (see the level options) | derived |
| `level` | **nowhere**: `storybook_pages.reading_level` is null, and `stories_text` has no column | Flesch-Kincaid grade computed at pair-build time, with `level_source` (chosen default, below) |
| `target_words` | sampled from the story's own content words, preferring words in `word_lists` for the age group, 2–3 per story to match the golden set | derived, so inclusion is satisfiable by construction |
| `pages` | picture books: `storybook_pages` grouped by `book_id`. TinyStories: sentences chunked to the age mode's sentences-per-page | derived |
| `characters` | not in any schema. v1 uses a capitalised-name heuristic, nullable | derived, weak |
| `setting` | not in any schema | **null in v1**. The prompt treats it as optional |

**Request side.** This extends the existing `StoryRequest` instead of replacing it:

- Kept fields: `prompt`, `age`, `target_words`, `subject`, `drawing_ref` and `description`.
- Added, all optional: `character_name` (KAN-92), `age_mode` (`learn_to_read` or `grow_vocabulary`,
  KAN-92), and `reading_level: float | None`, a Flesch-Kincaid grade. That is the scale the golden
  set's `target_reading_level` and `ReadabilityScorer.grade` both use.

**Output.** The model is constrained to this JSON Schema, kept in `story/configs/story_output.schema.json`:

```json
{
  "type": "object",
  "required": ["title", "pages"],
  "additionalProperties": false,
  "properties": {
    "title": {"type": "string", "maxLength": 80},
    "pages": {
      "type": "array", "minItems": 3, "maxItems": 12,
      "items": {
        "type": "object", "required": ["text"], "additionalProperties": false,
        "properties": {"text": {"type": "string", "minLength": 1, "maxLength": 400}}
      }
    }
  }
}
```

The age-mode preset tightens `minItems`/`maxItems` per request. A parser turns the JSON into
`Story(pages=tuple(page.text …), title=…)`. `Story.pages` stays `tuple[str, ...]` because
`build_book` passes each page straight to `illustrate` and `speak`. KAN-92's `Page` type is therefore
the page object in the JSON plus a parsed dataclass inside `story/`, not a change to `Story.pages`.
That is the chosen default in Decisions (extend, don't replace). Each training target is the pair's story rendered in this exact JSON, so the
fine-tune learns the format as well as the content.

## The missing `level` field — chosen default, revisable in review

Option B below is the chosen default (Decisions, 2026-11-14). Keith had been asked to decide; work
went ahead without waiting, and the choice stays open to change in review of this plan.

Neither the picture-book mirrors nor TinyStories carry a reading level
(`storybooks.py:297` writes `reading_level=pd.NA`). The golden set and the readability interface both
use a Flesch-Kincaid grade (0.0–4.3 in the golden book scenarios).

| Option | How | For | Against |
|---|---|---|---|
| **A. Wait for the readability back-fill** (brief B4) | member 3's scorer writes `storybook_pages.reading_level`; pairs are built after | the label comes from the same model that judges `reading_level_within1`; matches the brief's recommendation | blocks S2 on S3 (scorer) and B4, which need a bucket write by a person and a new data version; TinyStories is still unlabelled unless B4 also covers `stories_text`; nothing to train on before the MVP date |
| **B. Compute the Flesch-Kincaid grade at pair-build time** | `story/data.py` scores each story's text with the formula and records `level_source: "fk_formula"` | available today; deterministic; no bucket write; same scale as the golden set; every row gets a level, TinyStories included | FK is noisy on very short texts and can go negative; it is the baseline the scorer must beat, so the LoRA learns the weaker notion of level; slight circularity if the gate's proxy is also FK |
| **C. Proxy by source or age band** | e.g. TinyStories → 3–5, picture books → 5–8 | trivial | too coarse to support "within ±1"; bakes a source bias into level conditioning |
| **D. Drop `level` from the pair** | control level only with the prompt and regeneration | no labels needed | the model never learns level conditioning: §8's "LLM ignores level" failure, by design |
| **E. LLM-labelled levels** | ask a large model to grade each story | richer than FK | costs money; unvalidated scale; a new model to evaluate |

**Chosen default: B now, A as the drop-in upgrade.**

- Build pairs with the FK grade, rounded to 0.5, and a `level_source` column.
- When B4 lands, rebuild the pairs from the scorer's levels under a new data version and retrain.
  The pipeline does not change, only the column the label comes from.
- Report `level_source` in every MLflow run and in `CARD.md`, so no number is read as scorer-aligned
  when it is not.
- `age_group` is derived from level: grade < 1.5 → 3–5, otherwise 5–8. That cut is a proposal and
  goes on the same decision.

*If review changes this, only S2's label column and the `age_group` cut move. Nothing else in the
plan depends on which option is used.*

## Choosing the base model

All three candidates run **zero-shot**, with the same prompt builder (S3), the same constrained
decoding (S4) and the same sampler settings, over a **development request set**. That set is the
validation split's books turned into requests (S2), plus 40 hand-written requests that span both age
modes. **The golden set is not used to choose.** It stays the release gate, and tuning against it
would make the gate measure our tuning.

| | Qwen3-4B | Phi-4-mini-instruct | Llama 3.2 3B Instruct |
|---|---|---|---|
| Licence (model card; re-check in S6) | Apache-2.0 | MIT | Llama 3.2 Community License: acceptable-use policy and "Built with Llama" attribution |
| Notes | has a thinking mode; run with it **off** for latency and format | 3.8B | smallest, so most headroom for QLoRA on 8 GB |

**Measured per candidate**, logged as one MLflow run each in `yip-story`:

1. JSON-valid rate *without* constrained decoding. This shows whether the model needs the constraint.
2. Target-word inclusion, with `contains_target_word` from `golden.py`, so the number means the same
   as the gate's.
3. Level within ±1, FK formula until the scorer exists.
4. Page-count and sentence-length compliance against the age-mode preset.
5. Regeneration rate.
6. Judge quality (1–5) on a fixed 40-request subset.
7. p50/p95 generation latency, tokens per second, and peak VRAM on one named GPU.

**Selection rule**, fixed before the runs so the result cannot pick the rule:

1. **Hard filters.** Licence acceptable for a course project. 4-bit QLoRA fits on one L4 (24 GB) at our
   sequence length, measured, not estimated. p95 for a whole book is ≤ 30 s on an L4, which leaves room
   inside the 120 s `p95_full_book_s` for illustration and TTS.
2. **Primary metric.** The constraint score: the mean of target-word inclusion, level-within-±1 and
   format compliance.
3. **Within 2 points of the leader**, prefer the higher judge score, then the lower latency, then the
   smaller model (AGENTS.md: "Prefer a model that fine-tunes comfortably on one local GPU").

The comparison bar plot and every number are MLflow artifacts. The decision goes in Decisions, citing
the run ids.

## Constrained decoding and vocabulary-aware prompts

- **Format** is enforced by grammar-constrained decoding against the JSON Schema, so an invalid
  document cannot be produced. The library is chosen in S4 between `outlines`, `lm-format-enforcer`
  and `xgrammar`. The criteria: works with `transformers` 5.x (the repo's floor), and with vLLM if
  serving later moves there.
- **Target words cannot be enforced by a grammar** without wrecking fluency. They are enforced by:
  - the prompt (listed, with a one-line instruction per word);
  - a check after decoding, using the same `contains_target_word` as the gate;
  - regeneration of the whole story, up to `max_attempts` (config, default 3), then a typed error.
    Serving decides what to show; the writer never returns a story that fails the check.
  - An optional small positive logit bias on the target words' first tokens is a sensitivity knob
    (S8), not a default.
- **Vocabulary-aware prompts.** An offline build step reads `word_lists` at `v1.1` and writes one file
  per age mode into `story/configs/`:
  - Learn to Read (3–5): Dolch pre-primer to grade 1, plus Fry first 300.
  - Grow Vocabulary (5–8): Kuperman words with age of acquisition ≤ age + 1.
  - The prompt says "use mostly these words", without listing the whole file. Prompt length is measured
    in S3. The checker reports the out-of-list content-word rate as a `CARD.md` metric, not a gate.
  - Sentence length and page count come from the age-mode preset (KAN-92).

## Splits

- **By book, 80/10/10.** The split key is `book_key = f"{source}-{source_id}"` for `stories_text`, which
  is exactly `storybook_pages.book_id` for the picture books (Verified facts). One book therefore lands
  in the same split in both datasets. TinyStories stories are each their own book.
- Assignment is by a stable hash of `book_key` within strata of (level band × language × source), as
  §3.3 asks. It is deterministic, so a rebuild on the same data version gives the same split. The
  hash, not a seeded shuffle, is what makes a book's split independent of which other books exist.
- **Dedupe before splitting.** The pipeline drops exact duplicates only (`text_hash`). S2b adds the
  near-duplicate (MinHash) pass that §3.2 promises, so a near-copy cannot sit in train and test.
- **Source mix.** TinyStories will dwarf the picture books. The training sampler caps TinyStories to
  a configured ratio (start 3:1), which is a sensitivity axis in S8. The bias report's capped country
  weights apply when the picture books are sampled (KAN-80).

## Evaluation

| Target (§7) | Contract key | Owner | How this plan meets it |
|---|---|---|---|
| target words ≥ 95% | `target_word_inclusion` | story | check-and-regenerate makes misses rare; the fine-tune makes them cheap |
| reading level within ±1 for ≥ 85% | `reading_level_within1` | **readability** | the key is computed on our text by member 3's scorer. We report an FK-formula version in `CARD.md` and per slice. Until the scorer is real, the gate number for this key is not ours to claim |
| quality ≥ 4.0 | `story_quality_mean` | story | LLM-as-judge with a fixed rubric (age fit, coherence, uses the words naturally, gentle ending), calibrated against a human rating of 30 stories. The judge model is open question 3 |
| zero safety violations on the golden set | `safety_violations_golden` | **safety** | the story's job is to never return text the safety check flags: the writer regenerates on a flag, and S8 reports the flag rate per slice |

- `evaluate.py` writes the contract shape, with `baseline` set to the chosen model's zero-shot score
  from S6. That is model-development step 1, before any training.
- `golden.py` still drives `FakeStoryWriter` with `write(prompt, words, age) -> str`, which is not
  the `StoryWriter` protocol. S5 adds an adapter so a real `StoryWriter` can run in the golden set
  (`golden.py`'s own docstring calls this open work belonging with the first real area).

## Training: where it runs, and logging

- **This machine (MacBook Air, no NVIDIA GPU)** runs data building, prompts, tests, and the
  `--smoke` step on a tiny randomly initialised model, on CPU. It cannot run the bake-off at real
  size or QLoRA, because `bitsandbytes` 4-bit needs CUDA. MLX-based LoRA on Apple silicon exists,
  but it is a different stack from the area's container, so its numbers could not be reproduced or
  registered.
- **Bake-off and QLoRA** run on **one Spot L4 on Vertex AI** in Diya's `YIP_COMPUTE_PROJECT`.
  - Launched by a person with `train-vertex`, from `story/Dockerfile` (moved to a CUDA base in S7).
  - Checkpoints and adapters go to `gs://$YIP_ARTIFACT_BUCKET/models/story/<run>/`, never to the
    data bucket.
  - A teammate's RTX 4060 (8 GB) is the free option for exploration, and Llama 3.2 3B is the
    likeliest to fit there. A registered run must come from Vertex anyway (ADR clause 4).
- **Blockers outside this plan:** GPU quota in Diya's compute project (a new project starts at zero),
  and the shared MLflow server (`model-development` §1.4 and §10). Both are person-only. Until the
  shared server exists, runs log to a local, git-ignored store with
  `MLFLOW_TRACKING_URI=file:./.local-data/mlruns` (`.local-data/` is already in `.gitignore`). Every
  run still carries the full tag set, so it can be **re-logged** to the shared server when that
  exists. A number only in a local store does not count for a decision or a registration.
- **Every run** (bake-off, smoke, sweep, fine-tune) logs to experiment `yip-story` with these tags:
  - `stage=story`, `owner`, `data_version=v1.1`, `git_sha`, `base_model` with the pinned revision,
    `gate_group=story`, and `where` (`local` or `vertex`);
  - plus `level_source`, and the DVC object hash for `stories_text`, `storybook_pages` and
    `word_lists`, read from `data/*.dvc` at the `data-v1.1` tag.
  - Params are the whole config. Artifacts are `metrics.json`, the pair-build manifest (counts per
    split and stratum), the bar plot and `CARD.md`.
- **QLoRA starting point**, provisional, to be swept with Optuna in `story/configs/`: 4-bit NF4 with
  double quantisation, bf16 compute, LoRA r=16, α=32, dropout 0.05 on all attention and MLP
  projections, lr 2e-4 cosine, 2 epochs, max sequence length 1024, gradient checkpointing. Loss is
  computed on the JSON answer tokens only. `train.py` honours `max_steps` and a 30-minute wall clock
  (ADR clause 2) and checkpoints often enough that a Spot preemption costs minutes.

## Slices

Every slice is tracked on **KAN-39**. There are no sub-tasks (Decisions, 2026-11-14), so each
slice's branch, commits and PR carry `KAN-39`.

### S0 · This plan

*size: small*

Adds `docs/plans/story-generation-llm.md` and points `mvp-cloud-deploy` S5/S5b at it.

- **Acceptance** — `python scripts/check_docs.py` passes. The chosen defaults are in Decisions.
  `mvp-cloud-deploy` S5 and S5b point here.
- **Testing** — docs check only (no code).
- **Depends on** — nothing. **Jira** — KAN-39.

### S1 · Request and output types, age-mode presets, template generator (KAN-92)

*size: medium*

Additive `StoryRequest` fields; `Story.title`; `AgeMode` presets (pages, sentences per page, words per
sentence); `story_output.schema.json` plus `parse_story()`; `TemplateStoryWriter` implementing
`StoryWriter`; `python -m primer.story` printing JSON.

- **Why separate** — it fixes the interface every later slice codes to. It is the scope KAN-92
  describes, built by Diya and tracked on KAN-39.
- **Touches** — `story/interface.py`, new `story/presets.py`, `story/schema.py`, `story/template.py`,
  `story/__main__.py`, `story/configs/story_output.schema.json`, `tests/story/`.
- **Acceptance** — KAN-92's: `pytest tests/story` passes, and the CLI prints schema-valid JSON for both
  age modes. `tests/models/test_area_interfaces.py` is unchanged and passes.
- **Testing** — every target word present (fails if the template drops one); too many words for the
  mode raises `ValueError` (fails if the check is removed); same request gives byte-identical output;
  the parser rejects extra keys and empty pages.
- **Depends on** — S0. **Jira** — KAN-39 (scope of KAN-92).

### S2 · Instruction-pair builder and book splits

*size: medium*

`story/data.py`: `read_dataset(version, split)` returns pairs built from `stories_text` and
`storybook_pages` at a pinned version. It does the book-key split, page grouping and chunking, target-
word sampling, the level column (FK grade with `level_source`, per Decisions), and writes a manifest.

- **Why separate** — pure data work, testable without a model, and every later slice consumes it.
- **Touches** — `story/data.py`, `story/pairs.py`, `story/configs/pairs.yaml`, `tests/story/`.
- **Acceptance**
  - No `book_key` appears in two splits.
  - A picture book's `stories_text` row and its `storybook_pages` rows land in the same split.
  - Split ratios are within ±2 points of 80/10/10 on a 1,000-book synthetic frame.
  - Every pair's target words occur in its story.
  - Every row carries `license`, `attribution` and `level_source`.
- **Testing** — each acceptance line has a test on in-memory frames built through `conform()`, and
  each fails if the corresponding code is removed (for example, splitting by `record_id` instead of
  `book_key` breaks the cross-dataset test).
- **Depends on** — S1 (types). **Jira** — KAN-39.

### S2b · Near-duplicate and safety filtering of training text

*size: small*

MinHash near-dedupe across sources before splitting. Passes every story through the safety area's
`SafetyClassifier` (the fake in tests, the real one when member 3 lands it) and drops flagged stories,
with counts in the manifest.

- **Acceptance** — two stories differing by one word collapse to one. A story the fake classifier flags
  is absent from all splits.
- **Depends on** — S2. **Jira** — KAN-39.

### S3 · Prompt builder, vocabulary files, constraint checker

*size: medium*

Per-age-mode prompt templates, kept in config. An offline `build_vocabulary` step from `word_lists`
writes vocabulary files into `story/configs/`. `check_story(story, request)` returns target-word misses,
page and sentence violations, out-of-list rate, and FK grade.

- **Acceptance** — a prompt never exceeds a configured token budget for the 3-word golden case. The
  checker's word matching agrees with `golden.contains_target_word` on a shared table of cases.
- **Testing** — table tests (fail if matching becomes substring); the token budget is tested with a
  whitespace tokenizer stand-in.
- **Depends on** — S1. **Jira** — KAN-39.

### S4 · `StoryWriterModel`: constrained decoding and regenerate loop

*size: medium*

`model.py` lazy-loads a Hugging Face causal LM from config (base plus optional adapter), decodes under
the JSON grammar, parses, checks, and regenerates up to `max_attempts`. Adds a `story` extra to
`pyproject.toml`, not installed in CI.

- **Acceptance** — with a scripted fake backend: invalid JSON is never returned; a story missing a
  word triggers regeneration; N failures raise `StoryGenerationError`; importing `primer.story.model`
  does not import `torch`.
- **Testing** — the four cases above, plus an import-cost test like the other areas'.
- **Depends on** — S3. **Jira** — KAN-39.

### S5 · Evaluation harness and golden adapter

*size: medium*

`evaluate.py` writes the contract `metrics.json` (story keys, `baseline`, slices). `slices.py` reports
age mode, gender-swapped prompts and picture-book origin, through `primer.data.bias`. Adds the
LLM-as-judge rubric. An adapter lets `golden.py` drive a real `StoryWriter`; member 5 reviews that
change.

- **Acceptance** — `evaluate.py --model fake` writes a file that `contract.missing_keys` accepts for the
  story keys. The golden runner with `--fake-models` produces numbers identical to before the adapter.
- **Depends on** — S4. **Jira** — KAN-39.

### S6 · Candidate bake-off and model choice (*person-only for the GPU run*)

*size: medium*

One config per candidate with a pinned revision. A `bakeoff` CLI runs the development request set,
logs one MLflow run per candidate, and draws the bar plot. A person runs it on one L4. The result is
recorded in Decisions with run ids, VRAM as a dated Verified fact, and the winner's zero-shot score
becomes the `baseline`.

- **Acceptance** — three MLflow runs exist with identical prompt, decoding and data version; the
  selection rule applied to them is written in Decisions.
- **Testing** — the CLI is tested with a fake backend: one run per candidate, and the rule picks the
  expected winner from a fixed table.
- **Depends on** — S5; GPU quota; MLflow server. **Jira** — KAN-39.

### S7 · QLoRA training (*person-only for the Vertex launch*)

*size: large*

`train.py` uses `peft` and `bitsandbytes` with answer-only loss, and honours `max_steps` and the 30-min
cap. Optuna search space in config. `Dockerfile` moves to a CUDA base. `--smoke` runs one step on a tiny
model on CPU.

- **Acceptance** — `--smoke` passes in CI-equivalent conditions. One Vertex run logged with every tag
  above. The `story` gate group compared with S6's baseline in Verified facts. The adapter replaces
  the base only if the group improves.
- **Depends on** — S6. **Jira** — KAN-39.

### S8 · Bias slices and sensitivity (ablation)

*size: medium*

Per-slice metrics for age mode, gender-swapped prompts and origin. Ablations: drop each prompt
constraint in turn (word list, level, target-word instruction), temperature sweep, TinyStories ratio,
logit bias on/off. Results and mitigations go in `CARD.md`. SHAP is stated as inapplicable
(`model-development` §3).

- **Depends on** — S7 (or S6 for the zero-shot model). **Jira** — KAN-39.

### S9 · Register, container, model card

*size: small*

`register.py` (MLflow registry plus Artifact Registry push); `CARD.md` complete. Only a
`GATE_RESULT: PASS` run takes `candidate`.

- **Depends on** — S7 and S8. Deploy is `mvp-cloud-deploy` S10. **Jira** — KAN-39.

## Status

| Slice | State | PR | Notes |
|---|---|---|---|
| S0 | in progress | | this plan; five chosen defaults in Decisions, revisable in review |
| S1 | not started | | KAN-92's scope, built by Diya on KAN-39 |
| S2 | not started | | level = FK grade with `level_source` (chosen default) |
| S2b | not started | | |
| S3 | not started | | |
| S4 | not started | | |
| S5 | not started | | touches member 5's area |
| S6 | not started | | needs GPU quota + MLflow |
| S7 | not started | | Vertex GPU job, person-launched |
| S8 | not started | | |
| S9 | not started | | |

## Decisions

The five entries below are **chosen defaults, revisable in review**. Diya went ahead with them rather
than waiting, so a reviewer who disagrees changes them here, with a new dated entry below the one it
corrects. The base model is not decided here: S6 decides it, from logged runs.

- **2026-11-14** — *Level (chosen default, revisable in review).* `level` is the Flesch-Kincaid grade
  of the story text, computed when the instruction pairs are built (`story/data.py`), rounded to 0.5,
  with a `level_source` column (`fk_formula`). When the readability back-fill (brief B4) lands, pairs
  are rebuilt from the scorer's levels under a new data version. Chosen because it is available now,
  needs no bucket write, and uses the same scale as the golden set. Keith was asked to decide and has
  not yet. See "The missing `level` field" for the other options.
- **2026-11-14** — *KAN-92 (chosen default, revisable in review).* Diya builds KAN-92's types,
  age-mode presets and template generator herself, as S1 of this plan, tracked on KAN-39.
- **2026-11-14** — *Interface (chosen default, revisable in review).* The existing `StoryRequest` and
  `Story` are extended, not replaced. KAN-92's fields are added as optional fields, `Story.pages` stays
  `tuple[str, ...]`, and the JSON `Page` object lives inside `story/`. Chosen so `serving/book.py`
  and `tests/models/test_area_interfaces.py` keep working unchanged.
- **2026-11-14** — *Local MLflow (chosen default, revisable in review).* Until the shared MLflow
  server exists, runs log to `MLFLOW_TRACKING_URI=file:./.local-data/mlruns`, which is git-ignored
  (`.local-data/`), with the full tag set so they can be re-logged to the shared server later.
- **2026-11-14** — *Jira (chosen default, revisable in review).* No sub-tasks. Every slice is
  tracked on KAN-39.

## Verified facts

- **2026-11-14** — `src/primer/story/` exists with the full area layout. Only `interface.py` is
  implemented. `serving/book.py` and `tests/models/test_area_interfaces.py` import `StoryRequest`,
  `Story`, `StoryWriter` and `FakeStoryWriter`. Checked by reading the files and grepping `src` and
  `tests`.
- **2026-11-14** — `stories_text` and `storybook_pages` carry no usable level. `STORIES_TEXT` has no
  level column, and `storybooks.py:297` writes `storybook_pages.reading_level` as `pd.NA`. Checked in
  `schemas.py` and `storybooks.py`.
- **2026-11-14** — A picture book's `storybook_pages.book_id` is `f"{source}-{story_id}"`
  (`storybooks.py:264`), and its `stories_text` row has `source_id = story_id` (`storybooks.py:290`).
  So `f"{source}-{source_id}"` joins the two.
- **2026-11-14** — The golden set has 100 scenarios: 80 `book`, 20 `safety`. Book scenarios are
  spread over ages 3–8 (13–14 each), have 2 or 3 target words (40 each), and a `target_reading_level`
  from 0.0 to 4.3, with 23 of the 80 at 0.0. Counted with a Python script over
  `configs/golden/scenarios.jsonl`.
- **2026-11-14** — The data pipeline has no MinHash dedupe and no toxicity filter. Only `text_hash`
  exists. Checked by grepping `src/primer/data`.
- **2026-11-14** — `docs/plans/story-demo.md`, cited by KAN-92, does not exist on `main` or on any
  fetched branch. Checked with `git log --all -- docs/plans/story-demo.md`.

## Open questions

1. ~~KAN-92's types vs the merged interface.~~ Chosen default 2026-11-14: extend, don't replace
   (Decisions).
2. ~~The `level` field.~~ Chosen default 2026-11-14: FK grade now, back-fill later (Decisions).
   Still open to Keith in review.
3. **Which model judges `story_quality_mean`?** It must not be one of the candidates. A hosted API
   model costs money but is stronger; a larger open model costs GPU time. Either way it is calibrated
   against 30 human ratings. *Recommended:* a hosted model on a capped sample, rubric and prompt pinned
   in config. *Decides:* Diya with member 5, since the key flows through `golden.py`.
4. **Age 5 is in both modes** (§: 3–5 Learn to Read, 5–8 Grow Vocabulary). *Recommended:* the parent
   picks the mode; the default is by age, with 5 → Learn to Read. *Decides:* Diya.
5. **Where the GPU runs come from.** *Recommended:* Diya requests L4 quota in her compute project now,
   since a new project starts at zero. *Decides:* Diya.

## Out of scope

- Serving and the Cloud Run deploy of `yip-story`: `mvp-cloud-deploy` S9/S10.
- Back-filling `reading_level` in the bucket: brief B4, member 3, person-run.
- The image safety check and the safety classifier itself: member 3.
- Learner adaptation and target-word *selection*: §5.3 "word-selection logic", a later stage.
