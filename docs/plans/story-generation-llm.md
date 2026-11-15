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
    offline into a file in `story/configs/`, and `model.py` reads that file. Only offline code reads
    `primer.data`: `data.py`, the vocabulary build script, and `slices.py` and `evaluate.py` (which use
    `primer.data.bias`). `model.py` never does.
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
| `pages` | picture books: `storybook_pages` grouped by `(source, book_id)` — their `stories_text` rows are **not** read, so no book is counted twice. TinyStories: the `stories_text` row (`source == "tinystories"` only), sentences chunked to the age mode's sentences-per-page | derived; one pair per book |
| `characters` | not in any schema. v1 uses a capitalised-name heuristic, nullable | derived, weak |
| `setting` | not in any schema | **null in v1**. The prompt treats it as optional |

**Request side.** This extends the existing `StoryRequest` instead of replacing it:

- Kept fields: `prompt`, `age`, `target_words`, `subject`, `drawing_ref` and `description`.
- Added, all optional: `character_name` (KAN-92), `age_mode` (`learn_to_read` or `grow_vocabulary`,
  KAN-92), and `reading_level: float | None`, a Flesch-Kincaid grade. The golden set's
  `target_reading_level` and `ReadabilityScorer.grade` are both *named* as FK grades. The golden
  targets, though, are generated from age, not measured on text (see "The FK function" below).

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
| **A. Wait for the readability back-fill** (brief B4) | member 3's scorer writes `storybook_pages.reading_level`; pairs are built after | the label comes from the same model that judges `reading_level_within1`; matches the brief's recommendation | blocks S2 on S3 (the scorer) and on B4, which is more than a one-off write (review F7): **(b)** B4 has to be a pipeline stage in `src/primer/data`, because any new snapshot batch re-runs preprocess, which resets `reading_level` to NA (`storybooks.py:297`); labelling TinyStories as well needs a new `STORIES_TEXT` column and a `configs/schemas` change. **(c)** It also needs a new data version, which is a team decision (`model-development` §1.3), not this plan's. **(a)** Like every option, the Vertex job's service account needs read access to the data bucket (Blockers). There is also no mapping yet from the scorer's CLEAR `BT_easiness` target to the FK grade `ReadabilityScorer.grade` returns (question 3). Nothing to train on before the MVP date |
| **B. Compute the Flesch-Kincaid grade at pair-build time** | `story/data.py` scores each whole story with the one shared FK function (below) and records `level_source: "fk_formula@1"` | available today; deterministic; no bucket write; the same named scale as the golden set (whose targets are generated, not measured; see below); every row gets a level, TinyStories included | FK is noisy on very short texts and can go negative; it is the baseline the scorer must beat, so the LoRA learns the weaker notion of level; slight circularity if the gate's proxy is also FK |
| **C. Proxy by source or age band** | e.g. TinyStories → 3–5, picture books → 5–8 | trivial | too coarse to support "within ±1"; bakes a source bias into level conditioning |
| **D. Drop `level` from the pair** | control level only with the prompt and regeneration | no labels needed | the model never learns level conditioning: §8's "LLM ignores level" failure, by design |
| **E. LLM-labelled levels** | ask a large model to grade each story | richer than FK | costs money; unvalidated scale; a new model to evaluate |

**Chosen default: B now, A as the drop-in upgrade.**

- Build pairs with the FK grade from the one shared function (below), rounded to 0.5, and a
  `level_source` column that names the implementation version.
- When B4 lands as a pipeline stage, and the team has decided on a new data version (review F7), rebuild
  the pairs from the scorer's levels and retrain. The split does not move, because level is never
  part of the split key.
  The pair builder changes in one place: it reads `storybook_pages.reading_level`, which the schema
  stores as a **string** (`schemas.py:46`), parses it to a float grade, and sets
  `level_source: "readability_scorer"`. Nothing downstream of the pairs changes.
- Report `level_source` in every MLflow run and in `CARD.md`, so no number is read as scorer-aligned
  when it is not.
- `age_group` is derived from level: grade < 1.5 → 3–5, otherwise 5–8. That cut is a proposal and
  goes on the same decision.

*If review changes this, only S2's label column and the `age_group` cut move. Nothing else in the
plan depends on which option is used.*

### The FK function (review F5)

Nothing in `src/` computes Flesch-Kincaid today, and the repo has no `textstat` dependency. So the
plan pins **one function**, `primer.story.levels.fk_grade(text) -> float`, used by all three places
that need a level:

- the pair builder (S2);
- `check_story` (S3);
- bake-off metric 3 (S6).

How it works:

- **Formula.** `0.39 × words/sentences + 11.8 × syllables/words − 15.59`, computed on the **whole
  story**, never per page, because per-page FK on a 20-word page is mostly noise.
- **Clamped at 0**, as the golden generator clamps its targets (`scripts/gen_golden_scenarios.py:93`).
  Unclamped FK goes negative on very simple text.
- **Tokenisation.** Words and sentences use the same tokenisation as `primer.data.text.word_count`
  and `sentence_count`. Syllables use a vowel-group heuristic, written in the module, with no new
  dependency.
- **Versioning.** The implementation is versioned. `level_source` records `fk_formula@1`, and any
  change bumps it, so pairs built with different formulas cannot be mixed silently.

**"Same scale as the golden set" is true in name only.** The golden targets are not the FK of any
text. They are `max(0, age − 4 ± 0.3)` (`gen_golden_scenarios.py:93`), and 23 of the 80 book targets
are exactly 0.0. Whether FK-of-text and those targets line up is therefore an **assumption** until it
is measured. S2's manifest shows pair counts per 0.5-level bin next to the golden target bins, so
the overlap is visible before any training.

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
3. Level within ±1, using the shared `fk_grade` until the scorer exists.
4. Page-count and sentence-length compliance against the age-mode preset.
5. Regeneration rate.
6. Judge quality (1–5) on a fixed 40-request subset.
7. p50/p95 generation latency, tokens per second, and peak VRAM on one named GPU.
8. Peak VRAM of the model loaded in 4-bit for inference, and whether that alone fits in 8 GB (the
   laptop RTX 4060 AGENTS.md names). This is measured. QLoRA *training* memory cannot be measured in
   S6, because the training code arrives in S7, so it is estimated here and measured in S7.

**Selection rule**, fixed before the runs so the result cannot pick the rule:

1. **Hard filters.**
   - Licence acceptable for a course project.
   - 4-bit inference VRAM (item 8, measured), plus an estimate for LoRA adapters, optimiser state and
     activations at sequence length 1024, fits on one L4 (24 GB).
   - p95 for a whole book is ≤ 30 s on an L4, which leaves room inside the 120 s `p95_full_book_s`
     for illustration and TTS.
2. **Primary metric.** The constraint score: the mean of three rates in 0–1: target-word inclusion
   (item 2), level-within-±1 (item 3), and format compliance, which means item 4.
3. **Within 0.02 (2 percentage points) of the leader's constraint score**, prefer the higher judge
   score, then the lower latency, then the candidate that fits in 8 GB, then the smaller model
   (AGENTS.md: "Prefer a model that fine-tunes comfortably on one local GPU").

If S7's first real step shows the winner does not fit QLoRA on an L4 after all, S7 falls back to the
runner-up and records why in Decisions.

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
    (S8a), not a default.
- **What triggers regeneration in v0.** Exactly three things: invalid JSON (which constrained decoding
  should make impossible, so this is a backstop), a missing target word, and a page-count or
  sentence-length violation of the age-mode preset.
  - **Level is not a regenerate condition in v0.** It is steered by the prompt (S3 tests that the
    prompt carries the requested level) and the fine-tune, and is measured, not enforced. Rejecting
    on an FK formula that is noisy on short texts would mostly burn attempts. Revisit once the
    readability scorer is real.
  - **Safety is not checked inside the writer.** `serving/book.py:109-121` already classifies the
    story after `write` and refuses a flagged book. Checking again in the writer would duplicate
    member 3's area inside this one.
- **Vocabulary-aware prompts.** An offline build step reads `word_lists` at `v1.1` and writes one file
  per age mode into `story/configs/`.
  - **Each word keeps its `list_name`**, and each file carries the `license` and `attribution` of
    every list it draws from (Dolch, Fry, Kuperman), from the `word_lists` rows. This is the standing
    rule that every record carries both (review F9).
  - Learn to Read (3–5): Dolch pre-primer to grade 1, plus Fry first 300.
  - Grow Vocabulary (5–8): the Learn to Read words plus the rest of Fry's list. **Kuperman words
    (age of acquisition ≤ age + 1) are added only once they pass the safety area's check.**
    Kuperman is a general adult word list, so its words up to an age of acquisition of 9 have not
    been screened for a child's prompt (Keith's question 2). No safety blocklist exists yet:
    `src/primer/safety/` has only the `SafetyClassifier` protocol and its fake. Until it does, the
    Kuperman words stay out, and the build step runs every word it keeps through
    `SafetyClassifier.check` once a real one exists. The cost is a smaller Grow Vocabulary list for
    v0.
  - The prompt says "use mostly these words", without listing the whole file. Prompt length is measured
    in S3. The checker reports the out-of-list content-word rate as a `CARD.md` metric, not a gate.
  - Sentence length and page count come from the age-mode preset (KAN-92).

## Building the pairs and the splits

S2 runs these steps in this order. The order matters: dedupe before the split, gender-swap after it.

1. **One pair per book** (review F3). A picture book's `stories_text` row is the same text as its
   pages joined with `"\n\n"` (`storybooks.py:287`). Reading both datasets would count every picture
   book twice. So:
   - picture books come **only from `storybook_pages`**, grouped by `(source, book_id)`, with pages in
     `page_number` order;
   - from `stories_text`, **only `source == "tinystories"`** rows are read.
2. **TinyStories pool, sampled by hash.** `stories_text` holds about 1.8M TinyStories rows (Keith's
   count from the 2026-11-11 bias report: 1,799,236 of 1,799,990). The pool keeps the TinyStories
   whose **salted** hash fraction, from `sha256("pool:" + book_key)`, is below `pool_fraction`. The
   default is sized for the largest TinyStories ratio S8b sweeps (30:1 against about 758 picture
   books, so about 23k stories). The sample is a pure function of the key, so the pool is stable and
   cheap. Everything after this step runs on about 24k stories, not 1.8M. See question 1 under Open
   questions.
   - **The salt is required.** Step 4 splits on the *unsalted* hash of the same key (train below
     0.8). Without a salt, every pooled story (fraction below ~0.013) would land in train, leaving
     val and test with no TinyStories at all. With it, the pool is spread 80/10/10 like everything
     else.
3. **Cross-source exact dedupe** (review F4). The pipeline dedupes less than this plan first said:
   - `storybooks.py:298` drops duplicate `text_hash` only within one source's batch;
   - `storybook_pages` is never deduped;
   - TinyStories dedupes only against its own earlier batches (`tinystories.py:117-121`);
   - nothing dedupes across sources.

   S2 therefore computes `text_hash` (from `primer.data.text`, the pipeline's own function) over each
   pair's full story, and keeps the first row per hash in a fixed source order. That order is
   `african_storybook`, `storyweaver`, `tinystories`, so a duplicate keeps its picture-book
   attribution. The dropped count goes in the manifest. **Near**-duplicates (MinHash) stay in S2b.
4. **Split: a pure hash of the book key, 80/10/10** (review F1).
   - `book_key = f"{source}/{book_id}"`, with `book_id = source_id` for TinyStories. That is the
     format `split_of()` already uses (`illustration/consistency/experiment.py:83`), so a picture book
     lands in the same split for the illustration area and the story area.
   - The fraction is `int.from_bytes(sha256(book_key)[:8], "big") / 2**64`, `split_of`'s exact hash.
     It maps to train below 0.8, val below 0.9, and test otherwise.
   - **There are no strata, and level is never part of the split key.** A book's split depends only
     on its key: not on which other books exist, and not on its level. Swapping the FK level for the
     scorer's level (option B → A) therefore never moves a book between splits.
   - §3.3's "stratified by reading level and language" is met by reporting, not by assignment. The
     manifest shows the count per split for each (source × 0.5-level bin), so an imbalance is
     visible. Language adds nothing today: it is `en`-only in both `configs/schemas/stories_text.json`
     and `storybook_pages.json`.
   - `split_of` cannot be imported as it stands. It is private to another model area, and it returns
     only `val`/`test`. S2 inlines the same hash, and switches to the shared helper in
     `src/primer/data/` when **KAN-120** lands (Keith).
5. **Gender-swap balancing, after the split, within each split** (review F2). `bias.balance_by_swap`
   (`src/primer/data/bias.py:246`, by `source`) is the mitigation the bias report prescribes for
   stories (KAN-80). It runs separately on train, val and test, because a swapped copy shares its
   original's `book_key` and must stay in the original's split. Copies are marked `augmented=True`,
   and the manifest counts them.
   - **Correction:** an earlier draft said "the bias report's capped country weights apply when the
     picture books are sampled". That was wrong. `capped_weights` applies only to Quick, Draw!'s
     `country_code` (`bias.py:340`), picture books have no country column, and the report's §4 says
     origin needs no re-weighting.
6. **Source mix in training.** The training sampler draws TinyStories at a configured ratio to
   picture books (start 3:1), from the train split of the pool. The ratio is a sensitivity axis in
   S8b.

## Evaluation

| Target (§7) | Contract key | Owner | How this plan meets it |
|---|---|---|---|
| target words ≥ 95% | `target_word_inclusion` | story | check-and-regenerate makes misses rare; the fine-tune makes them cheap |
| reading level within ±1 for ≥ 85% | `reading_level_within1` | **readability** | the key is computed on our text by member 3's scorer. We report an FK-formula version in `CARD.md` and per slice. Until the scorer is real, the gate number for this key is not ours to claim |
| quality ≥ 4.0 | `story_quality_mean` | story | LLM-as-judge with a fixed rubric (age fit, coherence, uses the words naturally, gentle ending), calibrated against a human rating of 30 stories. The judge model is open question 3 |
| zero safety violations on the golden set | `safety_violations_golden` | **safety** | serving's existing check after `write` refuses a flagged story (see "What triggers regeneration"). This plan's part is to measure: S8a reports the safety-flag rate on our stories per slice |

- `evaluate.py` writes the contract shape, with `baseline` set to the **chosen model's zero-shot
  score on the golden set**, measured in S6 after the choice is made. That is model-development
  step 1, before any training. The dev set chooses; the golden set gives the baseline the fine-tune
  in S7 is compared against. Both numbers are on the same golden set, so the comparison is like for
  like.
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
- **Blockers outside this plan:**
  - GPU quota in Diya's compute project (a new project starts at zero).
  - The shared MLflow server (`model-development` §1.4 and §10).
  - **Read access for the training job** (review F7a). The Vertex job runs as
    `yip-trainer@$YIP_COMPUTE_PROJECT` (`.claude/skills/train-vertex/job_template.yaml:25`), and that
    account needs `roles/storage.objectViewer` on `gs://young-illustrators-primer-data` to read
    `v1.1`. `model-development` §6 grants read access only to the six people's accounts. Keith, as
    bucket owner, grants it.

  All three are person-only. Until the
  shared server exists, runs log to a local, git-ignored store with
  `MLFLOW_TRACKING_URI=file:./.local-data/mlruns` (`.local-data/` is already in `.gitignore`). Every
  run still carries the full tag set, so it can be **re-logged** to the shared server when that
  exists. A number only in a local store does not count for a decision or a registration.
- **Every run** (bake-off, smoke, sweep, fine-tune) logs to experiment `yip-story` with these tags:
  - `stage=story`, `owner`, `data_version=v1.1`, `git_sha`, `base_model` with the pinned revision,
    `gate_group=story`, and `where` (`local` or `vertex`);
  - plus `level_source`, and the DVC object hash for `stories_text`, `storybook_pages` and
    `word_lists`, read from `data/*.dvc` at the `data-v1.1` tag.
  - Params are the whole config. Artifacts are `metrics.json`, the pair-build manifest, the bar plot
    and `CARD.md`. The manifest holds counts per split for each (source × 0.5-level) reporting bin;
    these bins are reported only and never used to assign splits.
- **QLoRA starting point**, provisional, to be swept with Optuna in `story/configs/`: 4-bit NF4 with
  double quantisation, bf16 compute, LoRA r=16, α=32, dropout 0.05 on all attention and MLP
  projections, lr 2e-4 cosine, 2 epochs, max sequence length 1024, gradient checkpointing. Loss is
  computed on the JSON answer tokens only. `train.py` always honours `max_steps` from the config.
  The 30-minute wall clock (ADR clause 2) applies to **agent-local runs only**: it is a
  `--max-minutes` flag that the agent-local path sets and the Vertex job leaves unset, so a real
  2-epoch run is never cut at 30 minutes. `train.py` also checkpoints often enough that a Spot
  preemption costs minutes.

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

`story/data.py`: `read_dataset(version, split)` returns pairs built from `storybook_pages` (picture
books) and the TinyStories rows of `stories_text`, at a pinned version. It follows the six steps in
"Building the pairs and the splits":

- one pair per book;
- the hash-sampled TinyStories pool;
- cross-source exact dedupe;
- the pure-hash split;
- per-split gender-swap balancing;
- the level from `levels.fk_grade`, with page chunking and target-word sampling.

It writes a manifest.

- **Why separate** — pure data work, testable without a model, and every later slice consumes it.
- **Touches** — `story/data.py`, `story/pairs.py`, `story/levels.py` (the shared FK function),
  `story/configs/pairs.yaml`, `tests/story/`.
- **Acceptance**
  - **One pair per book.** A picture book present in both `storybook_pages` and `stories_text`
    yields exactly one pair. No `stories_text` row with a source other than `tinystories` is read.
  - **Pure-hash split.** A book's split is a function of `book_key = f"{source}/{book_id}"` alone.
    Adding or removing other books, or changing any book's level, never changes it. No `book_key`
    appears in two splits.
  - Split ratios are within ±2 points of 80/10/10 on a 1,000-book synthetic frame, **for the pooled
    TinyStories as well as the picture books**.
  - **Cross-source dedupe.** Two pairs from different sources with the same `text_hash` collapse to
    one, which keeps the picture-book source. The manifest counts the drop.
  - **Gender-swap copies stay in their original's split,** marked `augmented=True`.
  - `fk_grade` is ≥ 0 on every input, and pairs record `level_source = "fk_formula@1"`.
  - The manifest shows pair counts per split for each (source × 0.5-level bin), next to the golden
    target bins.
  - Every pair's target words occur in its story.
  - Every row carries `license`, `attribution` and `level_source`.
- **Testing** — each acceptance line has a test on in-memory frames built through `conform()`. Each
  test fails if its behaviour is reverted:
  - reading picture books from `stories_text` too breaks the one-pair test;
  - stratifying or ranking within a group breaks the "adding books doesn't move a book" test;
  - deduping per source breaks the cross-source test;
  - balancing before the split breaks the copy-stays test;
  - removing the clamp breaks the `fk_grade ≥ 0` test on a one-syllable-word text;
  - dropping the pool salt breaks the "pooled TinyStories reach val and test" ratio test.

  `fk_grade` also gets a table of hand-computed cases.
- **Depends on** — S1 (types). **Jira** — KAN-39.

### S2b · Near-duplicate and safety filtering of training text

*size: small*

MinHash near-dedupe across sources. It runs as **step 3b** of "Building the pairs and the splits":
after the exact dedupe (step 3) and before the split (step 4). Every story passes through the safety area's
`SafetyClassifier` (the fake in tests, the real one when member 3 lands it), and flagged stories are
dropped, with counts in the manifest.

**Scope** (review F6): this runs on the hash-sampled pool from step 2 (about 24k stories), **not**
on all ~1.8M TinyStories. That is what keeps it small.

- **Acceptance**
  - Two stories differing by one word collapse to one.
  - A story the fake classifier flags is absent from all splits.
  - Both steps run on the pool, never on the full `stories_text`.
- **Testing**
  - A near-duplicate pair on a synthetic frame (fails if MinHash is skipped).
  - A story containing the fake's marker is dropped (fails if the filter is skipped).
  - A spy classifier counts calls and asserts one per pooled story, so the test fails if the filter
    is moved before pooling.
- **Depends on** — S2. **Jira** — KAN-39.

### S3 · Prompt builder, vocabulary files, constraint checker

*size: medium*

Per-age-mode prompt templates, kept in config. An offline `build_vocabulary` step from `word_lists`
writes vocabulary files into `story/configs/`. `check_story(story, request)` returns target-word misses,
page and sentence violations, out-of-list rate, and FK grade.

- **Acceptance**
  - A prompt never exceeds a configured token budget for the 3-word golden case.
  - The checker's word matching agrees with `golden.contains_target_word` on a shared table of cases.
  - The checker's level comes from `levels.fk_grade`, the same function as S2.
  - Every vocabulary file carries `list_name` per word, plus the `license` and `attribution` of
    each list it draws from.
  - No Kuperman word is in any file until a real `SafetyClassifier` has checked it.
- **Testing**
  - Table tests, which fail if matching becomes substring matching.
  - The token budget is tested with a whitespace tokenizer stand-in.
  - The vocabulary build is tested on a synthetic `word_lists` frame:
    - missing attribution fails;
    - a Kuperman row is excluded when only the fake classifier is available;
    - a word the fake flags is dropped.
- **Depends on** — S1. **Jira** — KAN-39.

### S4 · `StoryWriterModel`: constrained decoding and regenerate loop

*size: medium*

`model.py` lazy-loads a Hugging Face causal LM from config (base plus optional adapter), decodes under
the JSON grammar, parses, checks, and regenerates up to `max_attempts`. Adds a `story` extra to
`pyproject.toml`, not installed in CI.

- **Acceptance** — with a scripted fake backend:
  - invalid JSON is never returned;
  - a story missing a word triggers regeneration;
  - a story that breaks the age-mode page count or sentence length triggers regeneration;
  - N failures raise `StoryGenerationError`;
  - importing `primer.story.model` does not import `torch`.

  Level and safety are deliberately not regenerate conditions (see "What triggers regeneration in
  v0").
- **Testing** — the five cases above, plus an import-cost test like the other areas'.
- **Depends on** — S3. **Jira** — KAN-39.

### S5 · Evaluation harness and golden adapter

*size: medium*

`evaluate.py` writes the contract `metrics.json` (story keys, `baseline`, slices). `slices.py` defines
the slice keys (age mode, gender-swapped prompts, picture-book origin) through `primer.data.bias` and
is exercised on the fake. Per-slice numbers for a real model come in S8a. Adds the LLM-as-judge
rubric. An adapter lets `golden.py` drive a real `StoryWriter`; member 5 reviews that change.

- **Acceptance**
  - `evaluate.py --model fake` writes a file in which the story keys
    (`{k for k, o in contract.METRIC_OWNERS.items() if o == "story"}`) are all present in `overall`,
    and `baseline` is present. (`contract.missing_keys` checks all eleven keys with no stage filter,
    so a story-only file can never pass it.)
  - The golden runner with `--fake-models` produces numbers identical to before the adapter.
- **Testing**
  - A test reads the written file and checks the story keys and `baseline` (fails if `evaluate.py`
    drops either).
  - A golden-runner regression test pins the fake-model metrics from `main` and compares (fails if
    the adapter changes any number).
  - `slices.py` is tested on a synthetic frame with known per-slice values.
  - The judge rubric is tested with a fake judge that returns fixed scores, so the mean is exact.
- **Depends on** — S4. **Jira** — KAN-39.

### S6 · Candidate bake-off and model choice (*person-only for the GPU run*)

*size: medium*

One config per candidate with a pinned revision. A `bakeoff` CLI runs the development request set,
logs one MLflow run per candidate, and draws the bar plot. A person runs it on one L4. The result is
recorded in Decisions with run ids, and VRAM as a dated Verified fact. Then the person runs the
**golden set** once with the zero-shot winner. That golden `story` group is the `baseline`.

- **Acceptance**
  - Three MLflow runs exist with identical prompt, decoding and data version.
  - The selection rule applied to them is written in Decisions.
  - A fourth MLflow run holds the zero-shot winner's golden-set `story` group, and its numbers are in
    Verified facts as the baseline S7 must beat. This also meets `mvp-cloud-deploy` S5's acceptance:
    golden `target_word_inclusion` and `story_quality_mean` reported, and the gate passed or the gap
    recorded.
- **Testing** — the CLI is tested with a fake backend: one run per candidate, and the rule picks the
  expected winner from a fixed table.
- **Depends on** — S5; GPU quota; MLflow server. **Jira** — KAN-39.

### S7 · QLoRA training (*person-only for the Vertex launch*)

*size: large*

`train.py` uses `peft` and `bitsandbytes` with answer-only loss. It always honours `max_steps`, and
the 30-minute cap only when `--max-minutes` is set (the agent-local path). Optuna search space in
config. `Dockerfile` moves to a CUDA base. `--smoke` runs one step on a tiny model on CPU.

- **Touches** — `story/train.py`, `story/configs/` (including `story_baseline.yaml`, still
  `data_version: unpinned`), `story/Dockerfile`, `pyproject.toml` (`story` extra), and
  `tests/models/test_area_interfaces.py::test_the_vertex_template_passes_flags_train_actually_accepts`.
  That test feeds `.claude/skills/train-vertex/job_template.yaml`'s arguments, including a `gs://`
  `--out`, to `train.main` and expects `NotImplementedError`. Once `train.py` is real, the test
  changes to parse-only through a `--dry-run` flag that validates the arguments and config and returns
  without training. It still fails if the template's flags drift.
- **Acceptance**
  - `.venv/bin/python -m pytest tests/story/test_train_smoke.py` passes locally with
    `pip install -e ".[dev,story]"`. CI does not install the `story` extra, so there the test is skipped
    by `pytest.importorskip`, as in `tests/drawing/test_siglip_encoder.py`. The PR says so and records
    the local run.
  - One Vertex run logged with every tag above.
  - The golden `story` group compared with S6's golden baseline in Verified facts.
  - The adapter replaces the base only if the group improves.
  - First real step confirms QLoRA VRAM on the L4; falls back to the runner-up if it does not fit.
- **Testing**
  - `tests/story/test_train_smoke.py` (`story` extra only): one step on a tiny randomly initialised
    model, asserting the loss is finite and the loss mask covers only the JSON answer tokens. It fails
    if the loss reverts to the whole sequence.
  - Hermetic tests that run in CI:
    - `--max-minutes` stops the loop with a fake clock (fails if the cap is ignored);
    - with the flag unset, no wall-clock cap applies (fails if the 30 minutes become unconditional);
    - `--dry-run` parses the train-vertex template and returns without training (the replacement for
      `test_the_vertex_template_passes_flags_train_actually_accepts`);
    - the MLflow tags are checked against a fake client (fails if any §2 tag is missing).
- **Depends on** — S6. **Jira** — KAN-39.

### S8a · Bias slices and sensitivity on the zero-shot model

*size: medium*

`model-development` §3 puts bias (step 3) and sensitivity (step 4) before training (step 5), so this
runs on S6's zero-shot winner:

- per-slice metrics for age mode, gender-swapped prompts and origin, including the safety-flag rate;
- ablations that drop each prompt constraint in turn (word list, level, target-word instruction);
- a temperature sweep, and logit bias on and off.

Results and mitigations go in `CARD.md`. SHAP is stated as inapplicable (`model-development` §3).

- **Acceptance**
  - One MLflow run per ablation and sweep point, with per-slice metrics logged as
    `slice/<key>=<value>/<metric>`.
  - Any slice of at least 100 rows whose `target_word_inclusion` is more than 5 points below overall
    is named in `CARD.md`, with a mitigation proposed or ruled out.
- **Testing**
  - The ablation driver is tested with a fake backend: dropping a constraint removes exactly that
    instruction from the prompt (fails if the ablation is a no-op).
  - The per-slice logger is tested with fixed numbers, including a 99-row slice that must not be
    judged (the 100-row floor from `primer.data.bias`).
- **Depends on** — S6. **Jira** — KAN-39.

### S8b · Training-dependent sensitivity

*size: small*

Re-runs S8a's slices on the fine-tuned adapter, and adds the axes that only exist with training: the
TinyStories ratio (3:1, 10:1, 30:1, all drawn from the step-2 pool) and the LoRA rank.

- **Acceptance**
  - One MLflow run per ratio and per rank.
  - `CARD.md` compares each slice before and after fine-tuning.
  - The chosen ratio is recorded in Decisions, with its run ids.
- **Testing**
  - The sampler is tested on a synthetic pool: each ratio yields the configured number of TinyStories
    per picture book, and all of them are drawn from the train split (fails if it reads val or test).
- **Depends on** — S7. **Jira** — KAN-39.

### S9 · Register, container, model card

*size: small*

`register.py` (MLflow registry plus Artifact Registry push); `CARD.md` complete. Only a
`GATE_RESULT: PASS` run takes `candidate`.

- **Acceptance**
  - `register.py --run-id <id>` refuses a run whose recorded gate result is not `PASS`, or which is
    missing any of the `model-development` §2 tags.
  - A passing run gets the `candidate` alias, and its image is tagged `<git-sha>`.
  - `CARD.md` has no `_not chosen_` / `_not pinned_` placeholders left.
- **Testing**
  - Fake MLflow and registry clients: a FAIL run is refused, a missing `data_version` tag is refused,
    and a PASS run is registered (each fails if its check is removed).
  - The real push is person-run and recorded in Verified facts.
- **Depends on** — S7 and S8b. Deploy is `mvp-cloud-deploy` S10. **Jira** — KAN-39.

## Status

| Slice | State | PR | Notes |
|---|---|---|---|
| S0 | in review | [#28](https://github.com/keithfernandes12/young-illustrators-primer/pull/28) | this plan; five chosen defaults (revisable in review) plus Keith's review fixes, all in Decisions |
| S1 | not started | | KAN-92's scope, built by Diya on KAN-39 |
| S2 | not started | | pure-hash split, one pair per book, cross-source dedupe, shared `fk_grade` (review F1–F5); helper moves to `src/primer/data` with KAN-120 |
| S2b | not started | | runs on the hash-sampled pool, not 1.8M rows |
| S3 | not started | | |
| S4 | not started | | |
| S5 | not started | | touches member 5's area |
| S6 | not started | | needs GPU quota + MLflow |
| S7 | not started | | Vertex GPU job, person-launched |
| S8a | not started | | before training, per model-development §3 |
| S8b | not started | | |
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
- **2026-11-14** — *Candidates up to 4B.* Scoping §5.3 says "Llama 3.x 1–3B / Qwen 2.5 / Phi".
  KAN-39 names Qwen3-4B and Phi-4-mini (3.8B), which are above that range. The KAN-39 list is used,
  and the selection rule's VRAM filter and 8 GB tie-break keep AGENTS.md's single-GPU preference.
  The submitted scoping doc is left as submitted.
- **2026-11-14** — *Regeneration conditions in v0.* The writer regenerates on a missing target word
  and on age-mode format violations. It does not regenerate on level, which is prompt-steered and
  measured, or on safety, which serving already checks after `write`. Chosen to avoid a noisy FK
  check burning attempts, and to avoid duplicating the safety area. Revisit when the readability
  scorer is real. (From review finding F3.)
- **2026-11-14** — *Split rule (Keith's review F1, corrects the original split text).* The split is a
  pure hash of `book_key = f"{source}/{book_id}"`, using `split_of`'s exact hash, at 80/10/10.
  - There are no strata, and level is never part of the key, so the option B → A swap can never move
    a book between splits.
  - Stratification is reported in the manifest, not used for assignment.
  - Chosen over hash-ranking within strata, which makes a book's split depend on the other books.
  - The helper moves to `src/primer/data/` under KAN-120.
- **2026-11-14** — *Pair sources and dedupe (Keith's review F3/F4).*
  - One pair per book: picture books only from `storybook_pages`, and only `tinystories` rows from
    `stories_text`.
  - Cross-source exact `text_hash` dedupe in S2, before the split.
  - Gender-swap balancing (`balance_by_swap`) after the split, within each split (F2).
- **2026-11-14** — *One FK function (Keith's review F5).* `primer.story.levels.fk_grade`: whole
  story, clamped at 0, versioned as `fk_formula@1`, and shared by S2, S3 and S6.
  - **This corrects the Level entry above:** "uses the same scale as the golden set" holds in name
    only, because the golden targets are generated from age (`max(0, age − 4 ± 0.3)`), not measured
    as FK on text.
  - S2's manifest shows level bins next to the golden target bins.
- **2026-11-14** — *Kuperman words wait for safety (Keith's question 2).* The vocabulary files hold
  only Dolch and Fry words until a real `SafetyClassifier` can screen Kuperman words.

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
- **2026-11-14** — *Note on the `book_id` join fact above:* it is still true, but S2 no longer uses
  the join (Keith's review F1/F3). Picture books are read only from `storybook_pages`, and the split
  key is `f"{source}/{book_id}"`. Also, the "Regeneration conditions in v0" decision's "review
  finding F3" refers to the self-review before Keith's review, not to Keith's F3.
- **2026-11-14** — Checked while answering Keith's review of PR #28, by reading the cited lines on
  `main`:
  - `split_of()` (`illustration/consistency/experiment.py:83`) hashes the first 8 bytes of SHA-256
    and returns only `val` or `test`, keyed on `f"{source}/{book_id}"`.
  - `capped_weights` is applied only to `drawings.country_code` (`bias.py:340`).
  - `balance_by_swap` exists (`bias.py:246`).
  - `language` is `en`-only in both schema files.
  - `stories_text` picture-book text is `"\n\n".join(pages)` (`storybooks.py:287`), deduped only
    within a batch (`storybooks.py:298`).
  - TinyStories dedupes only against its own batches (`tinystories.py:117-121`).
  - The golden targets are `max(0, age − 4 ± 0.3)` (`gen_golden_scenarios.py:93`).
  - Nothing in `src/` computes FK, and there is no `textstat` dependency.
  - `src/primer/safety/` has no blocklist yet.
  - Nothing maps the CLEAR `BT_easiness` target to an FK grade. The only link is the FK → BT
    straight-line fit in `bias.py:188`, used for analysis.
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
   Still open to Keith in review. Note that the brief (`model-stages-and-data-coverage`) recommends
   option A and names the **Language lead** as decider. If that is not Keith, the Language lead
   should see this choice in review too.
3. **Which model judges `story_quality_mean`?** It must not be one of the candidates. A hosted API
   model costs money but is stronger; a larger open model costs GPU time. Either way it is calibrated
   against 30 human ratings. *Recommended:* a hosted model on a capped sample, rubric and prompt pinned
   in config. Whichever is chosen, the judge only ever sees golden-set and development-set text,
   which is synthetic, and never a real request: `character_name` can be a child's own name (COPPA;
   run `privacy-guard` on S5). *Decides:* Diya with member 5, since the key flows through `golden.py`.
4. **Age 5 is in both modes** (§: 3–5 Learn to Read, 5–8 Grow Vocabulary). *Recommended:* the parent
   picks the mode; the default is by age, with 5 → Learn to Read. *Decides:* Diya.
5. **Where the GPU runs come from.** *Recommended:* Diya requests L4 quota in her compute project now,
   since a new project starts at zero. *Decides:* Diya.
6. ~~Serving has to handle `StoryGenerationError`.~~ **Tracked as KAN-121** (serving owner,
   `mvp-cloud-deploy` S9). After S4 the writer can raise once its attempts run out, and
   `serving/book.py:109` calls `writer.write` with no handling. (Review F11; earlier self-review F13.)
7. **Is ~2.3k TinyStories out of ~1.8M intended?** (Keith's question 1.) *Answer: yes, as a starting
   point, though the 3:1 ratio was a judgement, not a derived number.*
   - At 3:1 against about 758 picture books, roughly 2.3k TinyStories are used across all splits
     (about 1.8k in train). The target style is the picture books, and a few thousand pairs is an
     ordinary QLoRA size for teaching a format.
   - What is **not known** is whether more TinyStories would help constraint following or would
     drown out the picture-book style. S8b answers that by sweeping 3:1, 10:1 and 30:1 from the
     hash-sampled pool, which is sized for 30:1 so the sweep never re-runs S2b.
   - *Decides:* Diya, from S8b's runs.
8. **Where is the CLEAR `BT_easiness` → FK grade mapping defined?** (Keith's question 3.) *Answer:
   nowhere yet.*
   - `ReadabilityScorer.grade` is documented as returning an FK grade, but the scorer trains on CLEAR
     `score` (`BT_easiness`, higher means easier). No code maps one to the other.
   - The only link in the repo is the FK → BT straight-line fit in `bias.py:188`, which goes the
     other way and is used for analysis.
   - Option A cannot work until the readability owner defines that mapping: invert the fit, train the
     scorer on FK directly, or change the interface.
   - *Decides:* member 3 (readability). This plan's option B does not depend on it.

## Out of scope

- Serving and the Cloud Run deploy of `yip-story`: `mvp-cloud-deploy` S9/S10.
- Back-filling `reading_level` in the bucket: brief B4. Per review F7, this is **not** a one-off
  write:
  - **(b)** It has to be a pipeline stage in `src/primer/data` (the data owner's area), because a new
    snapshot batch re-runs preprocess and resets `reading_level` to NA (`storybooks.py:297`).
    Labelling TinyStories would also need a new `STORIES_TEXT` column and a `configs/schemas` change.
  - **(c)** The new data version it implies is a team decision (`model-development` §1.3).
  - It depends on the readability owner defining the `BT_easiness` → FK mapping (open question 8).
- The shared book-split helper in `src/primer/data/`: **KAN-120** (Keith). S2 inlines the same hash
  until then.
- Granting the Vertex service account read access to the data bucket: Keith, as bucket owner (review
  F7a; listed under Blockers).
- The image safety check and the safety classifier itself: member 3.
- Learner adaptation and target-word *selection*: §5.3 "word-selection logic", a later stage.
