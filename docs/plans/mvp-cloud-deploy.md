---
title: MVP on Google Cloud — models v0, serving, deploy, gates and monitoring
type: plan
created: 2026-11-07
updated: 2026-11-15
status: active
author: rishiparekh
tags: [mvp, gcp, serving, models, deploy, loops]
related: [ai-coding-framework, ci-hardening, autonomous-loops, story-generation-llm]
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
- **Not doing** — IP-Adapter and the style LoRA (stretch goals, §5.4). Whisper read-aloud was on
  this list and is **not any more**: the speech area landed on `main` while this plan was open
  (KAN-95/96/97/100 — `SpeechRecognizer` and page alignment, Whisper and wav2vec2 recognisers, an
  evaluation harness and model selection). This plan carries no slice for it, which is a statement
  about this plan and not about the project. The three
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

> **Planned in detail in `docs/plans/story-generation-llm.md`** (KAN-39), slices S1–S6 there. Update
> progress in that plan's status table; this entry stays as the MVP-level summary. Where the two
> differ, that plan is current:
> - **Candidates** are Qwen3-4B, Phi-4-mini and Llama 3.2 3B, so up to 4B rather than "1–3B".
> - **Dependencies:** it uses the Flesch-Kincaid formula and the safety fake until S3 and S4 here
>   land, so it does not wait on them.
> - **Regeneration** is on target words and the age-mode format only. Serving's existing safety
>   check refuses a flagged story, and level is steered by the prompt, not regenerated on.
>
> The text below is kept as originally written, except the Depends line, which now marks S3 and S4
> as non-blocking (review F8 on PR #28).

`src/primer/story/`: an instruct model (Llama 3.x 1–3B, Qwen 2.5 or Phi) with a prompt per age mode,
then check, and regenerate on failure, for target-word inclusion, readability (S3) and safety (S4).

- **Acceptance** — golden `target_word_inclusion` and `story_quality_mean` reported; gate group `story`
  passes or the gap is recorded under Verified facts.
- **Depends on** — S3, S4, as **non-blocking**. The detailed plan uses the Flesch-Kincaid formula
  and the safety fake until they land. **Jira** — KAN-39.

### S5b · Story LoRA fine-tune

*size: medium*

> **Planned in detail in `docs/plans/story-generation-llm.md`** (KAN-39), slices S7–S9 there. Where
> the two differ, that plan is current: the base is chosen from Qwen3-4B, Phi-4-mini and Llama 3.2 3B
> (up to 4B, not "1–3B"), and the method is QLoRA. The text below is kept as originally written.

LoRA on the chosen 1–3B base with the instruction dataset (TinyStories plus the picture-book corpora),
trained on Vertex AI with `train-vertex`; replaces the S5 base model only if the `story` gate group improves.

- **Acceptance** — MLflow run with data version; golden `story` metrics compared with S5 in Verified facts.
- **Depends on** — S1, S5. **Jira** — KAN-39.

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
inference path that normalises a decoded drawing into the form the recogniser sees, and the
training-time augmentation that makes a rendered Quick, Draw! canvas look like a real upload.
Settings in config, not in call sites.

**Arrays in, arrays out. This module does not decode.** Decoding an upload means JPEG, which means
Pillow, which lives in the `drawing` extra that CI does not install (`pyproject.toml`,
`.github/workflows/ci.yml`). Keeping decode outside is what lets the module be `numpy`-only and
therefore covered by CI — the same bargain S6b struck, and worth being exact about why: `render.py`
is library-free because it only ever *encodes* PNG, and its one test that needs a real decoder is
skipped wherever Pillow is absent. Decoding is the thing the renderer declined to do, not an
example of what it did.

Numbered between S6b and S6b2 because that is where it belongs in build order: S6b2 and S6c2 both
need it and neither can start without it.

- **Why separate** — it is KAN-87's argument a second time. If training degrades images one way and
  inference normalises them another, a measured number is a comparison between a model and two
  preprocessing paths, and nobody can tell which moved. One module, both directions, so they cannot
  drift. It is also the only thing standing between S6b2 and S6c2 and the "ready" column: KAN-31
  gates both and has had no slice, which by this plan's own rule — `loop-worker` builds slices, not
  tickets — meant neither could ever be scheduled.
- **Inference direction** — from a decoded array: optionally convert to greyscale, optionally crop
  and binarise to the renderer's black-on-white convention, and land on `drawing_render.yaml`'s
  `size`. Only `size` carries over from that file; `padding` and `source_grid` are stroke-space
  concepts with no meaning for a photograph, and `size: 224` is marked provisional there, so this
  output geometry moves if S6a's checkpoint does.
- **What already decodes, and what happens to it** — `SiglipEncoder.encode` opens the bytes with
  Pillow and composites them through `model.py`'s `flattened()` onto `alpha_background` before the
  processor sees them. That is already an inference preprocessing step, and **`flattened` stays
  where it is.** It is a Pillow function by necessity — it reads `.mode` and `.info`, calls
  `Image.new` and `alpha_composite`, and runs *before* an array exists — so moving it into
  `preprocess.py` would contradict this slice's own acceptance that the module imports no image
  library, and would put Pillow back in CI's path. Decode and composite are one step, both stay at
  the call site, and `preprocess.py` takes the composited RGB array. The single-entry guard below
  therefore covers **array-level** preprocessing only, and its test has to say so explicitly or it
  fails on day one against a path that legitimately exists.
- **Training direction** — paper texture, line wobble, partial erasure, colour fill and lighting
  over a rendered canvas, applied at training time only and seeded so a run reproduces. Its output
  has to stay in the domain the inference direction accepts — same dtype, same geometry, same
  value range — which is the constraint that keeps the two from drifting. That is a constraint on
  the shape, not on colour: augmentation may add colour the greyscale path then removes.
- **Not in this slice** — KAN-31 also covers Amateur Drawings derived sets: character crops, masks,
  normalised keypoints, OpenPose renders and auto-captions. Those are detection and segmentation
  rather than preprocessing, and **S6e** already owns that work. Say so when closing KAN-31 so the
  ticket is not read as done.
- **Acceptance** — a test asserts a single entry point per direction, as KAN-87's single-renderer
  guard does, so a second preprocessing path cannot appear later under another name. Augmentation is
  deterministic given a seed. The module imports no image library, so CI covers all of it without
  the `drawing` extra.
- **Where the risk is** — the augmentation is synthetic and nothing validates that it resembles a
  real upload. S6c2 is what would say, and S6c2 exercises only the inference direction, so build
  that first *within this slice* and treat the augmentation as unproven until there are numbers.
  Both directions ship together: `loop-worker` merges whole slices, so this is build order inside
  one PR, not a dependency anything else can key off.
- **Depends on** — S6b for the canvas convention. **Jira** — KAN-31.

### S6b3 · Amateur Drawings strata and samplers

*size: small*

`src/primer/data/strata.py`: strata over Amateur Drawings, plus `sample_uniform` and
`sample_proportional` over them,
beside `bias.py`, whose `stratify_key` and `split_strata` are the patterns to follow. Nothing here
embeds, clusters, captions or loads a model. It sits with the other `S6b*` slices because all of
them are about what reaches a model rather than about a model; it is before S6b2 because both
S6b2's consumer and S6c2's want a sampler, and neither wants to wait on the rendered split.

- **Why separate** — `quickdraw` picks rows with `max_per_category: 2000`, "first N drawings". For
  Quick, Draw! the damage is bounded to within-category order, which nobody has checked either;
  Amateur Drawings is one undifferentiated pile of 174,981 processed rows, so "first N" inherits Meta's
  upload order and whatever it correlates with. Both consumers need an answer — KAN-111's
  `humanoid` class if it is accepted, and S6c2's stability sample either way.
- **Four binned inputs, and the count is the design** — the **original long edge**
  (`max(original_width, original_height)`, the size *before* Meta's 512 px resize); the **original
  aspect ratio**; the **framing ratio**, `bbox_width × bbox_height` against `width × height`
  (derived — there is no `bbox_area` column); and segmentation **`area`** for figure complexity.
  Size and aspect are both needed and neither substitutes: a 3024×4032 phone photo and a 756×1008
  digital canvas share an aspect ratio and differ in long edge. Four bins each would allow 256 cells;
  **measured over the real corpus it is 126**, because `aspect` supports only two bins — see the
  Verified fact below. Pooling turns out to touch 0.7% of rows, not the routine path this slice
  first assumed, which is itself a reason its parameters stay a recorded Decision rather than
  a number fixed here.
- **The binning and pooling parameters are the implementer's, recorded as a Decision** — bin count
  and whether bins are quantile or fixed; `min_size`; what happens when `pd.qcut` finds duplicate
  edges on a concentrated column; the pooling candidate set, the tie-break, and the fallback when
  nothing is big enough. `split_strata` (`bias.py`) answers every one of those for its own case and
  is the shape to follow — a candidate set restricted to strata already at `min_size`, an explicit
  tie-break, a terminal "pool everything" fallback. **This plan deliberately does not pin the
  values**: nobody has looked at the distributions, and a number chosen here before anyone has
  would be a guess with a plan's authority behind it. What the slice requires is that the choices
  are recorded with their reasons, and that **the realised bin edges ship with the strata** so a
  later draw reproduces the same bins rather than re-deriving them from a different corpus version.
  The stratum key is the concatenation of bin labels, the way `stratify_key` concatenates.
- **A one-valued column raises, and that is a behavioural rule rather than a statistical one** —
  `n_keypoints` is the live example: `schemas.py` records it as "17 for every Amateur Drawings row
  today", so it cannot stratify anything. The failure is otherwise silent, and silent in a way a
  distribution test cannot catch: with a single stratum the uniform and the proportional draw
  return the *same* distribution, and each returns the distribution it claims, so such a test
  passes. So the rule is stated as behaviour — **building strata from a column with one distinct
  value raises, naming the column** — and the test asserts the raise.
- **Two samplers, not one** — uniform-per-stratum for **training** (coverage, so rare modes are
  learned rather than averaged away) and proportional for **evaluation** (fidelity, because an eval
  set's job is to predict production). Conflating them biases a measurement silently, which is why
  the split is in the acceptance rather than left to the caller. **Both take `seed: int = 0` and
  pass it as `random_state`**, as every sampling call site in `bias.py` does, so a draw is
  reproducible — S6c2 freezes 1,000 ids and S6d re-measures the same statistic on the same
  drawings, which a redrawn sample would make incomparable. Keyed on **strata**, so they work
  whether or not S6b4 ever runs. If S6b4 adopts a cluster id it joins as a **categorical**
  bin label rather than a binned one — the strata builder needs that path, since
  quartile-binning a cluster id is meaningless — and the samplers then take it unchanged.
- **What the strata do *not* establish** — that a size signature really separates photographs from
  scans from digital canvases. That is an assumption, so the slice reports **stratum occupancy** —
  how many drawings fall in each bin — as a dated Verified fact, and makes no claim about capture
  mode. That occupancy table is the only evidence anyone has for the assumption, which is why it is
  in the acceptance rather than left as prose. S6b4 is what would validate the assumption itself.
- **Acceptance** — six things, each of which can be shown false:
  1. The strata exist, with their realised bin edges recorded alongside.
  2. Building strata from a column with one distinct value raises, naming the column.
  3. Both samplers exist, take a seed, and the same seed returns the same rows twice.
  4. From a deliberately skewed fixture at a fixed seed, the proportional draw reproduces the skew
     and the uniform draw flattens it, each within a stated tolerance.
  5. The stratum-occupancy table is recorded as a dated Verified fact.
  6. Every derived record carries `license` and `attribution`, which the review checklist
     makes a standing rule and which matters here because the corpus is CC-BY-NC-4.0.
- **Testing** — clauses 1–4 and 6 above are tests; clause 5 is a markdown edit and is not
  one. Plus a fixture whose stratum columns are absent entirely,
  which must raise rather than silently return the whole corpus. Every test is seeded; an unseeded
  draw with no tolerance is either flaky or loose enough to pass either way.
- **Not a schema change** — the samplers return row subsets of `CHARACTER_DRAWINGS`, which already
  carries `license` and `attribution` (`schemas.py`). The strata and their edges are a **sidecar**
  keyed by `record_id`, carrying both fields like any derived record; nothing is added to the
  frozen schema, so no re-run and no new data version.
- **CI covers all of it** — pandas over columns that already exist, like `stratify_key`. `pandas`
  is a core dependency and CI installs `.[dev]`; no `torch`, no extra.
- **Depends on** — KAN-24's ingest. Not S6a, S6b or S6b1: no model, no rendering.
  **Jira** — KAN-117.

### S6b4 · Spike: does visual clustering beat the strata?

*size: medium · spike · **person-only***

**A spike, not a slice**, and the distinction is load-bearing: its deliverable is a recorded
verdict, not a feature, and "we tried and it did not help" closes it. Nothing downstream waits on
it — S6b3 ships the samplers, and S6b4 can only ever add a stratum.

- **The question** — do image embeddings find structure in Amateur Drawings that S6b3's size and
  framing signatures do not? If not, S6b3 is the answer and this closes.
- **Method** — embed and cluster with two encoders (SigLIP 2's tower, pinned and measured in S6a
  but circular since it is the model being trained; and DINOv2, an independent view), then
  **validate the clusters with captions**: caption centroid-nearest and boundary members
  separately, and report caption coherence and TF-IDF distinctive terms per cluster. Structure
  found in image space, checked in an independent modality.
- **Caption quality first, or not at all** — `docs/briefs/drawing-recognition-model-candidates.md`
  asks for 2–3 captioners over ~50 drawings and nobody has run it. **If captions come back
  uniformly generic the validation is vacuous and the spike stops there**, with that verdict as its
  result — which is what S6e needs before it commits to a captioner.
- **How "beat" is decided** — the comparison has to be like for like: score the stratum partition
  and each cluster partition with the **same** caption measures over **comparable samples**, which
  roughly doubles the caption budget. Record the method as a Decision *before* running it. **A
  comparison whose metric is chosen after seeing the numbers is not a comparison.**
- **The judge must not be SigLIP's text tower.** Its towers are trained aligned, so SigLIP's image
  clusters score higher in SigLIP's text space than DINOv2's do **by construction**. Use a text
  encoder independent of both candidates, or decide on the distinctive terms, which are neutral.
- **Why person-only** — the full sweep is 178,000 images × 2 encoders ≈ **12 hours** on this
  machine's CPU wheel (0.12 s is S6a's measured warm `analyse` for SigLIP, **assumed** for DINOv2,
  and counts forward passes only — not PNG decode, not reading ~48 GB), and captioning is
  autoregressive, so its per-image cost is well above a single forward pass and has **not been
  measured at all**. `bounded-local-training-for-agents.md` governs training and does not reach
  this, so it is precedent rather than rule. **An agent must not start the embedding sweep
  or the captioning run.** A subsample
  makes the sweep cheaper — 20,000 rows is ~80 minutes across both encoders and an agent-sized
  draw is nearer 7,000 (~28 minutes) — but it cannot produce an adoptable cluster id, because
  ids covering 7,000 of 178,000 rows stratify nothing. The subsample path can only ever reach
  "did not beat the strata".
- **Where it misleads** — a cluster can cohere on something irrelevant, brightness say, and score
  well, so a person reads the terms and recognises a wrong axis. And this is **not** KAN-111's
  shortcut test, which is cross-corpus; this clusters *within* Amateur Drawings and never touches a
  render.
- **If it is adopted** — the cluster id is a **sidecar** keyed by `record_id`, not a column:
  `CHARACTER_DRAWINGS` is frozen with its DVC pointer at a `gs://` path, so a new column would mean
  a re-run and a new data version. It carries `license` and `attribution` like any derived record,
  and S6b3's samplers take it as one more categorical bin label.
- **Privacy** — children's drawings, and AGENTS.md's rule is "No PII in logs, fixtures or commits".
  Nothing this spike produces goes in the repository: any images pulled for inspection stay on
  local scratch. **Captions need the same care** — children write their names on drawings and a
  captioner with OCR ability may transcribe them, so scrub before recording anything. TF-IDF
  terms are shareable in a Verified fact where the images are not.
- **Needs what nothing supplies** — DINOv2 and a captioner are external models that no slice
  delivers, and neither is in `pyproject.toml`. Part of why this is person-only.
- **How it stops** — a spike with no stop rule is how a spike becomes open-ended. **Two working days of a
  person's time** — that is the termination condition. **One full embedding sweep per encoder**
  is a cap on re-work rather than a stopping point: stopping when the sweep finishes would stop
  before any clustering, captioning or comparison had happened, which is the whole deliverable.
  `k` is swept over at most three values, chosen and recorded beforehand — not because of the
  twelve hours, since embeddings are computed once and cached and re-clustering at another `k`
  costs minutes, but because **captioning scales with cluster count** at ~50 captions per
  cluster, and captioning is the cost nobody has measured. If the budget is reached
  with no verdict, the Verified fact records **inconclusive** and the spike closes — which is a
  result, and leaves S6b3's strata standing as the answer.
- **Done when** — a dated Verified fact records the verdict, whichever it is: captions too generic
  to validate anything; clusters computed and beaten by the strata; clusters adopted, with a
  Decision naming the encoder, the method, `k`, and the comparison agreed beforehand; or
  inconclusive against the budget above.
- **Depends on** — S6b3 for the strata it is measured against, S6a for the SigLIP tower, KAN-24's
  ingest, and the two external models above. **Jira** — KAN-117.

### S6b2 · Rendered QuickDraw split for the recogniser

*size: medium*

`src/primer/drawing/data.py`: train/val/test over the frozen QuickDraw version through S6b's
renderer, with S6b1's paper/crayon augmentation applied at training time only.

- **Why separate** — it is the first slice that needs frozen data on disk, which S6b does not.
- **Acceptance** — `read_dataset(version, split)` returns each of a pinned version's three splits
  (one per call; `build_splits` makes all three from a frame already read), no record appears in
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
  metric is scored over `FakeDrawingClassifier`, which is handed the answer. This slice is what
  makes it mean anything.) KAN-64 delivers a child's own drawing, uploaded; S6c2 measures how far
  the preprocessing moves the answer, and until it has, "clears 0.85" means the gate passed, not
  that the stage works.
- **The gate ceiling is no longer capped.** KAN-116 is fixed and merged (#26): the golden set's
  unreachable labels were replaced, and counted on `main` its sixteen distinct `drawing_label`
  values are all in the 345, so a flawless recogniser reaches 1.0. Re-count before recording a
  baseline rather than assuming it, because the ceiling is a property of whatever
  `configs/golden/scenarios.jsonl` holds that day, not a constant.
- **Depends on** — S6a, S6b2. **Jira** — KAN-38.

### S6c2 · Preprocessing stability on the Amateur Drawings corpus

*size: medium*

`src/primer/drawing/evaluate.py`: measure how far the preprocessing path moves the recogniser's
answer on real children's drawings, using the Amateur Drawings
corpus. **No labels, and none are needed** — what is measured is how
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
  `person` too per KAN-110, so it would answer the corpus problem and not this one. Do not reach for a
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
- **Sample: 1,000 drawings, drawn once with a recorded seed and frozen by id.** Amateur Drawings
  holds ~178,000; scoring all of them under four variants is ~712,000 forward passes, which at
  S6a's measured 0.12 s warm on CPU is about 24 hours. That is longer than any unattended agent
  run should take on the project's only machine — `bounded-local-training-for-agents.md` sets a
  30-minute wall clock but governs *training*, so it is the precedent
  here rather than the rule. 1,000 is roughly 8 minutes
  and resolves a change rate to about ±3 points at 95%, finer than any effect worth acting on here.
  Freezing the ids matters as much as the size: S6d re-measures this statistic on the same
  drawings, and a redrawn sample makes the two incomparable.
  **Draw it with S6b3's *proportional* sampler, not the uniform-per-stratum one.** This is an
  evaluation sample, so it has to match the corpus it is standing in for; the uniform-per-stratum
  sampler is for training and would over-represent rare modes here, reporting a stability figure
  the product never sees. If S6b3 has not landed, a documented random draw is acceptable, and the
  Verified fact has to say the sample is unstratified so the number is read accordingly.
- **Decide while building** — which rank statistic "how far the top-5 ranking moves" is: Kendall's
  tau, top-5 set overlap, or mean rank displacement. Record the choice and the reason as a Decision.
- **Acceptance** — the pairwise stability figures, the margin figures, the sample size and its data
  version are reported from one MLflow run — naming **which sampler drew it**, which is the only
  thing that shows the S6b3 edge was honoured rather than quietly falling back to a random
  draw — or in Verified facts with a note that no host exists
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
- **Depends on** — S6a and **S6b1**; **S6b3 softly**, for the proportional sampler — a documented
  random draw is an acceptable fallback, so this does not park behind it. Not S6c: both slices
  write `evaluate.py`, but that is merge
  ordering rather than a dependency — the file already imports `primer.evaluation.contract` and
  derives `OWNED_METRICS` from it, so there is no contract shape for S6c to establish first. Adding
  the edge would also put this slice behind S6b2 and the whole QuickDraw split, which it never
  reads. Whoever takes both should land S6c first to keep the diffs apart. **Jira** — KAN-109.

| Variant | Differs from the row above by | Crop |
|---|---|---|
| As published by the source | — (the nearest thing we have to the product's path) | none |
| Converted to greyscale | colour | none |
| Binarised to black on white | photography, lighting, paper texture | none |
| Cropped to the annotated box, then binarised | framing | the dataset's `bbox_*` |

**Each row differs from the one above it by exactly one *operation*.** An earlier draft
folded cropping and binarisation into a single row, which made its
figure a combined effect attributable to neither — report pairs of adjacent rows, or name the
combination explicitly when reporting a non-adjacent pair.

The crop is Amateur Drawings' own user-confirmed `bbox_x/y/width/height` (`CHARACTER_DRAWINGS`),
which the product does not have at upload time — nothing in the inference path crops, and
`crop_ref`/`mask_ref` are S6e. That variant is therefore an **upper bound the product cannot reach
today**, not the real inference path, and its number has to be quoted that way.

The first row needs the same care. These are not uploads as the product would receive them: the
source docstring says Meta resized the published PNGs to 512 px on the long side, keeping the real
upload size only as `original_width`/`original_height` (`schemas.py`). A parent's phone sends a
full-resolution capture. So that row carries two confounds, not one — an unknown mix of
photographs, scans and digital drawings, and a resize nobody in this pipeline performed.

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
  **A gain on rendered Quick, Draw! together with a drop in stability is the outcome this stage
  most needs to catch** — it is what fine-tuning on clean renders is most likely to do — and
  recording only the gated number makes it invisible.
  **Read a stability *gain* with equal suspicion.** Stability is agreement between preprocessing
  variants, and a model that has collapsed onto one confident answer whatever it is shown is
  perfectly stable. Fine-tuning hard on clean renders, over a vocabulary that cannot name this
  corpus's subject, is a plausible route to exactly that, and it would present as an improvement.
  S6c2's confidence and margin figures are what tell the two apart: **stability rising while top-1
  confidence rises and margins widen, on a corpus the vocabulary cannot name, is evidence of
  collapse rather than robustness.** Report all three together or none of them. This failure has
  already happened once in this area — dropping one normalisation line in S6a made every drawing
  return `The Eiffel Tower`, with confidence.
  If S6c2 has not run yet, this slice may still proceed, but its Verified fact has to say the
  result is unvalidated against anything but rendered strokes rather than leaving a reader to assume otherwise.
  Stability is not accuracy either way: it can show the model became more brittle, or more rigid,
  never that it became wrong.
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
  real recogniser rather than the Fake. (The real-recogniser version is
  `test_every_character_sheet_field_reaches_the_illustrator_from_the_real_recogniser` in
  `tests/drawing/test_sheet.py`; the unsuffixed name is the Fake's, in
  `tests/models/test_area_interfaces.py`.)
- **Not in this slice** — `reference_image_ref` and `approved`, which the stylise-and-approve step
  sets; who owns that step is still open between KAN-38 and KAN-42.
- **Depends on** — S6a. **Jira** — KAN-38.

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
book even though `isinstance(model, Illustrator)` passes. Stage 6's `ConsistencyScorer` exists.
KAN-105 tracks wiring it into `IllustratorModel.consistency()`.

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
| S2 | merged | [#11](https://github.com/keithfernandes12/young-illustrators-primer/pull/11) | golden set + runner + gate done on fakes; real-model evaluation still open. Unblocks S3–S6, S11 |
| S3 | not started | | |
| S4 | not started | | |
| S5 | not started | | detailed in `story-generation-llm` (S1–S6). KAN-39 |
| S5b | not started | | Vertex GPU job; detailed in `story-generation-llm` (S7–S9). KAN-39 |
| S6a | merged | [#18](https://github.com/keithfernandes12/young-illustrators-primer/pull/18) | zero-shot SigLIP 2 baseline; its `drawing_top5_acc` is measured in S6c |
| S6b | merged | [#19](https://github.com/keithfernandes12/young-illustrators-primer/pull/19) | `render_strokes` + config + PNG; numpy only, so CI covers it. KAN-87 |
| S6b1 | merged | [#31](https://github.com/keithfernandes12/young-illustrators-primer/pull/31) | `prepare_drawing` + `augment_canvas`, numpy only, wired into the encoder; binarising is local after two global thresholds failed on real drawings. Unblocks S6b2 and S6c2. KAN-31 |
| S6b3 | merged | [#29](https://github.com/keithfernandes12/young-illustrators-primer/pull/29) | strata and two samplers, measured over the real 174,981 rows. KAN-117 |
| S6b4 | not started | | **spike, person-only**: waits on S6b3, S6a and KAN-24, plus DINOv2 and a captioner that no slice supplies. Can only add a stratum. KAN-117 |
| S6b2 | merged | [#32](https://github.com/keithfernandes12/young-illustrators-primer/pull/32) | `data.py`: hashed 80/10/10 split of v1.1 (506k/63k/63k), rendered on demand through `render_strokes` → `augment_canvas` (train only) → `prepare_drawing`. Unblocks S6c |
| S6c | merged | [#37](https://github.com/keithfernandes12/young-illustrators-primer/pull/37) | baseline `drawing_top5_acc` **0.5436** over all 63,310 rendered test drawings (GPU, bf16) under the prompt `a child's drawing of a {label}.` (switched 2026-11-16; was 0.4956 under the model card's prompt): gate FAIL against 0.85, chance 0.0145. MLflow run `b72bc442`, superseding `c2b68187`, in `yip-drawing`. The gate's number, not the product's. KAN-38 |
| S6c2 | merged | [#33](https://github.com/keithfernandes12/young-illustrators-primer/pull/33) | 1,000 frozen drawings (unstratified), four variants: top-1 moves 25% / 50% / 41% between adjacent variants; over half the drawings come out `squiggle`. Reproduces exactly. KAN-109 |
| S6d | not started | | first slice that trains; builds its own `--max-steps` cap. No longer blocked on hardware (CUDA torch, RTX 4060 8 GB, 2026-11-15); S6c merged (#37); must beat 0.5436 under the switched prompt. Reads S6c2's stability figures |
| S6e | merged | [#35](https://github.com/keithfernandes12/young-illustrators-primer/pull/35) | `sheet.py`: Florence-2 + SAM 2.1 + ViTPose beside SigLIP 2; every field filled on 82/82 real sheets; two figures found as two on 12/20. Unblocks KAN-42 |
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
- **2026-11-15 — Correction to the Decision above: Pillow *is* declared.** `pyproject.toml:36` has
  `drawing = ["torch>=2.4", "transformers>=5.0,<6", "pillow>=10.0"]`, landed with S6a. The clause
  "Pillow is declared nowhere in this project at all" was true when written and is not now, and
  S6b1 in this same plan states the corrected position: Pillow "lives in the `drawing` extra that
  CI does not install". **The Decision's reasoning survives unchanged** — the renderer avoids an
  image library so the whole strokes-to-bytes path runs in CI, which installs `.[dev]` only, and
  that is about what CI installs rather than about what is declared. The follow-on question below
  ("add `pillow` to the extra once S6a lands it") is **half** answered: `pillow` is in the extra,
  but its second half is not — `tests/drawing/test_render.py` still skips on
  `importorskip("PIL.Image")`, so the decoder check still arrives by accident rather than being
  opt-in. The question stays open on that half.
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
- **2026-11-15 — S6c: "the golden split" in S6c's title means S6b2's `test` split, not
  `configs/golden/`.** The slice body says "over S6b's test split", and the golden set has only a
  `drawing_label` per scenario, no images, so it cannot be scored by a real recogniser. The number is
  top-5 over `split_frames(read v1.1)["test"]`, the same membership `read_dataset("v1.1", "test")` uses.
- **2026-11-15 — S6c: the drawing metrics file is gated directly, not merged through `golden.py`.**
  `evaluate.py` writes the contract shape (`overall` / `baseline` / `slices`) and repeats
  `drawing_top5_acc` at the top level, because `check_gate.py` looks keys up there, and the file is
  gated with `--only drawing`. Merging into `golden.py`'s output was rejected: `check_gate.py` reads
  `fake_models` for the **whole file**, so a merged file either reports the real drawing number as
  `FAKE-…` or, without the flag, certifies the fake story, safety and readability numbers as real.
  `golden.py` is unchanged. Reversible: when a second area has a real number, a merge that carries
  provenance per key replaces this.
- **2026-11-15 — S6c: the `baseline` is chance, 5/345 = 0.0145.** `evaluate.py` says `baseline` is
  "what the model must beat", but the model scored here *is* S6a's baseline. The speech stage writes
  `baseline == overall` when a run is its own baseline. That would make the field say nothing here.
  Quick, Draw!'s own recogniser cannot serve either: every v1.1 drawing has `recognized=True`, so it
  scores 1.0 by selection. Chance is the floor any recogniser must clear. S6d's run passes S6c's
  `overall` as what it must beat.
- **2026-11-15 — S6c scores each drawing as an upload, once.** Each test drawing is rendered to PNG
  by `render_strokes_png` and handed to `SiglipEncoder.encode` as bytes, which decodes, composites
  and runs `prepare_drawing` exactly as for a child's upload. `RenderedSplit[i]` was not fed to the
  encoder: it has already run `prepare_drawing`, so passing it on would preprocess twice. At the
  shipped settings `prepare_drawing` returns a rendered canvas unchanged (S6b1), so the tower sees
  the same pixels training will. Labels are compared by vocabulary index (`data.label_index`,
  case-insensitive), never by string. The cut is a literal 5 (`evaluate.TOP_K`), not
  `DrawingConfig.top_k`.

- **2026-11-16 — The prompt template is `a child's drawing of a {label}.`, chosen on `val`.** S6f's
  sweep over 2,000 `val` drawings (MLflow `d99d75b5`) scored it +4.3 top-5 points [+2.9, +5.9] above
  the model card's `This is a photo of {label}.`, and the KAN-38 owner switched to it before S6d.
  It was chosen on `val` so that `test` stays a report: S6c was re-measured on `test` once under
  it, and no other template was scored there. It is kept exactly as measured, "a airplane"
  included; correcting the article would be a new, unmeasured prompt. Every earlier drawing figure
  (S6c's 0.4956, S6c2's table, S6f's perturbations) was taken under the old template and stays
  on record as that.

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
  `mermaid`, `snowman`, `teddy-bear`), and a prediction can only land on one of those. Which ones
  it actually favours is unmeasured — no model has been run over this corpus. So
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
  `FakeComponents`, whose `FakeDrawingClassifier.top5(true_label)` is handed the answer and echoes it back
  (`fakes.py:27-31`), so the vocabulary is never consulted. **S6c is where it becomes live**, and
  it would present as an unexplained plateau just below the gate. Tracked as KAN-116.
- **2026-11-15 — Superseded: the cap above is gone, KAN-116 is merged.** PR #26 (`cf48752`,
  2026-11-14 23:38Z) regenerated the scenarios, replacing `boat` with `sailboat` and `kite` with
  `cloud`. Re-counted on `origin/main`: 100 scenarios carry a `drawing_label`, 16 distinct values,
  **none absent from `quickdraw_categories.txt`**, so the ceiling is 1.0 and the 0.85 gate has its
  full headroom back. The fact above stayed in this branch for seventeen minutes after #26 merged
  because the branch is cut from `375d0cd` and never saw it — which is the lesson worth keeping:
  **a ceiling is a property of whatever `configs/golden/scenarios.jsonl` holds that day.** S6c
  re-counts rather than quoting either number.
- **2026-11-15 — Two corrections to the fact above, which was already on `main` and should not
  have been edited in place.** First: "100 scenarios carry a `drawing_label`" is true but names
  the wrong population — all 100 carry one, while `golden.py` scores only the **80 `kind: book`**
  scenarios, which is the 80 the superseded fact above it counts. The 16 distinct values and the
  1.0 ceiling are unchanged. Second: "seventeen minutes" is the gap to this branch's fifth
  commit `a8e0c51` on `docs/KAN-109-plan-photographed-eval` (PR #22); the gap to `c57d8eb`,
  which actually recorded the supersession, is **42 minutes**. A third pairing in the fact
  above is also loose: `cf48752` is the branch head at 23:26Z, while 23:38Z is when the
  merge commit `5569fe5` landed.

  The two claims corrected here — the population and the seventeen minutes — were both Both were
  rewritten in place in an earlier commit, against `documentation.md`'s
  "append, do not rewrite, the record"; this entry restores the original and corrects it here
  instead.
- **2026-11-14 — kiddraw is black-stroke only, so it does not answer the colour question.**
  Measured from the repository rather than the paper: `experiments/museumstation/trial-sequence.js`
  sets `path.strokeColor = 'black'`, `strokeCap = 'round'`, `strokeWidth = 10`, and the analysis
  code measures ink from the alpha channel (`img[:,:,3]`), which only works when every stroke is one
  colour on a transparent ground. It is still the only source found with drawings by real children
  **and** category labels: 37,770 drawings, ages 2–10, 48 categories of which 44 map onto our 345
  (39 exact, 5 by clear synonym), checked against the same categories file. It carries no licence
  file and so cannot be ingested as it stands — KAN-110. No PNG was opened; the colour finding is
  from source code.

- **2026-11-15 — The GPU is present; the installed `torch` cannot see it.** Measured today:
  `nvidia-smi` reports an **RTX 4060 Laptop, 8188 MiB, driver 592.82**, while `.venv`'s torch is
  `2.15.0+cpu` with `torch.version.cuda` `None` and `torch.cuda.is_available()` `False`. The cause
  is packaging, not hardware: the `drawing` extra asks for `torch>=2.4` with no build, and PyPI's
  default index serves the CPU wheel on Windows; a CUDA build needs
  `--index-url https://download.pytorch.org/whl/cu124` and roughly 2.5 GB. That remedy is
  **untested** — neither the channel's coverage of torch 2.15 nor the download size was checked.
  **Every "CPU-only" figure
  in the facts above is therefore a property of this environment, not a limit of this machine** —
  including S6d being unrunnable and the measured VRAM figure AGENTS.md asks for being
  unobtainable. Both would be one install away if the remedy works, which nobody has tried.

- **2026-11-14 — S6b1: Amateur Drawings is almost never square, and the recogniser's processor
  squashes.** Over all 174,981 rows of `processed/character_drawings` (batch
  `20261110T055432Z`, metadata only): 1.35% square, 88.17% portrait, 10.48% landscape, aspect
  p5/p50/p95 0.463/0.750/1.333. The real `SiglipImageProcessorPil` at the pinned revision turns a
  512 x 384 drawing into 224 x 224 — a squash, on nearly every drawing in the corpus.
  `prepare_drawing` therefore resizes the long side to `size` and pads with the drawing's own
  edge colour (per-channel median of the outermost pixels), not the renderer's white, which
  would frame a grey photograph in bars no training image has.
- **2026-11-14 — S6b1: binarising after padding turned whole photographs into ink.**
  *Hypothesis:* Otsu over the padded, resized drawing separates marks from paper. *Result:* no.
  On 60 drawings (seed `20261114`, an **unstratified** `numpy` draw without replacement —
  S6b3's sampler is in #29, not on `main`) the median drawing came out **74.2% ink** and the worst
  exactly **75.0%**, the share of a 384 x 512 drawing left after padding it square: the white
  padding was the brightest class, so the whole page fell on the ink side. Fixed by thresholding
  before padding. The order still matters under the local threshold that replaced Otsu (next
  fact) — padding first draws a bar of ink along each seam — and
  `test_binarising_a_non_square_photograph_thresholds_the_drawing_not_the_padding` pins it on
  grey paper: 268 ink pixels padding first, against the drawing's 12. (Its first version used
  light paper on an 8px canvas and, once the threshold went local, passed in either order; an
  independent review caught that, and a pad-first mutation now fails it.)
- **2026-11-14 — S6b1: a global threshold turns lighting into ink, so binarising is local.**
  *Hypothesis:* Otsu on the drawing alone is enough. *Result:* no. Same 60 drawings: in the
  median one **39.8%** of the photograph's own edge binarised to ink, and in 42 of 60 more than a
  quarter — shadows, vignetting, the table round the sheet. Replaced with a local threshold (ink
  when more than `binarise_offset` darker than the mean of a `binarise_window` square; Bradley and
  Roth's published 1/8 and 15%, **untuned**): edge-as-ink median **4.1%**, more than a quarter on
  5 of 60; ink overall p50/p95/max **5.4% / 13.6% / 36.1%** of the drawing, cropped to the
  annotated box 6.3% / 14.0% / 39.7%. Its cost is pinned by a test: a solid dark area binarises to
  its outline. S6c2's binarised row now measures this method, and should say so.
- **2026-11-14 — S6b1: 1.89% of the annotated boxes reach past their image, all by at most a
  pixel.** 3,311 of 174,981 once rounded outwards to whole pixels, maximum overshoot **1.00 px**,
  none with zero area. A strict check would have refused all of them, so `prepare_drawing` clamps
  within `BOX_TOLERANCE = 1.0` px and refuses beyond; corpus-wide that refuses none.
- **2026-11-14 — S6b1: the processor adds nothing after `prepare_drawing`.** The pinned
  processor gives identical `pixel_values` for a prepared array and for a PIL image of it
  (max |diff| 0), and both equal plain `(x / 255 − 0.5) / 0.5`: at 224 its resize is a no-op.
  So `prepare_drawing` is the only resampling between a decoded drawing and the tower, and
  handing the processor an array instead of an image changed nothing it computes.
- **2026-11-14 — S6b1: the inference direction leaves a rendered canvas alone, and the
  augmentation is cheap enough per drawing but not per epoch.** On 150 real Quick, Draw!
  drawings (the first 50 of `house`, `cat` and `bicycle` from the public bucket, read
  anonymously) `prepare_drawing(render_strokes(s))` equals the render on **150/150**.
  `augment_canvas` took **7.6–8.6 ms** mean, p95 10.4–12.3 ms over two runs, then
  `prepare_drawing` about 1 ms; the same seed gave identical output 20/20. At S6b2's ~500k
  records that is about **70 minutes an epoch single-threaded** against rendering's 3 — S6b2
  needs loader workers, and this, not rendering, is the number to profile first.
- **2026-11-14 — S6b1: preparing a full-resolution upload costs more than recognising it.** A
  4032 x 3024 x 3 array took **0.32–0.42 s** in colour and 0.12–0.28 s in greyscale across runs,
  against the tower's 0.11 s warm forward pass; a 512 px Amateur Drawing takes 31 ms mean
  (p95 58 ms). Amateur Drawings was resized by Meta, so no corpus figure shows this — a parent's
  phone will. Not optimised: an integer pre-shrink before the area resize is the obvious fix and
  is KAN-71's to ask for.
- **2026-11-14 — S6b1: the first lighting augmentation fired and did nothing.** It shifted
  brightness and stretched contrast both ways, and on white paper a brighter, punchier draw
  clipped the page back to 255: one seed in four on the test square changed no pixel. Lighting now
  only darkens and flattens (shading, exposure loss, contrast loss), which is what a photograph of
  a page does; a test checks eight seeds all reach the paper.
- **2026-11-14 — S6b1: the new path end to end, on one real drawing.** The first `house` record,
  rendered by `render_strokes_png` and run through `python -m primer.drawing.smoke` (decode,
  `flattened`, `prepare_drawing`, the pinned tower on CPU): `house` at **0.3026**, then `tooth`,
  `square`, `castle`, `barn`; 0.11 s warm. **Evidence about wiring, not quality** — one drawing,
  and still no `drawing_top5_acc` until S6c.
- **2026-11-14 — S6b1 departs from this plan in two places, both on purpose.** `ink` and
  `background` carry over from `drawing_render.yaml` as well as `size` — "binarise to the
  renderer's black-on-white convention" names them, and a copy would be the drift the module
  exists to prevent. And `flattened` stays in `model.py`, as this plan says and against KAN-31's
  description, which recommends moving it: it is Pillow, and moving it would break the ticket's
  own no-image-library acceptance. The shipped `greyscale: false` and `binarise: false` keep what
  S6a did and **decide neither question** — colour versus greyscale is the KAN-38 owner's, on
  S6c2's numbers. None of these figures is in MLflow: they measure preprocessing rather than
  compare models, and there was no host yet (KAN-107; one exists as of 2026-11-15). S6b2 inherits the remaining half of the
  single-path guard — asserting the training loader imports `augment_canvas` from here.
- **2026-11-15 — S6b1: the first independent review found the single-path guard half-blind.**
  *Hypothesis:* the guard catches a second preprocessing path in the drawing path. *Result:* not
  as written. It read only `node.module`, so `from torchvision import transforms` and
  `from scipy import ndimage` passed, as did `.convert(mode="L")` and every Pillow `resize`,
  `crop` and `thumbnail`: a reviewer added a torchvision `Compose([Grayscale, Resize])` to
  `drawing/data.py` and the guard stayed green. All four forms are now caught, each with a test,
  and the same mutation fails it. The guard is still a guard rather than a proof — a second path
  written in plain numpy under a new name passes it, as it passes KAN-87's. The same round found
  the encoder's `prepare=` argument and the processor-size coupling untested (both now pinned by
  a mutation-checked test), hard-coded augmentation constants (now in the YAML), a white paper
  tint that clipped 54.8% of the texture away (tints now leave room for it, enforced at load),
  and an augmentation config with inverted levels that drew blank pages (now refused). A second
  round confirmed all of it by mutation and found three smaller gaps the fixes left — three moved
  settings with no test that they are read, the size check never shown to accept a plain dict
  nor to explain a processor without a fixed square, and `import torchvision` followed by
  `torchvision.transforms.…` slipping past the guard — each now closed with a mutation-checked
  test.
- **2026-11-15 — S6b2: the frozen Quick, Draw! drawings, measured whole.** `v1.1`'s `drawings`
  (read through `primer.data.catalog` at its recorded generations, 15 s): **632,746** rows, all
  `quickdraw`, `source_id` and `record_id` both unique, 345 labels at **1,615–1,977** drawings
  each (median 1,858), `license` CC-BY-4.0 and `attribution` present on every row. *Hypothesis:*
  some records would trip `render_strokes`, since `valid_strokes` accepts three runs where the
  renderer requires two (the open question S6b handed on). *Result:* **none** — every record
  renders, and none carries a third run. So the loader raises on a refusal, naming the record,
  rather than skipping it: one would mean the pipeline and the renderer have diverged.
  Render plus `prepare_drawing` costs **1.19 ms mean, p95 1.85 ms** (every 50th record) — 3.3×
  S6b's 0.356 ms from one sparse category, as that fact warned.
- **2026-11-15 — S6b2: three labels differ from the vocabulary in case, and an exact match
  refused the whole version.** The data stores `the eiffel tower`, `the great wall of china` and
  `the mona lisa`; `quickdraw_categories.txt`, written in the bucket's listing order, has `The
  Eiffel Tower` and so on. Labels now match case-insensitively, and a vocabulary with two labels
  differing only in case is refused rather than guessed between.
- **2026-11-15 — S6b2: every drawing in the split was recognised by Quick, Draw!'s own model.**
  `recognized` is true on all 632,746 rows: the pipeline kept only drawings Google's classifier
  got right. The recogniser is trained and gated on sketches a classifier already found legible,
  so a `drawing_top5_acc` from S6c is measured on an easier set than the raw dataset — and
  further still from a child's drawing. S6c's fact should say so beside the number.
- **2026-11-15 — S6b2: the split, built on `v1.1`.** Salted SHA-256 of `source_id`
  (`configs/drawing_data.yaml`, salt `kan-38-s6b2`, 0.8/0.1/0.1): train **506,324** (80.02%),
  val **63,112** (9.97%), test **63,310** (10.01%); val and test each hold all 345 labels, at
  least 145 and 146 drawings per label. All three build in 1.8 s, since nothing is rendered until
  an item is read. The split is unstratified by design — a hash, not S6b3's samplers, which
  were in #29 and not yet on `main` when this was built — and per-label coverage above is what
  shows that did not cost a class.
  Not in MLflow: these describe the data, not a model, and there was no host then (KAN-107; one exists as of 2026-11-15).
- **2026-11-15 — S6b2: `v1.1` passes the stricter provenance check.** Review asked that blank and
  non-text `license` / `attribution` be refused, not only null ones; the earlier "present on every
  row" was a null check alone, so it was re-measured. `read_dataset("v1.1", split)` for all three
  splits (train 506,324, val 63,112, test 63,310 = **632,746**, 35 s including the bucket read):
  **0** rows blank or not text in either column (and, after a later review round made `source_id`
  and `label` require text too, 0 refused there: all 632,746 still load), each provenance column
  holding one distinct value
  (`CC-BY-4.0`; `Quick, Draw! Dataset, Google Creative Lab`). So the check refuses no row of the
  version this slice trains on. On each split `record(-1)` and `record(-n)` were accepted and
  `record(-n - 1)` raised `IndexError`.
- **2026-11-15 — S6c2: the sample.** 1,000 Amateur Drawings record ids, frozen in
  `src/primer/drawing/configs/stability_sample.txt`, drawn once from `v1.1`'s
  `character_drawings` (174,981 rows) as `default_rng(20261115).choice(sorted(record_id), 1000)`.
  **Unstratified** — a uniform random draw, the plan's stated fallback, because S6b3's
  proportional sampler is in #29 and not on `main`. It matches the corpus where checked: 1.6%
  square and 89.8% portrait against the corpus's 1.35% and 88.17%; all 1,000 CC-BY-NC-4.0. The
  same file in `v1` holds **174,983** rows against `v1.1`'s 174,981 — the two versions pin
  different generations of one path — so the sample names its version and S6d must read `v1.1`.
- **2026-11-15 — S6c2: preprocessing moves the baseline's answer on most children's drawings.**
  S6a's zero-shot baseline on CPU, 4,000 forward passes in 802 s, `v1.1`, the sample above.
  *Hypothesis:* on a corpus the vocabulary cannot name, preprocessing would flip top-1 often, but
  mostly where the margin is thin. *Result:* both halves, and the second less than expected.

  | Adjacent pair | top-1 changes | ... where the margin was above median | top-5 overlap |
  |---|---|---|---|
  | as published → greyscale | **25.5%** | 10.0% | 0.765 |
  | greyscale → binarised | **49.9%** | 35.8% | 0.393 |
  | binarised → cropped to the box | **41.3%** | 24.6% | 0.561 |

  | Variant | top-1 confidence median | top-1/top-2 margin median | distinct top-1 labels | most common top-1 share |
  |---|---|---|---|---|
  | as published | 0.103 | 0.051 | 108 | 53.3% |
  | greyscale | 0.081 | 0.037 | 110 | 50.8% |
  | binarised | 0.085 | 0.039 | 127 | 51.5% |
  | cropped, binarised | 0.093 | 0.044 | 123 | 54.6% |

  Colour alone moves one drawing in four, and one in ten even where the model had a preference.
  Binarising is the largest single move: half of all answers change and **only 39% of the top
  five survives** — so the local threshold (S6b1) changes what the model sees far more than
  colour does, and the binarised row is a different input, not a cleaned-up one. Cropping to the
  annotated box moves 41% again; that row is an upper bound the product cannot reach (no box at
  upload). Margins are thin everywhere (median 0.04–0.05 on sigmoid scores, against 0.098 between
  `house` and `barn` on S6a's smoke drawing), which is the "forced near-miss" the plan predicted —
  yet the above-median half still flips 10–36%, so the movement is not only tie-breaking.
  **Over half of all drawings get one label under every variant, and it is `squiggle`**
  (`top1_label_share` 0.51–0.55): asked to name a child's drawing of a person from a vocabulary
  with no person in it, the baseline reaches for the category that means "a scribble". That is
  KAN-111's gap made concrete, and a reason for that ticket rather than for this slice to act on.
  It also means the baseline is already concentrated before any fine-tune — the collapse
  signature S6d is told to watch for — so S6d must read its stability against this, not against
  zero, and a fine-tune that moves drawings *off* `squiggle` may show as *less* stable while
  being better. *Second hypothesis:* the measurement reproduces. *Result:* a full re-run gave
  every figure above **identically** (779 s), so S6d's comparison is against fixed numbers. **Not accuracy**, and nothing here says which variant is right. Not in MLflow:
  there was no host then (KAN-107; one exists as of 2026-11-15); the summary JSON's fields are what an MLflow run would log.
- **2026-11-15 — S6c2: the decision the plan asked for, and the one it did not.** The rank
  statistic is **top-5 set overlap** (`evaluate.py` docstring has why). Whether the 0.85 gate
  keeps reading the rendered Quick, Draw! number is **not decided here** — it is the KAN-38
  owner's, and the plan's acceptance asked for a recorded decision this slice is not entitled to
  make. What these numbers add to that decision: the gate's number is taken on drawings a
  classifier already recognised (S6b2), while on children's drawings the same model's answer
  moves 25–50% under preprocessing alone.
- **2026-11-15 — S6e: three models chosen, downloaded with approval, and measured before use.**
  From the brief, on licence and the 8 GB rule: **Florence-2 base** (MIT) for the description and
  character boxes, **SAM 2.1 small** (Apache-2.0) for masks, **ViTPose base** (Apache-2.0) for
  keypoints; `body_type` takes no model of its own. Each pinned by full revision in
  `configs/drawing_sheet.yaml`. Florence-2 comes from `florence-community/Florence-2-base`, not
  `microsoft/`: the Microsoft repo needs `trust_remote_code` and runs three Python files from it,
  while transformers 5.23 loads the converted repo natively — same model, MIT, weights and JSON
  only (chosen by the user). On disk: 0.47 / 0.18 / 0.34 GB. SAM 2's image processor needs
  **`torchvision`**, installed with approval as `0.29.1+cu130` to match the CUDA torch (6.6 MB,
  `--no-deps`, `pip check` clean). *Hypothesis:* photo-trained models would mostly work on
  children's drawings. *Result, on the first 40 drawings of S6c2's frozen sample against the
  dataset's own annotations, GPU:* Florence-2 `<OD>` finds the figure at IoU ≥ 0.5 on **37/40**
  (median 0.89), but also boxes `human face` 28 times and `footwear` 19 times beside `person` 33
  — so a box ≥ 80% inside a larger one is folded into it rather than made a character. SAM's mask
  matches the annotated segmentation at IoU ≥ 0.5 on **26/40** (median 0.68), slightly under the
  figure (31% of the box against the annotation's 40%). ViTPose places **61%** of annotated
  joints within 10% of the figure's size (PCK@0.1). SAM loads with a transformers warning that
  the checkpoint is `sam2_video` read into `Sam2Model`; the masks above are its output anyway.
- **2026-11-15 — S6e: the acceptance, measured — every field always filled, two characters found
  as two in 12 of 20.** `CharacterSheetRecognizer` on real drawings, GPU, warm, all four models
  resident (**2.00 GiB peak VRAM**). 40 single-figure drawings: one sheet on 37, two or three on
  3; every one of the 45 sheets carries candidates, embedding, crop, mask, a known body type and,
  for humanoids, 17 joints; a description on 40/40; **0.82 s** a drawing mean, p95 1.32 s. Body
  type: 33 humanoid, 5 creature, 7 object, against 40 annotated human figures — so about one in
  six children's figures is not read as a person, and KAN-42 would place rather than repose it.
  20 composites of two real figures side by side (in memory, never written): **exactly two
  sheets on 12/20**, one on 6, three or four on 2; all 37 sheets complete; 1.07 s a drawing.
  The plan's acceptance — "one sheet per character" — therefore holds on 60% of two-figure
  drawings, not on all of them; the misses are the detector boxing two figures as one. In CI the
  acceptance runs on fake weights through `build_book`, with a stand-in for the approve step
  (KAN-64). Not in MLflow (KAN-107). Unstratified sample, as S6c2's.
- **2026-11-15 — S6e: what review changed, and what it did not.** Sheets now come back **most
  confident first**, which is what `DrawingAnalysis.characters` and `.main` promise and what
  `build_book` takes as the story's subject; they had been in box-area order, so a large doubtful
  object could have become the subject ahead of a smaller, sure one. Joints are normalised
  against the pixel bounds the crop is actually saved at (floor of the top-left, ceil of the
  bottom-right) rather than the raw float box, so a fractional detection no longer shifts the
  skeleton by up to a pixel on the saved crop. A mask with no pixels falls back to the unmasked
  box instead of whitening the child's lines; a box wholly outside the drawing is dropped; a
  joint that is NaN or infinite is kept as not visible; and a failing model call now says which
  step, and which character, failed. None of these changes a count above (sheets per drawing,
  body types), but the figures were **not** re-measured with the real models afterwards.
  `HuggingFaceSheetModels` itself — the pose box conversion, the detection pairing — runs only
  against the real weights and is not exercised in CI; the measured figures above are its test.
- **2026-11-15 — S6c2: what the figures above were taken under, and what review changed.** The
  figures are from the **CPU, fp32** run (S6a's baseline, as stated), before the summary recorded
  its own execution. It now records the resolved `device` and `dtype`, the prompt template,
  count and a hash of the prompt text, every scoring setting, and each variant's preprocessing
  settings, and `stability --device cpu` pins the CPU baseline on a machine with a GPU, whose
  bf16 moves confidences about 20% (Verified fact on the CUDA torch). Review also fixed three
  things that do not move the table: top-5 overlap is divided by the labels there are when fewer
  than five (this vocabulary has 345, so the table is unchanged), a per-call `prepare` whose
  `render.size` differs from the checkpoint's resolution is now refused rather than resampled
  twice, and the `top1_change_rate_confident` docstring said the opposite of what it measures:
  flips that were only near-ties would pile up below the median margin and leave this rate near
  zero, and a rate close to the overall one would say the flips are spread evenly. In all three
  pairs it sits below the overall rate (10.0% against 25.5%, 35.8% against 49.9%, 24.6% against
  41.3%) but far above zero: some flips are near-ties, and confident answers moved too. The
  figures were **not** re-measured after these changes; none of them touches a number in the
  tables. A second review round added guards that cost nothing on the real data: each image's
  bytes must match its recorded `image_sha256` (checked on five sample drawings against the
  bucket: all match; `record_id` is unique across `v1.1`'s 174,981 rows, so the 1,000 ids match
  1,000 rows), scores must be sigmoid outputs in 0-1, `--out`'s folder must exist before the
  13-minute run starts, and each pair's summary carries `confident_drawings`, the size of the
  above-median subset, which is half the sample unless the median margin is 0 (a saturated or
  collapsed model, S6d's case to watch). Left for later, not done here: a `--data-uri` flag like
  `primer.data.cli` has, and the share of drawings whose top-1/top-2 margin is exactly 0, which
  would make a bf16 GPU run comparable with this CPU baseline.
- **2026-11-15 — Local GPU work runs: `.venv`'s torch is now a CUDA build.** *Supersedes* the
  2026-11-15 fact "The GPU is present; the installed `torch` cannot see it" and the CPU-only notes
  in earlier facts, which stay as the record of what was true then. `torch` went from
  `2.15.0+cpu` to **`2.14.1+cu130`** (CUDA 13.0) from `https://download.pytorch.org/whl/cu130` —
  2.14.1 because no Windows Python 3.13 CUDA build of 2.15 exists; it meets the `drawing` extra's
  `torch>=2.4`, and `pip check` is clean. `accelerate 1.15.0` was installed alongside, for
  Hugging Face training (device placement, mixed precision). Checked in this session, not only
  reported: `torch.cuda.is_available()` is True on an **RTX 4060 Laptop GPU, 8.0 GiB**, bf16
  supported (driver 592.82). `python -m primer.drawing.smoke --device cuda` ran the pinned SigLIP
  2 checkpoint in bf16 on the GPU **for the first time** — the CUDA path S6a's facts called
  untested — with the same top 5 as CPU (`house`, `tooth`, `square`, `castle`, `barn`), **0.01 s**
  warm against 0.12 s on CPU, and **1.23 GB peak VRAM**: the measured figure AGENTS.md asks for,
  for inference. Run twice, in two sessions, identically. The drawing tests pass on the new torch.
  **Caveat:** `house` scored **0.2451** in bf16 on the GPU against **0.3026** in fp32 on the CPU.
  The ranking is identical, so `drawing_top5_acc` is unaffected, but confidences move ~20%
  between devices — no confidence threshold or KAN-71 cache key is portable between them, as the
  `CONFIDENCE_DECIMALS` comment in `model.py` already warned for the embedding. Fine-tuning VRAM
  is still unmeasured; that is S6d's to record.
- **2026-11-15 — S6b3's strata, measured over the real corpus.** `dvc pull` of
  `data/character_drawings` gives **174,981 rows** — the source docstring's "over 178,000" is the
  annotation count, not what survives preprocessing. Building the four strata over all of them:
  **126 cells, not the 256 four bins would allow**, because `aspect` supports only **two** bins.
  Its 25th, 50th and 75th percentiles are all **0.75** — **61.8%** of the corpus is exactly
  3:4 portrait — so `pd.qcut` drops the duplicate edges. (An earlier draft of this fact said
  "three quarters"; the quartiles only need a bit over half the mass at one value, and the
  measured share is 108,087 of 174,981.) That path was specified defensively and turns
  out to be the live case, not an edge case.
  **Pooling is rarer than this slice assumed:** 39 of the 126 cells fall under `min_size = 100`,
  and together they hold **1,291 rows, 0.7% of the corpus**. The slice previously said pooling
  would be "a routine path and not an edge case"; that was wrong and is corrected above.
  After pooling: **87 strata**, largest 6.3% of rows, smallest 0.06%. Measured ranges:
  `long_edge` 133–12,000, `framing` 0.004–1.003 (a box can marginally exceed its frame), `area`
  **0.0**–258,572, and no nulls in any of the four.
- **2026-11-15 — The two samplers behave as specified on the real corpus.** Drawing 1,000 rows
  from all 87 strata: the **proportional** sampler puts 6.4% in the largest stratum against the
  corpus's own 6.3%, and the **uniform** sampler flattens that to 1.2%. Both return exactly 1,000
  rows, both reproduce their draw under the same seed, and both carry `license` and `attribution`
  through. This is the difference the slice exists to make legible: using the uniform draw for an
  evaluation set would over-represent the corpus's rare modes fivefold and report a number the
  product never sees.

- **2026-11-15 — The stratum-occupancy distribution, which acceptance clause 5 asks for.**
  Over the 87 pooled strata: smallest **106** rows, median **884**, largest **10,991**. Deciles of
  stratum size are 106 / 183.6 / 354 / 884 / 2,509.5 / 5,545.4 / 10,991 at the 0th, 10th,
  25th, 50th, 75th, 90th and 100th percentiles — interpolated, so only the 0th, 50th and
  100th are sizes an actual stratum has. (The 25th lands at position 21.5 of 87, between
  strata of 351 and 357 rows; it prints without a decimal only because their mean is whole.) Half the corpus sits in the largest **12** strata, 80% in the
  largest 31, 90% in the largest 46. So the distribution has a long thin tail rather than a few
  dominant modes, which is why the uniform and proportional samplers diverge as much as they do —
  the uniform draw gives the 46 smallest strata the same weight as the 12 that hold half the rows.
- **2026-11-15 — The `aspect` bin does not separate portrait from landscape, and the PR body was
  wrong to say it does.** The realised edge is **0.75**, not 1.0, because the bins are quantiles.
  So bin 1 is "wider than 3:4", which mixes orientations: of 174,981 rows, 154,290 are portrait,
  18,347 landscape and 2,344 square — and **7,927 portrait rows (5.1% of all portrait drawings)
  fall above the 0.75 edge** into the same bin as the landscape ones. What the bin actually
  separates is "exactly 3:4 or narrower" from "everything else", which is a mode indicator rather
  than an orientation one. A fixed edge at 1.0 would give a genuine orientation split; a quantile
  edge cannot. Whether that is worth having is the open question on this slice.
- **2026-11-15 — S6c: S6a's zero-shot baseline scores `drawing_top5_acc` = 0.4956 and fails the
  0.85 gate.** Measured with `python -m primer.drawing.evaluate --model
  src/primer/drawing/configs/drawing_baseline.yaml --version v1.1 --split test --device cuda` over
  **every** drawing of S6b2's test split, **63,310**, read from `gs://young-illustrators-primer-data`.
  No sample, so no sampler applies. Taken on the **RTX 4060 in bf16** at encoder revision `75de2d55`,
  prompts sha256 `9b82e482…`. It ran in 757 s, 0.012 s per drawing including render and PNG
  round-trip. The run started from an uncommitted tree identical to commit `b5e7294`, which the file records as its `git_sha`. `check_gate.py --only drawing` on the file prints
  `FAIL drawing_top5_acc >= 0.85: actual=0.4956`, a verdict rather than `metric missing`. The chance
  baseline is 0.0145.
  - **In MLflow (backfilled the same day).** The shared server came up later on 2026-11-15:
    MLflow 2.17.2 on VM `mlflow` in the shared project, reached through an IAP tunnel. The run is
    `c2b6818759af457e972d868b77019225` in experiment `yip-drawing`, run name
    `s6c-siglip2-zero-shot-baseline`. It was logged from the metrics file above, not re-scored.
    It carries the model-development tags (`where=local`), 23 params, `drawing_top5_acc`, the chance
    baseline as `baseline/drawing_top5_acc`, and per country `slice/country_code/<code>/drawing_top5_acc`
    and `.../n`. Artifacts are `eval/<metrics file>` and `CARD.md`. `check_gate.py --mlflow-run-id`
    then wrote `gate/drawing_top5_acc` and the tag `gate_passed=false`.
  - **Two deviations.** First, slice metrics use `slice/<key>/<value>/<metric>`, not
    model-development.md's `slice/<key>=<value>/<metric>`: MLflow rejects `=` and `:` in metric
    names. Second, the client is `mlflow-skinny==3.18.0`, against a 2.17.2 server. Both
    `mlflow==2.17.2` and `mlflow-skinny==2.17.2` were tried first. The full package pins
    `pandas<3` and `pyarrow<18`, and pyarrow 17 has no Python 3.13 wheel. The skinny 2.17.2 would
    downgrade `packaging` 26.3 → 24.2 and `protobuf` 7.36 → 5.29. The 3.18 client changed no
    installed package.
  - **What the number is over.** Top-5 on **rendered** Quick, Draw!: the gate's number, not the
    product's. Every v1.1 drawing has `recognized=True`, meaning Quick, Draw!'s own model already
    named it, so this set is easier than raw sketches. On children's drawings, S6c2 measured this
    same baseline's top-1 moving 25–50% between preprocessing variants alone, with 51–55% of
    drawings coming out `squiggle`. That is not accuracy, but it says 0.4956 is not a floor for the
    product.
  - **`squiggle` again.** On a random 300 of these drawings (seed 0, a diagnostic, not the
    result), `squiggle` was the top-1 for 82 (27%), and that 300 scored 0.52 top-5.
  - **Per country.** 165 countries appear. 54 have at least `SMALL_SLICE` (100) drawings and are
    judged; they hold 61,350 of the 63,310. Judged slices run from 0.4333 (BE, n=210) to 0.6471
    (CL, n=102). The largest are US 0.4774 (n=28,557), GB 0.4689, CA 0.4753, DE 0.5143 and AU
    0.5246. Differences of that size over a few hundred drawings are within noise, and nothing here
    is adjusted for label mix per country.
  - **Device and dtype matter for S6d.** bf16 on GPU ranks near-ties differently from fp32 on CPU:
    S6a's `house` scored 0.2451 against 0.3026. S6d's comparison has to be taken with `--device
    cuda` (bf16) to be like for like. S6c2's stability figures were taken on CPU in fp32.
  - **First attempt, for the record.** A `--limit 300` timing probe scored 0.0033, below chance. The
    split frame is ordered by label, and its first 300 rows are all `the eiffel tower` (192) and
    `the great wall of china` (108). Hypothesis: the pipeline was broken. Result: it was not; the
    labels are hard, and a random 300 scored 0.52. `--limit` takes the first N rows, which is why
    its help text says "never a result".
  - **What can be gated, after review (Copilot, 2026-11-15).** The top-level `drawing_top5_acc` is
    written only for the complete `test` split: a `--limit` probe and `--split val` both omit it, so
    `check_gate` says `metric missing` and `--mlflow-run-id` cannot tag them `gate_passed=true`
    (`val` chooses the model; it stays diagnostic). The ratio is written unrounded, because 53,811
    of 63,310 is 0.8499605 and rounds to a passing 0.85. An empty `--version` and a
    `prompt_template` without `{label}` are refused before any bucket or weights are read. The
    0.4956 above is unchanged: it was the four-place display of the same hits. Once the arguments
    parse, the previous run's `--out` file is moved to `<out>.previous` (the first free `.previous.N` if that is taken) before any other refusal or
    load, so a run that dies cannot leave an old result at `--out` for `check_gate` to read, and a
    probe reusing the path does not destroy a full-split result; only a file with `"stage":
    "drawing"` is moved, anything else is refused untouched. The result is written beside and
    renamed into place. The file also records `gate_eligible`, `data_uri`, and the package's own commit and
    whether its tree was dirty.
  - **The golden ceiling, re-counted.** `configs/golden/scenarios.jsonl` holds 80 book scenarios
    over 16 distinct `drawing_label`s. All 16 are in the 345, case-insensitively and exactly, so a
    flawless recogniser reaches 1.0. That is a property of the golden set, not of this number,
    which is not taken over it.

- **2026-11-16 — S6c under the switched prompt: `drawing_top5_acc` = 0.5436, still a gate FAIL.**
  The same command as S6c's 0.4956, over all **63,310** test drawings, RTX 4060 in bf16, encoder
  revision `75de2d55`, from clean commit `b749f34`, with only `prompt_template` changed (prompts
  sha256 `d3425b53…`). It ran in 942 s. `check_gate.py --only drawing` prints `FAIL
  drawing_top5_acc >= 0.85: actual=0.5436`. MLflow run `b72bc442f32f43cba857e9127f2b6c6f` in
  `yip-drawing`, run name `s6c-siglip2-zero-shot-child-prompt`, tagged `supersedes_run` =
  `c2b68187…` and `gate_passed=false`, logged from the metrics file by the same path as the
  backfill.
  - **+4.8 points on `test`** against +4.3 [+2.9, +5.9] on `val` in S6f: the gain carried over
    from the split it was chosen on, and it is a better prompt, not a better model. S6d's
    fine-tune must beat 0.5436 under this prompt, not 0.4956.
  - **Per country.** The same 54 judged slices now run from 0.4735 (RO, n=452) to 0.6765 (CL,
    n=102). US 0.5282 (n=28,557), GB 0.5183, CA 0.5343, DE 0.5573, AU 0.5712: every large slice
    rose by 4–6 points, so the switch did not open a gap between countries.

- **2026-11-16 — S6c2 under the switched prompt: children's drawings get less stable, and
  confidently wrong.** S6c2's command (`evaluate stability --version v1.1 --device cpu`), CPU fp32,
  the same 1,000 frozen drawings, 999.8 s, prompts sha256 `d3425b53…`. Between #33's merge and
  `b749f34` nothing on the stability path changed except comments and this template, and the old
  table reproduced exactly, so the template is the only variable. Old figures in brackets.

  | Adjacent pair | top-1 changes | ... where the margin was above median | top-5 overlap |
  |---|---|---|---|
  | as published → greyscale | **33.7%** [25.5%] | 22.2% [10.0%] | 0.745 [0.765] |
  | greyscale → binarised | **66.4%** [49.9%] | 60.2% [35.8%] | 0.534 [0.393] |
  | binarised → cropped to the box | **49.5%** [41.3%] | 44.0% [24.6%] | 0.681 [0.561] |

  | Variant | top-1 confidence median | margin median | distinct top-1 | most common top-1 share |
  |---|---|---|---|---|
  | as published | **0.929** [0.103] | 0.035 [0.051] | 160 [108] | 16.8% [53.3%] |
  | greyscale | 0.851 [0.081] | 0.053 [0.037] | 144 [110] | 20.3% [50.8%] |
  | binarised | 0.212 [0.085] | 0.032 [0.039] | 135 [127] | 42.8% [51.5%] |
  | cropped, binarised | 0.257 [0.093] | 0.035 [0.044] | 123 [141] | 36.2% [54.6%] |

  - **The most common answer is now `The Great Wall of China`**, not `squiggle`, under every
    variant. One possible mechanism, unmeasured: the template reads "a child's drawing of a The
    Great Wall of China.", and the shared prefix "a child's drawing of" fits every drawing here.
  - **Confidence jumped tenfold on unbinarised uploads** (median 0.10 → 0.93) while the margin did
    not grow, so a high sigmoid score now says little about which label is right. Nothing in
    `src/` thresholds the recogniser's confidence today; anything that later does would read it
    as certainty.
  - **Top-1 flips rose in every pair**, and in the confident half they roughly doubled. Top-5
    overlap rose for two pairs, so the top five as a set is steadier there while its order is not.
  - This is the pattern S6d's acceptance names: a gain on rendered Quick, Draw! (+4.8, above)
    together with a drop in stability. It is measured here on the prompt, before any training.

## Open questions

- **What S6c hands to S6d, found in review of #37.** None is a defect in S6c as shipped. S6d is
  KAN-38's, so these are hand-offs to its owner, not a claim that all four are done under that
  ticket. Recorded on KAN-38 on 2026-11-15. (1) *Ties, S6d acceptance:* `scored` breaks ties in
  vocabulary order, so a tied label at the fifth/sixth boundary can count as a hit; bf16 sigmoids
  make near-ties plausible. Record `boundary_ties` and a pessimistic accuracy beside the headline
  before S6d's number nears 0.85. (2) *The gate does not bind a file to a run, a design decision
  before S6d registers a model:* `check_gate --mlflow-run-id` tags any run from any drawing file
  that passes, and the file carries no checkpoint hash or run id. The code is in `check_gate.py`,
  which every area shares, so it may need the gate's owner or its own ticket. (3) *MLflow logging
  is not in the repo, S6d:* the backfill script was scratch; S6d needs a logging helper or flag so
  it logs the same params and `slice/<key>/<value>/<metric>` names. `model-development.md`'s
  `slice/<key>=<value>` text (changed in #38) and its contract paragraph (which still says
  `golden.py` merges each area's `overall`) need to follow, and are #38's file. (4) *The gate is
  device- and version-agnostic, S6d acceptance plus a team decision:* a CPU/fp32 run and a bf16
  GPU run are both gate-eligible, and nothing checks the drawing count against the version's
  expected test size. Decide which device and dtype certify.

- **What S6b1 hands to S6b2 and S6c2.** Recorded here so they are not found only in Verified
  facts. *S6b2:* assert the training loader imports `augment_canvas` from `preprocess.py`, the
  half of the single-path guard that needs a loader to exist; run augmentation in loader workers,
  since at ~8 ms a canvas it costs about 70 minutes an epoch single-threaded; and decide whether
  training should see non-square framing — 98.65% of Amateur Drawings reach the tower padded with
  bars of their own paper colour, while rendered and augmented canvases are always square, so
  training never sees a bar (raised on KAN-38). *S6c2:* its binarised row now measures a local
  threshold, not a global one, and its cropped row cannot run through the encoder at all —
  `Encoder.encode(bytes, prompts)` has no way to pass a box, so either the box travels through
  that seam or S6c2 calls `prepare_drawing` itself before encoding (raised on KAN-109).
- **What S6b2 hands to S6d.** Found in review of #32, none a defect in the loader as shipped.
  *Epochs:* `RenderedSplit.set_epoch` changes the dataset in the process that calls it, so a
  multi-worker `DataLoader` with `persistent_workers=True` never sees the new epoch; S6d should
  rebuild the workers each epoch or carry the epoch in the sampler. *Reads:* `read_dataset`
  re-reads and re-checks the whole version (about 15 s plus the frame check) for every split asked
  for, so three splits are three reads; a caller-supplied frame or a cache would avoid it once a
  training run wants all three. *Seeds:* an item's seed is 32 bits, so about 29 pairs of the 506k
  train items share one in an epoch, which is harmless but should not be mistaken for
  independence. Not raised on a Jira issue yet; S6d's owner decides.
- **Where does EXIF orientation get applied?** A phone JPEG with Orientation=6 reaches
  `prepare_drawing` unrotated, as it reached the tower before this slice. The standard fix,
  `PIL.ImageOps.exif_transpose` at the decode site, is decode rather than preprocessing, but the
  single-path guard bans `PIL.ImageOps` wholesale and would fail CI on it. *Recommended:* apply it
  in `encode` beside `flattened` and narrow the guard to allow exactly `exif_transpose`. Raised
  on KAN-64, which owns uploads.
- **Should a full-resolution upload be shrunk before the area resize?** A 4032 x 3024 photo costs
  0.3–0.4 s to prepare and peaks around 337 MB of allocations (the float64 copy), and the
  resampling-weight cache can hold 32 matrices of ~7 MB. An integer block-mean pre-shrink would
  cut both. *Recommended:* yes, when KAN-71 budgets the upload path. Raised on KAN-71.

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
  only. **Update, 2026-11-15:** a shared MLflow server now exists on the shared project (KAN-107,
  `docs/runbooks/mlflow-server.md`), reached through an IAP tunnel, and S6c's baseline is its first
  run. Who hosts it long term is still the project owner's to confirm
  (`docs/engineering/model-development.md`), and a cloud job cannot reach it yet (see the runbook).
- **Does the recogniser see colour, or greyscale?** The pipeline answers this inconsistently today
  — see the Verified fact above — and nothing has measured what the inconsistency costs. Zero-shot
  the asymmetry is probably mild, since SigLIP 2 was trained on photographs; the concern is after
  S6d, where a tower fine-tuned only on greyscale renders can regress on colour drawings while the
  gate, scored on those same renders, stays green. That is reasoning, not measurement.
  *Recommended:* give the recogniser greyscale, since S6b1's preprocessing module has to produce
  it anyway, and keep colour for the character sheet and KAN-42, where it matters far more than it
  does for naming an object. That needs a colour field on `Character`, which does not exist. S6c2's
  greyscale variant measures the cost either way. *Decides:* the KAN-38 owner with the KAN-31 owner,
  before S6d. Not before S6b1: that module is specified to make greyscale optional, so it can be
  built either way, and S6c2 — which needs it — is what produces the number this decision wants.
  Deciding earlier would mean deciding without the measurement.
- **Should KAN-117 be split in two?** It covers S6b3 — an ordinary slice an agent builds — and
  S6b4, a person-only spike that may conclude "we tried and it did not help". Different
  actors, different risk, different definitions of done, and when S6b3 lands and the issue
  transitions, S6b4 silently reads as done. *Recommended:* split it, for the same reason the
  slice was split. *Decides:* whoever picks up S6b3 first, before they transition anything.
- **Does the recogniser get a `humanoid` class?** **Tracked as KAN-111.** The 345 name no whole
  person while Amateur Drawings is ~178,000 drawings of human figures **whose label is free** — the
  corpus definition supplies it, so nothing needs hand-labelling. *Recommended:* add the class,
  drawing ~2,000 rows through S6b3's **uniform-per-stratum** sampler and holding the rest back as
  the first accuracy measurement on real children's drawings — **single-class recall, not
  345-way accuracy**, since every held-back row is humanoid by the corpus definition. Three
  riders, all load-bearing:

  1. **Both classes must pass through S6b1's preprocessing**, or the model separates them on paper
     texture and answers `humanoid` to every upload, dragons included. Test it rather than assume
     it: take held-out Quick, Draw! drawings, run them through KAN-31's photo-degradation
     augmentation and score them — if accuracy collapses when a known *non*-person starts looking
     photographed, the model was reading the paper. This is a cross-corpus test and S6b3 does not
     perform it.
  2. **Single-class recall is reported beside the 345-way number**, since answering `humanoid` to
     everything scores 100%.
  3. **The licence follows the weights.** Quick, Draw! is CC-BY-4.0; Amateur Drawings is
     **CC-BY-NC-4.0** (`configs/sources.yaml`, flagged there as metadata not checked), so
     training on it arguably makes the resulting weights non-commercial. The
     only rider with a legal consequence, and the only one attaching to a trained artefact rather
     than to a measurement. Fine for coursework; record it before it is discovered later.

  The cheaper alternative leaves the vocabulary alone and has KAN-64 suppress the label when
  `body_type` is humanoid — no accuracy number, but no change to the gate either. A third option,
  replacing the metric with one that assumes no fixed vocabulary, is larger than the MVP and
  neither of the first two forecloses it. *Decides:* the KAN-38 owner with the product owner,
  **before S6d**, because option one changes the gate's denominator to 346 and makes every prior
  run incomparable.
- **Does the 345-category vocabulary fit what children actually draw?** **Tracked as KAN-111.** It
  cannot name a whole person
  (see the Verified fact above), and KAN-64's flow is "name each character" — children draw
  people constantly. The recogniser will answer `angel` or `snowman`, confidently, and no amount of
  fine-tuning adds a word the label set does not contain. This is a product question before it is a
  model one, and it also decides what an accuracy number on real child drawings could ever mean:

  | Corpus | Real children | How it was captured | Subjects the 345 can name | Gives an accuracy number? |
  |---|---|---|---|---|
  | Quick, Draw!, rendered | no — adults, 20s timer | drawn in a browser | all 345 | yes, and it is the gate's |
  | Amateur Drawings | yes | **uploads, mix unknown** | **none today** — human figures only | **only if the `humanoid` class above is accepted** |
  | kiddraw (KAN-110) | yes, ages 2–10 | natively digital, museum kiosk | 44 of its 48 | yes, if the licence clears |

  The four kiddraw categories that fail to map are `bowl`, `person`, `lamp` and `bottle` — the
  output of the same check as the 44, recorded in the Verified fact above and in KAN-110. **`person`
  is one of them**, so kiddraw would answer the corpus problem and not this one: it supplies
  drawings by real children over subjects the vocabulary can name, and still cannot say what the
  recogniser should return for a child's drawing of a person.

  "Photographed" is not established for Amateur Drawings and this plan should stop implying it: the
  source docstring says Meta resized "the uploaded drawings" to 512 px, and `configs/sources.yaml`
  says only "PNG images". Uploads to a web demo plausibly mix photographs, scans and natively
  digital drawings, and the 512 px resize is a second confound on top of the first. The mix is
  checkable — `data/character_drawings.dvc` is frozen, though its dep is a `gs://` path with
  `push: false`, so checking needs a pull and bucket access, as S6c2's own run will. Both confounds
  are worth reporting alongside those figures.

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
- **S6e needs `torchvision`: decided 2026-11-15 by the KAN-38 owner.** SAM 2's image processor
  imports it, and like the CUDA torch it must match the torch build (`0.29.1+cu130` for
  `2.14.1+cu130`). Decision: a separate **`sheet` extra** (`torch>=2.4`, `transformers`,
  `torchvision>=0.19`) so the recogniser alone does not need it, and the CUDA pair documented
  in `AGENTS.md` → Commands rather than pinned in `pyproject.toml`. `pip install -e
  ".[sheet]"` now runs `CharacterSheetRecognizer` on CPU; the GPU needs the documented
  reinstall of torch and torchvision together. Not done, by that decision: pinning a build in
  the extra or `requirements.txt`.
- **Two figures read as one in 6 of 20 two-figure drawings.** Florence-2's detector sometimes
  boxes neighbouring figures together, and the containment rule then folds them. *Recommended:*
  measure on real multi-character uploads before tuning — composites of two single-figure
  drawings are a stand-in — and try Florence-2's phrase grounding or SAM's automatic masks if the
  rate holds. KAN-42 should treat a sheet count as a lower bound until then.
- **A fresh clone installs the CPU torch again.** The `drawing` extra says `torch>=2.4` and names no
  index, so `pip install -e ".[drawing]"` on Windows takes PyPI's CPU wheel, and the GPU fact above
  silently stops being true for the next person or agent. *Decided 2026-11-15 (KAN-38 owner):*
  document the step, do not pin. `AGENTS.md` → Commands now carries the torch and torchvision
  reinstall from the cu130 index in one command, and `pip check`. `accelerate`, which Hugging
  Face training wants (S6d), is still a separate `pip install accelerate`.
- **Should a part-label box that escapes its figure be a character, and should containment look at
  labels?** Found in review of #35, and left as it is because either answer moves the measured
  counts above. Containment ignores labels, so a person drawn inside a house box (80% or more
  within it) is folded into the house, and the house becomes the subject. And a part label whose
  box is less than 80% inside its figure — a face half outside the person — becomes a character
  of its own; `human face` contains `human`, so it is read as humanoid and ViTPose runs on a face
  box. The same happens to a lone face when the person box is missed (the figure is found at
  IoU ≥ 0.5 on 37 of the 40 drawings).
  *Recommended:* a list of part words (face, head, hand, shoe, ...) dropped before grouping, never
  scored or posed, and a label-aware fold, measured on the 40 drawings and 20 composites before it
  ships, since both can only lower the sheet counts. *Decides:* the KAN-38 owner.
- **When is a SAM mask too small to use?** Only an empty mask falls back to the unmasked box now.
  A mask covering a sliver of its box still whitens the rest, and SAM covers 31% of the box
  against the annotation's 40%, so a threshold needs the distribution of mask share over the 82
  sheets, which has not been taken. *Recommended:* measure it, then add `min_mask_share` to
  `drawing_sheet.yaml`.
- **The default artifact store is a process-lifetime dictionary.** `MemoryArtifactStore` never
  evicts a child's crops and masks, a `mem://` ref cannot be read by another process (KAN-42's
  worker), and a content-hash name means two children's identical crops would share one object
  once the GCS store exists, which complicates deleting one child's data. *Recommended:* serving
  passes a store explicitly and the GCS one decides its naming against the deletion path (KAN-64).

- **Keep `a child's drawing of a {label}.` now that S6c2 has been re-measured under it?** It gains
  +4.8 top-5 on the gate's split, but on children's drawings top-1 flips more often, the median
  confidence of an unbinarised upload goes from 0.10 to 0.93, and `The Great Wall of China`
  replaces `squiggle` as the most common answer (Verified facts, 2026-11-16). The switch was
  decided before these figures existed. *Options:* keep it and have S6d read its stability against
  the new table; revert to `This is a photo of {label}.`; or run S6c2 under `a sketch of a {label}.`
  (+3.3 on `val`) first, which selects on Amateur Drawings rather than on `test`. *Decides:* the
  KAN-38 owner.

## Out of scope

- IP-Adapter character consistency: after the MVP.
- Whisper read-aloud scoring: out of scope **for this plan**, not for the project — it shipped
  under KAN-95/96/97/100 and lives in `src/primer/speech/`.
- Bias slicing and drift alerts: PRs #7 and #8.
