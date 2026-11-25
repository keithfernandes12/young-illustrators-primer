# Model card: drawing

The pretrained zero-shot baseline (KAN-38, plan slice S6a). Nothing is registered yet: S6c measured
`drawing_top5_acc` over S6b's rendered test split (below), and there is no trained checkpoint until
S6d. Fill the remaining blanks in that slice rather than guessing them here.

| Field | Value |
|---|---|
| Stage | drawing |
| Purpose | recognise what a child drew |
| Base model | `google/siglip2-base-patch16-224` @ `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` (Apache-2.0) |
| Method | zero-shot: sigmoid image–text scores over the 345 Quick, Draw! categories |
| Data version | _not pinned_ — zero-shot, nothing trained |
| Owner | _unassigned_ |

## Why this base model

`docs/briefs/drawing-recognition-model-candidates.md` compares SigLIP 2, OpenCLIP, DINOv2 and
DINOv3 for this output. SigLIP 2 wins on quality among the Apache-2.0 candidates; the **base**
checkpoint rather than a larger one is the local-GPU rule in `AGENTS.md` — the member machine this
is iterated on is an 8 GB laptop, and S6d has to fine-tune whatever is chosen here, not only run it.

DINOv3 was not taken: its custom licence is a product risk the brief flags, and nothing it offers
over SigLIP 2 is worth taking that on for a course MVP.

## Size, measured

Loaded and counted, not estimated (2026-11-14):

| | Parameters | fp32 | bf16 (the CUDA path) |
|---|---|---|---|
| Whole model | 375,187,970 | 1.398 GB | 0.699 GB |
| Vision tower (what S6d fine-tunes) | 92,884,224 | 0.346 GB | 0.173 GB |
| Text tower | 282,303,744 | 1.052 GB | 0.526 GB |

**VRAM has not been measured.** The run that produced these numbers was on CPU — PyPI's Windows
`torch` wheel is CPU-only — so `smoke.py`'s `peak_vram_gb` returned `null`. AGENTS.md asks for a
measured VRAM figure before a candidate is trusted on the 8 GB machine; producing it needs a CUDA
build and belongs to S6d, which is also the slice where it actually binds.

## Metrics

Gated keys this stage owns are listed in `evaluate.py` as `OWNED_METRICS`, taken from
`primer.evaluation.contract`. Record the value, the baseline it beat, and the run id.

| Key | Value | Baseline | Run |
|---|---|---|---|
| `drawing_top5_acc` | 0.5436 over all 63,310 test drawings (**fails** the 0.85 gate) | chance 0.0145 (5 of 345); target 0.85 | MLflow `b72bc442f32f43cba857e9127f2b6c6f`, experiment `yip-drawing` |

Measured on 2026-11-16 with `python -m primer.drawing.evaluate`: data version v1.1, RTX 4060, bf16,
prompt template `a child's drawing of a {label}.`, commit `b749f34`. It is a number over **rendered
Quick, Draw! drawings**, not children's uploads, and only comparable with another run on the same
device, dtype and prompt (a CPU/fp32 run ranks near-ties differently and was not done).

**The previous figure, 0.4956** (2026-11-15, MLflow `c2b6818759af457e972d868b77019225`), was the
same model and split under the model card's template, `This is a photo of {label}.`. The template
was switched on S6f's `val` sweep (+4.3 points there), and the switch measured +4.8 on `test`. That
is a better prompt, not a better model: S6d's fine-tune must beat 0.5436.

**Do not read a number into the zero-shot setup.** The brief's survey of the sketch literature puts
plain CLIP near 43% top-1 on Quick, Draw!, and the gate wants 0.85 top-5. This baseline exists to
produce the floor S6d must beat and to prove the path end to end, not to clear the gate.

## Slices

Per-slice results for the keys in `slices.py` (`country_code`) and for category, and what was done about any gap.

The two keys `model-development.md` names for this stage are `country_code` and category (the
label). Both are reported by `python -m primer.drawing.bias_report` (S6g), from the per-drawing rows
of the S6c run. The rows must reproduce that run's metrics file exactly before anything is computed.
The run is v1.1 `test`, all 63,310 drawings, on GPU in bf16, under `a child's drawing of a {label}.`,
with top-5 0.5436. The figures below come from commit `8e7bdb0`, before this PR's review fixes, and
have not been re-run on those fixes (they need the 16-minute GPU run; see the plan's Open questions).

**Country: nine disparities named, so the earlier "within noise" is withdrawn.** 54 of 165 countries
have at least `SMALL_SLICE` (100) drawings and are judged. Each is compared with what its own label
mix predicts, from each label's accuracy in the *other* countries, and tested with Holm's correction
at a family-wise 5%.
- **Below their label mix:** RO −6.9, SA −6.4, GB −3.2 and **US −2.5** (n=28,557).
- **Above it:** RU +4.6, TH +5.0, KR +5.9, IN +7.0 and **JP +11.7**.
- **Controlling for label mix reveals gaps rather than explaining them away.** The raw comparison
  against the other countries' rate names only six of the nine, and misses RO, SA and RU.
- **Why, is not measured here.** The likeliest cause is drawing style, since how people draw an
  object varies by country. The US and GB being *below* their label mix matters most: they are 52%
  of the split and the bulk of S6d's training data.
- **Mitigation, proposed for S6d and not applied:** a zero-shot model has nothing to reweight.
  - Cap any one country's share of the training sample with `capped_weights`, so a fine-tune on a
    US-heavy split cannot buy US accuracy with the other countries'.
  - Re-run this report on every candidate. The cap is a guard; this bar is the control. For a
    country named below its label mix (in the baseline or the candidate), reject a candidate whose
    difference falls by more than the baseline's 95% half-width (from
    `difference_low` to `difference_high` in the report, not adjusted for the number of
    countries); for one above its label mix, report the move and do not reject on it. A
    yardstick, not a test: the candidate's and baseline's differences are two noisy estimates on
    the same drawings, so S6d may tighten it.
- **Read with two limits.** Drawings are treated as independent, and Quick, Draw! rows carry no
  drawer id, so a player who drew many items would make the p-values too small and this cannot be
  adjusted for. "Named" is statistical, not practical: at the US's size a gap of two or three
  points is named, so read the difference and its interval, not only the verdict.

**Category: per-class difficulty, not a fairness gap.** All 345 categories are judged (146–227
drawings each). Top-5 runs from 0.00 to 0.996.
- **Hardest:** `garden` (0.00), `pool`, `postcard`, `beach`, `firetruck`, `pond` and `ocean`
  (0.02–0.04). These are scenes and near-synonyms the vocabulary also names.
- **Easiest:** `line`, `triangle` and `zigzag` (0.99 and above).
- **Mitigation, proposed for S6d:** report per-category accuracy for every candidate, and do not
  accept a gain that comes only from categories already above the median.

## Mitigations and trade-offs

- **Zero-shot on sketches is weak**, and knowingly so — see above. The mitigation is S6d.
- **The prompt trades children's-drawing stability for the gate's number.** `a child's drawing of
  a {label}.` adds 4.8 top-5 points on rendered Quick, Draw!, but on S6c2's 1,000 children's
  drawings top-1 changes more often between preprocessing variants (34 / 66 / 50% against 25 / 50 /
  41%), the median top-1 confidence of an unbinarised upload is 0.93 rather than 0.10, and the
  most common answer is `The Great Wall of China` rather than `squiggle`. Kept knowingly
  (2026-11-16). A confidence near 1 from this model is not evidence the label is right.
- **Confidences are sigmoid, not softmax.** SigLIP's training loss is pairwise, so each score is
  "does this label fit", independent of the others. They do not sum to 1. Ranking is unaffected,
  which is all `drawing_top5_acc` reads, but anything that later thresholds a confidence must know
  which of the two it holds.
- **A transparent upload is composited onto white before scoring**, not flattened. `alpha_background`
  in `drawing_baseline.yaml` sets the grey level, and white is chosen to match what the model is
  measured on: the Quick, Draw! split is rendered as dark ink on white, and a photograph of a
  paper drawing is too. An upload composited onto anything else would be inverted relative to the
  split, so this is a setting that moves every confidence, not a display choice.
- **One character per drawing.** The whole image is the subject. A drawing with a child, a dog and
  a house returns one character until S6e adds detection, so the other two are silently lost.
- **The structural fields are empty**, not guessed: `body_type` stays `UNKNOWN`, which KAN-42 reads
  as "do not repose" rather than as a default branch, and `description` stays `""` rather than a
  sentence built from the label. S6e fills both from different models.
- **The vocabulary is fixed at the 345 Quick, Draw! categories.** An invented creature — which is
  what the product is *for* — can only come back as the nearest of those 345.
- **The embedding is a cache key (KAN-71), not the consistency signal.** KAN-48 measures
  consistency in DINOv2 + CLIP space. Reusing this one there would measure the encoder against
  itself.
- **That cache key is bound to the dtype it was taken at.** The tower runs in bf16 on CUDA and
  fp32 on CPU, and near a unit vector's typical component one bf16 step is 244 times the
  rounding `model.py` applies — so the same drawing analysed on a GPU and on a CPU produces two
  different `identity_key`s and two cache entries. Rounding cannot fix it: the two runs computed
  different numbers, rather than the same number with noise on top. Anything sharing a KAN-71
  cache across devices has to record which dtype the key came from.
- **The embedding is unit length in fp32, approximately so in bf16.** Measured on the real
  checkpoint, a bf16 embedding's norm is 1.0018 rather than 1.0 — the normalisation is exact in
  the arithmetic it is done in, and bf16 has eight bits of mantissa. Nothing here depends on the
  norm being exactly 1 (the ranking is a dot product of two vectors scaled the same way), but a
  consumer that assumes a unit vector should know it holds to three decimals, not to float
  precision.

## Sensitivity

Measured by `python -m primer.drawing.sensitivity` (S6f) on 2,000 `val` drawings of v1.1, on GPU in
bf16, under the shipped prompt `a child's drawing of a {label}.`. The run is MLflow
`dfaee7e263744e9e85ce939c30489128` (2026-11-16), and the full table is in the plan's Verified facts.
Each figure is the change in top-5 accuracy against the same drawings unperturbed, with a 95%
interval. Baseline top-5 on the sample is 0.518.

- **Missing strokes hurt most.** Dropping the last stroke costs 14.7 points [12.8, 16.6], and the
  last two cost 21.8 [19.6, 24.2]. A child who stops early, or a scan that loses faint lines, will
  cost far more accuracy than any change of pen.
- **Line thickness: no clear change for 1 and 5 px on this sample, and 7 px is worse.** 7 px
  costs 2.5 points [1.1, 3.8]. No equivalence margin is claimed, so this is consistent with 3 px,
  not proof that it is best. Top-1 still changes on 27–38% of drawings, so the label shown first
  is less stable than the gated number suggests.
- **The prompt is a real setting.** The shipped template beats the model card's `This is a photo of
  {label}.` by 4.3 points [2.9, 5.9], and `{label}` alone by 8.5. `a sketch of a {label}.` is
  within noise of it (-0.95 [-2.1, +0.25]).
- **The answer rests on a few regions.** In occlusion maps of 20 correctly ranked drawings, the
  three most important 32 px patches carry a median 58% of the confidence drop. The share counts
  positive drops only, and a drawing with three or fewer inked patches scores 1.0 trivially. Here
  the median drawing has 25.5 inked patches (range 12–37), so none is trivial, and no map had an
  undefined share (such a map reports `None`, is counted in `top3_share_undefined`, and is left out of the median).
- **Partial erasure is too weak to read.** The training erasure removes a median 0.5% of ink; on
  the 202 drawings that lost at least 5%, the change is -2.0 [-5.4, +1.0].
- **Uncorrected intervals.** About ten 95% intervals are reported with no multiplicity correction;
  the 7 px result is the one line-width interval that excludes zero, so treat it as suggestive
  given the uncorrected intervals.
- **Not covered:** training hyperparameters (S6d), and children's drawings, for which see S6c2's
  preprocessing stability.
- **History:** the first run (MLflow `d99d75b514934b18836e4ad94ddbc32f`, 2026-11-15) was under the
  old prompt and gave the same picture: -11.7 and -19.9 for strokes, no clear line-width change, and
  +4.3 for this prompt.
