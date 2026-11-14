# Model card: drawing

The pretrained zero-shot baseline (KAN-38, plan slice S6a). Nothing is registered yet: there is no
measured `drawing_top5_acc` until S6c runs this over S6b's rendered split, and no trained
checkpoint until S6d. Fill the blanks below in those slices rather than guessing them here.

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
DINOv3 for this output. SigLIP 2 wins on quality among the Apache-2.0 candidates; the **base** tower
rather than a larger one is the local-GPU rule in `AGENTS.md` — the member machine this is iterated
on is an 8 GB laptop, and S6d has to fine-tune whatever is chosen here, not only run it.

DINOv3 was not taken: its custom licence is a product risk the brief flags, and nothing it offers
over SigLIP 2 is worth taking that on for a course MVP.

## Metrics

Gated keys this stage owns are listed in `evaluate.py` as `OWNED_METRICS`, taken from
`primer.evaluation.contract`. Record the value, the baseline it beat, and the run id.

| Key | Value | Baseline | Run |
|---|---|---|---|
| `drawing_top5_acc` | _not measured_ — S6c | target 0.85 | |

**Do not read a number into the zero-shot setup.** The brief's survey of the sketch literature puts
plain CLIP near 43% top-1 on Quick, Draw!, and the gate wants 0.85 top-5. This baseline exists to
produce the floor S6d must beat and to prove the path end to end, not to clear the gate.

## Slices

Per-slice results for the keys in `slices.py`, and what was done about any gap. Empty until S6c.

## Mitigations and trade-offs

- **Zero-shot on sketches is weak**, and knowingly so — see above. The mitigation is S6d.
- **Confidences are sigmoid, not softmax.** SigLIP's training loss is pairwise, so each score is
  "does this label fit", independent of the others. They do not sum to 1. Ranking is unaffected,
  which is all `drawing_top5_acc` reads, but anything that later thresholds a confidence must know
  which of the two it holds.
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
