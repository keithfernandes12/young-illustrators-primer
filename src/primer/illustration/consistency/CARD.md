# Model card: character consistency (stage 6)

Pretrained, never trained. The tuned values are the threshold and the choice of embedder, both set by
`experiment.py` on held-out picture books. Numbers marked _pending_ are filled in from
`consistency_report.json` after a person runs the experiment, never by hand.

| Field | Value |
|---|---|
| Stage | 6 · character consistency, inside `illustration/` (owns no gated key of its own: `clip_consistency` belongs to `illustration` in `contract.METRIC_OWNERS`) |
| Purpose | Score each book: do the pages look like the character the child drew? One score per book, in the request path |
| Base model | Baseline **`clip-vit-b32`** (`openai/clip-vit-base-patch32`, image tower). Candidates `clip-vit-l14`, `dinov2-small` must beat it on val ROC-AUC |
| Weights revision | _pending_: pin the commit from the report's `revisions` into `configs/consistency.yaml` |
| Data version | `v1.1` (`storybook_pages`) |
| Threshold | _pending_. `thresholds.json` keeps `null` until the team agrees a value (KAN-42 needs human approval for it) |
| Owner | Rutvij Surti (stage 6, KAN-48); coordinates with the KAN-42 illustration owner |
| Jira | KAN-48 (benchmark): KAN-103 code, KAN-104 experiment run, KAN-105 serving and golden set; KAN-42 (where it is wired in) |

## How the score works

`ConsistencyScorer.score(references, pages)` embeds the character's reference images and every page in
one batch, then:

- **identity**: each page's cosine similarity to its closest reference, aggregated by `mean` (or the
  stricter `min`). This is the score.
- **cohesion**: the mean similarity of the pages to each other. It is reported, and weighted in only as
  a sensitivity knob (`identity_weight < 1`). CLIP cosines are compressed, so it lifts wrong-character
  books; see Sensitivity.
- No reference means a score of 0.

The references should be the parent-approved storybook-style reference (`reference_image_ref`, KAN-38),
not the raw crayon crop and not `Character.embedding`. That embedding is SigLIP 2, a different space,
so it is a latency cache key only.

## Metrics

| Metric | Split | Selected | Baseline | Run id |
|---|---|---|---|---|
| ROC-AUC, same book vs other book | test | _pending_ | _pending_ | _pending_ |
| Precision / recall / F1 at the threshold | test | _pending_ | _pending_ | |
| False-positive rate at the threshold | test | _pending_ | _pending_ | |
| `clip_consistency` on golden books | golden | _blocked: needs generated pages (KAN-42)_ | | |

## Slices

Reported per `source` (picture-book origin: African Storybook, StoryWeaver) and `language`. Slices under
`primer.data.bias.SMALL_SLICE` (100) pairs are listed but not judged. Record the largest ROC-AUC gap
from the report's `slice_gaps`, and what was done about it:

- _pending_

Proposed mitigation if one origin or language separates worse: a per-slice threshold, which the
experiment can compute from the same val pairs. Rejected by default, because serving does not know a
generated book's "origin". The more likely fix is a better embedder.

## Sensitivity

- **Threshold sweep** (0.50 to 1.00): what each candidate threshold trades in precision against recall.
- **Perturbations** applied to one image of each test pair: greyscale, 64 px downscale, centre crop,
  JPEG at quality 15. A large drop under greyscale means the score keys on colour, which suits a
  child's purple dragon but makes a lighting change look like a different character.
- **Raw CLIP cosines are compressed.** In a CPU smoke test on synthetic shapes (2026-11-14, not a
  measurement on real data), a purple circle scored 0.77 against green squares. Four green-square pages
  scored 0.83 as a book once cohesion was weighted at 0.25. The threshold must therefore come from the
  experiment: an intuitive 0.80 would pass the wrong character.
- SHAP and LIME do not apply: there is no feature vector to attribute.

## Limitations and trade-offs

- **The labels are a proxy.** "Same book" does not mean "same character on both pages", so the
  positives are noisy and ROC-AUC is a lower bound on how well the embedder tells characters apart.
- **Domain shift.** The pages are human-drawn illustrations, not diffusion output. Re-set the threshold
  from the spread of real generated books once KAN-42 produces them. That is the "set in experiments"
  step the stage map asks for.
- **Latency.** CLIP runs on every book. That is one image-tower pass per page plus the references, in one
  batch. L/14 costs several times what B/32 does, so it has to win on ROC-AUC by enough to pay for
  that. Fold the time into `p95_full_book_s` (owned by `system`), measured on the serving
  GPU.
- **Privacy.** Reference images are child data. The scorer keeps them in memory, and nothing here logs
  images or embeddings.
