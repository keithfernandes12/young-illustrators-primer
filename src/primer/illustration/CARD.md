# Model card: illustration

Pretrained baseline (KAN-42, plan slice S7a in `docs/plans/mvp-cloud-deploy.md`). Nothing is trained
yet, and nothing is registered.

| Field | Value |
|---|---|
| Stage | illustration |
| Purpose | illustrate each page so it keeps the shape of the child's drawing |
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` rev `451f4fe`, CreativeML OpenRAIL-M |
| Conditioning | `lllyasviel/control_v11p_sd15_scribble` rev `3564ec7` (OpenRAIL), on the child's sketch |
| Settings | `configs/illustration_baseline.yaml`: UniPC, 25 steps, guidance 7.5, ControlNet scale 1.0, seed 42, 512×512 |
| Data version | not pinned: a pretrained model, used as published |
| Owner | Adit Vakil (KAN-42) |

## How a page is made

The sketch is the in-memory upload when serving passes it, or it is re-read from `drawing_ref` (`gs://`
or `file://`). `sketch.py` makes it white strokes on black, pads it to a square and resizes it. Children
draw dark on light, and the scribble model was trained on light on dark. One diffusion call follows, and
the PNG is written under a random name. `characters` is accepted but not yet used.

## Metrics

Gated keys this stage owns are listed in `evaluate.py` as `OWNED_METRICS`, taken from
`primer.evaluation.contract`. `clip_consistency` has no threshold yet, and `evaluate.py` is still a
stub, so no gated number exists. Measured locally, not gated:

| Measure | Value | How |
|---|---|---|
| Seconds per page | 39.3–44.2 | Apple M4, MPS, fp16, 25 steps, three runs (2026-11-13 and 2026-11-14) |
| Load from cache | 3.4–5.9 s | same |
| Edge recall against the sketch | 0.993, against 0.619 for the sketch rotated 180° | baseline run, 2026-11-13 |

## Slices

None measured yet. The planned slices are subject category, and skin-tone and origin prompts.

## Mitigations and trade-offs

- **No safety check on generated images here.** Diffusers' checker is off because it silently returns a
  black page. The check belongs in serving, after `illustrate()` (KAN-94). Synthetic inputs only until
  then.
- **Shapes are kept, and details drift.** In the baseline, a single round window became three windows,
  and cloud ellipses became coloured orbs. The scribble model follows outlines, not what they mean.
- **Not usable from `build_book` yet.** `consistency()` raises until the stage 6 score exists, and
  `build_book` calls it on every book. Use `illustrate()` directly until then.
- **Faint and unusual drawings.** The stroke threshold is set relative to the paper, so pencil, yellow
  crayon, transparent canvas exports and dim photos keep their strokes. A page with no strokes is
  refused. Uneven lighting in a phone photo is not handled yet: a shadow can read as a stroke.
- **No character identity across pages.** IP-Adapter on `characters` is follow-up work; pages return
  `identity=None`, so `clip_consistency` cannot be scored against a character yet.
- **Too slow for the latency target on a laptop.** About 4–5 minutes for a 6-page book, against the
  120 s `p95_full_book_s` target. S7b needs the L4 or a few-step sampler.
