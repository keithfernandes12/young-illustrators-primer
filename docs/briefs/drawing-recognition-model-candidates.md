---
title: Open-source model candidates for the drawing recognition stage
type: brief
created: 2026-11-14
status: draft
author: rishiparekh
jira: [KAN-38]
tags: [vision, models, research]
related: []
---

# Open-source model candidates for the drawing recognition stage

## Problem

KAN-38 asks the drawing stage to return a **character sheet per character** plus **one description per
drawing**. That is not one model: the contract has six separable outputs, and each one has a different
set of open-source candidates, a different licence risk and a different "can we skip training entirely"
answer. This brief lists the candidates per output with links to how each works, so the model choice is
made against the contract rather than against whichever model was named first.

It stops short of choosing. Nothing here has been benchmarked on our data.

## Sources

Read:

- PR [#11](https://github.com/keithfernandes12/young-illustrators-primer/pull/11) — `src/primer/drawing/`
  (`interface.py`, `data.py`, `model.py`, `train.py`, `evaluate.py`, `slices.py`, `CARD.md`),
  `src/primer/serving/book.py`, `src/primer/evaluation/contract.py`,
  `.claude/skills/golden-gate/thresholds.json`.
- Jira: [KAN-38](https://northeastern-team-12.atlassian.net/browse/KAN-38) (the output contract),
  [KAN-42](https://northeastern-team-12.atlassian.net/browse/KAN-42) (the consumer),
  [KAN-64](https://northeastern-team-12.atlassian.net/browse/KAN-64) (confirmation UI, produces `approved`),
  [KAN-87](https://northeastern-team-12.atlassian.net/browse/KAN-87) (stroke → image rendering),
  [KAN-31](https://northeastern-team-12.atlassian.net/browse/KAN-31) (augmentation, derived sets),
  [KAN-24](https://northeastern-team-12.atlassian.net/browse/KAN-24) (ingestion, **Done**),
  [KAN-90](https://northeastern-team-12.atlassian.net/browse/KAN-90) (`n_keypoints` = 17 everywhere).

Not read: `docs/project_scoping.md` Section 7 in full, KAN-48 (consistency benchmark) and KAN-71
(latency/cost) beyond what KAN-38 and KAN-42 quote of them.

Assumed: Quick, Draw! is the label source and the `drawing_top5_acc` evaluation set; Amateur Drawings is
the source of boxes, masks and keypoints. Both come from KAN-24's description, not from reading the
frozen data.

## What the contract actually asks for

From `src/primer/drawing/interface.py` on PR #11, plus the fields KAN-38 says are still missing:

| Field | On PR #11? | Consumer | Needs a model? |
|---|---|---|---|
| `label` + `confidence` | yes (`Character`) | KAN-64, golden gate | **Yes** — gated at `drawing_top5_acc ≥ 0.85` |
| `description` (per drawing) | yes (`DrawingAnalysis`) | story area | **Yes** — a captioning VLM, not the classifier |
| `embedding` | yes (`Character`) | KAN-71 cache key | **Yes** — any image encoder; *not* the consistency signal |
| `body_type` | **no** | KAN-42 reposing branch | Yes, but 3 classes — zero-shot is plausible |
| `crop_ref` / `mask` | **no** | KAN-42, KAN-64 | **Yes** — detection + segmentation |
| `keypoints` | **no** | KAN-42 OpenPose reposing | **Yes** — humanoids only, 17 joints |
| `reference_image_ref` + `approved` | **no** | KAN-42 IP-Adapter | Rendering, not recognition — see open question |

Two structural notes from KAN-38 that change what we build, not just which weights we load:

1. **Ranking is within a character, not across characters.** `DrawingAnalysis.top_labels(k=5)` on PR #11
   reads the top of each character's list, so on a two-character drawing "top-5" silently means
   something else than it does on Quick, Draw!. `Character` needs its own ranked candidates.
2. **The description comes from a different model than the label.** KAN-38 already says this (Florence-2
   for the description, SigLIP 2 for the label). The `DrawingRecognizer` Protocol is one method, so
   `analyse()` will fan out to at least two models behind it.

## Candidates by output

### 1. Label + confidence — the gated metric

The target is top-5 accuracy ≥ 0.85 over Quick, Draw! classes. This is the one number the release gate
reads for our stage, so it is the one that must be trained and measured rather than borrowed.

| Candidate | How it works | Licence | Notes |
|---|---|---|---|
| **SigLIP 2** (named in KAN-38) | Sigmoid (not softmax) contrastive image–text pretraining, plus a caption-decoder loss, global–local and masked-prediction losses. Gives an image tower you can linear-probe or fully fine-tune. [Paper](https://arxiv.org/abs/2502.14786) · [HF `google/siglip2-*`](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107) · [timm image-only tower](https://huggingface.co/timm/vit_base_patch16_siglip_512.v2_webli) | Apache-2.0 (verify on the official card) | The original sigmoid-loss idea is [SigLIP, arXiv 2303.15343](https://arxiv.org/abs/2303.15343) |
| **OpenCLIP / CLIP** | Softmax contrastive image–text; the reference point every sketch paper compares against. [CLIP paper](https://arxiv.org/abs/2103.00020) · [open_clip repo](https://github.com/mlfoundations/open_clip) | MIT / Apache-2.0 | Cheap, well-understood, weaker than SigLIP 2 on recent benchmarks |
| **DINOv2 backbone + linear head** | Self-supervised, no text tower. Strong frozen features; you add a 345-way head. [Paper](https://arxiv.org/abs/2304.07193) · [repo](https://github.com/facebookresearch/dinov2) | Apache-2.0 | Simplest thing that can hit a fixed label set; no zero-shot |
| **DINOv3** | Same family, much larger pretraining (LVD-1689M), ViT and ConvNeXt variants, better dense features. [Meta blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/) | **Custom "DINOv3 License"**, not Apache-2.0 | Licence is the reason to think before defaulting to it |

**What the literature says about sketches specifically** — worth reading before assuming a photo-trained
encoder transfers:

- Zero-shot CLIP is *weak* on sketch benchmarks. Bandyopadhyay et al. report plain CLIP at ~45.6% on a
  combined Quick, Draw! / TU-Berlin / edgemap set, rising to ~63% with learned visual prompts on a frozen
  ViT — [arXiv 2407.03893](https://arxiv.org/abs/2407.03893). On Quick, Draw! alone the gap is smaller
  (43.2% → 47.4%). These are top-1 numbers, not our top-5, but the message holds: **fine-tuning is not
  optional.**
- Frozen pretrained features vs. supervised training on the same data: ~71.9% vs. 76.1% top-1 on
  Quick, Draw! ([arXiv 2210.15146](https://arxiv.org/abs/2210.15146), ResNet-50 encoder).
- Stroke-sequence models are a different family entirely, if we ever want to use the raw strokes rather
  than KAN-87's rendered images — [Sketch-BERT, arXiv 2005.09159](https://arxiv.org/abs/2005.09159).

**Recommendation to test first:** SigLIP 2 image tower, fine-tuned on KAN-87's rendered Quick, Draw!
images with KAN-31's crayon/paper augmentation. Keep a frozen-features + linear head run as the baseline
`CARD.md` has to beat, because it costs an afternoon and sets the floor.

### 2. Description — a captioning VLM, separate from the classifier

The story writer needs "a child's crayon drawing of a purple dragon with one wing bigger than the other",
which a 345-way classifier cannot say.

| Candidate | How it works | Licence | Notes |
|---|---|---|---|
| **Florence-2** (named in KAN-38) | One seq2seq model with task prompts (`<CAPTION>`, `<DETAILED_CAPTION>`, `<OD>`, `<REFERRING_EXPRESSION_SEGMENTATION>`). 0.23B / 0.77B. [Paper](https://arxiv.org/abs/2311.06242) · [HF `microsoft/Florence-2-large`](https://huggingface.co/microsoft/Florence-2-large) | MIT | Small, cheap, and its detection/segmentation prompts could also serve output 3 |
| **Qwen3-VL 8B Instruct** | General VLM, much better free-form description; MoE variants for more quality per active parameter. [HF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Apache-2.0 | Heavier; check against the KAN-71 latency budget |
| **Moondream 3 (preview)** | 9B MoE, ~2B active — designed for small-footprint VQA/captioning. [Repo](https://github.com/vikhyat/moondream) | Apache-2.0 (Moondream 2; confirm for 3) | Good quality-per-GPU-dollar if Florence-2 is too terse |
| **SmolVLM2** | 256M–2.2B, built for edge. [HF blog](https://huggingface.co/blog/smolvlm2) | Apache-2.0 | The floor option if the description must be ~free |

General landscape: [Hugging Face's 2025 VLM survey](https://huggingface.co/blog/vlms-2025).

**Caveat:** every one of these is trained overwhelmingly on photographs. Caption quality on crayon
drawings is an empirical question — compare 2–3 on ~50 real drawings before committing, and note that a
wrong description is worse than a terse one here, because the story is built on it.

### 3. Crop and mask

**The strongest option may need no training at all.** Meta's *Animated Drawings* project solved exactly
this problem on exactly this dataset and released the fine-tuned weights:

- [Repo `facebookresearch/AnimatedDrawings`](https://github.com/facebookresearch/AnimatedDrawings) (MIT) —
  "animation code and fine-tuned model weights for drawn human figure detection and pose estimation".
- [Paper, arXiv 2303.12741](https://arxiv.org/abs/2303.12741) · [Meta blog](https://ai.meta.com/blog/using-ai-to-bring-childrens-drawings-to-life/)
- Pipeline: Mask R-CNN (Detectron2) for the figure box → **morphological ops, not the predicted mask**,
  for the cutout → a ResNet-50 heatmap pose estimator fine-tuned on children's drawings.
- The Amateur Drawings dataset (~178k drawings with boxes, masks and joints) is the same data KAN-24
  already ingested, and its 17 joints is the `n_keypoints = 17` in KAN-90.

That makes it a zero-training baseline for `crop_ref`, `mask`, `keypoints` **and** `body_type`
(the detector only fires on humanlike figures). Its limit is in the name: humanoids. Dragons, houses and
spaceships are out of distribution.

For the general case:

| Candidate | How it works | Licence | Notes |
|---|---|---|---|
| **SAM 3** | Promptable *concept* segmentation: give a noun phrase or an image exemplar, get **every** instance. Zero-shot LVIS mask AP 48.8; ~30 ms/image on an H200. [Paper arXiv 2511.16719](https://arxiv.org/abs/2511.16719) · [repo](https://github.com/facebookresearch/sam3) · [Meta blog](https://ai.meta.com/blog/segment-anything-model-3/) · [Ultralytics docs](https://docs.ultralytics.com/models/sam-3/) | **Meta "SAM License"**, gated download on HF | Fits our "several characters per drawing" case better than anything else |
| **SAM 2** | Promptable segmentation from points/boxes; no text prompting, so it needs a detector in front. [Paper arXiv 2408.00714](https://arxiv.org/abs/2408.00714) · [repo](https://github.com/facebookresearch/sam2) | Apache-2.0 | The licence-safe fallback |
| **Grounding DINO (+ SAM)** | Open-vocabulary detection from text; pairs with SAM for masks. [Paper arXiv 2303.05499](https://arxiv.org/abs/2303.05499) · [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) | Apache-2.0 | The pre-SAM-3 way to do text-prompted instance masks |
| **BiRefNet** | High-resolution dichotomous segmentation / matting — one clean foreground cutout. [Repo](https://github.com/ZhengPeng7/BiRefNet) · [paper arXiv 2401.03407](https://arxiv.org/abs/2401.03407) | MIT | Good for single-character drawings on white paper |
| ~~BRIA RMBG-2.0~~ | BiRefNet architecture + proprietary data. [HF card](https://huggingface.co/briaai/RMBG-2.0) | **Non-commercial** (`bria-rmbg-2.0`) | Popular, but the licence rules it out unless the project stays non-commercial |

### 4. Keypoints (humanoids only)

Meta's drawing-specific pose weights above are the obvious first try. If they are not good enough, or if
we want a modern backbone to fine-tune on the Amateur Drawings joints:

| Candidate | How it works | Licence | Notes |
|---|---|---|---|
| **ViTPose / ViTPose++** | Plain ViT backbone + lightweight decoder, heatmap per joint. ~81% COCO AP for the large variants. [ViTPose arXiv 2204.12484](https://arxiv.org/abs/2204.12484) · [ViTPose++ arXiv 2212.04246](https://arxiv.org/abs/2212.04246) · in `transformers` as `VitPose` | Apache-2.0 | Accuracy-first on a GPU |
| **RTMPose** | Real-time top-down pose, SimCC-style coordinate classification; the deployment-friendly option. [arXiv 2303.07399](https://arxiv.org/abs/2303.07399) · [mmpose](https://github.com/open-mmlab/mmpose) | Apache-2.0 | Best accuracy/latency trade for KAN-71 |
| **Sapiens** | Human-centric foundation models (pose, seg, depth, normals), up to 308 keypoints. [arXiv 2408.12569](https://arxiv.org/abs/2408.12569) · [repo](https://github.com/facebookresearch/sapiens) | Check Meta licence | Overkill for 17 joints, but the strongest backbone |
| **MediaPipe Pose Landmarker** | On-device, very fast. [Docs](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) | Apache-2.0 | Trained on photos of people; likely poor on stick figures |
| ~~OpenPose~~ | The classic bottom-up PAF method. [Repo](https://github.com/CMU-Perceptual-Computing-Lab/openpose) | **Non-commercial academic licence** | KAN-42 says "OpenPose reposing" — that means the *pose format* ControlNet expects, not necessarily this codebase. Worth confirming with the KAN-42 owner |

Independent comparisons put ViTPose and Sapiens ahead on accuracy, RTMPose ahead on speed, and OpenPose
as a legacy baseline — e.g. an infant-movement study across MediaPipe/OpenPose/PCT/RTMPose/Sapiens/ViTPose
([Oxford ORA](https://ora.ox.ac.uk/objects/uuid:ff7ad4b5-3e9f-4d71-819d-15d19154a947)) and a sign-language
pose-system evaluation ([arXiv](https://arxiv.org/pdf/2604.24609)).

### 5. `body_type` — humanoid / creature / object

Three classes, and the branch KAN-42 picks from. Options, cheapest first: zero-shot SigLIP 2 or CLIP text
prompts; a mapping from the Quick, Draw! label we already predict (most of the 345 classes have an obvious
body type); a 3-way head on the same backbone as output 1. The label mapping is probably enough and costs
nothing — but it fails exactly where it matters, on the invented creatures the product is *for*.

### 6. `embedding` — and what it is not

KAN-38 is explicit: this is a **cache key for latency (KAN-71)**, not the consistency measurement.
KAN-48 benchmarks consistency with **DINOv2 + CLIP** similarity, which is a different space from
SigLIP 2's. So: take the embedding from whatever encoder output 1 ends up using, and do not let it leak
into the consistency metric. `Character.identity_key` in `interface.py` already exists for passing
identity downstream **without** copying the child-derived vector onto every page — keep using it.

## Vocabulary

- **crop** (`crop_ref`, `mask`) — the character cut out of the child's own crayon lines.
- **reference** (`reference_image_ref`) — that character redrawn in storybook style and approved.
  Never used interchangeably with *crop*; feeding the crop to IP-Adapter makes every page look like crayon.
- **character sheet** — one per character. A drawing has several; a drawing has exactly one *description*.
- **candidates** — the ranked labels *inside* one character, which is what `drawing_top5_acc` measures.

## Options

**A. One model, one job each (SigLIP 2 + Florence-2 + SAM 3 + RTMPose).** Best quality per output;
four sets of weights to load, version, and fit inside the KAN-71 latency budget.

**B. Reuse Meta's Animated Drawings weights for everything structural**, and train only the classifier.
Fastest to a working end-to-end path, and the only option where crop/mask/keypoints need no training at
all. Caps us at humanoid figures until replaced.

**C. Florence-2 for description *and* grounding.** One VLM, prompted twice, covers the description and a
referring-expression mask. Fewer models; quality on crayon drawings unproven.

## Questions to decide

1. **Who owns the stylise-and-approve step?** Both tickets call this a chicken-and-egg, and it is
   weaker than that: KAN-42's style LoRA trains on KAN-32 (CC-BY storybook illustrations) and never
   reads our output, so **KAN-42's training is not blocked on this stage**. What is genuinely unowned
   is the runtime step between us — render the child's crop in storybook style, show it to the parent,
   keep the approved result. It needs the crop (ours) *and* the style (theirs), and each ticket lists
   it as the other's job. *Recommended:* the step belongs to KAN-42 or the KAN-64 flow, not here —
   recognition should not grow a diffusion dependency — and version one uses pretrained SD + a style
   prompt with no LoRA, regenerated once the LoRA exists. `Character.approve()` exists so the field is
   readable from the sheet without recognition filling it. *Decides:* KAN-38 and KAN-42 owners.
2. **Is a non-OSI licence acceptable for the product** (SAM 3's Meta licence, DINOv3's custom licence)?
   *Recommended:* yes for the course, but keep an Apache-2.0 fallback named per output so the swap is a
   config change. *Decides:* the team.
3. **Does "OpenPose reposing" in KAN-42 mean the OpenPose *format* or the OpenPose *codebase*?**
   *Recommended:* the format — ControlNet's OpenPose conditioning — so we can produce it with
   Apache-2.0 RTMPose/ViTPose. *Decides:* KAN-42 owner.
4. **Does the product need non-humanoid keypoints?** *Recommended:* no — KAN-42 already branches on
   `body_type`, with keypoints for humanoids only. *Decides:* KAN-42 owner.
5. **Rendered images cached, or rendered on the fly?** (KAN-87.) Affects whether augmentation is a
   training-time transform or a frozen derived set. *Decides:* KAN-87 owner with the vision teammate.

## Candidate slices

1. ~~**Extend the interface first.**~~ **Done** on `feat/KAN-38-character-sheet-contract`:
   `Character` now carries `candidates`, `body_type`, `crop_ref`, `mask_ref`, `keypoints`,
   `reference_image_ref` and `approved`; `top_labels` ranks within one character; the Fake emits a
   full sheet and `FakeDrawingRecognizer(approved=True)` lets KAN-42 exercise IP-Adapter today.
2. **Classifier baseline.** Frozen SigLIP 2 features + linear head on KAN-87's rendered Quick, Draw!.
   Produces the first real `drawing_top5_acc` and the number `CARD.md` must beat.
3. **Classifier fine-tune**, with KAN-31 augmentation, against the 0.85 gate.
4. **Description**, bake-off of 2–3 VLMs on ~50 real drawings, scored by hand.
5. **Crop + mask + keypoints**, starting from Meta's released drawing-specific weights; swap in
   SAM 3 / RTMPose where they fall over.
6. **`body_type`**, once 2 and 5 exist and we can see where the label mapping fails.

## Recommendation

Slice 1 is done, which matters more for the team than for this stage: KAN-42 was never blocked on our
*model*, only on those fields existing, and it can now build both the reposing branch and IP-Adapter
conditioning against `FakeDrawingRecognizer` with no weights anywhere.

Next, run slices 2 and 5 in parallel. They are the two cheapest ways to find out whether this stage is
hard — slice 2 gives the gated number, slice 5 tells us whether a crop and 17 joints can be had for free.
Both are answers rather than guesses.

Before starting: KAN-87 must land, since every classifier slice reads its rendered images. Question 1
does not block us, but it should be settled before KAN-42 gets far, and question 3 (OpenPose format vs.
codebase) decides whether the pose model can be Apache-2.0.
