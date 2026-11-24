"""Sensitivity analysis for the drawing stage (KAN-38, slice S6f).

    python -m primer.drawing.sensitivity --version v1.1 --split val --sample 2000 --out sens.json [--mlflow --owner NAME]

How far the recogniser's answer moves when the **drawing** changes, and how far it moves when the
**inference-time setting** changes. These are the inputs `docs/engineering/model-development.md`
names for this stage -- stroke count, line thickness, partial erasure -- plus the prompt template,
which is the one hyperparameter a zero-shot recogniser has. **Training hyperparameters are S6d's**:
nothing here trains, so none of this answers the guidelines' hyperparameter-sensitivity item.

Each perturbation is compared with the unperturbed baseline **on the same drawings** (a paired
comparison), and the change in top-5 accuracy carries a paired bootstrap interval, so a 2-point
move over 2,000 drawings is not read as an effect when it is noise. ``pair_stats`` from S6c2 adds
how often top-1 changes and how much of the top 5 survives.

**Perturbations are applied on evaluation splits on purpose.** ``data.py`` refuses augmentation on
``val`` and ``test`` because an augmented evaluation measures the augmentation; here measuring it is
the point, and the baseline beside every number is the unaugmented drawing. Erasure goes through
``augment_canvas`` with every other operation off, rather than a second eraser written here.

**Runs on ``val`` by default.** The prompt sweep compares templates, and acting on a comparison
taken on ``test`` would be tuning on the test split.

**Occlusion** is the attribution step (the guidelines' SHAP/LIME, for an image): patches of the
canvas are painted paper-white -- an in-distribution edit, it erases that region -- and the drop in
the true label's confidence is recorded. A white patch over blank paper changes nothing, so only
patches holding ink are scored, and the report is **how concentrated** the drop is, not whether ink
matters more than paper (which would be true by construction). Only drawings the baseline already
places in its top five are occluded: a drop in a confidence that was near zero is noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from primer.drawing.evaluate import (
    Scored,
    _git_state,
    add_mlflow_arguments,
    check_mlflow_arguments,
    log_to_mlflow,
    pair_stats,
    scored,
)
from primer.drawing.preprocess import AugmentConfig, augment_canvas, default_augment_config
from primer.drawing.render import RenderConfig, default_config, render_strokes

#: The gated metric reads five guesses; the hit rate here is the same quantity.
TOP_K = 5

#: Stroke counts dropped from the end. Quick, Draw! records strokes in drawing order, so the last
#: ones are usually detail added to an outline. By count, not by fraction: on a three-stroke
#: drawing 25% and 50% would both drop one stroke and report the same thing twice.
DROP_LAST = (1, 2)

#: Stroke weights other than the shipped 3px. Odd, as `RenderConfig` requires, and within the
#: 8px padding. 1px is the near-invisible sketch the render config warns about.
LINE_WIDTHS = (1, 5, 7)

#: Erasure removing less ink than this is reported but not counted as an erasure: `augment_canvas`
#: places its rectangles at random, and many land on blank paper.
ERASURE_FLOOR = 0.05

#: Prompt templates compared with the shipped one, which is the baseline and is skipped here:
#: "{label}" alone, a sketch-worded one, the model card's, and the one shipped since 2026-11-16.
PROMPT_TEMPLATES = (
    "{label}",
    "a sketch of a {label}.",
    "This is a photo of {label}.",
    "a child's drawing of a {label}.",
)

#: Occlusion patch, in pixels of the 224px canvas: a 7 x 7 grid.
PATCH = 32

#: Confidences for every vocabulary prompt, in prompt order, for one canvas.
Scorer = Callable[[np.ndarray, Sequence[str]], Sequence[float]]


@dataclass(frozen=True)
class Drawing:
    """One drawing to perturb: its strokes, and its label's index in the vocabulary."""

    record_id: str
    strokes: Sequence[Any]
    label: int


def erasure_only(base: AugmentConfig | None = None) -> AugmentConfig:
    """``augment_canvas`` with erasure always on and every other operation off, on white paper."""
    base = base or default_augment_config()
    white = (base.render.background,) * 3
    return replace(
        base,
        wobble_probability=0.0,
        erasure_probability=1.0,
        ink_colour_probability=0.0,
        fill_probability=0.0,
        lighting_probability=0.0,
        paper_tints=(white,),
        paper_texture=0.0,
        paper_grain=0.0,
    )


def ink_mass(canvas: np.ndarray, render: RenderConfig) -> float:
    """How much ink a greyscale canvas holds: the summed darkness below the paper level."""
    return float(np.clip(render.background - canvas.astype(np.float64), 0, None).sum())


def erased(canvas: np.ndarray, seed: int, config: AugmentConfig) -> tuple[np.ndarray, float]:
    """``canvas`` with some ink erased, and the fraction of its ink mass that went."""
    rgb = augment_canvas(canvas, seed, config)
    grey = rgb[:, :, 0]
    if not (np.array_equal(grey, rgb[:, :, 1]) and np.array_equal(grey, rgb[:, :, 2])):
        raise ValueError("erasure-only augmentation produced colour; another operation is still on")
    before = ink_mass(canvas, config.render)
    removed = 0.0 if before == 0 else 1.0 - ink_mass(grey, config.render) / before
    return grey, removed


def hit(scores: Scored, label: int) -> bool:
    return label in scores.ranking[:TOP_K]


def paired_delta(
    before: Sequence[bool], after: Sequence[bool], seed: int, resamples: int = 1000
) -> dict[str, float]:
    """Top-5 accuracy after minus before, over the same drawings, with a 95% bootstrap interval."""
    if len(before) != len(after) or not before:
        raise ValueError("a paired delta needs the same, non-empty set of drawings on both sides")
    diff = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "top5_acc_before": round(float(np.mean(before)), 4),
        "top5_acc_after": round(float(np.mean(after)), 4),
        "top5_delta": round(float(diff.mean()), 4),
        "top5_delta_ci_low": round(float(low), 4),
        "top5_delta_ci_high": round(float(high), 4),
        "drawings": float(len(diff)),
    }


def _compare(
    base: Sequence[Scored], after: Mapping[int, Scored], labels: Sequence[int], seed: int
) -> dict[str, float]:
    """Baseline against a perturbation over the drawings the perturbation applied to."""
    kept = sorted(after)
    if not kept:
        return {"drawings": 0.0}
    before_scores = [base[i] for i in kept]
    after_scores = [after[i] for i in kept]
    return {
        **paired_delta(
            [hit(s, labels[i]) for s, i in zip(before_scores, kept, strict=True)],
            [hit(s, labels[i]) for s, i in zip(after_scores, kept, strict=True)],
            seed,
        ),
        **{key: round(value, 4) for key, value in pair_stats(before_scores, after_scores).items()},
    }


def occlusion(
    canvas: np.ndarray, label: int, scorer: Scorer, prompts: Sequence[str], render: RenderConfig
) -> dict[str, Any]:
    """The true label's confidence drop with each inked ``PATCH``-square painted paper-white.

    ``grid`` holds the drop per patch, row by row, and None for a patch with no ink -- whose drop
    is 0 by construction and so says nothing. ``top3_share`` is the share of the total positive
    drop (negative drops are left out) that the three largest patches carry: near 1 means the
    answer hangs on one region. A drawing with three or fewer inked patches scores 1.0 trivially,
    so read it beside ``inked_patches``. It is None when no patch lowers the confidence: the share
    is undefined there, and a 0.0 would drag the median down.
    """
    start = float(scorer(canvas, prompts)[label])
    rows = range(0, canvas.shape[0], PATCH)
    columns = range(0, canvas.shape[1], PATCH)
    grid: list[list[float | None]] = []
    drops: list[float] = []
    for top in rows:
        row: list[float | None] = []
        for left in columns:
            patch = canvas[top : top + PATCH, left : left + PATCH]
            if not (patch < render.background).any():
                row.append(None)
                continue
            covered = canvas.copy()
            covered[top : top + PATCH, left : left + PATCH] = render.background
            drop = start - float(scorer(covered, prompts)[label])
            row.append(round(drop, 4))
            drops.append(drop)
        grid.append(row)
    positive = sorted((d for d in drops if d > 0), reverse=True)
    total = sum(positive)
    return {
        "confidence": round(start, 4),
        "inked_patches": len(drops),
        "max_drop": round(max(drops), 4) if drops else None,
        "top3_share": round(sum(positive[:3]) / total, 4) if total > 0 else None,
        "grid": grid,
    }


def measure(
    drawings: Sequence[Drawing],
    scorer: Scorer,
    prompts: Sequence[str],
    *,
    prompt_sets: Mapping[str, Sequence[str]],
    seed: int = 0,
    occlude: int = 20,
    render: RenderConfig | None = None,
    augmentation: AugmentConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Score every drawing unperturbed and under every perturbation, and summarise.

    Loops perturbation-outermost, so each perturbation scores against one prompt set and the
    encoder's text cache is not thrown away between drawings.
    """
    if not drawings:
        raise ValueError("no drawings to measure")
    render = render or default_config()
    augmentation = augmentation or erasure_only()
    labels = [d.label for d in drawings]
    say = progress or (lambda _: None)

    canvases = [render_strokes(d.strokes, render) for d in drawings]
    base = [scored(scorer(c, prompts)) for c in canvases]
    say("baseline")
    results: dict[str, Any] = {}

    for k in DROP_LAST:
        after = {
            i: scored(scorer(render_strokes(d.strokes[:-k], render), prompts))
            for i, d in enumerate(drawings)
            if len(d.strokes) > k
        }
        results[f"drop_last_{k}"] = _compare(base, after, labels, seed)
        say(f"drop_last_{k}")

    for width in LINE_WIDTHS:
        config = replace(render, line_width=width)
        after = {
            i: scored(scorer(render_strokes(d.strokes, config), prompts)) for i, d in enumerate(drawings)
        }
        results[f"line_width_{width}"] = _compare(base, after, labels, seed)
        say(f"line_width_{width}")

    removed: list[float] = []
    erased_scores: dict[int, Scored] = {}
    for i, canvas in enumerate(canvases):
        image, fraction = erased(canvas, seed + i, augmentation)
        removed.append(fraction)
        if fraction >= ERASURE_FLOOR:
            erased_scores[i] = scored(scorer(image, prompts))
    results["erasure"] = {
        **_compare(base, erased_scores, labels, seed),
        "ink_removed_median": round(statistics.median(removed), 4),
        "below_floor": float(sum(1 for f in removed if f < ERASURE_FLOOR)),
        "floor": ERASURE_FLOOR,
    }
    say("erasure")

    for name, alternative in prompt_sets.items():
        after = {i: scored(scorer(c, alternative)) for i, c in enumerate(canvases)}
        results[f"prompt:{name}"] = _compare(base, after, labels, seed)
        say(f"prompt:{name}")

    correct = [i for i, s in enumerate(base) if hit(s, labels[i])][:occlude]
    maps = {
        drawings[i].record_id: occlusion(canvases[i], labels[i], scorer, prompts, render) for i in correct
    }
    shares = [m["top3_share"] for m in maps.values() if m["top3_share"] is not None]
    max_drops = [m["max_drop"] for m in maps.values() if m["max_drop"] is not None]
    say("occlusion")

    return {
        "drawings": len(drawings),
        "baseline_top5_acc": round(sum(hit(s, labels[i]) for i, s in enumerate(base)) / len(base), 4),
        "perturbations": results,
        "occlusion": {
            "drawings": len(maps),
            "patch": PATCH,
            "top3_share_median": round(statistics.median(shares), 4) if shares else None,
            "top3_share_undefined": len(maps) - len(shares),
            "inked_patches_median": statistics.median(m["inked_patches"] for m in maps.values())
            if maps
            else None,
            "max_drop_median": round(statistics.median(max_drops), 4) if max_drops else None,
            "maps": maps,
        },
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sensitivity", description="S6f: how far the drawing recogniser's answer moves"
    )
    p.add_argument("--version", required=True, help="the frozen primer-data version, e.g. v1.1")
    p.add_argument(
        "--split", default="val", choices=("val", "test"), help="val by default: see the module docstring"
    )
    p.add_argument("--sample", type=int, default=2000, help="drawings drawn at random from the split")
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--occlude", type=int, default=20, help="correctly ranked drawings to occlude")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--out", required=True, type=Path)
    # The shared --mlflow and --owner (S6f2): the same refusals and tags as S6c's and S6c2's runs.
    add_mlflow_arguments(p)
    return p


def main(argv: list[str] | None = None) -> int:
    from primer.data import catalog
    from primer.data.storage import Store
    from primer.drawing.data import check_frame, label_index, split_frames
    from primer.drawing.model import DrawingConfig, SiglipEncoder
    from primer.drawing.render import to_png

    cli = parser()
    args = cli.parse_args(argv)
    if not args.out.resolve().parent.is_dir():
        cli.error(f"--out's folder does not exist: {args.out.parent}")
    if args.sample < 1 or args.occlude < 0:
        cli.error("--sample must be at least 1 and --occlude at least 0")
    # Before the bucket or the GPU: a missing owner or tracking server used to fail only after
    # minutes of scoring, when the run tried to log.
    check_mlflow_arguments(cli, args)

    config = DrawingConfig.from_yaml()
    labels = config.labels()
    index = label_index(labels)
    frame = catalog.read_dataset("drawings", store=Store(), sources=["quickdraw"], version=args.version)
    if frame.empty:
        raise ValueError(f"version {args.version!r} holds no Quick, Draw! drawings")
    check_frame(frame, labels)
    part = split_frames(frame)[args.split]
    # A seeded uniform draw from one split of Quick, Draw!. S6b3's samplers stratify Amateur
    # Drawings, not this corpus, so they do not apply.
    sample = part.sample(n=min(args.sample, len(part)), random_state=args.seed)
    drawings = [
        Drawing(str(row.record_id), json.loads(row.strokes), index[row.label.casefold()])
        for row in sample.itertuples()
    ]

    encoder = SiglipEncoder(config, device=args.device)

    def scorer(canvas: np.ndarray, prompts: Sequence[str]) -> Sequence[float]:
        # PNG bytes, as an upload: the encoder decodes and runs `prepare_drawing` itself.
        return encoder.encode(to_png(canvas), prompts).confidences

    prompt_sets = {
        t: tuple(t.format(label=label) for label in labels)
        for t in PROMPT_TEMPLATES
        if t != config.prompt_template
    }
    prompts = config.prompts()
    git_sha, git_dirty = _git_state()
    start = time.perf_counter()
    result = measure(
        drawings,
        scorer,
        prompts,
        prompt_sets=prompt_sets,
        seed=args.seed,
        occlude=args.occlude,
        progress=lambda step: print(
            f"{step}: {time.perf_counter() - start:.0f}s", file=sys.stderr, flush=True
        ),
    )
    result.update(
        {
            "data_version": args.version,
            "split": args.split,
            "seed": args.seed,
            "sampler": "uniform random draw from the split (pandas .sample, seeded)",
            "shipped_prompt_template": config.prompt_template,
            "encoder": config.encoder,
            "encoder_revision": config.encoder_revision,
            "device": str(encoder.device),
            "dtype": encoder.dtype_name or "unknown",
            "prompts_sha256": hashlib.sha256("\n".join(prompts).encode()).hexdigest(),
            "render": asdict(default_config()),
            "license": sorted(set(sample["license"])),
            "attribution": sorted(set(sample["attribution"])),
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "seconds": round(time.perf_counter() - start, 1),
        }
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out}: baseline top-5 {result['baseline_top5_acc']} over {result['drawings']} drawings"
    )
    for line in iter_summary(result):
        print(line)
    if args.mlflow:
        log_to_mlflow("sensitivity", result, args.out, args.owner, "s6f-sensitivity")
    return 0


def iter_summary(result: Mapping[str, Any]) -> Iterable[str]:
    """One line per perturbation, for a person reading the run in a terminal."""
    for name, figures in result["perturbations"].items():
        if figures.get("drawings"):
            yield (
                f"{name:28s} n={int(figures['drawings']):5d}  top-5 {figures['top5_acc_before']:.3f} -> "
                f"{figures['top5_acc_after']:.3f}  delta {figures['top5_delta']:+.3f} "
                f"[{figures['top5_delta_ci_low']:+.3f}, {figures['top5_delta_ci_high']:+.3f}]"
            )


if __name__ == "__main__":
    sys.exit(main())
