"""Evaluate the drawing model. CLI: --model --split --out metrics.json.

Writes the shape in ``primer.evaluation.contract``: ``overall`` carries this stage's gated keys,
``baseline`` is what the model must beat and is never omitted, ``slices`` carries the per-slice
breakdown. ``primer.evaluation.golden`` merges these into the file the release gate reads.

**Preprocessing stability (KAN-109, slice S6c2)** lives here too, run as
``python -m primer.drawing.evaluate stability --version v1.1 --out stability.json``. It scores the
frozen Amateur Drawings sample in ``configs/stability_sample.txt`` under each variant in
``VARIANTS`` and reports, for each pair of adjacent variants, how often the top-1 label changes and
how much of the top 5 survives, and for each variant the top-1 confidence and the top-1/top-2
margin. **No labels and no accuracy**: the corpus is children's drawings of human figures and the
345 categories name no whole person, so there is nothing to be right about (KAN-111). What is
measured is how far the preprocessing moves the answer -- and the margins are what say whether a
moved answer was a preference perturbed or a tie broken differently.

**Rank statistic: top-5 set overlap** (the share of one variant's top five that is in the
other's). Chosen over Kendall's tau, which over 345 sigmoid scores is dominated by the order of
labels nobody will ever see, and over mean rank displacement, which is unbounded and dominated by
the same tail. The gated metric reads top-5 membership, so the overlap is in the gate's own units.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from primer.drawing.preprocess import PrepareConfig, default_prepare_config
from primer.evaluation import contract

#: The gated keys this stage owns, from the shared contract.
OWNED_METRICS = tuple(k for k, owner in contract.METRIC_OWNERS.items() if owner == "drawing")

SAMPLE_PATH = Path(__file__).parent / "configs" / "stability_sample.txt"

#: S6c2's variants, in the plan's table order: each differs from the one before it by exactly one
#: operation, so adjacent pairs are attributable. The crop is the dataset's own annotated box,
#: which the product does not have at upload time -- that row is an upper bound, not the product.
VARIANTS = ("as_published", "greyscale", "binarised", "cropped_binarised")
ADJACENT = tuple(itertools.pairwise(VARIANTS))


def variant_settings(base: PrepareConfig | None = None) -> dict[str, tuple[PrepareConfig, bool]]:
    """Each variant as ``(settings for prepare_drawing, whether it crops to the box)``."""
    base = base or default_prepare_config()
    plain = replace(base, greyscale=False, binarise=False)
    grey = replace(base, greyscale=True, binarise=False)
    binary = replace(base, greyscale=True, binarise=True)
    return {
        "as_published": (plain, False),
        "greyscale": (grey, False),
        "binarised": (binary, False),
        "cropped_binarised": (binary, True),
    }


@dataclass(frozen=True)
class Scored:
    """One drawing under one variant: label indices best first, and the two figures that matter."""

    ranking: tuple[int, ...]
    top1_confidence: float
    margin: float


def scored(confidences: Sequence[float]) -> Scored:
    """Rank ``confidences``. Ties keep vocabulary order, so a ranking is deterministic."""
    if len(confidences) < 2:
        raise ValueError("ranking needs at least two labels")
    ranking = tuple(sorted(range(len(confidences)), key=lambda i: (-confidences[i], i)))
    first, second = confidences[ranking[0]], confidences[ranking[1]]
    return Scored(ranking=ranking, top1_confidence=float(first), margin=float(first - second))


def pair_stats(before: Sequence[Scored], after: Sequence[Scored]) -> dict[str, float]:
    """How far moving from one variant to the next moves the answer, over the same drawings.

    ``top1_change_rate_confident`` restricts the change rate to drawings whose margin under
    ``before`` is at least its median: the half where the model had a preference to perturb.
    If that rate is as high as the overall one, the flips are tie-breaking noise, not effect.
    """
    if len(before) != len(after) or not before:
        raise ValueError("pair_stats needs the same, non-empty set of drawings under both variants")
    changed = [a.ranking[0] != b.ranking[0] for a, b in zip(before, after, strict=True)]
    overlap = [len(set(a.ranking[:5]) & set(b.ranking[:5])) / 5 for a, b in zip(before, after, strict=True)]
    median_margin = statistics.median(s.margin for s in before)
    confident = [flip for flip, s in zip(changed, before, strict=True) if s.margin >= median_margin]
    return {
        "top1_change_rate": sum(changed) / len(changed),
        "top1_change_rate_confident": sum(confident) / len(confident),
        "top5_overlap": statistics.fmean(overlap),
    }


def variant_stats(scores: Sequence[Scored]) -> dict[str, float]:
    """Confidence and margin under one variant, plus how concentrated its answers are.

    ``top1_label_share`` -- the share of drawings given the single most common label -- is there
    for S6d: a model collapsing onto one confident answer is perfectly stable, and this is the
    figure that shows it.
    """
    tops = Counter(s.ranking[0] for s in scores)
    return {
        "top1_confidence_mean": statistics.fmean(s.top1_confidence for s in scores),
        "top1_confidence_median": statistics.median(s.top1_confidence for s in scores),
        "margin_mean": statistics.fmean(s.margin for s in scores),
        "margin_median": statistics.median(s.margin for s in scores),
        "distinct_top1": float(len(tops)),
        "top1_label_share": tops.most_common(1)[0][1] / len(scores),
        "top1_label_index": float(tops.most_common(1)[0][0]),
    }


class VariantEncoder(Protocol):
    def encode(
        self, image: bytes, prompts: Sequence[str], *, prepare: PrepareConfig | None = ..., box: object = ...
    ) -> Any: ...


def measure_stability(
    drawings: Iterable[tuple[bytes, Sequence[float]]],
    encoder: VariantEncoder,
    prompts: Sequence[str],
    base: PrepareConfig | None = None,
) -> dict[str, Any]:
    """Score every drawing under every variant and summarise. ``drawings`` yields ``(png, box)``."""
    settings = variant_settings(base)
    scores: dict[str, list[Scored]] = {name: [] for name in VARIANTS}
    for image, box in drawings:
        for name, (prepare, crops) in settings.items():
            encoding = encoder.encode(image, prompts, prepare=prepare, box=box if crops else None)
            scores[name].append(scored(encoding.confidences))
    return {
        "drawings": len(scores[VARIANTS[0]]),
        "variants": {name: variant_stats(scores[name]) for name in VARIANTS},
        "pairs": {f"{a}->{b}": pair_stats(scores[a], scores[b]) for a, b in ADJACENT},
        "rank_statistic": "top-5 set overlap",
    }


def load_sample(path: Path = SAMPLE_PATH) -> list[str]:
    """The frozen sample's record ids, in file order. Comment lines carry its provenance."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [line for line in lines if line and not line.startswith("#")]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name} lists a record twice")
    return ids


def _stability(argv: list[str]) -> int:
    """The CLI: read the sample's drawings from a frozen version and write the summary."""
    from primer.data import catalog
    from primer.data.storage import Store
    from primer.drawing.model import DrawingConfig, SiglipEncoder

    parser = argparse.ArgumentParser(prog="evaluate stability")
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    store = Store()
    frame = catalog.read_dataset("character_drawings", store, ["amateur_drawings"], version=args.version)
    ids = load_sample()
    rows = frame.set_index("record_id").loc[ids]
    prefix = f"{store.uri}/"

    def drawings() -> Iterable[tuple[bytes, Sequence[float]]]:
        for row in rows.itertuples():
            if not row.image_uri.startswith(prefix):
                raise ValueError(f"{row.image_uri} is not in {store.uri}; point PRIMER_DATA_URI at it")
            relative = row.image_uri.removeprefix(prefix)
            yield store.read_bytes(relative), (row.bbox_x, row.bbox_y, row.bbox_width, row.bbox_height)

    config = DrawingConfig.from_yaml()
    start = time.perf_counter()
    summary = measure_stability(drawings(), SiglipEncoder(config), config.prompts())
    labels = config.labels()
    for stats in summary["variants"].values():
        # The name beside the index: "53% of drawings got label 207" says nothing to a reader.
        stats["top1_label"] = labels[int(stats["top1_label_index"])]
    summary.update(
        {
            "data_version": args.version,
            "sample": SAMPLE_PATH.name,
            "sampler": "uniform random draw (unstratified), seed 20261115 -- S6b3's sampler is not on main",
            "license": sorted(set(rows["license"])),
            "attribution": sorted(set(rows["attribution"])),
            "encoder": config.encoder,
            "encoder_revision": config.encoder_revision,
            "seconds": round(time.perf_counter() - start, 1),
        }
    )
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["stability"]:
        return _stability(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.parse_args(argv)
    raise NotImplementedError("drawing/evaluate.py is a stub")


if __name__ == "__main__":
    sys.exit(main())
