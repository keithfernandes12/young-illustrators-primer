"""Evaluate the drawing model.

**Top-5 accuracy (KAN-38, slice S6c)**:
``python -m primer.drawing.evaluate --model src/primer/drawing/configs/drawing_baseline.yaml
--version v1.1 --split test --out drawing_metrics.json``, then
``check_gate.py --metrics drawing_metrics.json --only drawing``. Every drawing of S6b2's split is
rendered by ``render_strokes`` and scored through the encoder as an upload would be.

Writes the shape in ``primer.evaluation.contract``: ``overall`` carries this stage's gated keys,
``baseline`` is what the model must beat and is never omitted -- chance, 5 of 345 -- and ``slices``
carries the per-country breakdown. The gated key is repeated at the top level so the release gate
can read this file directly; see ``metrics_file`` for why it is not merged into ``golden.py``'s.

**This is top-5 on rendered Quick, Draw!**, the gate's number and not the product's: every drawing
in v1.1 was already recognised by Quick, Draw!'s own model, and nothing here is a child's upload.

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
import hashlib
import itertools
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
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
    if not all(0.0 <= value <= 1.0 for value in confidences):  # False for NaN too
        raise ValueError("confidences are sigmoid scores in 0-1; got NaN or a value outside it")
    ranking = tuple(sorted(range(len(confidences)), key=lambda i: (-confidences[i], i)))
    first, second = confidences[ranking[0]], confidences[ranking[1]]
    return Scored(ranking=ranking, top1_confidence=float(first), margin=float(first - second))


def pair_stats(before: Sequence[Scored], after: Sequence[Scored]) -> dict[str, float]:
    """How far moving from one variant to the next moves the answer, over the same drawings.

    ``top1_change_rate_confident`` restricts the change rate to drawings whose margin under
    ``before`` is at least its median: the half where the model had a preference to perturb.
    Flips that were only near-ties broken the other way would pile up among the *low*-margin
    drawings, so this rate would then sit well below the overall one. A rate close to the overall
    one says the flips are not concentrated there: confident answers moved too, so the movement
    is not only tie-breaking.

    ``confident_drawings`` is the size of that subset. It is half the drawings unless the median
    margin is 0 -- a saturated or collapsed model, most scores exactly 1.0 -- when ``>= median``
    admits every drawing and the confident rate is the overall rate by construction, so read the
    rate with the count beside it.
    """
    if len(before) != len(after) or not before:
        raise ValueError("pair_stats needs the same, non-empty set of drawings under both variants")
    changed = [a.ranking[0] != b.ranking[0] for a, b in zip(before, after, strict=True)]
    # By the entries a ranking has, not by five: with fewer than five labels an unchanged ranking
    # shares all of its top-k and must score 1.0, not 3/5.
    top = min(5, len(before[0].ranking))
    overlap = [
        len(set(a.ranking[:top]) & set(b.ranking[:top])) / top for a, b in zip(before, after, strict=True)
    ]
    median_margin = statistics.median(s.margin for s in before)
    confident = [flip for flip, s in zip(changed, before, strict=True) if s.margin >= median_margin]
    return {
        "top1_change_rate": sum(changed) / len(changed),
        "top1_change_rate_confident": sum(confident) / len(confident),
        "confident_drawings": float(len(confident)),
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


def describe_run(config: Any, device: str, dtype: str, base: PrepareConfig | None = None) -> dict[str, Any]:
    """What a stability number was taken under, so two runs can be told apart or reproduced.

    A margin is a difference of two confidences, and confidences move about 20% between fp32 on a
    CPU and bf16 on a GPU (see `model.CONFIDENCE_DECIMALS`), so the device and dtype are part of
    the result. So are the prompts -- the template, how many, and a hash of the text itself -- and
    every preprocessing setting each variant used. ``config_dir`` is left out: it is a path on one
    person's machine and says nothing about the score.
    """
    return {
        **_scored_under(config, device, dtype),
        "preprocessing": {
            name: {**asdict(prepare), "crops_to_box": crops}
            for name, (prepare, crops) in variant_settings(base).items()
        },
    }


def _scored_under(config: Any, device: str, dtype: str) -> dict[str, Any]:
    """The device, dtype, scoring settings and prompts a number was taken under."""
    prompts = config.prompts()
    scoring = {key: value for key, value in asdict(config).items() if key != "config_dir"}
    return {
        "device": device,
        "dtype": dtype,
        "scoring": scoring,
        "prompt_count": len(prompts),
        "prompts_sha256": hashlib.sha256("\n".join(prompts).encode()).hexdigest(),
    }


def add_mlflow_arguments(parser: argparse.ArgumentParser) -> None:
    """``--mlflow`` and ``--owner``, the same on every drawing evaluation (S6c3)."""
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="after writing --out, log a copy of it to MLflow (experiment yip-drawing, server "
        "MLFLOW_TRACKING_URI). The file stays the record; the run is a copy",
    )
    parser.add_argument("--owner", default="", help="who ran it; required with --mlflow")


def check_mlflow_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse a bad --mlflow request before minutes of scoring, not after."""
    if not args.mlflow:
        return
    if not args.owner.strip():
        parser.error("--owner is required with --mlflow: runs from six machines are told apart by it")
    if getattr(args, "limit", None) is not None:
        # A first-N probe is label-ordered and "never a result" (S6c); a run on the shared server
        # would sit beside real ones with a drawing_top5_acc metric of its own.
        parser.error("--mlflow does not log a --limit probe: it is a timing check, never a result")


def log_to_mlflow(kind: str, result: dict[str, Any], out: Path, owner: str, run_name: str) -> str:
    """Log ``out`` and print the run id. Imported lazily: the package is only needed here."""
    from primer.drawing.tracking import log_run

    # The 3.x client prints an emoji when a run ends. On a cp1252 Windows console that raises
    # inside `end_run` and leaves the run RUNNING on the server; replace what cannot be printed.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")
    run_id = log_run(kind=kind, result=result, metrics_path=out, owner=owner, run_name=run_name)
    print(f"logged MLflow run {run_id}")
    return run_id


def stability_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluate stability")
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="default: cuda if available, else cpu. Pass cpu to reproduce the recorded CPU baseline "
        "(fp32) on a machine with a GPU, whose bf16 moves the thin margins",
    )
    add_mlflow_arguments(parser)
    return parser


def _stability(argv: list[str]) -> int:
    """The CLI: read the sample's drawings from a frozen version and write the summary."""
    from primer.data import catalog
    from primer.data.storage import Store
    from primer.drawing.model import DrawingConfig, SiglipEncoder

    parser = stability_parser()
    args = parser.parse_args(argv)
    check_mlflow_arguments(parser, args)
    out = Path(args.out)
    if not out.resolve().parent.is_dir():
        parser.error(f"--out's folder does not exist: {out.parent}; the run takes minutes, so say so now")

    store = Store()
    frame = catalog.read_dataset("character_drawings", store, ["amateur_drawings"], version=args.version)
    ids = load_sample()
    rows = frame.set_index("record_id").loc[ids]
    if len(rows) != len(ids):
        # `.loc` returns every row of a repeated record_id, which would score a drawing twice.
        raise ValueError(f"the sample's {len(ids)} ids matched {len(rows)} rows in {args.version}")
    prefix = f"{store.uri}/"

    def drawings() -> Iterable[tuple[bytes, Sequence[float]]]:
        for row in rows.itertuples():
            if not row.image_uri.startswith(prefix):
                raise ValueError(f"{row.image_uri} is not in {store.uri}; point PRIMER_DATA_URI at it")
            relative = row.image_uri.removeprefix(prefix)
            data = store.read_bytes(relative)
            if hashlib.sha256(data).hexdigest() != row.image_sha256:
                raise ValueError(f"{row.Index}: the image bytes do not match the recorded image_sha256")
            yield data, (row.bbox_x, row.bbox_y, row.bbox_width, row.bbox_height)

    config = DrawingConfig.from_yaml()
    # Read before the run, as S6c does: an edit meanwhile should not be recorded as the code.
    git_sha, git_dirty = _git_state()
    start = time.perf_counter()
    encoder = SiglipEncoder(config, device=args.device)
    summary = measure_stability(drawings(), encoder, config.prompts())
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
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            **describe_run(config, str(encoder.device), encoder.dtype_name or "unknown"),
        }
    )
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.mlflow:
        log_to_mlflow("stability", summary, out, args.owner, "s6c2-preprocessing-stability")
    return 0


# ------------------------------------------------------------------------------- accuracy (S6c)

#: The gated metric reads the top five guesses. A literal, not ``DrawingConfig.top_k``: that field
#: is how many candidates a `Character` carries, and widening it must not widen the metric.
TOP_K = 5
STAGE = "drawing"


class ScoringEncoder(Protocol):
    def encode(self, image: bytes, prompts: Sequence[str]) -> Any: ...


def top5_hits(
    drawings: Iterable[tuple[bytes, int]],
    encoder: ScoringEncoder,
    prompts: Sequence[str],
    progress: Any = None,
) -> list[bool]:
    """For each ``(png, label index)``, whether the label is among the encoder's top five.

    Compared by vocabulary **index**, never by string: the data spells three labels in lower case
    where the vocabulary capitalises them, and a string comparison would score those as misses.
    Ties keep vocabulary order (``scored``), so the same scores give the same hits on every run.
    """
    hits: list[bool] = []
    for image, label in drawings:
        ranking = scored(encoder.encode(image, prompts).confidences).ranking
        hits.append(label in ranking[:TOP_K])
        if progress is not None:
            progress(len(hits))
    if not hits:
        raise ValueError("no drawings to evaluate")
    return hits


def chance_baseline(vocabulary_size: int) -> dict[str, Any]:
    """Top-5 accuracy of guessing five labels at random: what any recogniser must beat.

    Not Quick, Draw!'s own recogniser, which would be the natural rival: every v1.1 drawing has
    ``recognized=True``, so its rate on this set is 1.0 by selection, not by skill.
    """
    if vocabulary_size < 1:
        raise ValueError("the vocabulary is empty")
    return {
        "drawing_top5_acc": round(min(TOP_K, vocabulary_size) / vocabulary_size, 4),
        "kind": f"chance: {TOP_K} labels of {vocabulary_size} guessed uniformly at random",
    }


def _accuracy(hits: Sequence[bool]) -> float:
    # Unrounded: the gate compares this to its threshold, and rounding 0.8499605 to 0.85 would pass
    # a run that misses. Only the printed summary rounds.
    return sum(hits) / len(hits)


def accuracy_by_slice(hits: Sequence[bool], groups: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    """``drawing_top5_acc`` per value of each slice key. ``groups[key][i]`` is drawing *i*'s value.

    Each slice carries ``judged``: False below ``SMALL_SLICE`` drawings, so it is reported, not
    judged -- 165 countries over 63,310 drawings leaves most of them small.
    """
    from primer.drawing.slices import SMALL_SLICE

    slices: list[dict[str, Any]] = []
    for key, values in groups.items():
        if len(values) != len(hits):
            raise ValueError(f"slice key {key!r} has {len(values)} values for {len(hits)} drawings")
        members: dict[str, list[bool]] = {}
        for value, hit in zip(values, hits, strict=True):
            members.setdefault(value, []).append(hit)
        slices.extend(
            {
                "key": key,
                "value": value,
                "n": len(group),
                "judged": len(group) >= SMALL_SLICE,
                "metrics": {"drawing_top5_acc": _accuracy(group)},
            }
            for value, group in sorted(members.items())
        )
    return slices


def metrics_file(
    hits: Sequence[bool],
    groups: Mapping[str, Sequence[str]],
    vocabulary_size: int,
    *,
    gated: bool,
    **provenance: Any,
) -> dict[str, Any]:
    """The ``primer.evaluation.contract`` shape, plus the gated key at the top level.

    ``check_gate.py`` looks keys up at the top level, so ``overall`` alone would read as ``metric
    missing``; the key is repeated there so this file can be gated directly with ``--only
    drawing``. There is deliberately no ``fake_models`` key: the gate reads that flag for the
    whole file, which is why a real number is not merged into ``golden.py``'s fake-scored one.

    With ``gated=False`` the top-level key is left out, so the gate says ``metric missing`` rather
    than passing or failing -- and never certifies an MLflow run. That is the case for a ``--limit``
    probe (a few hundred drawings that are not even a random sample of the split) and for the
    ``val`` split, which is used to choose the model and so is diagnostic only: the gate's
    population is the complete ``test`` split.
    """
    overall = {"drawing_top5_acc": _accuracy(hits), "drawings": len(hits)}
    top_level = {"drawing_top5_acc": overall["drawing_top5_acc"]} if gated else {}
    reserved = {"stage", "drawing_top5_acc", "gate_eligible", "overall", "baseline", "slices", "fake_models"}
    if clash := reserved & provenance.keys():
        # Not just ordering: with `gated=False` the top-level key is absent, so a provenance key of
        # that name would survive and put a gate-readable number in an ineligible file.
        raise ValueError(f"provenance cannot set {sorted(clash)}: the writer owns those keys")
    return {
        **provenance,
        "stage": STAGE,
        **top_level,
        # The one place eligibility is stated outright; the missing top-level key is how the gate
        # sees it, but a script reading `overall` would otherwise take a probe for a result.
        "gate_eligible": gated,
        "overall": overall,
        "baseline": chance_baseline(vocabulary_size),
        "slices": accuracy_by_slice(hits, groups),
    }


def _git_state() -> tuple[str, bool | None]:
    """This package's commit and whether its tree has uncommitted changes (None: unknown).

    Run from the package's own folder, not the caller's working directory, so a run started from
    another repository does not record that repository's commit.
    """
    import subprocess

    here = Path(__file__).resolve().parent

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], capture_output=True, text=True, check=False, timeout=10, cwd=here
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    sha = git("rev-parse", "--short", "HEAD")
    # Scoped to this package (cwd): the run's own output files elsewhere in the repo must not read
    # as an uncommitted change to the code.
    status = git("status", "--porcelain", "--", ".")
    return sha or "unknown", (None if status is None else bool(status))


def _set_previous_output_aside(parser: argparse.ArgumentParser, out: Path) -> None:
    """Move the last run's file to ``<out>.previous`` (or the first free ``.previous.N``) so a run
    that dies cannot leave it at ``out``.

    ``check_gate`` reads only ``out``, so a run that fails, or is refused after this point, leaves
    nothing there to be certified. The old file is kept, not deleted: a 12-minute full-split result
    is not something a ``--limit`` probe reusing the path should destroy. Only a previous drawing
    metrics file is moved; anything else (``--out <the model yaml>``) is refused untouched.
    """
    if not out.exists():
        return
    try:
        previous = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = None
    if not isinstance(previous, dict) or previous.get("stage") != STAGE:
        parser.error(f"--out {out} exists and is not a previous drawing metrics file; not overwriting it")
    # A name that cannot collide: `replace` overwrites silently, so a second reuse of --out would
    # destroy the full-split result the first one had set aside.
    aside = out.with_name(out.name + ".previous")
    for attempt in itertools.count(1):
        if not aside.exists():
            break
        aside = out.with_name(f"{out.name}.previous.{attempt}")
    out.replace(aside)


def accuracy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate", description="drawing_top5_acc over a rendered Quick, Draw! split (S6c)"
    )
    parser.add_argument(
        "--model", required=True, type=Path, help="a DrawingConfig yaml, e.g. configs/drawing_baseline.yaml"
    )
    parser.add_argument("--version", required=True, help="the frozen primer-data version, e.g. v1.1")
    parser.add_argument(
        "--split",
        default="test",
        choices=("val", "test"),
        help="never train: a number over the drawings the model learns from measures memory",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="the metrics file. An existing drawing metrics file here is moved to <out>.previous "
        "first, so a failed run leaves nothing at this path for check_gate to read",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="default: cuda if available, else cpu. Recorded in the output: bf16 on cuda and fp32 on "
        "cpu rank near-ties differently, so two numbers compare only on the same device and dtype",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score only the first N drawings (a timing probe, never a result)",
    )
    add_mlflow_arguments(parser)
    return parser


def _evaluate_accuracy(argv: list[str]) -> int:
    """The CLI: every drawing of a split, rendered and scored the way an upload is."""
    from primer.data import catalog
    from primer.data.storage import Store
    from primer.drawing.data import check_frame, label_index, split_frames
    from primer.drawing.model import DrawingConfig, SiglipEncoder
    from primer.drawing.render import render_strokes_png
    from primer.drawing.slices import SLICE_KEYS, slice_values

    parser = accuracy_parser()
    args = parser.parse_args(argv)
    if not args.out.resolve().parent.is_dir():
        parser.error(
            f"--out's folder does not exist: {args.out.parent}; the run takes minutes, so say so now"
        )
    # First thing after the arguments parse, before any other refusal or load: a run that exits
    # early must not leave the previous file at --out for the gate to certify. (An argument that
    # argparse itself rejects exits before this, with a non-zero status.)
    _set_previous_output_aside(parser, args.out)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    check_mlflow_arguments(parser, args)

    if not args.version.strip():
        # `required=True` only demands the flag: `--version ""` would read the unfrozen current
        # data (the catalog treats "" as no version) and still write a gate-eligible file.
        parser.error("--version must name a frozen primer-data version, e.g. v1.1")

    config = DrawingConfig.from_yaml(args.model)
    labels = config.labels()
    prompts = config.prompts()
    if len(set(prompts)) != len(prompts):
        # The same refusal `DrawingRecognizerModel` makes. A `prompt_template` without `{label}`
        # gives every label one prompt, so top-five hits would follow vocabulary order, not the
        # drawing. Checked before the bucket or the weights are touched.
        parser.error(
            f"prompt_template {config.prompt_template!r} does not give every label a distinct "
            "prompt; a template without {label} makes every score identical"
        )
    index = label_index(labels)

    store = Store()
    frame = catalog.read_dataset("drawings", store=store, sources=["quickdraw"], version=args.version)
    if frame.empty:
        raise ValueError(f"version {args.version!r} holds no Quick, Draw! drawings")
    check_frame(frame, labels)
    # The split's membership function itself, so these are exactly `read_dataset`'s drawings.
    part = split_frames(frame)[args.split].reset_index(drop=True)
    if args.limit is not None:
        part = part.head(args.limit)

    def drawings() -> Iterable[tuple[bytes, int]]:
        # Rendered to PNG and handed to the encoder as bytes: the encoder decodes, composites and
        # runs `prepare_drawing` exactly as it does for an upload, so this is the product's path
        # from the bytes onwards. At the shipped settings `prepare_drawing` returns a rendered
        # canvas unchanged, so this is also the image `read_dataset` would hand to training.
        for row in part.itertuples():
            try:
                png = render_strokes_png(json.loads(row.strokes))
            except Exception as error:
                # Abort rather than skip: a dropped drawing would quietly change the denominator.
                raise ValueError(f"record {row.record_id} could not be rendered: {error}") from error
            yield png, index[row.label.casefold()]

    def progress(done: int) -> None:
        if done % 5000 == 0 or done == len(part):
            print(
                f"{done}/{len(part)} scored, {time.perf_counter() - start:.0f}s", file=sys.stderr, flush=True
            )

    # Read before the run: it takes minutes, and an edit meanwhile should not be recorded as the code.
    git_sha, git_dirty = _git_state()
    start = time.perf_counter()
    encoder = SiglipEncoder(config, device=args.device)
    hits = top5_hits(drawings(), encoder, prompts, progress)
    groups = {key: list(slice_values(part, key)) for key in SLICE_KEYS}
    result = metrics_file(
        hits,
        groups,
        len(labels),
        gated=args.split == "test" and args.limit is None,
        model_uri=str(args.model),
        encoder=config.encoder,
        encoder_revision=config.encoder_revision,
        data_version=args.version,
        # `Store()` honours PRIMER_DATA_URI, so a version tag alone does not say which data was read.
        data_uri=store.uri,
        split=args.split,
        limit=args.limit,
        over="rendered Quick, Draw! (recognized=True only), not children's uploads",
        license=sorted(set(part["license"])),
        attribution=sorted(set(part["attribution"])),
        git_sha=git_sha,
        git_dirty=git_dirty,
        seconds=round(time.perf_counter() - start, 1),
        **_scored_under(config, str(encoder.device), encoder.dtype_name or "unknown"),
        preprocessing=asdict(encoder.prepare or default_prepare_config()),
    )
    # Written beside and renamed into place, so a crash mid-write cannot leave truncated JSON at --out.
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    partial.replace(args.out)
    judged = sum(1 for s in result["slices"] if s["judged"])
    print(
        f"wrote {args.out}: drawing_top5_acc={result['overall']['drawing_top5_acc']:.4f} over {len(hits)} drawings "
        f"(chance {result['baseline']['drawing_top5_acc']}; {judged} of {len(result['slices'])} slices judged)"
    )
    if args.mlflow:
        log_to_mlflow("accuracy", result, args.out, args.owner, f"s6c-drawing-top5-{args.split}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["stability"]:
        return _stability(argv[1:])
    return _evaluate_accuracy(argv)


if __name__ == "__main__":
    sys.exit(main())
