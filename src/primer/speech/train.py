"""Select the speech model. CLI: --config --out [--data-version --only NAME ... --smoke --no-mlflow].

ASR is a pretrained stage, so step 5 ("train and tune") is model selection: every candidate in the
config (checkpoint x decoding settings) is evaluated on the same clips, one MLflow run each, and the
best one under ``selection`` is chosen. The baseline candidate runs first and every other run records
its numbers as ``baseline``.

Writes, under ``--out``:

- ``<candidate>/metrics.json`` (the contract shape) and ``<candidate>/bias_report.md``
- ``comparison.md`` and, if matplotlib is installed, ``comparison.png``
- ``selection.json``: which candidate won and why, or why none qualified
- ``<selected>/sensitivity.md``: added noise and speed perturbation on the winner, or on the
  candidate ``--sensitivity-on`` names

It runs on a member's machine. An agent session may run it locally only inside the cap -- 30 minutes
of wall clock -- and never launches it as a cloud job (docs/adrs/bounded-local-training-for-agents.md).
``--smoke`` runs the whole path on a fake recogniser and synthetic clips, with no download.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from primer.speech import sensitivity
from primer.speech.evaluate import (
    CONFIG,
    STAGE,
    ReadAloudClip,
    evaluate,
    git_sha,
    load_config,
    metrics_file,
    model_uri,
)
from primer.speech.interface import SAMPLE_RATE, SpeechRecognizer, Transcript
from primer.speech.slices import bias_report

EXPERIMENT = "yip-speech"


def select_best(results: Mapping[str, Mapping[str, Any]], selection: Mapping[str, Any]) -> dict[str, Any]:
    """The winning candidate under `selection`, with the reason each other one lost.

    Eligible: WER at most ``max_wer`` and real-time factor at most ``max_rtf`` (an unknown RTF does not
    disqualify). Among those, the highest ``rank_by`` wins, ties broken by lower WER. A metric that is
    None ranks last rather than being treated as zero or one.
    """
    max_wer, max_rtf, rank_by = selection["max_wer"], selection["max_rtf"], selection["rank_by"]
    eligible, rejected = [], {}
    for name, overall in results.items():
        if overall["asr_wer"] > max_wer:
            rejected[name] = f"asr_wer {overall['asr_wer']} > {max_wer}"
        elif overall.get("asr_rtf") is not None and overall["asr_rtf"] > max_rtf:
            rejected[name] = f"asr_rtf {overall['asr_rtf']} > {max_rtf}"
        else:
            eligible.append(name)

    def key(name: str) -> tuple[float, float]:
        score = results[name].get(rank_by)
        return (float("inf") if score is None else -float(score), float(results[name]["asr_wer"]))

    ranked = sorted(eligible, key=key)
    return {
        "selected": ranked[0] if ranked else None,
        "ranking": ranked,
        "rejected": rejected,
        "rule": dict(selection),
    }


def comparison_table(results: Mapping[str, Mapping[str, Any]], selected: str | None, baseline: str) -> str:
    keys = (
        "asr_wer",
        "asr_misread_precision",
        "asr_misread_recall",
        "asr_misread_f1",
        "asr_runaway_rate",
        "asr_rtf",
    )
    lines = ["| candidate | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, overall in results.items():
        cells = ["n/a" if overall.get(k) is None else f"{overall[k]:.3f}" for k in keys]
        mark = " (selected)" if name == selected else " (baseline)" if name == baseline else ""
        lines.append(f"| {name}{mark} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _plot(results: Mapping[str, Mapping[str, Any]], path: Path) -> bool:
    """A bar chart of WER and misread F1 per candidate. Skipped without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    names = list(results)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.1), 4))
    for offset, metric in ((-0.2, "asr_wer"), (0.2, "asr_misread_f1")):
        ax.bar(x + offset, [results[n].get(metric) or 0.0 for n in names], width=0.4, label=metric)
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _numeric(prefix: str, values: Mapping[str, Any]) -> dict[str, float]:
    return {
        f"{prefix}{k}": float(v) for k, v in values.items() if isinstance(v, int | float) and v is not None
    }


def slice_metric_name(key: str, value: str, metric: str) -> str:
    """``slice/<key>/<value>/<metric>``. MLflow refuses ``=`` in a metric name, so it is a path segment."""
    clean = "".join(ch if ch.isalnum() or ch in "-_. " else "_" for ch in value)
    return f"slice/{key}/{clean}/{metric}"


def log_run(
    name: str,
    candidate: Mapping[str, Any],
    data: Mapping[str, Any],
    result: Mapping[str, Any],
    run_dir: Path,
    owner: str,
) -> str:
    """One MLflow run for one candidate. Returns the run id. Logs numbers and settings, never audio."""
    import mlflow

    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", EXPERIMENT))
    tags = {
        "stage": STAGE,
        "owner": owner,
        "data_version": result["data_version"],
        "git_sha": result["git_sha"],
        "base_model": candidate["checkpoint"],
        "gate_group": STAGE,
        "where": os.environ.get("YIP_WHERE", "local"),
    }
    with mlflow.start_run(run_name=name, tags=tags) as run:
        mlflow.log_params(
            {f"model.{k}": v for k, v in candidate.items()} | {f"data.{k}": v for k, v in data.items()}
        )
        mlflow.log_metrics(_numeric("", result["overall"]) | _numeric("baseline/", result["baseline"]))
        for s in result["slices"]:
            for metric, value in s["metrics"].items():
                if isinstance(value, int | float):
                    mlflow.log_metric(slice_metric_name(s["key"], s["value"], metric), float(value))
        mlflow.log_artifacts(str(run_dir))
        return str(run.info.run_id)


def _tag_selected(run_id: str) -> None:
    from mlflow.tracking import MlflowClient

    MlflowClient().set_tag(run_id, "selected", "true")


# ---------------------------------------------------------------------------------- smoke mode


class _SmokeRecognizer:
    """Hears a fixed sentence per clip length. Lets ``--smoke`` exercise the path with no model."""

    def __init__(self, by_length: Mapping[int, str]) -> None:
        self.by_length = by_length

    def transcribe(
        self, samples: ArrayLike, sample_rate: int = SAMPLE_RATE, prompt: str | None = None
    ) -> Transcript:
        from primer.speech.interface import FakeSpeechRecognizer

        return FakeSpeechRecognizer(self.by_length.get(len(np.asarray(samples)), "")).transcribe(samples)


def smoke_clips() -> tuple[list[ReadAloudClip], SpeechRecognizer]:
    """Synthetic clips (silence of distinct lengths) and a recogniser that 'hears' each reference."""
    from primer.speech.data import with_synthetic_misreads

    sentences = [
        "the cat sat on the mat",
        "a big red dog ran home",
        "we like to read books",
        "the sun is hot",
    ]
    clips = [
        ReadAloudClip(
            clip_id=f"smoke-{i}",
            speaker_id=f"smoke-speaker-{i % 2}",
            page_text=text,
            reference=text,
            misread_positions=frozenset(),
            samples=np.zeros(SAMPLE_RATE + i * 160, dtype=np.float32),
            slices={"age_band": "3-5" if i % 2 else "6-8"},
        )
        for i, text in enumerate(sentences)
    ]
    recognizer = _SmokeRecognizer({len(c.samples): c.reference for c in clips})
    return with_synthetic_misreads(clips, rate=0.5, seed=0), recognizer


# ------------------------------------------------------------------------------------------ CLI


def _run_candidate(
    name: str,
    candidate: Mapping[str, Any],
    recognizer: SpeechRecognizer,
    clips: Sequence[ReadAloudClip],
    data: Mapping[str, Any],
    version: str,
    baseline: Mapping[str, Any] | None,
    out: Path,
) -> dict[str, Any]:
    overall, slices = evaluate(recognizer, clips, prompt_with_page=bool(candidate.get("prompt_with_page")))
    result = metrics_file(
        model_uri=model_uri(candidate),
        data_version=version,
        git_sha=git_sha(),
        overall=overall,
        baseline=baseline,
        slices=slices,
    )
    run_dir = out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (run_dir / "bias_report.md").write_text(bias_report(slices), encoding="utf-8")
    (run_dir / "recognizer.json").write_text(json.dumps(dict(candidate), indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--data-version", default=None, help="recorded with each run; defaults to the config's"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--only", action="append", default=None, help="run just these candidates (repeatable)"
    )
    parser.add_argument("--max-clips", type=int, default=None, help="override the config's max_clips")
    parser.add_argument("--no-mlflow", action="store_true", help="write files only")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument(
        "--sensitivity-on",
        default=None,
        help="run the sensitivity analysis on this candidate instead of the selected one (it must be in the batch)",
    )
    parser.add_argument("--smoke", action="store_true", help="fake recogniser, synthetic clips, no download")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    data = dict(config["data"])
    if args.max_clips is not None:
        data["max_clips"] = args.max_clips
    baseline_name = config["baseline"]
    candidates = {c["name"]: c for c in config["candidates"]}
    names = args.only or list(candidates)
    unknown = set(names) - set(candidates)
    if unknown:
        parser.error(f"unknown candidates: {sorted(unknown)}")
    # The baseline runs first so every other run can record it. If it is not in this batch, an earlier
    # batch's file supplies it.
    names = sorted(names, key=lambda n: n != baseline_name)
    if args.sensitivity_on and args.sensitivity_on not in names:
        parser.error(f"--sensitivity-on {args.sensitivity_on!r} must be one of the candidates in this batch")
    baseline: dict[str, Any] | None = None
    if baseline_name not in names:
        earlier = args.out / baseline_name / "metrics.json"
        if not earlier.exists():
            parser.error(f"run the baseline {baseline_name!r} first, or include it with --only")
        baseline = json.loads(earlier.read_text(encoding="utf-8"))["overall"]

    use_mlflow = not (args.no_mlflow or args.smoke)
    if args.smoke:
        clips, smoke_recognizer = smoke_clips()
        version = "smoke"
    else:
        from primer.speech.data import data_version, load_clips

        clips = load_clips(data)
        version = args.data_version or data_version(data)
        print(f"loaded {len(clips)} clips ({version})")

    from primer.speech.model import RecognizerConfig, build

    results: dict[str, dict[str, Any]] = {}
    recognizers: dict[str, SpeechRecognizer] = {}
    run_ids: dict[str, str] = {}
    for name in names:
        candidate = candidates[name]
        settings = RecognizerConfig.from_dict(candidate)
        recognizer: SpeechRecognizer = smoke_recognizer if args.smoke else build(settings)
        print(f"evaluating {name} ...", flush=True)
        result = _run_candidate(name, asdict(settings), recognizer, clips, data, version, baseline, args.out)
        if name == baseline_name:
            baseline = result["overall"]
        results[name] = result["overall"]
        recognizers[name] = recognizer
        if use_mlflow:
            run_ids[name] = log_run(
                name, asdict(settings), data, result, args.out / name, config.get("owner", "")
            )
        print(
            f"  asr_wer={result['overall']['asr_wer']} asr_misread_f1={result['overall']['asr_misread_f1']}"
        )

    choice = select_best(results, config["selection"])
    choice["run_ids"] = run_ids
    selected = choice["selected"]
    (args.out / "selection.json").write_text(json.dumps(choice, indent=2) + "\n", encoding="utf-8")
    (args.out / "comparison.md").write_text(
        comparison_table(results, selected, baseline_name), encoding="utf-8"
    )
    _plot(results, args.out / "comparison.png")

    # A named candidate is analysed even when none met the selection rule: the analysis says how the
    # model behaves, which is worth knowing for the one the team chose. It is not tagged as selected.
    target = args.sensitivity_on or selected
    if target and not args.no_sensitivity:
        print(f"sensitivity analysis on {target} ...", flush=True)
        rows = sensitivity.run(
            recognizers[target], clips, prompt_with_page=bool(candidates[target].get("prompt_with_page"))
        )
        (args.out / target / "sensitivity.md").write_text(sensitivity.table(rows), encoding="utf-8")
        (args.out / target / "sensitivity.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
        if use_mlflow:
            import mlflow

            with mlflow.start_run(run_id=run_ids[target]):
                mlflow.log_artifact(str(args.out / target / "sensitivity.md"))
                for row in rows:
                    mlflow.log_metric(f"sensitivity/{row['perturbation']}/asr_wer", float(row["asr_wer"]))
    if selected and use_mlflow:
        _tag_selected(run_ids[selected])

    print(comparison_table(results, selected, baseline_name))
    print(f"selected: {selected or 'none qualified'} -> {args.out / 'selection.json'}")
    return 0 if selected else 1


if __name__ == "__main__":
    sys.exit(main())
