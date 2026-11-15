"""Evaluate the speech model. CLI: --model <candidate> --out metrics.json [--config --split --max-clips --baseline].

Writes the shape in ``primer.evaluation.contract``: ``overall`` carries this stage's metrics,
``baseline`` is what the model must beat and is never omitted, ``slices`` carries the per-slice
breakdown. ``primer.evaluation.golden`` merges these into the file the release gate reads.

The shared contract has no ASR key yet, so ``OWNED_METRICS`` is empty and this stage's numbers are
reported under ``asr_*`` keys in ``overall`` (and in ``CARD.md``) without being gated. Making one of
them gated means changing ``contract.py`` and ``thresholds.json``, agreed with the team.

Two things are measured, because they answer different questions:

- **Word error rate** compares the transcript with what the child actually said (``reference``, a
  human transcript). It is the recogniser's accuracy.
- **Misread precision and recall** compare the words ``align`` flags against the page words a person
  labelled as misread or skipped. It is what the learner profile sees, and it is where Whisper's
  habit of auto-correcting a misreading shows up: a low recall with a good word error rate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from primer.evaluation import contract
from primer.speech.alignment import align, normalize
from primer.speech.interface import SAMPLE_RATE, SpeechRecognizer, Transcript, Word

#: The gated keys this stage owns, from the shared contract.
OWNED_METRICS = tuple(k for k, owner in contract.METRIC_OWNERS.items() if owner == "speech")

STAGE = "speech"

#: A clip whose inserted words exceed this multiple of its reference words (plus the slack) ran away.
RUNAWAY_FACTOR = 2
RUNAWAY_SLACK = 3

#: The candidates config the CLIs read by default.
CONFIG = Path(__file__).resolve().parent / "configs" / "speech_baseline.yaml"


@dataclass(frozen=True)
class ReadAloudClip:
    """One labelled utterance of a page being read aloud.

    `reference` is what was actually said, transcribed by a person. `misread_positions` indexes the
    page words (as ``normalize`` splits them) that the child misread or skipped. `slices` carries the
    slice values for this clip, e.g. ``{"speaker_age": "5-6"}``; never a name or any other PII.
    """

    clip_id: str
    speaker_id: str
    page_text: str
    reference: str
    misread_positions: frozenset[int]
    samples: Sequence[float] | np.ndarray = ()
    sample_rate: int = SAMPLE_RATE
    slices: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WordErrors:
    """Edit counts between a reference and a hypothesis, summable across clips."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_words: int = 0

    def __add__(self, other: WordErrors) -> WordErrors:
        return WordErrors(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_words + other.reference_words,
        )

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Errors over reference words. Undefined with no reference words, unless nothing was heard."""
        if self.reference_words == 0:
            if self.errors == 0:
                return 0.0
            raise ValueError("word error rate is undefined with no reference words")
        return self.errors / self.reference_words


def word_errors(reference: str, hypothesis: str) -> WordErrors:
    """Minimum-edit counts between two utterances, after the same normalisation ``align`` uses."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    # cost[j] = (total, subs, dels, ins) for ref[:i] against hyp[:j]; ties prefer fewer substitutions.
    cost = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        previous, cost = cost, [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                match = previous[j - 1]
            else:
                t, s, d, n = previous[j - 1]
                match = (t + 1, s + 1, d, n)
            t, s, d, n = previous[j]
            delete = (t + 1, s, d + 1, n)
            t, s, d, n = cost[j - 1]
            insert = (t + 1, s, d, n + 1)
            cost.append(min(match, delete, insert))
    _, subs, dels, ins = cost[-1]
    return WordErrors(subs, dels, ins, len(ref))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate of one utterance. See ``WordErrors.rate`` for the empty cases."""
    return word_errors(reference, hypothesis).rate


def flagged_positions(page_text: str, heard: str) -> frozenset[int]:
    """Indexes of page words ``align`` marks misread or skipped for `heard`."""
    transcript = Transcript(words=tuple(Word(w, 0.0, 0.0, 1.0) for w in heard.split()))
    page_words = [w for w in align(0, page_text, transcript).words if w.expected is not None]
    return frozenset(i for i, w in enumerate(page_words) if w.status in ("misread", "skipped"))


def is_runaway(errors: WordErrors) -> bool:
    """Whether a clip's output ran away: more inserted words than ``RUNAWAY_FACTOR`` x what was said.

    A runaway is a different failure from misrecognition -- the recogniser invents speech -- and one
    of them can dominate a corpus WER, so it is counted on its own line (``asr_runaway_rate``).
    """
    return errors.insertions > RUNAWAY_FACTOR * max(errors.reference_words, 1) + RUNAWAY_SLACK


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _metrics(rows: Sequence[_Row]) -> dict[str, Any]:
    """The ``asr_*`` metrics over transcribed rows.

    Precision is None when nothing was flagged and recall is None when nothing was labelled: both are
    undefined there, and writing 0 or 1 would read as a measured result. F1 is None when either is.
    `asr_rtf` is transcription time over audio time (under 1 is faster than real time); it is None
    when the clips carry no audio, as with the fake. `asr_runaway_rate` is the share of clips where
    the recogniser invented far more words than were said (``is_runaway``).
    """
    errors = WordErrors()
    true_pos = flagged = labelled = runaway = 0
    seconds = audio_s = 0.0
    for row in rows:
        clip = row.clip
        clip_errors = word_errors(clip.reference, row.heard)
        errors += clip_errors
        runaway += is_runaway(clip_errors)
        predicted = flagged_positions(clip.page_text, row.heard)
        true_pos += len(predicted & clip.misread_positions)
        flagged += len(predicted)
        labelled += len(clip.misread_positions)
        seconds += row.seconds
        audio_s += len(clip.samples) / clip.sample_rate
    precision, recall = _ratio(true_pos, flagged), _ratio(true_pos, labelled)
    f1 = None
    if precision is not None and recall is not None:
        f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    return {
        "asr_wer": round(errors.rate, 4),
        "asr_misread_precision": precision,
        "asr_misread_recall": recall,
        "asr_misread_f1": f1,
        "asr_rtf": round(seconds / audio_s, 4) if audio_s else None,
        "asr_runaway_rate": round(runaway / len(rows), 4),
        "asr_clips": len(rows),
        "asr_reference_words": errors.reference_words,
    }


@dataclass(frozen=True)
class _Row:
    clip: ReadAloudClip
    heard: str
    seconds: float


def evaluate(
    recognizer: SpeechRecognizer,
    clips: Iterable[ReadAloudClip],
    prompt_with_page: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Transcribe every clip and return ``(overall, slices)``. The audio is not kept.

    With `prompt_with_page`, the page text is passed as the recogniser's prompt. Each slice carries
    `judged`: False when it has fewer clips than ``slices.SMALL_SLICE``, so it is reported, not judged.
    """
    from primer.speech.slices import SMALL_SLICE

    rows: list[_Row] = []
    for clip in clips:
        started = clock()
        heard = recognizer.transcribe(
            clip.samples, clip.sample_rate, prompt=clip.page_text if prompt_with_page else None
        ).text
        rows.append(_Row(clip, heard, clock() - started))
    if not rows:
        raise ValueError("no clips to evaluate")
    groups: dict[tuple[str, str], list[_Row]] = defaultdict(list)
    for row in rows:
        for key, value in row.clip.slices.items():
            groups[(key, value)].append(row)
    slices = [
        {
            "key": key,
            "value": value,
            "n": len(group),
            "judged": len(group) >= SMALL_SLICE,
            "metrics": _metrics(group),
        }
        for (key, value), group in sorted(groups.items())
    ]
    return _metrics(rows), slices


def metrics_file(
    *,
    model_uri: str,
    data_version: str,
    git_sha: str,
    overall: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    slices: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The contract shape. With no `baseline`, this run is the baseline and ``baseline == overall``."""
    return {
        "stage": STAGE,
        "model_uri": model_uri,
        "data_version": data_version,
        "git_sha": git_sha,
        "overall": dict(overall),
        "baseline": dict(overall if baseline is None else baseline),
        "slices": [dict(s) for s in slices],
    }


def read_baseline(path: Path) -> dict[str, Any]:
    """The ``overall`` of an earlier speech run's metrics file, which this run must beat."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("stage") != STAGE:
        raise ValueError(f"{path} is a {data.get('stage')!r} metrics file, not {STAGE!r}")
    return dict(data["overall"])


def git_sha() -> str:
    """The short commit this code runs at, or "unknown" outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=CONFIG.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def load_config(path: Path) -> dict[str, Any]:
    """A candidates config (``configs/speech_baseline.yaml``)."""
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = [c["name"] for c in config.get("candidates", [])]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: candidate names must be unique")
    if config.get("baseline") not in names:
        raise ValueError(f"{path}: baseline {config.get('baseline')!r} is not one of the candidates")
    return dict(config)


def model_uri(candidate: Mapping[str, Any]) -> str:
    """Where the configuration comes from, e.g. ``hf:openai/whisper-small.en@main#whisper-small-en``."""
    return f"hf:{candidate['checkpoint']}@{candidate.get('revision', 'main')}#{candidate['name']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="a candidate name from --config")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--split", default=None, help="override the config's data split")
    parser.add_argument("--max-clips", type=int, default=None, help="override the config's max_clips")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="an earlier speech metrics.json to beat")
    args = parser.parse_args(argv)

    from primer.speech.data import data_version, load_clips
    from primer.speech.model import RecognizerConfig, build

    config = load_config(args.config)
    candidate = next((c for c in config["candidates"] if c["name"] == args.model), None)
    if candidate is None:
        parser.error(f"no candidate named {args.model!r} in {args.config}")
    data = dict(config["data"])
    if args.split:
        data["split"] = args.split
    if args.max_clips is not None:
        data["max_clips"] = args.max_clips
    settings = RecognizerConfig.from_dict(candidate)
    overall, slices = evaluate(build(settings), load_clips(data), prompt_with_page=settings.prompt_with_page)
    result = metrics_file(
        model_uri=model_uri(candidate),
        data_version=data_version(data),
        git_sha=git_sha(),
        overall=overall,
        baseline=read_baseline(args.baseline) if args.baseline else None,
        slices=slices,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: asr_wer={overall['asr_wer']} asr_misread_f1={overall['asr_misread_f1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
