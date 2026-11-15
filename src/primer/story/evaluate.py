"""Evaluate a story writer over validation requests, and write the contract `metrics.json`.

Metric definitions (every rate is in [0, 1]; "requests" means all of them, failures included):

- `first_attempt_valid_json_rate` -- requests whose **first** reply parsed as the story JSON.
- `target_word_inclusion` -- **the gated key.** Target words used, over target words requested, across
  all requests. A request that failed every attempt is scored on its last parsed story, or as zero
  if nothing ever parsed, so failures pull the number down rather than vanishing from it.
- `success_rate` -- requests that returned a story (all three retry checks passed).
- `page_limit_rate`, `sentence_limit_rate` -- among returned stories, those within the age mode's page
  count and sentence/word limits. Measured, never retried on. `format_compliance_rate` is their mean.
- `fk_mean`, `fk_within1_rate` -- the shared `fk_grade` of returned stories, and the share within ±1
  of the request's target level.
- `raw_safety_flag_rate` -- attempts the guardrail flagged, over all attempts: how often the model
  *tried* to write something unsafe.
- `returned_safety_violation_rate` -- returned stories the guardrail flags. **Zero by construction**
  when the same guardrail drives the retries; it is reported so that a different, independent
  classifier can be passed in to measure it honestly (member 3's, once it exists).
- `mean_attempts`, `p95_latency_s` -- cost.

`story_quality_mean` (the other gated story key) needs the LLM-as-judge, which is not built in this
submission (`CARD.md`, "Not applicable / deferred"), so it is absent from `overall` rather than faked.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from primer.evaluation import contract
from primer.safety.interface import SafetyClassifier
from primer.story.checks import check_story, missing_target_words
from primer.story.data import ValidationRequest
from primer.story.presets import mode_for
from primer.story.writer import LLMStoryWriter, WriteReport

#: The gated keys this stage owns, from the shared contract.
OWNED_METRICS = tuple(k for k, owner in contract.METRIC_OWNERS.items() if owner == "story")


@dataclass(frozen=True)
class RequestResult:
    """One request's outcome. Holds no story text and no character name, so it is safe to log."""

    request_id: str
    source: str
    age: int
    ok: bool
    attempts: int
    outcomes: tuple[str, ...]
    first_valid_json: bool
    words_wanted: int
    words_found: int
    pages_ok: bool | None
    sentences_ok: bool | None
    level: float | None
    level_within1: bool | None
    returned_flagged: bool | None
    latency_s: float


def run_requests(
    writer: LLMStoryWriter,
    requests: Sequence[ValidationRequest],
    judge: SafetyClassifier | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> list[RequestResult]:
    """Write a story for each request and measure it. `judge` checks returned stories (default: the
    writer's own guardrail)."""
    judge = judge or writer.classifier
    assert judge is not None
    results = []
    for item in requests:
        request = item.to_request()
        mode = mode_for(request.age, request.age_mode)
        start = clock()
        report: WriteReport = writer.write_with_report(request)
        latency = clock() - start
        scored = report.story or report.last_parsed
        missing = missing_target_words(scored, request.target_words) if scored else request.target_words
        check = check_story(report.story, request.target_words, mode) if report.story else None
        within = None
        if check is not None and request.reading_level is not None:
            within = abs(check.level - request.reading_level) <= 1.0
        returned_flagged = None
        if report.story is not None:
            returned_flagged = judge.check(" ".join((report.story.title, *report.story.pages))).flagged
        results.append(RequestResult(
            request_id=item.request_id, source=item.source, age=item.age, ok=report.ok,
            attempts=len(report.attempts), outcomes=tuple(a.outcome for a in report.attempts),
            first_valid_json=report.attempts[0].outcome != "invalid_json",
            words_wanted=len(request.target_words), words_found=len(request.target_words) - len(missing),
            pages_ok=check.pages_ok if check else None, sentences_ok=check.sentences_ok if check else None,
            level=check.level if check else None, level_within1=within,
            returned_flagged=returned_flagged, latency_s=latency,
        ))
    return results


def _rate(values: Sequence[bool | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(sum(known) / len(known), 4) if known else None


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)], 4)


def summarise(results: Sequence[RequestResult]) -> dict[str, float | None]:
    """The metrics defined in the module docstring, over a set of results."""
    if not results:
        raise ValueError("no results to summarise")
    wanted = sum(r.words_wanted for r in results)
    attempts = [o for r in results for o in r.outcomes]
    pages = _rate([r.pages_ok for r in results])
    sentences = _rate([r.sentences_ok for r in results])
    levels = [r.level for r in results if r.level is not None]
    return {
        "first_attempt_valid_json_rate": _rate([r.first_valid_json for r in results]),
        "target_word_inclusion": round(sum(r.words_found for r in results) / wanted, 4) if wanted else 0.0,
        "success_rate": _rate([r.ok for r in results]),
        "page_limit_rate": pages,
        "sentence_limit_rate": sentences,
        "format_compliance_rate": round((pages + sentences) / 2, 4) if pages is not None and sentences is not None
        else None,
        "fk_mean": round(sum(levels) / len(levels), 4) if levels else None,
        "fk_within1_rate": _rate([r.level_within1 for r in results]),
        "raw_safety_flag_rate": round(attempts.count("unsafe") / len(attempts), 4) if attempts else 0.0,
        "returned_safety_violation_rate": _rate([r.returned_flagged for r in results]) or 0.0,
        "mean_attempts": round(sum(r.attempts for r in results) / len(results), 4),
        "p95_latency_s": _p95([r.latency_s for r in results]),
        "n_requests": float(len(results)),
    }


def metrics_file(
    *,
    model_uri: str,
    data_version: str,
    git_sha: str,
    overall: dict[str, Any],
    baseline: dict[str, Any],
    slices: list[dict[str, Any]],
    details: dict[str, Any],
) -> dict[str, Any]:
    """The contract shape (model-development section 2). `overall` and `baseline` carry only this
    stage's gated keys; everything else goes in `details`, which the gate ignores."""
    owned = [k for k in OWNED_METRICS if k in overall]
    return {
        "stage": "story",
        "model_uri": model_uri,
        "data_version": data_version,
        "git_sha": git_sha,
        "overall": {k: overall[k] for k in owned},
        "baseline": {k: baseline[k] for k in owned if k in baseline},
        "slices": slices,
        "details": details,
        "not_measured": [k for k in OWNED_METRICS if k not in overall],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.parse_args(argv)
    # The grid (`python -m primer.story.grid`) is the entry point that evaluates and writes metrics.json
    # for this submission; a single-model CLI arrives with registration (S9).
    raise NotImplementedError("use `python -m primer.story.grid --config ...` for this submission")


if __name__ == "__main__":
    sys.exit(main())
