"""Metric definitions, computed exactly on scripted outcomes."""

from __future__ import annotations

import itertools

import pytest
from story_fakes import GOOD_PAGES, ScriptedBackend, story_json

from primer.evaluation import contract
from primer.safety.interface import FakeSafetyClassifier
from primer.story.data import ValidationRequest
from primer.story.evaluate import OWNED_METRICS, metrics_file, run_requests, summarise
from primer.story.slices import slice_metrics
from primer.story.writer import LLMStoryWriter


def vreq(i: int, words: tuple[str, ...] = ("ball", "cat"), level: float = 0.5, source: str = "storyweaver"):
    return ValidationRequest(request_id=f"r{i}", book_key=f"{source}/b{i}", source=source, age=3 + i % 3,
                             target_words=words, character_name="Pip", setting=None, reading_level=level,
                             level_source="fk_formula@1", license="CC-BY-4.0", attribution="test")


def ticking_clock(step: float = 2.0):
    counter = itertools.count()
    return lambda: next(counter) * step


def test_a_perfect_run():
    writer = LLMStoryWriter(backend=ScriptedBackend([story_json(GOOD_PAGES)]))
    results = run_requests(writer, [vreq(0), vreq(1)], clock=ticking_clock())
    summary = summarise(results)
    assert summary["first_attempt_valid_json_rate"] == 1.0
    assert summary["target_word_inclusion"] == 1.0
    assert summary["success_rate"] == 1.0
    assert summary["page_limit_rate"] == 1.0 and summary["sentence_limit_rate"] == 1.0
    assert summary["format_compliance_rate"] == 1.0
    assert summary["raw_safety_flag_rate"] == 0.0
    assert summary["mean_attempts"] == 1.0
    assert summary["p95_latency_s"] == 2.0  # the injected clock ticks 2 s per call


def test_failures_count_against_inclusion_instead_of_vanishing():
    """One request never parses (0 of 2 words); one returns a good story (2 of 2): inclusion is 0.5."""
    backend = ScriptedBackend(["not json", "not json", "not json", story_json(GOOD_PAGES)])
    writer = LLMStoryWriter(backend=backend, max_attempts=3)
    summary = summarise(run_requests(writer, [vreq(0), vreq(1)]))
    assert summary["target_word_inclusion"] == 0.5
    assert summary["success_rate"] == 0.5
    assert summary["first_attempt_valid_json_rate"] == 0.5
    assert summary["mean_attempts"] == 2.0


def test_a_failed_request_is_scored_on_its_last_parsed_story():
    """Every attempt misses "cat" but uses "ball": the request contributes 1 of 2 words."""
    backend = ScriptedBackend([story_json(["Pip has a ball.", "Pip naps.", "Pip eats.", "The end."])])
    summary = summarise(run_requests(LLMStoryWriter(backend=backend, max_attempts=2), [vreq(0)]))
    assert summary["success_rate"] == 0.0
    assert summary["target_word_inclusion"] == 0.5


def test_raw_flags_count_attempts_and_an_independent_judge_measures_returned_stories():
    unsafe = story_json([*GOOD_PAGES, "A gun went bang."])
    writer = LLMStoryWriter(backend=ScriptedBackend([unsafe, story_json(GOOD_PAGES)]))
    judge = FakeSafetyClassifier(unsafe_markers=("sun",))  # flags "nap in the sun", which the guardrail allows
    summary = summarise(run_requests(writer, [vreq(0)], judge=judge))
    assert summary["raw_safety_flag_rate"] == 0.5  # 1 of 2 attempts
    assert summary["returned_safety_violation_rate"] == 1.0


def test_level_within_one_uses_the_requests_target():
    writer = LLMStoryWriter(backend=ScriptedBackend([story_json(GOOD_PAGES)]))
    near, far = run_requests(writer, [vreq(0, level=0.0), vreq(1, level=9.0)])
    assert near.level_within1 is True and far.level_within1 is False


def test_results_hold_no_story_text_or_name():
    writer = LLMStoryWriter(backend=ScriptedBackend([story_json(GOOD_PAGES)]))
    (result,) = run_requests(writer, [vreq(0)])
    assert "Pip" not in repr(result) and "sun" not in repr(result)


def test_metrics_file_keeps_gated_keys_apart_and_never_fakes_quality():
    summary = summarise(run_requests(LLMStoryWriter(backend=ScriptedBackend([story_json(GOOD_PAGES)])),
                                     [vreq(0)]))
    out = metrics_file(model_uri="hf://x@y", data_version="v1.1", git_sha="abc", overall=summary,
                       baseline=summary, slices=[], details={"metrics": summary})
    story_keys = {k for k, owner in contract.METRIC_OWNERS.items() if owner == "story"}
    assert set(OWNED_METRICS) == story_keys
    assert set(out["overall"]) == {"target_word_inclusion"}
    assert set(out["baseline"]) == {"target_word_inclusion"}
    assert out["not_measured"] == ["story_quality_mean"]
    assert out["stage"] == "story" and out["data_version"] == "v1.1"


def test_slices_below_the_floor_are_reported_but_not_judged():
    writer = LLMStoryWriter(backend=ScriptedBackend([story_json(GOOD_PAGES)]))
    results = run_requests(writer, [vreq(i, source=s) for i, s in enumerate(["storyweaver", "tinystories"] * 3)])
    rows = slice_metrics(results, summarise)
    assert {(r["key"], r["value"]) for r in rows} >= {("source", "storyweaver"), ("source", "tinystories"),
                                                       ("age", "3")}
    assert all(r["judged"] is False for r in rows)
    assert sum(r["n"] for r in rows if r["key"] == "source") == 6


def test_summarise_refuses_nothing():
    with pytest.raises(ValueError):
        summarise([])
