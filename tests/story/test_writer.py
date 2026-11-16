"""The writer's retry rules: invalid JSON, a missing target word, or a failed safety check -- and nothing else."""

from __future__ import annotations

import pytest
from story_fakes import GOOD_PAGES, ScriptedBackend, story_json

from primer.safety.interface import FakeSafetyClassifier
from primer.story.interface import StoryRequest, StoryWriter
from primer.story.writer import GenerationSettings, LLMStoryWriter, StoryGenerationError

REQUEST = StoryRequest(prompt="", age=4, target_words=("ball", "cat"), character_name="Pip")
GOOD = story_json(GOOD_PAGES)


def writer(*replies: str, **kwargs) -> tuple[LLMStoryWriter, ScriptedBackend]:
    backend = ScriptedBackend(replies)
    return LLMStoryWriter(backend=backend, **kwargs), backend


def test_the_writer_satisfies_the_story_protocol():
    assert isinstance(writer(GOOD)[0], StoryWriter)


def test_a_good_first_reply_needs_one_attempt():
    report = writer(GOOD)[0].write_with_report(REQUEST)
    assert report.ok and report.story is not None
    assert report.story.pages == GOOD_PAGES
    assert [a.outcome for a in report.attempts] == ["ok"]


@pytest.mark.parametrize(
    "first,outcome",
    [("not json", "invalid_json"),
     (story_json(["Pip sees a cat.", "Pip naps.", "Pip eats.", "The end."]), "missing_words"),
     (story_json([*GOOD_PAGES, "Then a gun went bang."]), "unsafe")],
    ids=["invalid-json", "missing-word", "unsafe"],
)
def test_each_failure_kind_is_retried(first, outcome):
    report = writer(first, GOOD)[0].write_with_report(REQUEST)
    assert report.ok
    assert [a.outcome for a in report.attempts] == [outcome, "ok"]


def test_format_limits_are_measured_not_retried():
    """Ten pages breaks the 3-5 page limit, but limits are not a retry condition (plan Decisions)."""
    ten_pages = story_json([*GOOD_PAGES, *(["We play."] * 6)])
    report = writer(ten_pages)[0].write_with_report(REQUEST)
    assert report.ok and len(report.attempts) == 1


def test_every_attempt_failing_raises_and_keeps_the_last_parsed_story():
    missing = story_json(["Pip sees a cat."])
    w, _ = writer("not json", missing, max_attempts=3)
    report = w.write_with_report(REQUEST)
    assert report.story is None
    assert report.last_parsed is not None and report.last_parsed.pages == ("Pip sees a cat.",)
    assert [a.outcome for a in report.attempts] == ["invalid_json", "missing_words", "missing_words"]
    with pytest.raises(StoryGenerationError, match="3 attempts"):
        w.write(REQUEST)


def test_each_retry_uses_a_new_seed():
    w, backend = writer("not json", "not json", GOOD, settings=GenerationSettings(seed=10, temperature=0.5))
    w.write(REQUEST)
    assert [call.seed for call in backend.calls] == [10, 11, 12]
    assert all(call.temperature == 0.5 for call in backend.calls)


def test_another_classifier_can_replace_the_guardrail():
    """Member 3's classifier, or any SafetyClassifier, plugs in through the same protocol."""
    w, _ = writer(GOOD, GOOD, classifier=FakeSafetyClassifier(unsafe_markers=("sun",)), max_attempts=2)
    report = w.write_with_report(REQUEST)
    assert not report.ok
    assert [a.outcome for a in report.attempts] == ["unsafe", "unsafe"]


def test_attempt_details_never_contain_the_story_or_the_name():
    w, _ = writer(story_json(["Pip sees a dog."]), max_attempts=1)
    report = w.write_with_report(REQUEST)
    assert all("Pip" not in a.detail and "dog" not in a.detail for a in report.attempts)


def test_ages_outside_three_to_five_are_refused():
    with pytest.raises(ValueError):
        writer(GOOD)[0].write(StoryRequest(prompt="", age=7, target_words=("cat",)))
