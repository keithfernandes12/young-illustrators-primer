"""The story JSON parser and the story stage's own avoid-list guardrail."""

from __future__ import annotations

import json

import pytest
from story_fakes import story_json

from primer.safety.interface import SafetyClassifier
from primer.story.guardrails import AvoidListGuardrail, load_avoid_list
from primer.story.schema import InvalidStoryJSON, parse_story


def test_a_valid_reply_parses():
    story = parse_story(story_json(["One.", "Two."], title="Hi"))
    assert story.title == "Hi"
    assert story.pages == ("One.", "Two.")


def test_a_markdown_fence_is_stripped():
    assert parse_story("```json\n" + story_json(["One."]) + "\n```").pages == ("One.",)


@pytest.mark.parametrize(
    "reply",
    [
        "Once upon a time",
        "[]",
        json.dumps({"title": "Hi", "pages": [{"text": "One."}], "moral": "be kind"}),
        json.dumps({"title": "", "pages": [{"text": "One."}]}),
        json.dumps({"title": "x" * 81, "pages": [{"text": "One."}]}),
        json.dumps({"title": "Hi", "pages": []}),
        json.dumps({"title": "Hi", "pages": [{"text": "  "}]}),
        json.dumps({"title": "Hi", "pages": [{"text": "One.", "image": "x"}]}),
        json.dumps({"title": "Hi", "pages": ["One."]}),
    ],
    ids=["prose", "array", "extra-key", "empty-title", "long-title", "no-pages", "blank-page", "page-extra",
         "page-not-object"],
)
def test_anything_else_is_invalid(reply):
    with pytest.raises(InvalidStoryJSON):
        parse_story(reply)


def test_the_guardrail_satisfies_the_safety_protocol():
    assert isinstance(AvoidListGuardrail(), SafetyClassifier)


@pytest.mark.parametrize("text,category", [("The knight drew a sword.", "weapons"),
                                           ("Then the dog DIED.", "death"),
                                           ("You are stupid.", "unkind")])
def test_avoid_list_words_are_flagged(text, category):
    verdict = AvoidListGuardrail().check(text)
    assert verdict.flagged
    assert verdict.category == category


@pytest.mark.parametrize("text", ["Mia has a new skill.", "They kick the ball.", "Shut the door, please.",
                                  "Look at the arrow on the sign.", "The drum goes beat, beat."])
def test_whole_words_only_and_no_innocent_false_positives(text):
    """"skill" must not match "kill"; the words removed on purpose must stay removed."""
    assert not AvoidListGuardrail().check(text).flagged


def test_the_verdict_never_quotes_the_text():
    """Verdicts can reach logs; the reason names a category, not the offending word."""
    verdict = AvoidListGuardrail().check("There was a gun.")
    assert "gun" not in (verdict.reason or "")


def test_every_category_has_a_prompt_phrase_and_words():
    avoid = load_avoid_list()
    assert avoid.version.startswith("avoid_list@")
    assert set(avoid.prompts) == set(avoid.words.values())
