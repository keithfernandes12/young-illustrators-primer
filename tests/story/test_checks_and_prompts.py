"""Story checks against the 3-5 preset, and the prompt built for a request."""

from __future__ import annotations

import pytest

from primer.evaluation.golden import contains_target_word as gate_contains
from primer.story.checks import check_story, contains_target_word, missing_target_words
from primer.story.guardrails import load_avoid_list
from primer.story.interface import Story, StoryRequest
from primer.story.levels import fk_grade
from primer.story.presets import LEARN_TO_READ, mode_for
from primer.story.prompts import build_messages

CASES = [
    ("The cat naps.", "cat"),
    ("A concatenated word.", "cat"),
    ("Ice cream!", "ice cream"),
    ("Cream and ice.", "ice cream"),
    ("Don't go.", "don't"),
    ("CAT!", "cat"),
    ("", "cat"),
    ("cat", ""),
]


@pytest.mark.parametrize("text,target", CASES)
def test_word_matching_agrees_with_the_gate(text, target):
    """`target_word_inclusion` here must mean what the golden gate means."""
    assert contains_target_word(text, target) == gate_contains(text, target)


def test_a_word_only_in_the_title_is_missing():
    """The child reads the pages; a target word only in the title is never read (review F2)."""
    story = Story(pages=("The dog runs.",), title="A cat")
    assert missing_target_words(story, ("cat", "dog", "nap")) == ("cat", "nap")


def _story(n_pages: int, page: str = "Pip sees a ball.") -> Story:
    return Story(pages=tuple(page for _ in range(n_pages)), title="T")


@pytest.mark.parametrize("n,ok", [(3, False), (4, True), (6, True), (7, False)])
def test_page_limits_follow_the_preset(n, ok):
    assert check_story(_story(n), (), LEARN_TO_READ).pages_ok is ok


def test_too_many_sentences_on_a_page_fails():
    assert not check_story(_story(4, "One. Two. Three."), (), LEARN_TO_READ).sentences_ok


def test_a_too_long_sentence_fails():
    long = "Pip and the cat and the dog and the hen ran to the big red barn."
    assert not check_story(_story(4, long), (), LEARN_TO_READ).sentences_ok


def test_the_level_is_the_shared_fk_on_the_whole_story():
    story = _story(4)
    assert check_story(story, (), LEARN_TO_READ).level == fk_grade(story.text)


def test_only_ages_three_to_five_are_built():
    assert mode_for(4) is LEARN_TO_READ
    with pytest.raises(ValueError):
        mode_for(7)
    with pytest.raises(ValueError):
        mode_for(4, "grow_vocabulary")


def test_the_prompt_carries_the_request_and_names_topics_not_words():
    avoid = load_avoid_list()
    request = StoryRequest(prompt="", age=4, target_words=("ball", "nap"), character_name="Pip",
                           setting="the garden")
    system, user = (m["content"] for m in build_messages(request, LEARN_TO_READ, avoid))
    assert "Pip" in user and "the garden" in user and "ball, nap" in user
    assert f"{LEARN_TO_READ.min_pages} to {LEARN_TO_READ.max_pages} pages" in system
    assert "weapons" in system, "the avoid-list topics must be in the prompt"
    assert " gun" not in system and "sword" not in system, "the avoid words themselves must not be"
    assert '"pages"' in system, "the JSON shape must be in the prompt"


@pytest.mark.parametrize("level,phrase", [(0.0, "kindergarten"), (1.0, "US grade 1"), (1.5, "US grade 1.5")])
def test_the_prompt_tells_the_model_the_target_level(level, phrase):
    """Review Q1: the level is measured against the request, so the model must be told it."""
    request = StoryRequest(prompt="", age=4, target_words=("ball",), reading_level=level)
    _, user = (m["content"] for m in build_messages(request, LEARN_TO_READ, load_avoid_list()))
    assert f"Reading level: {phrase} or easier." in user


def test_no_level_means_no_level_line():
    request = StoryRequest(prompt="", age=4, target_words=("ball",))
    _, user = (m["content"] for m in build_messages(request, LEARN_TO_READ, load_avoid_list()))
    assert "Reading level" not in user
