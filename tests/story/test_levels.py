"""The one shared FK function (Keith's reading-level decision, plan Decisions 2026-11-14)."""

from __future__ import annotations

import pytest

from primer.data import text as pipeline_text
from primer.story.levels import FK_VERSION, fk_grade, raw_fk, round_half, sentence_count, syllables, words

SAMPLES = [
    "The cat sat on the mat.",
    "Pip runs! Pip jumps? Pip naps.",
    "Elephants enjoy wonderful adventures in the faraway jungle.",
    "It's Mia’s hat. Don't touch it.\n\nThe end.",
    "",
    "one two three",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_tokenisation_matches_the_pipeline(text):
    """levels.py copies primer.data.text's rules; if either changes, this fails."""
    assert len(words(text)) == pipeline_text.word_count(text)
    assert sentence_count(text) == pipeline_text.sentence_count(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_grade_is_never_negative_and_always_a_multiple_of_a_half(text):
    grade = fk_grade(text)
    assert grade >= 0
    assert (grade * 2) == int(grade * 2), f"{grade} is not on the 0.5 grid"


def test_simple_text_is_clamped_at_zero():
    """Unclamped FK goes negative on very simple text. Fails if the clamp is removed."""
    text = "The cat sat. The dog ran. We had fun."
    assert raw_fk(text) < 0
    assert fk_grade(text) == 0.0


def test_hard_text_rounds_to_the_nearest_half():
    text = "Elephants enjoy wonderful adventures."
    # 4 words, 1 sentence, 3+2+3+4 = 12 syllables by the heuristic.
    expected_raw = 0.39 * 4 + 11.8 * 12 / 4 - 15.59
    assert raw_fk(text) == pytest.approx(expected_raw)
    assert fk_grade(text) == 21.5


@pytest.mark.parametrize("value,expected", [(0.24, 0.0), (0.25, 0.5), (0.74, 0.5), (0.75, 1.0), (1.0, 1.0)])
def test_round_half_is_half_up(value, expected):
    """Python's round() is banker's rounding (round(0.5) == 0); this must not be."""
    assert round_half(value) == expected


@pytest.mark.parametrize("word,expected", [("cat", 1), ("make", 1), ("little", 2), ("the", 1), ("happy", 2)])
def test_syllable_heuristic(word, expected):
    assert syllables(word) == expected


def test_the_version_is_pinned():
    assert FK_VERSION == "fk_formula@1"
