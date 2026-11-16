"""Checks on one story against its request: the numbers every metric is built from.

`contains_target_word` is the same rule as `primer.evaluation.golden.contains_target_word`, copied
rather than imported because this runs on the request path; `tests/story/test_checks_and_prompts.py`
asserts the two agree on a shared table, so `target_word_inclusion` here means what the gate means.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from primer.story.interface import Story
from primer.story.levels import fk_grade, sentence_count, words
from primer.story.presets import AgeMode

_TOKEN = re.compile(r"[a-z0-9']+")
_SENTENCE = re.compile(r"[^.!?]+[.!?]*")


def contains_target_word(text: str, target: str) -> bool:
    """Whether `text` uses `target` as a whole word (a multiword target as a contiguous run)."""
    found = _TOKEN.findall(text.lower())
    wanted = _TOKEN.findall(target.lower())
    if not wanted:
        return False
    return any(found[i : i + len(wanted)] == wanted for i in range(len(found) - len(wanted) + 1))


def missing_target_words(story: Story, target_words: tuple[str, ...]) -> tuple[str, ...]:
    """Target words the story's **pages** do not use.

    Pages only, not the title: `Story.text` is what serving illustrates, narrates and safety-checks,
    and what the area tests mean by inclusion. A word used only in the title is never read on a page.
    """
    return tuple(word for word in target_words if not contains_target_word(story.text, word))


@dataclass(frozen=True)
class StoryCheck:
    missing_words: tuple[str, ...]
    n_pages: int
    pages_ok: bool
    #: every page within the preset's sentence count, and every sentence within its word count
    sentences_ok: bool
    level: float


def check_story(story: Story, target_words: tuple[str, ...], mode: AgeMode) -> StoryCheck:
    """Measure a story against its request and age mode. Never raises on a parsed story."""
    sentences_ok = True
    for page in story.pages:
        if sentence_count(page) > mode.max_sentences_per_page:
            sentences_ok = False
            break
        if any(len(words(sentence)) > mode.max_words_per_sentence for sentence in _SENTENCE.findall(page)):
            sentences_ok = False
            break
    return StoryCheck(
        missing_words=missing_target_words(story, target_words),
        n_pages=len(story.pages),
        pages_ok=mode.min_pages <= len(story.pages) <= mode.max_pages,
        sentences_ok=sentences_ok,
        level=fk_grade(story.text),
    )
