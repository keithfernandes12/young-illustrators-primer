"""The one Flesch-Kincaid function every part of the story stage uses.

Keith's reading-level decision (plan Decisions, 2026-11-14) requires exactly one FK implementation,
shared by the pair builder, `check_story` and the bake-off level metric, so a level never differs
between where it is labelled and where it is checked. It is:

- computed on the **whole story**, never per page (per-page FK on a 20-word page is mostly noise);
- **clamped at 0**, as `scripts/gen_golden_scenarios.py` clamps the golden targets;
- **rounded to 0.5 inside the function**, half up;
- versioned: `FK_VERSION` goes into every pair's `level_source`, and any change to this file that can
  move a grade must bump it.

Tokenisation copies `primer.data.text` (the pipeline's own word and sentence rules) rather than
importing it, because this module runs at request time and nothing on the request path may import
`primer.data`. `tests/story/test_levels.py` asserts the two stay identical.
"""

from __future__ import annotations

import math
import re

FK_VERSION = "fk_formula@1"

# Same patterns as primer.data.text._WORD and _SENTENCE_END.
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def words(text: str) -> list[str]:
    """The words of `text`, by the pipeline's rule."""
    return _WORD.findall(text)


def sentence_count(text: str) -> int:
    """Sentences in `text`, by the pipeline's rule: at least 1 for non-empty text."""
    if not text:
        return 0
    return max(len(_SENTENCE_END.findall(text)), 1)


def syllables(word: str) -> int:
    """A vowel-group estimate of the syllables in one word. At least 1.

    Heuristic, deliberately simple and dependency-free: count runs of vowels, and drop a silent
    trailing "e" ("make" is one syllable) unless the word ends in consonant + "le" ("little" is two).
    """
    lowered = word.lower().replace("’", "'").split("'")[0]
    if not lowered:
        return 1
    count = len(_VOWEL_GROUP.findall(lowered))
    if lowered.endswith("e") and not lowered.endswith("le") and count > 1:
        count -= 1
    return max(count, 1)


def raw_fk(text: str) -> float:
    """The unclamped, unrounded Flesch-Kincaid grade. Only for tests and diagnostics."""
    tokens = words(text)
    if not tokens:
        return 0.0
    n_sentences = sentence_count(text)
    n_syllables = sum(syllables(token) for token in tokens)
    return 0.39 * len(tokens) / n_sentences + 11.8 * n_syllables / len(tokens) - 15.59


def round_half(value: float) -> float:
    """Round to the nearest 0.5, half up (Python's round() would round 0.25 to 0.0, banker's style)."""
    return math.floor(value * 2 + 0.5) / 2


def fk_grade(text: str) -> float:
    """The Flesch-Kincaid grade of a whole story: clamped at 0, rounded to 0.5. Version `FK_VERSION`."""
    return round_half(max(0.0, raw_fk(text)))
