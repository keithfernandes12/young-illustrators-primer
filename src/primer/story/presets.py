"""Age-mode presets: how long a story is and how short its sentences are.

Only Learn to Read (ages 3-5) is built in this submission (plan Decisions, 2026-11-14 scope change).
The numbers are provisional and live here, not at call sites, because changing one changes every
compliance metric: a `page_limit_rate` is only comparable with another taken under the same preset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgeMode:
    name: str
    min_age: int
    max_age: int
    min_pages: int
    max_pages: int
    max_sentences_per_page: int
    max_words_per_sentence: int
    #: The FK grade range a story for this mode should land in (the golden targets for ages 3-5 run
    #: 0.0-1.3). Measured, never retried on: see the plan's regeneration decision.
    max_level: float


LEARN_TO_READ = AgeMode(
    name="learn_to_read",
    min_age=3,
    max_age=5,
    min_pages=4,
    max_pages=6,
    max_sentences_per_page=2,
    max_words_per_sentence=10,
    max_level=1.5,
)

MODES = {LEARN_TO_READ.name: LEARN_TO_READ}


def mode_for(age: int, age_mode: str | None = None) -> AgeMode:
    """The preset for a request. Raises for anything this submission does not build."""
    if age_mode is not None:
        if age_mode not in MODES:
            raise ValueError(f"age mode {age_mode!r} is not built yet; only {sorted(MODES)}")
        return MODES[age_mode]
    if LEARN_TO_READ.min_age <= age <= LEARN_TO_READ.max_age:
        return LEARN_TO_READ
    raise ValueError(f"age {age} is outside the built age modes (Learn to Read is 3-5)")
