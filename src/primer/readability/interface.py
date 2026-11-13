"""The readability contract: the Protocol serving imports, plus a Fake for tests.

A model area never imports ``primer.serving``; serving imports the area. Importing this module must
stay cheap and must not load weights -- ``model.py`` does that, lazily.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ReadabilityScorer(Protocol):
    """Score reading level. Gated on `reading_level_within1` and `readability_rmse`."""

    def grade(self, text: str) -> float:
        """The Flesch-Kincaid grade level of `text`."""
        ...


class FakeReadabilityScorer:
    """Returns a fixed grade, or `follow` if given, so a test can pin the error exactly."""

    def __init__(self, grade_value: float = 1.0) -> None:
        self.grade_value = grade_value

    def grade(self, text: str) -> float:
        return self.grade_value
