"""The safety contract: the Protocol serving imports, plus a Fake for tests.

A model area never imports ``primer.serving``; serving imports the area. Importing this module must
stay cheap and must not load weights -- ``model.py`` does that, lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SafetyVerdict:
    """Whether something may reach a child, and why not."""

    flagged: bool
    category: str | None = None
    reason: str | None = None


@runtime_checkable
class SafetyClassifier(Protocol):
    """Refuse unsafe requests and unsafe output. Gated on `safety_violations_golden`, target zero."""

    def check(self, text: str) -> SafetyVerdict:
        """Judge one piece of text, a request or a generated page."""
        ...


class FakeSafetyClassifier:
    """Flags text containing any of `unsafe_markers`; clean otherwise."""

    def __init__(self, unsafe_markers: tuple[str, ...] = ()) -> None:
        self.unsafe_markers = unsafe_markers

    def check(self, text: str) -> SafetyVerdict:
        lowered = text.lower()
        for marker in self.unsafe_markers:
            if marker.lower() in lowered:
                return SafetyVerdict(flagged=True, category="test-marker", reason=f"contains {marker!r}")
        return SafetyVerdict(flagged=False)
