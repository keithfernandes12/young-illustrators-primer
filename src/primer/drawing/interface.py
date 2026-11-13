"""The drawing contract: the Protocol serving imports, plus a Fake for tests.

A model area never imports ``primer.serving``; serving imports the area. Importing this module must
stay cheap and must not load weights -- ``model.py`` does that, lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Prediction:
    """One candidate label for a drawing."""

    label: str
    confidence: float


@runtime_checkable
class DrawingRecognizer(Protocol):
    """Recognise what a child drew. Gated on `drawing_top5_acc`."""

    def top_k(self, image: bytes, k: int = 5) -> list[Prediction]:
        """The k most likely labels, most likely first."""
        ...


class FakeDrawingRecognizer:
    """Returns `label` first, then filler. No weights, no network."""

    def __init__(self, label: str = "cat") -> None:
        self.label = label

    def top_k(self, image: bytes, k: int = 5) -> list[Prediction]:
        filler = ["dog", "bird", "fish", "tree", "house"]
        labels = [self.label, *[f for f in filler if f != self.label]][:k]
        return [Prediction(label=name, confidence=1.0 - index * 0.1) for index, name in enumerate(labels)]
