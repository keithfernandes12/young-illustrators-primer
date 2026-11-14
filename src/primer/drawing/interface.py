"""The drawing contract: the Protocol serving imports, plus a Fake for tests.

A model area never imports ``primer.serving``; serving imports the area. Importing this module must
stay cheap and must not load weights -- ``model.py`` does that, lazily.

The recogniser has two outputs, and both are consumed downstream:

- a **description** of the drawing, which the story writer needs to write about what the child
  actually drew rather than about the prompt alone;
- an **embedding per main character**, which the illustration area conditions on to keep a
  character looking like itself across pages (KAN-42: IP-Adapter on the approved reference).

A character also carries a label, because `drawing_top5_acc` -- the gated metric in
`.claude/skills/golden-gate/thresholds.json` -- is still a classification accuracy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Character:
    """One character found in a drawing.

    ``embedding`` is the identity signal the illustrator conditions on. It is derived from the
    child's drawing, so treat it as child data: never log it, never put it in a URL or an object
    name, and do not persist it outside the paths the parent opted into.
    """

    label: str
    confidence: float
    embedding: tuple[float, ...] = ()

    @property
    def has_embedding(self) -> bool:
        return bool(self.embedding)

    @property
    def identity_key(self) -> str:
        """An opaque, stable handle for this character's identity.

        Downstream code needs to know that two pages were conditioned on the *same* character; it
        does not need the vector to answer that. Passing this instead keeps the child-derived
        embedding in the one place that conditions on it, rather than copying it onto every page
        of the finished book.
        """
        digest = hashlib.sha256(repr((self.label, self.embedding)).encode()).hexdigest()
        return f"{self.label}:{digest[:16]}"


@dataclass(frozen=True)
class DrawingAnalysis:
    """Everything the recogniser returns for one drawing."""

    #: What the whole drawing shows, in words the story writer can use.
    description: str = ""
    #: The main characters, most confident first.
    characters: tuple[Character, ...] = field(default_factory=tuple)

    @property
    def main(self) -> Character | None:
        """The most confident character, or None for a drawing with nothing in it."""
        return self.characters[0] if self.characters else None

    def top_labels(self, k: int = 5) -> list[str]:
        """The k most confident labels, for `drawing_top5_acc`."""
        return [character.label for character in self.characters[:k]]


@runtime_checkable
class DrawingRecognizer(Protocol):
    """Recognise what a child drew. Gated on `drawing_top5_acc`."""

    def analyse(self, image: bytes) -> DrawingAnalysis:
        """Describe the drawing and extract its main characters, most confident first."""
        ...


class FakeDrawingRecognizer:
    """Returns `labels` as characters with deterministic embeddings. No weights, no network."""

    def __init__(self, label: str = "cat", also: tuple[str, ...] = (), dimensions: int = 4) -> None:
        self.labels = (label, *also)
        self.dimensions = dimensions

    def _embedding(self, label: str) -> tuple[float, ...]:
        """Stable per label, so two pages of the same character compare equal."""
        seed = sum(ord(character) for character in label)
        return tuple(round(((seed + index) % 100) / 100, 2) for index in range(self.dimensions))

    def analyse(self, image: bytes) -> DrawingAnalysis:
        characters = tuple(
            Character(label=label, confidence=round(1.0 - index * 0.1, 2), embedding=self._embedding(label))
            for index, label in enumerate(self.labels)
        )
        described = " and a ".join(self.labels)
        return DrawingAnalysis(description=f"a child's drawing of a {described}", characters=characters)
