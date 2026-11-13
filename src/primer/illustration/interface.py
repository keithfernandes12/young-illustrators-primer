"""The illustration contract: the Protocol serving imports, plus a Fake for tests.

A model area never imports ``primer.serving``; serving imports the area. Importing this module must
stay cheap and must not load weights -- ``model.py`` does that, lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Illustration:
    """One page illustration. `image_ref` is a URI or cache key, never image bytes."""

    page_index: int
    image_ref: str
    alt_text: str


@runtime_checkable
class Illustrator(Protocol):
    """Illustrate pages and keep the character consistent. Gated on `clip_consistency`."""

    def illustrate(self, page_index: int, page_text: str, style: str) -> Illustration:
        """One illustration for one page."""
        ...

    def consistency(self, illustrations: list[Illustration]) -> float:
        """How alike the illustrations are, 0..1. The whole point of a picture book."""
        ...


class FakeIllustrator:
    """Deterministic refs and a fixed consistency score. Produces no images."""

    def __init__(self, consistency_score: float = 0.9) -> None:
        self.consistency_score = consistency_score

    def illustrate(self, page_index: int, page_text: str, style: str) -> Illustration:
        return Illustration(
            page_index=page_index,
            image_ref=f"fake://{style}/page-{page_index}",
            alt_text=page_text[:60],
        )

    def consistency(self, illustrations: list[Illustration]) -> float:
        return self.consistency_score if illustrations else 0.0
