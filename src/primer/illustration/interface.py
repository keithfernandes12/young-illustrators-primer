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
    #: What the page was conditioned on, so consistency can be scored against the drawing.
    subject: str | None = None


@runtime_checkable
class Illustrator(Protocol):
    """Illustrate pages and keep the character consistent. Gated on `clip_consistency`.

    ``subject`` and ``drawing_ref`` are the conditioning signal: the child's own drawing is what
    the illustrations have to look like. Without them an illustrator can only read the page text,
    and drawing-to-pages consistency cannot be scored at all.
    """

    def illustrate(
        self,
        page_index: int,
        page_text: str,
        style: str,
        subject: str | None = None,
        drawing_ref: str | None = None,
    ) -> Illustration:
        """One illustration for one page."""
        ...

    def consistency(self, illustrations: list[Illustration]) -> float:
        """How alike the illustrations are, 0..1. The whole point of a picture book."""
        ...


class FakeIllustrator:
    """Deterministic refs and a fixed consistency score. Produces no images."""

    def __init__(self, consistency_score: float = 0.9) -> None:
        self.consistency_score = consistency_score

    def illustrate(
        self,
        page_index: int,
        page_text: str,
        style: str,
        subject: str | None = None,
        drawing_ref: str | None = None,
    ) -> Illustration:
        return Illustration(
            page_index=page_index,
            image_ref=f"fake://{style}/{subject or 'unknown'}/page-{page_index}",
            alt_text=page_text[:60],
            subject=subject,
        )

    def consistency(self, illustrations: list[Illustration]) -> float:
        if not illustrations:
            return 0.0
        # A real scorer compares images; the fake reports perfect consistency only when every page
        # was conditioned on the same subject, so serving failing to thread it through is visible.
        subjects = {illustration.subject for illustration in illustrations}
        return self.consistency_score if len(subjects) == 1 else 0.0
