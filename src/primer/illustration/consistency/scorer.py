"""Score how much a book's pages look like the character they were drawn from.

A pretrained image embedder (CLIP by default) embeds the character's reference images and every page,
and the score is cosine similarity in that space. Nothing is trained; what is tuned is the threshold
below which a book counts as inconsistent, which ``experiment.py`` sets on held-out picture books.

Two numbers per book:

- **identity** -- each page against the closest reference image. This is the score by default: six
  identical pages of the *wrong* character must score low, and page-to-page similarity alone would
  score them 1.0.
- **cohesion** -- the pages against each other, reported alongside. Weighting it into the score
  (``identity_weight < 1``) is a sensitivity knob, off by default for two reasons. CLIP cosines are
  compressed (unrelated images still score around 0.75), so cohesion lifts a wrong-character book
  towards the threshold. And the experiment calibrates the threshold on page-to-page pairs, which is
  what identity measures.

This runs once per book in the request path, so a book is embedded in a single batch and the embedder
loads its weights lazily (``embedders.py``), never at import time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

AGGREGATES: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda values: float(np.mean(values)),
    # One off-model page is enough to break a picture book, so `min` is the stricter alternative.
    "min": lambda values: float(np.min(values)),
}


@runtime_checkable
class ImageEmbedder(Protocol):
    """Turns encoded images (PNG, JPEG) into vectors. Images are child data: never log them."""

    #: Short model name, e.g. ``clip-vit-b32``; used in reports and run tags.
    name: str

    def embed(self, images: Sequence[bytes]) -> np.ndarray:
        """One row per image, each of unit length."""
        ...


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length. A zero row stays zero rather than becoming NaN."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


@dataclass(frozen=True)
class ConsistencyResult:
    """One book's score. ``identity`` has one entry per page, in page order."""

    score: float
    identity: tuple[float, ...]
    cohesion: float | None
    threshold: float | None = None

    @property
    def consistent(self) -> bool | None:
        """Whether the book clears the threshold; None while no threshold has been agreed."""
        return None if self.threshold is None else self.score >= self.threshold

    @property
    def weakest_page(self) -> int | None:
        """Index of the page least like the character: the one to regenerate first."""
        return int(np.argmin(self.identity)) if self.identity else None


class ConsistencyScorer:
    """Scores books with one embedder. Satisfies what ``Illustrator.consistency()`` needs."""

    def __init__(
        self,
        embedder: ImageEmbedder,
        threshold: float | None = None,
        identity_weight: float = 1.0,
        aggregate: str = "mean",
    ) -> None:
        if not 0.0 <= identity_weight <= 1.0:
            raise ValueError(f"identity_weight must be between 0 and 1, got {identity_weight}")
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {sorted(AGGREGATES)}, got {aggregate!r}")
        self.embedder = embedder
        self.threshold = threshold
        self.identity_weight = identity_weight
        self.aggregate = aggregate

    def score(self, references: Sequence[bytes], pages: Sequence[bytes]) -> ConsistencyResult:
        """Score a book's pages against the character's reference images.

        With no reference there is nothing to hold the character to, so the book scores 0 -- the same
        rule ``FakeIllustrator.consistency()`` applies to a page with no character embedding.
        """
        if not pages or not references:
            return ConsistencyResult(0.0, tuple(0.0 for _ in pages), None, self.threshold)
        vectors = self.embedder.embed([*references, *pages])
        return self.score_embeddings(vectors[: len(references)], vectors[len(references):])

    def score_embeddings(self, references: np.ndarray, pages: np.ndarray) -> ConsistencyResult:
        """The same score from vectors already computed, e.g. cached per character."""
        refs, pgs = normalise(references), normalise(pages)
        identity = np.clip((pgs @ refs.T).max(axis=1), 0.0, 1.0)
        cohesion = None
        if len(pgs) >= 2:
            similarity = pgs @ pgs.T
            off_diagonal = similarity[~np.eye(len(pgs), dtype=bool)]
            cohesion = round(float(np.clip(off_diagonal.mean(), 0.0, 1.0)), 4)
        held = AGGREGATES[self.aggregate](identity)
        score = held if cohesion is None else self.identity_weight * held + (1 - self.identity_weight) * cohesion
        return ConsistencyResult(
            score=round(score, 4),
            identity=tuple(round(float(value), 4) for value in identity),
            cohesion=cohesion,
            threshold=self.threshold,
        )


def summarise(results: Sequence[ConsistencyResult]) -> dict[str, float]:
    """Book scores → the gated ``clip_consistency`` key, plus the pass rate once a threshold exists.

    ``clip_consistency`` is the mean book score, which is what ``thresholds.json`` compares. The pass
    rate is not a contract key; it belongs in ``CARD.md``.
    """
    if not results:
        return {"clip_consistency": 0.0}
    summary = {"clip_consistency": round(float(np.mean([result.score for result in results])), 4)}
    verdicts = [result.consistent for result in results if result.consistent is not None]
    if verdicts:
        summary["consistency_pass_rate"] = round(sum(verdicts) / len(verdicts), 4)
    return summary


class FakeEmbedder:
    """Deterministic embeddings from byte statistics. No weights, no network.

    Two images with the same mix of byte values get the same vector, so a test can build "pages of the
    same character" and "pages of another character" from plain bytes.
    """

    name = "fake"

    def __init__(self, dimensions: int = 16) -> None:
        self.dimensions = dimensions
        self.calls = 0

    def embed(self, images: Sequence[bytes]) -> np.ndarray:
        self.calls += 1
        rows = np.zeros((len(images), self.dimensions))
        for row, image in enumerate(images):
            values = np.frombuffer(image, dtype=np.uint8).astype(np.int64) % self.dimensions
            rows[row] = np.bincount(values, minlength=self.dimensions)
        return normalise(rows)
