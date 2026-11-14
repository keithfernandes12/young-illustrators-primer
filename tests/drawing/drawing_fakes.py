"""The recording encoder the drawing tests run against. No weights, no network, no `torch`.

The recogniser's one dependency on a checkpoint sits behind `model.Encoder`, so every test injects
this instead. That is also what proves the seam is real: if `analyse` ever reached a tower
directly, these tests would need a 1.6 GB download to run.

A module rather than a conftest fixture because the tests use `RecordingEncoder` as a class --
several build one per case inside a loop.
"""

from __future__ import annotations

from collections.abc import Sequence

from primer.drawing.model import Encoding

#: A short vocabulary for tests that assert on specific labels. The real file has 345.
TEST_LABELS = ("cat", "dog", "house", "dragon", "tree", "sun", "boat")


class RecordingEncoder:
    """Returns the scores it was given and remembers how it was called.

    ``scores`` is positional, like the real encoding: index *i* is the prompt at index *i*. A test
    that wants "dog" to win puts the highest number at the index of "dog", which is exactly the
    pairing a reordering bug would break.
    """

    def __init__(self, scores: Sequence[float], dimensions: int = 4) -> None:
        self.scores = tuple(scores)
        self.dimensions = dimensions
        self.calls: list[tuple[bytes, tuple[str, ...]]] = []

    def encode(self, image: bytes, prompts: Sequence[str]) -> Encoding:
        self.calls.append((image, tuple(prompts)))
        return Encoding(
            confidences=self.scores,
            # Deterministic and already normalised in magnitude, so a test can assert on it.
            embedding=tuple(round(1 / (index + 2), 8) for index in range(self.dimensions)),
        )

    @property
    def prompts(self) -> tuple[str, ...]:
        """The prompts of the last call."""
        return self.calls[-1][1]
