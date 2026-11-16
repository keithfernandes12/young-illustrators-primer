"""Book-level train / validation / test splits that every model area shares (KAN-120).

A book's split comes from a hash of its key and nothing else: not from which other books exist,
not from any label such as reading level, and not from a seed. So the same book lands in the same
split on every machine, in every run, in every model area, and when the data grows. Pages of one
book never straddle splits, as the review checklist requires ("split by book, never by page").

**The key** is ``book_key(source, book_id)``, ``"<source>/<book_id>"``. A ``book_id`` is only unique
within its source, so the source is part of the key. ``storybook_pages`` and the illustration area
already key books this way.

**The hash** is the first 8 bytes of SHA-256 read as an unsigned integer, divided by 2**64: a
number in [0, 1) that the fractions are laid along, train first. It is the same hash as
``primer.illustration.consistency.experiment.split_of`` and ``primer.story.data.hash_fraction``, so
both areas' existing assignments are reproduced exactly (tested), and they can switch to this module
without moving a single book.

Model areas must not import each other, which is why this lives in ``primer.data``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

SPLIT_NAMES = ("train", "val", "test")


def book_key(source: str, book_id: str) -> str:
    """The split key for a book: ``"<source>/<book_id>"``."""
    return f"{source}/{book_id}"


def hash_fraction(key: str, salt: str = "") -> float:
    """A stable number in [0, 1) for ``key``: the first 8 bytes of SHA-256 of ``salt + key``, over 2**64.

    ``salt`` gives an independent draw from the same key (the story area salts with ``"pool:"`` to
    pick a TinyStories pool that is independent of the split). Leave it empty for splits.
    """
    return int.from_bytes(hashlib.sha256((salt + key).encode()).digest()[:8], "big") / 2**64


@dataclass(frozen=True)
class SplitFractions:
    """The share of books in each split. Each is between 0 and 1, and together they sum to 1.

    A share may be 0: the illustration area uses ``SplitFractions(train=0.0, val=0.5, test=0.5)``,
    since it trains nothing.
    """

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        for name in SPLIT_NAMES:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a share between 0 and 1, not {value!r}")
            object.__setattr__(self, name, float(value))
        total = math.fsum(getattr(self, name) for name in SPLIT_NAMES)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"train + val + test must be 1, not {total}")

    def upper_bounds(self) -> tuple[tuple[str, float], ...]:
        """Each split with the top of its range, train first.

        Rounded to 12 decimals so float addition can't move a boundary: 0.8 + 0.1 gives
        0.9000000000000001, and the story area's existing table says 0.9. The last split always
        ends at exactly 1.0, so every fraction in [0, 1) has a split.
        """
        bounds, total = [], 0.0
        for name in SPLIT_NAMES:
            total = math.fsum([total, getattr(self, name)])
            bounds.append((name, round(total, 12)))
        bounds[-1] = (SPLIT_NAMES[-1], 1.0)
        return tuple(bounds)


DEFAULT_FRACTIONS = SplitFractions()


def split_of(key: str, fractions: SplitFractions = DEFAULT_FRACTIONS) -> str:
    """``"train"``, ``"val"`` or ``"test"`` for the book with this key. Depends on nothing else."""
    position = hash_fraction(key)
    return next(name for name, upper in fractions.upper_bounds() if position < upper)
