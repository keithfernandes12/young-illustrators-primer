"""The shared book-level train / validation / test split, for model areas that split books (KAN-120).

A book's split comes from a hash of its key and nothing else: not from which other books exist,
not from any label such as reading level, and not from a seed. So the same book lands in the same
split on every machine, in every run, and when the data grows, and pages of one book never straddle
splits (AGENTS.md: "split by book, never by page"). Areas that use the same fractions also put a
book in the same split as each other. Different fractions lay the same hash out differently: a
book at 0.3 is train under 80/10/10 but val under illustration's 0/50/50. That's harmless there,
because illustration trains nothing.

**Who uses it.** The story and illustration areas split books, and each still has its own copy
(``primer.story.data.split_of_book``, ``primer.illustration.consistency.experiment.split_of``). They
move to this module when their owners switch over. The drawing area's ``split_of`` is different: it
splits individual drawings with its own salt, so it isn't a book split.

**The key** is ``book_key(source, book_id)``, ``"<source>/<book_id>"``. A ``book_id`` is only unique
within its source, so the source is part of the key. The story and illustration areas already build
book keys this way.

**The hash** is the first 8 bytes of SHA-256 read as an unsigned integer, divided by 2**64: a
number in [0, 1) that the fractions are laid along, train first. It is the same hash as both areas'
copies, so their existing assignments are reproduced exactly (tested), and switching moves no book.

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

    ``salt`` gives an independent draw from the same key. For example, the story area orders its
    validation requests with ``"requests:"``, and its plan picks a TinyStories pool with ``"pool:"``,
    so neither depends on the split. Leave it empty for splits.
    """
    return int.from_bytes(hashlib.sha256((salt + key).encode()).digest()[:8], "big") / 2**64


@dataclass(frozen=True)
class SplitFractions:
    """The share of books in each split. Each is between 0 and 1, and together they sum to 1.

    A share may be 0. The illustration area's own ``split_of(key, val_fraction=0.5)`` is equivalent
    to ``SplitFractions(train=0.0, val=0.5, test=0.5)``, since it trains nothing.
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

        Rounded to 12 decimals so float addition can't move a boundary: 0.6 + 0.3 gives
        0.8999999999999999, not 0.9, so without rounding a book hashed at exactly 0.8999999999999999
        would land in test instead of val. The last split always ends at exactly 1.0, so every
        fraction in [0, 1) has a split.
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
