"""Datasets for the drawing stage.

``read_dataset`` returns train / val / test for a frozen data version. Splitting is by book, never
by page (`AGENTS.md`), so pages of one book never land in two splits.
"""

from __future__ import annotations

from typing import Any


def read_dataset(version: str, split: str = "train") -> Any:
    """Load the pinned dataset version. Never reads unfrozen data."""
    raise NotImplementedError("drawing/data.py is a stub; see docs/engineering/model-development.md")
