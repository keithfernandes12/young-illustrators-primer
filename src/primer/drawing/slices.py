"""Slice keys for the drawing stage, reusing ``primer.data.bias``.

Every gated metric is also reported per slice, so a model that is good on average and bad for one
group is visible rather than averaged away. A slice under ``SMALL_SLICE`` drawings is reported but
not judged -- the same floor the data pipeline's bias report and the speech stage use.

``country_code`` is the one key Quick, Draw! carries that says something about who drew: the
pipeline's bias report already measures Google's own recogniser per country
(``bias.recognition_by_country``), and this is the same question asked of ours.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from primer.data.bias import SMALL_SLICE, stratify_key

#: Slice keys this stage reports.
SLICE_KEYS: tuple[str, ...] = ("country_code",)


def slice_values(frame: pd.DataFrame, key: str) -> pd.Series:
    """Each row's value of ``key``, with a missing value as ``"unknown"`` (``bias.stratify_key``).

    A frame without the column raises rather than putting every drawing in one slice: a
    breakdown with a single group in it would look like a measured slice and be none.
    """
    if key not in frame.columns:
        raise KeyError(f"cannot slice by {key!r}: the frame has no such column")
    return stratify_key(frame, [key])


def slice_of(record: Mapping[str, Any], key: str) -> str | None:
    """The value of ``key`` for one record (a row as a mapping), or None when it does not carry it."""
    value = record.get(key)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def judged(slices: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The slices with enough drawings to draw a conclusion from."""
    return [s for s in slices if s["n"] >= SMALL_SLICE]
