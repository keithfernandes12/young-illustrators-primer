"""Slice keys for the story stage, and per-slice metrics.

Every gated metric is also reported per slice, so a model that is good on average and bad for one
group is visible rather than averaged away. The keys for this submission:

- `source` -- picture-book origin (`african_storybook`, `storyweaver`) or `tinystories`;
- `age` -- 3, 4 or 5.

`primer.data.bias` judges a slice only at 100 rows or more (`SMALL_SLICE`). The Mac submission has
20-30 requests, so **every slice here is below that floor**: the numbers are reported, and each row
says `judged: false`, rather than presenting a 10-request difference as a finding. Gender-swapped
prompts (the slice `model-development` names for stage 2) need paired requests and are deferred
(`CARD.md`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from primer.data.bias import SMALL_SLICE

#: Slice keys this stage reports.
SLICE_KEYS: tuple[str, ...] = ("source", "age")


def slice_of(record: object, key: str) -> str | None:
    """The value of `key` for `record`, or None when the record does not carry it."""
    value = getattr(record, key, None)
    return None if value is None else str(value)


def slice_metrics(results: Sequence[Any], summarise: Any) -> list[dict[str, Any]]:
    """`[{key, value, n, judged, metrics}]` for every slice key and value present in `results`."""
    rows = []
    for key in SLICE_KEYS:
        groups: dict[str, list[Any]] = {}
        for result in results:
            value = slice_of(result, key)
            if value is not None:
                groups.setdefault(value, []).append(result)
        for value in sorted(groups):
            members = groups[value]
            rows.append({"key": key, "value": value, "n": len(members), "judged": len(members) >= SMALL_SLICE,
                         "metrics": summarise(members)})
    return rows
