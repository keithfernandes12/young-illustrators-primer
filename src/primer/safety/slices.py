"""Slice keys for the safety stage, reusing ``primer.data.bias``.

Every gated metric is also reported per slice, so a model that is good on average and bad for one
group is visible rather than averaged away.
"""

from __future__ import annotations

#: Slice keys this stage reports. Extend with the keys that matter for safety.
SLICE_KEYS: tuple[str, ...] = ()


def slice_of(record: object, key: str) -> str | None:
    """The value of `key` for `record`, or None when the record does not carry it."""
    raise NotImplementedError("safety/slices.py is a stub")
