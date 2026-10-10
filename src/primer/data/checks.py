"""Reusable validation checks over lists of raw records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from primer.data.manifest import Check


def min_records(records: list[dict], minimum: int) -> Check:
    return Check("min_records", len(records) >= minimum, f"{len(records):,} records (minimum {minimum:,})")


def count_between(name: str, count: int, low: int, high: int | None = None) -> Check:
    ok = count >= low and (high is None or count <= high)
    expected = f"{low:,}+" if high is None else (f"{low:,}" if low == high else f"{low:,}-{high:,}")
    return Check(name, ok, f"{count:,} (expected {expected})")


def required_columns(columns: list[str], required: list[str]) -> Check:
    missing = [c for c in required if c not in columns]
    return Check("required_columns", not missing, f"missing: {', '.join(missing)}" if missing else "all present")


def required_fields(records: list[dict], fields: list[str]) -> Check:
    missing = Counter(f for r in records for f in fields if f not in r)
    detail = ", ".join(f"{f} missing in {n:,}" for f, n in missing.items())
    return Check("required_fields", not missing, detail or f"all records have {', '.join(fields)}")


def unique(records: list[dict], field: str) -> Check:
    counts = Counter(r.get(field) for r in records)
    dupes = sum(n - 1 for n in counts.values() if n > 1)
    return Check(f"unique_{field}", dupes == 0, f"{dupes:,} duplicate {field} values")


def max_ratio(name: str, records: list[dict], is_bad: Callable[[dict], bool], limit: float) -> Check:
    """Pass when the share of records where ``is_bad`` is true stays at or below ``limit``."""
    bad = sum(1 for r in records if is_bad(r))
    ratio = bad / len(records) if records else 0.0
    return Check(name, ratio <= limit, f"{bad:,} of {len(records):,} ({ratio:.1%}, limit {limit:.0%})")
