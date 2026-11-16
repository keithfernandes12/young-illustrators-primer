"""Statistics and schema generation, and anomaly detection over time.

For every processed dataset, after each pipeline run:

1. **Statistics** (``profile_frame``): rows, and per column its type, null fraction and either
   numeric statistics, text lengths, or the most common values. Saved to
   ``metadata/profiles/<dataset>/<run_id>.json``, so there is a history over time.
2. **Schema** (``infer_schema``): the expected columns, types, null limits, value ranges, minimum
   text lengths and allowed values, inferred once from good data with
   ``primer-data profile --write-schemas`` and then reviewed and committed in
   ``configs/schemas/<dataset>.json`` (like a TFDV schema).
3. **Validation** (``validate``): the schema becomes a Great Expectations suite, run on the
   current data. A failed expectation is an **error**: the data broke its contract. The top of a
   numeric range is the exception (``range_warnings``): growing sources legitimately go past what
   was seen when the schema was inferred, so exceeding it is a **warning**.
4. **Drift** (``drift``): the new profile is compared with the previous one; a large drop in rows,
   a jump in nulls, a shift in a numeric mean or a text length, a change in category shares, or a
   column crossing between categorical and open-ended is a **warning**.

``check_datasets`` runs all four and returns the anomalies. The Airflow DAG fails the run on
errors and emails the anomalies (see ``alerts.py``).

**Long text stays out of memory** (``load_dataset``, KAN-91). A text column whose mean length is
over ``LONG_TEXT`` (stories, Quick, Draw! strokes) is never held whole: it is read a chunk at a time
for exact statistics (nulls, lengths), and its contract (not null, not empty) is checked from those.
Every other column is read in full and validated by Great Expectations as before. Given all 1.8M
stories, Great Expectations alone needed over 7 GB and was killed inside Airflow.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from primer.data.catalog import DATASETS, dataset_files
from primer.data.config import REPO_ROOT
from primer.data.sources.amateur_drawings import N_KEYPOINTS
from primer.data.storage import Store

log = logging.getLogger(__name__)

SCHEMAS_DIR = REPO_ROOT / "configs" / "schemas"
PROFILES_DIR = "metadata/profiles"
ANOMALIES_DIR = "metadata/anomalies"

CATEGORICAL_MAX = 25         # a text column with at most this many distinct values is categorical
MIN_PER_CATEGORY = 20        # ...and gets an allowed set only if each value appears this often on average
OPEN_COLUMNS = {"batch_id", "translator"}  # new batches and new names arrive with every ingest
# Free-text credits a source may legitimately leave empty: a book with no "# " title line, an
# empty "* Illustration: []" credit, an empty caption cell. Empty there is not missing content.
CREDIT_COLUMNS = {"title", "author", "illustrator", "translator", "image_caption"}
LONG_TEXT = 200              # mean length above this: skip distinct counting, and stream it (stories, strokes)
TOP_VALUES = 10
# Numeric columns whose valid range is fixed by what they count, not by the data seen so far. Their
# contract uses these bounds instead of the observed range widened by 10% (KAN-90). n_keypoints counts
# the visible joints out of COCO's 17: every Amateur Drawings row has all 17 today, which would infer
# 15.3-18.7, and a later release with hidden joints is valid data, not a broken contract.
KNOWN_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "character_drawings": {"n_keypoints": (0.0, float(N_KEYPOINTS))},
}
# Rows per batch when streaming long text. It never changes results. It bounds the batches handed to
# the statistics, but pyarrow may still decode a whole row group of the column at once, so peak
# memory also depends on the files' row-group size (see the measured figures in dags/README.md).
TEXT_BATCH_ROWS = 50_000

# Drift thresholds (warnings, not errors).
ROW_DROP = 0.05              # rows fell by more than 5%
NULL_JUMP = 0.05             # null fraction rose by more than 5 points
MEAN_SHIFT_SD = 0.5          # numeric mean moved by more than half a standard deviation
LENGTH_SHIFT = 0.3           # mean text length changed by more than 30%
SHARE_SHIFT = 0.15           # total variation distance between category shares


# ------------------------------------------------------------------ statistics

def _sig(x: float, digits: int = 6, direction: str = "nearest") -> float:
    """Round to significant figures rather than decimal places, so tiny values (one null in millions
    of rows, a probability column) don't collapse to 0. "down"/"up" round outward, for bounds that
    must still contain the value they were inferred from."""
    if abs(x) < 1e-300 or not math.isfinite(x):   # 10 ** 300+ would overflow
        return float(x)
    scale = 10.0 ** (digits - 1 - math.floor(math.log10(abs(x))))
    scaled = round(x * scale, 6)   # drop float noise first, so 0.15 doesn't ceil to 0.1501
    if direction == "down":
        return math.floor(scaled) / scale
    if direction == "up":
        return math.ceil(scaled) / scale
    return round(scaled) / scale


def _kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "string"


def profile_column(series: pd.Series) -> dict[str, Any]:
    kind = _kind(series)
    nf = float(series.isna().mean())
    # Round toward 0.5, so only exactly none or exactly all null reads as 0 or 1.
    out: dict[str, Any] = {"kind": kind, "null_fraction": _sig(nf, direction="down" if nf > 0.5 else "up")}
    values = series.dropna()
    if values.empty:
        return out
    if kind in ("int", "float"):
        numbers = values.astype(float)
        out.update(min=float(numbers.min()), max=float(numbers.max()), mean=float(numbers.mean()),
                   std=float(numbers.std(ddof=0)), p01=float(numbers.quantile(0.01)),
                   p50=float(numbers.quantile(0.5)), p99=float(numbers.quantile(0.99)))
    elif kind == "bool":
        out["true_fraction"] = round(float(values.astype(bool).mean()), 6)
    elif kind == "datetime":
        out.update(min=str(values.min()), max=str(values.max()))
    else:
        lengths = values.astype(str).str.len()
        out.update(mean_length=float(lengths.mean()), min_length=int(lengths.min()),
                   max_length=int(lengths.max()))
        if out["mean_length"] <= LONG_TEXT:
            counts = values.astype(str).value_counts()
            out["distinct"] = int(counts.size)
            if counts.size <= CATEGORICAL_MAX:
                out["shares"] = {str(k): round(float(v), 6) for k, v in (counts / counts.sum()).items()}
            else:
                out["top"] = {str(k): round(float(v), 6)
                              for k, v in (counts.head(TOP_VALUES) / counts.sum()).items()}
    return out


def profile_frame(df: pd.DataFrame) -> dict[str, Any]:
    return {"rows": len(df), "columns": {col: profile_column(df[col]) for col in df.columns}}


# ------------------------------------------------------------------ loading without the long text

@dataclass
class TextStats:
    """Exact statistics of one text column, gathered a chunk at a time."""

    rows: int = 0
    nulls: int = 0
    total_length: int = 0
    min_length: int | None = None
    max_length: int | None = None
    # Non-null values shorter than the schema's ``min_length``. None means "not counted" (no
    # ``min_lengths`` given for this column), which ``text_contract`` refuses to read as "none".
    too_short: int | None = None

    @property
    def non_null(self) -> int:
        return self.rows - self.nulls

    @property
    def mean_length(self) -> float:
        return self.total_length / self.non_null if self.non_null else 0.0

    def profile(self) -> dict[str, Any]:
        """The entry ``profile_column`` gives a long text column, from the statistics alone."""
        nf = self.nulls / self.rows
        return {"kind": "string", "null_fraction": _sig(nf, direction="down" if nf > 0.5 else "up"),
                "mean_length": self.mean_length, "min_length": self.min_length, "max_length": self.max_length}


def _is_text(field_type) -> bool:
    import pyarrow as pa

    return bool(pa.types.is_string(field_type) or pa.types.is_large_string(field_type)
                or getattr(pa.types, "is_string_view", lambda _: False)(field_type))


@dataclass
class _Scan:
    stats: dict[str, TextStats]      # every column that is text in at least one file
    rows: int
    columns: list[str]               # every column, in file order
    not_text: set[str]               # columns with a non-text, non-null type in at least one file


def _scan_text(store: Store, paths: list[str], min_lengths: dict[str, int]) -> _Scan:
    """Statistics of every text column across ``paths``, one chunk in memory at a time.

    A column missing from some files counts as null there, as it would in the concatenated frame.
    A column that is another type in some file is listed in ``not_text``: its statistics here
    would miss those values, so ``load_dataset`` loads it whole instead.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    stats: dict[str, TextStats] = {}
    columns: list[str] = []
    not_text: set[str] = set()
    rows = 0
    for path in paths:
        with store.fs.open(store.path(path), "rb") as fh:
            parquet = pq.ParquetFile(fh)
            rows += parquet.metadata.num_rows
            columns += [f.name for f in parquet.schema_arrow if f.name not in columns]
            not_text |= {f.name for f in parquet.schema_arrow
                         if not _is_text(f.type) and not pa.types.is_null(f.type)}
            names = [f.name for f in parquet.schema_arrow if _is_text(f.type)]
            for batch in parquet.iter_batches(batch_size=TEXT_BATCH_ROWS, columns=names) if names else ():
                for name in names:
                    column = batch.column(name)
                    lengths = pc.utf8_length(column)
                    s = stats.setdefault(name, TextStats(too_short=0 if name in min_lengths else None))
                    s.rows += len(column)
                    s.nulls += column.null_count
                    if column.null_count == len(column):
                        continue
                    low, high = pc.min(lengths).as_py(), pc.max(lengths).as_py()
                    s.total_length += pc.sum(lengths).as_py()
                    s.min_length = low if s.min_length is None else min(s.min_length, low)
                    s.max_length = high if s.max_length is None else max(s.max_length, high)
                    if s.too_short is not None:
                        s.too_short += pc.sum(pc.less(lengths, min_lengths[name])).as_py() or 0
    for s in stats.values():
        s.nulls += rows - s.rows    # files without the column
        s.rows = rows
    return _Scan(stats, rows, columns, not_text)


@dataclass
class LoadedDataset:
    """A dataset with its long text columns held as statistics only."""

    frame: pd.DataFrame              # every column except the long text ones
    text: dict[str, TextStats]       # the long text columns
    rows: int
    columns: list[str]               # every column, in file order

    def profile(self) -> dict[str, Any]:
        """Equal to ``profile_frame`` of the whole dataset (tested), without loading the long text."""
        return {"rows": self.rows, "columns": {
            name: self.text[name].profile() if name in self.text else profile_column(self.frame[name])
            for name in self.columns}}


def load_dataset(store: Store, dataset: str, min_lengths: dict[str, int] | None = None) -> LoadedDataset:
    """Read a dataset like ``read_dataset`` does, except that text columns averaging over
    ``LONG_TEXT`` characters are streamed for statistics instead of loaded.

    ``min_lengths`` (from the schema, see ``min_lengths``) lets the stream also count values that
    are too short, so the contract can be checked without the text.
    """
    import pyarrow.parquet as pq

    paths = [p for group in dataset_files(store, dataset).values() for p in group]
    scan = _scan_text(store, paths, min_lengths or {})
    text = {name: s for name, s in scan.stats.items()
            if s.non_null and s.mean_length > LONG_TEXT and name not in scan.not_text}
    frames = []
    for path in paths:
        with store.fs.open(store.path(path), "rb") as fh:
            keep = [name for name in pq.read_schema(fh).names if name not in text]
            fh.seek(0)
            # pd.read_parquet, as Store.read_parquet uses, so the dtypes match read_dataset's
            frames.append(pd.read_parquet(fh, columns=keep))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return LoadedDataset(frame, text, scan.rows, scan.columns)


def min_lengths(schema: dict[str, Any] | None) -> dict[str, int]:
    """The schema's minimum text lengths, for ``load_dataset`` to count values below them."""
    return {name: spec["min_length"] for name, spec in (schema or {}).get("columns", {}).items()
            if "min_length" in spec}


# ------------------------------------------------------------------ schema

def infer_schema(dataset: str, profile: dict[str, Any], min_rows_fraction: float = 0.5) -> dict[str, Any]:
    """Expected contract for a dataset, inferred from one good profile. Review before committing.

    Numeric ranges are the observed range widened by 10% on each side, except that a non-negative
    column never gets a negative floor, and gets a floor of 0 only if 0 was observed (so a count
    collapsing to zero is caught). A column in ``KNOWN_BOUNDS`` gets those bounds instead, because
    what it counts fixes its range (``n_keypoints``: 0 to 17 visible joints). Text columns that were
    never empty must stay non-empty.
    Categorical columns get their observed values as the allowed set (unless the values are too
    rare to be real categories, like a handful of translator names); ``min_rows`` guards against a
    dataset shrinking.

    Null limits are the observed fraction plus 5 points. A column too null for that is marked
    ``not_all_null`` (it must keep at least one value), and one that was entirely null is marked
    ``nulls_unconstrained``, so "may be all null" is a visible decision, never a side effect.
    """
    columns: dict[str, Any] = {}
    for name, col in profile["columns"].items():
        spec: dict[str, Any] = {"kind": col["kind"]}
        nf = col["null_fraction"]
        limit = _sig(nf + 0.05, 4, "up")   # 5 points of headroom
        if nf == 0:
            spec["max_null_fraction"] = 0.0
        elif limit < 1:
            spec["max_null_fraction"] = limit
        elif nf < 1:
            # Mostly null, e.g. titles in stories_text, where TinyStories (no titles) grows every run:
            # any fraction would eventually fail on good data, so only "entirely null" is an error.
            spec["max_null_fraction"] = 1.0
            spec["not_all_null"] = True
        else:
            spec["max_null_fraction"] = 1.0
            spec["nulls_unconstrained"] = True
        if col["kind"] in ("int", "float") and "min" in col:
            pad = 0.1 * (col["max"] - col["min"]) or abs(col["max"]) * 0.1 or 1.0
            low = col["min"] - pad
            if col["min"] >= 0 and low < 0:
                # Counts and lengths can't go negative, and 0 is allowed only if it was seen. Any
                # value from 1 up stays valid: growing sources bring smaller counts than seen so far.
                low = min(col["min"] * 0.9, 1.0)
            spec["min"], spec["max"] = _sig(low, direction="down"), _sig(col["max"] + pad, direction="up")
            if name in KNOWN_BOUNDS.get(dataset, {}):
                spec["min"], spec["max"] = KNOWN_BOUNDS[dataset][name]
        if col["kind"] == "string" and col.get("min_length", 0) >= 1 and name not in CREDIT_COLUMNS:
            # Not empty; not "as long as before", since new books bring shorter pages. Truncated text
            # is drift (length_shift), not a contract break.
            spec["min_length"] = 1
        non_null = profile["rows"] * (1 - nf)
        if "shares" in col and name not in OPEN_COLUMNS and non_null >= MIN_PER_CATEGORY * len(col["shares"]):
            spec["allowed"] = sorted(col["shares"])
        columns[name] = spec
    return {"dataset": dataset, "min_rows": int(profile["rows"] * min_rows_fraction),
            "created_at": datetime.now(UTC).isoformat(), "columns": columns}


def load_schema(dataset: str, schemas_dir: Path = SCHEMAS_DIR) -> dict[str, Any] | None:
    path = Path(schemas_dir) / f"{dataset}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ------------------------------------------------------------------ anomalies

@dataclass
class Anomaly:
    dataset: str
    severity: str        # "error" (contract broken) or "warning" (drift)
    check: str
    column: str | None
    detail: str


def build_suite(schema: dict[str, Any]):
    os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
    import great_expectations as gx

    suite = gx.ExpectationSuite(name=f"{schema['dataset']}_contract")
    cols = schema["columns"]
    suite.add_expectation(gx.expectations.ExpectTableColumnsToMatchSet(column_set=list(cols), exact_match=True))
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=schema["min_rows"]))
    for name, spec in cols.items():
        if spec["max_null_fraction"] < 1.0:
            suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(
                column=name, mostly=_sig(1.0 - spec["max_null_fraction"], direction="down")))
        elif spec.get("not_all_null"):
            # mostly > 0: passes with a single value in any realistic table, fails only when all null
            suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=name, mostly=1e-9))
        if "min" in spec:
            # Only the floor is a contract; going above "max" is a warning (see range_warnings).
            suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column=name, min_value=spec["min"]))
        if "min_length" in spec:
            suite.add_expectation(gx.expectations.ExpectColumnValueLengthsToBeBetween(
                column=name, min_value=spec["min_length"]))
        if "allowed" in spec:
            suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column=name, value_set=spec["allowed"]))
    return suite


def _gx_ready(df: pd.DataFrame) -> pd.DataFrame:
    """GX handles plain numpy dtypes best: nullable ints/bools/strings become object/float."""
    out = df.copy()
    for col in out.columns:
        kind = _kind(out[col])
        if kind in ("int", "float"):
            out[col] = out[col].astype("float64")
        elif kind in ("bool", "string"):
            out[col] = out[col].astype(object).where(out[col].notna(), None)
    return out


def gx_context():
    """One in-memory Great Expectations context, quiet (no progress bars, analytics or INFO logs)."""
    os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
    logging.getLogger("great_expectations").setLevel(logging.WARNING)
    import great_expectations as gx

    context = gx.get_context(mode="ephemeral")
    try:
        from great_expectations.data_context.types.base import ProgressBarsConfig
        context.variables.progress_bars = ProgressBarsConfig(globally=False)
    except Exception:  # cosmetic only: progress bars stay on
        log.debug("Could not turn off Great Expectations progress bars", exc_info=True)
    context.data_sources.add_pandas("primer")
    return context


def validate(df: pd.DataFrame, schema: dict[str, Any], context=None,
             text: dict[str, TextStats] | None = None) -> list[Anomaly]:
    """Run the schema's Great Expectations suite on ``df``; each failed expectation is an error.

    ``text`` holds the long text columns that ``load_dataset`` left out of ``df``. Their part of
    the contract is checked from those statistics instead (``text_contract``), with the same check
    names, and the column-set expectation covers only the columns actually in ``df``.
    """
    text = text or {}
    context = context or gx_context()
    source = context.data_sources.get("primer")
    name = schema["dataset"]
    if name in [a.name for a in source.assets]:
        source.delete_asset(name)
    asset = source.add_dataframe_asset(name=name)
    batch = asset.add_batch_definition_whole_dataframe("all").get_batch(
        batch_parameters={"dataframe": _gx_ready(df)})
    in_frame = {**schema, "columns": {k: v for k, v in schema["columns"].items() if k not in text}}
    result = batch.validate(build_suite(in_frame))
    anomalies = text_contract(schema, text)
    for r in result.results:
        if r.success:
            continue
        cfg = r.expectation_config
        res = r.result or {}
        detail = cfg.type
        if "unexpected_percent" in res:
            detail += f": {res['unexpected_percent']:.2f}% unexpected"
            if res.get("partial_unexpected_list"):
                detail += f" (e.g. {res['partial_unexpected_list'][:3]})"
        elif "observed_value" in res:
            detail += f": observed {res['observed_value']}"
        if r.exception_info and r.exception_info.get("raised_exception"):
            detail += f": {r.exception_info.get('exception_message')}"
        anomalies.append(Anomaly(schema["dataset"], "error", cfg.type, cfg.kwargs.get("column"), detail))
    return _one_column_set_anomaly(anomalies)


def _one_column_set_anomaly(anomalies: list[Anomaly]) -> list[Anomaly]:
    """Merge column-set breaks into one, as the full-frame check reports a single one.

    An unexpected long column (``text_contract``) and a column-set break in the frame (Great
    Expectations) are the same contract failure seen from two sides.
    """
    check = "expect_table_columns_to_match_set"
    breaks = [a for a in anomalies if a.check == check]
    if len(breaks) < 2:
        return anomalies
    merged = Anomaly(breaks[0].dataset, "error", check, None, "; ".join(a.detail for a in breaks))
    first = anomalies.index(breaks[0])
    rest = [a for a in anomalies if a.check != check]
    return rest[:first] + [merged] + rest[first:]


def text_contract(schema: dict[str, Any], text: dict[str, TextStats]) -> list[Anomaly]:
    """The schema's expectations for long text columns, decided from exact statistics.

    The same rules ``build_suite`` gives Great Expectations: the non-null share must reach
    ``mostly``, and no value may be shorter than ``min_length``. (A ``not_all_null`` column always
    passes here: only columns with at least one value are streamed.) A column the schema doesn't
    have breaks the column set. A column the schema expects to be numeric or categorical (``min``,
    ``allowed``) that has turned into long text is reported as that check failing, as Great
    Expectations reports it on the full frame.

    Raises ``ValueError`` if the schema has a ``min_length`` that the stream did not count: that
    would otherwise pass every empty value silently. Load with ``min_lengths(schema)``.
    """
    dataset, out = schema["dataset"], []
    for name, s in text.items():
        spec = schema["columns"].get(name)
        if spec is None:
            out.append(Anomaly(dataset, "error", "expect_table_columns_to_match_set", None,
                               f"expect_table_columns_to_match_set: unexpected column {name!r}"))
            continue
        non_null = s.non_null / s.rows
        if spec["max_null_fraction"] < 1.0:
            mostly = _sig(1.0 - spec["max_null_fraction"], direction="down")
            if non_null < mostly:
                out.append(Anomaly(dataset, "error", "expect_column_values_to_not_be_null", name,
                                   f"expect_column_values_to_not_be_null: {100 * (1 - non_null):.2f}% unexpected"))
        if "min_length" in spec and s.too_short is None:
            raise ValueError(f"{dataset}.{name}: the schema has a min_length but the stream did not count "
                             "short values; load the dataset with min_lengths(schema)")
        if "min_length" in spec and s.too_short:
            out.append(Anomaly(dataset, "error", "expect_column_value_lengths_to_be_between", name,
                               f"expect_column_value_lengths_to_be_between: {100 * s.too_short / s.non_null:.2f}% "
                               f"unexpected (shorter than {spec['min_length']})"))
        for key, check in (("min", "expect_column_values_to_be_between"), ("allowed", "expect_column_values_to_be_in_set")):
            if key in spec:
                out.append(Anomaly(dataset, "error", check, name,
                                   f"{check}: the column is now long text (mean length {s.mean_length:.0f})"))
    return out


def range_warnings(schema: dict[str, Any], profile: dict[str, Any],
                   previous: dict[str, Any] | None = None) -> list[Anomaly]:
    """Warnings for numeric columns reaching a new high above the schema's ``max``.

    The maximum is inferred from the data seen so far, and both incremental sources (more Quick,
    Draw! categories) and snapshot sources (a longer book upstream) legitimately exceed it, so this
    is drift, not a broken contract. It warns once per new high, not on every run until the schema
    is regenerated: a value already above ``max`` in the previous profile was already reported.
    """
    out: list[Anomaly] = []
    prev_cols = (previous or {}).get("columns", {})
    for name, spec in schema["columns"].items():
        col = profile["columns"].get(name, {})
        if "max" not in spec or "max" not in col:
            continue
        ceiling = max(spec["max"], prev_cols.get(name, {}).get("max", spec["max"]))
        if col["max"] > ceiling:
            out.append(Anomaly(schema["dataset"], "warning", "above_range", name,
                               f"max {col['max']:.6g} is above the schema's {spec['max']:.6g}"))
    return out


def drift(dataset: str, previous: dict[str, Any], current: dict[str, Any]) -> list[Anomaly]:
    """Warnings for large changes between two profiles of the same dataset."""
    out: list[Anomaly] = []
    if previous["rows"] and current["rows"] < previous["rows"] * (1 - ROW_DROP):
        out.append(Anomaly(dataset, "warning", "row_drop", None,
                           f"rows fell from {previous['rows']:,} to {current['rows']:,}"))
    for name, cur in current["columns"].items():
        prev = previous["columns"].get(name)
        if prev is None:
            continue
        if cur["null_fraction"] - prev["null_fraction"] > NULL_JUMP:
            out.append(Anomaly(dataset, "warning", "null_jump", name,
                               f"nulls {prev['null_fraction']:.1%} → {cur['null_fraction']:.1%}"))
        if "mean" in cur and "mean" in prev and prev.get("std") is not None:
            sd = prev["std"]
            shift = abs(cur["mean"] - prev["mean"])
            if sd > 0 and shift > MEAN_SHIFT_SD * sd:
                out.append(Anomaly(dataset, "warning", "mean_shift", name,
                                   f"mean {prev['mean']:.3g} → {cur['mean']:.3g} ({shift / sd:.2f} sd)"))
            elif sd == 0 and not math.isclose(cur["mean"], prev["mean"], rel_tol=1e-9, abs_tol=1e-12):
                out.append(Anomaly(dataset, "warning", "mean_shift", name,
                                   f"mean {prev['mean']:.3g} → {cur['mean']:.3g} (was constant)"))
        if "mean_length" in cur and "mean_length" in prev and prev["mean_length"] > 0:
            change = cur["mean_length"] / prev["mean_length"] - 1
            if abs(change) > LENGTH_SHIFT:
                out.append(Anomaly(dataset, "warning", "length_shift", name,
                                   f"mean length {prev['mean_length']:.4g} → {cur['mean_length']:.4g} "
                                   f"({change:+.0%})"))
        if (name not in OPEN_COLUMNS and "distinct" in cur and "distinct" in prev
                and ("shares" in cur) != ("shares" in prev)):
            # Crossed CATEGORICAL_MAX either way: shares can't be compared, but the jump is the news.
            out.append(Anomaly(dataset, "warning", "cardinality_shift", name,
                               f"distinct values {prev['distinct']:,} → {cur['distinct']:,}"))
        if "shares" in cur and "shares" in prev:
            keys = set(cur["shares"]) | set(prev["shares"])
            tvd = 0.5 * sum(abs(cur["shares"].get(k, 0.0) - prev["shares"].get(k, 0.0)) for k in keys)
            if tvd > SHARE_SHIFT:
                out.append(Anomaly(dataset, "warning", "share_shift", name,
                                   f"category shares moved by {tvd:.2f} (total variation)"))
    return out


# ------------------------------------------------------------------ orchestration

def _safe(run_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)


def previous_profile(store: Store, dataset: str, exclude: str | None = None) -> dict[str, Any] | None:
    profiles = []
    for path in store.glob(f"{PROFILES_DIR}/{dataset}/*.json"):
        if path == exclude:
            continue
        try:
            profile = store.read_json(path)
        except ValueError:  # truncated by an interrupted write; must not break every later run
            log.warning("Ignoring unreadable profile %s", path)
            continue
        if isinstance(profile, dict):
            profiles.append(profile)
    return max(profiles, key=lambda p: p.get("created_at", "")) if profiles else None


def _check_dataset(store: Store, dataset: str, run_id: str, created_at: str, save: bool,
                   schemas_dir: Path, context) -> tuple[dict[str, Any] | None, list[Anomaly], Any]:
    """Profile, validate and drift-check one dataset.

    Returns its profile (None if it has no rows and was never profiled, e.g. a source not yet run
    in this data root), its anomalies, and the GX context, created on first use.
    """
    schema = load_schema(dataset, schemas_dir)
    data = load_dataset(store, dataset, min_lengths(schema))
    out_path = f"{PROFILES_DIR}/{dataset}/{_safe(run_id)}.json"
    prev = previous_profile(store, dataset, exclude=out_path)
    if data.rows == 0:
        if not prev:
            return None, [], context
        # Not saved as a profile, so the last good one stays the baseline and every empty run errors.
        return ({"rows": 0, "compared_with": prev.get("run_id")},
                [Anomaly(dataset, "error", "empty_dataset", None,
                         f"no rows; the last check with data ({prev.get('run_id')}) had {prev.get('rows', 0):,}")],
                context)
    log.info("Checking %s (%s rows%s)", dataset, f"{data.rows:,}",
             f"; streaming {', '.join(data.text)}" if data.text else "")
    profile = {"dataset": dataset, "run_id": run_id, "created_at": created_at, **data.profile()}
    anomalies: list[Anomaly] = []
    if schema is None:
        anomalies.append(Anomaly(dataset, "error", "no_schema", None,
                                 f"no configs/schemas/{dataset}.json, so nothing was validated; "
                                 "run `primer-data profile --write-schemas` and commit it"))
    else:
        context = context or gx_context()
        anomalies += validate(data.frame, schema, context, data.text) + range_warnings(schema, profile, prev)
    if prev:
        anomalies += drift(dataset, prev, profile)
    if save:
        store.write_json(out_path, profile)
    return {**profile, "compared_with": prev.get("run_id") if prev else None}, anomalies, context


def check_datasets(store: Store | None = None, run_id: str | None = None, save: bool = True,
                   schemas_dir: Path = SCHEMAS_DIR, datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    """Profile, validate and drift-check every dataset. With ``save``, write profiles and the anomaly
    report to the data root; without it, only read."""
    store = store or Store()
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {"run_id": run_id, "created_at": datetime.now(UTC).isoformat(),
                              "data_uri": store.uri, "datasets": {}, "anomalies": []}
    context = None
    for dataset in datasets:
        profile: dict[str, Any] | None
        try:
            profile, anomalies, context = _check_dataset(store, dataset, run_id, report["created_at"], save,
                                                         schemas_dir, context)
        except Exception as e:  # e.g. a corrupt parquet file: report it, check the rest, write the report
            log.exception("Checking %s failed", dataset)
            profile = {"rows": 0, "compared_with": None}
            anomalies = [Anomaly(dataset, "error", "check_failed", None, f"{type(e).__name__}: {e}")]
        if profile is None:
            continue
        report["datasets"][dataset] = {"rows": profile["rows"], "errors": sum(a.severity == "error" for a in anomalies),
                                       "warnings": sum(a.severity == "warning" for a in anomalies),
                                       "compared_with": profile["compared_with"]}
        report["anomalies"] += [asdict(a) for a in anomalies]
        log.info("%-16s %10s rows, %d errors, %d warnings", dataset, f"{profile['rows']:,}",
                 report["datasets"][dataset]["errors"], report["datasets"][dataset]["warnings"])
    report["errors"] = sum(a["severity"] == "error" for a in report["anomalies"])
    report["warnings"] = sum(a["severity"] == "warning" for a in report["anomalies"])
    if save:
        store.write_json(f"{ANOMALIES_DIR}/{_safe(run_id)}.json", report)
        report["report_uri"] = store.url(f"{ANOMALIES_DIR}/{_safe(run_id)}.json")
    return report


def write_schemas(store: Store | None = None, schemas_dir: Path = SCHEMAS_DIR,
                  datasets: tuple[str, ...] = DATASETS) -> list[Path]:
    """Infer a schema for each dataset from the current data and write it to ``schemas_dir``.

    Only reads the data root. The schemas are meant to be reviewed and committed.
    """
    store = store or Store()
    schemas_dir = Path(schemas_dir)
    schemas_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset in datasets:
        data = load_dataset(store, dataset)
        if data.rows == 0:
            continue
        schema = json.loads(json.dumps(infer_schema(dataset, data.profile()), default=_json_default))
        path = schemas_dir / f"{dataset}.json"
        if path.exists():
            # Keep the old date when nothing else changed, so an unchanged contract has no diff.
            old = json.loads(path.read_text(encoding="utf-8"))
            if {**old, "created_at": None} == {**schema, "created_at": None}:
                schema["created_at"] = old.get("created_at")
        path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8", newline="\n")
        written.append(path)
    return written


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def format_anomalies(report: dict[str, Any], limit: int = 50) -> str:
    """Plain-text summary for logs and alert emails."""
    lines = [(f"Data check {report['run_id']} on {report['data_uri']}: "
              f"{report['errors']} errors, {report['warnings']} warnings"), ""]
    for dataset, d in report["datasets"].items():
        lines.append(f"  {dataset:<16} {d['rows']:>10,} rows  {d['errors']} errors  {d['warnings']} warnings")
    if report["anomalies"]:
        lines.append("")
    for a in report["anomalies"][:limit]:
        col = f" [{a['column']}]" if a["column"] else ""
        lines.append(f"- {a['severity'].upper()} {a['dataset']}{col}: {a['detail']}")
    if len(report["anomalies"]) > limit:
        lines.append(f"... and {len(report['anomalies']) - limit} more")
    if report.get("report_uri"):
        lines += ["", f"Full report: {report['report_uri']}"]
    return "\n".join(lines)
