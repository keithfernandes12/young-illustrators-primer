"""Statistics and schema generation, and anomaly detection over time.

For every processed dataset, after each pipeline run:

1. **Statistics** (``profile_frame``): rows, and per column its type, null fraction and either
   numeric statistics, text lengths, or the most common values. Saved to
   ``metadata/profiles/<dataset>/<run_id>.json``, so there is a history over time.
2. **Schema** (``infer_schema``): the expected columns, types, null limits, value ranges and
   allowed values, inferred once from good data with ``primer-data profile --write-schemas`` and
   then reviewed and committed in ``configs/schemas/<dataset>.json`` (like a TFDV schema).
3. **Validation** (``validate``): the schema becomes a Great Expectations suite, run on the
   current data. A failed expectation is an **error**: the data broke its contract.
4. **Drift** (``drift``): the new profile is compared with the previous one; a large drop in rows,
   a jump in nulls, a shift in a numeric mean, or a change in category shares is a **warning**.

``check_datasets`` runs all four and returns the anomalies. The Airflow DAG fails the run on
errors and emails the anomalies (see ``alerts.py``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from primer.data.catalog import DATASETS, read_dataset
from primer.data.config import REPO_ROOT
from primer.data.storage import Store

log = logging.getLogger(__name__)

SCHEMAS_DIR = REPO_ROOT / "configs" / "schemas"
PROFILES_DIR = "metadata/profiles"
ANOMALIES_DIR = "metadata/anomalies"

CATEGORICAL_MAX = 25         # a text column with at most this many distinct values is categorical
MIN_PER_CATEGORY = 20        # ...and gets an allowed set only if each value appears this often on average
OPEN_COLUMNS = {"batch_id", "translator"}  # new batches and new names arrive with every ingest
LONG_TEXT = 200              # mean length above this: skip distinct counting (stories, strokes)
TOP_VALUES = 10

# Drift thresholds (warnings, not errors).
ROW_DROP = 0.05              # rows fell by more than 5%
NULL_JUMP = 0.05             # null fraction rose by more than 5 points
MEAN_SHIFT_SD = 0.5          # numeric mean moved by more than half a standard deviation
SHARE_SHIFT = 0.15           # total variation distance between category shares


# ------------------------------------------------------------------ statistics

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
    out: dict[str, Any] = {"kind": kind, "null_fraction": round(float(series.isna().mean()), 6)}
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
        out.update(mean_length=float(lengths.mean()), max_length=int(lengths.max()))
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


# ------------------------------------------------------------------ schema

def infer_schema(dataset: str, profile: dict[str, Any], min_rows_fraction: float = 0.5) -> dict[str, Any]:
    """Expected contract for a dataset, inferred from one good profile. Review before committing.

    Numeric ranges are the observed range widened by 10% on each side; categorical columns get
    their observed values as the allowed set (unless the values are too rare to be real
    categories, like a handful of translator names); ``min_rows`` guards against a dataset shrinking.
    """
    columns: dict[str, Any] = {}
    for name, col in profile["columns"].items():
        spec: dict[str, Any] = {"kind": col["kind"]}
        spec["max_null_fraction"] = 0.0 if col["null_fraction"] == 0 else round(min(1.0, col["null_fraction"] + 0.05), 4)
        if col["kind"] in ("int", "float") and "min" in col:
            pad = 0.1 * (col["max"] - col["min"]) or abs(col["max"]) * 0.1 or 1.0
            low = col["min"] - pad
            if col["min"] >= 0:
                low = max(0.0, low)  # counts and lengths can't go negative
            spec["min"], spec["max"] = round(low, 6), round(col["max"] + pad, 6)
        non_null = profile["rows"] * (1 - col["null_fraction"])
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
                column=name, mostly=round(1.0 - spec["max_null_fraction"], 6)))
        if "min" in spec:
            suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
                column=name, min_value=spec["min"], max_value=spec["max"]))
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
    except Exception:  # noqa: BLE001, S110 - cosmetic only
        pass
    context.data_sources.add_pandas("primer")
    return context


def validate(df: pd.DataFrame, schema: dict[str, Any], context=None) -> list[Anomaly]:
    """Run the schema's Great Expectations suite on ``df``; each failed expectation is an error."""
    context = context or gx_context()
    source = context.data_sources.get("primer")
    name = schema["dataset"]
    if name in [a.name for a in source.assets]:
        source.delete_asset(name)
    asset = source.add_dataframe_asset(name=name)
    batch = asset.add_batch_definition_whole_dataframe("all").get_batch(
        batch_parameters={"dataframe": _gx_ready(df)})
    result = batch.validate(build_suite(schema))
    anomalies = []
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
    return anomalies


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
        if "mean" in cur and "mean" in prev:
            sd = prev.get("std") or 0.0
            shift = abs(cur["mean"] - prev["mean"])
            if sd > 0 and shift > MEAN_SHIFT_SD * sd:
                out.append(Anomaly(dataset, "warning", "mean_shift", name,
                                   f"mean {prev['mean']:.3g} → {cur['mean']:.3g} ({shift / sd:.2f} sd)"))
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
    paths = [p for p in store.glob(f"{PROFILES_DIR}/{dataset}/*.json") if p != exclude]
    if not paths:
        return None
    profiles = [store.read_json(p) for p in paths]
    return max(profiles, key=lambda p: p.get("created_at", ""))


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
        df = read_dataset(dataset, store)
        if df.empty:
            continue
        log.info("Checking %s (%s rows)", dataset, f"{len(df):,}")
        profile = {"dataset": dataset, "run_id": run_id, "created_at": report["created_at"], **profile_frame(df)}
        out_path = f"{PROFILES_DIR}/{dataset}/{_safe(run_id)}.json"
        prev = previous_profile(store, dataset, exclude=out_path)
        schema = load_schema(dataset, schemas_dir)
        if schema and context is None:
            context = gx_context()
        anomalies = validate(df, schema, context) if schema else [
            Anomaly(dataset, "warning", "no_schema", None,
                    f"no configs/schemas/{dataset}.json yet; run `primer-data profile --write-schemas`")]
        if prev:
            anomalies += drift(dataset, prev, profile)
        if save:
            store.write_json(out_path, profile)
        report["datasets"][dataset] = {"rows": profile["rows"], "errors": sum(a.severity == "error" for a in anomalies),
                                       "warnings": sum(a.severity == "warning" for a in anomalies),
                                       "compared_with": prev.get("run_id") if prev else None}
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
        df = read_dataset(dataset, store)
        if df.empty:
            continue
        schema = infer_schema(dataset, profile_frame(df))
        path = schemas_dir / f"{dataset}.json"
        path.write_text(json.dumps(schema, indent=2, default=_json_default) + "\n", encoding="utf-8")
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
