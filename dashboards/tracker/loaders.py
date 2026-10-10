"""Load the project plan, dataset registry, and pipeline run manifests.

Kept free of Streamlit so it can be unit tested and reused by the pipeline.

Run manifest contract
---------------------
Every pipeline run writes one JSON file to ``<metadata_uri>/runs/``, where
``metadata_uri`` is a local folder or ``gs://<bucket>/metadata``:

    {
      "run_id": "20261014T030000Z-tinystories-ingest",
      "source": "tinystories",            # key in configs/sources.yaml
      "stage": "ingest",                  # one of STAGES
      "status": "success",                # running | success | failed
      "started_at": "2026-10-14T03:00:00Z",
      "finished_at": "2026-10-14T03:04:12Z",
      "records_in": 0,
      "records_out": 50000,               # records leaving this stage in this run
      "records_total": 50000,             # optional: dataset size after this run
      "dataset_version": null,            # optional: DVC tag, e.g. "v1"
      "output_uri": "gs://.../raw/tinystories/20261014/",
      "validation": {                     # optional, mainly for the validate stage
        "checks": [{"name": "no_empty_text", "passed": true, "detail": ""}]
      },
      "error": null                       # message when status is failed
    }
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import fsspec
import pandas as pd
import yaml

STAGES = ["ingest", "validate", "preprocess", "split", "version"]

RUN_COLUMNS = [
    "run_id", "source", "stage", "status", "started_at", "finished_at",
    "records_in", "records_out", "records_total", "dataset_version",
    "output_uri", "checks", "checks_passed", "checks_failed", "error",
]


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------- project tab

def load_project(path: Path) -> dict:
    return load_yaml(path)


def tasks_frame(project: dict, today: date) -> pd.DataFrame:
    names = {ws["id"]: ws["name"] for ws in project.get("workstreams", [])}
    tasks = pd.DataFrame(project.get("tasks", []))
    if tasks.empty:
        return pd.DataFrame(columns=["id", "workstream", "workstream_name", "title",
                                     "owner", "status", "due", "overdue"])
    tasks["due"] = pd.to_datetime(tasks["due"]).dt.date
    tasks["workstream_name"] = tasks["workstream"].map(names).fillna(tasks["workstream"])
    tasks["overdue"] = (tasks["status"] != "done") & (tasks["due"] < today)
    return tasks


def workstream_summary(project: dict, tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ws in project.get("workstreams", []):
        t = tasks[tasks["workstream"] == ws["id"]]
        total = len(t)
        done = int((t["status"] == "done").sum())
        rows.append({
            "Workstream": ws["name"],
            "Lead": ws.get("lead") or "TBD",
            "Progress": round(100 * done / total) if total else 0,
            "Done": done,
            "In progress": int((t["status"] == "in_progress").sum()),
            "Blocked": int((t["status"] == "blocked").sum()),
            "Overdue": int(t["overdue"].sum()),
            "Total": total,
        })
    return pd.DataFrame(rows)


def current_week(weeks: list[dict], today: date) -> dict | None:
    for week in weeks:
        if week["start"] <= today <= week["end"]:
            return week
    return None


def timeline_frame(weeks: list[dict], today: date) -> pd.DataFrame:
    rows = []
    for week in weeks:
        if week["end"] < today:
            state = "done"
        elif week["start"] <= today:
            state = "current"
        else:
            state = "upcoming"
        rows.append({
            "Week": week["label"],
            "Dates": f"{week['start']:%b %d} – {week['end']:%b %d}",
            "State": state,
            "Milestones": week["milestones"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- pipeline tab

def load_sources(path: Path) -> pd.DataFrame:
    sources = pd.DataFrame.from_dict(load_yaml(path).get("sources", {}), orient="index")
    sources.index.name = "key"
    if "priority" in sources:
        sources = sources.sort_values("priority", kind="stable")
    return sources


def empty_runs() -> pd.DataFrame:
    return pd.DataFrame(columns=RUN_COLUMNS + ["when"])


def load_runs(metadata_uri: str) -> pd.DataFrame:
    """Read every run manifest under ``<metadata_uri>/runs/``, newest first."""
    if "://" in metadata_uri:
        pattern = metadata_uri.rstrip("/") + "/runs/*.json"
    else:
        pattern = (Path(metadata_uri) / "runs" / "*.json").as_posix()

    try:
        files = fsspec.open_files(pattern, mode="r", encoding="utf-8")
    except FileNotFoundError:
        return empty_runs()

    records = []
    for open_file in files:
        with open_file as fh:
            try:
                record = json.load(fh)
            except json.JSONDecodeError:
                continue  # half-written manifest; it will be read on the next refresh
        checks = (record.pop("validation", None) or {}).get("checks", [])
        record["checks"] = checks
        record["checks_passed"] = sum(1 for c in checks if c.get("passed"))
        record["checks_failed"] = sum(1 for c in checks if not c.get("passed"))
        records.append(record)

    if not records:
        return empty_runs()

    runs = pd.DataFrame.from_records(records).reindex(columns=RUN_COLUMNS)
    for col in ("started_at", "finished_at"):
        runs[col] = pd.to_datetime(runs[col], utc=True, errors="coerce")
    for col in ("records_in", "records_out", "records_total"):
        runs[col] = pd.to_numeric(runs[col], errors="coerce")
    runs["when"] = runs["finished_at"].fillna(runs["started_at"])
    return runs.sort_values("when", ascending=False, ignore_index=True)


def _stage_rank(stage: str) -> int:
    return STAGES.index(stage) if stage in STAGES else -1


def _record_count(run: pd.Series) -> float:
    return run["records_total"] if pd.notna(run["records_total"]) else run["records_out"]


def source_summary(sources: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, src in sources.iterrows():
        source_runs = runs[runs["source"] == key]
        ok = source_runs[source_runs["status"] == "success"]
        latest = source_runs.iloc[0] if not source_runs.empty else None
        versions = source_runs["dataset_version"].dropna()
        rows.append({
            "Source": src.get("name", key),
            "Priority": src.get("priority"),
            "MVP": bool(src.get("mvp", False)),
            "Feeds": ", ".join(src.get("feeds") or []),
            "License": src.get("license"),
            "License verified": bool(src.get("license_verified", False)),
            "Furthest stage": max(ok["stage"], key=_stage_rank) if not ok.empty else None,
            "Last status": latest["status"] if latest is not None else "not started",
            "Last run": latest["when"] if latest is not None else pd.NaT,
            "Records": _record_count(ok.iloc[0]) if not ok.empty else None,
            "Version": versions.iloc[0] if not versions.empty else None,
        })
    return pd.DataFrame(rows)


STATUS_ICONS = {"success": "✅", "failed": "❌", "running": "⏳"}


def stage_matrix(sources: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Latest status of each stage for each source."""
    rows = {}
    for key, src in sources.iterrows():
        source_runs = runs[runs["source"] == key]
        row = {}
        for stage in STAGES:
            stage_runs = source_runs[source_runs["stage"] == stage]
            row[stage.title()] = (
                STATUS_ICONS.get(stage_runs.iloc[0]["status"], "?") if not stage_runs.empty else "—"
            )
        rows[src.get("name", key)] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def records_over_time(runs: pd.DataFrame) -> pd.DataFrame:
    """Dataset size per source over time, one column per source."""
    ok = runs[runs["status"] == "success"].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["records"] = ok["records_total"].fillna(ok["records_out"])
    ok = ok.dropna(subset=["records", "when"])
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.pivot_table(index="when", columns="source", values="records", aggfunc="max")
        .sort_index()
        .ffill()
    )
