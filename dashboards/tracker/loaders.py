"""Load the project plan, dataset registry, and pipeline run manifests.

Kept free of Streamlit so it can be unit tested.

Run manifest contract
---------------------
Every pipeline stage run writes one JSON file to ``<metadata_uri>/runs/``, where
``metadata_uri`` is ``gs://<bucket>/metadata`` or a local folder:

    {
      "run_id": "20261014T030000Z-tinystories-ingest",
      "source": "tinystories",            # key in configs/sources.yaml
      "stage": "ingest",                  # ingest | validate | preprocess | split | version
      "batch_id": "20261014T030000Z",     # the stages of one batch share this
      "status": "success",                # running | success | failed
      "started_at": "2026-10-14T03:00:00Z",
      "finished_at": "2026-10-14T03:04:12Z",
      "records_in": 0,
      "records_out": 50000,               # records leaving this stage in this run
      "records_total": 50000,             # optional: dataset size after this run
      "dataset_version": null,            # optional: DVC tag, e.g. "v1"
      "output_uri": "gs://.../raw/tinystories/20261014T030000Z",
      "validation": {"checks": [{"name": "no_empty_text", "passed": true, "detail": ""}]},
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
    "run_id", "source", "stage", "batch_id", "status", "started_at", "finished_at",
    "records_in", "records_out", "records_total", "dataset_version", "output_uri", "error",
]


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ----------------------------------------------------------------- project

def load_project(path: Path) -> dict:
    return load_yaml(path)


def tasks_frame(project: dict, today: date) -> pd.DataFrame:
    names = {ws["id"]: ws["name"] for ws in project.get("workstreams", [])}
    tasks = pd.DataFrame(project.get("tasks", []))
    if tasks.empty:
        return pd.DataFrame(columns=["id", "workstream", "workstream_name", "title", "owner",
                                     "status", "due", "done", "overdue"])
    if "done" not in tasks:
        tasks["done"] = None
    tasks["due"] = pd.to_datetime(tasks["due"]).dt.date
    tasks["workstream_name"] = tasks["workstream"].map(names).fillna(tasks["workstream"])
    tasks["overdue"] = (tasks["status"] != "done") & (tasks["due"] < today)
    return tasks


def current_week(weeks: list[dict], today: date) -> dict | None:
    for week in weeks:
        if week["start"] <= today <= week["end"]:
            return week
    return None


# ---------------------------------------------------------------- pipeline

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
                records.append(json.load(fh))
            except json.JSONDecodeError:
                continue  # half-written manifest; it will be read on the next refresh
    if not records:
        return empty_runs()

    runs = pd.DataFrame.from_records(records).reindex(columns=RUN_COLUMNS)
    for col in ("started_at", "finished_at"):
        runs[col] = pd.to_datetime(runs[col], utc=True, errors="coerce")
    for col in ("records_in", "records_out", "records_total"):
        runs[col] = pd.to_numeric(runs[col], errors="coerce")
    runs["when"] = runs["finished_at"].fillna(runs["started_at"])
    return runs.sort_values("when", ascending=False, ignore_index=True)


def load_versions(metadata_uri: str) -> list[dict]:
    """Dataset versions from ``<metadata_uri>/versions/*.json``, newest first."""
    if "://" in metadata_uri:
        pattern = metadata_uri.rstrip("/") + "/versions/*.json"
    else:
        pattern = (Path(metadata_uri) / "versions" / "*.json").as_posix()
    try:
        files = fsspec.open_files(pattern, mode="r", encoding="utf-8")
    except FileNotFoundError:
        return []
    versions = []
    for open_file in files:
        with open_file as fh:
            versions.append(json.load(fh))
    return sorted(versions, key=lambda v: v["created_at"], reverse=True)


def _records(run: pd.Series) -> float:
    return run["records_total"] if pd.notna(run["records_total"]) else run["records_out"]


def source_status(sources: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """One row per registered source: where it stands and how many records it has."""
    rows = []
    for key, src in sources.iterrows():
        source_runs = runs[runs["source"] == key]
        if source_runs.empty:
            status, updated, records = "— not started", pd.NaT, None
        else:
            latest = source_runs.iloc[0]
            if latest["status"] == "failed":
                status = f"❌ {latest['stage']} failed"
            elif latest["status"] == "running":
                status = f"⏳ {latest['stage']} running"
            else:
                ok = source_runs[source_runs["status"] == "success"]
                furthest = max(ok["stage"], key=lambda s: STAGES.index(s) if s in STAGES else -1)
                if furthest == "version":
                    tag = ok[ok["stage"] == "version"].iloc[0]["dataset_version"]
                    status = f"✅ processed, in {tag}"
                else:
                    status = "✅ processed" if furthest == "preprocess" else f"✅ {furthest}"
            processed = source_runs[(source_runs["stage"] == "preprocess") & (source_runs["status"] == "success")]
            records = _records(processed.iloc[0]) if not processed.empty else None
            updated = latest["when"]
        rows.append({
            "Source": src.get("name", key),
            "Status": status,
            "Records": records,
            "Updated": updated,
        })
    return pd.DataFrame(rows)


def activity(project: dict, runs: pd.DataFrame, sources: pd.DataFrame, versions: list[dict] | None = None,
             limit: int = 15) -> list[dict]:
    """Newest-first feed of pipeline batches, dataset versions, and completed tasks: {when, icon, text}."""
    names = sources["name"].to_dict() if "name" in sources else {}
    items = []

    for version in versions or []:
        sizes = ", ".join(f"{name} {d['rows']:,}" for name, d in version["datasets"].items())
        note = f" ({version['message']})" if version.get("message") else ""
        items.append({"when": pd.Timestamp(version["created_at"]).tz_convert("UTC"), "icon": "🏷️",
                      "text": f"**Dataset version {version['tag']}** frozen{note}: {sizes}"})

    batches = runs[runs["stage"] != "version"]  # versions are shown once each, above
    for (source, batch), group in batches.groupby(["source", "batch_id"], dropna=True):
        # Latest attempt at each stage, so a stage that failed and was re-run counts once.
        by_stage = {row["stage"]: row for _, row in group.sort_values("when").iterrows()}
        latest = pd.DataFrame(list(by_stage.values()))
        ingest = by_stage.get("ingest")
        if len(by_stage) == 1 and ingest is not None and ingest["status"] == "success" \
                and not ingest["records_out"]:
            continue  # checked the source, nothing new
        name = names.get(source, source)
        failed = latest[latest["status"] == "failed"]
        if not failed.empty:
            run = failed.iloc[0]
            error = str(run["error"]).removeprefix("Validation failed: ")
            icon, text = "❌", f"**{name}** {run['stage']} failed: {error}"
        elif (latest["status"] == "running").any():
            icon, text = "⏳", f"**{name}** {latest[latest['status'] == 'running'].iloc[0]['stage']} running"
        elif "preprocess" in by_stage:
            processed = by_stage["preprocess"]
            # Only when both count the same unit (picture books fetch stories but process pages).
            raw = f" from {int(ingest['records_out']):,} fetched" if ingest is not None \
                and pd.notna(ingest["records_out"]) and ingest["records_out"] >= processed["records_out"] else ""
            icon, text = "✅", f"**{name}**: {int(processed['records_out']):,} records processed{raw}"
        else:
            last = group.sort_values("when").iloc[-1]
            icon, text = "✅", f"**{name}** {last['stage']} done"
        items.append({"when": group["when"].max(), "icon": icon, "text": text})

    for task in project.get("tasks", []):
        if task.get("status") == "done" and task.get("done"):
            items.append({"when": pd.Timestamp(task["done"], tz="UTC"), "icon": "☑️",
                          "text": f"Task done: {task['title']} ({task.get('owner', 'TBD')})"})
    for decision in project.get("decisions", []):
        if decision.get("status") == "decided" and decision.get("decided"):
            items.append({"when": pd.Timestamp(decision["decided"], tz="UTC"), "icon": "🧭",
                          "text": f"Decided: {decision['decision']} → {decision.get('outcome', '')}"})

    items.sort(key=lambda i: i["when"], reverse=True)
    return items[:limit]
