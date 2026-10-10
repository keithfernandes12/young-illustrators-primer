"""Young Illustrator's Primer tracker: project progress and data pipeline status.

Run from the repo root:

    streamlit run dashboards/tracker/app.py
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import loaders
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PROJECT_FILE = HERE / "project.yaml"
SOURCES_FILE = REPO_ROOT / "configs" / "sources.yaml"
SAMPLE_METADATA = HERE / "examples" / "metadata"
DEFAULT_METADATA = os.environ.get("TRACKER_METADATA_URI", "gs://young-illustrators-primer-data/metadata")

TASK_ICONS = {"done": "✅", "in_progress": "🔄", "todo": "⬜", "blocked": "⛔"}
WEEK_ICONS = {"done": "✅", "current": "📍", "upcoming": ""}

st.set_page_config(page_title="Primer Tracker", page_icon="📚", layout="wide")


@st.cache_data(ttl=60)
def get_project() -> dict:
    return loaders.load_project(PROJECT_FILE)


@st.cache_data(ttl=60)
def get_sources() -> pd.DataFrame:
    return loaders.load_sources(SOURCES_FILE)


@st.cache_data(ttl=60)
def get_runs(metadata_uri: str) -> pd.DataFrame:
    return loaders.load_runs(metadata_uri)


def days_until(target: date, today: date) -> str:
    days = (target - today).days
    if days > 0:
        return f"{days} days"
    return "today" if days == 0 else "passed"


# ---------------------------------------------------------------- project tab

def render_project(project: dict, today: date) -> None:
    dates = project["dates"]
    tasks = loaders.tasks_frame(project, today)
    week = loaders.current_week(project["weeks"], today)

    done = int((tasks["status"] == "done").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current week", week["label"] if week else "—")
    c2.metric("MVP (Nov 15)", days_until(dates["mvp"], today))
    c3.metric("Code freeze (Nov 30)", days_until(dates["code_freeze"], today))
    c4.metric("Tasks done", f"{done} / {len(tasks)}")

    span = (dates["testing_end"] - dates["build_start"]).days
    elapsed = min(max((today - dates["build_start"]).days, 0), span)
    st.progress(elapsed / span, text=f"Project timeline: day {elapsed} of {span}")
    if week:
        st.info(f"**{week['label']} ({week['start']:%b %d} – {week['end']:%b %d}):** {week['milestones']}")

    st.subheader("Workstreams")
    st.dataframe(
        loaders.workstream_summary(project, tasks),
        hide_index=True,
        column_config={
            "Progress": st.column_config.ProgressColumn(
                "Progress", min_value=0, max_value=100, format="%d%%"
            ),
        },
    )

    st.subheader("Tasks")
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    ws_pick = f1.multiselect("Workstream", sorted(tasks["workstream_name"].unique()))
    status_pick = f2.multiselect("Status", list(TASK_ICONS))
    owner_pick = f3.multiselect("Owner", sorted(tasks["owner"].unique()))
    overdue_only = f4.checkbox("Overdue only")

    view = tasks
    if ws_pick:
        view = view[view["workstream_name"].isin(ws_pick)]
    if status_pick:
        view = view[view["status"].isin(status_pick)]
    if owner_pick:
        view = view[view["owner"].isin(owner_pick)]
    if overdue_only:
        view = view[view["overdue"]]

    view = view.assign(icon=view["status"].map(TASK_ICONS)).sort_values(["due", "id"])
    st.dataframe(
        view[["icon", "id", "title", "workstream_name", "owner", "status", "due", "overdue"]],
        hide_index=True,
        column_config={
            "icon": st.column_config.TextColumn("", width="small"),
            "id": "ID",
            "title": st.column_config.TextColumn("Task", width="large"),
            "workstream_name": "Workstream",
            "owner": "Owner",
            "status": "Status",
            "due": st.column_config.DateColumn("Due", format="MMM D"),
            "overdue": st.column_config.CheckboxColumn("Overdue"),
        },
    )
    st.caption("To update tasks, edit `dashboards/tracker/project.yaml` and open a PR.")

    left, right = st.columns(2)
    with left:
        st.subheader("MVP (due Nov 15)")
        for item in project.get("mvp", []):
            st.markdown(f"{TASK_ICONS.get(item['status'], '⬜')} {item['item']}")
    with right:
        st.subheader("Stretch goals (in order)")
        for i, item in enumerate(project.get("stretch", []), start=1):
            st.markdown(f"{TASK_ICONS.get(item['status'], '⬜')} {i}. {item['item']}")

    st.subheader("Decisions")
    decisions = pd.DataFrame(project.get("decisions", []))
    if not decisions.empty:
        decisions = decisions.sort_values(["status", "due"], ascending=[False, True])
        st.dataframe(
            decisions,
            hide_index=True,
            column_config={
                "decision": st.column_config.TextColumn("Decision", width="large"),
                "status": "Status",
                "outcome": "Outcome",
                "due": st.column_config.DateColumn("Due by", format="MMM D"),
            },
        )

    st.subheader("Timeline")
    timeline = loaders.timeline_frame(project["weeks"], today)
    timeline["State"] = timeline["State"].map(lambda s: f"{WEEK_ICONS[s]} {s}".strip())
    st.dataframe(
        timeline,
        hide_index=True,
        column_config={"Milestones": st.column_config.TextColumn(width="large")},
    )


# --------------------------------------------------------------- pipeline tab

def render_pipeline(sources: pd.DataFrame, runs: pd.DataFrame, metadata_uri: str, sample: bool) -> None:
    if sample:
        st.warning("Showing **sample** pipeline data. Turn off sample data in the sidebar to see real runs.")
    elif runs.empty:
        st.info(
            f"No pipeline runs found under `{metadata_uri}/runs/`. Runs appear here once the "
            "pipeline starts writing run manifests. Turn on sample data in the sidebar for a preview."
        )

    summary = loaders.source_summary(sources, runs)
    week_ago = pd.Timestamp.now(tz="UTC") - timedelta(days=7)
    recent_failures = runs[(runs["status"] == "failed") & (runs["when"] >= week_ago)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sources registered", len(sources))
    c2.metric("Sources ingested", int(summary["Furthest stage"].notna().sum()))
    c3.metric("Total records", f"{int(summary['Records'].fillna(0).sum()):,}")
    c4.metric("Failed runs (7 days)", len(recent_failures))

    st.subheader("Sources")
    st.dataframe(
        summary,
        hide_index=True,
        column_config={
            "MVP": st.column_config.CheckboxColumn("MVP"),
            "License verified": st.column_config.CheckboxColumn("License verified"),
            "Last run": st.column_config.DatetimeColumn("Last run", format="MMM D, HH:mm"),
            "Records": st.column_config.NumberColumn("Records", format="%d"),
        },
    )
    unverified = summary[summary["MVP"] & ~summary["License verified"]]
    if not unverified.empty:
        st.caption(f"⚠️ {len(unverified)} MVP sources still need their license verified "
                   "(`license_verified` in `configs/sources.yaml`).")

    st.subheader("Stage progress")
    st.caption(" → ".join(s.title() for s in loaders.STAGES) + "  ·  ✅ success  ❌ failed  ⏳ running  — not run")
    st.dataframe(loaders.stage_matrix(sources, runs))

    if runs.empty:
        return

    growth = loaders.records_over_time(runs)
    if not growth.empty:
        st.subheader("Records over time")
        st.line_chart(growth)

    st.subheader("Recent runs")
    st.dataframe(
        runs.head(50)[["when", "source", "stage", "status", "records_in", "records_out",
                       "checks_passed", "checks_failed", "dataset_version", "output_uri"]],
        hide_index=True,
        column_config={
            "when": st.column_config.DatetimeColumn("When", format="MMM D, HH:mm"),
            "records_in": st.column_config.NumberColumn("In", format="%d"),
            "records_out": st.column_config.NumberColumn("Out", format="%d"),
            "checks_passed": "Checks ✅",
            "checks_failed": "Checks ❌",
            "dataset_version": "Version",
            "output_uri": "Output",
        },
    )

    failed = runs[runs["status"] == "failed"].head(10)
    if not failed.empty:
        st.subheader("Failures")
        for _, run in failed.iterrows():
            with st.expander(f"❌ {run['source']} / {run['stage']} — {run['when']:%b %d, %H:%M}"):
                if pd.notna(run["error"]):
                    st.code(run["error"], language=None)
                bad = [c for c in run["checks"] if not c.get("passed")]
                if bad:
                    st.dataframe(pd.DataFrame(bad), hide_index=True)

    validated = runs[(runs["stage"] == "validate") & runs["checks"].map(bool)]
    if not validated.empty:
        st.subheader("Validation checks")
        source = st.selectbox("Source", sorted(validated["source"].unique()))
        latest = validated[validated["source"] == source].iloc[0]
        st.caption(f"Latest validate run: {latest['when']:%b %d, %H:%M} ({latest['run_id']})")
        checks = pd.DataFrame(latest["checks"])
        checks.insert(0, "", checks["passed"].map({True: "✅", False: "❌"}))
        st.dataframe(checks.drop(columns="passed"), hide_index=True)


# ----------------------------------------------------------------------- main

def main() -> None:
    with st.sidebar:
        st.header("Settings")
        metadata_uri = st.text_input(
            "Pipeline metadata location",
            DEFAULT_METADATA,
            help="Local folder or gs://<bucket>/metadata. Set TRACKER_METADATA_URI to change the default.",
        )
        sample = st.toggle("Show sample pipeline data", value=False)
        if st.button("Refresh now"):
            st.cache_data.clear()
        st.caption("Data refreshes automatically every 60 seconds.")

    project = get_project()
    st.title(f"📚 {project.get('project', 'Project')} tracker")

    project_tab, pipeline_tab = st.tabs(["Project", "Data pipeline"])
    with project_tab:
        render_project(project, date.today())
    with pipeline_tab:
        uri = str(SAMPLE_METADATA) if sample else metadata_uri
        try:
            runs = get_runs(uri)
        except Exception as exc:  # noqa: BLE001 - e.g. missing GCS credentials
            st.error(f"Couldn't read pipeline metadata from `{uri}`: {exc}")
            runs = loaders.empty_runs()
        render_pipeline(get_sources(), runs, uri, sample)


main()
