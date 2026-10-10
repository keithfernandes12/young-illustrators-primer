"""Young Illustrator's Primer tracker: what's happened, data status, open tasks.

Run from the repo root:

    streamlit run dashboards/tracker/app.py
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import loaders
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PROJECT_FILE = HERE / "project.yaml"
SOURCES_FILE = REPO_ROOT / "configs" / "sources.yaml"
METADATA_URI = os.environ.get("TRACKER_METADATA_URI", "gs://young-illustrators-primer-data/metadata")

st.set_page_config(page_title="Primer Tracker", page_icon="📚", layout="centered")


@st.cache_data(ttl=60)
def get_project() -> dict:
    return loaders.load_project(PROJECT_FILE)


@st.cache_data(ttl=60)
def get_sources() -> pd.DataFrame:
    return loaders.load_sources(SOURCES_FILE)


@st.cache_data(ttl=60)
def get_runs(uri: str) -> pd.DataFrame:
    return loaders.load_runs(uri)


@st.cache_data(ttl=60)
def get_versions(uri: str) -> list[dict]:
    return loaders.load_versions(uri)


def days_to(target: date, today: date) -> str:
    days = (target - today).days
    return f"{days} days" if days > 0 else ("today" if days == 0 else "passed")


def feed_line(item: dict) -> str:
    when = item["when"]
    stamp = f"{when:%b %d %H:%M}" if (when.hour or when.minute) else f"{when:%b %d}"
    return f"- `{stamp}` {item['icon']} {item['text']}"


def main() -> None:
    today = date.today()
    project, sources = get_project(), get_sources()
    try:
        runs, versions = get_runs(METADATA_URI), get_versions(METADATA_URI)
    except Exception as exc:  # noqa: BLE001 - e.g. missing GCS credentials
        st.error(f"Couldn't read pipeline runs from `{METADATA_URI}`: {exc}")
        runs, versions = loaders.empty_runs(), []

    tasks = loaders.tasks_frame(project, today)
    status = loaders.source_status(sources, runs)
    week = loaders.current_week(project["weeks"], today)
    dates = project["dates"]

    st.title("📚 Young Illustrator's Primer")
    st.caption(" · ".join([
        week["label"] if week else "Outside the build schedule",
        f"MVP in {days_to(dates['mvp'], today)}",
        f"code freeze in {days_to(dates['code_freeze'], today)}",
        f"data version {versions[0]['tag']}" if versions else "no data version yet",
    ]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tasks done", f"{int((tasks['status'] == 'done').sum())} / {len(tasks)}")
    c2.metric("Overdue", int(tasks["overdue"].sum()))
    c3.metric("Sources in", f"{int(status['Records'].notna().sum())} / {len(status)}")
    c4.metric("Records", f"{int(status['Records'].fillna(0).sum()):,}")

    st.subheader("What's happened")
    feed = loaders.activity(project, runs, sources, versions)
    st.markdown("\n".join(feed_line(i) for i in feed) if feed else "Nothing yet.")

    st.subheader("Data")
    st.dataframe(
        status,
        hide_index=True,
        column_config={
            "Records": st.column_config.NumberColumn(format="%d"),
            "Updated": st.column_config.DatetimeColumn(format="MMM D, HH:mm"),
        },
    )

    st.subheader("Open tasks")
    open_tasks = tasks[tasks["status"] != "done"].sort_values(["due", "id"])
    workstream = st.selectbox("Workstream", ["All", *sorted(open_tasks["workstream_name"].unique())],
                              label_visibility="collapsed")
    if workstream != "All":
        open_tasks = open_tasks[open_tasks["workstream_name"] == workstream]
    view = pd.DataFrame({
        "Due": open_tasks["due"],
        "Task": [("⚠️ " if late else "") + ("🔄 " if s == "in_progress" else "") + t
                 for t, s, late in zip(open_tasks["title"], open_tasks["status"], open_tasks["overdue"])],
        "Owner": open_tasks["owner"],
        "Workstream": open_tasks["workstream_name"],
    })
    st.dataframe(view, hide_index=True, column_config={"Due": st.column_config.DateColumn(format="MMM D")})

    open_decisions = [d for d in project.get("decisions", []) if d.get("status") == "open"]
    if open_decisions:
        st.markdown("**Open decisions**\n" + "\n".join(
            f"- {d['decision']} (by {d['due']:%b %d})" for d in open_decisions))

    st.caption("⚠️ overdue · 🔄 in progress · Edit `dashboards/tracker/project.yaml` to update tasks · "
               "Refreshes every 60 seconds")


main()
