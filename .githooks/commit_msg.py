"""Checks the commit subject line before git records the commit.

Git runs `.githooks/commit-msg`, a small shell script that picks a working Python (on Windows
`python3` is often the Microsoft Store stub) and runs this file. Enable once per clone:

    git config core.hooksPath .githooks

Format:  type(scope): imperative summary (KAN-<n>)
Example: feat(data): add Amateur Drawings ingestion (KAN-42)

The Jira key in the subject links every commit to its issue. CI (.github/workflows/pr-rules.yml)
runs the same check on pull requests, so a clone without this hook enabled is still caught there.
"""

from __future__ import annotations

import re
import sys

TYPES = ("feat", "fix", "docs", "refactor", "test", "chore", "ci", "perf")
SUBJECT = re.compile(rf"^({'|'.join(TYPES)})(\([a-z0-9-]+\))?!?: \S.* \(KAN-\d+\)$")
EXEMPT_PREFIXES = ("Merge ", "Revert \"", "fixup! ", "squash! ", "amend! ")

HELP = f"""Commit subject must look like:

    type(scope): imperative summary (KAN-<n>)
    feat(data): add Amateur Drawings ingestion (KAN-42)

type:  {", ".join(TYPES)}
scope: the area changed, e.g. data, drawing, story, readability, illustration, speech, safety,
       serving, app, dags, dashboards, infra, docs, agents, deps
See AGENTS.md -> Git and pull requests."""


def subject_of(message: str) -> str:
    for line in message.splitlines():
        if line.startswith("# ------------------------ >8"):
            break
        if line.strip() and not line.startswith("#"):
            return line.rstrip()
    return ""


def problem(subject: str) -> str | None:
    """Return why `subject` is not acceptable, or None if it is."""
    if subject.startswith(EXEMPT_PREFIXES):
        return None
    if not SUBJECT.match(subject):
        return f"Bad commit subject: {subject!r}"
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return 0
    with open(argv[1], encoding="utf-8") as f:
        subject = subject_of(f.read())
    issue = problem(subject)
    if issue:
        print(f"{issue}\n\n{HELP}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
