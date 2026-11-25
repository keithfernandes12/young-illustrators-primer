#!/usr/bin/env python3
"""List the plans' open questions that no slice and no Jira issue owns (cross-slice-coherence S8).

Usage:  python scripts/audit_open_questions.py            (one line per untracked question)
        python scripts/audit_open_questions.py --json     (the same, for the `seams` skill)

An open question, or a hand-off recorded as one, is how a plan says "someone has to decide or do
this". A question that names neither a Jira key nor a slice belongs to nobody: it waits until a
person happens to reread that plan. This lists those, so a gap found by a review or by
`scan_duplicates.py` has somewhere to become work.

**Tracked** means the bullet contains a `KAN-<n>` key or a slice label (`S5`, `S6c2`). That is the
lenient rule the plan chose to start with: the first list should be short and true. A bullet that
names a slice only in passing still counts as tracked.

It reads every `docs/plans/*.md` except `AGENTS.md` and templates (`_*.md`), and only the
`## Open questions` section. A bullet is a line starting with `- ` at the left margin; its indented
continuation lines belong to it.

It always exits 0. It is an audit, not a gate: an untracked question is a reason to file
something, not a reason to block a PR.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PLANS = Path(__file__).resolve().parents[1] / "docs" / "plans"
KEY = re.compile(r"\bKAN-\d+\b")
SLICE = re.compile(r"\bS\d+[a-z]?\d*\b")
SECTION = "## Open questions"


@dataclass(frozen=True)
class Question:
    plan: str
    line: int
    text: str
    tracked: bool


def questions(path: Path) -> list[Question]:
    """Every bullet under the plan's ``## Open questions`` heading, with its first line number."""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == SECTION)
    except StopIteration:
        return []
    found: list[Question] = []
    current: list[str] = []
    first = 0

    def close() -> None:
        if current:
            body = " ".join(part.strip() for part in current)
            tracked = bool(KEY.search(body) or SLICE.search(body))
            found.append(Question(path.name, first, body.removeprefix("- ").strip(), tracked))

    for number, line in enumerate(lines[start + 1 :], start=start + 2):
        if line.startswith("## "):
            break
        if line.startswith("- "):
            close()
            current, first = [line], number
        elif current and (line.startswith(" ") or not line.strip()):
            if line.strip():
                current.append(line)
        else:
            close()
            current = []
    close()
    return found


def audit(plans: Path = PLANS) -> list[Question]:
    """Every open question in every plan, tracked or not, in file order."""
    return [
        q
        for path in sorted(plans.glob("*.md"))
        if path.name != "AGENTS.md" and not path.name.startswith("_")
        for q in questions(path)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print every question as JSON")
    parser.add_argument("--plans", type=Path, default=PLANS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    found = audit(args.plans)
    untracked = [q for q in found if not q.tracked]
    if args.json:
        print(json.dumps({"total": len(found), "untracked": [asdict(q) for q in untracked]}, indent=2))
        return 0
    for q in untracked:
        print(f"docs/plans/{q.plan}:{q.line}: {q.text[:110]}")
    print(f"{len(untracked)} of {len(found)} open questions name neither a Jira key nor a slice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
