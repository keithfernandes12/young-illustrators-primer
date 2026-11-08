#!/usr/bin/env python3
"""Refuses Bash commands that target the production Google Cloud project.

Claude Code runs this after guard_bash.py, before every Bash tool call (see .claude/settings.json).
Staging is where agents deploy, train and experiment; production changes are made by a person.

"Production" is any of:
- the project id in YIP_PROD_PROJECT (set it in your shell or .claude/settings.local.json),
- `YIP_DEPLOY_ENV=prod` on the command line (the deploy-cloud-run scripts read it),
- a terraform command that selects the prod workspace or a *prod* var file.

A person who is deliberately working on prod starts the session with YIP_ALLOW_PROD=1, or types
the command with the `!` prefix, which does not pass through hooks.
Exit 0 to allow; print the reason to stderr and exit 2 to refuse, like guard_bash.py.
"""

from __future__ import annotations

import json
import os
import re
import sys

TERRAFORM_PROD = re.compile(r"\bterraform\b.*(workspace\s+select\s+prod\b|-var-file[= ]\S*prod)")


def check(command: str, environ: dict[str, str]) -> str | None:
    if environ.get("YIP_ALLOW_PROD") == "1" or not command:
        return None
    prod = environ.get("YIP_PROD_PROJECT", "").strip()
    if prod and re.search(rf"(^|[^A-Za-z0-9-]){re.escape(prod)}($|[^A-Za-z0-9-])", command):
        return f"this command names the production project {prod!r}."
    if re.search(r"(^|\s)YIP_DEPLOY_ENV=prod\b", command):
        return "this command sets YIP_DEPLOY_ENV=prod."
    if TERRAFORM_PROD.search(command):
        return "this terraform command selects the prod workspace or a prod var file."
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    reason = check(command, dict(os.environ))
    if reason:
        print(
            f"BLOCKED by .claude/hooks/guard_prod.py: {reason} Agents work in staging "
            "($YIP_STAGING_PROJECT). Ask the person to run the production step themselves.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
