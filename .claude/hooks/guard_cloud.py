#!/usr/bin/env python3
"""Refuses shell commands that would change Google Cloud in a way only a person should.

Claude Code runs this after guard_bash.py, before every Bash or PowerShell tool call
(see .claude/settings.json).

This project has ONE Google Cloud project, used for development and deployment alike, so there is
no staging/production split to guard. "Production" here is the live serving path, not a project id:
the risky act is sending traffic to a revision, deleting something, or touching IAM.

Two rules, in order:

1. The project must be the one we own. Only the *value* of ``--project`` / ``--project=``,
   ``gcloud config set project`` and ``CLOUDSDK_CORE_PROJECT`` is compared -- never the command
   text. Our project id is also the repository name, so a substring match would refuse
   ``cd .../young-illustrators-primer`` and ``cat young-illustrators-primer/AGENTS.md``.
   Set the id in .claude/settings.local.json as YIP_PROJECT (YIP_STAGING_PROJECT is also read, so
   the deploy scripts keep working).

2. Some operations are refused whatever the project: sending 100% of traffic to a revision,
   deploying without ``--no-traffic``, deletes, service-account keys and IAM policy changes.

Commands that invoke no cloud tool are not this hook's business and pass straight through.
A person deliberately doing one of these starts the session with YIP_ALLOW_PROD=1, or types the
command with the ``!`` prefix, which does not pass through hooks.
Exit 0 to allow; print the reason to stderr and exit 2 to refuse, like guard_bash.py.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys

# Tools whose commands can change Google Cloud. Anything else is none of our business.
CLOUD_TOOLS = {"gcloud", "gsutil", "bq", "terraform"}

# gcloud verbs that only read. Used to decide whether an unverifiable project is worth refusing;
# anything genuinely destructive is caught by RISKY below regardless of this list.
READ_ONLY_VERBS = {
    "list",
    "describe",
    "read",
    "get",
    "get-iam-policy",
    "info",
    "version",
    "tail",
    "print-access-token",
    "print-identity-token",
    "validate",
    "fmt",
    "plan",
    "show",
    "output",
    "state",
}

RISKY = [
    (
        re.compile(r"\brun\s+services\s+update-traffic\b(?=.*(--to-latest|=100\b))"),
        "sends all traffic to a revision, which is how a change goes live",
    ),
    (
        re.compile(r"\brun\s+deploy\b(?!.*--no-traffic)"),
        "deploys a revision that would take traffic immediately (use --no-traffic --tag for a canary)",
    ),
    (
        re.compile(r"\b(run|functions|sql|compute|container|artifacts|storage)\b.*\bdelete\b"),
        "deletes a cloud resource",
    ),
    (re.compile(r"\bprojects\s+delete\b"), "deletes a Google Cloud project"),
    (
        re.compile(r"\biam\s+service-accounts\s+keys\s+create\b"),
        "creates a long-lived service-account key",
    ),
    (
        re.compile(r"\b(add|remove|set)-iam-policy-binding\b"),
        "changes an IAM policy binding",
    ),
    (re.compile(r"\bterraform\b.*\bdestroy\b"), "destroys terraform-managed infrastructure"),
    (
        re.compile(r"\bterraform\b.*(workspace\s+select\s+prod\b|-var-file[= ]\S*prod)"),
        "selects a terraform prod workspace or var file",
    ),
]

# The deploy scripts read YIP_DEPLOY_ENV; both shells can set it.
DEPLOY_ENV_PROD = re.compile(
    r"(?:(?:^|\s)YIP_DEPLOY_ENV=|\$env:YIP_DEPLOY_ENV\s*=\s*['\"]?)prod\b", re.IGNORECASE
)

PROJECT_FLAG = re.compile(r"--project[= ]\s*([A-Za-z0-9][A-Za-z0-9-]*)")
CONFIG_SET_PROJECT = re.compile(r"\bgcloud\b[^|;&]*\bconfig\s+set\s+project\s+([A-Za-z0-9][A-Za-z0-9-]*)")
CORE_PROJECT_ENV = re.compile(r"CLOUDSDK_CORE_PROJECT\s*=\s*['\"]?([A-Za-z0-9][A-Za-z0-9-]*)")


def _tokens(command: str) -> list[str]:
    """Best-effort tokenisation. PowerShell syntax is not POSIX, so fall back to whitespace."""
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def uses_cloud_tool(tokens: list[str]) -> bool:
    for token in tokens:
        name = token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if name in CLOUD_TOOLS or name in {f"{t}.cmd" for t in CLOUD_TOOLS}:
            return True
    return False


def named_projects(command: str, tokens: list[str]) -> set[str]:
    """Every project id the command explicitly names, from flags and env vars only."""
    found: set[str] = set()
    for index, token in enumerate(tokens):
        if token.startswith("--project="):
            found.add(token.split("=", 1)[1])
        elif token == "--project" and index + 1 < len(tokens):
            found.add(tokens[index + 1])
    for pattern in (PROJECT_FLAG, CONFIG_SET_PROJECT, CORE_PROJECT_ENV):
        found.update(match.group(1) for match in pattern.finditer(command))
    return {p.strip("'\"") for p in found if p and not p.startswith("-")}


def is_read_only(tokens: list[str]) -> bool:
    """True when the command names a verb that only reads. Flags count: `gcloud --version` reads."""
    return any(token.lstrip("-") in READ_ONLY_VERBS for token in tokens)


def check(command: str, environ: dict[str, str]) -> str | None:
    """Return why `command` must be refused, or None to allow it."""
    if environ.get("YIP_ALLOW_PROD") == "1" or not command.strip():
        return None

    tokens = _tokens(command)
    if not uses_cloud_tool(tokens):
        # Not a cloud command. The project id is also the repo name, so this branch is what keeps
        # `cd .../young-illustrators-primer` and `gh pr list --repo .../young-illustrators-primer`
        # out of the guard entirely.
        return None

    allowed = (environ.get("YIP_PROJECT") or environ.get("YIP_STAGING_PROJECT") or "").strip()
    named = named_projects(command, tokens)

    foreign = sorted(p for p in named if p != allowed)
    if allowed and foreign:
        return (
            f"this command names the Google Cloud project {foreign[0]!r}, which is not this "
            f"project's ({allowed!r})."
        )

    for pattern, why in RISKY:
        if pattern.search(command):
            return f"this command {why}."

    if DEPLOY_ENV_PROD.search(command):
        return "this command sets YIP_DEPLOY_ENV=prod."

    if not allowed and not is_read_only(tokens):
        return (
            "no Google Cloud project is configured, so this command cannot be checked. Set "
            "YIP_PROJECT in .claude/settings.local.json."
        )

    return None


SHELL_TOOLS = {"Bash", "PowerShell"}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    # On Windows the PowerShell tool is the primary shell for claude.ai accounts, so both are checked.
    if payload.get("tool_name") not in SHELL_TOOLS:
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    reason = check(command, dict(os.environ))
    if reason:
        print(
            f"BLOCKED by .claude/hooks/guard_cloud.py: {reason} Agents may read cloud state and "
            "deploy canaries with --no-traffic; a person makes a change live. Run it yourself, or "
            "start the session with YIP_ALLOW_PROD=1.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
