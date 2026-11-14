#!/usr/bin/env python3
"""Refuses shell commands that would change Google Cloud in a way only a person should.

Claude Code runs this after guard_bash.py, before every Bash or PowerShell tool call
(see .claude/settings.json).

There is no staging/production split to guard. "Production" here is the live serving path, not a
project id: the risky act is moving traffic, deleting something, or touching IAM.

Six members build six model areas in parallel, so a session may legitimately name two projects
(docs/engineering/model-development.md section 6):

- ``YIP_SHARED_PROJECT`` -- the team project. Durable state only: the data bucket, Artifact
  Registry, the models bucket, MLflow, and the served Cloud Run revision. Builds push here.
- ``YIP_COMPUTE_PROJECT`` -- this member's own project. Compute only: Vertex AI custom jobs and
  Cloud Run GPU jobs, burning that member's own credits.

The rules, in order:

1. **Every project the command names must be ours.** Only the *value* of ``--project`` /
   ``--project=``, ``gcloud config set project`` and ``CLOUDSDK_CORE_PROJECT`` is compared -- never
   the command text. The shared project id is also the repository name, so a substring match would
   refuse ``cd .../young-illustrators-primer``. A value written as a shell variable
   (``--project "$YIP_SHARED_PROJECT"``) is resolved against the environment first, because that is
   the form every skill document uses; a reference that cannot be resolved is refused.
2. **A mutating command must establish its target project.** If it names none, the inherited
   ``CLOUDSDK_CORE_PROJECT`` is the effective target and must be ours. With no target at all, a
   mutating command is refused rather than sent wherever gcloud happens to point.
3. **Some operations are refused whatever the project**: any traffic change, deploying without
   ``--no-traffic``, deletes, service-account keys and any IAM policy write.

Compound commands are checked segment by segment, nested shells (``bash -c '...'``) are checked
recursively, and the deployment helper scripts are checked by what they do -- a hook never sees the
``gcloud`` call inside a script, so ``bash promote.sh`` has to be refused by name.

A person doing one of these deliberately starts the session with YIP_ALLOW_PROD=1, or types the
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

# Shells that take a command string, so the real command is one level down.
SHELL_WRAPPERS = {"bash", "sh", "zsh", "dash", "ash", "pwsh", "powershell"}

# Operations that only read. Compared against the parsed operation and against flag NAMES, never
# against flag values -- `--display-name list` must not make a job creation look like a read.
READ_ONLY_VERBS = {
    "list",
    "describe",
    "read",
    "get",
    "get-iam-policy",
    "info",
    "version",
    "help",
    "tail",
    "print-access-token",
    "print-identity-token",
    "validate",
    "fmt",
    "plan",
    "show",
    "output",
    "state",
    "cat",
    "du",
    "ls",
}

#: Deployment helpers that perform a refused operation inside themselves. Hooks do not run again
#: for what a script does, so the script has to be recognised by name.
HELPER_SCRIPTS = {
    "promote.sh": "runs a traffic promotion, which is how a change goes live",
    "rollback.sh": "shifts live traffic between revisions",
}

RISKY = [
    (
        re.compile(r"\brun\s+services\s+update-traffic\b"),
        "changes which revision serves live traffic",
    ),
    (
        # `replace` applies a service YAML that can set spec.traffic outright, and `update` creates
        # a revision that serves everything when the service routes to latest. Neither says
        # "traffic" on the command line.
        re.compile(r"\brun\s+services\s+(replace|update)\b"),
        "rewrites a live Cloud Run service, which can move traffic without naming it",
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
        re.compile(r"\b(add|remove|set)-iam-policy-binding\b|\bset-iam-policy\b"),
        "writes an IAM policy",
    ),
    (re.compile(r"\bterraform\b.*\bdestroy\b"), "destroys terraform-managed infrastructure"),
    (
        re.compile(r"\bterraform\b.*(workspace\s+select\s+prod\b|-var-file[= ]\S*prod)"),
        "selects a terraform prod workspace or var file",
    ),
]

# Nothing in the deploy scripts reads YIP_DEPLOY_ENV any more -- the staging/prod branching is
# gone. It is still refused because a person may have it set from the old workflow, and an
# agent inheriting it would be acting on an intent nobody restated.
DEPLOY_ENV_PROD = re.compile(
    r"(?:(?:^|\s)YIP_DEPLOY_ENV=|\$env:YIP_DEPLOY_ENV\s*=\s*['\"]?)prod\b", re.IGNORECASE
)

CONFIG_SET_PROJECT = re.compile(r"\bconfig\s+set\s+project\s+(\S+)")
CORE_PROJECT_ASSIGNMENT = re.compile(r"CLOUDSDK_CORE_PROJECT\s*=\s*(\S+)")

#: ``$NAME``, ``${NAME}``, ``$env:NAME`` (PowerShell) and ``%NAME%`` (cmd).
VARIABLE_REF = re.compile(
    r"^(?:\$\{(?P<braced>\w+)\}|\$env:(?P<ps>\w+)|\$(?P<bare>\w+)|%(?P<cmd>\w+)%)$"
)

#: In precedence order, the variables that name a project this session may touch.
PROJECT_VARS: tuple[str, ...] = (
    "YIP_COMPUTE_PROJECT",
    "YIP_SHARED_PROJECT",
    "YIP_PROJECT",
    "YIP_STAGING_PROJECT",
)

UNRESOLVED = "\x00unresolved"


def allowed_projects(environ: dict[str, str]) -> set[str]:
    """Every project id this session is configured for. Empty means nothing is configured."""
    return {value.strip() for var in PROJECT_VARS if (value := environ.get(var, "")).strip()}


def split_segments(command: str) -> list[str]:
    """Split on &&, ||, ;, | and newlines that appear outside quotes."""
    segments: list[str] = []
    current: list[str] = []
    quote = ""
    index = 0
    while index < len(command):
        char = command[index]
        if quote:
            current.append(char)
            if char == "\\" and quote == '"' and index + 1 < len(command):
                current.append(command[index + 1])
                index += 2
                continue
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char in ";\n" or command.startswith(("&&", "||"), index) or char == "|":
            segments.append("".join(current))
            current = []
            index += 2 if command.startswith(("&&", "||"), index) else 1
            continue
        else:
            current.append(char)
        index += 1
    segments.append("".join(current))
    return [segment.strip() for segment in segments if segment.strip()]


def _tokens(segment: str) -> list[str]:
    """Best-effort tokenisation. PowerShell syntax is not POSIX, so fall back to whitespace."""
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _basename(token: str) -> str:
    return token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def resolve_project_ref(raw: str, environ: dict[str, str]) -> str:
    """A literal project id stays itself; a variable reference is expanded from `environ`.

    Returns UNRESOLVED when the token is a variable the environment does not define, so the caller
    can fail closed instead of comparing a literal dollar sign against the allowlist.
    """
    match = VARIABLE_REF.match(raw)
    if not match:
        return raw
    name = next(value for value in match.groupdict().values() if value)
    value = environ.get(name, "").strip()
    return value or UNRESOLVED


def uses_cloud_tool(tokens: list[str]) -> bool:
    return any(_basename(token).removesuffix(".cmd") in CLOUD_TOOLS for token in tokens)


def nested_commands(tokens: list[str]) -> list[str]:
    """Every command string a wrapping shell or `eval` was asked to run.

    Matching an exact `-c` missed `bash -lc '...'` and `sh -ec '...'`: short options cluster, so
    the flag carrying the command can be any single-dash group containing `c`.
    """
    found: list[str] = []
    for index, token in enumerate(tokens):
        name = _basename(token).removesuffix(".exe")
        if name == "eval" and index + 1 < len(tokens):
            found.append(tokens[index + 1])
            continue
        if name not in SHELL_WRAPPERS:
            continue
        for offset in range(index + 1, len(tokens) - 1):
            candidate = tokens[offset]
            if candidate.startswith("--"):
                if candidate.lower() == "--command":
                    found.append(tokens[offset + 1])
                    break
                continue
            if candidate.startswith("-") and "c" in candidate[1:].lower():
                found.append(tokens[offset + 1])
                break
    return found


#: Operations refused on their own tokens rather than by a text pattern, because the text form is
#: either ambiguous or defeatable by a flag value.
def token_refusal(tokens: list[str]) -> str | None:
    operation, flags = operation_and_flags(tokens)
    # Drop everything up to and including the tool itself, so the first operation word is the
    # gcloud group ("run", "ai") rather than "gcloud" or a leading `env`/path prefix.
    for index, word in enumerate(operation):
        if _basename(word).removesuffix(".cmd") in CLOUD_TOOLS:
            operation = operation[index + 1 :]
            break

    # `--no-traffic` must be a real flag. A text search also matched it inside a value, so
    # `--labels=n=--no-traffic` passed as if the revision took no traffic.
    if operation[:2] == ["run", "deploy"] and "no-traffic" not in flags:
        return (
            "this command deploys a revision that would take traffic immediately "
            "(use --no-traffic --tag for a canary)."
        )

    # docs/adrs/where-model-training-happens.md: agent sessions never train. An ask rule is not
    # enough when a loop is running unattended.
    launches = {
        ("ai", "custom-jobs", "create"): "launches a Vertex AI training job",
        ("ai", "custom-jobs", "cancel"): "cancels a training job",
        ("ai", "hp-tuning-jobs", "create"): "launches a hyperparameter tuning job",
        ("run", "jobs", "create"): "creates a Cloud Run job",
        ("run", "jobs", "execute"): "executes a Cloud Run job",
        ("batch", "jobs", "submit"): "submits a Batch job",
    }
    for prefix, why in launches.items():
        if tuple(operation[: len(prefix)]) == prefix:
            return (
                f"this command {why}, and agents never train "
                "(docs/adrs/where-model-training-happens.md). Prepare the config and hand it over."
            )
    return None


def helper_refusal(tokens: list[str]) -> str | None:
    """Deployment helpers are refused by name: a hook never sees the gcloud call inside them."""
    for token in tokens:
        reason = HELPER_SCRIPTS.get(_basename(token))
        if reason:
            return f"this command {reason}."
    return None


def operation_and_flags(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Split a segment into its operation words and its flag names.

    A flag written without ``=`` consumes the next token, so that value is neither an operation
    word nor a flag. That is what keeps ``--display-name list`` from looking like a read.
    """
    operation: list[str] = []
    flags: list[str] = []
    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            name, _, inline_value = token.partition("=")
            flags.append(name.lstrip("-"))
            skip_next = not inline_value and index + 1 < len(tokens)
            continue
        operation.append(token)
    return operation, flags


def is_read_only(tokens: list[str]) -> bool:
    operation, flags = operation_and_flags(tokens)
    return bool((set(operation) | set(flags)) & READ_ONLY_VERBS)


def named_projects(segment: str, tokens: list[str], environ: dict[str, str]) -> set[str]:
    """Every project the segment explicitly names, resolved through the environment."""
    raw: set[str] = set()
    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token.startswith("--project="):
            raw.add(token.split("=", 1)[1])
        elif token == "--project" and index + 1 < len(tokens):
            raw.add(tokens[index + 1])
            skip_next = True
        elif token.startswith("CLOUDSDK_CORE_PROJECT="):
            raw.add(token.split("=", 1)[1])
    for pattern in (CONFIG_SET_PROJECT, CORE_PROJECT_ASSIGNMENT):
        raw.update(match.group(1).strip("'\"") for match in pattern.finditer(segment))
    return {resolve_project_ref(value, environ) for value in raw if value and not value.startswith("-")}


def _check_text(text: str, environ: dict[str, str], depth: int = 0) -> str | None:
    """Check one command line.

    Environment assignments and project selections apply to the whole line, not to the segment they
    appear in -- `CLOUDSDK_CORE_PROJECT=x; gcloud ...` and `$env:YIP_DEPLOY_ENV = "prod"; gcloud ...`
    both target what the earlier segment set. So those are read across the line, while the
    operations themselves are judged segment by segment.
    """
    segments = split_segments(text)
    token_sets = [_tokens(segment) for segment in segments]

    for tokens in token_sets:
        if depth >= 4:
            break
        for inner in nested_commands(tokens):
            reason = _check_text(inner, environ, depth + 1)
            if reason:
                return reason

    for tokens in token_sets:
        reason = helper_refusal(tokens)
        if reason:
            return reason

    if not any(uses_cloud_tool(tokens) for tokens in token_sets):
        # Not a cloud command. The shared project id is also the repository name, so this branch is
        # what keeps `cd .../young-illustrators-primer` out of the guard entirely.
        return None

    if DEPLOY_ENV_PROD.search(text):
        return "this command sets YIP_DEPLOY_ENV=prod."

    allowed = allowed_projects(environ)
    named: set[str] = set()
    for segment, tokens in zip(segments, token_sets, strict=True):
        named |= named_projects(segment, tokens, environ)

    if UNRESOLVED in named:
        return (
            "this command names a project through a variable that is not set, so the target "
            "cannot be checked. Set YIP_COMPUTE_PROJECT and YIP_SHARED_PROJECT."
        )

    foreign = sorted(project for project in named if project not in allowed)
    if allowed and foreign:
        ours = ", ".join(sorted(allowed))
        return (
            f"this command names the Google Cloud project {foreign[0]!r}, which is not one of "
            f"ours ({ours})."
        )

    for segment, tokens in zip(segments, token_sets, strict=True):
        if not uses_cloud_tool(tokens):
            continue
        for pattern, why in RISKY:
            if pattern.search(segment):
                return f"this command {why}."
        reason = token_refusal(tokens)
        if reason:
            return reason
        if is_read_only(tokens):
            continue
        # A mutating command must establish where it is going.
        if not allowed:
            return (
                "no Google Cloud project is configured, so this command cannot be checked. Set "
                "YIP_COMPUTE_PROJECT and YIP_SHARED_PROJECT in .claude/settings.local.json."
            )
        if not named:
            inherited = environ.get("CLOUDSDK_CORE_PROJECT", "").strip()
            if inherited and inherited not in allowed:
                return (
                    f"this command would target {inherited!r} through CLOUDSDK_CORE_PROJECT, "
                    "which is not one of ours."
                )
            if not inherited:
                return (
                    "this command changes cloud state without naming a project, so its target "
                    "cannot be established. Pass --project explicitly."
                )
    return None


def check(command: str, environ: dict[str, str]) -> str | None:
    """Return why `command` must be refused, or None to allow it."""
    if environ.get("YIP_ALLOW_PROD") == "1" or not command.strip():
        return None
    return _check_text(command, environ)


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
            "deploy canaries with --no-traffic; a person moves traffic. Run it yourself, or start "
            "the session with YIP_ALLOW_PROD=1.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
