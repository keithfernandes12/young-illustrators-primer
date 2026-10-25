#!/usr/bin/env python3
"""Refuses shell commands that AI agents must not run in this repo.

Claude Code runs this before every Bash tool call (see .claude/settings.json). It reads the
hook payload as JSON on stdin. To allow, it exits 0 silently. To refuse, it prints the reason
to stderr and exits 2, which Claude Code shows to the agent.

What it refuses, and why, is listed in AGENTS.md ("Rules that are easy to break" and "Git and
pull requests"). Commands a person types with the `!` prefix do not pass through this hook, so
a person can still run anything deliberately.

The command is split on shell separators outside quotes and tokenized with shlex, because a
regex over the raw string is easy to fool (for example `--title "--label x"` would look like a
label). If a command that touches a guarded tool cannot be parsed, it is refused rather than
guessed at.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable

REPO = "keithfernandes12/young-illustrators-primer"
JIRA_KEY = re.compile(r"\bKAN-\d+\b")
GUARDED_WORDS = ("gh", "git", "primer-data", "primer.data", "gsutil", "gcloud", "dvc")
PRIMER_WRITE_COMMANDS = {"run", "ingest", "validate", "preprocess", "version"}

# Flags that take a value as the next token. Skipping the value stops a quoted value that looks
# like a flag (`--title "--draft"`) from being read as that flag.
GH_VALUE_FLAGS = {
    "-R", "--repo", "-t", "--title", "-b", "--body", "-F", "--body-file", "-B", "--base",
    "-H", "--head", "-l", "--label", "-r", "--reviewer", "-a", "--assignee", "-m", "--milestone",
    "-p", "--project", "-T", "--template", "-X", "--method", "-f", "--field", "--raw-field",
    "-q", "--jq", "--input", "--hostname", "-S", "--search", "-L", "--limit", "-A", "--author",
    "-s", "--state", "--json", "--recover", "-e", "--editor",
}
GIT_GLOBAL_VALUE_FLAGS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}
GIT_PUSH_VALUE_FLAGS = {"--repo", "--receive-pack", "--exec", "-o", "--push-option"}
GSUTIL_BLOCKED = {
    "rm", "rb", "mb", "setmeta", "acl", "iam", "lifecycle", "versioning", "retention", "defacl",
    "label", "cors", "web", "notification", "kms", "logging", "ubla", "pap", "rpo", "autoclass",
    "bucketpolicyonly", "compose", "rewrite",
}
DVC_BLOCKED = {"push", "gc", "remove", "import-url", "destroy"}
PROGRAM_PREFIXES = {"sudo", "env", "command", "exec", "time", "nohup"}

GIT_RULES = "AGENTS.md -> Git and pull requests"
DATA_RULES = "AGENTS.md -> Rules that are easy to break"


def split_segments(command: str) -> list[str]:
    """Split on &&, ||, ;, |, & and newlines that appear outside quotes."""
    segments: list[str] = []
    current: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        c = command[i]
        if quote:
            current.append(c)
            if c == "\\" and quote == '"' and i + 1 < len(command):
                current.append(command[i + 1])
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
            continue
        if c == "\\" and i + 1 < len(command):
            current.append(command[i:i + 2])
            i += 2
            continue
        if c in "'\"":
            quote = c
            current.append(c)
        elif c in ";\n&|":
            segments.append("".join(current))
            current = []
            if i + 1 < len(command) and command[i + 1] == c and c in "&|":
                i += 1
        else:
            current.append(c)
        i += 1
    segments.append("".join(current))
    return [s.strip() for s in segments if s.strip()]


def substitutions(segment: str) -> list[str]:
    """Commands nested in $(...) or backticks, which run even inside an assignment."""
    found = re.findall(r"\$\(([^()]*)\)", segment)
    found += re.findall(r"`([^`]*)`", segment)
    return found


def positionals(args: list[str], value_flags: set[str]) -> tuple[list[str], list[str]]:
    """Return (positional words, flag tokens), skipping the values of value-taking flags."""
    words: list[str] = []
    flags: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--":
            words.extend(args[i + 1:])
            break
        if tok.startswith("-") and tok != "-":
            flags.append(tok)
            if tok in value_flags and i + 1 < len(args):
                flags.append(args[i + 1])
                i += 2
                continue
        else:
            words.append(tok)
        i += 1
    return words, flags


def flag_values(args: list[str], names: set[str]) -> list[str]:
    """Every value given to any of `names`, in `--x v` or `--x=v` form."""
    values: list[str] = []
    for i, tok in enumerate(args):
        if tok in names and i + 1 < len(args):
            values.append(args[i + 1])
        else:
            name, eq, value = tok.partition("=")
            if eq and name in names:
                values.append(value)
    return values


def has_flag(args: list[str], *names: str) -> bool:
    return any(tok in names or tok.partition("=")[0] in names for tok in args)


def normalize_repo(value: str) -> str:
    value = re.sub(r"^(https?://)?(www\.)?github\.com/", "", value.strip())
    return value.removesuffix(".git").strip("/").lower()


def check_gh(args: list[str], cwd: str) -> str | None:
    for repo in flag_values(args, {"-R", "--repo"}):
        if normalize_repo(repo) != REPO:
            return f"gh may only target {REPO} while an agent works here (got {repo!r}). See {GIT_RULES}."
    words, _ = positionals(args, GH_VALUE_FLAGS)
    if not words:
        return None
    top = words[0]
    sub = words[1] if len(words) > 1 else ""
    if top == "repo" and (sub != "view" or (len(words) > 2 and normalize_repo(words[2]) != REPO)):
        return f"Only `gh repo view` of this repo is allowed. See {GIT_RULES}."
    if top == "api" and sub:
        match = re.match(r"/?repos/([^/]+/[^/?]+)", sub)
        if match and "{" not in match.group(1) and normalize_repo(match.group(1)) != REPO:
            return f"`gh api` may only call {REPO}. See {GIT_RULES}."
    if top == "issue" and sub == "create":
        return "GitHub Issues are not used here; Jira project KAN is the tracker. Ask the person before creating a Jira issue."
    if top == "pr" and sub == "merge":
        return ("Agents never merge pull requests. Leave the PR with the `needs-reviewer` label; "
                "a teammate reviews it and merges. See " + GIT_RULES + ".")
    if top == "pr" and sub == "create":
        return check_pr_create(args, cwd)
    return None


def check_pr_create(args: list[str], cwd: str) -> str | None:
    usage = ("Use: gh pr create --draft --label agent-authored --label needs-reviewer "
             "--title \"type(scope): summary (KAN-<n>)\" --body-file <file from .github/pull_request_template.md>")
    if has_flag(args, "--body", "-b"):
        return "Write the PR body from the template into a file and pass --body-file, not --body.\n" + usage
    if not has_flag(args, "--draft", "-d"):
        return "Open pull requests as drafts (--draft). Mark ready after `ruff check .`, `mypy src` and `pytest` pass.\n" + usage
    body_files = flag_values(args, {"--body-file", "-F"})
    if not body_files or body_files[0] == "-":
        return "Pass the PR body as a file with --body-file.\n" + usage
    path = body_files[0] if os.path.isabs(body_files[0]) else os.path.join(cwd, body_files[0])
    try:
        with open(path, encoding="utf-8") as f:
            body = f.read()
    except OSError:
        return f"The --body-file {body_files[0]!r} could not be read.\n" + usage
    if not JIRA_KEY.search(body):
        return "The PR body must link a Jira issue key (KAN-<n>). Ask the person which issue this is.\n" + usage
    labels = {name.strip() for value in flag_values(args, {"--label", "-l"}) for name in value.split(",")}
    if "agent-authored" not in labels:
        return "PRs opened by an agent carry --label agent-authored.\n" + usage
    if "needs-reviewer" not in labels and not flag_values(args, {"--reviewer", "-r"}):
        return "Assign a reviewer (--reviewer) or add --label needs-reviewer so a teammate can claim it.\n" + usage
    return None


def current_branch(cwd: str) -> str:
    try:
        result = subprocess.run(["git", "symbolic-ref", "--short", "HEAD"], cwd=cwd,
                                capture_output=True, text=True, timeout=5, check=False)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def check_git(args: list[str], cwd: str, branch_of: Callable[[str], str]) -> str | None:
    words, _ = positionals(args, GIT_GLOBAL_VALUE_FLAGS)
    if not words or words[0] != "push":
        return None
    push_args = args[args.index("push") + 1:]
    for tok in push_args:
        if tok in ("--force", "--force-with-lease", "--force-if-includes", "--delete", "--mirror",
                   "--prune", "--all", "--tags") or tok.startswith("--force-with-lease="):
            return f"`git push {tok}` is not allowed for agents. See {GIT_RULES}."
        if re.fullmatch(r"-[a-zA-Z]+", tok) and set(tok[1:]) & {"f", "d"}:
            return f"`git push {tok}` (force or delete) is not allowed for agents. See {GIT_RULES}."
    refs, _ = positionals(push_args, GIT_PUSH_VALUE_FLAGS)
    refspecs = refs[1:]
    for spec in refspecs:
        if spec.startswith("+"):
            return f"Force-pushing ({spec}) is not allowed for agents. See {GIT_RULES}."
        if spec.startswith(":"):
            return f"Deleting a remote branch ({spec}) is not allowed for agents. See {GIT_RULES}."
        dest = spec.split(":")[-1].removeprefix("refs/heads/")
        if dest in ("main", "master") or (dest == "HEAD" and branch_of(cwd) in ("main", "master")):
            return f"Agents never push to main. Push your own branch and open a draft PR. See {GIT_RULES}."
        if dest.startswith(("refs/tags/", "data-")):
            return "Ask the person to push tags themselves (for example `! git push origin data-v1`)."
    if not refspecs and branch_of(cwd) in ("main", "master"):
        return f"You are on main; agents never push to main. Create a branch first. See {GIT_RULES}."
    return None


def check_primer(args: list[str], inline_env: dict[str, str], environ: dict[str, str]) -> str | None:
    words, _ = positionals(args, {"--data-uri", "--sources-file", "--batch-size", "--batch", "-m", "--message"})
    if not words or words[0] not in PRIMER_WRITE_COMMANDS:
        return None
    uri = (flag_values(args, {"--data-uri"}) or [None])[0]
    uri = uri or inline_env.get("PRIMER_DATA_URI") or environ.get("PRIMER_DATA_URI") or ""
    if not uri or uri.lower().startswith(("gs://", "gcs://")):
        target = uri or "the shared bucket gs://young-illustrators-primer-data (the default)"
        return (f"This would write to {target}. Run it locally: "
                "`primer-data --data-uri .local-data " + " ".join(words) + "` (the flag goes before the command). "
                "A person runs bucket writes themselves with `! primer-data ...`. See " + DATA_RULES + ".")
    return None


def check_gsutil(words: list[str]) -> str | None:
    if not words:
        return None
    sub, rest = words[0], words[1:]
    if sub in GSUTIL_BLOCKED:
        return f"`{sub}` changes or deletes bucket data or settings; a person runs it. See {DATA_RULES}."
    if sub in ("cp", "mv", "rsync") and rest:
        if rest[-1].lower().startswith("gs://"):
            return f"Writing into the bucket ({rest[-1]}) is for a person to do. See {DATA_RULES}."
        if sub == "mv" and any(w.lower().startswith("gs://") for w in rest[:-1]):
            return f"`mv` deletes the bucket source; a person runs it. See {DATA_RULES}."
    return None


def check_gcloud(args: list[str]) -> str | None:
    words, _ = positionals(args, {"--project", "--account", "--configuration", "--format", "--filter"})
    if not words:
        return None
    if words[0] == "auth":
        return "`gcloud auth` changes credentials; a person runs it."
    if "delete" in words:
        return "`gcloud ... delete` is for a person to run."
    if words[0] == "storage":
        rest = words[1:]
        if rest and rest[0] in ("buckets", "objects") and len(rest) > 1 and rest[1] != "list" and rest[1] != "describe":
            return f"`gcloud storage {' '.join(rest[:2])}` changes the bucket; a person runs it. See {DATA_RULES}."
        return check_gsutil(rest)
    return None


def strip_prefix(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate leading VAR=value assignments and wrappers like `env` or `sudo`."""
    env: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tok):
            name, _, value = tok.partition("=")
            env[name] = value
        elif tok in PROGRAM_PREFIXES:
            pass
        else:
            break
        i += 1
    return env, tokens[i:]


def program_name(token: str) -> str:
    name = re.split(r"[\\/]", token)[-1].lower()
    return name.removesuffix(".exe")


def check_command(command: str, cwd: str, environ: dict[str, str] | None = None,
                  branch_of: Callable[[str], str] = current_branch, depth: int = 0) -> str | None:
    """Return the reason to refuse `command`, or None to allow it."""
    environ = dict(os.environ) if environ is None else environ
    if depth > 3:
        return None
    for segment in split_segments(command):
        for inner in substitutions(segment):
            reason = check_command(inner, cwd, environ, branch_of, depth + 1)
            if reason:
                return reason
        try:
            tokens = shlex.split(segment)
        except ValueError:
            if any(re.search(rf"(^|[\s/]){re.escape(w)}\b", segment) for w in GUARDED_WORDS):
                return "Could not parse this command safely (unbalanced quotes?); rewrite it more simply."
            continue
        inline_env, rest = strip_prefix(tokens)
        if not rest:
            continue
        prog, args = program_name(rest[0]), rest[1:]
        reason = None
        if prog in ("bash", "sh", "zsh") and "-c" in args:
            idx = args.index("-c")
            if idx + 1 < len(args):
                reason = check_command(args[idx + 1], cwd, environ, branch_of, depth + 1)
        elif prog == "eval":
            reason = check_command(" ".join(args), cwd, environ, branch_of, depth + 1)
        elif prog == "gh":
            reason = check_gh(args, cwd)
        elif prog == "git":
            reason = check_git(args, cwd, branch_of)
        elif prog == "primer-data":
            reason = check_primer(args, inline_env, environ)
        elif prog.startswith("python") and args[:2] == ["-m", "primer.data.cli"]:
            reason = check_primer(args[2:], inline_env, environ)
        elif prog == "gsutil":
            reason = check_gsutil(positionals(args, {"-o", "-h", "-u"})[0])
        elif prog == "gcloud":
            reason = check_gcloud(args)
        elif prog == "dvc":
            words = positionals(args, set())[0]
            if words and words[0] in DVC_BLOCKED:
                reason = (f"`dvc {words[0]}` writes to or deletes from the shared DVC remote; a person runs it "
                          "(see the freeze-dataset-version skill).")
        if reason:
            return reason
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    reason = check_command(command, payload.get("cwd") or os.getcwd())
    if reason:
        print(f"BLOCKED by .claude/hooks/guard_bash.py: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
