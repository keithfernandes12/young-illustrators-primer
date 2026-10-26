---
title: AI coding framework — agent guidance, guardrails and docs
type: plan
created: 2026-10-25
status: active
author: rishiparekh
jira: [KAN-19]
tags: [agents, process, ci, docs]
related: [documentation, engineering-loop, review-checklist]
---

# AI coding framework — agent guidance, guardrails and docs

## Where things stand

The whole team uses Claude Code. Before this plan the repo had no agent guidance, no CI, no PR
template, no branch protection and no commit rules, and nothing stopped an agent from writing to
the shared bucket, merging its own PR or acting on another repository. This plan adds one shared
instruction file, guardrails that refuse the dangerous commands, the repo's first CI, a docs system
with checked frontmatter, and a set of skills. It is the record later agents read before changing
any of these pieces.

## Sources read

- The repo at `feature/data-pipeline` (6240f42): `pyproject.toml`, `src/primer/data/README.md`,
  `src/primer/data/cli.py`, `tests/data/conftest.py`, `.gitignore`, `.gitattributes`.
- `docs/project_scoping.md` §3.3 (splits), §8 (failure analysis), §9 (deployment).
- The team's answers to the setup questionnaire (tracker, rules, permissions, OS).
- Not read: Jira issue contents beyond the project structure; the other teammates' machines.

## Contract

- **Target behaviour** — an agent in this repo reads `AGENTS.md`, follows the docs and skills, and
  is refused (with the correct alternative) when it tries to: merge, push to `main`, force-push,
  delete branches, act on another repo, open a non-draft or Jira-less PR, write to the bucket, or
  change credentials. Commits and PRs follow one format, checked locally and in CI.
- **Not doing** — no Slack integration (the workspace is shared with people outside the team); no
  Cursor/Copilot files; no `AGENTS.md` in empty folders; no reformatting or type-fixing of existing
  code.
- **Patterns to follow** — scripts in Python (everyone has it; bash fails on plain Windows);
  scripts outside the package are loaded in tests by file path, as `tests/data/conftest.py` already
  does for the dashboard loaders.
- **Boundaries** — the guardrails live in `.claude/hooks/` (agents only) and `.githooks/` (everyone);
  CI mirrors the commit and PR rules because local hooks can be skipped.
- **Test plan** — `tests/agent_hooks/` (guard: refused and allowed command lists; commit-msg;
  session start), `tests/test_no_gcs.py`, `tests/test_docs.py`. All run in plain `pytest`.

## Slices

### S1 · Framework files

*size: large (mostly documents)*

Adds `AGENTS.md`, `CLAUDE.md`, `src/primer/data/AGENTS.md`, `.claude/settings.json`, the two Claude
hooks, `.githooks/commit-msg`, the GCS test guard, `scripts/check_docs.py`, the docs folders with
templates and process documents, the skills in `.claude/skills/`, `.mcp.json`, the PR template,
CODEOWNERS, `ci.yml`, `pr-rules.yml`, and mypy/ruff-format configuration. Also fixes the flaky
CLEAR test and the 5 existing `mypy` errors (one commit each), because `ci.yml` starts running
both `pytest` and `mypy src` on every PR.

- **Why separate** — it is the foundation; everything else configures or extends it.
- **Touches** — new files, plus `pyproject.toml` (dev tools), `.gitignore`, and the `xlsx_bytes`
  helper in `tests/data/test_sources.py`, and the type fixes in `src/primer/data/manifest.py`,
  `src/primer/data/versioning.py` and `src/primer/data/sources/storybooks.py`.
- **Acceptance** — `pytest` passes, including every refused/allowed guard case; `ruff check .`
  passes; an agent session that runs `gh pr merge 1` is refused with the reason; a commit subject
  without `(KAN-<n>)` is refused once `core.hooksPath` is set.
- **Testing** — removing the `merge` rule from `check_gh` fails `test_refused[gh pr merge 3]`;
  removing the autouse fixture in `tests/conftest.py` fails `tests/test_no_gcs.py`; deleting
  `author` from any document fails `test_repository_docs_are_valid`; restoring the original
  `xlsx_bytes` fails `test_xlsx_bytes_is_deterministic` (it builds the same frame on both sides of a
  wall-clock second boundary); removing any of the `mypy` fixes fails the `mypy src` step in CI.
- **Depends on** — `feature/data-pipeline` merged (or this branch based on it).
- **Jira** — KAN-19.

### S2 · Repository settings (person with admin rights)

*size: small*

Create the `needs-reviewer` and `agent-authored` labels and the `protect-main` ruleset (1 approval,
approval of the latest push, no force-push, no deletion). Commands are in "Admin commands" below.

- **Why separate** — needs admin rights an agent must not hold; it is not a code change.
- **Acceptance** — `gh api repos/keithfernandes12/young-illustrators-primer/rulesets` lists
  `protect-main`; a PR author cannot merge their own PR without another member's approval.
- **Depends on** — S1 merged (so the labels it refers to are documented).

### S3 · Required checks

*size: small*

After `ci.yml` (`test` job) and `pr-rules.yml` (`pr-rules` job) have each run on one PR, add them
as required status checks to `protect-main`.

- **Acceptance** — a PR with a failing test or a commit subject without a Jira key cannot merge.
- **Depends on** — S1, S2, and one PR that has run both workflows.

### Later slices (not started; pick up from here)

- `src/primer/serving/AGENTS.md` when FastAPI code lands: endpoints, auth, what must never be logged.
- A `docs/runbooks/` entry for each operated process as it appears (retraining, deploys, rollback).
- Run `ruff format .` across the codebase. Its own PR (large, mechanical diff).
- Pin ruff to a range (e.g. `ruff>=0.17,<0.18`): `ruff>=0.6` installs the newest release, whose
  default rule set has grown, so a new ruff release can turn CI red on unchanged code.
- When `app/` adds a Node toolchain: split the `AGENTS.md` commands section into Python and web.
- When GCP access exists: a secrets ADR (Secret Manager for shared secrets; agents read dev secrets
  only; no service-account key files) and alert routing (Cloud Monitoring email, not Slack).

## Status

| Slice | State | PR | Notes |
|---|---|---|---|
| S1 | in review | #1 | Windows 11 verified; awaiting first CI run |
| S2 | not started | | needs admin |
| S3 | not started | | after S2 and one PR run |

## Decisions

- **2026-10-25** — Jira (project KAN) is the only tracker; GitHub Issues stay off and
  `dashboards/tracker/project.yaml` is not authoritative. *Team rule.*
- **2026-10-25** — Agents never merge; every PR has a reviewer or the `needs-reviewer` placeholder;
  every PR body links a Jira key; `gh` targets only this repo. *Team rule.*
- **2026-10-25** — No Slack integration and agents never post to Slack: the free workspace is shared
  with unrelated people and keeps 90 days of history. *Team rule.*
- **2026-10-25** — Commit subjects are `type(scope): summary (KAN-<n>)`; branches are
  `<type>/KAN-<n>-<slug>`; PRs open as drafts. The PR title follows the commit format because a
  squash merge uses it as the commit on `main`.
- **2026-10-25** — Agent sessions default to a local data folder (`PRIMER_DATA_URI=.local-data` in
  `.claude/settings.json`); the guard hook is the second layer, so a hook that fails to start does
  not leave the bucket exposed.
- **2026-10-25** — Docs use seven type folders with checked frontmatter; no numbering bot. A check
  that runs in `pytest` replaces the validation that larger setups leave out and then drift without.
- **2026-10-25** — Skills live directly in `.claude/skills/` (everyone uses Claude Code); `AGENTS.md`
  stays tool-neutral so other tools still work if someone switches.

- **2026-10-25** — Code ownership of the agent and process files is every collaborator, not one
  named owner. A pull request touching them is reviewed by any collaborator except its author.
  GitHub never requests review from the author and never accepts their own approval for a
  code-owner requirement, so listing everyone expresses that rule; CODEOWNERS cannot state it
  directly. The repository is under a personal account, so teams are unavailable and the list is
  individual logins that must be kept in step with the collaborator list. *Team rule, resolving
  open question 1.*

- **2026-10-25** — CI rules for people and agents live in `docs/engineering/ci.md` (what runs, what
  is enforced versus convention, red builds, flaky tests, changing CI); improvements are a separate
  plan, `docs/plans/ci-hardening.md`. A `fix-ci` skill handles red builds. If ci-hardening S2
  (`ci-pass`) lands before S3 here, the admin requires `ci-pass` instead of `test` and `pr-rules`.

## Verified facts

- **2026-10-25** — `primer-data` reads its data root from `--data-uri` (a global flag that must come
  *before* the command) or the `PRIMER_DATA_URI` environment variable, defaulting to
  `gs://young-illustrators-primer-data`. Checked in `src/primer/data/cli.py`.
- **2026-10-25** — `run`, `ingest`, `validate`, `preprocess` and `version` write to the data root;
  `sources` and `versions` only read. An incremental `run` advances `metadata/state/<source>.json`;
  a `version` tag can never be overwritten. Checked in `cli.py` and `src/primer/data/README.md`.
- **2026-10-25** — The existing 33 tests pass with the GCS guard in place (none of them touch GCS).
- **2026-10-25** — The hook scripts, the commit-msg hook and the docs check were run under Python
  3.14 on Linux: 169 tests pass and `ruff check .` is clean. **Not yet verified on Windows or
  macOS.** The hook command first tries `python3 -c ""` and falls back to `python` only if that
  fails, then runs the hook once: on Windows `python3` is often the Microsoft Store stub, and on
  macOS `python` often does not exist (open question 2).
- **2026-10-25** — `mypy src` reports 5 errors in 3 files of the existing pipeline code (listed under
  Later slices); `mypy` is therefore configured but not yet run in CI.
- **2026-10-25** — ruff 0.17 (what `ruff>=0.6` installs today) enables more rules by default than
  0.6; the existing code passes them.
- **2026-10-25** — `test_clear_snapshot_validate_preprocess` is flaky **before** this framework:
  1 of 40 runs failed on the unmodified branch, and 1 of 40 with the framework, with the same
  error. Cause confirmed by building `xlsx_bytes(clear_frame())` twice 1.1 s apart: the bytes
  differ. Fix listed under Later slices.
- **2026-10-25** — The hook commands are tested as Claude Code runs them
  (`tests/agent_hooks/test_settings_commands.py`, including a fake Windows `python3` stub). An
  earlier `python3 … || python …` command printed "BLOCKED" but exited 0 — the fallback re-ran the
  hook with its input already consumed — so the refusal never reached Claude Code. The test fails
  against that command.
- **2026-10-25** — Correction to the flaky-test entry above: fixed in S1, not a later slice. An
  `.xlsx` varies in two places — every zip entry's write time, and the created/modified dates in
  `docProps/core.xml` — so `xlsx_bytes` now fixes both after pandas writes the file. Evidence:
  `test_xlsx_bytes_is_deterministic` fails against the original helper and passes with the fix; the
  data tests passed 80 of 80 runs afterwards (the old rate predicts about 2 failures). Full suite:
  170 passed.
- **2026-10-25** — Correction to the `mypy` entry above: the 5 errors are fixed in S1 and
  `mypy src` now runs in `ci.yml` (after `ruff check .`). None was a runtime bug: two dictionaries
  that mypy inferred too narrowly (`manifest.py`, `versioning.py` — now `dict[str, Any]`); a
  `tarfile.extractfile` result that can be `None` on paper but is already guarded by `isfile()`
  (`storybooks.py` — now an explicit `None` check that skips the entry); and two `list[str]` slices
  returned where `list[str | None]` is promised (`align_images` — now copied with `list(...)`, as its
  `extra == 0` branch already did). `mypy src`: no issues in 19 files; `pytest`: 170 passed.
- **2026-10-25** — The commit-msg hook is now a POSIX shell launcher (`.githooks/commit-msg`) that
  picks `python3` or `python` the same way as the Claude hooks, then runs `.githooks/commit_msg.py`.
  The first version started with `#!/usr/bin/env python3`, which on Windows usually resolves to
  the Microsoft Store stub, so every commit would have failed once `core.hooksPath` was set.
  `tests/agent_hooks/test_git_hook.py` makes git run the hook with a working and a stubbed
  `python3`. Still not run on a real Windows machine (open question 2).

- **2026-10-25** — Confirmed on real Windows (Windows 11, Git Bash, Python 3.13.5): the
  `python3`-then-`python` launcher starts both Claude hooks and the git `commit-msg` hook. A bad
  commit subject is refused with exit 1 and HEAD unchanged; `gh pr merge` and a `primer-data` run
  against a `gs://` root are both BLOCKED with the refusal reaching the agent; `PRIMER_DATA_URI` is
  `.local-data`; all 12 skills load. `pytest`: 164 passed, 10 skipped (the 10 fake a broken Windows
  `python3` and need POSIX); `ruff check .`, `mypy src` and `scripts/check_docs.py` all clean. This
  answers open question 2 for Windows only — macOS and WSL are still unverified.
- **2026-10-25** — `guard_bash.py` `split_segments` scans heredoc *content* as commands, so a command
  whose heredoc body merely quotes a refused command is itself refused. Hit twice while finishing S1:
  a heredoc mentioning the `gh pr merge` check was BLOCKED, both when writing the PR body and when
  editing this plan. Worked around by writing the text to a file first; raised as a reviewer
  question on the PR.

- **2026-10-25** — Correction to the Windows entry above: a clean `ruff check .` on Windows does
  not mean a clean one on Linux. The first CI run on PR #1 passed `pr-rules` but failed `test` at
  the `ruff check .` step with 4 x EXE001 ("shebang is present but file is not executable") for
  `guard_bash.py`, `session_start.py`, `check_docs.py` and `check_pr.py`. Ruff skips EXE001 on
  Windows, so the local run was clean. Root cause: the patch declares mode 100755 for three of
  those files, but a Windows clone has `core.fileMode=false`, so applying it dropped the bit and
  they were committed 100644. `check_docs.py` and `check_pr.py` are 100644 in the patch itself,
  with shebangs, so they would fail EXE001 on Linux from a fresh apply on any OS.
- **2026-10-25** — The same dropped bit left `.githooks/commit-msg` non-executable. Git silently
  skips a non-executable hook on POSIX, so a teammate cloning on WSL or macOS and running
  `git config core.hooksPath .githooks` would have had **no local commit-subject check at all**,
  with no error to tell them. `pr-rules.yml` still caught bad subjects at PR time, so the rule held;
  only the fast local feedback was lost. Fixed in this PR with `git update-index --chmod=+x` on all
  five files (blob hashes unchanged). `commit_msg.py` stays 100644: no shebang, invoked through the
  launcher. Open question 2 should therefore be checked on a real WSL and macOS clone, not inferred.

- **2026-10-25** — The Atlassian tool names in `.claude/settings.json` are now verified against
  the authenticated `atlassian` server (site `northeastern-team-12`, Jira project KAN), replacing
  names inferred before authentication. Three of the inferred names did not exist on this server
  (`createConfluenceContent`, `updateConfluenceContent`, `executeWrite`, `executeDestructive` are
  from a different Atlassian tool layout), so they matched nothing and guarded nothing. The real
  write tools now in `permissions.ask` are `createJiraIssue`, `editJiraIssue`,
  `transitionJiraIssue`, `createIssueLink`, `addWorklogToJiraIssue`, `createConfluencePage` and
  `updateConfluencePage`. The server exposes no delete tool for issues or pages. Comment tools
  (`addCommentToJiraIssue`, `createConfluenceFooterComment`, `createConfluenceInlineComment`) are
  deliberately left unguarded, per AGENTS.md. Lesson: MCP permission entries are silent when the
  tool name is wrong, so they must be checked against a live server rather than inferred.

- **2026-10-26** — The `permissions` block in `.claude/settings.json` did not apply in a Claude Code
  session running from a git worktree under `.claude/worktrees/`. In Manual (default) mode
  `gh run rerun 1` and `git tag --list` both ran with no prompt, although `Bash(gh run rerun:*)` and
  `Bash(git tag:*)` are both in `permissions.ask` — the latter since before that session started.
  Ruled out: permission mode, mid-session settings loading, the `cd ... &&` prefix,
  `settings.local.json`, user-level settings. `hooks` and `env` from the same file do apply, so the
  file is read. Cause unproven; the clean test is a fresh session in the repository root, Manual
  mode, `git tag --list`. Consequence: `guard_bash.py` is what actually enforces, and it covers
  every `deny` entry, so nothing lost its only protection. The `ask` entries have no hook behind
  them, so "ask first" for Jira writes, PR reviews and tags is weaker than the config suggests.

- **2026-10-26** — Refines the entry above, and lowers its severity. The session that saw no
  permission prompts had **no `.claude/` directory in its project root at all**: it ran from the
  worktree `.claude/worktrees/ai-coding-framework-setup-8f091b`, checked out at `a66b451` on a
  branch predating the framework, so `.claude/settings.json` did not exist there. There were never
  any project permission rules to apply, which is a far more ordinary explanation than "worktree
  sessions ignore permissions". A normal clone has the file at its root, so a teammate is probably
  unaffected — still unverified, because the test needs a session whose project root contains the
  file, and a session started from this worktree inherits the worktree. Run it from a terminal in
  the repository root instead. Unexplained: `hooks` and `env` did apply in that session despite the
  missing file, most likely configured by the desktop app from the origin checkout when it created
  the worktree session. Unchanged: `guard_bash.py` is what demonstrably enforces, and it covers
  every `deny` entry.

## Open questions

1. ~~Who owns the agent and process files in CODEOWNERS once workstream leads exist?~~
   **Resolved 2026-10-25**: every collaborator owns them; any collaborator except the author
   reviews. See Decisions. Revisit only if the team grows past the point where pinging everyone
   is reasonable, or if the repository moves to an organisation and teams become available.
2. Does the hook command start on every teammate's OS (PowerShell-only Windows, Git Bash, macOS)?
   *Each teammate runs `/hooks` once in Claude Code and confirms the two hooks are listed, then
   runs the check in AGENTS.md → "Check your setup".*
3. Retire `dashboards/tracker/project.yaml` or generate it from Jira? *Recommended: generate it, or
   delete it; two trackers drift.*
4. How long do a full `pytest` and a full ingest take? Needed before setting CI timeouts.

## Admin commands (for S2)

```bash
gh label create needs-reviewer --color FBCA04 --description "Waiting for a teammate to claim review"
gh label create agent-authored --color 5319E7 --description "Written by an AI coding agent"
gh api -X POST repos/keithfernandes12/young-illustrators-primer/rulesets --input .github/rulesets/protect-main.json
```

## Out of scope

- Model training, serving and frontend code — each gets its own plan.
- Cloud infrastructure and secrets — after GCP access exists.
