---
title: Getting started as a new teammate
type: engineering
created: 2026-10-27
status: active
author: rishiparekh
tags: [onboarding, process, setup, agents]
related: [engineering-loop, review-checklist, documentation, ci]
---

# Getting started as a new teammate

Day one: set the clone up, then read the right things in the right order. This is a path **through**
the other documents, not a copy of them — where it disagrees with `AGENTS.md`, `AGENTS.md` wins.

## 1. Before you clone

Needed once per machine: **Python 3.11+**, **git**, and the **GitHub CLI** signed in
(`gh auth status` should show the account that can push). On Windows also **Git for Windows** —
Claude Code runs its commands and hooks through Git Bash — and Python from python.org with
"Add python.exe to PATH" ticked.

**Pick Windows or WSL and stay there.** Use one environment per checkout: a virtual environment
made on Windows has `.venv\Scripts`, WSL expects `.venv/bin`, and using one checkout from both
sides also makes git see file-mode noise. In WSL, clone under your home directory; a repo under
`/mnt/c/...` works but every git operation crosses into the Windows file system and is several
times slower.

## 2. Set the clone up

```bash
git clone git@github.com:keithfernandes12/young-illustrators-primer.git
cd young-illustrators-primer
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,dvc]"
git config core.hooksPath .githooks
```

That last line is the one everyone forgets. It turns on the commit-message check, so a bad commit
subject is refused locally instead of failing in CI ten minutes later.

## 3. Start Claude Code and let it check the rest

Run `claude` **from the activated environment, in the repository root**. Not from a session
worktree: a worktree checked out at a commit from before this framework existed has no `.claude/`
of its own, so the permission rules never load and the file-editing tools refuse to touch the real
checkout.

A hook runs at the start of every session, checks this clone and asks you to run anything missing —
so there is no setup document to remember. Then run `/mcp` once and authenticate `atlassian` for
Jira.

To check by hand, see `AGENTS.md` → **Check your setup**: `git config core.hooksPath` prints
`.githooks`, `primer-data sources` runs, `/hooks` lists the two project hooks, and asking the agent
to run `git tag --list` in Manual permission mode should make it ask you first.

## 4. Read these, in this order

| Read | What it gives you |
|---|---|
| `AGENTS.md` | The rules. Read it all — everything below expands on part of it |
| `docs/engineering/engineering-loop.md` | How a change goes from request to merged PR, in seven steps |
| `docs/engineering/review-checklist.md` | What a reviewer will check, and what Blocking / Fix / Tidy / Follow-up mean |
| `docs/engineering/ci.md` | What CI runs, what blocks a merge, and how a new rule gets added |
| `docs/engineering/documentation.md` | Which folder a document belongs in, and the frontmatter rules |
| `src/primer/data/AGENTS.md` and `src/primer/data/README.md` | Only if you touch the data pipeline |
| `docs/project_scoping.md` | What we are building and why (§3.3 splits, §8 failure analysis) |

Then read the nearest `AGENTS.md` in every folder you touch, and the plan in `docs/plans/` for the
area you are working on — plans are the record of what was built and why, not just what is planned.

## 5. Your first change

1. Confirm the Jira issue in project **KAN**. Never create or move one without asking.
2. Branch `<type>/KAN-<n>-<slug>` from `main`, e.g. `feat/KAN-42-amateur-drawings`.
3. Build in slices. Before you push: `ruff check .`, `mypy src`, `pytest` — the same three CI runs.
4. Commit as `type(scope): summary (KAN-<n>)`. The hook refuses anything else and tells you why.
5. Open a **draft** PR from the template, with the Jira key in the body and either a named reviewer
   or the `needs-reviewer` label.
6. Someone who did not write it reviews and merges it. You never merge your own PR.

The skills in `.claude/skills/` do most of this for you — `engineering-loop` to plan, `create-pr`
to open the PR, `review-pr` to review your own branch first, `fix-feedback` to work through review
comments, `pr-status` to find out what is blocking something. Ask the agent "what skills does this
repo have?" for the full list.

## 6. The things most likely to bite you

- **Never run `primer-data run|ingest|validate|preprocess|version` without `--data-uri .local-data`**
  unless you mean to write to the shared bucket. It defaults to
  `gs://young-illustrators-primer-data`. An accidental run advances `metadata/state/<source>.json`,
  so the next real ingest silently skips records, and a version tag can never be overwritten.
- **Never modify or delete anything under `raw/` or `metadata/versions/`.** Every dataset version's
  picture-book `image_uri`s point into `raw/`.
- **No PII, child names, drawings or audio** in logs, fixtures or commits. This is a COPPA-sensitive
  children's product.
- **The guardrails only constrain agents, not you.** The hook that refuses merges, force-pushes and
  bucket writes applies to an agent's commands. A command you type yourself runs — that is
  deliberate, so a person can always act on purpose, but it means the care is yours.
- **Tests never touch GCS**, and that one *is* enforced for everyone by `tests/conftest.py`.

## Where to ask

Jira project **KAN** — comments on the issue or the PR. Not Slack: the workspace is free and shared
with people outside the team, so nothing about this project goes there.
