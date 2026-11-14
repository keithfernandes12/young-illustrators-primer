# AGENTS.md

How to work in this repo — for AI coding agents and for people. Read this first, then the nearest
`AGENTS.md` in every folder you touch. The rules marked *enforced* are checked by a hook, a test or
CI, and the message you get when you break one says what to do instead.

## What this is

Young Illustrator's Primer: a child draws characters and scenes; the system recognizes them, writes
a story at the child's reading level around parent-chosen target words, illustrates each page
keeping the child's design, and narrates it. Two age modes (3–5 Learn to Read, 5–8 Grow
Vocabulary), a parent dashboard, and an internal model-health dashboard. MLOps course project
(IE 7374), MVP due Nov 15. Scope and architecture authority: `docs/project_scoping.md` (§9
deployment, §8 failure analysis).

## Layout

- `src/primer/data/` — data pipeline, the only real code so far. Read `src/primer/data/AGENTS.md`.
- `src/primer/{drawing,story,readability,illustration,speech,safety,serving}/` — empty, planned.
- `configs/sources.yaml` — dataset registry. `data/` — DVC pointer files only, never data.
- `dags/` — Airflow DAGs (`dags/README.md`); `docker-compose.yaml` and `infra/airflow/` run them.
- `dashboards/tracker/` — Streamlit pipeline tracker. `app/`, `notebooks/` — empty.
- `docs/` — decisions, plans, procedures. See **Documents** below.
- `scripts/` — repo checks (`check_docs.py`, `check_pr.py`).
- `.claude/` — agent settings, guard hooks and skills. `.githooks/` — the commit-message check.
- New code goes under `src/primer/<area>/`. The old `src/<area>/` layout is closed.

## Commands

```bash
pip install -e ".[dev,dvc]"                  # setup (Python >= 3.11)
git config core.hooksPath .githooks          # once per clone: commit-message check
pytest                                       # all tests (includes hook, docs and no-GCS checks)
pytest tests/data/test_sources.py            # one file
ruff check .                                 # lint (line length 110)
ruff format .                                # format
mypy src                                     # type-check
python scripts/check_docs.py                 # docs frontmatter (also runs inside pytest)
python .claude/skills/verify/scripts/verify.py   # everything CI runs, plus privacy lint; one PASS/FAIL line
primer-data --data-uri .local-data run <source> --batch-size 2   # pipeline, locally
pip install -e ".[dashboard]" && streamlit run dashboards/tracker/app.py
```

`--data-uri` is a global flag: it goes **before** the command.

## Check your setup

At the start of every Claude Code session a hook checks this clone and, if something is missing,
the agent will ask you to run it. To check by hand: `git config core.hooksPath` prints `.githooks`;
`primer-data sources` runs; in Claude Code, `/hooks` lists the two hooks and `/mcp` shows
`atlassian` connected.

**Run Python through the project's virtualenv, not through whatever `python` is on `PATH`.** After
`pip install -e ".[dev,dvc]"` the dependencies live in `.venv`, and the agent's shell does not
inherit an activation — its state does not survive between commands, so `source .venv/…/activate`
in one call is gone by the next. Bare `python -m pytest` therefore fails at `tests/conftest.py`
with `ModuleNotFoundError: gcsfs` while `.venv/Scripts/python.exe -m pytest` passes. Name the
interpreter explicitly (`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` elsewhere);
`permissions.allow` carries both spellings for the gates, so an unattended loop is not stopped by a
prompt on every `verify`. Launching Claude Code from an already-activated shell also works, because
the process inherits `PATH`.

Once per clone, also check that the permission rules load: in a session in Manual permission mode,
ask the agent to run `git tag --list`. It should ask you before running it, because
`Bash(git tag:*)` is in `permissions.ask`. If it runs without asking, the `permissions` block in
`.claude/settings.json` is not reaching that session — the usual cause is a session rooted in a
worktree under `.claude/worktrees/`, which has no `.claude/` of its own, so open the session from
the repository root instead. The guard hook works either way; it is the "ask first" rules that are
lost, so an agent could write to Jira or submit a PR review without checking. Say what you find in
`docs/plans/ai-coding-framework.md`.

## Rules that are easy to break

- **Never write to the shared bucket from an agent session.** *Enforced.* `primer-data run|ingest|
  validate|preprocess|version` write to `PRIMER_DATA_URI`, which defaults to
  `gs://young-illustrators-primer-data`. An accidental `run` advances `metadata/state/<source>.json`
  so the next real ingest silently skips records; a `version` tag can never be overwritten. Agent
  sessions get `PRIMER_DATA_URI=.local-data` from `.claude/settings.json`, and a hook refuses
  bucket writes. A person runs bucket writes themselves by typing `! primer-data …`.
- **Never modify or delete anything under `raw/` or `metadata/versions/`.** Every dataset version's
  picture-book `image_uri`s point into `raw/`; versions are frozen by GCS object generation.
- **Tests never touch GCS.** *Enforced* by `tests/conftest.py`. Use the fake fetcher and the local
  `store` fixture.
- **Every processed record carries `license` and `attribution`.** A record without them is a legal
  risk.
- **Split by book, never by page**, so pages of one book never land in both train and test.
- **No PII in logs, fixtures or commits.** Child data is COPPA-sensitive; use pseudonymous IDs.
- **Agent sessions never launch a cloud training job.** Vertex AI custom jobs, hyperparameter
  tuning jobs and Cloud Run jobs are a person's action, whatever the project. An agent **may**
  train locally, on the machine the session runs on, bounded by the area's `max_steps` and a
  30-minute wall clock; record the result as a dated Verified fact, and treat it as evidence
  about wiring, not about model quality. A member trains freely on their own machine. A run that
  gets registered must be reproducible from the area's container on Vertex AI, with the same
  frozen data version, logged to the shared MLflow. See
  `docs/adrs/bounded-local-training-for-agents.md`. Cap local *pipeline* runs with `--batch-size`.
- **Never create service-account JSON keys** or commit credentials. Use Application Default
  Credentials (`gcloud auth application-default login`, run by a person).
- **Never delete anything outside this repository.** The only exception is a temporary file or
  folder you created yourself for the task at hand, which you delete by the exact path you created.
  Never delete by a path you did not write, never recursively above the repository root, and never
  widen a delete with a glob. *Partly enforced*: `.claude/settings.json` denies the catastrophic
  forms, but there is no hook that resolves an arbitrary path, so most of this rule is still on you.
  When a delete is not obviously yours to make, leave the files and say so.
- **Prefer a model that fine-tunes comfortably on one local GPU.** Model selection is constrained
  by what a member can actually iterate on: a laptop RTX 4060 is 8 GB of VRAM, so a candidate that
  only fine-tunes on an A100 is the wrong default however good its benchmark. Reach for the smaller
  checkpoint first (a base or large tower over a giant one), use bf16 and gradient checkpointing,
  and keep anything that will not fit as inference-only or as a later cloud run. Record the VRAM a
  candidate actually needed as a dated Verified fact, measured rather than estimated.
- **Every model comparison is logged, not remembered.** Any run that produces a number someone
  might repeat — a candidate bake-off, a hyperparameter change, a capped local run — logs to MLflow
  with its params, metrics and the data version, so two candidates can be compared later without
  re-running either. A number that appears only in a transcript did not happen.
- **The GPU is reachable from Git Bash and WSL, not from PowerShell.** `docker run --gpus all` is
  verified working from Git Bash (`docs/engineering/model-containers-on-windows.md`). PowerShell
  does not reliably see the GPU, which is one reason `CLAUDE_CODE_USE_POWERSHELL_TOOL=0` is set for
  the project. Run anything that touches the GPU from Git Bash.

## Git and pull requests

- Branch: `<type>/KAN-<n>-<slug>`, from `main`, e.g. `feat/KAN-42-amateur-drawings`.
- Commit subject: `type(scope): imperative summary (KAN-<n>)`, e.g.
  `feat(data): add Amateur Drawings ingestion (KAN-42)`. Types: `feat fix docs refactor test chore
  ci perf`. Scope: the area (`data`, `story`, `serving`, `docs`, `agents`, …). *Enforced* by
  `.githooks/commit-msg` (which runs `commit_msg.py`) and CI.
- PR title uses the same format (a squash merge makes it the commit on `main`). *Enforced* by CI.
- Open PRs as **drafts** from `.github/pull_request_template.md`, with a Jira key in the body, the
  `agent-authored` label if an agent wrote it, and either a reviewer or the `needs-reviewer` label.
  *Enforced* for agents. Use the `create-pr` skill.
- Mark ready only after `ruff check .`, `mypy src` and `pytest` pass and the diff has been re-read.
- **Agents never merge, never push to `main`, never force-push, never delete branches, and use `gh`
  only on this repo.** *Enforced.* A teammate who did not write the PR approves and merges it.
- Agents may commit and push their own branch, open draft PRs, and comment on PRs and Jira issues.
  They ask first before creating or moving a Jira issue, submitting a PR review, or pushing a tag.
- Stacked PRs are fine, at most 2 deep. Fixes go on the top branch; never rewrite a branch below one
  that has been reviewed.

## Tests and CI

CI (`.github/workflows/`) runs exactly the commands above — `ruff check .`, `mypy src`, `pytest` —
plus the PR title, commit and Jira-key checks, on every PR. Full rules, and which are enforced versus
conventions: `docs/engineering/ci.md`. Planned CI changes: `docs/plans/ci-hardening.md`.

- **A red CI run on the PR's latest commit means not ready.** Use the `fix-ci` skill: read the failing
  step, reproduce it locally, fix the cause. **Never re-run a failed job hoping it passes.**
- **Never make CI green by weakening it**: no `skip`/`xfail`, deleted tests, `# type: ignore`,
  `# noqa`, removed steps or lowered thresholds without a reason in a comment and a reviewer's
  explicit agreement — and never in the same PR as the code it unblocks.
- **Tests are hermetic.** No bucket (*enforced*), no network, no wall-clock time or randomness the test
  does not control, no real children's data.
- **A bug fix includes a test that fails without the fix.** Say so in the PR body.
- **Flaky tests are fixed or quarantined, never retried away.** Prove it with a loop and compare with
  `main`; see `docs/engineering/ci.md` → Flaky tests.
- **CI changes** (`.github/`) go in their own PR and follow `docs/engineering/ci.md` → Changing CI:
  stable job names (they are required checks), read-only `permissions`, `timeout-minutes`, actions
  pinned to a commit, event data passed through `env:`, new gates warn-only first.

## Documents (`docs/`)

`docs/` is the team's shared memory: agents read it before starting work and write to it as they
go, so the next person or agent can continue without rediscovering context. Full rules:
`docs/engineering/documentation.md`. *Frontmatter is enforced* by `scripts/check_docs.py` (runs in
`pytest`).

| Folder | Holds |
|---|---|
| `docs/plans/` | How we will build something decided, as PR-sized slices — and the record of what was built. **Read the plan before working on its area; update its status table in the same PR.** |
| `docs/adrs/` | One decision others must follow, and why |
| `docs/briefs/` | Shaping an idea before it is decided |
| `docs/runbooks/` | Operating something that runs (freeze data, retrain, roll back) |
| `docs/solutions/` | A solved problem likely to come back — check here before debugging |
| `docs/engineering/` | How we work: `onboarding.md`, `documentation.md`, `engineering-loop.md`, `review-checklist.md`, `ci.md` |
| `docs/meeting-notes/` | What a meeting decided, with owners |

Every document starts with frontmatter (`title`, `type`, `created`, `status`, `author`; optional
`updated`, `jira`, `tags`, `related`, `supersedes`, `superseded_by`) and is named with a kebab-case
slug. Each folder has an `AGENTS.md` and a `_template.md`. Use the `write-doc` skill.

## How work is done

For anything bigger than a one-file fix, follow `docs/engineering/engineering-loop.md` (the
`engineering-loop` skill): read the sources, write the contract in the plan, build one slice per
PR, review yourself with `review-pr`, then open the PR. Before asking for review, apply
`docs/engineering/review-checklist.md`.

## Skills (`.claude/skills/`)

| Skill | Use it to |
|---|---|
| `engineering-loop` | take a change from request to reviewable PR, with a plan |
| `write-doc` | create or update a document in `docs/` with correct frontmatter |
| `shape-feature` | turn a vague idea or Jira epic into a brief with options and slices |
| `create-pr` | commit, push and open a correct draft PR |
| `review-pr` | review a PR or your own branch against the checklist, with finding categories |
| `fix-feedback` | work through review findings and record a disposition for each |
| `fix-ci` | diagnose and fix a red CI run without weakening CI |
| `pr-status` | find out what is going on with a PR and what to do next |
| `commit-reorganize` | turn a messy branch into clean, ordered commits |
| `split-pr` | split an oversized branch, or move staged changes to their own PR |
| `freeze-dataset-version` | prepare and finish a dataset version (a person runs the bucket write) |
| `planning-poker` | size Jira issues as a team |
| `refine-comments` | clean up code comments and docstrings without changing logic |
| `verify` | run every CI check plus the privacy and infra gates; ends `VERIFY_RESULT: PASS/FAIL` |
| `golden-gate` | compare golden-set metrics with the §7 targets; ends `GATE_RESULT: PASS/FAIL` |
| `privacy-guard` | lint for child PII in logs, URLs, object names and stored audio |
| `loop-worker` | run one unattended engineering-loop iteration on a plan (for `/goal`, `/loop`, Ralph) |
| `train-vertex` | launch and track a Vertex AI training job |
| `deploy-cloud-run` | canary and smoke-test a Cloud Run service; a person promotes and rolls back |

Loop prompts for `/goal`, `/loop` and Ralph live in `.claude/loops/`; a bare `/loop` reads
`.claude/loop.md`. Unattended work follows `docs/plans/autonomous-loops.md`. Google Cloud is one
**shared** team project holding everything durable (`$YIP_SHARED_PROJECT`) plus one **compute**
project per member for their own jobs (`$YIP_COMPUTE_PROJECT`); `.claude/hooks/guard_cloud.py`
accepts both and refuses any third. It guards the live serving path rather than a project name:
agents may read cloud state and deploy canaries with `--no-traffic`, while traffic changes,
deletes, IAM writes and training launches are left to a person.

## Tracking and communication

- Jira project **KAN** is the tracker (issue types: Epic, Feature, Task, Subtask — there is no
  "Story"). Do not use GitHub Issues. `dashboards/tracker/project.yaml` is not authoritative.
- Jira is connected through `.mcp.json` (each person authenticates once with `/mcp`).
- Agents post updates as Jira or PR comments. **Agents never post to Slack**: the workspace is free
  and shared with people outside the team.
