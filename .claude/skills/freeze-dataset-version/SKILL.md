---
name: freeze-dataset-version
description: Freeze all processed datasets as a named, permanent data version (primer-data version) and commit the DVC pointer files and git tag — the agent checks preconditions and does the git work; a person runs the irreversible bucket write. Use when asked to freeze, tag, cut or version the data, or "make data v0.x".
---

# Freeze a dataset version

A version can never be overwritten, and running it half-way leaves the bucket and git out of step.
Reference: `src/primer/data/README.md` → Versions.

## 1. Preconditions (check all; stop on any failure)

- On a branch `chore/KAN-<n>-data-<tag>` from current `main`, working tree clean.
- The tag is new: `primer-data versions` (reads the bucket; needs the person's credentials) does not
  list it, and `git tag -l "data-<tag>"` is empty.
- The latest runs for every source validated (check the tracker dashboard or
  `metadata/runs/` manifests via `primer-data versions`).
- The person confirms the tag and the message ("what changed since the last version").

## 2. The person runs the bucket write

Agents cannot run this (the guard hook refuses it). Give the person the exact command to type in
this session:

```
! primer-data version <tag> -m "<message>"
```

Ask them to paste the last lines of output. It writes `metadata/versions/<tag>.json` and refreshes
`data/*.dvc`.

## 3. The agent finishes the git side

```bash
git status                       # expect changes to data/*.dvc only
git add data/*.dvc
git commit -m "chore(data): freeze data version <tag> (KAN-<n>)"
git push -u origin HEAD
```

Ask before tagging; then `git tag data-<tag>`. Pushing the tag is refused for agents; ask the person
to run `! git push origin data-<tag>`. Open the PR with `create-pr`, noting the version in the body.

## 4. If it went wrong

- Step 2 failed with "exists" → the tag was already used; pick another. Never try to delete a version.
- Step 2 succeeded but step 3 failed or was interrupted → **do not run `version` again** (it will
  refuse, and retrying with a new tag creates a second version). Tell the person what state things are
  in: the version exists in the bucket; the pointer files may be uncommitted. Commit them as in step 3.
- Anything else unexpected → stop and report; do not improvise against the bucket.

## 5. Record it

Add a dated line to the relevant plan's Verified facts (tag, date, row counts from the output).
