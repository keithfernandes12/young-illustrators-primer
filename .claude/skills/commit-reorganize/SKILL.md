---
name: commit-reorganize
description: Rewrite the current branch's history into clean, ordered commits — structure separate from behaviour, foundations first, each with a correct subject — without changing the final code, with a backup tag and a check that the result is identical. Use when asked to clean up, reorganize or squash commits, or before review when the branch has WIP commits.
---

# Reorganize commits

Changes history only; the final code must be byte-for-byte the same.

## Steps

1. **Set aside what does not belong.** `git status`. Stash scratch files:
   `git stash push -m "ephemeral: reorganize" -- <files>`. Commit any real uncommitted work as
   `chore: temp (KAN-<n>)`.
2. **Back up and flatten.**
   ```bash
   git fetch origin main
   git tag backup-<branch>-$(date +%Y%m%d%H%M%S)
   git rebase origin/main
   git reset --soft origin/main      # the whole branch is now one staged change
   ```
3. **Read the full change:** `git diff --cached`.
4. **Plan the commits.** Each is *either* structure (moves, renames, extraction, formatting) *or*
   behaviour, never both. Order: structure and tidying → core modules → their tests → features →
   feature tests → bug fixes → config and CI → docs (including the plan status update). Show the
   plan to the person.
5. **Commit each group.** `git reset`, then stage and commit group by group with
   `type(scope): summary (KAN-<n>)` and a short body saying why.
6. **Prove nothing changed:** `git diff backup-<tag>` must be empty. If not, stop and investigate.
7. **Push:** `git push --force-with-lease` is refused for agents by the guard hook. Tell the person
   the branch is ready and ask them to run `! git push --force-with-lease`.
8. **Restore scratch files:** `git stash list`, then `git stash pop` the "ephemeral: reorganize" entry.

## Recover

`git reset --hard backup-<tag>` restores the branch exactly. Delete the tag when the person is happy.

## Never

Edit code while reorganizing; mix structure and behaviour in one commit; leave out real work; commit
scratch files; rewrite a branch that a stacked branch above it has been reviewed on.
