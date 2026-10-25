---
name: split-pr
description: Split work into smaller pull requests — extract the staged changes into their own PR while keeping the rest, or split an oversized branch into 2–4 focused PRs (parallel or stacked) and prove they add back up to the original. Use when a PR is over ~400 lines, mixes concerns, a reviewer says it is too big, or when asked to extract, split, unbundle or stack changes.
---

# Split a pull request

## Mode A — extract staged changes

1. Note the current branch. Review `git diff --cached`.
2. `git stash push --keep-index -m "WIP: keep for <branch>"`.
3. Commit the staged changes (`type(scope): summary (KAN-<n>)`); note the hash.
4. `git switch -c <type>/KAN-<n>-<slug> origin/main` and `git cherry-pick <hash>`.
5. Push and open the PR with `create-pr`.
6. **Always** return: `git switch <original>`, `git stash pop` — even if a step failed.

## Mode B — split a branch

1. **Flatten** as in `commit-reorganize` steps 1–3 (backup tag, rebase, soft reset).
2. **Group files by concern:** tidy/refactor, fixes, features, tests with their code, CI/config,
   docs. Docs-only and CI-only changes are quick to review — give them their own PR.
3. **Choose a shape:**
   - *Parallel* — groups do not depend on each other; each branches from `main`. Prefer this.
   - *Stacked* — B needs A. At most 2 deep (repo rule).
   Size alone is not a reason to stack.
   Guide: 300–600 lines → 2 PRs, 600–1200 → 3, 1200–2000 → 4, more → reconsider the scope.
   Show the plan to the person before creating branches.
4. **Build each branch:** `git switch -c <type>/KAN-<n>-<slug> <base>`;
   `git checkout <source-branch> -- <files>`; commit structure, then behaviour.
5. **Each branch passes on its own:** `ruff check . && mypy src && pytest`, no imports of code that lives on
   another branch.
6. **Prove nothing was lost:** merge all new branches into a temporary branch off `main`;
   `git diff <source-branch>` must be empty. Delete the temporary branch.
7. **Open the PRs** with `create-pr`. For a stack, start each body with a **Stack** line: merge
   order, this PR's base, which PRs are independent. Update the plan's slices to match.

## Stacks

Fixes go on the top branch; never rewrite a branch below one that has been reviewed. When the lower
PR merges, rebase the upper branch on `main` and ask the person to force-push it
(`! git push --force-with-lease`).

## Never

Edit code while splitting; make more than 4 PRs; leave the person on a different branch than they
started.
