Maintenance pass for this repo (bare `/loop` reads this file). Do the first item that applies, then stop:

1. If any of my open agent-authored PRs has a red check on its latest commit, run `fix-ci` on the
   oldest one. Never re-run a failed job hoping it passes.
2. If one of my open PRs has unaddressed review comments, run `fix-feedback` on it.
3. If a PR of mine is behind `main` and has no conflicts, report it (do not rebase reviewed branches).
4. Otherwise run one `loop-worker` iteration on docs/plans/mvp-cloud-deploy.md.

End with one line: what you did and the PR it concerned, or `NOTHING_TO_DO`.
