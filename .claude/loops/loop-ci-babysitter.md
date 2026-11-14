/loop 15m For each of my open agent-authored PRs on this repo: if the latest commit's checks are red, run
fix-ci on it; if there are unaddressed review comments, run fix-feedback; otherwise do nothing.
One PR per run. Never re-run a failed job, never weaken CI, never mark a PR ready. End with one line:
the PR you touched and what you did, or NOTHING_TO_DO.
