# docs/runbooks

A runbook operates something that runs: freeze a data version, run a retrain, roll back a deploy.
A procedure for how the team works (review, docs, git) belongs in `docs/engineering/` instead.

- Start from `_template.md`. `type: runbook`; `status: active` or `deprecated`.
- Numbered steps with the exact commands, what you should see after each, and how to undo it.
- Mark which steps only a person may run (anything that writes to the bucket, deploys, or changes
  credentials). Agents prepare those steps; people run them with the `!` prefix.
