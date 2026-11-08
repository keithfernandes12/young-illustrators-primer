---
name: privacy-guard
description: COPPA-oriented privacy guardrail for Young Illustrator's Primer. Use whenever code touches child profiles, parent accounts, drawings, audio/ASR, logging, analytics, BigQuery exports, Firestore/Cloud SQL schemas, URLs/query params, or prompts sent to an LLM; and before every PR. Runs a heuristic linter for PII leaks and lists the human-review checklist.
---

# Privacy guard

The product is for children aged 3–8. Commitments from the scoping doc:
parent-owned accounts, pseudonymous child IDs, audio discarded by default,
no PII in logs, drawings stored only where the parent opted in.

## Run the linter

```bash
python .claude/skills/privacy-guard/scripts/privacy_lint.py src app
```

Exit code 1 and `PRIVACY_RESULT: FAIL` when it finds something. Silence a confirmed false
positive by appending `# privacy: ok <reason>` to that line (reviewers will check the reason).

## What it flags

- Logging/printing/tracing of fields such as `child_name`, `first_name`, `email`, `phone`,
  `address`, `birthdate`/`dob`, `age` with a name, raw `audio`, raw `drawing`/image bytes, `transcript`.
- PII placed in URLs, query strings, Pub/Sub attributes, or GCS object names.
- Audio written to GCS/disk without checking a parent opt-in flag on the same code path.
- Hard-coded secrets (API keys, private keys).

## Human checklist (the linter cannot do these)

- Child is referenced by `child_id` (random UUID) everywhere except the parent-visible profile.
- Prompts sent to the story LLM contain only the child's chosen nickname if the parent allowed it.
- Retention: drawings/audio have lifecycle rules on their GCS buckets; deleting a parent account deletes the child's data.
- Logs and BigQuery tables contain no free-text the child produced.
