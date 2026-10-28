---
name: planning-poker
description: Size Jira issues with T-shirt sizes, alone or as a team, and turn the sizes into hours. Use when asked to estimate, size, point or plan a sprint for Jira issues, or "planning poker".
---

# Planning poker

Size = time to an open PR, including implementation and automated tests.

| Size | Meaning | Standard hours | Conservative hours |
|---|---|---|---|
| XS | trivial, well understood | 1 | 2 |
| S | small, few unknowns | 2 | 3 |
| M | a day-ish, some unknowns | 3 | 5 |
| L | several days, real unknowns | 5 | 8 |
| XL | too big — break it down | 8 | 12 |

## Steps

1. Collect the issues: Jira keys (read them through Jira tools) or pasted summaries.
2. For each issue, show its summary and acceptance criteria and ask for a size, one issue at a time.
   **Team mode:** collect each teammate's size before revealing any; if sizes differ by two or more
   steps (S vs L), ask the high and low voters to explain, then re-vote.
3. Any XL: say it must be split, and suggest the split.
4. Ask which hours mapping to use.
5. Output a table (key, summary, size, hours), the total, and a list the person can paste into Jira.
   Do not change Jira issues without the person's OK.
