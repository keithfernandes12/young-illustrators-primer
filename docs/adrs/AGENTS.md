# docs/adrs

An ADR records one decision that others must follow, and why: a library, a model, a data split, an
architecture boundary.

- Start from `_template.md`. `type: adr`; `status: draft` → `accepted` → `deprecated` or
  `superseded`.
- File name is a plain slug (`split-by-book.md`); no numbers.
- Do not edit an accepted ADR to change its decision. Write a new ADR with `supersedes: <old-slug>`
  and set the old one's `superseded_by`.
- Link the ADR from the plans and code areas it governs (`related` in the plan; a line in the area's
  `AGENTS.md` if agents must know it).
