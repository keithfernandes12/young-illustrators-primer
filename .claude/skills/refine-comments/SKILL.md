---
name: refine-comments
description: Clean up comments and docstrings in given Python files — remove comments that restate the code or are outdated, add missing "why" — without changing any logic. Use when asked to improve, clean up or review comments or docstrings.
---

# Refine comments

Comments only. If you find a bug, report it; do not fix it here.

## Remove

- Comments that restate the code (`# increment i`, `# return the result`).
- Outdated comments that no longer match the code.
- Commented-out code (git keeps history).
- Docstring sections that repeat type hints (`:param x: int`).

## Keep or add

- **Why**, not what: a non-obvious decision, a trade-off, a workaround and what it works around.
- Edge cases and invariants: "Pages are matched to images by count; anything else is marked
  unaligned rather than guessed."
- Safety, privacy and data-integrity reasons: "Never rewrite raw/: version image_uris point here."
- Public functions and classes: a one-line docstring saying what it is for and any side effect
  (writes to the store, network calls).

## Steps

1. Read each file in full.
2. List removals and additions per file; show them to the person for larger files.
3. Edit. Run `ruff check .`, `mypy src` and `pytest` to prove nothing else changed.
4. Summarise per file: removed N, added N, and anything that looked like a bug.

Commit as `docs(<area>): clarify comments in <module> (KAN-<n>)`.
