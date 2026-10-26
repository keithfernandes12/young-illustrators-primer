---
name: shape-feature
description: Turn a vague idea, Jira epic, scoping section or meeting decision into a brief in docs/briefs/ with the problem, vocabulary, options, open questions and candidate slices, ready to become a plan and Jira tasks. Use when asked to shape, scope, break down or think through a feature before building it, or when acceptance criteria are missing.
---

# Shape a feature

Output: a brief in `docs/briefs/<slug>.md` (via `write-doc`), status `draft`.

## 1. Sources

Read the Jira epic or issue, the relevant sections of `docs/project_scoping.md`, related plans and
ADRs, and the code that exists for this area. In the brief, list **what you read, what you did not
read, what you inferred, and what is missing**.

## 2. Product pass

Work out, and write down:

- Who the user is (child 3–5, child 5–8, parent, team member) and what they will be able to do.
- Acceptance criteria — numbered, each one testable.
- Vocabulary: the words for the new concepts, and words we will not use, so code, UI and docs agree.
- The things involved (drawings, stories, pages, target words…), their states, and what moves them
  between states.
- Edge and negative cases: empty input, an unrecognisable drawing, unsafe content, a slow model.

## 3. Production-shape check

For each concept: what is the real data behind it? What is the source of truth? Who owns ordering
and concurrency? Does one user action change one record or many? Any unknown answer means a
decision is needed before building — list it as a question.

Separate **decisions** (new boundaries, stored names, privacy, contracts other modules rely on —
these become ADRs) from ordinary engineering questions.

## 4. Questions

Ask the person one question at a time, each with your recommended answer. Record answers in the
brief.

## 5. Slices

Propose candidate slices using three rules:

1. Every slice ships something usable on its own.
2. Later slices must not break what earlier ones delivered.
3. Promises that cannot be added later (privacy, data format, safety) go first.

Recommend one order and say why.

## 6. Finish

End the brief with a status line: **ready for a plan**, **needs answers** (list them), or **needs a
decision** (name the ADR to write). When decided, set the brief to `decided` and create the plan with
`write-doc`. Do not create Jira issues without the person's OK; list the proposed ones instead.
