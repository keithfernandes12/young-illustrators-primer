# Ralph overnight prompt

Start (from the repository root, with the ralph-loop plugin installed):

    /ralph-loop "Read .claude/loops/ralph-overnight.md and follow the prompt below its --- line." --max-iterations 12 --completion-promise "LOOP_EMPTY"

Cancel with `/cancel-ralph`. The plugin's Stop hook applies to every session in this project while
`.claude/ralph-loop.local.md` exists, so do not run a second interactive session in the same clone.

Prompt (everything below this line is what each iteration reads):

---

You are one iteration of an overnight run. Previous iterations left their work in git, in open PRs and
in docs/plans/mvp-cloud-deploy.md; read those, not your memory.

Run exactly one loop-worker iteration on docs/plans/mvp-cloud-deploy.md. Only slices with a Jira key,
not marked person-only, whose dependencies are merged. Stop early with LOOP_WAIT if 3 agent-authored
PRs are open.

If no slice is ready, output <promise>LOOP_EMPTY</promise>. Otherwise end with the loop-worker line.
