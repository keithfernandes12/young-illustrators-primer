"""A SKILL.md may not send an agent to a skill that is not here.

train-vertex named six "vendored" skills -- trl-training, huggingface-vision-trainer, two mlflow
ones, agent-platform-model-registry and gcloud -- none of which is in this repository. An agent
reading that either hunts for them or quietly assumes capabilities it does not have.

The convention these tests enforce:

- `Repo skills: `a`, `b`` names skills that must exist under .claude/skills/. Checked.
- Anything from outside is described as **not in this repository**, and what the skill actually
  needs from it is inlined.
- Nothing is called "vendored" unless it really is vendored here.

A free scan of backticked words cannot do this job: SKILL.md files are full of label names, branch
names and field names that look identical to skill names (`needs-reviewer`, `main`, `candidate`),
so the declaration line is what makes the reference machine-checkable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
SKILL_FILES = sorted(SKILLS_DIR.glob("*/SKILL.md"))

REPO_SKILLS = {path.parent.name for path in SKILL_FILES}
DECLARATION = re.compile(r"^Repo skills:\s*(.+)$", re.MULTILINE)
BACKTICKED = re.compile(r"`([a-z][a-z0-9-]*)`")


def test_skills_were_found():
    assert SKILL_FILES, "no SKILL.md files found"
    assert "train-vertex" in REPO_SKILLS


@pytest.mark.parametrize("skill_file", SKILL_FILES, ids=lambda p: p.parent.name)
def test_declared_repo_skills_resolve(skill_file):
    """Every skill named on a `Repo skills:` line must be a directory under .claude/skills/."""
    text = skill_file.read_text(encoding="utf-8")
    for declaration in DECLARATION.findall(text):
        named = BACKTICKED.findall(declaration)
        assert named, f"{skill_file.parent.name}: a Repo skills line names nothing: {declaration!r}"
        for name in named:
            assert name in REPO_SKILLS, (
                f"{skill_file.parent.name} declares repo skill {name!r}, which is not a directory "
                f"under .claude/skills/. Either add it or move it to the external paragraph."
            )


@pytest.mark.parametrize("skill_file", SKILL_FILES, ids=lambda p: p.parent.name)
def test_nothing_claims_to_be_vendored_unless_it_is(skill_file):
    """"Vendored" means a copy lives here. Six skills claimed it and none of them did."""
    text = skill_file.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "vendored" not in line.lower():
            continue
        # A line may say what is NOT vendored; it may not assert that something IS.
        assert re.search(r"not\s+vendored|nothing\s+else\s+is\s+vendored", line, re.IGNORECASE), (
            f"{skill_file.parent.name} calls something vendored: {line.strip()!r}. "
            f"If it is not a directory under .claude/skills/, say so and inline what is needed."
        )


ADR = REPO_ROOT / "docs" / "adrs" / "where-model-training-happens.md"
AGENT_INSTRUCTIONS = sorted((REPO_ROOT / ".claude").glob("*/*.md")) + SKILL_FILES

#: Phrases that tell an agent to start training. The ADR says agents never do.
LAUNCH_BUDGET = re.compile(
    r"jobs have been launched|launches per /goal|launch (?:up to )?\d+ (?:vertex|training|gpu)",
    re.IGNORECASE,
)


def test_the_training_adr_exists_and_says_agents_do_not_train():
    assert ADR.exists()
    text = ADR.read_text(encoding="utf-8")
    assert "Agent sessions never train" in text


@pytest.mark.parametrize("path", AGENT_INSTRUCTIONS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_nothing_tells_an_agent_to_launch_training(path):
    """`goal-golden-gate.md` told an agent to launch four Vertex jobs, which the ADR forbids.

    A loop prompt that contradicts an accepted ADR is worse than no prompt: the agent follows the
    nearest instruction, not the decision record.
    """
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if LAUNCH_BUDGET.search(line) and "person" not in line.lower():
            raise AssertionError(
                f"{path.relative_to(REPO_ROOT)}:{number} sets an agent training budget, but "
                f"docs/adrs/where-model-training-happens.md says agents never train:\n  {line.strip()}"
            )


@pytest.mark.parametrize(
    "skill_file",
    [p for p in SKILL_FILES if p.parent.name in {"train-vertex", "deploy-cloud-run"}],
    ids=lambda p: p.parent.name,
)
def test_cloud_skills_mark_their_external_dependencies(skill_file):
    """The two cloud skills lean on user-level installs; that has to be visible, not assumed."""
    text = skill_file.read_text(encoding="utf-8")
    assert "not in this repository" in text.lower(), (
        f"{skill_file.parent.name} names external skills without saying they are absent"
    )
    assert "gcloud help" in text, "the one habit worth inlining from the gcloud skill is missing"
