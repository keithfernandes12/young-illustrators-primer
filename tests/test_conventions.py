"""Shared conventions that independently built slices must not drift from (cross-slice-coherence S1).

Slices are built in parallel and never read each other. A slice built before a shared helper landed
re-implements it, and nothing notices until a merge (S6f wrote its own MLflow tags before S6c3 added
`drawing/tracking.py`). These tests turn "someone should notice" into a failing test.

Four checks, each a scan of `src/` with `ast` (no imports of the scanned code, no weights, no
network, no MLflow):

1. **Starting an MLflow run** -- every ``.start_run(`` call is in the shared helper or an allowlisted
   file, at no more than its allowlisted count.
2. **The seven required tags** (`docs/engineering/model-development.md`) are in every tag dict that is
   a literal at the call site, or a name bound to a literal in the same function. A dict built
   elsewhere cannot be checked statically; its allowlist entry says so.
3. **Git state comes from one function.** A new ``*git_sha*`` or ``*git_state*`` function fails.
4. **No literal owner.** A dict with an ``"owner"`` key and a string-literal value fails: every member
   would be logged as the one person whose name is in the code.

**The allowlists are today's exceptions, not permission.** Each entry names a reason and a Jira key.
An entry that no longer matches the code also fails, so the list shrinks as areas migrate instead of
silently keeping a stale exception. When an area moves onto the shared helper, delete its entry in
the same PR.

Each failure message names the helper to use instead.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: `docs/engineering/model-development.md` -> Tracking conventions.
REQUIRED_TAGS = frozenset({"stage", "owner", "data_version", "git_sha", "base_model", "gate_group", "where"})

#: Where a slice should start a run and read git state. S2 moves the git helper to a shared module;
#: changing these two constants is all this file needs then.
SHARED_LOGGER = "primer/drawing/tracking.py"
SHARED_GIT = ("primer/drawing/evaluate.py", "_git_state")

GIT_HELPER = re.compile(r"git_(sha|state)")


@dataclass(frozen=True)
class Allowed:
    """One existing exception: how many, why, and the ticket that will remove it."""

    count: int
    reason: str
    jira: str


#: ``.start_run(`` call sites outside the shared logger, by file, with how many each has today.
START_RUN_ALLOWED: dict[str, Allowed] = {
    "primer/drawing/sensitivity.py": Allowed(
        1, "S6f built its own run before tracking.py existed; PR #46 moves it onto tracking.py", "KAN-38"
    ),
    "primer/speech/train.py": Allowed(
        2, "speech's own log_run, and one re-open of a run by id; gaps listed in KAN-128", "KAN-128"
    ),
    "primer/story/grid.py": Allowed(2, "the story grid's MlflowLogger predates tracking.py", "KAN-39"),
    "primer/illustration/consistency/experiment.py": Allowed(
        1, "the consistency threshold experiment predates tracking.py", "KAN-103"
    ),
}

#: Tag dicts that cannot be read statically, by file. Checked only that they exist, so a site that
#: later builds a literal is checked from then on.
TAGS_UNCHECKED: dict[str, str] = {
    "primer/drawing/sensitivity.py": "passes dict(tags) to start_run; PR #46 moves it onto run_tags (KAN-38)",
    "primer/story/grid.py": "tags arrive as a parameter of MlflowLogger.log_run / log_files (KAN-39)",
}

#: Git-state functions other than the shared one, as (file, function name).
GIT_ALLOWED: dict[tuple[str, str], Allowed] = {
    ("primer/evaluation/golden.py", "_git_sha"): Allowed(
        1, "the golden runner's own copy; S2 of cross-slice-coherence moves it", "KAN-89"
    ),
    ("primer/speech/evaluate.py", "git_sha"): Allowed(1, "speech's copy, no dirty flag (KAN-128)", "KAN-97"),
    ("primer/story/grid.py", "git_sha"): Allowed(1, "the story grid's copy", "KAN-39"),
    ("primer/illustration/consistency/experiment.py", "git_sha"): Allowed(
        1, "the consistency experiment's copy", "KAN-103"
    ),
}

#: Files with a literal owner in a dict, with how many.
OWNER_ALLOWED: dict[str, Allowed] = {
    "primer/drawing/sensitivity.py": Allowed(1, "hard-coded owner; removed by PR #46", "KAN-38"),
}


# ------------------------------------------------------------------------------------------ scans


def _modules(root: Path) -> Iterator[tuple[str, ast.Module]]:
    for path in sorted(root.rglob("*.py")):
        yield path.relative_to(root).as_posix(), ast.parse(path.read_text(encoding="utf-8"), str(path))


def start_run_counts(root: Path) -> dict[str, int]:
    """``.start_run(`` calls per file: ``mlflow.start_run`` and any attribute so named
    (``self._mlflow.start_run`` in the story grid)."""
    counts: dict[str, int] = {}
    for name, tree in _modules(root):
        n = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "start_run"
        )
        if n:
            counts[name] = n
    return counts


def _literal_keys(node: ast.AST) -> set[str] | None:
    if isinstance(node, ast.Dict) and all(isinstance(k, ast.Constant) for k in node.keys):
        return {str(k.value) for k in node.keys if isinstance(k, ast.Constant)}
    return None


def _bound_literal(name: str, scope: ast.AST) -> ast.AST | None:
    """The dict literal last assigned to ``name`` in ``scope``, if that is how it was built."""
    found = None
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            found = node.value
    return found


def tag_dicts(root: Path) -> dict[str, list[set[str] | None]]:
    """Per file, the key sets of every tag dict given to ``.start_run(tags=…)`` or ``.set_tags(…)``.

    ``None`` stands for a dict that is not a literal and is not bound to one in the same function.
    """
    found: dict[str, list[set[str] | None]] = {}
    for name, tree in _modules(root):
        scopes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
        for scope in scopes:
            for node in ast.walk(scope):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr == "start_run":
                    given = next((k.value for k in node.keywords if k.arg == "tags"), None)
                elif node.func.attr == "set_tags" and node.args:
                    given = node.args[0]
                else:
                    continue
                if given is None:
                    continue  # a run with no tags here, e.g. re-opened by id
                if isinstance(given, ast.Name):
                    given = _bound_literal(given.id, scope) or given
                found.setdefault(name, []).append(_literal_keys(given))
    return found


def git_helpers(root: Path) -> set[tuple[str, str]]:
    """Every ``(file, function)`` whose name contains ``git_sha`` or ``git_state``."""
    return {
        (name, node.name)
        for name, tree in _modules(root)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and GIT_HELPER.search(node.name)
    }


def literal_owners(root: Path) -> dict[str, int]:
    """Per file, dicts whose ``"owner"`` key maps to a string literal."""
    counts: dict[str, int] = {}
    for name, tree in _modules(root):
        n = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "owner"
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    ):
                        n += 1
        if n:
            counts[name] = n
    return counts


# --------------------------------------------------------------------------------------- verdicts


def start_run_problems(counts: dict[str, int], allowed: dict[str, Allowed]) -> list[str]:
    problems = []
    for name, n in sorted(counts.items()):
        if name == SHARED_LOGGER:
            continue
        limit = allowed[name].count if name in allowed else 0
        if n > limit:
            problems.append(
                f"{name} starts {n} MLflow run(s), {limit} allowed: log through {SHARED_LOGGER} "
                "(`log_run`, `run_tags`, `run_params`) instead of a new `start_run`"
            )
    for name, entry in sorted(allowed.items()):
        if counts.get(name, 0) < entry.count:
            problems.append(
                f"{name}: the allowlist expects {entry.count} `start_run` call(s) and the code has "
                f"{counts.get(name, 0)}; lower or delete its entry ({entry.jira})"
            )
    return problems


def tag_problems(found: dict[str, list[set[str] | None]], unchecked: dict[str, str]) -> list[str]:
    problems = []
    for name, sets in sorted(found.items()):
        if name == SHARED_LOGGER:
            continue  # builds its tags with `run_tags`, checked directly by its own test below
        for keys in sets:
            if keys is None:
                if name not in unchecked:
                    problems.append(
                        f"{name} passes a tag dict that is not a literal, so the seven required tags "
                        f"cannot be checked: build it with {SHARED_LOGGER}'s `run_tags`, or add it to "
                        "TAGS_UNCHECKED with a reason"
                    )
                continue
            missing = sorted(REQUIRED_TAGS - keys)
            if missing:
                problems.append(
                    f"{name} logs a run without {missing}; `run_tags` in {SHARED_LOGGER} sets all seven"
                )
    for name in sorted(unchecked):
        if not any(keys is None for keys in found.get(name, [])):
            problems.append(
                f"{name} is in TAGS_UNCHECKED but has no uncheckable tag dict any more; delete its entry"
            )
    return problems


def git_problems(found: set[tuple[str, str]], allowed: dict[tuple[str, str], Allowed]) -> list[str]:
    shared = f"`{SHARED_GIT[1]}` in {SHARED_GIT[0]}"
    problems = [
        f"{name} defines {func}(): read the commit and dirty flag with {shared}, not a new copy"
        for name, func in sorted(found)
        if (name, func) != SHARED_GIT and (name, func) not in allowed
    ]
    problems += [
        f"{name}:{func} is allowlisted but no longer exists; delete its entry ({entry.jira})"
        for (name, func), entry in sorted(allowed.items())
        if (name, func) not in found
    ]
    if SHARED_GIT not in found:
        problems.append(f"the shared git helper {shared} is gone; point SHARED_GIT at its new home")
    return problems


def owner_problems(counts: dict[str, int], allowed: dict[str, Allowed]) -> list[str]:
    problems = []
    for name, n in sorted(counts.items()):
        limit = allowed[name].count if name in allowed else 0
        if n > limit:
            problems.append(
                f"{name} hard-codes an owner in a tag dict: take it from `--owner` and pass it to "
                f"`run_tags` in {SHARED_LOGGER}"
            )
    for name, entry in sorted(allowed.items()):
        if counts.get(name, 0) < entry.count:
            problems.append(
                f"{name} no longer hard-codes an owner; delete its OWNER_ALLOWED entry ({entry.jira})"
            )
    return problems


# ------------------------------------------------------------------------------------- on src/


def test_runs_are_started_only_by_the_shared_logger_or_a_listed_exception():
    assert start_run_problems(start_run_counts(SRC), START_RUN_ALLOWED) == []


def test_every_checkable_tag_dict_has_the_seven_required_tags():
    assert tag_problems(tag_dicts(SRC), TAGS_UNCHECKED) == []


def test_the_shared_logger_writes_the_seven_required_tags():
    from primer.drawing.tracking import run_tags

    tags = run_tags({"data_version": "v1.1", "encoder": "e", "git_sha": "abc"}, owner="m1", kind="accuracy")
    assert REQUIRED_TAGS <= set(tags)


def test_git_state_comes_from_one_function_or_a_listed_exception():
    assert git_problems(git_helpers(SRC), GIT_ALLOWED) == []


def test_no_owner_is_hard_coded_outside_a_listed_exception():
    assert owner_problems(literal_owners(SRC), OWNER_ALLOWED) == []


def test_every_exception_names_a_reason_and_a_jira_key():
    entries = [*START_RUN_ALLOWED.values(), *GIT_ALLOWED.values(), *OWNER_ALLOWED.values()]
    assert all(e.reason.strip() and re.fullmatch(r"KAN-\d+", e.jira) and e.count >= 1 for e in entries)
    assert all(re.search(r"KAN-\d+", reason) for reason in TAGS_UNCHECKED.values())


# ------------------------------------------------------------------------------ on fixture trees


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


TAGS_OK = (
    '{"stage": "s", "owner": owner, "data_version": "v", "git_sha": "g", '
    '"base_model": "b", "gate_group": "s", "where": "local"}'
)


def test_a_new_run_outside_the_shared_logger_is_caught(tmp_path):
    root = _tree(
        tmp_path,
        {"primer/new_area/train.py": "def f(m):\n    with m.start_run(run_name='x'):\n        pass\n"},
    )

    problems = start_run_problems(start_run_counts(root), {})

    assert len(problems) == 1 and SHARED_LOGGER in problems[0]


def test_a_second_run_in_an_allowlisted_file_is_caught(tmp_path):
    two = "def f(m):\n    m.start_run()\n    m.start_run()\n"
    root = _tree(tmp_path, {"primer/a.py": two})

    assert start_run_problems(start_run_counts(root), {"primer/a.py": Allowed(1, "r", "KAN-1")})
    assert not start_run_problems(start_run_counts(root), {"primer/a.py": Allowed(2, "r", "KAN-1")})


def test_an_allowlist_entry_the_code_no_longer_needs_is_caught(tmp_path):
    root = _tree(tmp_path, {"primer/a.py": "x = 1\n"})

    problems = start_run_problems(start_run_counts(root), {"primer/a.py": Allowed(1, "r", "KAN-1")})

    assert problems and "delete its entry" in problems[0]


def test_a_tag_dict_missing_a_required_tag_is_caught_directly_and_through_a_name(tmp_path):
    direct = "def f(m):\n    m.start_run(tags={'stage': 's'})\n"
    named = "def g(m, owner):\n    tags = {'stage': 's'}\n    m.start_run(tags=tags)\n"
    set_tags = "def h(m):\n    m.set_tags({'stage': 's'})\n"
    root = _tree(tmp_path, {"a.py": direct, "b.py": named, "c.py": set_tags})

    problems = tag_problems(tag_dicts(root), {})

    assert [p.split(" ")[0] for p in problems] == ["a.py", "b.py", "c.py"]


def test_a_complete_tag_dict_passes(tmp_path):
    root = _tree(tmp_path, {"a.py": f"def f(m, owner):\n    tags = {TAGS_OK}\n    m.start_run(tags=tags)\n"})

    assert tag_problems(tag_dicts(root), {}) == []


def test_an_uncheckable_tag_dict_needs_a_listed_reason(tmp_path):
    root = _tree(tmp_path, {"a.py": "def f(m, tags):\n    m.set_tags(tags)\n"})

    assert tag_problems(tag_dicts(root), {})
    assert tag_problems(tag_dicts(root), {"a.py": "param (KAN-1)"}) == []


def test_a_new_git_helper_is_caught_and_the_shared_one_is_not(tmp_path):
    shared_file, shared_name = SHARED_GIT
    root = _tree(
        tmp_path,
        {shared_file: f"def {shared_name}():\n    pass\n", "primer/other.py": "def _git_sha():\n    pass\n"},
    )

    problems = git_problems(git_helpers(root), {})

    assert len(problems) == 1 and "primer/other.py" in problems[0] and shared_name in problems[0]


def test_a_hard_coded_owner_is_caught_and_a_variable_one_is_not(tmp_path):
    root = _tree(
        tmp_path,
        {"a.py": "tags = {'owner': 'someone'}\n", "b.py": "def f(who):\n    return {'owner': who}\n"},
    )

    problems = owner_problems(literal_owners(root), {})

    assert len(problems) == 1 and problems[0].startswith("a.py")


@pytest.mark.parametrize("name", ["_git_sha", "git_sha", "_git_state", "read_git_state"])
def test_any_function_named_for_git_state_counts(tmp_path, name):
    root = _tree(tmp_path, {"a.py": f"def {name}():\n    pass\n"})

    assert git_helpers(root) == {("a.py", name)}
