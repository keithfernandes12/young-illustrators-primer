"""Shared conventions that independently built slices must not drift from (cross-slice-coherence S1).

Slices are built in parallel and never read each other. A slice built before a shared helper landed
re-implements it, and nothing notices until a merge (`mvp-cloud-deploy S6f` wrote its own MLflow tags
before `mvp-cloud-deploy S6c3` added `drawing/tracking.py`). These tests turn "someone should notice" into a failing test.

Four checks, each a scan of `src/` with `ast` (no imports of the scanned code, no weights, no
network, no MLflow):

1. **Starting an MLflow run** -- every ``.start_run(`` call is in the shared helper or an allowlisted
   file, at no more than its allowlisted count. What it cannot see: ``from mlflow import start_run as
   sr``, ``getattr(mlflow, "start_run")()``, ``s = mlflow.start_run; s()``. Any attribute named
   ``start_run`` counts, so ``timer.start_run()`` is a (tolerated) false positive.
2. **The seven required tags** (`docs/engineering/model-development.md`) are in every tag dict that is
   a literal at the call site, or a name bound to a literal in the same or an enclosing scope (keys
   added later with ``tags["k"] = v`` count, in source order). The calls read are
   ``start_run(tags=...)`` and ``.set_tags(...)`` on ``mlflow`` or ``self._mlflow``; another object's
   ``set_tags`` is not a tag write. A ``set_tags`` literal is held to all seven too, because any tag
   write is part of the run's identity: to add one tag to a run, pass ``{**run_tags(...), "k": v}``,
   which is not a literal and so takes a ``TAGS_UNCHECKED`` entry with its reason. A dict built
   elsewhere cannot be checked statically, and neither can one that ``tags.update(...)``,
   ``tags.setdefault(...)`` or ``tags |= ...`` changed: the scan marks it unreadable and its
   allowlist entry says so. What it cannot see: a dict passed through another function's argument,
   ``tags=`` given as the sixth positional argument or as ``**opts``, ``mlflow.set_tag("k", v)``
   (singular), and, where a name is bound in both branches of an ``if``, the earlier branch (the last
   binding wins).
3. **Git state comes from one function** among those named ``*git_sha*`` or ``*git_state*``; a new one
   fails. It does not see other names: ``data/versioning.py:_git_commit`` is a copy it deliberately
   leaves out (dataset-version metadata, not model-run provenance; see the plan's Verified facts).
4. **No literal owner in a tag dict.** A dict display that has an ``"owner"`` key with a string-literal
   value *and* at least three other required tag keys fails (so ``{"stage": ..., "owner": "x"}`` in a
   config is not a tag dict), as does a name bound to such a dict that later gets
   ``tags["owner"] = "x"``. Every member would be logged as the one person whose name is in the code.
   What it cannot see: ``mlflow.set_tag("owner", "x")``, ``dict(owner="x", ...)`` and
   ``{**base, "owner": "x"}``.

That the shared logger itself writes the seven tags is asserted once, by
`tests/drawing/test_tracking.py`; this file does not repeat it (nor import the scanned code).

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

#: Where a slice should start a run and read git state. cross-slice-coherence S2 moves the git helper to a shared module;
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
        1,
        "mvp-cloud-deploy S6f built its own run before tracking.py existed; PR #46 moves it onto it",
        "KAN-38",
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
        1, "the golden runner's own copy; cross-slice-coherence S2 moves it", "KAN-89"
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
    """``start_run(`` calls per file: ``mlflow.start_run``, any attribute so named
    (``self._mlflow.start_run`` in the story grid) and a bare ``start_run`` imported from mlflow."""
    counts: dict[str, int] = {}
    for name, tree in _modules(root):
        n = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call) and _callee(node) == "start_run")
        if n:
            counts[name] = n
    return counts


def _literal_keys(node: ast.AST) -> set[str] | None:
    if isinstance(node, ast.Dict) and all(isinstance(k, ast.Constant) for k in node.keys):
        return {str(k.value) for k in node.keys if isinstance(k, ast.Constant)}
    return None


def _callee(node: ast.Call) -> str | None:
    """The called name: ``x.start_run(...)`` and ``start_run(...)`` are both ``start_run``."""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _is_mlflow(func: ast.expr) -> bool:
    """Whether ``func`` is ``mlflow.<x>`` or ``self._mlflow.<x>``: the receivers that write MLflow tags."""
    if not isinstance(func, ast.Attribute):
        return False
    receiver = func.value
    if isinstance(receiver, ast.Name):
        return receiver.id == "mlflow"
    return isinstance(receiver, ast.Attribute) and receiver.attr == "_mlflow"


@dataclass
class _TagDict:
    """A dict literal as it stands at one point in the source: its keys and whether its owner is a
    string literal. ``keys`` is ``None`` for a dict that cannot be read (``**spread``, a call)."""

    keys: set[str] | None
    literal_owner: bool = False  # the display itself has ``"owner": "<str>"``
    owner_assigned: bool = False  # a later ``name["owner"] = "<str>"`` set it

    def snapshot(self) -> set[str] | None:
        return None if self.keys is None else set(self.keys)


def _literal(node: ast.AST) -> _TagDict | None:
    """The `_TagDict` for a dict display with constant keys, else ``None``."""
    keys = _literal_keys(node)
    if keys is None or not isinstance(node, ast.Dict):
        return None
    owner = any(
        isinstance(k, ast.Constant)
        and k.value == "owner"
        and isinstance(v, ast.Constant)
        and isinstance(v.value, str)
        for k, v in zip(node.keys, node.values, strict=True)
    )
    return _TagDict(keys, owner)


_Scope = dict[str, _TagDict | None]


class _TagScan(ast.NodeVisitor):
    """Visit every call once, in source order, and record the tag dict each ``start_run(tags=...)`` or
    ``set_tags(...)`` receives.

    A name resolves to the dict literal last bound to it in the nearest scope that binds it, with keys
    added by ``name["k"] = v`` up to that call. A function body is read after the scope that defines it
    (a closure sees the enclosing function's final bindings, as it would when called). Parameters shadow
    outer names and are unreadable.
    """

    def __init__(self) -> None:
        self.chain: list[_Scope] = [{}]
        self.deferred: list[tuple[ast.AST, list[_Scope]]] = []
        self.class_scopes: set[int] = set()
        self.given: list[_TagDict | None] = []
        self.bound: list[_TagDict] = []  # every dict literal bound to a name, as it ends up

    # scopes
    def scan(self, tree: ast.Module) -> list[_TagDict | None]:
        self._scope(tree.body, self.chain)
        return self.given

    def _scope(self, body: list[ast.stmt], chain: list[_Scope]) -> None:
        saved_chain, saved_deferred = self.chain, self.deferred
        self.chain, self.deferred = chain, []
        for stmt in body:
            self.visit(stmt)
        pending = self.deferred
        self.chain, self.deferred = saved_chain, saved_deferred
        for fn, fn_chain in pending:
            self._function(fn, fn_chain)

    def _function(self, fn: ast.AST, chain: list[_Scope]) -> None:
        assert isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
        args = fn.args
        params = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
        params += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
        stmts: list[ast.stmt] = fn.body if isinstance(fn.body, list) else [ast.Expr(fn.body)]
        self._scope(stmts, [*chain, dict.fromkeys(params)])

    def _defer(self, node: ast.AST) -> None:
        self.deferred.append((node, [s for s in self.chain if id(s) not in self.class_scopes]))

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._bind(node.name, None)
        self._defer(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._defer(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name, None)
        scope: _Scope = {}
        self.class_scopes.add(id(scope))
        self._scope(node.body, [*self.chain, scope])

    # bindings
    def _bind(self, name: str, value: _TagDict | None) -> None:
        self.chain[-1][name] = value

    def _lookup(self, name: str) -> _TagDict | None:
        for scope in reversed(self.chain):
            if name in scope:
                return scope[name]
        return None

    def _bind_literal(self, name: str, value: ast.AST) -> None:
        literal = _literal(value)
        if literal is not None:
            self.bound.append(literal)
        self._bind(name, literal)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._bind_literal(target.id, node.value)
            elif isinstance(target, ast.Subscript):
                self._add_key(target, node.value)
            else:
                self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            if isinstance(node.target, ast.Name):
                self._bind_literal(node.target.id, node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and (held := self._lookup(node.target.id)) is not None:
            held.keys = None  # ``tags |= {...}``: keys we do not track

    def _add_key(self, target: ast.Subscript, value: ast.AST) -> None:
        if not isinstance(target.value, ast.Name):
            return
        held = self._lookup(target.value.id)
        key = target.slice
        if held is None:
            return
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            held.keys = None  # a key we cannot read: the dict can no longer be checked
            return
        if held.keys is not None:
            held.keys.add(key.value)
        if key.value == "owner":
            held.owner_assigned = isinstance(value, ast.Constant) and isinstance(value.value, str)

    # calls
    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        callee = _callee(node)
        given: ast.AST | None
        if callee in ("update", "setdefault") and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and (held := self._lookup(node.func.value.id)):
                held.keys = None  # the dict changed in a way we do not track
            return
        if callee == "start_run":
            given = next((k.value for k in node.keywords if k.arg == "tags"), None)
        elif callee == "set_tags" and _is_mlflow(node.func) and node.args:
            given = node.args[0]
        else:
            return
        if given is None:
            return  # a run with no tags here, e.g. re-opened by id
        if isinstance(given, ast.Name):
            held = self._lookup(given.id)
            self.given.append(_TagDict(held.snapshot()) if held else None)
        else:
            self.given.append(_literal(given))


def tag_dicts(root: Path) -> dict[str, list[set[str] | None]]:
    """Per file, the key sets of every tag dict given to ``start_run(tags=...)`` or ``.set_tags(...)``.

    ``None`` stands for a dict that is not a literal and is not bound to one (see `_TagScan`).
    """
    found: dict[str, list[set[str] | None]] = {}
    for name, tree in _modules(root):
        for given in _TagScan().scan(tree):
            found.setdefault(name, []).append(None if given is None else given.keys)
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
    """Per file, tag dicts whose ``"owner"`` maps to a string literal.

    A tag dict has at least three other required tag keys, so a config dict is not counted. Counted:
    a dict display with the literal owner, and a name bound to a dict that later gets
    ``name["owner"] = "<str>"``.
    """
    counts: dict[str, int] = {}
    others = REQUIRED_TAGS - {"owner"}
    for name, tree in _modules(root):
        n = 0
        for node in ast.walk(tree):
            dict_ = _literal(node)
            if dict_ is not None and dict_.literal_owner and dict_.keys and len(dict_.keys & others) >= 3:
                n += 1
        scan = _TagScan()
        scan.scan(tree)
        n += sum(
            1
            for held in scan.bound
            if held.owner_assigned and not held.literal_owner and held.keys and len(held.keys & others) >= 3
        )
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
    set_tags = "import mlflow\n\n\ndef h():\n    mlflow.set_tags({'stage': 's'})\n"
    root = _tree(tmp_path, {"a.py": direct, "b.py": named, "c.py": set_tags})

    problems = tag_problems(tag_dicts(root), {})

    assert [p.split(" ")[0] for p in problems] == ["a.py", "b.py", "c.py"]


def test_a_complete_tag_dict_passes(tmp_path):
    root = _tree(tmp_path, {"a.py": f"def f(m, owner):\n    tags = {TAGS_OK}\n    m.start_run(tags=tags)\n"})

    assert tag_problems(tag_dicts(root), {}) == []


def test_an_uncheckable_tag_dict_needs_a_listed_reason(tmp_path):
    root = _tree(tmp_path, {"a.py": "import mlflow\n\n\ndef f(tags):\n    mlflow.set_tags(tags)\n"})

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
        {
            "a.py": "tags = {'stage': 's', 'data_version': 'v', 'git_sha': 'g', 'owner': 'someone'}\n",
            "b.py": "def f(who):\n    return {'stage': 's', 'data_version': 'v', 'git_sha': 'g', 'owner': who}\n",
        },
    )

    problems = owner_problems(literal_owners(root), {})

    assert len(problems) == 1 and problems[0].startswith("a.py")


def test_an_owner_in_a_config_dict_that_is_not_a_tag_dict_is_not_caught(tmp_path):
    root = _tree(tmp_path, {"a.py": "CFG = {'owner': 'team-a', 'lr': 0.1}\n"})

    assert literal_owners(root) == {}


def test_a_bare_start_run_imported_from_mlflow_is_counted(tmp_path):
    root = _tree(tmp_path, {"a.py": "from mlflow import start_run\n\n\ndef f():\n    start_run()\n"})

    assert start_run_counts(root) == {"a.py": 1}


def test_a_module_level_tag_call_is_checked(tmp_path):
    root = _tree(tmp_path, {"a.py": "import mlflow\n\nmlflow.set_tags({'stage': 's'})\n"})

    problems = tag_problems(tag_dicts(root), {})

    assert len(problems) == 1 and problems[0].startswith("a.py")


def test_keys_added_to_a_tag_dict_before_the_call_count(tmp_path):
    adds = "".join(f"    tags['{k}'] = 'x'\n" for k in sorted(REQUIRED_TAGS - {"stage"}))
    root = _tree(
        tmp_path, {"a.py": f"def f(m):\n    tags = {{'stage': 's'}}\n{adds}    m.start_run(tags=tags)\n"}
    )

    assert tag_problems(tag_dicts(root), {}) == []


def test_keys_added_after_the_call_do_not_count(tmp_path):
    late = "def f(m):\n    tags = {'stage': 's'}\n    m.start_run(tags=tags)\n    tags['owner'] = 'x'\n"
    root = _tree(tmp_path, {"a.py": late})

    assert tag_problems(tag_dicts(root), {})


def test_a_closure_is_checked_once_against_the_dict_its_outer_function_built(tmp_path):
    source = (
        "def outer(m, owner):\n"
        f"    tags = {TAGS_OK}\n"
        "    def inner():\n"
        "        m.start_run(tags=tags)\n"
        "    inner()\n"
    )
    root = _tree(tmp_path, {"a.py": source})

    assert tag_dicts(root) == {"a.py": [REQUIRED_TAGS]}
    assert tag_problems(tag_dicts(root), {}) == []


def test_a_closure_defined_before_the_dict_is_bound_still_sees_it(tmp_path):
    source = (
        "def outer(m, owner):\n"
        "    def inner():\n"
        "        m.start_run(tags=tags)\n"
        f"    tags = {TAGS_OK}\n"
        "    inner()\n"
    )
    root = _tree(tmp_path, {"a.py": source})

    assert tag_dicts(root) == {"a.py": [REQUIRED_TAGS]}


def test_a_parameter_shadows_a_tag_dict_of_the_same_name_outside(tmp_path):
    source = f"import mlflow\ntags = {TAGS_OK}\n\n\ndef f(tags):\n    mlflow.set_tags(tags)\n"
    root = _tree(tmp_path, {"a.py": source})

    assert tag_dicts(root) == {"a.py": [None]}


@pytest.mark.parametrize("name", ["_git_sha", "git_sha", "_git_state", "read_git_state"])
def test_any_function_named_for_git_state_counts(tmp_path, name):
    root = _tree(tmp_path, {"a.py": f"def {name}():\n    pass\n"})

    assert git_helpers(root) == {("a.py", name)}


# ---------------------------------------------------------------------- the stale and gone branches


def test_a_tags_unchecked_entry_with_no_uncheckable_dict_is_caught():
    problems = tag_problems({"a.py": [set(REQUIRED_TAGS)]}, {"a.py": "was a parameter (KAN-1)"})

    assert len(problems) == 1 and "TAGS_UNCHECKED" in problems[0] and "delete its entry" in problems[0]


def test_an_allowlisted_git_function_that_no_longer_exists_is_caught():
    allowed = {("primer/old.py", "_git_sha"): Allowed(1, "r", "KAN-1")}

    problems = git_problems({SHARED_GIT}, allowed)

    assert len(problems) == 1 and "no longer exists" in problems[0] and "KAN-1" in problems[0]


def test_a_missing_shared_git_helper_is_caught():
    problems = git_problems(set(), {})

    assert len(problems) == 1 and "is gone" in problems[0] and "SHARED_GIT" in problems[0]


def test_an_owner_allowlist_entry_the_code_no_longer_needs_is_caught():
    problems = owner_problems({}, {"a.py": Allowed(1, "r", "KAN-1")})

    assert len(problems) == 1 and "OWNER_ALLOWED" in problems[0] and "KAN-1" in problems[0]


# --------------------------------------------------------------- tag writes, updates and owners


def test_set_tags_on_mlflow_and_on_self_dot_mlflow_are_tag_writes_but_another_object_is_not(tmp_path):
    root = _tree(
        tmp_path,
        {
            "a.py": "import mlflow\n\nmlflow.set_tags({'stage': 's'})\n",
            "b.py": "class G:\n    def f(self):\n        self._mlflow.set_tags({'stage': 's'})\n",
            "c.py": "def f(span):\n    span.set_tags({'k': 1})\n",
        },
    )

    assert tag_dicts(root) == {"a.py": [{"stage"}], "b.py": [{"stage"}]}


def test_a_partial_set_tags_after_a_full_start_run_is_still_held_to_the_seven(tmp_path):
    source = (
        "import mlflow\n\n\n"
        "def f(owner):\n"
        f"    mlflow.start_run(tags={TAGS_OK})\n"
        "    mlflow.set_tags({'gate_eligible': 'false'})\n"
    )
    root = _tree(tmp_path, {"a.py": source})

    problems = tag_problems(tag_dicts(root), {})

    assert len(problems) == 1 and problems[0].startswith("a.py") and "gate_eligible" not in problems[0]


@pytest.mark.parametrize(
    "change",
    ["tags.update({'x': 1})", "tags.setdefault('x', 1)", "tags |= {'x': 1}"],
)
def test_a_tag_dict_changed_by_update_setdefault_or_or_assign_is_unreadable(tmp_path, change):
    source = (
        f"import mlflow\n\n\ndef f():\n    tags = {{'stage': 's'}}\n    {change}\n    mlflow.set_tags(tags)\n"
    )
    root = _tree(tmp_path, {"a.py": source})

    assert tag_dicts(root) == {"a.py": [None]}
    assert "not a literal" in tag_problems(tag_dicts(root), {})[0]


def test_an_owner_assigned_by_subscript_to_a_tag_dict_is_caught(tmp_path):
    base = "def f(m, who):\n    tags = {'stage': 's', 'data_version': 'v', 'git_sha': 'g'}\n"
    literal = base + "    tags['owner'] = 'me'\n    m.start_run(tags=tags)\n"
    variable = base + "    tags['owner'] = who\n    m.start_run(tags=tags)\n"
    root = _tree(tmp_path, {"a.py": literal, "b.py": variable})

    assert literal_owners(root) == {"a.py": 1}


def test_a_dict_with_a_generic_owner_key_and_only_one_other_tag_key_is_not_a_tag_dict(tmp_path):
    root = _tree(tmp_path, {"a.py": "pipeline = {'stage': 'train', 'owner': 'ml-team'}\n"})

    assert literal_owners(root) == {}
