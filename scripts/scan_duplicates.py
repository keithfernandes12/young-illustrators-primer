#!/usr/bin/env python3
"""Advisory scan for duplicated helpers and conventions (cross-slice-coherence S5).

Usage:  python scripts/scan_duplicates.py [--root PATH]

`verify` runs this as an advisory check: it prints `WARN  duplicates` when there are new findings
and never changes `VERIFY_RESULT`. It is heuristic by design, so it must not block a slice; the
hard rules are `tests/test_conventions.py` (cross-slice-coherence S1).

It reads `src/` and `scripts/` with `ast` and reports:

- **same-name** private module-level functions (``_name``) defined in two or more modules (methods
  are left out: ``_load`` on seven different classes is polymorphism, not a copy);
- **same-body** functions: two or more modules with an identical body (docstring removed, at least
  three statements), whatever they are called;
- **conventions written out by hand**: a dict holding four or more of the seven required MLflow
  tags (``tag-set``), ``slice/…`` metric names built with different separators (``slice-style``),
  and one ``yip-…`` experiment name in two modules (``experiment``). A generic "same literal in
  more than three modules" rule was tried first and dropped: on ``main`` it reported 82 literals,
  nearly all data-schema column names (``license``, ``record_id``) that modules share on purpose;
- **S1's blind spots**, the patterns its scans cannot see: ``git rev-parse`` run by any function,
  ``set_tag("owner", "<str>")``, ``dict(owner="<str>")``, ``{**x, "owner": "<str>"}``,
  ``.create_run(``, and a ``start_run`` reached by an alias (``from mlflow import start_run``,
  ``getattr(x, "start_run")``, ``s = mlflow.start_run``) or given ``tags`` positionally or in ``**``;
- **every ``start_run`` call**, one ``start-run`` finding each, so the run loggers outside the shared
  one stay visible.

A finding whose file S1 already allowlists for the same kind is marked ``known``. That match is per
file and kind, so a second copy added to an allowlisted file also reads ``known``; S1's own counts
still fail on it. Exit status is 1 when there is at least one new finding, so `verify` can print
WARN, 0 otherwise, and 2 when the scan itself crashed (the line starts with ``ERROR``).
"""

from __future__ import annotations

import argparse
import ast
import functools
import importlib.util
import re
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("src", "scripts")
#: `docs/engineering/model-development.md` -> Tracking conventions. Only a fallback: the scan reads
#: the set from `tests/test_conventions.py` when it can, so the two cannot drift.
REQUIRED_TAGS = frozenset({"stage", "owner", "data_version", "git_sha", "base_model", "gate_group", "where"})
TAG_SET_MIN = 4  # required-tag keys in one dict that make it a hand-written tag set
EXPERIMENT = re.compile(r"^yip-[a-z][a-z0-9-]*$")
MIN_BODY = 3  # statements, so `return x` is never a duplicate
#: The form `docs/engineering/model-development.md` fixes: slice/<key>/<value>/<metric>.
SLICE_SEPARATOR = "/"
#: Sites S1 deliberately does not scan, so the scan lists them as known: (kind, file).
#: `data/versioning.py` records a dataset version's full commit, not a run's provenance
#: (cross-slice-coherence Verified facts, 2026-11-23).
OWN_KNOWN = {("git-rev-parse", "src/primer/data/versioning.py")}


@dataclass(frozen=True)
class Finding:
    kind: str
    where: str
    detail: str
    known: bool = False

    def line(self) -> str:
        return f"{self.kind:<14} {self.where:<55} {self.detail}{'  [known]' if self.known else ''}"


def _modules(root: Path) -> Iterator[tuple[str, ast.Module]]:
    for folder in FOLDERS:
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if path.resolve() == Path(__file__).resolve():
                continue  # its own patterns and messages would match themselves
            yield path.relative_to(root).as_posix(), ast.parse(path.read_text(encoding="utf-8"), str(path))


def _functions(tree: ast.Module) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node


def _body_key(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # the docstring
    if len(body) < MIN_BODY:
        return None
    return "\n".join(ast.dump(stmt) for stmt in body)


def _docstrings(tree: ast.Module) -> set[int]:
    ids = set()
    for node in [tree, *ast.walk(tree)]:
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def _text(node: ast.AST | None) -> str | None:
    """The value of a string literal, or None for anything else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def blind_spots(name: str, tree: ast.Module, required: frozenset[str] = REQUIRED_TAGS) -> Iterator[Finding]:
    """The patterns `tests/test_conventions.py` cannot see, one finding each."""
    for node in ast.walk(tree):
        where = f"{name}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.List):
            head = [e.value for e in node.args[0].elts[:2] if isinstance(e, ast.Constant)]
            if head == ["git", "rev-parse"]:
                yield Finding(
                    "git-rev-parse",
                    where,
                    "reads the commit by hand (SHARED_GIT in tests/test_conventions.py is the shared helper)",
                )
        if isinstance(node, ast.ImportFrom) and node.module == "mlflow":
            for alias in node.names:
                if alias.name in ("start_run", "set_tags", "set_tag"):
                    yield Finding(
                        "mlflow-alias", where, f"imports {alias.name} directly, which S1 cannot follow"
                    )
        elif isinstance(node, ast.Call):
            func = node.func
            attr = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else ""
            )
            if (
                attr == "set_tag"
                and len(node.args) >= 2
                and _text(node.args[0]) == "owner"
                and _text(node.args[1]) is not None
            ):
                yield Finding("literal-owner", where, 'set_tag("owner", "<str>")')
            elif attr == "dict" and any(k.arg == "owner" and _text(k.value) is not None for k in node.keywords):
                yield Finding("literal-owner", where, 'dict(owner="<str>")')
            elif attr == "create_run":
                yield Finding("create-run", where, "starts a run through MlflowClient, not the shared logger")
            elif attr == "getattr" and len(node.args) >= 2 and _text(node.args[1]) == "start_run":
                yield Finding("mlflow-alias", where, 'getattr(..., "start_run")')
            elif attr == "start_run":
                if len(node.args) >= 6 or any(k.arg is None for k in node.keywords):
                    yield Finding(
                        "start-run-tags", where, "tags given positionally or in **, which S1 cannot read"
                    )
                else:
                    yield Finding("start-run", where, "starts an MLflow run; the shared logger is S1's")
        elif isinstance(node, ast.Dict):
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if len(keys & required) >= TAG_SET_MIN:
                yield Finding(
                    "tag-set", where, f"{len(keys & required)} of the {len(required)} required tags by hand"
                )
            if None in node.keys:
                for key, value in zip(node.keys, node.values, strict=True):
                    if _text(key) == "owner" and _text(value) is not None:
                        yield Finding("literal-owner", where, '{**x, "owner": "<str>"}')
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "start_run"
        ):
            yield Finding("mlflow-alias", where, "start_run assigned to a name, which S1 cannot follow")


def _slice_separator(node: ast.JoinedStr) -> str | None:
    """The text between the first two placeholders of an f-string that starts ``slice/``."""
    parts = node.values
    if not (parts and isinstance(parts[0], ast.Constant) and str(parts[0].value).startswith("slice/")):
        return None
    for i, part in enumerate(parts[1:], start=1):
        if isinstance(part, ast.FormattedValue):
            following = parts[i + 1] if i + 1 < len(parts) else None
            if isinstance(following, ast.Constant):
                return str(following.value)[:1]
            return None
    return None


def scan(root: Path = ROOT) -> list[Finding]:
    conventions = _conventions(root)
    required = frozenset(getattr(conventions, "REQUIRED_TAGS", REQUIRED_TAGS))
    names: dict[str, set[str]] = defaultdict(set)
    bodies: dict[str, set[str]] = defaultdict(set)
    experiments: dict[str, set[str]] = defaultdict(set)
    styles: dict[str, set[str]] = defaultdict(set)
    findings: list[Finding] = []
    for name, tree in _modules(root):
        for top in tree.body:  # module level only: a method is not a copied helper
            if (
                isinstance(top, ast.FunctionDef | ast.AsyncFunctionDef)
                and top.name.startswith("_")
                and not top.name.startswith("__")
            ):
                names[top.name].add(f"{name}:{top.lineno}")
        for func in _functions(tree):
            key = _body_key(func)
            if key is not None:
                bodies[key].add(f"{name}:{func.lineno} {func.name}")
        skip = _docstrings(tree)
        for node in ast.walk(tree):
            text = _text(node)
            if text is not None and id(node) not in skip and EXPERIMENT.match(text):
                experiments[text].add(name)
            elif isinstance(node, ast.JoinedStr):
                separator = _slice_separator(node)
                if separator is not None:
                    styles[separator].add(f"{name}:{node.lineno}")
        findings.extend(blind_spots(name, tree, required))
    for func_name, places in sorted(names.items()):
        if len({p.split(":")[0] for p in places}) >= 2:
            findings.append(
                Finding("same-name", ", ".join(sorted(places)), f"{func_name}() in {len(places)} modules")
            )
    for places in sorted(bodies.values(), key=sorted):
        if len({p.split(":")[0] for p in places}) >= 2:
            findings.append(Finding("same-body", ", ".join(sorted(places)), "identical bodies"))
    for experiment, modules in sorted(experiments.items()):
        if len(modules) >= 2:
            findings.append(
                Finding(
                    "experiment",
                    ", ".join(sorted(modules)),
                    f"{experiment!r} named in {len(modules)} modules",
                )
            )
    for separator, places in sorted(styles.items()):
        if separator == SLICE_SEPARATOR:
            continue  # the documented form is not a finding
        for place in sorted(places):
            findings.append(
                Finding(
                    "slice-style",
                    place,
                    f"separator {separator!r} between key and value; the documented form is "
                    "slice/<key>/<value>/<metric>",
                )
            )
    return findings


@functools.cache
def _conventions(root: Path) -> ModuleType | None:
    """`tests/test_conventions.py` loaded once, or None when it is missing or will not import."""
    path = root / "tests" / "test_conventions.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_conventions", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_conventions"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 -- advisory: a broken test file must not stop the scan
        return None
    return module


def _allowlisted(root: Path) -> set[tuple[str, str]]:
    """(kind, file) pairs S1 already lists, so the scan can say known rather than new."""
    module = _conventions(root)
    if module is None:
        return set()
    known = set(OWN_KNOWN)
    known |= {("git-rev-parse", f"src/{file}") for file, _ in getattr(module, "GIT_ALLOWED", {})}
    shared = getattr(module, "SHARED_GIT", ("", ""))[0]
    known.add(("git-rev-parse", f"src/{shared}"))
    # The shared logger is where the tag set, the slice name and the run start are meant to live.
    logger = getattr(module, "SHARED_LOGGER", "")
    known |= {(kind, f"src/{logger}") for kind in ("tag-set", "slice-style", "start-run")}
    known |= {("start-run", f"src/{file}") for file in getattr(module, "START_RUN_ALLOWED", {})}
    return known


def mark_known(findings: list[Finding], known: set[tuple[str, str]]) -> list[Finding]:
    return [
        Finding(f.kind, f.where, f.detail, known=(f.kind, f.where.split(":")[0]) in known) for f in findings
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    conventions = args.root / "tests" / "test_conventions.py"
    try:
        findings = mark_known(scan(args.root), _allowlisted(args.root))
    except Exception as exc:  # noqa: BLE001 -- name the crash; verify shows this line under WARN
        print(f"ERROR  the scan crashed: {exc!r}")
        return 2
    if conventions.exists() and _conventions(args.root) is None:
        print(f"warning: {conventions} did not import, so the allowlist is unavailable", file=sys.stderr)
    for finding in findings:
        print(finding.line())
    new = sum(1 for f in findings if not f.known)
    print(f"{len(findings)} findings ({new} new, {len(findings) - new} known)")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
