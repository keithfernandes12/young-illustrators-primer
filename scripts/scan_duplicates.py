#!/usr/bin/env python3
"""Advisory scan for duplicated helpers and conventions (cross-slice-coherence S5).

Usage:  python scripts/scan_duplicates.py [--root PATH]

`verify` runs this as an advisory check: it prints `WARN  duplicates` when there are new findings
and never changes `VERIFY_RESULT`. It is heuristic by design, so it must not block a slice; the
hard rules are `tests/test_conventions.py` (cross-slice-coherence S1).

It reads `src/` and `scripts/` with `ast` and reports:

- **same-name** private helpers (``_name``) defined in two or more modules;
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
  ``getattr(x, "start_run")``, ``s = mlflow.start_run``) or given ``tags`` positionally or in ``**``.

A finding whose file S1 already allowlists for the same kind is marked ``known``. Exit status is 1
when there is at least one new finding, so `verify` can print WARN, and 0 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("src", "scripts")
#: `docs/engineering/model-development.md` -> Tracking conventions.
REQUIRED_TAGS = frozenset({"stage", "owner", "data_version", "git_sha", "base_model", "gate_group", "where"})
TAG_SET_MIN = 4  # required-tag keys in one dict that make it a hand-written tag set
EXPERIMENT = re.compile(r"^yip-[a-z][a-z0-9-]*$")
MIN_BODY = 3  # statements, so `return x` is never a duplicate


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


def _str(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def blind_spots(name: str, tree: ast.Module) -> Iterator[Finding]:
    """The patterns `tests/test_conventions.py` cannot see, one finding each."""
    for node in ast.walk(tree):
        where = f"{name}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.List):
            head = [e.value for e in node.args[0].elts[:2] if isinstance(e, ast.Constant)]
            if head == ["git", "rev-parse"]:
                yield Finding(
                    "git-rev-parse",
                    where,
                    "reads the commit; the shared helper is drawing/evaluate._git_state",
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
                and _str(node.args[0])
                and node.args[0].value == "owner"
                and _str(node.args[1])
            ):  # type: ignore[union-attr]
                yield Finding("literal-owner", where, 'set_tag("owner", "<str>")')
            elif attr == "dict" and any(k.arg == "owner" and _str(k.value) for k in node.keywords):
                yield Finding("literal-owner", where, 'dict(owner="<str>")')
            elif attr == "create_run":
                yield Finding("create-run", where, "starts a run through MlflowClient, not the shared logger")
            elif (
                attr == "getattr"
                and len(node.args) >= 2
                and _str(node.args[1])
                and node.args[1].value == "start_run"
            ):  # type: ignore[union-attr]
                yield Finding("mlflow-alias", where, 'getattr(..., "start_run")')
            elif attr == "start_run" and (len(node.args) >= 6 or any(k.arg is None for k in node.keywords)):
                yield Finding(
                    "start-run-tags", where, "tags given positionally or in **, which S1 cannot read"
                )
        elif isinstance(node, ast.Dict):
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if len(keys & REQUIRED_TAGS) >= TAG_SET_MIN:
                yield Finding(
                    "tag-set", where, f"{len(keys & REQUIRED_TAGS)} of the 7 required tags written by hand"
                )
            if None in node.keys:
                for key, value in zip(node.keys, node.values, strict=True):
                    if _str(key) and key.value == "owner" and _str(value):  # type: ignore[union-attr]
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
    names: dict[str, set[str]] = defaultdict(set)
    bodies: dict[str, set[str]] = defaultdict(set)
    experiments: dict[str, set[str]] = defaultdict(set)
    styles: dict[str, set[str]] = defaultdict(set)
    findings: list[Finding] = []
    for name, tree in _modules(root):
        for func in _functions(tree):
            if func.name.startswith("_") and not func.name.startswith("__"):
                names[func.name].add(f"{name}:{func.lineno}")
            key = _body_key(func)
            if key is not None:
                bodies[key].add(f"{name}:{func.lineno} {func.name}")
        skip = _docstrings(tree)
        for node in ast.walk(tree):
            if _str(node) and id(node) not in skip and EXPERIMENT.match(node.value):  # type: ignore[union-attr]
                experiments[node.value].add(name)  # type: ignore[union-attr]
            elif isinstance(node, ast.JoinedStr):
                separator = _slice_separator(node)
                if separator is not None:
                    styles[separator].add(f"{name}:{node.lineno}")
        findings.extend(blind_spots(name, tree))
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
    if len(styles) > 1:
        for separator, places in sorted(styles.items()):
            for place in sorted(places):
                findings.append(
                    Finding(
                        "slice-style",
                        place,
                        f"separator {separator!r} between key and value ({len(styles)} styles in use)",
                    )
                )
    return findings


def _allowlisted(root: Path) -> set[tuple[str, str]]:
    """(kind, file) pairs S1 already lists, so the scan can say known rather than new."""
    path = root / "tests" / "test_conventions.py"
    if not path.exists():
        return set()
    spec = importlib.util.spec_from_file_location("_conventions", path)
    if spec is None or spec.loader is None:
        return set()
    module = importlib.util.module_from_spec(spec)
    sys.modules["_conventions"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 -- advisory: a broken test file must not stop the scan
        return set()
    known = {("git-rev-parse", f"src/{file}") for file, _ in getattr(module, "GIT_ALLOWED", {})}
    shared = getattr(module, "SHARED_GIT", ("", ""))[0]
    known.add(("git-rev-parse", f"src/{shared}"))
    # The shared logger is where the tag set and the slice name are meant to live.
    logger = getattr(module, "SHARED_LOGGER", "")
    known |= {("tag-set", f"src/{logger}"), ("slice-style", f"src/{logger}")}
    return known


def mark_known(findings: list[Finding], known: set[tuple[str, str]]) -> list[Finding]:
    return [
        Finding(f.kind, f.where, f.detail, known=(f.kind, f.where.split(":")[0]) in known) for f in findings
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    findings = mark_known(scan(args.root), _allowlisted(args.root))
    for finding in findings:
        print(finding.line())
    new = sum(1 for f in findings if not f.known)
    print(f"{len(findings)} findings ({new} new, {len(findings) - new} known)")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
