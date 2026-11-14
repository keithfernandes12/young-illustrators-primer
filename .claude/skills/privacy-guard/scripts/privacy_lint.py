#!/usr/bin/env python3
"""Heuristic PII linter for YIP. Usage: privacy_lint.py [paths...]  -> PRIVACY_RESULT: PASS|FAIL

For Python files the log rule is checked against the parsed syntax tree, not the raw line, so that
a privacy word inside a *message* is not mistaken for private data being logged. That distinction
matters: `log.warning("Email alerts are off (set PRIMER_ALERT_EMAILS)")` names no child and logs
nothing private, while `log.info("page for %s", child_name)` logs a name.

Rules that are genuinely about literal text -- a PII query parameter, a PII-keyed object name, a
hardcoded credential -- stay as regexes, because there the literal *is* the finding.

Add `# privacy: ok` to a line to accept it deliberately.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

PII = r"(child_?name|first_?name|last_?name|full_?name|e_?mail|phone|address|birth_?date|\bdob\b|transcript|audio_bytes|raw_audio|drawing_bytes|image_bytes|raw_drawing)"
PII_RE = re.compile(PII, re.IGNORECASE)

# Call targets that send data somewhere it will be kept: logs, traces, experiment tracking.
LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical"}
# Sinks whose first string argument names the field rather than describing the message, so the
# literal counts: span.set_attribute("child_name", v) records a field called child_name.
KEYED_SINKS = {"set_attribute", "log_param", "log_text", "log_dict"}
SINK_METHODS = LOG_METHODS | KEYED_SINKS
SINK_FUNCTIONS = {"print"}

# Rules whose subject is the literal text of the line, where a regex is the right tool.
TEXT_RULES = [
    (
        "pii-in-url",
        re.compile(r"(\?|&)[a-z_]*" + PII + r"\s*=|f[\"'][^\"']*/\{[^}]*" + PII, re.IGNORECASE),
    ),
    (
        "pii-in-object-name",
        re.compile(r"(blob|object_name|gcs_path|pubsub.*attributes?)\W.*" + PII, re.IGNORECASE),
    ),
    (
        "hardcoded-secret",
        re.compile(
            r"(AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35}|-----BEGIN (RSA |EC )?PRIVATE KEY-----|sk-[A-Za-z0-9]{32,})"
        ),
    ),
]

# The log rule for languages we do not parse. Same shape as the original line-based rule.
NON_PYTHON_LOG_RULE = re.compile(
    r"(log(ger|ging)?\.(debug|info|warning|warn|error|exception|critical)|print|console\.(log|info|warn|error)"
    r"|span\.set_attribute|mlflow\.log_(param|text|dict))\s*\(.*" + PII,
    re.IGNORECASE,
)

AUDIO_WRITE = re.compile(
    r"(upload_from_(file|filename|string)|\.write_bytes|open\([^)]*['\"]wb['\"]).*audio",
    re.IGNORECASE,
)
OPT_IN = re.compile(r"(save_audio|audio_opt_in|keep_audio|store_audio)", re.IGNORECASE)

EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".sql"}
SKIP_DIRS = {"node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".git", ".local-data"}


def is_test_path(path: Path) -> bool:
    """Tests legitimately contain fake PII. Match path parts, so `latest.py` is still linted."""
    return any(part == "tests" for part in path.parts) or path.name.startswith("test_")


def files(paths: list[str]):
    for raw in paths or ["."]:
        p = Path(raw)
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.suffix in EXTS and not SKIP_DIRS.intersection(f.parts) and not is_test_path(f):
                    yield f


def _sink_name(call: ast.Call) -> str | None:
    """The sink this call targets, or None when the call is not a sink we care about."""
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in SINK_METHODS:
        return func.attr
    if isinstance(func, ast.Name) and func.id in SINK_FUNCTIONS:
        return func.id
    return None


def _value_expressions(call: ast.Call, keyed: bool):
    """The call's arguments, with message templates removed.

    For a log call, a bare string argument is the message template: privacy words in it describe
    the message and are not data. Everything else -- names, attributes, f-strings, %-formats,
    .format() calls, dict and list literals -- can carry real values and is inspected.

    For a keyed sink the first string argument is the field name, so nothing is skipped.
    """
    args = [*call.args, *(kw.value for kw in call.keywords)]
    for arg in args:
        if not keyed and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            continue
        yield arg


def _pii_identifiers(node: ast.AST) -> list[str]:
    """Identifiers under `node` that look like private data. String literals are ignored."""
    hits: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and PII_RE.search(child.id):
            hits.append(child.id)
        elif isinstance(child, ast.Attribute) and PII_RE.search(child.attr):
            hits.append(child.attr)
        # A dict key or attribute name such as {"child_name": x} names the field being sent.
        elif (
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and PII_RE.fullmatch(child.value.replace(" ", "_"))
        ):
            hits.append(child.value)
    return hits


def opt_in_scopes(source: str) -> list[tuple[int, int]]:
    """Line ranges of functions that mention a parent opt-in.

    An opt-in identifier anywhere in a file used to suppress every audio-write finding in it, even
    when it was assigned False or belonged to an unrelated function. Consent on one code path is
    not consent on another. Narrowing it to the enclosing function is still a heuristic -- the real
    check is control flow -- but it no longer lets one mention cover a whole module.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    scopes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            segment = ast.get_source_segment(source, node) or ""
            if OPT_IN.search(segment):
                scopes.append((node.lineno, node.end_lineno or node.lineno))
    return scopes


def python_log_findings(path: Path, source: str, lines: list[str]) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        sink = _sink_name(node)
        if sink is None:
            continue
        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        if "privacy: ok" in line:
            continue
        expressions = _value_expressions(node, keyed=sink in KEYED_SINKS)
        hits = sorted({h for expression in expressions for h in _pii_identifiers(expression)})
        if hits:
            findings.append(
                f"{path}:{node.lineno}: [pii-in-log] {sink}(...) sends {', '.join(hits)}"
            )
    return findings


def main() -> int:
    findings: list[str] = []
    for f in files(sys.argv[1:]):
        try:
            source = f.read_text(errors="ignore")
        except OSError:
            continue
        lines = source.splitlines()

        if f.suffix == ".py":
            findings += python_log_findings(f, source, lines)
            consented = opt_in_scopes(source)
        else:
            # Nothing to parse, so the old file-wide rule is all that is available here.
            consented = [(1, len(lines))] if OPT_IN.search(source) else []

        for i, line in enumerate(lines, 1):
            if "privacy: ok" in line:
                continue
            if f.suffix != ".py" and NON_PYTHON_LOG_RULE.search(line):
                findings.append(f"{f}:{i}: [pii-in-log] {line.strip()[:160]}")
            for name, rx in TEXT_RULES:
                if rx.search(line):
                    findings.append(f"{f}:{i}: [{name}] {line.strip()[:160]}")
            if AUDIO_WRITE.search(line) and not any(start <= i <= end for start, end in consented):
                findings.append(f"{f}:{i}: [audio-without-opt-in] {line.strip()[:160]}")

    for x in findings:
        print(x)
    print(f"PRIVACY_RESULT: {'FAIL (' + str(len(findings)) + ' findings)' if findings else 'PASS'}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
