#!/usr/bin/env python3
"""Heuristic PII linter for YIP. Usage: privacy_lint.py [paths...]  -> PRIVACY_RESULT: PASS|FAIL"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PII = r"(child_?name|first_?name|last_?name|full_?name|e_?mail|phone|address|birth_?date|\bdob\b|transcript|audio_bytes|raw_audio|drawing_bytes|image_bytes|raw_drawing)"
RULES = [
    (
        "pii-in-log",
        re.compile(
            r"(log(ger|ging)?\.(debug|info|warning|warn|error|exception|critical)|print|console\.(log|info|warn|error)|span\.set_attribute|mlflow\.log_(param|text|dict))\s*\(.*"
            + PII,
            re.IGNORECASE,
        ),
    ),
    (
        "pii-in-url",
        re.compile(
            r"(\?|&)[a-z_]*" + PII + r"\s*=|f[\"'][^\"']*/\{[^}]*" + PII, re.IGNORECASE
        ),
    ),
    (
        "pii-in-object-name",
        re.compile(
            r"(blob|object_name|gcs_path|pubsub.*attributes?)\W.*" + PII, re.IGNORECASE
        ),
    ),
    (
        "hardcoded-secret",
        re.compile(
            r"(AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35}|-----BEGIN (RSA |EC )?PRIVATE KEY-----|sk-[A-Za-z0-9]{32,})"
        ),
    ),
]
AUDIO_WRITE = re.compile(
    r"(upload_from_(file|filename|string)|\.write_bytes|open\([^)]*['\"]wb['\"]).*audio",
    re.IGNORECASE,
)
OPT_IN = re.compile(r"(save_audio|audio_opt_in|keep_audio|store_audio)", re.IGNORECASE)
EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".sql"}
SKIP = {"node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".git"}


def files(paths: list[str]):
    for p in map(Path, paths or ["."]):
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in p.rglob("*"):
                if (
                    f.suffix in EXTS
                    and not SKIP.intersection(f.parts)
                    and "test" not in f.name
                ):
                    yield f


def main() -> int:
    findings = []
    for f in files(sys.argv[1:]):
        try:
            lines = f.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        text = "\n".join(lines)
        for i, line in enumerate(lines, 1):
            if "privacy: ok" in line:
                continue
            for name, rx in RULES:
                if rx.search(line):
                    findings.append(f"{f}:{i}: [{name}] {line.strip()[:160]}")
            if AUDIO_WRITE.search(line) and not OPT_IN.search(text):
                findings.append(f"{f}:{i}: [audio-without-opt-in] {line.strip()[:160]}")
    for x in findings:
        print(x)
    print(
        f"PRIVACY_RESULT: {'FAIL (' + str(len(findings)) + ' findings)' if findings else 'PASS'}"
    )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
