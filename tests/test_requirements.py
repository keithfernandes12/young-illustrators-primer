"""requirements.txt stays in step with pyproject.toml, the source of truth for dependencies."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def _pins() -> dict[str, list[str]]:
    """Package name -> pinned versions (more than one when pinned per platform)."""
    pins: dict[str, list[str]] = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        req = Requirement(line)
        [spec] = req.specifier                       # exactly one specifier...
        assert spec.operator == "==", line           # ...and it is an exact pin
        pins.setdefault(canonicalize_name(req.name), []).append(spec.version)
    return pins


def test_requirements_pin_every_declared_dependency_within_its_range():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    pins = _pins()
    for declared in project["dependencies"] + extras["dev"] + extras["dvc"]:
        req = Requirement(declared)
        name = canonicalize_name(req.name)
        assert name in pins, f"{req.name} is in pyproject.toml but not pinned in requirements.txt: regenerate it"
        for version in pins[name]:
            assert req.specifier.contains(version, prereleases=True), \
                f"requirements.txt pins {req.name}=={version}, outside pyproject.toml's {req.specifier}"
