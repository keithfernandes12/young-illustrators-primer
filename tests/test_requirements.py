"""requirements.txt stays in step with pyproject.toml, and the drawing extras pin transformers."""

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


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_requirements_pin_every_declared_dependency_within_its_range():
    project = _project()
    extras = project["optional-dependencies"]
    pins = _pins()
    for declared in project["dependencies"] + extras["dev"] + extras["dvc"]:
        req = Requirement(declared)
        name = canonicalize_name(req.name)
        assert name in pins, f"{req.name} is in pyproject.toml but not pinned in requirements.txt: regenerate it"
        for version in pins[name]:
            assert req.specifier.contains(version, prereleases=True), \
                f"requirements.txt pins {req.name}=={version}, outside pyproject.toml's {req.specifier}"


MEASURED_TRANSFORMERS = "5.23.0"


def test_the_drawing_extras_pin_transformers_to_the_version_the_numbers_were_measured_on():
    """Every recorded drawing number was measured on one transformers release (mvp-cloud-deploy).

    The drawing extras are not in requirements.txt (CI never installs torch), so the pin lives in
    pyproject.toml itself. A range would let a minor release move top-5 0.5436 without anyone
    changing a line here. Moving the pin means re-measuring, so it also means editing this constant.
    """
    extras = _project()["optional-dependencies"]
    for extra in ("drawing", "sheet"):
        [req] = [r for r in map(Requirement, extras[extra]) if canonicalize_name(r.name) == "transformers"]
        msg = f"the {extra} extra must pin transformers=={MEASURED_TRANSFORMERS} exactly, not {req.specifier}"
        assert len(req.specifier) == 1, msg
        [spec] = req.specifier
        assert spec.operator == "==" and "*" not in spec.version, msg
        assert spec.version == MEASURED_TRANSFORMERS, msg
