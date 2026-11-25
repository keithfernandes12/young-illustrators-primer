"""The advisory duplicate scan (scripts/scan_duplicates.py, cross-slice-coherence S5)."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scan_duplicates", ROOT / "scripts" / "scan_duplicates.py")
assert spec is not None and spec.loader is not None
scan_duplicates = importlib.util.module_from_spec(spec)
# Registered before it runs: a dataclass looks its module up in sys.modules while being defined.
sys.modules["scan_duplicates"] = scan_duplicates
spec.loader.exec_module(scan_duplicates)

HELPER = "def _load(path):\n    a = 1\n    b = 2\n    return a + b + len(path)\n"
TAGS = (
    '{"stage": "s", "owner": o, "data_version": "v", "git_sha": "g", '
    '"base_model": "b", "gate_group": "s", "where": "local"}'
)


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def _kinds(root: Path) -> list[str]:
    return [f.kind for f in scan_duplicates.scan(root)]


def test_two_identical_private_helpers_are_reported_and_one_is_not(tmp_path):
    one = _tree(tmp_path / "one", {"src/a.py": HELPER})
    two = _tree(tmp_path / "two", {"src/a.py": HELPER, "src/b.py": HELPER})

    assert _kinds(one) == []
    assert sorted(_kinds(two)) == ["same-body", "same-name"]


def test_a_short_body_is_never_a_duplicate(tmp_path):
    tiny = "def f(x):\n    return x\n"
    assert _kinds(_tree(tmp_path, {"src/a.py": tiny, "src/b.py": tiny})) == []


def test_a_hand_written_tag_set_is_reported(tmp_path):
    root = _tree(tmp_path, {"src/a.py": f"def f(o):\n    return {TAGS}\n"})
    assert _kinds(root) == ["tag-set"]


def test_one_slice_naming_style_is_fine_and_two_are_reported(tmp_path):
    slash = "def f(k, v, m):\n    return f'slice/{k}/{v}/{m}'\n"
    colon = "def g(k, v, m):\n    return f'slice/{k}:{v}/{m}'\n"

    assert _kinds(_tree(tmp_path / "one", {"src/a.py": slash})) == []
    found = scan_duplicates.scan(_tree(tmp_path / "two", {"src/a.py": slash, "src/b.py": colon}))
    assert sorted(f.detail.split("'")[1] for f in found if f.kind == "slice-style") == ["/", ":"]


def test_an_experiment_name_in_two_modules_is_reported(tmp_path):
    root = _tree(tmp_path, {"src/a.py": "E = 'yip-drawing'\n", "src/b.py": "E = 'yip-drawing'\n"})
    assert _kinds(root) == ["experiment"]


@pytest.mark.parametrize(
    "code, kind",
    [
        ("import subprocess\nsubprocess.run(['git', 'rev-parse', 'HEAD'])\n", "git-rev-parse"),
        ("m.set_tag('owner', 'someone')\n", "literal-owner"),
        ("t = dict(owner='someone')\n", "literal-owner"),
        ("t = {**base, 'owner': 'someone'}\n", "literal-owner"),
        ("MlflowClient().create_run('1')\n", "create-run"),
        ("from mlflow import start_run as sr\n", "mlflow-alias"),
        ("s = getattr(mlflow, 'start_run')\n", "mlflow-alias"),
        ("s = mlflow.start_run\n", "mlflow-alias"),
        ("mlflow.start_run(None, None, None, False, None, {'a': 1})\n", "start-run-tags"),
        ("mlflow.start_run(**opts)\n", "start-run-tags"),
    ],
)
def test_each_s1_blind_spot_is_reported(tmp_path, code, kind):
    assert _kinds(_tree(tmp_path, {"src/a.py": code})) == [kind]


def test_scripts_are_scanned_too(tmp_path):
    root = _tree(tmp_path, {"scripts/tool.py": "subprocess.run(['git', 'rev-parse', 'HEAD'])\n"})
    assert _kinds(root) == ["git-rev-parse"]


def test_a_finding_s1_already_lists_is_marked_known(tmp_path):
    found = scan_duplicates.scan(
        _tree(tmp_path, {"src/primer/a.py": "subprocess.run(['git', 'rev-parse'])\n"})
    )

    marked = scan_duplicates.mark_known(found, {("git-rev-parse", "src/primer/a.py")})

    assert [f.known for f in marked] == [True]


def test_the_cli_exits_1_only_for_new_findings(tmp_path, capsys):
    clean = _tree(tmp_path / "clean", {"src/a.py": "x = 1\n"})
    dirty = _tree(tmp_path / "dirty", {"src/a.py": HELPER, "src/b.py": HELPER})

    assert scan_duplicates.main(["--root", str(clean)]) == 0
    assert scan_duplicates.main(["--root", str(dirty)]) == 1
    assert "2 findings (2 new, 0 known)" in capsys.readouterr().out


def test_it_runs_on_the_real_tree_and_separates_known_from_new():
    # Not a count assertion: main changes weekly. S1's allowlisted git copies must read as known.
    found = scan_duplicates.mark_known(scan_duplicates.scan(), scan_duplicates._allowlisted(ROOT))
    known = {f.where.split(":")[0] for f in found if f.known and f.kind == "git-rev-parse"}
    assert "src/primer/evaluation/golden.py" in known
