"""The advisory duplicate scan (scripts/scan_duplicates.py, cross-slice-coherence S5)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scan_duplicates(load_script):
    return load_script("scripts/scan_duplicates.py", "scan_duplicates")


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


def _kinds(module, root: Path) -> list[str]:
    return [f.kind for f in module.scan(root)]


def test_two_identical_private_helpers_are_reported_and_one_is_not(scan_duplicates, tmp_path):
    one = _tree(tmp_path / "one", {"src/a.py": HELPER})
    two = _tree(tmp_path / "two", {"src/a.py": HELPER, "src/b.py": HELPER})

    assert _kinds(scan_duplicates, one) == []
    assert sorted(_kinds(scan_duplicates, two)) == ["same-body", "same-name"]


def test_a_short_body_is_never_a_duplicate(scan_duplicates, tmp_path):
    tiny = "def f(x):\n    return x\n"
    assert _kinds(scan_duplicates, _tree(tmp_path, {"src/a.py": tiny, "src/b.py": tiny})) == []


def test_a_hand_written_tag_set_is_reported(scan_duplicates, tmp_path):
    root = _tree(tmp_path, {"src/a.py": f"def f(o):\n    return {TAGS}\n"})
    assert _kinds(scan_duplicates, root) == ["tag-set"]


def test_only_a_slice_style_other_than_the_documented_one_is_reported(scan_duplicates, tmp_path):
    slash = "def f(k, v, m):\n    return f'slice/{k}/{v}/{m}'\n"
    colon = "def g(k, v, m):\n    return f'slice/{k}:{v}/{m}'\n"

    assert _kinds(scan_duplicates, _tree(tmp_path / "one", {"src/a.py": slash})) == []
    found = scan_duplicates.scan(_tree(tmp_path / "two", {"src/a.py": slash, "src/b.py": colon}))
    assert [(f.kind, f.where.split(":")[0]) for f in found] == [("slice-style", "src/b.py")]
    assert _kinds(scan_duplicates, _tree(tmp_path / "three", {"src/b.py": colon})) == ["slice-style"]


def test_an_experiment_name_in_two_modules_is_reported(scan_duplicates, tmp_path):
    root = _tree(tmp_path, {"src/a.py": "E = 'yip-drawing'\n", "src/b.py": "E = 'yip-drawing'\n"})
    assert _kinds(scan_duplicates, root) == ["experiment"]


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
        ("mlflow.start_run(run_name='x')\n", "start-run"),
    ],
)
def test_each_s1_blind_spot_is_reported(scan_duplicates, tmp_path, code, kind):
    assert _kinds(scan_duplicates, _tree(tmp_path, {"src/a.py": code})) == [kind]


def test_scripts_are_scanned_too(scan_duplicates, tmp_path):
    root = _tree(tmp_path, {"scripts/tool.py": "subprocess.run(['git', 'rev-parse', 'HEAD'])\n"})
    assert _kinds(scan_duplicates, root) == ["git-rev-parse"]


def test_a_finding_s1_already_lists_is_marked_known(scan_duplicates, tmp_path):
    found = scan_duplicates.scan(
        _tree(tmp_path, {"src/primer/a.py": "subprocess.run(['git', 'rev-parse'])\n"})
    )

    marked = scan_duplicates.mark_known(found, {("git-rev-parse", "src/primer/a.py")})

    assert [f.known for f in marked] == [True]


def test_the_cli_exits_1_only_for_new_findings(scan_duplicates, tmp_path, capsys):
    clean = _tree(tmp_path / "clean", {"src/a.py": "x = 1\n"})
    dirty = _tree(tmp_path / "dirty", {"src/a.py": HELPER, "src/b.py": HELPER})

    assert scan_duplicates.main(["--root", str(clean)]) == 0
    assert scan_duplicates.main(["--root", str(dirty)]) == 1
    assert "2 findings (2 new, 0 known)" in capsys.readouterr().out


def test_methods_with_the_same_name_in_different_classes_are_not_a_duplicate(scan_duplicates, tmp_path):
    one = "class A:\n    def _load(self):\n        a = 1\n        return a\n"
    two = "class B:\n    def _load(self):\n        b = 2\n        return b\n"

    assert _kinds(scan_duplicates, _tree(tmp_path, {"src/a.py": one, "src/b.py": two})) == []


CONVENTIONS = """
REQUIRED_TAGS = frozenset({"w", "x", "y", "z"})
GIT_ALLOWED = {("primer/a.py", "git_sha"): 1}
SHARED_GIT = ("primer/b.py", "_git_state")
START_RUN_ALLOWED = {"primer/c.py": 1}
SHARED_LOGGER = "primer/d.py"
"""


def test_the_required_tags_come_from_the_conventions_file_when_there_is_one(scan_duplicates, tmp_path):
    root = _tree(
        tmp_path,
        {
            "tests/test_conventions.py": CONVENTIONS,
            "src/a.py": "t = {'w': 1, 'x': 2, 'y': 3, 'z': 4}\n",
        },
    )

    assert _kinds(scan_duplicates, root) == ["tag-set"]


def test_s1s_allowlists_mark_start_run_and_git_sites_known(scan_duplicates, tmp_path):
    root = _tree(
        tmp_path,
        {
            "tests/test_conventions.py": CONVENTIONS,
            "src/primer/c.py": "mlflow.start_run()\n",
            "src/primer/d.py": "mlflow.start_run()\n",
            "src/primer/e.py": "mlflow.start_run()\n",
            "src/primer/a.py": "subprocess.run(['git', 'rev-parse', 'HEAD'])\n",
            "src/primer/data/versioning.py": "subprocess.run(['git', 'rev-parse', 'HEAD'])\n",
        },
    )

    found = scan_duplicates.mark_known(scan_duplicates.scan(root), scan_duplicates._allowlisted(root))

    new = [f.where.split(":")[0] for f in found if not f.known]
    assert new == ["src/primer/e.py"], "only the site no allowlist lists is new"


def test_a_conventions_file_that_will_not_import_is_named_on_stderr(scan_duplicates, tmp_path, capsys):
    root = _tree(tmp_path, {"tests/test_conventions.py": "raise RuntimeError('x')\n", "src/a.py": "x = 1\n"})

    assert scan_duplicates.main(["--root", str(root)]) == 0

    assert "allowlist is unavailable" in capsys.readouterr().err


def test_a_scan_that_crashes_says_so_and_exits_2(scan_duplicates, tmp_path, capsys):
    root = _tree(tmp_path, {"src/a.py": "def broken(:\n"})

    assert scan_duplicates.main(["--root", str(root)]) == 2

    assert capsys.readouterr().out.startswith("ERROR")


def test_it_runs_on_the_real_tree(scan_duplicates):
    # Not a count or a name: main changes weekly and S1's allowlists are meant to shrink.
    assert isinstance(scan_duplicates.scan(), list)
