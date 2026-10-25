"""The Bash guard refuses what AGENTS.md says agents must not do, and allows ordinary work."""

import io
import json

import pytest

GOOD_PR = ("gh pr create --draft --label agent-authored --label needs-reviewer "
           "--title \"feat(data): add source (KAN-42)\" --body-file body.md")

REFUSED = [
    # Merging, in any spelling
    "gh pr merge 3",
    "gh pr merge 3 --squash --delete-branch",
    "gh -R keithfernandes12/young-illustrators-primer pr merge 3",
    "git status && gh pr merge 3",
    "x=$(gh pr merge 3)",
    "bash -c 'gh pr merge 3'",
    # Other repositories
    "gh pr list -R someone/else",
    "gh pr list --repo=https://github.com/someone/else",
    "gh api repos/someone/else/pulls",
    "gh repo clone someone/else",
    "gh repo delete keithfernandes12/young-illustrators-primer --yes",
    "gh issue create --title x --body y",
    # PR creation rules
    # A label written inside the title is not a label
    "gh pr create --draft --label needs-reviewer --title \"--label agent-authored\" --body-file body.md",
    "gh pr create --draft --label agent-authored --title \"--label needs-reviewer\" --body-file body.md",
    "gh pr create --body \"x\" --draft",
    "gh pr create --label agent-authored --label needs-reviewer --body-file body.md",
    "gh pr create --draft --label agent-authored --label needs-reviewer --body-file no-key.md",
    "gh pr create --draft --label needs-reviewer --body-file body.md",
    "gh pr create --draft --label agent-authored --body-file body.md",
    "gh pr create --draft --label agent-authored --label needs-reviewer --body-file missing.md",
    # Pushes
    "git push origin main",
    "git push origin HEAD:main",
    "git push origin feat/x:refs/heads/main",
    "git push -f origin feat/x",
    "git push -uf origin feat/x",
    "git push --force-with-lease origin feat/x",
    "git push origin +feat/x",
    "git push origin :feat/x",
    "git push --delete origin feat/x",
    "git push --tags",
    "git push origin data-v0.2",
    "git -C . push origin main",
    # Bucket writes through the pipeline
    "primer-data run tinystories",
    "primer-data ingest quickdraw --batch-size 2",
    "primer-data --data-uri gs://young-illustrators-primer-data run quickdraw",
    "primer-data --data-uri=gs://young-illustrators-primer-data preprocess clear_corpus",
    "PRIMER_DATA_URI=gs://young-illustrators-primer-data primer-data run tinystories",
    "primer-data version v0.2 -m \"new drawings\"",
    "cd src && primer-data run tinystories",
    "python -m primer.data.cli run tinystories",
    # Bucket writes through Google tools
    "gsutil cp file.txt gs://young-illustrators-primer-data/raw/x",
    "gsutil -m rm -r gs://young-illustrators-primer-data/raw",
    "gsutil mv gs://young-illustrators-primer-data/a ./a",
    "gcloud storage rm -r gs://young-illustrators-primer-data/raw",
    "gcloud storage cp ./x gs://young-illustrators-primer-data/x",
    "gcloud storage buckets update gs://young-illustrators-primer-data --no-versioning",
    "gcloud auth login",
    "gcloud auth application-default login",
    "gcloud run services delete api",
    "dvc push",
    "dvc gc -w",
    "dvc import-url gs://young-illustrators-primer-data/processed data/x",
    # Unparseable commands that touch guarded tools
    "gh pr merge \"3",
]

ALLOWED = [
    "pytest",
    "ruff check . && pytest -q",
    "git status && git diff",
    "git push -u origin feat/KAN-42-amateur-drawings",
    "git push origin HEAD",
    "git commit -m \"fix(data): handle empty batch (KAN-7)\"",
    GOOD_PR,
    "gh pr create --draft --label agent-authored,needs-reviewer --title t --body-file body.md",
    "gh pr create --draft --label agent-authored --reviewer someone --title t --body-file body.md",
    "gh pr view 3",
    "gh pr list",
    "gh pr diff 3 --repo keithfernandes12/young-illustrators-primer",
    "gh api repos/keithfernandes12/young-illustrators-primer/pulls",
    "gh api repos/{owner}/{repo}/pulls/3/comments --paginate",
    "gh api graphql -f query='{ viewer { login } }'",
    "gh repo view",
    "primer-data --data-uri .local-data run tinystories --batch-size 2",
    "PRIMER_DATA_URI=./tmp primer-data run clear_corpus",
    "primer-data sources",
    "primer-data versions",
    "gsutil ls gs://young-illustrators-primer-data/metadata/runs",
    "gsutil cp gs://young-illustrators-primer-data/metadata/runs/x.json ./x.json",
    "gcloud storage ls gs://young-illustrators-primer-data",
    "dvc status",
    "echo \"gh pr merge 3\"",
    "grep -rn 'git push origin main' docs",
]


@pytest.fixture
def workdir(tmp_path):
    (tmp_path / "body.md").write_text("## Jira\n- KAN-42\n", encoding="utf-8")
    (tmp_path / "no-key.md").write_text("## Jira\n- none\n", encoding="utf-8")
    return str(tmp_path)


def on_branch(name):
    return lambda cwd: name


@pytest.mark.parametrize("command", REFUSED)
def test_refused(guard, workdir, command):
    reason = guard.check_command(command, workdir, environ={}, branch_of=on_branch("feat/x"))
    assert reason, f"should be refused: {command}"


@pytest.mark.parametrize("command", ALLOWED)
def test_allowed(guard, workdir, command):
    reason = guard.check_command(command, workdir, environ={}, branch_of=on_branch("feat/x"))
    assert reason is None, f"should be allowed: {command} -> {reason}"


def test_pipeline_uses_local_default_from_environment(guard, workdir):
    environ = {"PRIMER_DATA_URI": ".local-data"}
    assert guard.check_command("primer-data run tinystories", workdir, environ=environ) is None


def test_pipeline_bucket_default_from_environment_is_refused(guard, workdir):
    environ = {"PRIMER_DATA_URI": "gs://young-illustrators-primer-data"}
    assert guard.check_command("primer-data run tinystories", workdir, environ=environ)


def test_bare_push_on_main_is_refused(guard, workdir):
    assert guard.check_command("git push", workdir, environ={}, branch_of=on_branch("main"))
    assert guard.check_command("git push origin HEAD", workdir, environ={}, branch_of=on_branch("main"))
    assert guard.check_command("git push", workdir, environ={}, branch_of=on_branch("feat/x")) is None


def test_main_exit_codes(guard, monkeypatch, capsys):
    def run(command):
        payload = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": "."}
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
        return guard.main()

    assert run("gh pr merge 3") == 2
    assert "Agents never merge" in capsys.readouterr().err
    assert run("pytest") == 0


def test_other_tools_are_ignored(guard, monkeypatch):
    payload = {"tool_name": "Read", "tool_input": {"file_path": "x"}}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert guard.main() == 0
