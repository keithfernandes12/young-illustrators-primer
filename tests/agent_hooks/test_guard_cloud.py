"""guard_cloud.py refuses cloud changes only a person should make, and leaves everything else alone.

This project has one Google Cloud project whose id is also the repository name, so the hook must
compare the value of --project rather than searching the command text. The collision tests below
are the reason the hook was rewritten: the earlier guard refused 6 of 6 everyday commands.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / ".claude" / "hooks" / "guard_cloud.py"
spec = importlib.util.spec_from_file_location("guard_cloud", HOOK)
assert spec and spec.loader
guard_cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard_cloud)

PROJECT = "young-illustrators-primer"
ENV = {"YIP_PROJECT": PROJECT}

# The normal six-member setup: durable state in the team project, this member's compute in theirs.
COMPUTE = "member-dev"
SPLIT_ENV = {"YIP_COMPUTE_PROJECT": COMPUTE, "YIP_SHARED_PROJECT": PROJECT}


@pytest.mark.parametrize(
    "command",
    [
        # The project id is the repository name. None of these touch Google Cloud.
        "git clone git@github.com:keithfernandes12/young-illustrators-primer.git",
        "cd /c/Users/Rishi/Desktop/Project/young-illustrators-primer && pytest -q",
        "gh repo view keithfernandes12/young-illustrators-primer",
        "gh pr list --repo keithfernandes12/young-illustrators-primer",
        "cat young-illustrators-primer/AGENTS.md",
        "ruff check young-illustrators-primer/src",
    ],
)
def test_repo_name_is_not_a_cloud_command(command):
    assert guard_cloud.check(command, ENV) is None


@pytest.mark.parametrize(
    "command",
    [
        f"gcloud run services list --project {PROJECT}",
        f"gcloud run services describe yip-api --project {PROJECT} --region us-central1",
        f"gcloud logging read severity>=ERROR --project {PROJECT} --limit 10",
        f"gcloud run revisions list --project={PROJECT}",
        f"gcloud ai custom-jobs describe 123 --project {PROJECT}",
        # A canary takes no traffic, so an agent may deploy one.
        f"gcloud run deploy yip-api --project {PROJECT} --no-traffic --tag canary --image x",
        "terraform -chdir=infra validate",
    ],
)
def test_allowed_on_our_project(command):
    assert guard_cloud.check(command, ENV) is None


@pytest.mark.parametrize(
    "command",
    [
        "gcloud run services list --project someone-elses-prod",
        "gcloud run services describe yip-api --project=another-team-prod",
        "gcloud config set project another-team-prod",
        "CLOUDSDK_CORE_PROJECT=another-team-prod gcloud run services list",
    ],
)
def test_refuses_a_project_that_is_not_ours(command):
    reason = guard_cloud.check(command, ENV)
    assert reason and "not one of ours" in reason


@pytest.mark.parametrize(
    "command",
    [
        f"gcloud run services update-traffic yip-api --project {PROJECT} --to-latest",
        f"gcloud run services update-traffic yip-api --project {PROJECT} --to-revisions yip-api-003=100",
        f"gcloud run deploy yip-api --project {PROJECT} --image x",
        f"gcloud run services delete yip-api --project {PROJECT}",
        f"gcloud projects delete {PROJECT}",
        f"gcloud iam service-accounts keys create key.json --project {PROJECT} --iam-account a@b",
        f"gcloud projects add-iam-policy-binding {PROJECT} --member user:a@b --role roles/owner",
        "terraform -chdir=infra destroy",
        "terraform -chdir=infra workspace select prod",
        "terraform -chdir=infra plan -var-file=envs/prod.tfvars",
    ],
)
def test_refuses_risky_operations_even_on_our_project(command):
    assert guard_cloud.check(command, ENV)


def test_deploy_env_prod_is_refused_in_both_shells():
    """The deploy scripts read YIP_DEPLOY_ENV, and either shell can set it."""
    assert guard_cloud.check("YIP_DEPLOY_ENV=prod gcloud run deploy x --no-traffic", ENV)
    assert guard_cloud.check('$env:YIP_DEPLOY_ENV = "prod"; gcloud run services list', ENV)
    assert guard_cloud.check('$env:YIP_DEPLOY_ENV = "staging"; gcloud run services list', ENV) is None


def test_fails_closed_when_no_project_is_configured():
    """An unset project used to mean "allow everything"; it now means "cannot be checked"."""
    reason = guard_cloud.check("gcloud ai custom-jobs create --region us-central1", {})
    assert reason and "no Google Cloud project is configured" in reason


def test_reads_are_still_allowed_without_a_project():
    assert guard_cloud.check("gcloud run services list", {}) is None
    assert guard_cloud.check("gcloud --version", {}) is None


def test_staging_variable_is_still_honoured():
    """The kit's deploy scripts set YIP_STAGING_PROJECT; it keeps working as the project id."""
    env = {"YIP_STAGING_PROJECT": PROJECT}
    assert guard_cloud.check(f"gcloud run services list --project {PROJECT}", env) is None
    assert guard_cloud.check("gcloud run services list --project other-project", env)


# --------------------------------------------------------- compute project + shared project


@pytest.mark.parametrize(
    "command",
    [
        # Builds and the registry live in the shared team project.
        f"gcloud builds submit --project {PROJECT} --tag {PROJECT}/yip/story:abc123 .",
        f"gcloud storage ls gs://yip-models --project {PROJECT}",
        # Training burns this member's own credits, in their own project.
        f"gcloud ai custom-jobs create --project {COMPUTE} --region us-central1 --config job.yaml",
        f"gcloud ai custom-jobs stream-logs 42 --project={COMPUTE} --region us-central1",
    ],
)
def test_both_of_our_projects_are_allowed(command):
    assert guard_cloud.check(command, SPLIT_ENV) is None


@pytest.mark.parametrize(
    "command",
    [
        "gcloud ai custom-jobs create --project someone-elses-project --region us-central1",
        "gcloud builds submit --project another-team-prod .",
        "gcloud config set project someone-elses-project",
    ],
)
def test_a_third_project_is_still_refused(command):
    reason = guard_cloud.check(command, SPLIT_ENV)
    assert reason and "not one of ours" in reason


@pytest.mark.parametrize("project", [PROJECT, COMPUTE])
@pytest.mark.parametrize(
    "template",
    [
        "gcloud run services update-traffic yip-api --project {p} --to-latest",
        "gcloud run services delete yip-api --project {p}",
        "gcloud projects add-iam-policy-binding {p} --member user:a@b --role roles/owner",
        "gcloud iam service-accounts keys create k.json --project {p} --iam-account a@b",
    ],
)
def test_risky_verbs_are_refused_in_both_projects(template, project):
    """Splitting the projects must not open a door in either of them."""
    assert guard_cloud.check(template.format(p=project), SPLIT_ENV)


def test_repo_name_collision_does_not_return_with_two_projects():
    """The shared project id is still the repository name; that must stay harmless."""
    assert guard_cloud.check("cd /c/dev/young-illustrators-primer && pytest -q", SPLIT_ENV) is None
    assert guard_cloud.check("cat young-illustrators-primer/AGENTS.md", SPLIT_ENV) is None


def test_neither_configured_and_not_read_only_is_refused():
    reason = guard_cloud.check("gcloud ai custom-jobs create --region us-central1", {})
    assert reason and "no Google Cloud project is configured" in reason


def test_compute_project_alone_is_enough():
    """A member who has only been given their own project is still guarded, not locked out."""
    env = {"YIP_COMPUTE_PROJECT": COMPUTE}
    assert guard_cloud.check(f"gcloud ai custom-jobs list --project {COMPUTE}", env) is None
    assert guard_cloud.check(f"gcloud builds submit --project {PROJECT} .", env)


def test_override_allows_everything():
    env = {**ENV, "YIP_ALLOW_PROD": "1"}
    assert guard_cloud.check(f"gcloud run services delete yip-api --project {PROJECT}", env) is None


@pytest.mark.parametrize("tool_name", ["Bash", "PowerShell"])
def test_refusal_exits_2(tool_name):
    payload = {
        "tool_name": tool_name,
        "tool_input": {"command": f"gcloud run services delete yip-api --project {PROJECT}"},
    }
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        env={"YIP_PROJECT": PROJECT, "PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "guard_cloud.py" in result.stderr


def test_other_tools_pass_through():
    payload = {"tool_name": "Read", "tool_input": {"file_path": "x"}}
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        env={"YIP_PROJECT": PROJECT, "PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
