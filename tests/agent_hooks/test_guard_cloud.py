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


# ------------------------------------------------- findings from the Copilot review of PR #11
#
# Each case below was a confirmed bypass or a confirmed false refusal before the guard was
# rewritten. They are the reason the rewrite exists, so they are pinned individually.


@pytest.mark.parametrize(
    "command",
    [
        # Every skill document writes the project this way. The hook sees unexpanded shell text,
        # so before the rewrite the literal "$YIP_SHARED_PROJECT" counted as a foreign project and
        # the documented build and training commands were refused by our own guard.
        'gcloud builds submit --project "$YIP_SHARED_PROJECT" --tag img .',
        "gcloud ai custom-jobs create --project $YIP_COMPUTE_PROJECT --region us-central1 --config j.yaml",
        "gcloud run services list --project ${YIP_SHARED_PROJECT}",
        'gcloud run services list --project "$env:YIP_SHARED_PROJECT"',
    ],
)
def test_project_written_as_a_variable_is_resolved(command):
    assert guard_cloud.check(command, SPLIT_ENV) is None, command


def test_an_unset_project_variable_fails_closed():
    """A reference we cannot resolve is not evidence that the target is ours."""
    reason = guard_cloud.check("gcloud builds submit --project $YIP_NOT_SET --tag img .", SPLIT_ENV)
    assert reason and "not set" in reason


@pytest.mark.parametrize(
    "command",
    [
        f"gcloud run services update-traffic yip-api --project {PROJECT} --to-tags canary=10",
        f"gcloud run services update-traffic yip-api --project {PROJECT} --to-revisions yip-api-2=50",
        f"gcloud run services update-traffic yip-api --project {PROJECT} --to-latest",
    ],
)
def test_any_traffic_change_is_refused_not_only_full_promotion(command):
    """10% of live requests is still live requests; the boundary is traffic, not percentage."""
    assert guard_cloud.check(command, SPLIT_ENV)


def test_replacing_a_whole_iam_policy_is_refused():
    """set-iam-policy replaces everything, which the binding-only pattern never matched."""
    assert guard_cloud.check(f"gcloud projects set-iam-policy {PROJECT} policy.json", SPLIT_ENV)


@pytest.mark.parametrize(
    "command",
    [
        "bash .claude/skills/deploy-cloud-run/scripts/promote.sh yip-api",
        "bash .claude/skills/deploy-cloud-run/scripts/rollback.sh yip-api",
        "./.claude/skills/deploy-cloud-run/scripts/promote.sh yip-api",
    ],
)
def test_deployment_helpers_are_refused_by_name(command):
    """A hook never sees the gcloud call inside a script, so the script must be refused itself."""
    assert guard_cloud.check(command, SPLIT_ENV), command


def test_a_nested_shell_does_not_launder_the_command():
    inner = f"gcloud run services update-traffic yip-api --project {PROJECT} --to-latest"
    assert guard_cloud.check(f"bash -c '{inner}'", SPLIT_ENV)
    assert guard_cloud.check(f'sh -c "{inner}"', SPLIT_ENV)


def test_canary_deploy_is_still_allowed():
    """The fix must not take away what the loops legitimately do."""
    command = f"gcloud run deploy yip-api --project {PROJECT} --no-traffic --tag canary --image x"
    assert guard_cloud.check(command, SPLIT_ENV) is None


def test_inherited_core_project_is_the_effective_target():
    """Configuring an allowed YIP_* project does not configure gcloud's actual target."""
    env = {**SPLIT_ENV, "CLOUDSDK_CORE_PROJECT": "another-team-prod"}
    reason = guard_cloud.check("gcloud builds submit --tag img .", env)
    assert reason and "another-team-prod" in reason
    # Our own project inherited the same way is fine.
    ours = {**SPLIT_ENV, "CLOUDSDK_CORE_PROJECT": PROJECT}
    assert guard_cloud.check("gcloud builds submit --tag img .", ours) is None


def test_a_mutation_must_name_its_target():
    reason = guard_cloud.check("gcloud builds submit --tag img .", SPLIT_ENV)
    assert reason and "cannot be established" in reason


def test_a_flag_value_cannot_disguise_a_mutation_as_a_read():
    """`--display-name list` once made a job creation look read-only and skip the fail-closed check.

    Creating a job in our own compute project is allowed, so the bypass only mattered where
    read-only status is what exempts a command: an unconfigured session, and one with no target.
    """
    command = "gcloud ai custom-jobs create --config job.yaml --display-name list"
    assert guard_cloud.check(command, {}), "unconfigured session must still refuse"
    assert guard_cloud.check(command, SPLIT_ENV), "no project named, so the target is unknown"
    # The genuine read is still a read, and the genuine creation in our project is still allowed.
    assert guard_cloud.check(f"gcloud ai custom-jobs list --project {COMPUTE}", SPLIT_ENV) is None
    assert guard_cloud.check(f"{command} --project {COMPUTE}", SPLIT_ENV) is None


def test_every_segment_of_a_compound_command_is_checked():
    safe = f"gcloud run services list --project {PROJECT}"
    risky = f"gcloud run services delete yip-api --project {PROJECT}"
    assert guard_cloud.check(f"{safe} && {risky}", SPLIT_ENV)
    assert guard_cloud.check(f"{safe} ; {safe}", SPLIT_ENV) is None


def test_an_env_assignment_in_an_earlier_segment_still_counts():
    """`CLOUDSDK_CORE_PROJECT=x; gcloud ...` targets x, even though the segments are separate."""
    assert guard_cloud.check(
        f"CLOUDSDK_CORE_PROJECT=another-team-prod; gcloud run services list --project {PROJECT}",
        SPLIT_ENV,
    )


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
