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
    reason = guard_cloud.check("gcloud builds submit --tag img .", {})
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
        # Training itself is refused by the ADR rule; watching a person's run is the agent's
        # half of the handover, and it still has to name the member's own project.
        f"gcloud ai custom-jobs list --project {COMPUTE} --region us-central1",
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
    reason = guard_cloud.check("gcloud builds submit --tag img .", {})
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
        "gcloud ai custom-jobs list --project $YIP_COMPUTE_PROJECT --region us-central1",
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
    command = "gcloud builds submit --tag img . --display-name list"
    assert guard_cloud.check(command, {}), "unconfigured session must still refuse"
    assert guard_cloud.check(command, SPLIT_ENV), "no project named, so the target is unknown"
    # A genuine read is still a read, and the same build named at our project is allowed.
    assert guard_cloud.check(f"gcloud ai custom-jobs list --project {COMPUTE}", SPLIT_ENV) is None
    assert guard_cloud.check(f"{command} --project {PROJECT}", SPLIT_ENV) is None


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


# ------------------------------------------------- findings from Adit's review of PR #11 (d695651)


@pytest.mark.parametrize(
    "wrapper",
    [
        "bash -lc '{inner}'",
        "sh -ec '{inner}'",
        "bash --login -c '{inner}'",
        "eval '{inner}'",
    ],
)
def test_a_clustered_shell_flag_does_not_launder_the_command(wrapper):
    """Matching an exact `-c` missed `-lc` and `-ec`; short options cluster."""
    inner = f"gcloud run services update-traffic yip-api --to-latest --project {PROJECT}"
    assert guard_cloud.check(wrapper.format(inner=inner), SPLIT_ENV), wrapper


@pytest.mark.parametrize("verb", ["replace", "update"])
def test_rewriting_a_live_service_is_refused(verb):
    """Neither says "traffic", but a service YAML sets spec.traffic and update revisions latest."""
    command = f"gcloud run services {verb} svc.yaml --project {PROJECT}"
    assert guard_cloud.check(command, SPLIT_ENV)


def test_no_traffic_must_be_a_flag_not_a_substring():
    """`--labels=n=--no-traffic` matched the old text lookahead and passed as a canary."""
    sneaky = f"gcloud run deploy yip-api --image img --project {PROJECT} --labels=n=--no-traffic"
    assert guard_cloud.check(sneaky, SPLIT_ENV)
    real = f"gcloud run deploy yip-api --image img --project {PROJECT} --no-traffic --tag canary"
    assert guard_cloud.check(real, SPLIT_ENV) is None


@pytest.mark.parametrize(
    "command",
    [
        "gcloud ai custom-jobs create --project member-dev --region us-central1 --config j.yaml",
        "gcloud ai hp-tuning-jobs create --project member-dev --region us-central1",
        "gcloud run jobs execute yip-eval --project member-dev --region us-central1 --wait",
        "gcloud run jobs create yip-eval --project member-dev --region us-central1 --image x",
    ],
)
def test_agents_never_launch_training(command):
    """docs/adrs/where-model-training-happens.md. An ask rule is not enough in an unattended loop."""
    reason = guard_cloud.check(command, SPLIT_ENV)
    assert reason and "never train" in reason, command


def test_reading_a_job_is_still_allowed():
    """Monitoring somebody else's run is the agent's half of the handover."""
    for command in (
        f"gcloud ai custom-jobs list --project {COMPUTE} --region us-central1",
        f"gcloud ai custom-jobs describe 42 --project {COMPUTE} --region us-central1",
        f"gcloud ai custom-jobs stream-logs 42 --project {COMPUTE} --region us-central1",
    ):
        assert guard_cloud.check(command, SPLIT_ENV) is None, command


# --------------------------------- findings from an independent review-pr pass on 689fd6b


@pytest.mark.parametrize("track", ["beta", "alpha"])
@pytest.mark.parametrize(
    "rest",
    [
        "run deploy yip-api --image IMG",
        "ai custom-jobs create --config j.yaml",
        "ai hp-tuning-jobs create --config j.yaml",
        "run jobs execute yip-eval",
        "batch jobs submit j",
    ],
)
def test_a_release_track_does_not_defeat_the_token_rules(track, rest):
    """`gcloud beta run deploy` is the same operation, but it shifted every prefix by one.

    The regex rules survived because they search the whole segment; all four token-prefix rules
    did not, so deploy-without---no-traffic and every training launch were allowed.
    """
    command = f"gcloud {track} {rest} --project {PROJECT}"
    assert guard_cloud.check(command, SPLIT_ENV), command


def test_a_boolean_flag_does_not_swallow_the_next_flag():
    """Making flag values unable to vote made every valueless flag eat the flag after it.

    The command below is the GPU example from deploy_canary.sh's own header, and the guard was
    refusing it while telling the caller to use --no-traffic.
    """
    gpu = (
        f"gcloud run deploy yip-story --image IMG --gpu 1 --gpu-type nvidia-l4 --cpu 8 "
        f"--memory 32Gi --no-cpu-throttling --no-gpu-zonal-redundancy --max-instances 2 "
        f"--min-instances 0 --no-allow-unauthenticated --no-traffic --tag canary "
        f"--project {PROJECT} --region us-central1"
    )
    assert guard_cloud.check(gpu, SPLIT_ENV) is None, gpu
    # The value-cannot-vote fix it came from still holds.
    sneaky = f"gcloud run deploy yip-api --image IMG --project {PROJECT} --labels=n=--no-traffic"
    assert guard_cloud.check(sneaky, SPLIT_ENV)


@pytest.mark.parametrize("command", ["terraform -chdir=infra apply", "terraform -chdir=infra init"])
def test_terraform_is_not_asked_for_a_flag_it_does_not_have(command):
    """terraform has no --project, so demanding one made infra work impossible to run."""
    assert guard_cloud.check(command, SPLIT_ENV) is None, command


def test_bigquery_deletes_are_refused():
    """bq has its own delete verbs and none of the words the resource pattern looks for."""
    env = {**SPLIT_ENV, "CLOUDSDK_CORE_PROJECT": PROJECT}
    assert guard_cloud.check(f"bq rm -r -f {PROJECT}:primer_metrics", env)


def test_inline_interpreter_code_naming_a_cloud_tool_is_refused():
    """An interpreter can reach the cloud without the guard ever seeing a command.

    Partial by construction: the payload is code, not a command line. It stops the obvious shape.
    """
    inner = "import subprocess; subprocess.run(['gcloud','run','services','update-traffic','x'])"
    assert guard_cloud.check(f'python -c "{inner}"', SPLIT_ENV)
    assert guard_cloud.check(f"python3 -c '{inner}'", SPLIT_ENV)


@pytest.mark.parametrize(
    "command",
    [
        'python -c "print(1)"',
        "python -I scripts/gen_golden_scenarios.py configs/golden/scenarios.jsonl",
        "python -m pytest -q",
        "python -m primer.evaluation.golden --split golden --fake-models --out m.json",
    ],
)
def test_ordinary_python_is_not_blocked(command):
    """The repo runs Python constantly; the interpreter rule must not get in the way."""
    assert guard_cloud.check(command, SPLIT_ENV) is None, command


_TRAFFIC = f"gcloud run services update-traffic yip-api --to-latest --project {PROJECT}"
_REPLACE = f"gcloud run services replace s.yaml --project {PROJECT}"


@pytest.mark.parametrize(
    "command",
    [
        f"({_TRAFFIC})",
        f"x=$({_TRAFFIC})",
        f"x=$({_REPLACE})",
        f'echo "$({_TRAFFIC})"',
        f"echo `{_TRAFFIC}`",
        f"cat <({_TRAFFIC})",
        f"cat >({_TRAFFIC})",
        f'RESULT="$( {_TRAFFIC} )"',
    ],
)
def test_a_subshell_or_substitution_does_not_launder_the_command(command):
    """The shell runs these; the splitter did not look inside any of them.

    Command substitution happens inside double quotes too, so only single quotes stop the scan.
    """
    assert guard_cloud.check(command, SPLIT_ENV), command


@pytest.mark.parametrize(
    "command",
    [
        'echo "$(date)"',
        "VERSION=$(git rev-parse --short HEAD); echo $VERSION",
        "(cd /c/dev/young-illustrators-primer && pytest -q)",
        "diff <(sort a.txt) <(sort b.txt)",
        "python -c \"import subprocess; subprocess.run(['ls'])\"",
        f"echo '$({_TRAFFIC})'",  # single quotes: the shell never runs it
    ],
)
def test_ordinary_substitutions_are_not_blocked(command):
    """Substitution is everywhere in normal shell use; only the cloud commands inside matter."""
    assert guard_cloud.check(command, SPLIT_ENV) is None, command


def test_cancelling_a_job_is_refused_for_its_own_reason():
    """Not "agents never train" -- cancelling may stop someone else's run."""
    reason = guard_cloud.check(f"gcloud ai custom-jobs cancel 42 --project {COMPUTE}", SPLIT_ENV)
    assert reason and "someone else" in reason


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
