"""guard_prod.py refuses commands aimed at the production project and allows staging work."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / ".claude" / "hooks" / "guard_prod.py"
spec = importlib.util.spec_from_file_location("guard_prod", HOOK)
assert spec and spec.loader
guard_prod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard_prod)

ENV = {"YIP_PROD_PROJECT": "yip-prod-123"}


@pytest.mark.parametrize(
    "command",
    [
        "gcloud run deploy yip-api --project yip-prod-123 --image x",
        "gcloud run services update-traffic yip-api --project=yip-prod-123 --to-latest",
        "YIP_DEPLOY_ENV=prod bash .claude/skills/deploy-cloud-run/scripts/promote.sh yip-api",
        "terraform -chdir=infra workspace select prod",
        "terraform -chdir=infra plan -var-file=envs/prod.tfvars",
    ],
)
def test_refused(command):
    assert guard_prod.check(command, ENV)


@pytest.mark.parametrize(
    "command",
    [
        "gcloud run deploy yip-api --project yip-staging-456 --image x",
        "gcloud run services describe yip-api --project yip-prod-1234",  # a different project id
        "pytest -q",
        "terraform -chdir=infra plan -var-file=envs/staging.tfvars",
    ],
)
def test_allowed(command):
    assert guard_prod.check(command, ENV) is None


def test_override_allows_prod():
    assert (
        guard_prod.check("gcloud run deploy x --project yip-prod-123", {**ENV, "YIP_ALLOW_PROD": "1"}) is None
    )


def test_refusal_exits_2():
    payload = {"tool_name": "Bash", "tool_input": {"command": "gcloud run deploy x --project yip-prod-123"}}
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        env={"YIP_PROD_PROJECT": "yip-prod-123", "PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "guard_prod.py" in result.stderr
