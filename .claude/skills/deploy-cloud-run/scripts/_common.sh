#!/usr/bin/env bash
# Shared settings for YIP deploy scripts.
#
# There is one Google Cloud project, so there is nothing to select: $YIP_PROJECT is it
# ($YIP_STAGING_PROJECT is accepted as the older name). The boundary agents must not cross is
# traffic, not a project, and .claude/hooks/guard_cloud.py enforces that separately.
set -euo pipefail

PROJECT="${YIP_PROJECT:-${YIP_STAGING_PROJECT:-}}"
: "${PROJECT:?set YIP_PROJECT in .claude/settings.local.json}"
REGION="${YIP_REGION:-us-central1}"

# Pick an interpreter the way the repo's hooks do: on Windows `python3` is the Microsoft Store
# alias stub, which prints "Python was not found" and exits non-zero, so the JSON parsing below
# would silently produce empty strings and every smoke test would fail.
PY=python3
"$PY" -c "" 2>/dev/null || PY=python

STATE_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)/.local-data/deploy"
mkdir -p "$STATE_DIR"
G=(--project "$PROJECT" --region "$REGION")
tag_url() { # tag_url <service> <tag>
  gcloud run services describe "$1" "${G[@]}" --format=json | "$PY" -c '
import json,sys
svc=json.load(sys.stdin); tag=sys.argv[1]
for t in svc.get("status",{}).get("traffic",[]):
    if t.get("tag")==tag: print(t.get("url","")); break' "$2"
}
serving_revision() { # revision currently holding the most traffic
  gcloud run services describe "$1" "${G[@]}" --format=json | "$PY" -c '
import json,sys
t=[x for x in json.load(sys.stdin).get("status",{}).get("traffic",[]) if x.get("percent")]
print(max(t,key=lambda x:x["percent"]).get("revisionName","") if t else "")'
}
echo "[yip-deploy] project=$PROJECT region=$REGION" >&2
