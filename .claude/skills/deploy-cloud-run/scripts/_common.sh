#!/usr/bin/env bash
# Shared settings for YIP deploy scripts.
#
# The served Cloud Run service and Artifact Registry are durable state, so they live in the
# SHARED team project, not a member's compute project (model-development.md section 6).
# $YIP_PROJECT and the older $YIP_STAGING_PROJECT still work for a single-project checkout.
# The boundary agents must not cross is traffic, not a project; guard_cloud.py enforces that.
set -euo pipefail

PROJECT="${YIP_SHARED_PROJECT:-${YIP_PROJECT:-${YIP_STAGING_PROJECT:-}}}"
: "${PROJECT:?set YIP_SHARED_PROJECT in .claude/settings.local.json}"
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
tagged_revision() { # tagged_revision <service> <tag> -- which revision currently carries that tag
  gcloud run services describe "$1" "${G[@]}" --format=json | "$PY" -c '
import json,sys
svc=json.load(sys.stdin); tag=sys.argv[1]
for t in svc.get("status",{}).get("traffic",[]):
    if t.get("tag")==tag: print(t.get("revisionName","")); break' "$2"
}
echo "[yip-deploy] project=$PROJECT region=$REGION" >&2
