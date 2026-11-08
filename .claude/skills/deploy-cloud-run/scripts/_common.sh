#!/usr/bin/env bash
# Shared settings for YIP deploy scripts. Staging unless YIP_DEPLOY_ENV=prod (guard hook still applies).
set -euo pipefail
: "${YIP_STAGING_PROJECT:?set YIP_STAGING_PROJECT}"
REGION="${YIP_REGION:-us-central1}"
if [[ "${YIP_DEPLOY_ENV:-staging}" == "prod" ]]; then
  : "${YIP_PROD_PROJECT:?set YIP_PROD_PROJECT}"
  PROJECT="$YIP_PROD_PROJECT"
else
  PROJECT="$YIP_STAGING_PROJECT"
fi
STATE_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)/.local-data/deploy"
mkdir -p "$STATE_DIR"
G=(--project "$PROJECT" --region "$REGION")
tag_url() { # tag_url <service> <tag>
  gcloud run services describe "$1" "${G[@]}" --format=json | python3 -c '
import json,sys
svc=json.load(sys.stdin); tag=sys.argv[1]
for t in svc.get("status",{}).get("traffic",[]):
    if t.get("tag")==tag: print(t.get("url","")); break' "$2"
}
serving_revision() { # revision currently holding the most traffic
  gcloud run services describe "$1" "${G[@]}" --format=json | python3 -c '
import json,sys
t=[x for x in json.load(sys.stdin).get("status",{}).get("traffic",[]) if x.get("percent")]
print(max(t,key=lambda x:x["percent"]).get("revisionName","") if t else "")'
}
echo "[yip-deploy] project=$PROJECT region=$REGION" >&2
