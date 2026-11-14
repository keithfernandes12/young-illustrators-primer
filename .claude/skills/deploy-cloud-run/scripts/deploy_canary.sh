#!/usr/bin/env bash
# Usage: deploy_canary.sh <service> <image> [extra gcloud run deploy flags...]
# Example GPU worker:
#   deploy_canary.sh yip-story "$IMG" --gpu 1 --gpu-type nvidia-l4 --cpu 8 --memory 32Gi \
#     --no-cpu-throttling --no-gpu-zonal-redundancy --max-instances 2 --min-instances 0 --no-allow-unauthenticated
#
# The canary takes NO live traffic. It is reachable only on its tagged URL, which smoke_test.sh
# uses. Moving traffic to it is a person's action (docs/adrs/where-model-training-happens.md is the
# same principle for training); .claude/hooks/guard_cloud.py refuses every `update-traffic` from an
# agent, including a partial one, so this script must not call it.
source "$(dirname "$0")/_common.sh"
SERVICE="$1"; IMAGE="$2"; shift 2

# Reject the flags this script owns before doing anything else -- certainly before any gcloud call.
# Caller flags are forwarded, and gcloud takes the last occurrence of a repeated flag, so an extra
# --project or --tag would silently win and deploy somewhere nobody intended.
for flag in "$@"; do
  case "$flag" in
    --project|--project=*|--region|--region=*|--tag|--tag=*|--traffic|--traffic=*|--to-latest)
      echo "[yip-deploy] refusing: ${flag%%=*} is this script's to set, not the caller's" >&2
      exit 1
      ;;
  esac
done

# Fail closed. A describe that fails may mean the service is absent, but it may equally mean a
# permission problem or a transient API error -- and the old fallback deployed at 100% traffic in
# all three cases. Bootstrapping the first revision of a service is a person's job.
if ! DESCRIBE_ERROR="$(gcloud run services describe "$SERVICE" "${G[@]}" --format=none 2>&1)"; then
  cat >&2 <<ERROR
[yip-deploy] cannot describe $SERVICE in $PROJECT:
$DESCRIBE_ERROR

Refusing to deploy. If the service does not exist yet, a person creates the first revision; if this
is a permissions or API error, fix that first. This script only ever adds a zero-traffic revision to
a service that already exists.
ERROR
  exit 1
fi

PREV="$(serving_revision "$SERVICE")"
if [[ -z "$PREV" ]]; then
  echo "[yip-deploy] $SERVICE has no revision serving traffic; refusing (nothing to roll back to)" >&2
  exit 1
fi
echo "$PREV" > "$STATE_DIR/$SERVICE.prev"
echo "[yip-deploy] current serving revision: $PREV (saved for rollback)"

gcloud run deploy "$SERVICE" "$@" "${G[@]}" --image "$IMAGE" --no-traffic --tag canary

# Record exactly which revision was tagged, so promote.sh moves traffic to the revision that was
# smoke-tested rather than to whatever is newest by then.
CANARY_REV="$(tagged_revision "$SERVICE" canary)"
if [[ -z "$CANARY_REV" ]]; then
  echo "[yip-deploy] deployed, but could not read back the canary revision name" >&2
  exit 1
fi
echo "$CANARY_REV" > "$STATE_DIR/$SERVICE.canary"

URL="$(tag_url "$SERVICE" canary)"
echo "CANARY_REVISION: ${CANARY_REV}"
echo "CANARY_URL: ${URL}"
echo "CANARY_PERCENT: 0"
echo "[yip-deploy] the canary serves no live traffic; a person promotes it after the smoke test"
