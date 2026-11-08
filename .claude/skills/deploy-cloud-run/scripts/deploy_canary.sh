#!/usr/bin/env bash
# Usage: deploy_canary.sh <service> <image> [extra gcloud run deploy flags...]
# Example GPU worker:
#   deploy_canary.sh yip-story "$IMG" --gpu 1 --gpu-type nvidia-l4 --cpu 8 --memory 32Gi \
#     --no-cpu-throttling --no-gpu-zonal-redundancy --max-instances 2 --min-instances 0 --no-allow-unauthenticated
source "$(dirname "$0")/_common.sh"
SERVICE="$1"; IMAGE="$2"; shift 2
CANARY_PERCENT="${YIP_CANARY_PERCENT:-10}"

if gcloud run services describe "$SERVICE" "${G[@]}" >/dev/null 2>&1; then
  PREV="$(serving_revision "$SERVICE")"
  echo "$PREV" > "$STATE_DIR/$SERVICE.prev"
  echo "[yip-deploy] current serving revision: $PREV (saved for rollback)"
  gcloud run deploy "$SERVICE" "${G[@]}" --image "$IMAGE" --no-traffic --tag canary "$@"
  gcloud run services update-traffic "$SERVICE" "${G[@]}" --to-tags "canary=${CANARY_PERCENT}"
else
  echo "[yip-deploy] $SERVICE does not exist yet; first deploy takes 100% (nothing to canary against)"
  gcloud run deploy "$SERVICE" "${G[@]}" --image "$IMAGE" --tag canary "$@"
fi
URL="$(tag_url "$SERVICE" canary)"
echo "CANARY_URL: ${URL}"
echo "CANARY_PERCENT: ${CANARY_PERCENT}"
