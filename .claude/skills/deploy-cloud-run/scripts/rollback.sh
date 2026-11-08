#!/usr/bin/env bash
# Usage: rollback.sh <service> [revision]  -> 100% traffic back to the saved previous revision
source "$(dirname "$0")/_common.sh"
SERVICE="$1"; REV="${2:-$(cat "$STATE_DIR/$SERVICE.prev" 2>/dev/null || true)}"
[[ -z "$REV" ]] && { echo "no previous revision recorded; pass one explicitly (gcloud run revisions list)"; exit 1; }
gcloud run services update-traffic "$SERVICE" "${G[@]}" --to-revisions "${REV}=100"
echo "ROLLED_BACK: $SERVICE -> $REV"
