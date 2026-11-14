#!/usr/bin/env bash
# Usage: promote.sh <service> [revision]  -> 100% traffic to the smoke-tested canary revision
#
# A person runs this. .claude/hooks/guard_cloud.py refuses it from an agent session, by name,
# because the gcloud call below never reaches a hook of its own.
#
# It promotes the EXACT revision deploy_canary.sh recorded, not `--to-latest`. If a second deploy
# lands while the first canary is being watched, --to-latest would promote the revision nobody
# tested.
source "$(dirname "$0")/_common.sh"
SERVICE="$1"
REV="${2:-$(cat "$STATE_DIR/$SERVICE.canary" 2>/dev/null || true)}"

if [[ -z "$REV" ]]; then
  echo "no canary revision recorded for $SERVICE. Run deploy_canary.sh first, or pass the" >&2
  echo "revision explicitly: promote.sh $SERVICE <revision>  (gcloud run revisions list)" >&2
  exit 1
fi

CURRENT_CANARY="$(tagged_revision "$SERVICE" canary)"
if [[ -n "$CURRENT_CANARY" && "$CURRENT_CANARY" != "$REV" ]]; then
  echo "refusing: the canary tag now points at $CURRENT_CANARY, not the tested $REV." >&2
  echo "Something deployed since the smoke test. Re-test, or pass the revision explicitly." >&2
  exit 1
fi

gcloud run services update-traffic "$SERVICE" "${G[@]}" --to-revisions "${REV}=100"
echo "PROMOTED: $SERVICE -> $REV"
