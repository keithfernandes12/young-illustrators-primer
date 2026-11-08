#!/usr/bin/env bash
# Usage: smoke_test.sh <service> [tag]   -> prints SMOKE_RESULT: PASS|FAIL
source "$(dirname "$0")/_common.sh"
SERVICE="$1"; TAG="${2:-canary}"
URL="$(tag_url "$SERVICE" "$TAG")"
[[ -z "$URL" ]] && { echo "no URL for tag $TAG"; echo "SMOKE_RESULT: FAIL"; exit 1; }
TOKEN="$(gcloud auth print-identity-token 2>/dev/null || true)"
AUTH=(); [[ -n "$TOKEN" ]] && AUTH=(-H "Authorization: Bearer $TOKEN")
ok=1
code=$(curl -s -o /dev/null -w '%{http_code}' "${AUTH[@]}" "$URL/healthz" || echo 000)
echo "GET /healthz -> $code"; [[ "$code" == "200" ]] || ok=0
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
FIXTURE="$ROOT/tests/fixtures/smoke_book_request.json"
if [[ "$SERVICE" == "yip-api" && -f "$FIXTURE" ]]; then
  code=$(curl -s -o /tmp/yip_smoke.json -w '%{http_code}' "${AUTH[@]}" -H 'Content-Type: application/json' \
         -X POST --data @"$FIXTURE" "$URL/v1/books:dryRun" || echo 000)
  echo "POST /v1/books:dryRun -> $code"; head -c 400 /tmp/yip_smoke.json; echo
  [[ "$code" == "200" ]] || ok=0
fi
if [[ $ok -eq 1 ]]; then echo "SMOKE_RESULT: PASS"; else echo "SMOKE_RESULT: FAIL"; exit 1; fi
