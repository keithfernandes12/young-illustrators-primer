#!/usr/bin/env bash
# Usage: promote.sh <service>  -> 100% traffic to the latest (canary) revision
source "$(dirname "$0")/_common.sh"
SERVICE="$1"
gcloud run services update-traffic "$SERVICE" "${G[@]}" --to-latest
echo "PROMOTED: $SERVICE -> latest revision ($(serving_revision "$SERVICE"))"
