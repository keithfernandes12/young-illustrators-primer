/loop 5m Watch the canary of <service> on $YIP_SHARED_PROJECT. Read only: gcloud run services describe,
gcloud logging read for the canary revision over the last 10 minutes. Report error rate, p95 latency
and any safety-flag log lines versus the stable revision. If error rate is above 2% or a safety flag
appears, say ROLLBACK_RECOMMENDED and show the rollback.sh command (it is an ask rule; I will approve it).
After 3 consecutive healthy checks, say PROMOTE_RECOMMENDED and show promote.sh. Then stop the loop.
