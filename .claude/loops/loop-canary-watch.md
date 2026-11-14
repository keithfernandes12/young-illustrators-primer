/loop 5m Watch the canary of <service> on $YIP_SHARED_PROJECT. Read only: gcloud run services describe,
gcloud logging read for the canary revision over the last 10 minutes. Report error rate, p95 latency
and any safety-flag log lines versus the stable revision. If error rate is above 2% or a safety flag
appears, say ROLLBACK_RECOMMENDED and show the rollback.sh command. **Do not run it and do not wait
for an approval prompt** -- guard_cloud.py refuses it by name, because moving traffic is a person's
action. Print the command for a person to run in their own terminal.
After 3 consecutive healthy checks, say PROMOTE_RECOMMENDED and show promote.sh. Then stop the loop.
