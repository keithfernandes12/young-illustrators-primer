---
name: deploy-cloud-run
description: Deploy Young Illustrator's Primer services to Google Cloud Run with a canary, smoke test, promotion, and rollback. Use when building or deploying the FastAPI gateway, the GPU workers (story LLM, diffusion + ControlNet, TTS, ASR), or the web app; when asked to "ship", "deploy", "canary", "promote", or "roll back"; and when a loop is watching a canary. Staging is autonomous; production requires an explicit human go-ahead in the conversation.
---

# Deploy to Cloud Run

Pairs with the Google Cloud skills (user-level install, see the tutorial): follow `gcloud` (validate every leaf command
with `gcloud help <command>` before running it), `cloud-run-basics`,
`google-cloud-recipe-auth`, and `cloud-run-alert-configuration`.

## Environments

There is one Google Cloud project (`$YIP_PROJECT`, also read as `$YIP_STAGING_PROJECT`), so the
boundary is traffic, not a project id.

| Action | Who may do it |
|---|---|
| Deploy a canary revision with `--no-traffic --tag` | Claude, autonomously, through the scripts below |
| Smoke-test a canary by its tag URL | Claude, autonomously |
| Send traffic to a revision, delete a service, change IAM | A person, in their own terminal. `guard_cloud.py` refuses these whatever the project. |

Default region is `$YIP_REGION` (fallback `us-central1`, which has both L4 and
RTX PRO 6000 GPUs on Cloud Run).

## Service map (from the scoping doc's architecture)

| Service | Kind | Hardware | Notes |
|---|---|---|---|
| `yip-api` | Cloud Run service | CPU, 1 vCPU / 1 GiB | FastAPI gateway, auth, parent settings, enqueues jobs to Pub/Sub |
| `yip-story` | Cloud Run service | `nvidia-l4`, 8 vCPU / 32 GiB | Story LLM (LoRA) + readability/safety gate |
| `yip-illustrate` | Cloud Run service or worker pool | `nvidia-l4` | SD + ControlNet-Scribble (+ IP-Adapter); pull from Pub/Sub, write pages to GCS |
| `yip-tts` | Cloud Run service | CPU first, L4 only if latency needs it | Kokoro / Piper with word timestamps |
| `yip-web` | Cloud Run service or Firebase Hosting | CPU | Child reader + parent dashboard |

GPU services need instance-based billing (`--no-cpu-throttling`), at least 4 vCPU
and 16 GiB for an L4 (8 vCPU / 32 GiB recommended), and `--max-instances` below
the project's regional GPU quota. Keep `--min-instances 0` in staging; consider
`--min-instances 1` on `yip-story` only during the Dec 8–11 pilot to avoid cold starts.
Stage model weights in GCS and mount with Cloud Storage FUSE instead of baking them
into the image.

## Procedure

1. **Pre-flight**: `python .claude/skills/verify/scripts/verify.py` must end in
   `VERIFY_RESULT: PASS`. For model changes, `golden-gate` must end in
   `GATE_RESULT: PASS` -- which needs `primer.evaluation`, so model deploys are blocked until
   that lands (see docs/plans/autonomous-loops.md).
2. **Build** with Cloud Build into Artifact Registry (tag = short git SHA):
   ```bash
   gcloud builds submit --project "$YIP_STAGING_PROJECT" \
     --tag "$YIP_REGION-docker.pkg.dev/$YIP_STAGING_PROJECT/yip/<service>:$(git rev-parse --short HEAD)" <context-dir>
   ```
3. **Canary** (new revision, 0% traffic, tagged URL, then 10%):
   ```bash
   bash .claude/skills/deploy-cloud-run/scripts/deploy_canary.sh <service> <image> [extra gcloud flags...]
   ```
4. **Smoke test** the tagged canary URL:
   ```bash
   bash .claude/skills/deploy-cloud-run/scripts/smoke_test.sh <service>
   ```
   It calls `/healthz` and, for `yip-api`, `POST /v1/books:dryRun` with a fixture
   drawing, and prints `SMOKE_RESULT: PASS|FAIL`.
5. **Watch** for at least 15 minutes (a `/loop 5m` is ideal): error rate < 2%, p95
   latency within targets, zero safety flags. Use `cloud-logging-query-generation`
   to build the log query.
6. **Promote** or **roll back**:
   ```bash
   bash .claude/skills/deploy-cloud-run/scripts/promote.sh <service>
   bash .claude/skills/deploy-cloud-run/scripts/rollback.sh <service>   # sends 100% to the previous serving revision
   ```

## Never

- Never use `--allow-unauthenticated` on GPU workers; only `yip-api` and `yip-web`
  are public, and workers are invoked by a service account through Pub/Sub push or
  IAM-authenticated calls.
- Never put secrets in env vars or images; use Secret Manager (`--set-secrets`).
- Never log child names, raw drawings, or audio (see `privacy-guard`).
- Never delete revisions, services, buckets, or Artifact Registry images from a loop
  (`guard_bash.py` already refuses `gcloud ... delete`).
- Deploy scripts are bash; on Windows run them from Git Bash, as the team already does for hooks.
