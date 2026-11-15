---
title: Run and reach the shared MLflow tracking server
type: runbook
created: 2026-11-15
status: active
author: rishiparekh
tags: [mlflow, gcp, tracking, KAN-107]
related: [model-development]
---

# Run and reach the shared MLflow tracking server

## When to use it

- You want to log a run, or open the MLflow UI. Go straight to **Reach it from your machine**.
- You are granting a teammate access.
- The server is down and needs rebuilding from scratch.

One server holds every run the team logs, local runs included
(`docs/engineering/model-development.md`, *Tracking conventions*). It was first stood up on
2026-11-15. Its first run is S6c's drawing baseline, in experiment `yip-drawing`.

## What runs where

- **Project.** Everything is in the shared project, `young-illustrators-primer`.
- **VM.** `mlflow` in `us-central1-a`: an `e2-micro`, Debian 12, a 30 GB standard disk and a 2 GB
  swap file, running as service account `mlflow-server@…`.
- **Server.** MLflow **2.17.2** and Postgres 16 under Docker Compose, from `infra/mlflow/`. The
  run database is the `pgdata` volume on the VM.
- **Artifacts.** `gs://yip-mlflow-artifacts`. The service account holds
  `roles/storage.objectAdmin` on that bucket and nothing else. The server proxies every upload
  (`--serve-artifacts`), so no laptop needs bucket access.
- **Network.** MLflow has no application login, so the firewall is its only boundary. The intended
  design: the VM carries network tag `mlflow`; rule `allow-iap-mlflow` (priority 800) admits IAP's
  range (`35.235.240.0/20`) on ports 22 and 5000 to that tag only; rule `deny-non-iap-mlflow`
  (priority 900) denies everything else to those ports on that tag. The deny rule is needed
  because the default network's own rules (`default-allow-internal` for `10.128.0.0/9`,
  `default-allow-ssh` for `0.0.0.0/0`, both priority 65534) otherwise let any other VM reach port
  5000 and the whole internet reach port 22. The VM has an ephemeral external IP so it can pull
  images; that IP is billed.
- **Not yet true of the live server.** A read-only `gcloud compute firewall-rules list` on
  2026-11-15 showed `allow-iap-mlflow` with **no target tag** (so it applies to every VM in the
  network), no deny rule, and the VM untagged. Until a person runs **Harden the live server**
  below, any VM added to the default network can read and write every run, and SSH is open to the
  internet. Today `mlflow` is the project's only VM. Delete this bullet once that step is done.

## Before you start

- **Your own steps.** You need `gcloud` signed in as yourself, and on the shared project both
  `roles/iap.tunnelResourceAccessor` and read access to Compute Engine. The tunnel command looks
  the VM up first (`compute.instances.get`, `compute.instances.list`), which the IAP role does not
  include. `roles/compute.viewer` is enough.
- **Name the project every time.** Every `gcloud` line below passes
  `--project=young-illustrators-primer`. A teammate's default project is usually their own, and
  the create and delete commands would act there.
- **Person-only steps.** Every step that creates a resource or grants a role — steps 1–3,
  **Harden the live server** and **Grant a teammate access** — is for a person to run. Agents cannot make IAM grants
  (`.claude/settings.json`, `guard_cloud.py`) and must not create service-account keys.
- **Windows shells.** On Windows, the `gcloud` lines below are written so they run unchanged in
  both PowerShell and Git Bash. Three PowerShell traps cost time on the first setup:
  - `export` does not exist. Use `$env:NAME = "value"`, or `setx NAME value` to persist it.
  - A bare `$NAME` is an empty PowerShell variable. Use `$env:NAME`, or type the value out in full.
  - An unquoted comma splits an argument into an array. Quote it: `--allow="tcp:22,tcp:5000"`.
  - Do not append to `~/.bashrc` from Windows PowerShell 5.1: `>>` writes UTF-16, which Git Bash
    cannot read.

## Reach it from your machine

1. **Open the tunnel and leave the window open:**

   ```
   gcloud compute start-iap-tunnel mlflow 5000 --local-host-port=localhost:5000 --zone=us-central1-a --project=young-illustrators-primer
   ```

   It prints `Listening on port [5000]`. The UI is then at http://localhost:5000.

2. **Point your client at the tunnel and check it.** In Git Bash:

   ```
   export MLFLOW_TRACKING_URI=http://localhost:5000 PYTHONIOENCODING=utf-8
   curl -s localhost:5000/health
   ```

   The check prints `OK`.
   - **`PYTHONIOENCODING=utf-8` is required on Windows.** The 3.x client prints an emoji when a
     run ends. On a cp1252 console that raises inside `end_run` and leaves the run `RUNNING`.
     Your data is already logged by then: close the run with
     `MlflowClient().set_terminated(run_id, "FINISHED")`.

3. **The client is `mlflow-skinny==3.18.0`, not 2.17.2.** It talks to the 2.17.2 server over the
   same REST API for params, metrics, tags and artifacts. Both 2.17.2 packages were rejected on
   the project's Python 3.13 venv:
   - `mlflow==2.17.2` pins `pandas<3` and `pyarrow<18`, and pyarrow 17 has no 3.13 wheel.
   - `mlflow-skinny==2.17.2` would downgrade `packaging` 26.3 → 24.2 and `protobuf` 7.36 → 5.29.

   Check the plan before calling client APIs newer than the server, such as logged models or
   tracing.

4. **Name things as the conventions say.**
   - Experiment: `MLFLOW_EXPERIMENT_NAME=yip-<component>`.
   - Tags: the ones `model-development.md` lists.
   - Slice metrics: `slice/<key>/<value>/<metric>`. MLflow rejects `=` and `:` in metric names,
     so the documented `slice/<key>=<value>/<metric>` cannot be logged.

## Grant a teammate access (person-only)

Run by someone who can change the shared project's IAM. First check the teammate can already see
Compute Engine in the project (any of Viewer, Editor or `roles/compute.viewer` does); if not, grant
`roles/compute.viewer` too, because the tunnel command needs `compute.instances.get` and
`compute.instances.list`.

```
gcloud projects add-iam-policy-binding young-illustrators-primer --member="user:<their-google-account>" --role=roles/iap.tunnelResourceAccessor
gcloud projects add-iam-policy-binding young-illustrators-primer --member="user:<their-google-account>" --role=roles/compute.viewer
```

The second line is needed only when they have no Compute read access yet. It takes a minute or two
to apply. They then follow **Reach it from your machine**. If their SSH or tunnel still fails,
`gcloud compute ssh mlflow --zone=us-central1-a --project=young-illustrators-primer
--tunnel-through-iap --troubleshoot` names the missing permission.

## Harden the live server (person-only)

Do this once, before anyone adds another VM to the default network. It tags the VM, scopes the
allow rule to that tag and lets it win over the default rules, then denies the rest. `allow` is
priority 800 and `deny` is 900: the lower number wins, so IAP's range gets through and everyone else
is refused. Tag first, so the VM is never left without the rules.

```
gcloud compute instances add-tags mlflow --tags=mlflow --zone=us-central1-a --project=young-illustrators-primer
gcloud compute firewall-rules update allow-iap-mlflow --target-tags=mlflow --priority=800 --project=young-illustrators-primer
gcloud compute firewall-rules create deny-non-iap-mlflow --direction=INGRESS --action=DENY --rules="tcp:22,tcp:5000" --source-ranges=0.0.0.0/0 --target-tags=mlflow --priority=900 --network=default --project=young-illustrators-primer
```

Check it, then open the tunnel and `curl -s localhost:5000/health` to confirm IAP still gets in:

```
gcloud compute firewall-rules list --project=young-illustrators-primer --format="table(name,priority,sourceRanges.list(),allowed[].map().firewall_rule().list(),denied[].map().firewall_rule().list(),targetTags.list())"
```

`allow-iap-mlflow` and `deny-non-iap-mlflow` must both show target tag `mlflow`. This procedure has
not been run yet, so it is untested.

## Build it from scratch (person-only)

1. **Bucket and service account:**

   ```
   gcloud storage buckets create gs://yip-mlflow-artifacts --project=young-illustrators-primer --location=us-central1 --uniform-bucket-level-access
   gcloud iam service-accounts create mlflow-server --project=young-illustrators-primer
   gcloud storage buckets add-iam-policy-binding gs://yip-mlflow-artifacts --member="serviceAccount:mlflow-server@young-illustrators-primer.iam.gserviceaccount.com" --role=roles/storage.objectAdmin
   ```

2. **VM and firewall.** If `instances create` reports an API is off, first run
   `gcloud services enable compute.googleapis.com iap.googleapis.com --project=young-illustrators-primer`.

   The VM is created with the `mlflow` tag, and both rules target that tag, so they touch no other
   VM. Create the allow rule at priority 800 and the deny rule at 900 (see **Harden the live
   server** for why).

   ```
   gcloud compute instances create mlflow --project=young-illustrators-primer --zone=us-central1-a --machine-type=e2-micro --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=30GB --boot-disk-type=pd-standard --tags=mlflow --service-account=mlflow-server@young-illustrators-primer.iam.gserviceaccount.com --scopes=cloud-platform
   gcloud compute firewall-rules create allow-iap-mlflow --project=young-illustrators-primer --network=default --direction=INGRESS --source-ranges=35.235.240.0/20 --allow="tcp:22,tcp:5000" --target-tags=mlflow --priority=800
   gcloud compute firewall-rules create deny-non-iap-mlflow --project=young-illustrators-primer --network=default --direction=INGRESS --action=DENY --rules="tcp:22,tcp:5000" --source-ranges=0.0.0.0/0 --target-tags=mlflow --priority=900
   ```

3. **Copy the files up and connect.** The first `ssh` generates your key. If it fails with "Remote
   side unexpectedly closed", wait a minute: the guest agent has not added your user yet. The
   VM's serial log shows `Created user <you>` once it has.

   ```
   gcloud compute scp --recurse infra/mlflow mlflow:~ --zone=us-central1-a --project=young-illustrators-primer --tunnel-through-iap
   gcloud compute ssh mlflow --zone=us-central1-a --project=young-illustrators-primer --tunnel-through-iap
   ```

4. **On the VM: swap, Docker, password, start.** The e2-micro has 1 GB of RAM, so do not skip
   the swap.

   ```
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   sudo apt-get update && sudo apt-get install -y docker.io docker-compose
   cd ~/mlflow && echo "POSTGRES_PASSWORD=$(openssl rand -hex 24)" > .env && chmod 600 .env
   sudo docker-compose up -d --build
   sleep 20; curl -s localhost:5000/health
   ```

   The last line prints `OK`. If the `mlflow` container keeps restarting, run
   `sudo docker-compose logs --tail=40 mlflow`. One failed start while Postgres initialises is
   normal, and `restart: unless-stopped` retries it.

## If something goes wrong

- **The tunnel says permission denied, or cannot find the instance.** Your account lacks
  `roles/iap.tunnelResourceAccessor` or Compute read access
  (`compute.instances.get`, `compute.instances.list`); see **Grant a teammate access**.
- **The VM rebooted.** Both containers restart on their own. The swap file is in `/etc/fstab`.
- **Rebuild the server only.** `sudo docker-compose up -d --build` in `~/mlflow` rebuilds it. Runs
  survive, because they live in the `pgdata` volume and the bucket.
- **Cannot be undone:** `sudo docker-compose down -v`, or deleting the VM, destroys the `pgdata`
  volume and with it **every run's params, metrics and tags**. Only the artifacts in the bucket
  survive. There is no backup of the database yet; that is an open question in
  `docs/engineering/model-development.md`.
- **Tear it all down (person-only)**, in this order:
  1. `gcloud compute instances delete mlflow --zone=us-central1-a --project=young-illustrators-primer`
  2. `gcloud compute firewall-rules delete allow-iap-mlflow deny-non-iap-mlflow --project=young-illustrators-primer`
     (on a server that was never hardened, only `allow-iap-mlflow` exists; delete just that)
  3. `gcloud storage rm -r gs://yip-mlflow-artifacts`
  4. `gcloud iam service-accounts delete mlflow-server@young-illustrators-primer.iam.gserviceaccount.com --project=young-illustrators-primer`
