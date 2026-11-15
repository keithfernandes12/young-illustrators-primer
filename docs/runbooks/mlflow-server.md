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
  images; that IP is billed. It is public, but no rule admits the internet to port 5000, and after
  hardening none admits it to port 22 either.
- **Not yet true of the live server.** A read-only `gcloud compute firewall-rules list` on
  2026-11-15 showed `allow-iap-mlflow` with **no target tag** (so it applies to every VM in the
  network), no deny rule, and the VM untagged. Until a person runs **Harden the live server**
  below, any VM added to the default network can read and write every run, and SSH is open to the
  internet. Today `mlflow` is the project's only VM. The bucket was also created without
  `--public-access-prevention` (it reports `inherited`); apply it with
  `gcloud storage buckets update gs://yip-mlflow-artifacts --public-access-prevention`. When both
  are done and the checks under **Harden the live server** pass, edit these: delete this bullet,
  the "untested" sentence and the "has not been run either" line under **Harden the live server**,
  the "Today a job in the same VPC…" sentence in the **Cloud jobs** bullet, the "until then"
  wording in the port comment of `infra/mlflow/docker-compose.yml`, and the "once the runbook's
  hardening is applied" parenthetical in `docs/engineering/model-development.md`.
- **Cloud jobs.** The server's external IP admits no one on port 5000, and after hardening only
  IAP can reach it, so a Vertex or Cloud Run job has no path in. Today a job in the same VPC
  reaches it only through `default-allow-internal`, which hardening closes. How cloud runs log is an open decision (an
  internal-only rule for a tagged range, or a login proxy); until it is made, log from a machine
  with the tunnel open.

## Before you start

- **Your own steps.** You need `gcloud` signed in as yourself, and on the shared project both
  `roles/iap.tunnelResourceAccessor` and read access to Compute Engine. The tunnel command looks
  the VM up first (`compute.instances.get`, `compute.instances.list`), which the IAP role does not
  include. `roles/compute.viewer` is enough.
- **Name the project every time.** Every `gcloud compute`, `gcloud iam` and `gcloud services` line
  below passes `--project=young-illustrators-primer`; the `gcloud storage` lines name the bucket,
  which is global. A teammate's default project is usually their own, and the create and delete
  commands would act there.
- **Person-only steps.** Every step that creates a resource, changes the network or grants a role
  — **Build it from scratch**, **Harden the live server**, **Grant a teammate access** and the
  teardown — is for a person to run. Agents cannot make IAM grants (`.claude/settings.json`, `guard_cloud.py`) and must
  not create service-account keys.
- **Windows shells.** On Windows, the `gcloud` lines below are written so they run unchanged in
  both PowerShell and Git Bash. Four PowerShell traps cost time on the first setup:
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
   - Slice metrics: `slice/<key>/<value>/<metric>`. MLflow rejects `=` in metric names everywhere and `:` on Windows clients,
     so the earlier form `slice/<key>=<value>/<metric>` cannot be logged.

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
this command names the missing permission:

```
gcloud compute ssh mlflow --zone=us-central1-a --project=young-illustrators-primer --tunnel-through-iap --troubleshoot
```

**To revoke access** (someone leaves, or an account is wrong), remove both roles. The IAP role is
granted on the whole project, so while it stands that person can tunnel to any VM and port the
firewall admits, not only `mlflow`:

```
gcloud projects remove-iam-policy-binding young-illustrators-primer --member="user:<their-google-account>" --role=roles/iap.tunnelResourceAccessor
gcloud projects remove-iam-policy-binding young-illustrators-primer --member="user:<their-google-account>" --role=roles/compute.viewer
```

Skip the second line if they hold `compute.viewer` for another reason. A tunnel they already
opened keeps working until it is closed, so ask them to close it.

## Harden the live server (person-only)

Do this once, before anyone adds another VM to the default network. It tags the VM, scopes the
allow rule to that tag, then denies the rest. `allow` is priority 800 and `deny` is 900: the lower
number wins, so the allow rule beats the deny rule, IAP's range gets through and everyone else is
refused. (The default rules sit at 65534 and lose to both.) Tag first, so the VM is never left
without the rules.

```
gcloud compute instances add-tags mlflow --tags=mlflow --zone=us-central1-a --project=young-illustrators-primer
gcloud compute firewall-rules update allow-iap-mlflow --target-tags=mlflow --priority=800 --project=young-illustrators-primer
gcloud compute firewall-rules create deny-non-iap-mlflow --direction=INGRESS --action=DENY --rules="tcp:22,tcp:5000" --source-ranges=0.0.0.0/0 --target-tags=mlflow --priority=900 --network=default --project=young-illustrators-primer
```

Check it, then open the tunnel and `curl -s localhost:5000/health` to confirm IAP still gets in:

```
gcloud compute firewall-rules list --project=young-illustrators-primer --format="table(name,priority,sourceRanges.list(),allowed[].map().firewall_rule().list(),denied[].map().firewall_rule().list(),targetTags.list())"
```

`allow-iap-mlflow` and `deny-non-iap-mlflow` must both show target tag `mlflow`. That table and a
working tunnel only show that IAP still gets in. Also check that nobody else does, from a machine
that is **not** using the tunnel. The external IP changes if the VM is stopped and started, so read
it first:

```
gcloud compute instances describe mlflow --zone=us-central1-a --project=young-illustrators-primer --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
curl -m 5 -sS http://<that-ip>:5000/health
curl -m 5 -sS telnet://<that-ip>:22
```

Run these in Git Bash, or as `curl.exe` in PowerShell (there `curl` is an `Invoke-WebRequest` alias
and `-m` fails). Run the port-22 line once **before** hardening as well: it should print an SSH
banner, which shows the check can see through your network. A timeout is only meaningful after
that. Afterwards both `curl` lines must time out (a banner or `OK` means the port is open). On a laptop this proves
that port 22 is closed to the internet; port 5000 never had an internet-facing allow rule. The
exposure hardening exists for is another VM in the network, so also create a throwaway VM in the
default network, run the same two `curl` lines against `mlflow`'s **internal** IP, and delete the
throwaway VM. The same step closes the bucket gap listed under **Not yet true of the live server**:
run `gcloud storage buckets update gs://yip-mlflow-artifacts --public-access-prevention` too. This
procedure has not been run yet, so it is untested; drop that statement only after these checks pass.

**If the tunnel or SSH stops working afterwards**, undo it. Run all three lines: the deny rule is
not always the cause (for example, if the tag did not land on the VM), and together they return the
rules and the VM to how they were first built:

```
gcloud compute firewall-rules delete deny-non-iap-mlflow --project=young-illustrators-primer
gcloud compute firewall-rules update allow-iap-mlflow --clear-target-tags --priority=1000 --project=young-illustrators-primer
gcloud compute instances remove-tags mlflow --tags=mlflow --zone=us-central1-a --project=young-illustrators-primer
```

The `--clear-target-tags` flag has not been run either.

## Build it from scratch (person-only)

1. **Bucket and service account:**

   ```
   gcloud storage buckets create gs://yip-mlflow-artifacts --project=young-illustrators-primer --location=us-central1 --uniform-bucket-level-access --public-access-prevention
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

   **Do not re-run the `scp` on a server that is already running** if you have an
   `infra/mlflow/.env` locally: it overwrites the VM's `.env`. Postgres applies its password only
   when the `pgdata` volume is first created, so a different password locks MLflow out of its own
   database. To update a changed compose file or Dockerfile, copy just that file
   (`mlflow:~/mlflow/`) and run `sudo docker-compose up -d --build` on the VM.

4. **On the VM: swap, Docker, password, start.** The e2-micro has 1 GB of RAM, so do not skip
   the swap.

   ```
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   sudo apt-get update && sudo apt-get install -y docker.io docker-compose
   cd ~/mlflow && (umask 077; echo "POSTGRES_PASSWORD=$(openssl rand -hex 24)" > .env)
   sudo docker-compose up -d --build
   sleep 60; curl -s localhost:5000/health
   ```

   Run the `.env` line only on a fresh VM: on a running one it makes a new password that does not
   match the existing `pgdata` volume. The last line prints `OK`; if it prints nothing, wait a
   little and run the `curl` again, since the first image build on an e2-micro is slow. If the
   `mlflow` container keeps restarting, run `sudo docker-compose logs --tail=40 mlflow`. One failed start while Postgres initialises is
   normal, and `restart: unless-stopped` retries it.

## If something goes wrong

- **The tunnel says permission denied, or cannot find the instance.** Your account lacks
  `roles/iap.tunnelResourceAccessor` or Compute read access
  (`compute.instances.get`, `compute.instances.list`); see **Grant a teammate access**.
- **The VM rebooted.** Both containers restart on their own. The swap file is in `/etc/fstab`.
- **Rebuild the server only.** `sudo docker-compose up -d --build` in `~/mlflow` rebuilds it from
  the copy of the files on the VM, not from the repo; copy a changed file up first (see
  **Build it from scratch**, step 3). Runs survive, because they live in the `pgdata` volume and the bucket.
- **Cannot be undone:** `sudo docker-compose down -v`, or deleting the VM, destroys the `pgdata`
  volume and with it **every run's params, metrics and tags**. Only the artifacts in the bucket
  survive. There is no backup of the database yet; that is an open question in
  `docs/engineering/model-development.md`.
- **Tear it all down (person-only).** This deletes every run and every artifact. Export what you
  need first; step 3 removes the bucket and everything in it. Teammates' project-level
  `roles/iap.tunnelResourceAccessor` and `roles/compute.viewer` grants stay in place; remove them
  with the revoke lines under **Grant a teammate access** if they should go. In this order:
  1. `gcloud compute instances delete mlflow --zone=us-central1-a --project=young-illustrators-primer`
  2. `gcloud compute firewall-rules delete allow-iap-mlflow deny-non-iap-mlflow --project=young-illustrators-primer`
     (on a server that was never hardened, only `allow-iap-mlflow` exists; delete just that)
  3. `gcloud storage rm -r gs://yip-mlflow-artifacts`
  4. `gcloud iam service-accounts delete mlflow-server@young-illustrators-primer.iam.gserviceaccount.com --project=young-illustrators-primer`
