---
title: Running the model containers from Git Bash and from WSL
type: engineering
created: 2026-11-13
status: active
author: rishiparekh
jira: [KAN-89]
tags: [windows, wsl, docker, gpu, models, gcp]
related: [onboarding, bounded-local-training-for-agents]
---

# Running the model containers from Git Bash and from WSL

Every teammate is primarily on Windows, and the model path has to work from Git Bash and from WSL.
`onboarding.md` already says **pick Windows or WSL and stay there, one environment per checkout** —
that rule is about the *virtual environment*, and it does not change: a venv built on Windows has
`.venv\Scripts`, WSL expects `.venv/bin`, and using one checkout from both sides makes git see
file-mode noise.

What this document adds is the **container** path, which is what the model work actually uses. You do
not need a second checkout for it. The container is the unit of reproducibility the training ADR
requires (`docs/adrs/bounded-local-training-for-agents.md`), so it has to be launchable from whichever
shell you live in.

> `docs/engineering/model-development.md`, referred to below, is not on this branch yet. It is
> drafted elsewhere and lands separately; `related` above omits it until it does.

## What was measured, and where

Measured on one Windows 11 machine on 2026-11-13: Docker Desktop 29.8.0 (WSL 2 backend, `OSType:
linux`), NVIDIA GeForce RTX 4060 Laptop GPU, driver 592.82, WSL 2 with Ubuntu.

| Claim | Status |
|---|---|
| `docker run --gpus all` reaches the GPU **from Git Bash** | **Verified.** The container printed `NVIDIA GeForce RTX 4060 Laptop GPU`. |
| A `-v` bind mount from Git Bash needs care | **Verified**, and it is worse than a single flag — see below. |
| The same from **WSL** | **Not verified.** Entering the distro failed from the session that ran these checks (`Wsl/Service/0x80072747`, a socket/queue error). `wsl --status` and `wsl -l -v` worked; running a command inside Ubuntu did not. Someone should repeat the two checks below from WSL and update this table. |

**WSL is therefore not the only local-GPU path.** Git Bash works. That was an open assumption and it
is now settled for the Git Bash side.

## Git Bash: the two conversions that bite

MSYS rewrites POSIX-looking paths on the command line into Windows paths before `docker.exe` sees
them. It does this in **two** places, and most write-ups only mention the first:

1. the `-v source:destination` specification, and
2. **any absolute path you pass as an argument to the command inside the container.**

The second is the one that wastes an afternoon. Running `docker run -v "$PWD:/work" img ls /work`
from Git Bash fails with:

```
ls: cannot access 'C:/Program Files/Git/work': No such file or directory
```

That is not the mount failing. The mount is fine; MSYS rewrote the `/work` argument to `ls` into a
path under the Git installation. With a single-character destination it is louder still — `-v
"$dir:/w"` becomes `'...;W:\'` and Docker refuses it outright with `destination can't be '/'`.

### The form that works

```bash
export MSYS_NO_PATHCONV=1
docker run --rm --gpus all \
  -v "$(cygpath -m "$PWD"):/work" -w /work \
  "$IMAGE" python -m primer.story.train --config configs/story_lora.yaml --smoke
```

Both halves are required, for different reasons:

- `MSYS_NO_PATHCONV=1` stops the rewriting — of the `-v` spec *and* of `/work` as an argument.
- `cygpath -m` turns the checkout path into the Windows-with-forward-slashes form
  (`C:/Users/...`) that Docker Desktop understands.

**`MSYS_NO_PATHCONV=1` on its own is a trap.** With a raw MSYS source path (`/c/Users/...`) the run
*succeeds* — exit 0, no error — because Docker treats the source as a path inside its own VM, creates
an empty directory there and mounts that. Your code is not in the container and nothing says so. If
a container claims your source tree is empty, this is why.

Doubling the leading slash (`-v "$dir://work"`) does not rescue it on its own, because it does
nothing about the argument conversion.

## WSL: what to expect

Clone under your Linux home, not `/mnt/c/...` — `onboarding.md` explains why, and it matters more
for training, where every epoch re-reads the dataset. Paths need no conversion, so the plain form
works:

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work "$IMAGE" <cmd>
```

Unverified here: see the table above.

## Credentials and project are per environment

**Application Default Credentials do not cross the boundary.** They are stored per environment:

| Environment | ADC path |
|---|---|
| Windows / Git Bash | `%APPDATA%\gcloud\application_default_credentials.json` |
| WSL | `$HOME/.config/gcloud/application_default_credentials.json` |

`CLOUDSDK_CONFIG` overrides both, and pointing WSL at the Windows file is not worth it — the file
references a Windows path and the two gcloud installs disagree about it. Run
`gcloud auth application-default login` **once in each environment you use**. That is a person's
action; agents are denied `gcloud auth` (`AGENTS.md`).

**`gcloud config` is also per environment**, which is the part people trip over. Setting
`YIP_COMPUTE_PROJECT` in Git Bash does nothing for your WSL shell. If you use both, set both:

```bash
# in each environment
export YIP_COMPUTE_PROJECT=<your own project>     # your Vertex AI jobs, your credits
export YIP_SHARED_PROJECT=young-illustrators-primer   # data bucket, registry, MLflow
```

`.claude/settings.local.json` covers Claude Code sessions only; your own shells need the exports (or
your profile). The guard reads either variable and refuses a project that is neither
(`.claude/hooks/guard_cloud.py`).

To hand ADC to a container, mount it read-only and point `GOOGLE_APPLICATION_CREDENTIALS` at the
mount — from Git Bash, with the conversions above:

```bash
export MSYS_NO_PATHCONV=1
docker run --rm --gpus all \
  -v "$(cygpath -m "$APPDATA")/gcloud:/gcloud:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcloud/application_default_credentials.json \
  -v "$(cygpath -m "$PWD"):/work" -w /work "$IMAGE" <cmd>
```

## Keep the model cache off the long-path cliff

Hugging Face caches under `%USERPROFILE%\.cache\huggingface` by default. Model repositories nest
deeply, and Windows still refuses paths over 260 characters unless long paths are enabled, so a
download fails partway with a confusing `FileNotFoundError` on a file whose name is plainly there.

Set `HF_HOME` somewhere short, and mount it so six-gigabyte downloads survive a container restart:

```bash
export HF_HOME=/c/hf           # Git Bash;  C:\hf
export HF_HOME=$HOME/hf        # WSL
```

```bash
docker run --rm --gpus all -v "$(cygpath -m "$HF_HOME"):/hf" -e HF_HOME=/hf ... "$IMAGE" <cmd>
```

A short host path also keeps the cache off the slow `/mnt/c` crossing if you are in WSL.

## Checks that keep this true

`tests/agent_hooks/test_script_modes.py` asserts that no script under `.claude/skills/*/scripts/`
hardcodes `python3` or embeds an absolute `/c/...` or `C:\...` path, either of which would work on
one person's machine and nowhere else. `tests/agent_hooks/test_deploy_scripts.py` checks the
interpreter fallback those scripts use instead.
