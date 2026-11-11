---
title: Ingest every available source into the bucket
type: runbook
created: 2026-11-09
status: active
author: keithfernandes12
tags: [data, gcp, ingestion, KAN-24, KAN-83]
related: []
---

# Ingest every available source into the bucket

## When to use it

To bring the bucket up to date with everything the pipeline can fetch: finish TinyStories (2.12M
stories), ingest Amateur Drawings (178,166 drawings, a ~48 GB archive), re-check the snapshot sources,
then re-infer the schemas, save the first data-check baseline and freeze a version.

## Before you start

- A Windows laptop with the project venv (`%USERPROFILE%\.venvs\primer`, installed from
  `requirements.txt` plus `pip install -e . --no-deps`), `gcloud auth application-default login`
  done as an account that can write to `gs://young-illustrators-primer-data`, ~50 GB free disk
  **outside OneDrive**, and ~10 GB of free memory for the data check.
- Every command here that writes to the bucket is run **by a person**. Agents only prepare them.
- **Time:** the archive has to come down (48 GB) and go back up (48 GB). The upload is the slow part:
  about 2 hours at 50 Mbit/s upload, 5 hours at 20 Mbit/s. Keep the laptop plugged in, on a wired or
  strong connection, with sleep set to *Never* while plugged in (Settings → System → Power).
- Cost: the bucket grows by about 50 GB (about $1 a month). Uploads to GCS are free.

## Steps

Run everything in **PowerShell** from the repo root. First, in every PowerShell window you use:

```
$pd = "$env:USERPROFILE\.venvs\primer\Scripts\primer-data.exe"
& $pd sources       # amateur_drawings: "pipeline ready"
```

1. **Window 1:** download the Amateur Drawings files to a folder outside OneDrive. `-C -` resumes an
   interrupted download; just rerun the same line.
   ```
   New-Item -ItemType Directory -Force C:\primer-downloads\amateur_drawings
   cd C:\primer-downloads\amateur_drawings
   curl.exe -L -C - -O https://dl.fbaipublicfiles.com/amateur_drawings/amateur_drawings_annotations.json
   curl.exe -L -C - -O https://dl.fbaipublicfiles.com/amateur_drawings/amateur_drawings.tar
   Get-ChildItem | Select-Object Name, Length    # 288409774 and 48208397683
   ```

2. **Window 2**, while that downloads: finish TinyStories. 2,065,719 stories are left after the
   54,000 already in, so five batches of 500,000 cover it (each run ends with `Batch <id> done`; the
   last one is about 66,000). Each run streams past the stories already taken, so later runs start
   slower. The sixth command should say `Nothing new to ingest`.
   ```
   1..5 | ForEach-Object { & $pd run tinystories --batch-size 500000 }
   & $pd run tinystories
   ```

3. **Window 2:** re-check the other sources. Each should say `Nothing new to ingest` (Quick, Draw! is
   complete; the snapshots make a new batch only if the upstream file or repo changed).
   ```
   foreach ($s in "quickdraw", "clear_corpus", "sight_words", "kuperman_aoa", "storyweaver", "african_storybook") { & $pd run $s }
   ```

4. **Once step 1 has finished:** ingest Amateur Drawings from the local copies. It logs every 10,000
   images and ends with `Batch <id> done`. If validation fails it stops before anything reaches
   `processed/` and names the failed check.
   ```
   $env:PRIMER_AMATEUR_DRAWINGS_DIR = "C:\primer-downloads\amateur_drawings"
   & $pd -v run amateur_drawings
   ```

5. **After steps 2–4 are all done:** re-infer the schema contracts from the complete data and review
   the diff, then save the first data-check baseline (profiles and the anomaly report in the bucket;
   KAN-83). Expect `0 errors`.
   ```
   & $pd profile --write-schemas
   git diff --stat configs/schemas
   & $pd profile --save
   ```

6. Freeze the data as `v1` (also refreshes `data/*.dvc`):
   ```
   & $pd version v1 -m "All available sources: TinyStories complete, Amateur Drawings added"
   ```

7. Delete the local copy of the archive (the images are in the bucket now):
   ```
   Remove-Item -Recurse C:\primer-downloads
   ```

8. Commit the code, the reviewed `configs/schemas/*.json` and `data/*.dvc` on a branch, open a PR,
   and after it merges tag the commit `data-v1`.

## If something goes wrong

- **Laptop slept or the connection dropped during step 4:** rerun step 4. The ETags are recorded only
  after every image is uploaded, so it starts a fresh batch from the local copy (no re-download). The
  partial batch stays under `raw/amateur_drawings/<batch>/`; once the new batch has validated, a
  person may remove it with
  `gcloud storage rm -r gs://young-illustrators-primer-data/raw/amateur_drawings/<partial batch>/`
  (the bucket's soft delete keeps it recoverable for 7 days).
- **Validation fails:** nothing reached `processed/`. Keep the downloaded files and send the console
  output to whoever is fixing it.
- **The data check reports errors:** after `--write-schemas` the contracts match the data, so an error
  points at a real problem in one dataset; fix it before freezing `v1`.
- **Out of memory in step 5:** close other apps and rerun; each dataset is checked one at a time.
- **`primer-data version` fails on DVC:** run it with `--no-dvc`, then `& $pd dvc-pointers`.
- Steps that cannot be undone: removing a partial batch (step 7 only deletes your local copy).
  Ingests never modify or delete existing data, and a version can't be overwritten.
