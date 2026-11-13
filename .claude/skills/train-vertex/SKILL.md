---
name: train-vertex
description: Launch, monitor, and evaluate training/fine-tuning jobs for Young Illustrator's Primer on Google Cloud (Vertex AI custom jobs or Cloud Run GPU jobs). Use for the story LLM LoRA, drawing recognizer fine-tune, readability scorer, safety classifier, and any hyperparameter sweep; when a /goal asks to "train until the golden gate passes"; or when checking a running job. Always logs to MLflow, records the DVC data version, and finishes with golden-gate.
---

# Train on Vertex AI

Pairs with vendored skills: `trl-training` (LoRA/SFT recipes), `huggingface-vision-trainer`
(drawing recognizer), `mlflow-onboarding`, `mlflow-trace-instrumentation`,
`agent-platform-model-registry`, `gcloud`.

## Before launching anything

1. `python .claude/skills/verify/scripts/verify.py` passes.
2. Run a CPU smoke test with a tiny model and about 20 examples
   (`python -m primer.story.train --config configs/story_lora.yaml --smoke`). This is a
   seconds-long import-and-one-step check, not training: `AGENTS.md` forbids local model
   training. A job that crashes in step 1 on a GPU costs real credits.
3. The training data is a frozen dataset version (`primer-data versions`; freezing is the
   `freeze-dataset-version` skill and a person runs the bucket write). Never train on unfrozen data.
4. Write the hypothesis for this run in the model's plan under "Verified facts" once it is checked.

Jobs run in **your own** project, `$YIP_COMPUTE_PROJECT`, burning your own credits. Everything
durable -- the data bucket, Artifact Registry, the models bucket and MLflow -- is in the shared
team project, `$YIP_SHARED_PROJECT` (`docs/engineering/model-development.md` section 6). Keeping
compute personal is what makes consolidation a one-line change later.

## Option A: Vertex AI custom job (preferred for training)

Write `infra/vertex/<run>.yaml` (template in `job_template.yaml` next to this file), then:

```bash
gcloud help ai custom-jobs create        # validate flags first, per the gcloud skill
gcloud ai custom-jobs create \
  --project "$YIP_COMPUTE_PROJECT" --region "${YIP_REGION:-us-central1}" \
  --display-name "story-lora-$(git rev-parse --short HEAD)" \
  --config infra/vertex/<run>.yaml
gcloud ai custom-jobs list --project "$YIP_COMPUTE_PROJECT" --region "${YIP_REGION:-us-central1}" --limit 5
gcloud ai custom-jobs stream-logs <JOB_ID> --project "$YIP_COMPUTE_PROJECT" --region "${YIP_REGION:-us-central1}"
```

Machine guidance: 1–3B LoRA fits on `g2-standard-8` + 1x `NVIDIA_L4` (24 GB) with 4-bit or
bf16 + gradient checkpointing. Use Spot scheduling when the job checkpoints to GCS every N
steps and can resume; confirm the current field name with `gcloud help ai custom-jobs create`
and the Vertex docs before relying on it. Check GPU quota in the region first.

## Option B: Cloud Run GPU job (short evals, batch illustration, small fine-tunes)

```bash
gcloud run jobs create yip-eval --project "$YIP_COMPUTE_PROJECT" --region "${YIP_REGION:-us-central1}" \
  --image "$IMAGE" --gpu 1 --gpu-type nvidia-l4 --cpu 8 --memory 32Gi \
  --no-gpu-zonal-redundancy --task-timeout 3600 --max-retries 0
gcloud run jobs execute yip-eval --project "$YIP_COMPUTE_PROJECT" --region "${YIP_REGION:-us-central1}" --wait
```

## Every run must

- Set `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME=yip-<component>`; log params, metrics,
  the git SHA, and `data_version` (the frozen version tag, e.g. `v1`).
- Write checkpoints and final adapters to a separate models bucket
  (`gs://$YIP_ARTIFACT_BUCKET/models/<component>/<run>/`), never into `gs://young-illustrators-primer-data`.
  The job's service account writes; the agent session does not.
- Finish with `golden-gate` on the golden split. Only a `GATE_RESULT: PASS` run may be
  registered as `candidate` in MLflow / Vertex Model Registry.
- Record the result (run id, metrics, cost estimate, pass/fail, next idea) as a dated Verified fact in the plan.

## Loop budget rules

- At most 1 GPU job running per component at a time, and at most `YIP_MAX_TRAIN_JOBS`
  (default 4) launches per /goal. Stop and report when the budget is spent.
- Never cancel or delete another person's job. Never change thresholds to make a run pass.
