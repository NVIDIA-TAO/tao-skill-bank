---
name: tao-finetune-anomalygennext
description: >-
  Validate a user AnomalyGenNext dataset and recipe, freeze its validation set,
  and prepare a canonical texture fine-tuning recipe. Use when AnomalyGenNext
  task weights do not yet exist. Do not use for ordinary defect generation.
license: Apache-2.0
compatibility: Requires the AnomalyGenNext 1.1 container and its model checkpoints.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags: [tao, data, anomalygen-next, fine-tuning, synthetic-data]
---

# Fine-tune AnomalyGenNext

This leaf prepares a user-owned dataset and optional recipe for task-specific
AnomalyGenNext LoRA training. The pinned image is declared in
`references/skill_info.yaml`; do not replace it with the older 1.0 release.

## Inputs

```text
DATASET/
  defect_spec.jsonl
  TEXTURE/
    clean_image/*
    anomaly_image/DEFECT/*
    mask/DEFECT/*
VALIDATION/testcase.jsonl
```

Also supply the Cosmos3-Nano base checkpoint directory, `Wan2.2_VAE.pth`, and
a local `facebook/dinov2-large` checkpoint directory. The validation JSONL must
contain `image_filename`, `mask_filename`, and `anomaly_type`; each trained
`TEXTURE+DEFECT` needs at least three rows. A separately stored defect spec is
accepted with `--defect-spec`.

An optional user `recipe.yaml` is a template. Custom training settings remain,
while dataset, checkpoint, validation, type order, and iteration-zero validation
are replaced by validated identities. Without a template, the packaged recipe
uses the established 5000-step defaults.

## Prepare

After the common launch review, run the `prepare_recipe` action:

```bash
scripts/prepare_finetune_recipe.py \
  --dataset-root /data/my_dataset \
  --validation-testcase /data/validation/testcase.jsonl \
  --base-checkpoint /models/Cosmos3-Nano \
  --vae-path /models/Wan2.2_VAE.pth \
  --nn-backbone /models/facebook/dinov2-large \
  --dataset-name my_dataset \
  --recipe-template /data/recipe.yaml \
  --output /results/canonical_recipe.yaml
```

The action freezes absolute validation paths, validates anomaly images and
masks, checks type agreement with `defect_spec.jsonl`, refuses output reuse,
and emits a recipe plus metadata. `validation_iter` must be a multiple of
`save_iter`; `max_iter` must reach a post-baseline validation.

## Train and accept

Invoke `tao-launch-workflow`, review the platform, image, mounts, GPU shape,
runtime, and exact recipe, then submit the `train` action. Bind the selected
DINOv2 directory read-only at the fixed container path declared in
`skill_info.yaml`; the public image does not bundle it. An optional Hugging Face
cache supplies the Qwen tokenizer for offline execution.

```bash
scripts/finetune_anomalygennext.sh \
  --recipe /results/canonical_recipe.yaml \
  --results-dir /new/training_results \
  --num-gpus 1
```

The wrapper uses the upstream trainer already inside the pinned container. It
refuses existing publication outputs and, unless `--resume` is explicit, an
existing runtime directory. Accept success only when `training_handoff.json`
is `COMPLETE`: iteration zero and a later scored checkpoint must exist,
`best_checkpoint.txt` must select the maximum `Average.nn_score`, and the score
must improve strictly. The handoff hashes the copied adapter and recipe.

Read `references/nn-score-contract.md` before changing metric or acceptance
settings. A 1000-step run is the smallest quality smoke; shorter runs prove
wiring only.
