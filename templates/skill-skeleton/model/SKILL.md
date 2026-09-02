---
name: REPLACE-WITH-SKILL-NAME
description: >-
  One-to-three-sentence description of what the model does and when to use it.
  Use when the user asks to "fine-tune REPLACE-WITH-NETWORK", "train
  REPLACE-WITH-NETWORK on REPLACE-WITH-DATA-TYPE", or mentions
  REPLACE-WITH-DOMAIN-TERMS. Include literal trigger phrases.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit + NGC API key.
metadata:
  author: REPLACE-WITH-AUTHOR-NAME
  version: "0.1.0"
allowed-tools: Read Bash
---

# Skill Name

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Two-line summary of the model. What it is, what it produces.

## External dependencies

| Dependency | Purpose | Install |
|---|---|---|
| docker | Run the training container | https://docs.docker.com/engine/install/ |
| nvidia-container-toolkit | GPU access in containers | https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html |
| NGC API key | Pull `nvcr.io` images | https://ngc.nvidia.com/ |

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Quick start (Docker)

### Train

```bash
set -a; source /path/to/.env; set +a   # omit if already exported
docker run --gpus all --rm \
  -e HF_TOKEN \
  -v /path/to/spec.yaml:/spec.yaml \
  -v /path/to/data:/data \
  -v /path/to/results:/results \
  nvcr.io/nvidia/tao/tao-toolkit:<tag> \
  <entrypoint-cmd> train -e /spec.yaml
```

### Evaluate

```bash
set -a; source /path/to/.env; set +a   # omit if already exported
docker run --gpus all --rm \
  -e HF_TOKEN \
  -v /path/to/eval_spec.yaml:/spec.yaml \
  -v /path/to/checkpoint:/checkpoint \
  -v /path/to/results:/results \
  nvcr.io/nvidia/tao/tao-toolkit:<tag> \
  <entrypoint-cmd> evaluate -e /spec.yaml
```

### Inference (dry-run)

```bash
docker run --gpus all --rm \
  -v /path/to/inference_spec.yaml:/spec.yaml \
  -v /path/to/test_dir:/test \
  -v /path/to/output:/output \
  nvcr.io/nvidia/tao/tao-toolkit:<tag> \
  <entrypoint-cmd> inference -e /spec.yaml
```

Container image and per-action command are in `references/skill_info.yaml`. See `tao-skill-bank:tao-run-on-docker` for `docker run` conventions.

## CLI Reference

| Argument | Required | Default | Description |
|---|---|---|---|
| `-e, --experiment_spec` | Yes | — | Path to YAML spec file (mounted into container) |
| `-r, --results_dir` | No | from spec | Output directory |
| `-k, --key` | No | — | TLT encryption key (legacy models) |

## Output structure

```
<results_dir>/
├── train/
│   ├── checkpoints/
│   ├── logs/
│   └── metrics.json
├── evaluate/
│   └── results.json
└── inference/
    └── predictions/
```

## Credentials

- **`HF_TOKEN`** (if gated weights) — describe what the token is for.
- **`NGC_KEY`** — for pulling `nvcr.io` images.

## Pretrained weights

| Model | Path in container | Source |
|---|---|---|
| Base | `<container_path>` | NGC / HuggingFace / URL |

## Data format

Describe expected input structure — directory layout, file conventions, annotation formats.

## Critical overrides

If the model's schema has defaults that don't work out of the box, document them as a table.

| Parameter | Schema default | Required value | Why |
|---|---|---|---|
| ... | ... | ... | ... |

## Important parameters (reference)

Group by subsystem — training loop, model, optimization, vision, checkpointing, etc.

## Hardware

- Minimum: <GPU count> × <VRAM>
- Recommended: <GPU count> × <GPU type>

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError` on pretrained model | Missing mount | Add `-v` for the checkpoint dir |
| `OOM` during training | Batch size too large | Reduce `train.batch_size` or add GPUs |

## Running on a platform

This skill describes the model, its container command, and the spec. To run with
job tracking, S3 staging, or on a remote/multi-node cluster, hand the spec to a
**platform skill** — `tao-run-on-docker`, `-slurm`, `-kubernetes`, `-brev`,
`-virtualenv`, or any installed external platform skill —
which executes it through the four-verb contract (`submit`/`status`/`logs`/
`cancel`) with no SDK. See `tao-skill-bank:tao-launch-workflow`.
