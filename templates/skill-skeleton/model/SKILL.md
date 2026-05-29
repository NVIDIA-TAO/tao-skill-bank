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
  version: "0.1"
allowed-tools: Read Bash
---

# Skill Name

Two-line summary of the model. What it is, what it produces.

## External dependencies

| Dependency | Purpose | Install |
|---|---|---|
| docker | Run the training container | https://docs.docker.com/engine/install/ |
| nvidia-container-toolkit | GPU access in containers | https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html |
| NGC API key | Pull `nvcr.io` images | https://ngc.nvidia.com/ |

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only; otherwise default to `auto`. When `automl_policy: auto`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Quick start (Docker)

### Train

```bash
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

## Optional: running via the TAO SDK

If `nvidia-tao-sdk` is installed and you want job tracking + S3 I/O wrapping:

```python
from tao_sdk.platforms.brev import BrevSDK   # or: from tao_sdk.platforms.lepton import LeptonSDK
sdk = BrevSDK()
job = sdk.create_job(
    image='nvcr.io/nvidia/tao/tao-toolkit:<tag>',
    command='<entrypoint-cmd> train -e /spec.yaml',
    gpu_count=1,
    env_vars={'HF_TOKEN': os.environ['HF_TOKEN']},
    inputs={'/spec.yaml': 's3://bucket/specs/train.yaml',
            '/data/':     's3://bucket/datasets/...'},
    outputs=['/results/'],
)
```

See `tao-skill-bank:tao-run-platform` for full SDK semantics.
