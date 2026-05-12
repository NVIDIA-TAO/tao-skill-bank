---
name: normal-train
description: Standard single-step train/eval/export workflow for any TAO model. Use when training a model on a dataset without
  iterative augmentation.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Sub-skills declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash Write
tags:
- training
- single-step
- generic
---

# Normal Train

Standard supervised fine-tuning: train a model on a labeled dataset, optionally evaluate, then optionally export. The most common TAO workflow for adapting a pretrained model to a new dataset.

## Steps

1. **train** — always executed
2. **eval** — executed if `eval_dataset_uri` is resolved
3. **export** — optional, on user request after training

## Prerequisites

### Required
- **model**: A compatible TAO model (e.g., clip, nvdinov2, grounding_dino)
- **train_dataset_uri**: URI of the training dataset (e.g., `s3://bucket/train/`)
- **platform**: Ask from the generated supported-platform list:
  `${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py --format text`
- **container image confirmation**: resolve the default image from the selected
  model/action config, show it to the user, and require confirmation or
  `image=<override>` before creating runner files or submitting training.

### Optional
- **eval_dataset_uri**: Some model skills mark this as required — check the resolved model skill before treating it as optional.
- **base_checkpoint**: If not provided, defaults to the NGC pretrained checkpoint listed in the model skill, or trains from scratch if no NGC checkpoint exists.
- **image override**: Use `image=<override>` to pin a specific TAO toolkit build
  after reviewing the resolved default.

## Launch Intake

After the user confirms they want this standard train/eval/export workflow,
ask which supported platform they intend to run on. Generate the choices with
`scripts/list_tao_platforms.py --format text`; do not scan platform docs or
folders.

Also ask whether long-running monitoring should stay enabled and how many
minutes between status updates. Defaults: enabled, 5 minutes.

After the model/action are known, run `scripts/resolve_tao_image.py --model
<network> --action train --format text` and ask whether to use the resolved
image or an `image=<override>`. Do not create the normal-train runner until the
image is confirmed.

After platform selection, run
`scripts/list_tao_platforms.py --platform <platform> --format text` and ask
only for credentials relevant to that platform, plus any selected-model
credentials. Do not ask for unrelated platform credentials.
