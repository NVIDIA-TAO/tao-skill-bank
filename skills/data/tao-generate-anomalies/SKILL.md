---
name: tao-generate-anomalies
license: Apache-2.0
compatibility: NVIDIA GPU, Docker runtime, and the images.metropolis_sdg.anomalygen image in versions.yaml.
metadata:
  author: NVIDIA Corporation
  version: "1.2.0"
allowed-tools: Read Bash
description: >-
  Prepare anomaly inputs, fine-tune an AnomalyGen 1.2 adapter, generate synthetic
  anomaly images, evaluate native quality metrics, and search per-sample settings.
  One entry point with prepare, full, finetune_only, and inference_only modes.
  Supports texture datasets and OD gap preparation, public pretrained adapters,
  or a user checkpoint with its exact training recipe and data.
tags: [tao, data]
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TAO Generate Anomalies

Use this skill for every AnomalyGen data operation. The application owns its
DEFT loop, mining strategy, training-pool mutation, and acceptance KPI; this skill
owns preparation, native 1.2 training/generation, and auditable output contracts.
If this is a standalone install, initialize `tao-setup` first.

Resolve the image with `python scripts/resolve_versions_key.py images.metropolis_sdg.anomalygen`.
The sole runtime is `nvcr.io/nvidia/cdf-anomalygen:1.2.0`; native repository root <!-- versions-key: images.metropolis_sdg.anomalygen -->
inside that image is `/workspace/cdf-anomalygen`. Never launch a legacy image.
Delegate container lifecycle to the selected platform skill's `create`, `run`,
`wait`, and `remove` operations. Keep job identity, bounded wait, persistent logs,
state, and cleanup receipts. See [container-runtime.md](references/container-runtime.md).

| Mode | Behavior |
|---|---|
| `prepare` | Freeze/filter native testcases or prepare masks; `--source od-gaps` supports inputs, AMP, and finalize stages. No model training/generation. |
| `full` | Prepare validation, train, verify NN improvement and native best checkpoint, then prepare/generate/evaluate/search. |
| `finetune_only` | Prepare validation, train, and publish a hash-bound checkpoint/recipe handoff. |
| `inference_only` | Generate from a checkpoint/run directory plus its exact training recipe; evaluate and search without training. |

## Run contract

Entry point: `python3 scripts/run_anomalygen.py MODE --name NAME --output-dir NEWDIR`.
Paths are compute-visible absolute paths. Use a fresh output and run name; never
silently reuse a completed or failed run. `--repo` defaults to the image root.
The complete machine contract is [skill_info.yaml](references/skill_info.yaml).
For structured actions, optional `pseudo_label_flag` and `skip_quality_flag`
default to empty strings (omit their argument entirely). Set the former to the
single token `--pseudo-label` for COCO output, or the latter to `--skip-quality`
for an explicit generation-only smoke. Do not append `true`/`false` values to
these flags. Both options affect only `full` and `inference_only` generation.

- `--dataset-dir ROOT`: native `TEXTURE/{clean_image,anomaly_image/DEFECT,mask/DEFECT}` data.
- `--checkpoint PATH --recipe YAML`: inference adapter or run directory and its exact training recipe. A similarly named recipe is insufficient.
- `--input-data-path JSONL`: explicit one-output-per-testcase generation inputs; otherwise use `--num-sdg N` and mask placement.
- `--validation-testcase JSONL`: supplied validation masks; otherwise preparation uses the validation placement route.
- `--base-checkpoint DCP --vae-path FILE`: explicit native base/VAE overrides where required; checkpoint geometry and class stamps remain authoritative.
- `--expected-dataset-inventory JSON`: optional exact reference-dataset inventory, a sorted array of `{path, sha256, bytes}` using relative paths. Verify input and filtered-copy membership/bytes before generation, evaluation and refinement; a mismatch stops. The source must already exclude protected files when this contract is supplied. Omit to retain the normal filtering behavior.
- `--exclude-stems TXT`: exact filename stems, one per line. Applied before training, clean/reference selection, mask selection and generation; frozen with a hash. No dataset-specific identity rules live here.
- `--parallelism ddp --compile true`: preserve TAO training runtime controls, with native environment forwarding.
- `--num-gpus 1 --base-seed 1 --amp-seed 43`: explicit reproducible generation and placement controls.
- `--mask-backend amp` is the default, preserving the previous TAO route. AMG is opt-in and requires explicit generator, judge, and model registry; it does not silently launch services.
- `--num-search-run 3 --score nn`: native 1.2 per-sample search. No automatic threshold/drop or regeneration/top-up stage. `filter.py --drop_ratio` is a separate explicit native operation.
- `--pseudo-label`: native mask-based COCO export with `--no_caption`; OD application then converts/merges labels. `--skip-quality` is explicit generation-only smoke behavior, never an unreported quality pass.

Read [input-contract.md](references/input-contract.md), then select
[aoi-texture.md](references/aoi-texture.md) or [od-defects.md](references/od-defects.md).
The default public reproduction example is PCB-Nano; `nvidia/Cosmos-AnomalyGen-Metal-Nano` and
`nvidia/Cosmos-AnomalyGen-Glass-Nano` are additional public examples.

## Verification and completion

The native generator keeps guardrails enabled. Do not bypass a missing guardrail
asset or suppress a native failure. Verify the native timing receipt, and report
its actual capabilities: 1.2 has text blocking and image face blurring;
`image_guardrail_enforcing=false` must not be described as image content blocking.
Guardrail-blocked testcases produce no reconstructed image and are not retried to
make an unsafe request pass.

Validate identity, not just totals: each testcase index appears exactly once in
generation CSV or blocked CSV; source image/mask/type must agree; generated files
must match the CSV. Require **generated + guardrail-blocked = testcases**.
`anomalygen_contract.py` performs this verification. A fully blocked batch can be
complete with `training_ready=false`; applications must not train on it.

Preserve native CSVs, timing, stamps and selected-output layout. Record recipe,
testcase and exclusion hashes, checkpoint identity, commands, wall time, native
phase timing, selected quality policy and training handoff. Never claim quality
or compatibility from a structural unit test alone. See [checkpoints.md](references/checkpoints.md)
for stamped 1.2 and legacy 1.1 compatibility.

The AnomalyGen and ChangeNet AOI skills are unsigned on this development branch;
their stale pre-migration signature files have been removed.
