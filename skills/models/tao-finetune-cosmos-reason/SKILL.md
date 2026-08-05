---
name: tao-finetune-cosmos-reason
description: >-
  Shared Cosmos3 frontend that explicitly routes Cosmos Framework and
  Cosmos-RL, validates runtime model/WTS/AETC/SLURM inputs, builds clean
  repository-derived images, prepares checkpoints, gates full training on a
  smoke run, and returns token-weighted losses and task-aware accuracy.
license: Apache-2.0
compatibility: Docker with NVIDIA Container Toolkit, or SLURM with Pyxis/Enroot and a user-supplied shared-storage configuration.
metadata:
  author: NVIDIA Corporation
  version: "0.3.0"
allowed-tools: Read Bash
tags: [cosmos, vlm, sft, peft, wts, aetc, slurm]
---

# Cosmos3 TAO training

Keep this as one shared model-facing frontend. Shared concepts live here;
backend-native runtime contracts remain separate in
`references/cosmos-framework-backend.yaml` and
`references/cosmos-rl-backend.yaml`. Never translate one backend's TOML into
the other backend's schema.

## Mandatory runtime intake

Before planning training, collect all of the following. Do not infer a path
from history, another user, a prior job, an image, or a developer checkout.

- `base_model_path_or_uri`; require `base_model_revision` for a URI/model ID.
- `base_model_format`; for a URI this must be `qwen3_vl` or `cosmos3_omni`.
- optional `prepared_checkpoint_path`; validate it instead of silently
  replacing it.
- WTS: training/validation annotation paths and media roots.
- AETC: one or more training/validation annotation paths, media roots, and
  optional task selection.
- explicit `backend` for a comparison; `cosmos-framework` or `cosmos-rl`.
- `training_mode`; `dense` or `peft`. PEFT also requires rank, alpha, dropout,
  target modules, bias, RS-LoRA, modules-to-save, and adapter precision.
- user-owned `results_dir`, `checkpoint_dir`, `cache_dir`, and, for SLURM,
  `sqsh_cache_dir`, `sqsh_path`, `ssh_key_path`, container mounts, and all
  scheduler settings.
- clean repository paths and exact commits for the selected native backend,
  TAO integration, DAFT, and TAO Core; image tags/base images/build context are
  runtime inputs.

The planner preserves each original path and reports an accessible `realpath`.
Missing paths fail. No fallback dataset, checkpoint, cache, result directory,
image, partition, account, mount, SSH key, or shared-storage root is allowed.

## Backend selection

Run `scripts/cosmos_workflow.py resolve` first.

| Request | Automatic selection |
|---|---|
| Cosmos3-Nano plain train | Cosmos Framework |
| Cosmos3-Nano AutoML/HPO | Cosmos-RL |
| Nano evaluate/inference/quantize | Cosmos-RL |
| Cosmos3-Edge native train | Cosmos Framework |

An explicit supported backend wins. Comparative runs reject `auto`, so both
sides of an experiment are deliberately forced. Framework-trained checkpoints
use the native exact-key exporter, then the repository-backed TAO evaluation
adapter. That does not make Framework a Cosmos-RL version.

## Required gates

Execute these stages in order and persist their outputs.

1. Resolve model/backend/action and load the selected backend contract.
2. Check credentials by presence only. Never read or persist credential
   values. Require a token only for the operation that needs it.
3. Validate host tools, clean repository commits/trees, build context, free
   storage, every original/resolved path, and image build inputs.
4. Build the selected native image and TAO action image from clean commits.
   Inspect `/opt/tao/image-provenance.json`; reject dirty, missing, or mismatched
   source. Never mount a host source checkout into training.
5. If needed, prepare the model with the converter packaged in the clean
   Framework image. URI downloads require immutable revisions. Validate exact
   tensor/config keys and fingerprint model, tokenizer, and processor files.
6. Validate every annotation and referenced media file, record counts,
   duplicates, train/validation overlap, task selection, and fingerprints.
   Verify the resolved inputs again from an allocated compute node.
7. For Cosmos-RL full video runs, prewarm train and validation caches using
   separate deterministic dataset+model+processor keys. Require complete
   manifests and resumable entries. Never reuse an unproven cache.
8. Generate backend-native TOML, environment, topology, preflight commands,
   parity data, and machine-readable job metadata. Full specs must contain no
   sample limit.
9. Convert the newly built image to a new SQSH when SLURM is selected. Record
   image ID/digest and SQSH SHA256; verify Pyxis/Enroot, mounts, non-root Python,
   packages, decoder, GPU memory/type, CUDA/PyTorch, NCCL, and storage on the
   allocated node.
10. Run a smoke job for every distinct backend × dataset adapter × training
    mode × checkpoint/evaluator path. Continue only on child exit zero,
    structured `SUCCESS`, finite global train/validation loss, checkpoint
    completion, and evaluator accuracy coverage.
11. Materialize a fresh full spec with all smoke limits removed. Launch with
    `afterok` only after the smoke gate, monitor scheduler and structured TAO
    state to a terminal result, and preserve the child exit code independently
    of scheduler state.
12. Export/evaluate the selected checkpoint with identical prompt,
    preprocessing, generation, normalization, and task scoring. Extract final
    metrics with `scripts/extract_cosmos_metrics.py`.

## Dataset contracts

WTS accepts an explicit annotation JSON array plus explicit media root for each
split. Records require media and at least a user/assistant conversation pair.
Validate every record and media reference, not a prefix sample.

AETC accepts DAFT `tao-vl-reason-v1.0` item envelopes and the supported tasks:
BCQ, MCQ, open-ended BCQ/MCQ, open QA, scene description, video
summarization, temporal localization, temporal description, and causal
linkage. BCQ/MCQ have deterministic accuracy. Generative tasks report their
defined text/task metrics and are excluded from aggregate accuracy with a
reason. Aggregate AETC accuracy is example-weighted over accuracy-defined
tasks.

## Dense and PEFT contracts

Dense SFT must have no active LoRA block and must report trainable, frozen, and
total parameter counts. PEFT must represent the same rank, alpha, dropout,
target modules, bias, RS-LoRA, modules-to-save, precision, and trainable count
on both backends. Reject a paired PEFT run when semantics cannot be matched.

For fair comparisons, force the same logical model, train/validation records,
media, prompt, frames, sequence length, precision, seed, epochs, effective
global batch, optimizer, learning rate, schedule, warmup, weight decay,
clipping, loss masking, validation/checkpoint cadence, evaluated checkpoint,
and generation/normalization settings. Classify differences as equivalent
syntax, unavoidable implementation difference, or invalid mismatch; an invalid
mismatch blocks the full pair.

## Metrics and completion

The required primary metrics are:

- complete-run globally reduced token-weighted training loss, with numerator
  and valid-label denominator;
- final-validation globally reduced token-weighted loss, with numerator and
  valid-label denominator;
- repository-evaluator validation accuracy, with correct/total, coverage,
  per-task metrics, aggregation definition, exclusions, and evaluator version.

Do not average console lines or rank means. A step loss is not the average
training loss. A validation heartbeat is not final validation loss. A
generative exact match is not accuracy unless the task defines it.

The native Framework callback and native Cosmos-RL logger own early failure,
checkpoint, progress, metric, and terminal events. Do not stage a status bridge
or patch status at container startup. `COMPLETED` from SLURM is failure when the
child exit code is nonzero or the terminal TAO state is not successful.

## SLURM invariants

Generated jobs use `#!/usr/bin/env bash` and are syntax-checked by Bash. They
use SQSH via Pyxis, disable requeue by default, run one launcher task per node,
write stdout/stderr to runtime-supplied paths, and exit with the training child
code. Framework topology is shard=`gpus_per_node`, replica=`nodes`.
Cosmos-RL uses one controller on node zero and its policy-worker topology.
Asynchronous distributed checkpointing is rejected for multi-node runs.

Every job metadata record must validate against
`schemas/cosmos-job-metadata.schema.json`. It records supplied/resolved paths,
model/data/image fingerprints, source commits, config and SQSH checksums,
requested/allocated topology, scheduler and child states, logs/results,
timestamps, and terminal TAO status without credentials.

## Source-affecting recovery

If a run exposes a code or image defect, stop the affected path, change the
owning repository, add a test, commit it, rebuild both image and SQSH from a
clean checkout, rerun smoke, and rerun every affected full job. Never edit a
running container, patch an existing image, reuse an old SQSH after a source
change, or rely on a temporary launch script as the implementation.
