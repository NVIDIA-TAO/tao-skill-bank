---
name: tao-run-dinov3-ssl-deft
description: Run the concrete, resumable DINOv3 SSL DEFT workflow with either GRIT-score or multi-task round-robin mining. Use for iterative DINOv3 self-supervised refinement from an unlabeled source pool. The skill selects a reviewed recipe and invokes the code-based workflow; it does not implement loop transitions itself.
license: Apache-2.0
compatibility: Requires the selected platform's native launcher and a TAO Data Services image containing the DINOv3 SSL DEFT modules; no host TAO or Python data-stack installation.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- application
- workflow
- deft
- dinov3
- ssl
---

# DINOv3 SSL DEFT

This skill invokes `nvidia_tao_ds.mining.dinov3.workflow`, packaged in TAO Data
Services. Its Python controller, not the agent, owns stage order, state, retries,
stopping, and artifact checks. `scripts/run_workflow.py` is only a compatibility
shim; it contains no orchestration and performs no installation.

## Instructions

1. Resolve the user's DINOv3 model and choose exactly one known pattern:
   - `grit-score` for label-free target scoring.
   - `multi-task-round-robin` for normalized per-task weakness allocation.
2. Run the shared `tao-launch-workflow` preflight and let the user choose among
   supported installed platforms. Do not default a platform.
3. Read `references/tao-container-integration.md` and preflight the selected
   execution profile. In the DS controller run `python -m
   nvidia_tao_ds.mining.dinov3.workflow preflight`; add `--gpu` only when that
   allocation also performs local GPU compute. For external runners, validate
   GPU operations and the chosen scoring backend in the actual compute image,
   not the CPU-only controller. Use the module's `init` command to generate a
   packaged recipe, fill paths and reviewed resources, then run `validate` and
   `plan`.
4. Inspect the declared manifests and storage destination, then show the compact
   approval contract below. Do not ask for approval with unknown inventory,
   materialization, cache, iteration, sampling, or monitoring behavior.
5. After approval, invoke `python -m nvidia_tao_ds.mining.dinov3.workflow run <config>`
   in the allocated DS runtime. Use `status`, `logs`,
   `resume`, `cancel`, and `report` for later interactions. Never edit `state.json`.
   A release-change continuation must use a new run directory and the explicit
   `adopt-training` command before `run`; never alter prior locks or state.

## Approval Contract

Present these six short items, populated with concrete values from `plan` and
preflight rather than generic descriptions:

1. **Loop**: score current model -> select weak targets -> retrieve relevant
   source neighbors -> publish cumulative training manifest -> initialize a
   candidate from the configured checkpoint policy -> train -> optional
   evaluate; state the maximum rounds, every stop condition, and any evaluation
   metric patience policy. Metric patience retains the best evaluated
   checkpoint for delivery; it does not gate individual rounds.
2. **Data**: target and immutable source snapshots, verified embedding-shard
   inventory, row/shard counts, parent-history input, storage locations, and
   estimated bytes/inodes required on the execution platform.
3. **Sampling**: per-round target total, normalized task weights and resolved
   integer quotas (or GRIT fraction), neighbors per target, relevance radius and
   floor, duplicate threshold, and whether search is exact or audited ANN. For
   ANN, name the persistent index and recall audit plus the approved probe count
   and candidate depth; state that final decisions use exact float32 reranking.
4. **Training**: passes per round, image resolution, checkpoint policy, immutable
   base checkpoint, fixed or sample-count-resolved platform resources, and the
   optimizer-update floor when node graduation is enabled. The shipped policy is
   `base_checkpoint_each_round`: mined data accumulates, while every candidate
   model is reinitialized from `model.base_checkpoint`. Explain that
   `training_manifest.parquet` is a cumulative, sample-ID-deduplicated set of
   file/archive locators so prior mined data remains in later rounds without
   copying payloads. For multi-node launches, the TAO refinement adapter seals
   one resolved input spec and gives every launcher a private temporary copy;
   explicit result and resource overrides bind all ranks to the same output.
5. **Cache and outputs**: name any platform dataset cache that must be created.
   The controller reuses images and C-RADIO embeddings read-only and stores only
   scores, selections, neighbors, manifests, checkpoints, metrics, logs, state,
   and reports under `run_dir`; it creates no image copies or symlinks.
6. **Monitoring**: state the polling cadence and promise updates at stage/round
   transitions with mining yield, task metrics, retries/failures, and terminal reason.

For multi-task weighting, `targets_per_round` fixes cost. Positive
`task_weights` divide that total with deterministic largest-remainder rounding;
unspecified tasks have weight 1.0 and every configured task receives at least
one target. `targets_per_task` remains accepted only for older configurations.
With `multi_task.policy: balanced`, an inactive task's quota is left unused
instead of being donated to another task. The unique cumulative manifest keeps
lineage and deduplication intact, while a derived training view repeats minority
task-provenance groups to equal exposure without copying image payloads.

Optional `training.node_scaling` graduates through an explicit `allowed_nodes`
list. For each cumulative manifest, the controller reads the base spec's batch
size and selects the largest allocation that preserves at least
`target_optimizer_updates`. The resolved nodes, GPUs, world size, and expected
updates are committed before launching the training leaf; round number alone
never changes resources.

Every action has the same portable execution choice. With
`execution_mode: runner`, the configured platform runner owns the requested
resources. With `execution_mode: adapter_managed`, the runner launches one
durable wrapper and the adapter owns the resources declared in `resources`.
The workflow records both allocations and never requests the delegated
resources a second time. `wrapper_resources` must describe one node.

## Monitored Runs

When the user asks to monitor, keep the interaction attached until the workflow
is terminal. Poll at the requested cadence; otherwise use roughly 5 minutes for
pending/short stages and 20-30 minutes for long search or training stages. Send
an update on every stage or round transition, meaningful mining/KPI result,
retry, failure, and final stop. Do not treat successful submission as completion
and do not silently detach after a fixed elapsed time.

## Scientific Contract

- GRIT is a within-domain ordinal instability rank. Do not call it calibrated
  uncertainty, generic representation quality, or a guaranteed KPI driver.
- The frozen outcome study used ViT-B. Relative-depth probes make the action
  operational on every TAO DINOv3 variant, but do not constitute independent
  KPI validation of GRIT on L, H+, or 7B.
- Multi-task acquisition inputs must be disjoint from the sealed benchmark by
  acquisition unit. The benchmark reports outcomes and never generates targets.
- Diagnostic replay may reuse acquisition inputs, but it must be labeled
  `diagnostic_replay` and never presented as held-out evidence.
- Fixed C-RADIO embeddings determine source relevance for both strategies.
- The default training policy is `base_checkpoint_each_round`: scoring follows
  the latest candidate, but training never inherits its weights. Continual
  adaptation requires the explicit `previous_round_checkpoint` policy.
- Distributed training must not launch multiple TAO entrypoints against one
  mutable experiment YAML. The platform runner supplies one launcher per node,
  an exact TAO node-level rendezvous, and a fresh gang-attempt launch ID. Any
  failure cancels and restarts the full static gang; partial node retry is
  invalid. The packaged adapter verifies shared inputs, prepares on node rank
  zero, launches private spec copies, and finalizes on rank zero; any allocation
  mismatch is terminal.
- Runtime placement, code transport, cache placement, and cleanup are platform
  adapter responsibilities. The generic controller only requires immutable
  inputs, durable committed outputs, and the declared stage-result contract.
- Persistent target suppression yields `no_actionable_targets`; it is not corpus
  exhaustion.
- Optional evaluation early stopping uses a weighted, direction-normalized mean
  of explicitly named evaluator metrics. It stops after the configured patience
  without meaningful improvement and delivers the best evaluated checkpoint.
  It cannot be enabled when evaluation is absent, and diagnostic replay remains
  diagnostic rather than held-out evidence.
- Only exact search over every declared shard may produce `radius_exhausted`.
  ANN depth limits must use `search_budget_exhausted`.
- Large-pool ANN retrieval and exact reranking are separate committed stages.
  Resume from saved candidates after a rerank failure; never silently rebuild
  or substitute unaudited probe/depth settings.

## Interfaces

For TAO Docker/Slurm deployment, read `references/tao-container-integration.md`.
The minimal profile is one allocated DS container: DS already includes TAO
PyTorch/Core and the data-stack dependencies. With `execution.backend: local`,
leaf commands run inside that allocation; it does not launch nested containers.
Use the existing external runner for multiple nodes or separate-image jobs.
The selected image must contain the new DEFT modules: do not install or patch
packages at startup to hide a release mismatch.

Conversational use generates the same YAML consumed directly by:

```bash
python -m nvidia_tao_ds.mining.dinov3.workflow validate run.yaml
python -m nvidia_tao_ds.mining.dinov3.workflow plan run.yaml
python -m nvidia_tao_ds.mining.dinov3.workflow adopt-training continuation.yaml
python -m nvidia_tao_ds.mining.dinov3.workflow run run.yaml
```

Python integrations import `WorkflowConfig` and `RefinementWorkflow` from
`nvidia_tao_ds.mining.dinov3.workflow`. Skill Bank is not a runtime dependency.
All interfaces produce the same run artifacts.

Customer heads, task losses, sealed evaluation, and large-scale indexed search
plug in as leaf commands. Read `references/adapter-contracts.md`; do not put
customer model logic into the controller or the conversational layer.

## Examples

```bash
python -m nvidia_tao_ds.mining.dinov3.workflow init --recipe grit-score --output run.yaml
python -m nvidia_tao_ds.mining.dinov3.workflow init --recipe multi-task-round-robin --output run.yaml
```

## Related Skills

- `tao-launch-workflow`
- `tao-run-on-docker`
- `tao-run-on-slurm`
- `tao-run-on-kubernetes`
- `tao-data-io`
