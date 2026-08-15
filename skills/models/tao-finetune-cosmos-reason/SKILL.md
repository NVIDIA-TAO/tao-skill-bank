---
name: tao-finetune-cosmos-reason
description: >-
  Shared Cosmos3 frontend that explicitly routes Cosmos Framework and
  Cosmos-RL, validates runtime model/video-dataset/SLURM inputs, builds clean
  repository-derived images, prepares checkpoints, gates full training on a
  smoke run, and returns token-weighted losses and task-aware accuracy.
license: Apache-2.0
compatibility: Docker with NVIDIA Container Toolkit, or SLURM with Pyxis/Enroot and a user-supplied shared-storage configuration.
metadata:
  author: NVIDIA Corporation
  version: "0.3.3"
allowed-tools: Read Bash
tags: [cosmos, vlm, sft, peft, video, reasoning, slurm]
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
- `base_model_format`; Nano URI inputs use `qwen3_vl` or `cosmos3_omni`.
  Cosmos3-Edge is inferred as `cosmos3_edge` from the resolved model ID.
- optional `prepared_checkpoint_path`; validate it instead of silently
  replacing it.
- training/validation annotation paths and media roots for conversation-style
  or task-aware video supervision, plus optional task selection.
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

### Public Cosmos3-Edge checkpoint contract

For Cosmos3-Edge, accept the public Hugging Face model ID plus an immutable
revision, or a complete local snapshot of that same public model. Do not ask
the user for a second workload-specific checkpoint and do not copy or edit
`processor_config.json` or `video_preprocessor_config.json` to create one.
Keep the base-model fingerprint separate from the processor-profile
fingerprint.

The model-aware TAO profile supplies Edge runtime defaults: 6 sampled video
frames, a 1280 x 720 per-frame reference budget (5,529,600 aggregate video
pixels), sequence length 16,000, and `flash_attention_2`. These are runtime
settings, not checkpoint contents. Record whether each value came from the
skill profile or an explicit user override, include it in parity and cache
keys, and require the normal compute-node smoke gate before full training.
Framework receives the pixel budget through `TAO_VIDEO_MAX_PIXELS`; Nano keeps its native processor limit unless the user
explicitly overrides it.

## Backend selection

Run `scripts/cosmos_workflow.py resolve` first.

| Request | Automatic selection |
|---|---|
| Cosmos3-Nano plain train | Cosmos-RL (compatibility default) |
| Cosmos3-Nano AutoML/HPO | Cosmos-RL |
| Nano Framework-DCP export | Cosmos Framework |
| Nano evaluate/inference/microservice with no explicit backend | Cosmos-RL |
| Nano quantize | Cosmos-RL |
| Cosmos3-Edge train/export/evaluate/inference/microservice | Cosmos Framework |

An explicit supported backend wins, so users can select Cosmos Framework for
Nano training without changing model ownership. Comparative runs reject
`auto`, so both sides of an experiment are deliberately forced.
Framework-trained checkpoints use the native exact-key exporter, then the
repository-backed TAO evaluation adapter. That does not make Framework a
Cosmos-RL version.

## Evaluation intake and inheritance

For every `evaluate` action, run `scripts/evaluation_workflow.py` before
materializing action TOML. Never submit
`references/spec_template_evaluate.yaml` directly; it is a dataset-neutral
shape template with intentionally unresolved semantic fields.

When evaluating a selected fine-tuning job, give the helper that job's sealed
training plan and structured terminal status. It automatically inherits the
exact validation annotation/media paths and fingerprint, system prompt
(including recorded empty), frame count, pixel budget, precision, seed,
validation batch size, task/answer/metric profile, backend, training mode,
prepared base model, and GPU count. It records field-level provenance and
blocks a checksum-invalid plan. Do not ask the user to repeat an inherited
value and do not replace it with a template value.

The helper writes all genuinely missing fields to `required_user_inputs` in
one pass. Ask once for only those entries: normally the new results directory,
an exact checkpoint/epoch when checkpoint events are ambiguous, generation
maximum tokens when the training plan did not record an evaluation contract,
or task/metric semantics that annotation inspection could not prove. If the
user selects a different evaluation corpus, also require its exact annotation
and media paths and re-run structural/fingerprint inspection; do not carry the
old dataset's prompt or scorer semantics into it.

Entries in `automated_actions` are not questions for the user. In particular,
the skill owns Framework DCP export/verification and deterministic recovery of
a Cosmos-RL PEFT base model from training provenance. It also owns
deterministic full-coverage materialization when the sealed validation split
contains multiple manifests or media roots; never ask the user to choose a
subset. Rerun the helper with the
Framework pre-action's verified `action_model_path`; only a `ready=true` plan
may write or launch the evaluation TOML. Preserve the plan/config SHA256 values
in the evaluation job record. See `references/cosmos-reason-evaluate.md` for
the complete action flow.

## Framework checkpoint pre-action

Before every Framework `evaluate`, `inference`, or `inference_microservice`
action, invoke `scripts/framework_checkpoint_action.py plan`. Do not ask the
user to find or run an exporter manually. The helper preserves the supplied
checkpoint path, detects a complete HF safetensors directory versus a native
Framework DCP, infers the saved Framework config only from the checkpoint's
standard run layout, and otherwise requires an explicit `config_file`.

For DCP input, run the helper's `prepare` verb in the newly built Framework
TAO action image before starting the requested action. Its runner may be local
Docker or an `srun` Pyxis command supplied through `--command-prefix`. On
SLURM, stage this checked-in helper and `cosmos_common.py` with checksums in the
job input directory; mount only that job directory, not a source checkout.
The helper invokes the repository-owned exact-key exporter, verifies the DCP
metadata/config/base-model/revision/exported weights and manifest, writes
`.tao_export_complete`, and returns `action_model_path`. Put that verified path
into `model.model_name` for evaluation or `model_path` for inference and the
microservice. Run `verify` once more from the target compute frame before the
action child starts.

An already verified export is reused without conversion. A stale or partial
export is never silently accepted: `prepare` creates and verifies a sibling
temporary export, preserves the invalid directory under an `.invalid-*` name,
then atomically installs the replacement. Capture the pre-action JSON, child
exit code, export manifest checksum, source checkpoint/config fingerprints,
and final action child code in the job metadata. A failed export blocks the
evaluate or inference allocation/action and emits terminal `FAILURE`.

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
   When SLURM storage is not mounted on the launch host, let
   `cosmos_workflow.py` stream its checked-in `cosmos_common.py` inspector to a
   login host over SSH. It runs from stdin, preserves remote `realpath` values,
   and creates no remote script or source overlay. Do not require local Lustre,
   `sbatch`, or `srun` on an SSH-based launch host. Run this expensive input
   inspection exactly once with the `plan` verb and pass a local
   `--plan-artifact <path>` so the resolved request and inspection results are
   sealed for the remaining launch verbs.
7. Prepare the decoder input selected by the structural dataset contract.
   Cosmos-RL defaults to direct on-demand sample processing so training starts
   without a dataset-cache prewarm phase. Conversation-style runs prewarm only
   when the user explicitly selects `--rl-dataset-cache-mode prewarm`; that
   opt-in uses separate deterministic train and validation keys.
   `--rl-video-profile auto` selects `pynv-device-rgbp` for
   `video_conversation` and `system-pyav` for
   `task_aware_video_reasoning`; record the resolved profile and rationale.
   The fast profile uses the source-baked PyNvVideoCodec device-RGBP/DLPack
   path, one spawned DataLoader worker, prefetch two, and four order-preserving
   in-process batch threads. Its two capacities are derived from the larger
   inspected split's unique-media count unless explicitly supplied. The video
   LRU stores processed `fetch_video` outputs in rank-local memory and the
   decoder cache stores rank-local native sessions; both populate during
   ordinary training, persist no video files, and require no prewarm.
   The explicit `system-pyav` profile remains the sparse software route for
   codec-policy-constrained runs. Its packaged reader must register in the
   controller and every spawned worker; the image installs the checksum-pinned
   official PyAV wheel and proves that generic `h264` and `hevc` resolve to
   software decoders. A source-built wheel resolving those names to `*_cuvid`
   is an image defect. Never combine the two runtime profiles or describe the
   PyNv route as satisfying a software-codec policy without a separate review.
   The JSON plan emits exact `decoder_artifact.preparation_command` and
   `decoder_artifact.validation_command` values for the selected clean image;
   re-plan once with their map, manifest, and artifact fingerprint outputs,
   seal that plan, and never reuse another run's cache or override artifact.
8. Generate backend-native TOML, environment, topology, preflight commands,
   parity data, resolved video-runtime profile, and machine-readable job
   metadata. Full specs must contain no
   sample limit. `plan` and `preflight` are read-only with respect to the
   target compute frame; the explicitly requested controller-side plan artifact
   is their handoff. Invoke `preflight`, post-review `materialize`, and
   `render-slurm` with that same `--plan-artifact` and no repeated original
   input arguments. `materialize` atomically creates the TOML and any
   merged/smoke manifest in the verified compute frame. For SSH-based
   SLURM, the checked-in helper is streamed to the verified login host and the
   generated files are written directly to user-supplied shared storage; do
   not read a remote annotation through the launch host or copy a temporary
   source patch to the cluster. The planner derives all in-container runtime
   paths from explicit mount mappings and rejects an explicit mapping mismatch.
9. Convert the newly built image to a new SQSH when SLURM is selected. Record
   image ID/digest and SQSH SHA256; verify Pyxis/Enroot, mounts, non-root Python,
   packages, decoder, GPU memory/type, CUDA/PyTorch, NCCL, and storage on the
   allocated node.
10. Run a smoke job for every distinct backend × structural dataset family × training
    mode × checkpoint/evaluator path. Continue only on child exit zero,
    structured `SUCCESS`, finite global train/validation loss, checkpoint
    completion, and evaluator accuracy coverage.
11. Create one fresh sealed full plan after the smoke gate, with all smoke
    limits removed. Materialize its full spec once and verify its SHA256 in the
    compute frame before rendering the job from the same plan artifact. Launch with
    `afterok` only after the smoke gate, monitor scheduler and structured TAO
    state to a terminal result, and preserve the child exit code independently
    of scheduler state.
12. Resolve evaluation with `scripts/evaluation_workflow.py`. Inherit exact
    fine-tuning artifacts, collect only its remaining user inputs, run its
    backend-owned automated checkpoint pre-actions, and require `ready=true`.
    Evaluate the selected checkpoint with identical prompt, preprocessing,
    generation, normalization, and task scoring. Extract final metrics with
    `scripts/extract_cosmos_metrics.py`.

## Dataset contracts

Resolve datasets by structure, not by project, benchmark, directory, or file
name. The supported families are:

- `video_conversation`: a JSON array with media and at least two ShareGPT,
  LLaVA, or OpenAI-style conversation turns;
- `task_aware_video_reasoning`: one or more item-envelope or array annotation
  files with media, task identity, and conversation/response targets.

Default `dataset_family` to `auto`, inspect every annotation, and require train
and validation to resolve to the same family. Capture record count, unique
media count, media reuse, extensions, byte-size distribution, task/metric
metadata, and any declared width, height, FPS, and duration. Select processor,
cache, smoke-size, and resource profiles from those characteristics and the
model tier. Never branch on a customer dataset name.

Tasks declaring accuracy participate in deterministic accuracy; common binary
and multiple-choice task types are recognized. Generative tasks report their
declared metrics and are excluded from aggregate accuracy with a reason.
Aggregate accuracy is example-weighted over records with an accuracy definition.

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

Use `references/cosmos-reproducibility-gates.md` as the source-owner and test
map before proposing a workaround in a fresh session.
