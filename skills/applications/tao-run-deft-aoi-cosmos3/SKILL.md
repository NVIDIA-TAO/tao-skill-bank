---
name: tao-run-deft-aoi-cosmos3
description: >
  Run the disk-backed NVIDIA DEFT AOI improvement loop for Cosmos3 Nano with
  Cosmos Framework: evaluate canonical NVPAW JSONL, use Proxy errors for RCCA
  and task-aware real-image mining, assemble monotonic real-only training
  JSONL, full-parameter BF16 fine-tune to synchronous Framework DCP, and gate
  only on the frozen Benchmark using the recorded exact five-component F1
  evaluator. Use for "tao-deft-aoi", "run DEFT AOI", or "improve Cosmos3
  NVPAW AOI". Do not use for one-off generic training.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Requires the TAO skill bank, Python with numpy/pyarrow/yaml, Cosmos Framework image, data-services image, and one selected platform native CLI.
metadata:
  author: NVIDIA Corporation
  version: "0.2.0"
allowed-tools: Read Task Bash Write
tags: [application, workflow, deft, aoi, cosmos-framework]
---

# DEFT AOI with Cosmos Framework

The user-facing shorthand `tao-deft-aoi` resolves to this canonical skill,
`tao-run-deft-aoi-cosmos3`. This application owns orchestration. The model
implementation is `tao-finetune-cosmos-reason` with
`workload=deft-aoi`, which must resolve train, evaluate, and inference to the
`cosmos-framework` backend.

## Immutable workflow contract

The loop is:

```text
Benchmark evaluate -> exact F1 gate
  pass: loop_stop
  fail: Proxy evaluate -> RCCA -> routing -> data_mining -> assemble_data
        -> validate_data -> CFW train -> next Benchmark evaluate
```

Only real records selected from the six supported classification/detection
families in `annotations/mining.jsonl` enter training. Canonical Mining-only
count/segmentation rows are ignored with auditable counts; they are never
converted, copied, or admitted to Train. Proxy and Benchmark remain strict.
Every later iteration retains the preceding training JSONL and adds at least
one current Mining record. Proxy and Benchmark targets are excluded. The
Benchmark file and metric contract are SHA-256 sealed at initialization.

## Required workspace

Use `/home/seanlin/projects/deft/workspace` unless the user explicitly selects
another workspace with the same contract:

```text
annotations/mining.jsonl
annotations/benchmark.jsonl
annotations/proxy_kpi.jsonl
eval/calculate_f1_metrics.py
models/Cosmos3-Nano-VLM/
specs/train_spec.toml
specs/evaluate_spec_proxy.toml
specs/evaluate_spec_benchmark.toml
results/<run>/
```

Every JSONL row has a unique `id`, supported `task_type`, native `messages`,
one or two ordered image parts, and integer `min_pixels`/`max_pixels`. A
reference-based row orders images as golden then target. Never convert these
files to a JSON array at the runtime boundary.

## Preflight and approval

1. Resolve the model with `$PYTHON scripts/resolve_tao_model.py --model
   nvidia/Cosmos3-Nano --action <action> --workload deft-aoi`. Show the chosen
   backend and rationale.
2. This application supports multiple platforms. Ask once among supported,
   installed peers; never choose one by default. Read the selected platform's
   `SKILL.md` and run its Preflight.
3. Read `references/preflight.md`. Validate annotations, model snapshot,
   evaluator path/hash, spec paths, Python dependencies, writable results, and
   image keys. Resolve `images.tao_toolkit.cosmos_framework` and
   `images.tao_toolkit.data_services` from `versions.yaml`; the launch plan
   records an immutable digest.
4. Invoke `tao-launch-workflow` before any side-effecting action. Show one
   launch review containing concrete nested config, image digests, mounts,
   resources, outputs, and credential names. Wait for explicit approval.
5. Never ask for credential values. Never pull, download, log in, submit, or
   launch before that approval.

## State and resume

Initialize state once with `$PYTHON scripts/init_deft_state.py`; never overwrite or
hand-edit it. State schema version 7 records the Framework backend, immutable
image references/digests, exact recipe, annotation/evaluator hashes, DCP manifests,
prediction JSONL, raw exact-evaluator reports, and committed events. An older
backend or stage schema cannot resume; initialize a new run.

Before each stage or after context compaction, run `$PYTHON scripts/deft_context.py`
against `${RESULTS_DIR}/deft_state.json`. Its `next_stage` is authoritative.
Commit a successful stage with `$PYTHON scripts/commit_stage.py` only after all
required artifacts validate. Durations must be measured positive seconds.

## Train contract

Render nested TOML with `$PYTHON scripts/render_cfw_sft.py`, passing the
canonical workspace root as `--media-root`, and plan with
`$PYTHON scripts/cfw_action_plan.py`. Full profile is fixed:

- experiment `nvpaw_omni_vlm_sft`, BF16, full parameters;
- 8 GPUs on one node, FSDP shard 8 / replicate 1;
- micro-batch 4 per rank, gradient accumulation 16, global batch 512;
- fused AdamW, LR `1e-6`, weight decay `0.05`, betas `0.9/0.999`, merger
  multiplier 20;
- vision encoder frozen; projector and language model trainable;
- full activation checkpointing;
- 500 iterations, save every 100, cycle 500, warmup 5,
  `f_start=.05`, `f_max=1`, `f_min=.1`;
- synchronous DCP.

The smoke profile must be explicitly named. It may reduce rows, iterations,
checkpoint interval, and GPU count, but does not change precision,
full-parameter tuning, freeze policy, direct JSONL semantics, or DCP format.

The packaged `scripts/nvpaw_cfw` adapter seals JSONL path, row count, SHA-256,
image-item count and pixel ranges; preserves multi-image order; masks loss to
assistant tokens; deterministically shuffles/resumes; and deterministically
resamples over-context rows. The trainer writes `iter_#########` synchronous
DCP. Validate it with `$PYTHON scripts/cfw_dcp.py` before commit.

## Evaluate, inference, and KPI

Render multi-task evaluation config with
`$PYTHON scripts/render_cfw_evaluate.py`, passing the canonical workspace root
as `--media-root`. Plan evaluation and single-media inference through
`cfw_action_plan.py`. Both execute the packaged `cfw_jsonl_runtime.py` inside
the Cosmos Framework image so the canonical JSONL is streamed directly and
one/two-image message order plus pixel bounds are preserved. Both paths use
BF16, the same 1024-token generation budget, and the same preprocessing and
checkpoint handoff. For a trained DCP, the model skill's
`framework_checkpoint_action.py prepare` must first create a verified
exact-key action model; the runtime command consumes that exported directory,
never the DCP path itself. Output is atomically normalized to `id`,
`task_type`, ordered source prompt `message`, `GT`, and `raw_prediction`.
`cfw_predictions.py` remains the standalone strict coverage validator for
externally produced/sharded Framework rows.

`$PYTHON scripts/exact_f1_adapter.py` invokes the recorded absolute
`eval/calculate_f1_metrics.py`, preserves its raw JSON report, binds the
committed report by absolute path and SHA-256, verifies the evaluator SHA-256,
and extracts exactly:

- `non_reference_based.tasks.BCQ.macro_f1`
- `non_reference_based.tasks.MCQ.macro_f1`
- `non_reference_based.tasks.DET.f1`
- `reference_based.tasks.BCQ.macro_f1`
- `reference_based.tasks.DET.f1`

All five must meet the frozen component threshold, and missing or unknown
prediction counts must be zero. The app never recalculates F1. Only a frozen
Benchmark metric result may stop the loop; Proxy results drive RCCA/mining.

## Platform execution

Application renderers emit only platform-neutral image, command, config,
mount, resource, pre-action, and output descriptors. The selected platform
owns native submission. Every GPU stage uses the four verbs
`submit`/`status`/`logs`/`cancel`, with the job-record opened before launch.
Monitor the backend and map states to `PENDING RUNNING COMPLETE ERROR CANCELED
UNKNOWN`.

## Completion

A run is complete only after a successful `loop_stop` commit, terminal
`deft_state.json`, a final Benchmark metric result, and a rendered
`DEFT_Loop_Report.html`. A prepared plan, submitted job, checkpoint alone, or
intermediate Mining artifact is not completion.

Read the focused references as needed:

- `references/preflight.md`
- `references/pipeline-and-state.md`
- `references/cosmos-reason.md`
- `references/aoi-annotation.md`
- `references/metric-contract.md`
- `references/scripts-and-agents.md`
