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

### Single-image Defect Detection ablation

Enable this launch-recorded policy with `init_deft_state.py
--defect-detection-ablation`. It requires `task_strict` routing and forbids
Component Count replay. `analyze_gaps.py` annotates only single-image `Defect
Detection` proxy failures with box-level evidence: false negatives,
best-overlap predictions with `0 < IoU <= 0.5`, and false positives. The
packaged KPI evaluator and its `f1_cohort_balanced_v1` contract are unchanged;
`task_balanced_v1` remains only a state-recorded alias.

After task-strict top-K routing, materialize with
`$PYTHON scripts/defect_detection_ablation.py`. Pass the routed candidate
parquet, canonical Mining/Proxy JSONL, both Proxy and Benchmark as validation
inputs, the canonical media root, the launch global batch as `--row-multiple`,
the launch epochs/global batch, and the launch-recorded minimum accepted rows.
The selector:

- can augment deficit-weighted selected gaps at launch time through
  `route_selected_gaps.py --defect-detection-supplement GAP_CANDIDATES
  --supplement-summary SUMMARY --defect-detection-anchor-policy POLICY`;
  `hard_only` preserves the historical approved-evidence supplement, while
  `all_proxy_severity` replaces the selected DD subset with every Proxy DD row
  ordered as FN/partial-overlap, FP, then correctly handled rows. Correct rows
  carry explicit `proxy_correct` provenance so their candidates remain
  auditable and trainable;

- applies the launch-recorded default top-K to the five maintenance tasks and
  an independently launch-recorded Defect Detection top-K override;

- reserves the configured fraction (at least 50 percent) for exactly `Defect
  Detection`; `Component Detection` is one of five maintenance tasks and never
  counts toward that reserve;
- admits positive Defect Detection rows from proxy-FN, partial-overlap, or
  correctly handled DD routes, while proxy-FP routes provide empty hard
  negatives; when the launch
  explicitly authorizes direct Mining-pool box calibration, separately labeled
  `calibration_empty_ground_truth` candidates may fill only the residual empty
  quota and are never reported as task-strict hard negatives;
- balances positive marginal quotas over source, defect phenotype,
  source-by-phenotype, 1024-canvas box-area quartile, local-contrast quartile,
  and GT-count bins `1`, `2-3`, `4+`, reporting capacity shortages rather than
  filling them with another task;
- matches the Proxy single-image Defect Detection empty-GT rate after integer
  rounding; and
- treats each ordered `(golden, target)` reference sample as one atomic unit
  across embedding, retrieval, de-duplication, history, leakage exclusion, and
  canonical two-image materialization; and
- rejects atomic-sample, content-SHA, perceptual near-duplicate, Proxy, and
  Benchmark collisions while leaving every selected source record and its
  `official_v1` messages, coordinates, box order, and image controls unchanged.

The command writes the materialized JSONL and a bound
`defect_detection_quota_manifest_v1`. By default a shortfall leaves the
manifest on disk with `verified=false` and exits nonzero. When
`--minimum-rows` is launch-recorded, the selector instead accepts the largest
feasible global-batch-aligned corpus at or above that raw minimum, records the
requested/accepted shortfall and every quota shortage, and rejects anything
smaller. For an enabled ablation,
`commit_stage.py ... --stage assemble_data` requires `--quota-manifest`, and
`render_cfw_sft.py` must receive both `--quota-manifest` and
`--require-defect-detection-quota-manifest`; each gate re-hashes the JSONL and
recomputes the optimizer schedule. Never launch Train unless both gates pass.

### Repetition blend

Training materialization can apply a launch-recorded repetition blend after
exact-duplicate and Proxy/Benchmark leakage exclusion. Configure it in TOML or
JSON, or map a launch prompt to `--repetition-blend`, `--repetition-policy`,
`--repetition-rep-min`, `--repetition-rep-max`,
`--repetition-never-repeat-empty-gt`, and repeatable
`--repetition-explicit-multiplier TASK=MULTIPLIER` arguments; the existing
maximum-training-row cap is its `row_cap`. The default
`deficit_proportional` policy normalizes task deficits from the current
`gaps_summary.json` (equal weights when unavailable), while `explicit` uses
the supplied task multipliers. A launch prompt may request
`deficit-proportional repetition (max rep N)` or explicit multipliers. The
fixed-seed fractional sampler repeats only accepted rows, never reapplies a
perceptual-hash filter, keeps empty-ground-truth calibration rows at one
occurrence by default, and writes a bound `repetition_blend_manifest.json`
alongside the compatible `defect_detection_quota_manifest_v1`.

## Train contract

Render nested TOML with `$PYTHON scripts/render_cfw_sft.py`, passing the
canonical workspace root as `--media-root`, and plan with
`$PYTHON scripts/cfw_action_plan.py`. Before rendering a full profile, run a
short platform probe and pass its largest stable `--micro-batch-per-rank`.
The full profile contract is:

- experiment `nvpaw_omni_vlm_sft`, BF16, full parameters;
- 8 GPUs on one node, FSDP shard 8 / replicate 1;
- the probed micro-batch per rank plus launch-recorded gradient accumulation,
  minimum-global-batch, and learning-rate policy; the legacy defaults remain
  accumulation 16, a global-batch floor of 512, and linear scaling from
  `1e-6`, while an operator-reviewed launch may explicitly disable the floor
  and use a fixed recipe LR;
- fused AdamW, weight decay `0.05`, betas `0.9/0.999`, merger multiplier 20;
- vision encoder frozen; projector and language model trainable;
- full activation checkpointing;
- native `num_epochs` / `steps_per_epoch` scheduling, with the requested epoch
  count per DEFT iteration; materialized rows must be an exact global-batch
  multiple so epochs end on optimizer-update boundaries;
- one synchronous checkpoint at the final epoch, an LR cycle spanning the
  epoch-derived update count, warmup at most 5 updates, `f_start=.05`,
  `f_max=1`, `f_min=.1`;
- synchronous DCP.

Image augmentation is launch-configurable as `exp40_photometric` or `off`.
The `off` profile sets the augmentation switch and every photometric/geometric
probability or magnitude to zero and is recorded in state and the rendered
descriptor.

The smoke profile must be explicitly named. It may reduce rows, updates,
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

Evaluation defaults to `[evaluation] row_order = "task_length_sorted"`, which
orders canonical rows by task type, prompt image count, prompt text length, and
id before applying stride sharding; use `row_order = "source"` or the runtime
`--row-order=source` override only when source-order replay is required. The
per-rank and merged evaluation summaries record both the selected order and its
sort key, while merge-by-id restores canonical source order and exact coverage.
On the recorded 20,657-row v3b iteration-5 benchmark this default reduced wall
time from 58:20 to 38:56 (-33%), with every cohort F1 delta at most 0.30 and
therefore within evaluator noise.

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

Every rendered launch descriptor also carries the required OneLogger runtime
bootstrap. Cosmos Framework is standalone rather than NeMo-descendant, so the
platform installs `one-logger-utils` from the internal MLWFO PyPI index under
the `SLURM_LOCALID == 0` guard while nonzero ranks wait 30 seconds. When the
container does not inherit the submitter's home, mount the existing host
`.netrc` at `/root/.netrc`; never export `WANDB_API_KEY`. Smoke/probe launches
set `ONE_LOGGER_JOB_CATEGORY=test`; real DEFT launches leave that variable
unset.

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
