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

The loop is parameterized by `benchmark_cadence = every | final_and_best`:

```text
reusable zero-shot Benchmark evaluate -> exact F1 gate -> Proxy KPI + RCCA
  -> routing -> data_mining -> assemble_data -> validate_data -> CFW train
  -> Proxy KPI + RCCA
       final_and_best: Benchmark only for a Proxy-best or final checkpoint
       every: Benchmark for every checkpoint
  -> pass: loop_stop; otherwise route the next iteration
```

Only real records selected from the six supported classification/detection
families in `annotations/mining.jsonl` enter training. Canonical Mining-only
count/segmentation rows are ignored with auditable counts; they are never
converted, copied, or admitted to Train. Proxy and Benchmark remain strict.
Every later iteration retains the preceding training JSONL and adds at least
one current Mining record. Proxy and Benchmark targets are excluded. The
Benchmark file and metric contract are SHA-256 sealed at initialization.
Zero-shot predictions may be reused only when their recorded Benchmark,
model, image, evaluator, and prediction checksums all match.

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
  counts toward that reserve. The fraction is a lower bound, not an exact
  split: for a candidate total `T`, take at least `ceil(T * fraction)` DD rows
  and increase DD as needed when the available maintenance rows cannot fill
  the remainder;
- admits positive Defect Detection rows from proxy-FN, partial-overlap, or
  correctly handled DD routes, while proxy-FP routes provide empty hard
  negatives; when the launch explicitly authorizes direct Mining-pool box
  calibration, separately labeled `calibration_empty_ground_truth` and
  `calibration_few_box_ground_truth` candidates may fill the Proxy-bound
  detection quotas and are never reported as task-strict hard negatives;
- balances positive marginal quotas over source, defect phenotype,
  source-by-phenotype, 1024-canvas box-area quartile, local-contrast quartile,
  and GT-count bins `1`, `2-3`, `4+`, reporting capacity shortages rather than
  filling them with another task;
- derives and matches the Proxy empty-GT rate independently for single-image
  and reference Defect Detection after deterministic integer rounding;
- labels empty reference pairs as
  `calibration_reference_no_change_ground_truth` negatives and keeps the
  ordered golden/target pair atomic;
- applies the 512 empty, 512 few-box, and 500 reference calibration limits to
  current additions only. Pass the previous cumulative Train as
  `--previous-jsonl`; exact prior rows are excluded from the selector because
  the assembler retains them independently;
- treats each ordered `(golden, target)` reference sample as one atomic unit
  across embedding, retrieval, de-duplication, history, leakage exclusion, and
  canonical two-image materialization; and
- rejects atomic-sample, content-SHA, Proxy, and Benchmark collisions while
  leaving every selected source record and its `official_v1` messages,
  coordinates, box order, and image controls unchanged. Perceptual
  near-duplicate filtering is applied only when configured; use
  `--no-near-duplicate-filter` when the launch disables it.

The selector writes only the current `data/mined.jsonl` and its bound
`defect_detection_quota_manifest_v1`; it never writes
`assemble_data/train.jsonl`. Generate the two-command handoff with
`render_iteration_mining_runner.py`: the second command is the sole assembly
boundary, passes the previous Train path and SHA-256 to
`assemble_training_json.py`, and emits the cumulative Train plus a final
`defect_detection_quota_manifest_v2`. By default a shortfall leaves the
current manifest on disk with `verified=false` and exits nonzero. When
`--minimum-rows` is launch-recorded, the selector instead accepts the largest
feasible global-batch-aligned corpus at or above that raw minimum, records the
requested/accepted shortfall and every quota shortage, and rejects anything
smaller. Feasibility uses `dd_take = max(ceil(T * fraction), T -
available_maintenance)` and therefore permits more than the minimum DD share;
an exact share would require a separate explicit policy. For an enabled ablation,
`commit_stage.py ... --stage assemble_data` requires `--quota-manifest`, and
`render_cfw_sft.py` must receive both `--quota-manifest` and
`--require-defect-detection-quota-manifest`; each gate re-hashes the JSONL and
recomputes the optimizer schedule. After iteration 1, both `data_mining` and
`assemble_data` commits also require the preceding cumulative Train lineage;
the assembly summary binds its path/hash/count and proves every preceding row
and fingerprint remains present. Never launch Train unless both gates pass.

Cached source routing and ShareGPT emission are path/atomic-ID lookups, never
pair-canvas rendering stages. Keep the recorded source pair-asset cache root
separate from the run's query output root; only embedding-input preparation
in `route_selected_gaps.py` and `build_mining_source_pool.py` should create
missing canvases for embedding. `render_iteration_mining_runner.py` can carry
both roots and render a runner with flushed start/end timings for each mining,
selector, and assembler stage; see
[the mining reference](references/tao-mine-aoi-images.md#task-aware-routing)
for the request fields. Generating that runner does not authorize its execution.

Reference-task mining records `[mining] pair_similarity` as `canvas` (the
compatibility default) or `two_vector`. Canvas mode embeds one deterministic
1024x512 golden/test composite but reduces each board's effective resolution;
two-vector mode instead embeds the golden and test boards independently at the
same full single-image resolution and with the same encoder used by ordinary
mining, reuses matching cached test-board embeddings, and embeds each shared
golden only once. It ranks only canonical atomic pairs by the `mean` (default)
or `min` of golden/test cosine similarity, records `sim_golden`, `sim_test`,
and `sim_pair` for every candidate plus same-board-type hit rates per reference
query, and never changes pair identity, de-duplication, leakage, budget, or
two-image materialization. A launch prompt may request `two_vector pair
similarity (mean|min)`.

### Repetition blend

After exact-duplicate and Proxy/Benchmark leakage exclusion, the launch-recorded
`deficit_proportional` repetition blend rebalances the task *mix*: its default
budget is the number of available unique rows (`budget_multiplier=1.0`), with
the existing `row_cap` only an upper bound, so abundant tasks may be seeded,
deterministically downsampled below `rep=1` while scarce tasks may be repeated
up to `rep_max`; unclamped tasks are redistributed by default and total
target-minus-realized share gaps above five percentage points are warned in
both manifests and stage summaries. Configure the budget, tolerance,
redistribution, repetition bounds,
empty-GT policy, seed, or explicit multipliers in TOML/JSON or their matching
`--repetition-*` arguments; empty-GT calibration rows may be downsampled but
are never repeated by default, and no perceptual-hash filter is reapplied. A
launch prompt may request `deficit-proportional repetition (rebalance task
shares, max rep N)`. In a cumulative iteration, retained copies from the
previous Train are mandatory lineage but do not inflate the unique-row budget.
`render_iteration_mining_runner.py` moves every repetition and gap-weight
control off the current-row selector and onto the sole cumulative assembler,
so the blend runs exactly once over previous plus new rows. The assembler
writes the schema-v2 bound `repetition_blend_manifest.json`, and the final
`defect_detection_quota_manifest_v2` carries that cumulative blend while
preserving the selector's disabled current-only blend under `current_selection`.

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
prediction counts must be zero. The app never recalculates F1. Proxy is scored
after every checkpoint and is the only gap-analysis input. Only a frozen
Benchmark metric result may stop the loop.

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
