---
name: tao-run-deft-aoi-cosmos3
description: >
  Run the disk-backed DEFT AOI improvement loop for NVIDIA Cosmos Reason 3 /
  Cosmos3 models, using Nano by default and Edge or Super when explicitly
  requested: evaluate the base model on Proxy and frozen Benchmark splits,
  mine real image pairs from Proxy gaps, assemble a per-iteration Train JSON
  from selected Mining samples, train with cosmos-rl LoRA SFT, and repeat
  through the selected platform's submit/status/logs/cancel contract.
  This migration supports bare labels only: the assistant response must be
  exactly OK or NG. Use for "run Cosmos3 DEFT AOI", "CR3 AOI loop", or
  "improve Cosmos3 PCB inspection with bare OK/NG"; do not use for
  rich/reasoning annotation, one-off Cosmos training, or generic anomaly
  generation.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Requires the companion TAO skill-bank skills from `eval.config`, host Python with `pyarrow` and `yaml`, and the selected platform's native CLI.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Task Bash Write
tags:
- application
- workflow
- deft
- aoi
- cosmos
---

# Skill: tao-run-deft-aoi-cosmos3

## Installation

Install this application together with the companion TAO skills listed in
`eval.config` so they resolve as `~/.claude/{models,data,platform,core}/...`.
Provision a host Python with `pyarrow` and `yaml`; run bundled validation with
`python -m unittest tests.test_cosmos3_bare` (pytest is optional, not required).

## Execution Contract

Treat a run as a disk-backed state machine.

1. Preserve every explicit user value and show the source of each effective
   value (`user`, `spec`, or `default`) in the Pre-Flight Summary.
2. Ask which installed platform to use. Do not select Docker, SLURM,
   Kubernetes, Brev, virtualenv, or an external platform by default.
3. Read and run the selected platform skill's Preflight before constructing
   launch commands. Stop on a missing system/native-CLI prerequisite. A small
   missing Python helper may be installed with `python -m pip install ...`,
   then Preflight must be rerun.
4. Before any mutation or launch, invoke `tao-launch-workflow` and show its
   launch review plus this skill's Pre-Flight Summary. Wait for one explicit
   approval.
5. After approval, initialize `${RESULTS_DIR}/deft_state.json` exactly once
   with `scripts/init_deft_state.py`. Never reinitialize a resumed run or edit
   `deft_state.json` / `loop_log.jsonl` by hand.
6. Before every stage, after context compaction, and before a completion claim,
   run:

   ```bash
   <skill_root>/scripts/deft_python.sh \
     <skill_root>/scripts/audit_deft_run.py --results-dir "${RESULTS_DIR}"
   ```

   Stop and repair the listed disk inconsistency when it prints
   `DEFT_RUN_STATUS=INVALID`. Read only the reported `read_before_action`
   reference before continuing.
7. Submit each GPU stage through the chosen platform's four verbs:
   `submit` / `status` / `logs` / `cancel`. The `submit` verb must open the
   job-record before native launch; the returned id is the only launch handle.
   Poll the backend, not the job-record, and map state to
   `PENDING RUNNING COMPLETE ERROR CANCELED UNKNOWN`.
8. Commit every completed or failed DEFT stage with
   `scripts/commit_stage.py`. It updates state and log atomically, then runs the
   audit and rolls back on inconsistency.
9. Claim completion only after this exits zero:

   ```bash
   <skill_root>/scripts/deft_python.sh \
     <skill_root>/scripts/audit_deft_run.py \
     --results-dir "${RESULTS_DIR}" --require-complete
   ```

Never place secrets in a spec, command, transcript, job-record, or chat. Check
credential presence only, for example
`[ -n "$HF_TOKEN" ] && echo SET || echo UNSET`. Read no credential file and
load no `.env`.

## Cosmos3 Model Contract

- Model skill: `tao-finetune-cosmos-reason`.
- Supported canonical base models:
  - `nvidia/Cosmos3-Nano` — default;
  - `nvidia/Cosmos3-Edge` — only when explicitly requested;
  - `nvidia/Cosmos3-Super` — only when explicitly requested.
- Normalize the user aliases `nano`, `edge`, and `super` to those canonical
  IDs. Preserve any variant selected in the prompt. When no variant is
  selected, use Nano.
- Give hardware recommendations for the selected variant and report when the
  available compute is insufficient. If the prompt asks for a variant
  recommendation based on hardware or workload, recommend one with the
  tradeoff, but require an explicit selection before state initialization.
  Never silently switch or fall back to another variant.
- Keep the selected canonical ID as source-model lineage, but do not pass the
  native online checkpoint directly to Cosmos-RL.
- The published Cosmos Reason 3 reasoners ship in Cosmos3's own native Omni
  format (`model_type="cosmos3_omni"`), which Cosmos-RL cannot load. After
  launch approval and before baseline evaluation, run the model skill's
  `scripts/prepare_cosmos3_vlm_checkpoint.py` to convert the selected reasoner
  into a Qwen3-VL safetensors PTM, or validate and reuse an existing prepared
  output.
- Use the prepared PTM consistently for zero-shot evaluation, Train
  `policy.model_name_or_path`, and LoRA `model.base_model_path`. The model
  being trained is still the selected Cosmos Reason 3 reasoner — keep its
  canonical ID as checkpoint lineage; the Qwen3-VL PTM is only the on-disk
  format Cosmos-RL consumes.
- Nano may use the helper's packaged Qwen3-VL default. Edge and Super require
  a variant-specific, validated VLM base; never reuse Nano's conversion
  arguments.
- Container key: `images.tao_toolkit.cosmos_rl` in `versions.yaml`.
- Train action: `cosmos-rl --config <spec.toml>
  /opt/cosmos_rl/tao_sft_example.py`.
- The pinned image caps vLLM evaluation at one image per prompt, which this
  two-image contract cannot satisfy. Run `scripts/patch_eval_image_cap.py`
  before the first evaluate job and mount its output read-only into every
  evaluation container; see `references/cosmos-reason.md`.
- Workflow override: `automl_policy: off`. DEFT owns iteration and checkpoint
  selection; this is a workflow argument, not a TOML key.
- Default adaptation: LoRA over the language-side projections
  `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
  "down_proj"]`, leaving the vision tower's pretrained weights untouched. The
  schema also accepts `"all-linear"`, which additionally adapts the vision
  linear layers; use it only when the user explicitly requests it. Derive all
  other Train defaults from the model skill's current template.
- Every spec is a nested dictionary serialized to TOML. Never write literal
  flat dotted keys into a spec.
- Do not mount user data over `/workspace`; cosmos-rl is installed there.
- Run every container that writes into the results tree as the invoking user,
  not root, or the run's own outputs become undeletable by their owner. See
  `references/cosmos-reason.md`.

Read `skills/models/tao-finetune-cosmos-reason/SKILL.md` and its
`references/skill_info.yaml` before authoring a spec. Start from the model
skill's current packaged template for the selected action and apply only the
AOI workflow overrides in `references/cosmos-reason.md`. Replace every
dataset/output path with the chosen platform's compute-frame path. Prove that
the selected Cosmos-RL image can load the prepared PTM and train the requested
variant; do not reuse Nano conversion, parallelism, or memory assumptions for
Edge or Super.

## Bare OK/NG Contract

This migration supports one annotation mode: `bare_okng`.

- Each record is ShareGPT JSON with exactly two images in
  `[AOI, golden_reference]` order.
- The first human/user turn contains the inspection prompt.
- The final assistant/gpt response is exactly `OK` or `NG`; reasoning,
  prefixes, explanations, and final-answer wrappers are invalid training
  labels.
- `NG` is the positive class. `NG -> OK` is a false accept; `OK -> NG` is a
  false reject.
- Evaluation may normalize a model response by its last standalone `OK`/`NG`
  token, but training labels remain exact.
- Rich, reasoning, BCQ/MCQ, and task fan-out modes are outside this migration.
  Stop instead of silently accepting them.

Run `scripts/validate_sharegpt.py` on Proxy, Benchmark, Mining, and each
generated iteration training file. There is no input Train annotation.
Run `scripts/validate_split_contract.py` to prove that Proxy, Benchmark, and
Mining targets are disjoint and that the frozen Benchmark annotation hash has
not changed. When a generated Train file is supplied, the same validator
requires its targets to come from Mining, the immediate `--previous-train`
seed, or the current iteration's `--synthetic` AnomalyGen output, and to remain
disjoint from Proxy and Benchmark. For iteration N>1, `--previous-train` is
required and the validator proves that every preceding Train record was
retained.

## KPI Isolation

- **Proxy:** `annotations/proxy_kpi.json`. It is the only error source
  for RCCA, routing, mining targets, and data-mixture decisions. It never stops
  the loop.
- **Benchmark:** `annotations/benchmark_kpi.json`. It is frozen, evaluated
  at baseline and every iteration, and is the only stop-gate source. Benchmark
  sample errors never feed routing or mining.
- Default gate: `recall_ng >= 1.0`. If the user asks for accuracy, use
  `accuracy >= <target>`.
- Unknown model responses block the gate through the
  `unknown_predictions <= 0` metric constraint.

`scripts/analyze_gaps.py` writes Proxy RCCA artifacts or Benchmark aggregate
metrics plus `metric_result.json`. `scripts/record_metric_result.py` binds the
Benchmark metric evidence to the configured metric contract.

## Workspace Contract

```text
workspace/
├── annotations/                 # user-supplied
│   ├── benchmark_kpi.json
│   ├── proxy_kpi.json
│   └── mining_pool.json
├── images/                      # user-supplied
└── specs/                       # produced by this workflow, after approval
    ├── train_spec.toml
    ├── evaluate_spec_proxy.toml
    └── evaluate_spec_benchmark.toml
```

Per-role evaluate specs are preferred over one shared `evaluate_spec.toml`;
both are accepted. See `references/data-layout.md`.

The user supplies annotations and images. The specs are **not** an input to
ask for: build them from the `tao-finetune-cosmos-reason` templates plus the
AOI overrides, and write them after the approval gate and before
`init_deft_state.py`, which refuses to initialize without them. A workspace
carrying its own specs is still valid — reuse them rather than overwriting —
but their absence is normal and is never a reason to stop and ask the user for
a TOML file.

Non-default paths are valid when passed explicitly to
`scripts/init_deft_state.py`; downstream stages must read the recorded paths
instead of re-inferring conventions. Record absolute host/compute-frame
artifact paths under `${RESULTS_DIR}/baseline` or
`${RESULTS_DIR}/iterN`.

Read `references/data-layout.md` for the dataset roles, allowed source
categories, and commercial-training eligibility.

## Launch Intake and Pre-Flight

Read `references/preflight.md` and run every ordered check:

1. select and preflight the platform;
2. resolve workspace, annotations, media root, and `max_iterations`;
3. validate bare ShareGPT and Proxy/Benchmark/Mining target isolation;
4. hash and freeze Benchmark annotations;
5. resolve current Cosmos-RL and data-services images from `versions.yaml`;
6. plan conversion of the selected Cosmos Reason 3 reasoner into a Qwen3-VL
   PTM, and that output's platform-visible path;
7. check only required environment-variable presence;
8. construct Proxy / Benchmark TOML specs and validate the Train template;
9. verify compute shape and path visibility from the selected platform;
10. run the model/platform launch preflight;
11. show the full Pre-Flight Summary and stop for approval.

No results directory, state file, spec mutation, dependency install, image
pull, or native launch is allowed before this gate, except the TAO policy's
small-Python-helper remediation.

## Workflow

The full transition graph is in `references/pipeline-and-state.md`.

The frozen Benchmark gate is always evaluated before any Proxy work. Proxy
evaluate and RCCA exist only to seed the next iteration's mining, so they run
only when the gate is unmet. A run that passes the gate stops without spending
a Proxy evaluation.

Baseline starts with zero-shot frozen Benchmark evaluation of the unmodified
base model, which establishes the zero-shot KPI:

1. `evaluate_benchmark`
2. `benchmark_metrics` — stop here when the gate already passes.
3. `evaluate_proxy` — only when the gate is unmet.
4. `proxy_rcca`

For each `iterN` when the frozen Benchmark gate is unmet:

1. `routing` — derive mining targets from Proxy false accepts/rejects only.
   Write both formats from the same rows: `mining_targets.json` for state
   (`--mining-targets` takes the JSON) and a `filepath[,label]` parquet for the
   embedding container. Gap rows carry no image paths, so join back to Proxy by
   `id` — see `references/gap-analysis.md`.
2. `anomalygen` — generate synthetic defects with `paidf-anomalygen` in
   `inference_only` mode, then turn each generated pair into a bare `NG`
   record with `scripts/emit_sdg_sharegpt.py`. `--skip` is permitted only when
   the driving Proxy RCCA recorded zero false accepts, and even then generating
   is often still worthwhile — see `references/paidf-anomalygen.md`.
3. `data_mining` — invoke `tao-mine-aoi-images`, then apply the configured
   cosine floor with `scripts/filter_mined_by_cosine.py`.
4. `assemble_data` — align mined target paths to Mining source prompts,
   golden references, and exact labels with `scripts/emit_mined_sharegpt.py`;
   create `train_iter_1.json` from the mined and synthetic records only after
   Proxy RCA and Mining selection, then append monotonically into
   `train_iter_N.json` in later iterations with
   `scripts/assemble_training_json.py`.
5. `validate_data` — validate exact bare labels, files, duplicates, and
   generated-Train lineage plus Proxy/Benchmark leakage.
6. `train`
7. `evaluate_benchmark`
8. `benchmark_metrics` — stop here when the gate passes or
   `N = max_iterations`.
9. `evaluate_proxy` — only when the loop continues.
10. `proxy_rcca`

After every iteration, render `DEFT_Loop_Report.html` with the reporter agent.
Stop when the Benchmark contract passes, `max_iterations` is reached, or a
hard stop occurs. Commit `loop_stop`, run the completion audit, then render one
final report.

## Stage References

| Stage | Producer | Read first |
|---|---|---|
| Train | `tao-finetune-cosmos-reason` train, `automl_policy: off` | `references/cosmos-reason.md`, `references/example_lora_config.toml` |
| Proxy / Benchmark evaluate | `tao-finetune-cosmos-reason` evaluate | `references/cosmos-reason.md` |
| Proxy RCCA / Benchmark metric | bundled `analyze_gaps.py` | `references/gap-analysis.md` |
| Routing / mining | Proxy gaps + `tao-mine-aoi-images` | `references/tao-mine-aoi-images.md` |
| AnomalyGen | `paidf-anomalygen`, `mode=inference_only` | `references/paidf-anomalygen.md` |
| Assemble / validate | bundled bare ShareGPT scripts | `references/aoi-annotation.md` |
| State/log/report | bundled commit/audit scripts + reporter | `references/scripts-and-agents.md` |

## Hard Stops

Commit an error stage and do not auto-retry for: invalid disk state; a rich or
non-exact training label; a JSONL or non-array annotation input; an
an unconverted Cosmos Reason 3 checkpoint still in native Omni format at a
Cosmos-RL boundary;
missing/ambiguous mined-to-source alignment; target overlap among
Proxy/Benchmark/Mining; a generated Train target outside Mining and AnomalyGen
output, or overlapping Proxy/Benchmark; a changed Benchmark hash; any Benchmark
error used for routing; missing/empty mining output; a failed or empty
AnomalyGen run while Proxy false accepts remain outstanding; an `anomalygen`
skip not backed by zero false accepts in the driving RCCA; a synthetic record
whose label is not `NG` or whose paired image is missing; a checkpoint outside
the iteration result tree; an invalid nested TOML spec; unknown evaluator
ground truth; or a program error.

Infrastructure errors may follow the chosen platform skill's bounded retry
policy with a new job-record linked by `--retry-of`; the DEFT stage is committed
only once, after a successful terminal backend result.
