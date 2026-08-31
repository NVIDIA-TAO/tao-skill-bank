---
name: tao-run-deft-aoi-cosmos3
description: >
  Run the disk-backed DEFT AOI improvement loop for NVIDIA Cosmos Reason 3 /
  Cosmos3 models, using Nano by default and Edge or Super when explicitly
  requested: evaluate the base model on Proxy and frozen Benchmark splits,
  mine real image pairs from Proxy gaps, assemble a per-iteration Train JSON
  from selected Mining samples, train with cosmos-rl LoRA SFT, and repeat
  through the selected platform's submit/status/logs/cancel contract.
  The default bare_okng profile uses exact OK/NG labels; the opt-in
  nvpaw_multitask_v1 profile supports component/defect classification and
  normalized-box detection with optional golden references, task-balanced
  KPI gates, and pluggable gap-analysis ablations. Use for "run Cosmos3 DEFT
  AOI", "CR3 AOI loop", or "improve Cosmos3 PCB inspection"; do not use for
  one-off Cosmos training or generic anomaly generation.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Requires the companion TAO skill-bank skills from `eval.config`, host Python with `numpy`, `pyarrow`, and `yaml`, and the selected platform's native CLI.
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

Install this application as part of the full TAO skill-bank root, not as only
the companion skill folders: `TAO_SKILL_BANK_PATH` must point at a directory
containing `versions.yaml`, `scripts/resolve_versions_key.py`, and the
`skills/{applications,models,data,platform,core}/...` tree listed in
`eval.config`. Run bundled validation with the skill Python so dependencies
match runtime: `PYTHON=$(scripts/deft_python.sh); "$PYTHON" -m unittest
tests.test_cosmos3_bare tests.test_cosmos3_nvpaw
tests.test_gap_analysis_profiles`. Resolve network mode first. Missing air-gap imports
are a hard stop; network-enabled setup lives only in
`references/network-bootstrap.md`.

## Execution Contract

Treat a run as a disk-backed state machine.

1. Preserve every explicit user value and show the source of each effective
   value (`user`, `spec`, or `default`) in the Pre-Flight Summary.
2. Ask which installed platform to use. Do not select Docker, SLURM,
   Kubernetes, Brev, virtualenv, or an external platform by default.
3. Resolve network mode, then read exactly one of `references/air-gap.md` or
   `references/network-bootstrap.md`. Run the selected platform skill's
   Preflight and stop on a missing system/native-CLI prerequisite.
4. Before any mutation or launch, invoke `tao-launch-workflow` and show its
   launch review plus this skill's Pre-Flight Summary. Wait for one explicit
   approval.
5. After approval, initialize `${RESULTS_DIR}/deft_state.json` exactly once
   with `scripts/init_deft_state.py`. Pass the exact GPU model reported by the
   selected platform's Preflight through `--gpu-model` (include accelerator
   memory when available), plus the resolved network mode/source and selected
   absolute Python. Never reinitialize a resumed run or edit
   `deft_state.json` by hand.
6. Before every stage, after context compaction, and before a completion claim,
   run `scripts/deft_context.py --state ... --stage ...`. Use its durable
   `next_stage` and the state file's `status`,
   `current_iteration`, `iterations.*.status`, `stage_completed`, and latest
   `events` entry to resume. Do not infer progress from assistant prose or
   from an artifact that is not recorded in state.
7. Run every command that can install, fetch, log in, or launch a local
   container through `scripts/deft_exec.py --state ... -- <command>`. In an
   air-gap it rejects egress/package operations and enforces no-pull. Remote
   platforms must apply the equivalent immutable no-pull/offline policy.
8. Submit each GPU stage through the chosen platform's four verbs:
   `submit` / `status` / `logs` / `cancel`. The `submit` verb must open the
   job-record before native launch; the returned id is the only launch handle.
   Poll the backend, not the job-record, and map state to
   `PENDING RUNNING COMPLETE ERROR CANCELED UNKNOWN`.
9. Commit every completed or failed DEFT stage with
   `scripts/commit_stage.py`. It verifies the stage inputs and atomically
   updates both the resume snapshot and ordered `events` array in state. Every
   commit requires a positive, measured `--duration-sec`: use backend elapsed
   wall time for submitted jobs and a host wall-clock timer for inline stages.
   Missing or zero durations are rejected.
10. Claim completion only after `scripts/finalize_run.py` verifies final
   Benchmark evidence, successfully commits `loop_stop`, and a fresh
   read of `deft_state.json` shows `status == "complete"`,
   `iterations.baseline.status == "complete"`, and the final iteration's
   `status == "complete"`.

Never place secrets in a spec, command, transcript, job-record, or chat. Check
credential presence only, for example
`[ -n "$HF_TOKEN" ] && echo SET || echo UNSET`. Credentials come from the
user's exported shell environment or from a user-approved env file —
`~/.tao/secrets.env`, `~/.config/tao/.env`, or a path the user points at, never
one merely found in the workspace — loaded with
`set -a; source /path/to/.env; set +a`. Never print the file's contents or any
credential value.

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

## Annotation Profiles

Profile selection is explicit and frozen in state. `bare_okng` remains the
default; never infer a profile from annotation contents.

### `bare_okng`

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

### `nvpaw_multitask_v1`

- Supports the six task types in `references/nvpaw-prompt-formats.md`:
  component/defect classification and detection, each with the documented
  single-image or golden-then-target role contract.
- `id` identifies one prompt/answer record; `target_id` identifies one physical
  target. Multiple records may share a target without duplicate embedding work.
- Classification answers are prompt-local semantic choice sets, including
  valid `[]`. Detection answers are labeled `xyxy` integer boxes normalized to
  `[0,1000]`, also allowing `[]`.
- JSONL OpenAI `messages` is an authoring format only. Run
  `materialize_nvpaw_annotations.py`; Cosmos consumes the deterministic JSON
  array it produces.
- Pass `--annotation-profile nvpaw_multitask_v1` through validation, analysis,
  routing, emit, assembly, and state commands. Rich mode is never automatic.
- Freeze `--mining-router-mode` in state for every run. `image_only` preserves
  the visual-similarity baseline; `task_strict` restricts each query to Mining
  targets carrying at least one matching task; `task_then_fallback` fills a
  short strict neighborhood from the global image pool and records every such
  row as `route_tier=fallback`. The default remains `image_only` for backward
  compatibility. See `assets/mining-router.svg` for the operator-facing flow.
- Freeze `--anomalygen-policy auto|disabled` in state. `auto` is the backward-
  compatible default and keeps the gap-evidence skip gate. `disabled` makes
  the AnomalyGen skip unconditional for every iteration: do not resolve its
  image or assets, do not launch it, commit the stage with `--skip`, and keep
  training from mined records.

Run `scripts/validate_sharegpt.py` with the selected profile on Proxy,
Benchmark, Mining, and each generated iteration training file. There is no
input Train annotation.
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
- Bare default gate: `recall_ng >= 1.0`. If the user asks for accuracy, use
  `accuracy >= <target>`.
- Rich default gate: `task_balanced_v1`. Its scalar is the worst task
  attainment; missing, duplicate, unknown, or unparsable predictions block the
  gate. `task_dataset_balanced_v1` is an explicit experimental alternative and
  enforces `--min-group-support` for every task×dataset cell.

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
3. validate the selected annotation profile and Proxy/Benchmark/Mining target isolation;
4. hash and freeze Benchmark annotations;
5. resolve current Cosmos-RL and data-services images from `versions.yaml`,
   plus AnomalyGen only when `--anomalygen-policy auto`;
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

1. `routing` — derive mining targets from Proxy gaps only.
   Write both formats from the same rows: `mining_targets.json` for state
   (`--mining-targets` takes the JSON) and a `filepath[,label]` parquet for the
   embedding container. Bare rows join back to Proxy by `id`; rich rows already
   carry explicit target identity and collapse record gaps by `target_id` — see
   `references/gap-analysis.md`.
2. `anomalygen` — when the frozen policy is `disabled`, commit an unconditional
   `--skip` and continue to mining. Under `auto`, generate synthetic defects
   with `paidf-anomalygen` in `inference_only` mode or use its evidence-gated
   skip. Bare output becomes exact `NG`; rich output is limited to defect-
   classification tasks with a known label. Detection SDG is a hard error
   without validated geometry. See `references/paidf-anomalygen.md`.
3. `data_mining` — use `tao-mine-aoi-images` to embed each unique Proxy target
   and Mining source image once. Bare mode keeps the native nearest-neighbor
   plus `filter_mined_by_cosine.py` path. Rich mode passes both embedding
   artifacts through `task_mining_router.py`, which applies the frozen
   `image_only`, `task_strict`, or `task_then_fallback` policy, top-K, and cosine
   floor while retaining routed task provenance. Then run the mapped skill's
   history-aware post-processing. The default top-K remains 5.
4. `assemble_data` — align mined target paths to Mining source prompts,
   golden references, and exact labels with `scripts/emit_mined_sharegpt.py`;
   task-aware rows fan out only to `routed_task_types`, while `image_only` and
   explicit fallback rows retain the source target's available tasks;
   create `train_iter_1.json` from mined records plus synthetic records only
   when AnomalyGen produced them, after Proxy RCA and Mining selection; then
   append monotonically into
   `train_iter_N.json` in later iterations with
   `scripts/assemble_training_json.py`.
5. `validate_data` — validate the selected profile, files, record-level
   duplicates, generated-Train lineage, and Proxy/Benchmark target leakage.
6. `train`
7. `evaluate_benchmark`
8. `benchmark_metrics` — stop here when the gate passes or
   `N = max_iterations`.
9. `evaluate_proxy` — only when the loop continues.
10. `proxy_rcca`

`init_deft_state.py` writes the first `DEFT_Loop_Report.html`; every successful
`commit_stage.py` call then refreshes it through the deterministic
`scripts/render_report.py` post-commit hook. Stop when the Benchmark contract
passes, `max_iterations` is reached, or a hard stop occurs. For an ordinary
stop, run `scripts/finalize_run.py` with the explicit reason, then run
`render_report.py --require-terminal` after optional token alignment.
The report includes a bounded prompt showcase; rich runs also show task KPI
attainment, task×dataset diagnostics, and prediction coverage. Keep every other visual convention aligned with
ChangeNet. See `references/REPORT_RENDERING.md`. Never delegate or hand-author
report rendering.

## Stage References

| Stage | Producer | Read first |
|---|---|---|
| Train | `tao-finetune-cosmos-reason` train, `automl_policy: off` | `references/cosmos-reason.md`, `references/example_lora_config.toml` |
| Proxy / Benchmark evaluate | `tao-finetune-cosmos-reason` evaluate | `references/cosmos-reason.md` |
| Proxy RCCA / Benchmark metric | bundled `analyze_gaps.py` | `references/gap-analysis.md` |
| Routing / mining | Proxy gaps + `tao-mine-aoi-images` | `references/tao-mine-aoi-images.md` |
| AnomalyGen | `paidf-anomalygen`, `mode=inference_only` | `references/paidf-anomalygen.md` |
| Assemble / validate | bundled profile-aware ShareGPT scripts | `references/aoi-annotation.md` |
| State/report | bundled state commit + deterministic report hook | `references/scripts-and-agents.md` |

## Hard Stops

Commit an error stage and do not auto-retry for: invalid disk state; a label or
answer invalid for the selected profile; a non-array materialized annotation input; an
an unconverted Cosmos Reason 3 checkpoint still in native Omni format at a
Cosmos-RL boundary;
missing/ambiguous mined-to-source alignment; missing/tampered mining history,
cross-iteration mined filepath duplication; target overlap among
Proxy/Benchmark/Mining; a generated Train target outside Mining and AnomalyGen
output, or overlapping Proxy/Benchmark; a changed Benchmark hash; any Benchmark
error used for routing; missing/empty mining output; attempting to run
AnomalyGen when its frozen policy is `disabled`; under `auto`, a failed or
empty AnomalyGen run while eligible gaps remain outstanding, or an
`anomalygen` skip without the required zero-gap evidence; a synthetic record
whose label is not `NG` or whose paired image is missing; a checkpoint outside
the iteration result tree; an invalid nested TOML spec; unknown evaluator
ground truth; or a program error.

Infrastructure errors may follow the chosen platform skill's bounded retry
policy with a new job-record linked by `--retry-of`; the DEFT stage is committed
only once, after a successful terminal backend result.
