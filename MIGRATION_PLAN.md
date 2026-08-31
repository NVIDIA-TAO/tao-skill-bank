# DEFT AOI Cosmos Framework Migration Plan — rebased revision

Status: **approved by Sean; implementation complete on `dev/deft-aoi-cfw`;
runtime launch remains approval-gated and the canonical exact-evaluator
preflight is blocked by source data described below**

This revision supersedes the earlier approved plan because Sean changed both
the source base and the application scope. The branch is now rebased onto the
latest `origin/main`, and AnomalyGen is to be removed from the NVPAW DEFT AOI
application rather than migrated.

## Provenance and approval boundary

- Migration worktree: `~/projects/deft/tao-skill-bank-worktrees/dev-deft-aoi-cfw`
- Migration branch: `dev/deft-aoi-cfw`
- Latest main base: `origin/main` at
  `95e0295a494469197ac5538d6d713a0965a78006`
- Rebased task-balanced head:
  `fdc73655d7f70652b3431de8d0293bc92f82f17c`
- Replayed NVPAW commits: task-aware routing, component-count history,
  frozen-task-group evaluation, and the task-balanced run recipe. The final
  rebased state retains the six current NVPAW classification/detection tasks.
- Comparison-only branch: `origin/dev/deft_aoi_cr_framework_semicon` at
  `659fd4daa5f11bc19fd001642ace92009810565b`
- Read-only CFW recipe ground truth:
  `/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/yichengl/proj/cosmos3-reasoner/outputs/NVPAW_v1_cosmos_framework/scripts`
- Canonical workspace: `~/projects/deft/workspace`

The dirty main worktree (`.codex-plugin/plugin.json`) was not touched. Sean
approved this plan, so implementation performed code/config/doc changes and
non-side-effecting tests. No image pull, model/data download, registry login,
train, evaluate, inference, or platform submission is authorized. Every
runtime launch still requires its own launch review and explicit confirmation.

## Target outcome

`tao-run-deft-aoi-cosmos3` must use Cosmos Framework for its train, evaluate,
and inference runtime paths, consume the canonical NVPAW JSONL files directly,
produce and hand off Framework DCP checkpoints, and use the workspace
`eval/calculate_f1_metrics.py` as the only KPI authority.

The iterative application becomes a real-mining-only loop:

```text
Benchmark gate -> Proxy RCCA -> routing -> data_mining -> assemble_data
               -> validate_data -> CFW train -> Benchmark gate -> ...
```

There is no AnomalyGen/SDG stage, configuration, checkpoint, Guardrail,
synthetic-data producer, artifact, or failure mode in the migrated app.

## Implementation validation result

- The adapted existing application suite passes all 19 tests. New migration
  coverage is limited to one lightweight CLI contract smoke per runtime surface
  (train, evaluate, inference). Model/workload routing passes both resolver
  tests, repository skill validation passes, and static searches find no
  removed runtime or application-local AnomalyGen surface.
- Train renders the reviewed Framework recipe and a platform-neutral DCP launch
  descriptor. Evaluate and inference share the packaged
  `cfw_jsonl_runtime.py`, which streams canonical JSONL, preserves ordered
  one/two-image messages and pixel bounds, consumes a verified exported action
  model after DCP handoff, and atomically emits the exact prediction schema.
- Canonical annotation and media validation passes for 3,000 Proxy rows, 2,145
  Benchmark rows, and 970,839 eligible Mining rows. The complete Mining source
  has 972,799 rows; 1,960 count/segmentation rows are outside the approved six
  task types and are ignored in memory with auditable counts. The complete
  canonical file remains hash-sealed and unmodified.
- The exact evaluator preflight correctly remains a hard stop on the current
  canonical workspace: `calculate_f1_metrics.py` rejects seven classification
  Benchmark rows whose ground truth is `[]` (the first is
  `Real-IAD/images/pcb/test/OK/S0096/pcb_0096_OK_C2_20231027165818.jpg#qa0`).
  The migrated application does not patch, copy, or replace the evaluator and
  does not reinterpret those labels. The canonical annotations/evaluator must
  be reconciled before any approved runtime launch.

## Done versus remaining

Done in this branch:

- rebased the NVPAW DEFT source onto latest main and routed only
  `workload=deft-aoi` train/evaluate/inference to Cosmos Framework;
- migrated training to the reviewed 8-GPU full-parameter BF16/FSDP recipe,
  direct indexed canonical JSONL, synchronous DCP, and strict DCP/action-model
  handoff;
- migrated evaluate/inference to the shared Framework JSONL/multi-image
  runtime and exact output schema;
- made the workspace evaluator the only F1 authority, with evaluator and raw
  report path/SHA-256 binding and fail-closed coverage;
- removed all application-local AnomalyGen, SDG, Guardrail, synthetic-data,
  and Cosmos-RL compatibility surfaces; and
- updated state schema, real-mining-only transitions, reports, references,
  eval fixtures, and the adapted existing test suite.

Remaining outside the code migration:

- reconcile the seven canonical Benchmark `GT: []` classification rows with
  `eval/calculate_f1_metrics.py`; until then exact-evaluator preflight must stop;
- after that reconciliation, obtain a separate launch review/approval for the
  container/GPU smoke (train to DCP, base and trained evaluate/inference, exact
  F1, and DCP resume); and
- push or open an MR only if Sean explicitly requests it.

## Latest-main inventory findings

The rebase materially changes the earlier plan:

- `versions.yaml` already pins
  `images.tao_toolkit.cosmos_framework` to the current CFW image.
- `tao-finetune-cosmos-reason/references/skill_info.yaml` already declares
  separate CFW and Cosmos-RL image pins and backend contracts.
- The current model skill already provides the Framework native trainer,
  CFW DCP contract, exact-key exporter, checkpoint pre-action, image-digest
  resolution, and platform-neutral planning infrastructure.
- Generic Nano train/evaluate/inference still select Cosmos-RL by default.
  There is no `deft-aoi` workload rule in either model resolver.
- The application preflight explicitly asks `resolve_tao_image.py` for
  `--backend cosmos-rl`; app state stores `config.containers.cosmos_rl`.
- Current NVPAW JSONL is still treated as an authoring format and converted
  into Cosmos-RL JSON arrays before runtime.
- Current training examples are Cosmos-RL LoRA TOML. The application expects
  an exported HF/adapter checkpoint rather than native Framework DCP.
- Current evaluation uses `cosmos-rl-evaluate` and the source-patching
  `patch_eval_image_cap.py` compatibility path.
- Current rich KPI is calculated locally by `multitask_metrics.py` and exposed
  as `task_balanced_v1`; this conflicts with the required exact workspace F1
  evaluator.
- The current state graph contains `anomalygen` between `routing` and
  `data_mining`, and the app resolves the PAIDF image/checkpoints/assets,
  accepts synthetic records, and renders SDG evidence.

## Dependency and replacement map

| Dependency point | Rebased current state | Approved-target replacement | Impacted files | Verification | Main risk |
| --- | --- | --- | --- | --- | --- |
| DEFT backend routing | Generic Nano defaults to Cosmos-RL; no DEFT workload route | Route `workload=deft-aoi` train/evaluate/inference to `cosmos-framework`; explicit supported backend still wins; AutoML/HPO and non-DEFT Nano defaults remain Cosmos-RL | `scripts/resolve_tao_model.py`; model `scripts/cosmos_workflow.py`; model `references/skill_info.yaml`; routing tests | Three DEFT action cases select CFW; generic Nano/AutoML regression matrix stays unchanged | Broadly changing model defaults would regress unrelated consumers |
| CFW image | Latest main already has the CFW version pin and backend image metadata | Reuse the existing pin and latest-main digest resolver; update only the app's requested backend/state fields | App `references/preflight.md`; `scripts/init_deft_state.py`; `scripts/tests/test_deft_versions_contract.py`; image resolver tests | Resolved tag and immutable runtime digest are recorded; no duplicate pin | Importing stale reference-branch pin changes over main |
| Cosmos3 checkpoint intake | App documents Cosmos-RL preparation and a prepared PTM as its runtime boundary | For the DEFT CFW path, accept the complete workspace Qwen3-VL HF snapshot directly or use the CFW-native converter for Omni; do not invoke a Cosmos-RL preparation entrypoint | App `SKILL.md`; `references/cosmos-reason.md`; preflight; model CFW action metadata/tests | Structural config/tensor/processor/provenance check; planned command contains no Cosmos-RL runtime | Workspace converted snapshot may differ from YC's Omni source path |
| Training spec | Cosmos-RL LoRA TOML, epoch output, and single-GPU examples | CFW full and smoke TOML rendered from nested data; full profile exactly matches YC's full-parameter 8-GPU recipe | Replace `references/example_sft_config.toml` and `example_lora_config.toml` with CFW templates; add CFW renderer/planner and tests | Golden TOML values and `4 x 8 x 16 = 512`; no active LoRA block in full profile | Reference branch's one-image LoRA recipe is incompatible |
| Training driver | `cosmos-rl --config ...` and packaged Cosmos-RL hook | Package reviewed NVPAW CFW experiment/dataset/distributor/processor hooks and call `cosmos_framework.scripts.train` through the existing model/backend planner | New app adapter package and planner; model CFW contract if an application hook field is needed | Import/descriptor tests, config validation, then separately approved GPU smoke | Image API drift or missing registration hooks |
| Canonical data boundary | JSONL is materialized to Cosmos-RL JSON arrays | Consume `annotations/{mining,benchmark,proxy_kpi}.jsonl` directly with a deterministic byte-offset index; filter only Mining's out-of-scope count/segmentation rows in memory and report them | `scripts/materialize_nvpaw_annotations.py`; `nvpaw_annotations.py`; `check_annotations.py`; `validate_sharegpt.py`; new CFW data adapter; data docs/tests | Seal full-source path/row count/SHA-256/image count; preserve one/two images and pixel bounds; strict Proxy/Benchmark | Path/mount identity or over-context samples |
| Iteration training data | Mined plus optional synthetic records are assembled into a JSON array | Assemble real Mining selections only into canonical JSONL accepted by the CFW adapter; retain monotonic lineage and split isolation | `assemble_training_json.py`; `emit_mined_sharegpt.py`; `validate_split_contract.py`; annotation/state docs/tests | JSONL round trip, no Proxy/Benchmark leakage, prior-iteration monotonicity | Removing synthetic inputs must not weaken real-mining lineage checks |
| Train output | HF/adapter directory such as `safetensors/epoch_10` | Framework synchronous DCP `iter_#########`, with model/optim/scheduler/trainer metadata, per-rank loader state, and latest pointer | `commit_stage.py`; new DCP validator; state/output docs/tests | Complete/incomplete/corrupt DCP fixtures and resume tests | A partial DCP could be committed |
| Checkpoint handoff | Evaluate/inference consume a Cosmos-RL-style exported checkpoint | Use latest-main `framework_checkpoint_action.py` to verify or exact-key materialize DCP before CFW evaluate/inference; record the manifest | Model CFW metadata/helper; app evaluate/inference planners; state | DCP/HF detection, exact-key manifest, idempotent reuse, base-vs-trained selection | Silent key mismatch or stale export reuse |
| Evaluation | `cosmos-rl-evaluate` plus an evaluator source patch for multi-image | CFW evaluation adapter preserving complete message/image order and emitting the workspace evaluator's prediction JSONL schema | Remove `patch_eval_image_cap.py`; add CFW evaluation renderer/normalizer; update app/model docs/tests | One- and two-image fixtures; no source patch/mount; base and DCP planning tests | Reference evaluator is one-image bare OK/NG and cannot be copied verbatim |
| Inference | Cosmos-RL backend action | CFW inference using the same preprocessing, checkpoint, generation, and output-normalization contract as evaluation | App inference planner/normalizer; model action metadata; tests/docs | Base/DCP planning and prediction-schema tests | Eval/inference preprocessing could diverge |
| KPI authority | Local `multitask_metrics.py` computes six task groups and `balanced_score` | Invoke the exact workspace `eval/calculate_f1_metrics.py`, record its path/SHA-256, preserve raw JSON, and never recalculate F1 in app code | Refactor/remove `multitask_metrics.py`; update `analyze_gaps.py`, `metric_contract.py`, `record_metric_result.py`, state/report/docs/tests | Raw report equality; missing/unknown predictions fail closed; search ensures no second F1 implementation | Exact evaluator groups by family/cohort rather than all six current task types |
| Stop gate | `task_balanced_v1` uses six locally calculated task groups | Use schema-distinct `f1_cohort_balanced_v1`: freeze five required evaluator paths and take minimum threshold attainment without recomputing F1 | State/metric/report/gap-control files and tests | Required paths cannot disappear; vector, coverage, threshold, and scalar projection are recorded | Non-reference component and defect DET are pooled by the exact evaluator |
| State/provenance | State version 5/6 records Cosmos-RL, LoRA, materialized JSON, adapter checkpoint, and AnomalyGen | Bump schema; record CFW tag/digest, full recipe, JSONL/evaluator hashes, DCP/export, predictions/raw F1; omit AnomalyGen entirely | `references/deft_state.json`; `init_deft_state.py`; `commit_stage.py`; `deft_context.py`; `render_report.py` | New-state golden test; old backend/stage state fails closed for resume | Implicitly resuming an old Cosmos-RL/AnomalyGen run |
| Platform submission | App is platform-neutral, while the reference adds direct Docker submit helpers | Keep latest-main four-verb/job-record ownership; app emits only image/command/config/mount/resource descriptors | CFW app renderers/planners; `SKILL.md`; preflight; planner tests | No native submit/pull command in renderers; selected platform consumes descriptor | Copying reference's Docker-only assumptions |
| Air-gap/eval packaging | Bundles Cosmos-RL and AnomalyGen assets and eval dependencies | Bundle CFW plus app adapter/evaluator provenance only; remove AnomalyGen dependency and artifacts | `references/air-gap.md`; `eval.config`; `evals/evals.json`; packaging tests | Dependency list and artifact manifest contain neither runtime | Leaving a hidden bootstrap/download path |

## AnomalyGen removal — explicit scope

Follow the removal shape on the comparison branch, adapted to the rebased
task-balanced and latest-main contracts. This is deletion, not a CFW port.

### State machine and runtime

- Remove `anomalygen` from `STAGES`, `SKIPPABLE_STAGES`, CLI choices, state
  snapshots, event fields, and next-stage transitions.
- Change the transition to `routing -> data_mining`.
- Remove `--anomalygen-policy`, project/image/checkpoint/dataset/base-model
  arguments and all AnomalyGen image resolution/preflight.
- Set training provenance to `mined_real_samples_only`; assembly requires at
  least one validated mined record and has no synthetic input.
- Remove SDG/AMP/Guardrail asset checks, PAIDF compatibility checks, skip
  evidence, retry/failure text, and runtime mount/network instructions.

### Files to delete

- `skills/applications/tao-run-deft-aoi-cosmos3/scripts/emit_sdg_sharegpt.py`
- `skills/applications/tao-run-deft-aoi-cosmos3/references/tao-generate-anomalies.md`
- `skills/applications/tao-run-deft-aoi-cosmos3/tests/test_sdg_path_compat.py`

`patch_eval_image_cap.py` and `test_eval_image_cap_compat.py` are also deleted,
but as part of the Cosmos-RL evaluator removal rather than AnomalyGen.

### Files to strip or rewrite

- Application surface: `SKILL.md`, `skill-card.md`, `eval.config`, and
  `evals/evals.json`.
- References/reporting: `DEFT_Loop_Report.html`, `RCCA_REPORT_TEMPLATE.md`,
  `REPORT_RENDERING.md`, `air-gap.md`, `aoi-annotation.md`,
  `cosmos-reason.md`, `data-layout.md`, `deft_state.json`,
  `nvpaw-prompt-formats.md`, `pipeline-and-state.md`, `preflight.md`, and
  `scripts-and-agents.md`.
- Runtime/state: `assemble_training_json.py`, `commit_stage.py`,
  `deft_context.py`, `init_deft_state.py`, `nvpaw_annotations.py`,
  `render_report.py`, `route_selected_gaps.py`, and
  `validate_split_contract.py`.
- Tests: remove/update AnomalyGen and synthetic cases in
  `test_cosmos3_bare.py`, `test_cosmos3_nvpaw.py`,
  `test_cosmos3_init_state_contract.py`,
  `test_cosmos3_rcca_report_contract.py`,
  `test_cosmos3_report_rendering.py`, `test_validator_truthfulness.py`, and
  packaging/eval tests.
- Repository contract test: remove `images.metropolis_sdg.paidf_anomalygen`
  only from the Cosmos3 application entry in
  `scripts/tests/test_deft_versions_contract.py`.

Do not delete the global `tao-generate-anomalies` skill or the global
`images.metropolis_sdg.paidf_anomalygen` version key; other applications still
own them.

## Direct Cosmos-RL application dependencies to eliminate

Active mentions occur in:

- `SKILL.md`, `skill-card.md`, `eval.config`
- `references/air-gap.md`, `aoi-annotation.md`, `cosmos-reason.md`,
  `data-layout.md`, `deft_state.json`, `example_lora_config.toml`,
  `example_sft_config.toml`, `preflight.md`, `scripts-and-agents.md`, and the
  soon-to-be-deleted AnomalyGen reference
- `scripts/check_annotations.py`, `init_deft_state.py`,
  `materialize_nvpaw_annotations.py`, `nvpaw_annotations.py`,
  `patch_eval_image_cap.py`, and `validate_sharegpt.py`
- `tests/test_cosmos3_bare.py` and `test_eval_image_cap_compat.py`

The final application tree may contain a concise historical note, but no
Cosmos-RL command, import, image, backend selection, dataset serialization,
checkpoint, metric, patch, or output contract. Generic model-skill Cosmos-RL
support remains for unrelated workflows.

## YC CFW recipe to encode

- Experiment: `nvpaw_omni_vlm_sft`
- Precision: BF16
- Full-parameter tuning (`keys_to_select = []`), not LoRA
- 8 GPUs; FSDP shard 8, replicate 1
- Micro-batch per rank 4; gradient accumulation 16; global batch 512
- Fused AdamW; LR `1e-6`; weight decay `0.05`; betas `0.9`, `0.999`;
  merger LR multiplier 20
- Freeze vision encoder true; multimodal projector false; language model false
- Full activation checkpointing
- 500 iterations; save every 100; cycle 500; warmup 5;
  `f_start=0.05`, `f_max=1`, `f_min=0.1`
- Synchronous DCP
- Direct indexed JSONL with sealed row count, SHA-256, and image-item count
- Deterministic shuffle, DP-rank batches, resumable distributor state,
  assistant-token masking, multi-image/pixel-bound preservation, and
  deterministic over-context resampling

The smoke profile may reduce rows, iterations, checkpoints, and—only when
explicitly named—GPU topology. It must not silently change precision,
full-parameter tuning, freeze policy, data semantics, or checkpoint format.

## Implementation phases after renewed approval

### 1. Tests first: routing, removal, and F1 contracts

1. Add failing resolver tests for DEFT CFW train/evaluate/inference and
   non-DEFT regression cases.
2. Add failing state-machine tests proving `routing -> data_mining` and that
   no AnomalyGen/synthetic CLI or state field exists.
3. Add failing exact-evaluator adapter tests and freeze the five required F1
   JSON paths:
   - `non_reference_based.tasks.BCQ.macro_f1`
   - `non_reference_based.tasks.MCQ.macro_f1`
   - `non_reference_based.tasks.DET.f1`
   - `reference_based.tasks.BCQ.macro_f1`
   - `reference_based.tasks.DET.f1`
4. Add failing CFW recipe/data/DCP/planner contract tests.

### 2. Remove AnomalyGen and simplify the loop

Apply the explicit removal section above. Preserve Proxy-driven routing,
task-aware mining, history deduplication, real-record assembly, Benchmark
isolation, durable reports, and latest-main CLI/digest safeguards.

### 3. Route DEFT to the existing CFW backend

Add the application workload rule to both resolvers and metadata. Update the
app preflight/state to request and record CFW. Do not alter generic Nano,
AutoML/HPO, quantize, or explicit-backend behavior.

### 4. Package the direct NVPAW CFW data/training adapter

Package reviewed equivalents of YC's dataset index, distributor, processor,
experiment registration, training entrypoint, and artifact validation. The
YC directory stays read-only and is not a runtime dependency. Render the full
and smoke TOML from nested data and emit a platform-neutral launch descriptor.

### 5. Implement DCP, evaluation, inference, and F1

Validate DCP completeness and use the existing Framework checkpoint action for
materialization. Normalize CFW evaluation/inference responses to JSONL rows
with `id`, `task_type`, source message(s), `GT`, and `raw_prediction`. Invoke
the recorded exact evaluator and store its raw report, coverage, component
vector, and deterministic minimum-attainment gate.

### 6. State/docs/cleanup

Bump state schema, fail closed on old backend/stage state, update all docs and
eval fixtures, delete obsolete files, and verify application-tree searches
for both Cosmos-RL runtime dependencies and AnomalyGen/SDG dependencies.

## Validation plan

### Rebased baseline observed during inventory

```text
PYTHON=$(bash scripts/deft_python.sh)
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m unittest discover -s tests -p 'test_*.py'
```

Result: **87 tests run; 84 passed and 3 failed**. All failures are confined to
the now-removed AnomalyGen surface:

- the task-balanced test expects the pre-main `policy_disabled` skip reason;
  latest main records the required human summary instead;
- the flattened-install wrapper repeats that same failure;
- the AnomalyGen reference has one pre-main executable-bit-dependent script
  invocation.

These are recorded baseline incompatibilities from the rebase, not CFW
implementation failures. The revised scope deletes the affected stage,
reference, and expectations. All non-AnomalyGen tests passed.

### Static and unit gates

- All app tests and relevant repository resolver/version tests pass.
- DEFT action routing selects CFW; generic/AutoML Cosmos-RL routes do not move.
- Application stage lists, CLI help, state, report, eval config, and docs contain
  no AnomalyGen/SDG/Guardrail/PAIDF contract.
- Full CFW golden config matches every YC value and global batch 512.
- JSONL fixtures prove indexing, hashes, image counts, one/two-image ordering,
  pixel bounds, masking, shuffle, and resume.
- Real-mining assembly remains monotonic and split-isolated with no synthetic
  input path.
- DCP fixtures cover complete, partial, corrupt, pointer, resume, and exact-key
  materialization cases.
- CFW evaluation/inference fixtures preserve message/image structure and emit
  the exact prediction schema.
- Exact workspace evaluator output is preserved; application code does not
  implement an alternative F1 calculation; missing/unknown coverage fails.
- New state records CFW image digest, JSONL/evaluator hashes, DCP/export, and
  F1 evidence; old Cosmos-RL/AnomalyGen state cannot resume as the new schema.
- App renderers contain no native platform submit/pull operation.
- `scripts/validate-skills.sh` and packaging/exec-bit tests pass.

### Separately approval-gated smoke

Stage deterministic read-only subsets under a new job results/staging area:

- mining: 500 rows
- benchmark: 100 rows
- proxy KPI: 180 rows

Do not modify canonical workspace files. After a separate platform/resource,
image/digest, model, mount, command, output, credential-presence, and cost
review, run CFW preflight; short full-parameter training to DCP; base and DCP
evaluate/inference; exact F1/coverage comparison; and DCP resume. No smoke
launch is authorized by plan or implementation approval alone.

## Risks and mitigations

| Risk | Mitigation / decision gate |
| --- | --- |
| Rebase accidentally reintroduces stale release/image behavior | Latest main is the ancestor; use existing pin/digest helpers and retain main regression tests |
| Reference branch implements a one-image LoRA/Docker workflow | Reuse only routing, DCP, and removal patterns; recipe/data tests come from YC and the NVPAW workspace |
| AnomalyGen removal leaves an unreachable state/report/eval dependency | Contract searches plus stage/CLI/state/report golden tests; delete dedicated files and synthetic inputs |
| Removing synthetic support weakens lineage validation | Keep all real Mining origin, history, monotonicity, and split-isolation checks; require non-empty mined input |
| CFW image lacks packaged NVPAW registrations or APIs drift | Package reviewed adapter, run import/contract probe before any approved GPU work, no runtime source patch |
| Workspace Qwen3-VL snapshot differs from YC's Omni source | Validate config/tensor/processor/provenance and CFW loading; stop instead of silently converting/substituting |
| Multi-image or pixel bounds are lost | Direct JSONL adapter plus explicit one/two-image and bounds fixtures |
| Partial DCP or stale export is accepted | Strict component/rank/metadata/pointer checks and existing exact-key manifest validation |
| Exact F1 evaluator pools non-reference detection tasks | Use the already approved schema-distinct five-component cohort gate; retain raw report, components, and task-aware routing evidence |
| Platform-neutral contract is bypassed by copied submit helpers | Render descriptors only; selected platform retains four verbs and record-before-launch |
| Generic Cosmos-RL or global AnomalyGen users regress | Do not remove global backends/skills/image keys; run explicit non-DEFT resolver and version tests |

## Renewed approval checkpoint

Sean should approve or amend this rebased revision before implementation.
Approval would authorize the code/config/doc changes and non-side-effecting
tests above, including deletion of the application-local AnomalyGen and
Cosmos-RL compatibility files. It would not authorize container pulls,
downloads, registry login, train/evaluate/inference, push, or any platform
submission.
