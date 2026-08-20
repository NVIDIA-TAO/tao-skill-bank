# Scripts, Stage Ownership, and Recovery

Despite this file's compatibility-preserving name, the IAA workflow does not
need a reporter agent or stage subagents. One agent runs a small linear flow
through deterministic scripts.

## Contents

- [Script interfaces](#script-interfaces)
- [Stage ownership](#stage-ownership)
- [Platform action contract](#platform-action-contract)
- [Bundled adapter contract](#bundled-adapter-contract)
- [Path invariants](#path-invariants)
- [Bounded recovery](#bounded-recovery)

## Script interfaces

| Script | Role | Runtime |
|---|---|---|
| `deft_python.sh` | select a Python per call and cap BLAS/OpenMP threads | shell |
| `check_iaa_cuda_runtime.py` | allocate and synchronize CUDA tensors, then verify image-specific TAO CLI entrypoints | selected image or approved TAO virtualenv |
| `manage_iaa_virtualenv.py` | non-mutating lock plan, approved hash-locked install, or full profile/CUDA verification | control Python plus selected profile |
| `virtualenv_runtime.py` | enforce the schema-validated `pyt`/`ds` ABI, package, console-script, import, pip, and CUDA contract | control Python plus selected profile |
| `prepare_deft_config.py` | copy templates into a run and apply approved values | IAA runtime |
| `init_deft_state.py` | validate config/metric inputs and create schema-v3 state once | IAA runtime |
| `audit_deft_run.py` | read-only state, transition, evidence, and artifact audit | IAA runtime (parquet/YAML validation) |
| `run_deft_action.py` | prepare one platform-neutral TAO bundle and finalize native job/output evidence | control Python |
| `run_deft_cli.py` | verified TAO CLI/path adapter used only by the virtualenv platform | action-selected `pyt` or `ds` profile |
| `run_deft_container.py` | legacy Docker-only compatibility adapter for schema-v1 runs | control Python |
| `run_iaa_stage.py` | expose bundled IAA host operations as named subcommands | IAA runtime |
| `iaa_deft/` | bundled IAA gap, mining, selection, visualization, config, and checkpoint implementation | imported only through `run_iaa_stage.py` |
| `commit_stage.py` | validate and atomically commit one stage | IAA runtime for parquet checks |
| `recover_commit.py` | roll back a journaled state/log commit interrupted by process death | IAA runtime (runs the audit) |
| `parse_iaa_metrics.py` | parse one exact CSV row/column into iteration-bound JSON | control Python |
| `render_deft_report.py` | audit and atomically render a deterministic HTML report | IAA runtime (its audit validates parquet) |
| `metric_contract.py`, `command_contract.py`, `checkpoint_contract.py`, `record_metric_result.py`, `log_stage.py` | internal validation/commit helpers | never invoke directly |

Use the control plane for platform-action preparation/finalization and metric parsing:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/<control-script>.py" ...
```

Add `--runtime` for config preparation, initialization, audit, bundled IAA
adapters, stage commit/recovery, and reporting:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/<runtime-script>.py" ...
```

Every path argument must be absolute. Workflow paths and decisions may not
depend on a prior `cd` or hidden env-file load. Credential-requiring commands
consume only variables inherited from the launching process; no wrapper opens
or sources a credentials file, and no value is written to argv, status, state,
or logs.

## Stage ownership

| State stage | Producer | Adapter or launch | Read first |
|---|---|---|---|
| `dataset_setup` | archive `rebuild.py` plus bundled IAA data utilities | `run_iaa_stage.py dataset-materialize` | `data-layout.md` |
| `pool_embed` | TAO data services | `run_deft_action.py` + selected platform | `mining.md`, `platform-execution.md` |
| `evaluate` | TAO CLIP plus bound parser | `run_iaa_stage.py eval-config`, platform action, parser | `clip-train-eval.md`, `metric-contract.md`, `platform-execution.md` |
| `gap_analysis` | bundled IAA gap analysis | `run_iaa_stage.py gap-analysis` | `gap-analysis.md` |
| `data_mining` | TAO text embedding/k-NN plus bundled IAA data utilities | two platform actions, then `mining-postprocess` | `mining.md`, `platform-execution.md` |
| `history_select` | bundled IAA history selection | `run_iaa_stage.py history-select` | `mining.md` |
| `visualize` | bundled IAA visualization plus optional TAO image embedding | `visualize-prepare`, platform actions, `visualize-finish` | `visualization.md`, `platform-execution.md` |
| `train` | bundled config/checkpoint helpers plus TAO CLIP | `train-config`, platform action, `publish-checkpoint` | `clip-train-eval.md`, `platform-execution.md` |
| `loop_stop` / report | bundled control scripts | commit, audit, renderer | `pipeline-and-state.md` |

The canonical workflow executes plain `clip train`, so there is no AutoML branch.
Never add `automl_policy`, `workflow`, or another orchestration key to a TAO
YAML schema.

## Platform action contract

Always prepare and finalize TAO through `run_deft_action.py`, and execute through
the selected platform skill. Never assemble an untracked native command. The
producer reads the platform, image, resource shape, mounts, exact argv, output
set, and credential-forwarding policy from immutable state and emits a
schema-valid spec bundle.

The following is a launch shape, not a literal copy/paste command; each stage
reference supplies the exact stable name, outputs, command, and independently
reconstructs credential arguments from immutable approval:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image <pyt|ds> \
    --stage-dir "$STAGE_DIR" --name <stable-stage-name> \
    "${HF_ARGS[@]}" \
    --fresh-output "$ABSOLUTE_EXPECTED_FILE" \
    -- <TAO command and arguments>
```

Repeat `--fresh-output` for every exact file whose freshness proves the action.
After preparation, follow `platform-execution.md`: reconcile interrupted
launch state; stage remote inputs with delete semantics and attest the exact
durable results scope; open the request-owned job-record; bind it while it is
still PENDING; then run the selected platform's four verbs, capture logs,
synchronize outputs, and finalize. A native exit zero without fresh, valid
output is failure; a file without successful matching job/action evidence is
not resumable evidence. Never submit when reconciliation reports an existing
bound job.

Action and host-adapter statuses carry a stable `attempt` value. Attempt 1 is
the initial call, attempt 2 is the single evidence-based correction, and the
producer refuses attempt 3. Keep the documented stable command names; changing
a name does not create a new retry budget.

Required stable names are `pool_embed`, `target_embed`, `knn`,
`viz_weak_embed`, `viz_mined_embed`, `viz_previous_embed`, `train`, and
`evaluate`. Logs and statuses live in the matching `--stage-dir`.

## Bundled adapter contract

`run_iaa_stage.py` is the only supported entry to bundled IAA host operations:

```text
dataset-materialize  create eval/val/pool splits and source_pool.parquet
gap-analysis         analyze the prior evaluation into iter_N/gaps
mining-postprocess   summarize k-NN and create uncapped/capped candidates
history-select       apply novel/replay budget and leakage check
visualize-prepare    contact sheets and image-embedding input parquets
visualize-finish     t-SNE from completed image embeddings
train-config         generate an iteration train YAML
publish-checkpoint   choose the best checkpoint and normalize it
eval-config          generate baseline or iteration eval YAML
iteration-summary    bind an iteration's gap/mining/checkpoint summary
```

Each command takes `--results-dir` and `--deft-config`; iteration commands also
take `--iter-num`, except `eval-config`, which takes `--iter-label`. Gap
analysis derives the iteration budget from immutable state. Do not recreate
these calls with inline Python.

## Path invariants

- Baseline state label `baseline` maps to `${RESULTS_DIR}/zs`.
- State label `iterN` maps to `${RESULTS_DIR}/iter_N`.
- Split files are exactly under `iaa_splits/`; source pool/embeddings are
  exactly under `embeddings/source/`.
- The action request maps `${RESULTS_DIR}` to `/results` and its approved
  absolute path, the dataset parent to `/data` and its approved absolute path,
  the immutable config to `/specs`, compatibility code to `/patches`, and the
  workspace cache to `/cache`. Every container platform must render all aliases.
- Container YAML and commands use `/results`, `/data`, and `/specs`; state and
  commit arguments use absolute host paths.
- Iteration evidence must come from that iteration. Never satisfy iterN with a
  prior iteration's checkpoint, parquet, status, or summary.

## Bounded recovery

After any nonzero command, inspect the last meaningful error block in its log.
Do not guess a different CLI. One corrected retry is allowed only when evidence
supports the correction.

### Interrupted canonical commit

If the audit reports `.deft_commit_transaction.json`, run no workload stage.
Restore the saved canonical state/log pair once:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/recover_commit.py" --results-dir "$RESULTS_DIR"
```

The recovery validates journal ownership, restores both files atomically, and
accepts only a valid nonterminal `IN_PROGRESS` or `FAILED` audit. Then run the
normal audit and follow its single `next_action`. Never edit or delete the
journal by hand.

### History write preceded commit

`history-select` records iteration N in `mining_selection_history.json` before
the outer stage transaction. If a process dies in that narrow window, state
still reports the stage uncommitted while the history entry and selection
outputs exist. Run:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/run_iaa_stage.py" history-select \
    --results-dir "$RESULTS_DIR" \
    --deft-config "$RESULTS_DIR/config/deft_config.yaml" \
    --iter-num "$N" --resume
```

Then run the audit and commit normally. The adapter verifies the saved
selection and leakage. If resume rejects the history or artifacts, hard-stop;
never remove/edit a history entry.

### GPU contention

Immediately before train, inspect selected GPU occupancy. If a launch fails
with OOM and occupancy changed after approval, wait for the selected GPUs to
be free and retry the same approved command once. Changing GPU IDs/count is a
parameter change and requires an updated approval summary. A second OOM on
otherwise free GPUs is terminal.

### Registry/network interruption

Retry a pull or public model download once only after connectivity or registry
state demonstrably changes. Never downgrade an image/model or switch sources
speculatively.

### Non-retryable failures

Checksum/rebuild failure, unsupported CUDA architecture, missing or malformed
split evidence, empty/schema-invalid mining output, eval leakage, cross-label
metric evidence, stale output, config hash mismatch, and state/log corruption
are hard stops. Preserve logs and provide the auditor's recovery statement.
