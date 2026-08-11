# Scripts, Stage Ownership, and Recovery

Despite this file's compatibility-preserving name, the IAA workflow does not
need a reporter agent or stage subagents. One agent runs a small linear flow
through deterministic scripts.

## Contents

- [Script interfaces](#script-interfaces)
- [Stage ownership](#stage-ownership)
- [Container launch contract](#container-launch-contract)
- [Bundled adapter contract](#bundled-adapter-contract)
- [Path invariants](#path-invariants)
- [Bounded recovery](#bounded-recovery)

## Script interfaces

| Script | Role | Runtime |
|---|---|---|
| `deft_python.sh` | select a Python per call and cap BLAS/OpenMP threads | shell |
| `prepare_deft_config.py` | copy templates into a run and apply approved values | IAA runtime |
| `init_deft_state.py` | validate config/metric inputs and create schema-v3 state once | IAA runtime |
| `audit_deft_run.py` | read-only state, transition, evidence, and artifact audit | IAA runtime (parquet/YAML validation) |
| `run_deft_container.py` | launch one pinned TAO container and atomically record log/status/fresh outputs | control Python |
| `run_iaa_stage.py` | expose bundled IAA host operations as named subcommands | IAA runtime |
| `iaa_deft/` | bundled IAA gap, mining, selection, visualization, config, and checkpoint implementation | imported only through `run_iaa_stage.py` |
| `commit_stage.py` | validate and atomically commit one stage | IAA runtime for parquet checks |
| `recover_commit.py` | roll back a journaled state/log commit interrupted by process death | IAA runtime (runs the audit) |
| `parse_iaa_metrics.py` | parse one exact CSV row/column into iteration-bound JSON | control Python |
| `render_deft_report.py` | audit and atomically render a deterministic HTML report | IAA runtime (its audit validates parquet) |
| `metric_contract.py`, `command_contract.py`, `checkpoint_contract.py`, `record_metric_result.py`, `log_stage.py` | internal validation/commit helpers | never invoke directly |

Use the control plane for container launch and metric parsing:

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
| `pool_embed` | TAO data services | `run_deft_container.py` | `mining.md` |
| `evaluate` | TAO CLIP plus bound parser | `run_iaa_stage.py eval-config`, container wrapper, parser | `clip-train-eval.md`, `metric-contract.md` |
| `gap_analysis` | bundled IAA gap analysis | `run_iaa_stage.py gap-analysis` | `gap-analysis.md` |
| `data_mining` | TAO text embedding/k-NN plus bundled IAA data utilities | two container launches, then `mining-postprocess` | `mining.md` |
| `history_select` | bundled IAA history selection | `run_iaa_stage.py history-select` | `mining.md` |
| `visualize` | bundled IAA visualization plus optional TAO image embedding | `visualize-prepare`, container launches, `visualize-finish` | `visualization.md` |
| `train` | bundled config/checkpoint helpers plus TAO CLIP | `train-config`, container launch, `publish-checkpoint` | `clip-train-eval.md` |
| `loop_stop` / report | bundled control scripts | commit, audit, renderer | `pipeline-and-state.md` |

The canonical workflow executes plain `clip train`, so there is no AutoML branch.
Never add `automl_policy`, `workflow`, or another orchestration key to a TAO
YAML schema.

## Container launch contract

Always launch TAO through the wrapper, not an assembled `docker run`. It reads
the image and standard mounts from state, forwards the existing `HF_TOKEN`
only with the explicit option below and without exposing its value, mounts the
compatibility patch, writes the full log, and records exact command status.

The following is a launch shape, not a literal copy/paste command; each stage
reference supplies the exact stable name, outputs, command, and independently
reconstructs credential arguments from immutable approval:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_container.py" \
    --results-dir "$RESULTS_DIR" --image <pyt|ds> \
    --stage-dir "$STAGE_DIR" --name <stable-stage-name> \
    "${HF_ARGS[@]}" \
    --fresh-output "$ABSOLUTE_EXPECTED_FILE" \
    -- <container command and arguments>
```

Repeat `--fresh-output` for every exact file whose freshness proves the
launch. The wrapper deletes only those non-symlink, results-scoped files before
launch, verifies they were recreated after launch, and records the exact argv
plus its digest. Pass its generated `<name>.status.json` to
`commit_stage.py`. A zero Docker exit without fresh, valid output is failure;
a file without successful matching command status is not resumable evidence.

The wrapper uses a deterministic Docker name, CID file, and nonblocking launch
lock. If its process dies while Docker continues, a later call inspects that
container read-only and refuses to overlap it. Wait for the named container to
exit; never kill it automatically. `HF_TOKEN` is not forwarded unless the
approved command needs it and `--pass-hf-token` is supplied explicitly.
Docker and host-adapter statuses also carry a stable `attempt` value. Attempt
1 is the initial call, attempt 2 is the single evidence-based correction, and
either wrapper refuses attempt 3. Keep the documented stable command names;
changing a name does not create a valid new retry budget.

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
- The wrapper mounts `${RESULTS_DIR}` as `/results`, the dataset parent as
  `/data` and at its absolute host path, the immutable config as `/specs`, and
  workspace cache as `/cache`.
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
