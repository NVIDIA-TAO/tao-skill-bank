# Gap Analysis

On every platform, dispatch this adapter as the allowlisted zero-GPU
`gap_analysis` action; never execute the mutator on the controller.

Run gap analysis only after an evaluation fails its KPI gate (or has no target)
and another iteration remains.

The stage is committed under the label whose evaluation it analyzes, but its
artifact belongs to the iteration it feeds:

| Evaluated label | `--iter-label` at commit | Feed iteration / output |
|---|---|---|
| baseline | `baseline` | `iter_1/gaps/kpi_gaps.parquet` |
| iterN | `iterN` | `iter_<N+1>/gaps/kpi_gaps.parquet` |

The platform action label is the feed iteration (`iter1`, `iter2`, ...), not
the evaluated label. Thus baseline gap analysis is dispatched with
`--label iter1`, while its subsequent `commit_stage.py` call still uses
`--iter-label baseline`. This binds the adapter command, stage directory, and
fresh output to the iteration it creates.

Do not run a gap stage after a passing KPI or after the final allowed
iteration. The transition validator rejects both paths.

## Command

Let `FEED_N=1` for baseline, otherwise the evaluated iteration plus one:

```bash
STAGE_DIR="$RESULTS_DIR/iter_$FEED_N/gaps"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
    --results-dir "$RESULTS_DIR" --image ds \
    --stage-dir "$STAGE_DIR" --name gap_analysis \
    --fresh-output "$STAGE_DIR/kpi_gaps.parquet" -- \
    python3 /iaa-runtime/run_iaa_compute.py gap_analysis \
      --results-dir /results --label "iter$FEED_N"
```

Execute and finalize this request through `platform-execution.md`.

The adapter resolves the exact preceding eval directory through the bundled IAA runtime,
uses the configured IAA eval split and gap metric, applies caption-diversity
limits, and writes the iteration-scoped parquet. It also maintains
`caption_selection_history.json` using the runtime's idempotent iteration
semantics.

For Airflow orchestration with SLURM compute, the bridge synchronizes this
run-wide mutable ledger separately from the absent-before-run parquet output.
It verifies the remote size and digest, atomically fetches the ledger, archives
the prior local ledger before a later iteration replaces it, and writes
`gap_analysis.history-sync.json`. If the compute action completed before this
specialized synchronization was recorded, use the bridge's bounded
`recover-gap-history-sync` operation; do not rerun the completed action or copy
the ledger manually.

The output must be a non-empty parquet with `filepath`, `text`, and
`weak_attribute`. If the metric gate is unmet but no gap row can be produced,
hard-stop: the query type, metric, or slicing contract is inconsistent. Do not
mine the full pool as a fallback.

Commit:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label "$EVALUATED_LABEL" \
    --stage gap_analysis \
    --gaps-parquet "$RESULTS_DIR/iter_$FEED_N/gaps/kpi_gaps.parquet" \
    --caption-history "$RESULTS_DIR/caption_selection_history.json" \
    --gap-analysis-status \
      "$RESULTS_DIR/iter_$FEED_N/gaps/gap_analysis.status.json" \
    --summary "$EVALUATED_LABEL gap analysis prepared iteration $FEED_N"
```

Run the audit. The only legal next action is `iter$FEED_N/data_mining`.
If a process ended after the caption ledger write but before this commit, the
audit reports the one in-flight feed as a warning. Rerun the same idempotent
adapter once and commit; never edit the ledger manually.
