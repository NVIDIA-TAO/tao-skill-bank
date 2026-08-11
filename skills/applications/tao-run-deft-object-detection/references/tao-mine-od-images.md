# DEFT OD — Mining Stage Overlay

Layers loop conventions on top of `tao-skill-bank:tao-mine-od-images`. Read that skill's `SKILL.md` for the full field reference.

## When to invoke

Iteration stage 3, after `embed`. Mines the source pool for images resembling this iteration's weak set.

## Inputs from state

| State field | Description |
|---|---|
| `iterations.iter${N}.embeddings_parquet` | This iteration's `weak_images_embeddings.parquet` — the mining **target**. |
| `config.source_pool_embeddings` | Pre-embedded source pool — the mining **source**. Must use the same encoder as the target. |
| `iterations.iter$((N-1)).exclude_parquet` | Previous cumulative mined set. **Absent on iteration 1.** |
| `config.source_detection_file` | Source-pool COCO JSON — required for `class_stratified`. |
| `config.target_detection_file` | KPI COCO JSON — required for `class_stratified`. |
| `config.rare_class_list` | Comma-separated rare classes, e.g. `"person,bicycle"`. |

## `exclude_path` must be `null` on iteration 1

This is a hard failure, not a warning. TAO DS `load_datasets` raises `FileNotFoundError` when `exclude_path` is set but is not a file — and on iteration 1 no cumulative parquet exists yet. Pointing at the not-yet-written path kills the stage before any mining happens.

- **Iteration 1**: emit `exclude_path: null`.
- **Iteration N > 1**: emit the absolute path to `${RESULTS_DIR}/iter$((N-1))/mined_cumulative.parquet`, and confirm the file exists before writing the spec.

> The reference pipeline's miner silently skipped a missing exclude file; TAO DS raises instead. Its KFP wrapper only worked because it pre-guarded with `os.path.exists`. Reproduce that guard, not the bare path.

## `detection_format` is never inferred

TAO DS raises `detection_format is required (coco or kitti) when a detection file is provided` rather than guessing. Set it explicitly to `coco` and supply COCO JSONs — a KITTI label directory passed as a detection file would be misparsed.

Note this stage uses **COCO** detection files while `gap_analysis` in the same loop reads **KITTI** inference labels. That split is correct: they are different inputs to different steps.

## Rare classes are derived here, and should not be

`class_stratified` allocation needs to know which classes are rare. The loop derives that
from the prepared pool: `pool_report.json` records `annotations_by_class`, and any target
class holding a below-mean share of the pool's annotations is treated as rare. Scarcity in
the **pool** is the right signal — stratified allocation exists so that classes the pool holds
few of still get their share of the budget, while a class the pool is full of will be found by
global allocation anyway.

> **Improvement to make in TAO DS:** this belongs in `tmm unique_neighbor_matching`, not in a
> caller's glue code. The mining action already reads the source pool and the
> `source_detection_file` COCO, so it holds everything needed to compute the class
> distribution itself — every caller wanting stratified allocation currently has to
> re-derive the same thing from outside, and each will pick a slightly different rule
> (median vs mean vs a fixed percentile), so the same pool yields different allocations
> depending on who asked. Exposing `rare_class_list: auto`, with the threshold rule owned
> and documented by TAO DS, would make the behaviour reproducible across callers and remove
> the need for a pool report to travel alongside the pool.
>
> Until then, `rare_class_list` stays an explicit input, and the loop's derivation is
> reported as a warning naming both the chosen classes and the counts behind them, so the
> decision is visible rather than silent.

## Spec

Write per-iteration under `${RESULTS_DIR}/iter${N}/mining/unique_neighbor_matching.yaml`:

```yaml
source_path:            <config.source_pool_embeddings>
target_path:            <iterations.iter${N}.embeddings_parquet>
output_dir:             <absolute path to ${RESULTS_DIR}/iter${N}/mining>
desired_unique_count:   <from prepare_budget_for_mining.py>
allocation_policy:      class_stratified     # or global — see below
distance_metric:        euclidean
candidate_expansion_factor: 5
source_embedding_column: embedding
target_embedding_column: embedding
source_filepath_column: filepath
target_filepath_column: filepath
exclude_path:           null                 # iteration 1 ONLY; else the prev cumulative path
source_detection_file:  <config.source_detection_file>
target_detection_file:  <config.target_detection_file>
detection_format:       coco
rare_class_list:        "person,bicycle"
save_embeddings:        false
visualize:              false
```

**Allocation policy.** Use `class_stratified` when the user supplied rare classes — budget is apportioned by target class ratio using largest-remainder, with a non-rare fallback pool, and each source is owned by the first class that claims it. Use `global` when no rare classes are configured. The reference pipeline's `class_balanced` mode maps to `class_stratified`.

**`candidate_expansion_factor` is not the loop's iteration count.** It seeds the miner's internal candidate-pool growth (starting at 5, widening each internal retry, capped at 8 internal passes). It is unrelated to `max_iterations`.

## Invocation

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" \
  tmm unique_neighbor_matching -e "$MINE_SPEC"
```

## Outputs

| Artifact | Path |
|---|---|
| `final_unique_files.parquet` | `${RESULTS_DIR}/iter${N}/mining/final_unique_files.parquet` |
| `summary.json` | `${RESULTS_DIR}/iter${N}/mining/summary.json` |

`final_unique_files.parquet` is a single column named by `source_filepath_column` (`filepath`). It feeds the `stage` step.

Read `coverage_pct` from `summary.json`:

- `retrieved_unique_count == 0` with weak images present → **hard stop**. The source pool has nothing left to give, or the exclude set has consumed it.
- `coverage_pct < 50` → warn in the iteration summary; the pool is running dry and later iterations will add little.
- Otherwise proceed.

## Commit

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label "iter${N}" --stage mine \
  --mining-output "${RESULTS_DIR}/iter${N}/mining/final_unique_files.parquet" \
  --mining-summary "${RESULTS_DIR}/iter${N}/mining/summary.json" \
  --summary "mined <retrieved>/<desired> unique images (<coverage_pct>% coverage)"
```
