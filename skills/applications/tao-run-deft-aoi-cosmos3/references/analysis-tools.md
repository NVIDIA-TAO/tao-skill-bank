# Offline analysis tools

These opt-in tools do not read or update DEFT state, change selection policy,
or replace the official KPI evaluator. Run them separately from the loop.
Existing outputs are never overwritten; select a fresh output directory/name.

## A: annotation target profile

```bash
python3 "$SKILL_ROOT/scripts/build_target_profile.py" \
  --input "$WORKSPACE/annotations/benchmark.jsonl" \
  --output-dir "$ANALYSIS_DIR" --name benchmark
python3 "$SKILL_ROOT/scripts/build_target_profile.py" \
  --compare "$ANALYSIS_DIR/benchmark_profile.json" "$ANALYSIS_DIR/pool_profile.json" \
  --budget 3000
```

Writes `<name>_profile.json` and `<name>_profile.md`; comparison prints a
Markdown cell table with requested, available, achievable, and shortage rows.
Profiles count rows/exposures, including exact repetitions in training files.
Only the six classification/detection tasks are profiled; other tasks are
counted in `ignored_task_rows`.

The geometry/bin logic follows the operator's `/tmp/box_profile.py`: effective
side is `sqrt(width * height)` with left-inclusive edges 16/33/66/130/260
(the compatibility label `>260` includes 260). Coordinates are already in
[0,1000], so relative area is divided by **1,000,000**, not 1024². Quantiles
use linear interpolation over all boxes, without the reference's first-200K
truncation. Invalid GT fails closed instead of silently becoming empty.
Each row contributes one cell: task, empty status, count bin, largest-box size
bin, pair kind, and dataset. Empty/classification size is `NA`; classification
count is `NA`, and its label histogram records raw canonical GT answers.
Classification empty means explicit `[]`/`{}` or the canonical negative direct
BCQ answer. All shares in JSON are fractions, not percentages.

Generated-edit path markers take precedence over augmented-view markers, then
identical paths and different photos. Dataset lookup prefers any ordered pair
path's `/NVPAW_pair/<sub>/` as `pair:<sub>` over `/datasets/<name>/` and finally
the row's `dataset` field. Comparison aggregates away dataset and uses
largest-remainder integer budgets over the requested five-dimensional cells;
shortage cells are not backfilled from unrelated cells.
