# Offline analysis tools

These opt-in tools do not read or update DEFT state, change selection policy,
or replace the official KPI evaluator. Run them separately from the loop.
Existing analysis data outputs are never overwritten; select a fresh output
directory/name. Full-pool G may re-render its scheduling script and rerun an
incomplete chunk; checksum-verified complete chunks are retained.

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

## E: disjoint validation panel

```bash
python3 "$SKILL_ROOT/scripts/build_validation_panel.py" \
  --input "$DEFT_ROOT/nvpaw/outputs/test_full.jsonl" \
  --benchmark "$WORKSPACE/annotations/benchmark.jsonl" \
  --proxy "$WORKSPACE/annotations/proxy_kpi.jsonl" \
  --media-root "$WORKSPACE" --per-task 500 --seed 17 \
  --output-dir "$DEFT_ROOT/nvpaw/outputs"
```

Writes only `validation_panel.jsonl`, `validation_panel_profile.json` (using A,
with an embedded `panel_manifest`), and `PANEL_MANIFEST.md`. Input hashes, panel
hash, exclusions, excluded families, target/realized shares, and shortages for
every benchmark stratum are recorded. The tool never reads predictions.

Exclusions use id **or any image path**, including either side of a pair,
against all Benchmark and Proxy records. Paths normalize separators and dot
segments; `--media-root` joins relative paths to match canonical absolute paths
without opening media. A stable seed/id hash reservoir limits retained
candidates to the requested per-stratum quota and makes selection independent
of input ordering; duplicated candidate IDs fail closed.

The target is 500 rows per task, allocated by benchmark shares over
dataset family × empty status × count × largest-box size. Missing or exhausted
families/strata are not replaced by unrelated cells. Per task, the effective
family cap is `max(0.35, largest benchmark target share among eligible families)`;
eligibility requires usable supply in the requested strata after exclusions.
Shares keep the full benchmark-task denominator, not just surviving families.
Thus a benchmark dominated by one of two eligible families is not emptied by
the base 35% cap. `family_cap_effective`, `family_cap_relaxed`, and
`family_cap_reason` appear in each task's JSON and the Markdown table. Relaxed
caps round their integer limits up (43.7% of 500 permits 219 rows); unrelaxed
caps keep the original floor calculation and byte-identical seeded selections.
`family_cap_rounding` records that distinction in JSON. When supply is short,
shrink to the largest available total satisfying the effective cap against the
**realized** task size, then allocate closest to target shares within each
bounded family. Shortages always compare to the original
500-per-task request, not the shrunken budget. Output records retain their
native messages and image controls; this tool does not freeze or install the
panel into the loop. The operator reviews and freezes its hash separately.

## G: pool residual scoring (sample default, full pool optional)

Both `prepare` modes only write CPU-side analysis inputs and execution plans.
Neither calls Slurm, starts a container, downloads assets, or loads a model.
`score` and `merge` only score existing predictions on CPU. The explicit
`run-chunk` entrypoint is for the operator's allocation, not preparation: it
delegates to `cfw_jsonl_runtime.py` and `merge_cfw_prediction_shards.py`.

Set `DEFT_ROOT` to the project root, `WORKSPACE=$DEFT_ROOT/workspace`, and
`SKILL_ROOT` to this application skill's directory in the matfix checkout (or
an immutable installed copy of this revision). `CFW_IMAGE_DIGEST` is the
operator-approved `nvcr.io/...@sha256:<64 hex>` image, and `MODEL_PATH` is its
already staged, action-ready zero-shot model or checkpoint directory. A
Framework DCP must first use the existing model-owned export/handoff; these
tools do not convert checkpoints. All paths must have the same absolute
locations inside the eventual container. CPU scoring requires PyArrow; no
GPU dependencies are needed until the operator executes the evaluation plan.

### Sample: capped, reproducible, cheap default

```bash
python3 "$SKILL_ROOT/scripts/score_pool_residual.py" prepare --mode sample \
  --input "$WORKSPACE/annotations/mining.jsonl" \
  --model-path "$MODEL_PATH" --media-root "$WORKSPACE" \
  --image "$CFW_IMAGE_DIGEST" --per-group 2000 --seed 17 --batch-size 1 \
  --output-dir "$RESIDUAL_DIR"
```

Writes `pool_sample.jsonl`, `pool_sample_manifest.json`, `evaluate_pool.toml`,
and `pool_scoring_plan.json`. Stable seed/id hash reservoirs select at most
2,000 rows per task/dataset (pair-aware), independently of input order; duplicate
IDs fail closed. Native row dictionaries/messages/image controls are preserved.
G supports the six benchmark tasks plus the pool's Component Count task;
segmentation is excluded. Unknown non-segmentation tasks fail closed in full
mode instead of silently reducing whole-pool coverage.
The manifest binds input/sample hashes, row counts, configuration, and model
path. The descriptor flags samples exceeding the approximate 60K-row budget
for operator review rather than silently truncating a dataset.

After the normal platform launch gate, the operator executes the descriptor's
`command` then `merge_command` in one approved eight-GPU allocation, with a
pre-staged image and data. This reuses native task/length-sorted evaluation,
1,024 generation tokens, eight torchrun workers, and the existing exact-coverage
shard merger. The sample descriptor requests interactive and 03:59:00.
After `predictions.jsonl` is complete:

```bash
python3 "$SKILL_ROOT/scripts/score_pool_residual.py" score \
  --sample "$RESIDUAL_DIR/pool_sample.jsonl" \
  --predictions "$RESIDUAL_DIR/predictions.jsonl" \
  --checkpoint "$MODEL_PATH" \
  --evaluator "$WORKSPACE/eval/calculate_f1_metrics.py" \
  --output-dir "$RESIDUAL_DIR"
```

### Full: resumable array, bounded resources, whole-pool merge

Use a fresh `FULL_RESIDUAL_DIR`. `CFW_SQSH` must be an existing, operator-verified
SquashFS of `CFW_IMAGE_DIGEST`; the renderer checks its magic and never falls
back to a registry pull inside a GPU allocation. `CFW_MOUNTS` is the explicit
Pyxis comma-separated source:target mount list covering **all** image roots
(including Yi-Cheng's paths outside this workspace), the model, skill code,
and analysis output. For a shared Lustre deployment this may be
`/lustre:/lustre`; the operator must verify the paths on the target cluster.

```bash
python3 "$SKILL_ROOT/scripts/score_pool_residual.py" prepare --mode full \
  --input "$WORKSPACE/annotations/mining.jsonl" \
  --model-path "$MODEL_PATH" --media-root "$WORKSPACE" \
  --image "$CFW_IMAGE_DIGEST" --sqsh "$CFW_SQSH" \
  --container-mounts "$CFW_MOUNTS" --partition interactive --constraint h100 \
  --chunks 16 --concurrency 8 --cpus-per-task 64 --batch-size 1 --seed 17 \
  --output-dir "$FULL_RESIDUAL_DIR"
```

`--partition`, the site-specific H100 `--constraint`, CPU request and optional
`--account` are operator-controlled. The generated `pool_array.sbatch` has
`--array=0-15%8`, one exclusive node/eight GPUs per element, and a fixed
03:50:00 limit. There is no cross-node collective: these are independent
chunks, not a sixteen-node distributed model. The script requires `TAO_JOB_ID`
from the normal operator-owned preflight/record-before-submit workflow; it does
not open a record or submit itself. Preflight the cached image's runtime
dependencies (including the descriptor's OneLogger contract), model, all image
mounts, and partition/GPU feature names before submitting. No credentials are
embedded; offline model loading and the no-home-mount convention are preserved.

Full preparation streams into balanced source-order round-robin chunks and
sorts each chunk using the existing evaluator batching key. It preserves every
eligible ID once and records exact segmentation exclusions, without hardcoding
the operator's current row counts. `pool_full_manifest.json` binds the input,
model path, code hashes, each chunk's source/config/plan hashes, and row counts.
Each `chunks/NNNN/` directory contains `source.jsonl`, `evaluate_pool.toml`,
`plan.json`, then operator-produced rank shards and `predictions.jsonl`.

Only full, canonical prediction coverage produces `COMPLETE.json`, bound to
the source, config, plan, checkpoint path, output checksum, and row count.
Re-running the identical `prepare --mode full` command re-renders only the
array's unfinished indices; it does not rewrite finished chunks. A partial or
corrupted output is rerun **whole**, with all eight ranks; stale shards do not
constitute completion. `run-chunk` rechecks completion under a per-chunk lock
and preserves failed child exit codes. Changing input/model/code/semantic
parameters requires a fresh output directory. Scheduling options may change
on re-render; the array script is authoritative for scheduling. Do not
re-render or merge while another process is publishing outputs to that root.

After the operator's array finishes, run the CPU merge:

```bash
python3 "$SKILL_ROOT/scripts/score_pool_residual.py" merge \
  --output-dir "$FULL_RESIDUAL_DIR" \
  --evaluator "$WORKSPACE/eval/calculate_f1_metrics.py"
```

Merge requires every chunk's verified completion, concatenates canonical
predictions, checks globally unique IDs and full coverage, and produces one
`predictions.jsonl`, one `pool_residual.parquet`, `residual_profile.json`, and
`RESIDUAL_REPORT.md`. It processes one chunk at a time, spools scores to disk,
and writes parquet in 8,192-row groups; it does not retain a million native
messages in RAM. Label-pattern review statistics span all chunks. The rough
operator estimate is 31 node-hours / about four hours at eight concurrent
nodes, queue permitting; this tool does not promise that throughput.

### Scoring and consumer contract

Parquet schema version `nvpaw_pool_residual_v1` includes string `id`, `task_type`,
`dataset`, `error_type`, `size_bin`, `count_bin`, `pair_kind`, `checkpoint`,
`scored_at`; double `row_score`; int64 `gt_boxes` (the GT count),
`matched_boxes`, `false_positives`, `false_negatives`; boolean `parse_ok`,
`is_residual`, `review_label_noise`; nullable double `prediction_confidence`;
and review/label-pattern strings. All scored rows are retained, keyed uniquely
by id; consumers filter `is_residual` instead of assuming a failures-only file.
Parquet metadata and the profile/report record provenance including evaluator
code and prediction SHA-256. `score`/`merge` accept `--evaluator-sha256` to
require a specific evaluator revision. The checkpoint field identifies the
staged model path, not a cryptographic hash of its weights: use immutable model
directories and operator-recorded checkpoint evidence.

**Component Count extension:** the authoritative evaluator has no count-row
metric, but the full-pool request excludes only segmentation. G therefore
scores these rows by exact nonnegative integer equality, using `wrong_class`
for integer mismatches and `parse_failure` for noninteger answers. This is
explicitly recorded as an analysis-only extension, not an official KPI. Nullable
int64 `gt_count`/`predicted_count` carry the scalar counts; `gt_boxes=0` and
size/count bins are `NA` because no boxes exist. G augments A's failure profile
with count-task row/dataset cells; A and E retain their original six-task
contracts. Count values are never reinterpreted as ground-truth boxes.

Canonical classification parsers and strict threshold-aware one-to-one box
matching come directly from the supplied authoritative evaluator. Coordinates
stay xyxy at scale 1.0; detection labels are ignored, just as in official F1.
Exact classification sets score 1, mismatches 0. Detection correctness requires
all GT and prediction boxes matched at **IoU > 0.5**, or a parseable explicit
empty on empty GT. Incorrect detection score is TP/(TP+FP+FN), penalizing both
misses and extras; parse failures score 0. This diagnostic is **not** F1.
Error precedence is parse_failure, correct, empty_on_positive, boxes_on_empty,
misaligned, missed_boxes, extra_boxes (classification uses wrong_class).
Misaligned means additional one-to-one matches become possible at IoU >0.3
versus >0.5, avoiding double-counting an already matched box. FP/FN columns
retain overlapping evidence even when one primary error is selected.

Failure geometry/dataset profiles reuse A unchanged. The report shows
task/dataset row accuracy and error mix. The native runtime emits **no
confidence**, so the tool never claims the model is confidently correct about
a disputed label. It offers human-review hints only: optional supplied
confidence >=0.9 on parseable disagreements, or >=95% consistent alternative
classification predictions among at least 20 rows sharing task, dataset,
GT labels and option meanings. A consistency hint is not calibrated confidence
or proof of a bad label. No labels, training data, or loop state are modified.

G's CPU tests load the real evaluator at the workspace path above; set
`DEFT_EXACT_EVALUATOR` when running the suite with a different local copy.
