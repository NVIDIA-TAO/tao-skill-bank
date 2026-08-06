# DEFT OD — KPI Analyze Stage Overlay

Layers loop conventions on top of `tao-skill-bank:tao-analyze-detection-kpi`. Read that skill's `SKILL.md` for the full field reference and pitfalls.

## When to invoke

Twice per phase boundary:

- **Baseline**: after the zero-shot `inference`, as the final baseline stage.
- **Iteration N**: after `inference`, as the final stage of the iteration.

## No pre-KPI label filtering

The reference pipeline optionally filtered inference labels before scoring — exclude-ROI (masking out a region of the frame) and exclude-PIC (dropping person-inside-car detections) — and its `exp7` configuration ran with PIC enabled.

**This loop applies neither.** Both are dataset-specific label cleanups for one intersection deployment, not general OD loop mechanics, so `kpi_analyze` scores the raw inference labels as written.

The practical consequence: **mAP here will not match `exp7`'s numbers**, because exp7's persons-inside-cars were removed from ground truth before scoring. The trend across iterations is still valid — it's a consistent measurement — but do not compare the absolute values against the reference experiment's.

## mAP is reported, not gated

This loop does **not** early-exit on a metric target. `kpi_analyze` records where the model stands so the report can show the trend across iterations; a regression does not stop the loop and does not trigger a retry. The loop runs until `max_iterations` or a hard-stop gate.

Do not add metric-based stopping logic here. If the user wants target-gated stopping, that is a change to the workflow contract, not something to improvise mid-run.

## Spec

Write per-phase under `${RESULTS_DIR}/<phase>/kpi/kpi_analyze.yaml`:

```yaml
data:
  input_format: KITTI        # UPPERCASE — see below
  kpi_sources:
  - image_dir: <absolute path to the KPI image dir>
    ground_truth_ann_path: <absolute path to KPI ground-truth labels>
    inference_ann_path: <absolute path to ${RESULTS_DIR}/<phase>/inference/labels>
  mapping: <absolute path to the class-mapping YAML>
visualize:
  platform: local            # avoids a wandb dependency inside the loop
  tag: <phase>
kpi:
  iou_threshold: 0.5
  conf_threshold: 0.3
  num_recall_points: 11
  ignore_sqwidth: 40
  filter: false
  is_internal: false
results_dir: <absolute path to ${RESULTS_DIR}/<phase>/kpi>
```

**`input_format` is uppercase here.** `analytics kpi_analyze` accepts only `KITTI` or `COCO`. This differs from `gap_analysis object_detection` in the same container, which takes lowercase `kitti` / `coco`. The two stages in this loop therefore spell the same format differently — that is expected, not a typo to "fix".

**Keep `visualize.platform: local`.** The `wandb` path needs credentials and a reachable endpoint; a loop stage should not depend on either.

**`is_internal` must stay `false`.** Setting it true drops every class except `person` and appends a `Summary` row, silently changing what the report means.

## Invocation

KPI analysis is CPU-only — do not request a GPU.

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" \
  analytics kpi_analyze -e "$KPI_SPEC" 2>&1 | tee "${RESULTS_DIR}/<phase>/kpi/kpi_analyze.log"
```

Tee the log: the aggregate **mAP is printed to stdout only** and is not written into `kpi_calc.csv`. Parse it from the log line `mAP: <value>` and record it in state; otherwise the trend across iterations cannot be reported.

## Outputs

| Artifact | Path |
|---|---|
| Per-class metrics | `${RESULTS_DIR}/<phase>/kpi/kpi_calc.csv` |
| PR curve plot | `${RESULTS_DIR}/<phase>/kpi/` |
| Captured log (mAP source) | `${RESULTS_DIR}/<phase>/kpi/kpi_analyze.log` |

`kpi_calc.csv` columns: `Sequence Name`, `TP`, `FP`, `FN`, `TN`, `Pr`, `Re`, `Acc`, `AP`.

`Sequence Name` is derived as the **second-to-last** component of `image_dir`, not configured. Keep `image_dir` free of a trailing slash and laid out so that component is the sequence identifier you want; two sources can otherwise collide under one name.

## Commit

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label "<phase>" --stage kpi_analyze \
  --kpi-csv "${RESULTS_DIR}/<phase>/kpi/kpi_calc.csv" \
  --kpi-log "${RESULTS_DIR}/<phase>/kpi/kpi_analyze.log" \
  --map-value "<parsed mAP>" \
  --summary "kpi: mAP=<value>"
```
