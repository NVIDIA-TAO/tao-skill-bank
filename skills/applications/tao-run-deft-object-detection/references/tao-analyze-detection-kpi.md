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

Do not hand-write it. Emit `analytics default_specs`, then apply the overlay —
`assets/overlays/kpi_analyze.yaml` holds every documented setting, so a field
cannot keep a TAO default just because nobody typed it:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/emit_default_spec.py \
  --stage kpi_analyze --ds-image "$TAO_DS_IMAGE" \
  --out "${RESULTS_DIR}/<phase>/kpi_spec.yaml"

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
  --spec "${RESULTS_DIR}/<phase>/kpi_spec.yaml" \
  --apply-workflow-defaults <skill_root>/assets/overlays/kpi_analyze.yaml \
  --set results_dir="${RESULTS_DIR}/<phase>/kpi" \
  --set visualize.tag=<phase> \
  --set data.kpi_sources="[{image_dir: <KPI images>, ground_truth_ann_path: <KPI labels>, inference_ann_path: ${RESULTS_DIR}/<phase>/inference/labels}]" \
  --set data.mapping=<class-mapping YAML> \
  --set data.image_dir=<KPI images> \
  --set data.ann_path=<KPI labels> \
  --report-json "${RESULTS_DIR}/<phase>/kpi_overrides.json" \
  --require-no-mandatory
```

Everything in the overlay is fixed; everything in `--set` is a path or a phase
label. A `--set` naming a key the overlay already set is rejected — that
collision is the drift the split exists to prevent.

The resulting `kpi:` block:

```yaml
kpi:
  iou_threshold: 0.5
  conf_threshold: 0.0        # the full PR curve — see below
  num_recall_points: 11      # 11-point interpolated AP; 101 selects COCO-style
  ignore_sqwidth: 40         # TAO emits 0, which scores smaller boxes
  filter: false
  is_internal: false
```

### These values are the leaf skill's defaults too

`tao-analyze-detection-kpi` ships the same `conf_threshold: 0.0`, `num_recall_points: 11`
and `ignore_sqwidth: 40`, so delegating to it and applying this overlay agree. Apply the
overlay anyway: the agreement is a fact about the current versions, not a guarantee, and
a spec that states its own scoring settings is auditable after the fact.

`conf_threshold: 0.0` is safe on the pinned image. An undetected ground-truth box used to
be carried as `(t=1, p=0.0)` against a `p >= conf_threshold` check, so zero scored every
missed box as a true positive; unmatched ground truth now uses a `-1.0` sentinel and lands
in FN at any threshold. On a build predating that fix, use a small positive value.

## Scoring at more than one confidence threshold

Inference runs at `0.0`, so the labels carry every detection and this stage can score them
at any threshold without re-running inference — re-apply the overlay with
`--set kpi.conf_threshold=<t> --allow-workflow-default-override` and a distinct
`results_dir` per point, since `kpi_calc.csv` is written unconditionally and a shared
directory keeps only the last. `conf_threshold` selects from what inference wrote; it cannot
recover a box inference dropped.

Only the `0.0` result is comparable to the loop's reported mAP. The loop itself scores once,
at `0.0`; sweeps are a diagnostic.

**`input_format` is uppercase here.** `analytics kpi_analyze` accepts only `KITTI` or `COCO`. This differs from `gap_analysis object_detection` in the same container, which takes lowercase `kitti` / `coco`. The two stages in this loop therefore spell the same format differently — that is expected, not a typo to "fix".

**Keep `visualize.platform: local`.** The `wandb` path needs credentials and a reachable endpoint; a loop stage should not depend on either.

**`is_internal` must stay `false`.** Setting it true drops every class except `person` and appends a `Summary` row, silently changing what the report means.

## Narrow the mapping to the target classes first

The user's mapping usually names more classes than the run targets. Scoring an
untargeted class adds a constant-0 AP row and averages it into the mAP, so the number
stops being comparable with any run that did not:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/prepare_mapping_for_kpi_analyze.py \
  --mapping "<the user's class mapping>" \
  --target-classes "<comma-separated target classes>" \
  --out "${RESULTS_DIR}/<phase>/kpi/mapping.yaml"
```

Pass the narrowed file as `data.mapping`, not the user's original.

## Invocation

**Launch it detached.** This stage runs ~22 minutes on a 14k-image KPI set and the mAP
appears *only* on stdout. A foreground `docker run | tee` loses the number if the client
dies, while the container keeps running — name the container, redirect to the log, and
wait on the log with `await_stage.py`:

```bash
docker run -d --name "deft_${PHASE}_kpi" ... > /dev/null
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/await_stage.py \
  --artifact "${RESULTS_DIR}/<phase>/kpi/kpi_calc.csv" --timeout-sec 5400
docker logs "deft_${PHASE}_kpi" > "${RESULTS_DIR}/<phase>/kpi/kpi_analyze.log" 2>&1
```

Pass `--gpus all` even though the scoring itself is CPU-bound: the TAO launcher calls
`nvidia-smi -L` unconditionally at startup, so omitting it fails with
`FileNotFoundError: 'nvidia-smi'` before any work begins.

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" \
  analytics kpi_analyze -e "$KPI_SPEC" 2>&1 | tee "${RESULTS_DIR}/<phase>/kpi/kpi_analyze.log"
# mAP appears only on stdout, so the tee is required — but a pipeline reports the
# exit status of `tee`, not of the container. Without one of these, a failed
# kpi_analyze looks like a success:
#   set -o pipefail   (before the pipeline), or
#   [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "kpi_analyze failed"; exit 1; }
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
