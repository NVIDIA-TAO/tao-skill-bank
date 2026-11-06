# Dual object-detection gap analysis

Read the `tao-analyze-gaps-od-map` skill and its structured metadata before
launching either gap job. The data-services action owns matching; do not replace
it with an application-local IoU implementation.

`prepare_deft_od_aoi_measurement.py` projects the frozen KPI COCO to KITTI,
including empty label files for clean KPI images, and emits:

```text
kpi_inference.yaml
test_inference.yaml
gap_loose.yaml
gap_strict.yaml
measurement_manifest.json
```

Run KPI inference first. Its KITTI label directory must match the
`--kpi-predictions` path frozen into both gap specs. Then submit the loose and
strict specs independently through the selected platform. They may run in
parallel after KPI inference completes.

Each gap job must reach `COMPLETE` and publish the leaf's required artifacts.
Routing consumes `box_gaps.parquet`, requiring `gap_type`, `filepath`,
`bbox`, and `best_iou`. Empty parquets are valid. Do not route from the
weak-image list.

The class map has exactly two lines: `background`, then `defect`. KPI image
stems must be unique for KITTI projection. Test inference remains report-only
and is never an input to gap routing or checkpoint selection.

AnomalyGen 1.2 synthesis uses the single `tao-generate-anomalies` skill:
`prepare --source od-gaps` for inputs/AMP/finalize, `finetune_only` for one-time
bootstrap, and `inference_only --pseudo-label` for each finalized group. Resolve
`images.metropolis_sdg.anomalygen`; mount the complete selected-stage cache at
`/workspace/cdf-anomalygen/checkpoints`. Keep the prepared manifest and its
hash-bound artifacts. Emit COCO with application `scripts/emit_sdg_coco.py`
using `--runs-manifest` and `--prepared-manifest`; admission uses the converter's
`pseudo_labels/coco_annotations_od_defect.json`. Pass caller `--exclude-stems`
at preparation, generation and conversion, and set `synthesis.exclude_stems`
for admission. The preparation CLI option overrides the frozen YAML exclusion path; omission uses the application path. The original source/donor provenance
remains in the prepared bundle. Existing NN improvement, box-quality, fraction,
whole-FN planning and split/KPI policies are unchanged. Fully blocked output is
COMPLETE with zero synthetic contribution; preserve blocked IDs without retry
or top-up. Guardrail capability limitations remain explicit.
