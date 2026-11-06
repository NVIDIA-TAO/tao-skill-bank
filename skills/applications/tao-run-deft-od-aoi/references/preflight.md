# Preflight and launch review

Complete this gate before creating a job record or submitting any action.

## Contract review

- Select an installed supported platform and run its preflight.
- Confirm the four normalized COCO roles and their resolved image roots.
- Confirm the trainable RT-DETR base checkpoint and maximum iterations.
- Confirm KPI is used for selection and test is report-only.
- Resolve the RT-DETR, Data Services, SigLIP embedding, mining, and optional
  AnomalyGen 1.2 images from their owning skills.
- Confirm GPU shape, storage mappings, runtime estimate, and durable
  `results_dir`.
- Confirm synthesis is disabled or has exact pixel masks, a defect
  specification, a reference pool with at least one supported image under
  `TEXTURE/clean_image`, Cosmos3-Nano assets, and either existing task weights
  or complete one-time fine-tuning routes.

Do not create output directories that actions require to be absent. Existing
DEFT results are resumed only through their committed
`deft_state.json`; never reinitialize them.

## Read-only validation

Before launch:

1. Validate all local source and checkpoint paths.
2. Run `init_deft_od_aoi.py` only after the approved policy has been written
   to a new location.
3. Inspect every emitted YAML as nested dictionaries; reject dotted keys at the
   container boundary.
4. Verify the class map is `background` followed by `defect`.
5. Confirm no KPI or test image identity appears in training sources.
6. Confirm each planned action's required predecessor artifact exists and its
   owning job reached `COMPLETE`.
7. For synthesis, require every boxed KPI annotation to resolve a real,
   nonempty `dataset_id` from its image or annotation metadata. An ID absent
   from `synthesis.routes` is valid and remains on the normal real-data path.
8. Require every boxed KPI annotation in a configured route to
   declare `texture_id`, `defect_class`, and an existing `fn_mask_source` path.
   Mask-content eligibility remains a per-FN preparation decision.
9. Resolve every synthesis route before iteration 0 and verify handoff
   hashes for newly trained adapters.

## Launch

Invoke `tao-launch-workflow`, present the consolidated launch review, obtain
confirmation, create the job record, then use the selected platform's
`submit/status/logs/cancel` verbs. Poll the backend for live state. A
scheduler success does not override a missing or invalid completion artifact.

Platform-specific staging, cache, filesystem, scheduler, and recovery policy
belongs to the selected platform skill; it is intentionally not duplicated
here.

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
