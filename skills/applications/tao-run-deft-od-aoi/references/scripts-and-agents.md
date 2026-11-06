# Bundled scripts and delegated skills

Use this map to locate the owner of each stage. Application scripts prepare or
validate contracts; model and data skills own their container actions.

## Application scripts

| Script | Purpose |
|---|---|
| `prepare_deft_od_aoi_sources.py` | Canonicalize customer shards, delegate role merging to Data Services `annotations merge`, validate KPI/test/real/clean roles, and emit `sources.json`. |
| `init_deft_od_aoi.py` | Validate normalized roles, freeze the policy, and initialize durable state. |
| `prepare_deft_od_aoi_retrieval.py` | Build candidate crops/specs or per-iteration gap-query/mining specs. |
| `deft_od_aoi_round_robin_selection.py` | Materialize deterministic per-query selections into the standard retrieval-stage artifacts. |
| `admit_deft_od_aoi_coco.py` | Admit unique real/clean/synthetic sources into cumulative binary COCO. |
| `prepare_deft_od_aoi_measurement.py` | Emit KPI/test RT-DETR inference and loose/strict gap specs. |
| `prepare_deft_od_aoi_training.py` | Emit direct main training or deterministic probe specs. |
| `select_deft_od_aoi_training.py` | Select the best probe or KPI-best checkpoint and optional extension. |
| `commit_deft_od_aoi_stage.py` | Verify artifacts and atomically advance application state. |
| `prepare_deft_od_aoi_synthesis.py` | Normalize exact strict FNs into AnomalyGen 1.2 preparation input. |
| `resolve_deft_od_aoi_synthesis.py` | Resolve task weights or emit one-time fine-tuning requests. |

## Delegated stages

| Stage | Owning skill |
|---|---|
| RT-DETR inference and training | `tao-train-rtdetr` |
| Loose and strict OD gap analysis | `tao-analyze-gaps-od-map` |
| Candidate and query embedding | `tao-generate-image-embeddings` |
| Global unique-image mining (`max_similarity`) | `tao-mine-od-images` |
| Per-query selection (`round_robin_similarity`) | Application-owned deterministic selector |
| AnomalyGen 1.2 preparation and AMP | `tao-generate-anomalies` |
| Missing task-weight training | `tao-generate-anomalies` |
| Synthetic OD generation | `tao-generate-anomalies` |
| Authorization, records, and dispatch | `tao-launch-workflow` plus selected platform |

Read the owning skill and its `references/skill_info.yaml` before dispatch.
Do not guess container commands or inline a replacement implementation.

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
