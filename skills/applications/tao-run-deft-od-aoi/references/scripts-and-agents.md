# Bundled scripts and delegated skills

Use this map to locate the owner of each stage. Application scripts prepare or
validate contracts; model and data skills own their container actions.

## Application scripts

| Script | Purpose |
|---|---|
| `init_deft_od_aoi.py` | Validate normalized roles, freeze the policy, and initialize durable state. |
| `prepare_deft_od_aoi_retrieval.py` | Build candidate crops/specs or per-iteration gap-query/mining specs. |
| `admit_deft_od_aoi_coco.py` | Admit unique real/clean/synthetic sources into cumulative binary COCO. |
| `prepare_deft_od_aoi_measurement.py` | Emit KPI/test RT-DETR inference and loose/strict gap specs. |
| `prepare_deft_od_aoi_training.py` | Emit direct main training or deterministic probe specs. |
| `select_deft_od_aoi_training.py` | Select the best probe or KPI-best checkpoint and optional extension. |
| `commit_deft_od_aoi_stage.py` | Verify artifacts and atomically advance application state. |

## Delegated stages

| Stage | Owning skill |
|---|---|
| RT-DETR inference and training | `tao-train-rtdetr` |
| Loose and strict OD gap analysis | `tao-analyze-gaps-od-map` |
| Candidate and query embedding | `tao-generate-image-embeddings` |
| Unique-image mining | `tao-mine-od-images` |
| Authorization, records, and dispatch | `tao-launch-workflow` plus selected platform |

Read the owning skill and its `references/skill_info.yaml` before dispatch.
Do not guess container commands or inline a replacement implementation.
