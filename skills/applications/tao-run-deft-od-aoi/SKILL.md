---
name: tao-run-deft-od-aoi
description: >-
  Run a binary industrial-inspection DEFT loop with TAO RT-DETR: measure on
  frozen KPI/test roles, mine gap-similar real and clean images with SigLIP,
  accumulate admitted COCO data, retrain from one base checkpoint, and select
  by KPI AP50. Use for iterative AOI defect detection, not generic multiclass OD.
license: Apache-2.0
compatibility: Requires TAO RT-DETR and Data Services images, CUDA GPUs, and normalized COCO roles.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags: [application, workflow, deft, object-detection, aoi, rtdetr]
---

# TAO DEFT OD AOI

This application is a disk-backed RT-DETR loop for one foreground class,
`defect`. Its core is real-data-only; AnomalyGenNext synthesis is an optional
route with separate preparation, generation, and admission gates.

## References

Read only the references needed for the current stage:

- intake and launch: `references/defaults.md`, `references/data-contract.md`,
  `references/source-manifest.md`, and `references/preflight.md`;
- orchestration: `references/pipeline.md` and
  `references/scripts-and-agents.md`;
- gaps and retrieval: `references/gap-routing.md` and
  `references/tao-analyze-gaps-od-map.md`;
- training and selection: `references/training-policy.md`;
- optional synthesis: `references/anomalygen-pool.md`.

## Start

Select an installed platform, read its skill, then invoke
`tao-launch-workflow`. The single launch review must include the four normalized
COCO roles, trainable RT-DETR base checkpoint, maximum iterations, image and
Data Services containers, GPU shape, and expected runtime. After approval,
copy `assets/default_policy.yaml`, fill its required values, and initialize once:

```bash
scripts/init_deft_od_aoi.py \
  --config /workspace/deft_policy.yaml \
  --output-dir /new/results/deft_contract
```

Never reinitialize an existing result. The validator requires disjoint KPI,
test, defective-real, and verified-clean roles; every COCO must declare only
`defect`. KPI and test may mix boxed and boxless images because they never enter
training. Every defective-real image needs at least one box, while clean images
remain explicit zero-annotation COCO entries.

## Loop boundary

The loop composes existing bank actions:

1. `tao-train-rtdetr` inference on KPI and test.
2. Two `tao-analyze-gaps-od-map` actions: loose confidence for FP routing and
   strict confidence for FN routing.
3. `tao-generate-image-embeddings` with one frozen SigLIP encoder, followed by
   `tao-mine-od-images` unique-neighbor matching against the real or clean role.
4. Application-owned admission and cumulative binary COCO assembly.
5. Direct `tao-train-rtdetr` training from the same frozen base checkpoint,
   then KPI-only checkpoint selection. Test remains report-only.

All specs are nested YAML dictionaries. Every GPU/Data Services action uses the
selected platform's `submit/status/logs/cancel` contract and a job record.
Stop on missing artifacts, role overlap, empty enabled mining, class drift, or
failed training. Never infer live state from the JSON record alone.

## Retrieval preparation

Build the reusable candidate cache once, then prepare queries after each pair
of loose/strict gap jobs:

```bash
scripts/prepare_deft_od_aoi_retrieval.py candidates \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --output-dir "$RESULTS/candidates"

scripts/prepare_deft_od_aoi_retrieval.py queries \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --strict-gaps "$ITER/strict/box_gaps.parquet" \
  --loose-gaps "$ITER/loose/box_gaps.parquet" \
  --iteration 1 --candidate-root "$RESULTS/candidates" \
  --output-dir "$ITER/retrieval"
```

Run every emitted embedding spec through `tao-generate-image-embeddings`, then
each enabled mining spec through `tao-mine-od-images`. Defective candidates
and gap queries use contextual crops; clean candidates use the frozen grid.
Strict FNs and loose near-miss FPs route to real data. Background-like loose
FPs route only to the verified-clean role. Empty roles emit no mining action.

## Admission and cumulative COCO

After both enabled miners complete, admit their selected candidate crops back
to unique source images and publish the next cumulative dataset:

```bash
scripts/admit_deft_od_aoi_coco.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --candidate-root "$RESULTS/candidates" \
  --retrieval-root "$ITER/retrieval" \
  --previous-coco "$PREVIOUS/train.json" \
  --output-dir "$ITER/training_data"
```

Omit `--previous-coco` only for iteration 1. The helper recomputes maximum
cosine similarity from the frozen candidate/query embeddings, applies the
minimum similarity, deduplicates crop hits to source images, excludes prior
sources, and caps cumulative clean negatives against cumulative real defects.
It retains every prior image and box and emits one binary COCO with explicit
zero-annotation clean images. Use `--link-mode hardlink` only when source and
output share a filesystem; portable staging should keep the copy default.

## Measurement specs

For baseline use the frozen base checkpoint; after training use the selected
iteration checkpoint. Prepare both inference specs and the two gap-analysis
specs together:

```bash
scripts/prepare_deft_od_aoi_measurement.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --checkpoint "$CHECKPOINT" \
  --kpi-predictions "$MEASURE/kpi/inference/labels" \
  --results-root "$MEASURE" --output-dir "$MEASURE/specs"
```

Submit KPI/test inference through `tao-train-rtdetr`. Once KPI labels exist,
submit `gap_loose.yaml` and `gap_strict.yaml` through
`tao-analyze-gaps-od-map`. The helper projects the frozen KPI COCO to KITTI,
keeps the two-line inference class map durable, and emits only nested specs.
Test inference is report-only and its output cannot influence routing or
checkpoint selection.

## RT-DETR training specs

Build nested specs from the current cumulative COCO:

```bash
scripts/prepare_deft_od_aoi_training.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" --iteration 1 \
  --train-coco "$ITER/training_data/train.json" \
  --train-images "$ITER/training_data/images" \
  --results-root "$ITER/training" --output-dir "$ITER/train_specs"
```

Invoke the RT-DETR leaf as direct training with `automl_policy: off`; the AOI
application already owns its frozen probe policy. Iterations 1–2 use 36 epochs.
Later iterations use the clipped size-adaptive budget. When probes are enabled,
the helper emits independent incumbent, data-growth-scaled, and deterministic
jitter specs, all initialized from the same frozen base checkpoint. Run all
three before selecting and materializing `train.yaml`; never initialize the
next iteration from a previous iteration checkpoint.

Select all three probes before main training:

```bash
scripts/select_deft_od_aoi_training.py probes \
  --manifest "$SPECS/training_manifest.json" \
  --main-template "$SPECS/main_template.yaml" \
  --status "$P0/status.json" --status "$P1/status.json" --status "$P2/status.json" \
  --output-dir "$SPECS/selected"
```

After main training, select the KPI-best checkpoint from structured status
rows. Repeat `--status` for resumed phases. If the best epoch is within the
frozen late window, the first call emits one same-iteration `extension.yaml`
from the terminal checkpoint; submit it, then reselect with
`--extension-applied`. The checkpoint carried to measurement is always the
maximum KPI `val_mAP50`, never the latest checkpoint or test result.

After each stage succeeds, commit at least one completion artifact with
`commit_deft_od_aoi_stage.py`. It accepts only the next frozen stage, verifies
and hashes every named file, atomically updates `deft_state.json`, and appends
`loop_log.jsonl`. The final iteration becomes `COMPLETE` only after its gap
artifacts are committed. Poll native backends for live state; this record is
durable workflow history, not a scheduler substitute.

## Optional synthesis with existing task weights

Enable `synthesis` only when KPI annotations carry `dataset_id`, `texture_id`,
`defect_class`, and a pixel `fn_mask_source`. Each configured dataset route
must provide an existing AnomalyGenNext checkpoint and matching recipe. After
strict gap analysis, normalize exact FN/annotation matches:

```bash
scripts/prepare_deft_od_aoi_synthesis.py \
  --policy "$RESULTS/deft_od_aoi_policy.yaml" \
  --strict-gaps "$MEASURE/gap_strict/box_gaps.parquet" \
  --output-dir "$ITER/synthesis_request"
```

Pass the emitted filtering YAML through `tao-prepare-anomalygennext-inputs`,
then its finalized generation plan through `tao-generate-od-defects`. Commit
`iteration_synthesis` before training. Re-run admission with the generated
native COCO and image root; synthetic categories are folded to `defect`, and
the frozen cumulative fraction cap is applied against admitted real defects.
Boxes alone never substitute for the required pixel mask.

## Missing AnomalyGenNext task weights

A synthesis route may replace `checkpoint` and `recipe` with a `finetune` block
containing `dataset_root`, frozen `validation_testcase`, Cosmos3-Nano
`base_checkpoint`, `vae_path`, `nn_backbone`, future `result_handoff`, and
optional user `recipe_template`/`defect_spec`. These are AnomalyGenNext inputs;
the application policy is not an upstream training recipe.

Run `resolve_deft_od_aoi_synthesis.py` before the candidate cache. If a handoff
is absent, it emits `finetune_requests.json`; execute each request through
`tao-finetune-anomalygennext` once. Resolve again into a new directory, verify
the hash-bound handoff, commit `synthesis_bootstrap`, and use the emitted
`resolved_synthesis_policy.yaml` for all later synthesis preparation. Never
start or resume AnomalyGenNext training inside a DEFT iteration.
