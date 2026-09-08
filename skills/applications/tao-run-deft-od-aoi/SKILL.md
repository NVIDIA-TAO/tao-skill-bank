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
`defect`. Its core is real-data-only. Synthetic generation is an optional later
integration and is not part of this contract.

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
`defect`. KPI/test pixels never enter training, and clean images remain explicit
zero-annotation COCO entries.

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

The following review slices add the query router, cumulative assembler, and
RT-DETR spec/selection helpers. This first slice freezes only the durable input
and policy boundary.
