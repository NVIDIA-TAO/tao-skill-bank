<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OD gap preparation and publication

OD generation uses the same `inference_only` engine as texture generation. The
OD application owns class projection and training-pool updates, not another
AnomalyGen generator.

1. `prepare --source od-gaps --prepare-stage inputs --config filtering.yaml
   --pool-dataset-root POOL [--exclude-stems EXCLUDE]`: validate normalized FN
   identity, isolate FN masks, select same-type donor masks, freeze clean/FN
   embedding specs and the input contract. Invalid masks or missing clean pools
   are reported as skipped; a batch without eligible FNs is `SKIPPED`.
2. Delegate the emitted clean/FN embedding specs to `tao-encode-images` on the
   selected platform. Retain box-level FN identity even when embeddings share
   the same original image. Remount the pool at its frozen compute path.
3. `prepare --source od-gaps --prepare-stage amp --prepared-root PREPARED
   --pool-dataset-root POOL --checkpoint-root /workspace/cdf-anomalygen/checkpoints`:
   rank distinct clean images, keep both FN and same-type mask branches, run
   native AMP with configured model/seed, and retain pair identities.
4. `prepare --source od-gaps --prepare-stage finalize --prepared-root PREPARED`:
   validate aligned masks, apply configured neighbor budget, and emit
   `prepared_anomalygen_inputs/anomalygen_generation_plan.json` and its hashed
   `prepared_inputs_manifest.json`. No pool mutation occurs.
5. For each generation-plan group, call `inference_only` with its checkpoint,
   exact recipe, testcase and real dataset root. Preserve the finalized manifest
   hashes; validate them before consuming a prepared handoff. Add `--pseudo-label`
   (structured action: `pseudo_label_flag: --pseudo-label`); its empty default
   omits COCO export.
6. Write a JSON list of `{dataset_id, manifest}` entries pointing at completed
   shared-run `manifest.json` files. The application runs
   `skills/applications/tao-run-deft-od-aoi/scripts/emit_sdg_coco.py
   --runs-manifest runs.json --prepared-manifest prepared_inputs_manifest.json
   --output-dir NEWDIR`. It validates native accounting
   and COCO identities/boxes, then emits native categories and an OD `defect`
   projection. The converter does not run model inference or mask generation.

Every prepare command also supplies `--name` and a new `--output-dir`; for AMP
and finalize these identify the operation while `--prepared-root` is the mutable
prepared bundle. Retain it across all stages. `assets/default_filtering.yaml`
is a generic configuration example; bind user-provided checkpoint/recipe/data.

OD preparation configuration requires an actual adapter **file** so its frozen
artifact manifest can hash the file. If the user supplies a native run directory,
resolve its authoritative `best_checkpoint.txt` and select that adapter file with
the exact saved training recipe before preparation. Shared `inference_only`
also accepts run directories directly; OD preparation does not perform that
resolution or prove its compatibility.

Original donor identity survives mask isolation and alignment: preparation records
`source_mask` and `source_mask_sha256` for the original FN mask or the exact
selected sampled donor, alongside the copied `mask_path` and its hash. Finalized
provenance retains original `source_mask`, scratch `copied_mask`, and placed
`aligned_mask` separately with hashes. Finalization, conversion, and admission
reject changed donors and current exclusions against those explicit identities.
A mask-selection bundle lacking original identity/hashes must be prepared again.
