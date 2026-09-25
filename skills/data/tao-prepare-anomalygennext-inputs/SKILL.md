---
name: tao-prepare-anomalygennext-inputs
description: >-
  Prepare normalized object-detection false negatives, source masks, and clean-image
  embedding inputs for AnomalyGenNext. Use when turning box-level FN gaps into a
  frozen, pair-preserving preparation plan. Do not use for training or generation.
license: Apache-2.0
compatibility: Requires the AnomalyGenNext 1.1 image, pandas, pyarrow, NumPy, Pillow, and PyYAML.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags: [tao, data, anomalygen-next, object-detection, input-preparation]
---

# Prepare AnomalyGenNext Inputs

The first action freezes eligible false-negative identities, isolates each FN
mask to its bounding box, selects one deterministic same-type mask, and emits
the clean and FN embedding specs. The second action consumes those embedding
results, preserves every FN-to-clean pair, and runs native automatic mask
placement (AMP) from the AnomalyGenNext container. The final action accepts
only pairs for which both mask branches passed AMP, then freezes the exact
generation inputs and their hashes.

## Input contract

Start from `assets/default_filtering.yaml`. The gap parquet must contain:

```text
image_id filepath gap_type bbox class split
dataset_id texture_id defect_class anomaly_type fn_mask_source
```

Identity is explicit: `anomaly_type` must equal
`texture_id+defect_class`. Preparation intentionally has no filename or
directory-layout inference. A caller adapting an unfamiliar dataset must
normalize these fields before this boundary.

Each selected dataset maps to an existing fine-tuned checkpoint and recipe.
The recipe must declare the exact anomaly type, and `defect_spec.jsonl` must
contain its placement definition. Text-routed definitions require a nonempty
`roi_prompt_defect_location`. The pool layout is:

```text
POOL/TEXTURE/clean_image/*
POOL/TEXTURE/mask/DEFECT/*
```

## Action

Run through the selected platform after the common launch review:

```bash
scripts/prepare_anomalygennext_inputs.py \
  --config /path/to/filtering.yaml \
  --pool-dataset-root /path/to/anomalygen_pool \
  --output-dir /new/result/root
```

The output root must not exist. The action emits `fn_queries.parquet`,
`selected_fn_queries.parquet`, `mask_selection.parquet`, `clean_pool.parquet`,
two `tao-generate-image-embeddings` specs, the copied filtering config, and
`input_contract.json`.
`pool_dataset_root` is one user-level folder input for the workflow. The
platform stages it once and retains the binding between its source and the
compute-visible path passed to preparation. Preparation freezes that compute
path into `filtering_config.yaml`, every clean filepath in `clean_pool.parquet`,
and `input_contract.json`. The contract declares the same typed folder as a
read-only downstream input for clean-image embedding and AMP. Each later
container must remount the original source at that exact compute path; the
generated script arguments are platform plumbing, not new user choices.

Run both emitted specs through `tao-generate-image-embeddings`, preserving the
same encoder. The clean-image embedding action must consume
`input_contract.json.downstream_inputs.clean_embeddings` so the pool paths in
its parquet resolve. The FN embedding action likewise needs mounts for the
source-image paths already carried by its input parquet. Place both outputs
under `embeddings/` as named by the specs, then submit the `run_amp` action:

```bash
scripts/run_anomalygennext_amp.py \
  --prepared-root /temporary/execution/root \
  --published-root /persistent/result/root \
  --pool-dataset-root /path/to/anomalygen_pool \
  --sam2-checkpoint /models/sam2.1_hiera_large.pt
```

`run_amp` always loads the frozen config below `--prepared-root`; it accepts no
second config. Its typed pool argument confirms that the platform remounted the
folder at the path already frozen into the config and parquets. This writes
`knn_candidates.parquet`, the native AMP request, and `amp/testcase.jsonl`.
When execution uses temporary storage, `--published-root` records the
persistent locations that will contain the saved results. Finalize the saved
output in that persistent result root:

```bash
scripts/finalize_anomalygennext_inputs.py --prepared-root /existing/result/root
```

The finalizer writes pair status, selected pairs, copied aligned masks,
per-dataset testcase and provenance JSONL, and a generation plan plus integrity
manifest. A clean image is unique only within one FN; separate FNs may reuse it.
Platform skills own staging and cache placement.

## Gates

- Preserve each box-level FN as a distinct `fn_id`.
- Produce exactly two masks per FN: isolated FN and deterministic same-type.
- Embed a repeated source image once while preserving every FN query row.
- Require successful, nonempty, non-full-image aligned masks for both branches.
- Never infer an anomaly type or placement prompt.
- Refuse output reuse and never mutate a training pool.
- Platform skills own transient staging, caches, and copy-back.

Read `references/input-contract.md` when authoring the normalized parquet or
dataset mapping.
