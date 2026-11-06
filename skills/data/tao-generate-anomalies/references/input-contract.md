<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Input and exclusion contracts

Native generation JSONL has `image_filename`, `mask_filename`, `anomaly_type`
(`TEXTURE+DEFECT`) and optionally `index`. Relative filenames resolve against the native repository working directory, then
are frozen as absolute paths before native execution. Files must exist in the same compute namespace. Each row requests one output (`num_generated_images=1`). Preset
indices must be unique nonnegative integers and are preserved. Without preset
indices, the filtered JSONL defines native row indices; the filtering manifest
also records original row positions.

A dataset uses `TEXTURE/clean_image`, `TEXTURE/anomaly_image/DEFECT`, and
`TEXTURE/mask/DEFECT`. Training masks are binary 0/255 and nonempty. Native training requires
`mask/DEFECT/<image-stem>_mask<image-extension>` or the PNG fallback
`mask/DEFECT/<image-stem>_mask.png`; an unsuffixed mask filename is rejected
before launch. Explicit validation/generation JSONL instead names its mask file
directly. CAD routes also need clean-aligned CAD masks and
`semantic_segmentation_labels.json`. Defect specs use native routing values
`free`, `text`, `cad`, or `cad_tht`; text requires its ROI prompt. Do not change
native routing enums to suit a particular application.

`--exclude-stems` contains exact stems, blank lines and # comments allowed;
paths are rejected. `sample` excludes sample image and sample_mask. Dataset views
copy eligible inputs to a new destination and keep original data unchanged;
symlinks escaping the source dataset are rejected. Training, validation, clean
references, donor masks, testcases, and output provenance share the exclusion
contract. OD freezes values/hash at its inputs stage and rejects reintroduction
at AMP/finalization; later stages do not accept a replacement exclusion list.

Never infer validation or benchmark eligibility from a folder name. The calling
application supplies and records the approved exclusion list and split policy.
