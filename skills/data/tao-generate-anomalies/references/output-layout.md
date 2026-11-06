<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Output layout and receipts

The shared runner writes `status.json`, `commands.jsonl` (commands, exit codes,
start/end times), and completion `manifest.json`. Native generation is `sdg/`;
quality rounds are `rounds/` and selected outputs `searched/`. Optional COCO is
`pseudo_labels/coco_annotations.json`. Training also publishes
`training/{training_handoff.json,nn_improvement_summary.json,canonical_recipe.yaml,best_model_checkpoint.pt}`.

Each native generation bucket keeps `texture_ft_generation_result.csv`, optional
`guardrail_blocked.csv`, `timing_summary.json`, `reconstructed_image/`, originals,
masks and native debug files. Do not rename these to legacy SDG filenames.
The verifier joins each native index to source image/mask/type and checks disk
images. Whole-text blocked rows use output_idx=-1; per-output blocks use 0 for
the supported one-output contract. Blocked rows have no reconstructed image.

The manifest identifies the selected output, exact recipe hash, input hash,
exclusion receipt, generation counts and native timing. Host wall time and
native phase timing are separate measurements. Unit fixtures and mocked
commands are not GPU performance evidence.
