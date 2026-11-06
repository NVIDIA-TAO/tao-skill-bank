<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Texture preparation, training, and generation

Use native `scripts/skill_utility/prepare_dataset.py --dataset pcb` and the
public PCB-Nano adapter for a public smoke run; check that script's `--help` in
the pinned image before selecting local/HF preparation options. Native public
PCB preparation may require HF authentication even for public assets. Metal and
Glass-Nano public adapters and user datasets use the same generic folder/recipe contract.

With the repository's skill scripts mounted at `/skill/scripts`, inference is:

```bash
python3 /skill/scripts/run_anomalygen.py inference_only \
  --name pcb-nano --output-dir /output/pcb-nano \
  --checkpoint /models/pcb-nano/iter_000015000.pt \
  --recipe /models/pcb-nano/training_recipe.yaml \
  --input-data-path /data/inference/testcase.jsonl --dataset-dir /data/pcb \
  --base-checkpoint /workspace/cdf-anomalygen/checkpoints/Cosmos3-Nano \
  --num-gpus 1 --num-search-run 3 --score nn
```

The checkpoint filename is illustrative: bind the actual downloaded file and
its paired recipe; do not fabricate a filename or substitute a generic template.
For a bounded generation smoke, add `--skip-quality` and report that quality was
not evaluated. For COCO output add `--pseudo-label`.

For training, use `full` or `finetune_only`, `--dataset-dir`, an optional native
`--recipe` training template, `--validation-testcase`, and base/VAE paths.
`--num-sdg` is required when validation or inference placement is needed. The
runner uses the native train controller, waits in 30-second slices, requires
iteration-zero NN baseline and a scored best checkpoint, and saves the exact
run recipe. The independent helper `prepare_finetune_recipe.py` can validate and
freeze a recipe before launch, including custom native knobs. Its explicit
`--max-iter` also adjusts cycle length unless a user template set that length.
`assets/default_recipe.yaml` is native 1.2 Nano geometry; do not transplant 1.1
shift/input-size defaults into a newly trained 1.2 adapter.

Fresh TAO training retains the 5000-iteration default with matching cycle length;
this is independent of the pretrained public PCB adapter's 15000-iteration
checkpoint. `--parallelism ddp|fsdp` and `--compile true|false` forward native
runtime environment controls. For a failed native training run, inspect its
controller PID/log and exact saved recipe before explicitly resuming with the
native controller; the shared runner refuses to overwrite an existing run.
