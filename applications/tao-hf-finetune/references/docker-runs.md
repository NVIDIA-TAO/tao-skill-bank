<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Docker Run Catalog

Canonical `docker run` invocations used across the pipeline. All commands assume
the image was built once with `docker build -t run-<short>:latest .`. All commands
mount `$OUTPUT_DIR` (or `$(pwd)` when invoked from a generated rerun skill) at
`/workspace`.

`<short>` = `model_short_name` from `config.yaml`.

---

## Why `--entrypoint /bin/bash -lc "..."`

Raw NGC images set `/opt/nvidia/nvidia_entrypoint.sh` as ENTRYPOINT and do **not**
wrap commands in `bash -c`. Passing a command string without
`--entrypoint /bin/bash -lc` produces:

```
exec: "cd /workspace && ...": No such file or directory
```

`--entrypoint /bin/bash -lc` works whether or not the image was built from the
provided Dockerfile.

## Why `--shm-size=16g`

Without it, PyTorch DataLoader with `num_workers > 0` crashes:

```
RuntimeError: DataLoader worker (pid N) is killed by signal: Bus error
```

Bump higher for very large batch sizes.

## Why `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Reduces fragmentation under variable-shape inputs (detection, VLM). Always pass
on training runs.

---

## 1. Build image (once)

```bash
docker build -t run-<short>:latest .
```

## 2. Prepare data

```bash
docker run --rm --gpus all --shm-size=16g --entrypoint /bin/bash \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd)/$OUTPUT_DIR:/workspace \
  run-<short>:latest \
  -lc "cd /workspace && python prepare_data.py --config config.yaml"
```

For `source = local`, also bind-mount the dataset path read-only:

```bash
  -v <local_dataset_path>:<local_dataset_path>:ro \
```

## 3. Smoke test (1 step on real data)

```bash
docker run --rm --gpus all --shm-size=16g --entrypoint /bin/bash \
  -e HF_TOKEN=$HF_TOKEN -e WANDB_MODE=disabled \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/$OUTPUT_DIR:/workspace \
  run-<short>:latest \
  -lc "set -o pipefail; cd /workspace && python train.py --config config.yaml --smoke --max_steps 1 2>&1 | tee logs/smoke.log"
```

Pass criteria in `logs/smoke.log`:
- No exception
- Loss is finite (not `0.0`, not `NaN`)
- `grad_norm > 0` at step 1

Any failure → STOP. Do not launch full training.

## 4. Baseline (zero-shot) eval

```bash
docker run --rm --gpus all --shm-size=16g --entrypoint /bin/bash \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd)/$OUTPUT_DIR:/workspace \
  run-<short>:latest \
  -lc "cd /workspace && python run_eval.py --config config.yaml \
       --checkpoint $MODEL_ID --output reports/baseline_results.json"
```

Skip if `skip_baseline: true` in `config.yaml`.

## 5. Full training (detached)

```bash
docker run -d --name hft_train --gpus all --shm-size=16g --entrypoint /bin/bash \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY -e WANDB_PROJECT=$WANDB_PROJECT \
  -e WANDB_RUN_NAME=$WANDB_RUN_NAME \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/$OUTPUT_DIR:/workspace \
  run-<short>:latest \
  -lc "set -o pipefail; cd /workspace && python train.py --config config.yaml 2>&1 | tee logs/train.log"

docker logs -f hft_train      # watch loss descend within 10-20 steps
```

Multi-GPU: prepend `torchrun --nproc_per_node=$gpu_count` to `python train.py`.

## 6. LoRA merge (VLM only)

```bash
docker run --rm --gpus all --entrypoint /bin/bash \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd)/$OUTPUT_DIR:/workspace \
  run-<short>:latest \
  -lc "cd /workspace && python merge_lora.py --base_model $MODEL_ID \
       --adapter checkpoints/final --output checkpoints/merged"
```

Subsequent eval / infer / push must use `checkpoints/merged` instead of
`checkpoints/final`.

## 7. Post-training eval

```bash
docker run --rm --gpus all --shm-size=16g --entrypoint /bin/bash \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd)/$OUTPUT_DIR:/workspace \
  run-<short>:latest \
  -lc "cd /workspace && python run_eval.py --config config.yaml \
       --checkpoint checkpoints/final --output reports/eval_results.json"
```

For LoRA, replace `checkpoints/final` → `checkpoints/merged`.

## 8. Inference samples (5 held-out)

```bash
docker run --rm --gpus all --shm-size=16g --entrypoint /bin/bash \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd)/$OUTPUT_DIR:/workspace \
  run-<short>:latest \
  -lc "cd /workspace && python infer.py --config config.yaml \
       --checkpoint checkpoints/final --n_samples 5 --output reports/inference_samples/"
```

Each sample writes: input image, overlay (bbox / mask / depth / caption),
`meta.json` with the raw prediction dict.

---

## Defaults summary

| Flag | Used by | Why |
|---|---|---|
| `--gpus all` | every GPU command | passes through host GPUs |
| `--shm-size=16g` | DataLoader workers | avoid Bus error on collate |
| `--entrypoint /bin/bash` + `-lc` | every command | bypass NGC entrypoint |
| `-e HF_TOKEN` | data, train, eval, infer, merge | HF Hub auth |
| `-e WANDB_*` | training only | metrics logging |
| `-e WANDB_MODE=disabled` | smoke only | no run pollution |
| `-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | train, smoke | allocator |
| `-d --name hft_train` | full training only | survive shell disconnect |
| `--rm` | every other command | one-shot cleanup |
