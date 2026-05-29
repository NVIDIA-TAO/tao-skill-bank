---
name: tao-train-depth-anything-v2
description: Monocular depth estimation using Metric Depth Anything v2 or Relative Depth Anything architectures. Predicts
  per-pixel depth from single RGB images. Use when training, evaluating, exporting, or running inference for a TAO
  monocular depth model. Trigger phrases include "train monocular depth", "DepthAnything v2", "metric depth from single
  image", "monocular depth estimation".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: '0.1'
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- monocular
- depth
- estimation
---

# Depth Net Mono

Monocular depth estimation using Metric Depth Anything v2 or Relative Depth Anything architectures. Predicts per-pixel depth from single RGB images.

Pretrained checkpoint loading varies by model variant and use case — see **Pretrained checkpoint loading — use case matrix** under [Important Parameters](#important-parameters) below.

The mono and stereo skills both invoke the unified TAO `depth_net` CLI inside the container; the mono/stereo family is selected via `model.model_type` (see Important Parameters below).

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `deploy/SKILL.md` first. The deploy spec template lives in this skill's `references/spec_template_deploy.yaml`.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only; otherwise default to `auto`. When `automl_policy: auto`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Workflow

### Prerequisites — data accessibility

Your dataset (RGB images + GT depth files) must be reachable from inside the container:
- **SDK runner**: place files at the S3 paths the runner resolves (the `S3_TRAIN` / `S3_EVAL` placeholders shown in **Typical Spec Overrides**). The runner handles S3 → container-path mounting transparently.
- **Direct `docker run`** (e.g. local testing): mount the host dataset root read-only at the same in-container path:

```
docker run ... -v <host_data_root>:<host_data_root>:ro <container> ...
```

The same accessibility requirement applies to the `<output_dir>` written by all actions.

### Step 1 — Annotation file

Per-line annotation file referenced by `data_sources[*].data_file`:

| Columns | Format | Use |
|---|---|---|
| 1 | `<image>` | Mono inference (no GT) |
| 2 | `<image> <gt_depth>` | Mono with GT |

If you already have one, point to it. Otherwise generate via `depth_net convert`:

```
depth_net convert -e <convert_spec.yaml>
```

`convert_spec.yaml` template:

```yaml
data_root: <directory whose immediate children are scene/sample folders that contain your image+depth files; convert walks data_root recursively but expects per-scene subdirectories at one level below>
image_dir_pattern: [<substring matching left/RGB image paths>]
depth_dir_pattern: [<substring matching GT depth paths>]
image_extension: ''     # optional .endswith filter, e.g. '.jpg'
depth_extension: ''     # optional, swapped during depth derivation, e.g. '.png'
split_ratio: 0.0        # 0.0/1.0 = test-only; 0.8 = 80/20 train+val
```

`convert` walks `data_root` recursively, selects paths whose path-string contains *all* substrings in `image_dir_pattern` (AND-filter), then derives the depth path by replacing `image_dir_pattern[0]` with `depth_dir_pattern[0]` and `image_extension` with `depth_extension`. Inspect your dataset's directory layout and identify the substring distinguishing RGB images from depth files (e.g. `rgb_` vs `sync_depth_`).

`data_root` must point at the parent that contains the per-scene subdirectories (e.g. for NYU eval, use `/data/nyu_v2/eval/test`, not `/data/nyu_v2/eval/test/bathroom` — the latter limits the walk to a single scene). Always include the leading dot in `image_extension` / `depth_extension` (e.g. `'.jpg'` not `'jpg'`); the substring swap is form-sensitive and a mismatch silently corrupts derived paths.

### Step 2 — Pair `model_type` and `dataset_name` based on your data

Default — generic class for each task:

| Data category | `model_type` | `dataset_name` |
|---|---|---|
| Disparity-encoded data (pixels) | `RelativeDepthAnything` | `RelativeMonoDataset` |
| Metric depth (meters) | `MetricDepthAnything` | `MetricMonoDataset` |
| Mono inference (no GT, any image) | matches train choice | `RelativeMonoDataset` or `MetricMonoDataset` |

Dataset-specific class — switch when the data needs preprocessing the generic class does not perform:

| Special case | `model_type` | `dataset_name` | What the class adds |
|---|---|---|---|
| NYU `sync_depth_*.png` (raw uint16 millimetres) — relative | `RelativeDepthAnything` | `NYUDV2Relative` | mm→m unit conversion + Eigen evaluation crop |
| NYU `sync_depth_*.png` (raw uint16 millimetres) — metric | `MetricDepthAnything` | `NYUDV2` | same |

Using a generic class on data that requires unit conversion (e.g. raw NYU uint16 PNGs) results in an empty valid mask and silent `train_loss = NaN`. Match the class to your data's encoding.

### Step 3 — Write spec yaml from Typical Spec Overrides

Copy the action block from **Training Requirements → Typical Spec Overrides**. Replace:
- `model.model_type` from Step 2
- `dataset.<...>.data_sources[*].dataset_name` from Step 2
- `data_sources[*].data_file` with the path from Step 1 (S3 path under SDK runner, host path for direct docker)
- For metric finetune: additionally apply **Metric Variant Finetuning Recipe**.

For mono training set `train.precision: fp32` (recommended) or `bf16` (Ampere SM80+, alternative).

### Step 4 — Run

```
docker run --gpus 'device=0' --shm-size 16G --ipc=host \
  --user $(id -u):$(id -g) \
  -v <data_root>:<data_root>:ro \
  -v <output_dir>:<output_dir> \
  <container> \
  depth_net <action> -e <spec.yaml>
```

Without `--user $(id -u):$(id -g)` the container writes outputs as `nobody:nogroup`, blocking host-side cleanup and retry.

### Step 5 — Verify

- Container exit code 0
- `status.json` `kpi` block populated
- For `train`: inspect per-step `train_loss` directly — the entrypoint reports `Execution status: PASS` even when `train_loss = NaN` (see Metric Variant Finetuning Recipe → Sanity-run PASS criteria)
- For `evaluate` / `inference`: artifacts under `results_dir`

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `deploy/SKILL.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Training Requirements

- **Valid `dataset_name` values for mono `data_sources`** (case-insensitive): `ThreeDVLM`, `FSD`, `NvCLIP`, `IssacStereo`, `Crestereo`, `Middlebury`, `NYUDV2`, `NYUDV2Relative`, `RelativeMonoDataset`, `MetricMonoDataset`. `NYUDV2` carries metric depth GT (meters) — pair with `MetricDepthAnything`; `NYUDV2Relative` is the same data with relative-depth conventions — pair with `RelativeDepthAnything`.
- **Monitoring metric:** val/loss

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.test_dataset.data_sources | eval_dataset | data_file: annotations.txt + dataset_name | Yes |
| inference | dataset.infer_dataset.data_sources | inference_dataset | data_file: annotations.txt + dataset_name | Yes |
| quantize | dataset.train_dataset.data_sources | train_datasets | data_file: annotations.txt + dataset_name | Yes |
| quantize | dataset.val_dataset.data_sources | eval_dataset | data_file: annotations.txt + dataset_name | Yes |
| quantize | dataset.quant_calibration_dataset.images_dir | train_datasets | images.tar.gz | No |
| train | dataset.train_dataset.data_sources | train_datasets | data_file: annotations.txt + dataset_name | Yes |
| train | dataset.val_dataset.data_sources | eval_dataset | data_file: annotations.txt + dataset_name | Yes |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`. Each `data_sources` entry is a dict with **two mandatory fields**: `data_file` and `dataset_name`.

```python
S3_TRAIN = "aws://bucket/data/train"
S3_EVAL = "aws://bucket/data/eval"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 10,
    "train.precision": "fp32",
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "model.model_type": "RelativeDepthAnything",
    "model.encoder": "vitl",
    "dataset.train_dataset.batch_size": 4,
    "dataset.train_dataset.workers": 4,
    "dataset.train_dataset.augmentation.crop_size": [518, 518],
    "dataset.train_dataset.data_sources": [
        {"data_file": f"{S3_TRAIN}/annotations.txt", "dataset_name": "RelativeMonoDataset"}
    ],
    "dataset.val_dataset.batch_size": 1,
    "dataset.val_dataset.workers": 4,
    "dataset.val_dataset.data_sources": [
        {"data_file": f"{S3_EVAL}/annotations.txt", "dataset_name": "RelativeMonoDataset"}
    ],
}
```

**Precision recommendation (relative variant)**: use `fp32` (recommended). `bf16` is supported as an alternative on Ampere SM80+ hardware.

**evaluate (mandatory data sources):**
```python
{
    "model.model_type": "RelativeDepthAnything",
    "dataset.test_dataset.batch_size": 1,
    "dataset.test_dataset.workers": 4,
    "dataset.test_dataset.data_sources": [
        {"data_file": f"{S3_EVAL}/annotations.txt", "dataset_name": "NYUDV2Relative"}
    ],
}
```

**export:**
```python
{
    "model.model_type": "RelativeDepthAnything",
    "export.input_channel": 3,
    "export.input_height": 518,
    "export.input_width": 518,
    "export.opset_version": 16,
    "export.on_cpu": False,
    "export.gpu_id": 0,
}
```

Defaults sourced from `nvidia_tao_pytorch/cv/depth_net/experiment_specs/experiment_mono_relative.yaml` (export block). Override only when the deployment target requires a different ONNX shape, opset, or export device.

**inference (mandatory data sources):**
```python
{
    "model.model_type": "RelativeDepthAnything",
    "dataset.infer_dataset.batch_size": 1,
    "dataset.infer_dataset.workers": 4,
    "dataset.infer_dataset.data_sources": [
        {"data_file": f"{S3_EVAL}/annotations.txt", "dataset_name": "RelativeMonoDataset"}
    ],
    "inference.save_raw_pfm": False,
}
```

`inference.save_raw_pfm` controls whether raw single-channel disparity is written as `.pfm` files alongside the visualization output. Default `False` — the action emits a 240×960 RGB JPG triptych (input | predicted disp | overlay-style panel) at 320×240 per panel, mirroring the source dataset's directory tree under `<results_dir>/inference/inference_images/`. Set `True` to additionally write `.pfm` files for downstream metric computation; raw disparity is unbounded scale-shift-invariant for `RelativeDepthAnything` and bounded to `[min_depth, max_depth]` for `MetricDepthAnything`.

**quantize (mandatory data sources):**
```python
{
    "dataset.train_dataset.data_sources": [
        {"data_file": f"{S3_TRAIN}/annotations.txt", "dataset_name": "RelativeMonoDataset"}
    ],
    "dataset.val_dataset.data_sources": [
        {"data_file": f"{S3_EVAL}/annotations.txt", "dataset_name": "RelativeMonoDataset"}
    ],
    "dataset.quant_calibration_dataset.images_dir": f"{S3_TRAIN}/images.tar.gz",
}
```
## Eval Dataset

Optional. Val dataset configured via `dataset.val_dataset.data_sources` (each entry needs `data_file` and `dataset_name`).

## Important Parameters

- **model.model_type**: Model architecture. Options: `MetricDepthAnything`, `RelativeDepthAnything`. Default `MetricDepthAnything`.
- **model.encoder**: Backbone encoder (top-level `model` field, not nested under `mono_backbone`). Options: `vits`, `vitb`, `vitl`, `vitg`. Default `vitl`.
- **model.mono_backbone.pretrained_path**: Path to **DINOv2 ViT-L encoder weights** (used for Relative train-from-scratch only — Metric and Relative finetune use `train.pretrained_model_path` + a TAO ckpt instead; see use-case matrix below). Architecturally identical to the DepthAnything v2 encoder (same ViT-L), but the weights differ: DINOv2 is the self-supervised pretraining used to initialize the Relative DepthAnything encoder before depth-supervised training. Set to an empty string (`""`) to skip the backbone-only weight load — use this when the full TAO checkpoint is supplied via `train.pretrained_model_path` (Pytorch-Lightning state) or `evaluate.checkpoint` / `inference.checkpoint`, since those carry the backbone state already. Setting both is redundant; the backbone-only load happens first and is then overwritten by the full-state load.
- **model.mono_backbone.use_bn** / **model.mono_backbone.use_clstoken**: Backbone toggles. Booleans. Defaults: `use_bn: False`, `use_clstoken: False` (matches the released `RelativeDepthAnything` and `MetricDepthAnything` checkpoint architectures). Override only when training a custom variant whose checkpoint was produced with the alternate setting.
- **train.optim.lr**: Learning rate. Default 1e-4 (AdamW).
- **train.lr_scheduler**: LR scheduler. Options: MultiStepLR, StepLR, CustomMultiStepLRScheduler, LambdaLR, PolynomialLR, OneCycleLR, CosineAnnealingLR.
- **train.precision**: Training precision. Options: fp32 (recommended), bf16 (Ampere SM80+, alternative), fp16.
- **train.distributed_strategy**: Distribution strategy. Options: ddp, fsdp.
- **train.activation_checkpoint**: Enable activation checkpointing. Default False.
- **dataset.dataset_name**: Top-level dataset family identifier (e.g., `MonoDataset`).
- **dataset.{train,val,test,infer}_dataset.batch_size**: Per-split batch size.
- **dataset.{train,val,test,infer}_dataset.workers**: Per-split DataLoader worker count (the field name is `workers`, not `num_workers`).
- **dataset.{train,val,test,infer}_dataset.augmentation.crop_size**: Per-split crop size. Default `[518, 518]`.
- **dataset.{train,val,test,infer}_dataset.data_sources**: List of `{data_file, dataset_name}` dicts. Both fields are mandatory per entry.
- **dataset.max_depth** / **dataset.min_depth**: Top-level depth range for metric depth estimation.
- **export.input_channel**: ONNX input channel count. Default `3` (RGB), matching the runtime input expected by `RelativeDepthAnythingV2` / `MetricDepthAnythingV2`. Source: `experiment_mono_relative.yaml` export block.
- **export.input_height** / **export.input_width**: ONNX input spatial dims. Default `518` / `518`, matching the model's training-time crop. Override only when targeting a different deployment input shape — the model's positional embeddings constrain practical shapes to multiples of the patch size (14 for ViT-L).
- **export.opset_version**: ONNX opset target. Default `17` (native LayerNormalization op for fp16 stability). Source: `experiment_mono_relative.yaml` export block.
- **export.on_cpu**: Whether ONNX export runs on CPU. Default `False` (uses `export.gpu_id`). Source: `experiment_mono_relative.yaml` export block.
- **export.gpu_id**: GPU device index for ONNX export when `on_cpu: False`. Default `0`. Source: `experiment_mono_relative.yaml` export block. Should match the `--gpus '"device=N"'` flag passed to `docker run`.
- **export.batch_size**: ONNX batch size. `1` = static, `-1` = batch axis dynamic. Height and width are always taken from the trace shape; H/W dynamic is not supported. Default `-1`.
- **inference.save_raw_pfm**: Whether the inference action additionally writes raw single-channel disparity as `.pfm` files alongside the visualization JPGs. Default `False`. Source: `experiment_mono_relative.yaml` inference block. Set `True` for downstream metric computation; raw disparity is unbounded scale-shift-invariant for `RelativeDepthAnything` and bounded to `[min_depth, max_depth]` for `MetricDepthAnything`. With the default, the inference action emits a 240×960 RGB JPG triptych under `<results_dir>/inference/inference_images/` mirroring the source dataset's directory tree.

### Pretrained checkpoint loading — use case matrix

| Use case | `model.mono_backbone.pretrained_path` | `train.pretrained_model_path` |
|---|---|---|
| Relative — train from scratch (DINOv2 backbone weights only) | `<DINOv2 ViT-L weights>` | `""` |
| Relative — finetune from TAO relative checkpoint | `""` | `<TAO relative ckpt>` |
| Metric — train from scratch on top of relative backbone (sanity) | `<TAO relative ckpt>` | `""` |
| Metric — finetune from TAO metric checkpoint | `""` | `<TAO metric ckpt>` |

Setting both keys is redundant: the backbone-only load happens first and is overwritten by the full-state load. The metric variant requires the `MetricDepthAnythingV2` head naming (`metric_depth_head.*`); see **Checkpoint compatibility** under the Metric Variant Finetuning Recipe.

## Relative Variant Finetuning Recipe

Relative finetune from a TAO-trained `RelativeDepthAnything` checkpoint:

| Spec key | Value | Notes |
|---|---|---|
| `model.model_type` | `RelativeDepthAnything` | |
| `model.encoder` | `vitl` | matches the released TAO relative checkpoint |
| `model.mono_backbone.pretrained_path` | `""` | the full TAO checkpoint already carries the backbone state; setting this is redundant and is overwritten by the full-state load |
| `train.pretrained_model_path` | `<TAO relative ckpt>` | full Pytorch-Lightning state load |
| `train.precision` | `fp32` (recommended) or `bf16` (alternative on Ampere SM80+) | |
| `train.optim.lr` | `5e-6` | The released relative checkpoint is already converged; the AdamW default `1e-4` listed in Important Parameters is an order of magnitude too aggressive for finetune from a converged backbone, and degrades the released checkpoint's accuracy on a short adaptation run. Use `5e-6` and a gentle scheduler (`LambdaLR`) when adapting to a new dataset. |
| `train.optim.lr_scheduler` | `LambdaLR` | gentle warmup + decay; matches the Metric Variant Recipe |

The dataset block follows **Step 2 — Pair `model_type` and `dataset_name`** above. Use `RelativeMonoDataset` for generic relative data and `NYUDV2Relative` for raw NYU `sync_depth_*.png` data.

If the goal is a sanity check (1-epoch loss-decreasing, exit 0) rather than convergent finetune, use the released checkpoint directly for `evaluate` / `inference` / `export` instead of running `train` — a 1-epoch finetune at any LR is unlikely to reach the released benchmark and will measure the warmup transient, not skill correctness.

The relative variant emits scale-shift-invariant disparity (unbounded). The deploy-side evaluator runs LSQ alignment + GT disparity inversion; ensure the deploy spec sets `model.model_type: RelativeDepthAnything` so those paths engage (see deploy/SKILL.md).

## Metric Variant Finetuning Recipe

**Checkpoint compatibility**: The Metric variant only loads checkpoints trained with TAO's `MetricDepthAnythingV2` model definition. Public Depth Anything v2 metric checkpoints (e.g., from the Depth Anything V2 GitHub release) use a different head attribute naming convention and will fail with `Unexpected key(s) in state_dict: "model.depth_head.*"` when passed to `train.pretrained_model_path`, `evaluate.checkpoint`, `inference.checkpoint`, or `export.checkpoint`. Use a TAO-trained metric checkpoint (or a TAO-converted equivalent) for all metric actions.

Metric finetuning uses a pretrained `RelativeDepthAnything` ViT-L backbone via `model.mono_backbone.pretrained_path`, with the metric head (`metric_depth_head`) initialized from scratch and no full PL state load (`train.pretrained_model_path: ""`). Because the backbone weights are already well-trained, the optimizer must step gently to preserve those features while the metric head converges; use `train.optim.lr: 5e-6` (20× lower than the AdamW default `1e-4` listed in Important Parameters) with `LambdaLR`.

The TAO repository ships an authoritative reference spec at `nvidia_tao_pytorch/cv/depth_net/experiment_specs/experiment_mono_metric.yaml`; metric finetuning **must** mirror its optimizer settings unless the user has empirical evidence to deviate.

**Required overrides for metric finetuning from a relative backbone:**

| Spec key | Recommended value | Source |
|---|---|---|
| `train.optim.lr` | `0.000005` (5e-6) | `experiment_mono_metric.yaml:39` — preserves the pretrained relative backbone while the from-scratch metric head converges. The AdamW default `1e-4` is too aggressive on this backbone-pretrained setup. |
| `train.optim.lr_scheduler` | `LambdaLR` | `experiment_mono_metric.yaml:40` |
| `model.mono_backbone.pretrained_path` | `<RelativeDepthAnything TAO ckpt>` | `experiment_mono_metric.yaml:45` — backbone-only load via `parse_lighting_checkpoint_to_backbone`; metric head reinitializes |
| `train.pretrained_model_path` | `""` | omit a full PL state load to keep the metric head from inheriting any pre-existing head weights |

**Dataset normalization block — required in train AND export specs:**

```yaml
dataset:
  dataset_name: MonoDataset
  normalize_depth: false   # NYU-trained metric checkpoint default
  min_depth: 0.001
  max_depth: 10.0
```

These three fields must mirror the values from the trained checkpoint's training spec in **both** the `train` action spec **and** the `export` action spec. The export pipeline reads `dataset.{normalize_depth, min_depth, max_depth}` to build the model graph the ONNX is traced from; omitting them makes the export silently use schema defaults that do not match the checkpoint, producing a serialized graph whose deploy-side evaluator output is non-physical even though the export action itself returns exit 0. Read the authoritative values from the checkpoint's sibling `experiment.yaml`.

**Defaults already enforced by the TAO trainer (do not need to be set):**

- `train.clip_grad_norm: 0.1` (clip-by-value at the Lightning `Trainer(gradient_clip_val=..., gradient_clip_algorithm="value")` level — `nvidia_tao_pytorch/cv/depth_net/scripts/train.py:94-95`).
- `train.optim.warmup_steps: 20` (linear LR warmup before the configured scheduler engages).
- `train.optim.weight_decay: 1e-4` (AdamW).

**Precision**: use `fp32` for the metric finetune. The from-scratch metric head + low lr combination is fragile under reduced precision; `fp32` is the safe default for this Recipe.

**Sanity-run override** (1-epoch loss-decreasing check on a small NYU subset):

```yaml
train:
  num_epochs: 1
  pretrained_model_path: ""
  precision: fp32
  optim:
    lr: 0.000005
    lr_scheduler: LambdaLR
model:
  model_type: MetricDepthAnything
  encoder: vitl
  mono_backbone:
    pretrained_path: /workspace/models/<relative_ckpt>.pth
    use_bn: False
    use_clstoken: False
```

A 1-epoch run with `metric_depth_head` random init will not reach released-checkpoint metric quality (that requires multi-epoch training); the recipe's purpose is functional sanity (`exit 0` + loss decreasing + no NaN).

**Sanity-run PASS criteria — entrypoint `Execution status: PASS` is not sufficient**:

The trainer's `Execution status: PASS` only signals epoch completion — it does not check for `train_loss = NaN`. A from-scratch metric head with low learning rate can produce `train_loss = NaN` while `val/loss` and the entrypoint PASS remain misleadingly clean. Inspect the `train_loss_step` values in the run log directly; PASS means *only* if the values are finite and decreasing.

Mitigations to try in order if NaN is observed:
- Increase `dataset.train_dataset.batch_size` to 2 or higher (the per-batch variance computation has unstable degrees-of-freedom at batch_size 1).
- Increase `train.optim.warmup_steps` from the default 20 (the LambdaLR factor at step 0 is 0, producing a no-op first update; the second step then sees a head still at random init).
- If both mitigations fail, fall back to reusing a pre-trained TAO metric checkpoint via `train.pretrained_model_path: <metric_ckpt>` and skip the from-scratch metric-head path entirely.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

- `ddp` with activation checkpointing: `find_unused_parameters=False`
- `ddp` without: `find_unused_parameters=True`
- `fsdp` forces precision to FP16

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Export / TRT Defaults

- TRT data types: FP32, BF16 (Ampere SM80+). FP16 is not supported for the ViT-L mono backbone.
- Recommended TRT precision: `bf16`. Use `fp32` if BF16 hardware is unavailable.

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 24GB+ VRAM per GPU. ViT-Large encoder is memory intensive. Use `fp32` (recommended) or `bf16` (Ampere SM80+, alternative) for training. Activation checkpointing is available for larger inputs.

## Error Patterns

**Depth range mismatch**: Ensure `dataset.max_depth` / `dataset.min_depth` match the actual depth range in your data.

**Missing pretrained weights**: DepthAnything v2 encoder requires `model.mono_backbone.pretrained_path` to be set for fine-tuning.

**`Key 'encoder' not in 'MonoBackBone'`**: `encoder` is a top-level `model.encoder` field, not under `mono_backbone`. See Important Parameters.

**`Key 'dataset_name' is not in struct`** under `data_sources`: every `data_sources` entry must include both `data_file` and `dataset_name`.

**`bash: exec: depth_net_mono: not found`**: the unified entrypoint is `depth_net` (no `_mono` / `_stereo` suffix). The skill's `command` already uses the correct form; check any user-supplied wrapper.

**Metric variant hyperparameter sourcing** (`dataset.normalize_depth`, `dataset.train_dataset.augmentation.input_mean`, `dataset.train_dataset.augmentation.input_std`): `MetricDepthAnything` requires depth normalization and ImageNet input statistics that match the checkpoint's training run. These are model- and dataset-specific (not skill-level defaults) — read them from the checkpoint's sibling `experiment.yaml` (or the upstream training spec). Common NYU-trained values: `normalize_depth: false`, `max_depth: 10.0`, `min_depth: 0.001`, `input_mean: [0.485, 0.456, 0.406]`, `input_std: [0.229, 0.224, 0.225]`. Mirror the depth-range values into the export spec — see Metric Variant Finetuning Recipe → Dataset normalization block.

**Export refuses to overwrite an existing ONNX file**: `ValueError: Default onnx file <path> already exists`. The mono export action refuses to overwrite a prior artifact at `export.onnx_file`. Delete or rename the existing file, or change the spec's `export.onnx_file` to a fresh path before re-running.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `depth_net_mono.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `evaluate.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| gen_trt_engine | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| gen_trt_engine | `gen_trt_engine.onnx_file` | `parent_model` | model file inferred from the parent job results folder |
| gen_trt_engine | `gen_trt_engine.trt_engine` | `create_engine_file` | output TensorRT engine path |
| gen_trt_engine | `results_dir` | `output_dir` | current job results directory |
| inference | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| train | `model.mono_backbone.pretrained_path` | `{'link': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth', 'destination_path': '/ptm/depth_net/mono_backbone/dinov2_vitl14_pretrain.pth'}` | {'link': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth', 'destination_path': '/ptm/depth_net/mono_backbone/dinov2_vitl14_pretrain.pth'} |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
