---
name: tao-train-foundation-stereo
description: Stereo depth estimation using FoundationStereo. Predicts disparity maps from stereo image pairs for 3D
  reconstruction. Use when training, evaluating, exporting, or running inference for a TAO FoundationStereo model. Trigger
  phrases include "train stereo depth", "FoundationStereo", "stereo disparity estimation", "3D reconstruction from stereo".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: '0.1'
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- stereo
- depth
- estimation
---

# Depth Net Stereo

Stereo depth estimation using FoundationStereo architecture. Predicts disparity maps from stereo image pairs for 3D reconstruction.

Uses pretrained Depth Anything v2 and EdgeNeXt encoders. Set `model.stereo_backbone.depth_anything_v2_pretrained_path` and `model.stereo_backbone.edgenext_pretrained_path`.

The mono and stereo skills both invoke the unified TAO `depth_net` CLI inside the container; the mono/stereo family is selected via `model.model_type` (e.g., `FoundationStereo`).

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `deploy/SKILL.md` first. The deploy spec template lives in this skill's `references/spec_template_deploy.yaml`.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only; otherwise default to `auto`. When `automl_policy: auto`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Workflow

### Prerequisites — data accessibility

Your dataset (left + right images + GT disparity) must be reachable from inside the container:
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
| 2 | `<left> <right>` | Stereo inference (no GT) |
| 3 | `<left> <right> <disparity>` | Stereo with GT |
| 4 | `<left> <right> <disparity> <occlusion_mask>` | Stereo with GT and occlusion mask |

If you already have one, point to it. Otherwise generate via `depth_net convert`:

```
depth_net convert -e <convert_spec.yaml>
```

`convert_spec.yaml` template (stereo):

```yaml
data_root: <directory whose immediate children are scene folders that contain your image+depth files; convert walks data_root recursively but expects per-scene subdirectories at one level below>
image_dir_pattern: [<substring matching left image paths>]
right_dir_pattern: [<substring matching right image paths>]
depth_dir_pattern: [<substring matching GT disparity paths>]
nocc_dir_pattern: []                 # optional, occlusion mask paths
image_extension: '.png'  # always include the leading dot
depth_extension: '.png'  # form must match image_extension (the swap is a substring replace)
nocc_extension: ''
split_ratio: 0.0        # 0.0/1.0 = test-only; 0.8 = 80/20 train+val
```

`convert` walks `data_root` recursively, selects paths whose path-string contains *all* substrings in `image_dir_pattern` (AND-filter), then derives right / depth / mask paths by replacing `image_dir_pattern[0]` with the corresponding pattern's first element plus extension swap. Inspect your dataset's directory layout and identify the substrings distinguishing left, right, and GT (e.g. `im0` vs `im1` vs `disp0GT` for Middlebury).

### Step 2 — Pair `model_type` and `dataset_name` based on your data

Prefer the dataset-specific class when your layout matches a supported one — it applies class-specific path conventions, evaluation crops, and (where applicable) occlusion-mask handling. Fall back to `GenericDataset` only for layouts that do not match any registered class.

| Data category | `model_type` | `dataset_name` |
|---|---|---|
| Middlebury data | `FoundationStereo` | `Middlebury` |
| KITTI data | `FoundationStereo` | `Kitti` |
| ETH3D data | `FoundationStereo` | `Eth3d` |
| FSD synthetic data | `FoundationStereo` | `FSD` |
| IsaacReal synthetic data | `FoundationStereo` | `IsaacRealDataset` |
| Crestereo synthetic data | `FoundationStereo` | `Crestereo` |
| Other / non-canonical layout | `FoundationStereo` | `GenericDataset` |

See **Training Requirements → Formats** for the full registered-class list. The same `dataset_name` value applies across train and evaluate actions (all of which use 3-column or 4-column annotations with GT disparity). The deploy-side `evaluate` action follows the same rule — see `deploy/SKILL.md`. For inference with 2-column annotations (left + right, no GT), use `dataset_name: GenericDataset` regardless of data layout — the dataset-specific classes (`Middlebury` / `Kitti` / `Eth3d` / `FSD` / `IsaacRealDataset` / `Crestereo`) require 3-column input and reject 2-column annotations at the dataloader level. For inference with 3-column annotations (left + right + GT), the dataset-specific class is fine.

### Step 3 — Write spec yaml from Typical Spec Overrides

Copy the action block from **Training Requirements → Typical Spec Overrides**. Replace:
- `model.model_type` from Step 2 (typically `FoundationStereo`)
- `dataset.<...>.data_sources[*].dataset_name` from Step 2
- `dataset.<...>.data_sources[*].data_file` with the path from Step 1
- For deploy-side `evaluate`: enforce `dataset.test_dataset.batch_size: 1` (see `deploy/SKILL.md`).

Shape consistency: the `crop_size` in `dataset.test_dataset.augmentation.crop_size` should match `export.input_height` / `input_width` so the trained-model evaluator and the deploy-side TensorRT evaluator operate at the same shape — see **Shape consistency** below in this file.

### Step 4 — Run

```
docker run --gpus 'device=0' --shm-size 16G --ipc=host \
  --user $(id -u):$(id -g) \
  -v <data_root>:<data_root>:ro \
  -v <output_dir>:<output_dir> \
  <container> \
  depth_net <action> -e <spec.yaml>
```

Without `--user $(id -u):$(id -g)` the container writes outputs as `nobody:nogroup`, blocking host-side cleanup / retry.

### Step 5 — Verify

- Container exit code 0
- `status.json` `kpi` block populated
- For `train`: inspect per-step `train_loss` directly (the entrypoint reports `Execution status: PASS` even when loss is NaN)
- For `evaluate`: rely on `epe` / `bp1` / `bp2` / `bp3` / `d1` / `rmse` (the evaluator also emits `abs_rel` / `sq_rel` / `rmse_log` which are non-meaningful for stereo — see **Evaluation Metrics** below)
- For `inference`: artifacts under `results_dir`

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `deploy/SKILL.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Training Requirements

- **Valid `dataset_name` values for stereo `data_sources`** (case-insensitive): `FSD`, `IsaacRealDataset`, `Crestereo`, `Middlebury`, `Eth3d`, `Kitti`, `GenericDataset`
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
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "model.model_type": "FoundationStereo",
    "model.encoder": "vits",
    "dataset.train_dataset.batch_size": 1,
    "dataset.train_dataset.workers": 4,
    "dataset.train_dataset.augmentation.crop_size": [320, 736],
    "dataset.train_dataset.data_sources": [
        {"data_file": f"{S3_TRAIN}/annotations.txt", "dataset_name": "Middlebury"}
    ],
    "dataset.val_dataset.batch_size": 1,
    "dataset.val_dataset.workers": 4,
    "dataset.val_dataset.augmentation.crop_size": [320, 736],
    "dataset.val_dataset.data_sources": [
        {"data_file": f"{S3_EVAL}/annotations.txt", "dataset_name": "Middlebury"}
    ],
}
```

**evaluate (mandatory data sources):**
```python
{
    "model.model_type": "FoundationStereo",
    "model.encoder": "vits",
    "dataset.test_dataset.batch_size": 1,
    "dataset.test_dataset.workers": 4,
    "dataset.test_dataset.augmentation.crop_size": [320, 736],
    "dataset.test_dataset.data_sources": [
        {"data_file": f"{S3_EVAL}/annotations.txt", "dataset_name": "Middlebury"}
    ],
}
```

**export:**
```python
{
    "model.model_type": "FoundationStereo",
    "model.encoder": "vits",
    "export.batch_size": 1,
    "export.input_height": 320,
    "export.input_width": 736,
}
```

**gen_trt_engine:**
```python
{
    "gen_trt_engine.batch_size": 1,
}
```

**inference (mandatory data sources):**
```python
{
    "model.model_type": "FoundationStereo",
    "model.encoder": "vits",
    "dataset.infer_dataset.batch_size": 1,
    "dataset.infer_dataset.workers": 4,
    "dataset.infer_dataset.data_sources": [
        {"data_file": f"{S3_EVAL}/annotations.txt", "dataset_name": "GenericDataset"}
    ],
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.train_dataset.data_sources": [
        {"data_file": f"{S3_TRAIN}/annotations.txt", "dataset_name": "Middlebury"}
    ],
    "dataset.val_dataset.data_sources": [
        {"data_file": f"{S3_EVAL}/annotations.txt", "dataset_name": "Middlebury"}
    ],
    "dataset.quant_calibration_dataset.images_dir": f"{S3_TRAIN}/images.tar.gz",
}
```
## Eval Dataset

Optional. Val dataset configured via `dataset.val_dataset.data_sources` (each entry needs `data_file` and `dataset_name`).

## Important Parameters

- **model.model_type**: Architecture. Default `FoundationStereo` for stereo. Only `FoundationStereo` is selectable in the current release.
- **model.encoder**: Backbone encoder (top-level `model` field, not nested under `stereo_backbone`). Options: `vits`, `vitb`, `vitl`, `vitg`. Schema default `vitl`; **FS small NGC ckpt requires `vits` — must override explicitly** (silent shape mismatch on `patch_embed` / ViT block keys without it).
- **model.max_disparity**: Maximum disparity range. Default 416, range 1-416.
- **model.hidden_dims**: Hidden dimensions in GRU refinement. Default `[128, 128, 128]`.
- **model.train_iters**: GRU refinement iterations during training. Default 22.
- **model.volume_dim**: Cost volume dimension. Schema default `32`, but the `FoundationStereo` class hardcodes `volume_dim = 28` at construction (`foundation_stereo.py:51`) — the schema field is currently a no-op for FS. Override is unnecessary; the model always builds at 28.
- **model.low_memory**: Memory optimization level. Range 0-4. Higher = less memory.
- **dataset.dataset_name**: Top-level dataset family identifier (e.g., `StereoDataset`).
- **dataset.baseline**: Stereo camera baseline. Default `193.001/1e3` meters.
- **dataset.focal_x**: Camera focal length X. Default `1998.842`.
- **dataset.{train,val,test,infer}_dataset.batch_size**: Per-split batch size.
- **dataset.{train,val,test,infer}_dataset.workers**: Per-split DataLoader worker count (the field name is `workers`, not `num_workers`).
- **dataset.{train,val,test,infer}_dataset.augmentation.crop_size**: Per-split crop size (e.g., `[320, 736]`). Match `export.input_height`/`export.input_width` and the deploy-side `evaluate` crop_size for end-to-end shape consistency (see `deploy/SKILL.md` for the deploy-side shape table).
- **dataset.{train,val,test,infer}_dataset.data_sources**: List of `{data_file, dataset_name}` dicts. Both fields are mandatory per entry.
- **train.optim.lr**: Learning rate. Default 1e-4 (AdamW).
- **train.precision**: Training precision. Options: fp32 (recommended), fp16. (bf16 is not supported by the FS trainer.)
- **train.distributed_strategy**: Distribution strategy. Options: ddp, fsdp.
- **export.batch_size**: ONNX batch size. `1` = static (matches NGC release), `-1` = batch axis dynamic (height and width are always taken from the trace shape; the DINOv2 + EdgeNeXt backbone constant-folds the patch count, so H/W dynamic is not supported). Default `-1`.

### Evaluation Metrics

`StereoDepthEvaluator` (`nvidia_tao_deploy/cv/depth_net/evaluation/stereo_evaluator.py`) emits a fixed metric set; only the disparity-domain metrics are meaningful for stereo:

| Metric | Meaning | Use |
|---|---|---|
| `epe` | mean End-Point-Error in pixels | primary stereo metric |
| `bp1` / `bp2` / `bp3` | fraction of pixels with EPE > 1 / 2 / 3 px | quality thresholds |
| `d1` | KITTI-style outlier rate (EPE > 3 px AND > 5% of GT disparity) | KITTI-comparable headline |
| `rmse` | RMSE on disparity values | sensitivity to large errors |

The same evaluator also emits `abs_rel`, `sq_rel`, `rmse_log`. These are formulated for monocular depth (relative-error normalised by GT depth in metres) and produce numerically large, **non-meaningful** values when applied to disparity tensors. Ignore them for stereo evaluation; rely on `epe` / `bp*` / `d1` / `rmse`.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

Same DDP/FSDP behavior as depth-net-mono. Multi-node requires `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` env vars.

## Export / TRT Defaults

- TRT data types: FP32, FP16.
- Static-shape ONNX (`export.batch_size: 1`): `fp16` supported (recommended, best EPE).
- Batch-only dynamic ONNX (`export.batch_size: -1`): `fp16` supported. Engine accepts variable batch size; height and width are pinned to the trace shape.
- Height and width are always pinned to the trace shape; H/W-dynamic engines are not supported. Build separate engines for different (H, W) targets.
- For the NGC release (576×960), set `export.batch_size: 1`, `export.opset_version: 17`, `export.on_cpu: True` (CPU export is required at 576×960 to avoid GPU OOM during the trace).
- For user-trained fp16 export, pair `opset_version` to `on_cpu`: `on_cpu: True` (CPU trace) accepts either opset 16 or 17 deterministically; `on_cpu: False` (GPU trace) accepts only opset 16 (opset 17 + on_cpu=False is broken on TRT 10.13 fp16). At `on_cpu=False + opset 16` the fp16 build is occasionally non-deterministic — re-run on a `costTensor::indexOfMin` or `optimizer::reduce` assertion. fp32 builds are unaffected. See `deploy/SKILL.md` for the validation table.
- `export.on_cpu` is driven by GPU trace memory: `False` for ≤320×736 (fits 47 GB VRAM), `True` for ≥480×736 (PyTorch trace OOMs at GPU). Prefer `on_cpu: True` whenever feasible — fp16 builds at `on_cpu=True` are empirically deterministic at every tested shape (including NGC release 576×960).
- See `deploy/SKILL.md` for the three supported deploy paths (NGC static / user-trained static / user-trained batch-only-dynamic).

## Hardware

Minimum 1 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. Stereo matching is memory intensive due to cost volume. Use `model.low_memory > 0` for constrained GPUs. fp32 recommended for training.

## Error Patterns

**Disparity overflow**: Reduce `model.max_disparity` if targets exceed range or OOM occurs.

**Missing pretrained paths**: Both `model.stereo_backbone.depth_anything_v2_pretrained_path` and `model.stereo_backbone.edgenext_pretrained_path` should be set for fine-tuning.

**`Key 'encoder' not in 'StereoBackBone'`**: `encoder` is a top-level `model.encoder` field, not under `stereo_backbone`. See Important Parameters.

**`Key 'dataset_name' is not in struct`** under `data_sources`: every `data_sources` entry must include both `data_file` and `dataset_name`.

**`bash: exec: depth_net_stereo: not found`**: the unified entrypoint is `depth_net` (no `_mono` / `_stereo` suffix). The skill's `command` already uses the correct form; check any user-supplied wrapper.

**Pyt `evaluate` runs at native image resolution (`crop_size` is decorative on the pyt test path)**: the stereo data module's test transform is built with `split='infer'` (`pl_stereo_data_module.py`), which applies only `NormalizeImage` + `PrepareForNet` — no `Resize`/`Crop`. So `dataset.test_dataset.augmentation.crop_size` is read but **not consumed** for the pyt `evaluate` action; samples are fed at the annotation file's native shape. For variable-aspect datasets like Middlebury, point the test annotation file at a resolution that fits GPU memory (e.g., MiddEval3-data-Q at 718×496 instead of MiddEval3-data-H at 1428×988 for the small variant on 24–48 GB GPUs). This asymmetry is pyt-only — `crop_size` IS authoritative on the deploy `evaluate` side (the deploy runtime reads it; see `deploy/SKILL.md`).

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `depth_net_stereo.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `evaluate.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `model.model_type` | `FoundationStereo` | FoundationStereo |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `model.model_type` | `FoundationStereo` | FoundationStereo |
| export | `results_dir` | `output_dir` | current job results directory |
| gen_trt_engine | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| gen_trt_engine | `gen_trt_engine.onnx_file` | `parent_model` | model file inferred from the parent job results folder |
| gen_trt_engine | `gen_trt_engine.trt_engine` | `create_engine_file` | output TensorRT engine path |
| gen_trt_engine | `model.model_type` | `FoundationStereo` | FoundationStereo |
| gen_trt_engine | `results_dir` | `output_dir` | current job results directory |
| inference | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `model.model_type` | `FoundationStereo` | FoundationStereo |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| quantize | `model.model_type` | `FoundationStereo` | FoundationStereo |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| train | `model.model_type` | `FoundationStereo` | FoundationStereo |
| train | `model.stereo_backbone.depth_anything_v2_pretrained_path` | `{'link': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth', 'destination_path': '/ptm/depth_net/stereo_backbone/depth_anything_v2_vits.pth'}` | {'link': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth', 'destination_path': '/ptm/depth_net/stereo_backbone/depth_anything_v2_vits.pth'} |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
