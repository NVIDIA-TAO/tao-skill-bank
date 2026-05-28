---
name: tao-train-fast-foundation-stereo
description: Real-time stereo depth estimation using FastFoundationStereo (FFS), the distilled bp2 commercial variant of
  FoundationStereo. Predicts disparity maps from stereo image pairs with ~10× lower latency than full FoundationStereo. Use
  when training, evaluating, exporting, or running inference for a TAO FastFoundationStereo (FFS) model. Trigger phrases
  include "train fast stereo", "real-time stereo disparity", "FastFoundationStereo", "distilled stereo depth".
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
- realtime
- distilled
---

# Depth Net Fast Stereo

Real-time stereo depth estimation using **FastFoundationStereo (FFS)** — the bp2 commercial distilled variant of FoundationStereo. Predicts disparity maps from rectified stereo image pairs with per-layer pruned widths for real-time inference.

The mono / stereo / fast-stereo skills share the unified TAO `depth_net` CLI; FFS is selected via `model.model_type: FastFoundationStereo`. FFS differs from `FoundationStereo` only in pruned per-layer widths and a serialized forward path; everything else (entrypoint, action verbs, dataset classes, deploy chain) is identical to `depth-net-stereo`.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, TensorRT `inference`), read `deploy/SKILL.md` first. The deploy spec template lives at `references/spec_template_deploy.yaml`.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only; otherwise default to `auto`. When `automl_policy: auto`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Two Use Cases

FFS ships with a pre-trained bp2 commercial checkpoint (`model_best_bp2_serialize.pth`).

1. **Raw deploy** — use the bp2 ckpt as-is. Skip `train`; run `inference` / `evaluate` / `export` / `gen_trt_engine` directly with the bp2 file as the action's checkpoint.
2. **Finetune on user data** — set `train.pretrained_model_path` to the bp2 file, train on user data, then verify + deploy on the resulting ckpt. The full 7-action sequence (train → evaluate pyt → inference pyt → export → gen_trt_engine → inference deploy → evaluate deploy) is supported.

## Workflow

### Prerequisites — data accessibility

Your dataset (left + right images + GT disparity for train / evaluate, left + right only for inference) must be reachable from inside the container:
- **SDK runner**: place files at the S3 paths the runner resolves (`S3_TRAIN` / `S3_EVAL` placeholders shown in **Typical Spec Overrides**).
- **Direct `docker run`** (e.g. local testing): mount the host dataset root read-only at the same in-container path:

```
docker run ... -v <host_data_root>:<host_data_root>:ro <container> ...
```

The same accessibility requirement applies to the `<output_dir>` written by all actions, and to the bp2 checkpoint path.

### Step 1 — Annotation file

Per-line annotation file referenced by `data_sources[*].data_file`. Schema is identical to `depth-net-stereo`:

| Columns | Format | Use |
|---|---|---|
| 2 | `<left> <right>` | Stereo inference (no GT) |
| 3 | `<left> <right> <disparity>` | Stereo with GT |
| 4 | `<left> <right> <disparity> <occlusion_mask>` | Stereo with GT and occlusion mask |

Generate via `depth_net convert` if needed; see the parent `depth-net-stereo` skill for `convert_spec.yaml` template.

### Step 2 — Pair `model_type` and `dataset_name` based on your data

Use `model_type: FastFoundationStereo` for FFS. The `dataset_name` choice mirrors the parent stereo skill — pick the dataset-specific class when your layout matches a registered one, otherwise `GenericDataset`.

| Data category | `model_type` | `dataset_name` |
|---|---|---|
| Middlebury | `FastFoundationStereo` | `Middlebury` |
| KITTI | `FastFoundationStereo` | `Kitti` |
| ETH3D | `FastFoundationStereo` | `Eth3d` |
| FSD synthetic | `FastFoundationStereo` | `FSD` |
| IsaacReal synthetic | `FastFoundationStereo` | `IsaacRealDataset` |
| Crestereo synthetic | `FastFoundationStereo` | `Crestereo` |
| Other / non-canonical | `FastFoundationStereo` | `GenericDataset` |

For inference with 2-column annotations (left + right, no GT), use `dataset_name: GenericDataset` regardless of layout.

### Step 3 — Set the bp2 distilled width overrides

FFS requires 15 model-section width override fields whose values match the bp2 commercial checkpoint exactly. Omitting any field falls back to TAO defaults that do **not** match the bp2 ckpt and produce shape-mismatch errors at forward time.

```yaml
model:
  model_type: FastFoundationStereo
  encoder: vitl
  hidden_dims: [128]                    # 1-layer GRU; NOT [128,128,128]
  n_gru_layers: 1                       # bp2 single-GRU
  corr_radius: 4
  corr_levels: 2
  n_downsample: 2
  valid_iters: 8
  max_disparity: 192                    # bp2 commercial; NOT 416 (full FS default)
  volume_dim: 28                       # bp2 ckpt invariant; NOT 32 (full FS default)
  mixed_precision: false                # see "Important Parameters" below
  gwc_feature_normalize: true           # see "Important Parameters" below

  # 15 bp2 distilled width overrides — copy as-is
  motion_encoder_widths: [56, 96, 16, 12]
  motion_encoder_final: 48
  gru_hidden: 60
  gru_gating_conv_widths: [100, 168]
  disp_head_input_dim: 60
  disp_head_intermediate: 36
  disp_head_pwconv1_widths: [212, 244]
  mask_widths: [32, 16]
  stem_2_widths: [12, 16]
  spx_2_gru_widths: [16, 12, 16, 24]
  spx_gru_out: 9
  classifier_mid: 14
  cnet_conv04_widths: [60, 48]
  cam_mid_channels: 8
  cost_agg_conv_patch_padding: [0, 0, 0]
```

The spec templates at `references/spec_template_*.yaml` carry this block as the canonical source.

### Step 4 — Write spec yaml from Typical Spec Overrides

Copy the action block from **Typical Spec Overrides**. Replace:
- `model.model_type: FastFoundationStereo` (already set)
- `dataset.<...>.data_sources[*].dataset_name` from Step 2
- `dataset.<...>.data_sources[*].data_file` with the path from Step 1
- For raw deploy use cases (no train): set `<action>.checkpoint` to the bp2 file path
- For finetune use cases: set `train.pretrained_model_path` to the bp2 file path

**Chained train → next action checkpoint path**: For local Docker chaining (no SDK runner), the trained checkpoint lives at `<train.results_dir>/<task>/dn_model_latest.pth` — Lightning `ModelCheckpoint` nests under the task name. Example: `train.results_dir: /workspace/results/finetune/train` produces `/workspace/results/finetune/train/train/dn_model_latest.pth`. Use that nested path for the next action's `<action>.checkpoint`. SDK-runner deploys resolve this automatically via `parent_job_id` — see "Spec Param / Parent Model Inference" below.

Shape consistency: `crop_size` in `dataset.test_dataset.augmentation.crop_size` should match `export.input_height` / `input_width` for end-to-end pyt-vs-deploy comparability — see `deploy/SKILL.md`'s shape table.

### Step 5 — Run

```
docker run --gpus 'device=0' --shm-size 16G --ipc=host \
  --user $(id -u):$(id -g) \
  -v <data_root>:<data_root>:ro \
  -v <output_dir>:<output_dir> \
  -v <bp2_ckpt_dir>:<bp2_ckpt_dir>:ro \
  <container> \
  depth_net <action> -e <spec.yaml>
```

Without `--user $(id -u):$(id -g)` the container writes outputs as `nobody:nogroup`, blocking host-side cleanup / retry.

**Local bind-mount tip (QA / development only)**: When bind-mounting a modified TAO repo (`tao-pytorch`, `tao-core`, `tao-deploy`) into the container, stale `__pycache__/*.pyc` files from a previous container run can shadow your patched `.py` source. The symptom is a cryptic TRT-side error (e.g., `IOptimizationProfile::setDimensions Error Code 3`) when the new code path should have produced something different. Clear the caches before launching the container:

```bash
find /path/to/tao-pytorch -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find /path/to/tao-core    -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find /path/to/tao-deploy  -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

SDK-runner production deployments are not affected — the runner copies sources fresh per job.

### Step 6 — Verify

- Container exit code 0
- `status.json` `kpi` block populated
- For `train`: inspect per-step `train_loss` directly (the entrypoint reports `Execution status: PASS` even when loss is NaN)
- For `evaluate`: rely on `epe` / `bp1` / `bp2` / `bp3` / `d1` / `rmse` (the evaluator also emits `abs_rel` / `sq_rel` / `rmse_log` which are non-meaningful for stereo)
- For `inference`: artifacts under `results_dir`
- **KPI namespace difference between pyt and deploy**: pyt `evaluate` writes the metric set under `kpi.val/epe`, `kpi.val/bp1`, etc. (namespaced by Lightning's `val/` prefix). Deploy `evaluate` (TRT engine path) writes the same metric set under `kpi.epe`, `kpi.bp1`, etc. (no `val/` prefix). Downstream verification scripts that read `status.json` need to handle both shapes.
- **Validate drift on your own dataset**: if you compare TAO FFS deploy (`gen_trt_engine` + TRT `evaluate`) against the upstream FFS deploy path on the same input, expect a small residual mean_abs disparity drift (TAO export graph + TRT 10.13 interaction; not improvable at the source-code level). The exact magnitude is dataset and hardware dependent — measure on your own data and decide whether the drift is acceptable for your downstream task.

### 7-action deploy flow

```
train (optional)            → finetuned ckpt
evaluate (pyt)              → PyT eager EPE / bp on val GT
inference (pyt)             → PyT eager disparity samples (visual sanity)
export                      → static fp32 ONNX (recommended at 480×736 or 320×736)
gen_trt_engine              → fp16 TRT engine on static ONNX path
inference (deploy)          → TRT disparity samples
evaluate (deploy)           → TRT EPE / bp drift vs PyT eager fp32
```

Skip `train` for raw-bp2 deploy. The remaining 6 actions (or the 4 deploy-only verbs starting from `export`) cover both use cases.

## Training Requirements

- **Valid `dataset_name` values for stereo `data_sources`** (case-insensitive): `FSD`, `IsaacRealDataset`, `Crestereo`, `Middlebury`, `Eth3d`, `Kitti`, `GenericDataset`
- **Monitoring metric:** val/loss

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.test_dataset.data_sources | eval_dataset | data_file: annotations.txt + dataset_name | Yes |
| inference | dataset.infer_dataset.data_sources | inference_dataset | data_file: annotations.txt + dataset_name | Yes |
| train | dataset.train_dataset.data_sources | train_datasets | data_file: annotations.txt + dataset_name | Yes |
| train | dataset.val_dataset.data_sources | eval_dataset | data_file: annotations.txt + dataset_name | Yes |

### Typical Spec Overrides

Data source overrides are **mandatory for every action**. Each `data_sources` entry is a dict with **two mandatory fields**: `data_file` and `dataset_name`. The `model.*` width fields below are also mandatory — see Step 3.

```python
S3_TRAIN = "aws://bucket/data/train"
S3_EVAL = "aws://bucket/data/eval"
BP2_CKPT = "/workspace/models/ffs/model_best_bp2_serialize.pth"

FFS_MODEL_BLOCK = {
    "model.model_type": "FastFoundationStereo",
    "model.encoder": "vitl",
    "model.hidden_dims": [128],
    "model.n_gru_layers": 1,
    "model.corr_radius": 4,
    "model.corr_levels": 2,
    "model.n_downsample": 2,
    "model.valid_iters": 8,
    "model.max_disparity": 192,
    "model.volume_dim": 28,
    "model.mixed_precision": False,
    "model.gwc_feature_normalize": True,
    "model.motion_encoder_widths": [56, 96, 16, 12],
    "model.motion_encoder_final": 48,
    "model.gru_hidden": 60,
    "model.gru_gating_conv_widths": [100, 168],
    "model.disp_head_input_dim": 60,
    "model.disp_head_intermediate": 36,
    "model.disp_head_pwconv1_widths": [212, 244],
    "model.mask_widths": [32, 16],
    "model.stem_2_widths": [12, 16],
    "model.spx_2_gru_widths": [16, 12, 16, 24],
    "model.spx_gru_out": 9,
    "model.classifier_mid": 14,
    "model.cnet_conv04_widths": [60, 48],
    "model.cam_mid_channels": 8,
    "model.cost_agg_conv_patch_padding": [0, 0, 0],
}
```

**train (finetune from bp2):**
```python
{
    **FFS_MODEL_BLOCK,
    "train.num_epochs": 1,
    "train.checkpoint_interval": 1,
    "train.validation_interval": 1,
    "train.num_gpus": 1,
    "train.precision": "fp32",
    "train.pretrained_model_path": BP2_CKPT,
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

**evaluate (raw bp2 — no train job parent):**
```python
{
    **FFS_MODEL_BLOCK,
    "evaluate.checkpoint": BP2_CKPT,
    "dataset.test_dataset.batch_size": 1,
    "dataset.test_dataset.workers": 4,
    "dataset.test_dataset.augmentation.crop_size": [480, 736],
    "dataset.test_dataset.data_sources": [
        {"data_file": f"{S3_EVAL}/annotations.txt", "dataset_name": "Middlebury"}
    ],
}
```

**inference (raw bp2 — 2-col annotations, no GT):**
```python
{
    **FFS_MODEL_BLOCK,
    "inference.checkpoint": BP2_CKPT,
    "dataset.infer_dataset.batch_size": 1,
    "dataset.infer_dataset.workers": 4,
    "dataset.infer_dataset.data_sources": [
        {"data_file": f"{S3_EVAL}/annotations.txt", "dataset_name": "GenericDataset"}
    ],
}
```

**export (raw bp2):**
```python
{
    **FFS_MODEL_BLOCK,
    "export.checkpoint": BP2_CKPT,
    "export.batch_size": 1,
    "export.input_height": 480,
    "export.input_width": 736,
    "export.opset_version": 17,
    "export.on_cpu": False,
}
```

For finetuned-ckpt actions (post-train), drop the explicit `<action>.checkpoint` and let the SDK resolve it from `parent_job_id` via `parent_model` (see **Spec Param / Parent Model Inference** below).

## Eval Dataset

Optional. Val dataset configured via `dataset.val_dataset.data_sources` (each entry needs `data_file` and `dataset_name`).

## Important Parameters

- **model.model_type**: Must be `FastFoundationStereo` for this skill.
- **model.encoder**: ViT backbone size; bp2 ckpt was trained with `vitl`. Other sizes will fail to load the bp2 weights.
- **model.hidden_dims**: bp2 uses `[128]` (single-GRU). Do **not** use the full-FS default `[128, 128, 128]` — shape-mismatch on the GRU head.
- **model.n_gru_layers**: bp2 uses `1`. Pair with `hidden_dims: [128]`.
- **model.max_disparity**: bp2 commercial uses `192`. The TAO Core schema default for this field is `416` — if the spec yaml's `model:` block does not explicitly set `max_disparity: 192`, OmegaConf falls back to the schema default and the cost volume is built with 2× the correct number of disparity levels (~104 vs the bp2-trained 48 at 1/4 scale). The model still loads and runs, but per-pixel disparity drifts severely from upstream because the cost-volume softmax peak shifts out of the trained regime. **Always set `model.max_disparity: 192` explicitly in the spec for FFS-bp2 deploy** — do not rely on the schema default. The setting on `dataset.max_disparity` is a separate dataset-side knob and does not propagate to the model.
- **model.mixed_precision**: Recommend `false` for FFS-bp2 train and pyt eval. The bp2 commercial ckpt was distilled upstream with bf16 amp, but the FS trainer in TAO does not support bf16 (only fp32 and fp16). Using `mixed_precision: false` (= fp32 forward) gives the cleanest pyt-vs-deploy parity check.
- **model.gwc_feature_normalize**: Must be `true` for FFS-bp2. The bp2 model was trained with normalized group-wise correlation cost volume, and the model code without this flag produces broken disparity (negative values, large drift from upstream baseline). Required for both pyt and deploy paths.
- **model.train_iters**: GRU refinement iterations during training. Default 22.
- **model.valid_iters**: GRU refinement iterations during inference / eval. bp2 ckpt was distilled targeting `8`; values higher than 8 do not improve quality.
- **model.volume_dim**: Cost volume Conv output channels. Schema default `32` (full-FS); FFS bp2 ckpt requires `28` — must override explicitly. Changing breaks bp2 ckpt key-shape match.
- **model.low_memory**: Memory optimization level. Range 0-4. Higher = less memory, slower.
- **dataset.dataset_name**: Top-level dataset family identifier (`StereoDataset`).
- **dataset.{train,val,test,infer}_dataset.batch_size**: Per-split batch size. Use `1` for variable-aspect datasets (Middlebury / KITTI / ETH3D) and during eval / TRT comparison; larger batch sizes are fine for fixed-shape synthetic data.
- **dataset.{train,val,test,infer}_dataset.workers**: Per-split DataLoader worker count.
- **dataset.{train,val,test,infer}_dataset.augmentation.crop_size**: Per-split crop. Match `export.input_height` / `export.input_width` and the deploy-side `evaluate` crop_size for end-to-end shape consistency.
- **dataset.{train,val,test,infer}_dataset.data_sources**: List of `{data_file, dataset_name}` dicts.
- **train.optim.lr**: Learning rate. Default 1e-4 (AdamW). For bp2 finetune, prefer `1e-5` (matches upstream).
- **train.precision**: Training precision. Options: `fp32` (recommended for FFS-bp2), `fp16`. (bf16 is not supported by the FS trainer.)
- **train.distributed_strategy**: Distribution strategy. Options: ddp, fsdp.
- **inference.save_raw_pfm**: Pyt inference action only — when `true`, the per-image disparity is dumped as a raw `.pfm` next to the colorized `.png`. Deploy inference (TRT engine path) emits only the colorized `.png` under `predicted_depth/<scene>_im0.png`; the `save_raw_pfm` knob is not consumed there. Use the pyt inference path if raw `.pfm` output is required.

### Evaluation Metrics

`StereoDepthEvaluator` emits a fixed metric set; only the disparity-domain metrics are meaningful:

| Metric | Meaning | Use |
|---|---|---|
| `epe` | mean End-Point-Error in pixels | primary stereo metric |
| `bp1` / `bp2` / `bp3` | fraction of pixels with EPE > 1 / 2 / 3 px | quality thresholds |
| `d1` | KITTI-style outlier rate (EPE > 3 px AND > 5% of GT disparity) | KITTI-comparable headline |
| `rmse` | RMSE on disparity values | sensitivity to large errors |

The same evaluator also emits `abs_rel`, `sq_rel`, `rmse_log` — these are formulated for monocular metric depth and produce non-meaningful values on disparity. Ignore them for stereo evaluation.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers). Same DDP / FSDP behavior as `depth-net-stereo`.

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

Multi-node requires `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` env vars.

## Export / TRT Defaults

- TRT data types: FP32, FP16.
- Recommended TRT precision for FFS-bp2: `fp16` on the static-shape ONNX path (lowest drift). Dynamic-shape path supports both `fp32` (default; static-fp32 parity) and `fp16` (latency-critical multi-resolution; higher drift than static fp16, may NaN under some checkpoint states — fall back to fp32 if observed). See `deploy/SKILL.md` deployment matrix.
- `export` always emits a **fp32 ONNX** regardless of `model.mixed_precision`. The fp16 vs fp32 selection happens at the `gen_trt_engine` step via `gen_trt_engine.tensorrt.data_type`.
- For static-shape FFS at 480×736: `export.batch_size: 1`, `export.opset_version: 17`, `export.on_cpu: False`.
- **`export.batch_size`**: positive int (default `1`) — static batch dimension; `-1` enables a dynamic batch axis on the ONNX input.
- **`export.dynamic_hw`**: bool (default `false`) — `true` enables dynamic H/W axes on the ONNX input. **FFS only.** FS / mono models ignore this flag with a warning and fall back to static H/W (their DINOv2 backbone constant-folds positional embeddings into the trace, so dynamic H/W at runtime would produce a wrong-shape pos-embed mismatching the actual patch tokens — silent crash). FFS uses EdgeNeXt only and is safe.

### Export use-case matrix

`export.batch_size` and `export.dynamic_hw` are independent. The four combinations:

| Use case | `batch_size` | `dynamic_hw` | Resulting ONNX |
|---|---|---|---|
| Fixed-batch fixed-resolution (most common, production fp16) | `1` (positive) | `false` | static `[1, 3, H, W]` |
| Variable-batch fixed-resolution | `-1` | `false` | dynamic batch only |
| Variable-resolution single-batch (FFS only) | `1` (positive) | `true` | dynamic H/W only |
| Variable-resolution + variable-batch (FFS only) | `-1` | `true` | both batch and H/W dynamic |

For FS / mono models, `dynamic_hw: true` is automatically ignored with a warning and the engine falls back to static H/W. Only `FastFoundationStereo` supports dynamic H/W due to its EdgeNeXt-only encoder.

## Hardware

- Minimum 1 GPU, 24 GB+ VRAM per GPU recommended (A6000 / A100). FFS is ~10× lower-memory than full FoundationStereo at the same input shape, but cost-volume convolution still dominates peak VRAM during training.
- For inference / deploy on edge: A2 / Orin-class GPUs handle FFS at 480×736 fp16 within real-time budget.
- `model.low_memory > 0` for constrained GPUs at training time.
- fp32 recommended for training (bf16 unsupported by FS trainer).

## Error Patterns

**`shape mismatch` at forward**: A `model.*` width override field is missing or wrong. Re-check Step 3 — all 15 fields must be set to the bp2 distilled values exactly.

**`Key 'gwc_feature_normalize' not in 'DepthNetModelConfig'`**: TAO Core too old. The `gwc_feature_normalize` knob requires the FFS-support TAO Core release; upgrade your container or remove the flag (which leaves the model in the broken-output state — see "Important Parameters → gwc_feature_normalize").

**`dynamic_hw: true` warning on FS / mono export**: Expected behavior, not an error. FS / mono models use a DINOv2 backbone that constant-folds positional embeddings into the trace, so dynamic H/W at runtime produces a fixed-size pos-embed mismatching the actual patch tokens (silent crash). The export path detects the model type, emits a warning, and falls back to static H/W. FFS uses EdgeNeXt only and supports `dynamic_hw: true` as documented in the Export use-case matrix.

**`Key 'encoder' not in 'StereoBackBone'`**: `encoder` is a top-level `model.encoder` field, not nested under `stereo_backbone`.

**`Key 'dataset_name' is not in struct`** under `data_sources`: every `data_sources` entry must include both `data_file` and `dataset_name`.

**Negative disparity in pyt evaluate / inference output**: `gwc_feature_normalize: true` is missing or `false`. The bp2 ckpt was trained with normalization on; without it, ~7-8% of pixels predict negative disparity (physically meaningless for stereo).

**Disparity drift much larger than expected vs upstream baseline**: The spec yaml's `model:` block is missing `max_disparity: 192`. OmegaConf falls back to the TAO Core schema default of `416`, which builds a cost volume with 2× the disparity levels the bp2 ckpt was trained for. The model loads and runs, no error is raised, but per-pixel disparity is shifted out of the trained regime. Fix: add `max_disparity: 192` under `model:` (separate from any `dataset.max_disparity` setting — they don't propagate to each other).

**`bash: exec: depth_net_stereo: not found`**: the unified entrypoint is `depth_net` (no `_mono` / `_stereo` / `_fast` suffix).

**Pyt `evaluate` runs at native image resolution (`crop_size` is decorative on the pyt test path)**: same asymmetry as `depth-net-stereo` — the test transform applies only `NormalizeImage` + `PrepareForNet`, no `Resize` / `Crop`. So `dataset.test_dataset.augmentation.crop_size` is read but **not consumed** for the pyt `evaluate` action; samples are fed at the annotation file's native shape. `crop_size` IS authoritative on the deploy side.

**`Failed to import SAM3` warning**: cosmetic only. SAM3 is an unrelated TAO model whose import is attempted at startup; the warning surfaces several times per pyt action (entrypoint init + Lightning callback init + … ). Safe to ignore for FFS — has no effect on training, evaluation, inference, or export.

**Dynamic deploy inference fails silently on stride-incompatible images**: see `deploy/SKILL.md` → "Common errors" → "Dynamic engine inference shape mismatch (silent failure)". Input H × W must be divisible by both 32 (encoder) and 4 (cost-volume); inputs that violate stride-32 produce empty `predicted_depth/` despite `status.json` "finished successfully".

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`.

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder (or set explicitly to bp2 ckpt path for raw deploy) |
| evaluate | `evaluate.trt_engine` | `parent_model` | TRT engine inferred from parent gen_trt_engine job |
| evaluate | `model.model_type` | `FastFoundationStereo` | FastFoundationStereo |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| export | `export.checkpoint` | `parent_model` | model file inferred from parent train job (or bp2 path for raw deploy) |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `model.model_type` | `FastFoundationStereo` | FastFoundationStereo |
| export | `results_dir` | `output_dir` | current job results directory |
| gen_trt_engine | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| gen_trt_engine | `gen_trt_engine.onnx_file` | `parent_model` | model file inferred from parent export job |
| gen_trt_engine | `gen_trt_engine.trt_engine` | `create_engine_file` | output TRT engine path |
| gen_trt_engine | `model.model_type` | `FastFoundationStereo` | FastFoundationStereo |
| gen_trt_engine | `results_dir` | `output_dir` | current job results directory |
| inference | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| inference | `inference.checkpoint` | `parent_model` | pyt path: model file inferred from parent train job (or bp2 path for raw deploy) |
| inference | `inference.trt_engine` | `parent_model` | deploy path: TRT engine inferred from parent gen_trt_engine job |
| inference | `model.model_type` | `FastFoundationStereo` | FastFoundationStereo |
| inference | `results_dir` | `output_dir` | current job results directory |
| train | `dataset.dataset_name` | `StereoDataset` | StereoDataset |
| train | `model.model_type` | `FastFoundationStereo` | FastFoundationStereo |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM (bp2 ckpt) when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train / export / AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. For raw-bp2 use cases without a parent train job, set the `<action>.checkpoint` field explicitly to the bp2 file path. Do not patch generated runner scripts to guess checkpoint paths.
