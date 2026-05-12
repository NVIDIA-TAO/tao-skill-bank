---
name: centerpose
description: CenterPose for keypoint / pose estimation. Detects object centers and regresses keypoint locations. Used for
  6-DoF object pose estimation.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: '0.1'
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- pose
- estimation
---

# CenterPose

CenterPose for keypoint / pose estimation. Detects object centers and regresses keypoint locations. Used for 6-DoF object pose estimation.

Set model.backbone.pretrained_backbone_path.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `deploy/SKILL.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. For AutoML, `schemas/train.schema.json` and `references/spec_template_train.yaml` must exist and parse; otherwise AutoML is unsupported for this model in the plugin workflow. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Training Requirements

- **Dataset type:** centerpose
- **Formats:** default
- **Monitoring metric:** val_3DIoU

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.test_data | eval_dataset | test.tar.gz | No |
| gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset | train.tar.gz | Yes |
| inference | dataset.inference_data | inference_dataset | val.tar.gz | No |
| train | dataset.train_data | train_datasets | train.tar.gz | No |
| train | dataset.val_data | eval_dataset | val.tar.gz | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "dataset.category": "bike",
    "dataset.batch_size": 4,
    "dataset.train_data": f"{S3_TRAIN}/train.tar.gz",
    "dataset.val_data": f"{S3_EVAL}/val.tar.gz",
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.category": "bike",
    "dataset.test_data": f"{S3_EVAL}/test.tar.gz",
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.category": "bike",
    "dataset.inference_data": f"{S3_EVAL}/val.tar.gz",
}
```

**gen_trt_engine (mandatory data sources):**
```python
{
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}/train.tar.gz"],
}
```
## Eval Dataset

Optional. Val and test datasets are provided as separate tarballs.

## Important Parameters

- **dataset.num_classes**: Number of object categories. Default 1.
- **dataset.num_joints**: Number of keypoints per object. Fixed at 8 (bbox keypoints). Valid range: exactly 8.
- **dataset.input_res**: Input resolution. Fixed at 512. Output resolution fixed at 128.
- **dataset.category**: Object category name. Default "cereal_box".
- **model.backbone.model_type**: Default fan_small. Backbone options limited in schema.
- **train.optim.lr**: Learning rate. Default 6e-5. MultiStep scheduler with lr_steps=[90, 120], lr_decay=0.1.
- **train.loss_config**: Rich loss config with toggles: mse_loss, obj_scale, obj_scale_uncertainty, hps_uncertainty, reg_bbox, hm_hp. Weights: wh_weight=0.1, off_weight=1, hp_weight=1.
- **inference.use_pnp**: Use PnP for 6-DoF pose. Default True. Requires camera intrinsics (focal_length_x/y, principle_point_x/y).
- **export.input_width**: Export input size. Fixed at 512x512. opset_version=16.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |

- Strategy: `auto` (Lightning picks the best strategy automatically)
- No explicit `num_nodes` or `distributed_strategy` config — single-node only
- No `sync_batchnorm`

## Export / TRT Defaults

- Export input: 512x512 (fixed), opset 16
- TRT data types: FP32, FP16, INT8
- TRT opt_batch_size: 4, max_batch_size: 8

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 16GB+ VRAM per GPU. CenterPose is moderately memory-intensive depending on input resolution and number of keypoints.

## Error Patterns

**num_joints mismatch**: Ensure dataset.num_joints matches the keypoint count in your annotations.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `centerpose.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `evaluate.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `encryption_key` | `key` | encryption key |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| gen_trt_engine | `encryption_key` | `key` | encryption key |
| gen_trt_engine | `gen_trt_engine.onnx_file` | `parent_model` | model file inferred from the parent job results folder |
| gen_trt_engine | `gen_trt_engine.tensorrt.calibration.cal_cache_file` | `create_cal_cache` | calibration cache path |
| gen_trt_engine | `gen_trt_engine.trt_engine` | `create_engine_file` | output TensorRT engine path |
| gen_trt_engine | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `model.backbone.pretrained_backbone_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
