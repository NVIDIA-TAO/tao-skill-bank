---
name: tao-train-dino
description: DINO (DETR with Improved DeNoising Anchor Boxes) for 2D object detection. Transformer-based detector with
  denoising training, multi-scale features, and optional distillation support. Use when training, evaluating, exporting,
  distilling, quantizing, or running inference for a TAO DINO detector. Trigger phrases include "train DINO", "DETR object
  detection", "TAO 2D detection", "DINO with distillation".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: '0.1'
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- object
- detection
---

# DINO

DINO (DETR with Improved DeNoising Anchor Boxes) for 2D object detection. Transformer-based detector with denoising training, multi-scale features, and optional distillation support.

Uses pretrained backbone weights (e.g. ResNet-50 ImageNet). Set `model.pretrained_backbone_path` for backbone-only or `train.pretrained_model_path` for full model.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and
TensorRT `inference`), read `deploy/SKILL.md` first. Deploy spec templates live
in this skill's `references/` folder with the `spec_template_deploy_*.yaml`
prefix.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only; otherwise default to `auto`. When `automl_policy: auto`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

The agent MUST read this section before generating any training or AutoML script for DINO.

- **Dataset type:** object_detection
- **Formats:** coco, coco_raw
- **Accepted dataset intents:** training, evaluation, testing, calibration
- **Monitoring metric:** val_mAP50

**Required datasets — MUST resolve both:**

| Dataset | Required | Why |
|---|---|---|
| Train dataset URI | Yes | Training data (COCO format) |
| Validation dataset URI | **Yes — ALWAYS** | DINO unconditionally builds a val dataloader. Omitting `val_data_sources` causes `FileNotFoundError` at startup regardless of the metric or workflow. If the user has no separate eval split, reuse the train URI. |

**Required inputs before generating any training spec:**

1. **Train dataset URI** — S3 path to COCO-format training data
2. **Validation dataset URI** — S3 path to COCO-format val data (can be same as train)
3. **`num_classes`** — How many object classes? Default 91 (COCO). Must be >= `max(category_id) + 1`. Too low causes `CUDA error: device-side assert triggered`.

Resolve these from the user request or the default profile below. Prompt only
for values that are still missing after applying the profile rules.

**Bankable local default profile for DINO AutoML smoke runs:**

Use this profile only when the user asks to run DINO AutoML and does not provide
dataset or class-count inputs. This profile is intentionally small and local to
this skill bank; it is for smoke/iteration runs, not a production benchmark.
Do not search previous runners, logs, session state, shell history, or the home
directory to recover these values.

```python
DINO_AUTOML_PROFILE = {
    "train_dataset_uri": "s3://nvcf-storage-handling/data/tao_od_synthetic_subset_train_no_convert",
    "validation_dataset_uri": "s3://nvcf-storage-handling/data/tao_od_synthetic_subset_val_no_convert",
    "object_classes": 4,
    "dataset_num_classes": 5,
    "image_archive": "images.tar.gz",
    "annotation_file": "annotations.json",
    "max_recommendations": 10,
    "train_num_epochs": 10,
    "train_checkpoint_interval": 10,
    "train_validation_interval": 1,
    "train_num_gpus": 1,
}
```

If the user supplies any dataset URI or class-count value, prefer the user value
and ask for any remaining required DINO value. Do not partially mix a user's
custom dataset with this profile's class count unless the user confirms it.

**Do not prompt for image layout for the standard DINO dataset.** The standard
TAO DINO dataset artifact is `images.tar.gz` plus `annotations.json`. Use
`images.tar.gz` in the remote `image_dir` spec override. The SDK downloads the
archive and rewrites the runtime spec to the extracted folder named after the
archive stem (`images.tar.gz` -> `images`). Only deviate if the user explicitly
provides a different image artifact name.

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| distill | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| distill | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| evaluate | evaluate.checkpoint | trained_model | DINO .pth/.tlt checkpoint | No |
| evaluate | dataset.test_data_sources.image_dir | eval_dataset | images.tar.gz | No |
| evaluate | dataset.test_data_sources.json_file | eval_dataset | annotations.json | No |
| gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset | images.tar.gz | Yes |
| inference | dataset.infer_data_sources.image_dir | inference_dataset | images.tar.gz | Yes |
| inference | dataset.infer_data_sources.classmap | inference_dataset | label_map.txt | No |
| quantize | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| quantize | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| quantize | dataset.quant_calibration_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | No |
| train | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| train | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — DINO's `config.json` has empty `data_sources` because the runner cannot auto-resolve array-of-objects spec keys (see Internal Details). The agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_VAL = "s3://bucket/data/val"    # can be same as S3_TRAIN
S3_EVAL = "s3://bucket/data/eval"  # for evaluate/inference

# Standard DINO dataset artifact. Pass the archive path as the remote input.
# At runtime the SDK extracts it and points DINO at the extracted "images" folder.
IMAGE_ARCHIVE = "images.tar.gz"
```

**train (mandatory):**
```python
{
    "dataset.train_data_sources": [
        {"image_dir": f"{S3_TRAIN}/{IMAGE_ARCHIVE}", "json_file": f"{S3_TRAIN}/annotations.json"}
    ],
    "dataset.val_data_sources": [
        {"image_dir": f"{S3_VAL}/{IMAGE_ARCHIVE}", "json_file": f"{S3_VAL}/annotations.json"}
    ],
    "dataset.num_classes": "<num_classes> + 1",
    "train.num_epochs": 10,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
}
```

**evaluate (mandatory checkpoint + data sources):**
```python
{
    "evaluate.checkpoint": "<checkpoint_uri>",
    "dataset.test_data_sources.image_dir": f"{S3_EVAL}/{IMAGE_ARCHIVE}",
    "dataset.test_data_sources.json_file": f"{S3_EVAL}/annotations.json",
    "dataset.num_classes": "<num_classes> + 1",
    "model.backbone": "<backbone used for training>",
    "model.num_queries": "<num_queries used for training>",
    "model.dropout_ratio": "<dropout_ratio used for training>",
}
```

For standard DINO eval datasets, do not search S3 to discover filenames. Build
the eval image and annotation URIs directly from the eval dataset base URI using
`images.tar.gz` and `annotations.json`, unless the user explicitly provides a
different layout.

For a DINO model trained by this SDK or by an AutoML child train job, prefer
microservices-style parent model inference instead of hardcoding the checkpoint
URI. Use this model-MD inference mapping:

```json
"spec_params": {
  "evaluate": {
    "evaluate.checkpoint": "parent_model"
  }
}
```

Use the train job id, or the AutoML best child train job id, as
`parent_job_id`. The SDK will list the parent result folder, filter `.pth`
checkpoints, and select the model file:

```python
checkpoint_uri = sdk.resolve_spec_param(
    eval_job_id,
    "parent_model",
    network_arch="dino",
    parent_job_id=train_job_id,
)
```

Equivalently, when resolving the checkpoint outside a spec-param loop:

```python
checkpoint_uri = sdk.get_model_results_path(train_job_id, network_arch="dino")
```

If cloud listing is unavailable but only the training job id is known, the
expected DINO fallback location is:

```python
checkpoint_uri = f"s3://{S3_BUCKET_NAME}/results/{train_job_id}/results_dir/train/dino_model_latest.pth"
```

Do not use `s3://<bucket>/results/<train_job_id>/dino_model_latest.pth`; DINO
training uploads checkpoints under `results_dir/train/`.

When evaluating an AutoML-trained model, carry forward the winning rec's
structural model settings into the eval spec. At minimum copy
`model.backbone`, `model.num_queries`, `model.dropout_ratio`, and
`dataset.num_classes`. If future HPO runs tune additional structural model
fields, copy those too so the checkpoint shape matches the evaluation model.

**export:**
```python
{
    "dataset.num_classes": "<num_classes> + 1",
}
```

**gen_trt_engine (mandatory data sources):**
```python
{
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}/{IMAGE_ARCHIVE}"],
    "gen_trt_engine.tensorrt.data_type": "FP16",
    "dataset.num_classes": "<num_classes> + 1",
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.infer_data_sources.image_dir": [f"{S3_EVAL}/{IMAGE_ARCHIVE}"],
    "dataset.infer_data_sources.classmap": f"{S3_EVAL}/label_map.txt",
    "dataset.num_classes": "<num_classes> + 1",
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.train_data_sources": [
        {"image_dir": f"{S3_TRAIN}/{IMAGE_ARCHIVE}", "json_file": f"{S3_TRAIN}/annotations.json"}
    ],
    "dataset.val_data_sources": [
        {"image_dir": f"{S3_VAL}/{IMAGE_ARCHIVE}", "json_file": f"{S3_VAL}/annotations.json"}
    ],
    "dataset.quant_calibration_data_sources": {
        "image_dir": f"{S3_TRAIN}/{IMAGE_ARCHIVE}", "json_file": f"{S3_TRAIN}/annotations.json"
    },
    "dataset.num_classes": "<num_classes> + 1",
}
```

**distill (mandatory data sources):**
```python
{
    "dataset.train_data_sources": [
        {"image_dir": f"{S3_TRAIN}/{IMAGE_ARCHIVE}", "json_file": f"{S3_TRAIN}/annotations.json"}
    ],
    "dataset.val_data_sources": [
        {"image_dir": f"{S3_VAL}/{IMAGE_ARCHIVE}", "json_file": f"{S3_VAL}/annotations.json"}
    ],
    "dataset.num_classes": "<num_classes> + 1",
}
```

## Dataset

COCO JSON format. train_data_sources and val_data_sources are lists supporting multiple data source entries. Each entry has image_dir and json_file (COCO annotations JSON).

**`image_dir` remote path**: For the standard TAO DINO dataset, set
`image_dir` to the archive path, e.g. `s3://bucket/data/images.tar.gz`.
The SDK downloads and extracts it, then rewrites the runtime training spec to
the extracted folder path, e.g. `/mnt/lustre/.../images`.

Do not ask the user whether to use `images` or `images.tar.gz` for standard
DINO datasets. Use `images.tar.gz`. If the user explicitly supplies a different
archive filename, derive the runtime folder from the archive stem:
`<name>.tar.gz` -> `<name>`, `<name>.tgz` -> `<name>`, `<name>.tar` -> `<name>`.

Supported formats: coco, coco_raw.

### Train Data Sources

- **image_dir**: `images.tar.gz` remote archive; runtime folder is `images`
- **json_file**: `annotations.json`

### Val Data Sources (ALWAYS required)

- **image_dir**: `images.tar.gz` remote archive; runtime folder is `images`
- **json_file**: `annotations.json`

### Inference Data Sources

- **image_dir**: `images.tar.gz` remote archive; runtime folder is `images`
- **classmap**: `label_map.txt`

### Evaluate Data Sources

- **checkpoint**: `evaluate.checkpoint`, a `.pth` or `.tlt` model file. For SDK
  train jobs and AutoML child train jobs, resolve it with `parent_model`
  inference so the SDK lists the result folder and selects an actual checkpoint
  file. If listing is unavailable, fall back to
  `results_dir/train/dino_model_latest.pth` under the training job's uploaded
  result directory.
- **image_dir**: `images.tar.gz` remote archive; runtime folder is `images`
- **json_file**: `annotations.json`

## Important Parameters

- **dataset.num_classes**: Number of object classes. Default is 91 (COCO). Must be >= `max(category_id) + 1`. Too low causes `CUDA error: device-side assert triggered`.
- **model.backbone**: Backbone architecture. Default resnet_50. Supported: resnet_34, resnet_50, fan_small_12_p4_hybrid, fan_base_16_p4_hybrid, fan_large_16_p4_hybrid, gcvit_tiny, gcvit_small, gcvit_base, gcvit_large, nvdinov2_vit_large_legacy, swin_tiny_224_1k, swin_small_224_1k, swin_base_224_22k, swin_large_224_22k, efficientvit_l2_224, efficientvit_l2_384.
- **train.optim.lr**: Learning rate. Default 2e-4 (AdamW). lr_backbone defaults to 2e-5 (10x lower). Reduce both if training diverges.
- **train.num_epochs**: DINO typically needs 30-50+ epochs for good mAP on real datasets. The default of 10 is suitable for quick iteration.
- **train.optim.lr_steps**: MultiStep LR decay schedule. Default [11]. For longer training, set to e.g. [30, 40] for a 50-epoch run.
- **model.num_queries**: Number of object queries. Default 300. Increase for dense scenes with many objects per image. num_select must be < num_queries * num_classes.
- **dataset.batch_size**: Per-GPU batch size. Default 4. Reduce to 2 if OOM on 16GB GPUs. Total batch = batch_size * num_gpus.

## Default Values

- **num_epochs**: `10`
- **batch_size**: `4`
- **learning_rate**: `2e-4`
- **lr_backbone**: `2e-5`
- **num_classes**: `91`
- **backbone**: `resnet_50`

## Evaluate Defaults

Use `references/spec_template_evaluate.yaml` (when present) as the base spec
for `action="evaluate"`, then apply the mandatory checkpoint and data-source
overrides above. `references/skill_info.yaml` declares the required evaluate
inputs so the SDK script runner downloads and rewrites them before running
the container. This model MD also documents
`evaluate.checkpoint = parent_model`, so generated runners should infer the
checkpoint from the parent job result files before submission:

```json
{
  "evaluate.checkpoint": {"type": "file"},
  "dataset.test_data_sources.image_dir": {"type": "file"},
  "dataset.test_data_sources.json_file": {"type": "file"}
}
```

## Export Defaults

- **input_width**: `640`
- **input_height**: `640`
- **opset_version**: `17`
- **trt_data_types**: `[FP32, FP16, INT8]`
- **trt_workspace_size_mb**: `1024`

## Hardware

- **Minimum**: 1 GPU
- **Recommended**: 4 GPUs
- **GPU Memory**: 24GB+ (A100 recommended)

Transformer-based detection is memory-intensive. batch_size=4 fits on 24GB GPUs. For 16GB GPUs, reduce to batch_size=2. Multi-GPU with 4+ GPUs recommended for datasets > 10k images.

## Error Patterns

**CUDA out of memory**: Reduce dataset.batch_size (4 -> 2 -> 1). DINO uses multi-scale features that consume significant GPU memory, especially with high-resolution images (default max 1333px).

**num_select must be < num_queries * num_classes**: Ensure model.num_select (default 300) is less than num_queries * dataset.num_classes.

**Error merging spec.yaml with schema**: Hydra/OmegaConf validation error. num_epochs and num_gpus must be under 'train.*', not at spec root. Use the SDK spec_shorthand_keys mapping.

**Dataset size smaller than total batch size**: Total batch = batch_size * num_gpus. If val dataset has fewer samples, reduce dataset.batch_size or num_gpus. The agent should proactively check this.

**return_interm_indices length must match num_feature_levels**: Default is [1,2,3,4] with num_feature_levels=4. If changing one, update the other.

**`FileNotFoundError` on images**: The archive extraction/cache and annotation paths are out of sync. For standard DINO datasets, pass remote `images.tar.gz`; the SDK should rewrite the runtime spec to `images`. If DINO looks under `/mnt/lustre/.../images/<file>.jpg` and files are missing, clear the stale `<images.tar.gz>.extracted` marker and re-extract/download the archive, or inspect the archive top-level layout.

**`FileNotFoundError` at startup (val)**: `val_data_sources` missing or pointing to non-existent data. DINO unconditionally builds a val dataloader — this is required even when only optimizing `train_loss`.

**`CUDA device-side assert`**: `num_classes` too low. Set `num_classes >= max(category_id) + 1`.

**S3 inputs not downloaded inside container**: When the agent invokes DINO via SDK orchestration, `references/skill_info.yaml` must declare `actions.train.inputs` with `[0]`-indexed spec keys (see "Optional: SDK orchestration internals"). Use `s3://...` for S3-compatible datasets; do not generate `aws://...` URIs.

**Evaluate checkpoint not found at result root**: DINO train jobs upload
checkpoints under `results_dir/train/`. If eval fails with `FileNotFoundError`
for `s3://<bucket>/results/<train_job_id>/dino_model_latest.pth`, set
`evaluate.checkpoint` to
`s3://<bucket>/results/<train_job_id>/results_dir/train/dino_model_latest.pth`.

## AutoML / HPO Notes

AutoML runs training — all requirements from **Training Requirements** above apply. The agent must read that section first.

For no-input local DINO AutoML smoke runs, use `DINO_AUTOML_PROFILE` from
**Training Requirements**. Do not inspect previous AutoML runs to infer dataset
URIs, `num_classes`, recommendation count, or interval settings.

**Recommended AutoML metric:** use explicit `metric="mAP50"` with
`direction="maximize"` and pass a custom `metric_extractor` that reads
`Validation mAP50`. Do not rely on `metric="kpi"` for generated DINO runners
unless you have verified the local resolver maps it to mAP50; loose fallback
parsing can otherwise optimize `val_loss`.

```python
import re

def extract_dino_map50(logs, metric_name):
    matches = re.findall(
        r"Validation mAP50\s*:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
        logs,
    )
    return float(matches[-1]) if matches else None

runner.run(
    ...,
    automl_settings={"metric": "mAP50", "direction": "maximize", ...},
    metric_extractor=extract_dino_map50,
)
```

**Recommended hyperparameters:**

```python
automl_hyperparameters=[
    "train.optim.lr",
    "train.optim.weight_decay",
    "model.backbone",
    "model.num_queries",
    "model.dropout_ratio",
]
custom_param_ranges={
    "train.optim.lr": {"valid_min": 1e-5, "valid_max": 5e-4},
    "model.backbone": {
        "valid_options": ["resnet_50", "resnet_34"],
        "option_weights": [0.75, 0.25],
    },
    "model.num_queries": {"valid_min": 100, "valid_max": 900},
    "model.dropout_ratio": {"valid_min": 0.0, "valid_max": 0.3},
}
```

`train.optim.weight_decay` is not in the default DINO spec schema — the runner accepts it with a warning. It still works; the DINO training code picks it up from the config.

**Backbone constraint for AutoML:** The LLM brain may propose backbone names not in the supported list (see Important Parameters above), e.g. `fan_small`, `fan_tiny`, `efficientvit_b2`. These cause training failures. Use `custom_param_ranges` to constrain categorical params when possible.

## Optional: SDK orchestration internals

The following details are only relevant when running DINO via the TAO SDK
(`script_runner` orchestration, S3 I/O wrapping, AutoML). Skills consumed by
the SDK read `references/skill_info.yaml` for these mappings. Skip this
section if running locally with `docker run`.

### Internal Details

#### Spec templates

DINO ships without `references/spec_template_train.yaml` or
`references/spec_template_evaluate.yaml`. To use SDK orchestration, generate
them from upstream:

- `spec_template_train.yaml` ← `tao-pytorch/nvidia_tao_pytorch/cv/dino/experiment_specs/train.yaml` (replace `"???"` placeholders with empty strings).
- `spec_template_evaluate.yaml` ← `tao-pytorch/nvidia_tao_pytorch/cv/dino/experiment_specs/evaluate.yaml` plus the shared `evaluate.checkpoint` field expected by `initialize_evaluation_experiment()`.

#### Data Sources Gap

DINO's `config.json` has `"data_sources": {}` (empty). The runner's `_apply_data_sources()` only handles flat spec keys (like cosmos-rl's `custom.train_dataset.annotation_path`), but DINO's data sources are **arrays of objects** (`dataset.train_data_sources[{image_dir, json_file}]`). The tao-core microservices config (`tao-core/nvidia_tao_core/microservices/handlers/network_configs/dino.config.json`) has the full mapping using a `mapping` sub-structure, but the runner doesn't support that format.

**Consequence:** The runner cannot auto-resolve data URIs for DINO. Data paths MUST be set manually via `spec_overrides` (see Training Requirements above). The skill's `config.json` instead declares `inputs` in the train action with `[0]`-indexed spec keys so the SDK's script_runner downloads S3 data at runtime:

```json
"inputs": {
    "dataset.train_data_sources[0].image_dir": {"type": "file"},
    "dataset.train_data_sources[0].json_file": {"type": "file"},
    "dataset.val_data_sources[0].image_dir": {"type": "file"},
    "dataset.val_data_sources[0].json_file": {"type": "file"}
}
```

The skill also declares evaluate inputs so generated eval runners do not need
to patch `script_runner` by hand:

```json
"inputs": {
    "evaluate.checkpoint": {"type": "file"},
    "dataset.test_data_sources.image_dir": {"type": "file"},
    "dataset.test_data_sources.json_file": {"type": "file"}
}
```

This model MD is the source of truth for DINO checkpoint inference:

```text
checkpoint format: pth
evaluate.checkpoint: parent_model
```

All model-specific metadata (dataset type, formats, metrics, required datasets) is documented in the **Training Requirements** section above.

**TODO:** Extend the runner's `_apply_data_sources()` to handle the `mapping` sub-structure from tao-core so DINO can use auto-resolved data sources like cosmos-rl does.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `dino.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| distill | `distill.pretrained_teacher_model_path` | `parent_model` | model file inferred from the parent job results folder |
| distill | `encryption_key` | `key` | encryption key |
| distill | `results_dir` | `output_dir` | current job results directory |
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
| quantize | `encryption_key` | `key` | encryption key |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `model.pretrained_backbone_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
