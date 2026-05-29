---
name: tao-train-visual-changenet
description: Visual ChangeNet for binary image classification and segmentation in AOI defect detection. Use when training,
  evaluating, exporting, or running inference for PCB defect detection or visual inspection, comparing image pairs for
  PASS/NO_PASS classification, or producing change-segmentation masks. Trigger phrases include "train Visual ChangeNet",
  "ChangeNet classify", "ChangeNet segment", "AOI defect detection", "PCB inspection model".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash
tags:
- pcb
- aoi
- defect
- classification
- segmentation
- siamese
- visual-inspection
---

# Visual ChangeNet

Visual ChangeNet is a TAO Toolkit model for visual inspection and defect detection. It supports two tasks:

- **Classify** — Binary image classification using a siamese-style architecture with a shared backbone (C-RADIO ViT) and a learnable difference module. Compares image pairs to classify defects as PASS/NO_PASS.
- **Segment** — Pixel-level change segmentation using a ViT-Large NVDINOv2 backbone. Compares before/after image pairs to produce a binary change mask.

The backbone weight (`c_radio_v2_vit_base_patch16_224`) is the `nvidia/C-RADIOv2-B` model from HuggingFace, distributed as `model.safetensors` (~393 MB). **The TAO 7.0.0-rc container does not auto-fetch from HF URLs** — `ptm_utils.load_pretrained_weights()` hands the `pretrained_backbone_path` value to `torch.load(path)` / `safetensors.torch.load_file(path)` directly. Passing an `https://huggingface.co/...` URL or a repo id produces `FileNotFoundError` and the run fails with `Execution status: FAIL` within a few seconds. Stage the file locally before launch:

```bash
python3 -c "from huggingface_hub import hf_hub_download; import shutil; \
shutil.copy(hf_hub_download('nvidia/C-RADIOv2-B', 'model.safetensors'), '<workspace>/backbone/c_radio_v2_b.safetensors')"
```

Mount it into the container (`-v <workspace>/backbone/c_radio_v2_b.safetensors:/data/pretrained_models/C-RADIOv2_B.safetensors`) and set the spec `model.backbone.pretrained_backbone_path` to the container path. `HF_TOKEN` is only needed at staging time, not at training time.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only; otherwise default to `auto`. When `automl_policy: auto`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference` for classify and segment variants), read `deploy/SKILL.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Training Requirements

Visual ChangeNet has two separate task modes with different dataset types and data source structures.

### Classify

- **Dataset type:** visual_changenet_classify
- **Formats:** default
- **Accepted dataset intents:** training, evaluation, testing, calibration
- **Monitoring metric:** val_loss

#### Per-Action Dataset Requirements (Classify)

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | dataset.classify.train_dataset.images_dir | train_datasets | images.tar.gz | No |
| train | dataset.classify.train_dataset.csv_path | train_datasets | dataset.csv | No |
| train | dataset.classify.validation_dataset.images_dir | eval_dataset | images.tar.gz | No |
| train | dataset.classify.validation_dataset.csv_path | eval_dataset | dataset.csv | No |
| quantize | dataset.classify.train_dataset.images_dir | train_datasets | images.tar.gz | No |
| quantize | dataset.classify.train_dataset.csv_path | train_datasets | dataset.csv | No |
| quantize | dataset.classify.validation_dataset.images_dir | eval_dataset | images.tar.gz | No |
| quantize | dataset.classify.validation_dataset.csv_path | eval_dataset | dataset.csv | No |
| quantize | dataset.classify.quant_calibration_dataset.images_dir | train_datasets | images.tar.gz | No |
| evaluate | dataset.classify.validation_dataset.images_dir | eval_dataset | images.tar.gz | No |
| evaluate | dataset.classify.validation_dataset.csv_path | eval_dataset | dataset.csv | No |
| evaluate | dataset.classify.test_dataset.images_dir | eval_dataset | images.tar.gz | No |
| evaluate | dataset.classify.test_dataset.csv_path | eval_dataset | dataset.csv | No |
| inference | dataset.classify.infer_dataset.images_dir | inference_dataset | images.tar.gz | No |
| inference | dataset.classify.infer_dataset.csv_path | inference_dataset | dataset.csv | No |
| gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset | images.tar.gz | Yes |

### Segment

- **Dataset type:** visual_changenet_segment
- **Formats:** default
- **Accepted dataset intents:** training, calibration
- **Monitoring metric:** val_loss

Segment uses a paired directory structure (`A/`, `B/`, `list/`, `label/`) instead of CSV + images. The `root_dir` spec key points to the top-level directory containing all four subdirectories.

**Required files per dataset:** `A.tar.gz`, `B.tar.gz`, `list.tar.gz`, `label.tar.gz`

#### Per-Action Dataset Requirements (Segment)

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | dataset.segment.root_dir | train_datasets | (root directory) | No |
| quantize | dataset.segment.root_dir | train_datasets | (root directory) | No |
| quantize | dataset.segment.quant_calibration_dataset.images_dir | train_datasets | (root directory) | No |
| evaluate | dataset.segment.root_dir | train_datasets | (root directory) | No |
| inference | dataset.segment.root_dir | train_datasets | (root directory) | No |
| gen_trt_engine | dataset.segment.root_dir | train_datasets | (root directory) | No |
| gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset | images.tar.gz | Yes |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
```

**train (classify, mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "train.use_distributed_sampler": False,
    "train.sync_batchnorm": False,
    "dataset.classify.train_dataset.images_dir": f"{S3_TRAIN}/images.tar.gz",
    "dataset.classify.train_dataset.csv_path": f"{S3_TRAIN}/dataset.csv",
    "dataset.classify.validation_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.classify.validation_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
}
```

**train (segment, mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "train.use_distributed_sampler": False,
    "train.sync_batchnorm": False,
    "dataset.segment.root_dir": f"{S3_TRAIN}",
}
```

**export (classify):**
```python
{
    "export.input_height": 896,
    "export.input_width": 224,
}
```

**export (segment):**
```python
{
    "export.input_height": 224,
    "export.input_width": 224,
}
```

**quantize (classify, mandatory data sources):**
```python
{
    "dataset.classify.train_dataset.images_dir": f"{S3_TRAIN}/images.tar.gz",
    "dataset.classify.train_dataset.csv_path": f"{S3_TRAIN}/dataset.csv",
    "dataset.classify.validation_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.classify.validation_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
    "dataset.classify.quant_calibration_dataset.images_dir": f"{S3_TRAIN}/images.tar.gz",
}
```

**evaluate (classify, mandatory data sources):**
```python
{
    "dataset.classify.validation_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.classify.validation_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
    "dataset.classify.test_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.classify.test_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
}
```

**inference (classify, mandatory data sources):**
```python
{
    "dataset.classify.infer_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.classify.infer_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
}
```

**gen_trt_engine (classify, mandatory data sources):**
```python
{
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}/images.tar.gz"],
}
```

**quantize (segment, mandatory data sources):**
```python
{
    "dataset.segment.root_dir": f"{S3_TRAIN}",
    "dataset.segment.quant_calibration_dataset.images_dir": f"{S3_TRAIN}",
}
```

**evaluate (segment, mandatory data sources):**
```python
{
    "dataset.segment.root_dir": f"{S3_TRAIN}",
}
```

**inference (segment, mandatory data sources):**
```python
{
    "dataset.segment.root_dir": f"{S3_TRAIN}",
}
```

**gen_trt_engine (segment, mandatory data sources):**
```python
{
    "dataset.segment.root_dir": f"{S3_TRAIN}",
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}/images.tar.gz"],
}
```
## Local Docker Invocation

When running without the TAO SDK (local docker), resolve the TAO pyt image from `versions.yaml` and invoke directly:

```bash
set -a; source <workspace>/.env; set +a

# Resolve the TAO pyt container URI from versions.yaml (single source of truth).
TAO_PYT_IMAGE=$("${TAO_SKILL_BANK_PATH:?}/scripts/resolve_versions_key.py" images.tao_toolkit.pyt)

docker run --rm --gpus all --shm-size=8g \
    -e NGC_API_KEY="${NGC_API_KEY}" \
    -v <workspace>:/data/workspace \
    -v <workspace>/results:/results \
    -v <workspace>/kpi/images:/data/datasets/NV_PCB_Siamese/images \
    -v <workspace>/train/base:/data/datasets/NV_PCB_Siamese/csv \
    -v <workspace>/kpi:/data/datasets/NV_PCB_Siamese/kpi \
    -v <workspace>/augmentation/backbone/c_radio_v2_b.ckpt:/data/pretrained_models/C-RADIOv2_B.pth \
    "$TAO_PYT_IMAGE" \
    visual_changenet <action> -e /data/workspace/specs/<spec>.yaml \
    [key=value overrides...]
```

**`--shm-size=8g` is required** — without it, dataloader workers crash with `Unexpected bus error encountered in worker` due to insufficient shared memory.

**Backbone mount**: mount the `.ckpt` file directly as a single file (not the directory), aliased to `/data/pretrained_models/C-RADIOv2_B.pth`.

Override checkpoint and results_dir on the command line to avoid editing the spec:
```bash
visual_changenet inference -e /data/workspace/specs/spec.yaml \
    inference.checkpoint=/results/<iter>/train/model_epoch_<EEE>_step_<SSS>.pth \
    inference.results_dir=/results/<iter>/inference/<label>
```

## Tasks

### Classify (default)

Uses actions: `train`, `evaluate`, `inference`. Defaults template: `references/spec_template_train.yaml`.

### Segment

Uses actions: `segment_train`, `segment_evaluate`, `segment_inference`. Defaults template: `references/spec_template_segment.yaml`.

Segmentation requires compiling custom CUDA ops (`MultiScaleDeformableAttention`) on first run, which takes ~5 minutes. The ViT adapter backbone uses these for multi-scale feature extraction.

Dataset structure for segmentation differs from classify — uses paired directories (`A/`, `B/`, `list/`, `label/`) instead of CSV files. See `dataset.segment.root_dir` in the defaults.

## Data Format

### Classify Inputs

The model needs two things from the dataset: a CSV file and an images directory. Find these in the user's dataset and set the corresponding spec fields:

| Spec field | What to set it to | Description |
|------------|-------------------|-------------|
| `dataset.classify.train_dataset.csv_path` | S3 path to the training CSV | 4-column CSV: `input_path,golden_path,label,object_name` |
| `dataset.classify.train_dataset.images_dir` | S3 path to the images directory | Contains subdirectories referenced by CSV paths |
| `dataset.classify.validation_dataset.csv_path` | S3 path to the validation CSV (optional) | Same 4-column format |
| `dataset.classify.validation_dataset.images_dir` | S3 path to the images directory (optional) | Can be same as training images_dir |

**How to find the right files:** List the dataset URI with `aws s3 ls <uri>` (or your storage CLI equivalent). Look for:
- A CSV with 4 columns (`input_path`, `golden_path`, `label`, `object_name`) — may be in a subdirectory, may have a descriptive name
- An `images/` directory (or similar) containing the image subdirectories referenced by the CSV

### Classify CSV Format

```csv
input_path,golden_path,label,object_name
data/defect,data/golden,bridge,bridge_PCB+solder_00000
```

- **input_path**: Directory path (relative to `images_dir`) containing the test/defect image.
- **golden_path**: Directory path (relative to `images_dir`) containing the golden/reference image.
- **label**: Defect class label (e.g., `bridge`, `PASS`, `NO_PASS`). For binary classification with `num_classes: 2`, the downstream loader collapses all defect labels into one class.
- **object_name**: Filename stem (no extension, no light suffix). TAO constructs the full path as: `{images_dir}/{input_path}/{object_name}_{light_suffix}{image_ext}`.

### Evaluate / Inference Inputs

| Spec field | What to set it to |
|------------|-------------------|
| `dataset.classify.test_dataset.csv_path` | S3 path to test CSV (evaluate) |
| `dataset.classify.test_dataset.images_dir` | S3 path to images (evaluate) |
| `dataset.classify.infer_dataset.csv_path` | S3 path to inference CSV (inference) |
| `dataset.classify.infer_dataset.images_dir` | S3 path to images (inference) |
| `evaluate.checkpoint` | S3 path to trained checkpoint (evaluate) |
| `inference.checkpoint` | S3 path to trained checkpoint (inference) |

### Segment Inputs

| Spec field | What to set it to |
|------------|-------------------|
| `dataset.segment.root_dir` | S3 path to root directory containing `A/`, `B/`, `list/`, `label/` subdirectories |

### Lighting Conventions

TAO builds file paths by string concatenation:

```
{images_dir}/{input_path}/{object_name}_SolderLight.jpg
```

The `input_map` config controls which lighting conditions are loaded and their channel indices. The `object_name` in the CSV must NOT include the light suffix or file extension — TAO appends those.

### Segment Data Layout

Segmentation uses a directory structure instead of CSV:

```
{root_dir}/
  A/           # Before images
  B/           # After images (same filenames as A/)
  list/        # Split files: train.txt, val.txt, test.txt
  label/       # Binary mask PNGs (0=unchanged, 255=changed)
```

The `image_ext` field in the spec (default `.jpg`) must match the actual file extensions in your dataset. If your images are `.png`, set `dataset.classify.image_ext: .png`.

## Lighting Conditions (input_map)

Visual ChangeNet supports multi-lighting-condition input via `dataset.classify.input_map`. Each key is a lighting condition name and the value is its channel index:

```yaml
input_map:
  SolderLight: 0
```

For single-lighting setups, use one entry with index 0. For multi-lighting (e.g., inspection with multiple illumination angles), add entries:

```yaml
input_map:
  SolderLight: 0
  WhiteLight: 1
  UVLight: 2
num_input: 3
```

Set `dataset.classify.num_input` to match the number of lighting conditions. The `grid_map` controls how multi-input images are tiled (default 2x2).

## Important Parameters

- **train.validation_interval**: Default 50. Run validation every N epochs. **IMPORTANT: must be ≤ num_epochs**, otherwise no validation runs and training may fail or produce no metrics. For short runs (e.g., 10 epochs), set to 5.
- **train.checkpoint_interval**: Default 200. Save checkpoint every N epochs. **IMPORTANT: must be ≤ num_epochs**, otherwise no checkpoint is saved and the training output is lost. For short runs, set to match num_epochs or lower.
- **train.num_epochs**: Default 100. Defect detection datasets are typically small, so training may converge in 50-100 epochs. Monitor validation metrics to avoid overfitting.
- **model.classify.train_margin_euclid**: Margin for the Euclidean distance loss during training (default 2.0). Larger values push embeddings further apart. Increase if the model struggles to separate defective from non-defective.
- **model.classify.eval_margin**: Classification threshold during evaluation (default 0.3). Samples with embedding distance below this margin are classified as non-defective; above as defective. This is the primary knob for precision/recall tradeoff -- lower values increase recall (catch more defects), higher values increase precision (fewer false alarms).
- **model.classify.embedding_vectors**: Number of embedding dimensions (default 5). Increase for more complex defect patterns; decrease for simpler binary tasks.
- **dataset.classify.batch_size**: Default 16. Can be increased for small images (224x224) on GPUs with sufficient VRAM.
- **dataset.classify.fpratio_sampling**: False positive ratio for balanced sampling during training (default 0.25). Controls the ratio of non-defective to defective samples in each batch.
- **train.classify.cls_weight**: Class weights for cross-entropy loss (default [1.0, 10.0]). The higher weight on class 1 (defective) compensates for class imbalance typical in defect detection datasets.

## Hardware

- **Minimum**: 1 GPU with 16GB+ VRAM (V100 or A100). Single-GPU training works for small datasets (<10k images).
- **Recommended**: 8 GPUs for production training on larger datasets. Visual ChangeNet uses DDP (DistributedDataParallel) across GPUs.
- GPU count is managed internally by TAO -- do not set `gpu_spec_key` in the spec. The `num_nodes` field (default 1) controls multi-node training.

## Error Patterns

**Checkpoint not found**: The evaluate and inference actions require a valid checkpoint path. If training output was moved or the results_dir changed, update `evaluate.checkpoint` or `inference.checkpoint` to the correct path. The default template `${results_dir}/train/changenet_model_classify_latest.pth` resolves at runtime -- ensure results_dir is set correctly.

**CSV format mismatch**: The CSV must have exactly three columns: `input_path`, `object_name`, `label`. Missing columns or extra headers cause a silent failure or KeyError. Verify the CSV has no BOM characters and uses comma delimiters (not semicolons or tabs).

**Image extension mismatch**: If `dataset.classify.image_ext` is `.jpg` but the actual images are `.png` (or vice versa), the data loader will find zero samples and training will fail with an empty dataset error. Always verify the extension matches your data.

**OOM during training**: Reduce `dataset.classify.batch_size` (16 -> 8 -> 4). With the default image size of 224x224, batch_size=16 typically fits on a 16GB GPU. If using larger images via `image_width`/`image_height`, reduce batch size proportionally.

**Low evaluation accuracy with correct training loss**: The `eval_margin` threshold may be miscalibrated for your data. After training, run inference on a validation set and inspect the embedding distance distribution to pick an appropriate threshold. The default 0.3 is tuned for the reference dataset and may not generalize.

**`AssertionError: Contrastive loss only supports Euclidean distance module`** at evaluate/inference: the spec dropped the `train` subtree. Model `__init__` reads `train.classify.loss` regardless of action; omitting it falls back to contrastive loss, which then conflicts with non-default `model.classify.difference_module` (e.g. `learnable`) saved in the checkpoint. Keep `train.classify.loss` (and `train.classify.cls_weight`) in the spec for evaluate and inference too.

**Training does not converge**: Check that `train.classify.cls_weight` is appropriate for your class distribution. If defects are very rare (<1% of samples), increase the defective class weight. Also verify that `fpratio_sampling` is not too low, which would under-sample the majority class.

**OSError: Could not load MultiScaleDeformableAttention...so** (segment only): CUDA ops not compiled. The ViT adapter backbone requires custom CUDA kernels that must be compiled on first run. Run `python setup.py develop` inside the container (~5 min compilation). This only applies to the segmentation task.

**MisconfigurationException: current_epoch=N, but max_epochs=M**: Old checkpoints in results directory. PyTorch Lightning auto-resumes from checkpoints and crashes if the new `max_epochs` is lower than a previous run's epoch. Fix: use a fresh results directory or unique run name.

**PYTHONPATH / ModuleNotFoundError: nvidia_tao_pytorch**: The TAO entrypoint spawns subprocesses that don't source `.bashrc`. Pass `PYTHONPATH` explicitly via environment variables, not shell init files. The TAO pyt container resolved from `versions.yaml::images.tao_toolkit.pyt` has PYTHONPATH pre-configured.

**Epoch defaults**: Classify training typically uses 100-2000 epochs depending on dataset size. Segmentation uses 200 epochs by default. For small datasets (<1k images), 100 epochs may suffice. For large production datasets, 2000 epochs with early stopping is common. Monitor validation metrics to determine convergence.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from this model skill:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `results_dir` | `output_dir` | current job results directory |
| inference | `results_dir` | `output_dir` | current job results directory |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
