---
name: vila
description: VILA vision-language model for multimodal understanding tasks. Supports video and image-based question answering,
  captioning, and reasoning. Fine-tunable with LoRA or full fine-tuning on custom VLM datasets.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: '0.1'
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- vision
- language
- model
---

# VILA

VILA vision-language model for multimodal understanding tasks. Supports video and image-based question answering, captioning, and reasoning. Fine-tunable with LoRA or full fine-tuning on custom VLM datasets.

Set model_path to the base VILA model checkpoint.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. For AutoML, `schemas/train.schema.json` and `references/spec_template_train.yaml` must exist and parse; otherwise AutoML is unsupported for this model in the plugin workflow. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Training Requirements

- **Dataset type:** vlm
- **Formats:** default, raw
- **Accepted dataset intents:** training, evaluation, testing
- **Monitoring metric:** val_acc

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | train.dataset.dataset_yaml_path | train_datasets | dataset.yaml | No |
| evaluate | eval.dataset_yaml_path | eval_dataset | dataset.yaml | No |
| inference | inference.media | inference_dataset | (dataset root) | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
```

**train (mandatory data sources):**
```python
{
    "train.dataset.dataset_yaml_path": f"{S3_TRAIN}/dataset.yaml",
    "train.num_epochs": 1,
    "train.batch_size": 8,
    "train.learning_rate": 1e-4,
    "train.vision_learning_rate": 1e-5,
    "train.lora_r": 16,
    "train.llm_mode": "lora",
}
```

**evaluate (mandatory data sources):**
```python
{
    "eval.dataset_yaml_path": f"{S3_EVAL}/dataset.yaml",
}
```

**inference (mandatory data sources):**
```python
{
    "inference.media": f"{S3_EVAL}/",
}
```

## Eval Dataset

Optional. Evaluation dataset configured via evaluate.dataset_yaml_path.

## Important Parameters

- **model_path**: Path to base VILA model. Required.
- **train.num_epochs**: Training epochs. Default 1 (VLM fine-tuning is typically short).
- **train.batch_size**: Per-GPU batch size. Default 8.
- **train.learning_rate**: LLM learning rate. Default 1e-4.
- **train.vision_learning_rate**: Vision encoder learning rate. Default 1e-5 (lower for stability).
- **train.lora_r**: LoRA rank. Default 16.
- **train.llm_mode**: LLM training mode. Default "lora". Options: lora, ft (full).
- **train.vision_mode**: Vision encoder mode. Default "ft" (full fine-tune).
- **train.model_max_length**: Maximum sequence length. Default 32768.
- **train.max_tiles**: Maximum image tiles. Default 12.
- **train.video_max_tiles**: Maximum video tiles. Default 6.
- **train.num_video_frames**: Number of video frames. Default 8.
- **train.gradient_accumulation_steps**: Gradient accumulation. Default 2.
- **train.warmup_ratio**: LR warmup ratio. Default 0.03.
- **train.dataset.dataset_yaml_path**: Path to dataset YAML configuration.
- **evaluate.task**: Evaluation benchmark. Default "youcook2_val".
- **inference.text**: Inference prompt. Default "What is this video about?"
- **inference.conv_mode**: Conversation mode. Default "auto".

## Hardware

Minimum 1 GPU(s), recommended 8 GPU(s). 40GB+ (A100 80GB recommended) VRAM per GPU. VLM fine-tuning is very memory intensive. LoRA mode (llm_mode=lora) significantly reduces memory. Use multi-GPU with DDP for reasonable training times. Full fine-tune (llm_mode=ft) requires more GPUs.

## Error Patterns

**CUDA out of memory**: Switch to llm_mode=lora, reduce batch_size, or reduce model_max_length / max_tiles.

**Missing dataset YAML**: Ensure train.dataset.dataset_yaml_path points to a valid YAML file.

**Video frame loading**: Ensure num_video_frames and video_max_tiles are compatible with available GPU memory.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `vila.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `model_base` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| evaluate | `model_path` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| inference | `model_base` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| inference | `model_path` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| train | `model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
