---
name: cosmos-predict-2-5
description: Cosmos Predict 2.5 text-to-video generation for synthetic data augmentation. Use when generating synthetic training
  videos from text captions or augmenting video datasets.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash
tags:
- video
- generation
- text-to-video
- cosmos
- sdg
---

# Cosmos Predict 2.5

NVIDIA's text-to-video generation model. Cosmos Predict 2.5 generates synthetic training videos from text captions, serving as a core component of the DEFT (Data-Efficient Fine-Tuning) pipeline.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. For AutoML, `schemas/train.schema.json` and `references/spec_template_train.yaml` must exist and parse; otherwise AutoML is unsupported for this model in the plugin workflow. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Purpose

Given text captions describing desired video content, Cosmos Predict 2.5 generates realistic synthetic videos. These generated videos augment real training data to improve downstream model performance, particularly for tasks where collecting real-world video data is expensive or dangerous.

## Inputs

- **input_prompt_json**: JSONL file with `video_id` and `caption` fields. Each line describes one video to generate.
- **weak_data_list**: Parquet file with `video_id`, `question`, and `ground_truth` fields. Identifies the gap data that needs synthetic augmentation.

## Outputs

- **output_dir**: Directory containing generated video files and an output parquet mapping video IDs to generated file paths.

## Parallel Execution

Supports splitting the input JSONL across multiple GPU groups for parallel generation. Each split runs independently, and results are merged after all splits complete.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_gpus | 8 | Number of GPUs for generation |
| num_output_frames | 500 | Frames per generated video |
| guidance | 7 | Guidance scale — higher values produce videos more faithful to the caption but potentially less diverse |

## Guardrails

Cosmos Predict 2.5 includes safety guardrails that block generation of certain content. For DEFT collision detection pipelines, guardrails **must be disabled** via the `disable-guardrails` flag (configured in `config.json`).

**Why:** Prompts describing vehicle collisions trigger the guardrail, which blocks the prompt on rank 0 and skips the diffusion step. However, other GPU ranks have already entered the NCCL collective for the diffusion denoising. Rank 0 moves to the next sample while ranks 1-N are stuck waiting, causing an NCCL collective timeout and SIGABRT on all ranks. This crashes the entire job even though non-collision prompts would succeed.

## Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| NCCL timeout / SIGABRT | Guardrail blocks a prompt mid-batch, desynchronizing GPU ranks | Disable guardrails (`--disable-guardrails` flag) |
| CUDA OOM | Video resolution or batch too large for GPU memory | Reduce batch size or num_output_frames |
| HF auth failure | Missing or invalid HF_TOKEN | Ensure HF_TOKEN is set in your environment (e.g., in `~/.config/tao/.env`) with a valid HuggingFace token that has access to the model |
| Low quality / artifacts | Guidance scale too low or too high | Adjust guidance parameter (typical range: 5-10) |
| Slow generation | Insufficient GPU count | Increase num_gpus or use faster GPU hardware (H100 preferred) |

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

No TAO Core `spec_params` mapping is available for this model. If an action consumes a model produced by an upstream job, resolve that model from the parent job id instead of hardcoding a result path:

```python
checkpoint_uri = sdk.get_model_results_path(parent_job_id, network_arch="cosmos-predict-2-5")
```

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
