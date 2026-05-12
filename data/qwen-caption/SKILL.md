---
name: qwen-caption
description: VLM captioning via Qwen endpoint for generating text descriptions of images/videos. Use when generating captions
  for synthetic video generation or describing visual content.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash
tags:
- captioning
- vlm
- qwen
- text-generation
---

# Qwen Caption

Generates text captions for videos using the Qwen3-VL-235B VLM (Vision Language Model) endpoint. Used in the DEFT pipeline to describe gap-identified videos before synthetic video generation with Cosmos Predict 2.5.

## Purpose

Before generating synthetic videos, the pipeline needs text descriptions of what kind of videos to create. The Qwen captioner takes gap-identified videos (videos where the model performs poorly) and generates detailed text captions describing their content. These captions then feed into Cosmos Predict 2.5 as generation prompts.

## Architecture

The actual VLM inference runs on an external Qwen3-VL-235B endpoint. This container handles:
1. Loading video files from the weak video list
2. Constructing prompts from the template
3. Making API calls to the VLM endpoint
4. Collecting and formatting caption responses into JSONL output

The endpoint URL must be provided by the user in the experiment configuration.

## Inputs

- **weak_video_list**: Parquet file with a `video_id` column containing paths to weak/gap videos that need captioning.
- **endpoint**: URL of the Qwen VLM inference endpoint (user-provided).
- **model**: Model name for the endpoint (default: `Qwen/Qwen3-VL-235B-A22B-Instruct`).
- **prompt_file**: Path to the captioning prompt template file.

## Outputs

- **output_json**: JSONL file with `video_id` and `caption` fields. Each line contains the generated caption for one video.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| model | Qwen/Qwen3-VL-235B-A22B-Instruct | VLM model identifier for the endpoint |
| endpoint | (required, user-provided) | URL of the Qwen VLM inference service |
| prompt_file | (required, user-provided) | Path to the prompt template for captioning |

## Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| Connection refused / timeout | VLM endpoint is down or unreachable | Verify endpoint URL is correct and the service is running |
| 401/403 from endpoint | Authentication failure on VLM service | Check endpoint authentication configuration |
| Empty captions | Prompt template issue or video loading failure | Verify prompt_file path and video file accessibility |
