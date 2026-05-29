---
name: tao-deploy-image-classification
description: >-
  Classification PyT deploy workflow for TensorRT engine generation, TensorRT evaluation, and TensorRT inference using TAO Deploy. Use
  when the user asks to deploy Classification PyT, build a Classification PyT TensorRT engine,
  run Classification PyT TRT inference, or evaluate a Classification PyT TRT engine.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit + NGC API key.
metadata:
  version: "0.1"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- image
- classification
- deployment
- tensorrt
---

# Classification PyT Deploy

Classification PyT deploy covers the TAO Deploy actions for an exported image classification model. Use the parent `classification-pyt` model skill for training, checkpoint evaluation, quantization, distillation, pruning, export, or non-TensorRT inference where those actions exist. Use this deploy sub-skill after export when the input artifact is an ONNX model and the desired output is a TensorRT engine or TensorRT-backed predictions.

Supported actions: `gen_trt_engine`, `evaluate`, `inference`.

## Quick Start

### Generate TensorRT Engine

```bash
docker run --gpus all --rm --shm-size=16g \
  -v /path/to/specs:/specs \
  -v /path/to/export:/models \
  -v /path/to/results:/results \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy \
  classification_pyt gen_trt_engine -e /specs/classification-pyt_deploy_gen_trt_engine.yaml
```

### Evaluate TensorRT Engine

```bash
docker run --gpus all --rm --shm-size=16g \
  -v /path/to/specs:/specs \
  -v /path/to/eval:/data \
  -v /path/to/results:/results \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy \
  classification_pyt evaluate -e /specs/classification-pyt_deploy_evaluate.yaml
```

### TensorRT Inference

```bash
docker run --gpus all --rm --shm-size=16g \
  -v /path/to/specs:/specs \
  -v /path/to/inference:/data \
  -v /path/to/results:/results \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy \
  classification_pyt inference -e /specs/classification-pyt_deploy_inference.yaml
```

Deploy action metadata is in `skill_info.yaml`. Deploy spec templates live in the parent references folder:

- `../references/spec_template_deploy_gen_trt_engine.yaml`
- `../references/spec_template_deploy_evaluate.yaml`
- `../references/spec_template_deploy_inference.yaml`

## Deploy Workflow

1. Train and export with the parent `classification-pyt` skill.
2. Keep the exported ONNX artifact and any sidecar files together in the mounted model directory.
3. Build the TensorRT engine with this sub-skill.
4. Run TensorRT `evaluate` or `inference` from the engine artifact produced by `gen_trt_engine`.

Direct TAO Launcher spelling is `tao deploy classification_pyt gen_trt_engine`, `tao deploy classification_pyt evaluate`, `tao deploy classification_pyt inference`.

## Required Inputs

| Action | Required artifact or data | Spec key |
|---|---|---|
| `gen_trt_engine` | Exported ONNX model | `gen_trt_engine.onnx_file` |
| `gen_trt_engine` | Output engine path | `gen_trt_engine.trt_engine` |
| `evaluate` | TensorRT engine | `evaluate.trt_engine` |
| `evaluate` | Image classification test folder | `dataset.test_dataset.images_dir` |
| `inference` | TensorRT engine | `inference.trt_engine` |
| `inference` | Image classification test folder | `dataset.test_dataset.images_dir` |

For direct Docker runs, mount input folders at the same paths used in the spec. For chained jobs, map exported ONNX artifacts into `gen_trt_engine.onnx_file` and map the engine artifact into `evaluate.trt_engine` or `inference.trt_engine` where those actions are available.

## Spec Overrides

Carry structural model and dataset settings forward from the train/export spec. The deploy defaults are templates, not a substitute for the model-specific values used to produce the ONNX file.

Recommended starting overrides:

```python
{
    'gen_trt_engine.tensorrt.data_type': 'fp16',
    'inference.batch_size': 1,
    'gen_trt_engine.tensorrt.min_batch_size': 1,
    'gen_trt_engine.tensorrt.opt_batch_size': 1,
    'gen_trt_engine.tensorrt.max_batch_size': 8,
}
```

Model-specific notes:

- Use `fp16` for the starter-kit TensorRT engine path unless INT8 calibration is explicitly requested.
- For TensorRT inference, set the runtime batch size to 1 unless the engine profile was built for the larger batch.

## Job Chain Mapping

| Action | Spec field | Parent or output |
|---|---|---|
| `gen_trt_engine` | `gen_trt_engine.onnx_file` | export job ONNX |
| `gen_trt_engine` | `gen_trt_engine.trt_engine` | new engine output path |
| `gen_trt_engine` INT8 | calibration image/cache fields | calibration dataset and new cache output |
| `evaluate` | `evaluate.trt_engine` | engine job output |
| `inference` | `inference.trt_engine` | engine job output |

## Outputs

| Action | Output |
|---|---|
| `gen_trt_engine` | TensorRT engine at `gen_trt_engine.trt_engine` |
| `evaluate` | Top-K classification metrics under `results_dir` |
| `inference` | Classification predictions under `results_dir` |

## Known Pitfalls

**Engine profile mismatch:** Runtime batch size for evaluate or inference must fit within the TensorRT min/opt/max profile used during `gen_trt_engine`.

**Template class or shape mismatch:** Copy class count, input resolution, backbone, and post-processing settings from train/export before running TAO Deploy.

**INT8 calibration missing:** INT8 builds need an extracted calibration image directory, a writable cache path, and enough images for `cal_batch_size * cal_batches`.

**Mounted paths do not exist:** TAO Deploy checks local paths inside the container. Make sure every path in the spec has a matching Docker mount or job artifact mapping.
