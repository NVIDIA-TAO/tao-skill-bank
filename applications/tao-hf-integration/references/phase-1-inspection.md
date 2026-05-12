<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

Full Phase 1 walkthrough for the `tao-hf-integration` skill — credential gathering, branch creation, isolated venv setup, model/dataset inspection, and the Phase 1 gate. See `hf-inspection.md` for a generic HF-inspection cheat sheet.

## Phase 1 — Information Gathering & Validation

### 1.1 Gather credentials, targets, and locate repos
Ask the user for:
- **HuggingFace Model ID** — e.g., `google/vit-base-patch16-224`
- **HuggingFace Access Token** (`HF_TOKEN`) — required for gated models
- **Model short-name** for TAO — a `snake_case` identifier used for directory names and class names (e.g., `vit_base_p16`)
- **Do you already have the TAO repos cloned locally?** Ask for the paths to `tao-core`, `tao-pytorch`, `tao-deploy`, and `tao-dataservices`. If the user provides paths, verify they exist and use them. Only clone repos that are missing.

If any repos need to be cloned, ask the user where they'd like them cloned to (default: current working directory), then clone only the missing ones:
```bash
# Only clone what's needed — skip repos the user already has
git clone <tao-core-url> /path/to/tao-core
git clone <tao-pytorch-url> /path/to/tao-pytorch
git clone <tao-deploy-url> /path/to/tao-deploy
git clone <tao-dataservices-url> /path/to/tao-dataservices
```

After cloning, each repo (tao-pytorch, tao-deploy, tao-dataservices) will have a `tao-core/` submodule inside it. This submodule points to the original commit and should NOT be used — always use our top-level `tao-core/` clone instead (see "Submodule Override Strategy" above).

### 1.2 Create a consistent working branch across all repos

Before any implementation work, create a new branch in **every** repo so changes are isolated and consistent:

1. Ask the user for:
   - **Branch name** — e.g., `feature/add-vit-base-p16`
   - **Base branch** — default is `main`. Ask if they want a different base.

2. Create the branch in all repos:
```bash
for repo in tao-core tao-pytorch tao-deploy tao-dataservices; do
  cd /path/to/$repo
  git checkout <base_branch>
  git pull origin <base_branch>
  git checkout -b <branch_name>
  cd -
done
```

**Important:** This branch is local only — it will NOT be pushed. It just keeps changes organized and makes it easy to diff against the base branch.

### 1.3 Set up isolated environment for HF inspection

All Phase 1 Python work runs in a temporary venv — do NOT install into the host Python:
```bash
python3 -m venv /tmp/tao-hf-inspect-venv
source /tmp/tao-hf-inspect-venv/bin/activate
pip install --quiet transformers huggingface_hub torch onnx timm
```

### 1.4 Validate that the model is a Computer Vision model
```python
from huggingface_hub import model_info
info = model_info("<MODEL_ID>", token="<HF_TOKEN>")
print(info.pipeline_tag)   # must be: image-classification, object-detection, image-segmentation, etc.
```
**Hard stop:** If `pipeline_tag` is an NLP, audio, or LLM task, halt and inform the user. TAO Toolkit currently supports Computer Vision models only.

### 1.5 Fetch the model architecture and checkpoint
```python
from transformers import AutoModel, AutoConfig
import torch

config = AutoConfig.from_pretrained("<MODEL_ID>", token="<HF_TOKEN>")
model  = AutoModel.from_pretrained("<MODEL_ID>", token="<HF_TOKEN>")
state_dict = model.state_dict()
```
- Print `config` to extract: `model_type`, `image_size`, `hidden_size`, `num_labels`, `num_hidden_layers`, `patch_size`
- Print the top-level `state_dict` keys and shapes to understand HF naming conventions
- Assess whether the HF task head is separable from the backbone
- Draft a key-name remapping plan for the HF-to-TAO `state_dict` conversion

### 1.6 Verify ONNX exportability
```python
# Use the model's native image_size (extracted in 1.5), not a hardcoded value
img_size = getattr(config, "image_size", 224)
if isinstance(img_size, int):
    img_size = (img_size, img_size)
dummy = torch.randn(1, 3, *img_size)
model.eval()
torch.onnx.export(model, dummy, "/tmp/tao_hf_test.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17)
```
If this fails, identify the problematic ops and apply workarounds **before** starting TAO integration:
- **Unsupported op** → Replace with ONNX-compatible equivalent (e.g., replace `torch.einsum` with explicit `matmul`/`permute`, replace custom CUDA kernels with pure PyTorch ops)
- **Dynamic control flow** (if/else on tensor values) → Rewrite as static ops or use `torch.where()`
- **Unsupported attention variant** → Rewrite using standard `nn.MultiheadAttention` or explicit Q/K/V matmuls
- **Try higher opset** → `opset_version=17` or `18` supports more ops than older versions
- **TensorRT compatibility** → After ONNX export succeeds, test with `trtexec` inside the prepared tao-deploy container (the host does not have TensorRT):
  ```bash
  docker run --rm --gpus all -v /tmp:/tmp tao-deploy-base:latest trtexec --onnx=/tmp/tao_hf_test.onnx --buildOnly
  ```
  If TRT fails on specific layers, those ops will need to be rewritten in the TAO implementation — record them now
- **If export fundamentally cannot work** (e.g., architecture uses dynamic shapes that vary per-input), inform the user — the model may not be suitable for TensorRT deployment

### 1.7 Clean up Phase 1 environment

After all inspection is complete and findings are recorded:
```bash
deactivate

# Remove the venv
rm -rf /tmp/tao-hf-inspect-venv

# Remove temp ONNX file
rm -f /tmp/tao_hf_test.onnx

# Optionally remove the cached HF model (can be multi-GB)
# Only do this if you've saved the state_dict keys and config — you won't need the raw HF weights again
rm -rf ~/.cache/huggingface/hub/models--<org>--<model_name>
```

### Phase 1 Gate — Confirm before proceeding:
- [ ] All 4 TAO repos located or cloned
- [ ] Consistent working branch created across all repos
- [ ] `pipeline_tag` is a supported CV task
- [ ] `model_type`, `image_size`, `hidden_size`, `num_labels` extracted
- [ ] Top-level `state_dict` keys documented, remapping plan drafted
- [ ] ONNX export sanity check passed (or failure mode understood)
- [ ] User confirmed the model short-name and task type

**Present findings to the user and get confirmation before proceeding to implementation.**

---

