# Cosmos Reason 3 Framework action contract

## Image and base model

Resolve `images.tao_toolkit.cosmos_framework` from the bank `versions.yaml`
and inspect its repository digest. The same immutable image runs Train,
Evaluate, and Inference.

The model path is a complete local HF-format VLM directory beneath the
workspace with `config.json`, tokenizer/processor files required by the model,
and safetensor weights. Pass this directory directly to all baseline actions
and iteration-1 Train. Do not prepare another weight tree.

## Train

Render `references/cosmos_framework_sft_full.toml` with
`scripts/render_cfw_sft.py`. The validation profile is single-GPU BF16 VLM
LoRA, one image per record, 10 epochs, and language-side projection targets.
The native command is:

```bash
cosmos-framework-train --sft-toml=/tao/config/train.toml
```

Iteration 1 leaves `checkpoint.load_path="???"`. Later iterations set it to
the preceding native DCP directory and retain
`checkpoint.keys_to_skip_loading=[]`. Submit through
`scripts/submit_cfw_train.py`; record the rendered TOML and selected DCP path.

## Evaluate

Render an exact nested TOML with `scripts/render_cfw_evaluate.py`. The native
command is:

```bash
cosmos-framework-evaluate --config /tao/config/evaluate.toml
```

The H200 profile has one GPU, BF16, batch size 1, one frame,
`torchcodec-cuda-on-demand`, temperature 0, and at most four output tokens.
Use the original local HF path directly for baseline. For an iteration DCP:

- `model.model_name`: DCP directory;
- `model.config_file`: saved Train TOML from that iteration;
- `model.export_dir`: action-local writable model directory;
- `model.vit_checkpoint_path`: original local HF base directory;
- `model.enable_lora=false`.

The public action handles the DCP locally when it starts. Mount the action
model parent read-write and the rest of the workspace read-only. Submit with
`scripts/submit_cfw_evaluate.py`.

## Inference

The native one-image command is:

```bash
cosmos-framework-inference \
  --model_path MODEL --torch_dtype bfloat16 --device_map auto --num_gpus 1 \
  --type image --media IMAGE --prompt 'Return exactly OK or NG.' \
  --max_new_tokens 4 --results_dir RESULTS --enable_lora false
```

For a DCP add `--config_file`, `--export_dir`, and
`--vit_checkpoint_path` with the same meanings as Evaluate. Use
`scripts/submit_cfw_inference.py`.

## Container invariants

Use the job-record id as the Docker name and label. Run detached, poll Docker
through `status`, read Docker output through `logs`, and stop through `cancel`.
Run as host UID:GID, use a writable stage working directory, keep user data off
`/workspace`, mount the workspace read-only, and expose only recorded results
and action-model directories read-write.
