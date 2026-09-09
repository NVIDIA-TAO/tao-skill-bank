# Container runtime

Read this before submitting AnomalyGenNext fine-tuning.

The action uses the public image declared by `skill_info.yaml`:

```text
nvcr.io/nvidia/paidf-anomalygen:1.1.0
```

Use the source tree and Python environment baked into that image at
`/workspace/paidf-anomalygen`. Do not overlay a host checkout or virtualenv.
Mount or stage the dataset, canonical recipe, validation JSONL, Cosmos3-Nano
checkpoint, VAE, optional Hugging Face cache, and durable results directory.

The image does not contain the DINOv2 checkpoint used by the validation
callback. Expose the selected directory read-only at:

```text
/workspace/paidf-anomalygen/checkpoints/facebook/dinov2-large
```

It must contain `config.json` and either `model.safetensors` or
`pytorch_model.bin`. For offline execution, preflight the cache supplied with
`--hf-cache` for the Qwen tokenizer required by the pinned image.

The selected platform owns image import, execution storage, cache placement,
GPU allocation, and result publication. Keep credentials out of recipes,
commands, logs, and job records. Preserve only the canonical recipe, selected
adapter, validation metrics, status, training curves, and
`training_handoff.json` declared by the action.

A 1000-step run is the minimum quality smoke because it includes iteration-zero
and later validation. Shorter runs demonstrate wiring only.
