# AnomalyGenNext 1.1 container runtime

Read this before submitting generation.

Resolve the image through `references/skill_info.yaml`:

```text
nvcr.io/nvidia/paidf-anomalygen:1.1.0
```

The image contains AnomalyGenNext 1.1 and its runtime at
`/workspace/paidf-anomalygen`. Do not substitute the older 1.0 image and do
not overlay an external checkout or host virtualenv.

Expose the testcase or prepared-input result, task checkpoint and recipe,
Cosmos3-Nano base checkpoint, complete checkpoint root, this skill directory,
and a new durable output directory. Preserve absolute paths or rewrite all
related paths consistently inside the compute frame. The platform owns
writable temporary storage and caches.

Mount the complete checkpoint root read-only at
`/workspace/paidf-anomalygen/checkpoints`. The action rejects any other mount
location and validates `Qwen/Qwen3-VL-8B-Instruct`,
`Qwen/Qwen3Guard-Gen-0.6B`, `nvidia/Cosmos-Guardrail1`, and
`nvidia/Cosmos3-Edge` in the Hugging Face cache under `hf/`, plus a
Transformers-compatible `facebook/dinov2-large`. It then forces Hugging Face
and Transformers offline. Missing assets fail before sampling. Keep registry
credentials out of specs, commands, logs, and job records.

Invoke the command declared in `skill_info.yaml`, for example inside the
container:

```bash
python /opt/tao-generate-od-defects/scripts/generate_od_defects.py \
  --inputs-dir /inputs/prepared \
  --base-checkpoint /models/Cosmos3-Nano \
  --checkpoint-root /workspace/paidf-anomalygen/checkpoints \
  --output-dir /results/generation \
  --num-gpus 1
```

The exposed GPU count must match `--num-gpus`. The output directory must not
already exist. The base path must contain `checkpoint.json` and the `model/`
checkpoint directory; it is not the shared checkpoint root. Read
`execution-contract.md` for the completion and accounting gates.
