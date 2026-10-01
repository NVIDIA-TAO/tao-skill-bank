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
Cosmos3-Nano base checkpoint, optional real-image root, optional Hugging Face
cache, this skill directory, and a new durable output directory. Preserve
absolute paths or rewrite all related paths consistently inside the compute
frame. The platform owns writable temporary storage and caches.

The image defaults to the non-root `anomalygen` user with UID 10000. Every
input mount must be readable and traversable by the effective container user;
every output, temporary, and cache mount must be writable. Before starting
Python, use the selected image, mounts, and runtime identity to create and
remove a nested temporary file in each writable mount. Stop on failure and
report the affected host path with its ownership and mode. If the platform
maps the process to the host user instead, that identity must resolve to a
username and all home and cache locations must remain writable. For the local
Docker identity-mapping recipe, see
[Running in Docker](../../tao-generate-anomalies/references/docker.md).

For offline use, preflight the selected Hugging Face cache for the tokenizer
and guardrail assets resolved by the pinned image. Missing assets can fail
before sampling. Keep registry and Hugging Face credentials out of specs,
commands, logs, and job records.

Invoke the command declared in `skill_info.yaml`, for example inside the
container:

```bash
python /opt/tao-generate-od-defects/scripts/generate_od_defects.py \
  --inputs-dir /inputs/prepared \
  --base-checkpoint /models/Cosmos3-Nano \
  --output-dir /results/generation \
  --num-gpus 1
```

The exposed GPU count must match `--num-gpus`. The output directory must not
already exist. The base path must contain `checkpoint.json` and the `model/`
checkpoint directory. Read `execution-contract.md` for the completion and
accounting gates.
