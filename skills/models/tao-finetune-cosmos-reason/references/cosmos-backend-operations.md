# Cosmos backend operations and recovery

## Clean build and source equivalence

For every repository, record branch, commit, tree, and dirty state. Refuse a
reproducibility build when any packaged source is dirty. The Framework path
first builds its native Dockerfile, then builds the TAO action layer using that
exact base. The Cosmos-RL path builds `Dockerfile.cosmos_rl` with exact native
RL, TAO actions, DAFT, and TAO Core commits.

Inspect `/opt/tao/image-provenance.json` after build. Verify repository commits,
source-manifest checksum, dependency inputs, Python/package locations, and
non-root imports. Resolve and record image ID/digest. SLURM then converts that
exact digest to a newly named SQSH in the runtime-supplied cache directory and
records its SHA256. A source change invalidates both image and SQSH.

For Cosmos-RL, image verification must import both `deep_ep` and
`deep_ep_cpp`, inspect the compiled extension for the internode mask-buffer
symbols referenced by the Python bindings, and verify that vLLM uses the
linear-equivalent Qwen3-VL Conv3D path for every PyTorch version at or above
2.9. A successful `nvidia-smi` or `cosmos-rl --help` check does not cover these
ABI and dispatch contracts.

## Model preparation

The logical base model is always supplied. A local Qwen3-VL safetensors model
can be used directly after config, tensor-index, tokenizer, and processor
validation. A URI requires an immutable revision and is snapshotted to the
runtime checkpoint area. Cosmos3 Omni inputs use the native Framework
converter and an explicitly supplied, revision-pinned architecture model. The
conversion manifest proves the common source model and fingerprints the
prepared representation. Framework DCP evaluation uses the native exact-key
VLM exporter; PEFT adapters are reconstructed and merged before shared
evaluation.

For every Framework evaluate, inference, or inference-microservice request,
the skill runs `scripts/framework_checkpoint_action.py plan` and then
`prepare` before constructing the action command. The helper skips native HF
safetensors inputs, but a Framework DCP is exported with the Framework
repository's `cosmos_framework.scripts.export_vlm_dcp` entry point. The
default output name includes the DCP-metadata and saved-config fingerprint.
Reuse requires a matching export manifest, checkpoint record, DCP metadata
hash, config path/hash, base-model identity/revision, and complete indexed
weights. The returned `action_model_path` replaces the model field in the
evaluate/inference request. Export failure is an action failure, not a reason
to load an older export or checkpoint.

## SLURM preflight

Validate SSH configuration without reading key contents, scheduler reachability,
partition/account/QOS/reservation, shared paths, free space, Pyxis, Enroot,
SQSH readability, mount mapping, work directory, and non-root Python imports.
On a short allocation validate the allocated GPU count/type/memory, driver,
CUDA, PyTorch CUDA, architecture, NCCL initialization, decoder/library, and
the model/data paths through the container.

Jobs explicitly use Bash. Use one launcher task per node and preserve the
training child code. Record scheduler state/reason/exit independently. Requeue
is off unless separately validated. Scheduler `COMPLETED` never overrides a
nonzero child exit or missing/failed TAO terminal state.

Framework topology is shard degree equal to GPUs per node and replicate degree
equal to nodes. Cosmos-RL uses one controller on node zero and policy workers
with the declared shard/replica topology. Multi-node asynchronous checkpointing
is rejected. The environment records deterministic seed/hash settings, NCCL
diagnostics/error handling, CUDA allocator settings, driver capabilities, and
resource limits without credentials.

## Decoder and cache recovery

Framework uses its native CUDA TorchCodec path. Cosmos-RL uses repository-owned
PyNvVideoCodec with correct pitch/stride handling. Worker count zero requires
prefetch to be absent or null. Full Cosmos-RL video runs prewarm separate train
and validation caches before model allocation. Cache keys combine dataset,
model, and processor fingerprints; completeness manifests and every entry are
validated before training.

Do not recover a full video run by falling back silently to CPU decoding or by
reusing another run's cache. A decoder, cache, or media failure is a failed
smoke gate.

## Failure classes

- request/input: missing runtime model, revision, dataset, media, path, or
  scheduler value;
- source/provenance: dirty checkout, mismatched image source, old SQSH, or host
  source import;
- platform: SSH, scheduler, Pyxis/Enroot, mount, permission, GPU, CUDA, NCCL,
  decoder, or storage failure;
- model/data: incompatible checkpoint keys/config, missing media, duplicate or
  overlapping records, incompatible structural family, or missing task metadata;
- experiment parity: model, dataset, optimization, prompt/preprocessing, or
  evaluator mismatch;
- runtime: OOM, distributed timeout, decoder error, checkpoint failure,
  nonfinite metric, child nonzero, or missing terminal status.

A source/image defect is fixed only in its owning repository, with a test and a
new clean build/SQSH. Never patch an existing container or rely on a generated
script as the sole fix.
