# Ordered preflight

Preflight is read-only except for the bank-wide small-Python-helper exception.
Do not create the run directory, specs, state, pull/login, download, or launch
until the review is confirmed.

## 1. Resolve model, backend, and platform

Resolve the model ID with `resolve_tao_model.py --model MODEL --action ACTION
--backend cosmos-framework --workload deft-aoi`. Resolve Train, Evaluate, and Inference through
`cosmos_workflow.py --backend cosmos-framework --action ACTION --workload
deft-aoi`. Record the backend rationale.

The fixed validation profile selects Docker. Otherwise, if more than one
supported installed platform remains, ask once among those peers. Read the
selected platform SKILL and run its Preflight.

For Docker validate without mutation:

```bash
command -v docker
docker version
docker info
docker run is a launch and is deferred until confirmation
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Require one NVIDIA H200 and record its exact reported name and memory. Verify
host UID/GID, free disk, and that absolute workspace paths are visible to the
Docker daemon.

## 2. Resolve network mode and Python

Honor `AIR_GAPPED=1`; otherwise use the explicit operator choice or
network-enabled default. Run `scripts/deft_python.sh` and verify imports for
`yaml` and `pyarrow`. If a missing Python helper is small and pip-installable,
install it in the selected environment, report it, and rerun the import. A
missing system CLI is a hard stop.

Never read credential values. Check presence only. Source only a user-approved
bare `KEY=value` file in the same shell call that consumes it.

## 3. Inspect workspace and local model

Resolve `${WORKSPACE}` and require:

- `annotations/proxy_kpi.json`;
- `annotations/benchmark_kpi.json`;
- `annotations/mining_pool.json`;
- referenced image files;
- one selected model directory beneath `models/` with `config.json` and
  safetensor weights.

The model is already in the required HF VLM format. Record its absolute path
and use it directly. Do not plan a preparation command.

## 4. Validate annotation and split contracts

Run `validate_sharegpt.py` on all three splits, with `--require-files` and
`--require-id` on Proxy/Benchmark. Require one image and exact OK/NG labels.
Run `validate_split_contract.py` and record pairwise overlap counts. Hash the
Benchmark file and show the frozen hash.

The media root is the workspace root when stored paths begin with `images/`.

## 5. Resolve immutable images

Resolve these keys from `${TAO_SKILL_BANK_PATH}/versions.yaml`:

```text
images.tao_toolkit.cosmos_framework
images.tao_toolkit.data_services
```

Inspect local Docker metadata for repository digests. If an image is absent,
mark `WILL_PULL_AFTER_APPROVAL`; do not pull during preflight. One Framework
image serves Train, Evaluate, and Inference.

## 6. Construct nested specs in memory

Plan the full Train TOML from `cosmos_framework_sft_full.toml` and per-role
Evaluate TOMLs from `render_cfw_evaluate.py`. For the validation profile record
5 iterations, 10 epochs, one H200, top-K 15 per label, `filter_by_label`, and
accuracy target 0.99. Show:

- baseline local HF model path;
- iteration DCP output convention;
- DCP Evaluate fields `config_file`, writable `export_dir`, and original HF
  `vit_checkpoint_path`;
- next-Train `checkpoint.load_path` handoff;
- H200 decoder and one-frame evaluation settings;
- exact native action entrypoints.

Only after approval write the planned specs beneath workspace/specs.

## 7. Launch review

Show one table with effective value, value source, and evidence for:

- model ID and absolute local path;
- backend and immutable Framework image digest;
- Docker, one H200, UID:GID, and mounts;
- Proxy/Benchmark/Mining paths and record counts;
- frozen Benchmark SHA-256;
- exact OK/NG contract;
- five iterations, ten epochs, top-K 15 per OK/NG query, cosine floor;
- accuracy and unknown-response constraints;
- results directory and job-record plan;
- Train/Evaluate/Inference native commands and DCP handoff;
- network mode and credential presence states.

Then invoke the shared `tao-launch-workflow` review and wait for explicit
confirmation. After approval, perform any declared pull, write specs, initialize
state, and begin the four-verb workflow.
