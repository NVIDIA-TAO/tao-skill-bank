---
name: tao-run-platform
description: TAO Execution SDK for submitting and monitoring GPU training jobs on supported platforms (Lepton, Brev, SLURM,
  local Docker, Kubernetes). Use when the user wants to run TAO jobs through the SDK, get job tracking, S3 I/O wrapping,
  multi-node distributed training, or platform-specific features that docker-run can't provide. Trigger phrases include
  "use the TAO SDK", "call tao_sdk", "AutoMLRunner", "ActionWorkflow", "Job handles", "S3 I/O wrapping", "TAO platform run".
license: Apache-2.0
compatibility: Requires Python 3.10+ and the nvidia-tao-sdk package (pip install nvidia-tao-sdk[all]).
metadata:
  author: NVIDIA Corporation
  version: '0.2'
allowed-tools: Read Bash
tags:
- platform
- tao
- sdk
---

# TAO Execution SDK

The SDK is the **optional** Python layer for users who need job handles, S3 I/O wrapping, or platform-specific features (Lepton multi-node, SLURM/Lustre queues, Kubernetes Jobs, local Docker debugging, Brev instance reuse). Most TAO skills run with just `docker run` and don't need it. Reach for the SDK when:

- You want a `Job` handle to poll status and stream logs over time.
- The platform is API-only (Lepton has no docker-run equivalent).
- You need S3-aware input download / output upload baked into the entrypoint.
- You're chaining multiple jobs and want persisted state.

## Preflight

Install `nvidia-tao-sdk[all]` before using this platform — the `[all]` extra pulls in every platform-specific dependency (Lepton, Brev, S3 utilities, etc.):

```bash
python -c "import tao_sdk" 2>/dev/null || {
  echo "MISSING: nvidia-tao-sdk not installed. Run:"
  echo "  pip install nvidia-tao-sdk[all]"
  exit 1
}
```

The package index is environment-specific — the runner/container is expected to have a working `pip` configuration (e.g. `~/.pip/pip.conf`, `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, or proxy). If the install fails for index/network reasons, that's a runner setup issue; this skill stays agnostic to the registry.

If missing, the agent prompts the user to authorize the install via Bash, then re-runs the preflight. Never auto-install silently.

## Setup

Credentials come from **environment variables** — sourced from `~/.config/tao/.env` (auto-loaded by the skill bank's SessionStart hook).

```python
from tao_sdk.platforms.lepton import LeptonSDK   # DGX Cloud
from tao_sdk.platforms.brev   import BrevSDK     # Brev GPU instances

sdk = LeptonSDK()    # reads LEPTON_WORKSPACE_ID, LEPTON_AUTH_TOKEN
# or
sdk = BrevSDK()      # reads BREV_API_TOKEN (optional — falls back to brev login)
```

Both SDKs validate credentials lazily on first use and raise `CredentialError` with a clear message if a required env var is missing. Required env vars:

| Platform | Required | Optional |
|---|---|---|
| Lepton | `LEPTON_WORKSPACE_ID`, `LEPTON_AUTH_TOKEN` | — |
| Brev | — (manual `brev login` works) | `BREV_API_TOKEN` |
| S3 I/O (any platform) | `S3_BUCKET_NAME`, `ACCESS_KEY`, `SECRET_KEY` | `S3_ENDPOINT_URL`, `CLOUD_REGION` |
| Container env | `NGC_KEY` | `HF_TOKEN` |

The agent never reads credential values — it only checks presence with `[ -n "$VAR_NAME" ]`.

## Workflow Launch Intake

For any TAO workflow or action launch, first confirm the user goal. Then ask
for platform and monitoring preferences before credentials or launch details.
Generate the supported platform choices from the packaged helper, not by
scanning platform docs or folders:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} --format text
```

Ask:

1. Which supported platform should run this workflow?
2. Should long-running monitoring stay enabled? Default: enabled. This means
   the agent remains attached and posts status until terminal state, including
   long `PENDING` queue waits.
3. How many minutes between status updates? Default: 5 minutes.

After the model/action are known, resolve the default container image from the
packaged metadata and ask the user to confirm it or provide `image=<override>`
before creating runner files:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/resolve_tao_image.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --model <network_arch> --action <action> --format text
```

For train-capable model workflows, inspect model-level AutoML metadata before
creating a plain training job:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_models.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --scope automl --format json
```

If the selected model has `automl_enabled: true` and a valid train schema,
route training through `applications/tao-run-automl` by default. A workflow should
only bypass AutoML when its run settings include `automl_policy: off`, the user
explicitly asks for a plain run, or the model metadata says AutoML is enabled
but the train schema is not packaged yet.

After the platform is selected, get the credential filter:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --platform <platform> --format text
```

Ask only for credentials returned for the selected platform. For example, SLURM
needs `SLURM_USER` and `SLURM_HOSTNAME`; it does not need Lepton credentials.
Kubernetes and local Docker do not need Lepton or SLURM credentials. Ask storage
credentials such as S3 keys only when the selected platform and the data/result
URIs require them.

## Core API

All platform SDKs implement the same core shape:

```python
sdk.create_job(image, command, gpu_count=1, env_vars=None, inputs=None, outputs=None, **kwargs) -> Job
sdk.get_job_status(job_id) -> JobStatus
sdk.get_job_logs(job_id, tail=None) -> str
sdk.cancel_job(job_id) -> bool
sdk.get_failure_analysis(job_id) -> dict | None
sdk.get_job_results_dir(job_id) -> str
sdk.check_path(remote_path) -> bool
sdk.list_path(remote_path) -> list[str]
```

Lepton-only:
- `sdk.get_job_replicas(job_id)` — replica-level diagnostics for stuck-pending jobs.

Brev-only:
- `sdk.delete_instance(instance_id)` — clean up an ephemeral instance.
- `sdk.list_instances()` — list active instances.

## Submitting a Job

The agent always **constructs the container command via `build_entrypoint`** before calling `create_job`. The agent reads the action's schema from `skill_info.yaml` (`command`, `config_format`, `inputs`, `outputs`, `upload_excludes`) and passes those fields as kwargs. `build_entrypoint` then bakes:

1. The in-container `script_runner` runtime (inlined as a base64 heredoc — no need for `tao_sdk` to be installed in the container).
2. The CLI invocation that, at runtime in the container, will: download declared inputs (S3 / HF-Hub / NGC), write the spec file at `{config_path}` with remote URIs rewritten to local paths, run the user command, and upload outputs.

Output destinations are resolved at runtime from env vars the SDK injects (see "Where outputs go" below). The platform SDK's `create_job` runs the resulting command **as-is** — no inputs/outputs kwargs, no implicit wrapping. The data flow is visible in the agent's code.

### Where outputs go (resolved at runtime — agents don't manage it)

The SDK injects `TAO_JOB_ID` (matches `Job.id`) and, when a persistent mount is attached, `TAO_RESULTS_ROOT` into the container env. Inside the container, `script_runner` resolves output destinations:

| Container env | Result |
|---|---|
| `TAO_RESULTS_ROOT` set (Lustre / PVC / bind / NFS) | Outputs at `{TAO_RESULTS_ROOT}/<job_id>/<key>/`; no upload |
| `S3_BUCKET_NAME` set (cloud, no mount) | Outputs at `s3://{bucket}/results/<job_id>/<key>/`; uploaded at end of run |
| Neither | Outputs at `/results/<job_id>/<key>/` (container-ephemeral) with a loud end-of-run warning |

Per-platform policy:

| SDK | What gets injected |
|---|---|
| `SlurmSDK` | `TAO_RESULTS_ROOT={SLURM_BASE_RESULTS_DIR}/results` (always — Lustre, never S3, avoids GPU-idle scheduler kill) |
| `LeptonSDK` | `TAO_RESULTS_ROOT={mount}/results` if a workspace volume is attached; otherwise S3 fallback |
| `KubernetesSDK` / `DockerSDK` / `BrevSDK` | `TAO_RESULTS_ROOT=/results` if a mount targets `/results`; otherwise S3 fallback |

Agents who want a custom destination can put an `s3://...` URI or absolute path directly at the output spec key — explicit values override the auto-fill. Otherwise, model-natural defaults like cosmos-rl's `output_dir: "output"` or DINO's empty `results_dir` are auto-rewritten by `script_runner`.

### The spec is nested dicts, NOT flat dotted keys

This is the most common mistake when constructing a spec. The dotted notation that appears in `skill_info.yaml`'s `inputs:` / `outputs:` blocks (e.g. `section.subsection.key`) is a **path into** a nested spec — `script_runner` looks values up at that path. It's not the spec's own shape. The spec mirrors whatever shape the model's container expects (typically a nested TOML/YAML).

```python
# ✓ CORRECT — nested dicts
specs = {
    "section": {
        "subsection": {"key": "value"},
    },
}

# ✗ WRONG — flat top-level key with dots. TOML/YAML emits this as a
# quoted bare-string key, the model sees an empty `section` table, and
# any input declared at "section.subsection.key" silently fails to
# download because _get_nested(specs, "section.subsection.key") → None.
specs = {
    "section.subsection.key": "value",
}
```

The two shapes look superficially similar but mean different things. When in doubt, open the model's `references/` directory (e.g. a default-spec TOML or YAML) — that's the literal nested structure the spec dict needs to mirror. The `inputs:` / `outputs:` declarations in `skill_info.yaml` are *paths into* the nested spec, not key names.

### Constructing the spec / args

The skill's action declares its config mechanism in `skill_info.yaml`'s `actions.<action>.mode` field (defaulting to `config` when absent). The agent's construction strategy follows from that:

| `mode` | How to construct |
|---|---|
| `args` | Copy the `actions.<a>.args` block from `skill_info.yaml` as your template. Substitute placeholders (`{storage_root}`, `{split_id}`, `{num_gpus}`, etc.) with the user's runtime values. Pass to `build_entrypoint(args=...)`. |
| `config` + `references/spec_template_<a>.yaml` exists | Load the template via `yaml.safe_load(...)` as the base spec; apply user overrides on top. Pass to `build_entrypoint(specs=...)`. |
| `config`, no template | Follow the model's `SKILL.md` — typically a "Critical Overrides" section lists which keys must be set. Construct the spec accordingly. Pass to `build_entrypoint(specs=...)`. |
| `passthrough` | Bare command + path-keyed `inputs={container_path: uri}` / `outputs=[paths]`. Pass to `build_entrypoint(inputs=..., outputs=...)`. |

**Recommended decision order:**

1. Read `action_cfg = skill_info["actions"][action]`. Check `action_cfg.get("mode", "config")`.
2. For `config` mode: check `references/spec_template_<action>.yaml`. If it exists, **load it as your base** — don't rebuild from scratch.
3. Apply user overrides on top (plus any "Critical Overrides" rows from the model's `SKILL.md`).
4. For `args` mode: copy `action_cfg["args"]`, fill placeholders, hand to `build_entrypoint(args=...)`.

```python
import yaml
from pathlib import Path

skill_dir = Path(bank) / "models/<model>"
skill_info = yaml.safe_load((skill_dir / "references/skill_info.yaml").read_text())
action_cfg = skill_info["actions"][action]
mode = action_cfg.get("mode", "config")

if mode == "args":
    args = dict(action_cfg["args"])
    args["weak-video-list"] = args["weak-video-list"].format(storage_root=user_storage)
    # ... substitute remaining placeholders
    ep = build_entrypoint(command=action_cfg["command"], args=args, ...)

elif mode == "config":
    template = skill_dir / f"references/spec_template_{action}.yaml"
    specs = yaml.safe_load(template.read_text()) if template.exists() else {}
    # apply user overrides on top
    specs.setdefault("policy", {})["model_name_or_path"] = user_model
    # ... etc
    ep = build_entrypoint(command=action_cfg["command"], specs=specs, ...)
```

### Spec-driven jobs

The skill's action declares a config file (`config_format`, `command: ... {config_path} ...`). Covers TAO models (DINO, BEVFusion, classification-pyt, …) and cosmos-rl — anything whose container reads a spec file and writes outputs to declared spec keys. Use whichever platform SDK fits the target backend; the `build_entrypoint` call is identical across platforms.

```python
import yaml
from tao_sdk.script_runner import build_entrypoint
from tao_sdk.versions import resolve_container_image
# pick the SDK matching your target platform:
from tao_sdk.platforms.lepton     import LeptonSDK     # or
from tao_sdk.platforms.slurm      import SlurmSDK      # or
from tao_sdk.platforms.kubernetes import KubernetesSDK # or
from tao_sdk.platforms.docker     import DockerSDK     # or
from tao_sdk.platforms.brev       import BrevSDK

skill_info = yaml.safe_load(open(f"{bank}/models/tao-train-dino/references/skill_info.yaml"))
action_cfg = skill_info["actions"]["train"]

specs = {
    "dataset": {
        "train_data_sources": [{
            "image_dir":  "s3://my-bucket/coco/train/images",
            "json_file":  "s3://my-bucket/coco/train/annotations.json",
        }],
        "val_data_sources": [{
            "image_dir":  "s3://my-bucket/coco/val/images",
            "json_file":  "s3://my-bucket/coco/val/annotations.json",
        }],
        "num_classes": 80,
    },
    "train": {"num_epochs": 10, "num_gpus": 8},
    # No results_dir — script_runner auto-fills at runtime.
}

ep = build_entrypoint(
    command=action_cfg["command"],                       # e.g. "dino train -e {config_path}"
    specs=specs,                                          # → infers config mode
    inputs=action_cfg["inputs"],                          # spec-keyed dict from skill_info.yaml
    outputs=action_cfg["outputs"],
    config_format=action_cfg["config_format"],            # "yaml" / "toml" / "json"
    upload_excludes=action_cfg.get("upload_excludes", []),
)

sdk = ...   # one of the SDKs above
job = sdk.create_job(
    image=resolve_container_image(skill_info["container_image"]),
    command=ep["command"],
    gpu_count=8,
    # Platform-specific kwargs go here — see each platform's SKILL.md:
    #   Lepton:     dedicated_node_group, resource_shape, num_nodes
    #   SLURM:      partition, account, num_nodes
    #   Kubernetes: namespace, node_selector, tolerations, num_nodes
    #   Docker:     mounts
    #   Brev:       instance_id, gpu_type, cloud_cred_id, workspace_group_id
)
print(f"Job submitted: {job.id}    Results: {job.results_dir}")
```

### Path-keyed jobs (no config file)

The skill's action does not write a spec file — inputs are passed as `{container_path: uri}` and outputs as a list of container paths. Covers HF inference scripts, custom commands, anything that takes its inputs via direct paths rather than a config file.

```python
ep = build_entrypoint(
    command="python infer.py --model /models/cosmos --input /data/in --output /results",
    inputs={                                              # path-keyed → infers passthrough mode
        "/models/cosmos": "hf_model://nvidia/Cosmos-Reason2-8B",   # HF Hub
        "/data/in":       "s3://bucket/test/in",                    # S3
        # also supported: "ngc://..."
    },
    outputs=["/results/"],
)
sdk.create_job(image=img, command=ep["command"], gpu_count=1)
```

In passthrough mode the runtime dispatches each input URI by scheme — `s3://`, `hf_model://`, `ngc://` — to the right downloader. No spec rewriting, no `{config_path}`. After the command, listed output paths are uploaded per the same destination resolution rules (S3 if `S3_BUCKET_NAME`, else mount, else container-ephemeral with warning).

### Mode inference (you don't pass `mode`)

`build_entrypoint` infers the mode from what the agent passes:

| What the agent passes | Inferred mode |
|---|---|
| `specs=...` (with optional spec-keyed `inputs` / `outputs`) | `config` — write spec file, rewrite URIs, run command |
| `args=...` (with optional spec-keyed `inputs` / `outputs`) | `args` — substitute CLI args into the command template |
| `inputs=...` and/or `outputs=...` only (path-keyed) | `passthrough` — download to listed paths, run, upload |
| nothing extra (just `command`) | `passthrough` with no I/O — bare command |

One helper, one signature.

## Resolving container images

Skills declare images either by key (`tao_toolkit.pyt`) or as an absolute URI (`nvcr.io/...`). Use `resolve_container_image()` to handle both:

```python
from tao_sdk.versions import resolve_container_image
image = resolve_container_image(skill_info["container_image"])
```

Behind the scenes it walks `versions.yaml` for keys; absolute URIs are returned as-is.

## Monitoring

```python
status = sdk.get_job_status(job.id)
print(status.status)   # Pending, Running, Complete, Error, Canceled
print(status.message)  # platform-specific detail

logs = sdk.get_job_logs(job.id, tail=200)
print(logs)
```

For stuck-Pending Lepton jobs, replica diagnostics reveal the cause (image pull, scheduling, mount errors):

```python
for r in sdk.get_job_replicas(job.id):
    issue = r["status"].get("readiness_issue")
    if issue:
        print(issue["reason"], issue["message"])
        # e.g. "InProgress" / "Pulling image"  (normal for big images)
        #      "Failed"     / "ImagePullBackOff" (NGC_KEY problem)
        #      "ConfigError" / "Mount point not found" (bad node)
```

On failure, `get_failure_analysis()` classifies the root cause:

```python
analysis = sdk.get_failure_analysis(job.id)
if analysis:
    print(analysis["err_class"])   # ERR_PROGRAM, ERR_INFRA, etc.
    print(analysis["suggestion"])  # human-readable fix
    for event in analysis.get("job_failure_by_node_event", []):
        print(event["node_event_name"], event["message"])  # OOM, GPU error, etc.
```

## Polling pattern

For interactive runs where the user wants to watch:

```python
import time
status_interval_minutes = status_interval_minutes or 5
while True:
    status = sdk.get_job_status(job.id)
    if status.status in ("Complete", "Error", "Canceled"):
        break
    print(f"  {status.status}")
    time.sleep(status_interval_minutes * 60)

if status.status == "Error":
    print(sdk.get_job_logs(job.id, tail=100))
    print(sdk.get_failure_analysis(job.id))
```

With long-running monitoring enabled, do not stop after 30 minutes or after a
few unchanged polls. Keep emitting updates every `status_interval_minutes`
until the job finishes, fails, is canceled, or the user asks to detach/stop.
If the chat/runtime cannot remain open that long, say so explicitly and provide
the durable workflow/log path for manual status refresh.

Do not use a final response for non-terminal monitored jobs. Finalizing the
turn detaches the chat watcher. Keep non-terminal status messages in progress
updates and continue polling; only finalize at terminal state, explicit user
detach/stop, or a real runtime limit that prevents further polling.

For background runs, persist `job.id` and the `state_file` path, then re-attach later by constructing the same SDK and calling `get_job_status(job_id)` — job state is read from the on-disk store.

## Orchestration patterns

Multi-step workflows, parallel sweeps, and run-folder durability via
`ActionWorkflow` live in
[`references/orchestration-patterns.md`](references/orchestration-patterns.md).
Read it before chaining `create_job` calls, sweeping a parameter, or
persisting run state across context breaks.

## Dataset utilities

When the skill's documented filenames don't match the user's layout, list the dataset to confirm:

```python
assert sdk.check_path("s3://my-bucket/coco/")
files = sdk.list_path("s3://my-bucket/coco/train/")
# Use the actual paths to set spec fields.
```

For S3 paths, strip trailing slashes when concatenating to avoid `//`:

```python
base = dataset_uri.rstrip("/")
specs["dataset"]["train_csv"] = f"{base}/train.csv"   # nested — see "spec is nested dicts"
```

## Platform-specific notes

### Lepton (`from tao_sdk.platforms.lepton import LeptonSDK`)
- Jobs run as containers on DGX Cloud.
- NFS/Lustre mounts auto-detected from the node group; the SDK builds the appropriate `Mount` objects.
- `gpu_count` resolves to a Lepton resource shape; or pass `dedicated_node_group="<name>"` for guaranteed allocation.
- `num_nodes=N` (N>1) enables distributed training.

### Brev (`from tao_sdk.platforms.brev import BrevSDK`)
- Jobs run on GPU instances via `brev exec`.
- No shared storage — S3 only.
- Pass `instance_id="<id>"` in kwargs to reuse an existing instance (skip 2–5 min boot).
- Pass `gpu_type="L40S"` to control instance class for ephemeral instances.
- Pass `cloud_cred_id="<id>"` and `workspace_group_id="<id>"` on multi-credential
  or multi-workspace accounts. Without them, `brev create` rejects with a
  placement error. Discover via `brev orgs --json` (cloud cred) and
  `brev ls --json` (workspace group). See `platform/tao-run-on-brev/SKILL.md` →
  *Creating an instance — placement info* for the full lookup recipe.
- The handler waits for both `status=RUNNING` and `brev exec ... -- true`
  before returning, so a `create_job` → `get_job_logs` sequence won't race
  sshd bring-up. The first remote exec uses a 600s timeout to absorb the
  container-pull window; reused instances use 30s.
- Use `sdk.delete_instance(instance_id)` when done with an ephemeral one.

### SLURM
- Jobs submit over SSH to a login node with `sbatch` and run containers through
  Pyxis/Enroot `srun --container-image`.
- Use the platform helper output to ask only for SLURM credentials and storage
  settings. Do not ask for Lepton, Brev, or Kubernetes credentials.
- Dataset paths must be visible from the cluster job, usually absolute Lustre or
  shared filesystem paths; do not pass agent-host local paths to SLURM jobs.
- Use the packaged SLURM runtime defaults unless the user gives a validated
  override. For the common `polar,polar3,polar4,grizzly` queues, prefer the
  four-hour default rather than generating 12-hour wrappers.

### Kubernetes
- Jobs run as Kubernetes Jobs on a configured GPU cluster.
- Auth uses kubeconfig (`KUBECONFIG` or `~/.kube/config`) or an in-cluster
  service account.
- Requires NVIDIA GPU Operator or equivalent `nvidia.com/gpu` device plugin.
- Do not ask for Lepton, Brev, or SLURM credentials for Kubernetes runs.
- A local path on the agent host is not proof that the path is mounted inside
  the job pod.

### Local Docker
- Jobs run on the local Docker daemon host.
- Multi-node is not supported; multi-GPU on the local host is supported.
- Verify local dataset paths, Docker daemon access, and NVIDIA runtime before
  generating or launching runner artifacts.

## Error patterns

SDK error → root cause → fix mappings are in
[`references/error-patterns.md`](references/error-patterns.md). Read when
you hit a `CredentialError`, image-pull failure, stuck-Pending job, or
similar — the entries map exception text to the underlying cause.

## What the SDK does NOT do

Scope guardrails (no skill-reading, no HPO, no spec opinions, no
auto-platform-selection, no workflow orchestration) live in
[`references/scope.md`](references/scope.md).
