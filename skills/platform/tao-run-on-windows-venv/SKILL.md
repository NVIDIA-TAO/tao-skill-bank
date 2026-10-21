---
name: tao-run-on-windows-venv
description: >-
  Run a model action natively on a Windows host in a plain Python venv: no WSL,
  no Docker, and no container daemon. Provides reusable Windows process
  lifecycle (submit/status/logs/cancel), local-path checks, GPU visibility, and
  native ARM64 capacity probing, plus a generic vLLM Windows ARM64 source-build
  route. The selected model skill owns its dependency pins, CUDA ABI, model
  preflight, launch arguments, and support boundary.
license: Apache-2.0
compatibility: Native Windows 11 or Windows Server host with a model-compatible NVIDIA driver, a model-compatible x64 or ARM64 Python venv, and local NTFS inputs and outputs. Single GPU and single node with stock Windows PyTorch wheels. No WSL, Docker, nvidia-tao-sdk, GPU reservation, cgroup, or MIG support. POSIX hosts use tao-run-on-virtualenv instead.
metadata:
  author: NVIDIA Corporation
  version: "0.2.1"
allowed-tools: Read Bash
tags: [platform, windows, venv, local, native, arm64]
---

# Native Windows venv execution

> **Standalone install?** If this session was not initialized by the TAO skill
> bank plugin, run `tao-setup` first for host preflight, credentials, and
> cross-skill discovery.

Platform slug: **`windows-venv`**. This skill owns how a native Windows process
is launched and observed. It does not select a model, install model packages,
or turn a model's x64 compatibility evidence into ARM64 support.

## Use with a model skill

1. Select the model skill and read its native-Windows reference. It supplies the
   supported Python architecture, Torch/CUDA ABI, packages and DLL directories,
   source revision, GPU minimum, action command, environment variables, and
   acceptance checks.
2. Run the model's preflight inside the target venv. For an ARM64 model route,
   also run `references/windows-arm64.md`'s explicit allocation probe.
3. Use the runner below to create, observe, and cancel one local process tree.
   Keep inputs, outputs, the venv, and the working directory on local NTFS.

For Cosmos3, the model-specific route is
`skills/models/tao-finetune-cosmos-reason/references/cosmos3-windows.md`. It
owns all Cosmos pins, TorchCodec setup, source patches, LoRA settings, and WTS
validation evidence; do not copy those model details into this platform skill.

When a supported model needs vLLM on native Windows ARM64 and no compatible
wheel exists, read `references/vllm-windows-arm64-build.md`. That reference owns
the reusable source-build decision and delegates the exact build mechanics to
the selected vLLM Windows source tree. Return to the model skill for model load,
modalities, serving arguments, correctness checks, and performance validation.

## Preconditions

- Confirm the target interpreter is native to the intended architecture. ARM64
  validation requires ARM64 Python, not AMD64 emulation.
- Confirm the model's driver, Torch, CUDA, and compiled-extension ABI before
  installing anything. A successful `torch.cuda.is_available()` is not a proof
  that the model's allocation or extension imports work.
- Store all job inputs and outputs in paths visible to this host. This platform
  is local tier A; stage remote data before invoking it.
- Serialize GPU stages yourself. `--gpu-ids` sets `CUDA_VISIBLE_DEVICES`; it
  does not reserve memory or isolate a user from other Windows processes.

## Quick start

Use the venv interpreter directly. Do not activate a shell environment and do
not run the target through `cmd.exe` or PowerShell string evaluation.

```powershell
$BANK = $env:TAO_SKILL_BANK_PATH
$VENV = "C:\path\to\model-venv"
$RUNNER = "$BANK\skills\platform\tao-run-on-windows-venv\references\windows_job_runner.py"

# Run the selected model's own preflight here.

$JOB_ID = & "$VENV\Scripts\python.exe" "$BANK\scripts\tao_job_record.py" open --platform windows-venv --image $VENV --network-arch <model> --action <action> --storage-tier A --results-root <local-results-root>
$RESULTS_DIR = "<local-results-root>\$JOB_ID"

& "$VENV\Scripts\python.exe" $RUNNER submit --job-dir $RESULTS_DIR --venv $VENV `
  --module <model.python.module> --job-id $JOB_ID --cwd <model-worktree> `
  --arg=<model-argument> --gpu-ids 0 -e <MODEL_ENV_NAME>
```

Exactly one selector is required: `--script`, `--module`, or `--entrypoint`.
Use `--arg=TOKEN` for an argument that begins with `-`, including a bare `--`.
The runner accepts `{config_path}`, `{results_dir}`, and `{job_id}` inside an
`--arg` token. Pass secret environment variables by name with `-e`; never put
their values on argv or in job files.

## Four-verb lifecycle

Open the job record before launch. The runner operates on a local job directory
and returns the fixed vocabulary `PENDING`, `RUNNING`, `COMPLETE`, `ERROR`,
`CANCELED`, or `UNKNOWN`.

```powershell
& "$VENV\Scripts\python.exe" $RUNNER status --job-dir $RESULTS_DIR
& "$VENV\Scripts\python.exe" $RUNNER logs --job-dir $RESULTS_DIR --tail 200
& "$VENV\Scripts\python.exe" $RUNNER cancel --job-dir $RESULTS_DIR
```

After a terminal state, mark the TAO job record with that state and the runner
PID or terminal detail. A retry receives a new record and a new job directory;
never submit a second target into an existing directory.

```powershell
& "$VENV\Scripts\python.exe" "$BANK\scripts\tao_job_record.py" mark $JOB_ID --state <terminal-state> --source agent
```

`status` is a pure query. The runner uses `OpenProcess` and
`WaitForSingleObject`, not `os.kill(pid, 0)`: on Windows, CPython has no
signal-zero special case and that call can terminate the target. It also records
process creation time to distinguish PID reuse.

`cancel` uses a named Windows Job Object and `TerminateJobObject`, which removes
the launcher and every descendant. This is not graceful: a canceled job cannot
be promised a final checkpoint or `atexit` handler. `UNKNOWN` means the durable
exit record and the named Job Object are both unavailable; inspect logs and any
surviving processes before resubmitting.

## Windows constraints

- Stock Windows PyTorch wheels do not provide a validated CUDA-distributed path
  for this platform. Treat it as one GPU and one node. A model requires its own
  separately documented distributed solution before this constraint can change.
- Windows reports process failures as unsigned DWORDs. The raw code is retained
  in `exit_status.json`; an access violation is typically `3221225477`
  (`0xC0000005`), not a POSIX signal.
- There is no cgroup, MIG, fractional GPU, or scheduler equivalent. Check for
  other GPU work and serialize expensive stages.
- A model's toolchain, extension DLLs, codecs, visualizers, and service stack
  are model dependencies. Keep their setup and validation in that model skill.

## References

- `references/windows_job_runner.py` — native Windows four-verb process runner.
- `references/windows-arm64.md` — generic native ARM64 interpreter and CUDA
  capacity gate.
- `references/vllm-windows-arm64-build.md` — generic native Windows ARM64 vLLM
  source-build and wheel-acceptance contract.
- `references/skill_info.yaml` — machine-readable platform metadata.
