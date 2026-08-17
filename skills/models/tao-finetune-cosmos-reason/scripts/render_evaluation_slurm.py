#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a deterministic Pyxis SLURM job for a READY Cosmos evaluation plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class RenderError(ValueError):
    """A deterministic evaluation-launch contract failure."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RenderError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seconds(value: str, field: str = "child timeout") -> int:
    try:
        hours, minutes, seconds = (int(part) for part in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise RenderError(f"{field} must use HH:MM:SS") from exc
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise RenderError(f"{field} must use valid HH:MM:SS fields")
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise RenderError(f"{field} must be positive")
    return total


def _directive_token(name: str, value: str, pattern: str) -> str:
    if not re.fullmatch(pattern, value):
        raise RenderError(f"invalid {name} for an SBATCH directive: {value!r}")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mount_target(mount: str) -> str:
    fields = mount.split(":")
    if len(fields) < 2 or not fields[0] or not fields[1]:
        raise RenderError(f"invalid container mount; expected host:container[:flags]: {mount}")
    return fields[1]


def render(args: argparse.Namespace) -> str:
    job_id = _directive_token(
        "TAO job id", args.tao_job_id, r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
    )
    partition = _directive_token(
        "partition", args.partition, r"[A-Za-z0-9][A-Za-z0-9_,.-]*"
    )
    account = _directive_token("account", args.account, r"[A-Za-z0-9][A-Za-z0-9_.-]*")
    _seconds(args.time_limit, "time limit")
    plan = _load_object(args.evaluation_plan.expanduser().resolve())
    if plan.get("schema_version") != 1 or plan.get("ready") is not True:
        raise RenderError("evaluation plan must be READY schema version 1")
    backend = plan.get("backend")
    if backend not in {"cosmos-framework", "cosmos-rl"}:
        raise RenderError(f"unsupported Cosmos evaluation backend: {backend!r}")
    if plan.get("required_user_inputs") or plan.get("blockers"):
        raise RenderError("READY evaluation plan still contains unresolved inputs or blockers")

    config_path = args.config.expanduser().resolve()
    expected_config_sha = str(plan.get("config_sha256") or "")
    actual_config_sha = _sha256(config_path)
    if not expected_config_sha or actual_config_sha != expected_config_sha:
        raise RenderError(
            f"evaluation config checksum mismatch: expected {expected_config_sha or '<missing>'}, "
            f"found {actual_config_sha}"
        )
    config = plan.get("config")
    if not isinstance(config, dict):
        raise RenderError("evaluation plan has no resolved config object")
    results_dir = Path(str(config.get("results_dir") or ""))
    if (
        not results_dir.is_absolute()
        or results_dir.name != job_id
        or any(character in str(results_dir) for character in "\r\n")
    ):
        raise RenderError(
            "evaluation results_dir must be absolute and end with the TAO job id"
        )
    runtime_config = Path(args.runtime_config_path)
    if not runtime_config.is_absolute() or not runtime_config.is_relative_to(results_dir):
        raise RenderError("runtime config path must be absolute and inside results_dir")

    checkpoint = plan.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RenderError("evaluation plan has no checkpoint contract")
    manifest = checkpoint.get("action_model_manifest")
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    if not isinstance(manifest, dict) or manifest.get("status") != "VERIFIED":
        raise RenderError("evaluation plan lacks a VERIFIED action checkpoint manifest")
    if manifest.get("action_model_path") != model.get("model_name"):
        raise RenderError("verified checkpoint manifest does not match model.model_name")
    if manifest.get("backend") != backend:
        raise RenderError("verified checkpoint manifest backend does not match evaluation backend")

    framework_runtime_gate = ""
    evaluator = "cosmos-rl-evaluate"
    if backend == "cosmos-framework":
        vision = config.get("vision") if isinstance(config.get("vision"), dict) else {}
        expected_runtime = {
            "video_decoder": "torchcodec-cuda-on-demand",
            "process_threads": 8,
            "decoder_threads": 1,
            "decoder_device": "cuda",
            "dataloader_num_workers": 1,
            "dataloader_prefetch_factor": 2,
            "dataloader_multiprocessing_context": "spawn",
            "dataloader_persistent_workers": True,
        }
        mismatches = {
            key: {"expected": value, "actual": vision.get(key)}
            for key, value in expected_runtime.items()
            if vision.get(key) != value
        }
        maximum_pixels = int(vision.get("max_pixels") or 0)
        minimum_pixels = int(vision.get("min_pixels") or 0)
        if (
            mismatches
            or int(vision.get("video_cache_size") or 0) <= 0
            or maximum_pixels <= 0
            or minimum_pixels != maximum_pixels
        ):
            raise RenderError(
                "Framework evaluation config does not preserve the sealed TorchCodec runtime "
                "and compatible Qwen pixel bounds: "
                f"runtime={json.dumps(mismatches, sort_keys=True)} "
                f"min_pixels={minimum_pixels} max_pixels={maximum_pixels}"
            )
        evaluator = "cosmos-framework-evaluate"
        # This is a read-only capability attestation against the code baked in
        # the selected SQSH.  It never stages or overlays evaluator source.
        probe = (
            "import inspect; "
            "from cosmos_rl.evaluation.base import BaseEvaluator; "
            "from cosmos_rl.framework.runtime import CosmosFrameworkRuntime; "
            "from cosmos_rl.utils.framework_torchcodec_video import FrameworkTorchCodecVideoPreprocessor; "
            "assert 'torchcodec-cuda-on-demand' in inspect.getsource(BaseEvaluator.load_model); "
            "assert '_framework_decoded_media' in inspect.getsource(CosmosFrameworkRuntime._task_conversation); "
            "preprocessor_source = inspect.getsource(FrameworkTorchCodecVideoPreprocessor); "
            "assert 'iter_prepared_batches' in preprocessor_source; "
            "assert 'persistent_workers=self.dataloader_persistent_workers' in preprocessor_source; "
            "assert 'multiprocessing_context=self.dataloader_multiprocessing_context' in preprocessor_source; "
            "assert FrameworkTorchCodecVideoPreprocessor.__name__ == 'FrameworkTorchCodecVideoPreprocessor'"
        )
        framework_runtime_gate = (
            "/workspace/.venv/bin/python -c " + shlex.quote(probe)
            + " && echo TAO_FRAMEWORK_EVALUATOR_RUNTIME_OK"
        )

    total_gpus = int(config.get("num_gpus") or 0)
    if total_gpus <= 0 or args.gpus_per_node <= 0 or total_gpus % args.gpus_per_node:
        raise RenderError(
            f"num_gpus={total_gpus} must be a positive multiple of gpus_per_node={args.gpus_per_node}"
        )
    nodes = total_gpus // args.gpus_per_node
    if nodes <= 0:
        raise RenderError("derived SLURM node count must be positive")
    if not Path(args.sqsh_path).is_absolute() or not args.sqsh_path.endswith(".sqsh"):
        raise RenderError("sqsh_path must be an absolute existing-image .sqsh identity")
    if args.cpus_per_task <= 0:
        raise RenderError("cpus_per_task must be positive")

    results_parent = results_dir.parent
    mounts = list(args.mount)
    if any(_mount_target(value) == "/results" for value in mounts):
        raise RenderError("do not supply a /results mount; the renderer owns its persistent binding")
    mounts.append(f"{results_parent}:/results")
    if len({_mount_target(value) for value in mounts}) != len(mounts):
        raise RenderError("container mount targets must be unique")

    timeout_seconds = _seconds(args.child_timeout)
    environment = {
        "NCCL_DEBUG": "WARN",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
        "PYTHONHASHSEED": str(config.get("evaluation", {}).get("seed", 0)),
        "PYTHONUNBUFFERED": "1",
        "TAO_API_JOB_ID": job_id,
        "TAO_API_RESULTS_DIR": "/results",
        "TAO_JOB_ID": job_id,
        "TAO_RESULTS_ROOT": "/results",
        "TAO_STATUS_FILE": f"/results/{job_id}/status.json",
        "TAO_STATUS_PATH": f"/results/{job_id}/status.json",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    container_env = sorted(environment)
    if nodes > 1:
        container_env.extend(["MASTER_ADDR", "MASTER_PORT"])
    env_lines = [f"export {key}={shlex.quote(value)}" for key, value in environment.items()]

    if nodes == 1:
        torchrun = (
            f"torchrun --standalone --nnodes=1 --nproc-per-node={args.gpus_per_node} "
            f'"$(command -v {evaluator})" '
            f"--config {shlex.quote(str(runtime_config))}"
        )
    else:
        torchrun = (
            f'torchrun --nnodes="${{SLURM_NNODES:?}}" --nproc-per-node={args.gpus_per_node} '
            '--node-rank="${SLURM_PROCID:?}" --master-addr="${MASTER_ADDR:?}" '
            f'--master-port="${{MASTER_PORT:?}}" "$(command -v {evaluator})" '
            f"--config {shlex.quote(str(runtime_config))}"
        )
    wrapped = "; ".join(
        [
            'export HOME="/tmp/tao-${TAO_JOB_ID:?}-${SLURM_PROCID:-0}"',
            'mkdir -p -m 700 "$HOME"',
            "ulimit -n 65536",
            "ulimit -s unlimited",
            *([framework_runtime_gate] if framework_runtime_gate else []),
            torchrun,
        ]
    )
    cpu_value = '"$step_cpus_per_task"' if nodes == 1 else str(args.cpus_per_task)
    srun = " ".join(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            f"{timeout_seconds}s",
            "srun",
            f"--nodes={nodes}",
            f"--ntasks={nodes}",
            "--ntasks-per-node=1",
            f"--gpus-per-node={args.gpus_per_node}",
            f"--cpus-per-task={cpu_value}",
            "--no-container-remap-root",
            "--no-container-mount-home",
            f"--container-env={','.join(sorted(container_env))}",
            f"--container-image={shlex.quote(args.sqsh_path)}",
            f"--container-mounts={shlex.quote(','.join(mounts))}",
            "bash -lc",
            shlex.quote(wrapped),
        ]
    )

    lines = [
        "#!/usr/bin/env bash",
        f"# evaluation_backend={backend}",
        f"# evaluation_config_sha256={actual_config_sha}",
        f"#SBATCH --job-name={job_id}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --account={account}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={nodes}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gpus-per-node={args.gpus_per_node}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --time={args.time_limit}",
        "#SBATCH --no-requeue",
        "#SBATCH --exclusive",
        f"#SBATCH --output={results_dir}/slurm-%j.out",
        f"#SBATCH --error={results_dir}/slurm-%j.err",
        "",
        "set -Eeuo pipefail",
        "export SLURM_EXPORT_ENV=ALL",
        *env_lines,
        f"mkdir -p -- {shlex.quote(str(results_dir))}",
    ]
    if nodes == 1:
        lines.extend(
            [
                f"requested_cpus_per_task={args.cpus_per_task}",
                'slurm_job_record="$(scontrol show job -o "${SLURM_JOB_ID:?}")"',
                'if [[ "$slurm_job_record" =~ NumCPUs=([0-9]+) ]]; then',
                '  step_cpus_per_task="${BASH_REMATCH[1]}"',
                "else",
                '  echo "Unable to resolve allocated CPUs for $SLURM_JOB_ID" >&2',
                "  exit 2",
                "fi",
                'if (( step_cpus_per_task < requested_cpus_per_task )); then exit 2; fi',
            ]
        )
    else:
        lines.extend(
            [
                'export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"',
                'export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"',
            ]
        )
    lines.extend(
        [
            "child_rc=0",
            "set +e",
            srun,
            'child_rc="$?"',
            "set -e",
            f"printf '%s\\n' \"$child_rc\" > {shlex.quote(str(results_dir / 'child_exit_code'))}",
            'exit "$child_rc"',
            "",
        ]
    )
    script = "\n".join(lines)
    checked = subprocess.run(
        ["bash", "-n"], input=script, text=True, capture_output=True, check=False
    )
    if checked.returncode:
        raise RenderError(f"generated SLURM script is invalid: {checked.stderr}")
    return script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-config-path", required=True)
    parser.add_argument("--tao-job-id", required=True)
    parser.add_argument("--sqsh-path", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--cpus-per-task", type=int, required=True)
    parser.add_argument("--time-limit", default="04:00:00")
    parser.add_argument("--child-timeout", default="03:48:00")
    parser.add_argument("--mount", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        script = render(args)
        _atomic_write(args.output, script)
        print(json.dumps({"output": str(args.output.resolve()), "sha256": _sha256(args.output)}))
        return 0
    except (OSError, RenderError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
