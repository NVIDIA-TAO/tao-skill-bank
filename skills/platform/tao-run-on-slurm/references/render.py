#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a spec-bundle into an sbatch script, and map sacct state.

Contract: `skills/core/tao-launch-workflow/references/bundle-rendering.md`.
The conventions encoded here are this skill's own: Pyxis/Enroot container args,
a Lustre `.sqsh` image, the mode-600 credential sidecar the template shreds on
exit, and `sbatch --parsable` for the handle.

The sbatch body is NOT written here — it is `templates/slurm/singlenode.sbatch.tmpl`
with its `@@NAME@@` placeholders substituted, so the credential sidecar handling
and the shred-on-exit trap stay in one reviewed place.
"""

from __future__ import annotations

import pathlib
import re
import shlex
import subprocess
from typing import Any

PLATFORM = "slurm"

TEMPLATE = "templates/slurm/singlenode.sbatch.tmpl"

# sacct states that are not terminal; everything else folds by exit code.
STATE_VOCAB = {
    "PENDING": "PENDING",
    "CONFIGURING": "PENDING",
    "REQUEUED": "PENDING",
    "RUNNING": "RUNNING",
    "COMPLETING": "RUNNING",
    "SUSPENDED": "RUNNING",
    "COMPLETED": "COMPLETE",
    "FAILED": "ERROR",
    "TIMEOUT": "ERROR",
    "OUT_OF_MEMORY": "ERROR",
    "NODE_FAIL": "ERROR",
    "BOOT_FAIL": "ERROR",
    "CANCELLED": "CANCELED",
    "DEADLINE": "CANCELED",
    "PREEMPTED": "CANCELED",
}


def _mounts(bundle: dict[str, Any], results_dir: str) -> str:
    """Pyxis --container-mounts: src:dst pairs, same path on both sides."""
    pairs: list[str] = []
    seen: set[str] = set()
    for item in bundle.get("declared_inputs") or []:
        uri = str(item["uri"])
        if "://" in uri:
            raise ValueError(
                f"declared_input {item['spec_key']} is {uri!r}; stage it onto "
                "Lustre with tao-data-io first and declare that path"
            )
        if not uri.startswith("/"):
            raise ValueError(
                f"declared_input {item['spec_key']} must be absolute, got {uri!r}"
            )
        if uri not in seen:
            pairs.append(f"{uri}:{uri}")
            seen.add(uri)
    pairs.append(f"{results_dir}:{results_dir}")
    return ",".join(pairs)


def render(bundle: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Bundle -> a rendered sbatch script plus the ssh sbatch command."""
    job_id = ctx["job_id"]
    results_dir = ctx["results_dir"]
    bank = pathlib.Path(ctx["bank"])
    template = (bank / TEMPLATE).read_text(encoding="utf-8")

    shape = bundle["compute_shape"]
    if int(shape["nodes"]) > 1:
        raise ValueError(
            "multi-node bundles need templates/slurm/multinode.sbatch.tmpl and the "
            "NCCL probe; render() covers the single-node template only"
        )

    command = " ".join(
        shlex.quote(token)
        for token in [*shlex.split(bundle["command"]), *(bundle.get("args") or [])]
    )
    job_dir = ctx.get("job_dir") or results_dir
    substitutions = {
        "JOB_NAME": job_id,
        "NUM_GPUS": str(int(shape["gpus"])),
        "CPUS_PER_TASK": str(ctx.get("cpus_per_task", 8)),
        "TIME": str(ctx.get("time_limit", "04:00:00")),
        "LOG_DIR": f"{job_dir}/logs",
        "IMAGE": bundle["image"],
        "CONTAINER_MOUNTS": _mounts(bundle, results_dir),
        "COMMAND": command,
        "SBATCH_EXTRA": str(ctx.get("sbatch_extra", "")),
        "ENV_FILE": str(ctx.get("env_file", "")),
        # Same convention as the k8s templates and virtualenv_runner: the
        # workload finds its output path here, so the bundle never names it.
        "EXTRA_ENV": "\n".join(
            filter(None, [f"export TAO_RESULTS_ROOT={shlex.quote(results_dir)}",
                          str(ctx.get("extra_env", ""))])
        ),
    }
    body = template
    for name, value in substitutions.items():
        body = body.replace(f"@@{name}@@", value)

    # Only @@UPPER_CASE@@ are real slots; the template header documents the
    # convention with a literal "@@<NAME>@@", which must not trip this check.
    remaining = sorted(set(re.findall(r"@@[A-Z][A-Z0-9_]*@@", body)))
    if remaining:
        raise ValueError(f"unsubstituted template placeholders: {', '.join(remaining)}")

    script = f"{job_dir}/sbatch/job_{job_id}.sbatch"
    login = ctx["login"]
    return {
        "files": {script: body},
        "argv": ["ssh", login, f"sbatch --parsable {shlex.quote(script)}"],
        "backend_ref": None,  # sbatch --parsable prints the id
    }


def status(backend_ref: str, ctx: dict[str, Any]) -> tuple[str, int]:
    """Map `sacct` state into the fixed vocabulary."""
    probe = subprocess.run(
        ["ssh", ctx["login"],
         f"sacct -j {shlex.quote(backend_ref)} --format=State,ExitCode -n -P | head -1"],
        capture_output=True, text=True, check=False,
    )
    line = probe.stdout.strip()
    if probe.returncode != 0 or not line:
        return "UNKNOWN", 0
    state, _, exit_field = line.partition("|")
    # sacct decorates cancelled states, e.g. "CANCELLED by 12345".
    state = state.strip().split()[0].rstrip("+")
    code = 0
    if ":" in exit_field:
        code = int(exit_field.split(":")[0] or 0)
    return STATE_VOCAB.get(state, "UNKNOWN"), code
