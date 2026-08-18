#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a spec-bundle for docker-free virtualenv execution.

Contract: `skills/core/tao-launch-workflow/references/bundle-rendering.md`.
This is the one platform with no container, so the bundle's `image` is the venv
interpreter rather than a registry URI, and `declared_inputs` need no mounting —
the process already sees the filesystem. What must still hold is everything the
four verbs depend on: the job is named after the minted id, and lifecycle goes
through the packaged `virtualenv_runner.py`, which owns the detached start, the
identity record and the process-group cleanup.

A GPU request is meaningless here: there is no container to place, so a bundle
asking for GPUs is refused rather than silently run on whatever the host has.
"""

from __future__ import annotations

import pathlib
import shlex
from typing import Any

PLATFORM = "virtualenv"

RUNNER = "skills/platform/tao-run-on-virtualenv/references/virtualenv_runner.py"

STATE_VOCAB = {
    "pending": "PENDING",
    "running": "RUNNING",
    "complete": "COMPLETE",
    "completed": "COMPLETE",
    "error": "ERROR",
    "failed": "ERROR",
    "canceled": "CANCELED",
    "cancelled": "CANCELED",
}


def prepare(bundle: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """There is no image to fetch; check the interpreter exists.

    The failure this prevents is a job that reaches RUNNING and dies instantly
    because the venv was never provisioned. This platform never installs
    anything — see this skill's SKILL.md — so a missing interpreter is fatal,
    not something to fix here.
    """
    interpreter = pathlib.Path(bundle["image"])
    if not interpreter.is_file():
        raise ValueError(
            f"venv interpreter {interpreter} does not exist; provision the "
            "environment first — this platform never installs packages"
        )
    return {"image": str(interpreter), "notes": ["interpreter present"]}


def render(bundle: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Bundle -> `virtualenv_runner.py submit ...`."""
    job_id = ctx["job_id"]
    results_dir = ctx["results_dir"]
    runner = str(pathlib.Path(ctx["bank"]) / RUNNER)

    if int(bundle["compute_shape"]["gpus"]) > 0:
        raise ValueError(
            "virtualenv has no container to place a GPU into; a GPU bundle "
            "belongs on docker, slurm or kubernetes"
        )
    for item in bundle.get("declared_inputs") or []:
        uri = str(item["uri"])
        if "://" in uri:
            raise ValueError(
                f"declared_input {item['spec_key']} is {uri!r}; stage it with "
                "tao-data-io first and declare the local path"
            )

    # `image` is the interpreter on this platform, per this skill's SKILL.md
    # (`--image "$VENV/bin/python"`). A registry URI here means the bundle was
    # authored for a container platform; running it would silently derive a
    # nonsense venv root, so say so instead.
    interpreter = pathlib.Path(bundle["image"])
    if not bundle["image"].startswith("/") or interpreter.parent.name != "bin":
        raise ValueError(
            f"virtualenv expects `image` to be a venv interpreter path such as "
            f"/opt/venv/bin/python, got {bundle['image']!r}; a registry URI "
            "belongs on docker, slurm, kubernetes or brev"
        )

    tokens = [*shlex.split(bundle["command"]), *(bundle.get("args") or [])]
    if not tokens:
        raise ValueError("bundle command is empty")
    # `image` is the interpreter for this platform; the runner takes the script.
    script, script_args = tokens[0], tokens[1:]
    argv = [
        "python3", runner, "submit",
        "--job-dir", results_dir,
        "--venv", str(interpreter.parent.parent),
        "--script", script,
        "--job-id", job_id,
    ]
    for name in ctx.get("env_passthrough") or []:
        argv += ["--env", name]
    if script_args:
        argv += ["--", *script_args]
    return {"files": {}, "argv": argv, "backend_ref": None}


def status(backend_ref: str, ctx: dict[str, Any]) -> tuple[str, int]:
    """Ask the packaged runner, which derives status from durable state."""
    import json
    import subprocess

    runner = str(pathlib.Path(ctx["bank"]) / RUNNER)
    probe = subprocess.run(
        ["python3", runner, "status", "--job-dir", ctx["results_dir"]],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        return "UNKNOWN", 0
    try:
        payload = json.loads(probe.stdout)
    except json.JSONDecodeError:
        native = probe.stdout.strip().split()[0]
        return STATE_VOCAB.get(native.lower(), "UNKNOWN"), 0
    native = str(payload.get("state") or payload.get("status") or "").lower()
    return STATE_VOCAB.get(native, "UNKNOWN"), int(payload.get("exit_code") or 0)
