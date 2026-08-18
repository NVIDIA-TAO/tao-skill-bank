#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a spec-bundle for Brev, by delegating the container-how to docker.

Contract: `skills/core/tao-launch-workflow/references/bundle-rendering.md`.
This skill is documented as "a compound over Docker", so it does not restate a
single docker flag: it asks `tao-run-on-docker`'s renderer for the container
argv and wraps it for remote execution.

The wrap is the whole subtlety. `brev exec [instance...] <command>` treats only
the LAST positional as the command, which is why this skill's SKILL.md warns
that a single-token probe passes while a real `docker run ...` does not. So the
container argv is collapsed into ONE shell-quoted string argument.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shlex
import subprocess
from typing import Any

PLATFORM = "brev"


def _docker_renderer():
    """Reuse the docker skill's renderer rather than restating its conventions."""
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "tao-run-on-docker/references/render.py"
    )
    if not path.is_file():
        raise ValueError(f"brev delegates to the docker renderer; not found at {path}")
    spec = importlib.util.spec_from_file_location("tao_render_docker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(bundle: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Bundle -> `brev exec <instance> '<docker run ...>'`."""
    instance = ctx["instance"]
    inner = _docker_renderer().render(bundle, ctx)
    if inner["files"]:
        raise ValueError("the docker renderer returned files; brev cannot stage them")
    command = " ".join(shlex.quote(token) for token in inner["argv"])
    return {
        "files": {},
        # One quoted positional: brev takes only the last one as the command.
        "argv": ["brev", "exec", instance, command],
        "backend_ref": None,
    }


def status(backend_ref: str, ctx: dict[str, Any]) -> tuple[str, int]:
    """Ask docker on the instance, and map with the docker vocabulary."""
    docker = _docker_renderer()
    probe = subprocess.run(
        ["brev", "exec", ctx["instance"],
         f"docker inspect --format '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' "
         f"{shlex.quote(backend_ref)}"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        return "UNKNOWN", 0
    parts = probe.stdout.strip().split()
    native = parts[0]
    exit_code = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else 0
    if native in docker.STATE_VOCAB:
        return docker.STATE_VOCAB[native], exit_code
    if native == "exited":
        return ("COMPLETE" if exit_code == 0 else "ERROR"), exit_code
    return "UNKNOWN", exit_code
