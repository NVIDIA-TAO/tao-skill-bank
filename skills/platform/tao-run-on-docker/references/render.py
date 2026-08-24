#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a spec-bundle into a local `docker run`, and map docker state.

The contract this implements is documented in
`skills/core/tao-launch-workflow/references/bundle-rendering.md`; the docker
conventions it encodes (GPU flags, mounts, naming) are the ones in this skill's
SKILL.md. Keeping the code here means a new platform never requires an edit to
whatever workflow is calling it.
"""

from __future__ import annotations

import getpass
import os
import pathlib
import re
import shlex
import subprocess
from typing import Any

PLATFORM = "docker"

OFFLINE_ENV = {
    "AIR_GAPPED": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PIP_NO_INDEX": "1",
}

# Native docker states that are not terminal. `exited` and `dead` are resolved
# by exit code at call time, so they are deliberately absent.
STATE_VOCAB = {
    "created": "PENDING",
    "restarting": "PENDING",
    "running": "RUNNING",
    "paused": "RUNNING",
}


# ── Host identity and runtime environment ───────────────────────────────────
# This skill's SKILL.md documents these as REQUIRED for any writable bind
# mount, and render() emitted none of them. They are not cosmetic:
#
#  * Without --user, the container writes checkpoints as root and the
#    submitting user cannot delete their own results tree.
#  * Without USER/LOGNAME, torch 2.x crashes at IMPORT -- getpass.getuser()
#    runs during `torch/_dynamo` inductor cache setup, and an arbitrary --user
#    UID has no /etc/passwd entry in the image, so it raises
#    KeyError: 'getpwuid(): uid not found: <uid>' before any workload code.
#  * Without --shm-size, torchrun and PyTorch DataLoaders exhaust docker's
#    64 MB /dev/shm default and die with `Bus error`.
#
# None of this belongs in the spec-bundle: it is how DOCKER expresses things
# other platforms get for free (enroot is rootless and exposes the host tmpfs;
# kubernetes has a dshm emptyDir and a securityContext).

RUNTIME_SUBDIR = ".tao-runtime/home"

# Frameworks that would otherwise write into image-owned paths like /root,
# which an arbitrary --user UID cannot create.
CACHE_ENV = {
    "XDG_CACHE_HOME": "",
    "HF_HOME": "huggingface",
    "TORCH_HOME": "torch",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "MPLCONFIGDIR": "matplotlib",
}


def runtime_home(results_dir: str) -> str:
    return f"{results_dir.rstrip('/')}/{RUNTIME_SUBDIR}"


def effective_identity(ctx: dict[str, Any]) -> tuple[int, int]:
    """The uid:gid a launch would use. ctx wins; otherwise this process."""
    uid, gid = ctx.get("uid"), ctx.get("gid")
    if uid is None or gid is None:
        uid, gid = os.getuid(), os.getgid()
    return int(uid), int(gid)


def preflight_launch(bundle: dict[str, Any], ctx: dict[str, Any]) -> None:
    """Refuse to LAUNCH as UID 0. Called immediately before argv runs.

    SKILL.md: "Refuse UID 0 for the canonical writable-bind path. If the
    launcher itself is root, obtain the verified non-root submitting UID:GID
    explicitly; never infer it from the output-directory owner."

    This is a launch gate, NOT a render gate. Rendering is not launching: a CI
    container inspecting or asserting on a command legitimately runs as root,
    and nothing it does can create a root-owned results tree. Putting the
    refusal in render() made every render-only test fail the moment the suite
    ran somewhere as root -- which is exactly what happened.
    """
    uid, _ = effective_identity(ctx)
    if uid == 0:
        raise ValueError(
            "refusing a writable docker launch as UID 0; pass the verified "
            "non-root submitting uid/gid in ctx (uid=, gid=)"
        )


def identity_args(ctx: dict[str, Any]) -> list[str]:
    """--user plus supplementary groups.

    Returns [] when the only identity available is root: there is no safe
    non-root id to infer, and SKILL.md forbids guessing one from the
    output-directory owner. preflight_launch() is what stops such a launch;
    this function only has to avoid emitting `--user 0:0`.
    """
    uid, gid = effective_identity(ctx)
    if uid == 0:
        return []
    args = ["--user", f"{uid}:{gid}"]
    groups = ctx.get("groups")
    if groups is None:
        groups = os.getgroups() if hasattr(os, "getgroups") else []
    # Preserve supplementary access to shared datasets and workspaces.
    for group in groups:
        if int(group) != gid:
            args += ["--group-add", str(int(group))]
    return args


def runtime_env(results_dir: str, ctx: dict[str, Any]) -> list[str]:
    """HOME, USER/LOGNAME and cache redirects onto the writable mount."""
    home = runtime_home(results_dir)
    name = str(ctx.get("user_name") or getpass.getuser() or "tao")
    env = ["-e", f"HOME={home}",
           # Any non-empty name satisfies getpass.getuser(); it need not exist
           # in the image.
           "-e", f"USER={name}", "-e", f"LOGNAME={name}"]
    for var, leaf in CACHE_ENV.items():
        env += ["-e", f"{var}={home}/.cache" + (f"/{leaf}" if leaf else "")]
    return env


def prepare(bundle: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Make the image locally available. Pull only when it is missing.

    This is the `docker image inspect … || docker pull` idiom from this skill's
    SKILL.md, hoisted out of the run: an implicit first-time pull inside
    `docker run` is billed as idle GPU time on a metered host, and it hides an
    auth failure inside a training log.
    """
    # SKILL.md: "Prepare these directories on the writable mount before
    # launch." An arbitrary --user UID may not be able to create them itself
    # once the container is running, and a framework that cannot write its
    # cache fails deep inside an import rather than at startup.
    results_dir = ctx.get("results_dir")
    if results_dir and pathlib.Path(results_dir).is_dir():
        home = pathlib.Path(runtime_home(str(results_dir)))
        for leaf in CACHE_ENV.values():
            target = home / ".cache" / leaf if leaf else home / ".cache"
            try:
                target.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(
                    f"cannot prepare the runtime cache dir {target}: {exc}. "
                    "results_dir must be writable by the submitting user "
                    "before launch"
                ) from exc

    image = bundle["image"]
    present = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True, check=False,
    ).returncode == 0
    if present:
        return {"image": image, "notes": ["image already present"]}
    if ctx.get("airgap"):
        raise ValueError(
            f"{image} is not present locally and air-gap forbids a pull; "
            "pre-stage the image on this host"
        )
    pulled = subprocess.run(
        ["docker", "pull", image], capture_output=True, text=True, check=False
    )
    if pulled.returncode != 0:
        raise ValueError(f"docker pull {image} failed: {pulled.stderr.strip()}")
    return {"image": image, "notes": ["pulled"]}


def input_env(bundle: dict[str, Any]) -> dict[str, str]:
    """Declared inputs as TAO_INPUT_<SPEC_KEY>, mirroring TAO_RESULTS_ROOT.

    A bundle declares its inputs by spec_key and URI, but the path the WORKLOAD
    sees is chosen by the platform, so a command naming a path directly is
    guessing at a layout it cannot see. When it guesses wrong the input is
    simply absent, and a command that does not check produces empty output and
    exits 0 -- a job reporting COMPLETE having read nothing.

    Outputs never had this problem because TAO_RESULTS_ROOT already exists.
    This is the same convention for the other direction.
    """
    env: dict[str, str] = {}
    for item in bundle.get("declared_inputs") or []:
        key = re.sub(r"[^A-Za-z0-9]+", "_", str(item["spec_key"])).strip("_").upper()
        if key:
            env[f"TAO_INPUT_{key}"] = str(item["uri"])
    return env


def render(bundle: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Bundle -> `docker run` argv. No files needed; the handle is stdout."""
    job_id = ctx["job_id"]
    results_dir = ctx["results_dir"]

    mounts: list[str] = []
    seen: set[str] = set()
    for item in bundle.get("declared_inputs") or []:
        uri = str(item["uri"])
        if "://" in uri:
            raise ValueError(
                f"declared_input {item['spec_key']} is {uri!r}; stage it with "
                "tao-data-io first and declare the compute-frame path"
            )
        if not uri.startswith("/"):
            raise ValueError(
                f"declared_input {item['spec_key']} must be absolute, got {uri!r}"
            )
        if uri not in seen:
            # Same absolute path on both sides: a path written into a CSV or a
            # spec must resolve identically inside and outside the container.
            mounts += ["-v", f"{uri}:{uri}:ro"]
            seen.add(uri)
    mounts += ["-v", f"{results_dir}:{results_dir}"]

    gpus = int(bundle["compute_shape"]["gpus"])
    resources = ["--gpus", "all"] if gpus > 0 else []
    # Air-gap is a docker convention here: no implicit pull, and the offline
    # env the workload libraries read. It belongs with the other docker flags
    # rather than in whatever workflow happens to be calling render().
    if ctx.get("airgap"):
        resources += ["--pull=never"]
        resources += [f"--env={name}={value}" for name, value in OFFLINE_ENV.items()]
    # The bundle is authored before the job id exists, but results_dir contains
    # it. TAO_RESULTS_ROOT is how the bank already tells a workload where to
    # write (k8s templates set it, virtualenv_runner sets it, tao-data-io keys
    # its upload decision on it), so a bundle never has to name the path.
    env = ["-e", f"TAO_RESULTS_ROOT={results_dir}"]
    # Same convention for the other direction: the workload finds its declared
    # inputs by spec_key instead of guessing this platform's mount layout.
    env += [arg for name, value in input_env(bundle).items()
            for arg in ("-e", f"{name}={value}")]
    env += [arg for name in ctx.get("env_passthrough") or [] for arg in ("-e", name)]

    workdir = ["-w", bundle["workdir"]] if bundle.get("workdir") else []
    # Docker's 64MB /dev/shm default is what makes this necessary; slurm and
    # kubernetes each solve it their own way, so it is rendered, not declared.
    shm = ["--shm-size", str(ctx.get("shm_size", "8g"))]
    argv = ["docker", "run", "--name", job_id, "-d", *shm,
            *identity_args(ctx), *runtime_env(results_dir, ctx),
            *resources, *env, *mounts, *workdir, bundle["image"]]
    argv += shlex.split(bundle["command"])
    argv += list(bundle.get("args") or [])
    return {"files": {}, "argv": argv, "backend_ref": None}


def status(backend_ref: str, ctx: dict[str, Any]) -> tuple[str, int]:
    """Map the container's native state into the fixed vocabulary."""
    probe = subprocess.run(
        ["docker", "inspect", "--format",
         "{{.State.Status}} {{.State.ExitCode}}", backend_ref],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        return "UNKNOWN", 0
    native, _, code = probe.stdout.strip().partition(" ")
    exit_code = int(code or 0)
    if native in STATE_VOCAB:
        return STATE_VOCAB[native], exit_code
    if native == "exited":
        return ("COMPLETE" if exit_code == 0 else "ERROR"), exit_code
    return "UNKNOWN", exit_code


def logs(backend_ref: str, ctx: dict[str, Any], tail: int = 200) -> str:
    """Bounded tail of the container's output.

    Both streams: docker keeps the workload's stdout and stderr apart, and a
    diagnosis almost always needs them interleaved -- a traceback on stderr is
    meaningless without the progress line on stdout that preceded it.
    """
    probe = subprocess.run(
        ["docker", "logs", "--tail", str(int(tail)), backend_ref],
        capture_output=True, text=True, check=False,
    )
    return (probe.stdout + probe.stderr).strip()


def cancel(backend_ref: str, ctx: dict[str, Any]) -> bool:
    """Stop the container, leaving it inspectable.

    `docker rm -f` would delete it and with it the exit code status() reads, so
    the job would go permanently UNKNOWN instead of settling at CANCELED. The
    caller marks the record; this only has to stop the work.
    """
    stopped = subprocess.run(
        ["docker", "stop", backend_ref], capture_output=True, text=True, check=False,
    )
    return stopped.returncode == 0
