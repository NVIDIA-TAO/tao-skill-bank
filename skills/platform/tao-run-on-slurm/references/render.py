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


SQSH_MAGIC = "hsqs"

# Scheduling identity a cluster may require on EVERY allocation. Held in one
# place because this renderer issues more than one: the job itself, and the
# image conversion. A conversion that silently omits the account fails with a
# scheduler error deep inside prepare(), which reads as "conversion is broken"
# rather than "this cluster wants an account". Values are supplied per cluster
# via ctx; nothing here names a site.
SCHEDULING_KEYS = ("account", "qos", "reservation")
SRUN_FLAG = {"account": "-A", "qos": "--qos", "reservation": "--reservation"}


def scheduling_srun_flags(ctx: dict[str, Any]) -> list[str]:
    """Scheduler identity as srun flags, for any allocation this module makes."""
    flags: list[str] = []
    for key in SCHEDULING_KEYS:
        value = str(ctx.get(key) or "").strip()
        if value:
            flags += [SRUN_FLAG[key], value]
    return flags


def scheduling_sbatch_directives(ctx: dict[str, Any]) -> str:
    """The same identity as #SBATCH lines, so both paths cannot diverge."""
    lines = []
    for key in SCHEDULING_KEYS:
        value = str(ctx.get(key) or "").strip()
        if value:
            lines.append(f"#SBATCH --{key}={value}")
    partition = str(ctx.get("partition") or "").strip()
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    return "\n".join(lines)


def enroot_uri(image: str) -> str:
    """Translate a canonical `registry/path:tag` into enroot's `registry#path:tag`.

    The spec-bundle requires a fully-qualified image URI, and docker consumes
    that form directly. Enroot does not: its syntax separates the registry with
    `#`, so `docker://docker.io/library/alpine:3.20` is read as an image whose
    PATH begins with "docker.io", and the registry request becomes
    `/v2/docker.io/library/alpine/manifests/...` — which fails as a 401 rather
    than a not-found, so it reads like a credentials problem.

    A leading component is a registry when it looks like a host: it contains a
    dot or a port, or is localhost. Otherwise the reference is already
    registry-relative and is passed through.
    """
    if "#" in image:
        return image
    head, slash, rest = image.partition("/")
    if slash and ("." in head or ":" in head or head == "localhost"):
        return f"{head}#{rest}"
    return image


def sqsh_path(image: str, ctx: dict[str, Any]) -> str:
    """Deterministic Lustre path for an image's converted squashfs.

    Pure, so the caching decision is testable without a cluster.
    """
    slug = image.replace("://", "_").replace("/", "_").replace(":", "_").replace("#", "_")
    return f"{ctx['sqsh_dir'].rstrip('/')}/{slug}.sqsh"


def prepare(bundle: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Reuse the cached .sqsh; convert only when it is missing or corrupt.

    Conversion is expensive and one-time-per-image, so the default path does
    nothing. Two things make "already there" untrustworthy on its own:

    * A conversion killed by a wall-time cap leaves a TRUNCATED file that
      `test -e` happily accepts. This skill's SKILL.md says the SQSH "is
      validated by `hsqs` magic" -- that guard was documented but implemented
      nowhere in the repo, so this is it: read the 4-byte squashfs magic.
    * The wall limit must exceed the conversion time, and the trap is the
      partition DEFAULT rather than its maximum. On CS-OCI-ORD every partition
      has DefaultTime=00:31:00 while `cpu` allows a full day, so a conversion
      submitted without an explicit `-t` is capped at 31 minutes no matter which
      partition it lands on -- changing partition alone fixes nothing. Hence the
      explicit `-t` below; `ctx["conversion_minutes"]` is a ceiling, not an
      estimate, and costs nothing when the work finishes sooner.

    Conversion is submitted as its own recorded job when `ctx["record_child"]`
    is supplied, so a 2-hour queue wait is observable instead of a silent hang.
    """
    image = bundle["image"]
    if image.endswith(".sqsh"):
        return {"image": image, "notes": ["image is already a squashfs path"]}
    target = sqsh_path(image, ctx)
    login = ctx["login"]

    magic = subprocess.run(
        ["ssh", login, f"head -c4 {shlex.quote(target)} 2>/dev/null || true"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if magic == SQSH_MAGIC:
        return {"image": target, "notes": ["reused cached sqsh"]}

    if ctx.get("airgap"):
        raise ValueError(
            f"{target} is missing or truncated and air-gap forbids the registry "
            "import that would rebuild it; pre-stage the .sqsh on Lustre"
        )

    note = "converted" if not magic else "reconverted (corrupt or truncated sqsh)"
    partition = ctx.get("conversion_partition", "cpu")
    minutes = int(ctx.get("conversion_minutes", 120))
    scheduling = " ".join(shlex.quote(f) for f in scheduling_srun_flags(ctx))
    convert = (
        f"srun -n1 -p {shlex.quote(partition)} -t {minutes} "
        f"{scheduling} enroot import -o {shlex.quote(target)} "
        f"docker://{enroot_uri(image)}"
    ).replace("  ", " ")
    result = subprocess.run(
        ["ssh", login, convert], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.strip()
        hint = ""
        if not scheduling_srun_flags(ctx):
            # The common first-run failure on a cluster that requires an
            # account: the message names enroot, so it reads as a broken
            # conversion rather than a missing scheduler setting.
            hint = (
                " — no scheduling identity was supplied; if this cluster "
                "requires one, pass it in ctx (account / qos / reservation) so "
                "it reaches the conversion allocation as well as the job"
            )
        raise ValueError(f"enroot import failed{hint}: {detail}")

    verify = subprocess.run(
        ["ssh", login, f"head -c4 {shlex.quote(target)} 2>/dev/null || true"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if verify != SQSH_MAGIC:
        # Never fall back to the registry URI: that puts the pull back inside
        # the GPU allocation, exactly when something is already wrong.
        raise ValueError(
            f"conversion produced no valid squashfs at {target} "
            f"(magic {verify!r}); raise --conversion-minutes or use a longer partition"
        )
    return {"image": target, "notes": [note]}


def _limit_minutes(time_limit: str) -> int:
    """SLURM wall limit -> minutes. Accepts D-HH:MM:SS and HH:MM:SS."""
    days, _, clock = time_limit.partition("-")
    if not clock:
        days, clock = "0", time_limit
    parts = [int(x) for x in clock.split(":")]
    while len(parts) < 3:
        parts.append(0)
    hours, minutes, seconds = parts
    return int(days) * 1440 + hours * 60 + minutes + (1 if seconds else 0)


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
    time_limit = str(ctx.get("time_limit", "04:00:00"))
    # `timeout` must fire before SLURM's own --time or the job dies in TIMEOUT,
    # which --requeue does not cover. The SDK used 3.8h against a 4h limit; 95%
    # generalises that to any wall limit.
    timeout_minutes = int(ctx.get("timeout_minutes") or max(1, int(_limit_minutes(time_limit) * 0.95)))
    if timeout_minutes >= _limit_minutes(time_limit):
        raise ValueError(
            f"timeout_minutes={timeout_minutes} is not under the wall limit "
            f"{time_limit}; the job would die in TIMEOUT instead of requeueing"
        )
    substitutions = {
        "JOB_NAME": job_id,
        "NUM_GPUS": str(int(shape["gpus"])),
        "CPUS_PER_TASK": str(ctx.get("cpus_per_task", 8)),
        "TIME": time_limit,
        "TIMEOUT_MINUTES": str(timeout_minutes),
        "RESULTS_DIR": results_dir,
        # Empty disables auto-resume. The key is network-specific, so the
        # producer supplies it; a bundle that cannot resume simply omits it.
        "RESUME_KEY": str(ctx.get("resume_key", "")),
        "LOG_DIR": f"{job_dir}/logs",
        "IMAGE": bundle["image"],
        "CONTAINER_MOUNTS": _mounts(bundle, results_dir),
        "COMMAND": command,
        # Derived from the same ctx keys as the conversion flags above, so a
        # cluster's scheduling identity is stated once and reaches every
        # allocation. `sbatch_extra` remains for anything else.
        "SBATCH_EXTRA": "\n".join(
            filter(None, [scheduling_sbatch_directives(ctx),
                          str(ctx.get("sbatch_extra", ""))])
        ),
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
