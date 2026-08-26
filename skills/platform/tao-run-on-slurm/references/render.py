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


def input_env(bundle: dict[str, Any]) -> dict[str, str]:
    """Declared inputs as TAO_INPUT_<SPEC_KEY>, mirroring TAO_RESULTS_ROOT.

    A bundle declares its inputs by spec_key and URI, but the path the WORKLOAD
    sees is chosen by the platform -- slurm mounts identity paths, docker and
    k8s may not -- so a command that names a path directly is guessing at a
    layout it cannot see. When it guesses wrong the input is simply absent, and
    a command that does not check will happily produce empty output and exit 0:
    a job that reports COMPLETE having read nothing.

    Outputs never had this problem because TAO_RESULTS_ROOT already exists.
    This is the same convention for the other direction.
    """
    env: dict[str, str] = {}
    for item in bundle.get("declared_inputs") or []:
        key = re.sub(r"[^A-Za-z0-9]+", "_", str(item["spec_key"])).strip("_").upper()
        if key:
            env[f"TAO_INPUT_{key}"] = str(item["uri"])
    return env


def _mounts(bundle: dict[str, Any], results_dir: str) -> str:
    """Pyxis --container-mounts: src:dst pairs, same path on both sides."""
    pairs: list[str] = []
    seen: set[tuple[str, str]] = set()
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
        # Read-only, matching the docker renderer. A declared INPUT a stage can
        # write to is a cross-platform trap: docker binds these :ro, so the same
        # bundle would fail there and silently succeed here. Writes belong under
        # results_dir. `target` overrides the in-container path.
        target = str(item.get("target") or uri)
        if (uri, target) not in seen:
            pairs.append(f"{uri}:{target}:ro")
            seen.add((uri, target))
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


# ── Scheduler discovery ─────────────────────────────────────────────────────
# Preflight used to verify only that we could REACH the cluster: ssh, enroot
# credentials, writable storage. None of that is schedulability, so a run could
# pass preflight completely and still die at submit on a partition that does
# not exist or an account the site requires. Both of this skill's real-cluster
# failures were of exactly that kind. Ask the scheduler instead of shipping one
# site's answers as defaults.

SINFO_FORMAT = "%R|%l|%L|%a|%G"


def parse_sinfo_minutes(value: str) -> int | None:
    """sinfo time limit -> minutes; None when unbounded or unparseable.

    Deliberately NOT `_limit_minutes`. That one parses a limit we ASKED for and
    is always HH:MM:SS, while sinfo reports the cluster's own dialect: it emits
    `infinite`, and it prints a bare `31:00` meaning MM:SS. Reusing the other
    parser would read that as 31 hours -- silently generous in the one place
    the number exists to be a ceiling.
    """
    text = (value or "").strip().lower()
    if not text or text in ("n/a", "none", "not_set"):
        return None
    if text.startswith(("infinite", "unlimited")):
        return None
    days = 0
    if "-" in text:
        head, _, text = text.partition("-")
        if not head.isdigit():
            return None
        days = int(head)
    fields = text.split(":")
    if not all(f.isdigit() for f in fields) or not 1 <= len(fields) <= 3:
        return None
    nums = [int(f) for f in fields]
    if len(fields) == 3:
        hours, minutes, seconds = nums
    elif len(fields) == 2:
        # A day component makes the clock HH:MM; on its own it is MM:SS.
        hours, minutes, seconds = (nums[0], nums[1], 0) if days else (0, nums[0], nums[1])
    else:
        hours, minutes, seconds = 0, nums[0], 0
    return days * 1440 + hours * 60 + minutes + (1 if seconds else 0)


def discover_scheduler(login: str) -> dict[str, dict[str, Any]]:
    """Ask the cluster which partitions exist and what limits they carry.

    Returns {name: {max_minutes, default_minutes, available, gres}}. An empty
    dict means sinfo told us nothing; callers must treat that as "unknown", not
    as "no partitions", so a transient ssh failure cannot look like a cluster
    with nothing on it.
    """
    probe = subprocess.run(
        ["ssh", login, f"sinfo -h -o {shlex.quote(SINFO_FORMAT)}"],
        capture_output=True, text=True, check=False,
    )
    partitions: dict[str, dict[str, Any]] = {}
    for line in probe.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4:
            continue
        # A default partition is starred in sinfo output; the star is display,
        # not part of the name, and passing it to -p is rejected.
        name = fields[0].strip().rstrip("*")
        if not name:
            continue
        gres = fields[4].strip() if len(fields) > 4 else ""
        partitions[name] = {
            "max_minutes": parse_sinfo_minutes(fields[1]),
            "default_minutes": parse_sinfo_minutes(fields[2]),
            "available": fields[3].strip().lower() in ("up", "avail"),
            "gres": "" if gres in ("(null)", "n/a") else gres,
        }
    return partitions


def require_partition(partitions: dict[str, dict[str, Any]], requested: str) -> None:
    """Fail before submitting, naming what the cluster actually offers.

    `-p nosuch` fails at submit with a bare "invalid partition specified" that
    does not say what IS valid, so the next step is always another round trip.
    """
    if not partitions or not requested:
        return
    missing = [p for p in requested.split(",") if p.strip().rstrip("*")
               and p.strip().rstrip("*") not in partitions]
    if missing:
        raise ValueError(
            f"partition {','.join(missing)} does not exist on this cluster; "
            f"available: {','.join(sorted(partitions))}"
        )


def choose_conversion_partition(
    partitions: dict[str, dict[str, Any]], requested: str | None = None
) -> str | None:
    """Pick where the one-time image conversion runs.

    Conversion is pure CPU work, so it should not sit in a GPU queue burning
    allocation the job has not started needing. Prefer a GPU-free partition,
    longest wall limit first, since the conversion's failure mode is being cut
    short. Returns None when discovery found nothing -- the caller then leaves
    the choice to the cluster's own default rather than inventing one.
    """
    if requested:
        require_partition(partitions, requested)
        return requested
    usable = [(n, m) for n, m in partitions.items() if m["available"]]
    if not usable:
        return None
    cpu_only = [(n, m) for n, m in usable if not m["gres"]] or usable
    # None (unbounded) sorts highest: it is the most headroom, not the least.
    return sorted(
        cpu_only,
        key=lambda nm: (nm[1]["max_minutes"] is None, nm[1]["max_minutes"] or 0, nm[0]),
        reverse=True,
    )[0][0]


def conversion_minutes(
    partitions: dict[str, dict[str, Any]], partition: str | None, ceiling: int
) -> int:
    """Clamp the requested ceiling to what the partition will actually grant.

    Asking for more than MaxTime is rejected outright, so an over-generous
    ceiling does not fail safe -- it fails at submit.
    """
    meta = partitions.get(partition or "") or {}
    cap = meta.get("max_minutes")
    return min(ceiling, cap) if cap else ceiling


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
    # A large image converts for tens of minutes with no output anywhere: the
    # stream goes back over ssh into the caller's buffer and is printed only on
    # failure, so a slow conversion and a hung one look identical while it runs.
    # Land it next to the .sqsh instead, where it can be tailed from any shell.
    log_path = f"{target[:-5]}.import.log"
    # Only now -- on the rare path that actually converts -- is discovery worth
    # an ssh round trip. The cached path stays a single `head -c4`.
    partitions = ctx.get("partitions")
    if partitions is None:
        partitions = discover_scheduler(login)
    partition = choose_conversion_partition(partitions, ctx.get("conversion_partition"))
    minutes = conversion_minutes(
        partitions, partition, int(ctx.get("conversion_minutes", 120))
    )
    # enroot unpacks layers through a temp dir before writing the .sqsh. Left
    # at its default that is the submit CWD -- here a Lustre/home path under
    # quota -- and the write fails mid-layer as `curl: (23) Failed writing
    # body`, which reads as a network fault rather than a placement one. The
    # dir must also be job-unique: a fixed name is deleted by cleanup from
    # another allocation, and enroot then fails whiteout conversion after
    # fetching every layer. Both requirements are this skill's documented ones.
    script = "\n".join([
        "set -Eeuo pipefail",
        "export TMPDIR=/tmp",
        'export ENROOT_TEMP_PATH="/tmp/enroot-tao-${SLURM_JOB_ID:-$$}"',
        'export SLURM_ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH}"',
        'mkdir -p "${ENROOT_TEMP_PATH}"',
        # Node-local scratch is not reclaimed for us; a failed import would
        # otherwise leave a partial layer tree behind on every retry.
        "trap 'rm -rf \"${ENROOT_TEMP_PATH}\"' EXIT",
        "cd /tmp",
        # Tee, do not redirect: the caller still needs the output on failure,
        # and PIPESTATUS preserves enroot's exit code past the pipe (a plain
        # pipeline would report tee's success and mask a failed import).
        f"enroot import -o {shlex.quote(target)} docker://{enroot_uri(image)}"
        f" 2>&1 | tee -a {shlex.quote(log_path)}",
        'exit "${PIPESTATUS[0]}"',
    ])
    # Built as a token list rather than an f-string: the previous form ended in
    # `.replace("  ", " ")` to tidy an empty flag slot, which would corrupt any
    # embedded script that contained a double space.
    argv = ["srun", "--chdir=/tmp", "-n1"]
    if partition:
        argv += ["-p", partition]
    # An explicit -t always, even when the partition was not chosen here: the
    # trap is the partition DEFAULT, not its maximum, so omitting -t silently
    # caps the conversion at DefaultTime no matter where it lands.
    argv += ["-t", str(minutes)]
    # enroot downloads and extracts layers in PARALLEL (ENROOT_MAX_CONNECTIONS
    # defaults to 10), so peak memory tracks concurrency, not image size. With
    # no --mem the step gets the partition's per-CPU default -- which for a
    # single task is small enough that a multi-layer image is OOM-killed at the
    # extract stage, after every layer has already been fetched. The scheduler
    # reports "Out Of Memory" against the step, but enroot's own output ends at
    # "Downloading N missing layers...", so it reads as a download failure.
    # Megabytes, matching skill_info's sqsh_conversion_memory_mb. Memory is
    # charged per CPU on some clusters, so an oversized request does not fail
    # fast -- it sits pending on a QOS group limit. The default is a measured
    # profile, not a generous guess.
    argv += ["--mem", f"{int(ctx.get('conversion_memory_mb', 7200))}M"]
    # enroot's final stage runs mksquashfs with `-processors 8`. `srun -n1`
    # allocates ONE core, so the cgroup time-slices all eight compression
    # threads onto it -- the conversion still succeeds, just far slower than
    # the tool was asking to go, and nothing reports the mismatch. Match the
    # allocation to the concurrency enroot actually requests.
    argv += ["--cpus-per-task", str(int(ctx.get("conversion_cpus_per_task", 4)))]
    argv += scheduling_srun_flags(ctx)
    argv += ["bash", "-c", script]
    convert = " ".join(shlex.quote(token) for token in argv)
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
        if "Could not process JSON input" in detail:
            # Measured on enroot 3.4.1 against Docker Hub: auth succeeds, the
            # manifest fetch returns 200 with valid JSON, and enroot still
            # fails here. The document is an OCI image index
            # (application/vnd.oci.image.index.v1+json), which older enroot
            # cannot parse -- and Docker Hub now serves it even to a client
            # whose Accept header offers ONLY the Docker manifest-list type,
            # so there is no request-side workaround. jq then exits and curl
            # reports `Failed writing body`, which is a broken pipe, not a
            # transport fault. Neither message names the real cause.
            hint += (
                " — the registry served a manifest this enroot cannot parse, "
                "most often an OCI image index. Docker Hub returns one for "
                "official images regardless of the Accept header, so this is "
                "not fixable by retrying or by changing the request. Use an "
                "image published with Docker manifest-list media types "
                "(nvcr.io does) or upgrade enroot on the compute nodes. "
                "Confirm with: curl -o /dev/null -w '%{content_type}' "
                "<registry manifest URL>"
            )
        elif "Failed writing body" in detail:
            hint += (
                " — enroot fetched but could not write the layer. Check free "
                "space on the conversion node's TMPDIR, and for a private "
                "registry that ~/.config/enroot/.credentials exists and is "
                "well-formed on the COMPUTE nodes (it is read there, not on "
                "the login node; NGC_KEY in the job env is not consulted)"
            )
        raise ValueError(
            f"enroot import failed{hint}: {detail} (full output: {log_path})"
        )

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
    return {"image": target, "notes": [note], "import_log": log_path}


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



def _toml_value(value: Any) -> str:
    """Serialize one TOML scalar or array. Booleans BEFORE ints on purpose:
    bool is a subclass of int in Python, so the order matters."""
    import json as _json

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return _json.dumps(value)          # TOML basic strings match JSON escaping
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"no TOML representation for {type(value).__name__}")


def dumps_toml(spec: dict[str, Any], _prefix: str = "") -> str:
    """Minimal TOML writer for a nested spec dict.

    Python ships tomllib to READ toml and nothing to write it, and this bank
    keeps its dependency set to pyyaml/jsonschema. Cosmos-RL specs are
    config_format=toml, so without this every cosmos train stage would fail at
    render. The supported shape is exactly what those specs contain: scalars,
    homogeneous arrays, and nested tables.

    Scalars are emitted before sub-tables at each level -- a key written after a
    [table] header would silently belong to that table instead of its parent,
    which is the classic way a hand-rolled TOML writer corrupts a config.
    """
    scalars, tables = [], []
    for key, value in spec.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            scalars.append(f"{key} = {_toml_value(value)}")
    out = "\n".join(scalars)
    for key, value in tables:
        name = f"{_prefix}{key}"
        body = dumps_toml(value, f"{name}.")
        out += f"\n\n[{name}]\n{body}" if out else f"[{name}]\n{body}"
    return out.strip() + "\n"


def config_file(bundle: dict[str, Any], job_id: str, config_root: str) -> tuple[str, str]:
    """Serialize a mode=config spec and return (path, content).

    The contract says the CONSUMER writes the spec file and substitutes its
    compute-frame path into `{config_path}`. No renderer did, so every
    mode=config bundle -- train, evaluate, inference, rca and all three mining
    stages -- reached its container with a literal `{config_path}` argument and
    failed on a file of that name.

    The file goes into the rendered `files` map, so it is placed by the same
    mechanism as every other rendered file: locally for docker and kubernetes,
    over ssh for slurm. That keeps ONE placement path rather than a per-platform
    write.
    """
    import json as _json

    fmt = str(bundle.get("config_format") or "yaml").lower()
    spec = bundle.get("spec") or {}
    if fmt == "json":
        return f"{config_root.rstrip('/')}/configs/{job_id}.json", _json.dumps(spec, indent=2)
    if fmt == "toml":
        return f"{config_root.rstrip('/')}/configs/{job_id}.toml", dumps_toml(spec)
    import yaml as _yaml

    return (f"{config_root.rstrip('/')}/configs/{job_id}.yaml",
            _yaml.safe_dump(spec, sort_keys=False))


def substitute_config_path(tokens: list[str], config_path: str) -> list[str]:
    return [t.replace("{config_path}", config_path) for t in tokens]



def place_files(files: dict[str, str], ctx: dict[str, Any]) -> None:
    """Write the rendered files on the CLUSTER, not on the launch host.

    render() returns paths on the shared filesystem and an argv that runs
    `sbatch` over ssh, so the script has to exist THERE. A generic caller that
    writes rendered files locally -- the correct behaviour for kubernetes,
    whose manifest is read by a local kubectl -- creates a stray tree on the
    launch host instead, and sbatch then fails on a path it cannot see. Where
    the launch host happens to have a writable parent of the same name, it is
    worse than an error: submission silently picks up whichever file is
    already on the cluster.

    Content travels on stdin, never argv, because a rendered sbatch body can
    carry credential material and argv is visible to anyone running `ps`.
    `umask 077` so it lands mode 600, matching the credential sidecar the
    template shreds on exit.
    """
    login = ctx["login"]
    for path, content in files.items():
        parent = str(pathlib.PurePosixPath(path).parent)
        placed = subprocess.run(
            ["ssh", login,
             f"mkdir -p {shlex.quote(parent)} && umask 077 && "
             f"cat > {shlex.quote(path)}"],
            input=content, capture_output=True, text=True, check=False,
        )
        if placed.returncode != 0:
            raise ValueError(
                f"could not write {path} on {login}: {placed.stderr.strip()}"
            )


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

    job_dir = ctx.get("job_dir") or results_dir
    tokens = [*shlex.split(bundle["command"]), *(bundle.get("args") or [])]
    extra_files: dict[str, str] = {}
    if bundle.get("mode") == "config":
        # Written under results_dir, which is bind-mounted into the container at
        # the same path, and placed on the CLUSTER by place_files() -- the same
        # route the sbatch script itself takes.
        config_path, content = config_file(bundle, job_id, results_dir)
        extra_files[config_path] = content
        tokens = substitute_config_path(tokens, config_path)

    command = " ".join(shlex.quote(token) for token in tokens)
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
        # Pyxis' spelling of docker's -w. Empty renders nothing, so a bundle
        # without a workdir keeps the image's own default.
        "CONTAINER_WORKDIR": (
            f"--container-workdir={shlex.quote(bundle['workdir'])} "
            if bundle.get("workdir") else ""
        ),
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
                          *(f"export {name}={shlex.quote(value)}"
                            for name, value in input_env(bundle).items()),
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
        "files": {script: body, **extra_files},
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


def logs(backend_ref: str, ctx: dict[str, Any], tail: int = 200) -> str:
    """Tail the sbatch --output/--error files for this job id.

    The template writes them under <job_dir>/logs/<job-name>-<jobid>/, and the
    job name is not derivable here, so match on the id and let the remote shell
    expand the glob. backend_ref is validated first precisely because it is
    NOT quoted -- an unquoted shell metacharacter would otherwise run.
    """
    if not re.fullmatch(r"[0-9][0-9_.+]*", backend_ref):
        raise ValueError(f"implausible SLURM job id {backend_ref!r}")
    pattern = f"{str(ctx['job_dir']).rstrip('/')}/logs/*-{backend_ref}/main.*"
    probe = subprocess.run(
        ["ssh", ctx["login"], f"tail -n {int(tail)} {pattern} 2>/dev/null"],
        capture_output=True, text=True, check=False,
    )
    return probe.stdout.strip()


def cancel(backend_ref: str, ctx: dict[str, Any]) -> bool:
    """scancel the allocation. Idempotent: cancelling a finished job is fine."""
    cancelled = subprocess.run(
        ["ssh", ctx["login"], f"scancel {shlex.quote(backend_ref)}"],
        capture_output=True, text=True, check=False,
    )
    return cancelled.returncode == 0
