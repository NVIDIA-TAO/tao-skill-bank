#!/usr/bin/env python3
"""Minimal TAO MCP server — runs TAO training containers on the host GPU.

The NemoClaw sandbox agent calls this over MCP (Streamable HTTP); the host runs
the container. The agent never touches Docker, the GPU, or NGC credentials.

Security boundary (this server IS the boundary):
  - Only images under nvcr.io/ (the NVIDIA NGC registry) may run. This admits
    TAO images plus QA/staging and data-generation images from other NGC orgs;
    it still refuses arbitrary registries (Docker Hub, private, etc.).
  - Data and results are confined to subpaths of a fixed --workspace-root that
    the agent cannot change or escape (no absolute paths, no `..`).
  - Reachability is gated by the OpenShell egress policy: the sandbox reaches
    this server only through the host bridge on the one allowed port. A bearer
    token is therefore optional — set TAO_MCP_TOKEN to require one, or leave it
    unset for the host-local bridge topology (the VSS pattern).

Run (plain HTTP; the sandbox reaches it via host.openshell.internal):
  uv run --with mcp --with uvicorn python server.py \
    --workspace-root /home/tao-dev/tao-workspace --host 0.0.0.0 --port 9901
"""

import argparse
import errno
import json
import os
import re
import stat
import subprocess
import uuid
from pathlib import Path, PurePosixPath

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

if __package__:
    from .docker_runtime import (
        build_docker_run_args,
        current_host_identity,
        ensure_workspace_directory,
        JOB_NAME_PREFIX,
        MANAGED_LABEL,
        prepare_isolated_results,
        remove_managed_results_tree,
        RESULTS_DEVICE_LABEL,
        RESULTS_INODE_LABEL,
        RESULTS_PATH_LABEL,
        RESULTS_TOKEN_LABEL,
        validate_managed_results_source,
        validate_volume_subpath,
        WORKSPACE_VOLUME_LABEL,
        workspace_volume_name,
    )
else:  # ``python server.py`` execution path
    from docker_runtime import (
        build_docker_run_args,
        current_host_identity,
        ensure_workspace_directory,
        JOB_NAME_PREFIX,
        MANAGED_LABEL,
        prepare_isolated_results,
        remove_managed_results_tree,
        RESULTS_DEVICE_LABEL,
        RESULTS_INODE_LABEL,
        RESULTS_PATH_LABEL,
        RESULTS_TOKEN_LABEL,
        validate_managed_results_source,
        validate_volume_subpath,
        WORKSPACE_VOLUME_LABEL,
        workspace_volume_name,
    )

NGC_IMAGE_PREFIX = "nvcr.io/"
_JOB_REFERENCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")

# tao_exec (CPU shell) constants. The shell mounts the whole workspace volume at
# /workspace and runs as the host user, so its writable HOME/cache tree lives
# under the workspace (owned by that user) rather than at an image-owned /root.
SHELL_CONTAINER_HOME = "/workspace/.tao-shell/home"
# Secrets forwarded into the shell by NAME only (docker reads the value from the
# bridge's own environment), so staging tools can reach gated HF / NGC artifacts
# without the value ever appearing on the docker command line.
_SHELL_PASSTHROUGH_ENV = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "NGC_API_KEY",
    "NGC_CLI_API_KEY",
)

# The sandbox reaches this server as host.openshell.internal (the OpenShell
# bridge). The MCP SDK's DNS-rebinding guard validates the Host header, so that
# name must be allowlisted or every request fails with "Invalid Host header".
mcp = FastMCP(
    "tao",
    transport_security=TransportSecuritySettings(
        allowed_hosts=["host.openshell.internal", "host.openshell.internal:9901"],
        allowed_origins=["http://host.openshell.internal:9901"],
    ),
)
WORKSPACE_ROOT: Path  # set from the required --workspace-root flag at startup

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


def _workspace_parts(subpath: str) -> tuple[str, ...]:
    if subpath in {"", "."}:
        return ()
    path = PurePosixPath(subpath)
    if path.is_absolute() or ".." in path.parts or "\x00" in subpath:
        raise ValueError(f"path escapes the workspace root: {subpath}")
    return path.parts


def _open_child_dir(parent_fd: int, name: str, *, create: bool) -> int:
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    try:
        return os.open(
            name,
            os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                f"workspace paths cannot traverse symlinks: {name}"
            ) from exc
        raise


def _open_workspace_object(subpath: str) -> int:
    """Open an existing object beneath WORKSPACE_ROOT without link races."""
    parts = _workspace_parts(subpath)
    current_fd = os.open(
        WORKSPACE_ROOT.resolve(), os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC
    )
    try:
        for part in parts[:-1]:
            child_fd = _open_child_dir(current_fd, part, create=False)
            os.close(current_fd)
            current_fd = child_fd
        if not parts:
            return current_fd
        object_fd = os.open(
            parts[-1], os.O_RDONLY | _NOFOLLOW | _CLOEXEC, dir_fd=current_fd
        )
        os.close(current_fd)
        return object_fd
    except OSError as exc:
        os.close(current_fd)
        if exc.errno == errno.ELOOP:
            raise ValueError(
                f"workspace paths cannot traverse symlinks: {subpath}"
            ) from exc
        raise
    except BaseException:
        os.close(current_fd)
        raise


def _open_workspace_parent(subpath: str) -> tuple[int, str]:
    """Open/create a writable object's parent with no-follow directory walks."""
    parts = _workspace_parts(subpath)
    if not parts:
        raise ValueError("cannot write the workspace root")
    current_fd = os.open(
        WORKSPACE_ROOT.resolve(), os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC
    )
    try:
        for part in parts[:-1]:
            child_fd = _open_child_dir(current_fd, part, create=True)
            os.close(current_fd)
            current_fd = child_fd
        return current_fd, parts[-1]
    except BaseException:
        os.close(current_fd)
        raise


def _resolve_under_root(subdir: str) -> str:
    """Resolve a caller-supplied subpath under the workspace root, or raise."""
    if subdir in ("", "."):
        target = WORKSPACE_ROOT
    else:
        if Path(subdir).is_absolute():
            raise ValueError(f"path must be relative to the workspace root: {subdir}")
        target = (WORKSPACE_ROOT / subdir).resolve()
    root = WORKSPACE_ROOT.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes the workspace root: {subdir}")
    return str(target)


def _docker(*args: str, timeout=120) -> subprocess.CompletedProcess:
    # timeout bounds the docker CLI call, not the workload: `run -d` returns once
    # the container starts, so training is unaffected. 120s suits every
    # control-plane call; pass timeout=None for `pull` (a cold image is GBs).
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )


def _validate_job_reference(job_id: str) -> str:
    if not _JOB_REFERENCE_RE.fullmatch(job_id):
        raise ValueError("invalid managed TAO job reference")
    return job_id


def _workspace_relative(path: str | Path) -> str:
    """Return one resolved workspace path in Docker volume-subpath form."""
    workspace = WORKSPACE_ROOT.resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"path escapes the workspace root: {path}") from exc
    return validate_volume_subpath(relative.as_posix() or ".")


def _inspect_workspace_volume(volume_name: str) -> dict:
    """Verify the fixed-root local volume used for traversal-safe subpaths."""
    expected_name = workspace_volume_name(WORKSPACE_ROOT)
    if volume_name != expected_name:
        raise ValueError("managed job references an unexpected workspace volume")
    proc = _docker("volume", "inspect", volume_name)
    if proc.returncode != 0:
        raise RuntimeError(f"managed workspace volume is unavailable: {volume_name}")
    try:
        info = json.loads(proc.stdout)[0]
        labels = info.get("Labels") or {}
        options = info.get("Options") or {}
    except (IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid managed workspace volume metadata") from exc
    expected_options = {
        "device": str(WORKSPACE_ROOT.resolve()),
        "o": "bind",
        "type": "none",
    }
    if (
        info.get("Name") != expected_name
        or info.get("Driver") != "local"
        or labels.get(MANAGED_LABEL) != "true"
        or options != expected_options
    ):
        raise ValueError("refusing an untrusted managed workspace volume")
    return info


def _ensure_workspace_volume() -> str:
    """Create or verify the one fixed-root volume used by this bridge."""
    volume_name = workspace_volume_name(WORKSPACE_ROOT)
    create_args = (
        "volume",
        "create",
        "--driver",
        "local",
        "--label",
        f"{MANAGED_LABEL}=true",
        "--opt",
        "type=none",
        "--opt",
        "o=bind",
        "--opt",
        f"device={WORKSPACE_ROOT.resolve()}",
        volume_name,
    )
    try:
        created = _docker(*create_args)
    except subprocess.TimeoutExpired:
        created = None
    try:
        _inspect_workspace_volume(volume_name)
    except (RuntimeError, ValueError) as exc:
        detail = "timed out" if created is None else created.stderr.strip()
        raise RuntimeError(
            f"could not create a trusted workspace volume ({detail})"
        ) from exc
    if created is not None and created.returncode != 0:
        raise RuntimeError(f"docker volume create failed: {created.stderr.strip()}")
    return volume_name


@mcp.tool()
def tao_ls(subdir: str = "") -> dict:
    """List the host workspace so the agent can find datasets and results.

    subdir: path relative to the workspace root (the container's /data). Returns
    entries with name, type (file/dir), and size in bytes. Read-only; confined
    to the workspace root.
    """
    try:
        target_fd = _open_workspace_object(subdir)
    except FileNotFoundError as exc:
        raise ValueError(f"path does not exist under workspace: {subdir}") from exc
    try:
        metadata = os.fstat(target_fd)
        if stat.S_ISREG(metadata.st_mode):
            return {
                "entries": [
                    {
                        "name": PurePosixPath(subdir).name,
                        "type": "file",
                        "size": metadata.st_size,
                    }
                ]
            }
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"not a file or directory under workspace: {subdir}")
        entries = []
        with os.scandir(target_fd) as directory:
            for entry in sorted(directory, key=lambda item: item.name):
                entry_metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(entry_metadata.st_mode):
                    entry_type = "dir"
                    size = None
                elif stat.S_ISREG(entry_metadata.st_mode):
                    entry_type = "file"
                    size = entry_metadata.st_size
                else:
                    entry_type = "symlink" if entry.is_symlink() else "other"
                    size = None
                entries.append(
                    {"name": entry.name, "type": entry_type, "size": size}
                )
        return {"path": subdir or ".", "entries": entries}
    finally:
        os.close(target_fd)


@mcp.tool()
def tao_read(subpath: str, max_bytes: int = 65536) -> dict:
    """Read a small text file from the host workspace (e.g. an annotations or
    spec file) so the agent can inspect data and fill in a training spec.

    subpath: path relative to the workspace root. Returns up to max_bytes of
    UTF-8 text (capped at 1 MiB). Read-only; confined to the workspace root.
    """
    max_bytes = max(1, min(max_bytes, 1_048_576))
    try:
        target_fd = _open_workspace_object(subpath)
    except FileNotFoundError as exc:
        raise ValueError(f"not a file under workspace: {subpath}") from exc
    try:
        metadata = os.fstat(target_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"not a file under workspace: {subpath}")
        data = os.read(target_fd, max_bytes)
        return {
            "subpath": subpath,
            "size": metadata.st_size,
            "truncated": metadata.st_size > max_bytes,
            "text": data.decode("utf-8", errors="replace"),
        }
    finally:
        os.close(target_fd)


@mcp.tool()
def tao_write(subpath: str, content: str) -> dict:
    """Write a text file into the host workspace (e.g. an experiment spec the
    container will read at /data/... or a state file). Creates parent dirs.
    Confined to the workspace root; refuses to overwrite a directory.

    subpath: path relative to the workspace root.
    content: UTF-8 text to write (capped at 4 MiB).
    """
    if len(content.encode("utf-8")) > 4_194_304:
        raise ValueError("content exceeds 4 MiB")
    encoded = content.encode("utf-8")
    parent_fd, name = _open_workspace_parent(subpath)
    try:
        try:
            target_fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NOFOLLOW | _CLOEXEC,
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            if exc.errno in {errno.EISDIR, errno.ELOOP}:
                raise ValueError(f"refusing unsafe write path: {subpath}") from exc
            raise
        try:
            with os.fdopen(target_fd, "wb", closefd=False) as target:
                target.write(encoded)
        finally:
            os.close(target_fd)
    finally:
        os.close(parent_fd)
    return {"subpath": subpath, "bytes_written": len(encoded)}


def _resolve_shell_image(image: str) -> str:
    """Resolve the CPU-shell image: explicit arg, else $TAO_SHELL_IMAGE."""
    resolved = (image or os.environ.get("TAO_SHELL_IMAGE", "")).strip()
    if not resolved:
        raise ValueError(
            "no shell image available: pass image=... or set TAO_SHELL_IMAGE to "
            "the TAO pyt image (the images.tao_toolkit.pyt entry in the skill "
            "bank versions.yaml)"
        )
    if not resolved.startswith(NGC_IMAGE_PREFIX):
        raise ValueError(f"image must start with {NGC_IMAGE_PREFIX}")
    return resolved


def _ensure_shell_home() -> None:
    """Create a host-owned, writable HOME/cache tree for the CPU shell.

    Made with the same no-follow directory walk tao_run uses, so a symlink
    planted in the workspace cannot redirect the shell's HOME outside the root.
    """
    ensure_workspace_directory(
        str(WORKSPACE_ROOT / ".tao-shell" / "home" / ".cache" / "huggingface"),
        WORKSPACE_ROOT,
    )


@mcp.tool()
def tao_exec(
    command: str,
    workdir: str = "/workspace",
    timeout: int = 600,
    image: str = "",
) -> dict:
    """Run a shell command in a CPU-only container over the host workspace.

    This is the agent's shell + filesystem for everything that is NOT GPU
    compute: inspecting data (ls/cat/head), making directories, copying/moving
    files (binaries included), unpacking archives, authoring specs, and staging
    models/datasets (huggingface_hub / ngcsdk / curl). The whole host workspace
    is mounted at /workspace and the command runs as the MCP server's host
    UID:GID, so any file it creates is owned by that user (never root) — the
    same ownership invariant as tao_run, and removable later without sudo. No
    GPU is attached; heavy compute is a tao_run job that shares this workspace.

    command: a bash command line (run via `bash -c`, no TAO entrypoint).
    workdir: working directory inside the container (default /workspace); must
      be under /workspace and must already exist.
    timeout: seconds allowed for the command (default 600). Background long
      downloads (`nohup ... > log 2>&1 &`) or raise this for big transfers.
    image: TAO pyt image to run as the shell; defaults to $TAO_SHELL_IMAGE.
      Must already be on the host (call tao_pull first).
    Returns {stdout, stderr, exit_code}.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if not (1 <= timeout <= 86400):
        raise ValueError("timeout must be between 1 and 86400 seconds")
    resolved_image = _resolve_shell_image(image)

    identity = current_host_identity()
    if identity.uid == 0:
        raise PermissionError(
            "refusing to run a writable shell as root; run the MCP bridge as "
            "the submitting host user"
        )
    if _docker("image", "inspect", resolved_image).returncode != 0:
        raise RuntimeError(
            f"shell image not present on host: {resolved_image}. Call "
            f"tao_pull({resolved_image!r}) first."
        )

    workdir = workdir or "/workspace"
    workdir_path = PurePosixPath(workdir)
    if not workdir_path.is_absolute():
        workdir_path = PurePosixPath("/workspace") / workdir_path
    workspace_mount = PurePosixPath("/workspace")
    if workdir_path != workspace_mount and workspace_mount not in workdir_path.parents:
        raise ValueError("workdir must be under /workspace")

    workspace_volume = _ensure_workspace_volume()
    _ensure_shell_home()

    args = [
        "run",
        "--rm",
        "--security-opt",
        "no-new-privileges:true",
        "--cap-drop",
        "ALL",
        "--user",
        f"{identity.uid}:{identity.gid}",
    ]
    for group_id in identity.supplementary_gids:
        args.extend(("--group-add", str(group_id)))
    shell_env = {
        "HOME": SHELL_CONTAINER_HOME,
        "USER": identity.username,
        "LOGNAME": identity.username,
        "XDG_CACHE_HOME": f"{SHELL_CONTAINER_HOME}/.cache",
        "HF_HOME": f"{SHELL_CONTAINER_HOME}/.cache/huggingface",
    }
    for name, value in shell_env.items():
        args.extend(("-e", f"{name}={value}"))
    for name in _SHELL_PASSTHROUGH_ENV:
        if os.environ.get(name):
            args.extend(("-e", name))  # pass-through: value stays off the cmdline
    args.extend(
        (
            "--mount",
            f"type=volume,src={workspace_volume},dst=/workspace",
            "-w",
            str(workdir_path),
            "--entrypoint",
            "bash",
            resolved_image,
            "-c",
            command,
        )
    )
    try:
        proc = _docker(*args, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"tao_exec timed out after {timeout}s; raise timeout or background "
            "the command (nohup ... > log 2>&1 &)"
        ) from exc
    return {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "exit_code": proc.returncode,
    }


@mcp.tool()
def tao_pull(image: str) -> dict:
    """Pull an NGC image to the host cache so a later tao_run starts instantly.

    This is the one intentionally-slow tool: a cold TAO image is several GB and
    can take minutes. Call it BEFORE tao_run — tao_run never pulls, it fails
    fast if the image is absent. Idempotent: returns at once if already present.

    image: full nvcr.io/... image reference.
    """
    if not image.startswith(NGC_IMAGE_PREFIX):
        raise ValueError(f"image must start with {NGC_IMAGE_PREFIX}")
    if _docker("image", "inspect", image).returncode == 0:
        return {"image": image, "pulled": False, "status": "already present"}
    proc = _docker("pull", image, timeout=None)  # no cap — pulls can take minutes
    if proc.returncode != 0:
        raise RuntimeError(f"docker pull failed: {proc.stderr.strip()}")
    return {"image": image, "pulled": True, "status": "pulled"}


@mcp.tool()
def tao_run(
    image: str,
    command: list[str],
    data_subdir: str = "",
    results_subdir: str = "results",
    gpus: int = 1,
    shm_size: str = "8g",
) -> dict:
    """Launch a TAO training container on the host GPU. Returns a job_id.

    image: full nvcr.io/... image reference (any NVIDIA NGC image — TAO,
      QA/staging, or data-generation).
    command: container command as an argument list (no shell).
    data_subdir/results_subdir: paths relative to the host workspace root.
      data_subdir is mounted at /data. A unique child of results_subdir is
      mounted at /results, so cleanup of one interrupted experiment cannot
      delete another run's checkpoints. The returned results_subdir is the
      exact host-workspace path for this job. /data remains read-write because
      some TAO workflows stage pretrained backbones beside the dataset.
    gpus: number of GPUs to attach.
    shm_size: /dev/shm size (e.g. "8g", "16g"). TAO PyTorch DataLoaders need a
      large shared-memory segment; the docker default (64m) causes "Bus error"
      / "DataLoader worker exited". Raise for many workers or multi-GPU.

    The container runs as the MCP server's host UID:GID. HOME and common cache
    directories are prepared below /results/.tao-runtime so bind-mounted output
    remains writable and removable by the same host user.
    """
    if not image.startswith(NGC_IMAGE_PREFIX):
        raise ValueError(f"image must start with {NGC_IMAGE_PREFIX}")
    if not isinstance(command, list) or not all(isinstance(a, str) for a in command):
        raise ValueError("command must be a list of strings")
    if not (1 <= gpus <= 8):
        raise ValueError("gpus must be between 1 and 8")
    if not re.fullmatch(r"\d+[bkmg]?", shm_size, re.IGNORECASE):
        raise ValueError("shm_size must look like '8g', '16g', '512m'")

    identity = current_host_identity()
    if identity.uid == 0:
        raise PermissionError(
            "refusing to launch a writable TAO job as root; run the MCP "
            "bridge as the submitting host user"
        )
    # Never let ``docker run`` turn a cold multi-GB image pull into an
    # ambiguous launch. Preflight warms the cache explicitly with tao_pull.
    if _docker("image", "inspect", image).returncode != 0:
        raise RuntimeError(
            f"image not present on host: {image}. Call tao_pull({image!r}) first "
            "(a cold TAO image is several GB and would otherwise block the launch)."
        )

    workspace_volume = _ensure_workspace_volume()
    data = _resolve_under_root(data_subdir)
    results_base = _resolve_under_root(results_subdir)
    # Create both writable roots with no-follow directory walks as the bridge
    # user. Docker's fixed-root volume then resolves their subpaths beneath the
    # workspace without accepting a swapped host symlink.
    data = str(ensure_workspace_directory(data, WORKSPACE_ROOT))
    results_base = str(ensure_workspace_directory(results_base, WORKSPACE_ROOT))
    data_volume_subpath = _workspace_relative(data)
    _workspace_relative(results_base)  # validate mount syntax before allocation
    token = uuid.uuid4().hex
    prepared_results = prepare_isolated_results(
        results_base, WORKSPACE_ROOT, identity, token
    )
    results = prepared_results.path
    results_volume_subpath = _workspace_relative(results)

    try:
        proc = _docker(
            *build_docker_run_args(
                image=image,
                command=command,
                workspace_volume=workspace_volume,
                data_subpath=data_volume_subpath,
                results_subpath=results_volume_subpath,
                results_device=prepared_results.device,
                results_inode=prepared_results.inode,
                gpus=gpus,
                shm_size=shm_size,
                identity=identity,
                job_token=token,
            )
        )
    except subprocess.TimeoutExpired as exc:
        # The Docker daemon may have accepted the create even though its client
        # timed out. The deterministic managed name lets the agent reconcile or
        # stop that job rather than launching a duplicate writer.
        job_name = f"{JOB_NAME_PREFIX}{token}"
        recovered = _reconcile_managed_launch(job_name, token, results)
        if recovered is not None:
            return _reconciled_launch_result(recovered, results)
        raise RuntimeError(
            "docker run response timed out; reconcile the possibly active job "
            f"as {job_name} before retrying; its results are at "
            f"{results.relative_to(WORKSPACE_ROOT)}"
        ) from exc
    if proc.returncode != 0:
        job_name = f"{JOB_NAME_PREFIX}{token}"
        recovered = _reconcile_managed_launch(job_name, token, results)
        if recovered is not None:
            return _reconciled_launch_result(recovered, results)
        # A completed client error can still follow an accepted daemon request
        # (for example, a daemon restart after create). Keep the isolated tree
        # and deterministic name; deleting now could race a late writer.
        raise RuntimeError(
            "docker run response was unsuccessful and launch state is "
            f"ambiguous ({proc.stderr.strip()}); reconcile {job_name} before "
            f"retrying; its results are at {results.relative_to(WORKSPACE_ROOT)}"
        )
    return {
        "job_id": proc.stdout.strip(),
        "results_subdir": str(results.relative_to(WORKSPACE_ROOT)),
    }


def _validate_managed_container_inspect(
    proc: subprocess.CompletedProcess, job_id: str
) -> dict:
    """Validate one Docker inspect response as belonging to this bridge."""
    try:
        payload = json.loads(proc.stdout)
        info = payload[0]
        image = info["Config"]["Image"]
        labels = info["Config"].get("Labels") or {}
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Docker metadata for job_id: {job_id}") from exc
    if not image.startswith(NGC_IMAGE_PREFIX):
        raise ValueError(f"refusing: {job_id} is not an nvcr.io container")
    if labels.get(MANAGED_LABEL) != "true":
        raise ValueError(f"refusing: {job_id} was not launched by this TAO bridge")
    if labels.get(WORKSPACE_VOLUME_LABEL) != workspace_volume_name(WORKSPACE_ROOT):
        raise ValueError(f"refusing: {job_id} belongs to another TAO workspace")
    return info


def _inspect_managed_container(job_id: str) -> dict:
    """Return Docker inspect data for a container created by this bridge."""
    _validate_job_reference(job_id)
    proc = _docker("inspect", job_id)
    if proc.returncode != 0:
        raise RuntimeError(f"unknown job_id: {job_id}")
    return _validate_managed_container_inspect(proc, job_id)


def _managed_results_path(info: dict) -> Path:
    """Resolve and validate the isolated /results workspace-volume subpath."""
    labels = info["Config"].get("Labels") or {}
    token = labels.get(RESULTS_TOKEN_LABEL, "")
    volume_name = labels.get(WORKSPACE_VOLUME_LABEL, "")
    results_subpath = validate_volume_subpath(labels.get(RESULTS_PATH_LABEL, ""))
    try:
        expected_device = int(labels[RESULTS_DEVICE_LABEL])
        expected_inode = int(labels[RESULTS_INODE_LABEL])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("managed TAO job has invalid result identity labels") from exc
    _inspect_workspace_volume(volume_name)
    mounts = [
        mount
        for mount in info.get("Mounts", [])
        if mount.get("Destination") == "/results"
    ]
    if len(mounts) != 1:
        raise ValueError("managed TAO job must have exactly one /results mount")
    mount = mounts[0]
    if (
        mount.get("Type") != "volume"
        or mount.get("Name") != volume_name
        or mount.get("RW") is not True
    ):
        raise ValueError(
            "managed TAO /results mount must use its trusted writable volume"
        )
    results = WORKSPACE_ROOT.resolve() / results_subpath
    return validate_managed_results_source(
        results,
        WORKSPACE_ROOT,
        token,
        expected_device=expected_device,
        expected_inode=expected_inode,
    )


def _reconcile_managed_launch(job_name: str, token: str, results: Path) -> str | None:
    """Return a verified late-created job ID, or None while still ambiguous."""
    try:
        proc = _docker("inspect", job_name)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    info = _validate_managed_container_inspect(proc, job_name)
    labels = info["Config"].get("Labels") or {}
    if labels.get(RESULTS_TOKEN_LABEL) != token:
        raise ValueError("recovered container token does not match this launch")
    if _managed_results_path(info) != results:
        raise ValueError("recovered container results do not match this launch")
    return str(info.get("Id") or job_name)


def _reconciled_launch_result(job_id: str, results: Path) -> dict:
    return {
        "job_id": job_id,
        "results_subdir": str(results.relative_to(WORKSPACE_ROOT)),
        "reconciled": True,
    }


@mcp.tool()
def tao_status(job_id: str) -> dict:
    """Return the state and exit code of a job started by tao_run."""
    info = _inspect_managed_container(job_id)
    state = info.get("State") or {}
    return {
        "state": state.get("Status", "unknown"),
        "exit_code": int(state.get("ExitCode", 0)),
    }


@mcp.tool()
def tao_logs(job_id: str, tail: int = 100) -> dict:
    """Return the last `tail` lines of a job's logs."""
    _inspect_managed_container(job_id)
    proc = _docker("logs", "--tail", str(tail), job_id)
    if proc.returncode != 0:
        raise RuntimeError(f"unknown job_id: {job_id}")
    return {"text": proc.stdout + proc.stderr}


@mcp.tool()
def tao_list() -> dict:
    """List every job this server launched (running or exited), newest first.

    Recovers job_ids the agent may have lost after an ambiguous launch response.
    Filters on the managed and workspace labels, so it never returns unrelated
    host containers or jobs from another TAO workspace.
    """
    volume_name = workspace_volume_name(WORKSPACE_ROOT)
    proc = _docker(
        "ps",
        "-a",
        "--filter",
        f"label={MANAGED_LABEL}=true",
        "--filter",
        f"label={WORKSPACE_VOLUME_LABEL}={volume_name}",
        "--no-trunc",
        "--format",
        "{{.ID}}\t{{.Image}}\t{{.State}}\t{{.Status}}",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker ps failed: {proc.stderr.strip()}")
    jobs = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = (line.split("\t") + ["", "", "", ""])[:4]
        jobs.append(
            {
                "job_id": parts[0],
                "image": parts[1],
                "state": parts[2],
                "status": parts[3],
            }
        )
    return {"jobs": jobs}


def _assert_ngc_container(job_id: str) -> dict:
    """Confirm a job is both NGC-hosted and created by this bridge."""
    return _inspect_managed_container(job_id)


@mcp.tool()
def tao_stop(job_id: str) -> dict:
    """Stop a running TAO job. Only nvcr.io containers may be stopped."""
    _assert_ngc_container(job_id)
    proc = _docker("stop", job_id)
    if proc.returncode != 0:
        raise RuntimeError(f"docker stop failed: {proc.stderr.strip()}")
    return {"stopped": proc.stdout.strip()}


@mcp.tool()
def tao_rm(job_id: str, force: bool = False) -> dict:
    """Remove a TAO job's container and writable layer. Bind-mounted data,
    results, caches, and checkpoints are never deleted. Only nvcr.io containers
    may be removed. Set force=True to remove one that is still running."""
    _assert_ngc_container(job_id)
    args = ["rm", job_id] if not force else ["rm", "-f", job_id]
    proc = _docker(*args)
    if proc.returncode != 0:
        raise RuntimeError(f"docker rm failed: {proc.stderr.strip()}")
    return {"removed": proc.stdout.strip()}


@mcp.tool()
def tao_cleanup_results(job_id: str) -> dict:
    """Remove a terminal job and delete only its isolated result tree.

    Use this after an interrupted or disposable experiment once any useful
    artifacts have been retained elsewhere. Running jobs are rejected. The
    bridge verifies its managed labels, fixed workspace volume, and exact
    /results directory identity before deleting checkpoints, outputs, and
    caches as the submitting host user (no sudo), then removes the container.
    The container remains available as cleanup authorization if deletion fails.
    """
    info = _assert_ngc_container(job_id)
    state = (info.get("State") or {}).get("Status", "unknown")
    if state not in {"created", "exited", "dead"}:
        raise ValueError(
            f"refusing to clean results while job is {state}; stop it first"
        )
    restart_policy = (
        ((info.get("HostConfig") or {}).get("RestartPolicy") or {}).get("Name")
    )
    if restart_policy not in {"", "no"}:
        raise ValueError("refusing to clean a job with an automatic restart policy")
    results = _managed_results_path(info)

    # Revalidate immediately before deletion. The persisted device/inode labels
    # reject a different directory swapped into this pathname, while the
    # descriptor-based remover never follows links in the tree.
    token = (info["Config"].get("Labels") or {})[RESULTS_TOKEN_LABEL]
    labels = info["Config"].get("Labels") or {}
    validate_managed_results_source(
        results,
        WORKSPACE_ROOT,
        token,
        expected_device=int(labels[RESULTS_DEVICE_LABEL]),
        expected_inode=int(labels[RESULTS_INODE_LABEL]),
    )

    try:
        remove_managed_results_tree(
            results,
            WORKSPACE_ROOT,
            token,
            expected_device=int(labels[RESULTS_DEVICE_LABEL]),
            expected_inode=int(labels[RESULTS_INODE_LABEL]),
        )
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"result cleanup failed for {results}; the terminal container "
            f"was retained so cleanup can be retried: {exc}"
        ) from exc

    removed = job_id
    try:
        proc = _docker("rm", job_id)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"results were deleted but container removal is ambiguous for {job_id}; "
            "reconcile it with tao_status/tao_rm"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"results were deleted but docker rm failed for {job_id}: "
            f"{proc.stderr.strip()}; retry tao_rm"
        )
    if proc.stdout.strip():
        removed = proc.stdout.strip()
    return {
        "removed": removed,
        "deleted_results_subdir": str(results.relative_to(WORKSPACE_ROOT)),
    }


class BearerAuth(BaseHTTPMiddleware):
    def __init__(self, app, token: str):
        super().__init__(app)
        self.expected = f"Bearer {token}"

    async def dispatch(self, request, call_next):
        if request.headers.get("authorization") != self.expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def main() -> None:
    global WORKSPACE_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace-root", required=True,
                    help="host directory that confines all data/results mounts")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--ssl-keyfile")
    ap.add_argument("--ssl-certfile")
    args = ap.parse_args()

    WORKSPACE_ROOT = Path(args.workspace_root).resolve()
    if not WORKSPACE_ROOT.is_dir():
        raise SystemExit(f"workspace root does not exist: {WORKSPACE_ROOT}")

    app = mcp.streamable_http_app()
    token = os.environ.get("TAO_MCP_TOKEN")
    if token:  # optional — enforced only when a token is provided
        app.add_middleware(BearerAuth, token=token)
    uvicorn.run(
        app, host=args.host, port=args.port,
        ssl_keyfile=args.ssl_keyfile, ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
