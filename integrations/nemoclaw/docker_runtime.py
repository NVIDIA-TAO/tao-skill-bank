#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only Docker launch helpers for the NemoClaw TAO bridge.

Keeping command construction here lets ownership and path-confinement behavior
be tested without importing the MCP or ASGI dependencies used by ``server.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import shutil
import stat
from typing import Mapping, Sequence


CONTAINER_HOME = "/results/.tao-runtime/home"
JOBS_DIRNAME = ".tao-jobs"
JOB_NAME_PREFIX = "tao-nemoclaw-"
MANAGED_LABEL = "com.nvidia.tao.nemoclaw.managed"
RESULTS_TOKEN_LABEL = "com.nvidia.tao.nemoclaw.results-token"
RESULTS_PATH_LABEL = "com.nvidia.tao.nemoclaw.results-path"
RESULTS_DEVICE_LABEL = "com.nvidia.tao.nemoclaw.results-device"
RESULTS_INODE_LABEL = "com.nvidia.tao.nemoclaw.results-inode"
WORKSPACE_VOLUME_LABEL = "com.nvidia.tao.nemoclaw.workspace-volume"
WORKSPACE_VOLUME_PREFIX = "tao-nemoclaw-workspace-"
_JOB_TOKEN_RE = re.compile(r"[0-9a-f]{32}")
_VOLUME_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class HostIdentity:
    """Numeric host identity used for files written through bind mounts."""

    uid: int
    gid: int
    username: str
    supplementary_gids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PreparedResults:
    """A newly created result directory and its immutable launch identity."""

    path: Path
    device: int
    inode: int


def current_host_identity() -> HostIdentity:
    """Return the identity of the host process running the MCP server."""
    uid = os.getuid()
    gid = os.getgid()
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError:
        # USER/LOGNAME still need a stable, non-empty value in containers whose
        # passwd database does not contain the numeric host uid.
        username = f"tao-{uid}"
    privileged_gids: set[int] = set()
    try:
        docker_socket = os.stat("/var/run/docker.sock", follow_symlinks=True)
        if stat.S_ISSOCK(docker_socket.st_mode):
            privileged_gids.add(docker_socket.st_gid)
    except OSError:
        pass
    if gid in privileged_gids:
        raise PermissionError(
            "refusing to pass the Docker socket group as the workload's "
            "primary GID; run the bridge with a non-Docker primary group"
        )
    supplementary_gids = tuple(
        group_id
        for group_id in sorted({int(group) for group in os.getgroups()})
        if group_id != gid and group_id not in privileged_gids
    )
    return HostIdentity(
        uid=uid,
        gid=gid,
        username=username,
        supplementary_gids=supplementary_gids,
    )


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _open_child_directory(
    parent_fd: int, name: str, *, create: bool, mode: int = 0o700
) -> int:
    if name in {"", ".", ".."} or "/" in name:
        raise ValueError(f"invalid workspace directory component: {name}")
    if create:
        try:
            os.mkdir(name, mode=mode, dir_fd=parent_fd)
        except FileExistsError:
            pass
    try:
        return os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                f"workspace directory component cannot be a symlink: {name}"
            ) from exc
        raise


def _open_workspace_directory(
    workspace_root: str | Path,
    target_dir: str | Path,
    *,
    create: bool,
) -> tuple[int, Path]:
    """Open one directory beneath a fixed root without following symlinks."""
    workspace = Path(workspace_root).resolve()
    target = Path(target_dir)
    if not target.is_absolute():
        raise ValueError("workspace directory must be absolute")
    try:
        relative = target.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"directory escapes workspace root: {target}") from exc

    current_fd = os.open(workspace, _DIRECTORY_OPEN_FLAGS)
    try:
        for part in relative.parts:
            child_fd = _open_child_directory(current_fd, part, create=create)
            os.close(current_fd)
            current_fd = child_fd
        actual = Path(os.readlink(f"/proc/self/fd/{current_fd}")).resolve()
        if not _is_within(actual, workspace):
            raise ValueError(f"directory escapes workspace root: {actual}")
        return current_fd, actual
    except BaseException:
        os.close(current_fd)
        raise


def ensure_workspace_directory(
    target_dir: str | Path, workspace_root: str | Path
) -> Path:
    """Create/open one workspace directory using no-follow component walks."""
    directory_fd, actual = _open_workspace_directory(
        workspace_root, target_dir, create=True
    )
    os.close(directory_fd)
    return actual


def _prepare_runtime_home_fd(results_fd: int, results: Path) -> tuple[Path, int]:
    runtime_fd = _open_child_directory(results_fd, ".tao-runtime", create=True)
    try:
        home_fd = _open_child_directory(runtime_fd, "home", create=True)
    finally:
        os.close(runtime_fd)
    try:
        for relative in (
            ".cache/huggingface",
            ".cache/torch",
            ".cache/triton",
            ".cache/torchinductor",
            ".cache/matplotlib",
            ".config",
            ".local",
        ):
            current_fd = os.dup(home_fd)
            try:
                for part in PurePosixPath(relative).parts:
                    child_fd = _open_child_directory(current_fd, part, create=True)
                    os.close(current_fd)
                    current_fd = child_fd
            finally:
                os.close(current_fd)
        return results / ".tao-runtime" / "home", home_fd
    except BaseException:
        os.close(home_fd)
        raise


def validate_job_token(token: str) -> str:
    """Validate the opaque token used to bind a container to its result tree."""
    if not _JOB_TOKEN_RE.fullmatch(token):
        raise ValueError("invalid managed TAO job token")
    return token


def workspace_volume_name(workspace_root: str | Path) -> str:
    """Return a stable Docker volume name for one resolved workspace root."""
    workspace = str(Path(workspace_root).resolve())
    digest = hashlib.sha256(workspace.encode("utf-8")).hexdigest()[:24]
    return f"{WORKSPACE_VOLUME_PREFIX}{digest}"


def validate_volume_subpath(subpath: str) -> str:
    """Validate a path used with Docker's traversal-safe volume-subpath."""
    if subpath in ("", "."):
        return "."
    path = PurePosixPath(subpath)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "," in subpath
        or "\x00" in subpath
    ):
        raise ValueError(f"invalid workspace volume subpath: {subpath}")
    return path.as_posix()


def _volume_mount(volume_name: str, destination: str, subpath: str) -> str:
    if not _VOLUME_NAME_RE.fullmatch(volume_name):
        raise ValueError("invalid managed workspace volume name")
    validated = validate_volume_subpath(subpath)
    fields = [f"type=volume", f"src={volume_name}", f"dst={destination}"]
    if validated != ".":
        fields.append(f"volume-subpath={validated}")
    return ",".join(fields)


def prepare_isolated_results(
    base_results_dir: str | Path,
    workspace_root: str | Path,
    identity: HostIdentity,
    token: str,
) -> PreparedResults:
    """Create a unique result tree that can later be deleted as one unit.

    A caller-selected results directory may be shared by many experiments.  A
    per-job child prevents cleanup of an interrupted run from deleting another
    run's best checkpoint or final output.
    """
    validate_job_token(token)
    workspace = Path(workspace_root).resolve()
    base_fd, actual_base = _open_workspace_directory(
        workspace, base_results_dir, create=True
    )
    try:
        jobs_fd = _open_child_directory(base_fd, JOBS_DIRNAME, create=True)
    finally:
        os.close(base_fd)
    try:
        try:
            os.mkdir(token, mode=0o700, dir_fd=jobs_fd)
        except FileExistsError as exc:
            raise FileExistsError(
                f"managed results token already exists: {token}"
            ) from exc
        job_fd = _open_child_directory(jobs_fd, token, create=False)
        try:
            metadata = os.fstat(job_fd)
            job_results = actual_base / JOBS_DIRNAME / token
            home, home_fd = _prepare_runtime_home_fd(job_fd, job_results)
            try:
                if not os.access(
                    f"/proc/self/fd/{home_fd}", os.W_OK | os.X_OK
                ):
                    raise PermissionError(
                        f"runtime HOME is not writable by host uid:gid "
                        f"{identity.uid}:{identity.gid}: {home}"
                    )
            finally:
                os.close(home_fd)
        except BaseException:
            os.close(job_fd)
            if shutil.rmtree.avoids_symlink_attacks:
                shutil.rmtree(token, dir_fd=jobs_fd, ignore_errors=True)
            raise
        else:
            os.close(job_fd)
    finally:
        os.close(jobs_fd)
    return PreparedResults(job_results, metadata.st_dev, metadata.st_ino)


def validate_managed_results_source(
    source: str | Path,
    workspace_root: str | Path,
    token: str,
    expected_device: int | None = None,
    expected_inode: int | None = None,
) -> Path:
    """Validate a managed /results path before destructive cleanup."""
    validate_job_token(token)
    workspace = Path(workspace_root).resolve()
    source_path = Path(source)
    if not source_path.is_absolute():
        raise ValueError("managed results path must be absolute")
    if source_path.is_symlink():
        raise ValueError("managed results path cannot be a symlink")
    resolved = source_path.resolve()
    if source_path != resolved:
        raise ValueError("managed results path must use its resolved path")
    if not _is_within(resolved, workspace):
        raise ValueError("managed results path escapes the workspace")
    if resolved.name != token or resolved.parent.name != JOBS_DIRNAME:
        raise ValueError("managed results path does not match its container token")
    if resolved.exists():
        metadata = resolved.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("managed results path must be a directory")
        if expected_device is not None and metadata.st_dev != expected_device:
            raise ValueError("managed results directory identity changed")
        if expected_inode is not None and metadata.st_ino != expected_inode:
            raise ValueError("managed results directory identity changed")
    return resolved


def _delete_directory_contents(directory_fd: int) -> None:
    """Delete one opened directory tree without following pathname symlinks."""
    os.fchmod(directory_fd, 0o700)
    with os.scandir(directory_fd) as entries:
        names = [entry.name for entry in entries]
    for name in names:
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(metadata.st_mode):
            os.chmod(
                name,
                0o700,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            child_fd = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd)
            try:
                opened_metadata = os.fstat(child_fd)
                if not os.path.samestat(metadata, opened_metadata):
                    raise RuntimeError(
                        "managed results directory changed during cleanup"
                    )
                _delete_directory_contents(child_fd)
            finally:
                os.close(child_fd)
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not os.path.samestat(opened_metadata, current):
                raise RuntimeError(
                    "managed results directory changed during cleanup"
                )
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


def remove_managed_results_tree(
    source: str | Path,
    workspace_root: str | Path,
    token: str,
    expected_device: int,
    expected_inode: int,
) -> bool:
    """Delete the expected result inode using no-follow directory descriptors."""
    results = validate_managed_results_source(
        source,
        workspace_root,
        token,
        expected_device=expected_device,
        expected_inode=expected_inode,
    )
    try:
        results_metadata = os.lstat(results)
    except FileNotFoundError:
        return False
    parent_fd, _actual_parent = _open_workspace_directory(
        workspace_root, results.parent, create=False
    )
    try:
        current = os.stat(token, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != expected_device
            or current.st_ino != expected_inode
            or not os.path.samestat(results_metadata, current)
        ):
            raise ValueError("managed results directory identity changed")
        os.chmod(token, 0o700, dir_fd=parent_fd, follow_symlinks=False)
        results_fd = os.open(token, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
        try:
            opened = os.fstat(results_fd)
            if not os.path.samestat(current, opened):
                raise RuntimeError(
                    "managed results directory changed during cleanup"
                )
            _delete_directory_contents(results_fd)
        finally:
            os.close(results_fd)
        current = os.stat(token, dir_fd=parent_fd, follow_symlinks=False)
        if not os.path.samestat(opened, current):
            raise RuntimeError("managed results directory changed during cleanup")
        os.rmdir(token, dir_fd=parent_fd)
        return True
    finally:
        os.close(parent_fd)


def prepare_runtime_home(
    results_dir: str | Path,
    workspace_root: str | Path,
    identity: HostIdentity,
) -> Path:
    """Create a host-owned writable HOME/cache tree below ``results_dir``.

    Both paths are resolved again here so a pre-existing ``.tao-runtime``
    symlink cannot redirect the cache mount outside the configured workspace or
    results tree. Existing root-owned results fail early with a useful error
    instead of letting training fail later or create more inaccessible files.
    """
    results_fd, results = _open_workspace_directory(
        workspace_root, results_dir, create=False
    )
    try:
        home, home_fd = _prepare_runtime_home_fd(results_fd, results)
    except OSError as exc:
        raise PermissionError(
            f"results directory is not writable by host uid:gid "
            f"{identity.uid}:{identity.gid}: {results}. Normalize its ownership "
            "before launching another TAO container."
        ) from exc
    finally:
        os.close(results_fd)
    try:
        if not os.access(f"/proc/self/fd/{home_fd}", os.W_OK | os.X_OK):
            raise PermissionError(
                f"runtime HOME is not writable by host uid:gid "
                f"{identity.uid}:{identity.gid}: {home}. Normalize the results "
                "directory ownership before launching another TAO container."
            )
        return home
    finally:
        os.close(home_fd)


# Kept in sync with server.py's _SHELL_PASSTHROUGH_ENV: the CPU shell and GPU
# jobs must see the same credential names, or a workload that stages its own
# models succeeds under tao_exec and fails under tao_run for no visible reason.
_CREDENTIAL_PASSTHROUGH_ENV = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "NGC_API_KEY",
    "NGC_CLI_API_KEY",
    "NGC_KEY",
)


_RESERVED_MOUNT_TARGETS = ("/data", "/results", "/workspace")
# Names whose values must never reach a command line. Credentials are forwarded
# by name instead (_CREDENTIAL_PASSTHROUGH_ENV), so the value stays in the
# bridge's environment and out of `ps` and the container's own argv.
_SECRET_NAME_SEGMENTS = frozenset(
    {"KEY", "KEYS", "TOKEN", "TOKENS", "SECRET", "SECRETS",
     "PASSWORD", "PASSWD", "CREDENTIAL", "CREDENTIALS", "APIKEY"}
)
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def validate_container_path(path: str, what: str) -> str:
    """Absolute, normalized, traversal-free container-side path."""
    if not path or not path.startswith("/"):
        raise ValueError(f"{what} must be an absolute container path: {path!r}")
    if "\x00" in path or "\n" in path:
        raise ValueError(f"{what} contains an illegal character")
    normalized = PurePosixPath(path).as_posix()
    if ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"{what} must not contain '..': {path!r}")
    if normalized == "/":
        raise ValueError(f"{what} must not be the container root")
    return normalized


def validate_container_env(env: Mapping[str, str] | None) -> dict[str, str]:
    """Non-secret KEY=VALUE pairs safe to place on the docker command line."""
    validated: dict[str, str] = {}
    for name, value in (env or {}).items():
        if not _ENV_NAME_RE.fullmatch(str(name)):
            raise ValueError(f"invalid environment variable name: {name!r}")
        # Match underscore-delimited segments, not substrings: TOKENIZERS_PARALLELISM
        # is a standard HuggingFace variable and a substring test rejects it (as it
        # does MONKEY_PATCH and TURKEY). Names the bridge forwards by name are always
        # refused here — passing those by value is what this guards against.
        segments = set(str(name).upper().split("_"))
        if segments & _SECRET_NAME_SEGMENTS or str(name) in _CREDENTIAL_PASSTHROUGH_ENV:
            raise ValueError(
                f"{name} looks like a credential; pass secrets by name through the "
                "bridge environment instead of as a job argument"
            )
        text = str(value)
        if "\x00" in text or "\n" in text:
            raise ValueError(f"value for {name} contains an illegal character")
        validated[str(name)] = text
    return validated


def build_docker_run_args(
    *,
    image: str,
    command: Sequence[str],
    workspace_volume: str,
    data_subpath: str,
    results_subpath: str,
    results_device: int,
    results_inode: int,
    gpus: int,
    shm_size: str,
    identity: HostIdentity,
    job_token: str,
    workdir: str = "",
    extra_env: Mapping[str, str] | None = None,
    extra_mounts: Sequence[Mapping[str, object]] | None = None,
) -> list[str]:
    """Build arguments passed after the ``docker`` executable.

    The numeric user mapping is the ownership invariant: anything written to a
    writable workspace volume is owned by the same host user that runs this
    bridge. Explicit HOME/cache redirects avoid writes to image-owned locations
    such as ``/root`` when the image's baked-in user is overridden.
    """
    environment = {
        "HOME": CONTAINER_HOME,
        "USER": identity.username,
        "LOGNAME": identity.username,
        "XDG_CACHE_HOME": f"{CONTAINER_HOME}/.cache",
        "HF_HOME": f"{CONTAINER_HOME}/.cache/huggingface",
        "TORCH_HOME": f"{CONTAINER_HOME}/.cache/torch",
        "TRITON_CACHE_DIR": f"{CONTAINER_HOME}/.cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{CONTAINER_HOME}/.cache/torchinductor",
        "MPLCONFIGDIR": f"{CONTAINER_HOME}/.cache/matplotlib",
    }

    args = [
        "run",
        "-d",
        "--pull",
        "never",
        "--restart",
        "no",
        "--security-opt",
        "no-new-privileges:true",
        "--cap-drop",
        "ALL",
        "--gpus",
        str(gpus),
        "--user",
        f"{identity.uid}:{identity.gid}",
    ]
    validate_job_token(job_token)
    args.extend(
        (
            "--name",
            f"{JOB_NAME_PREFIX}{job_token}",
            "--label",
            f"{MANAGED_LABEL}=true",
            "--label",
            f"{RESULTS_TOKEN_LABEL}={job_token}",
            "--label",
            f"{WORKSPACE_VOLUME_LABEL}={workspace_volume}",
            "--label",
            f"{RESULTS_PATH_LABEL}={validate_volume_subpath(results_subpath)}",
            "--label",
            f"{RESULTS_DEVICE_LABEL}={int(results_device)}",
            "--label",
            f"{RESULTS_INODE_LABEL}={int(results_inode)}",
        )
    )
    for group_id in identity.supplementary_gids:
        args.extend(("--group-add", str(group_id)))
    args.extend(("--shm-size", shm_size))
    for name, value in environment.items():
        args.extend(("-e", f"{name}={value}"))
    # Registry/model credentials, forwarded BY NAME so the value never reaches
    # the command line (docker reads it from the bridge's own environment).
    # GPU jobs need these for the same reason the CPU shell does: some workloads
    # bootstrap their own model cache on first run — paidf-anomalygen's
    # `download_checkpoints.sh` fetches gated Cosmos weights from inside the
    # container. Without this the documented auto-download path is unreachable
    # through the bridge and fails as an opaque 401 / RepositoryNotFoundError.
    for name in _CREDENTIAL_PASSTHROUGH_ENV:
        if os.environ.get(name):
            args.extend(("-e", name))
    # Caller-supplied non-secret environment. Values land on the docker command
    # line and are therefore visible in `ps`, so credentials must keep going
    # through the by-name allowlist above; validate_container_env rejects any
    # name that looks like a secret.
    for name, value in validate_container_env(extra_env).items():
        args.extend(("-e", f"{name}={value}"))

    if workdir:
        args.extend(("--workdir", validate_container_path(workdir, "workdir")))

    args.extend(
        (
            "--mount",
            _volume_mount(workspace_volume, "/data", data_subpath),
            "--mount",
            _volume_mount(workspace_volume, "/results", results_subpath),
            # The whole workspace, at the same path tao_exec uses. Without it the
            # two tools disagree on what an absolute path means: assets staged by
            # tao_exec under /workspace/... dangle inside a tao_run container,
            # where the workspace is only visible as /data. Writes still belong in
            # /results, which stays the job's isolated tree.
            "--mount",
            _volume_mount(workspace_volume, "/workspace", "."),
        )
    )
    # Extra mounts exist for images with a baked-in path contract — the
    # paidf-anomalygen container reads its Cosmos base checkpoints from
    # /workspace/paidf-anomalygen/checkpoints and cannot be told otherwise, so
    # without this the stage is unrunnable through the bridge at any price.
    # Sources still come from the one fixed-root workspace volume through
    # traversal-safe subpaths; only the container-side destination is free.
    for spec in extra_mounts or ():
        target = validate_container_path(str(spec.get("target", "")), "mount target")
        if any(target == r or target.startswith(r + "/") for r in _RESERVED_MOUNT_TARGETS):
            raise ValueError(
                f"mount target {target} is managed by the bridge; choose another path"
            )
        mount = _volume_mount(workspace_volume, target, str(spec.get("subdir", "")))
        if spec.get("ro"):
            mount += ",readonly"
        args.extend(("--mount", mount))

    args.extend((image, *command))
    return args
