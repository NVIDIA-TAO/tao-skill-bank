#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministically pack, transfer, and safely promote Brev trees."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import pathlib
import posixpath
import re
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, Callable, Iterator


SCHEMA_VERSION = "1"
WORKFLOW = "tao-brev-archive-transfer"
BLOCK_SIZE = 1024 * 1024
CHUNK_MANIFEST_KIND = "archive_chunks"
CHUNK_NAME = re.compile(r"chunk-([0-9]{6})\.part")
CHUNK_TEMP_NAME = re.compile(r"\.chunk-[0-9]{6}\.part\.[A-Za-z0-9_-]+")
SAFE_REMOTE_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SAFE_REMOTE_PATH = re.compile(r"/[A-Za-z0-9._/-]+")
MAX_TRANSFER_STREAMS = 4
INDEPENDENT_SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "ControlPersist=no",
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _canonical_sha256(payload: dict[str, Any], digest_key: str) -> str:
    unsigned = {key: value for key, value in payload.items() if key != digest_key}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_root(path: pathlib.Path, name: str, *, exists: bool) -> pathlib.Path:
    expanded = path.expanduser()
    if not expanded.is_absolute() or expanded == pathlib.Path("/") or ".." in expanded.parts:
        raise ValueError(f"{name} must be an absolute, normalized, non-root path")
    resolved = expanded.resolve(strict=exists)
    if exists and (not resolved.is_dir() or resolved.is_symlink()):
        raise ValueError(f"{name} must be a non-symlink directory")
    return resolved


def _destination(path: pathlib.Path) -> pathlib.Path:
    expanded = path.expanduser().absolute()
    if expanded == pathlib.Path("/") or ".." in expanded.parts:
        raise ValueError("--destination must be an absolute, normalized, non-root path")
    if expanded.is_symlink():
        raise ValueError("--destination must not be a symlink")
    parent = expanded.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("--destination parent must be a non-symlink directory")
    return parent / expanded.name


def _entries(root: pathlib.Path) -> Iterator[tuple[pathlib.Path, pathlib.PurePosixPath, os.stat_result]]:
    pending = [root]
    while pending:
        directory = pending.pop()
        children = sorted(os.scandir(directory), key=lambda item: item.name, reverse=True)
        for child in children:
            path = pathlib.Path(child.path)
            relative = pathlib.PurePosixPath(path.relative_to(root).as_posix())
            info = child.stat(follow_symlinks=False)
            mode = info.st_mode
            if stat.S_ISLNK(mode):
                _source_link(path, root, relative)
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                if not stat.S_ISLNK(mode):
                    raise ValueError(f"source contains an unsupported file type: {relative}")
            yield path, relative, info
            if stat.S_ISDIR(mode):
                pending.append(path)


def _source_link(
    path: pathlib.Path, root: pathlib.Path, relative: pathlib.PurePosixPath
) -> str:
    target_text = os.readlink(path)
    target = pathlib.PurePosixPath(target_text)
    if not target_text or "\x00" in target_text or target.is_absolute():
        raise ValueError(f"source symlink target must be relative: {relative}")
    lexical = pathlib.Path(os.path.normpath(path.parent / target_text))
    if not lexical.is_relative_to(root):
        raise ValueError(f"source symlink target escapes the source root: {relative}")
    parent = lexical.parent
    while parent != root:
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f"source symlink target has an unsafe parent: {relative}")
        parent = parent.parent
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"source symlink is dangling or cyclic: {relative}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"source symlink must resolve to an in-root regular file: {relative}")
    return target_text


def _normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    info.mode = 0o755 if info.isdir() or info.issym() else 0o644
    return info


def pack(
    source: pathlib.Path,
    archive: pathlib.Path,
    receipt: pathlib.Path,
    *,
    max_members: int,
    max_bytes: int,
) -> dict[str, Any]:
    source = _safe_root(source, "--source", exists=True)
    archive = archive.expanduser().absolute()
    receipt = receipt.expanduser().absolute()
    if archive == pathlib.Path("/") or receipt == pathlib.Path("/"):
        raise ValueError("archive and receipt must be non-root paths")
    archive.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{archive.name}.", suffix=".tmp", dir=archive.parent)
    os.close(fd)
    temporary = pathlib.Path(raw)
    members = total_bytes = 0
    try:
        with tarfile.open(
            temporary, "w", format=tarfile.PAX_FORMAT, dereference=False
        ) as bundle:
            for path, relative, info in _entries(source):
                members += 1
                if members > max_members:
                    raise ValueError(f"source exceeds --max-members={max_members}")
                if stat.S_ISREG(info.st_mode):
                    total_bytes += info.st_size
                    if total_bytes > max_bytes:
                        raise ValueError(f"source exceeds --max-bytes={max_bytes}")
                bundle.add(path, arcname=relative.as_posix(), recursive=False, filter=_normalized)
        archive_sha256 = _sha256(temporary)
        archive_bytes = temporary.stat().st_size
        os.replace(temporary, archive)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "workflow": WORKFLOW,
            "source": str(source),
            "archive": str(archive),
            "archive_sha256": archive_sha256,
            "archive_bytes": archive_bytes,
            "members": members,
            "payload_bytes": total_bytes,
            "limits": {"max_members": max_members, "max_bytes": max_bytes},
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        _atomic_json(receipt, payload)
        return payload
    finally:
        temporary.unlink(missing_ok=True)


def digest_tree(
    source: pathlib.Path,
    allowed_root: pathlib.Path,
    receipt: pathlib.Path,
    *,
    max_members: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Hash a tree, allowing file symlinks only within an approved cache root."""
    source = _safe_root(source, "--source", exists=True)
    allowed_root = _safe_root(allowed_root, "--allowed-root", exists=True)
    if not source.is_relative_to(allowed_root):
        raise ValueError("--source must be below --allowed-root")
    digest = hashlib.sha256()
    members = total_bytes = 0
    for directory, directory_names, file_names in os.walk(source, followlinks=False):
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            candidate = pathlib.Path(directory) / name
            if candidate.is_symlink():
                raise ValueError(f"tree contains a directory symlink: {candidate.relative_to(source)}")
        for name in file_names:
            candidate = pathlib.Path(directory) / name
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(allowed_root) or not resolved.is_file():
                raise ValueError(f"tree file escapes --allowed-root: {candidate.relative_to(source)}")
            members += 1
            if members > max_members:
                raise ValueError(f"tree exceeds --max-members={max_members}")
            size = resolved.stat().st_size
            total_bytes += size
            if total_bytes > max_bytes:
                raise ValueError(f"tree exceeds --max-bytes={max_bytes}")
            relative = candidate.relative_to(source).as_posix().encode()
            content_sha256 = _sha256(resolved).encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(size.to_bytes(8, "big"))
            digest.update(content_sha256)
    if members == 0:
        raise ValueError("tree contains no regular files")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "kind": "tree_digest",
        "source": str(source),
        "allowed_root": str(allowed_root),
        "tree_sha256": digest.hexdigest(),
        "members": members,
        "payload_bytes": total_bytes,
        "limits": {"max_members": max_members, "max_bytes": max_bytes},
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    _atomic_json(receipt.expanduser().absolute(), payload)
    return payload


def _validated_chunk_manifest(
    path: pathlib.Path, *, max_chunks: int, max_chunk_bytes: int
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "workflow", "kind", "archive_sha256", "archive_bytes",
        "chunk_bytes", "chunks", "max_chunks", "completed_at", "manifest_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("chunk manifest has missing or unexpected fields")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["workflow"] != WORKFLOW
        or payload["kind"] != CHUNK_MANIFEST_KIND
    ):
        raise ValueError("chunk manifest identity is unsupported")
    if payload["manifest_sha256"] != _canonical_sha256(payload, "manifest_sha256"):
        raise ValueError("chunk manifest digest is invalid")
    if not isinstance(payload["archive_sha256"], str) or len(payload["archive_sha256"]) != 64:
        raise ValueError("chunk manifest archive digest is invalid")
    for name in ("archive_bytes", "chunk_bytes", "max_chunks"):
        if not isinstance(payload[name], int) or isinstance(payload[name], bool) or payload[name] < 1:
            raise ValueError(f"chunk manifest {name} is invalid")
    if payload["chunk_bytes"] > max_chunk_bytes or payload["max_chunks"] > max_chunks:
        raise ValueError("chunk manifest exceeds approved bounds")
    chunks = payload["chunks"]
    if not isinstance(chunks, list) or not chunks or len(chunks) > max_chunks:
        raise ValueError("chunk manifest chunk count is invalid")
    offset = 0
    names: set[str] = set()
    for index, chunk in enumerate(chunks):
        required_chunk = {"index", "name", "offset", "size", "sha256"}
        if not isinstance(chunk, dict) or set(chunk) != required_chunk:
            raise ValueError("chunk manifest entry is malformed")
        expected_name = f"chunk-{index:06d}.part"
        if chunk["index"] != index or chunk["name"] != expected_name or chunk["offset"] != offset:
            raise ValueError("chunk manifest entries are missing, duplicate, or out of order")
        if chunk["name"] in names:
            raise ValueError("chunk manifest contains duplicate chunk names")
        names.add(chunk["name"])
        if (
            not isinstance(chunk["size"], int) or isinstance(chunk["size"], bool)
            or chunk["size"] < 1 or chunk["size"] > payload["chunk_bytes"]
        ):
            raise ValueError("chunk manifest entry size is invalid")
        if index < len(chunks) - 1 and chunk["size"] != payload["chunk_bytes"]:
            raise ValueError("non-final chunk has an unexpected size")
        if not isinstance(chunk["sha256"], str) or len(chunk["sha256"]) != 64:
            raise ValueError("chunk manifest entry digest is invalid")
        offset += chunk["size"]
    if offset != payload["archive_bytes"]:
        raise ValueError("chunk manifest total does not match archive size")
    return payload


def _validate_chunk_directory(directory: pathlib.Path, manifest: dict[str, Any]) -> None:
    expected = {chunk["name"] for chunk in manifest["chunks"]}
    actual: set[str] = set()
    for entry in os.scandir(directory):
        if entry.name not in expected:
            raise ValueError(f"chunk directory contains an unexpected entry: {entry.name}")
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ValueError(f"chunk is not a regular file: {entry.name}")
        actual.add(entry.name)
    if actual != expected:
        raise ValueError("chunk directory has missing chunks")
    for chunk in manifest["chunks"]:
        path = directory / chunk["name"]
        if path.stat().st_size != chunk["size"] or _sha256(path) != chunk["sha256"]:
            raise ValueError(f"chunk size or digest mismatch: {chunk['name']}")


def _remote_chunk_root(value: str) -> str:
    pure = pathlib.PurePosixPath(value)
    if (
        not value
        or value != pure.as_posix()
        or not pure.is_absolute()
        or value == "/"
        or ".." in pure.parts
        or not SAFE_REMOTE_PATH.fullmatch(value)
    ):
        raise ValueError("--remote-chunks-dir must be an absolute, normalized remote path")
    return value.rstrip("/")


def _local_chunk_root(path: pathlib.Path) -> pathlib.Path:
    expanded = path.expanduser().absolute()
    if expanded == pathlib.Path("/") or ".." in expanded.parts or expanded.is_symlink():
        raise ValueError("--local-chunks-dir must be a non-symlink, normalized, non-root path")
    expanded.mkdir(parents=True, exist_ok=True)
    if not expanded.is_dir() or expanded.is_symlink():
        raise ValueError("--local-chunks-dir must be a non-symlink directory")
    return expanded.resolve(strict=True)


def _chunk_transfer_argv(
    *,
    rsync_executable: str,
    ssh_executable: str,
    remote_host: str,
    remote_chunks_dir: str,
    local_chunks_dir: pathlib.Path,
    chunk_name: str,
) -> list[str]:
    if not SAFE_REMOTE_HOST.fullmatch(remote_host):
        raise ValueError("--remote-host contains unsupported characters")
    if not CHUNK_NAME.fullmatch(chunk_name):
        raise ValueError("chunk name is not canonical")
    ssh_command = shlex.join([ssh_executable, *INDEPENDENT_SSH_OPTIONS])
    return [
        rsync_executable,
        "--partial",
        "--append-verify",
        "-e",
        ssh_command,
        f"{remote_host}:{remote_chunks_dir}/{chunk_name}",
        str(local_chunks_dir / chunk_name),
    ]


def transfer_chunks(
    manifest_path: pathlib.Path,
    remote_host: str,
    remote_chunks_dir: str,
    local_chunks_dir: pathlib.Path,
    output_receipt: pathlib.Path,
    *,
    max_chunks: int,
    max_chunk_bytes: int,
    streams: int,
    timeout_seconds: int = 7200,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    ssh_executable: str = "ssh",
    rsync_executable: str = "rsync",
) -> dict[str, Any]:
    """Resume manifest-owned chunks over independent SSH connections."""
    if streams < 1 or streams > MAX_TRANSFER_STREAMS:
        raise ValueError(f"--streams must be between 1 and {MAX_TRANSFER_STREAMS}")
    manifest = _validated_chunk_manifest(
        manifest_path.expanduser().resolve(strict=True),
        max_chunks=max_chunks,
        max_chunk_bytes=max_chunk_bytes,
    )
    if not SAFE_REMOTE_HOST.fullmatch(remote_host):
        raise ValueError("--remote-host contains unsupported characters")
    remote_chunks_dir = _remote_chunk_root(remote_chunks_dir)
    local_chunks_dir = _local_chunk_root(local_chunks_dir)
    output_receipt = output_receipt.expanduser().absolute()
    receipt_parent = output_receipt.parent.resolve(strict=False)
    if receipt_parent == local_chunks_dir or receipt_parent.is_relative_to(local_chunks_dir):
        raise ValueError("--output-receipt must be outside --local-chunks-dir")
    expected = {chunk["name"]: chunk for chunk in manifest["chunks"]}
    for entry in os.scandir(local_chunks_dir):
        if entry.name not in expected:
            raise ValueError(f"chunk directory contains an unexpected entry: {entry.name}")
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ValueError(f"chunk is not a regular file: {entry.name}")

    pending: list[dict[str, Any]] = []
    for chunk in manifest["chunks"]:
        target = local_chunks_dir / chunk["name"]
        if not target.exists():
            pending.append(chunk)
            continue
        size = target.stat().st_size
        if size > chunk["size"]:
            raise ValueError(f"partial chunk exceeds its declared size: {chunk['name']}")
        if size == chunk["size"]:
            if _sha256(target) != chunk["sha256"]:
                raise ValueError(f"completed chunk digest mismatch: {chunk['name']}")
            continue
        pending.append(chunk)

    def copy_one(chunk: dict[str, Any]) -> tuple[str, subprocess.CompletedProcess[Any]]:
        argv = _chunk_transfer_argv(
            rsync_executable=rsync_executable,
            ssh_executable=ssh_executable,
            remote_host=remote_host,
            remote_chunks_dir=remote_chunks_dir,
            local_chunks_dir=local_chunks_dir,
            chunk_name=chunk["name"],
        )
        completed = runner(
            argv, capture_output=True, text=True, check=False, timeout=timeout_seconds
        )
        return chunk["name"], completed

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(streams, len(pending) or 1)) as pool:
        futures = {pool.submit(copy_one, chunk): chunk["name"] for chunk in pending}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                _, completed = future.result()
            except (OSError, subprocess.SubprocessError) as exc:
                failures.append(f"{name}: transport error {type(exc).__name__}")
                continue
            if completed.returncode != 0:
                failures.append(f"{name}: rsync exit {completed.returncode}")
    if failures:
        raise RuntimeError(
            "chunk transfer failed; resumable partials retained: "
            + "; ".join(sorted(failures))
        )

    _validate_chunk_directory(local_chunks_dir, manifest)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "kind": "archive_chunk_transfer",
        "manifest_sha256": manifest["manifest_sha256"],
        "archive_sha256": manifest["archive_sha256"],
        "remote_host": remote_host,
        "remote_chunks_dir": remote_chunks_dir,
        "local_chunks_dir": str(local_chunks_dir),
        "chunks": len(manifest["chunks"]),
        "streams": min(streams, len(manifest["chunks"])),
        "timeout_seconds": timeout_seconds,
        "ssh_options": list(INDEPENDENT_SSH_OPTIONS),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    _atomic_json(output_receipt, payload)
    return payload


def split_archive(
    archive: pathlib.Path,
    pack_receipt: pathlib.Path,
    output_dir: pathlib.Path,
    manifest_path: pathlib.Path,
    *,
    chunk_bytes: int,
    max_chunks: int,
) -> dict[str, Any]:
    archive = archive.expanduser().resolve(strict=True)
    receipt = _validated_receipt(pack_receipt.expanduser().resolve(strict=True))
    if archive.stat().st_size != receipt["archive_bytes"]:
        raise ValueError("archive size does not match the pack receipt")
    count = (receipt["archive_bytes"] + chunk_bytes - 1) // chunk_bytes
    if count < 1 or count > max_chunks:
        raise ValueError(f"archive exceeds --max-chunks={max_chunks}")
    output_dir = output_dir.expanduser().absolute()
    if output_dir == pathlib.Path("/") or output_dir.is_symlink():
        raise ValueError("--output-dir must be a non-symlink, non-root path")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path.expanduser().absolute()
    if manifest_path.parent.resolve(strict=True) == output_dir.resolve(strict=True):
        raise ValueError("--manifest must be outside --output-dir")
    if manifest_path.exists():
        existing = _validated_chunk_manifest(
            manifest_path.resolve(strict=True), max_chunks=max_chunks, max_chunk_bytes=chunk_bytes
        )
        if (
            existing["archive_sha256"] != receipt["archive_sha256"]
            or existing["archive_bytes"] != receipt["archive_bytes"]
            or existing["chunk_bytes"] != chunk_bytes
        ):
            raise ValueError("existing chunk manifest does not match this archive")
        _validate_chunk_directory(output_dir, existing)
        return existing
    allowed_names = {f"chunk-{index:06d}.part" for index in range(count)}
    for entry in os.scandir(output_dir):
        if CHUNK_TEMP_NAME.fullmatch(entry.name):
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ValueError(f"interrupted chunk temporary is unsafe: {entry.name}")
            pathlib.Path(entry.path).unlink()
            continue
        if entry.name not in allowed_names:
            raise ValueError(f"chunk directory contains an unexpected entry: {entry.name}")
    chunks: list[dict[str, Any]] = []
    archive_digest = hashlib.sha256()
    with archive.open("rb") as source:
        for index in range(count):
            data = source.read(chunk_bytes)
            if not data:
                raise ValueError("archive ended before the expected chunk count")
            archive_digest.update(data)
            name = f"chunk-{index:06d}.part"
            digest = hashlib.sha256(data).hexdigest()
            target = output_dir / name
            if target.exists() or target.is_symlink():
                if target.is_symlink() or not target.is_file():
                    raise ValueError(f"existing chunk is not a regular file: {name}")
                if target.stat().st_size != len(data) or _sha256(target) != digest:
                    raise ValueError(f"existing chunk does not match the archive: {name}")
            else:
                fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=output_dir)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.chmod(temporary, 0o600)
                    os.replace(temporary, target)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            chunks.append({
                "index": index, "name": name, "offset": index * chunk_bytes,
                "size": len(data), "sha256": digest,
            })
        if source.read(1):
            raise ValueError("archive contains bytes beyond the expected size")
    if archive_digest.hexdigest() != receipt["archive_sha256"]:
        raise ValueError("archive SHA-256 does not match the pack receipt")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "kind": CHUNK_MANIFEST_KIND,
        "archive_sha256": receipt["archive_sha256"],
        "archive_bytes": receipt["archive_bytes"],
        "chunk_bytes": chunk_bytes,
        "chunks": chunks,
        "max_chunks": max_chunks,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    payload["manifest_sha256"] = _canonical_sha256(payload, "manifest_sha256")
    _atomic_json(manifest_path, payload)
    return payload


def join_archive(
    chunks_dir: pathlib.Path,
    manifest_path: pathlib.Path,
    archive: pathlib.Path,
    output_receipt: pathlib.Path,
    *,
    max_chunks: int,
    max_chunk_bytes: int,
    cleanup_chunks: bool,
) -> dict[str, Any]:
    chunks_dir = _safe_root(chunks_dir, "--chunks-dir", exists=True)
    manifest = _validated_chunk_manifest(
        manifest_path.expanduser().resolve(strict=True),
        max_chunks=max_chunks, max_chunk_bytes=max_chunk_bytes,
    )
    _validate_chunk_directory(chunks_dir, manifest)
    archive = archive.expanduser().absolute()
    if archive == pathlib.Path("/") or archive.is_symlink():
        raise ValueError("--archive must be a non-symlink, non-root path")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.parent / f".{archive.name}.join-{manifest['archive_sha256'][:12]}.tmp"
    if archive.exists():
        if not archive.is_file() or archive.stat().st_size != manifest["archive_bytes"] or _sha256(archive) != manifest["archive_sha256"]:
            raise ValueError("existing archive does not match the chunk manifest")
    else:
        if temporary.exists() or temporary.is_symlink():
            raise ValueError("join staging file exists; inspect it before retrying")
        digest = hashlib.sha256()
        try:
            with temporary.open("xb") as output:
                for chunk in manifest["chunks"]:
                    with (chunks_dir / chunk["name"]).open("rb") as source:
                        for block in iter(lambda: source.read(BLOCK_SIZE), b""):
                            digest.update(block)
                            output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != manifest["archive_bytes"] or digest.hexdigest() != manifest["archive_sha256"]:
                raise ValueError("reassembled archive does not match the chunk manifest")
            os.chmod(temporary, 0o600)
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "kind": "archive_join",
        "archive": str(archive),
        "archive_sha256": manifest["archive_sha256"],
        "archive_bytes": manifest["archive_bytes"],
        "manifest_sha256": manifest["manifest_sha256"],
        "chunks": len(manifest["chunks"]),
        "joined_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    _atomic_json(output_receipt.expanduser().absolute(), payload)
    if cleanup_chunks:
        for chunk in manifest["chunks"]:
            (chunks_dir / chunk["name"]).unlink()
    return payload


def _validated_receipt(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "workflow", "source", "archive", "archive_sha256",
        "archive_bytes", "members", "payload_bytes", "limits", "completed_at",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("pack receipt has missing or unexpected fields")
    if payload["schema_version"] != SCHEMA_VERSION or payload["workflow"] != WORKFLOW:
        raise ValueError("pack receipt identity is unsupported")
    if not isinstance(payload["archive_sha256"], str) or len(payload["archive_sha256"]) != 64:
        raise ValueError("pack receipt archive_sha256 is invalid")
    for name in ("archive_bytes", "members", "payload_bytes"):
        if not isinstance(payload[name], int) or isinstance(payload[name], bool) or payload[name] < 0:
            raise ValueError(f"pack receipt {name} is invalid")
    return payload


def _member_path(staging: pathlib.Path, name: str) -> pathlib.Path:
    pure = pathlib.PurePosixPath(name)
    if not name or name != pure.as_posix() or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"archive contains an unsafe path: {name!r}")
    return staging.joinpath(*pure.parts)


def _link_target(name: str, link_text: str) -> str:
    target = pathlib.PurePosixPath(link_text)
    if not link_text or "\x00" in link_text or target.is_absolute():
        raise ValueError(f"archive symlink target must be relative: {name!r}")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), link_text))
    pure = pathlib.PurePosixPath(resolved)
    if resolved in {"", "."} or pure.is_absolute() or resolved == ".." or resolved.startswith("../"):
        raise ValueError(f"archive symlink target escapes staging: {name!r}")
    return pure.as_posix()


def _validate_member_graph(
    kinds: dict[str, str], links: dict[str, tuple[str, str]]
) -> None:
    for name in kinds:
        parts = pathlib.PurePosixPath(name).parts
        for index in range(1, len(parts)):
            if kinds.get("/".join(parts[:index])) == "symlink":
                raise ValueError(f"archive member traverses a symlink parent: {name!r}")
    resolved: dict[str, str] = {}

    def resolve(name: str, active: set[str]) -> str:
        if name in resolved:
            return resolved[name]
        kind = kinds.get(name)
        if kind == "file":
            return name
        if kind != "symlink":
            raise ValueError(f"archive symlink does not resolve to a regular member: {name!r}")
        if name in active:
            raise ValueError(f"archive symlink graph contains a cycle: {name!r}")
        target = links[name][1]
        final = resolve(target, {*active, name})
        resolved[name] = final
        return final

    for name in links:
        resolve(name, set())


def extract(
    archive: pathlib.Path,
    pack_receipt: pathlib.Path,
    destination: pathlib.Path,
    output_receipt: pathlib.Path,
    *,
    max_members: int,
    max_bytes: int,
    replace_existing: bool,
) -> dict[str, Any]:
    archive = archive.expanduser().resolve(strict=True)
    pack_receipt = pack_receipt.expanduser().resolve(strict=True)
    destination = _destination(destination)
    receipt = _validated_receipt(pack_receipt)
    if archive.stat().st_size != receipt["archive_bytes"] or _sha256(archive) != receipt["archive_sha256"]:
        raise ValueError("archive size or SHA-256 does not match the pack receipt")
    if receipt["members"] > max_members or receipt["payload_bytes"] > max_bytes:
        raise ValueError("pack receipt exceeds the approved extraction limits")
    staging = destination.parent / f".{destination.name}.extract-{receipt['archive_sha256'][:12]}"
    backup = destination.parent / f".{destination.name}.previous-{receipt['archive_sha256'][:12]}"
    if staging.exists() or staging.is_symlink() or backup.exists() or backup.is_symlink():
        raise ValueError("staging or backup path already exists; inspect it before retrying")
    staging.mkdir(parents=True)
    seen: set[str] = set()
    kinds: dict[str, str] = {}
    links: dict[str, tuple[str, str]] = {}
    members = total_bytes = 0
    promoted = False
    try:
        with tarfile.open(archive, "r|") as bundle:
            for member in bundle:
                members += 1
                if members > max_members:
                    raise ValueError(f"archive exceeds --max-members={max_members}")
                if member.name in seen:
                    raise ValueError(f"archive contains a duplicate path: {member.name!r}")
                seen.add(member.name)
                if member.isdir():
                    kinds[member.name] = "directory"
                    continue
                if member.issym():
                    links[member.name] = (
                        member.linkname, _link_target(member.name, member.linkname)
                    )
                    kinds[member.name] = "symlink"
                    continue
                if not member.isreg():
                    raise ValueError(f"archive contains an unsupported member type: {member.name!r}")
                kinds[member.name] = "file"
                total_bytes += member.size
                if total_bytes > max_bytes:
                    raise ValueError(f"archive exceeds --max-bytes={max_bytes}")
        if members != receipt["members"] or total_bytes != receipt["payload_bytes"]:
            raise ValueError("archive member or payload totals do not match the pack receipt")
        _validate_member_graph(kinds, links)

        with tarfile.open(archive, "r|") as bundle:
            for member in bundle:
                target = _member_path(staging, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=False)
                    os.chmod(target, 0o755)
                    continue
                if member.issym():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"archive member has no payload: {member.name!r}")
                with source, target.open("xb") as stream:
                    shutil.copyfileobj(source, stream, length=BLOCK_SIZE)
                if target.stat().st_size != member.size:
                    raise ValueError(f"archive member size changed during extraction: {member.name!r}")
                os.chmod(target, 0o644)
        for name, (link_text, _) in links.items():
            target = _member_path(staging, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(link_text)
        for name in links:
            resolved = _member_path(staging, name).resolve(strict=True)
            if not resolved.is_relative_to(staging) or not resolved.is_file():
                raise ValueError(f"extracted symlink is not bound to a staged regular file: {name!r}")
        if destination.exists() or destination.is_symlink():
            if not replace_existing:
                raise ValueError("destination exists; use --replace-existing for recoverable promotion")
            os.replace(destination, backup)
        os.replace(staging, destination)
        promoted = True
        payload = {
            **receipt,
            "destination": str(destination),
            "backup": str(backup) if backup.exists() else None,
            "extracted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        _atomic_json(output_receipt.expanduser().absolute(), payload)
        return payload
    except Exception:
        if not promoted:
            shutil.rmtree(staging, ignore_errors=True)
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
        raise


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    pack_parser = subparsers.add_parser("pack")
    pack_parser.add_argument("--source", required=True, type=pathlib.Path)
    pack_parser.add_argument("--archive", required=True, type=pathlib.Path)
    pack_parser.add_argument("--receipt", required=True, type=pathlib.Path)
    pack_parser.add_argument("--max-members", required=True, type=_positive)
    pack_parser.add_argument("--max-bytes", required=True, type=_positive)
    digest_parser = subparsers.add_parser("digest-tree")
    digest_parser.add_argument("--source", required=True, type=pathlib.Path)
    digest_parser.add_argument("--allowed-root", required=True, type=pathlib.Path)
    digest_parser.add_argument("--receipt", required=True, type=pathlib.Path)
    digest_parser.add_argument("--max-members", required=True, type=_positive)
    digest_parser.add_argument("--max-bytes", required=True, type=_positive)
    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--archive", required=True, type=pathlib.Path)
    split_parser.add_argument("--pack-receipt", required=True, type=pathlib.Path)
    split_parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    split_parser.add_argument("--manifest", required=True, type=pathlib.Path)
    split_parser.add_argument("--chunk-bytes", required=True, type=_positive)
    split_parser.add_argument("--max-chunks", required=True, type=_positive)
    transfer_parser = subparsers.add_parser("transfer-chunks")
    transfer_parser.add_argument("--manifest", required=True, type=pathlib.Path)
    transfer_parser.add_argument("--remote-host", required=True)
    transfer_parser.add_argument("--remote-chunks-dir", required=True)
    transfer_parser.add_argument("--local-chunks-dir", required=True, type=pathlib.Path)
    transfer_parser.add_argument("--output-receipt", required=True, type=pathlib.Path)
    transfer_parser.add_argument("--max-chunks", required=True, type=_positive)
    transfer_parser.add_argument("--max-chunk-bytes", required=True, type=_positive)
    transfer_parser.add_argument("--streams", type=_positive, default=4)
    transfer_parser.add_argument("--timeout-seconds", type=_positive, default=7200)
    transfer_parser.add_argument("--ssh-executable", default="ssh")
    transfer_parser.add_argument("--rsync-executable", default="rsync")
    join_parser = subparsers.add_parser("join")
    join_parser.add_argument("--chunks-dir", required=True, type=pathlib.Path)
    join_parser.add_argument("--manifest", required=True, type=pathlib.Path)
    join_parser.add_argument("--archive", required=True, type=pathlib.Path)
    join_parser.add_argument("--output-receipt", required=True, type=pathlib.Path)
    join_parser.add_argument("--max-chunks", required=True, type=_positive)
    join_parser.add_argument("--max-chunk-bytes", required=True, type=_positive)
    join_parser.add_argument("--cleanup-chunks", action="store_true")
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--archive", required=True, type=pathlib.Path)
    extract_parser.add_argument("--pack-receipt", required=True, type=pathlib.Path)
    extract_parser.add_argument("--destination", required=True, type=pathlib.Path)
    extract_parser.add_argument("--output-receipt", required=True, type=pathlib.Path)
    extract_parser.add_argument("--max-members", required=True, type=_positive)
    extract_parser.add_argument("--max-bytes", required=True, type=_positive)
    extract_parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    if args.operation == "pack":
        payload = pack(args.source, args.archive, args.receipt, max_members=args.max_members, max_bytes=args.max_bytes)
    elif args.operation == "digest-tree":
        payload = digest_tree(
            args.source, args.allowed_root, args.receipt,
            max_members=args.max_members, max_bytes=args.max_bytes,
        )
    elif args.operation == "split":
        payload = split_archive(
            args.archive, args.pack_receipt, args.output_dir, args.manifest,
            chunk_bytes=args.chunk_bytes, max_chunks=args.max_chunks,
        )
    elif args.operation == "transfer-chunks":
        payload = transfer_chunks(
            args.manifest, args.remote_host, args.remote_chunks_dir,
            args.local_chunks_dir, args.output_receipt,
            max_chunks=args.max_chunks, max_chunk_bytes=args.max_chunk_bytes,
            streams=args.streams, timeout_seconds=args.timeout_seconds,
            ssh_executable=args.ssh_executable,
            rsync_executable=args.rsync_executable,
        )
    elif args.operation == "join":
        payload = join_archive(
            args.chunks_dir, args.manifest, args.archive, args.output_receipt,
            max_chunks=args.max_chunks, max_chunk_bytes=args.max_chunk_bytes,
            cleanup_chunks=args.cleanup_chunks,
        )
    else:
        payload = extract(
            args.archive, args.pack_receipt, args.destination, args.output_receipt,
            max_members=args.max_members, max_bytes=args.max_bytes,
            replace_existing=args.replace_existing,
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
