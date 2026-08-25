# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomically stage a large local directory tree onto a SLURM shared filesystem.

The normal rsync path is appropriate for ordinary datasets.  A validated
action-required subset is materially faster as one tar stream when its source
tree contains millions of unrelated small files or symlinks.  This helper keeps
that optimization bounded: one explicit source, one explicit remote target,
optional digest-bound manifest, exact-stream hashing, atomic promotion, and a
durable receipt.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import stat
import subprocess
import tempfile
from collections.abc import Sequence
from typing import Any


RECEIPT_NAME = ".tao-stage-receipt.json"
SCHEMA_VERSION = 1


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _safe_source(raw: str) -> pathlib.Path:
    source = pathlib.Path(raw).expanduser()
    if not source.is_absolute():
        raise ValueError("source must be an absolute path")
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source must be a non-symlink directory")
    return source.resolve(strict=True)


def _safe_remote_target(raw: str) -> pathlib.PurePosixPath:
    target = pathlib.PurePosixPath(raw)
    if not target.is_absolute() or ".." in target.parts:
        raise ValueError("remote target must be an absolute path without '..'")
    if len(target.parts) < 5 or target in {
        pathlib.PurePosixPath("/"),
        pathlib.PurePosixPath("/home"),
        pathlib.PurePosixPath("/lustre"),
    }:
        raise ValueError("remote target is too broad for atomic replacement")
    if target.name in {"", ".", ".."}:
        raise ValueError("remote target must name one dataset directory")
    return target


def _safe_login(raw: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._@:")
    if not raw or any(char not in allowed for char in raw):
        raise ValueError("login contains unsupported characters")
    return raw


def inventory_tree(source: pathlib.Path) -> dict[str, Any]:
    """Return a deterministic content inventory without following symlinks."""

    digest = hashlib.sha256()
    counts = {"directories": 0, "regular_files": 0, "symlinks": 0}
    regular_bytes = 0

    def add(kind: str, relative: str, payload: bytes = b"") -> None:
        digest.update(kind.encode() + b"\0" + relative.encode("utf-8", "surrogateescape"))
        digest.update(b"\0" + str(len(payload)).encode() + b"\0" + payload + b"\0")

    for root, dirnames, filenames in os.walk(source, topdown=True, followlinks=False):
        dirnames.sort()
        filenames.sort()
        root_path = pathlib.Path(root)

        traversed: list[str] = []
        for name in dirnames:
            path = root_path / name
            relative = path.relative_to(source).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                counts["symlinks"] += 1
                add("L", relative, os.readlink(path).encode("utf-8", "surrogateescape"))
            elif stat.S_ISDIR(mode):
                counts["directories"] += 1
                add("D", relative)
                traversed.append(name)
            else:
                raise ValueError(f"unsupported directory entry type: {path}")
        dirnames[:] = traversed

        for name in filenames:
            path = root_path / name
            if path.name == RECEIPT_NAME:
                continue
            relative = path.relative_to(source).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                counts["symlinks"] += 1
                add("L", relative, os.readlink(path).encode("utf-8", "surrogateescape"))
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(f"unsupported file type: {path}")
            counts["regular_files"] += 1
            size = path.stat().st_size
            regular_bytes += size
            digest.update(b"F\0" + relative.encode("utf-8", "surrogateescape") + b"\0")
            digest.update(str(size).encode() + b"\0")
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")

    return {
        "sha256": digest.hexdigest(),
        "counts": counts,
        "regular_bytes": regular_bytes,
    }


def _internal_symlinks_only(source: pathlib.Path) -> bool:
    """Return true when every source symlink resolves inside the source tree."""
    for root, dirnames, filenames in os.walk(source, topdown=True, followlinks=False):
        root_path = pathlib.Path(root)
        for name in [*dirnames, *filenames]:
            path = root_path / name
            if not path.is_symlink():
                continue
            try:
                path.resolve(strict=True).relative_to(source)
            except (OSError, ValueError):
                return False
        dirnames[:] = [name for name in dirnames if not (root_path / name).is_symlink()]
    return True


def _manifest_entries(path: pathlib.Path, source: pathlib.Path) -> tuple[list[str], str]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("manifest must be an absolute regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError("manifest digest mismatch")
    if payload.get("schema_version") != 1 or payload.get("source") != str(source):
        raise ValueError("manifest source/schema does not match this staging request")
    entries = payload.get("entries")
    if (
        not isinstance(entries, list)
        or not entries
        or entries != sorted(set(entries))
        or payload.get("entry_count") != len(entries)
    ):
        raise ValueError("manifest entries must be a non-empty sorted unique list")
    for entry in entries:
        relative = pathlib.PurePosixPath(entry) if isinstance(entry, str) else None
        if (
            relative is None
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or entry == RECEIPT_NAME
        ):
            raise ValueError(f"unsafe manifest entry: {entry!r}")
    return entries, expected


def _action_snapshot_entries(
    request_path: pathlib.Path, source: pathlib.Path, field: str
) -> tuple[list[str], str]:
    """Validate a complete producer-bound snapshot from an IAA action request."""
    if field not in {"controller_snapshot", "patches_snapshot"}:
        raise ValueError("snapshot field must be controller_snapshot or patches_snapshot")
    if not request_path.is_absolute() or request_path.is_symlink() or not request_path.is_file():
        raise ValueError("action request must be an absolute regular file")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    expected_request = request.get("request_sha256")
    unsigned_request = dict(request)
    unsigned_request.pop("request_sha256", None)
    if expected_request != hashlib.sha256(
        json.dumps(unsigned_request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest():
        raise ValueError("action request digest mismatch")
    if request.get("workflow") != "tao-run-deft-iaa" or request.get("platform") != "slurm":
        raise ValueError("snapshot request is not an IAA SLURM action")
    manifest = request.get(field)
    if not isinstance(manifest, dict) or manifest.get("root") != str(source):
        raise ValueError("snapshot root does not match staging source")
    records = manifest.get("entries")
    if not isinstance(records, list) or not records:
        raise ValueError("snapshot entries must be a non-empty list")
    if manifest.get("sha256") != hashlib.sha256(
        json.dumps({"entries": records}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest():
        raise ValueError("snapshot manifest digest mismatch")
    entries: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise ValueError("snapshot entry has an invalid shape")
        relative = pathlib.PurePosixPath(record["path"])
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"unsafe snapshot entry: {record['path']!r}")
        path = source.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"snapshot entry is missing or changed: {record['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"snapshot entry digest mismatch: {record['path']}")
        entries.append(record["path"])
    if entries != sorted(set(entries)):
        raise ValueError("snapshot entries must be sorted and unique")
    actual = sorted(
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if actual != entries:
        raise ValueError("snapshot tree contains missing or unrelated files")
    return entries, manifest["sha256"]


def inventory_paths(source: pathlib.Path, entries: list[str]) -> dict[str, Any]:
    """Inventory only explicit action-required leaves and reject escape links."""

    digest = hashlib.sha256()
    counts = {"directories": 0, "regular_files": 0, "symlinks": 0}
    regular_bytes = 0
    entry_set = set(entries)
    for relative in entries:
        path = source / relative
        mode = path.lstat().st_mode
        encoded = relative.encode("utf-8", "surrogateescape")
        if stat.S_ISLNK(mode):
            resolved = path.resolve(strict=True)
            try:
                resolved_relative = resolved.relative_to(source).as_posix()
            except ValueError as exc:
                raise ValueError(f"manifest symlink escapes source: {relative}") from exc
            if resolved_relative not in entry_set:
                raise ValueError(
                    f"manifest omits the in-source target of symlink {relative}: "
                    f"{resolved_relative}"
                )
            target = os.readlink(path).encode("utf-8", "surrogateescape")
            counts["symlinks"] += 1
            digest.update(b"L\0" + encoded + b"\0" + target + b"\0")
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"manifest entry is not a regular file or symlink: {relative}")
        size = path.stat().st_size
        counts["regular_files"] += 1
        regular_bytes += size
        digest.update(b"F\0" + encoded + b"\0" + str(size).encode() + b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "counts": counts, "regular_bytes": regular_bytes}


def _receipt_core(
    source: pathlib.Path,
    target: pathlib.PurePosixPath,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "target": str(target),
        "inventory": inventory,
    }


def receipt_reusable(receipt: Any, core: dict[str, Any]) -> bool:
    if not isinstance(receipt, dict) or not isinstance(receipt.get("archive_sha256"), str):
        return False
    return all(receipt.get(key) == value for key, value in core.items())


def _ssh_capture(login: str, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", login, command],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _read_remote_receipt(login: str, target: pathlib.PurePosixPath) -> Any:
    marker = target / RECEIPT_NAME
    command = f"test -f {shlex.quote(str(marker))} && sed -n '1p' {shlex.quote(str(marker))}"
    completed = _ssh_capture(login, command)
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def _stream_archive(
    login: str,
    source: pathlib.Path,
    temp_target: pathlib.PurePosixPath,
    entries: list[str] | None = None,
) -> str:
    remote_command = "set -Eeuo pipefail; mkdir -p {parent}; if test -e {temp}; then chmod -R u+rwX -- {temp}; rm -rf -- {temp}; fi; mkdir -p {temp}; tar --no-same-owner -C {temp} -xf -".format(
        parent=shlex.quote(str(temp_target.parent)), temp=shlex.quote(str(temp_target))
    )
    with tempfile.TemporaryFile() as tar_stderr, tempfile.TemporaryFile() as ssh_stderr:
        manifest_file = None
        tar_args = ["tar", "--format=pax", "-C", str(source), "-cf", "-"]
        if entries is None:
            tar_args.append(".")
        else:
            manifest_file = tempfile.NamedTemporaryFile()
            manifest_file.write(b"\0".join(item.encode("utf-8") for item in entries) + b"\0")
            manifest_file.flush()
            tar_args.extend(
                [
                    "--null",
                    "--verbatim-files-from",
                    "--no-recursion",
                    "-T",
                    manifest_file.name,
                ]
            )
        tar_process = subprocess.Popen(
            tar_args,
            stdout=subprocess.PIPE,
            stderr=tar_stderr,
        )
        assert tar_process.stdout is not None
        ssh_process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", login, remote_command],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=ssh_stderr,
        )
        assert ssh_process.stdin is not None
        archive_digest = hashlib.sha256()
        try:
            while chunk := tar_process.stdout.read(1024 * 1024):
                archive_digest.update(chunk)
                ssh_process.stdin.write(chunk)
        except BrokenPipeError:
            pass
        finally:
            ssh_process.stdin.close()
            tar_process.stdout.close()
        tar_code = tar_process.wait()
        ssh_code = ssh_process.wait()
        if tar_code or ssh_code:
            tar_stderr.seek(0)
            ssh_stderr.seek(0)
            detail = (tar_stderr.read() + ssh_stderr.read()).decode(errors="replace")[-4000:]
            raise RuntimeError(
                f"archive staging failed (tar={tar_code}, ssh={ssh_code}): {detail}"
            )
        if manifest_file is not None:
            manifest_file.close()
    return archive_digest.hexdigest()


def _valid_seed_receipt(
    receipt: Any, source: pathlib.Path, target: pathlib.PurePosixPath,
) -> bool:
    return (
        isinstance(receipt, dict)
        and receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("source") == str(source)
        and receipt.get("target") == str(target)
        and isinstance(receipt.get("inventory"), dict)
        and re.fullmatch(r"[0-9a-f]{64}", str(receipt["inventory"].get("sha256", "")))
        is not None
        and re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("archive_sha256", "")))
        is not None
    )


def _seed_and_rsync(
    login: str,
    source: pathlib.Path,
    target: pathlib.PurePosixPath,
    temp_target: pathlib.PurePosixPath,
    inventory_sha256: str,
) -> str:
    """Create a copy-on-write candidate and send only content changes."""
    seed = f"""set -Eeuo pipefail
test -d {shlex.quote(str(target))}
test ! -L {shlex.quote(str(target))}
mkdir -p {shlex.quote(str(temp_target.parent))}
if test -e {shlex.quote(str(temp_target))}; then
  test ! -L {shlex.quote(str(temp_target))}
  chmod -R u+rwX -- {shlex.quote(str(temp_target))}
  rm -rf -- {shlex.quote(str(temp_target))}
fi
cp -al -- {shlex.quote(str(target))} {shlex.quote(str(temp_target))}
rm -f -- {shlex.quote(str(temp_target / RECEIPT_NAME))}
"""
    seeded = _ssh_capture(login, seed)
    if seeded.returncode != 0:
        raise RuntimeError(
            "copy-on-write remote staging seed failed: " + seeded.stderr[-4000:]
        )
    base = [
        "rsync", "--archive", "--checksum", "--delete", "--delete-excluded",
        "--safe-links", f"--exclude=/{RECEIPT_NAME}", "--",
        str(source) + "/", f"{login}:{temp_target}/",
    ]
    synchronized = subprocess.run(
        base, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if synchronized.returncode != 0:
        raise RuntimeError(
            "incremental remote staging failed: " + synchronized.stderr[-4000:]
        )
    # A zero rsync exit is the protocol's receiver-acknowledged completion of
    # every checksum comparison, changed-file transfer, deletion, and rename.
    # Repeating the same full-tree checksum as a dry run adds no new failure
    # boundary and doubles staging time once checkpoints are multi-gigabyte.
    return hashlib.sha256(
        b"rsync-cow-v1\0" + inventory_sha256.encode("ascii")
    ).hexdigest()


def _promote(
    login: str,
    target: pathlib.PurePosixPath,
    temp_target: pathlib.PurePosixPath,
    backup_target: pathlib.PurePosixPath,
    receipt: dict[str, Any],
) -> None:
    encoded = base64.b64encode(_canonical(receipt)).decode("ascii")
    marker = temp_target / RECEIPT_NAME
    command = f"""set -Eeuo pipefail
python3 -c {shlex.quote('import base64,pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(base64.b64decode(sys.argv[2]))')} {shlex.quote(str(marker))} {shlex.quote(encoded)}
if test -e {shlex.quote(str(backup_target))}; then
  chmod -R u+rwX -- {shlex.quote(str(backup_target))}
  rm -rf -- {shlex.quote(str(backup_target))}
fi
if test -e {shlex.quote(str(target))}; then mv -- {shlex.quote(str(target))} {shlex.quote(str(backup_target))}; fi
if mv -- {shlex.quote(str(temp_target))} {shlex.quote(str(target))}; then
  if test -e {shlex.quote(str(backup_target))}; then
    chmod -R u+rwX -- {shlex.quote(str(backup_target))}
    rm -rf -- {shlex.quote(str(backup_target))}
  fi
else
  if test -e {shlex.quote(str(backup_target))} && ! test -e {shlex.quote(str(target))}; then mv -- {shlex.quote(str(backup_target))} {shlex.quote(str(target))}; fi
  exit 1
fi
"""
    completed = _ssh_capture(login, command)
    if completed.returncode != 0:
        raise RuntimeError(f"atomic remote promotion failed: {completed.stderr[-4000:]}")


def _write_local_receipt(path: pathlib.Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp.write(_canonical(receipt))
        temp_name = pathlib.Path(tmp.name)
    os.replace(temp_name, path)


def stage_tree(
    *,
    source_raw: str,
    login_raw: str,
    target_raw: str,
    receipt_raw: str,
    manifest_raw: str | None = None,
    action_request_raw: str | None = None,
    snapshot_field: str | None = None,
    incremental_existing: bool = False,
) -> dict[str, Any]:
    source = _safe_source(source_raw)
    login = _safe_login(login_raw)
    target = _safe_remote_target(target_raw)
    receipt_path = pathlib.Path(receipt_raw).expanduser()
    if not receipt_path.is_absolute():
        raise ValueError("receipt must be an absolute local path")

    entries = None
    manifest_sha256 = None
    if manifest_raw is not None and action_request_raw is not None:
        raise ValueError("manifest and action-request snapshot modes are mutually exclusive")
    if action_request_raw is None and snapshot_field is not None:
        raise ValueError("snapshot-field requires action-request")
    if action_request_raw is not None and snapshot_field is None:
        raise ValueError("action-request requires snapshot-field")
    if action_request_raw is not None:
        entries, manifest_sha256 = _action_snapshot_entries(
            pathlib.Path(action_request_raw).expanduser(), source, str(snapshot_field)
        )
        inventory = inventory_paths(source, entries)
    elif manifest_raw is None:
        inventory = inventory_tree(source)
    else:
        entries, manifest_sha256 = _manifest_entries(
            pathlib.Path(manifest_raw).expanduser(), source
        )
        inventory = inventory_paths(source, entries)
    # Cache relocation changes an action-bound manifest digest without changing
    # selected content. Reuse is therefore bound to the exact deterministic
    # source/target/content inventory; the manifest digest remains evidence on
    # a newly staged receipt but is not part of the reuse identity.
    core = _receipt_core(source, target, inventory)
    remote_receipt = _read_remote_receipt(login, target)
    if receipt_reusable(remote_receipt, core):
        _write_local_receipt(receipt_path, remote_receipt)
        return {"status": "reused", "receipt": str(receipt_path), **core}

    suffix = inventory["sha256"][:16]
    temp_target = target.with_name(f".{target.name}.staging-{suffix}")
    backup_target = target.with_name(f".{target.name}.previous-{suffix}")
    use_incremental = (
        incremental_existing
        and entries is None
        and _internal_symlinks_only(source)
        and _valid_seed_receipt(remote_receipt, source, target)
    )
    archive_sha256 = (
        _seed_and_rsync(login, source, target, temp_target, inventory["sha256"])
        if use_incremental
        else _stream_archive(login, source, temp_target, entries)
    )
    receipt = {
        **core,
        "archive_sha256": archive_sha256,
        "transport": "rsync-cow-v1" if use_incremental else "tar-stream-v1",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if manifest_sha256 is not None:
        receipt["manifest_sha256"] = manifest_sha256
    _promote(login, target, temp_target, backup_target, receipt)
    confirmed = _read_remote_receipt(login, target)
    if confirmed != receipt:
        raise RuntimeError("remote staging receipt did not survive atomic promotion")
    _write_local_receipt(receipt_path, receipt)
    result = {"status": "staged", "receipt": str(receipt_path), **core}
    if manifest_sha256 is not None:
        result["manifest_sha256"] = manifest_sha256
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--login", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--action-request")
    parser.add_argument("--incremental-existing", action="store_true")
    parser.add_argument(
        "--snapshot-field",
        choices=("controller_snapshot", "patches_snapshot"),
    )
    args = parser.parse_args(argv)
    try:
        result = stage_tree(
            source_raw=args.source,
            login_raw=args.login,
            target_raw=args.target,
            receipt_raw=args.receipt,
            manifest_raw=args.manifest,
            action_request_raw=args.action_request,
            snapshot_field=args.snapshot_field,
            incremental_existing=args.incremental_existing,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"slurm_stage_tree: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
