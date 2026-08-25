# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import io
import json
import os
import pathlib
import shlex
import shutil
import stat
import subprocess
import tarfile
import threading
import time

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skills/platform/tao-run-on-brev/scripts/brev_archive_transfer.py"
SPEC = importlib.util.spec_from_file_location("brev_archive_transfer", SCRIPT)
assert SPEC and SPEC.loader
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


def _pack(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, dict]:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "a.txt").write_text("alpha")
    (source / "nested/b.bin").write_bytes(b"beta")
    archive = tmp_path / "payload.tar"
    receipt = tmp_path / "pack.json"
    payload = transfer.pack(source, archive, receipt, max_members=10, max_bytes=100)
    return archive, receipt, payload


def test_pack_is_deterministic_and_receipted(tmp_path: pathlib.Path) -> None:
    archive, _, first = _pack(tmp_path)
    second_archive = tmp_path / "payload-two.tar"
    second_receipt = tmp_path / "pack-two.json"
    source = tmp_path / "source"
    second = transfer.pack(source, second_archive, second_receipt, max_members=10, max_bytes=100)
    assert first["archive_sha256"] == second["archive_sha256"]
    assert archive.read_bytes() == second_archive.read_bytes()
    assert first["members"] == 3 and first["payload_bytes"] == 9


def test_pack_rejects_bounds(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_text("payload")
    with pytest.raises(ValueError, match="max-bytes"):
        transfer.pack(source, tmp_path / "y.tar", tmp_path / "y.json", max_members=10, max_bytes=1)


def test_mode_0500_snapshot_routes_through_archive_without_mutation(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "request-snapshot"
    nested = source / "skills"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text("signed snapshot", encoding="utf-8")
    os.chmod(nested, 0o500)
    os.chmod(source, 0o500)

    # A mode-first recursive copy publishes the non-writable directory before
    # its children. This is the ordering used by the failing Brev copy path.
    destination_mode = stat.S_IMODE(source.stat().st_mode)
    assert destination_mode == 0o500
    assert not destination_mode & stat.S_IWUSR
    assert any(source.iterdir())

    def mode_first_recursive_copy(
        source_root: pathlib.Path, destination_root: pathlib.Path
    ) -> None:
        destination_root.mkdir()
        os.chmod(destination_root, stat.S_IMODE(source_root.stat().st_mode))
        destination_writable = destination_root.stat().st_mode & stat.S_IWUSR
        if any(source_root.iterdir()) and not destination_writable:
            raise PermissionError("destination mode applied before child creation")

    with pytest.raises(PermissionError, match="before child creation"):
        mode_first_recursive_copy(source, tmp_path / "mode-first-copy")

    archive = tmp_path / "snapshot.tar"
    receipt = tmp_path / "snapshot.pack.json"
    packed = transfer.pack(
        source, archive, receipt, max_members=4, max_bytes=1024
    )
    destination = tmp_path / "published-snapshot"
    transfer.extract(
        archive,
        receipt,
        destination,
        tmp_path / "snapshot.extract.json",
        max_members=4,
        max_bytes=1024,
        replace_existing=False,
    )

    assert (destination / "skills/SKILL.md").read_text() == "signed snapshot"
    assert packed["members"] == 2
    assert stat.S_IMODE(source.stat().st_mode) == 0o500
    assert stat.S_IMODE(nested.stat().st_mode) == 0o500


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("absolute", "relative"), ("escape", "escapes"),
        ("dangling", "dangling"), ("directory", "regular file"),
        ("unsafe_parent", "unsafe parent"), ("cycle", "cyclic"),
    ],
)
def test_pack_rejects_unsafe_source_links(
    tmp_path: pathlib.Path, case: str, message: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside")
    if case == "absolute":
        (source / "link").symlink_to(outside)
    elif case == "escape":
        (source / "link").symlink_to("../outside")
    elif case == "dangling":
        (source / "link").symlink_to("missing")
    elif case == "directory":
        (source / "directory").mkdir()
        (source / "link").symlink_to("directory")
    elif case == "unsafe_parent":
        (source / "real").mkdir()
        (source / "real/file").write_text("payload")
        (source / "alias").symlink_to("real")
        (source / "link").symlink_to("alias/file")
    else:
        (source / "a").symlink_to("b")
        (source / "b").symlink_to("a")
    with pytest.raises(ValueError, match=message):
        transfer.pack(source, tmp_path / "x.tar", tmp_path / "x.json", max_members=10, max_bytes=100)


def test_safe_relative_symlink_roundtrip_preserves_exact_text(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source"
    (source / "raw").mkdir(parents=True)
    (source / "images").mkdir()
    (source / "raw/image.jpg").write_bytes(b"pixels")
    (source / "images/image.jpg").symlink_to("../raw/image.jpg")
    archive = tmp_path / "links.tar"
    pack_receipt = tmp_path / "links.pack.json"
    packed = transfer.pack(source, archive, pack_receipt, max_members=10, max_bytes=100)
    destination = tmp_path / "destination"
    transfer.extract(
        archive, pack_receipt, destination, tmp_path / "links.extract.json",
        max_members=10, max_bytes=100, replace_existing=False,
    )
    assert (destination / "images/image.jpg").is_symlink()
    assert (destination / "images/image.jpg").readlink().as_posix() == "../raw/image.jpg"
    assert (destination / "images/image.jpg").read_bytes() == b"pixels"
    assert packed["payload_bytes"] == 6


def test_digest_tree_accepts_in_root_blob_links_and_is_content_bound(tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "cache"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots/revision"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / "weight"
    blob.write_bytes(b"weights")
    (snapshot / "model.bin").symlink_to(blob)
    receipt = tmp_path / "tree.json"
    first = transfer.digest_tree(snapshot, cache, receipt, max_members=2, max_bytes=100)
    assert first["members"] == 1 and first["payload_bytes"] == 7
    blob.write_bytes(b"changed")
    second = transfer.digest_tree(snapshot, cache, tmp_path / "tree-two.json", max_members=2, max_bytes=100)
    assert second["tree_sha256"] != first["tree_sha256"]


def test_digest_tree_rejects_escaping_symlink(tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "cache"
    snapshot = cache / "snapshot"
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("secret")
    (snapshot / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        transfer.digest_tree(snapshot, cache, tmp_path / "tree.json", max_members=2, max_bytes=100)


def test_extract_validates_digest_and_promotes_with_backup(tmp_path: pathlib.Path) -> None:
    archive, receipt, payload = _pack(tmp_path)
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "old").write_text("recoverable")
    output = tmp_path / "extract.json"
    result = transfer.extract(
        archive, receipt, destination, output,
        max_members=payload["members"], max_bytes=payload["payload_bytes"],
        replace_existing=True,
    )
    assert (destination / "a.txt").read_text() == "alpha"
    assert pathlib.Path(result["backup"]).joinpath("old").read_text() == "recoverable"
    assert json.loads(output.read_text())["archive_sha256"] == payload["archive_sha256"]

    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        transfer.extract(
            archive, receipt, tmp_path / "other", tmp_path / "other.json",
            max_members=10, max_bytes=100, replace_existing=False,
        )


@pytest.mark.parametrize("kind", ["traversal", "absolute_symlink", "duplicate", "hardlink"])
def test_extract_rejects_unsafe_members(tmp_path: pathlib.Path, kind: str) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as bundle:
        if kind == "traversal":
            info = tarfile.TarInfo("../escape")
            info.size = 1
            bundle.addfile(info, io.BytesIO(b"x"))
        elif kind == "absolute_symlink":
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            bundle.addfile(info)
        elif kind == "duplicate":
            for value in (b"a", b"b"):
                info = tarfile.TarInfo("same")
                info.size = 1
                bundle.addfile(info, io.BytesIO(value))
        else:
            info = tarfile.TarInfo("hard")
            info.type = tarfile.LNKTYPE
            info.linkname = "target"
            bundle.addfile(info)
    receipt = tmp_path / "unsafe.json"
    payload = {
        "schema_version": "1", "workflow": transfer.WORKFLOW,
        "source": "/remote/source", "archive": "/remote/archive.tar",
        "archive_sha256": transfer._sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "members": 2 if kind == "duplicate" else 1,
        "payload_bytes": 2 if kind == "duplicate" else (1 if kind == "traversal" else 0),
        "limits": {"max_members": 10, "max_bytes": 100}, "completed_at": "now",
    }
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsafe path|unsupported member|duplicate|must be relative"):
        transfer.extract(
            archive, receipt, tmp_path / "destination", tmp_path / "out.json",
            max_members=10, max_bytes=100, replace_existing=False,
        )
    assert not (tmp_path / "destination").exists()


def test_extract_rejects_symlink_cycle_and_symlink_parent(tmp_path: pathlib.Path) -> None:
    for case in ("cycle", "parent"):
        archive = tmp_path / f"{case}.tar"
        with tarfile.open(archive, "w") as bundle:
            first = tarfile.TarInfo("a")
            first.type = tarfile.SYMTYPE
            first.linkname = "b" if case == "cycle" else "target"
            bundle.addfile(first)
            if case == "cycle":
                second = tarfile.TarInfo("b")
                second.type = tarfile.SYMTYPE
                second.linkname = "a"
                bundle.addfile(second)
            else:
                target = tarfile.TarInfo("target")
                target.type = tarfile.DIRTYPE
                bundle.addfile(target)
                nested = tarfile.TarInfo("a/file")
                nested.size = 1
                bundle.addfile(nested, io.BytesIO(b"x"))
        receipt = tmp_path / f"{case}.json"
        receipt.write_text(json.dumps({
            "schema_version": "1", "workflow": transfer.WORKFLOW,
            "source": "/remote/source", "archive": "/remote/archive.tar",
            "archive_sha256": transfer._sha256(archive), "archive_bytes": archive.stat().st_size,
            "members": 2 if case == "cycle" else 3,
            "payload_bytes": 0 if case == "cycle" else 1,
            "limits": {"max_members": 10, "max_bytes": 100}, "completed_at": "now",
        }))
        with pytest.raises(ValueError, match="cycle|symlink parent"):
            transfer.extract(
                archive, receipt, tmp_path / f"out-{case}", tmp_path / f"out-{case}.json",
                max_members=10, max_bytes=100, replace_existing=False,
            )


def _split(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict]:
    archive, pack_receipt, _ = _pack(tmp_path)
    chunks = tmp_path / "chunks"
    manifest_path = tmp_path / "chunks.json"
    manifest = transfer.split_archive(
        archive, pack_receipt, chunks, manifest_path,
        chunk_bytes=1024, max_chunks=32,
    )
    return archive, chunks, manifest_path, manifest


def test_chunk_split_and_join_are_digest_bound_and_atomic(tmp_path: pathlib.Path) -> None:
    archive, chunks, manifest_path, manifest = _split(tmp_path)
    joined = tmp_path / "joined.tar"
    receipt = tmp_path / "joined.json"
    result = transfer.join_archive(
        chunks, manifest_path, joined, receipt,
        max_chunks=32, max_chunk_bytes=1024, cleanup_chunks=False,
    )
    assert joined.read_bytes() == archive.read_bytes()
    assert result["archive_sha256"] == manifest["archive_sha256"]
    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    assert not list(tmp_path.glob(".joined.tar.join-*.tmp"))


def test_chunk_split_resumes_verified_chunks_and_rejects_tamper(tmp_path: pathlib.Path) -> None:
    archive, chunks, manifest_path, first = _split(tmp_path)
    retained = chunks / first["chunks"][0]["name"]
    retained_stat = retained.stat()
    missing = chunks / first["chunks"][-1]["name"]
    missing.unlink()
    manifest_path.unlink()
    resumed = transfer.split_archive(
        archive, tmp_path / "pack.json", chunks, manifest_path,
        chunk_bytes=1024, max_chunks=32,
    )
    assert resumed["archive_sha256"] == first["archive_sha256"]
    assert retained.stat().st_ino == retained_stat.st_ino
    assert missing.exists()

    manifest_path.unlink()
    retained.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="existing chunk does not match"):
        transfer.split_archive(
            archive, tmp_path / "pack.json", chunks, manifest_path,
            chunk_bytes=1024, max_chunks=32,
        )


def test_chunk_split_recovers_only_its_interrupted_temporary(tmp_path: pathlib.Path) -> None:
    archive, chunks, manifest_path, _ = _split(tmp_path)
    manifest_path.unlink()
    interrupted = chunks / ".chunk-000001.part.deadbeef"
    interrupted.write_bytes(b"partial")
    resumed = transfer.split_archive(
        archive, tmp_path / "pack.json", chunks, manifest_path,
        chunk_bytes=1024, max_chunks=32,
    )
    assert resumed["archive_bytes"] == archive.stat().st_size
    assert not interrupted.exists()


def test_chunk_split_requires_manifest_outside_chunk_directory(tmp_path: pathlib.Path) -> None:
    archive, pack_receipt, _ = _pack(tmp_path)
    chunks = tmp_path / "chunks"
    with pytest.raises(ValueError, match="outside"):
        transfer.split_archive(
            archive, pack_receipt, chunks, chunks / "manifest.json",
            chunk_bytes=1024, max_chunks=32,
        )


@pytest.mark.parametrize("case", ["missing", "tampered", "reordered", "duplicate"])
def test_chunk_join_rejects_incomplete_or_mismatched_sets(
    tmp_path: pathlib.Path, case: str
) -> None:
    _, chunks, manifest_path, manifest = _split(tmp_path)
    if case == "missing":
        (chunks / manifest["chunks"][0]["name"]).unlink()
    elif case == "tampered":
        (chunks / manifest["chunks"][0]["name"]).write_bytes(b"tampered")
    else:
        changed = json.loads(manifest_path.read_text())
        if case == "reordered":
            changed["chunks"][0], changed["chunks"][1] = changed["chunks"][1], changed["chunks"][0]
        else:
            changed["chunks"][1]["name"] = changed["chunks"][0]["name"]
        changed["manifest_sha256"] = transfer._canonical_sha256(changed, "manifest_sha256")
        manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="missing|mismatch|out of order|duplicate"):
        transfer.join_archive(
            chunks, manifest_path, tmp_path / "joined.tar", tmp_path / "joined.json",
            max_chunks=32, max_chunk_bytes=1024, cleanup_chunks=False,
        )
    assert not (tmp_path / "joined.tar").exists()


def test_chunk_manifest_bounds_and_cleanup_are_explicit(tmp_path: pathlib.Path) -> None:
    _, chunks, manifest_path, manifest = _split(tmp_path)
    with pytest.raises(ValueError, match="bounds|count"):
        transfer.join_archive(
            chunks, manifest_path, tmp_path / "too-small.tar", tmp_path / "too-small.json",
            max_chunks=1, max_chunk_bytes=1024, cleanup_chunks=False,
        )
    joined = tmp_path / "joined.tar"
    transfer.join_archive(
        chunks, manifest_path, joined, tmp_path / "joined.json",
        max_chunks=32, max_chunk_bytes=1024, cleanup_chunks=True,
    )
    assert joined.exists()
    assert all(not (chunks / chunk["name"]).exists() for chunk in manifest["chunks"])


def test_chunk_transfer_uses_independent_manifest_owned_streams_and_resumes(
    tmp_path: pathlib.Path,
) -> None:
    _, remote_chunks, manifest_path, manifest = _split(tmp_path)
    local_chunks = tmp_path / "local-chunks"
    local_chunks.mkdir()
    first = manifest["chunks"][0]
    second = manifest["chunks"][1]
    shutil.copyfile(remote_chunks / first["name"], local_chunks / first["name"])
    second_data = (remote_chunks / second["name"]).read_bytes()
    (local_chunks / second["name"]).write_bytes(second_data[: len(second_data) // 2])

    calls: list[list[str]] = []
    active = 0
    peak = 0
    lock = threading.Lock()

    def runner(argv, **kwargs):
        nonlocal active, peak
        with lock:
            calls.append(argv)
            active += 1
            peak = max(peak, active)
        target = pathlib.Path(argv[-1])
        source = remote_chunks / target.name
        offset = target.stat().st_size if target.exists() else 0
        with source.open("rb") as input_stream, target.open("ab") as output_stream:
            input_stream.seek(offset)
            shutil.copyfileobj(input_stream, output_stream)
        time.sleep(0.01)
        with lock:
            active -= 1
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    receipt_path = tmp_path / "transfer.json"
    result = transfer.transfer_chunks(
        manifest_path,
        "brev-instance",
        "/remote/chunks",
        local_chunks,
        receipt_path,
        max_chunks=32,
        max_chunk_bytes=1024,
        streams=4,
        runner=runner,
        ssh_executable="/usr/bin/ssh",
        rsync_executable="/usr/bin/rsync",
    )

    destinations = [argv[-1] for argv in calls]
    assert len(destinations) == len(set(destinations))
    assert str(local_chunks / first["name"]) not in destinations
    assert str(local_chunks / second["name"]) in destinations
    assert 2 <= peak <= 4
    for argv in calls:
        assert argv[1:3] == ["--partial", "--append-verify"]
        ssh_argv = shlex.split(argv[argv.index("-e") + 1])
        assert ["-o", "ControlMaster=no"] == ssh_argv[3:5]
        assert ["-o", "ControlPath=none"] == ssh_argv[5:7]
        assert ["-o", "ControlPersist=no"] == ssh_argv[7:9]
        assert pathlib.Path(argv[-1]).name in {
            chunk["name"] for chunk in manifest["chunks"]
        }
    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    assert result["streams"] == 4
    assert json.loads(receipt_path.read_text())["kind"] == "archive_chunk_transfer"
    transfer._validate_chunk_directory(local_chunks, manifest)  # noqa: SLF001


def test_chunk_transfer_failure_retains_only_manifest_partial(
    tmp_path: pathlib.Path,
) -> None:
    _, _, manifest_path, manifest = _split(tmp_path)
    local_chunks = tmp_path / "local-chunks"
    local_chunks.mkdir()
    partial = local_chunks / manifest["chunks"][0]["name"]
    partial.write_bytes(b"partial")

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 23, stdout="", stderr="network")

    receipt = tmp_path / "transfer.json"
    with pytest.raises(RuntimeError, match="resumable partials retained"):
        transfer.transfer_chunks(
            manifest_path,
            "brev-instance",
            "/remote/chunks",
            local_chunks,
            receipt,
            max_chunks=32,
            max_chunk_bytes=1024,
            streams=4,
            runner=runner,
        )
    assert partial.read_bytes() == b"partial"
    assert not receipt.exists()


def test_chunk_transfer_rejects_receipt_below_chunk_directory(
    tmp_path: pathlib.Path,
) -> None:
    _, _, manifest_path, _ = _split(tmp_path)
    local_chunks = tmp_path / "local-chunks"

    with pytest.raises(ValueError, match="outside --local-chunks-dir"):
        transfer.transfer_chunks(
            manifest_path,
            "brev-instance",
            "/remote/chunks",
            local_chunks,
            local_chunks / "evidence" / "transfer.json",
            max_chunks=32,
            max_chunk_bytes=1024,
            streams=4,
        )
