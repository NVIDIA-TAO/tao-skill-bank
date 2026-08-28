# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end regression tests for immutable IAA/PAS archive inputs."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / "skills" / "applications" / "tao-run-deft-iaa"
SCRIPTS = SKILL_ROOT / "scripts"
PYT_IMAGE = "nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.2.0-rc-53-multiarch"  # versions-key: images.tao_toolkit.deft_pas_pyt
DS_IMAGE = "nvcr.io/nvstaging/tao/tao-toolkit-ds:7.2.0-rc-52-multiarch"  # versions-key: images.tao_toolkit.deft_pas_data_services


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tar(path: Path, payload: bytes, *, gzip: bool = False) -> None:
    mode = "w:gz" if gzip else "w"
    with tarfile.open(path, mode) as archive:
        info = tarfile.TarInfo("payload.txt")
        info.size = len(payload)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(payload))


def _run(script: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *(str(arg) for arg in args)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def iaa_run(tmp_path: Path) -> dict[str, object]:
    workspace = tmp_path / "workspace"
    archive_root = tmp_path / "archives"
    workspace.mkdir()
    (workspace / "data").mkdir()
    archive_root.mkdir()
    images_archive = archive_root / "images_raw.tar"
    metadata_archive = archive_root / "meta.tar.gz"
    _write_tar(images_archive, b"approved\n")
    _write_tar(metadata_archive, b"metadata\n", gzip=True)
    return {
        "workspace": workspace,
        "results": workspace / "results" / "run",
        "dataset": workspace / "data" / "iaa",
        "images": images_archive,
        "metadata": metadata_archive,
        "images_sha256": _sha256(images_archive),
        "metadata_sha256": _sha256(metadata_archive),
    }


def _prepare(run: dict[str, object], **overrides: object) -> subprocess.CompletedProcess[str]:
    values = {**run, **overrides}
    args: list[object] = [
        "prepare_deft_config.py",
        "--workspace",
        values["workspace"],
        "--results-dir",
        values["results"],
        "--dataset-root",
        values["dataset"],
        "--images-archive",
        values["images"],
        "--images-archive-sha256",
        values["images_sha256"],
        "--metadata-archive",
        values["metadata"],
        "--metadata-archive-sha256",
        values["metadata_sha256"],
        "--max-iterations",
        1,
    ]
    if values.get("checksums") is not None:
        args.extend(("--checksums-file", values["checksums"]))
    return _run(str(args[0]), *args[1:])


def _initialize(run: dict[str, object]) -> subprocess.CompletedProcess[str]:
    results = Path(run["results"])
    args: list[object] = [
        "init_deft_state.py",
        "--results-dir",
        results,
        "--workspace",
        run["workspace"],
        "--dataset-root",
        run["dataset"],
        "--images-archive",
        run["images"],
        "--images-archive-sha256",
        run["images_sha256"],
        "--metadata-archive",
        run["metadata"],
        "--metadata-archive-sha256",
        run["metadata_sha256"],
        "--max-iterations",
        1,
        "--platform",
        "docker",
        "--pyt-image",
        PYT_IMAGE,
        "--ds-image",
        DS_IMAGE,
        "--deft-config",
        results / "config" / "deft_config.yaml",
        "--tao-spec",
        results / "config" / "tao_spec.yaml",
    ]
    if run.get("checksums") is not None:
        args.extend(("--checksums-file", run["checksums"]))
    return _run(str(args[0]), *args[1:])


def _prepare_and_initialize(run: dict[str, object]) -> None:
    prepared = _prepare(run)
    assert prepared.returncode == 0, prepared.stderr
    initialized = _initialize(run)
    assert initialized.returncode == 0, initialized.stderr


def test_prepare_rejects_bytes_that_do_not_match_approved_digest(iaa_run):
    wrong = "0" * 64
    prepared = _prepare(iaa_run, images_sha256=wrong)
    assert prepared.returncode == 2
    assert "changed after approval" in prepared.stderr
    assert not Path(iaa_run["results"]).exists()


def test_preflight_helper_emits_canonical_archive_identity(iaa_run):
    inspected = _run("archive_contract.py", "--archive", iaa_run["images"])
    assert inspected.returncode == 0, inspected.stderr
    assert inspected.stdout.strip() == iaa_run["images_sha256"]


def test_internal_bindings_coexist_with_optional_publisher_manifest(iaa_run):
    manifest = Path(iaa_run["images"]).parent / "SHA256SUMS"
    manifest.write_text(
        f"{iaa_run['images_sha256']}  images_raw.tar\n"
        f"{iaa_run['metadata_sha256']}  meta.tar.gz\n"
    )
    iaa_run["checksums"] = manifest

    _prepare_and_initialize(iaa_run)
    audit = _run(
        "audit_deft_run.py", "--results-dir", iaa_run["results"], "--json"
    )
    assert audit.returncode == 0, audit.stderr
    assert json.loads(audit.stdout)["status"] == "IN_PROGRESS"


def test_initialization_rejects_archive_changed_after_config_preparation(iaa_run):
    prepared = _prepare(iaa_run)
    assert prepared.returncode == 0, prepared.stderr
    _write_tar(Path(iaa_run["metadata"]), b"tampered\n", gzip=True)

    initialized = _initialize(iaa_run)
    assert initialized.returncode == 2
    assert "changed after approval" in initialized.stderr
    assert not (Path(iaa_run["results"]) / "deft_state.json").exists()


def test_same_path_same_size_mutation_is_rejected_at_every_dataset_gate(iaa_run):
    _prepare_and_initialize(iaa_run)
    results = Path(iaa_run["results"])
    image_archive = Path(iaa_run["images"])
    original_size = image_archive.stat().st_size

    before = _run("audit_deft_run.py", "--results-dir", results, "--json")
    assert before.returncode == 0, before.stderr
    assert json.loads(before.stdout)["status"] == "IN_PROGRESS"

    _write_tar(image_archive, b"mutated!\n")
    assert image_archive.stat().st_size == original_size
    assert _sha256(image_archive) != iaa_run["images_sha256"]

    audit = _run("audit_deft_run.py", "--results-dir", results, "--json")
    assert audit.returncode == 1
    report = json.loads(audit.stdout)
    assert report["status"] == "INVALID"
    assert any("images_archive changed after initialization" in item for item in report["errors"])

    materialize = _run(
        "run_iaa_stage.py",
        "dataset-materialize",
        "--results-dir",
        results,
        "--deft-config",
        results / "config" / "deft_config.yaml",
    )
    assert materialize.returncode == 2
    assert "images_archive changed after initialization" in materialize.stderr

    commit = _run(
        "commit_stage.py",
        "--results-dir",
        results,
        "--iter-label",
        "baseline",
        "--stage",
        "dataset_setup",
        "--summary",
        "must not commit mutated input",
    )
    assert commit.returncode == 2
    assert "images_archive changed after initialization" in commit.stderr


def test_new_state_and_approval_bind_both_archive_digests(iaa_run):
    _prepare_and_initialize(iaa_run)
    results = Path(iaa_run["results"])
    approval = json.loads((results / "config" / "approval.json").read_text())
    state = json.loads((results / "deft_state.json").read_text())

    assert approval["schema_version"] == "3"
    for field, source in (
        ("images_archive_sha256", "images_sha256"),
        ("metadata_archive_sha256", "metadata_sha256"),
    ):
        assert approval[field] == iaa_run[source]
        assert state["config"][field] == iaa_run[source]


def test_uncommitted_legacy_state_cannot_enter_dataset_setup(iaa_run):
    _prepare_and_initialize(iaa_run)
    results = Path(iaa_run["results"])
    state_path = results / "deft_state.json"
    approval_path = results / "config" / "approval.json"
    state = json.loads(state_path.read_text())
    approval = json.loads(approval_path.read_text())
    for field in ("images_archive_sha256", "metadata_archive_sha256"):
        state["config"].pop(field)
        approval.pop(field)
    approval["schema_version"] = "2"
    approval_path.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n")
    state["config"]["spec_sha256"]["approval.json"] = _sha256(approval_path)
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    audit = _run("audit_deft_run.py", "--results-dir", results, "--json")
    assert audit.returncode == 1
    report = json.loads(audit.stdout)
    assert report["status"] == "INVALID"
    assert any("legacy state does not content-bind" in item for item in report["errors"])
