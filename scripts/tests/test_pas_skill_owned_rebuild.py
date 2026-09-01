# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for NVBug 6597479's skill-owned PAS builder."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / "skills" / "applications" / "tao-run-deft-pas"
REBUILD = SKILL_ROOT / "scripts" / "rebuild.py"


def _write_export(root: Path, *, unsafe_image_path: str | None = None) -> None:
    for split in ("train", "val", "test"):
        unique_name = f"{split}_00000000.jpg"
        source_split = "eval" if split == "test" else split
        image_path = unsafe_image_path or f"images/source/{split}.jpg"
        source = root / "images_raw" / source_split / "images" / "source" / f"{split}.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"representative-image")
        row = {
            "unique_name": unique_name,
            "source_split": source_split,
            "image_path": image_path,
            "caption": f"caption for {split}",
        }
        (root / f"{split}_pairs.json").write_text(
            json.dumps([row]), encoding="utf-8"
        )
        (root / f"{split}_list.txt").write_text(
            unique_name + "\n", encoding="utf-8"
        )


def _run(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REBUILD),
            "--dataset-root",
            str(root),
            "--workers",
            "1",
            *extra,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_builder_materializes_export_without_archive_executable(tmp_path: Path) -> None:
    _write_export(tmp_path)
    assert not (tmp_path / "rebuild.py").exists()

    completed = _run(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "VERIFY: PASS" in completed.stdout
    for split in ("train", "val", "test"):
        link = tmp_path / "images" / f"{split}_00000000.jpg"
        assert link.is_symlink()
        source_split = "eval" if split == "test" else split
        assert os.readlink(link) == (
            f"../images_raw/{source_split}/images/source/{split}.jpg"
        )
        assert (tmp_path / "captions" / f"{split}_00000000.txt").read_text(
            encoding="utf-8"
        ) == f"caption for {split}\n"

    resumed = _run(tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert "existing=        1" in resumed.stdout
    assert "VERIFY: PASS" in resumed.stdout


def test_documented_extraction_ignores_legacy_archive_builder(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    _write_export(export)
    (export / "rebuild.py").write_text(
        "from pathlib import Path\nPath('legacy-builder-ran').write_text('bad')\n",
        encoding="utf-8",
    )
    archive = tmp_path / "meta.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(export.glob("*_pairs.json")) + sorted(
            export.glob("*_list.txt")
        ):
            handle.add(path, arcname=path.name)
        handle.add(export / "rebuild.py", arcname="rebuild.py")

    dataset_root = tmp_path / "materialized"
    shutil.copytree(export / "images_raw", dataset_root / "images_raw")
    extracted = subprocess.run(
        [
            "tar",
            "-xzf",
            str(archive),
            "-C",
            str(dataset_root),
            "--exclude=rebuild.py",
            "--exclude=./rebuild.py",
            "--exclude=*/rebuild.py",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert extracted.returncode == 0, extracted.stderr
    assert not (dataset_root / "rebuild.py").exists()

    completed = _run(dataset_root)

    assert completed.returncode == 0, completed.stderr
    assert "VERIFY: PASS" in completed.stdout
    assert not (dataset_root / "legacy-builder-ran").exists()


def test_builder_rejects_conflicting_existing_output(tmp_path: Path) -> None:
    _write_export(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    os.symlink("../images_raw/train/wrong.jpg", images / "train_00000000.jpg")

    completed = _run(tmp_path)

    assert completed.returncode == 2
    assert "existing image entry does not match PAS metadata" in completed.stderr


def test_builder_rejects_symlinked_caption_output(tmp_path: Path) -> None:
    _write_export(tmp_path)
    captions = tmp_path / "captions"
    captions.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("must remain unchanged\n", encoding="utf-8")
    os.symlink(outside, captions / "train_00000000.txt")

    completed = _run(tmp_path)

    assert completed.returncode == 2
    assert "unsafe existing caption output" in completed.stderr
    assert outside.read_text(encoding="utf-8") == "must remain unchanged\n"


@pytest.mark.parametrize(
    "unsafe_path",
    ("../outside.jpg", "/absolute.jpg", "images//double.jpg", "images\\windows.jpg"),
)
def test_builder_rejects_unsafe_metadata_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    _write_export(tmp_path, unsafe_image_path=unsafe_path)

    completed = _run(tmp_path)

    assert completed.returncode == 2
    assert "image_path must" in completed.stderr
    assert not (tmp_path / "images").exists()


def test_dataset_contract_executes_only_the_skill_owned_builder() -> None:
    reference = (SKILL_ROOT / "references" / "data-layout.md").read_text(
        encoding="utf-8"
    )

    assert '"$SKILL_ROOT/scripts/rebuild.py"' in reference
    assert '"$DATASET_ROOT/rebuild.py"' not in reference
    assert "--exclude='rebuild.py'" in reference
    assert "meta.tar.gz` | yes | pair/list metadata and README/vocabulary files" in reference
