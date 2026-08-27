# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the skill-owned IAA dataset rebuild."""

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REBUILD = (
    REPO_ROOT
    / "skills/applications/tao-run-deft-iaa/scripts/iaa_deft/rebuild.py"
)


def test_bundled_rebuild_uses_export_metadata_without_export_code(tmp_path):
    metadata = tmp_path / "metadata"
    output = tmp_path / "dataset"
    metadata.mkdir()
    decoy = metadata / "rebuild.py"
    decoy.write_text("raise RuntimeError('export-owned code was executed')\n")
    for split in ("train", "val", "test"):
        unique_name = f"nested/{split}.jpg" if split == "train" else f"{split}.jpg"
        image_path = f"source/{split}.jpg"
        raw = output / "images_raw" / split / image_path
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(split.encode())
        rows = [
            {
                "unique_name": unique_name,
                "source_split": split,
                "image_path": image_path,
                "caption": f"{split} caption",
            }
        ]
        (metadata / f"{split}_pairs.json").write_text(json.dumps(rows))
        (metadata / f"{split}_list.txt").write_text(unique_name + "\n")

    result = subprocess.run(
        [
            sys.executable,
            str(REBUILD),
            "--metadata-root",
            str(metadata),
            "--out",
            str(output),
            "--workers",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "VERIFY: PASS" in result.stdout
    assert (output / "images/nested/train.jpg").resolve() == (
        output / "images_raw/train/source/train.jpg"
    )
    assert (output / "captions/nested/train.txt").read_text() == "train caption\n"
    assert decoy.read_text() == "raise RuntimeError('export-owned code was executed')\n"

    subset_verify = subprocess.run(
        [
            sys.executable,
            str(REBUILD),
            "--metadata-root",
            str(metadata),
            "--out",
            str(output),
            "--workers",
            "1",
            "--splits",
            "train",
            "--verify-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "VERIFY: PASS" in subset_verify.stdout


def test_rebuild_refuses_metadata_paths_that_escape_owned_roots(tmp_path):
    metadata = tmp_path / "metadata"
    output = tmp_path / "dataset"
    metadata.mkdir()
    raw = output / "images_raw/train/source.jpg"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"image")
    (metadata / "train_pairs.json").write_text(
        json.dumps(
            [
                {
                    "unique_name": "../escape.jpg",
                    "source_split": "train",
                    "image_path": "source.jpg",
                    "caption": "caption",
                }
            ]
        )
    )
    (metadata / "train_list.txt").write_text("../escape.jpg\n")

    result = subprocess.run(
        [
            sys.executable,
            str(REBUILD),
            "--metadata-root",
            str(metadata),
            "--out",
            str(output),
            "--workers",
            "1",
            "--splits",
            "train",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must be a non-empty relative path" in result.stderr
    assert not (output / "escape.jpg").exists()
