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
    for split in ("train", "val", "test"):
        unique_name = f"{split}.jpg"
        raw = output / "images_raw" / split / unique_name
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(split.encode())
        rows = [
            {
                "unique_name": unique_name,
                "source_split": split,
                "image_path": unique_name,
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
    assert (output / "images/train.jpg").resolve() == (
        output / "images_raw/train/train.jpg"
    )
    assert (output / "captions/train.txt").read_text() == "train caption\n"
    assert not (metadata / "rebuild.py").exists()
