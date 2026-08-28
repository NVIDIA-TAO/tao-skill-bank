# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for PAS dataset-layout transparency."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
IAA_SCRIPTS = (
    REPO_ROOT / "skills" / "applications" / "tao-run-deft-iaa" / "scripts"
)
sys.path.insert(0, str(IAA_SCRIPTS))

from iaa_deft.dataset_layout import report_dataset_layout  # noqa: E402


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "pas"
    images = root / "images" / "camera_a"
    captions = root / "captions" / "camera_a"
    images.mkdir(parents=True)
    captions.mkdir(parents=True)
    (images / "one.JPG").write_bytes(b"one")
    (images / "two.png").write_bytes(b"two")
    (captions / "one.txt").write_text("red shirt\n", encoding="utf-8")
    (captions / "two.txt").write_text("blue shirt\n", encoding="utf-8")
    (root / "train_pairs.json").write_text(
        json.dumps(
            [
                {
                    "image_path": "images/camera_a/one.JPG",
                    "caption": "red shirt",
                    "query_type": "easy",
                },
                {
                    "image_path": "images/camera_a/two.png",
                    "caption": "blue shirt",
                    "query_type": "hard",
                },
            ]
        ),
        encoding="utf-8",
    )
    (root / "val_pairs.json").write_text(
        '[\n{"image_path":"images/camera_a/one.JPG","caption":"red","query_type":"easy"},\n'
        '{"image_path":"images/camera_a/two.png","caption":"blue","query_type":"easy"}\n]\n',
        encoding="utf-8",
    )
    (root / "manifest.json").write_text('{"version": 1}', encoding="utf-8")
    (root / "broken.json").write_text("{", encoding="utf-8")
    return root


def test_report_exposes_notebook_dataset_components(tmp_path, capsys):
    root = _dataset(tmp_path)

    report = report_dataset_layout(root)

    output = capsys.readouterr().out
    assert "Crops (image directories): 1" in output
    assert "images/camera_a: 2 images" in output
    assert "Queries (pairs/query JSON files): 2" in output
    assert "train_pairs.json: 2 rows (easy=1, hard=1)" in output
    assert "val_pairs.json: 2 rows (easy=2)" in output
    assert "Attribute metadata (caption/.txt directories): 1" in output
    assert "captions/camera_a: 2 .txt files" in output
    assert len(report["image_dirs"]) == 1
    assert len(report["query_files"]) == 2
    assert len(report["text_dirs"]) == 1


def test_report_cli_is_read_only_and_rejects_invalid_bounds(tmp_path):
    root = _dataset(tmp_path)
    before = sorted(
        (path.relative_to(root), path.stat().st_mtime_ns)
        for path in root.rglob("*")
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(IAA_SCRIPTS)

    result = subprocess.run(
        [sys.executable, "-m", "iaa_deft.dataset_layout", str(root)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "Dataset layout report" in result.stdout
    after = sorted(
        (path.relative_to(root), path.stat().st_mtime_ns)
        for path in root.rglob("*")
    )
    assert after == before
    with pytest.raises(ValueError, match="max_depth"):
        report_dataset_layout(root, max_depth=-1)
    with pytest.raises(ValueError, match="top_n"):
        report_dataset_layout(root, top_n=0)
