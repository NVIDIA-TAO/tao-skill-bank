# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for DEFT AOI assembled-training CSV validation."""

import csv
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "skills"
    / "applications"
    / "tao-run-deft-aoi"
    / "scripts"
    / "validate_training_csv.py"
)
SPEC = importlib.util.spec_from_file_location("deft_aoi_validate_csv", MODULE_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


FIELDS = ["input_path", "golden_path", "label", "object_name"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _stage_pair(workspace: Path, input_dir: str, golden_dir: str, obj: str) -> None:
    for directory in (input_dir, golden_dir):
        target = workspace / directory / f"{obj}_SolderLight.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"image")


def test_duplicate_sample_identity_is_rejected(tmp_path):
    _stage_pair(tmp_path, "kpi/images/sample", "kpi/images/golden", "C1@1")
    row = {
        "input_path": "kpi/images/sample",
        "golden_path": "kpi/images/golden",
        "label": "PASS",
        "object_name": "C1@1",
    }
    train_csv = tmp_path / "train.csv"
    _write_csv(train_csv, [row, dict(row)])

    errors = validator.validate(train_csv, tmp_path)

    assert any("duplicate sample row" in error for error in errors)


def test_leakage_normalizes_base_and_workspace_path_coordinates(tmp_path):
    _stage_pair(tmp_path, "kpi/images/sample", "kpi/images/golden", "C1@1")
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    _write_csv(
        train_csv,
        [{
            "input_path": "kpi/images/sample",
            "golden_path": "kpi/images/golden",
            "label": "PASS",
            "object_name": "C1@1",
        }],
    )
    _write_csv(
        validation_csv,
        [{
            "input_path": "sample",
            "golden_path": "golden",
            "label": "PASS",
            "object_name": "C1@1",
        }],
    )

    errors = validator.validate(train_csv, tmp_path, validation_csv)

    assert any("train/val leak" in error for error in errors)
