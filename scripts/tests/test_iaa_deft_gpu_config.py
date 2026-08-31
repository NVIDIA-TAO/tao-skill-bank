# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for IAA DEFT GPU-shape materialization."""

import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
IAA_ROOT = REPO_ROOT / "skills/applications/tao-run-deft-iaa"
PREPARE = IAA_ROOT / "scripts/prepare_deft_config.py"
INITIALIZE = IAA_ROOT / "scripts/init_deft_state.py"


def _base_args(tmp_path: Path, selected: str):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    images = tmp_path / "images_raw.tar"
    metadata = tmp_path / "meta.tar.gz"
    images.write_bytes(b"images")
    metadata.write_bytes(b"metadata")
    results = workspace / "results/run"
    dataset = workspace / "data/iaa"
    args = [
        sys.executable,
        str(PREPARE),
        "--workspace",
        str(workspace),
        "--results-dir",
        str(results),
        "--dataset-root",
        str(dataset),
        "--images-archive",
        str(images),
        "--metadata-archive",
        str(metadata),
        "--max-iterations",
        "1",
        "--num-gpus",
        str(len(selected.split(","))),
        "--gpu-ids",
        selected,
    ]
    return args, workspace, results, dataset, images, metadata


def test_selected_host_gpu_shape_is_materialized_as_dense_container_ordinals(tmp_path):
    args, workspace, results, dataset, images, metadata = _base_args(
        tmp_path, "2,3"
    )
    prepared = subprocess.run(args, check=True, capture_output=True, text=True)
    report = json.loads(prepared.stdout)
    approval = json.loads((results / "config/approval.json").read_text())
    spec = yaml.safe_load((results / "config/tao_spec.yaml").read_text())
    template = yaml.safe_load((IAA_ROOT / "specs/tao_spec.yaml").read_text())

    assert report["gpu_ids"] == [2, 3]
    assert report["container_gpu_ids"] == [0, 1]
    assert approval["host_gpu_ids"] == [2, 3]
    assert approval["container_gpu_ids"] == [0, 1]
    assert spec["train"]["gpu_ids"] == [0, 1]
    assert spec["evaluate"]["gpu_ids"] == [0, 1]
    assert spec["inference"]["gpu_ids"] == [0, 1]
    assert spec["inference"]["num_gpus"] == 2
    assert template["train"]["num_gpus"] == "???"
    assert template["evaluate"]["gpu_ids"] == "???"
    assert template["inference"]["num_gpus"] == "???"

    initialized = subprocess.run(
        [
            sys.executable,
            str(INITIALIZE),
            "--results-dir",
            str(results),
            "--workspace",
            str(workspace),
            "--dataset-root",
            str(dataset),
            "--images-archive",
            str(images),
            "--metadata-archive",
            str(metadata),
            "--max-iterations",
            "1",
            "--pyt-image",
            approval["pyt_image"],
            "--ds-image",
            approval["ds_image"],
            "--deft-config",
            str(results / "config/deft_config.yaml"),
            "--tao-spec",
            str(results / "config/tao_spec.yaml"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert initialized.returncode == 0
    state = json.loads((results / "deft_state.json").read_text())
    assert state["config"]["gpu_ids"] == [2, 3]
    assert state["config"]["container_gpu_ids"] == [0, 1]


def test_selection_rejects_duplicate_host_gpu_ids(tmp_path):
    args, *_ = _base_args(tmp_path, "2,2")
    result = subprocess.run(args, capture_output=True, text=True)

    assert result.returncode == 2
    assert "distinct non-negative IDs" in result.stderr
