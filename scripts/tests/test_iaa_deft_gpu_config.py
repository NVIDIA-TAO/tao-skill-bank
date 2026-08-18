# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for IAA DEFT GPU inventory and selection."""

import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
IAA_ROOT = REPO_ROOT / "skills/applications/tao-run-deft-iaa"
PREPARE = IAA_ROOT / "scripts/prepare_deft_config.py"
INITIALIZE = IAA_ROOT / "scripts/init_deft_state.py"
PYT_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt"
DS_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services"


def _base_args(tmp_path: Path, selected: str, visible: str):
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
        "--platform",
        "docker",
        "--max-iterations",
        "1",
        "--num-gpus",
        str(len(selected.split(","))),
        "--gpu-ids",
        selected,
        "--visible-gpu-ids",
        visible,
        "--image-edit-gpu-ids",
        selected.split(",")[0],
        "--vlm-gpu-ids",
        selected.split(",")[0],
        "--llm-gpu-ids",
        selected.split(",")[0],
    ]
    return args, workspace, results, dataset, images, metadata


def test_selected_gpu_shape_is_materialized_once_and_inventory_is_recorded(tmp_path):
    args, workspace, results, dataset, images, metadata = _base_args(
        tmp_path, "0,2", "0,1,2,3,4,5,6,7"
    )
    prepared = subprocess.run(args, check=True, capture_output=True, text=True)
    report = json.loads(prepared.stdout)
    approval = json.loads((results / "config/approval.json").read_text())
    spec = yaml.safe_load((results / "config/tao_spec.yaml").read_text())
    template = yaml.safe_load((IAA_ROOT / "specs/tao_spec.yaml").read_text())

    assert report["gpu_ids"] == [0, 2]
    assert report["visible_gpu_count"] == 8
    assert approval["visible_gpu_ids"] == list(range(8))
    assert spec["train"]["gpu_ids"] == [0, 2]
    assert spec["evaluate"]["gpu_ids"] == [0, 2]
    assert spec["inference"]["gpu_ids"] == [0, 2]
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
            PYT_IMAGE,
            "--ds-image",
            DS_IMAGE,
            "--deft-config",
            str(results / "config/deft_config.yaml"),
            "--tao-spec",
            str(results / "config/tao_spec.yaml"),
            "--sdg-config",
            str(results / "config/sdg_config.yaml"),
            "--platform",
            "docker",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert initialized.returncode == 0
    state = json.loads((results / "deft_state.json").read_text())
    assert state["config"]["visible_gpu_ids"] == list(range(8))
    assert state["config"]["visible_gpu_count"] == 8
    assert state["config"]["gpu_ids"] == [0, 2]


def test_selection_rejects_gpu_absent_from_host_inventory(tmp_path):
    args, *_ = _base_args(tmp_path, "0,4", "0,1,2,3")
    result = subprocess.run(args, capture_output=True, text=True)

    assert result.returncode == 2
    assert "absent from --visible-gpu-ids: 4" in result.stderr
