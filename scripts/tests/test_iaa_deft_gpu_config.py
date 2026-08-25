# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for IAA DEFT GPU inventory and selection."""

import json
import importlib.util
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


def _run_stage_module():
    scripts = IAA_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("iaa_run_stage_gpu_test", scripts / "run_iaa_stage.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _base_args(tmp_path: Path, selected: str, visible: str, *, platform: str = "docker"):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    images = tmp_path / "images_raw.tar"
    metadata = tmp_path / "meta.tar.gz"
    images.write_bytes(b"images")
    metadata.write_bytes(b"metadata")
    results = workspace / "results/run"
    dataset = workspace / "data/iaa"
    distributed = platform in {"slurm", "brev", "kubernetes"}
    image_edit_ids = "0,1,2,3,4,5,6,7" if distributed else selected.split(",")[0]
    vlm_ids = "0" if distributed else selected.split(",")[0]
    llm_ids = "1" if distributed else selected.split(",")[0]
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
        platform,
        "--max-iterations",
        "1",
        "--num-gpus",
        str(len(selected.split(","))),
        "--gpu-ids",
        selected,
        "--visible-gpu-ids",
        visible,
        "--image-edit-gpu-ids",
        image_edit_ids,
        "--vlm-gpu-ids",
        vlm_ids,
        "--llm-gpu-ids",
        llm_ids,
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


def test_remote_platform_does_not_compare_scheduler_ids_to_control_host(tmp_path):
    args, workspace, results, dataset, images, metadata = _base_args(
        tmp_path, "6,7", "0,1,2,3", platform="slurm"
    )
    result = subprocess.run(args, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["gpu_ids"] == [6, 7]
    assert report["visible_gpu_ids"] == [0, 1, 2, 3]

    initialized = subprocess.run(
        [
            sys.executable,
            str(INITIALIZE),
            "--results-dir", str(results),
            "--workspace", str(workspace),
            "--dataset-root", str(dataset),
            "--images-archive", str(images),
            "--metadata-archive", str(metadata),
            "--max-iterations", "1",
            "--pyt-image", PYT_IMAGE,
            "--ds-image", DS_IMAGE,
            "--deft-config", str(results / "config/deft_config.yaml"),
            "--tao-spec", str(results / "config/tao_spec.yaml"),
            "--sdg-config", str(results / "config/sdg_config.yaml"),
            "--platform", "slurm",
        ],
        capture_output=True,
        text=True,
    )
    assert initialized.returncode == 0, initialized.stderr


def test_single_host_brev_materializes_exact_four_one_one_two_gpu_layout(tmp_path):
    args, _, results, *_ = _base_args(
        tmp_path, "6,7", "0,1,2,3,4,5,6,7", platform="brev"
    )
    replacements = {
        "--image-edit-gpu-ids": "0,1,2,3",
        "--vlm-gpu-ids": "4",
        "--llm-gpu-ids": "5",
    }
    for option, value in replacements.items():
        args[args.index(option) + 1] = value

    completed = subprocess.run(args, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    sdg = yaml.safe_load((results / "config/sdg_config.yaml").read_text())
    approval = json.loads((results / "config/approval.json").read_text())
    assert sdg["endpoints"]["gpu_ids"] == {
        "image_edit": [0, 1, 2, 3], "vlm": [4], "llm": [5],
    }
    assert sdg["generation"]["generation_nodes"] == 1
    assert sdg["generation"]["gpus_per_generation_node"] == 4
    assert approval["sdg"]["gpus_per_generation_node"] == 4


def test_single_host_brev_rejects_distributed_worker_gpu_mapping(tmp_path):
    args, *_ = _base_args(
        tmp_path, "6,7", "0,1,2,3,4,5,6,7", platform="brev"
    )
    completed = subprocess.run(args, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "single-host Brev requires --image-edit-gpu-ids 0,1,2,3" in completed.stderr


def test_container_generated_config_uses_cuda_local_ordinals(tmp_path, monkeypatch):
    module = _run_stage_module()
    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump({"train": {"gpu_ids": [6, 7]}, "evaluate": {"gpu_ids": [6, 7]}})
    )
    monkeypatch.setattr(
        module,
        "_state",
        lambda _results: {"config": {"platform": "docker", "num_gpus": 2}},
    )

    module._normalize_generated_gpu_ids(config, tmp_path)

    payload = yaml.safe_load(config.read_text())
    assert payload["train"] == {"gpu_ids": [0, 1], "num_gpus": 2}
    assert payload["evaluate"] == {"gpu_ids": [0, 1], "num_gpus": 2}


def test_virtualenv_generated_config_uses_cuda_local_ordinals(tmp_path, monkeypatch):
    module = _run_stage_module()
    config = tmp_path / "train.yaml"
    original = {"train": {"num_gpus": 2, "gpu_ids": [6, 7]}}
    config.write_text(yaml.safe_dump(original))
    monkeypatch.setattr(
        module,
        "_state",
        lambda _results: {"config": {"platform": "virtualenv", "num_gpus": 2}},
    )

    module._normalize_generated_gpu_ids(config, tmp_path)

    payload = yaml.safe_load(config.read_text())
    assert payload["train"] == {"num_gpus": 2, "gpu_ids": [0, 1]}
