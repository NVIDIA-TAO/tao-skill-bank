# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for IAA DEFT host/container GPU scoping."""

import json
import sys
from pathlib import Path

import pytest
import yaml

IAA_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "applications"
    / "tao-run-deft-iaa"
    / "scripts"
)
sys.path.insert(0, str(IAA_SCRIPTS))
import run_deft_container as container  # noqa: E402
import init_deft_state as state  # noqa: E402
import prepare_deft_config as prepare  # noqa: E402


def test_docker_gpu_args_use_exact_approved_devices():
    assert container._docker_gpu_args(  # noqa: SLF001
        {"gpu_ids": [0, 2], "num_gpus": 2}
    ) == ["--gpus", '"device=0,2"']


def test_nonzero_host_devices_become_dense_container_ordinals(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    results = workspace / "results" / "run"
    dataset = workspace / "data" / "iaa"
    images_archive = tmp_path / "images_raw.tar"
    metadata_archive = tmp_path / "meta.tar.gz"
    images_archive.write_bytes(b"images")
    metadata_archive.write_bytes(b"metadata")

    assert prepare.main(
        [
            "--workspace",
            str(workspace),
            "--results-dir",
            str(results),
            "--dataset-root",
            str(dataset),
            "--images-archive",
            str(images_archive),
            "--metadata-archive",
            str(metadata_archive),
            "--platform",
            "docker",
            "--max-iterations",
            "1",
            "--num-gpus",
            "2",
            "--gpu-ids",
            "1,3",
        ]
    ) == 0

    tao_spec = yaml.safe_load((results / "config" / "tao_spec.yaml").read_text())
    approval = json.loads((results / "config" / "approval.json").read_text())
    assert tao_spec["train"]["gpu_ids"] == [0, 1]
    assert tao_spec["evaluate"]["gpu_ids"] == [0, 1]
    assert approval["host_gpu_ids"] == [1, 3]
    assert approval["container_gpu_ids"] == [0, 1]

    assert state.main(
        [
            "--workspace",
            str(workspace),
            "--results-dir",
            str(results),
            "--dataset-root",
            str(dataset),
            "--images-archive",
            str(images_archive),
            "--metadata-archive",
            str(metadata_archive),
            "--max-iterations",
            "1",
            "--platform",
            "docker",
            "--pyt-image",
            prepare.PINNED_PYT_IMAGE,
            "--ds-image",
            prepare.PINNED_DS_IMAGE,
            "--deft-config",
            str(results / "config" / "deft_config.yaml"),
            "--tao-spec",
            str(results / "config" / "tao_spec.yaml"),
        ]
    ) == 0
    persisted = json.loads((results / "deft_state.json").read_text())
    assert persisted["config"]["gpu_ids"] == [1, 3]
    assert persisted["config"]["container_gpu_ids"] == [0, 1]
    assert container._docker_gpu_args(persisted["config"]) == [  # noqa: SLF001
        "--gpus",
        '"device=1,3"',
    ]


@pytest.mark.parametrize("value", [None, "0,1", [], [True], [0, 0], [-1]])
def test_approval_gpu_namespaces_reject_malformed_values(value):
    with pytest.raises(ValueError):
        state._approval_gpu_ids(value, "host_gpu_ids")  # noqa: SLF001


@pytest.mark.parametrize(
    "config",
    [
        {"gpu_ids": [], "num_gpus": 0},
        {"gpu_ids": [0, 0], "num_gpus": 2},
        {"gpu_ids": [0, 2], "num_gpus": 1},
        {"gpu_ids": [True], "num_gpus": 1},
    ],
)
def test_docker_gpu_args_reject_invalid_or_mismatched_state(config):
    with pytest.raises(ValueError):
        container._docker_gpu_args(config)  # noqa: SLF001
