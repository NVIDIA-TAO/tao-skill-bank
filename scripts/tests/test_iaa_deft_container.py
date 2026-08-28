# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for IAA DEFT GPU scoping and Docker execution evidence."""

import argparse
import hashlib
import json
import shutil
import sys
import time
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
from command_contract import command_sha256, expected_container_command  # noqa: E402

IAA_ROOT = IAA_SCRIPTS.parent


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


def test_interrupted_wrapper_reconciles_completed_auto_removed_container(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    results = workspace / "results/run"
    config_dir = results / "config"
    dataset = workspace / "data/iaa"
    config_dir.mkdir(parents=True)
    dataset.mkdir(parents=True)

    hashes = {}
    for name in container.RUN_SPEC_NAMES:
        destination = config_dir / name
        if name == "approval.json":
            destination.write_text("{}\n")
        else:
            shutil.copyfile(IAA_ROOT / "specs" / name, destination)
        hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()

    state_config = {
        "workspace": str(workspace),
        "dataset_root": str(dataset),
        "config_dir": str(config_dir),
        "deft_config": str(config_dir / "deft_config.yaml"),
        "tao_spec": str(config_dir / "tao_spec.yaml"),
        "spec_sha256": hashes,
        "platform": "docker",
        "pyt_image": container.PINNED_IMAGES["pyt"],
        "ds_image": container.PINNED_IMAGES["ds"],
        "num_gpus": 1,
        "gpu_ids": [0],
        "requires_hf_token": False,
        "mining_topn": 25,
        "knn_metric": "cosine",
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "deft_state.json").write_text(
        json.dumps(
            {
                "schema_version": "3",
                "workflow": "tao-run-deft-iaa",
                "results_dir": str(results),
                "max_iterations": 1,
                "config": state_config,
            }
        )
    )

    stage = results / "embeddings/source"
    stage.mkdir(parents=True)
    output = stage / "embeddings.parquet"
    output.write_bytes(b"completed parquet")
    log = stage / "pool_embed.log"
    log.write_text(
        "work complete\nExecution status: PASS\n" + "trailing diagnostics\n" * 6000
    )
    command = expected_container_command("pool_embed", "baseline", state_config)
    identity = hashlib.sha256(
        f"{results}\0{stage.relative_to(results)}\0pool_embed".encode()
    ).hexdigest()[:20]
    container_name = f"tao-deft-{identity}"
    started_ns = time.time_ns() - 1_000_000_000
    status_path = stage / "pool_embed.status.json"
    status_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "workflow": "tao-run-deft-iaa",
                "kind": "container",
                "name": "pool_embed",
                "attempt": 1,
                "image_kind": "ds",
                "image": container.PINNED_IMAGES["ds"],
                "command": command,
                "command_sha256": command_sha256(command),
                "passed_hf_token": False,
                "container_name": container_name,
                "cidfile": str(stage / "pool_embed.cid"),
                "started_at": "2026-08-13T00:00:00+00:00",
                "started_ns": started_ns,
                "finished_at": None,
                "status": "running",
                "exit_code": None,
                "log_path": str(log),
                "fresh_outputs": [str(output)],
            }
        )
    )
    args = argparse.Namespace(
        results_dir=results,
        image="ds",
        stage_dir=stage,
        name="pool_embed",
        pass_hf_token=False,
        fresh_output=[output],
        command=command,
    )

    monkeypatch.setattr(container, "_container_is_running", lambda _: False)
    monkeypatch.setattr(container, "_container_exit_code", lambda _: None)

    def unexpected_launch(*_args, **_kwargs):
        raise AssertionError("reconciliation must not launch a second container")

    monkeypatch.setattr(container.subprocess, "run", unexpected_launch)
    _, _, returncode = container.run(args)

    reconciled = json.loads(status_path.read_text())
    assert returncode == 0
    assert output.read_bytes() == b"completed parquet"
    assert reconciled["attempt"] == 1
    assert reconciled["status"] == "ok"
    assert reconciled["exit_code"] == 0
    assert reconciled["reconciled_after_wrapper_exit"] is True
    assert reconciled["reconciliation_source"] == "container_log"


def test_complete_log_scan_returns_the_final_status_beyond_tail_boundaries(tmp_path):
    log = tmp_path / "stage.log"
    log.write_text(
        "Execution status: PASS\n"
        + "later diagnostic output\n" * 6000
        + "Execution status: FAIL\n"
    )

    assert log.stat().st_size > 64 * 1024
    assert container._last_execution_status(log) == "FAIL"  # noqa: SLF001


def test_interrupted_wrapper_does_not_trust_pass_log_without_fresh_output(
    tmp_path, monkeypatch
):
    log = tmp_path / "stage.log"
    log.write_text("Execution status: PASS\n")
    monkeypatch.setattr(container, "_container_exit_code", lambda _: None)

    reconciled = container._reconcile_interrupted_success(  # noqa: SLF001
        {"started_ns": time.time_ns() - 1_000_000_000},
        status_path=tmp_path / "stage.status.json",
        log_path=log,
        fresh_outputs=[str(tmp_path / "missing.parquet")],
        container_name="removed-container",
    )

    assert reconciled is False
    assert not (tmp_path / "stage.status.json").exists()
