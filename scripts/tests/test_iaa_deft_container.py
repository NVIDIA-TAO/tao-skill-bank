# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for IAA DEFT Docker execution evidence."""

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
from command_contract import command_sha256, expected_container_command  # noqa: E402

IAA_ROOT = IAA_SCRIPTS.parent


def test_docker_gpu_args_use_exact_approved_devices():
    assert container._docker_gpu_args(  # noqa: SLF001
        {"gpu_ids": [0, 2], "num_gpus": 2}
    ) == ["--gpus", '"device=0,2"']


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


def test_mining_template_has_no_competing_runtime_defaults():
    mining = yaml.safe_load((IAA_ROOT / "specs/mining_spec.yaml").read_text())

    assert mining["topn"] == "???"
    assert mining["knn_metric"] == "???"


def test_embedding_specs_name_the_same_siglip2_model():
    models = {
        yaml.safe_load((IAA_ROOT / "specs" / name).read_text())["model"]
        for name in ("image_embed_spec.yaml", "text_embed_spec.yaml")
    }
    assert models == {"SigLIP2"}


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
    log.write_text("work complete\nExecution status: PASS\n")
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


def test_metric_contract_uses_the_pinned_containers_pas_filenames():
    relevant_suffixes = {".py", ".md", ".json"}
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in IAA_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in relevant_suffixes
        and "__pycache__" not in path.parts
    )

    assert "nvidia_iaa_metrics" not in text
    assert "nvidia_pas_metrics.csv" in text
    assert "nvidia_pas_metrics_aggregate.csv" in text
