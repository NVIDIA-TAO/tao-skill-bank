# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

import pytest
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "skills" / "applications" / "tao-run-deft-iaa" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "iaa_airflow_sdg_action", SCRIPT_DIR / "airflow_sdg_action.py"
)
assert SPEC and SPEC.loader
SDG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SDG)


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, *, nodes: int = 3):
    shared = tmp_path / "airflow" / "shared"
    workspace = shared / "workspace"
    results = workspace / "results" / "run-01"
    dataset = workspace / "data" / "iaa"
    config_dir = results / "config"
    stage = results / "iter_1" / "datagen"
    runtime = stage / ".tao-runtime" / "controller"
    for directory in (
        config_dir, results / "iter_1" / "mining", results / "iter_1" / "gaps",
        results / "iaa_splits", dataset, runtime / "iaa_deft", shared / "cache",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    mined = results / "iter_1" / "mining" / "mined_pairs.json"
    gaps = results / "iter_1" / "gaps" / "kpi_gaps.parquet"
    eval_list = results / "iaa_splits" / "eval_list.txt"
    eval_pairs = results / "iaa_splits" / "eval_pairs.json"
    vocab = dataset / "attribute_vocab.json"
    mined.write_text('[{"unique_name":"source.jpg"}]\n')
    gaps.write_bytes(b"parquet")
    eval_list.write_text("eval.jpg\n")
    eval_pairs.write_text('[{"unique_name":"eval.jpg"}]\n')
    vocab.write_text('{"attributes":["shoe color"]}\n')
    (runtime / "run_sdg_stage.py").write_text("# staged runtime\n")
    (runtime / "iaa_deft" / "impl.py").write_text("VALUE = 1\n")
    runtime_digest = SDG._python_tree_sha256(runtime / "iaa_deft")
    images = {
        "augmentation": "nvcr.io/nvstaging/tao/augmentation:1@sha256:" + "a" * 64,
        "auto_labeling": "nvcr.io/nvstaging/tao/auto-labeling:1@sha256:" + "b" * 64,
        "image_edit_serving": "vllm/vllm-omni:1@sha256:" + "c" * 64,
        "text_serving": "vllm/vllm-openai:1@sha256:" + "d" * 64,
    }
    models = {
        "image_edit": {
            "id": "Qwen/Qwen-Image-Edit-2511", "revision": "1" * 40,
            "backend": "vllm-omni", "port": 18102, "min_vram_mib": 38000,
        },
        "vlm": {
            "id": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", "revision": "2" * 40,
            "backend": "vllm", "port": 18100, "min_vram_mib": 52000,
        },
        "llm": {
            "id": "Qwen/Qwen2.5-14B-Instruct", "revision": "3" * 40,
            "backend": "vllm", "port": 18101, "min_vram_mib": 28000,
        },
    }
    local = nodes == 1
    endpoint_gpu_ids = (
        {"image_edit": [0, 1, 2, 3], "vlm": [4], "llm": [5]}
        if local else {"image_edit": list(range(8)), "vlm": [0], "llm": [1]}
    )
    tao_gpu_ids = [6, 7] if local else [0, 1]
    config = {
        "schema_version": "1", "enabled": True, "images": images, "models": models,
        "endpoints": {
            "ownership": "managed", "reuse_requested": False,
            "startup_timeout_s": 600, "request_timeout_s": 180,
            "retry_interval_s": 15, "cache_dir": str(shared / "cache"),
            "gpu_ids": endpoint_gpu_ids,
        },
        "generation": {
            "generation_nodes": nodes, "gpus_per_generation_node": 4 if local else 8,
            "image_edit_request_timeout_s": 600, "verification_max_attempts": 2,
            "max_samples_per_iteration": 10,
        },
    }
    config_path = config_dir / "sdg_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    digest = _sha(config_path)
    state = {
        "schema_version": "3", "workflow": SDG.WORKFLOW,
        "started_at": "2026-08-22T00:00:00+00:00",
        "results_dir": str(results), "max_iterations": 3, "current_iteration": 1,
        "gate_met": False, "loop_stop_reason": None,
        "active_runtime_sha256": runtime_digest,
        "config": {
            "platform": "airflow", "dataset_root": str(dataset),
            "sdg_config": str(config_path), "sdg_config_sha256": digest,
            "spec_sha256": {"sdg_config.yaml": digest},
            "iaa_deft_bundle_sha256": runtime_digest,
            "requires_hf_token": True, "gpu_ids": tao_gpu_ids,
            "ds_image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
            "sdg": {
                "endpoint_mode": "managed", "reuse_requested": False,
                "generation_nodes": nodes, "gpus_per_generation_node": 4 if local else 8,
                "gpu_ids": config["endpoints"]["gpu_ids"],
                "models": models, "images": images,
            },
        },
        "iterations": {
            "baseline": {"status": "complete"},
            "iter1": {
                "status": "in_progress", "stage_completed": "history_select",
                "mined_pairs": str(mined),
            },
        },
    }
    state_path = results / "deft_state.json"
    state_path.write_text(json.dumps(state))
    state_dir = tmp_path / "tao-state"
    (state_dir / "jobs").mkdir(parents=True)
    monkeypatch.setenv("TAO_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TAO_IAA_AIRFLOW_SHARED_ROOT", str(shared))
    monkeypatch.setenv("AIRFLOW_BASE_URL", "https://airflow.example.test")
    monkeypatch.setenv("AIRFLOW_API_TOKEN", "airflow-secret")
    return argparse.Namespace(
        deft_state=state_path, sdg_config=config_path, iteration=1,
        runtime_root=runtime, cpu_pool="iaa-cpu", tao_gpu_pool="iaa-tao-gpu",
        image_worker_pool="iaa-image-workers", coordinator_pool="iaa-coordinator",
        dag_timeout_s=14400, output=stage / "airflow_sdg.action.json",
    ), state_dir


def test_prepare_is_deterministic_and_binds_three_independent_workers(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch, nodes=3)
    first = SDG.prepare_request(args)
    second = SDG.prepare_request(args)
    assert first["status"] == "created"
    assert second["status"] == "reused"
    request = first["payload"]
    assert request["generation_nodes"] == 3
    assert request["resources"]["gpus_per_image_worker"] == 8
    assert request["resources"]["image_worker_capacity"] == 8
    assert request["resources"]["vlm_gpus"] == 1
    assert request["resources"]["llm_gpus"] == 1
    assert request["resources"]["tao_gpus"] == 2
    assert request["request_sha256"] == SDG._canonical_sha256(request)
    assert len(request["expected_outputs"]) == 6
    assert "job_state_dir" not in request


def test_prepare_binds_single_host_four_one_one_two_gpu_layout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch, nodes=1)
    request = SDG.prepare_request(args)["payload"]
    assert request["generation_nodes"] == 1
    assert request["resources"]["gpus_per_image_worker"] == 4
    assert request["resources"]["image_edit_gpu_ids"] == [0, 1, 2, 3]
    assert request["resources"]["vlm_gpu_ids"] == [4]
    assert request["resources"]["llm_gpu_ids"] == [5]
    assert request["resources"]["tao_gpu_ids"] == [6, 7]


def test_prepare_rejects_partial_image_worker_gpu_shape(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch)
    config = yaml.safe_load(args.sdg_config.read_text())
    config["endpoints"]["gpu_ids"]["image_edit"] = list(range(7))
    args.sdg_config.write_text(yaml.safe_dump(config, sort_keys=True))
    state = json.loads(args.deft_state.read_text())
    digest = _sha(args.sdg_config)
    state["config"]["sdg_config_sha256"] = digest
    state["config"]["spec_sha256"]["sdg_config.yaml"] = digest
    state["config"]["sdg"]["gpu_ids"] = config["endpoints"]["gpu_ids"]
    args.deft_state.write_text(json.dumps(state))
    with pytest.raises(SDG.ContractError, match="topology is incomplete"):
        SDG.prepare_request(args)


def test_submit_binds_pending_job_and_keeps_credentials_out_of_conf(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, state_dir = _fixture(tmp_path, monkeypatch)
    prepared = SDG.prepare_request(args)
    request = prepared["payload"]
    job_path = state_dir / "jobs" / "deft-iaa-sdg-job.json"
    job_path.write_text(json.dumps({
        "id": "deft-iaa-sdg-job", "platform": "airflow",
        "image": request["images"]["controller"], "network_arch": "iaa-sdg",
        "action": request["action_id"], "results_dir": request["paths"]["stage_dir"],
        "backend_ref": None, "terminal_state": None,
        "transitions": [{"state": "PENDING"}],
    }))
    calls = []

    def fake_request(self, method, path, payload=None, **kwargs):
        calls.append((method, path, payload))
        if method == "GET" and "/pools/" in path:
            name = path.rsplit("/", 1)[-1]
            return {"name": name, "slots": 8, "open_slots": 8}
        if method == "GET":
            return {
                "dag_id": SDG.airflow.DEFAULT_DAG_ID, "is_paused": False,
                "tags": [SDG.airflow.CONTRACT],
            }
        return {"state": "queued"}

    monkeypatch.setattr(SDG.airflow.AirflowClient, "_request", fake_request)
    assert SDG.submit(argparse.Namespace(
        request=pathlib.Path(prepared["request"]), job_record=job_path,
    )) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PENDING"
    binding = pathlib.Path(request["job_binding_path"])
    assert binding.is_file()
    post = [call for call in calls if call[0] == "POST"]
    assert len(post) == 1
    assert "airflow-secret" not in json.dumps(post[0][2], sort_keys=True)
    assert post[0][2]["conf"]["kind"] == SDG.KIND
    assert post[0][2]["logical_date"] is None


def test_request_rejects_credential_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch)
    request = SDG.prepare_request(args)["payload"]
    request["resources"]["cpu_pool"] = "airflow-secret"
    request["request_sha256"] = SDG._canonical_sha256(request)
    with pytest.raises(SDG.airflow.AirflowContractError, match="credential value"):
        SDG.validate_request(request)
