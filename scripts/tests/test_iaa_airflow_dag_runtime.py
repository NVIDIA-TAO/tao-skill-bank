# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "skills" / "applications" / "tao-run-deft-iaa" / "scripts"
    / "airflow_dag_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("iaa_airflow_dag_runtime", SCRIPT)
assert SPEC and SPEC.loader
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def _standard_conf(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    shared = tmp_path / "shared"
    stage = shared / "results" / "run-1" / "iter_1" / "gaps"
    mount = shared / "results" / "run-1"
    jobs = shared / "state" / "jobs"
    for directory in (stage, mount, jobs):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TAO_IAA_AIRFLOW_SHARED_ROOT", str(shared))
    request_path = stage / "gap_analysis.attempt-1.action.json"
    binding_path = stage / "gap_analysis.attempt-1.job-binding.json"
    log_path = stage / "gap_analysis.attempt-1.log"
    output = stage / "kpi_gaps.parquet"
    job_path = jobs / "deft-iaa-gap-analysis-test.json"
    request = {
        "schema_version": "1", "workflow": RUNTIME.WORKFLOW, "platform": "airflow",
        "name": "gap_analysis", "attempt": 1, "label": "iter1", "runtime_sha256": "a" * 64,
        "record_image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
        "workload_image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
        "gpu_ids": [6, 7], "forward_env": [], "environment": {"HOME": "/tmp"},
        "spec_bundle": {
            "network_arch": "iaa-adapter", "action": "deft-iaa-gap-analysis-test",
            "image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
            "mode": "args", "command": "python3", "args": ["-c", "pass"],
            "compute_shape": {"gpus": 2, "nodes": 1},
        },
        "mounts": [{"source": str(mount), "target": "/results", "read_only": False}],
        "job_binding_path": str(binding_path), "log_path": str(log_path),
        "fresh_outputs": [str(output)], "started_ns": 1,
    }
    request["request_sha256"] = RUNTIME._canonical_sha256(request, "request_sha256")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    job = {
        "schema_version": "1", "id": "deft-iaa-gap-analysis-test", "platform": "airflow",
        "image": request["record_image"], "network_arch": "iaa-adapter",
        "action": request["spec_bundle"]["action"], "results_dir": str(stage),
        "storage_tier": "C", "upload_excludes": [".tao-runtime/"],
        "submitted_at": "2026-08-22T00:00:00+00:00",
    }
    job_path.write_text(json.dumps(job), encoding="utf-8")
    job_identity = {field: job.get(field) for field in RUNTIME.JOB_IDENTITY_FIELDS}
    binding = {
        "request_path": str(request_path), "job_record_path": str(job_path),
        "job_id": job["id"], "results_scope": str(stage),
        "staging_receipt_sha256": "b" * 64,
        "job_identity_sha256": hashlib.sha256(
            json.dumps(job_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    binding["binding_sha256"] = RUNTIME._canonical_sha256(binding, "binding_sha256")
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    conf = {
        "contract": RUNTIME.CONTRACT, "job_id": job["id"],
        "request_sha256": request["request_sha256"],
        "binding_sha256": binding["binding_sha256"], "request": request,
        "job_identity": job_identity,
        "resolved_mounts": [{
            "source": str(mount), "target": "/results", "read_only": False,
            "declared_source_sha256": hashlib.sha256(str(mount).encode()).hexdigest(),
        }],
        "results_scope": str(stage), "staging_receipt_sha256": "b" * 64,
    }
    return conf, output, log_path


def test_standard_executor_preserves_explicit_gpu_ids_and_validates_outputs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    conf, output, log_path = _standard_conf(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    inspect_payload = [{
        "Config": {"Labels": {
            "tao-job": conf["job_id"], "tao-platform": "airflow",
            "tao-request-sha256": conf["request_sha256"],
        }},
        "State": {"Running": True, "ExitCode": 0},
    }]

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[:2] == ["docker", "inspect"]:
            prior_launch = any(call[:3] == ["docker", "run", "-d"] for call in calls[:-1])
            return subprocess.CompletedProcess(argv, 0 if prior_launch else 1,
                                               json.dumps(inspect_payload) if prior_launch else "", "")
        if argv[:3] == ["docker", "run", "-d"]:
            return subprocess.CompletedProcess(argv, 0, "c" * 64 + "\n", "")
        if argv[:2] == ["docker", "wait"]:
            output.write_bytes(b"parquet")
            return subprocess.CompletedProcess(argv, 0, "0\n", "")
        if argv[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(argv, 0, "work complete\n", "")
        raise AssertionError(argv)

    monkeypatch.setattr(RUNTIME.subprocess, "run", fake_run)
    result = RUNTIME.execute_standard(conf)
    launch = next(call for call in calls if call[:3] == ["docker", "run", "-d"])
    assert launch[launch.index("--gpus") + 1] == '"device=6,7"'
    assert "--gpus all" not in " ".join(launch)
    assert result["exit_code"] == 0
    assert log_path.read_text().endswith("AIRFLOW_DOCKER_EXIT_CODE=0\n")


def test_standard_executor_rejects_credentials_in_conf(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    conf, _, _ = _standard_conf(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "do-not-serialize-this-token")
    conf["request"]["environment"]["BAD"] = "do-not-serialize-this-token"
    conf["request"]["request_sha256"] = RUNTIME._canonical_sha256(
        conf["request"], "request_sha256"
    )
    conf["request_sha256"] = conf["request"]["request_sha256"]
    with pytest.raises(RUNTIME.RuntimeContractError):
        RUNTIME.execute_standard(conf)


def test_dispatch_captures_prelaunch_contract_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    conf, _, log_path = _standard_conf(tmp_path, monkeypatch)
    conf["resolved_mounts"][0]["source"] = str(tmp_path / "shared" / "missing")
    with pytest.raises(RUNTIME.RuntimeContractError, match="does not exist"):
        RUNTIME.dispatch(conf)
    text = log_path.read_text(encoding="utf-8")
    assert "AIRFLOW_DISPATCH_ERROR=RuntimeContractError" in text
    assert "does not exist" in text


def test_sdg_validator_rejects_overlapping_single_host_gpu_roles(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = tmp_path / "shared"
    results = shared / "results" / "run-1"
    stage = results / "iter_1" / "datagen"
    runtime = stage / ".tao-runtime" / "controller"
    dataset = shared / "data"
    cache = shared / "cache"
    for directory in (runtime, dataset, cache, results / "config"):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TAO_IAA_AIRFLOW_SHARED_ROOT", str(shared))
    files = {
        "config_path": results / "config" / "sdg_config.yaml",
        "mined_pairs": stage / "mined.json", "gaps_parquet": stage / "gaps.parquet",
        "eval_list": stage / "eval.txt", "eval_pairs": stage / "eval.json",
        "attribute_vocab": dataset / "attribute_vocab.json",
    }
    stage.mkdir(parents=True, exist_ok=True)
    for path in files.values():
        path.write_text("{}", encoding="utf-8")
    state = results / "deft_state.json"
    state.write_text("{}", encoding="utf-8")
    binding_path = stage / "airflow-sdg.job-binding.json"
    request_path = stage / "airflow_sdg.action.json"
    request = {
        "workflow": RUNTIME.WORKFLOW, "platform": "airflow", "kind": RUNTIME.SDG_KIND,
        "run_id": "run-1", "generation_nodes": 1, "forward_env": [],
        "paths": {
            "results_dir": str(results), "stage_dir": str(stage),
            "dataset_root": str(dataset), "runtime_root": str(runtime),
            "cache_dir": str(cache), **{name: str(path) for name, path in files.items()},
        },
        "bindings": {
            "state_sha256": RUNTIME._file_sha256(state),
            "config_sha256": RUNTIME._file_sha256(files["config_path"]),
            "runtime_sha256": "a" * 64,
        },
        "resources": {
            "image_edit_gpu_ids": [0, 1, 2, 3], "vlm_gpu_ids": [3],
            "llm_gpu_ids": [5], "tao_gpu_ids": [6, 7],
        },
        "job_binding_path": str(binding_path),
    }
    request["request_sha256"] = RUNTIME._canonical_sha256(request, "request_sha256")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    binding = {"request_path": str(request_path), "job_id": "sdg-job"}
    binding["binding_sha256"] = RUNTIME._canonical_sha256(binding, "binding_sha256")
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    conf = {
        "contract": RUNTIME.CONTRACT, "kind": RUNTIME.SDG_KIND, "job_id": "sdg-job",
        "request_sha256": request["request_sha256"],
        "binding_sha256": binding["binding_sha256"], "request": request,
    }
    with pytest.raises(RUNTIME.RuntimeContractError, match="overlap"):
        RUNTIME._validate_sdg_conf(conf)
