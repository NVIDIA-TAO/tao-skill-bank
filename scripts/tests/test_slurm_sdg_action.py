# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import urllib.error

import pytest
import yaml


REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "skills/platform/tao-run-on-slurm/scripts/slurm_sdg_action.py"
SPEC = importlib.util.spec_from_file_location("slurm_sdg_action", SCRIPT)
assert SPEC and SPEC.loader
sdg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sdg)


def _request(tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict]:
    results = tmp_path / "results" / "run_1"
    stage = results / "iter_1" / "datagen"
    stage.mkdir(parents=True)
    payload = {
        "schema_version": "1",
        "workflow": "tao-run-deft-iaa",
        "kind": "slurm_sdg_action",
        "platform": "slurm",
        "name": "sdg_execute",
        "action_id": "deft-iaa-sdg-aabbccdd",
        "started_at": "2026-08-19T12:00:00+00:00",
        "started_ns": 1776772800000000000,
        "generation_nodes": 3,
        "run_id": "run_1",
        "iteration": 1,
        "attempt": 1,
        "results_dir": str(results),
        "stage_dir": str(stage),
        "dataset_root": str(tmp_path / "data" / "iaa"),
        "config_path": str(results / "config" / "sdg_config.yaml"),
        "config_sha256": "4" * 64,
        "runtime_root": str(tmp_path / "runtime"),
        "runtime_sha256": "5" * 64,
        "cache_dir": str(tmp_path / "cache"),
        "images": {
            "augmentation": "/lustre/images/augmentation.sqsh",
            "auto_labeling": "/lustre/images/auto-labeling.sqsh",
            "image_edit": "/lustre/images/vllm-omni.sqsh",
            "text_serving": "/lustre/images/vllm-openai.sqsh",
        },
        "component_sources": {
            "augmentation": "nvcr.io/nvstaging/tao/augmentation:1@sha256:" + "a" * 64,
            "auto_labeling": "nvcr.io/nvstaging/tao/auto-labeling:1@sha256:" + "b" * 64,
            "image_edit": "vllm/vllm-omni:1@sha256:" + "c" * 64,
            "text_serving": "vllm/vllm-openai:1@sha256:" + "d" * 64,
        },
        "models": {
            "image_edit": {
                "id": "Qwen/Qwen-Image-Edit-2511", "revision": "1" * 40,
                "backend": "vllm-omni", "port": 18102, "tensor_parallel": 1,
            },
            "vlm": {
                "id": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", "revision": "2" * 40,
                "backend": "vllm", "port": 18100, "tensor_parallel": 1,
            },
            "llm": {
                "id": "Qwen/Qwen2.5-14B-Instruct", "revision": "3" * 40,
                "backend": "vllm", "port": 18101, "tensor_parallel": 1,
            },
        },
        "resources": {
            "coordinator_nodes": 1, "coordinator_gpus": 2,
            "image_worker_nodes": 1, "image_worker_gpus": 8,
            "image_worker_capacity": 8,
            "image_worker_cpus_per_task": 64,
            "coordinator_cpus_per_task": 60,
            "time_minutes": 240,
        },
        "scheduler": {"account": None, "partition": None},
        "limits": {
            "startup_timeout_s": 10, "retry_interval_s": 1,
            "request_timeout_s": 2, "image_edit_request_timeout_s": 600,
            "verification_max_attempts": 2,
            "component_max_attempts": 2,
        },
        "forward_env": ["HF_TOKEN"],
        "expected_outputs": [
            str(stage / "dataset" / "sdg_manifest.json"),
            str(stage / "dataset" / "sdg_pairs.json"),
            str(stage / "dataset" / "sdg_image_list.txt"),
            str(stage / "sdg_execution_manifest.json"),
        ],
    }
    payload["request_sha256"] = sdg._canonical_sha256(payload)
    path = tmp_path / "sdg.action.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _record(tmp_path: pathlib.Path, payload: dict, job_id: str = "tao-job-123") -> pathlib.Path:
    path = tmp_path / "job.json"
    path.write_text(json.dumps({
        "id": job_id, "platform": "slurm", "action": payload["action_id"],
        "status": "PENDING",
    }), encoding="utf-8")
    return path


def _completed(argv, rc=0, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess(argv, rc, stdout, stderr)


def _pool(payload: dict, base_job_id: str = "tao-job-123") -> dict:
    endpoints = []
    base_port = payload["models"]["image_edit"]["port"]
    for node in range(payload["generation_nodes"]):
        for gpu in range(8):
            endpoints.append({
                "id": f"img-{node:03d}-gpu-{gpu}",
                "url": f"http://node-{node:03d}.cluster:{base_port + gpu}/v1",
                "capacity": 1,
                "gpu_identity": f"node-{node:03d}.cluster/gpu-{gpu}",
                "owner": {
                    "native_id": str(1000 + node),
                    "name": f"{base_job_id}-img-{node:03d}",
                },
            })
    return {
        "schema_version": "1", "platform": "slurm",
        "model": {
            "id": payload["models"]["image_edit"]["id"],
            "revision": payload["models"]["image_edit"]["revision"],
        },
        "required_capacity": payload["generation_nodes"] * 8,
        "auth_env": "IMAGE_EDIT_API_KEY", "endpoints": endpoints,
        "created_at": "2026-08-19T12:01:00Z",
        "request_sha256": payload["request_sha256"],
    }


def _image_owners(payload: dict, base_job_id: str = "tao-job-123") -> dict:
    return {
        "schema_version": "1", "request_sha256": payload["request_sha256"],
        "generation_nodes": payload["generation_nodes"],
        "workers": [
            {
                "role": "image-worker", "index": index,
                "name": f"{base_job_id}-img-{index:03d}",
                "native_id": str(1000 + index), "reconciled": False,
            }
            for index in range(payload["generation_nodes"])
        ],
    }


def _prepare_fixture(tmp_path: pathlib.Path) -> argparse.Namespace:
    results = tmp_path / "run_1"
    config_dir = results / "config"
    config_dir.mkdir(parents=True)
    config = {
        "schema_version": "1", "enabled": True,
        "images": {
            "augmentation": "nvcr.io/nvstaging/tao/augmentation:1@sha256:" + "a" * 64,
            "auto_labeling": "nvcr.io/nvstaging/tao/auto-labeling:1@sha256:" + "b" * 64,
            "image_edit_serving": "vllm/vllm-omni:1@sha256:" + "c" * 64,
            "text_serving": "vllm/vllm-openai:1@sha256:" + "d" * 64,
        },
        "models": {
            "image_edit": {
                "id": "Qwen/Qwen-Image-Edit-2511", "revision": "1" * 40,
                "backend": "vllm-omni", "port": 18102,
            },
            "vlm": {
                "id": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", "revision": "2" * 40,
                "backend": "vllm", "port": 18100,
            },
            "llm": {
                "id": "Qwen/Qwen2.5-14B-Instruct", "revision": "3" * 40,
                "backend": "vllm", "port": 18101,
            },
        },
        "endpoints": {
            "ownership": "managed", "reuse_requested": False,
            "startup_timeout_s": 1800, "request_timeout_s": 180,
            "retry_interval_s": 15,
        },
        "generation": {
            "generation_nodes": 3, "gpus_per_generation_node": 8,
            "image_edit_request_timeout_s": 600,
            "verification_max_attempts": 2,
        },
    }
    config_path = config_dir / "sdg_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    state = {
        "schema_version": "3", "workflow": "tao-run-deft-iaa",
        "started_at": "2026-08-19T12:00:00+00:00",
        "results_dir": str(results), "max_iterations": 3,
        "current_iteration": 1, "gate_met": False,
        "config": {
            "platform": "slurm", "dataset_root": "/lustre/data/iaa",
            "sdg_config": str(config_path), "requires_hf_token": True,
            "iaa_deft_bundle_sha256": "9" * 64,
            "spec_sha256": {"sdg_config.yaml": digest},
        },
        "iterations": {
            "baseline": {"status": "complete"},
            "iter1": {"status": "in_progress", "stage_completed": "history_select"},
        },
    }
    state_path = results / "deft_state.json"
    state_path.write_text(json.dumps(state))
    stage = results / "iter_1" / "datagen"
    return argparse.Namespace(
        deft_state=state_path, sdg_config=config_path, iteration=1,
        runtime_root=stage / ".tao-runtime" / "approved-runtime",
        cache_dir=pathlib.Path("/lustre/cache/huggingface"),
        augmentation_image=pathlib.Path("/lustre/images/augmentation.sqsh"),
        auto_labeling_image=pathlib.Path("/lustre/images/auto-labeling.sqsh"),
        image_edit_image=pathlib.Path("/lustre/images/image-edit.sqsh"),
        text_serving_image=pathlib.Path("/lustre/images/text-serving.sqsh"),
        account="acct", partition="gpu", image_worker_cpus_per_task=64,
        coordinator_cpus_per_task=60, time_minutes=240,
        output=stage / ".tao-runtime" / "controller" / "sdg.action.json",
    )


def test_render_fans_out_independent_eight_gpu_workers_and_two_gpu_coordinator(tmp_path):
    path, _ = _request(tmp_path)
    request = sdg.load_request(path)
    common = dict(
        request=request, worker=pathlib.Path("/lustre/run/slurm_sdg_action.py"),
        remote_request=pathlib.Path("/lustre/run/sdg.action.json"),
        auth_file=pathlib.Path("/lustre/run/endpoint-auth.env"),
        env_file=pathlib.Path("/lustre/run/job.env"), account="acct", partition="gpu",
    )
    image = sdg._render(
        mode="image-worker", job_id="tao-job-123-img-000", worker_index=0, **common,
    )
    coordinator = sdg._render(
        mode="coordinator", job_id="tao-job-123-coord", base_job_id="tao-job-123",
        job_group=pathlib.Path("/lustre/run/image-owners.json"),
        **common,
    )
    assert "#SBATCH --nodes=1" in image and "#SBATCH --gres=gpu:8" in image
    assert "#SBATCH --cpus-per-task=64" in image
    assert "image-worker" in image and '--worker-index "0"' in image
    assert "#SBATCH --nodes=1" in coordinator and "#SBATCH --gres=gpu:2" in coordinator
    assert "#SBATCH --cpus-per-task=60" in coordinator
    assert "#SBATCH --dependency=" not in coordinator
    assert 'coordinator --request' in coordinator and '--job-id "tao-job-123"' in coordinator
    for rendered in (image, coordinator):
        assert "CUDA_VISIBLE_DEVICES" not in rendered and "--gpus all" not in rendered
        assert subprocess.run(["bash", "-n", "/dev/stdin"], input=rendered, text=True).returncode == 0


def test_prepare_request_uses_distinct_worker_and_coordinator_cpu_defaults():
    args = sdg._parser().parse_args(  # noqa: SLF001
        [
            "prepare-request", "--deft-state", "/run/deft_state.json",
            "--sdg-config", "/run/config/sdg_config.yaml", "--iteration", "1",
            "--runtime-root", "/run/runtime", "--cache-dir", "/cache",
            "--augmentation-image", "/images/augmentation.sqsh",
            "--auto-labeling-image", "/images/auto-labeling.sqsh",
            "--image-edit-image", "/images/image-edit.sqsh",
            "--text-serving-image", "/images/text-serving.sqsh",
            "--output", "/run/sdg.action.json",
        ]
    )
    assert args.image_worker_cpus_per_task == 64
    assert args.coordinator_cpus_per_task == 60


def test_each_image_node_runs_eight_independent_single_gpu_services(tmp_path):
    path, _ = _request(tmp_path)
    request = sdg.load_request(path)
    commands = [
        sdg._endpoint_command(request, "image_edit", port=18102 + gpu) for gpu in range(8)
    ]
    assert len({command[command.index("--port") + 1] for command in commands}) == 8
    for command in commands:
        assert "--gpus=1" in command and "--exclusive" in command and "--exact" in command
        assert "--cpus-per-task=8" in command
        assert command[command.index("--tensor-parallel-size") + 1] == "1"
        assert "--container-env=HF_TOKEN,VLLM_API_KEY,MASTER_PORT" in command
        assert "all" not in command and not any("CUDA_VISIBLE_DEVICES" in token for token in command)
    assert sum(int(command[command.index("--cpus-per-task=8")].split("=", 1)[1])
               for command in commands) == request["resources"]["image_worker_cpus_per_task"]


def test_every_managed_vllm_server_receives_auth_by_environment_name_only(tmp_path):
    path, _ = _request(tmp_path)
    request = sdg.load_request(path)
    for role in sdg.ROLES:
        command = sdg._endpoint_command(request, role)
        env_tokens = [token for token in command if token.startswith("--container-env=")]
        assert len(env_tokens) == 1
        assert "VLLM_API_KEY" in env_tokens[0].split("=", 1)[1].split(",")
        assert "server-secret" not in " ".join(command)


def test_image_worker_assigns_distinct_internal_master_port_per_gpu(
    tmp_path, monkeypatch
):
    path, payload = _request(tmp_path)
    launched = []

    class Process:
        next_pid = 1000

        def __init__(self, argv, **kwargs):
            self.argv = argv
            self.kwargs = kwargs
            self.pid = Process.next_pid
            Process.next_pid += 1
            launched.append(self)

        def poll(self):
            return None

        def wait(self, timeout=None):
            return -15

    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("SLURM_JOB_NAME", "tao-job-123-img-000")
    monkeypatch.setattr(sdg, "_verify_signed_inputs", lambda request: {})
    checked_ports = []
    monkeypatch.setattr(
        sdg, "_port_available", lambda port: checked_ports.append(port) or True
    )
    monkeypatch.setattr(sdg, "_probe_role", lambda *args, **kwargs: None)
    monkeypatch.setattr(sdg.subprocess, "Popen", Process)
    monkeypatch.setattr(sdg.os, "killpg", lambda *args: None)
    monkeypatch.setattr(
        sdg.time, "sleep",
        lambda _seconds: (_ for _ in ()).throw(InterruptedError("stop test worker")),
    )

    with pytest.raises(InterruptedError, match="stop test worker"):
        sdg.image_worker(argparse.Namespace(
            request=path, job_id="tao-job-123-img-000", worker_index=0,
        ))

    expected_master_ports = [
        sdg.IMAGE_MASTER_PORT_BASE + gpu * sdg.IMAGE_MASTER_PORT_STRIDE
        for gpu in range(8)
    ]
    assert len(launched) == 8
    assert [int(process.kwargs["env"]["MASTER_PORT"]) for process in launched] == (
        expected_master_ports
    )
    assert checked_ports == [
        *range(payload["models"]["image_edit"]["port"],
               payload["models"]["image_edit"]["port"] + 8),
        *expected_master_ports,
    ]
    assert all(
        "MASTER_PORT" in next(
            token for token in process.argv if token.startswith("--container-env=")
        ).split("=", 1)[1].split(",")
        for process in launched
    )


def test_text_probes_require_environment_auth_for_models_and_inference(tmp_path, monkeypatch):
    path, _ = _request(tmp_path)
    request = sdg.load_request(path)
    calls = []

    def protected(url, *, timeout, payload=None, auth_env=None):
        calls.append((url, auth_env, payload))
        if auth_env != "VLLM_API_KEY" or not os.environ.get(auth_env):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
        return (
            {"data": [{"id": request["models"]["llm"]["id"]}]}
            if url.endswith("/models") else {"choices": [{}]}
        )

    monkeypatch.setattr(sdg, "_request_json", protected)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    with pytest.raises(urllib.error.HTTPError) as error:
        sdg._probe_role(request, "llm")
    assert error.value.code == 401
    monkeypatch.setenv("VLLM_API_KEY", "text-endpoint-secret")
    sdg._probe_role(request, "llm")
    assert len(calls) == 3
    assert all(call[1] == "VLLM_API_KEY" for call in calls)
    assert all("text-endpoint-secret" not in repr(call) for call in calls)


def test_component_text_auth_is_forwarded_by_name_only(tmp_path, monkeypatch):
    path, _ = _request(tmp_path)
    request = sdg.load_request(path)
    monkeypatch.setenv("VLLM_API_KEY", "server-secret")
    monkeypatch.setenv("VLM_API_KEY", "vlm-secret")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    augment = sdg._component_command(
        request, "augment", source_key="sample-1", attempt=1,
        target_attributes={}, image_edit_url="http://node:8002/v1",
    )
    label = sdg._component_command(request, "label", source_key="sample-1")
    joined = " ".join([*augment, *label])
    assert "--container-env=IMAGE_EDIT_API_KEY,VLM_API_KEY,LLM_API_KEY,OPENAI_API_KEY" in augment
    assert "--container-env=VLM_API_KEY,LLM_API_KEY,OPENAI_API_KEY" in label
    assert "--container-remap-root" not in augment
    assert "--container-remap-root" in label
    assert "--no-container-mount-home" not in augment
    assert "--no-container-mount-home" in label
    assert f"--container-mounts={request['stage_dir']}/label_inputs:/input:ro,{request['stage_dir']}:/output" in label
    assert "--container-mounts=/root" not in " ".join(label)
    assert "UV_CACHE_DIR=/app/data/out/.tao-runtime/uv-cache" in augment
    assert "UV_CACHE_DIR=/output/.tao-runtime/uv-cache" in label
    assert augment[augment.index("UV_CACHE_DIR=/app/data/out/.tao-runtime/uv-cache") - 1] == "env"
    assert label[label.index("UV_CACHE_DIR=/output/.tao-runtime/uv-cache") - 1] == "env"
    for secret in ("server-secret", "vlm-secret", "llm-secret", "openai-secret"):
        assert secret not in joined


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.update({"shell": "rm -rf /"}), "unexpected"),
        (lambda p: p.update({"forward_env": ["HF_TOKEN", "NGC_API_KEY"]}), "forward_env"),
        (lambda p: p["resources"].update({"image_worker_gpus": 4}), "image_worker_gpus"),
        (lambda p: p["resources"].update({"image_worker_cpus_per_task": 60}),
         "image_worker_cpus_per_task"),
        (lambda p: p["resources"].update({"coordinator_cpus_per_task": 4}),
         "coordinator_cpus_per_task"),
        (lambda p: p["resources"].update({"cpus_per_task": 64}), "exactly"),
        (lambda p: p["models"]["image_edit"].update({"tensor_parallel": 2}), "tensor_parallel"),
        (lambda p: p.update({"generation_nodes": 0}), "generation_nodes"),
        (lambda p: p.update({"started_ns": 0}), "started_ns"),
    ],
)
def test_request_rejects_arbitrary_shell_secrets_and_resource_widening(tmp_path, mutation, match):
    _, payload = _request(tmp_path)
    mutation(payload)
    payload["request_sha256"] = sdg._canonical_sha256(payload)
    with pytest.raises(ValueError, match=match):
        sdg.validate_request(payload)


def test_request_digest_detects_mutation(tmp_path):
    _, payload = _request(tmp_path)
    payload["limits"]["startup_timeout_s"] += 1
    with pytest.raises(ValueError, match="does not match"):
        sdg.validate_request(payload)


def test_prepare_request_is_deterministic_idempotent_and_never_reads_credentials(tmp_path, monkeypatch):
    args = _prepare_fixture(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "must-not-appear")
    first = sdg.prepare_request(args)
    original = args.output.read_bytes()
    second = sdg.prepare_request(args)
    assert first["status"] == "written" and second["status"] == "unchanged"
    assert args.output.read_bytes() == original
    payload = json.loads(original)
    assert payload["generation_nodes"] == 3
    assert payload["scheduler"] == {"account": "acct", "partition": "gpu"}
    assert payload["resources"]["image_worker_cpus_per_task"] == 64
    assert payload["resources"]["coordinator_cpus_per_task"] == 60
    assert payload["forward_env"] == ["HF_TOKEN"]
    assert b"must-not-appear" not in original
    assert sdg._canonical_sha256(payload) == payload["request_sha256"]


def test_prepare_request_maps_only_signed_backend_paths_for_airflow_slurm(tmp_path):
    args = _prepare_fixture(tmp_path)
    controller_results = args.deft_state.parent
    backend_results = pathlib.Path("/lustre/workspace/results") / controller_results.name
    backend_dataset = pathlib.Path("/lustre/datasets/iaa")
    args.backend_results_dir = backend_results
    args.backend_dataset_root = backend_dataset
    args.runtime_root = backend_results / "iter_1/datagen/.tao-runtime/runtime"

    prepared = sdg.prepare_request(args)
    request = prepared["request"]

    assert pathlib.Path(prepared["output"]).is_relative_to(controller_results)
    assert request["results_dir"] == str(backend_results)
    assert request["stage_dir"] == str(backend_results / "iter_1/datagen")
    assert request["config_path"] == str(backend_results / "config/sdg_config.yaml")
    assert request["dataset_root"] == str(backend_dataset)
    assert all(
        pathlib.Path(path).is_relative_to(backend_results)
        for path in request["expected_outputs"]
    )


def test_terminal_output_sync_maps_exact_remote_artifacts_to_controller(
    tmp_path, monkeypatch
):
    local_results = tmp_path / "controller/results/run_1"
    local_results.mkdir(parents=True)
    remote_results = pathlib.Path("/lustre/workspace/results/run_1")
    remote_stage = remote_results / "iter_1/datagen"
    request = {
        "results_dir": str(remote_results),
        "stage_dir": str(remote_stage),
        "expected_outputs": [
            str(remote_stage / "dataset/sdg_manifest.json"),
            str(remote_stage / "dataset/sdg_pairs.json"),
            str(remote_stage / "dataset/sdg_image_list.txt"),
            str(remote_stage / "sdg_execution_manifest.json"),
        ],
    }
    dataset_remotes = [pathlib.Path(path) for path in request["expected_outputs"][:3]]
    remotes = [
        pathlib.Path(request["expected_outputs"][3]),
        remote_stage / "endpoint_pool.json",
        remote_stage / "endpoint_manifest.json",
        remote_stage / "status/sdg-normalize.slurm.status.json",
    ]
    job_id = "iaa-sdg-test-123"
    status_path = remote_stage / "status/sdg-normalize.slurm.status.json"
    endpoint_path = remote_stage / "endpoint_manifest.json"
    extra_remotes = [
        remote_stage / "logs/sdg-normalize.slurm.log",
        remote_stage / "status/sdg-normalize.slurm.pre-action.json",
        remote_stage / f".tao-runtime/sdg.action.{job_id}.json",
        remote_stage / f"slurm_sdg_terminal.{job_id}.json",
    ]
    remotes.extend(extra_remotes)
    bodies = {str(path): f"verified:{path.name}\n".encode("utf-8") for path in remotes}
    bodies[str(endpoint_path)] = json.dumps({"job_id": job_id}).encode()
    bodies[str(status_path)] = json.dumps({
        "log_path": str(extra_remotes[0]),
        "pre_action": {"path": str(extra_remotes[1]), "sha256": "a" * 64},
    }).encode()

    def ssh(_login, command, **_kwargs):
        remote = next(path for path in remotes if str(path) in command)
        body = bodies[str(remote)]
        output = f"{len(body)}\n{hashlib.sha256(body).hexdigest()}  {remote}\n".encode()
        return _completed([], stdout=output)

    def run(argv, **_kwargs):
        remote = argv[-2].split(":", 1)[1]
        pathlib.Path(argv[-1]).write_bytes(bodies[remote])
        return _completed(argv)

    monkeypatch.setattr(sdg, "_ssh", ssh)
    monkeypatch.setattr(sdg, "_run", run)

    def sync_dataset(_login, remote_dataset, controller, *, remote_results):
        assert remote_dataset == remote_stage / "dataset"
        rows = []
        for remote in dataset_remotes:
            local = controller / remote.relative_to(remote_results)
            local.parent.mkdir(parents=True, exist_ok=True)
            body = f"verified:{remote.name}\n".encode()
            local.write_bytes(body)
            rows.append({
                "kind": "dataset_file", "local": str(local), "remote": str(remote),
                "size": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            })
        return rows

    monkeypatch.setattr(sdg, "_synchronize_controller_dataset_tree", sync_dataset)
    evidence = sdg._synchronize_controller_outputs(
        "user@login", request, local_results
    )

    assert len(evidence) == 11
    for remote in [*dataset_remotes, *remotes]:
        relative = remote.relative_to(remote_results)
        expected = bodies[str(remote)] if str(remote) in bodies else f"verified:{remote.name}\n".encode()
        assert (local_results / relative).read_bytes() == expected
    assert all(row["remote"].startswith(str(remote_results)) for row in evidence)
    assert all(row["local"].startswith(str(local_results)) for row in evidence)


def test_dataset_tree_sync_includes_generated_images_and_captions(tmp_path, monkeypatch):
    remote_results = pathlib.Path("/lustre/workspace/results/run_1")
    remote_dataset = remote_results / "iter_1/datagen/dataset"
    local_results = tmp_path / "controller/results/run_1"
    local_results.mkdir(parents=True)
    bodies = {
        "sdg_manifest.json": b"{}\n",
        "images/generated.jpg": b"generated-image",
        "captions/generated.txt": b"generated caption\n",
    }
    rows = []
    for relative, body in sorted(bodies.items()):
        rows.append(
            f"{hashlib.sha256(body).hexdigest()}|{len(body)}|"
            f"{base64.b64encode(relative.encode()).decode()}"
        )

    monkeypatch.setattr(
        sdg, "_ssh", lambda *_args, **_kwargs: _completed([], stdout=("\n".join(rows) + "\n").encode())
    )

    def copy_tree(argv, **_kwargs):
        target = pathlib.Path(argv[-1]) / remote_dataset.name
        for relative, body in bodies.items():
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        return _completed(argv)

    monkeypatch.setattr(sdg, "_run", copy_tree)
    evidence = sdg._synchronize_controller_dataset_tree(
        "user@login", remote_dataset, local_results, remote_results=remote_results
    )
    assert len(evidence) == 3
    for relative, body in bodies.items():
        assert (local_results / "iter_1/datagen/dataset" / relative).read_bytes() == body


def test_prepare_attempt2_requires_authoritative_failed_attempt1(tmp_path, monkeypatch):
    args = _prepare_fixture(tmp_path)
    first = sdg.prepare_request(args)["request"]
    prior_request_path = args.output
    prior_bytes = args.output.read_bytes()
    prior_job = "tao-job-123"
    record = tmp_path / f"{prior_job}.json"
    record.write_text(json.dumps({
        "schema_version": 1, "id": prior_job, "platform": "slurm",
        "backend_ref": "2004", "action": first["action_id"],
        "results_dir": first["stage_dir"], "terminal_state": "ERROR",
        "err_class": "ERR_INFRA", "redacted": True,
        "terminal_write_by": "poller",
        "transitions": [
            {"state": "PENDING"}, {"state": "RUNNING"}, {"state": "ERROR"},
        ],
    }))
    group = {
        "schema_version": "1", "request_sha256": first["request_sha256"],
        "job_id": prior_job,
        "coordinator": {"role": "coordinator", "name": f"{prior_job}-coord",
                        "native_id": "2004", "reconciled": False},
        "image_workers": _image_owners(first)["workers"],
    }
    monkeypatch.setattr(sdg, "_remote_job_group", lambda *args: group)
    monkeypatch.setattr(sdg, "_assert_job_ownership", lambda *args: None)
    monkeypatch.setattr(sdg, "_native_state", lambda *args: "FAILED")
    args.retry_from_request = args.output
    args.retry_from_job_record = record
    args.retry_login = "user@login"
    args.output = args.output.with_name("sdg.attempt-2.action.json")
    second = sdg.prepare_request(args)["request"]
    assert second["attempt"] == 2 and second["action_id"] != first["action_id"]
    assert second["action_id"] == "deft-iaa-sdg-" + hashlib.sha256(
        json.dumps(second["retry"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    assert second["retry"]["job_id"] == prior_job
    assert second["retry"]["backend_ref"] == "2004"
    assert second["retry"]["prior_partition"] == first["scheduler"]["partition"]
    assert second["retry"]["new_partition"] == second["scheduler"]["partition"]
    assert sdg._resume_sha256(second) == sdg._resume_sha256(first)
    assert sdg.prepare_request(args)["status"] == "unchanged"
    assert args.retry_from_request.read_bytes() == prior_bytes
    assert record.is_file()
    attempt2_record = tmp_path / "attempt2-job.json"
    attempt2_record.write_text(json.dumps({
        "id": prior_job, "platform": "slurm", "action": second["action_id"],
        "schema_version": 1, "redacted": True, "results_dir": second["stage_dir"],
        "retry_of": prior_job, "terminal_state": None,
        "transitions": [{"state": "PENDING"}],
    }))
    with pytest.raises(ValueError, match="distinct job record"):
        sdg._load_job_record(attempt2_record, second, prior_job)
    attempt2_record.write_text(json.dumps({
        "id": "tao-job-456", "platform": "slurm", "action": second["action_id"],
        "schema_version": 1, "redacted": True, "results_dir": second["stage_dir"],
        "retry_of": prior_job, "terminal_state": None,
        "transitions": [{"state": "PENDING"}],
    }))
    assert sdg._load_job_record(attempt2_record, second, "tao-job-456")["id"] == "tao-job-456"
    args.retry_from_request = args.output
    args.output = args.output.with_name("sdg.attempt-3.action.json")
    with pytest.raises(ValueError, match="original attempt-1"):
        sdg.prepare_request(args)
    args.retry_from_request = prior_request_path
    args.output = args.output.with_name("sdg.attempt-2.invalid.json")
    bad = json.loads(record.read_text())
    bad["terminal_state"] = "CANCELED"
    record.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="terminal infrastructure ERROR"):
        sdg.prepare_request(args)
    bad["terminal_state"] = "ERROR"
    bad["terminal_write_by"] = "controller"
    record.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="terminal infrastructure ERROR"):
        sdg.prepare_request(args)


def test_attempt2_may_rebind_only_the_evidenced_scheduler_partition(
    tmp_path, monkeypatch
):
    args = _prepare_fixture(tmp_path)
    first = sdg.prepare_request(args)["request"]
    native_states = {"2004": "FAILED", "2005": "CANCELLED"}
    retry = {
        "job_id": "attempt-1-job", "action_id": first["action_id"],
        "backend_ref": "2004", "request_sha256": first["request_sha256"],
        "job_record_sha256": "a" * 64, "job_group_sha256": "b" * 64,
        "native_states_sha256": hashlib.sha256(json.dumps(
            native_states, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_states": native_states, "terminal_evidence": None,
    }
    monkeypatch.setattr(sdg, "_retry_lineage", lambda *unused: (first, retry))
    args.retry_from_request = args.output
    args.retry_from_job_record = tmp_path / "attempt-1-job.json"
    args.retry_login = "user@login"
    args.partition = "polar4"
    args.output = args.output.with_name("sdg.attempt-2.action.json")

    second = sdg.prepare_request(args)["request"]

    assert second["scheduler"]["partition"] == "polar4"
    assert second["retry"]["prior_partition"] == first["scheduler"]["partition"]
    assert second["retry"]["new_partition"] == "polar4"
    comparable = json.loads(json.dumps(first))
    comparable["scheduler"]["partition"] = "polar4"
    assert sdg._resume_sha256(second) == sdg._resume_sha256(comparable)

    args.output = args.output.with_name("sdg.attempt-2.invalid.json")
    args.time_minutes -= 1
    with pytest.raises(ValueError, match="apart from the selected retry partition"):
        sdg.prepare_request(args)


def test_prepare_bounded_unstarted_pool_rebind_repair_stays_attempt2(
    tmp_path, monkeypatch
):
    args = _prepare_fixture(tmp_path)
    first = sdg.prepare_request(args)["request"]
    native_states = {"2004": "FAILED", "2005": "CANCELLED"}
    retry = {
        "job_id": "attempt-1-job", "action_id": first["action_id"],
        "backend_ref": "2004", "request_sha256": first["request_sha256"],
        "job_record_sha256": "a" * 64, "job_group_sha256": "b" * 64,
        "native_states_sha256": hashlib.sha256(json.dumps(
            native_states, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_states": native_states, "terminal_evidence": None,
    }
    monkeypatch.setattr(sdg, "_retry_lineage", lambda *unused: (first, retry))
    args.retry_from_request = args.output
    args.retry_from_job_record = tmp_path / "attempt-1-job.json"
    args.retry_login = "user@login"
    args.output = args.output.with_name("sdg.attempt-2.action.json")
    second = sdg.prepare_request(args)["request"]

    repair = {
        "kind": sdg.POOL_REBIND_REPAIR_KIND,
        "job_id": "attempt-2-job", "action_id": second["action_id"],
        "backend_ref": "3004", "request_sha256": second["request_sha256"],
        "job_record_sha256": "c" * 64, "job_group_sha256": "d" * 64,
        "native_states_sha256": hashlib.sha256(json.dumps(
            native_states, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_states": native_states, "terminal_sha256": "e" * 64,
        "cleanup_sha256": "f" * 64, "execute_log_sha256": "1" * 64,
        "progress_sha256": "2" * 64,
    }
    repair["evidence_sha256"] = hashlib.sha256(json.dumps(
        repair, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    monkeypatch.setattr(
        sdg, "_pool_rebind_repair_lineage", lambda *unused: (second, repair)
    )
    state = json.loads(args.deft_state.read_text())
    evidence = pathlib.Path(state["results_dir"]) / "runtime_rebind/validation-1.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps({"result": "PASS", "runtime_sha256": "8" * 64}))
    state["active_runtime_sha256"] = "8" * 64
    state["runtime_lineage"] = [{
        "schema_version": "1", "sequence": 1,
        "old_sha256": "9" * 64, "new_sha256": "8" * 64,
        "rebound_at": "2026-08-24T00:00:00+00:00", "reason": "approved repair",
        "evidence_path": str(evidence),
        "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "plugin_base_version": "0.1.12", "skill_version": "0.6.0",
    }]
    args.deft_state.write_text(json.dumps(state))
    args.retry_from_request = None
    args.retry_from_job_record = None
    args.retry_login = None
    args.repair_from_request = args.output
    args.repair_from_job_record = tmp_path / "attempt-2-job.json"
    args.repair_login = "user@login"
    args.output = args.output.with_name("sdg.attempt-2-repair.action.json")
    repaired = sdg.prepare_request(args)["request"]

    assert repaired["attempt"] == 2
    assert repaired["retry"] == retry
    assert repaired["repair"] == repair
    assert repaired["repair"]["runtime_rebind_sha256"] == hashlib.sha256(json.dumps(
        state["runtime_lineage"][0], sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    assert repaired["action_id"] not in {first["action_id"], second["action_id"]}
    comparable_second = dict(second)
    comparable_second["runtime_sha256"] = repaired["runtime_sha256"]
    assert sdg._resume_sha256(repaired) == sdg._resume_sha256(comparable_second)
    assert sdg.prepare_request(args)["status"] == "unchanged"


def test_prepare_single_shorter_scheduler_reschedule_preserves_attempt2_repair(
    tmp_path, monkeypatch
):
    args = _prepare_fixture(tmp_path)
    first = sdg.prepare_request(args)["request"]
    native_states = {"2004": "FAILED", "2005": "CANCELLED"}
    retry = {
        "job_id": "attempt-1-job", "action_id": first["action_id"],
        "backend_ref": "2004", "request_sha256": first["request_sha256"],
        "job_record_sha256": "a" * 64, "job_group_sha256": "b" * 64,
        "native_states_sha256": hashlib.sha256(json.dumps(
            native_states, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_states": native_states, "terminal_evidence": None,
    }
    repair = {
        "kind": sdg.POOL_REBIND_REPAIR_KIND,
        "job_id": "attempt-2-job", "action_id": "deft-iaa-sdg-repaired",
        "backend_ref": "3004", "request_sha256": "c" * 64,
        "job_record_sha256": "d" * 64, "job_group_sha256": "e" * 64,
        "native_states_sha256": hashlib.sha256(json.dumps(
            native_states, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_states": native_states, "terminal_sha256": "f" * 64,
        "cleanup_sha256": "1" * 64, "execute_log_sha256": "2" * 64,
        "progress_sha256": "3" * 64, "runtime_rebind_sha256": "4" * 64,
    }
    repair["evidence_sha256"] = hashlib.sha256(json.dumps(
        repair, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    repaired = dict(first)
    repaired.update({
        "attempt": 2, "retry": retry, "repair": repair,
        "action_id": "deft-iaa-sdg-repaired",
    })
    repaired["request_sha256"] = sdg._canonical_sha256(repaired)
    accounting = {
        "4001": {"state": "CANCELLED", "elapsed_raw": 0},
        "4002": {"state": "CANCELLED", "elapsed_raw": 0},
    }
    reschedule = {
        "kind": sdg.SCHEDULER_RESCHEDULE_KIND,
        "job_id": "attempt-2-repair-job", "action_id": repaired["action_id"],
        "backend_ref": "4001", "request_sha256": repaired["request_sha256"],
        "job_record_sha256": "5" * 64, "job_group_sha256": "6" * 64,
        "native_accounting_sha256": hashlib.sha256(json.dumps(
            accounting, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_accounting": accounting, "prior_time_minutes": 240,
        "new_time_minutes": 60, "progress_sha256": repair["progress_sha256"],
    }
    reschedule["evidence_sha256"] = hashlib.sha256(json.dumps(
        reschedule, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    monkeypatch.setattr(
        sdg, "_scheduler_reschedule_lineage", lambda *unused: (repaired, reschedule)
    )
    args.time_minutes = 60
    args.reschedule_from_request = tmp_path / "sdg.attempt-2-repair.action.json"
    args.reschedule_from_job_record = tmp_path / "attempt-2-repair-job.json"
    args.reschedule_login = "user@login"
    args.output = args.output.with_name("sdg.attempt-2-reschedule-60.action.json")
    prepared = sdg.prepare_request(args)["request"]

    assert prepared["attempt"] == 2
    assert prepared["retry"] == retry
    assert prepared["repair"] == repair
    assert prepared["reschedule"] == reschedule
    assert prepared["resources"]["time_minutes"] == 60
    comparable = json.loads(json.dumps(repaired))
    comparable["resources"]["time_minutes"] = 60
    assert sdg._resume_sha256(prepared) == sdg._resume_sha256(comparable)
    job_record = tmp_path / "rescheduled-job.json"
    job_record.write_text(json.dumps({
        "schema_version": 1, "id": "rescheduled-job", "platform": "slurm",
        "action": prepared["action_id"], "results_dir": prepared["stage_dir"],
        "retry_of": reschedule["job_id"], "redacted": True,
        "terminal_state": None, "transitions": [{"state": "PENDING"}],
    }), encoding="utf-8")
    assert sdg._load_job_record(
        job_record, prepared, "rescheduled-job", require_pending=True
    )["retry_of"] == reschedule["job_id"]
    assert sdg.prepare_request(args)["status"] == "unchanged"


def test_scheduler_reschedule_requires_owned_zero_elapsed_terminal_group(
    tmp_path, monkeypatch
):
    prior_path, prior = _request(tmp_path)
    native_states = {"1000": "FAILED", "1001": "CANCELLED"}
    prior["attempt"] = 2
    prior["retry"] = {
        "job_id": "attempt-1-job", "action_id": "deft-iaa-sdg-attempt1",
        "backend_ref": "9000", "request_sha256": "a" * 64,
        "job_record_sha256": "b" * 64, "job_group_sha256": "c" * 64,
        "native_states_sha256": hashlib.sha256(json.dumps(
            native_states, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_states": native_states, "terminal_evidence": None,
    }
    repair = {
        "kind": sdg.POOL_REBIND_REPAIR_KIND,
        "job_id": "attempt-2-job", "action_id": "deft-iaa-sdg-attempt2",
        "backend_ref": "9100", "request_sha256": "d" * 64,
        "job_record_sha256": "e" * 64, "job_group_sha256": "f" * 64,
        "native_states_sha256": hashlib.sha256(json.dumps(
            native_states, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_states": native_states, "terminal_sha256": "1" * 64,
        "cleanup_sha256": "2" * 64, "execute_log_sha256": "3" * 64,
        "progress_sha256": "4" * 64, "runtime_rebind_sha256": "5" * 64,
    }
    repair["evidence_sha256"] = hashlib.sha256(json.dumps(
        repair, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    prior["repair"] = repair
    prior["action_id"] = "deft-iaa-sdg-repaired"
    prior["request_sha256"] = sdg._canonical_sha256(prior)
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    job_id = "attempt-2-repair-job"
    record_path = tmp_path / f"{job_id}.json"
    record_path.write_text(json.dumps({
        "schema_version": 1, "id": job_id, "platform": "slurm",
        "action": prior["action_id"], "results_dir": prior["stage_dir"],
        "retry_of": repair["job_id"], "backend_ref": "9200",
        "terminal_state": "ERROR", "err_class": "ERR_INFRA", "redacted": True,
        "terminal_write_by": "poller",
        "transitions": [
            {"state": "PENDING"}, {"state": "RUNNING"}, {"state": "ERROR"},
        ],
    }), encoding="utf-8")
    group = {
        "schema_version": "1", "request_sha256": prior["request_sha256"],
        "job_id": job_id,
        "coordinator": {
            "role": "coordinator", "name": f"{job_id}-coord",
            "native_id": "9200", "reconciled": False,
        },
        "image_workers": [{
            "role": "image-worker", "index": 0, "name": f"{job_id}-img-000",
            "native_id": "9201", "reconciled": False,
        }],
    }
    monkeypatch.setattr(sdg, "_remote_job_group", lambda *unused: group)
    monkeypatch.setattr(sdg, "_assert_job_ownership", lambda *unused: None)
    monkeypatch.setattr(
        sdg, "_remote_file_sha256", lambda *unused: repair["progress_sha256"]
    )
    monkeypatch.setattr(sdg, "_ssh", lambda *unused, **kwargs: _completed([]))
    monkeypatch.setattr(
        sdg, "_native_accounting",
        lambda _login, _native: {"state": "CANCELLED", "elapsed_raw": 0},
    )

    loaded, lineage = sdg._scheduler_reschedule_lineage(
        prior_path, record_path, "user@login", 60
    )
    assert loaded == prior
    assert lineage["prior_time_minutes"] == 240
    assert lineage["new_time_minutes"] == 60
    assert set(lineage["native_accounting"]) == {"9200", "9201"}

    monkeypatch.setattr(
        sdg, "_native_accounting",
        lambda _login, _native: {"state": "CANCELLED", "elapsed_raw": 1},
    )
    with pytest.raises(ValueError, match="canceled before execution"):
        sdg._scheduler_reschedule_lineage(
            prior_path, record_path, "user@login", 60
        )


def test_prepare_image_master_port_launch_repair_preserves_all_prior_lineage(
    tmp_path, monkeypatch
):
    args = _prepare_fixture(tmp_path)
    first = sdg.prepare_request(args)["request"]
    native_states = {"1000": "FAILED", "1001": "CANCELLED"}
    retry = {
        "job_id": "attempt-1-job", "action_id": first["action_id"],
        "backend_ref": "1000", "request_sha256": "a" * 64,
        "job_record_sha256": "b" * 64, "job_group_sha256": "c" * 64,
        "native_states_sha256": hashlib.sha256(json.dumps(
            native_states, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_states": native_states, "terminal_evidence": None,
    }
    repair = {
        "kind": sdg.POOL_REBIND_REPAIR_KIND, "job_id": "attempt-2-job",
        "action_id": "deft-iaa-sdg-attempt2", "backend_ref": "2000",
        "request_sha256": "d" * 64, "job_record_sha256": "e" * 64,
        "job_group_sha256": "f" * 64,
        "native_states_sha256": retry["native_states_sha256"],
        "native_states": native_states, "terminal_sha256": "1" * 64,
        "cleanup_sha256": "2" * 64, "execute_log_sha256": "3" * 64,
        "progress_sha256": "4" * 64, "runtime_rebind_sha256": "5" * 64,
    }
    repair["evidence_sha256"] = hashlib.sha256(json.dumps(
        repair, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    accounting = {"3000": {"state": "CANCELLED", "elapsed_raw": 0}}
    reschedule = {
        "kind": sdg.SCHEDULER_RESCHEDULE_KIND, "job_id": "repair-job",
        "action_id": "deft-iaa-sdg-repair", "backend_ref": "3000",
        "request_sha256": "6" * 64, "job_record_sha256": "7" * 64,
        "job_group_sha256": "8" * 64,
        "native_accounting_sha256": hashlib.sha256(json.dumps(
            accounting, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_accounting": accounting, "prior_time_minutes": 240,
        "new_time_minutes": 60, "progress_sha256": repair["progress_sha256"],
    }
    reschedule["evidence_sha256"] = hashlib.sha256(json.dumps(
        reschedule, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    prior = json.loads(json.dumps(first))
    prior["resources"]["time_minutes"] = 60
    prior.update({
        "attempt": 2, "retry": retry, "repair": repair,
        "reschedule": reschedule, "action_id": "deft-iaa-sdg-rescheduled",
    })
    prior["request_sha256"] = sdg._canonical_sha256(prior)
    launch_states = {"4000": "CANCELLED", "4001": "FAILED"}
    launch_repair = {
        "kind": sdg.IMAGE_MASTER_PORT_REPAIR_KIND,
        "job_id": "rescheduled-job", "action_id": prior["action_id"],
        "backend_ref": "4000", "request_sha256": prior["request_sha256"],
        "job_record_sha256": "9" * 64, "job_group_sha256": "a" * 64,
        "native_states_sha256": hashlib.sha256(json.dumps(
            launch_states, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "native_states": launch_states, "terminal_sha256": "b" * 64,
        "cleanup_sha256": "c" * 64,
        "failure_evidence": [{
            "worker_name": "rescheduled-job-img-000", "native_id": "4001",
            "endpoint_id": "img-000-gpu-5", "worker_log_sha256": "d" * 64,
            "endpoint_log_sha256": "e" * 64,
        }],
        "descriptor_sha256": {"/lustre/descriptor.json": "f" * 64},
        "progress_sha256": repair["progress_sha256"],
    }
    launch_repair["evidence_sha256"] = hashlib.sha256(json.dumps(
        launch_repair, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    monkeypatch.setattr(
        sdg, "_image_master_port_repair_lineage",
        lambda *unused: (prior, launch_repair),
    )
    args.time_minutes = 60
    args.launch_repair_from_request = tmp_path / "rescheduled.action.json"
    args.launch_repair_from_job_record = tmp_path / "rescheduled-job.json"
    args.launch_repair_login = "user@login"
    args.output = args.output.with_name("sdg.attempt-2-launch-repair.action.json")
    prepared = sdg.prepare_request(args)["request"]

    assert prepared["attempt"] == 2
    assert prepared["retry"] == retry
    assert prepared["repair"] == repair
    assert prepared["reschedule"] == reschedule
    assert prepared["launch_repair"] == launch_repair
    assert sdg._resume_sha256(prepared) == sdg._resume_sha256(prior)
    record = tmp_path / "launch-repair-job.json"
    record.write_text(json.dumps({
        "schema_version": 1, "id": "launch-repair-job", "platform": "slurm",
        "action": prepared["action_id"], "results_dir": prepared["stage_dir"],
        "retry_of": launch_repair["job_id"], "redacted": True,
        "terminal_state": None, "transitions": [{"state": "PENDING"}],
    }), encoding="utf-8")
    assert sdg._load_job_record(
        record, prepared, "launch-repair-job", require_pending=True
    )["retry_of"] == launch_repair["job_id"]

def test_prepare_request_rejects_different_existing_output(tmp_path):
    args = _prepare_fixture(tmp_path)
    sdg.prepare_request(args)
    payload = json.loads(args.output.read_text())
    payload["resources"]["time_minutes"] += 1
    payload["request_sha256"] = sdg._canonical_sha256(payload)
    args.output.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differs"):
        sdg.prepare_request(args)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda state, args: state["config"].update({"platform": "docker"}), "platform must be slurm"),
        (lambda state, args: state["iterations"]["iter1"].update({"stage_completed": "sdg"}), "not rerunnable"),
        (lambda state, args: state["config"]["spec_sha256"].update({"sdg_config.yaml": "0" * 64}), "digest"),
        (lambda state, args: setattr(args, "runtime_root", pathlib.Path("/tmp/outside")), "under"),
    ],
)
def test_prepare_request_binds_state_commit_and_safe_paths(tmp_path, mutation, match):
    args = _prepare_fixture(tmp_path)
    state = json.loads(args.deft_state.read_text())
    mutation(state, args)
    args.deft_state.write_text(json.dumps(state))
    with pytest.raises(ValueError, match=match):
        sdg.prepare_request(args)


def test_job_record_binds_action_and_exact_job_id(tmp_path):
    _, payload = _request(tmp_path)
    record = _record(tmp_path, payload)
    loaded = sdg._load_job_record(record, payload, "tao-job-123")
    assert loaded["action"] == payload["action_id"]
    bad = json.loads(record.read_text())
    bad["action"] = "other-action"
    record.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="action_id"):
        sdg._load_job_record(record, payload, "tao-job-123")


def test_attempt2_job_record_is_pending_only_for_submit_and_evolves_for_status(tmp_path):
    path, payload = _request(tmp_path)
    payload["attempt"] = 2
    payload["retry"] = {
        "job_id": "attempt1-job", "action_id": "attempt1-action",
        "backend_ref": "1004", "request_sha256": "1" * 64,
        "job_record_sha256": "2" * 64, "job_group_sha256": "3" * 64,
        "native_states_sha256": sdg.hashlib.sha256(
            json.dumps({"1004": "FAILED"}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "native_states": {"1004": "FAILED"},
        "terminal_evidence": None,
    }
    payload["request_sha256"] = sdg._canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    record = tmp_path / "attempt2-job.json"
    base = {
        "schema_version": 1, "id": "attempt2-job", "platform": "slurm",
        "backend_ref": None, "action": payload["action_id"],
        "results_dir": payload["stage_dir"], "retry_of": "attempt1-job",
        "terminal_state": None, "redacted": True,
        "transitions": [{"state": "PENDING"}],
    }
    record.write_text(json.dumps(base))
    sdg._load_job_record(record, payload, "attempt2-job", require_pending=True)
    running = dict(base)
    running["backend_ref"] = "2004"
    running["transitions"] = [*base["transitions"], {"state": "RUNNING"}]
    record.write_text(json.dumps(running))
    sdg._load_job_record(record, payload, "attempt2-job")
    with pytest.raises(ValueError, match="fresh pending"):
        sdg._load_job_record(record, payload, "attempt2-job", require_pending=True)
    complete = dict(running)
    complete["terminal_state"] = "COMPLETE"
    complete["transitions"] = [*running["transitions"], {"state": "COMPLETE"}]
    record.write_text(json.dumps(complete))
    sdg._load_job_record(record, payload, "attempt2-job")


def test_readiness_timeout_is_bounded_and_actionable(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    payload["limits"]["startup_timeout_s"] = 1
    payload["request_sha256"] = sdg._canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    request = sdg.load_request(path)

    class Process:
        def poll(self):
            return None

    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(sdg.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(sdg.time, "sleep", lambda _: None)
    monkeypatch.setattr(sdg, "_probe_role", lambda *_, **__: (_ for _ in ()).throw(OSError("not ready")))
    with pytest.raises(TimeoutError, match="readiness deadline exceeded"):
        sdg._wait_readiness(request, {role: Process() for role in sdg.ROLES})


def test_malformed_models_response_enters_bounded_readiness_classification(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    payload["limits"]["startup_timeout_s"] = 1
    payload["request_sha256"] = sdg._canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    request = sdg.load_request(path)

    class Process:
        def poll(self):
            return None

    ticks = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(sdg.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(sdg.time, "sleep", lambda _: None)
    monkeypatch.setenv("VLLM_API_KEY", "server-secret")
    monkeypatch.setattr(sdg, "_request_json", lambda *_, **__: [])
    with pytest.raises(TimeoutError, match="malformed model metadata"):
        sdg._wait_readiness(request, {"llm": Process()})


def test_malformed_minimal_inference_enters_bounded_readiness_classification(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    payload["limits"]["startup_timeout_s"] = 1
    payload["request_sha256"] = sdg._canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    request = sdg.load_request(path)

    class Process:
        def poll(self):
            return None

    ticks = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(sdg.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(sdg.time, "sleep", lambda _: None)
    monkeypatch.setenv("VLLM_API_KEY", "server-secret")
    monkeypatch.setattr(
        sdg, "_request_json",
        lambda url, **_: ({"data": [{"id": request["models"]["llm"]["id"]}]}
                           if url.endswith("/models") else []),
    )
    with pytest.raises(TimeoutError, match="minimal inference returned no choices"):
        sdg._wait_readiness(request, {"llm": Process()})


def test_worker_cleanup_polls_squeue_then_sacct_until_terminal(tmp_path, monkeypatch):
    path, _ = _request(tmp_path)
    request = sdg.load_request(path)
    accounting = iter((b"RUNNING|\n", b"CANCELLED|\n"))
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "squeue":
            return _completed(argv, stdout=b"")
        return _completed(argv, stdout=next(accounting))

    monkeypatch.setattr(sdg, "_run", run)
    monkeypatch.setattr(sdg.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(sdg.time, "sleep", lambda _: None)
    records = [{"native_id": "1000", "cleanup": "accepted"}]
    sdg._poll_worker_terminations(request, records)
    assert records[0]["cleanup"] == "canceled"
    assert records[0]["native_state"] == "CANCELLED"
    assert [call[0][0] for call in calls] == ["squeue", "sacct", "squeue", "sacct"]
    assert all(call[1]["timeout_s"] == 10 for call in calls)


def test_worker_cleanup_timeout_is_bounded_and_preserves_last_state(tmp_path, monkeypatch):
    path, _ = _request(tmp_path)
    request = sdg.load_request(path)
    clock = [0.0]
    monkeypatch.setattr(sdg.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sdg, "_local_worker_state", lambda native, **kwargs: "RUNNING")
    monkeypatch.setattr(sdg.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    records = [
        {"native_id": str(native), "cleanup": "accepted"}
        for native in (1000, 1001, 1002)
    ]
    sdg._poll_worker_terminations(request, records)
    assert clock[0] == 10  # request startup timeout, once globally rather than once per worker
    assert all(record["cleanup"] == "failed" for record in records)
    assert all("deadline exceeded in state RUNNING" in record["error"] for record in records)


def test_cleanup_scancels_every_owned_worker_before_polling(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    request = sdg.load_request(path)
    workers = _image_owners(payload)["workers"]
    events = []
    monkeypatch.setattr(
        sdg, "_local_job_name",
        lambda native: f"tao-job-123-img-{int(native)-1000:03d}",
    )
    monkeypatch.setattr(
        sdg, "_run",
        lambda argv, **kwargs: events.append(f"scancel:{argv[1]}") or _completed(argv),
    )
    monkeypatch.setattr(
        sdg, "_local_worker_state",
        lambda native, **kwargs: events.append(f"poll:{native}") or "CANCELLED",
    )
    records = sdg._cleanup_image_workers(request, workers)
    assert events[:3] == ["scancel:1000", "scancel:1001", "scancel:1002"]
    assert events[3:] == ["poll:1000", "poll:1001", "poll:1002"]
    assert all(record["cleanup"] == "canceled" for record in records)


def test_cleanup_retries_transient_ownership_and_state_query_failures(
    tmp_path, monkeypatch,
):
    path, payload = _request(tmp_path)
    request = sdg.load_request(path)
    worker = _image_owners(payload)["workers"][:1]
    clock = [0.0]
    ownership_calls = [0]
    state_calls = [0]

    def job_name(native_id):
        ownership_calls[0] += 1
        if ownership_calls[0] == 1:
            raise subprocess.TimeoutExpired(["scontrol"], 10)
        return "tao-job-123-img-000"

    def worker_state(native_id, **kwargs):
        state_calls[0] += 1
        if state_calls[0] == 1:
            raise ValueError("image-worker squeue query timed out")
        return "CANCELLED"

    monkeypatch.setattr(sdg, "_local_job_name", job_name)
    monkeypatch.setattr(sdg, "_local_worker_state", worker_state)
    monkeypatch.setattr(sdg, "_run", lambda argv, **kwargs: _completed(argv))
    monkeypatch.setattr(sdg.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        sdg.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    records = sdg._cleanup_image_workers(request, worker)
    assert ownership_calls == [2]
    assert state_calls == [2]
    assert records[0]["owned"] is True
    assert records[0]["cleanup"] == "canceled"
    assert records[0]["native_state"] == "CANCELLED"
    assert "last_ownership_error" not in records[0]
    assert "last_query_error" not in records[0]


def test_cleanup_reconciles_timed_out_scancel_and_retries_only_exact_active_worker(
    tmp_path, monkeypatch,
):
    path, payload = _request(tmp_path)
    request = sdg.load_request(path)
    worker = _image_owners(payload)["workers"][:1]
    clock = [0.0]
    cancel_calls = [0]
    states = iter(("RUNNING", "CANCELLED"))

    monkeypatch.setattr(sdg, "_local_job_name", lambda native: "tao-job-123-img-000")
    monkeypatch.setattr(sdg, "_local_worker_state", lambda *args, **kwargs: next(states))

    def run(argv, **kwargs):
        if argv[:1] == ["scancel"]:
            cancel_calls[0] += 1
            if cancel_calls[0] == 1:
                raise subprocess.TimeoutExpired(argv, 10)
        return _completed(argv)

    monkeypatch.setattr(sdg, "_run", run)
    monkeypatch.setattr(sdg.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        sdg.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    records = sdg._cleanup_image_workers(request, worker)
    assert cancel_calls == [2]
    assert records[0]["owned"] is True
    assert records[0]["cleanup"] == "canceled"
    assert records[0]["native_state"] == "CANCELLED"
    assert "last_cancel_error" not in records[0]


def test_pool_requires_all_24_ordered_capacity_one_endpoints(tmp_path):
    path, payload = _request(tmp_path)
    stage = pathlib.Path(payload["stage_dir"])
    pool_path = stage / "endpoint_pool.json"
    pool_path.write_text(json.dumps(_pool(payload)))
    validated = sdg._load_endpoint_pool(sdg.load_request(path), pool_path)
    assert validated["required_capacity"] == 24
    assert len(validated["endpoints"]) == 24
    assert all(endpoint["capacity"] == 1 for endpoint in validated["endpoints"])
    broken = _pool(payload)
    broken["endpoints"].pop()
    pool_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="service count"):
        sdg._load_endpoint_pool(sdg.load_request(path), pool_path)


def test_coordinator_uses_ready_subset_of_approved_worker_maximum(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    request = sdg.load_request(path)
    owners = _image_owners(payload)["workers"]
    stage = pathlib.Path(payload["stage_dir"])
    pool = _pool(payload)
    for node in (1, 2):
        descriptor = {
            "schema_version": "1", "request_sha256": payload["request_sha256"],
            "model": pool["model"], "worker_index": node,
            "endpoints": pool["endpoints"][node * 8:(node + 1) * 8],
        }
        target = (
            stage / ".tao-runtime" / "image-workers"
            / f"tao-job-123-img-{node:03d}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(descriptor))
    clock = [0.0]
    monkeypatch.setattr(sdg.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        sdg.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(
        sdg, "_local_job_name",
        lambda native_id: f"tao-job-123-img-{int(native_id)-1000:03d}",
    )
    probes = []
    monkeypatch.setattr(
        sdg, "_probe_role", lambda request, role, **kw: probes.append(kw["base_url"]),
    )

    result = sdg._build_endpoint_pool(request, owners)

    assert result["required_capacity"] == 16
    assert len(result["endpoints"]) == len(probes) == 16
    assert result["endpoints"][0]["id"] == "img-001-gpu-0"
    assert result["endpoints"][-1]["id"] == "img-002-gpu-7"


def test_coordinator_builds_pool_only_after_strict_owner_and_reachability_checks(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    request = sdg.load_request(path)
    owners = _image_owners(payload)["workers"]
    stage = pathlib.Path(payload["stage_dir"])
    pool = _pool(payload)
    for node, worker in enumerate(owners):
        descriptor = {
            "schema_version": "1", "request_sha256": payload["request_sha256"],
            "model": pool["model"], "worker_index": node,
            "endpoints": pool["endpoints"][node * 8:(node + 1) * 8],
        }
        target = (
            stage / ".tao-runtime" / "image-workers"
            / f"tao-job-123-img-{node:03d}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(descriptor))
    monkeypatch.setattr(sdg, "_local_job_name", lambda native_id: f"tao-job-123-img-{int(native_id)-1000:03d}")
    probes = []
    monkeypatch.setattr(sdg, "_probe_role", lambda request, role, **kw: probes.append(kw["base_url"]))
    result = sdg._build_endpoint_pool(request, owners)
    assert result["required_capacity"] == 24 and len(probes) == 24
    assert (stage / "endpoint_pool.json").is_file()
    shared_scripts = REPO / "skills/applications/tao-run-deft-iaa/scripts"
    sys.path.insert(0, str(shared_scripts))
    try:
        from iaa_deft.sdg import validate_image_edit_endpoint_pool

        shared = validate_image_edit_endpoint_pool(json.loads((stage / "endpoint_pool.json").read_text()))
    finally:
        sys.path.remove(str(shared_scripts))
    assert shared["required_capacity"] == 24
    assert shared["endpoints"][0]["gpu_identity"] == "node-000.cluster/gpu-0"
    (stage / "endpoint_pool.json").unlink()
    monkeypatch.setattr(
        sdg, "_probe_role",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unreachable")),
    )
    with pytest.raises(OSError, match="unreachable"):
        sdg._build_endpoint_pool(request, owners)
    assert not (stage / "endpoint_pool.json").exists()


def test_signed_config_runtime_sqsh_and_commit_component_evidence_bind_end_to_end(tmp_path):
    path, payload = _request(tmp_path)
    config_path = pathlib.Path(payload["config_path"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "1",
        "images": {
            "augmentation": payload["component_sources"]["augmentation"],
            "auto_labeling": payload["component_sources"]["auto_labeling"],
            "image_edit_serving": payload["component_sources"]["image_edit"],
            "text_serving": payload["component_sources"]["text_serving"],
        },
        "models": {
            role: {"id": model["id"], "revision": model["revision"]}
            for role, model in payload["models"].items()
        },
        "generation": {"generation_nodes": 3, "gpus_per_generation_node": 8},
    }
    config_path.write_text(yaml.safe_dump(config))
    payload["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    runtime = pathlib.Path(payload["runtime_root"]) / "iaa_deft"
    runtime.mkdir(parents=True)
    (runtime / "runtime.py").write_text("VALUE = 1\n")
    payload["runtime_sha256"] = sdg._python_tree_sha256(runtime)
    for role in payload["images"]:
        image = tmp_path / "sqsh" / f"{role}.sqsh"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"hsqs-image")
        payload["images"][role] = str(image)
    payload["request_sha256"] = sdg._canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    request = sdg.load_request(path)
    immutable = sdg._verify_signed_inputs(request)
    pool = _pool(payload)
    auxiliary = {
        "request_sha256": payload["request_sha256"],
        "image_edit_pool": {
            "requested_capacity": 24, "requested_nodes": 3,
            "required_capacity": 24, "active_nodes": 3,
        },
        "roles": {
            role: {"model": payload["models"][role]["id"], "ready": True}
            for role in ("vlm", "llm")
        },
        "components": sdg._component_evidence(request, immutable),
    }
    shared_scripts = REPO / "skills/applications/tao-run-deft-iaa/scripts"
    sys.path.insert(0, str(shared_scripts))
    try:
        from commit_stage import _validated_sdg_endpoint_evidence

        _validated_sdg_endpoint_evidence(pool, auxiliary, config, "slurm")
    finally:
        sys.path.remove(str(shared_scripts))


def test_component_executes_exactly_once_because_shared_runtime_owns_retries(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return _completed(argv, rc=7)

    monkeypatch.setattr(sdg.subprocess, "run", run)
    stage = pathlib.Path(payload["stage_dir"])
    result = sdg.component(argparse.Namespace(
        request=path, job_id="tao-job-123", action="preprocess",
        input_root=stage / "source_ids", output_root=stage,
        source_key=None, attempt=1, target_attributes_json="{}",
        image_edit_endpoint_id=None, image_edit_url=None,
    ))
    assert result == 7
    assert len(calls) == 1
    assert calls[0][:6] == [
        "srun", "--overlap", "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=4",
    ]


def test_component_rejects_noncanonical_attributes_and_path_substitution(tmp_path):
    path, payload = _request(tmp_path)
    stage = pathlib.Path(payload["stage_dir"])
    base = dict(
        request=path, job_id="tao-job-123", action="augment",
        input_root=stage / "source_ids", output_root=stage,
        source_key="sample-1", attempt=1,
        target_attributes_json='{"top outer color": "red"}',
        image_edit_endpoint_id="img-000-gpu-0",
        image_edit_url="http://node-000.cluster:18102/v1",
    )
    with pytest.raises(ValueError, match="canonical"):
        sdg.component(argparse.Namespace(**base))
    base.update(target_attributes_json='{"top outer color":"red"}', output_root=tmp_path / "other")
    with pytest.raises(ValueError, match="signed SDG stage"):
        sdg.component(argparse.Namespace(**base))


def test_shared_runtime_receives_only_narrow_component_executor_contract(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    runtime = pathlib.Path(payload["runtime_root"])
    runtime.mkdir(parents=True)
    (runtime / "run_sdg_stage.py").write_text("# staged runtime\n")
    for output in payload["expected_outputs"]:
        candidate = pathlib.Path(output)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("{}\n")
    calls = []
    monkeypatch.setattr(
        sdg.subprocess, "run",
        lambda argv, **kwargs: calls.append(argv) or _completed(argv),
    )
    runtime_python = pathlib.Path(sys.executable)
    sdg._execute_shared(payload, path.resolve(), "tao-job-123", runtime_python)
    assert len(calls) == 2
    prepare, command = calls
    assert prepare[:3] == [str(runtime_python), str(runtime / "run_sdg_stage.py"), "prepare"]
    assert prepare[prepare.index("--config") + 1] == payload["config_path"]
    assert prepare[prepare.index("--output-root") + 1] == payload["stage_dir"]
    assert prepare[prepare.index("--dataset-root") + 1] == payload["dataset_root"]
    assert prepare[prepare.index("--gaps-parquet") + 1].endswith(
        "/iter_1/gaps/kpi_gaps.parquet"
    )
    assert prepare[prepare.index("--eval-pairs") + 1].endswith(
        "/iaa_splits/eval_pairs.json"
    )
    assert command[1:3] == [str(runtime / "run_sdg_stage.py"), "execute"]
    assert command[0] == str(runtime_python)
    assert command[command.index("--config") + 1] == prepare[prepare.index("--config") + 1]
    assert command[command.index("--output-root") + 1] == prepare[prepare.index("--output-root") + 1]
    assert command[command.index("--mined-pairs") + 1] == prepare[prepare.index("--mined-pairs") + 1]
    assert command[command.index("--eval-list") + 1] == prepare[prepare.index("--eval-list") + 1]
    assert command[command.index("--attribute-vocab") + 1] == prepare[prepare.index("--attribute-vocab") + 1]
    assert command[command.index("--execution-platform") + 1] == "slurm"
    assert command[command.index("--component-executor-request") + 1] == str(path.resolve())
    assert command[command.index("--component-executor-job-id") + 1] == "tao-job-123"
    assert command[command.index("--image-edit-endpoint-pool") + 1] == str(pathlib.Path(payload["stage_dir"]) / "endpoint_pool.json")
    assert "--explicit-unstarted-pool-rebind" not in command


def test_shared_runtime_declares_only_allowlisted_unstarted_pool_repair(
    tmp_path, monkeypatch
):
    path, payload = _request(tmp_path)
    runtime = pathlib.Path(payload["runtime_root"])
    runtime.mkdir(parents=True)
    (runtime / "run_sdg_stage.py").write_text("# staged runtime\n")
    for output in payload["expected_outputs"]:
        candidate = pathlib.Path(output)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("{}\n")
    payload["repair"] = {"kind": sdg.POOL_REBIND_REPAIR_KIND}
    calls = []
    monkeypatch.setattr(
        sdg.subprocess, "run",
        lambda argv, **kwargs: calls.append(argv) or _completed(argv),
    )
    sdg._execute_shared(payload, path.resolve(), "tao-job-123", pathlib.Path(sys.executable))
    assert "--explicit-unstarted-pool-rebind" in calls[1]


def test_shared_runtime_stops_before_execute_when_prepare_fails(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    runtime = pathlib.Path(payload["runtime_root"])
    runtime.mkdir(parents=True)
    (runtime / "run_sdg_stage.py").write_text("# staged runtime\n")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return _completed(argv, rc=7)

    monkeypatch.setattr(sdg.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="shared SDG prepare exited 7"):
        sdg._execute_shared(
            payload, path.resolve(), "tao-job-123", pathlib.Path(sys.executable),
        )
    assert len(calls) == 1
    assert calls[0][2] == "prepare"
    assert not (pathlib.Path(payload["stage_dir"]) / "sdg_plan.json").exists()


def test_runtime_python_is_only_selected_from_signed_workspace(tmp_path, monkeypatch):
    _, payload = _request(tmp_path)
    workspace = pathlib.Path(payload["results_dir"]).parent.parent
    runtime_python = workspace / ".venv/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("#!/bin/sh\n")
    runtime_python.chmod(0o755)
    probes = []

    def run(argv, **kwargs):
        probes.append(argv)
        return _completed(argv)

    monkeypatch.setattr(sdg, "_run", run)
    assert sdg._resolve_runtime_python(payload) == runtime_python
    assert len(probes) == 1 and probes[0][:2] == [str(runtime_python), "-c"]
    assert "pandas" in probes[0][2] and "pyarrow" in probes[0][2]


def test_runtime_python_rejects_missing_dependencies_without_system_fallback(
    tmp_path, monkeypatch,
):
    _, payload = _request(tmp_path)
    workspace = pathlib.Path(payload["results_dir"]).parent.parent
    for name in ("python", "python3"):
        candidate = workspace / ".venv/bin" / name
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("#!/bin/sh\n")
        candidate.chmod(0o755)
    calls = []
    monkeypatch.setattr(
        sdg, "_run", lambda argv, **kwargs: calls.append(argv) or _completed(argv, rc=1),
    )
    with pytest.raises(ValueError, match="no approved IAA runtime interpreter"):
        sdg._resolve_runtime_python(payload)
    assert [pathlib.Path(call[0]).name for call in calls] == ["python", "python3"]
    assert all(call[0] != sys.executable for call in calls)


def test_augment_component_rejects_endpoint_not_bound_to_pool(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    stage = pathlib.Path(payload["stage_dir"])
    (stage / "endpoint_pool.json").write_text(json.dumps(_pool(payload)))
    args = argparse.Namespace(
        request=path, job_id="tao-job-123", action="augment",
        input_root=stage / "source_ids", output_root=stage,
        source_key="sample-1", attempt=1, target_attributes_json="{}",
        image_edit_endpoint_id="img-000-gpu-0",
        image_edit_url="http://wrong-node:18102/v1",
    )
    with pytest.raises(ValueError, match="not bound"):
        sdg.component(args)
    args.image_edit_url = "http://node-000.cluster:18102/v1"
    monkeypatch.setattr(sdg.subprocess, "run", lambda argv, **kwargs: _completed(argv))
    assert sdg.component(args) == 0


def test_cancel_requires_exact_owned_job_and_confirmation(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    record = _record(tmp_path, payload)
    calls = []

    group = {
        "schema_version": "1", "request_sha256": payload["request_sha256"],
        "job_id": "tao-job-123",
        "coordinator": {"role": "coordinator", "name": "tao-job-123-coord", "native_id": "999", "reconciled": False},
        "image_workers": _image_owners(payload)["workers"],
    }

    def ssh(login, command):
        calls.append(command)
        if "slurm-job-group.tao-job-123.json" in command:
            return _completed([], stdout=json.dumps(group).encode())
        if "scontrol show job" in command:
            native_id = command.split("show job -o ", 1)[1].split()[0]
            name = "tao-job-123-coord" if native_id == "999" else f"tao-job-123-img-{int(native_id)-1000:03d}"
            return _completed([], stdout=f"TAO_JOB_NAME={name}\n".encode())
        return _completed([])

    monkeypatch.setattr(sdg, "_ssh", ssh)
    args = argparse.Namespace(
        request=path, job_record=record, job_id="tao-job-123", login="user@login",
        backend_ref="999", confirm=False,
    )
    with pytest.raises(ValueError, match="--confirm"):
        sdg.cancel(args)
    args.confirm = True
    result = sdg.cancel(args)
    assert result["status"] == "CANCELED"
    assert "scancel 999 1000 1001 1002" in calls
    assert "endpoint-auth.tao-job-123.env" in calls[-1]


def test_image_owner_manifest_rejects_silent_capacity_reduction(tmp_path):
    path, payload = _request(tmp_path)
    owner_path = tmp_path / "image-owners.json"
    owners = _image_owners(payload)
    owners["workers"].pop()
    owner_path.write_text(json.dumps(owners))
    with pytest.raises(ValueError, match="wrong worker count"):
        sdg._load_image_owners(sdg.load_request(path), owner_path, "tao-job-123")


def test_submit_creates_n_independent_workers_then_distinct_coordinator(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    record = _record(tmp_path, payload)
    submitted = []
    staged_json = []
    staged_runtime = []
    verified_inputs = []
    monkeypatch.setattr(
        sdg, "_verify_remote_submit_inputs",
        lambda login, request: verified_inputs.append((login, request["request_sha256"])),
    )
    monkeypatch.setattr(
        sdg, "_stage_file",
        lambda login, local, remote: hashlib.sha256(local.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        sdg,
        "_stage_shared_runtime",
        lambda login, source, remote, digest: staged_runtime.append(
            (login, source, remote, digest)
        ) or {"run_sdg_stage.py": "d" * 64},
    )
    monkeypatch.setattr(sdg, "_ensure_remote_auth_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(sdg, "_exact_job_ids", lambda *args: [])
    monkeypatch.setattr(sdg, "_stage_json", lambda login, body, remote: staged_json.append((body, remote)) or "b" * 64)
    monkeypatch.setattr(sdg, "_acquire_remote_lock", lambda *args: None)
    monkeypatch.setattr(sdg, "_release_remote_lock", lambda *args: None)

    def submit_rendered(login, *, rendered, remote_script, job_name, **kwargs):
        submitted.append((job_name, rendered, remote_script))
        return str(2000 + len(submitted)), "c" * 64, False, "e" * 64

    monkeypatch.setattr(sdg, "_submit_rendered", submit_rendered)
    result = sdg.submit(argparse.Namespace(
        request=path, job_record=record, job_id="tao-job-123", login="user@login",
        remote_script=pathlib.Path("/lustre/run/job"), env_file=None,
        account=None, partition=None,
    ))
    assert [item[0] for item in submitted] == [
        "tao-job-123-img-000", "tao-job-123-img-001", "tao-job-123-img-002",
        "tao-job-123-coord",
    ]
    assert all("#SBATCH --gres=gpu:8" in item[1] for item in submitted[:3])
    assert "#SBATCH --gres=gpu:2" in submitted[3][1]
    assert all("sdg.action.tao-job-123.json" in item[1] for item in submitted)
    assert all("slurm_sdg_action.tao-job-123.py" in item[1] for item in submitted)
    assert all("endpoint-auth.tao-job-123.env" in item[1] for item in submitted)
    assert result["backend_ref"] == "2004" and len(result["image_workers"]) == 3
    assert staged_json[-1][0]["coordinator"]["native_id"] == "2004"
    assert staged_json[-2][1].name == "image-owners.tao-job-123.json"
    assert staged_json[-1][1].name == "slurm-job-group.tao-job-123.json"
    assert result["job_group"].endswith("/slurm-job-group.tao-job-123.json")
    assert len(staged_runtime) == 1
    assert staged_runtime[0][2] == pathlib.Path(payload["runtime_root"])
    assert result["runtime_files_sha256"] == {"run_sdg_stage.py": "d" * 64}
    assert len(result["submit_intent_sha256"]) == 4
    assert verified_inputs == [("user@login", payload["request_sha256"])]


def test_remote_submit_preflight_rejects_missing_cache_before_gpu_work(tmp_path, monkeypatch):
    request_path, payload = _request(tmp_path)
    cache = pathlib.Path(payload["cache_dir"])
    dataset = pathlib.Path(payload["dataset_root"])
    config = pathlib.Path(payload["config_path"])
    cache.mkdir(parents=True)
    dataset.mkdir(parents=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("schema_version: '1'\n")
    payload["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    for key in sorted(payload["images"]):
        image = tmp_path / "images" / f"{key}.sqsh"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"hsqs-prepared-image")
        payload["images"][key] = str(image)
    payload["request_sha256"] = sdg._canonical_sha256(payload)
    request_path.write_text(json.dumps(payload))
    request = sdg.load_request(request_path)
    monkeypatch.setattr(
        sdg,
        "_ssh",
        lambda login, command: subprocess.run(
            ["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        ),
    )
    sdg._verify_remote_submit_inputs("unused", request)
    cache.rmdir()
    with pytest.raises(ValueError, match="cache_dir must be an existing"):
        sdg._verify_remote_submit_inputs("unused", request)


def test_remote_submit_lock_is_atomic_owner_checked_and_reusable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sdg,
        "_ssh",
        lambda login, command: subprocess.run(
            ["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        ),
    )
    lock = tmp_path / "runtime" / "submit.job.lock"
    sdg._acquire_remote_lock("unused", lock, "owner-one")
    assert lock.read_text() == "owner-one\n"
    with pytest.raises(ValueError, match="holds the request lock"):
        sdg._acquire_remote_lock("unused", lock, "owner-two")
    with pytest.raises(ValueError, match="request-lock release"):
        sdg._release_remote_lock("unused", lock, "owner-two")
    assert lock.read_text() == "owner-one\n"
    sdg._release_remote_lock("unused", lock, "owner-one")
    assert not lock.exists()
    sdg._acquire_remote_lock("unused", lock, "owner-two")
    sdg._release_remote_lock("unused", lock, "owner-two")


def test_submit_lock_blocks_overlap_and_releases_after_failure(tmp_path, monkeypatch):
    path, _ = _request(tmp_path)
    args = argparse.Namespace(request=path, job_id="tao-job-123", login="user@login")
    events = []
    monkeypatch.setattr(
        sdg, "_acquire_remote_lock", lambda login, lock, token: events.append(("acquire", lock, token)),
    )
    monkeypatch.setattr(
        sdg, "_release_remote_lock", lambda login, lock, token: events.append(("release", lock, token)),
    )
    monkeypatch.setattr(
        sdg, "_submit_unlocked",
        lambda _: (_ for _ in ()).throw(ValueError("submission failed")),
    )
    with pytest.raises(ValueError, match="submission failed"):
        sdg.submit(args)
    assert [event[0] for event in events] == ["acquire", "release"]
    assert events[0][1] == events[1][1]
    assert events[0][2] == events[1][2]

    events.clear()
    monkeypatch.setattr(
        sdg,
        "_acquire_remote_lock",
        lambda *unused: (_ for _ in ()).throw(ValueError("request lock")),
    )
    with pytest.raises(ValueError, match="request lock"):
        sdg.submit(args)
    assert events == []


def test_duplicate_submit_recovery_is_explicit_exact_and_quarantines_terminal(
    tmp_path, monkeypatch,
):
    request_path, payload = _request(tmp_path)
    job_id = "tao-job-123"
    record = _record(tmp_path, payload, job_id)
    record.write_text(json.dumps({
        "id": job_id, "platform": "slurm", "action": payload["action_id"],
        "backend_ref": "2004", "terminal_state": None,
    }))
    group = {
        "schema_version": "1", "request_sha256": payload["request_sha256"],
        "job_id": job_id,
        "coordinator": {
            "role": "coordinator", "name": f"{job_id}-coord",
            "native_id": "2005", "reconciled": False,
        },
        "image_workers": _image_owners(payload, job_id)["workers"],
    }
    ids = {
        f"{job_id}-coord": ["2004", "2005"],
        **{
            f"{job_id}-img-{index:03d}": [str(1000 + index)]
            for index in range(payload["generation_nodes"])
        },
    }
    monkeypatch.setattr(sdg, "_remote_job_group", lambda *unused: group)
    monkeypatch.setattr(sdg, "_exact_job_ids", lambda login, name: ids[name])
    monkeypatch.setattr(sdg, "_assert_job_ownership", lambda *unused: None)
    monkeypatch.setattr(sdg, "_native_state", lambda *unused: "FAILED")
    monkeypatch.setattr(sdg, "_acquire_remote_lock", lambda *unused: None)
    monkeypatch.setattr(sdg, "_release_remote_lock", lambda *unused: None)
    monkeypatch.setattr(
        sdg,
        "_remote_exists",
        lambda login, path, name, **kwargs: "slurm_sdg_terminal" in str(path),
    )

    def remote_json(login, path, name):
        job_name = path.name.removeprefix("submit-intent.").removesuffix(".json")
        return {
            "schema_version": "1", "request_sha256": payload["request_sha256"],
            "action_id": payload["action_id"], "attempt": 1,
            "job_id": job_id, "job_name": job_name, "script_sha256": "a" * 64,
        }

    monkeypatch.setattr(sdg, "_remote_json_file", remote_json)
    terminal_sha = "b" * 64
    commands = []

    def ssh(login, command):
        commands.append(command)
        if "duplicate terminal" in command:
            raise AssertionError("operation labels are not shell commands")
        if "mv --" in command and "duplicate-submit-evidence" in command:
            return _completed([], stdout=f"715\n{terminal_sha}  terminal.json\n".encode())
        return _completed([])

    monkeypatch.setattr(sdg, "_ssh", ssh)
    staged = []
    monkeypatch.setattr(
        sdg, "_stage_json", lambda login, body, remote: staged.append((body, remote)) or "c" * 64,
    )
    monkeypatch.setattr(
        sdg, "_load_duplicate_recovery", lambda *unused: staged[-1][0],
    )
    args = argparse.Namespace(
        request=request_path, login="user@login", job_id=job_id,
        job_record=record, confirm=False,
    )
    with pytest.raises(ValueError, match="requires --confirm"):
        sdg.recover_duplicate_submit(args)
    args.confirm = True
    recovered = sdg.recover_duplicate_submit(args)
    assert recovered["record_backend_ref"] == "2004"
    assert recovered["group_backend_ref"] == "2005"
    assert recovered["coordinator_native_ids"] == ["2004", "2005"]
    assert recovered["worker_native_ids"] == ["1000", "1001", "1002"]
    assert recovered["native_states"] == {
        "1000": "FAILED", "1001": "FAILED", "1002": "FAILED",
        "2004": "FAILED", "2005": "FAILED",
    }
    assert recovered["quarantined_terminal"]["size"] == 715
    assert recovered["quarantined_terminal"]["sha256"] == terminal_sha
    unsigned = dict(recovered)
    evidence_digest = unsigned.pop("evidence_sha256")
    assert evidence_digest == hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert not any(command.startswith("scancel ") for command in commands)


def test_remote_presence_probe_never_treats_transport_failure_as_absence(
    tmp_path, monkeypatch,
):
    states = iter((1, 0, 255))
    monkeypatch.setattr(
        sdg, "_ssh",
        lambda login, command: _completed([], rc=next(states), stderr=b"transport unavailable"),
    )
    path = tmp_path / "evidence.json"
    assert sdg._remote_exists("user@login", path, "evidence") is False
    assert sdg._remote_exists("user@login", path, "evidence") is True
    with pytest.raises(ValueError, match="status 255"):
        sdg._remote_exists("user@login", path, "evidence")


def test_retry_lineage_accepts_only_digest_bound_duplicate_recovery(tmp_path, monkeypatch):
    prior_path, prior = _request(tmp_path)
    job_id = "tao-job-123"
    record_path = tmp_path / f"{job_id}.json"
    record_path.write_text(json.dumps({
        "schema_version": 1, "id": job_id, "platform": "slurm",
        "backend_ref": "2004", "action": prior["action_id"],
        "results_dir": prior["stage_dir"], "terminal_state": "ERROR",
        "err_class": "ERR_INFRA", "redacted": True,
        "terminal_write_by": "agent",
        "transitions": [
            {"state": "PENDING"}, {"state": "RUNNING"}, {"state": "ERROR"},
        ],
    }))
    group = {
        "schema_version": "1", "request_sha256": prior["request_sha256"],
        "job_id": job_id,
        "coordinator": {
            "role": "coordinator", "name": f"{job_id}-coord",
            "native_id": "2005", "reconciled": False,
        },
        "image_workers": _image_owners(prior, job_id)["workers"],
    }
    group_sha = hashlib.sha256(
        json.dumps(group, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    recovery = {
        "record_backend_ref": "2004", "group_backend_ref": "2005",
        "job_group_sha256": group_sha,
        "native_states": {
            "1000": "FAILED", "1001": "FAILED", "1002": "FAILED",
            "2004": "FAILED", "2005": "FAILED",
        },
    }
    monkeypatch.setattr(sdg, "_remote_job_group", lambda *unused: group)
    monkeypatch.setattr(sdg, "_load_duplicate_recovery", lambda *unused: recovery)
    _, lineage = sdg._retry_lineage(prior_path, record_path, "user@login")
    assert lineage["native_states"] == recovery["native_states"]
    assert lineage["backend_ref"] == "2004"
    assert lineage["job_group_sha256"] == group_sha

    ordinary_group = json.loads(json.dumps(group))
    ordinary_group["coordinator"]["native_id"] = "2004"
    monkeypatch.setattr(sdg, "_remote_job_group", lambda *unused: ordinary_group)
    monkeypatch.setattr(sdg, "_assert_job_ownership", lambda *unused: None)
    monkeypatch.setattr(sdg, "_native_state", lambda *unused: "FAILED")
    monkeypatch.setattr(
        sdg, "_agent_cleanup_terminal_evidence",
        lambda *unused: (_ for _ in ()).throw(ValueError("missing cleanup evidence")),
    )
    with pytest.raises(ValueError, match="missing cleanup evidence"):
        sdg._retry_lineage(prior_path, record_path, "user@login")

    terminal_evidence = {
        "kind": "coordinator-cleanup-failure",
        "terminal_sha256": "1" * 64,
        "cleanup_sha256": "2" * 64,
        "expected_outputs_sha256": {
            output: "3" * 64 for output in prior["expected_outputs"]
        },
    }
    terminal_evidence["evidence_sha256"] = hashlib.sha256(json.dumps(
        terminal_evidence, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    monkeypatch.setattr(
        sdg, "_agent_cleanup_terminal_evidence", lambda *unused: terminal_evidence,
    )
    _, ordinary_lineage = sdg._retry_lineage(prior_path, record_path, "user@login")
    assert ordinary_lineage["terminal_evidence"] == terminal_evidence

    monkeypatch.setattr(sdg, "_remote_job_group", lambda *unused: group)
    recovery["job_group_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not bind retry ownership"):
        sdg._retry_lineage(prior_path, record_path, "user@login")


def test_duplicate_recovery_loader_rejects_mutation_and_verifies_quarantine(
    tmp_path, monkeypatch,
):
    _, request = _request(tmp_path)
    job_id = "tao-job-123"
    stage = pathlib.Path(request["stage_dir"])
    terminal = stage / f"slurm_sdg_terminal.{job_id}.json"
    archive = stage / ".tao-runtime" / f"duplicate-submit-evidence.{job_id}" / "terminal.json"
    payload = {
        "schema_version": "1", "workflow": "tao-run-deft-iaa",
        "kind": "slurm_sdg_duplicate_submit_recovery",
        "request_sha256": request["request_sha256"],
        "action_id": request["action_id"], "attempt": 1, "job_id": job_id,
        "job_record_sha256": "a" * 64, "record_backend_ref": "2004",
        "job_group_sha256": "b" * 64, "group_backend_ref": "2005",
        "worker_native_ids": ["1000", "1001", "1002"],
        "coordinator_native_ids": ["2004", "2005"],
        "native_states": {
            "1000": "FAILED", "1001": "FAILED", "1002": "FAILED",
            "2004": "FAILED", "2005": "FAILED",
        },
        "submit_intent_sha256": {
            f"{job_id}-img-000": "c" * 64,
            f"{job_id}-img-001": "d" * 64,
            f"{job_id}-img-002": "e" * 64,
            f"{job_id}-coord": "f" * 64,
        },
        "quarantined_terminal": {
            "original": str(terminal), "archive": str(archive),
            "size": 715, "sha256": "9" * 64,
        },
        "recorded_at": "2026-08-21T09:00:00+00:00",
    }
    payload["evidence_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    current = [payload]
    monkeypatch.setattr(sdg, "_remote_json_file", lambda *unused: current[0])
    monkeypatch.setattr(
        sdg, "_ssh",
        lambda login, command: _completed(
            [], stdout=("715\n" + "9" * 64 + "  terminal.json\n").encode(),
        ),
    )
    assert sdg._load_duplicate_recovery("user@login", request, job_id) == payload

    changed = json.loads(json.dumps(payload))
    changed["native_states"]["2005"] = "COMPLETED"
    current[0] = changed
    with pytest.raises(ValueError, match="digest mismatch"):
        sdg._load_duplicate_recovery("user@login", request, job_id)

    changed["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in changed.items() if key != "evidence_sha256"},
            sort_keys=True, separators=(",", ":"),
        ).encode()
    ).hexdigest()
    current[0] = changed
    monkeypatch.setattr(
        sdg, "_ssh", lambda login, command: _completed([], stdout=b"714\n" + b"9" * 64 + b"  terminal.json\n"),
    )
    with pytest.raises(ValueError, match="changed after recovery"):
        sdg._load_duplicate_recovery("user@login", request, job_id)


def test_agent_terminalized_retry_requires_bound_cleanup_failure_evidence(
    tmp_path, monkeypatch,
):
    _, request = _request(tmp_path)
    job_id = "tao-job-123"
    backend_ref = "2004"
    workers = _image_owners(request, job_id)["workers"]
    group = {
        "schema_version": "1", "request_sha256": request["request_sha256"],
        "job_id": job_id,
        "coordinator": {
            "role": "coordinator", "name": f"{job_id}-coord",
            "native_id": backend_ref, "reconciled": False,
        },
        "image_workers": workers,
    }
    terminal = {
        "schema_version": "1", "workflow": sdg.WORKFLOW, "kind": sdg.KIND,
        "status": "error", "job_id": job_id, "action_id": request["action_id"],
        "coordinator_native_id": backend_ref,
        "request_sha256": request["request_sha256"],
        "resume_sha256": sdg._resume_sha256(request), "attempt": 1,
        "started_at": request["started_at"], "started_ns": request["started_ns"],
        "worker_started_at": "2026-08-19T12:01:00+00:00",
        "finished_at": "2026-08-19T12:02:00+00:00",
        "error": "owned image-worker cleanup did not complete",
    }
    cleanup = {
        "schema_version": "1", "job_id": job_id,
        "action_id": request["action_id"],
        "request_sha256": request["request_sha256"],
        "steps": [
            {
                "role": "image-worker", "native_id": worker["native_id"],
                "name": worker["name"], "owned": True,
                "cleanup": "failed" if index == 0 else "canceled",
            }
            for index, worker in enumerate(workers)
        ],
    }
    monkeypatch.setattr(
        sdg, "_remote_json_file",
        lambda login, path, name: terminal if "terminal" in name else cleanup,
    )
    monkeypatch.setattr(sdg, "_remote_file_sha256", lambda *unused: "a" * 64)
    evidence = sdg._agent_cleanup_terminal_evidence(
        "user@login", request, job_id, backend_ref, group,
    )
    assert evidence["kind"] == "coordinator-cleanup-failure"
    assert set(evidence["expected_outputs_sha256"]) == set(request["expected_outputs"])
    body = dict(evidence)
    digest = body.pop("evidence_sha256")
    assert digest == hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()

    terminal["request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="coordinator cleanup-failure"):
        sdg._agent_cleanup_terminal_evidence(
            "user@login", request, job_id, backend_ref, group,
        )
    terminal["request_sha256"] = request["request_sha256"]
    for step in cleanup["steps"]:
        step["cleanup"] = "canceled"
    with pytest.raises(ValueError, match="failed worker group"):
        sdg._agent_cleanup_terminal_evidence(
            "user@login", request, job_id, backend_ref, group,
        )


def test_cleanup_recovery_authors_digest_bound_receipt_and_recovered_terminal(
    tmp_path, monkeypatch,
):
    request_path, request = _request(tmp_path)
    job_id = "tao-job-123"
    record_path = tmp_path / f"{job_id}.json"
    record_path.write_text(json.dumps({
        "id": job_id, "platform": "slurm", "action": request["action_id"],
        "backend_ref": "2004", "terminal_state": None, "terminal_write_by": None,
    }))
    workers = _image_owners(request, job_id)["workers"]
    group = {
        "schema_version": "1", "request_sha256": request["request_sha256"],
        "job_id": job_id,
        "coordinator": {
            "role": "coordinator", "name": f"{job_id}-coord",
            "native_id": "2004", "reconciled": False,
        },
        "image_workers": workers,
    }
    terminal_evidence = {
        "kind": "coordinator-cleanup-failure", "terminal_sha256": "1" * 64,
        "cleanup_sha256": "2" * 64,
        "expected_outputs_sha256": {
            output: "3" * 64 for output in request["expected_outputs"]
        },
    }
    terminal_evidence["evidence_sha256"] = hashlib.sha256(json.dumps(
        terminal_evidence, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    staged = []
    published = []
    monkeypatch.setattr(sdg, "_remote_job_group", lambda *unused: group)
    monkeypatch.setattr(sdg, "_acquire_remote_lock", lambda *unused: None)
    monkeypatch.setattr(sdg, "_release_remote_lock", lambda *unused: None)
    monkeypatch.setattr(sdg, "_remote_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(sdg, "_agent_cleanup_terminal_evidence", lambda *unused, **kwargs: terminal_evidence)
    monkeypatch.setattr(sdg, "_assert_job_ownership", lambda *unused: None)
    monkeypatch.setattr(
        sdg, "_native_state",
        lambda login, native: "FAILED" if native == "2004" else "CANCELLED",
    )
    monkeypatch.setattr(
        sdg, "_ssh",
        lambda *args, **kwargs: _completed(
            [], stdout=(("a" * 64) + "  terminal.error.json\n").encode(),
        ),
    )
    monkeypatch.setattr(
        sdg, "_stage_json", lambda login, payload, path: staged.append((payload, path)) or "a" * 64,
    )
    monkeypatch.setattr(sdg, "_load_cleanup_recovery", lambda *unused: staged[-1][0])
    monkeypatch.setattr(
        sdg, "_publish_cleanup_recovered_terminal",
        lambda login, req, jid, recovery: published.append(recovery) or {},
    )
    result = sdg.recover_cleanup_failure(argparse.Namespace(
        request=request_path, login="user@login", job_id=job_id,
        job_record=record_path, confirm=True,
    ))
    assert result["kind"] == "slurm_sdg_cleanup_recovery"
    assert result["native_states"] == {
        "2004": "FAILED", "1000": "CANCELLED", "1001": "CANCELLED", "1002": "CANCELLED",
    }
    assert result["terminal_evidence"] == terminal_evidence
    assert result["archived_terminal_sha256"] == "a" * 64
    body = dict(result)
    digest = body.pop("evidence_sha256")
    assert digest == hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    assert published == [result]


def test_status_maps_failed_coordinator_complete_only_with_valid_cleanup_recovery(
    tmp_path, monkeypatch,
):
    request_path, request = _request(tmp_path)
    job_id = "tao-job-123"
    record = _record(tmp_path, request, job_id)
    group = {
        "schema_version": "1", "request_sha256": request["request_sha256"],
        "job_id": job_id,
        "coordinator": {
            "role": "coordinator", "name": f"{job_id}-coord",
            "native_id": "2004", "reconciled": False,
        },
        "image_workers": _image_owners(request, job_id)["workers"],
    }
    monkeypatch.setattr(sdg, "_remote_job_group", lambda *unused: group)
    monkeypatch.setattr(sdg, "_assert_job_ownership", lambda *unused: None)
    monkeypatch.setattr(sdg, "_native_state", lambda *unused: "FAILED")
    monkeypatch.setattr(sdg, "_remote_exists", lambda *args, **kwargs: True)
    loaded = []
    monkeypatch.setattr(
        sdg, "_load_cleanup_recovery", lambda *args: loaded.append(args) or {"kind": "recovery"},
    )
    result = sdg.status(argparse.Namespace(
        request=request_path, login="user@login", backend_ref="2004",
        job_id=job_id, job_record=record,
    ))
    assert result == {
        "status": "COMPLETE", "native_state": "FAILED", "backend_ref": "2004",
        "recovered_cleanup": True,
    }
    assert loaded


def test_cleanup_recovery_loader_revalidates_archive_group_and_terminal_evidence(
    tmp_path, monkeypatch,
):
    _, request = _request(tmp_path)
    job_id = "tao-job-123"
    workers = _image_owners(request, job_id)["workers"]
    group = {
        "schema_version": "1", "request_sha256": request["request_sha256"],
        "job_id": job_id,
        "coordinator": {
            "role": "coordinator", "name": f"{job_id}-coord",
            "native_id": "2004", "reconciled": False,
        },
        "image_workers": workers,
    }
    terminal_evidence = {
        "kind": "coordinator-cleanup-failure", "terminal_sha256": "1" * 64,
        "cleanup_sha256": "2" * 64,
        "expected_outputs_sha256": {
            output: "3" * 64 for output in request["expected_outputs"]
        },
        "evidence_sha256": "4" * 64,
    }
    archive = (
        pathlib.Path(request["stage_dir"]) / ".tao-runtime"
        / f"cleanup-recovery-evidence.{job_id}" / "terminal.error.json"
    )
    payload = {
        "schema_version": "1", "workflow": sdg.WORKFLOW,
        "kind": "slurm_sdg_cleanup_recovery",
        "request_sha256": request["request_sha256"],
        "action_id": request["action_id"], "attempt": request["attempt"],
        "job_id": job_id, "job_record_sha256": "5" * 64,
        "job_group_sha256": hashlib.sha256(json.dumps(
            group, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "coordinator_native_id": "2004",
        "native_states": {
            "2004": "FAILED", "1000": "CANCELLED",
            "1001": "CANCELLED", "1002": "CANCELLED",
        },
        "terminal_evidence": terminal_evidence,
        "archived_terminal": str(archive),
        "archived_terminal_sha256": "6" * 64,
        "recorded_at": "2026-08-21T13:30:00+00:00",
    }
    payload["evidence_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    current = [payload]
    monkeypatch.setattr(sdg, "_remote_json_file", lambda *unused: current[0])
    monkeypatch.setattr(sdg, "_remote_file_sha256", lambda *unused: "6" * 64)
    monkeypatch.setattr(
        sdg, "_agent_cleanup_terminal_evidence", lambda *unused, **kwargs: terminal_evidence,
    )
    assert sdg._load_cleanup_recovery("user@login", request, job_id, group) == payload
    current[0] = {**payload, "action_id": "other-action"}
    with pytest.raises(ValueError, match="digest mismatch"):
        sdg._load_cleanup_recovery("user@login", request, job_id, group)


def test_submit_rendered_reconciles_lost_response_without_duplicate_submit(tmp_path, monkeypatch):
    path, _ = _request(tmp_path)
    request = sdg.load_request(path)
    remote = pathlib.Path("/lustre/run/job.sbatch")
    intent = pathlib.Path("/lustre/run/submit-intent.tao-job-123-img-000.json")
    matches = iter(([], [], [], ["2001"]))
    ssh_calls = []
    clock = [0.0]
    monkeypatch.setattr(sdg, "_exact_job_ids", lambda *args: next(matches))
    monkeypatch.setattr(
        sdg, "_stage_file",
        lambda login, local, remote: hashlib.sha256(local.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(sdg, "_ensure_remote_intent", lambda *args, **kwargs: "b" * 64)
    monkeypatch.setattr(sdg, "_assert_job_ownership", lambda *args: None)
    monkeypatch.setattr(sdg, "_run", lambda argv, **kwargs: _completed(argv))
    monkeypatch.setattr(sdg.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sdg.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def ssh(login, command):
        ssh_calls.append(command)
        if command.startswith("sbatch --parsable"):
            return _completed([], rc=255, stderr=b"connection lost")
        return _completed([])

    monkeypatch.setattr(sdg, "_ssh", ssh)
    native, _, reconciled, _ = sdg._submit_rendered(
        "user@login", rendered="# job\n", remote_script=remote,
        job_name="tao-job-123-img-000", intent_path=intent,
        intent_binding={"request_sha256": request["request_sha256"]},
        retry_interval_s=2, reconcile_timeout_s=10,
    )
    assert native == "2001" and reconciled is True
    assert clock[0] == 2
    assert sum(command.startswith("sbatch --parsable") for command in ssh_calls) == 1


def test_submit_intent_is_create_once_idempotent_and_rejects_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sdg, "_ssh",
        lambda login, command: subprocess.run(
            ["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        ),
    )
    path = tmp_path / "runtime" / "submit-intent.job.json"
    payload = {"schema_version": "1", "job_name": "job", "request_sha256": "a" * 64}
    first = sdg._ensure_remote_intent("unused", path, payload)
    original = path.read_bytes()
    assert sdg._ensure_remote_intent("unused", path, payload) == first
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="submit-intent"):
        sdg._ensure_remote_intent("unused", path, {**payload, "request_sha256": "b" * 64})
    absent = tmp_path / "runtime" / "absent-intent.json"
    with pytest.raises(ValueError, match="submit-intent"):
        sdg._ensure_remote_intent("unused", absent, payload, allow_create=False)
    assert not absent.exists()


def test_submit_rendered_reuses_one_intent_owned_existing_job_and_rejects_ambiguity(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        sdg, "_stage_file",
        lambda *args: (_ for _ in ()).throw(AssertionError("existing job script overwritten")),
    )
    creations = []
    monkeypatch.setattr(
        sdg, "_ensure_remote_intent",
        lambda *args, **kwargs: creations.append(kwargs["allow_create"]) or "b" * 64,
    )
    monkeypatch.setattr(sdg, "_run", lambda argv, **kwargs: _completed(argv))
    owned = []
    monkeypatch.setattr(sdg, "_assert_job_ownership", lambda *args: owned.append(args))
    monkeypatch.setattr(sdg, "_exact_job_ids", lambda *args: ["2001"])
    result = sdg._submit_rendered(
        "user@login", rendered="# job\n", remote_script=pathlib.Path("/lustre/job"),
        job_name="tao-job-123-img-000", intent_path=pathlib.Path("/lustre/intent"),
        intent_binding={"request_sha256": "a" * 64},
    )
    assert result[0] == "2001" and result[2] is True and len(owned) == 1
    assert creations == [False]
    monkeypatch.setattr(sdg, "_exact_job_ids", lambda *args: ["2001", "2002"])
    with pytest.raises(ValueError, match="ambiguous existing SLURM ownership"):
        sdg._submit_rendered(
            "user@login", rendered="# job\n", remote_script=pathlib.Path("/lustre/job"),
            job_name="tao-job-123-img-000", intent_path=pathlib.Path("/lustre/intent"),
            intent_binding={"request_sha256": "a" * 64},
        )
    assert creations == [False]


def test_partial_submit_failure_preserves_jobs_intents_and_sidecar(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    record = _record(tmp_path, payload)
    calls = []
    monkeypatch.setattr(sdg, "_verify_remote_submit_inputs", lambda *args: None)
    monkeypatch.setattr(sdg, "_stage_file", lambda *args: "a" * 64)
    monkeypatch.setattr(sdg, "_stage_shared_runtime", lambda *args: {})
    monkeypatch.setattr(
        sdg, "_ensure_remote_auth_file",
        lambda *args, **kwargs: calls.append(("auth", kwargs["allow_create"])),
    )
    monkeypatch.setattr(sdg, "_exact_job_ids", lambda *args: [])
    monkeypatch.setattr(sdg, "_acquire_remote_lock", lambda *args: None)
    monkeypatch.setattr(sdg, "_release_remote_lock", lambda *args: None)

    attempts = [
        ("2001", "b" * 64, False, "c" * 64),
        ValueError("ambiguous existing SLURM ownership"),
    ]

    def submit_rendered(*args, **kwargs):
        result = attempts.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(sdg, "_submit_rendered", submit_rendered)
    monkeypatch.setattr(sdg, "_ssh", lambda login, command: calls.append(command) or _completed([]))
    with pytest.raises(ValueError, match="ambiguous"):
        sdg.submit(argparse.Namespace(
            request=path, job_record=record, job_id="tao-job-123", login="user@login",
            remote_script=pathlib.Path("/lustre/run/job"), env_file=None,
            account=None, partition=None,
        ))
    assert calls == [("auth", True)]


def test_remote_job_group_is_scoped_to_exact_job_record(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    requested = []
    group = {
        "schema_version": "1", "request_sha256": payload["request_sha256"],
        "job_id": "tao-job-123",
        "coordinator": {
            "role": "coordinator", "name": "tao-job-123-coord",
            "native_id": "999", "reconciled": False,
        },
        "image_workers": _image_owners(payload)["workers"],
    }

    def ssh(login, command):
        requested.append(command)
        return _completed([], stdout=json.dumps(group).encode())

    monkeypatch.setattr(sdg, "_ssh", ssh)
    loaded = sdg._remote_job_group("user@login", sdg.load_request(path), "tao-job-123")
    assert loaded == group
    assert "slurm-job-group.tao-job-123.json" in requested[0]
    assert "slurm-job-group.json" not in requested[0]
    with pytest.raises(ValueError, match="unsupported characters"):
        sdg._remote_job_group("user@login", sdg.load_request(path), "../other")


def test_shared_runtime_staging_is_exact_hash_bound_and_remotely_verified(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    package = source / "iaa_deft"
    package.mkdir(parents=True)
    (source / "run_sdg_stage.py").write_text("# runner\n")
    (package / "__init__.py").write_text("# package\n")
    (package / "sdg.py").write_text("VALUE = 1\n")
    (package / "ignored.txt").write_text("not runtime\n")
    staged = []
    remote_commands = []

    def stage(login, local, remote):
        staged.append((local.relative_to(source), remote))
        return hashlib.sha256(local.read_bytes()).hexdigest()

    monkeypatch.setattr(sdg, "_stage_file", stage)
    monkeypatch.setattr(
        sdg,
        "_ssh",
        lambda login, command: remote_commands.append(command) or _completed([]),
    )
    remote = pathlib.Path("/lustre/run/runtime")
    result = sdg._stage_shared_runtime(
        "user@login", source, remote, sdg._python_tree_sha256(package)
    )
    assert [item[0] for item in staged] == [
        pathlib.Path("run_sdg_stage.py"),
        pathlib.Path("iaa_deft/__init__.py"),
        pathlib.Path("iaa_deft/sdg.py"),
    ]
    assert set(result) == {str(item[0]) for item in staged}
    assert len(remote_commands) == 1 and "python3 -c" in remote_commands[0]
    with pytest.raises(ValueError, match="disagrees with initialized state"):
        sdg._stage_shared_runtime("user@login", source, remote, "0" * 64)


def test_coordinator_terminal_preserves_signed_start_evidence_on_failure(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    legacy_terminal = pathlib.Path(payload["stage_dir"]) / "slurm_sdg_terminal.json"
    legacy_terminal.write_text(json.dumps({
        "status": "error", "job_id": "prior-resource-failure",
        "request_sha256": "0" * 64, "attempt": 1,
    }))
    owners_path = tmp_path / "image-owners.json"
    owners_path.write_text(json.dumps(_image_owners(payload)))
    monkeypatch.setenv("SLURM_JOB_ID", "2004")
    monkeypatch.setenv("SLURM_JOB_NAME", "tao-job-123-coord")

    class Process:
        next_pid = 500

        def __init__(self, *args, **kwargs):
            self.pid = Process.next_pid
            Process.next_pid += 1
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode

    monkeypatch.setattr(sdg, "_port_available", lambda _: True)
    monkeypatch.setattr(sdg, "_verify_signed_inputs", lambda request: {"images": {
        "augmentation": request["component_sources"]["augmentation"],
        "auto_labeling": request["component_sources"]["auto_labeling"],
    }})
    monkeypatch.setattr(
        sdg, "_resolve_runtime_python", lambda request: pathlib.Path(sys.executable),
    )
    monkeypatch.setattr(sdg.subprocess, "Popen", Process)
    monkeypatch.setattr(sdg, "_wait_readiness", lambda *_: (_ for _ in ()).throw(TimeoutError("bounded")))
    monkeypatch.setattr(sdg, "_local_job_name", lambda native: f"tao-job-123-img-{int(native)-1000:03d}")
    monkeypatch.setattr(sdg, "_local_worker_state", lambda native, **kwargs: "CANCELLED")
    monkeypatch.setattr(sdg, "_run", lambda argv, **kwargs: _completed(argv))
    monkeypatch.setattr(sdg.os, "killpg", lambda *_: None)
    with pytest.raises(TimeoutError, match="bounded"):
        sdg.coordinator(argparse.Namespace(
            request=path, job_id="tao-job-123", job_group=owners_path,
        ))
    terminal = json.loads((
        pathlib.Path(payload["stage_dir"]) / "slurm_sdg_terminal.tao-job-123.json"
    ).read_text())
    assert terminal["started_at"] == payload["started_at"]
    assert terminal["started_ns"] == payload["started_ns"]
    assert terminal["coordinator_native_id"] == "2004"
    assert terminal["status"] == "error"
    assert json.loads(legacy_terminal.read_text())["job_id"] == "prior-resource-failure"
    assert (
        pathlib.Path(payload["stage_dir"])
        / "endpoint_cleanup.tao-job-123.json"
    ).is_file()
    cleanup = json.loads((
        pathlib.Path(payload["stage_dir"])
        / "endpoint_cleanup.tao-job-123.json"
    ).read_text())
    worker_steps = [step for step in cleanup["steps"] if step["role"] == "image-worker"]
    assert len(worker_steps) == 3
    assert all(step["owned"] is True and step["cleanup"] == "canceled" for step in worker_steps)


def test_coordinator_cannot_report_success_when_owned_worker_cancel_fails(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    owners_path = tmp_path / "image-owners.json"
    owners_path.write_text(json.dumps(_image_owners(payload)))
    monkeypatch.setenv("SLURM_JOB_ID", "2004")
    monkeypatch.setenv("SLURM_JOB_NAME", "tao-job-123-coord")

    class Process:
        next_pid = 700

        def __init__(self, *args, **kwargs):
            self.pid = Process.next_pid
            Process.next_pid += 1
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode

    monkeypatch.setattr(sdg, "_port_available", lambda _: True)
    monkeypatch.setattr(sdg, "_verify_signed_inputs", lambda request: {"images": {
        "augmentation": request["component_sources"]["augmentation"],
        "auto_labeling": request["component_sources"]["auto_labeling"],
    }})
    monkeypatch.setattr(sdg, "_resolve_runtime_python", lambda request: pathlib.Path(sys.executable))
    monkeypatch.setattr(sdg.subprocess, "Popen", Process)
    monkeypatch.setattr(sdg, "_wait_readiness", lambda *_: None)
    monkeypatch.setattr(sdg, "_build_endpoint_pool", lambda *_: {"required_capacity": 24})
    monkeypatch.setattr(sdg, "_execute_shared", lambda *_: None)
    monkeypatch.setattr(sdg, "_local_job_name", lambda native: f"tao-job-123-img-{int(native)-1000:03d}")
    clock = [0.0]
    monkeypatch.setattr(sdg.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        sdg.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(
        sdg, "_local_worker_state",
        lambda native, **kwargs: "RUNNING" if native == "1001" else "CANCELLED",
    )

    def run(argv, **kwargs):
        if argv[:1] == ["scancel"] and argv[1] == "1001":
            return _completed(argv, rc=1, stderr=b"scheduler rejected cancel")
        return _completed(argv)

    monkeypatch.setattr(sdg, "_run", run)
    monkeypatch.setattr(sdg.os, "killpg", lambda *_: None)
    with pytest.raises(RuntimeError, match="cleanup did not complete"):
        sdg.coordinator(argparse.Namespace(request=path, job_id="tao-job-123", job_group=owners_path))
    terminal = json.loads((
        pathlib.Path(payload["stage_dir"]) / "slurm_sdg_terminal.tao-job-123.json"
    ).read_text())
    assert terminal["status"] == "error"
    cleanup = json.loads((
        pathlib.Path(payload["stage_dir"]) / "endpoint_cleanup.tao-job-123.json"
    ).read_text())
    workers = [step for step in cleanup["steps"] if step["role"] == "image-worker"]
    assert [step["cleanup"] for step in workers] == ["canceled", "failed", "canceled"]


def test_coordinator_rejects_runtime_before_endpoint_processes(tmp_path, monkeypatch):
    path, payload = _request(tmp_path)
    owners_path = tmp_path / "image-owners.json"
    owners_path.write_text(json.dumps(_image_owners(payload)))
    monkeypatch.setenv("SLURM_JOB_ID", "2004")
    monkeypatch.setenv("SLURM_JOB_NAME", "tao-job-123-coord")
    monkeypatch.setattr(
        sdg, "_verify_signed_inputs", lambda request: {"images": {
            "augmentation": request["component_sources"]["augmentation"],
            "auto_labeling": request["component_sources"]["auto_labeling"],
        }},
    )
    monkeypatch.setattr(
        sdg, "_resolve_runtime_python",
        lambda request: (_ for _ in ()).throw(ValueError("runtime dependencies absent")),
    )
    monkeypatch.setattr(
        sdg.subprocess, "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("endpoint started")),
    )
    with pytest.raises(ValueError, match="runtime dependencies absent"):
        sdg.coordinator(argparse.Namespace(
            request=path, job_id="tao-job-123", job_group=owners_path,
        ))
    assert not (pathlib.Path(payload["stage_dir"]) / "endpoint-logs").exists()


def test_log_sanitizer_redacts_tokens_and_keys():
    text = (
        "HF_" + "TO" + "KEN=hf_abcdefghijk API_" + "KEY=supersecret "
        "pass" + "word=hunter2 Authorization: Bearer endpoint-secret safe=ok"
    )
    clean = sdg._sanitize(text)
    assert "abcdefghijk" not in clean
    assert "supersecret" not in clean
    assert "hunter2" not in clean
    assert "endpoint-secret" not in clean
    assert "safe=ok" in clean
