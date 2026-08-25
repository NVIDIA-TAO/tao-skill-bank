# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skills/platform/tao-run-on-brev/scripts/brev_sdg_action.py"
SPEC = importlib.util.spec_from_file_location("brev_sdg_action", SCRIPT)
assert SPEC and SPEC.loader
sdg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sdg)
SHARED_RUNTIME = ROOT / "skills/applications/tao-run-deft-iaa/scripts"
sys.path.insert(0, str(SHARED_RUNTIME))
from iaa_deft import sdg as shared_sdg  # noqa: E402
import run_sdg_stage as shared_stage  # noqa: E402
import manage_sdg_endpoints as shared_manager  # noqa: E402


def _sign(payload: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(payload)
    payload.pop("request_sha256", None)
    payload["request_sha256"] = sdg._canonical_sha256(payload)
    return payload


def _request(
    tmp_path: pathlib.Path, *, single_host: bool = False,
) -> tuple[pathlib.Path, dict[str, Any]]:
    results = tmp_path / "results" / "run-01"
    output = results / "iter_1" / "datagen"
    dataset = tmp_path / "dataset"
    runtime = tmp_path / "runtime"
    payload = {
        "schema_version": "1", "workflow": sdg.WORKFLOW, "kind": sdg.KIND,
        "platform": "brev", "name": sdg.NAME, "action_id": "action-01",
        "run_id": "run-01", "iteration": 1, "attempt": 1,
        "started_at": "2026-08-19T20:00:00Z", "started_ns": 1,
        "local": {
            "results_dir": str(results), "stage_dir": str(output),
            "expected_outputs": [
                str(output / "dataset" / "sdg_manifest.json"),
                str(output / "dataset" / "sdg_pairs.json"),
                str(output / "dataset" / "sdg_image_list.txt"),
                str(output / "sdg_execution_manifest.json"),
            ],
        },
        "remote": {
            "results_dir": str(results), "stage_dir": str(output),
            "dataset_root": str(dataset),
            "config_path": str(results / "config" / "sdg_config.yaml"),
            "config_sha256": "1" * 64, "runtime_root": str(runtime),
            "runtime_sha256": "2" * 64, "cache_dir": str(tmp_path / "cache"),
            "controller_python": str(tmp_path / ".venv" / "bin" / "python"),
            "mined_pairs": str(results / "iter_1" / "mined_pairs.json"),
            "eval_list": str(results / "iaa_splits" / "eval_list.txt"),
            "gaps_parquet": str(results / "iter_1" / "gaps" / "kpi_gaps.parquet"),
            "eval_pairs": str(results / "iaa_splits" / "eval_pairs.json"),
            "attribute_vocab": str(dataset / "attribute_vocab.json"),
            "smoke_image": str(dataset / "smoke.jpg"),
            "endpoint_pool_path": str(output / "endpoint_pool.json"),
            "expected_outputs": [
                str(output / "dataset" / "sdg_manifest.json"),
                str(output / "dataset" / "sdg_pairs.json"),
                str(output / "dataset" / "sdg_image_list.txt"),
                str(output / "sdg_execution_manifest.json"),
            ],
        },
        "topology": "single_host" if single_host else "multi_host",
        "generation_nodes": 1 if single_host else 2,
        "coordinator": ({
            "instance": "brev-a", "gpu_ids": {"vlm": [4], "llm": [5], "tao": [6, 7]},
            "gpu_count": 8, "gpu_memory_mib": [81920] * 8,
        } if single_host else {
            "instance": "brev-a", "gpu_ids": {"vlm": [0], "llm": [1]},
        }),
        "workers": ([{
            "id": "worker-0", "instance": "brev-a", "address": "127.0.0.1",
            "gpu_ids": [0, 1, 2, 3], "ports": list(range(18102, 18106)),
        }] if single_host else [
            {
                "id": f"worker-{index}", "instance": f"brev-worker-{index}",
                "address": f"10.0.0.{index + 10}", "gpu_ids": list(range(8)),
                "ports": list(range(18102, 18110)),
            }
            for index in range(2)
        ]),
        "resources": ({
            "generation_nodes": 1, "gpus_per_worker": 4, "capacity_per_worker": 4,
            "coordinator_vlm_gpus": 1, "coordinator_llm_gpus": 1,
            "tao_gpus": 2, "host_gpu_count": 8, "host_min_vram_mib": 80000,
        } if single_host else {
            "generation_nodes": 2, "gpus_per_worker": 8, "capacity_per_worker": 8,
            "coordinator_vlm_gpus": 1, "coordinator_llm_gpus": 1,
        }),
        "models": {
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
        },
        "limits": {
            "startup_timeout_s": 600, "request_timeout_s": 180,
            "retry_interval_s": 15, "image_edit_request_timeout_s": 600,
            "verification_max_attempts": 2, "max_samples_per_iteration": 10,
        },
        "bindings": {"state_sha256": "3" * 64, "inventory_sha256": "4" * 64},
        "forward_env": ["HF_TOKEN"],
        "timeouts": {"controller_s": 3600, "worker_s": 600, "readiness_s": 300, "cancel_s": 10},
        "resume": {"max_controller_attempts": 2},
    }
    payload = _sign(payload)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _prepare_inputs(
    tmp_path: pathlib.Path, *, requires_hf_token: bool = True,
    single_host: bool = False,
) -> argparse.Namespace:
    workspace = tmp_path / "workspace"
    results = workspace / "results" / "run-01"
    dataset = workspace / "data" / "iaa"
    (results / "config").mkdir(parents=True)
    (results / "iter_1" / "mining").mkdir(parents=True)
    (results / "iter_1" / "gaps").mkdir(parents=True)
    (results / "iaa_splits").mkdir(parents=True)
    (dataset / "images").mkdir(parents=True)
    image = dataset / "images" / "source.jpg"
    image.write_bytes(b"jpeg")
    (dataset / "attribute_vocab.json").write_text('{"attributes": ["shoe color"]}\n')
    (results / "iaa_splits" / "eval_list.txt").write_text("eval.jpg\n")
    (results / "iaa_splits" / "eval_pairs.json").write_text('[{"unique_name":"eval.jpg"}]\n')
    (results / "iter_1" / "gaps" / "kpi_gaps.parquet").write_bytes(b"parquet-evidence")
    mined_pairs = results / "iter_1" / "mining" / "mined_pairs.json"
    mined_pairs.write_text(json.dumps([{
        "unique_name": "source.jpg", "image_path": "images/source.jpg",
    }]))
    sdg_config = {
        "endpoints": {
            "ownership": "managed", "reuse_requested": False,
            "startup_timeout_s": 600, "request_timeout_s": 180,
            "retry_interval_s": 15,
            "gpu_ids": (
                {"image_edit": [0, 1, 2, 3], "vlm": [4], "llm": [5]}
                if single_host else {"image_edit": list(range(8)), "vlm": [0], "llm": [1]}
            ),
        },
        "generation": {
            "generation_nodes": 1 if single_host else 2,
            "gpus_per_generation_node": 4 if single_host else 8,
            "image_edit_request_timeout_s": 600,
            "verification_max_attempts": 2, "max_samples_per_iteration": 10,
        },
        "models": {
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
        },
    }
    config_path = results / "config" / "sdg_config.yaml"
    config_path.write_text(sdg.yaml.safe_dump(sdg_config, sort_keys=True))
    state = {
        "schema_version": "3", "workflow": sdg.WORKFLOW,
        "started_at": "2026-08-19T20:00:00+00:00",
        "results_dir": str(results), "max_iterations": 3, "current_iteration": 1,
        "config": {
            "workspace": str(workspace), "dataset_root": str(dataset),
            "platform": "brev", "requires_hf_token": requires_hf_token,
            "gpu_ids": [6, 7] if single_host else [0],
            "sdg_config": str(config_path),
            "sdg_config_sha256": sdg._file_sha256(config_path),
            "sdg": {
                "endpoint_mode": "managed", "reuse_requested": False,
                "generation_nodes": 1 if single_host else 2,
                "gpus_per_generation_node": 4 if single_host else 8,
                "gpu_ids": sdg_config["endpoints"]["gpu_ids"],
                "models": sdg_config["models"],
            },
        },
        "iterations": {"iter1": {
            "status": "in_progress", "stage_completed": "history_select",
            "mined_pairs": str(mined_pairs),
        }},
    }
    state_path = results / "deft_state.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    inventory = {
        "schema_version": "1", "platform": "brev", "status": "resolved",
        "topology": "single_host" if single_host else "multi_host",
        "coordinator": ({
            "instance": "brev-coordinator", "gpu_count": 8,
            "gpu_memory_mib": [81920] * 8,
        } if single_host else {"instance": "brev-coordinator"}),
        "workers": ([{
            "id": "worker-0", "instance": "brev-coordinator", "address": "127.0.0.1",
        }] if single_host else [
            {"id": f"worker-{index}", "instance": f"brev-worker-{index}",
             "address": f"10.20.0.{index + 10}"}
            for index in range(2)
        ]),
    }
    inventory["inventory_sha256"] = sdg._inventory_sha256(inventory)
    inventory_path = tmp_path / "brev-inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    return argparse.Namespace(
        state=state_path, iteration=1, inventory=inventory_path,
        remote_root=pathlib.Path("/srv/tao-workspace"),
        remote_cache=pathlib.Path("/srv/tao-cache"),
        remote_controller_python=pathlib.Path("/srv/tao-workspace/.venv/bin/python"),
        output=tmp_path / "prepared-request.json",
    )


def _record(tmp_path: pathlib.Path, request: dict[str, Any]) -> pathlib.Path:
    path = tmp_path / "job.json"
    path.write_text(json.dumps({
        "id": "job-01", "platform": "brev", "action": request["action_id"]
    }), encoding="utf-8")
    return path


@pytest.mark.parametrize("requires_hf_token,expected", [(True, ["HF_TOKEN"]), (False, [])])
def test_prepare_request_is_deterministic_and_derives_signed_topology(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
    requires_hf_token: bool, expected: list[str],
) -> None:
    args = _prepare_inputs(tmp_path, requires_hf_token=requires_hf_token)
    monkeypatch.setenv("HF_TOKEN", "must-not-be-read-or-persisted")
    assert sdg.prepare_request(args) == 0
    first = args.output.read_bytes()
    request = sdg.load_request(args.output)
    assert sdg.prepare_request(args) == 0
    assert args.output.read_bytes() == first
    assert request["generation_nodes"] == 2
    assert request["coordinator"]["gpu_ids"] == {"vlm": [0], "llm": [1]}
    assert all(worker["gpu_ids"] == list(range(8)) for worker in request["workers"])
    assert all(worker["ports"] == list(range(18102, 18110)) for worker in request["workers"])
    assert request["models"]["image_edit"]["revision"] == "1" * 40
    assert request["limits"]["max_samples_per_iteration"] == 10
    assert request["forward_env"] == expected
    assert "must-not-be-read-or-persisted" not in first.decode()
    assert request["remote"]["results_dir"] == "/srv/tao-workspace/results/run-01"
    assert request["bindings"]["state_sha256"] == sdg._file_sha256(args.state)


def test_prepare_request_binds_state_and_refuses_different_existing_output(
    tmp_path: pathlib.Path,
) -> None:
    args = _prepare_inputs(tmp_path)
    assert sdg.prepare_request(args) == 0
    state = json.loads(args.state.read_text())
    args.state.write_text(json.dumps(state, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="different existing"):
        sdg.prepare_request(args)


def test_prepare_request_cli_has_only_read_only_constructor_inputs() -> None:
    parsed = sdg._parser().parse_args([
        "prepare-request", "--state", "/run/deft_state.json",
        "--iteration", "2", "--inventory", "/run/brev-inventory.json",
        "--remote-root", "/srv/workspace", "--remote-cache", "/srv/cache",
        "--remote-controller-python", "/srv/workspace/.venv/bin/python",
        "--output", "/run/sdg.action.json",
    ])
    assert parsed.verb == "prepare-request" and parsed.iteration == 2
    assert not any("token" in name or "key" in name or "credential" in name for name in vars(parsed))


def test_prepare_request_rejects_committed_sdg(tmp_path: pathlib.Path) -> None:
    args = _prepare_inputs(tmp_path)
    state = json.loads(args.state.read_text())
    state["iterations"]["iter1"]["stage_completed"] = "sdg"
    args.state.write_text(json.dumps(state) + "\n")
    with pytest.raises(ValueError, match="already committed"):
        sdg._prepared_request(args)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda inventory: inventory.update({"platform": "slurm"}), "platform=brev"),
        (lambda inventory: inventory["workers"][1].update({"instance": inventory["workers"][0]["instance"]}), "instances must be distinct"),
        (lambda inventory: inventory["workers"][0].update({"address": "127.0.0.1"}), "directly reachable"),
    ],
)
def test_prepare_request_rejects_wrong_or_unsafe_inventory(
    tmp_path: pathlib.Path, mutation: Any, match: str,
) -> None:
    args = _prepare_inputs(tmp_path)
    inventory = json.loads(args.inventory.read_text())
    mutation(inventory)
    inventory["inventory_sha256"] = sdg._inventory_sha256(inventory)
    args.inventory.write_text(json.dumps(inventory) + "\n")
    with pytest.raises(ValueError, match=match):
        sdg._prepared_request(args)


def test_prepare_request_rejects_unapproved_worker_count_and_bad_inventory_digest(
    tmp_path: pathlib.Path,
) -> None:
    args = _prepare_inputs(tmp_path)
    inventory = json.loads(args.inventory.read_text())
    inventory["workers"].pop()
    inventory["inventory_sha256"] = sdg._inventory_sha256(inventory)
    args.inventory.write_text(json.dumps(inventory) + "\n")
    with pytest.raises(ValueError, match="exactly the approved generation_nodes"):
        sdg._prepared_request(args)
    inventory["inventory_sha256"] = "0" * 64
    args.inventory.write_text(json.dumps(inventory) + "\n")
    with pytest.raises(ValueError, match="canonical inventory"):
        sdg._prepared_request(args)


def _worker_evidence(request: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for worker in request["workers"]:
        endpoints = []
        for gpu_id, port in zip(worker["gpu_ids"], worker["ports"]):
            endpoints.append({
                "id": f"{worker['id']}-gpu{gpu_id}",
                "url": f"http://{worker['address']}:{port}/v1", "capacity": 1,
                "gpu_identity": f"{worker['instance']}/gpu:{gpu_id}",
                "owner": {"native_id": f"native{gpu_id}", "name": f"slot-{gpu_id}"},
            })
        result.append({
            "state": "READY", "worker_id": worker["id"], "capacity": len(worker["gpu_ids"]),
            "model": {
                "id": request["models"]["image_edit"]["id"],
                "revision": request["models"]["image_edit"]["revision"],
            },
            "endpoints": endpoints,
        })
    return result


def test_signed_request_binds_two_workers_with_eight_independent_slots_each(tmp_path: pathlib.Path) -> None:
    path, _ = _request(tmp_path)
    request = sdg.load_request(path)
    assert request["generation_nodes"] == 2
    assert all(worker["gpu_ids"] == list(range(8)) and len(worker["ports"]) == 8 for worker in request["workers"])
    assert request["resources"]["capacity_per_worker"] == 8
    serialized = json.dumps(request)
    assert "operations" not in request and "serve_args" not in serialized
    assert "--gpus all" not in serialized


def test_single_host_request_binds_all_eight_gpu_roles_without_relay(
    tmp_path: pathlib.Path,
) -> None:
    path, _ = _request(tmp_path, single_host=True)
    request = sdg.load_request(path)
    assert request["topology"] == "single_host"
    assert request["coordinator"]["instance"] == request["workers"][0]["instance"]
    assert request["coordinator"]["gpu_ids"] == {
        "vlm": [4], "llm": [5], "tao": [6, 7],
    }
    assert request["workers"][0]["gpu_ids"] == [0, 1, 2, 3]
    assert request["workers"][0]["address"] == "127.0.0.1"
    assert request["resources"] == {
        "generation_nodes": 1, "gpus_per_worker": 4, "capacity_per_worker": 4,
        "coordinator_vlm_gpus": 1, "coordinator_llm_gpus": 1,
        "tao_gpus": 2, "host_gpu_count": 8, "host_min_vram_mib": 80000,
    }
    assert sdg._required_capacity(request) == 4
    assert "--gpus all" not in json.dumps(request)


def test_prepare_request_derives_canonical_single_host_topology(
    tmp_path: pathlib.Path,
) -> None:
    args = _prepare_inputs(tmp_path, single_host=True)
    request = sdg._prepared_request(args)
    assert request["topology"] == "single_host"
    assert request["workers"] == [{
        "id": "worker-0", "instance": "brev-coordinator", "address": "127.0.0.1",
        "gpu_ids": [0, 1, 2, 3], "ports": [18102, 18103, 18104, 18105],
    }]
    assert request["coordinator"]["gpu_ids"]["tao"] == [6, 7]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p["coordinator"].update({"gpu_memory_mib": [79999] * 8}), "80000 MiB"),
        (lambda p: p["workers"][0].update({"gpu_ids": [0, 1, 2, 4]}), "gpu_ids"),
        (lambda p: p["coordinator"]["gpu_ids"].update({"tao": [0, 1]}), "coordinator.gpu_ids"),
        (lambda p: p["workers"][0].update({"address": "10.0.0.9"}), "127.0.0.1"),
    ],
)
def test_single_host_rejects_wrong_hardware_or_gpu_ownership(
    tmp_path: pathlib.Path, mutation: Any, match: str,
) -> None:
    _, payload = _request(tmp_path, single_host=True)
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        sdg.validate_request(_sign(payload))


def test_single_host_interface_staging_deduplicates_shared_instance(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, request = _request(tmp_path, single_host=True)
    staged: list[str] = []
    monkeypatch.setattr(
        sdg, "stage_interface", lambda instance, *_args: staged.append(instance),
    )
    sdg.stage_all_interfaces(path, request)
    assert staged == ["brev-a"]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p["resources"].update({"capacity_per_worker": 7}), "resources"),
        (lambda p: p["workers"][0].update({"gpu_ids": list(range(7))}), "gpu_ids"),
        (lambda p: p.update({"shell": "rm -rf /"}), "unexpected"),
        (lambda p: p["remote"].update({"runtime_sha256": "bad"}), "runtime_sha256"),
        (lambda p: p.update({"forward_env": ["PATH"]}), "credential"),
    ],
)
def test_request_rejects_widening_arbitrary_commands_and_unbound_runtime(
    tmp_path: pathlib.Path, mutate: Any, match: str
) -> None:
    _, payload = _request(tmp_path)
    mutate(payload)
    payload = _sign(payload)
    with pytest.raises(ValueError, match=match):
        sdg.validate_request(payload)


def test_request_digest_and_canonical_outputs_are_immutable(tmp_path: pathlib.Path) -> None:
    _, payload = _request(tmp_path)
    payload["timeouts"]["controller_s"] += 1
    with pytest.raises(ValueError, match="immutable"):
        sdg.validate_request(payload)
    payload = _sign(payload)
    payload["remote"]["expected_outputs"][0] += ".wrong"
    payload = _sign(payload)
    with pytest.raises(ValueError, match="canonical"):
        sdg.validate_request(payload)


def test_generation_node_count_and_instance_identities_are_strict(tmp_path: pathlib.Path) -> None:
    _, payload = _request(tmp_path)
    payload["generation_nodes"] = 3
    payload = _sign(payload)
    with pytest.raises(ValueError, match="exactly generation_nodes"):
        sdg.validate_request(payload)
    _, payload = _request(tmp_path)
    payload["workers"][1]["instance"] = payload["workers"][0]["instance"]
    payload = _sign(payload)
    with pytest.raises(ValueError, match="distinct"):
        sdg.validate_request(payload)


def test_endpoint_pool_has_one_entry_per_instance_gpu_slot(tmp_path: pathlib.Path) -> None:
    _, request = _request(tmp_path)
    pool = sdg._pool_candidate(request, _worker_evidence(request))
    assert pool["required_capacity"] == 16 and len(pool["endpoints"]) == 16
    assert pool["auth_env"] == "IMAGE_EDIT_API_KEY"
    assert all(item["capacity"] == 1 and set(item) == {"id", "url", "capacity", "gpu_identity", "owner"} for item in pool["endpoints"])
    assert {
        item["gpu_identity"]
        for item in pool["endpoints"]
    } == {
        f"{worker['instance']}/gpu:{gpu_id}"
        for worker in request["workers"] for gpu_id in range(8)
    }


def test_endpoint_pool_is_consumed_by_shared_validator_and_brev_loader(tmp_path: pathlib.Path) -> None:
    _, request = _request(tmp_path)
    pool = sdg._pool_candidate(request, _worker_evidence(request))
    assert shared_sdg.validate_image_edit_endpoint_pool(pool) == pool
    path = tmp_path / "endpoint_pool.json"
    path.write_text(json.dumps(pool))
    entries = shared_stage._runtime_image_edit_endpoint_pool(
        {"models": {"image_edit": pool["model"]}}, path, "brev"
    )
    assert len(entries) == 16 and all(item["capacity"] == 1 for item in entries)


def test_worker_start_delegates_one_eight_slot_pool_to_shared_manager(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = _request(tmp_path)
    monkeypatch.setenv("IMAGE_EDIT_API_KEY", "ephemeral-secret")
    commands: list[list[str]] = []
    def run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        output = pathlib.Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        if argv[2] == "plan":
            output.write_text(json.dumps({
                "commands": {
                    f"slot-{gpu_id}": [
                        "docker", "run", "--gpus", f'"device={gpu_id}"',
                        "-p", f"0.0.0.0:{port}:{port}", "image",
                    ]
                    for gpu_id, port in zip(request["workers"][0]["gpu_ids"], request["workers"][0]["ports"])
                }
            }))
        else:
            worker_pool = sdg._pool_candidate(request, _worker_evidence(request))
            worker_pool["required_capacity"] = 8
            worker_pool["endpoints"] = worker_pool["endpoints"][:8]
            output.write_text(json.dumps({"image_edit_endpoint_pool": worker_pool}))
        return subprocess.CompletedProcess(argv, 0, "{}", "")
    monkeypatch.setattr(sdg.subprocess, "run", run)
    manifest = sdg._worker_start(request, "job-01", 0)
    assert len(commands) == 2 and len(manifest["endpoints"]) == 8
    command = commands[1]
    assert command[command.index("--roles") + 1] == "image_edit"
    assert command[command.index("--platform") + 1] == "brev"
    assert command[command.index("--service-host") + 1] == "10.0.0.10"
    assert command[command.index("--gpu-identity-prefix") + 1] == "brev-worker-0"
    assert "ephemeral-secret" not in command and "docker" not in command
    parsed = shared_manager._parser().parse_args(command[2:])
    assert parsed.action == "start" and parsed.roles == "image_edit"
    assert parsed.gpu_identity_prefix == "brev-worker-0"


def test_worker_plan_rejects_loopback_binding_before_start(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = _request(tmp_path)
    monkeypatch.setenv("IMAGE_EDIT_API_KEY", "ephemeral-secret")
    calls = []
    def run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        output = pathlib.Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "commands": {
                f"slot-{gpu}": ["docker", "run", "--gpus", f'"device={gpu}"', "-p", f"127.0.0.1:{port}:{port}"]
                for gpu, port in zip(request["workers"][0]["gpu_ids"], request["workers"][0]["ports"])
            }
        }))
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(sdg.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="directly reachable"):
        sdg._worker_start(request, "job-01", 0)
    assert len(calls) == 1 and calls[0][2] == "plan"


def test_resume_recreation_is_explicit_and_never_used_for_fresh_submit(tmp_path: pathlib.Path) -> None:
    _, request = _request(tmp_path)
    fresh = sdg.build_worker_helper_command(
        request, 0, "start", tmp_path / "fresh.json"
    )
    resumed = sdg.build_worker_helper_command(
        request, 0, "start", tmp_path / "resume.json", recreate_owned=True
    )
    assert "--recreate-owned" not in fresh and "--recreate-owned" in resumed
    assert request["action_id"][:24] in resumed[resumed.index("--run-id") + 1]
    private = sdg._worker_command(
        request, "_worker_start", "job-01", 0, recreate_owned=True
    )
    assert "--recreate-owned" in private
    parsed = shared_manager._parser().parse_args(resumed[2:])
    assert parsed.recreate_owned is True and parsed.roles == "image_edit"


def test_job_record_binds_action_platform_and_job_id(tmp_path: pathlib.Path) -> None:
    _, request = _request(tmp_path)
    record = _record(tmp_path, request)
    assert sdg.validate_job_record(record, request, "job-01")["action"] == "action-01"
    record.write_text(json.dumps({"id": "job-01", "platform": "slurm", "action": "action-01"}))
    with pytest.raises(ValueError, match="platform"):
        sdg.validate_job_record(record, request, "job-01")


def test_submit_reconciles_and_does_not_duplicate_active_controller(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, request = _request(tmp_path)
    record = _record(tmp_path, request)
    calls = []
    monkeypatch.setattr(sdg, "probe_controller_runtime", lambda *_: [{"status": "PASS"}])
    monkeypatch.setattr(sdg, "stage_all_interfaces", lambda *args: None)
    monkeypatch.setattr(sdg, "remote_reconcile", lambda *args: calls.append(args) or {"state": "ACTIVE", "pid": 4})
    monkeypatch.setattr(sdg, "_remote_json", lambda *args, **kwargs: pytest.fail("duplicate start"))
    args = argparse.Namespace(request=path, instance="brev-a", job_id="job-01", job_record=record, resume=False)
    assert sdg.submit(args) == 0 and len(calls) == 1


def test_controller_runtime_probe_uses_only_signed_virtualenv(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = _request(tmp_path, single_host=True)
    calls = []

    def remote(instance: str, command: str, **_kwargs: Any):
        calls.append((instance, command))
        return subprocess.CompletedProcess([], 0, '{"status":"PASS","prefix":"/run/.venv"}\n', "")

    monkeypatch.setattr(sdg, "run_remote", remote)
    evidence = sdg.probe_controller_runtime(request)
    assert evidence == [{
        "instance": "brev-a", "status": "PASS",
        "runtime": request["remote"]["controller_python"],
    }]
    assert request["remote"]["controller_python"] in calls[0][1]
    assert "/usr/bin/python3" not in calls[0][1]


def test_controller_runtime_probe_stops_on_missing_dependency(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = _request(tmp_path, single_host=True)
    monkeypatch.setattr(
        sdg, "run_remote",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 1, "", "ModuleNotFoundError: No module named pandas"
        ),
    )
    with pytest.raises(RuntimeError, match="provision the signed workspace virtualenv"):
        sdg.probe_controller_runtime(request)


def test_submit_passes_secret_only_in_transport_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, request = _request(tmp_path)
    record = _record(tmp_path, request)
    captured: dict[str, Any] = {}
    worker_captured: dict[str, Any] = {}
    monkeypatch.setattr(sdg, "probe_controller_runtime", lambda *_: [{"status": "PASS"}])
    monkeypatch.setenv("HF_TOKEN", "secret-value")
    monkeypatch.setattr(sdg, "stage_all_interfaces", lambda *args: None)
    monkeypatch.setattr(sdg, "remote_reconcile", lambda *args: {"state": "NONE"})
    monkeypatch.setattr(
        sdg, "_fanout_workers",
        lambda *args, **kwargs: worker_captured.update(kwargs) or _worker_evidence(request),
    )
    monkeypatch.setattr(sdg, "_stage_json_replace", lambda *args: None)
    def remote(_instance: str, command: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(command=command, kwargs=kwargs)
        return {"state": "STARTED", "pid": 5}
    monkeypatch.setattr(sdg, "_remote_json", remote)
    args = argparse.Namespace(request=path, instance="brev-a", job_id="job-01", job_record=record, resume=False)
    assert sdg.submit(args) == 0
    assert "secret-value" not in captured["command"]
    controller_environment = captured["kwargs"]["environment"]
    assert controller_environment["HF_TOKEN"] == "secret-value"
    assert controller_environment["IMAGE_EDIT_API_KEY"] == controller_environment["VLLM_API_KEY"]
    assert worker_captured["environment"]["IMAGE_EDIT_API_KEY"] == worker_captured["environment"]["VLLM_API_KEY"]
    ephemeral_key = worker_captured["environment"]["IMAGE_EDIT_API_KEY"]
    assert controller_environment["IMAGE_EDIT_API_KEY"] == ephemeral_key
    assert ephemeral_key not in captured["command"]
    assert ephemeral_key not in json.dumps(request)
    assert ephemeral_key not in capsys.readouterr().out


def test_interface_staging_uses_digest_checked_temporary_promotion(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local.json"
    local.write_text("{}\n")
    commands: list[str] = []
    monkeypatch.setattr(
        sdg, "run_remote",
        lambda _instance, command, **_kwargs: commands.append(command)
        or subprocess.CompletedProcess([], 0, "", ""),
    )
    copied: list[list[str]] = []
    monkeypatch.setattr(
        sdg.subprocess, "run",
        lambda argv, **_kwargs: copied.append(argv)
        or subprocess.CompletedProcess(argv, 0, "", ""),
    )
    digest = sdg._stage_file("brev-a", local, pathlib.Path("/remote/state/request.json"))
    assert digest == sdg._file_sha256(local)
    assert copied[0][0] == "scp" and ".request.json." in copied[0][-1]
    assert digest in commands[-1] and "sha256sum" in commands[-1]


def test_controller_delegates_to_shared_helpers_and_runtime(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = _request(tmp_path)
    state = sdg._state_paths(request)["root"]
    commands: list[list[str]] = []
    controller = sdg.Controller(request, "job-01")
    monkeypatch.setattr(controller, "_validate_staged_runtime", lambda: None)
    monkeypatch.setattr(controller, "_validate_endpoint_manifest", lambda: None)
    monkeypatch.setattr(controller, "_validate_endpoint_status", lambda: None)
    monkeypatch.setattr(controller, "_validate_and_probe_pool", lambda: {})
    monkeypatch.setattr(controller, "_validate_aux_plan", lambda: None)
    monkeypatch.setattr(controller, "_outputs_complete", lambda: True)
    def run(argv: list[str], _log: pathlib.Path) -> None:
        commands.append(argv)
    monkeypatch.setattr(controller, "_run", run)
    monkeypatch.setattr(controller, "cleanup", lambda: None)
    assert controller.run() == 0
    assert commands[0][2] == "prepare"
    for flag, key in (
        ("--mined-pairs", "mined_pairs"), ("--gaps-parquet", "gaps_parquet"),
        ("--attribute-vocab", "attribute_vocab"), ("--dataset-root", "dataset_root"),
        ("--eval-list", "eval_list"), ("--eval-pairs", "eval_pairs"),
    ):
        assert commands[0][commands[0].index(flag) + 1] == request["remote"][key]
    actions = [command[2] for command in commands[1:4]]
    assert actions == ["plan", "start", "status"]
    coordinator_args = shared_manager._parser().parse_args(commands[2][2:])
    assert coordinator_args.roles == "vlm,llm" and coordinator_args.service_host == "127.0.0.1"
    runtime = commands[4]
    assert runtime[2] == "execute"
    assert runtime[runtime.index("--execution-platform") + 1] == "brev"
    assert "docker" not in runtime and "augment" not in runtime
    progress = json.loads((state / "progress.json").read_text())
    assert progress["runtime_complete"] is True


def test_single_host_auxiliary_plan_preserves_signed_gpu_ids_and_loopback(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request = _request(tmp_path, single_host=True)
    controller = sdg.Controller(request, "job-01")
    controller.paths["root"].mkdir(parents=True)
    monkeypatch.setenv("HF_TOKEN", "must-not-be-in-command")
    commands = {}
    for role, gpu_id in (("vlm", 4), ("llm", 5)):
        port = request["models"][role]["port"]
        commands[role] = [
            "docker", "run", "--gpus", f'"device={gpu_id}"',
            "-p", f"127.0.0.1:{port}:{port}", "image",
        ]
    sdg._atomic_json(controller.paths["endpoint_plan"], {"commands": commands})
    controller._validate_aux_plan()
    commands["vlm"][commands["vlm"].index("--gpus") + 1] = '"device=all"'
    sdg._atomic_json(controller.paths["endpoint_plan"], {"commands": commands})
    with pytest.raises(RuntimeError, match="signed topology"):
        controller._validate_aux_plan()


def test_single_host_worker_plan_and_pool_use_only_image_edit_gpus_zero_to_three(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request = _request(tmp_path, single_host=True)
    monkeypatch.setenv("IMAGE_EDIT_API_KEY", "ephemeral-secret")
    def run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        output = pathlib.Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        if argv[2] == "plan":
            output.write_text(json.dumps({"commands": {
                f"slot-{gpu}": [
                    "docker", "run", "--gpus", f'"device={gpu}"',
                    "-p", f"0.0.0.0:{port}:{port}", "image",
                ]
                for gpu, port in zip(
                    request["workers"][0]["gpu_ids"], request["workers"][0]["ports"]
                )
            }}))
        else:
            pool = sdg._pool_candidate(request, _worker_evidence(request))
            output.write_text(json.dumps({"image_edit_endpoint_pool": pool}))
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(sdg.subprocess, "run", run)
    result = sdg._worker_start(request, "job-01", 0)
    assert result["capacity"] == 4
    assert {item["gpu_identity"] for item in result["endpoints"]} == {
        f"brev-a/gpu:{gpu}" for gpu in range(4)
    }


def test_single_host_resume_reconciles_exact_four_owned_slots(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request = _request(tmp_path, single_host=True)
    monkeypatch.setenv("IMAGE_EDIT_API_KEY", "ephemeral-secret")
    actions: list[str] = []
    def run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        action = argv[2]
        actions.append(action)
        output = pathlib.Path(argv[argv.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        if action == "status":
            output.write_text(json.dumps({
                "request_sha256": request["request_sha256"],
                "containers": {f"slot-{gpu}": {"owned": True} for gpu in range(4)},
            }))
        elif action == "plan":
            output.write_text(json.dumps({"commands": {
                f"slot-{gpu}": [
                    "docker", "run", "--gpus", f'"device={gpu}"',
                    "-p", f"0.0.0.0:{port}:{port}", "image",
                ]
                for gpu, port in zip(range(4), request["workers"][0]["ports"])
            }}))
        else:
            output.write_text(json.dumps({
                "image_edit_endpoint_pool": sdg._pool_candidate(
                    request, _worker_evidence(request),
                )
            }))
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(sdg.subprocess, "run", run)
    result = sdg._worker_start(request, "job-01", 0, recreate_owned=True)
    assert actions == ["status", "plan", "start"]
    assert result["recovery"] == "recreated_exact_owned_for_ephemeral_auth"


def test_single_host_readiness_smokes_all_four_loopback_endpoints_without_relay(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request = _request(tmp_path, single_host=True)
    controller = sdg.Controller(request, "job-01")
    controller.paths["root"].mkdir(parents=True)
    sdg._atomic_json(
        controller.paths["pool_candidate"],
        sdg._pool_candidate(request, _worker_evidence(request)),
    )
    smoke = pathlib.Path(request["remote"]["smoke_image"])
    smoke.parent.mkdir(parents=True)
    smoke.write_bytes(b"jpeg")
    monkeypatch.setenv("IMAGE_EDIT_API_KEY", "ephemeral-secret")
    calls: list[str] = []
    class Response:
        def __init__(self, payload: dict[str, Any]): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *_args: Any): return False
        def read(self) -> bytes: return json.dumps(self.payload).encode()
    def urlopen(req: Any, timeout: int) -> Response:
        calls.append(req.full_url)
        if req.full_url.endswith("/models"):
            return Response({"data": [{"id": request["models"]["image_edit"]["id"]}]})
        return Response({"data": [{"b64_json": "ok"}]})
    monkeypatch.setattr(sdg.urllib.request, "urlopen", urlopen)
    pool = controller._validate_and_probe_pool()
    assert len(pool["endpoints"]) == 4 and len(calls) == 8
    assert all(url.startswith("http://127.0.0.1:") for url in calls)
    readiness = json.loads(controller.paths["pool_readiness"].read_text())
    assert readiness["ready_capacity"] == 4
    assert readiness["control_host_relay"] is False


def test_coordinator_commits_pool_only_after_direct_models_and_inference_smoke(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = _request(tmp_path)
    controller = sdg.Controller(request, "job-01")
    controller.paths["root"].mkdir(parents=True)
    sdg._atomic_json(controller.paths["pool_candidate"], sdg._pool_candidate(request, _worker_evidence(request)))
    smoke = pathlib.Path(request["remote"]["smoke_image"])
    smoke.parent.mkdir(parents=True)
    smoke.write_bytes(b"jpeg")
    monkeypatch.setenv("IMAGE_EDIT_API_KEY", "ephemeral-secret")
    calls: list[str] = []
    class Response:
        def __init__(self, payload: dict[str, Any]): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *_args: Any): return False
        def read(self) -> bytes: return json.dumps(self.payload).encode()
    def urlopen(req: Any, timeout: int) -> Response:
        calls.append(req.full_url)
        assert req.get_header("Authorization") == "Bearer ephemeral-secret"
        if req.full_url.endswith("/models"):
            return Response({"data": [{"id": request["models"]["image_edit"]["id"]}]})
        return Response({"data": [{"b64_json": "ok"}]})
    monkeypatch.setattr(sdg.urllib.request, "urlopen", urlopen)
    pool = controller._validate_and_probe_pool()
    assert len(pool["endpoints"]) == 16 and len(calls) == 32
    assert pathlib.Path(request["remote"]["endpoint_pool_path"]).is_file()
    readiness = json.loads(controller.paths["pool_readiness"].read_text())
    assert readiness["ready_capacity"] == 16 and readiness["control_host_relay"] is False


def test_coordinator_worker_network_failure_is_actionable_and_does_not_commit_pool(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = _request(tmp_path)
    controller = sdg.Controller(request, "job-01")
    controller.paths["root"].mkdir(parents=True)
    sdg._atomic_json(controller.paths["pool_candidate"], sdg._pool_candidate(request, _worker_evidence(request)))
    smoke = pathlib.Path(request["remote"]["smoke_image"])
    smoke.parent.mkdir(parents=True)
    smoke.write_bytes(b"jpeg")
    monkeypatch.setenv("IMAGE_EDIT_API_KEY", "ephemeral-secret")
    monkeypatch.setattr(
        sdg.urllib.request, "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sdg.urllib.error.URLError("blocked")),
    )
    with pytest.raises(RuntimeError, match="private Brev networking/firewall"):
        controller._validate_and_probe_pool()
    assert not pathlib.Path(request["remote"]["endpoint_pool_path"]).exists()


def test_staged_config_and_runtime_digests_are_checked(tmp_path: pathlib.Path) -> None:
    _, request = _request(tmp_path)
    config = pathlib.Path(request["remote"]["config_path"])
    config.parent.mkdir(parents=True)
    config.write_text("schema_version: '1'\n")
    runtime = pathlib.Path(request["remote"]["runtime_root"])
    runtime.mkdir()
    (runtime / "manage_sdg_endpoints.py").write_text("# helper\n")
    (runtime / "run_sdg_stage.py").write_text("# runtime\n")
    request["remote"]["config_sha256"] = sdg._file_sha256(config)
    request["remote"]["runtime_sha256"] = sdg._tree_sha256(runtime)
    request = _sign(request)
    sdg.Controller(request, "job-01")._validate_staged_runtime()
    (runtime / "run_sdg_stage.py").write_text("# changed\n")
    with pytest.raises(RuntimeError, match="runtime digest"):
        sdg.Controller(request, "job-01")._validate_staged_runtime()


def test_cleanup_uses_shared_owned_only_stop_helper(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, request = _request(tmp_path)
    controller = sdg.Controller(request, "job-01")
    captured: list[str] = []
    def run(argv: list[str], **_kwargs: Any) -> Any:
        captured.extend(argv)
        return type("Result", (), {"returncode": 0, "stdout": "{}", "stderr": ""})()
    monkeypatch.setattr(sdg.subprocess, "run", run)
    controller.paths["root"].mkdir(parents=True)
    controller.cleanup()
    assert captured[2] == "stop" and "manage_sdg_endpoints.py" in captured[1]


def test_endpoint_status_requires_exact_owned_running_roles(tmp_path: pathlib.Path) -> None:
    _, request = _request(tmp_path)
    controller = sdg.Controller(request, "job-01")
    controller.paths["root"].mkdir(parents=True)
    sdg._atomic_json(controller.paths["endpoint_manifest"], {
        "status": "success", "ownership": "managed",
    })
    sdg._atomic_json(controller.paths["endpoint_status"], {
        "ownership": "managed",
        "containers": {
            role: {"owned": True, "running": role != "llm"}
            for role in ("vlm", "llm")
        },
    })
    with pytest.raises(RuntimeError, match="stopped"):
        controller._validate_endpoint_status()


def test_terminal_sync_binds_remote_and_local_canonical_outputs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = _request(tmp_path)
    copied: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sdg, "_sync_file",
        lambda _instance, remote, local: copied.append((str(remote), str(local))) or "a" * 64,
    )
    report = sdg.sync_outputs("brev-a", request)
    assert report["status"] == "synced" and len(copied) == 4
    assert copied == list(zip(
        request["remote"]["expected_outputs"], request["local"]["expected_outputs"]
    ))


def test_controller_cleanup_failure_controls_terminal_exit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = _request(tmp_path)
    controller = sdg.Controller(request, "job-01")
    monkeypatch.setattr(controller, "_validate_staged_runtime", lambda: (_ for _ in ()).throw(RuntimeError("startup failed")))
    monkeypatch.setattr(controller, "cleanup", lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed")))
    controller.paths["endpoint_manifest"].parent.mkdir(parents=True)
    controller.paths["endpoint_manifest"].write_text("{}")
    assert controller.run() == 1
    status = json.loads(controller.paths["status"].read_text())
    assert status["status"] == "ERROR" and "cleanup failed" in status["message"]
    assert status["job_id"] == "job-01" and status["action_id"] == request["action_id"]
    assert status["request_sha256"] == request["request_sha256"]
    assert status["started_at"] == request["started_at"] and status["started_ns"] == request["started_ns"]


def test_resume_does_not_repeat_committed_shared_runtime(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = _request(tmp_path)
    controller = sdg.Controller(request, "job-01")
    controller.paths["root"].mkdir(parents=True)
    sdg._atomic_json(controller.paths["progress"], {
        **controller.identity, "controller_attempt": 1, "runtime_complete": True,
    })
    monkeypatch.setattr(controller, "_validate_staged_runtime", lambda: None)
    monkeypatch.setattr(controller, "_outputs_complete", lambda: True)
    monkeypatch.setattr(controller, "_run", lambda *_: pytest.fail("shared runtime repeated"))
    monkeypatch.setattr(controller, "cleanup", lambda: None)
    assert controller.run() == 0


def test_redaction_cancel_confirmation_and_status_mapping(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert "secret-value" not in sdg._redact("token=secret-value", ["secret-value"])
    with pytest.raises(ValueError, match="confirm"):
        sdg.cancel(argparse.Namespace(confirm=False))
    path, request = _request(tmp_path)
    record = _record(tmp_path, request)
    monkeypatch.setattr(sdg, "remote_reconcile", lambda *args: {"state": "RESUMABLE"})
    monkeypatch.setattr(sdg, "_fanout_workers", lambda *args, **kwargs: _worker_evidence(request))
    args = argparse.Namespace(request=path, instance="brev-a", job_id="job-01", job_record=record)
    assert sdg.status(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"
