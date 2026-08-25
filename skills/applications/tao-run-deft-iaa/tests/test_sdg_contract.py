# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from iaa_deft.sdg import (  # noqa: E402
    QUERY_LEVELS,
    accepted_augmentations,
    bind_resumable_endpoint_pool,
    build_component_command,
    build_endpoint_command,
    container_name,
    normalize_generated_pairs,
    port_available,
    readiness_probe,
    residual_attribute_assignments,
    validate_config,
    validate_image_edit_endpoint_pool,
    validate_normalized_dataset,
    validate_gpu_inventory,
    wait_until_ready,
)
import manage_sdg_endpoints as endpoint_manager  # noqa: E402
import run_sdg_stage as stage_runner  # noqa: E402
from runtime_binding import python_tree_sha256  # noqa: E402


def config(ownership: str = "managed") -> dict:
    payload = yaml.safe_load((ROOT / "specs" / "sdg_config.yaml").read_text())
    payload["endpoints"]["ownership"] = ownership
    if ownership == "managed":
        payload["endpoints"]["reuse_requested"] = False
        payload["endpoints"]["gpu_ids"] = {"image_edit": [2], "vlm": [4], "llm": [6]}
        payload["endpoints"]["external_urls"] = {role: "" for role in ("image_edit", "vlm", "llm")}
    else:
        payload["endpoints"]["reuse_requested"] = True
        payload["endpoints"]["gpu_ids"] = {role: [] for role in ("image_edit", "vlm", "llm")}
        payload["endpoints"]["external_urls"] = {
            "image_edit": "http://127.0.0.1:18002/v1",
            "vlm": "http://127.0.0.1:18000/v1",
            "llm": "http://127.0.0.1:18001/v1",
        }
    return validate_config(payload)


def vocab(path: pathlib.Path) -> None:
    attributes = [
        "top outer color", "top outer type", "bottom color", "bottom type",
        "shoe color", "shoe type", "viewpoint",
    ]
    values = {
        "top outer color": {"red": 0}, "top outer type": {"jacket": 0},
        "bottom color": {"blue": 0}, "bottom type": {"pants": 0},
        "shoe color": {"black": 0}, "shoe type": {"sneakers": 0},
        "viewpoint": {"front": 0},
    }
    path.write_text(json.dumps({
        "attributes": attributes, "value_to_id": values,
        "id_to_value": {attribute: [next(iter(mapping))] for attribute, mapping in values.items()},
    }))


def metadata(path: pathlib.Path, passed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "attribute_verification": {
            "passed": passed,
            "details": {"results": [
                {"variable": key, "value": value, "passed": passed}
                for key, value in {
                    "top_outer_color": "red", "top_outer_type": "jacket",
                    "bottom_color": "blue", "bottom_type": "pants",
                    "shoe_color": "black", "shoe_type": "sneakers",
                }.items()
            ]},
        },
        "selections": {
            "top_outer_color": "red", "top_outer_type": "jacket",
            "bottom_color": "blue", "bottom_type": "pants",
            "shoe_color": "black", "shoe_type": "sneakers",
        },
    }))


class EndpointContractTests(unittest.TestCase):
    def test_unstarted_pool_rebind_requires_explicit_repair(self):
        cfg = config("external")
        old = [{
            "id": "old-gpu", "url": "http://old.internal:19000/v1", "capacity": 1,
            "gpu_identity": "old/gpu-0", "owner": {"native_id": "old", "name": "old"},
        }]
        new = [{
            "id": "new-gpu", "url": "http://new.internal:19000/v1", "capacity": 1,
            "gpu_identity": "new/gpu-0", "owner": {"native_id": "new", "name": "new"},
        }]
        binding = {
            "schema_version": "1", "platform": "slurm",
            "model": {
                "id": cfg["models"]["image_edit"]["id"],
                "revision": cfg["models"]["image_edit"]["revision"],
            },
            "required_capacity": 1, "auth_env": None, "endpoints": new,
            "created_at": "2026-08-24T00:00:00Z", "request_sha256": "a" * 64,
        }
        progress = {
            "schema_version": "1", "preprocessed": False, "augmentation": {},
            "split": False, "labeling": {}, "command_attempts": {"preprocess:batch:1": 1},
            "endpoint_attempts": {}, "image_edit_endpoints": old,
            "image_edit_endpoint_pool": {}, "image_edit_endpoint_history": [],
        }
        selected = [{"source_key": "sample"}]
        with self.assertRaisesRegex(ValueError, "explicit unfinished resume"):
            bind_resumable_endpoint_pool(progress, selected, new, binding)
        bind_resumable_endpoint_pool(
            progress, selected, new, binding, allow_unstarted_rebind=True,
        )
        self.assertEqual(progress["image_edit_endpoints"], new)
        self.assertEqual(progress["image_edit_endpoint_history"][0]["endpoints"], old)

    def test_image_edit_port_range_covers_every_gpu_service(self):
        cfg = config()
        cfg["models"]["image_edit"]["port"] = 65535
        cfg["generation"]["gpus_per_generation_node"] = 2
        with self.assertRaisesRegex(ValueError, "base port"):
            validate_config(cfg)

    def test_image_edit_port_range_cannot_overlap_auxiliary_roles(self):
        cfg = config()
        cfg["models"]["image_edit"]["port"] = 8000
        cfg["models"]["vlm"]["port"] = 8001
        cfg["models"]["llm"]["port"] = 8010
        cfg["generation"]["gpus_per_generation_node"] = 4
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            validate_config(cfg)

    @staticmethod
    def _owned_state(role: str, *, exit_code: int = 0, owned: bool = True) -> dict:
        labels = {
            "com.nvidia.tao.workflow": "tao-run-deft-iaa",
            "com.nvidia.tao.run": "run_42" if owned else "other_run",
            "com.nvidia.tao.role": role,
        }
        return {
            "Config": {"Labels": labels},
            "State": {
                "Status": "exited", "Running": False, "ExitCode": exit_code,
                "OOMKilled": False, "Error": "", "FinishedAt": "2026-08-19T01:02:03Z",
            },
        }

    @staticmethod
    def _success_manifest(cfg: dict, *, restart_count: int = 0) -> dict:
        return {
            "status": "success", "attempt": 2, "restart_count": restart_count,
            "ownership": "managed",
            "containers": {
                role: {
                    "name": container_name("run_42", role), "owned": True,
                    "model": cfg["models"][role]["id"],
                    "probe": {"models_ok": True, "inference_ok": role != "image_edit"},
                } for role in ("image_edit", "vlm", "llm")
            },
        }

    @classmethod
    def _configured_container(
        cls, cfg: dict, role: str, *, exit_code: int = 0, oom: bool = False,
        owned: bool = True, model_override: str | None = None,
    ) -> dict:
        model = cfg["models"][role]
        model_id = model_override or model["id"]
        command = [
            "--model", model_id, "--host", "0.0.0.0", "--port", str(model["port"]),
            "--revision", model["revision"], "--served-model-name", model_id,
        ]
        if role == "image_edit":
            command = [
                "vllm", "serve", model_id, "--omni", "--host", "0.0.0.0",
                "--port", str(model["port"]), "--revision", model["revision"],
                "--served-model-name", model_id,
            ]
        record = cls._owned_state(role, exit_code=exit_code, owned=owned)
        record["State"]["OOMKilled"] = oom
        record["Config"]["Image"] = cfg["images"][
            "image_edit_serving" if role == "image_edit" else "text_serving"
        ]
        record["Config"]["Cmd"] = command
        record["HostConfig"] = {"DeviceRequests": [{
            "DeviceIDs": [str(item) for item in cfg["endpoints"]["gpu_ids"][role]],
        }]}
        return record

    @staticmethod
    def _write_execution_receipt(root: pathlib.Path, cfg: dict, manifest: pathlib.Path) -> pathlib.Path:
        execution = root / "sdg_execution_manifest.json"
        execution.write_text(json.dumps({"selected_sources": 10, "accepted_sources": 9}))
        status = root / "sdg-normalize.host.status.json"
        status.write_text(json.dumps({"status": "ok", "exit_code": 0}))
        receipt = root / "deft_state.json"
        receipt.write_text(json.dumps({
            "workflow": "tao-run-deft-iaa",
            "config": {"sdg": {
                "endpoint_mode": "managed", "models": cfg["models"],
                "gpu_ids": cfg["endpoints"]["gpu_ids"], "images": cfg["images"],
            }},
            "iterations": {"iter1": {
                "endpoint_manifest": str(manifest.resolve()),
                "sdg_execution_manifest": str(execution.resolve()),
                "sdg_status": str(status.resolve()),
                "stage_completed": "sdg", "status": "in_progress",
            }},
        }))
        return receipt

    def test_external_endpoints_require_explicit_reuse_evidence(self):
        cfg = config("external")
        cfg["endpoints"]["reuse_requested"] = False
        with self.assertRaisesRegex(ValueError, "explicit user-requested reuse"):
            validate_config(cfg)

    def test_component_images_are_customer_pullable_and_immutable(self):
        cfg = config("external")
        for role in ("augmentation", "auto_labeling"):
            image = cfg["images"][role]
            self.assertTrue(image.startswith("nvcr.io/nvstaging/tao/"))
            self.assertRegex(image, r":[^@]+@sha256:[0-9a-f]{64}$")

    def test_endpoint_ownership_is_exact(self):
        inspect = {"Config": {"Labels": {
            "com.nvidia.tao.workflow": "tao-run-deft-iaa",
            "com.nvidia.tao.run": "run_42",
            "com.nvidia.tao.role": "vlm",
        }}}
        self.assertTrue(endpoint_manager._owned(inspect, "run_42", "vlm"))
        self.assertFalse(endpoint_manager._owned(inspect, "run_43", "vlm"))
        self.assertFalse(endpoint_manager._owned(inspect, "run_42", "llm"))

    def test_commands_preserve_explicit_gpu_ids_and_pins(self):
        cfg = config()
        cfg["endpoints"]["gpu_ids"]["image_edit"] = [0, 1, 2, 3]
        for role, expected in (("image_edit", '"device=0,1,2,3"'), ("vlm", '"device=4"'), ("llm", '"device=6"')):
            argv = build_endpoint_command(cfg, role, "run_42", pathlib.Path("/tmp/hf"))
            self.assertIn(expected, argv)
            self.assertNotIn("all", argv)
            self.assertNotIn(":latest", " ".join(argv))
            self.assertIn(cfg["models"][role]["revision"], argv)
            if role == "image_edit":
                self.assertEqual(argv[argv.index("--shm-size") + 1], "16g")
            else:
                self.assertNotIn("--shm-size", argv)
        self.assertEqual(container_name("run_42", "vlm"), "tao-deft-iaa-run-42-vlm")

    def test_endpoint_token_forwarding_is_bound_by_immutable_config(self):
        cfg = config()
        with mock.patch.dict(os.environ, {"HF_TOKEN": "present-but-not-approved"}):
            argv = build_endpoint_command(cfg, "vlm", "run_42", pathlib.Path("/tmp/hf"))
        self.assertNotIn("HF_TOKEN", argv)
        cfg["endpoints"]["forward_hf_token"] = True
        with mock.patch.dict(os.environ, {"HF_TOKEN": "approved-token"}):
            argv = build_endpoint_command(cfg, "vlm", "run_42", pathlib.Path("/tmp/hf"))
        self.assertIn("HF_TOKEN", argv)

    def test_image_edit_workers_are_single_gpu_tp1_with_unique_ports_and_pool(self):
        cfg = config()
        cfg["endpoints"]["gpu_ids"]["image_edit"] = [2, 3, 7]
        cfg["generation"]["gpus_per_generation_node"] = 3
        instances = endpoint_manager._instances(cfg, "run_42", ("image_edit",))
        containers = {}
        for ordinal, instance in enumerate(instances):
            argv = build_endpoint_command(
                cfg, "image_edit", "run_42", pathlib.Path("/tmp/hf"),
                image_edit_gpu_id=instance["gpu_id"], image_edit_ordinal=ordinal,
            )
            self.assertEqual(argv[argv.index("--gpus") + 1], f'"device={instance["gpu_id"]}"')
            self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "1")
            self.assertIn(f"127.0.0.1:{8002 + ordinal}:{8002 + ordinal}", argv)
            self.assertEqual(argv[argv.index("--name") + 1], container_name("run_42", "image_edit", instance["gpu_id"]))
            containers[instance["key"]] = {"id": f"native-{ordinal}"}
        pool = endpoint_manager._endpoint_pool(
            cfg, instances, containers, platform="docker", service_host="127.0.0.1",
            request_sha256="a" * 64, gpu_identity_prefix="host/run_42",
        )
        validated = validate_image_edit_endpoint_pool(pool)
        self.assertEqual(validated["required_capacity"], 3)
        self.assertEqual([item["capacity"] for item in validated["endpoints"]], [1, 1, 1])
        self.assertEqual(
            [item["gpu_identity"] for item in validated["endpoints"]],
            ["host/run_42/gpu:2", "host/run_42/gpu:3", "host/run_42/gpu:7"],
        )

    def test_remote_image_worker_requires_auth_and_uses_reachable_publish_binding(self):
        cfg = config()
        with self.assertRaisesRegex(ValueError, "requires API authentication"):
            build_endpoint_command(
                cfg, "image_edit", "run_42", pathlib.Path("/tmp/hf"),
                image_edit_gpu_id=2, image_edit_ordinal=0, publish_host="0.0.0.0",
            )
        argv = build_endpoint_command(
            cfg, "image_edit", "run_42", pathlib.Path("/tmp/hf"),
            image_edit_gpu_id=2, image_edit_ordinal=0, publish_host="0.0.0.0",
            authenticated=True,
        )
        self.assertIn("0.0.0.0:8002:8002", argv)
        self.assertEqual(argv[argv.index("-e") + 1], "VLLM_API_KEY")
        self.assertNotIn("IMAGE_EDIT_API_KEY", " ".join(argv))

    def test_remote_auth_rotation_recreates_only_exact_owned_image_worker(self):
        cfg = config()
        instance = endpoint_manager._instances(cfg, "run_42", ("image_edit",))[0]
        existing = self._owned_state("image_edit")
        existing["Id"] = "old-native-id"
        existing["State"].update(Status="running", Running=True)
        existing["HostConfig"] = {"DeviceRequests": [{"DeviceIDs": ["2"]}]}
        calls = []

        def fake_run(argv, check=True):
            calls.append(argv)
            return mock.Mock(
                returncode=0,
                stdout="new-native-id\n" if argv[:2] == ["docker", "run"] else "",
                stderr="",
            )

        with (
            mock.patch.object(endpoint_manager, "preflight", return_value={"ownership": "managed"}),
            mock.patch.object(endpoint_manager, "_require_component_images", return_value={}),
            mock.patch.object(endpoint_manager, "_inspect", return_value=existing),
            mock.patch.object(endpoint_manager, "_run", side_effect=fake_run),
            mock.patch.object(endpoint_manager, "wait_until_ready", return_value={"models_ok": True}),
        ):
            report = endpoint_manager.start(
                cfg, "run_42", pathlib.Path("/tmp/hf"), ("image_edit",),
                platform="brev", service_host="worker.internal", request_sha256="a" * 64,
                gpu_identity_prefix="instance-1", recreate_owned=True,
            )
        self.assertIn(["docker", "stop", "--time", "30", instance["name"]], calls)
        self.assertIn(["docker", "rm", instance["name"]], calls)
        self.assertEqual(report["recoveries"][0]["disposition"], "removed_and_recreated_for_run_scoped_auth_rotation")
        self.assertNotIn("old-native-id", json.dumps(report))

        foreign = json.loads(json.dumps(existing))
        foreign["Config"]["Labels"]["com.nvidia.tao.run"] = "other"
        with (
            mock.patch.object(endpoint_manager, "preflight", return_value={}),
            mock.patch.object(endpoint_manager, "_require_component_images", return_value={}),
            mock.patch.object(endpoint_manager, "_inspect", return_value=foreign),
            self.assertRaisesRegex(ValueError, "refusing to recreate"),
        ):
            endpoint_manager.start(
                cfg, "run_42", pathlib.Path("/tmp/hf"), ("image_edit",),
                platform="brev", service_host="worker.internal", request_sha256="a" * 64,
                gpu_identity_prefix="instance-1", recreate_owned=True,
            )

    def test_main_auth_rotation_bypasses_successful_running_early_return(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            config_path = root / "sdg.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            output = root / "manifest.json"
            output.write_text(json.dumps({
                "schema_version": "1", "status": "success", "attempt": 1,
                "restart_count": 0, "ownership": "managed",
                "containers": {"image_edit_gpu_2": {
                    "name": container_name("run_42", "image_edit", 2), "owned": True,
                    "model": cfg["models"]["image_edit"]["id"],
                    "probe": {"models_ok": True, "inference_ok": False},
                }},
            }))
            replacement = {
                "ownership": "managed", "containers": json.loads(output.read_text())["containers"],
            }
            with (
                mock.patch.dict(os.environ, {
                    "VLLM_API_KEY": "test-secret", "IMAGE_EDIT_API_KEY": "test-secret",
                }),
                mock.patch.object(endpoint_manager, "_all_managed_roles_running", return_value=True),
                mock.patch.object(endpoint_manager, "start", return_value=replacement) as start,
            ):
                code = endpoint_manager.main([
                    "start", "--config", str(config_path), "--run-id", "run_42",
                    "--cache-dir", str(root / "cache"), "--output", str(output),
                    "--platform", "brev", "--roles", "image_edit",
                    "--service-host", "worker.internal", "--request-sha256", "a" * 64,
                    "--gpu-identity-prefix", "instance-1", "--recreate-owned",
                ])
            self.assertEqual(code, 0)
            self.assertTrue(start.call_args.kwargs["recreate_owned"])
            self.assertEqual(json.loads(output.read_text())["restart_count"], 0)
            completed = json.loads(output.read_text())
            completed["auth_rotation"] = {
                "request_sha256": "a" * 64, "recreated": [container_name("run_42", "image_edit", 2)],
                "status": "complete",
            }
            output.write_text(json.dumps(completed))
            with (
                mock.patch.dict(os.environ, {
                    "VLLM_API_KEY": "test-secret", "IMAGE_EDIT_API_KEY": "test-secret",
                }),
                mock.patch.object(endpoint_manager, "_all_managed_roles_running", return_value=True),
                mock.patch.object(endpoint_manager, "start") as repeated_start,
            ):
                repeated = endpoint_manager.main([
                    "start", "--config", str(config_path), "--run-id", "run_42",
                    "--cache-dir", str(root / "cache"), "--output", str(output),
                    "--platform", "brev", "--roles", "image_edit",
                    "--service-host", "worker.internal", "--request-sha256", "a" * 64,
                    "--gpu-identity-prefix", "instance-1", "--recreate-owned",
                ])
            self.assertEqual(repeated, 0)
            repeated_start.assert_not_called()

    def test_role_scoped_inventory_and_per_gpu_image_edit_vram(self):
        cfg = config()
        cfg["endpoints"]["gpu_ids"]["image_edit"] = [0, 1]
        image_inventory = [
            {"index": 0, "compute_capability": 8.0, "memory_free_mib": 39000},
            {"index": 1, "compute_capability": 8.0, "memory_free_mib": 39000},
        ]
        validate_gpu_inventory(cfg, image_inventory, selected_roles=("image_edit",))
        image_inventory[1]["memory_free_mib"] = 37999
        with self.assertRaisesRegex(ValueError, "GPU 1.*38000"):
            validate_gpu_inventory(cfg, image_inventory, selected_roles=("image_edit",))
        # Missing coordinator GPUs are irrelevant to an image-only worker.
        with self.assertRaisesRegex(ValueError, "vlm requests missing GPU 4"):
            validate_gpu_inventory(cfg, image_inventory, selected_roles=("vlm", "llm"))

    def test_preflight_role_selection_isolates_worker_and_coordinator_allocations(self):
        cfg = config()
        inventories = {
            ("image_edit",): [{"index": 2, "compute_capability": 8.0,
                               "memory_total_mib": 40000, "memory_free_mib": 40000}],
            ("vlm", "llm"): [
                {"index": 4, "compute_capability": 8.0,
                 "memory_total_mib": 60000, "memory_free_mib": 60000},
                {"index": 6, "compute_capability": 8.0,
                 "memory_total_mib": 40000, "memory_free_mib": 40000},
            ],
        }
        for roles, inventory in inventories.items():
            with self.subTest(roles=roles):
                with (
                    mock.patch.object(endpoint_manager.shutil, "which", return_value="/bin/tool"),
                    mock.patch.object(endpoint_manager, "_run", return_value=mock.Mock(
                        returncode=0, stdout=json.dumps({"nvidia": {}}), stderr="",
                    )),
                    mock.patch.object(endpoint_manager.shutil, "disk_usage", return_value=mock.Mock(
                        free=200 * 1024**3,
                    )),
                    mock.patch.object(endpoint_manager, "_inventory", return_value=inventory),
                    mock.patch.object(endpoint_manager, "_inspect", return_value=None),
                    mock.patch.object(endpoint_manager, "port_available", return_value=True),
                    mock.patch.object(endpoint_manager, "component_status", return_value={"components": {}}),
                ):
                    report = endpoint_manager.preflight(cfg, "run_42", pathlib.Path("/tmp/hf"), roles)
            expected = {item["key"] for item in endpoint_manager._instances(cfg, "run_42", roles)}
            self.assertEqual(set(report["commands"]), expected)

    def test_cache_capacity_reuses_only_exact_revision_files(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as temporary:
            cache = pathlib.Path(temporary)
            model = cfg["models"]["image_edit"]
            snapshot = (
                cache / "hub" / ("models--" + model["id"].replace("/", "--"))
                / "snapshots" / model["revision"]
            )
            blobs = cache / "hub" / "blobs"
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            blob = blobs / "weight"
            blob.write_bytes(b"weight-bytes")
            (snapshot / "model.safetensors").symlink_to(blob)
            wrong = snapshot.parent / "wrong-revision"
            wrong.mkdir()
            (wrong / "ignored.bin").write_bytes(b"ignored")

            reusable = endpoint_manager._cached_revision_bytes(
                cfg, cache, ("image_edit",)
            )

        self.assertEqual(reusable, len(b"weight-bytes"))

    @staticmethod
    def _write_exact_model_cache(cfg: dict, cache: pathlib.Path, roles=("image_edit", "vlm", "llm")):
        for role in roles:
            model = cfg["models"][role]
            snapshot = (
                cache / "hub" / ("models--" + model["id"].replace("/", "--"))
                / "snapshots" / model["revision"]
            )
            snapshot.mkdir(parents=True)
            (snapshot / "weights.bin").write_bytes((role + "-weights").encode())

    @classmethod
    def _running_instances(cls, cfg: dict, run_id: str = "run_42") -> dict[str, dict]:
        result = {}
        for instance in endpoint_manager._instances(cfg, run_id, ("image_edit", "vlm", "llm")):
            record = cls._configured_container(cfg, instance["role"])
            record["Id"] = "native-" + instance["key"]
            record["State"].update({
                "Status": "running", "Running": True, "ExitCode": 0,
                "Paused": False, "Restarting": False, "Dead": False,
            })
            if instance["role"] == "image_edit":
                record["Config"]["Cmd"][record["Config"]["Cmd"].index("--port") + 1] = str(instance["port"])
                record["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = [str(instance["gpu_id"])]
            record["HostConfig"]["PortBindings"] = {
                f"{instance['port']}/tcp": [{
                    "HostIp": "127.0.0.1", "HostPort": str(instance["port"]),
                }]
            }
            result[instance["name"]] = record
        return result

    def _low_space_preflight(self, cfg: dict, cache: pathlib.Path, inspections: dict):
        def inspect(name):
            return inspections.get(name)

        with (
            mock.patch.object(endpoint_manager.shutil, "which", return_value="/bin/tool"),
            mock.patch.object(endpoint_manager, "_run", return_value=mock.Mock(
                returncode=0, stdout=json.dumps({"nvidia": {}}), stderr="",
            )),
            mock.patch.object(endpoint_manager.shutil, "disk_usage", return_value=mock.Mock(
                free=1 * 1024**3,
            )),
            mock.patch.object(endpoint_manager, "_inspect", side_effect=inspect),
            mock.patch.object(endpoint_manager, "readiness_probe", side_effect=lambda _cfg, role, **_kw: {
                "role": role, "models_ok": True, "inference_ok": role != "image_edit",
            }),
            mock.patch.object(endpoint_manager, "component_status", return_value={"components": {}}),
        ):
            return endpoint_manager.preflight(cfg, "run_42", cache, platform="virtualenv")

    def test_healthy_exact_owned_reuse_skips_only_acquisition_capacity(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as raw:
            cache = pathlib.Path(raw) / "cache"
            self._write_exact_model_cache(cfg, cache)
            report = self._low_space_preflight(
                cfg, cache, self._running_instances(cfg),
            )
        self.assertEqual(report["disposition"], "reuse_no_acquisition")
        self.assertEqual(report["cache_capacity"]["acquisition_gate"], "not_applicable")
        self.assertEqual(set(report["cache_receipts"]), {"image_edit", "vlm", "llm"})

    def test_healthy_reuse_path_performs_no_endpoint_or_cache_mutation(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as raw:
            cache = pathlib.Path(raw) / "cache"
            self._write_exact_model_cache(cfg, cache)
            inspections = self._running_instances(cfg)
            docker_calls = []

            def run(argv, check=True):
                docker_calls.append(argv)
                return mock.Mock(returncode=0, stdout=json.dumps({"nvidia": {}}), stderr="")

            with (
                mock.patch.object(endpoint_manager.shutil, "which", return_value="/bin/tool"),
                mock.patch.object(endpoint_manager, "_run", side_effect=run),
                mock.patch.object(endpoint_manager.shutil, "disk_usage", return_value=mock.Mock(
                    free=1 * 1024**3,
                )),
                mock.patch.object(endpoint_manager, "_inspect", side_effect=lambda name: inspections.get(name)),
                mock.patch.object(endpoint_manager, "readiness_probe", side_effect=lambda _cfg, role, **_kw: {
                    "role": role, "models_ok": True, "inference_ok": role != "image_edit",
                }),
                mock.patch.object(endpoint_manager, "_require_component_images", return_value={}),
                mock.patch.object(endpoint_manager, "component_status", return_value={"components": {}}),
            ):
                report = endpoint_manager.start(
                    cfg, "run_42", cache, platform="virtualenv",
                    gpu_identity_prefix="host/run_42",
                )
        self.assertEqual(report["disposition"], "reuse_no_acquisition")
        self.assertEqual(docker_calls, [["docker", "info", "--format", "{{json .Runtimes}}"]])

    def test_unhealthy_endpoint_enforces_full_capacity_gate(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as raw:
            cache = pathlib.Path(raw) / "cache"
            self._write_exact_model_cache(cfg, cache)
            inspections = self._running_instances(cfg)
            inspections[container_name("run_42", "vlm")]["State"]["Running"] = False
            with self.assertRaisesRegex(ValueError, "require at least 150 GiB"):
                self._low_space_preflight(cfg, cache, inspections)

    def test_wrong_revision_endpoint_enforces_full_capacity_gate(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as raw:
            cache = pathlib.Path(raw) / "cache"
            self._write_exact_model_cache(cfg, cache)
            inspections = self._running_instances(cfg)
            record = inspections[container_name("run_42", "llm")]
            command = record["Config"]["Cmd"]
            command[command.index("--revision") + 1] = "wrong-revision"
            with self.assertRaisesRegex(ValueError, "require at least 150 GiB"):
                self._low_space_preflight(cfg, cache, inspections)

    def test_missing_exact_cache_receipt_enforces_full_capacity_gate(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as raw:
            cache = pathlib.Path(raw) / "cache"
            self._write_exact_model_cache(cfg, cache, roles=("image_edit", "vlm"))
            with self.assertRaisesRegex(ValueError, "require at least 150 GiB"):
                self._low_space_preflight(cfg, cache, self._running_instances(cfg))

    def test_generation_topology_fields_are_positive_integers(self):
        for field, value in (("generation_nodes", 0), ("gpus_per_generation_node", True)):
            cfg = config()
            cfg["generation"][field] = value
            with self.assertRaisesRegex(ValueError, field):
                validate_config(cfg)

    def test_pool_created_at_accepts_python_utc_and_kubernetes(self):
        cfg = config()
        payload = {
            "schema_version": "1", "platform": "kubernetes",
            "model": {"id": cfg["models"]["image_edit"]["id"],
                      "revision": cfg["models"]["image_edit"]["revision"]},
            "required_capacity": 1, "auth_env": "IMAGE_EDIT_API_KEY",
            "endpoints": [{
                "id": "gpu-2", "url": "http://worker:8002/v1", "capacity": 1,
                "gpu_identity": "pod/gpu:2", "owner": {"native_id": "pod-1", "name": "worker"},
            }],
            "created_at": "2026-08-19T01:02:03+00:00", "request_sha256": "a" * 64,
        }
        self.assertEqual(validate_image_edit_endpoint_pool(payload)["created_at"], "2026-08-19T01:02:03Z")

    def test_only_owned_never_started_gpu_parse_failure_is_recreated(self):
        cfg = config()
        cfg["endpoints"]["gpu_ids"]["image_edit"] = [0, 1, 2, 3]
        cfg["generation"]["gpus_per_generation_node"] = 4
        failed = {
            "Id": "abcdef1234567890",
            "Config": {"Labels": {
                "com.nvidia.tao.workflow": "tao-run-deft-iaa",
                "com.nvidia.tao.run": "run_42",
                "com.nvidia.tao.role": "image_edit",
            }},
            "State": {
                "Status": "created", "Running": False,
                "StartedAt": "0001-01-01T00:00:00Z",
                "Error": "cannot set both Count and DeviceIDs on device request",
            },
            "HostConfig": {"DeviceRequests": [{"Count": 3, "DeviceIDs": ["0"]}]},
        }
        run_results = []

        def fake_run(argv, check=True):
            run_results.append(argv)
            stdout = "new-container-id\n" if argv[:2] == ["docker", "run"] else ""
            return mock.Mock(returncode=0, stdout=stdout, stderr="")

        with (
            mock.patch.object(endpoint_manager, "preflight", return_value={"ownership": "managed"}),
            mock.patch.object(endpoint_manager, "_require_component_images", return_value={}),
            mock.patch.object(endpoint_manager, "_inspect", side_effect=[failed, None, None, None, None, None]),
            mock.patch.object(endpoint_manager, "_run", side_effect=fake_run),
            mock.patch.object(endpoint_manager, "wait_until_ready", return_value={"inference_ok": True}),
        ):
            report = endpoint_manager.start(cfg, "run_42", pathlib.Path("/tmp/hf"))
        self.assertIn(["docker", "rm", "tao-deft-iaa-run-42-image-edit-gpu-0"], run_results)
        image_run = next(argv for argv in run_results if argv[:2] == ["docker", "run"])
        self.assertEqual(image_run[image_run.index("--gpus") + 1], '"device=0"')
        self.assertNotIn("all", image_run)
        self.assertEqual(report["recoveries"][0]["log_capture"], "empty")

    def test_recovery_never_applies_to_foreign_or_started_container(self):
        cfg = config()
        base = {
            "Config": {"Labels": {
                "com.nvidia.tao.workflow": "tao-run-deft-iaa",
                "com.nvidia.tao.run": "run_42",
                "com.nvidia.tao.role": "image_edit",
            }},
            "State": {
                "Status": "created", "Running": False,
                "StartedAt": "0001-01-01T00:00:00Z",
                "Error": "cannot set both Count and DeviceIDs on device request",
            },
            "HostConfig": {"DeviceRequests": [{"Count": 3, "DeviceIDs": ["0"]}]},
        }
        self.assertTrue(endpoint_manager._recoverable_gpu_parse_failure(base, "run_42", "image_edit", [0, 1, 2, 3]))
        self.assertFalse(endpoint_manager._recoverable_gpu_parse_failure(base, "other", "image_edit", [0, 1, 2, 3]))
        base["State"]["StartedAt"] = "2026-08-18T20:00:00Z"
        self.assertFalse(endpoint_manager._recoverable_gpu_parse_failure(base, "run_42", "image_edit", [0, 1, 2, 3]))

    def test_repair_created_removes_without_starting_workload(self):
        cfg = config()
        cfg["endpoints"]["gpu_ids"]["image_edit"] = [0, 1, 2, 3]
        failed = {
            "Config": {"Labels": {
                "com.nvidia.tao.workflow": "tao-run-deft-iaa", "com.nvidia.tao.run": "run_42",
                "com.nvidia.tao.role": "image_edit",
            }},
            "State": {"Status": "created", "Running": False, "StartedAt": "0001-01-01T00:00:00Z",
                      "Error": "cannot set both Count and DeviceIDs on device request\n"
                               "cannot set both Count and DeviceIDs on device request"},
            "HostConfig": {"DeviceRequests": [{"Count": 3, "DeviceIDs": ["0"]}]},
        }
        calls = []

        def fake_run(argv, check=True):
            calls.append(argv)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with (
            mock.patch.object(endpoint_manager, "_inspect", side_effect=[failed, None, None]),
            mock.patch.object(endpoint_manager, "_run", side_effect=fake_run),
        ):
            report = endpoint_manager.repair_created(cfg, "run_42")
        self.assertEqual(report["recoveries"][0]["disposition"], "removed_for_runtime_rebind")
        self.assertIn(["docker", "rm", "tao-deft-iaa-run-42-image-edit"], calls)
        self.assertFalse(any(argv[:2] == ["docker", "run"] for argv in calls))

    def test_success_manifest_restarts_two_later_crashed_owned_endpoints(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            config_path = root / "sdg.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            output = root / "manifest.json"
            prior = {
                "status": "success", "attempt": 2, "restart_count": 0, "ownership": "managed",
                "containers": {
                    role: {
                        "name": container_name("run_42", role), "owned": True,
                        "model": cfg["models"][role]["id"],
                        "probe": {"models_ok": True, "inference_ok": role != "image_edit"},
                    } for role in ("image_edit", "vlm", "llm")
                },
            }
            output.write_text(json.dumps(prior))
            with (
                mock.patch.object(endpoint_manager, "_load", return_value=cfg),
                mock.patch.object(endpoint_manager, "_all_managed_roles_running", return_value=False),
                mock.patch.object(endpoint_manager, "_clean_owned_shutdown_evidence", return_value=None),
                mock.patch.object(endpoint_manager, "_inspect", return_value=None),
                mock.patch.object(endpoint_manager, "start", return_value={"ownership": "managed", "containers": prior["containers"]}) as start,
            ):
                code = endpoint_manager.main([
                    "start", "--config", str(config_path), "--run-id", "run_42",
                    "--cache-dir", str(root / "cache"), "--output", str(output),
                ])
            self.assertEqual(code, 0)
            start.assert_called_once()
            refreshed = json.loads(output.read_text())
            self.assertEqual(refreshed["attempt"], 2)
            self.assertEqual(refreshed["restart_count"], 1)

            output.write_text(json.dumps(refreshed))
            with (
                mock.patch.object(endpoint_manager, "_load", return_value=cfg),
                mock.patch.object(endpoint_manager, "_all_managed_roles_running", return_value=False),
                mock.patch.object(endpoint_manager, "_clean_owned_shutdown_evidence", return_value=None),
                mock.patch.object(endpoint_manager, "_inspect", return_value=None),
                mock.patch.object(endpoint_manager, "start", return_value={"ownership": "managed", "containers": prior["containers"]}) as second_start,
            ):
                code = endpoint_manager.main([
                    "start", "--config", str(config_path), "--run-id", "run_42",
                    "--cache-dir", str(root / "cache"), "--output", str(output),
                ])
            self.assertEqual(code, 0)
            second_start.assert_called_once()
            refreshed = json.loads(output.read_text())
            self.assertEqual(refreshed["restart_count"], 2)

            output.write_text(json.dumps(refreshed))
            with (
                mock.patch.object(endpoint_manager, "_load", return_value=cfg),
                mock.patch.object(endpoint_manager, "_all_managed_roles_running", return_value=False),
                mock.patch.object(endpoint_manager, "_clean_owned_shutdown_evidence", return_value=None),
                mock.patch.object(endpoint_manager, "_inspect", return_value=None),
                mock.patch.object(endpoint_manager, "start") as second_start,
            ):
                code = endpoint_manager.main([
                    "start", "--config", str(config_path), "--run-id", "run_42",
                    "--cache-dir", str(root / "cache"), "--output", str(output),
                ])
            self.assertEqual(code, 2)
            second_start.assert_not_called()

    def test_three_planned_stop_start_cycles_do_not_consume_crash_budget(self):
        cfg = config()
        clean = {
            "state": "intentionally_stopped",
            "containers": {
                role: {"name": container_name("run_42", role), "owned": True,
                       "status": "exited", "exit_code": 0, "oom_killed": False,
                       "finished_at": "2026-08-19T01:02:03Z"}
                for role in ("image_edit", "vlm", "llm")
            },
        }
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            config_path = root / "sdg.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            output = root / "manifest.json"
            output.write_text(json.dumps(self._success_manifest(cfg)))
            for _ in range(3):
                with (
                    mock.patch.object(endpoint_manager, "_load", return_value=cfg),
                    mock.patch.object(endpoint_manager, "stop", return_value={
                        "ownership": "managed", "stopped": list(clean["containers"]), "removed": [],
                    }),
                    mock.patch.object(endpoint_manager, "_clean_owned_shutdown_evidence", return_value=clean),
                ):
                    self.assertEqual(endpoint_manager.main([
                        "stop", "--config", str(config_path), "--run-id", "run_42",
                        "--output", str(output),
                    ]), 0)
                stopped = json.loads(output.read_text())
                self.assertEqual(stopped["lifecycle"]["state"], "intentionally_stopped")
                with (
                    mock.patch.object(endpoint_manager, "_load", return_value=cfg),
                    mock.patch.object(endpoint_manager, "_all_managed_roles_running", return_value=False),
                    mock.patch.object(endpoint_manager, "_clean_owned_shutdown_evidence", return_value=clean),
                    mock.patch.object(endpoint_manager, "start", return_value={
                        "ownership": "managed", "containers": stopped["containers"],
                    }),
                ):
                    self.assertEqual(endpoint_manager.main([
                        "start", "--config", str(config_path), "--run-id", "run_42",
                        "--cache-dir", str(root / "cache"), "--output", str(output),
                    ]), 0)
                resumed = json.loads(output.read_text())
                self.assertEqual(resumed["restart_count"], 0)
                self.assertEqual(resumed["attempt"], 2)
                self.assertEqual(resumed["lifecycle"], {"state": "running"})

    def test_legacy_clean_owned_stop_is_resumable_without_marker(self):
        cfg = config()
        previous = self._success_manifest(cfg, restart_count=1)
        states = [self._owned_state(role) for role in ("image_edit", "vlm", "llm")]
        with mock.patch.object(endpoint_manager, "_inspect", side_effect=states):
            evidence = endpoint_manager._intentional_shutdown_resume(previous, cfg, "run_42")
        self.assertEqual(evidence["state"], "intentionally_stopped")

    def test_clean_shutdown_requires_complete_exact_owned_zero_exit_set(self):
        cfg = config()
        cases = (
            [self._owned_state("image_edit"), self._owned_state("vlm")],
            [self._owned_state("image_edit"), self._owned_state("vlm", owned=False), self._owned_state("llm")],
            [self._owned_state("image_edit"), self._owned_state("vlm", exit_code=1), self._owned_state("llm")],
        )
        for states in cases:
            with self.subTest(states=len(states)):
                padded = states + [None] * (3 - len(states))
                with mock.patch.object(endpoint_manager, "_inspect", side_effect=padded):
                    self.assertIsNone(endpoint_manager._clean_owned_shutdown_evidence(cfg, "run_42"))

    def test_external_stop_never_mutates_user_managed_endpoints(self):
        cfg = config("external")
        with mock.patch.object(endpoint_manager, "_inspect") as inspect:
            report = endpoint_manager.stop(cfg, "run_42")
        inspect.assert_not_called()
        self.assertEqual(report["ownership"], "external")
        self.assertEqual(report["stopped"], [])

    def test_recovers_exact_old_helper_overwrite_with_committed_receipt(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            manifest = root / "manifest.json"
            original = {
                "schema_version": "1", "status": "error", "action": "start", "attempt": 1,
                "restart_count": 3,
                "error": f"endpoint restart budget exhausted; inspect {manifest.resolve()}",
            }
            manifest.write_text(json.dumps(original))
            receipt = self._write_execution_receipt(root, cfg, manifest)
            states = [self._configured_container(cfg, role) for role in ("image_edit", "vlm", "llm")]
            with mock.patch.object(endpoint_manager, "_inspect", side_effect=states + states):
                recovered = endpoint_manager.recover_overwritten_stop(
                    cfg, "run_42", manifest.resolve(), receipt,
                )
            evidence = root / "manifest.restart-budget-error.json"
            self.assertEqual(json.loads(evidence.read_text()), original)
            self.assertEqual(recovered["status"], "success")
            self.assertEqual(recovered["restart_count"], 2)
            self.assertEqual(recovered["lifecycle"]["state"], "intentionally_stopped")
            self.assertEqual(
                recovered["recovery"]["disposition"],
                "intentional_shutdown_restored_without_starting_endpoints",
            )

    def test_overwrite_recovery_rejects_missing_readiness_receipt(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": "1", "status": "error", "action": "start", "attempt": 1,
                "restart_count": 3,
                "error": f"endpoint restart budget exhausted; inspect {manifest.resolve()}",
            }))
            receipt = root / "deft_state.json"
            receipt.write_text(json.dumps({"workflow": "tao-run-deft-iaa", "config": {"sdg": {
                "endpoint_mode": "managed", "models": cfg["models"],
                "gpu_ids": cfg["endpoints"]["gpu_ids"], "images": cfg["images"],
            }}, "iterations": {}}))
            states = [self._configured_container(cfg, role) for role in ("image_edit", "vlm", "llm")]
            with (
                mock.patch.object(endpoint_manager, "_inspect", side_effect=states + states),
                self.assertRaisesRegex(ValueError, "no committed successful SDG stage"),
            ):
                endpoint_manager.recover_overwritten_stop(cfg, "run_42", manifest.resolve(), receipt)

    def test_overwrite_recovery_rejects_changed_config_or_container_identity(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": "1", "status": "error", "action": "start", "attempt": 1,
                "restart_count": 3,
                "error": f"endpoint restart budget exhausted; inspect {manifest.resolve()}",
            }))
            receipt = self._write_execution_receipt(root, cfg, manifest)
            changed = config()
            changed["models"]["llm"]["revision"] = "changed-revision"
            clean = [self._configured_container(changed, role) for role in ("image_edit", "vlm", "llm")]
            with (
                mock.patch.object(endpoint_manager, "_inspect", side_effect=clean),
                self.assertRaisesRegex(ValueError, "does not match immutable config"),
            ):
                endpoint_manager.recover_overwritten_stop(changed, "run_42", manifest.resolve(), receipt)

            foreign = [self._configured_container(cfg, role) for role in ("image_edit", "vlm", "llm")]
            foreign[1] = self._configured_container(cfg, "vlm", owned=False)
            with (
                mock.patch.object(endpoint_manager, "_inspect", side_effect=foreign),
                self.assertRaisesRegex(ValueError, "cleanly stopped"),
            ):
                endpoint_manager.recover_overwritten_stop(cfg, "run_42", manifest.resolve(), receipt)

            wrong_model = [
                self._configured_container(cfg, "image_edit"),
                self._configured_container(cfg, "vlm"),
                self._configured_container(cfg, "llm", model_override="different/model"),
            ]
            with (
                mock.patch.object(endpoint_manager, "_inspect", side_effect=wrong_model),
                self.assertRaisesRegex(ValueError, "identity does not match"),
            ):
                endpoint_manager.recover_overwritten_stop(cfg, "run_42", manifest.resolve(), receipt)

    def test_overwrite_recovery_rejects_nonzero_oom_and_other_errors(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            manifest = root / "manifest.json"
            exact = {
                "schema_version": "1", "status": "error", "action": "start", "attempt": 1,
                "restart_count": 3,
                "error": f"endpoint restart budget exhausted; inspect {manifest.resolve()}",
            }
            manifest.write_text(json.dumps(exact))
            receipt = self._write_execution_receipt(root, cfg, manifest)
            for bad in (
                self._configured_container(cfg, "vlm", exit_code=1),
                self._configured_container(cfg, "vlm", oom=True),
            ):
                states = [self._configured_container(cfg, "image_edit"), bad,
                          self._configured_container(cfg, "llm")]
                with (
                    mock.patch.object(endpoint_manager, "_inspect", side_effect=states),
                    self.assertRaisesRegex(ValueError, "cleanly stopped"),
                ):
                    endpoint_manager.recover_overwritten_stop(cfg, "run_42", manifest.resolve(), receipt)
            other = dict(exact, error="some other failure")
            manifest.write_text(json.dumps(other))
            with self.assertRaisesRegex(ValueError, "exact old-helper"):
                endpoint_manager.recover_overwritten_stop(cfg, "run_42", manifest.resolve(), receipt)

    def test_stopped_image_edit_with_default_shm_is_recreated(self):
        cfg = config()
        stale = {
            "Id": "abcdef1234567890",
            "Config": {"Labels": {
                "com.nvidia.tao.workflow": "tao-run-deft-iaa",
                "com.nvidia.tao.run": "run_42",
                "com.nvidia.tao.role": "image_edit",
            }},
            "State": {"Status": "exited", "Running": False, "ExitCode": 135},
            "HostConfig": {"ShmSize": 64 * 1024**2},
        }
        calls = []

        def fake_run(argv, check=True):
            calls.append(argv)
            return mock.Mock(returncode=0, stdout="new-container-id\n", stderr="")

        with (
            mock.patch.object(endpoint_manager, "preflight", return_value={"component_images": {}}),
            mock.patch.object(endpoint_manager, "_require_component_images", return_value={}),
            mock.patch.object(endpoint_manager, "_inspect", side_effect=[stale, None, None]),
            mock.patch.object(endpoint_manager, "_run", side_effect=fake_run),
            mock.patch.object(endpoint_manager, "wait_until_ready", return_value={"models_ok": True}),
        ):
            report = endpoint_manager.start(cfg, "run_42", pathlib.Path("/tmp/hf"))
        self.assertIn(["docker", "rm", container_name("run_42", "image_edit", 2)], calls)
        self.assertEqual(report["recoveries"][0]["previous_shm_bytes"], 64 * 1024**2)
        self.assertEqual(report["recoveries"][0]["disposition"], "removed_and_recreated_with_required_shared_memory")

    def test_external_endpoints_are_never_startable(self):
        with self.assertRaisesRegex(ValueError, "never started"):
            build_endpoint_command(config("external"), "llm", "r", pathlib.Path("/tmp/hf"))

    def test_gpu_capacity_and_compute_checks(self):
        cfg = config()
        inventory = [
            {"index": 2, "compute_capability": 9.0, "memory_free_mib": 80000},
            {"index": 4, "compute_capability": 9.0, "memory_free_mib": 60000},
            {"index": 6, "compute_capability": 9.0, "memory_free_mib": 30000},
        ]
        validate_gpu_inventory(cfg, inventory)
        inventory[0]["memory_free_mib"] = 1000
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_gpu_inventory(cfg, inventory)
        inventory[0].update(memory_free_mib=80000, compute_capability=7.5)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_gpu_inventory(cfg, inventory)

    def test_port_collision(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        try:
            self.assertFalse(port_available(sock.getsockname()[1]))
        finally:
            sock.close()

    def test_readiness_success_wrong_model_malformed_and_timeout(self):
        cfg = config("external")
        model = cfg["models"]["llm"]["id"]

        def good(url, **kwargs):
            return {"data": [{"id": model}]} if url.endswith("/models") else {"choices": [{}]}

        self.assertTrue(readiness_probe(cfg, "llm", good)["inference_ok"])
        with self.assertRaisesRegex(ValueError, "does not serve"):
            readiness_probe(cfg, "llm", lambda *args, **kwargs: {"data": [{"id": "wrong"}]})
        with self.assertRaises((AttributeError, ValueError)):
            readiness_probe(cfg, "llm", lambda *args, **kwargs: [])
        with mock.patch("iaa_deft.sdg.time.monotonic", side_effect=[0.0, 2.0]):
            with self.assertRaises(TimeoutError):
                wait_until_ready(lambda: (_ for _ in ()).throw(ValueError("not ready")), 1, 0, lambda _: None)

    def test_component_commands_use_no_gpu_widening_or_secret_values(self):
        cfg = config("external")
        with mock.patch.dict("os.environ", {"VLM_API_KEY": "super-secret"}, clear=False):
            argv = build_component_command(
                cfg, "augment", input_root=pathlib.Path("/tmp/in"),
                output_root=pathlib.Path("/tmp/out"), source_key="p1", attempt=2,
                target_attributes={"top outer color": "red"},
            )
        self.assertIn("VLM_API_KEY", argv)
        self.assertNotIn("super-secret", argv)
        self.assertNotIn("--gpus", argv)
        self.assertEqual(argv[argv.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
        groups = [argv[index + 1] for index, value in enumerate(argv) if value == "--group-add"]
        self.assertEqual(groups, ["10000", "1000"])
        self.assertEqual(argv[argv.index("--entrypoint") + 1], "uv")
        self.assertEqual(argv.count("uv"), 1)
        self.assertIn("/tmp/in:/app/data/in:ro", argv)
        self.assertIn("/tmp/out:/app/data/out", argv)
        self.assertIn("/app/data/out/panes/p1.jpg", " ".join(argv))
        self.assertIn("--no-sync", argv)
        self.assertIn("pipeline.request_timeout=600", argv)
        self.assertIn("captioning.llm.variables.top_outer_color=[red]", argv)
        with self.assertRaisesRegex(ValueError, "approved"):
            build_component_command(cfg, "augment", input_root=pathlib.Path("/tmp/in"), output_root=pathlib.Path("/tmp/out"), source_key="p1", attempt=3)
        label = build_component_command(
            cfg, "label", input_root=pathlib.Path("/tmp/in"),
            output_root=pathlib.Path("/tmp/out"), source_key="p1",
        )
        self.assertIn("/tmp/in:/input:ro", label)
        self.assertIn("/tmp/out:/output", label)
        self.assertIn("data.0.inputs.video_path=/input/p1.jpg", label)
        self.assertIn("--no-sync", label)

    def test_prebuilt_component_images_are_presence_checked_only(self):
        cfg = config("external")
        with mock.patch.object(endpoint_manager, "_inspect_image", return_value=None):
            report = endpoint_manager.component_status(cfg)
            self.assertFalse(report["components"]["augmentation"]["present"])
            with self.assertRaisesRegex(ValueError, "pull the approved prebuilt images"):
                endpoint_manager._require_component_images(cfg)
        record = {"Id": "sha256:built", "Config": {"Labels": {}}}
        with mock.patch.object(endpoint_manager, "_inspect_image", return_value=record):
            report = endpoint_manager.component_status(cfg)
        self.assertTrue(report["components"]["auto_labeling"]["present"])
        self.assertEqual(report["components"]["auto_labeling"]["image"], cfg["images"]["auto_labeling"])


class GenerationContractTests(unittest.TestCase):
    def test_endpoint_lineage_policy_is_in_digest_bound_runtime_only(self):
        runner = (ROOT / "scripts" / "run_sdg_stage.py").read_text()
        policy = (ROOT / "scripts" / "iaa_deft" / "sdg.py").read_text()
        self.assertIn("bind_resumable_endpoint_pool", runner)
        self.assertNotIn("MAX_CONTROLLER_POOL_SNAPSHOTS", runner)
        self.assertIn("MAX_CONTROLLER_POOL_SNAPSHOTS", policy)
        with tempfile.TemporaryDirectory() as raw:
            copied = pathlib.Path(raw) / "iaa_deft"
            shutil.copytree(ROOT / "scripts" / "iaa_deft", copied)
            before = python_tree_sha256(copied)
            with (copied / "sdg.py").open("a", encoding="utf-8") as stream:
                stream.write("\n# digest binding probe\n")
            self.assertNotEqual(before, python_tree_sha256(copied))

    def _pool_manifest(self, cfg, endpoints, platform="host"):
        return {
            "schema_version": "1", "platform": platform,
            "model": {
                "id": cfg["models"]["image_edit"]["id"],
                "revision": cfg["models"]["image_edit"]["revision"],
            },
            "required_capacity": len(endpoints), "auth_env": None,
            "endpoints": endpoints, "created_at": "2026-08-19T12:00:00Z",
            "request_sha256": "a" * 64,
        }

    def _execution_fixture(self, root, cfg, keys):
        output = root / "datagen"
        (output / "panes").mkdir(parents=True)
        selected = []
        for key in keys:
            (output / "panes" / f"{key}.jpg").write_bytes(b"pane")
            selected.append({
                "source_key": key,
                "mined_unique_name": f"{key}.jpg",
                "source_attribute_values": [0] * 7,
                "target_attributes": {"top outer color": "red"},
            })
        (output / "sdg_plan.json").write_text(json.dumps({"selected": selected}))
        (output / "sdg_progress.json").write_text(json.dumps({
            "schema_version": "1", "preprocessed": True, "augmentation": {},
            "split": False, "labeling": {}, "command_attempts": {},
        }))
        args = __import__("argparse").Namespace(
            output_root=output, mined_pairs=None, attribute_vocab=root / "vocab.json",
            eval_list=root / "eval.txt", accepted_manifest=None, labels_root=None,
        )
        return output, args

    def _component_double(
        self, output, calls, active, lock, *, fail_once=None, reject_once=None,
        reject_always=None, fail_endpoint_once=None, fail_always_source=None,
        augment_barrier=None,
    ):
        def component(
            config_payload, action, input_root, output_root, log_path,
            source_key="", attempt=1, target_attributes=None,
            image_edit_url=None, image_edit_endpoint_id=None,
        ):
            if action == "augment":
                with lock:
                    calls.setdefault(source_key, []).append(attempt)
                    active.setdefault("endpoints_used", {}).setdefault(source_key, []).append(
                        image_edit_endpoint_id
                    )
                    active["current"] += 1
                    active["maximum"] = max(active["maximum"], active["current"])
                    endpoint_active = active.setdefault("endpoint_active", {})
                    endpoint_maximum = active.setdefault("endpoint_maximum", {})
                    endpoint_active[image_edit_endpoint_id] = endpoint_active.get(image_edit_endpoint_id, 0) + 1
                    endpoint_maximum[image_edit_endpoint_id] = max(
                        endpoint_maximum.get(image_edit_endpoint_id, 0),
                        endpoint_active[image_edit_endpoint_id],
                    )
                    active.setdefault("intervals", {}).setdefault(source_key, []).append([time.monotonic(), None])
                try:
                    if augment_barrier is not None and source_key != "smoke":
                        augment_barrier.wait(timeout=5)
                    time.sleep(0.04)
                    if (
                        fail_always_source == source_key
                        or
                        (fail_once == source_key and len(calls[source_key]) == 1)
                        or (
                            fail_endpoint_once == image_edit_endpoint_id
                            and active["endpoints_used"][source_key].count(image_edit_endpoint_id) == 1
                        )
                    ):
                        raise ValueError("injected component failure")
                    attempt_root = output / "augmentation" / source_key / f"attempt_{attempt}"
                    passed = not (
                        reject_always == source_key
                        or (reject_once == source_key and attempt == 1)
                    )
                    metadata(attempt_root / "output_metadata.json", passed)
                    (attempt_root / "output.jpg").write_bytes(b"generated")
                    (attempt_root / "output.txt").write_text("caption")
                finally:
                    with lock:
                        active["intervals"][source_key][-1][1] = time.monotonic()
                        active["endpoint_active"][image_edit_endpoint_id] -= 1
                        active["current"] -= 1
                return
            if action == "split":
                progress = json.loads((output / "sdg_progress.json").read_text())
                for key, outcome in sorted(progress["augmentation"].items()):
                    if outcome["status"] == "accepted":
                        crop = output / "augmented_dataset" / "augmented_imgs" / f"{key}_aug" / "0.jpg"
                        crop.parent.mkdir(parents=True, exist_ok=True)
                        crop.write_bytes(b"crop")
                return
            if action == "label":
                qa = output / "labels" / source_key / "task" / "open_qa.json"
                qa.parent.mkdir(parents=True, exist_ok=True)
                qa.write_text('{"items": []}')
                return
            raise AssertionError(action)

        return component

    def _execute_with_double(self, args, cfg, component):
        with (
            mock.patch.object(stage_runner, "_component_call", side_effect=component),
            mock.patch.object(stage_runner, "validate_labels", return_value={}),
            mock.patch.object(stage_runner, "normalize", return_value={"status": "ok"}),
        ):
            return stage_runner.execute(args, cfg)

    def test_augmentation_smoke_is_single_then_default_workers_overlap_deterministically(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cfg = config()
            cfg["endpoints"]["gpu_ids"]["image_edit"] = [0, 1, 2, 3]
            keys = ["smoke", "zeta", "beta", "delta", "alpha"]
            output, args = self._execution_fixture(root, cfg, keys)
            calls, active, lock = {}, {"current": 0, "maximum": 0}, threading.Lock()
            component = self._component_double(output, calls, active, lock, reject_once="beta")

            self._execute_with_double(args, cfg, component)

            self.assertEqual(stage_runner._augmentation_max_in_flight(cfg, 4), 4)
            smoke_end = active["intervals"]["smoke"][0][1]
            batch_start = min(active["intervals"][key][0][0] for key in keys[1:])
            self.assertLessEqual(smoke_end, batch_start)
            self.assertGreaterEqual(active["maximum"], 2)
            self.assertEqual(calls["beta"], [1, 2])
            beta_first, beta_second = active["intervals"]["beta"]
            self.assertLessEqual(beta_first[1], beta_second[0])
            progress = json.loads((output / "sdg_progress.json").read_text())
            self.assertEqual(list(progress["augmentation"]), sorted(keys))
            self.assertEqual(
                json.loads((output / "augmentation_smoke.json").read_text())["source_key"],
                "smoke",
            )

    def test_failed_smoke_exhausts_its_bound_without_starting_batch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cfg = config()
            cfg["endpoints"]["gpu_ids"]["image_edit"] = [0, 1, 2, 3]
            output, args = self._execution_fixture(root, cfg, ["smoke", "never-started"])
            calls, active, lock = {}, {"current": 0, "maximum": 0}, threading.Lock()
            component = self._component_double(
                output, calls, active, lock, reject_always="smoke"
            )

            with self.assertRaisesRegex(ValueError, "smoke source smoke did not pass"):
                self._execute_with_double(args, cfg, component)

            self.assertEqual(calls, {"smoke": [1, 2]})
            self.assertEqual(active["maximum"], 1)

    def test_max_in_flight_one_never_overlaps(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cfg = config()
            cfg["endpoints"]["gpu_ids"]["image_edit"] = [0, 1, 2, 3]
            cfg["generation"]["max_in_flight"] = 1
            output, args = self._execution_fixture(root, cfg, ["smoke", "b", "c", "d"])
            calls, active, lock = {}, {"current": 0, "maximum": 0}, threading.Lock()

            self._execute_with_double(
                args, cfg, self._component_double(output, calls, active, lock)
            )

            self.assertEqual(active["maximum"], 1)

    def test_concurrent_failure_is_isolated_and_resume_skips_accepted_sources(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cfg = config()
            cfg["endpoints"]["gpu_ids"]["image_edit"] = [0, 1, 2, 3]
            output, args = self._execution_fixture(root, cfg, ["smoke", "fails", "kept"])
            calls, active, lock = {}, {"current": 0, "maximum": 0}, threading.Lock()
            first = self._component_double(
                output, calls, active, lock, fail_always_source="fails"
            )

            with self.assertRaisesRegex(ValueError, "fails: injected component failure"):
                self._execute_with_double(args, cfg, first)
            self.assertEqual(calls, {"smoke": [1], "fails": [1, 1], "kept": [1]})
            progress = json.loads((output / "sdg_progress.json").read_text())
            self.assertEqual(progress["augmentation"]["kept"]["status"], "accepted")

            second = self._component_double(output, calls, active, lock)
            self._execute_with_double(args, cfg, second)
            self.assertEqual(calls["smoke"], [1])
            self.assertEqual(calls["kept"], [1])
            self.assertEqual(calls["fails"], [1, 1, 2])

    def test_changed_pool_resumes_terminal_augmentation_for_unfinished_labeling(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cfg = config("external")
            output, args = self._execution_fixture(root, cfg, ["smoke", "kept"])
            pool = root / "pool.json"
            first_entries = [{
                "id": "old-gpu", "url": "http://old.internal:19000/v1", "capacity": 1,
                "gpu_identity": "old/gpu-0", "owner": {"native_id": "old", "name": "old"},
            }]
            pool.write_text(json.dumps(self._pool_manifest(cfg, first_entries)))
            args.image_edit_endpoint_pool = pool
            args.execution_platform = "host"
            calls, active, lock = {}, {"current": 0, "maximum": 0}, threading.Lock()
            component = self._component_double(output, calls, active, lock)
            self._execute_with_double(args, cfg, component)

            progress = json.loads((output / "sdg_progress.json").read_text())
            progress["labeling"] = {}
            (output / "sdg_progress.json").write_text(json.dumps(progress))
            second_entries = [{
                "id": "new-gpu", "url": "http://new.internal:19000/v1", "capacity": 1,
                "gpu_identity": "new/gpu-0", "owner": {"native_id": "new", "name": "new"},
            }]
            pool.write_text(json.dumps(self._pool_manifest(cfg, second_entries)))
            before = {key: list(value) for key, value in calls.items()}
            self._execute_with_double(args, cfg, component)

            self.assertEqual(calls, before)
            resumed = json.loads((output / "sdg_progress.json").read_text())
            self.assertEqual(resumed["image_edit_endpoint_history"][0]["endpoints"], first_entries)
            self.assertEqual(resumed["image_edit_endpoints"], second_entries)
            self.assertTrue(all(value == "accepted" for value in resumed["labeling"].values()))

    def test_changed_pool_partial_resume_skips_completed_augmentation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cfg = config("external")
            output, args = self._execution_fixture(root, cfg, ["smoke", "kept", "fails"])
            pool = root / "pool.json"
            old = [{
                "id": "old-gpu", "url": "http://old.internal:19000/v1", "capacity": 1,
                "gpu_identity": "old/gpu-0", "owner": {"native_id": "old", "name": "old"},
            }]
            pool.write_text(json.dumps(self._pool_manifest(cfg, old)))
            args.image_edit_endpoint_pool = pool
            args.execution_platform = "host"
            calls, active, lock = {}, {"current": 0, "maximum": 0}, threading.Lock()
            with self.assertRaisesRegex(ValueError, "fails"):
                self._execute_with_double(
                    args, cfg, self._component_double(
                        output, calls, active, lock, fail_always_source="fails"
                    ),
                )
            old_counts = {key: len(value) for key, value in calls.items()}
            new = [{
                "id": "new-gpu", "url": "http://new.internal:19000/v1", "capacity": 1,
                "gpu_identity": "new/gpu-0", "owner": {"native_id": "new", "name": "new"},
            }]
            pool.write_text(json.dumps(self._pool_manifest(cfg, new)))
            self._execute_with_double(
                args, cfg, self._component_double(output, calls, active, lock)
            )
            self.assertEqual(len(calls["smoke"]), old_counts["smoke"])
            self.assertEqual(len(calls["kept"]), old_counts["kept"])
            self.assertEqual(active["endpoints_used"]["fails"][-1], "new-gpu")

    def test_changed_pool_history_is_bounded_by_controller_attempt_budget(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cfg = config("external")
            output, args = self._execution_fixture(root, cfg, ["smoke", "fails"])
            pool = root / "pool.json"
            args.image_edit_endpoint_pool = pool
            args.execution_platform = "host"
            calls, active, lock = {}, {"current": 0, "maximum": 0}, threading.Lock()
            for name in ("first", "second"):
                entries = [{
                    "id": f"{name}-gpu", "url": f"http://{name}.internal:19000/v1",
                    "capacity": 1, "gpu_identity": f"{name}/gpu-0",
                    "owner": {"native_id": name, "name": name},
                }]
                pool.write_text(json.dumps(self._pool_manifest(cfg, entries)))
                with self.assertRaises(ValueError):
                    self._execute_with_double(
                        args, cfg, self._component_double(
                            output, calls, active, lock, fail_always_source="fails"
                        ),
                    )
            third = [{
                "id": "third-gpu", "url": "http://third.internal:19000/v1", "capacity": 1,
                "gpu_identity": "third/gpu-0", "owner": {"native_id": "third", "name": "third"},
            }]
            pool.write_text(json.dumps(self._pool_manifest(cfg, third)))
            with self.assertRaisesRegex(ValueError, "resume attempt budget exhausted"):
                self._execute_with_double(
                    args, cfg, self._component_double(output, calls, active, lock)
                )

    def test_max_in_flight_validation_is_bounded_by_known_workers(self):
        cfg = config()
        cfg["generation"]["max_in_flight"] = 2
        with self.assertRaisesRegex(ValueError, "worker count"):
            validate_config(cfg)
        external = config("external")
        external["generation"]["max_in_flight"] = 2
        validate_config(external)
        with self.assertRaisesRegex(ValueError, "worker count"):
            stage_runner._augmentation_max_in_flight(external, 2)

    def test_runtime_pool_scales_to_twenty_four_one_gpu_endpoints(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cfg = config("external")
            cfg["generation"]["max_in_flight"] = 24
            manifest = root / "endpoint-pool.json"
            endpoints = [
                    {
                        "id": f"node-{index // 8}-gpu-{index % 8}",
                        "url": f"http://worker-{index // 8}.internal:{19000 + index}/v1",
                        "capacity": 1,
                        "gpu_identity": f"node-{index // 8}/gpu-{index % 8}",
                        "owner": {"native_id": f"job-{index // 8}", "name": f"worker-{index // 8}"},
                    }
                    for index in range(24)
                ]
            manifest.write_text(json.dumps(self._pool_manifest(cfg, endpoints)))
            keys = ["smoke"] + [f"source-{index:02d}" for index in range(24)]
            output, args = self._execution_fixture(root, cfg, keys)
            args.image_edit_endpoint_pool = manifest
            calls, active, lock = {}, {"current": 0, "maximum": 0}, threading.Lock()

            self._execute_with_double(
                args, cfg, self._component_double(
                    output, calls, active, lock, augment_barrier=threading.Barrier(24)
                )
            )

            self.assertEqual(active["maximum"], 24)
            self.assertEqual(len(active["endpoint_maximum"]), 24)
            self.assertTrue(all(value == 1 for value in active["endpoint_maximum"].values()))
            progress = json.loads((output / "sdg_progress.json").read_text())
            self.assertEqual(list(progress["augmentation"]), sorted(keys))
            self.assertEqual(len(progress["image_edit_endpoints"]), 24)

    def test_unhealthy_endpoint_is_quarantined_and_same_attempt_moves_once(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cfg = config("external")
            cfg["generation"]["max_in_flight"] = 2
            manifest = root / "endpoint-pool.json"
            endpoints = [
                    {"id": "bad-gpu", "url": "http://bad.internal:19000/v1", "capacity": 1,
                     "gpu_identity": "node-0/gpu-0",
                     "owner": {"native_id": "job-bad", "name": "bad-worker"}},
                    {"id": "good-gpu", "url": "http://good.internal:19000/v1", "capacity": 1,
                     "gpu_identity": "node-0/gpu-1",
                     "owner": {"native_id": "job-good", "name": "good-worker"}},
                ]
            manifest.write_text(json.dumps(self._pool_manifest(cfg, endpoints)))
            output, args = self._execution_fixture(root, cfg, ["smoke", "next"])
            args.image_edit_endpoint_pool = manifest
            calls, active, lock = {}, {"current": 0, "maximum": 0}, threading.Lock()

            self._execute_with_double(
                args, cfg, self._component_double(
                    output, calls, active, lock, fail_endpoint_once="bad-gpu"
                ),
            )

            self.assertEqual(calls["smoke"], [1, 1])
            self.assertEqual(active["endpoints_used"]["smoke"], ["bad-gpu", "good-gpu"])
            progress = json.loads((output / "sdg_progress.json").read_text())
            self.assertEqual(progress["endpoint_attempts"]["augment:smoke:1"], [
                {"endpoint_id": "bad-gpu", "gpu_identity": "node-0/gpu-0",
                 "owner": {"native_id": "job-bad", "name": "bad-worker"},
                 "status": "quarantined", "url": "http://bad.internal:19000/v1"},
                {"endpoint_id": "good-gpu", "gpu_identity": "node-0/gpu-1",
                 "owner": {"native_id": "job-good", "name": "good-worker"},
                 "status": "completed", "url": "http://good.internal:19000/v1"},
            ])
            self.assertEqual(progress["augmentation"]["smoke"]["endpoint_id"], "good-gpu")

    def test_runtime_pool_validation_requires_unique_one_gpu_endpoints(self):
        cfg = config("external")
        endpoints = [
            {"id": "node-0-gpu-0", "url": "http://node.internal:19000/v1",
             "capacity": 1, "gpu_identity": "node-0/gpu-0",
             "owner": {"native_id": "job-0", "name": "node-0"}},
        ]
        valid = self._pool_manifest(cfg, endpoints)
        self.assertEqual(validate_image_edit_endpoint_pool(valid)["endpoints"][0]["capacity"], 1)
        for field, value, message in (
            ("capacity", 8, "capacity"),
            ("url", "http://token@node.internal/v1", "credentials"),
            ("gpu_identity", "", "gpu_identity"),
        ):
            invalid = json.loads(json.dumps(valid))
            invalid["endpoints"][0][field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_image_edit_endpoint_pool(invalid)
        wrong_model = json.loads(json.dumps(valid))
        wrong_model["model"]["revision"] = "b" * 40
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw) / "pool.json"
            path.write_text(json.dumps(wrong_model))
            with self.assertRaisesRegex(ValueError, "model does not match"):
                stage_runner._runtime_image_edit_endpoint_pool(cfg, path, "host")

    def test_structured_component_executor_receives_no_shell_or_image_arguments(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            executor = root / "slurm_sdg_action.py"
            request = root / "sdg.action.json"
            executor.write_text("#!/usr/bin/env python3\n")
            request.write_text("{}\n")
            output = root / "output"
            input_root = root / "input"
            output.mkdir()
            input_root.mkdir()
            completed = mock.Mock(returncode=0)
            with mock.patch.object(stage_runner.subprocess, "run", return_value=completed) as run:
                stage_runner._component_call(
                    config(), "augment", input_root, output, root / "component.log",
                    "source-1", 2, {"top outer color": "red"},
                    executor, request, "job-123",
                    image_edit_url="http://worker.internal:19000/v1",
                    image_edit_endpoint_id="node-0-gpu-0",
                )
            argv = run.call_args.args[0]
            self.assertEqual(argv[:3], [sys.executable, str(executor), "component"])
            self.assertIn("--request", argv)
            self.assertIn("--job-id", argv)
            self.assertIn("--target-attributes-json", argv)
            self.assertIn("--image-edit-url", argv)
            self.assertIn("--image-edit-endpoint-id", argv)
            self.assertNotIn("docker", argv)
            self.assertFalse(any(token in {"sh", "bash"} for token in argv))
            encoded = argv[argv.index("--target-attributes-json") + 1]
            self.assertEqual(encoded, '{"top outer color":"red"}')

    def test_platform_status_and_manifest_record_selected_execution_frame(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            output = root / "artifact.json"

            def handler():
                output.write_text("{}\n")
                return {"ok": True}

            status = stage_runner._run_once(
                root, "sdg-normalize", [], [output], handler,
                execution_platform="brev",
            )
            self.assertEqual(status["execution_platform"], "brev")
            self.assertTrue((root / "status" / "sdg-normalize.brev.status.json").is_file())

    def test_sdg_commit_converts_validated_config_string_back_to_path(self):
        source = (ROOT / "scripts" / "commit_stage.py").read_text()
        self.assertNotIn(
            '_required_file(sdg_config_path, "state.config.sdg_config").read_text()',
            source,
        )
        log_stage = (ROOT / "scripts" / "log_stage.py").read_text()
        self.assertIn('"sdg",', log_stage)

    def test_generation_frame_uses_selected_platform(self):
        skill = (ROOT / "SKILL.md").read_text()
        reference = (ROOT / "references" / "local-sdg.md").read_text()

        self.assertIn("Every workload runs in the selected platform's compute frame", skill)
        self.assertIn("single-request slots", skill)
        self.assertIn("capacity-one endpoints", reference)

    def test_commit_binds_iteration_endpoint_pool_to_platform_topology_and_auxiliary_evidence(self):
        spec = importlib.util.spec_from_file_location("checked_commit_pool", ROOT / "scripts" / "commit_stage.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        cfg = config()
        cfg["generation"]["generation_nodes"] = 3
        cfg["generation"]["gpus_per_generation_node"] = 8
        endpoints = [{
            "id": f"node-{index}-gpu", "url": f"http://node-{index}:8002/v1", "capacity": 1,
            "gpu_identity": f"node-{index}/gpu:0",
            "owner": {"native_id": f"native-{index}", "name": f"worker-{index}"},
        } for index in range(24)]
        pool = {
            "schema_version": "1", "platform": "slurm",
            "model": {"id": cfg["models"]["image_edit"]["id"],
                      "revision": cfg["models"]["image_edit"]["revision"]},
            "required_capacity": 24, "auth_env": "IMAGE_EDIT_API_KEY",
            "endpoints": endpoints, "created_at": "2026-08-19T12:00:00Z",
            "request_sha256": "a" * 64,
        }
        auxiliary = {
            "request_sha256": "a" * 64,
            "image_edit_pool": {
                "requested_capacity": 24, "requested_nodes": 3,
                "required_capacity": 24, "active_nodes": 3,
            },
            "roles": {
                role: {"model": cfg["models"][role]["id"], "ready": True}
                for role in ("vlm", "llm")
            },
            "components": {
                role: cfg["images"][role] for role in ("augmentation", "auto_labeling")
            },
        }
        module._validated_sdg_endpoint_evidence(pool, auxiliary, cfg, "slurm")
        subset_pool = json.loads(json.dumps(pool))
        subset_pool["required_capacity"] = 8
        subset_pool["endpoints"] = subset_pool["endpoints"][:8]
        subset_auxiliary = json.loads(json.dumps(auxiliary))
        subset_auxiliary["image_edit_pool"].update({
            "required_capacity": 8, "active_nodes": 1,
        })
        module._validated_sdg_endpoint_evidence(
            subset_pool, subset_auxiliary, cfg, "slurm"
        )
        for field, replacement, endpoint_count, message in (
            ("platform", "brev", 24, "platform"),
            ("required_capacity", 7, 7, "complete-worker subset"),
            ("required_capacity", 25, 25, "complete-worker subset"),
        ):
            invalid = json.loads(json.dumps(pool))
            invalid[field] = replacement
            if field == "required_capacity":
                invalid["endpoints"] = [
                    {
                        "id": f"invalid-{index}",
                        "url": f"http://invalid-{index}:8002/v1",
                        "capacity": 1,
                        "gpu_identity": f"invalid-{index}/gpu:0",
                        "owner": {"native_id": f"bad-{index}", "name": f"bad-{index}"},
                    }
                    for index in range(endpoint_count)
                ]
            with self.assertRaisesRegex(ValueError, message):
                module._validated_sdg_endpoint_evidence(invalid, auxiliary, cfg, "slurm")
        wrong_active = json.loads(json.dumps(subset_auxiliary))
        wrong_active["image_edit_pool"]["active_nodes"] = 2
        with self.assertRaisesRegex(ValueError, "active topology"):
            module._validated_sdg_endpoint_evidence(
                subset_pool, wrong_active, cfg, "slurm"
            )
        wrong_aux = json.loads(json.dumps(auxiliary))
        wrong_aux["components"]["augmentation"] = "wrong@sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "augmentation"):
            module._validated_sdg_endpoint_evidence(pool, wrong_aux, cfg, "slurm")
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw) / "endpoint_pool.json"
            path.write_text(json.dumps(pool, sort_keys=True))
            binding = module._endpoint_pool_binding(path, pool)
            self.assertEqual(binding["path"], str(path.resolve()))
            self.assertEqual(binding["request_sha256"], "a" * 64)
            self.assertEqual(binding["required_capacity"], 24)
            self.assertRegex(binding["sha256"], r"^[0-9a-f]{64}$")
            remote = dict(binding)
            remote["path"] = f"/remote/results/{path.parent.parent.name}/{path.parent.name}/{path.name}"
            # A remote path is accepted only when the stable run-relative
            # suffix and all immutable binding fields are preserved.
            canonical = pathlib.Path("/controller/results") / path.parent.parent.name / path.parent.name / path.name
            canonical_binding = dict(binding, path=str(canonical))
            self.assertTrue(module._endpoint_pool_binding_matches(remote, canonical_binding, "slurm"))
            self.assertFalse(module._endpoint_pool_binding_matches(remote, canonical_binding, "docker"))
            wrong_digest = dict(remote, sha256="0" * 64)
            self.assertFalse(module._endpoint_pool_binding_matches(wrong_digest, canonical_binding, "slurm"))
            wrong_suffix = dict(remote, path=remote["path"].replace("endpoint_pool.json", "other.json"))
            self.assertFalse(module._endpoint_pool_binding_matches(wrong_suffix, canonical_binding, "slurm"))

    def test_audit_registers_canonical_iteration_endpoint_evidence_and_legacy_exception(self):
        spec = importlib.util.spec_from_file_location("checked_audit_pool", ROOT / "scripts" / "audit_deft_run.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        results = pathlib.Path("/tmp/run")
        cfg = {"platform": "brev"}
        self.assertEqual(
            module._expected_artifact_path("endpoint_pool", "iter2", results, cfg),
            results / "iter_2" / "datagen" / "endpoint_pool.json",
        )
        self.assertEqual(
            module._expected_artifact_path("endpoint_manifest", "iter2", results, cfg),
            results / "iter_2" / "datagen" / "endpoint_manifest.json",
        )
        self.assertIn("endpoint_pool", module.STAGE_REQUIRED_FIELDS["sdg"])
        self.assertEqual(module.FIELD_STAGE["endpoint_pool"], "sdg")
        canonical = {
            "path": "/controller/results/run_1/iter_2/datagen/endpoint_pool.json",
            "sha256": "a" * 64, "request_sha256": "b" * 64,
            "required_capacity": 24,
        }
        remote = dict(
            canonical,
            path="/lustre/workspace/results/run_1/iter_2/datagen/endpoint_pool.json",
        )
        self.assertTrue(module._endpoint_pool_binding_matches(remote, canonical, "slurm"))
        self.assertFalse(module._endpoint_pool_binding_matches(remote, canonical, "docker"))
        mapped = module._synced_remote_local_path(
            "/lustre/workspace/results/run_1/iter_2/datagen/logs/sdg.log",
            pathlib.Path("/controller/results/run_1"),
        )
        self.assertEqual(
            mapped,
            pathlib.Path("/controller/results/run_1/iter_2/datagen/logs/sdg.log"),
        )

    def test_synthetic_prepare_validate_label_normalize_and_resume(self):
        import argparse
        import pandas as pd

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cfg = config("external")
            cfg["generation"]["max_samples_per_iteration"] = 1
            cfg["generation"]["scale_factor"] = 1.0
            dataset = root / "source"
            (dataset / "images").mkdir(parents=True)
            (dataset / "images" / "source.jpg").write_bytes(b"source")
            mined = root / "mined.json"
            mined.write_text(json.dumps([{
                "unique_name": "source.jpg", "image_path": "images/source.jpg",
                "person_key": "person1", "image_attr_values": [0] * 7,
            }]))
            gaps = root / "gaps.parquet"
            pd.DataFrame({"image_attr_vector": [[0] * 7]}).to_parquet(gaps)
            vocab_path = root / "vocab.json"
            vocab(vocab_path)
            eval_list = root / "eval.txt"
            eval_list.write_text("eval.jpg\n")
            output = root / "datagen"
            args = argparse.Namespace(
                output_root=output, mined_pairs=mined, gaps_parquet=gaps,
                attribute_vocab=vocab_path, eval_list=eval_list, eval_pairs=None,
                dataset_root=dataset,
            )
            stage_runner.prepare_inputs(args, cfg)
            stage_runner.prepare_inputs(args, cfg)  # committed host operation is reused
            attempt = output / "augmentation" / "person1" / "attempt_1"
            metadata(attempt / "output_metadata.json", True)
            (attempt / "output.jpg").write_bytes(b"generated")
            args.augmentation_root = output / "augmentation"
            stage_runner.validate_augmentation(args, cfg)
            args.accepted_manifest = output / "accepted_manifest.json"
            args.labels_root = output / "labels"
            qa = args.labels_root / "person1" / "task" / "open_qa.json"
            qa.parent.mkdir(parents=True)
            qa.write_text(json.dumps({
                "version": "1.0", "metadata": {"task_type": "open_qa"},
                "items": [
                    {"video_id": "person1", "question": f"{level}-{index}", "answer": f"{level} caption {index}"}
                    for level in QUERY_LEVELS for index in range(3)
                ],
            }))
            stage_runner.validate_labels(args, cfg)
            stage_runner.normalize(args, cfg)
            manifest = json.loads((output / "dataset" / "sdg_manifest.json").read_text())
            self.assertEqual(manifest["num_pairs"], 9)
            self.assertEqual(manifest["rejected_samples_included"], 0)

    def test_prepare_joins_gap_vectors_from_bound_eval_pairs(self):
        import argparse
        import pandas as pd

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            cfg = config("external")
            cfg["generation"]["max_samples_per_iteration"] = 1
            dataset = root / "source"
            (dataset / "images").mkdir(parents=True)
            (dataset / "images" / "source.jpg").write_bytes(b"source")
            mined = root / "mined.json"
            mined.write_text(json.dumps([{
                "unique_name": "source.jpg", "image_path": "images/source.jpg",
                "person_key": "person1", "image_attr_values": [0] * 7,
            }]))
            gaps = root / "gaps.parquet"
            pd.DataFrame({"unique_name": ["eval.jpg"]}).to_parquet(gaps)
            eval_pairs = root / "eval_pairs.json"
            eval_pairs.write_text(json.dumps([{
                "unique_name": "eval.jpg", "image_attr_values": [0] * 7,
            }]))
            vocab_path = root / "vocab.json"
            vocab(vocab_path)
            eval_list = root / "eval.txt"
            eval_list.write_text("eval.jpg\n")
            output = root / "datagen"
            args = argparse.Namespace(
                output_root=output, mined_pairs=mined, gaps_parquet=gaps,
                attribute_vocab=vocab_path, eval_list=eval_list,
                eval_pairs=eval_pairs, dataset_root=dataset,
            )

            stage_runner.prepare_inputs(args, cfg)

            plan = json.loads((output / "sdg_plan.json").read_text())
            self.assertEqual(plan["residual_distribution"]["weak_rows"], 1)

    def test_canonical_transition_places_sdg_before_training(self):
        for script_name in ("commit_stage.py", "audit_deft_run.py"):
            spec = importlib.util.spec_from_file_location("checked_" + script_name[:-3], ROOT / "scripts" / script_name)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            state = {"max_iterations": 2, "iterations": {"iter1": {}}}
            self.assertEqual(
                module._expected_next({"iteration": "iter1", "stage": "history_select", "status": "ok"}, state),
                {("iter1", "sdg")},
            )
            self.assertEqual(
                module._expected_next({"iteration": "iter1", "stage": "sdg", "status": "ok"}, state),
                {("iter1", "visualize")},
            )

    def test_residual_distribution_is_deterministic_and_subtracts_mined(self):
        attributes = ["top outer color", "top outer type"]
        vocab_payload = {
            "attributes": attributes,
            "id_to_value": {
                "top outer color": ["missing", "red", "blue"],
                "top outer type": ["missing", "shirt", "jacket"],
            },
        }
        weak = [[1, 1], [1, 1], [2, 2]]
        mined = [[1, 1]]
        first, evidence = residual_attribute_assignments(weak, mined, vocab_payload, 3, 1.0)
        second, _ = residual_attribute_assignments(weak, mined, vocab_payload, 3, 1.0)
        self.assertEqual(first, second)
        self.assertEqual(evidence["weak_rows"], 3)
        self.assertTrue(any(item.get("top outer color") == "blue" for item in first))

    def test_residual_assignments_only_request_component_verified_attributes(self):
        attributes = ["top outer color", "shoe color", "viewpoint"]
        vocab_payload = {
            "attributes": attributes,
            "id_to_value": {
                "top outer color": ["red"], "shoe color": ["black"],
                "viewpoint": ["front"],
            },
        }
        assignments, _ = residual_attribute_assignments(
            [[0, 0, 0]], [[0, 0, 0]], vocab_payload, 1, 1.0,
        )
        self.assertEqual(assignments, [{"top outer color": "red"}])

    def test_acceptance_and_bounded_rejection(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            metadata(root / "p1" / "attempt_1" / "output_metadata.json", False)
            (root / "p1" / "attempt_1" / "output.jpg").write_bytes(b"x")
            metadata(root / "p1" / "attempt_2" / "output_metadata.json", True)
            (root / "p1" / "attempt_2" / "output.jpg").write_bytes(b"x")
            accepted, rejected = accepted_augmentations(root, 2)
            self.assertEqual((len(accepted), len(rejected)), (1, 1))
            metadata(root / "p2" / "attempt_3" / "output_metadata.json", False)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                accepted_augmentations(root, 2)

    def test_normalization_excludes_rejected_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            source = root / "accepted.jpg"
            source.write_bytes(b"jpeg")
            meta = root / "accepted.json"
            metadata(meta, True)
            labels = root / "labels" / "person1" / "task"
            labels.mkdir(parents=True)
            labels.joinpath("open_qa.json").write_text(json.dumps({
                level: [f"{level} caption {idx}" for idx in range(3)]
                for level in ("easy", "medium", "hard")
            }))
            accepted = root / "accepted_manifest.json"
            accepted.write_text(json.dumps({"accepted": [{
                "source_key": "person1", "source_unique_name": "source.jpg", "attempt": 2,
                "image": str(source), "metadata": str(meta), "metadata_sha256": "a" * 64,
                "source_attribute_values": [0] * 7,
                "target_attributes": {
                    "top outer color": "red", "top outer type": "jacket",
                    "bottom color": "blue", "bottom type": "pants",
                },
            }]}))
            vocab_path = root / "vocab.json"
            vocab(vocab_path)
            output = root / "dataset"
            result = normalize_generated_pairs(accepted, root / "labels", output, vocab_path, set())
            payload = json.loads(result.read_text())
            pairs = json.loads((output / "sdg_pairs.json").read_text())
            self.assertEqual(payload["num_pairs"], 9)
            self.assertEqual(len(pairs), 9)
            self.assertTrue(all(row["source_unique_name"] == "source.jpg" for row in pairs))
            self.assertEqual(
                validate_normalized_dataset(
                    result, output / "sdg_pairs.json", output / "sdg_image_list.txt"
                ),
                {"pairs": 9, "sources": 1, "images": 9},
            )
            self.assertEqual(normalize_generated_pairs(accepted, root / "labels", output, vocab_path, set()), result)
            (output / "images" / pairs[0]["unique_name"]).unlink()
            with self.assertRaisesRegex(ValueError, "file set"):
                validate_normalized_dataset(
                    result, output / "sdg_pairs.json", output / "sdg_image_list.txt"
                )
            with self.assertRaisesRegex(ValueError, "evaluation"):
                normalize_generated_pairs(accepted, root / "labels", root / "blocked", vocab_path, {"accepted.jpg"})

    def test_rejected_manifest_cannot_normalize(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            source = root / "rejected.jpg"
            source.write_bytes(b"jpeg")
            meta = root / "rejected.json"
            metadata(meta, False)
            accepted = root / "accepted.json"
            accepted.write_text(json.dumps({"accepted": [{
                "source_key": "p", "attempt": 1, "image": str(source),
                "metadata": str(meta), "metadata_sha256": "b" * 64,
                "source_attribute_values": [0] * 7,
                "target_attributes": {"top outer color": "red"},
            }]}))
            vocab_path = root / "vocab.json"
            vocab(vocab_path)
            with self.assertRaisesRegex(ValueError, "rejected"):
                normalize_generated_pairs(accepted, root, root / "out", vocab_path, set())


if __name__ == "__main__":
    unittest.main()
