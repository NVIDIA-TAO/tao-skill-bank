from __future__ import annotations

import importlib.util
import argparse
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "kubernetes_sdg_action.py"
SPEC = importlib.util.spec_from_file_location("kubernetes_sdg_action", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def unsigned_request(nodes: int = 3) -> dict:
    stage = "/mnt/tao/results/iter_1/datagen"
    digest = "a" * 64
    models = {
        "image_edit": {"id": "Qwen/Qwen-Image-Edit-2511", "revision": "1" * 40,
                       "backend": "vllm-omni", "port": 8002},
        "vlm": {"id": "Qwen/Qwen3-VL", "revision": "2" * 40,
                "backend": "vllm", "port": 8000},
        "llm": {"id": "Qwen/Qwen2.5", "revision": "3" * 40,
                "backend": "vllm", "port": 8001},
    }
    return {
        "schema_version": "1", "workflow": "tao-run-deft-iaa",
        "kind": "kubernetes_sdg_action", "platform": "kubernetes", "name": "sdg_execute",
        "action_id": "action-1", "run_id": "run-1", "iteration": 1, "attempt": 1,
        "started_at": "2026-08-19T12:00:00Z", "started_ns": 123456,
        "generation_nodes": nodes, "namespace": "tao", "pvc_claim": "tao-results",
        "pvc_mount": "/mnt/tao", "service_account": "tao-sdg",
        "images": {key: f"nvcr.io/nvstaging/tao/{key}:1@sha256:{digest}"
                   for key in ("augmentation", "auto_labeling", "image_edit", "text_serving", "controller")},
        "component_sources": {
            key: f"nvcr.io/nvstaging/tao/{key}:1@sha256:{digest}"
            for key in ("augmentation", "auto_labeling", "image_edit", "text_serving")
        },
        "bindings": {"state_sha256": "b" * 64, "config_sha256": "c" * 64,
                     "runtime_sha256": "d" * 64},
        "models": models,
        "paths": {
            "results_dir": "/mnt/tao/results", "stage_dir": stage,
            "dataset_root": "/mnt/tao/dataset", "config_path": "/mnt/tao/results/config/sdg_config.yaml",
            "runtime_root": "/mnt/tao/runtime", "cache_dir": "/mnt/tao/cache",
            "mined_pairs": "/mnt/tao/results/iter_1/mining/mined_pairs.json",
            "eval_list": "/mnt/tao/results/iaa_splits/eval_list.txt",
            "attribute_vocab": "/mnt/tao/results/config/attribute_vocab.json",
        },
        "limits": {"startup_timeout_s": 1800, "retry_interval_s": 10,
                   "request_timeout_s": 600, "component_max_attempts": 2,
                   "ttl_seconds": 3600},
        "forward_env": [],
        "expected_outputs": [
            f"{stage}/dataset/sdg_manifest.json", f"{stage}/dataset/sdg_pairs.json",
            f"{stage}/dataset/sdg_image_list.txt", f"{stage}/sdg_execution_manifest.json",
            f"{stage}/endpoint_pool.json", f"{stage}/endpoint_manifest.json",
        ],
    }


class KubernetesSdgContractTests(unittest.TestCase):
    def request(self, nodes: int = 3) -> dict:
        return MODULE.sign_request(unsigned_request(nodes))

    def test_signed_request_is_deterministic_and_strict(self):
        first = self.request()
        second = self.request()
        self.assertEqual(first, second)
        self.assertRegex(first["request_sha256"], r"^[0-9a-f]{64}$")
        invalid = dict(first, generation_nodes=0)
        invalid["request_sha256"] = MODULE._canonical_sha256(invalid)
        with self.assertRaisesRegex(ValueError, "generation_nodes"):
            MODULE.validate_request(invalid)
        extra = dict(first, secret="forbidden")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            MODULE.validate_request(extra)

    def test_prepare_derives_from_committed_state_and_reuses_identical_output(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = pathlib.Path(root_text).resolve()
            results = root / "run-1"
            config_dir = results / "config"
            config_dir.mkdir(parents=True)
            dataset = root / "data" / "iaa"
            dataset.mkdir(parents=True)
            (dataset / "attribute_vocab.json").write_text("{}\n")
            (results / "iaa_splits").mkdir(parents=True)
            (results / "iaa_splits" / "eval_list.txt").write_text("eval.jpg\n")
            stage = results / "iter_1" / "datagen"
            mined = results / "iter_1" / "mining" / "mined_pairs.json"
            mined.parent.mkdir(parents=True)
            mined.write_text("[]\n")
            digest = "a" * 64
            images = {
                "augmentation": f"registry/augmentation:1@sha256:{digest}",
                "auto_labeling": f"registry/auto:1@sha256:{digest}",
                "image_edit_serving": f"registry/image:1@sha256:{digest}",
                "text_serving": f"registry/text:1@sha256:{digest}",
            }
            models = {
                "image_edit": {"id": "image", "revision": "1" * 40,
                               "backend": "vllm-omni", "port": 8002},
                "vlm": {"id": "vlm", "revision": "2" * 40,
                        "backend": "vllm", "port": 8000},
                "llm": {"id": "llm", "revision": "3" * 40,
                        "backend": "vllm", "port": 8001},
            }
            gpu_ids = {"image_edit": list(range(8)), "vlm": [0], "llm": [1]}
            sdg = {
                "schema_version": "1", "enabled": True, "images": images, "models": models,
                "endpoints": {"ownership": "managed", "reuse_requested": False,
                              "startup_timeout_s": 1800, "request_timeout_s": 180,
                              "retry_interval_s": 15, "cache_dir": str(root / "cache"),
                              "gpu_ids": gpu_ids},
                "generation": {"generation_nodes": 3, "gpus_per_generation_node": 8},
            }
            config_path = config_dir / "sdg_config.yaml"
            config_path.write_text(MODULE.yaml.safe_dump(sdg, sort_keys=True))
            config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
            approved = {
                "endpoint_mode": "managed", "reuse_requested": False,
                "generation_nodes": 3, "gpus_per_generation_node": 8,
                "gpu_ids": gpu_ids, "models": models, "images": images,
            }
            state = {
                "schema_version": "3", "workflow": MODULE.WORKFLOW,
                "started_at": "2026-08-19T12:00:00+00:00", "results_dir": str(results),
                "max_iterations": 3, "current_iteration": 1, "gate_met": False,
                "config": {"platform": "kubernetes", "dataset_root": str(dataset),
                           "sdg_config": str(config_path), "sdg_config_sha256": config_digest,
                           "spec_sha256": {"sdg_config.yaml": config_digest},
                           "iaa_deft_bundle_sha256": "d" * 64,
                           "requires_hf_token": False, "sdg": approved},
                "iterations": {"iter1": {"status": "in_progress",
                                           "stage_completed": "history_select",
                                           "mined_pairs": str(mined)}},
            }
            state_path = results / "deft_state.json"
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            args = argparse.Namespace(
                deft_state=state_path, sdg_config=config_path, iteration=1,
                namespace="tao", pvc_claim="tao-results", pvc_mount=str(root),
                service_account="tao-sdg", runtime_root=stage / ".tao-runtime" / "controller",
                augmentation_image=images["augmentation"],
                auto_labeling_image=images["auto_labeling"],
                image_edit_image=images["image_edit_serving"],
                text_serving_image=images["text_serving"],
                controller_image=f"registry/controller:1@sha256:{digest}",
                ttl_seconds=3600, output=stage / "kubernetes_request.json",
            )
            first = MODULE.prepare_request(args)
            original_bytes = args.output.read_bytes()
            second = MODULE.prepare_request(args)
            self.assertEqual(first["status"], "created")
            self.assertEqual(second["status"], "reused")
            self.assertEqual(args.output.read_bytes(), original_bytes)
            request = first["request"]
            self.assertEqual(request["generation_nodes"], 3)
            self.assertEqual(request["bindings"]["config_sha256"], config_digest)
            self.assertEqual(request["component_sources"]["image_edit"], images["image_edit_serving"])
            self.assertEqual(request["paths"]["runtime_root"], str(args.runtime_root))

            mismatched = argparse.Namespace(**vars(args))
            mismatched.augmentation_image = f"registry/other:1@sha256:{digest}"
            with self.assertRaisesRegex(ValueError, "differs from immutable"):
                MODULE.prepare_request(mismatched)

            state["iterations"]["iter1"]["stage_completed"] = "sdg"
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "already committed"):
                MODULE.prepare_request(args)

    def test_render_is_native_sdg_topology_not_distributed_training(self):
        request = self.request(3)
        rendered = MODULE.render_resources(request)
        items = rendered["items"]
        worker = next(item for item in items if item["kind"] == "Job" and
                      item["metadata"]["labels"]["tao.nvidia.com/role"] == "image-worker")
        coordinator = next(item for item in items if item["kind"] == "Job" and
                           item["metadata"]["labels"]["tao.nvidia.com/role"] == "coordinator")
        service = next(item for item in items if item["kind"] == "Service" and
                       item["metadata"]["labels"]["tao.nvidia.com/role"] == "image-worker")
        self.assertEqual(worker["spec"]["completionMode"], "Indexed")
        self.assertEqual(worker["spec"]["parallelism"], 3)
        containers = worker["spec"]["template"]["spec"]["containers"]
        self.assertEqual(len(containers), 8)
        self.assertEqual(
            [item["resources"]["limits"]["nvidia.com/gpu"] for item in containers], [1] * 8
        )
        self.assertEqual(
            [item["args"][item["args"].index("--tensor-parallel-size") + 1] for item in containers],
            ["1"] * 8,
        )
        first = containers[0]
        self.assertEqual(first["command"], ["vllm", "serve", request["models"]["image_edit"]["id"]])
        self.assertEqual(first["args"][:5], ["--omni", "--host", "0.0.0.0", "--port", "8002"])
        self.assertEqual([row["port"] for row in service["spec"]["ports"]], list(range(8002, 8010)))
        urls = MODULE.expected_endpoint_urls(
            request, rendered["_tao"]["worker_job"], rendered["_tao"]["worker_job"]
        )
        self.assertEqual(len(urls), 24)
        self.assertEqual(len({row[2] for row in urls}), 24)
        self.assertEqual({row[0] for row in urls}, {0, 1, 2})
        sidecars = coordinator["spec"]["template"]["spec"]["initContainers"]
        self.assertEqual(len(sidecars), 2)
        self.assertTrue(all(item["restartPolicy"] == "Always" for item in sidecars))
        self.assertEqual(sum(item["resources"]["limits"]["nvidia.com/gpu"] for item in sidecars), 2)
        text = json.dumps(rendered)
        self.assertNotIn("WORLD_SIZE", text)
        self.assertNotIn("torchrun", text)
        self.assertNotIn("--gpus all", text)

    def test_secret_material_is_stdin_only_and_never_in_report_or_argv(self):
        request = self.request(1)
        calls = []

        def fake_json(args, stdin=None):
            calls.append((args, stdin))
            if args[:2] == ["get", "job,service,secret"]:
                return {"items": []}
            if stdin and json.loads(stdin).get("kind") == "Secret":
                return {"metadata": {"uid": "secret-uid"}}
            return {"items": [{"metadata": {"uid": "object-uid"}}]}

        with (
            mock.patch.object(MODULE, "_pvc_and_capacity_preflight"),
            mock.patch.object(MODULE, "_kubectl_json", side_effect=fake_json),
        ):
            report = MODULE.submit(request)
        secret_calls = [(argv, body) for argv, body in calls if body and json.loads(body).get("kind") == "Secret"]
        self.assertEqual(len(secret_calls), 1)
        argv, body = secret_calls[0]
        self.assertNotIn("VLLM_API_KEY", " ".join(argv))
        secret_values = set(json.loads(body)["stringData"].values())
        self.assertTrue(secret_values)
        report_text = json.dumps(report)
        rendered_text = json.dumps(MODULE.render_resources(request))
        self.assertTrue(all(value not in report_text and value not in rendered_text for value in secret_values))

    def test_preflight_requires_bound_rwx_and_full_gpu_shape(self):
        request = self.request(2)
        responses = iter([
            {"metadata": {"name": "tao-sdg"}},
            {"status": {"phase": "Bound", "accessModes": ["ReadWriteMany"]}},
            {"items": [
                {"status": {"allocatable": {"nvidia.com/gpu": "8"}}},
                {"status": {"allocatable": {"nvidia.com/gpu": "8"}}},
                {"status": {"allocatable": {"nvidia.com/gpu": "2"}}},
            ]},
        ])

        def fake_json(args, stdin=None):
            if args[:2] == ["version", "-o"]:
                return {"serverVersion": {"major": "1", "minor": "30"}}
            return next(responses)

        with (
            mock.patch.object(MODULE, "_kubectl_json", side_effect=fake_json),
            mock.patch.object(MODULE, "_run", return_value=mock.Mock(stdout="yes\n")),
        ):
            MODULE._pvc_and_capacity_preflight(request)

        responses = iter([
            {"metadata": {}}, {"status": {"phase": "Bound", "accessModes": ["ReadWriteOnce"]}},
        ])
        with (
            mock.patch.object(MODULE, "_kubectl_json", side_effect=lambda args, stdin=None:
                {"serverVersion": {"major": "1", "minor": "30"}} if args[:2] == ["version", "-o"] else next(responses)),
            mock.patch.object(MODULE, "_run", return_value=mock.Mock(stdout="yes\n")),
            self.assertRaisesRegex(ValueError, "ReadWriteMany"),
        ):
            MODULE._pvc_and_capacity_preflight(request)

    def test_cancel_refuses_mixed_ownership(self):
        request = self.request(1)
        foreign = {"metadata": {"uid": "foreign", "labels": {}}}
        with (
            mock.patch.object(MODULE, "_kubectl_json", return_value={"items": [foreign]}),
            self.assertRaisesRegex(ValueError, "foreign"),
        ):
            MODULE.cancel(request, True)

    def test_submit_refuses_partial_owned_resume(self):
        request = self.request(1)
        partial = {"kind": "Job", "metadata": {
            "name": MODULE.render_resources(request)["_tao"]["worker_job"], "uid": "uid-1",
            "labels": MODULE._labels(request, "image-worker"),
        }}
        with (
            mock.patch.object(MODULE, "_pvc_and_capacity_preflight"),
            mock.patch.object(MODULE, "_kubectl_json", return_value={"items": [partial]}),
            self.assertRaisesRegex(ValueError, "partial"),
        ):
            MODULE.submit(request)

    def test_readiness_probes_models_and_minimal_inference(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(self.payload).encode()

        for role, inference in (("image_edit", {"data": [{"b64_json": "x"}]}),
                                ("llm", {"choices": [{"message": {"content": "READY"}}]})):
            requests = []

            def fake_open(request, timeout):
                requests.append(request)
                return Response({"data": [{"id": "model"}]}) if len(requests) == 1 else Response(inference)

            with (
                mock.patch.dict(MODULE.os.environ, {"IMAGE_EDIT_API_KEY": "secret"}),
                mock.patch.object(MODULE.urllib.request, "urlopen", side_effect=fake_open),
            ):
                MODULE._probe_model("http://endpoint/v1", "model", role, 10)
            self.assertEqual(len(requests), 2)
            self.assertTrue(requests[0].full_url.endswith("/models"))
            self.assertTrue(
                requests[1].full_url.endswith("/images/edits" if role == "image_edit" else "/chat/completions")
            )

    def test_status_never_reports_partial_pool_complete(self):
        request = self.request(1)
        job = {"metadata": {"labels": MODULE._labels(request, "coordinator")},
               "status": {"succeeded": 1}}
        with (
            mock.patch.object(MODULE, "_kubectl_json", return_value={"items": [job]}),
            mock.patch.object(pathlib.Path, "is_file", return_value=False),
        ):
            self.assertNotEqual(MODULE.status(request)["state"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
