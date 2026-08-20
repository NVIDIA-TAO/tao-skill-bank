# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import pathlib
import socket
import sys
import tempfile
import unittest
from unittest import mock

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from iaa_deft.sdg import (  # noqa: E402
    QUERY_LEVELS,
    accepted_augmentations,
    build_component_command,
    build_endpoint_command,
    container_name,
    normalize_generated_pairs,
    port_available,
    readiness_probe,
    residual_attribute_assignments,
    validate_config,
    validate_gpu_inventory,
    wait_until_ready,
)
import manage_sdg_endpoints as endpoint_manager  # noqa: E402
import run_sdg_stage as stage_runner  # noqa: E402


def config(ownership: str = "managed") -> dict:
    payload = yaml.safe_load((ROOT / "specs" / "sdg_config.yaml").read_text())
    payload["endpoints"]["ownership"] = ownership
    if ownership == "managed":
        payload["endpoints"]["gpu_ids"] = {"image_edit": [2], "vlm": [4], "llm": [6]}
        payload["endpoints"]["external_urls"] = {role: "" for role in ("image_edit", "vlm", "llm")}
    else:
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
        "attribute_verification": {"passed": passed},
        "selections": {
            "top outer color": "red", "top outer type": "jacket",
            "bottom color": "blue", "bottom type": "pants",
            "shoe color": "black", "shoe type": "sneakers",
            "viewpoint": "front", "accessories": [],
        },
    }))


class EndpointContractTests(unittest.TestCase):
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
        for role, expected in (("image_edit", "device=2"), ("vlm", "device=4"), ("llm", "device=6")):
            argv = build_endpoint_command(cfg, role, "run_42", pathlib.Path("/tmp/hf"))
            self.assertIn(expected, argv)
            self.assertNotIn("all", argv)
            self.assertNotIn(":latest", " ".join(argv))
            self.assertIn(cfg["models"][role]["revision"], argv)
        self.assertEqual(container_name("run_42", "vlm"), "tao-deft-iaa-run-42-vlm")

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
        self.assertIn("/tmp/in:/app/data/in:ro", argv)
        self.assertIn("/tmp/out:/app/data/out", argv)
        self.assertIn("/app/data/out/panes/p1.jpg", " ".join(argv))
        self.assertIn("--no-sync", argv)
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
    def test_generation_frame_is_distinct_from_tao_platform(self):
        skill = (ROOT / "SKILL.md").read_text()
        reference = (ROOT / "references" / "local-sdg.md").read_text()

        self.assertIn(
            "selected TAO platform and the generation execution frame are distinct",
            skill,
        )
        self.assertIn("control host's Docker daemon", skill)
        self.assertIn(
            "Do not silently point the Docker CLI at a remote daemon",
            reference,
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
                attribute_vocab=vocab_path, eval_list=eval_list, dataset_root=dataset,
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
            self.assertEqual(normalize_generated_pairs(accepted, root / "labels", output, vocab_path, set()), result)
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
            }]}))
            vocab_path = root / "vocab.json"
            vocab(vocab_path)
            with self.assertRaisesRegex(ValueError, "rejected"):
                normalize_generated_pairs(accepted, root, root / "out", vocab_path, set())


if __name__ == "__main__":
    unittest.main()
