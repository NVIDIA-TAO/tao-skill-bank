#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Cosmos3 Omni baseline preparation (NVBug 6630303)."""

from __future__ import annotations

import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import prepare_cosmos_reason_evaluate as prepare_evaluate  # noqa: E402
import prepare_cosmos_reason_model as prepare_model  # noqa: E402
import prepare_cosmos_reason_train as prepare_train  # noqa: E402
import resume_position  # noqa: E402
from workflow_common import load_toml, prepared_baseline_checkpoint  # noqa: E402


RUNTIME_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-rl"
FRAMEWORK_IMAGE = "local/cosmos-framework:verified"
FRAMEWORK_DIGEST = "sha256:" + "a" * 64
ARCHITECTURE_REVISION = "b" * 40


def write_json(path: pathlib.Path, payload: object) -> pathlib.Path:
    """Write one JSON fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_qwen_checkpoint(path: pathlib.Path) -> pathlib.Path:
    """Write the smallest complete indexed Qwen3-VL checkpoint fixture."""
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "config.json", {"model_type": "qwen3_vl"})
    write_json(path / "tokenizer_config.json", {})
    write_json(path / "tokenizer.json", {})
    write_json(path / "preprocessor_config.json", {})
    write_json(
        path / "model.safetensors.index.json",
        {"weight_map": {"model.layer.weight": "model-00001-of-00001.safetensors"}},
    )
    (path / "model-00001-of-00001.safetensors").write_bytes(b"safetensors-fixture")
    return path


def write_workflow(path: pathlib.Path, baseline: pathlib.Path) -> pathlib.Path:
    """Write the workflow fields used by model and evaluation preparation."""
    payload = {
        "kpi_dataset": {
            "annotations_path": str(path.parent.parent / "data/kpi/annotations.json"),
            "media_dir": str(path.parent.parent / "data/kpi/media"),
        },
        "cosmos_reason": {
            "baseline_model_path": str(baseline),
            "base_evaluate_toml": str(path.parent / "base_evaluate.toml"),
            "continual_model": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def write_conversion_provenance(path: pathlib.Path, source: pathlib.Path) -> None:
    """Write provenance matching the current shared converter contract."""
    write_json(
        path / "tao_conversion_provenance.json",
        {
            "base_model": {"original": str(source), "resolved": str(source.resolve())},
            "architecture_model": {
                "original": prepare_model.DEFAULT_ARCHITECTURE_MODEL,
                "resolved": None,
            },
            "architecture_model_revision": ARCHITECTURE_REVISION,
            "framework_image": FRAMEWORK_IMAGE,
            "framework_image_digest": FRAMEWORK_DIGEST,
        },
    )


class ModelPreparationTests(unittest.TestCase):
    def test_reuses_complete_qwen_checkpoint_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary)
            baseline = write_qwen_checkpoint(workspace / "model/baseline")
            workflow = write_workflow(workspace / "specs/workflow.yaml", baseline)
            run_dir = workspace / "results/run"
            run_dir.mkdir(parents=True)
            runtime_result = {"image": RUNTIME_IMAGE, "status": "passed"}

            with mock.patch.object(
                prepare_model,
                "validate_with_runtime",
                return_value=runtime_result,
            ) as runtime_check:
                manifest = prepare_model.prepare_model(
                    workspace=workspace,
                    workflow_yaml=workflow,
                    run_dir=run_dir,
                    runtime_image=RUNTIME_IMAGE,
                    framework_image="",
                    framework_image_digest="",
                    model_preparation_script=pathlib.Path("/unused"),
                    architecture_model=prepare_model.DEFAULT_ARCHITECTURE_MODEL,
                    architecture_revision=prepare_model.DEFAULT_ARCHITECTURE_REVISION,
                )

            self.assertEqual(manifest["preparation"], "reused_qwen3_vl")
            self.assertEqual(manifest["prepared_model_path"], str(baseline))
            self.assertEqual(manifest["runtime_validation"], runtime_result)
            runtime_check.assert_called_once_with(baseline, RUNTIME_IMAGE)
            self.assertEqual(
                prepared_baseline_checkpoint(run_dir, workspace),
                baseline,
            )

    def test_converts_omni_with_current_interface_then_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary)
            source = workspace / "model/baseline"
            source.mkdir(parents=True)
            write_json(source / "config.json", {"model_type": "cosmos3_omni"})
            (source / "model.safetensors").write_bytes(b"omni-weights")
            workflow = write_workflow(workspace / "specs/workflow.yaml", source)
            run_dir = workspace / "results/run"
            run_dir.mkdir(parents=True)
            shared_helper = workspace / "shared_prepare.py"
            shared_helper.write_text("# fixture\n", encoding="utf-8")

            def complete_conversion(command: list[str], check: bool) -> None:
                self.assertTrue(check)
                output = pathlib.Path(command[command.index("--output-path") + 1])
                write_qwen_checkpoint(output)
                write_conversion_provenance(output, source)

            with mock.patch.object(
                prepare_model,
                "resolve_architecture_revision",
                return_value=ARCHITECTURE_REVISION,
            ), mock.patch.object(prepare_model, "verify_image_digest"), mock.patch.object(
                prepare_model,
                "validate_with_runtime",
                return_value={"image": RUNTIME_IMAGE, "status": "passed"},
            ), mock.patch.object(
                prepare_model.subprocess,
                "run",
                side_effect=complete_conversion,
            ) as conversion:
                first = prepare_model.prepare_model(
                    workspace=workspace,
                    workflow_yaml=workflow,
                    run_dir=run_dir,
                    runtime_image=RUNTIME_IMAGE,
                    framework_image=FRAMEWORK_IMAGE,
                    framework_image_digest=FRAMEWORK_DIGEST,
                    model_preparation_script=shared_helper,
                    architecture_model=prepare_model.DEFAULT_ARCHITECTURE_MODEL,
                    architecture_revision="main",
                )

            self.assertEqual(first["preparation"], "converted_cosmos3_omni")
            self.assertEqual(first["architecture_revision"], ARCHITECTURE_REVISION)
            command = conversion.call_args.args[0]
            self.assertIn("--base-model-path-or-uri", command)
            self.assertIn("--vlm-architecture-model-path-or-uri", command)
            self.assertIn("--vlm-architecture-model-revision", command)
            self.assertIn("--framework-image-digest", command)
            self.assertNotIn("--checkpoint-path", command)
            self.assertNotIn("--validate-with-image", command)
            output = pathlib.Path(command[command.index("--output-path") + 1])
            self.assertEqual(output.name, "prepared")
            self.assertEqual(output.parent.parent.parent, workspace / "model/prepared")

            with mock.patch.object(
                prepare_model,
                "resolve_architecture_revision",
                return_value=ARCHITECTURE_REVISION,
            ), mock.patch.object(
                prepare_model,
                "validate_with_runtime",
                return_value={"image": RUNTIME_IMAGE, "status": "passed"},
            ), mock.patch.object(prepare_model.subprocess, "run") as conversion:
                second = prepare_model.prepare_model(
                    workspace=workspace,
                    workflow_yaml=workflow,
                    run_dir=run_dir,
                    runtime_image=RUNTIME_IMAGE,
                    framework_image=FRAMEWORK_IMAGE,
                    framework_image_digest=FRAMEWORK_DIGEST,
                    model_preparation_script=shared_helper,
                    architecture_model=prepare_model.DEFAULT_ARCHITECTURE_MODEL,
                    architecture_revision="main",
                )

            self.assertEqual(second["preparation"], "reused_converted_cosmos3_omni")
            conversion.assert_not_called()

    def test_rejects_unknown_model_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary)
            source = workspace / "model/baseline"
            source.mkdir(parents=True)
            write_json(source / "config.json", {"model_type": "unknown"})
            (source / "model.safetensors").write_bytes(b"unknown-weights")
            workflow = write_workflow(workspace / "specs/workflow.yaml", source)
            run_dir = workspace / "results/run"
            run_dir.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "unsupported Cosmos Reason"):
                prepare_model.prepare_model(
                    workspace=workspace,
                    workflow_yaml=workflow,
                    run_dir=run_dir,
                    runtime_image=RUNTIME_IMAGE,
                    framework_image="",
                    framework_image_digest="",
                    model_preparation_script=pathlib.Path("/unused"),
                    architecture_model=prepare_model.DEFAULT_ARCHITECTURE_MODEL,
                    architecture_revision="main",
                )

    def test_resolves_mutable_hugging_face_revision_to_commit(self) -> None:
        response = io.BytesIO(json.dumps({"sha": ARCHITECTURE_REVISION}).encode("utf-8"))
        with mock.patch.object(prepare_model.urllib.request, "urlopen", return_value=response):
            resolved = prepare_model.resolve_architecture_revision(
                prepare_model.DEFAULT_ARCHITECTURE_MODEL,
                "main",
            )
        self.assertEqual(resolved, ARCHITECTURE_REVISION)

    def test_framework_digest_must_match_inspected_image(self) -> None:
        inspected = json.dumps([{"Id": FRAMEWORK_DIGEST, "RepoDigests": []}])
        completed = subprocess.CompletedProcess([], 0, stdout=inspected, stderr="")
        with mock.patch.object(prepare_model.shutil, "which", return_value="/usr/bin/docker"), \
                mock.patch.object(prepare_model.subprocess, "run", return_value=completed):
            prepare_model.verify_image_digest(FRAMEWORK_IMAGE, FRAMEWORK_DIGEST)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                prepare_model.verify_image_digest(
                    FRAMEWORK_IMAGE,
                    "sha256:" + "c" * 64,
                )


class WorkflowIntegrationTests(unittest.TestCase):
    def test_baseline_evaluate_and_noncontinual_train_use_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary)
            raw_omni = workspace / "model/baseline"
            raw_omni.mkdir(parents=True)
            write_json(raw_omni / "config.json", {"model_type": "cosmos3_omni"})
            prepared = write_qwen_checkpoint(workspace / "model/prepared/fixture/prepared")
            workflow = write_workflow(workspace / "specs/workflow.yaml", raw_omni)
            base_evaluate = workspace / "specs/base_evaluate.toml"
            base_evaluate.write_text(
                'results_dir = ""\n[dataset]\nannotation_path = ""\nmedia_dir = ""\n'
                '[model]\nmodel_name = ""\n',
                encoding="utf-8",
            )
            annotations = write_json(workspace / "data/kpi/annotations.json", [])
            media_dir = workspace / "data/kpi/media"
            media_dir.mkdir(parents=True)
            run_dir = workspace / "results/run"
            run_dir.mkdir(parents=True)
            write_json(
                run_dir / "baseline/model_preparation.json",
                {
                    "status": "ready",
                    "source_model_path": str(raw_omni),
                    "prepared_model_path": str(prepared),
                },
            )

            with mock.patch.object(
                sys,
                "argv",
                [
                    "prepare_cosmos_reason_evaluate.py",
                    "--workspace",
                    str(workspace),
                    "--workflow-yaml",
                    str(workflow),
                    "--run-dir",
                    str(run_dir),
                ],
            ):
                self.assertEqual(prepare_evaluate.main(), 0)
            evaluate = load_toml(run_dir / "baseline/evaluate/specs/evaluate.toml")
            self.assertEqual(evaluate["model"]["model_name"], str(prepared))
            self.assertEqual(evaluate["dataset"]["annotation_path"], str(annotations))
            self.assertEqual(evaluate["dataset"]["media_dir"], str(media_dir))

            config = yaml.safe_load(workflow.read_text(encoding="utf-8"))
            self.assertEqual(
                prepare_train.training_checkpoint(config, workspace, run_dir, 1),
                prepared,
            )
            self.assertEqual(
                prepare_train.training_checkpoint(config, workspace, run_dir, 2),
                prepared,
            )

    def test_model_preparation_is_first_resumable_stage(self) -> None:
        self.assertEqual(resume_position.INITIAL_STAGES[0], "prepare_cosmos_reason_model")


if __name__ == "__main__":
    unittest.main()
