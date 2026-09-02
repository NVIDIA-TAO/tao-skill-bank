#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import init_deft_state  # noqa: E402


class Cosmos3InitStateContractTests(unittest.TestCase):
    @staticmethod
    def _workspace(root: pathlib.Path) -> pathlib.Path:
        workspace = root / "workspace"
        (workspace / "annotations").mkdir(parents=True)
        (workspace / "specs").mkdir()
        (workspace / "eval").mkdir()
        model = workspace / "models/Cosmos3-Nano-VLM"
        model.mkdir(parents=True)
        for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
            (model / name).write_text("{}\n", encoding="utf-8")
        (model / "config.json").write_text('{"model_type":"qwen3_vl"}\n', encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"weights")
        for filename in ("proxy_kpi.jsonl", "benchmark.jsonl", "mining.jsonl"):
            (workspace / "annotations" / filename).write_text("{}\n", encoding="utf-8")
        for filename in ("train_spec.toml", "evaluate_spec.toml"):
            (workspace / "specs" / filename).write_text("value = 1\n", encoding="utf-8")
        (workspace / "eval/calculate_f1_metrics.py").write_text("pass\n", encoding="utf-8")
        return workspace

    @staticmethod
    def _argv(root: pathlib.Path, workspace: pathlib.Path, *extra: str) -> list[str]:
        immutable = "example/image:1@sha256:" + "a" * 64
        return [
            "--results-dir", str(root / "results"),
            "--workspace", str(workspace),
            "--platform", "docker",
            "--max-iterations", "1",
            "--num-gpus", "1",
            "--num-nodes", "1",
            "--recipe-profile", "smoke",
            "--gpu-model", "NVIDIA H100 80GB",
            "--framework-container", immutable,
            "--mining-container", immutable,
            *extra,
        ]

    def test_local_model_is_recorded_in_framework_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            rc = init_deft_state.main(self._argv(root, workspace))
            self.assertEqual(rc, 0)
            state = json.loads((root / "results/deft_state.json").read_text())
            self.assertEqual(state["version"], 7)
            self.assertEqual(state["config"]["training"]["backend"], "cosmos-framework")
            self.assertEqual(
                state["config"]["training"]["annotation_source"],
                "mined_real_samples_only",
            )

    def test_requested_epoch_and_probed_batch_policy_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            rc = init_deft_state.main(
                self._argv(
                    root,
                    workspace,
                    "--epochs-per-iteration", "5",
                    "--micro-batch-per-rank", "8",
                    "--gradient-accumulation", "16",
                    "--max-training-rows-per-iteration", "20000",
                    "--mining-pool-fraction-cap", "0.5",
                )
            )
            self.assertEqual(rc, 0)
            training = json.loads(
                (root / "results/deft_state.json").read_text()
            )["config"]["training"]
            self.assertEqual(training["epochs_per_iteration"], 5)
            self.assertEqual(training["micro_batch_per_rank"], 8)
            self.assertEqual(training["gradient_accumulation"], 16)
            self.assertEqual(training["global_batch"], 128)
            self.assertEqual(training["optimizer"]["learning_rate"], 2.5e-7)
            self.assertEqual(training["learning_rate_scaling"], "linear_from_global_batch_512")
            mining = json.loads(
                (root / "results/deft_state.json").read_text()
            )["config"]["mining"]
            self.assertEqual(mining["pool_fraction_cap"], 0.5)
            self.assertEqual(mining["max_training_rows_per_iteration"], 20_000)
            self.assertEqual(mining["calibration_policy"], "empty_and_few_box_from_mining")

    def test_invalid_local_model_is_rejected_before_state_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            invalid = root / "invalid-model"
            invalid.mkdir()
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = init_deft_state.main(
                    self._argv(root, workspace, "--base-model", str(invalid))
                )
            self.assertEqual(rc, 2)
            self.assertFalse((root / "results/deft_state.json").exists())
            self.assertIn("config, tokenizer, and processor", stderr.getvalue())

    def test_venv_python_symlink_survives_state_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            venv_bin = root / "venv/bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python3").symlink_to(sys.executable)
            venv_python = venv_bin / "python"
            venv_python.symlink_to("python3")
            rc = init_deft_state.main(
                self._argv(root, workspace, "--python-executable", str(venv_python))
            )
            self.assertEqual(rc, 0)
            state = json.loads((root / "results/deft_state.json").read_text())
            self.assertEqual(state["execution_policy"]["python_executable"], str(venv_python))


if __name__ == "__main__":
    unittest.main()
