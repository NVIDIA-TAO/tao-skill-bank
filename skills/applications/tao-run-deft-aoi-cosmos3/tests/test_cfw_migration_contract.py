# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
IMAGE = "nvcr.io/example/cosmos-framework@sha256:" + "a" * 64


class CosmosFrameworkSurfaceSmokeTests(unittest.TestCase):
    def _run(self, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [sys.executable, str(SCRIPTS / script), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed

    def test_train_surface_emits_framework_dcp_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = root / "train.toml"
            rendered = root / "train.json"
            self._run(
                "render_cfw_sft.py",
                "--profile", "smoke",
                "--model-path", "/models/qwen3-vl",
                "--train-jsonl", "/data/mining.jsonl",
                "--media-root", "/data",
                "--index-path", "/results/index/mining.u64",
                "--expected-rows", "1",
                "--expected-sha256", "b" * 64,
                "--expected-image-items", "1",
                "--run-name", "smoke",
                "--results-dir", "/results/train",
                "--num-gpus", "1",
                "--output", str(config),
                "--descriptor-output", str(rendered),
            )
            request = root / "request.json"
            output = root / "plan.json"
            descriptor = json.loads(rendered.read_text(encoding="utf-8"))
            request.write_text(
                json.dumps(
                    {
                        "action": "train",
                        "parameters": {
                            "image": IMAGE,
                            "config_path": str(config),
                            "results_dir": "/results/train",
                            "adapter_root": "/inputs/nvpaw_cfw",
                            "model_path": "/models/qwen3-vl",
                            "train_jsonl": "/data/mining.jsonl",
                            "media_root": "/data",
                            "index_path": "/results/index/mining.u64",
                            "hydra_overrides": descriptor["hydra_overrides"],
                            "num_gpus": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self._run("cfw_action_plan.py", "--request", str(request), "--output", str(output))
            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(plan["backend"], "cosmos-framework")
            self.assertEqual(plan["outputs"]["checkpoint_format"], "framework_dcp")

    def test_evaluate_surface_launches_jsonl_validation_and_declares_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            model = root / "model"
            model.mkdir()
            media = root / "image.png"
            media.write_bytes(b"image")
            source = root / "benchmark.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "task_type": "Component Classification",
                        "messages": [
                            {"role": "user", "content": [
                                {"type": "image", "image": media.name, "min_pixels": 1, "max_pixels": 1},
                                {"type": "text", "text": "inspect"},
                            ]},
                            {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
                        ],
                    }
                ) + "\n",
                encoding="utf-8",
            )
            config = root / "evaluate.toml"
            output = root / "predictions.jsonl"
            self._run(
                "render_cfw_evaluate.py",
                "--annotation-path", str(source),
                "--media-root", str(root),
                "--model-path", str(model),
                "--results-dir", str(root / "results"),
                "--output", str(config),
            )
            completed = self._run(
                "cfw_jsonl_runtime.py",
                "--action", "evaluate",
                "--config", str(config),
                "--model-path", str(model),
                "--output-jsonl", str(output),
                "--validate-only",
            )
            report = json.loads(completed.stdout)
            self.assertEqual(report["backend"], "cosmos-framework")
            self.assertEqual(report["selected_rows"], 1)

    def test_inference_surface_launches_same_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            model = root / "model"
            model.mkdir()
            media = root / "image.png"
            media.write_bytes(b"image")
            completed = self._run(
                "cfw_jsonl_runtime.py",
                "--action", "inference",
                "--model-path", str(model),
                "--output-jsonl", str(root / "predictions.jsonl"),
                "--media", str(media),
                "--prompt", "inspect",
                "--validate-only",
            )
            report = json.loads(completed.stdout)
            self.assertEqual(report["action"], "inference")
            self.assertEqual(report["selected_rows"], 1)


if __name__ == "__main__":
    unittest.main()
