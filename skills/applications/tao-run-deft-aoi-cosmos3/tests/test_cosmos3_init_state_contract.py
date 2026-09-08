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
    def test_final_and_best_is_the_default_benchmark_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            rc = init_deft_state.main(self._argv(root, workspace))

            self.assertEqual(rc, 0)
            state = json.loads((root / "results/deft_state.json").read_text())
            self.assertEqual(
                state["config"]["evaluation"]["benchmark_cadence"],
                "final_and_best",
            )

    def test_calibration_quotas_are_bound_to_proxy_rates_by_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            rc = init_deft_state.main(self._argv(root, workspace))

            self.assertEqual(rc, 0)
            contract = json.loads(
                (root / "results/deft_state.json").read_text()
            )["config"]["mining"]["calibration_quota_contract"]
            self.assertEqual(contract["policy"], "proxy_empty_rate_by_reference_cohort")
            self.assertEqual(
                contract["cohorts"]["non_reference_based"]["empty_rate"], 0.5
            )
            self.assertEqual(
                contract["cohorts"]["reference_based"]["empty_rate"], 0.5
            )
            self.assertEqual(
                contract["reference_empty_semantics"],
                "identical_or_no_change_pair_negative",
            )

    def test_near_duplicate_filter_can_be_explicitly_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            rc = init_deft_state.main(
                self._argv(root, workspace, "--no-near-duplicate-filter")
            )

            self.assertEqual(rc, 0)
            ablation = json.loads(
                (root / "results/deft_state.json").read_text()
            )["config"]["mining"]["defect_detection_ablation"]
            self.assertFalse(ablation["near_duplicate_filter_enabled"])
            self.assertIsNone(ablation["near_duplicate_hamming_distance"])

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
        def detection_row(record_id: str, task: str, boxes: list[dict]) -> dict:
            images = []
            if task.startswith("Ref_based"):
                images.append(
                    {
                        "type": "image",
                        "image": f"images/{record_id}-golden.png",
                        "min_pixels": 1,
                        "max_pixels": 1,
                    }
                )
            images.append(
                {
                    "type": "image",
                    "image": f"images/{record_id}.png",
                    "min_pixels": 1,
                    "max_pixels": 1,
                }
            )
            return {
                "id": record_id,
                "task_type": task,
                "messages": [
                    {
                        "role": "user",
                        "content": [*images, {"type": "text", "text": "detect"}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "```json\n" + json.dumps(boxes) + "\n```",
                            }
                        ],
                    },
                ],
            }

        one_box = [{"bbox_2d": [0, 0, 1, 1], "label": "x"}]
        proxy_rows = [
            detection_row("single-empty", "Defect Detection", []),
            detection_row("single-few", "Defect Detection", one_box),
            detection_row("reference-empty", "Ref_based Defect Detection", []),
            detection_row("reference-few", "Ref_based Defect Detection", one_box),
        ]
        (workspace / "annotations/proxy_kpi.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in proxy_rows), encoding="utf-8"
        )
        for filename in ("benchmark.jsonl", "mining.jsonl"):
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

    def test_operator_batch_lr_override_and_kpi_alias_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            rc = init_deft_state.main(
                self._argv(
                    root,
                    workspace,
                    "--micro-batch-per-rank", "8",
                    "--gradient-accumulation", "1",
                    "--learning-rate", "1e-6",
                    "--learning-rate-policy", "fixed",
                    "--minimum-global-batch", "0",
                    "--kpi-profile", "task_balanced_v1",
                    "--component-count-replay-per-iteration", "1000",
                )
            )
            self.assertEqual(rc, 0)
            state = json.loads((root / "results/deft_state.json").read_text())
            training = state["config"]["training"]
            self.assertEqual(training["global_batch"], 8)
            self.assertEqual(training["gradient_accumulation"], 1)
            self.assertEqual(training["minimum_global_batch"], 0)
            self.assertEqual(training["learning_rate_scaling"], "fixed")
            self.assertEqual(training["optimizer"]["learning_rate"], 1e-6)
            self.assertEqual(state["config"]["kpi"]["requested_profile"], "task_balanced_v1")
            self.assertEqual(state["config"]["kpi"]["profile"], "f1_cohort_balanced_v1")
            self.assertEqual(
                state["config"]["mining"]["component_count_replay_per_iteration"],
                1000,
            )

    def test_defect_detection_ablation_controls_are_launch_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            rc = init_deft_state.main(
                self._argv(
                    root,
                    workspace,
                    "--defect-detection-ablation",
                    "--defect-detection-anchor-policy", "all_proxy_severity",
                    "--defect-detection-top-k-per-target", "100",
                    "--defect-detection-minimum-fraction", "0.5",
                    "--minimum-training-rows-per-iteration", "3000",
                    "--near-duplicate-hamming-distance", "3",
                    "--augmentation-profile", "off",
                )
            )
            self.assertEqual(rc, 0)
            state = json.loads((root / "results/deft_state.json").read_text())
            mining = state["config"]["mining"]
            self.assertTrue(mining["defect_detection_ablation"]["enabled"])
            self.assertEqual(
                mining["defect_detection_ablation"]["minimum_materialized_fraction"],
                0.5,
            )
            self.assertEqual(
                mining["defect_detection_ablation"]["positive_evidence"],
                [
                    "proxy_false_negative",
                    "best_overlap_0_lt_iou_lte_0p5",
                ],
            )
            self.assertEqual(
                mining["defect_detection_ablation"]["anchor_policy"],
                "all_proxy_severity",
            )
            self.assertEqual(mining["top_k_per_task"], {"Defect Detection": 100})
            self.assertEqual(mining["minimum_training_rows_per_iteration"], 3000)
            self.assertEqual(state["config"]["training"]["augmentation_profile"], "off")

    def test_repetition_blend_controls_are_launch_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = self._workspace(root)
            rc = init_deft_state.main(
                self._argv(
                    root,
                    workspace,
                    "--repetition-blend",
                    "--repetition-policy", "explicit",
                    "--repetition-rep-min", "0.5",
                    "--repetition-rep-max", "3",
                    "--repetition-never-repeat-empty-gt",
                    "--repetition-explicit-multiplier", "Defect Detection=3",
                    "--repetition-seed", "29",
                    "--max-training-rows-per-iteration", "12000",
                )
            )

            self.assertEqual(rc, 0)
            state = json.loads((root / "results/deft_state.json").read_text())
            repetition = state["config"]["mining"]["repetition_blend"]
            self.assertEqual(
                repetition,
                {
                    "enabled": True,
                    "policy": "explicit",
                    "rep_min": 0.5,
                    "rep_max": 3.0,
                    "never_repeat_empty_gt": True,
                    "explicit_multipliers": {"Defect Detection": 3.0},
                    "row_cap": 12000,
                    "seed": 29,
                },
            )

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
