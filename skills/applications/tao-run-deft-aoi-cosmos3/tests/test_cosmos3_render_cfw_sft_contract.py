# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import sys
import tomllib
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import render_cfw_sft  # noqa: E402


class Cosmos3RenderCfwSftContractTests(unittest.TestCase):
    @staticmethod
    def _full(**overrides: object) -> dict:
        values: dict[str, object] = {
            "model_path": "/models/qwen3-vl",
            "train_jsonl": "/data/train.jsonl",
            "media_root": "/data",
            "index_path": "/results/index/train.u64",
            "expected_rows": 19_456,
            "expected_sha256": "a" * 64,
            "expected_image_items": 19_456,
            "run_name": "iter1",
            "results_dir": "/results/iter1/train",
            "num_gpus": 8,
            "epochs_per_iteration": 5,
            "micro_batch_per_rank": 8,
            "gradient_accumulation": 16,
        }
        values.update(overrides)
        return render_cfw_sft.build_profile("full", **values)

    def test_full_profile_uses_native_epochs_scaled_batch_lr_and_final_checkpoint(self) -> None:
        descriptor = self._full()
        config = descriptor["config"]

        self.assertEqual(config["trainer"]["num_epochs"], 5)
        self.assertEqual(config["trainer"]["steps_per_epoch"], 19)
        self.assertEqual(config["trainer"]["max_iter"], 95)
        self.assertEqual(config["checkpoint"]["save_freq_in_epoch"], 5)
        self.assertEqual(config["checkpoint"]["save_iter"], 95)
        self.assertEqual(config["scheduler"]["cycle_lengths"], [95])
        self.assertEqual(descriptor["data"]["global_batch"], 1024)
        self.assertEqual(config["optimizer"]["lr"], 2.0e-6)
        self.assertEqual(
            config["optimizer"]["lr_multipliers"],
            {"model.visual": 20.0},
        )
        self.assertIn(
            "dataloader_train.distributor.micro_batch_size=8",
            descriptor["hydra_overrides"],
        )

    def test_toml_omits_literal_multiplier_owned_by_python_adapter(self) -> None:
        descriptor = self._full()

        parsed = tomllib.loads(render_cfw_sft.dump_toml(descriptor["config"]))

        self.assertNotIn("lr_multipliers", parsed["optimizer"])
        self.assertEqual(
            descriptor["config"]["optimizer"]["lr_multipliers"],
            {"model.visual": 20.0},
        )

    def test_full_profile_rejects_global_batch_below_floor(self) -> None:
        with self.assertRaisesRegex(ValueError, "global batch must be at least 512"):
            self._full(
                expected_rows=16_384,
                micro_batch_per_rank=2,
                gradient_accumulation=16,
            )

    def test_full_profile_requires_exact_epoch_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of global batch"):
            self._full(expected_rows=20_000)


if __name__ == "__main__":
    unittest.main()
