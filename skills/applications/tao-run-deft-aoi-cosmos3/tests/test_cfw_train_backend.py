#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import render_cfw_evaluate  # noqa: E402
import render_cfw_sft  # noqa: E402
import submit_cfw_evaluate  # noqa: E402
import submit_cfw_inference  # noqa: E402
import submit_cfw_train  # noqa: E402


class FrameworkActionBackendTests(unittest.TestCase):
    def test_sft_profiles_use_direct_hf_then_direct_dcp_resume(self) -> None:
        for profile, expected_steps in (("smoke", 5), ("full", 500)):
            rendered = render_cfw_sft.render_profile(
                profile,
                model_path="/tao/model",
                annotation_path="/tao/data/train.json",
                media_root="/tao/media",
                run_name=f"iter1-{profile}",
            )
            parsed = render_cfw_sft.tomllib.loads(rendered)
            render_cfw_sft.validate_config(parsed, profile=profile)
            self.assertEqual(parsed["trainer"]["max_iter"], expected_steps)
            self.assertEqual(parsed["checkpoint"]["load_path"], "???")
            self.assertEqual(parsed["custom"]["images_per_record"], 1)
            self.assertNotIn("__TAO_CR3_", rendered)

        resumed = render_cfw_sft.tomllib.loads(
            render_cfw_sft.render_profile(
                "full",
                model_path="/tao/model",
                annotation_path="/tao/data/train.json",
                media_root="/tao/media",
                run_name="iter2-full",
                resume_checkpoint="/tao/iter1/train/checkpoints/iter_000000500",
            )
        )
        self.assertEqual(
            resumed["checkpoint"]["load_path"],
            "/tao/iter1/train/checkpoints/iter_000000500",
        )
        self.assertEqual(resumed["checkpoint"]["keys_to_skip_loading"], [])

        ten_epoch = render_cfw_sft.tomllib.loads(
            render_cfw_sft.render_profile(
                "full",
                model_path="/tao/model",
                annotation_path="/tao/data/train.json",
                media_root="/tao/media",
                run_name="iter1-full",
                num_epochs=10,
                train_records=37,
            )
        )
        self.assertEqual(ten_epoch["trainer"]["max_iter"], 370)
        self.assertEqual(ten_epoch["checkpoint"]["save_iter"], 37)
        self.assertEqual(ten_epoch["custom"]["num_epochs"], 10)

    def test_train_submit_invokes_native_entrypoint(self) -> None:
        config = render_cfw_sft.tomllib.loads(
            render_cfw_sft.render_profile(
                "smoke",
                model_path="/tao/model",
                annotation_path="/tao/data/train.json",
                media_root="/tao/media",
                run_name="iter1-smoke",
            )
        )
        command = submit_cfw_train.build_docker_argv(
            immutable_image="example/framework@sha256:" + "b" * 64,
            job_id="docker-train-test",
            config_host=pathlib.Path("/host/train.toml"),
            model_host=pathlib.Path("/host/model"),
            annotation_host=pathlib.Path("/host/train.json"),
            media_host=pathlib.Path("/host/media"),
            results_dir=pathlib.Path("/host/results"),
            adapter_host=pathlib.Path("/host/cfw_cr3_aoi_adapter.py"),
            config=config,
            gpus="all",
            identity_mounts=[],
            extra_mounts=[],
            resume_mount=None,
            offline=True,
        )
        self.assertIn("/workspace/.venv/bin/cosmos-framework-train", command)
        self.assertIn("--sft-toml=/tao/config/train.toml", command)
        self.assertIn("--user", command)
        self.assertIn("WORLD_SIZE=1", command)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertIn("example/framework@sha256:", " ".join(command))

    def test_evaluate_render_and_submit_are_single_gpu_h200_safe(self) -> None:
        config = render_cfw_evaluate.build_config(
            annotation_path="/workspace/annotations/benchmark.json",
            media_root="/workspace",
            model_path="/workspace/models/base",
            results_dir="/workspace/results/iter1/evaluate_benchmark",
        )
        rendered = render_cfw_evaluate.dump_toml(config)
        parsed = render_cfw_evaluate.tomllib.loads(rendered)
        self.assertEqual(parsed["num_gpus"], 1)
        self.assertEqual(parsed["vision"]["video_decoder"], "torchcodec-cuda-on-demand")
        self.assertFalse(parsed["model"]["enable_lora"])
        self.assertEqual(parsed["generation"]["max_tokens"], 4)

        command = submit_cfw_evaluate.build_docker_argv(
            immutable_image="example/framework@sha256:" + "c" * 64,
            job_id="docker-eval-test",
            config_host=pathlib.Path("/workspace/specs/evaluate.toml"),
            workspace=pathlib.Path("/workspace"),
            results_dir=pathlib.Path("/workspace/results/iter1/evaluate_benchmark"),
            writable_export_parent=pathlib.Path("/workspace/results/iter1/train"),
            gpus="all",
            offline=False,
        )
        self.assertIn("/workspace/.venv/bin/cosmos-framework-evaluate", command)
        self.assertIn("--config", command)
        self.assertIn(
            "type=bind,src=/workspace/results/iter1/train,dst=/workspace/results/iter1/train",
            command,
        )

    def test_inference_submit_uses_native_entrypoint_and_dcp_handoff(self) -> None:
        command = submit_cfw_inference.build_docker_argv(
            immutable_image="example/framework@sha256:" + "d" * 64,
            job_id="docker-infer-test",
            workspace=pathlib.Path("/workspace"),
            results_dir=pathlib.Path("/workspace/results/inference"),
            model_path=pathlib.Path("/workspace/results/iter1/train/checkpoint"),
            media=pathlib.Path("/workspace/images/part.png"),
            prompt="Return exactly OK or NG.",
            media_type="image",
            max_new_tokens=4,
            framework_config=pathlib.Path("/workspace/results/iter1/train/train.toml"),
            action_model_dir=pathlib.Path("/workspace/results/iter1/train/action_model"),
            base_model_path=pathlib.Path("/workspace/models/base"),
            gpus="all",
            offline=False,
        )
        self.assertIn("/workspace/.venv/bin/cosmos-framework-inference", command)
        for flag in ("--model_path", "--config_file", "--export_dir", "--vit_checkpoint_path"):
            self.assertIn(flag, command)
        self.assertIn("--enable_lora", command)
        self.assertIn("false", command)

    def test_training_annotations_accept_exactly_one_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            media = root / "media"
            media.mkdir()
            image = media / "part.png"
            image.write_bytes(b"image")
            annotation = root / "train.json"
            annotation.write_text(
                json.dumps([{"images": [str(image)], "conversations": []}]),
                encoding="utf-8",
            )
            self.assertEqual(
                submit_cfw_train.annotation_identity_mounts(annotation, media),
                [(media.resolve(), str(media.resolve()), True)],
            )


if __name__ == "__main__":
    unittest.main()
