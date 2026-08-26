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

import export_cfw_checkpoint  # noqa: E402
import render_cfw_sft  # noqa: E402
import submit_cfw_train  # noqa: E402


class FrameworkTrainBackendTests(unittest.TestCase):
    def test_smoke_and_full_profiles_render_as_native_lora_toml(self) -> None:
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
            self.assertEqual(parsed["model"]["precision"], "bfloat16")
            self.assertEqual(parsed["optimizer"]["keys_to_select"], ["lora_"])
            self.assertEqual(parsed["checkpoint"]["keys_to_skip_loading"], [])
            self.assertEqual(parsed["custom"]["fsdp_master_dtype"], "bfloat16")
            self.assertEqual(parsed["custom"]["model_max_length"], 40960)
            self.assertEqual(parsed["custom"]["images_per_record"], 2)
            self.assertNotIn("__TAO_CR3_", rendered)

    def test_train_command_preserves_identity_and_native_entrypoint(self) -> None:
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
            identity_mounts=[
                (pathlib.Path("/host/media"), "/host/media", True)
            ],
            extra_mounts=[],
            offline=True,
        )
        joined = " ".join(command)
        self.assertIn("--user", command)
        self.assertIn("USER=", joined)
        self.assertIn("LOGNAME=", joined)
        self.assertIn("HOME=/tmp", command)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", command)
        for expected in (
            "RANK=0",
            "WORLD_SIZE=1",
            "LOCAL_RANK=0",
            "LOCAL_WORLD_SIZE=1",
            "MASTER_ADDR=127.0.0.1",
            "MASTER_PORT=29500",
        ):
            self.assertIn(expected, command)
        self.assertIn("/etc/passwd:/etc/passwd:ro", command)
        self.assertIn("/etc/group:/etc/group:ro", command)
        self.assertIn(submit_cfw_train.CONTAINER_ADAPTER, joined)
        self.assertIn("cosmos_framework.scripts.train", command)
        self.assertIn("--sft-toml=/tao/config/train.toml", command)
        self.assertIn("model.config.parallelism.fsdp_master_dtype=bfloat16", command)
        self.assertIn("model.config.policy.model_max_length=40960", command)
        self.assertIn("example/framework@sha256:", joined)
        self.assertIn(
            "type=bind,src=/host/media,dst=/host/media,readonly", joined
        )
        source = (SCRIPTS / "submit_cfw_train.py").read_text(encoding="utf-8")
        self.assertIn("images.tao_toolkit.cosmos_framework", source)
        self.assertNotIn("nvcr.io/nvstaging/tao/cosmos-framework:", source)

    def test_absolute_annotation_paths_receive_identity_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            media = root / "workspace"
            external = root / "external"
            inside = media / "images/a.jpg"
            outside = external / "golden/a.jpg"
            inside.parent.mkdir(parents=True)
            outside.parent.mkdir(parents=True)
            inside.write_bytes(b"image")
            outside.write_bytes(b"golden")
            annotation = root / "train.json"
            annotation.write_text(
                json.dumps(
                    [
                        {
                            "images": [str(inside), str(outside)],
                            "conversations": [],
                        }
                    ]
                )
            )

            mounts = submit_cfw_train.annotation_identity_mounts(
                annotation, media
            )
            self.assertEqual(
                mounts,
                [
                    (media.resolve(), str(media.resolve()), True),
                    (outside.parent.resolve(), str(outside.parent), True),
                ],
            )

    def test_native_export_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            checkpoint = root / "checkpoint"
            metadata = checkpoint / "model/.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_bytes(b"dcp-metadata")
            output = root / "export"
            output.mkdir()
            (output / "config.json").write_text('{"model_type":"qwen3_vl"}\n')
            (output / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            (output / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layer.weight": "model-00001-of-00001.safetensors"
                        }
                    }
                )
            )
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
            ):
                (output / name).write_text("{}\n")
            (output / "checkpoint.json").write_text(
                json.dumps({"checkpoint_path": "/tao/checkpoint", "checkpoint_type": "vlm_dcp"})
            )
            (output / "export_manifest.json").write_text(
                json.dumps(
                    {
                        "format": "cosmos-framework-vlm-dcp",
                        "checkpoint_metadata_sha256": export_cfw_checkpoint.sha256_file(metadata),
                        "lora": {"enabled": True},
                        "merged_adapters": 252,
                    }
                )
            )
            verified = export_cfw_checkpoint.verify_export(checkpoint, output)
            self.assertEqual(verified["status"], "VERIFIED")
            self.assertEqual(verified["expected_shard_count"], 1)
            self.assertEqual(verified["evaluation_backend"], "cosmos-rl-vllm")
            self.assertFalse(verified["evaluation_model_contract"]["enable_lora"])

            command = export_cfw_checkpoint.build_docker_argv(
                immutable_image="example/framework@sha256:" + "c" * 64,
                checkpoint=checkpoint,
                config=root / "config.yaml",
                output=output,
                base_model=root / "base",
                gpus="all",
            )
            self.assertIn(export_cfw_checkpoint.EXPORTER, command)
            self.assertEqual(export_cfw_checkpoint.EXPORTER, "cosmos_framework.scripts.export_vlm_dcp")

    def test_interim_adapter_enforces_two_image_bare_contract(self) -> None:
        source = (SCRIPTS / "cfw_cr3_aoi_adapter.py").read_text(encoding="utf-8")
        self.assertIn("exactly two string image paths", source)
        self.assertIn('{"OK", "NG"}', source)
        self.assertIn("Remove it when the pinned image provides", source)


if __name__ == "__main__":
    unittest.main()
