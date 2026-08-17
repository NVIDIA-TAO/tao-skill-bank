# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluation_workflow.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("evaluation_workflow_backend_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _plan(backend: str) -> dict:
    plan = {
        "action": "train",
        "backend": backend,
        "experiment_id": f"{backend}-training",
        "training": {
            "training_mode": "dense",
            "frames": 8,
            "precision": "bfloat16",
            "sequence_length": 40960,
            "seed": 42,
            "system_prompt": "You are a helpful assistant.",
        },
        "datasets": {"validation": {"dataset_fingerprint": "validation-fingerprint"}},
        "model": {"fingerprint": "model-fingerprint"},
        "compute": {"total_gpus": 8},
        "processor_profile": {"model_tier": "nano", "max_video_pixels": 81920},
        "evaluation_contract": {
            "frames": 8,
            "precision": "bfloat16",
            "seed": 42,
            "batch_size": 1,
            "system_prompt": "You are a helpful assistant.",
            "validation_annotations": ["/data/validation.json"],
            "validation_media_roots": ["/data/videos"],
            "generation": {
                "max_tokens": None,
                "temperature": 0.0,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
            "task_profile": {
                "inferred_task_type": "mcq",
                "answer_type": "letter",
                "metric_names": ["accuracy"],
                "requires_user_input": [],
                "unresolved_accuracy_tasks": [],
            },
        },
    }
    if backend == "cosmos-framework":
        plan["framework_video_runtime"] = {
            "selected_profile": "torchcodec-cuda-on-demand",
            "decoder_device_binding": "explicit_local_rank",
            "decoder_device": "cuda",
            "decoder_threads": 1,
            "sft_process_threads": 8,
            "video_cache_size": 341,
            "dataloader_num_workers": 1,
            "dataloader_prefetch_factor": 2,
            "dataloader_multiprocessing_context": "spawn",
            "dataloader_persistent_workers": True,
        }
    else:
        plan["datasets"]["validation"]["profile"] = {
            "family": "video_conversation"
        }
        plan["rl_video_runtime"] = {
            "selected_profile": "pynv-device-rgbp",
            "frame_transfer": "device_rgbp",
            "video_cache_size": 341,
            "decoder_cache_size": 341,
        }
        plan["decoder_artifact"] = {
            "enabled": True,
            "path": "/data/video_override_map.json",
        }
    return plan


def _args(tmp_path: Path, plan_path: Path, backend: str) -> argparse.Namespace:
    action_model_path = "/exports/model" if backend == "cosmos-framework" else "/exports/rl-model"
    action_model_manifest = None
    if backend == "cosmos-rl":
        action_model_manifest = tmp_path / "rl-checkpoint-manifest.json"
        action_model_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "VERIFIED",
                    "backend": "cosmos-rl",
                    "source_checkpoint": "/checkpoints/epoch_1",
                    "action_model_path": action_model_path,
                    "epoch": 1,
                    "training_mode": "dense",
                    "checkpoint_kind": "hf_dense_safetensors",
                    "files": [{"path": "config.json", "size": 1}],
                }
            ),
            encoding="utf-8",
        )
    else:
        action_model_manifest = tmp_path / "framework-checkpoint-manifest.json"
        action_model_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "VERIFIED",
                    "backend": "cosmos-framework",
                    "source_checkpoint": "/checkpoints/epoch_1",
                    "action_model_path": action_model_path,
                    "verification": {
                        "ok": True,
                        "action_model_path": action_model_path,
                        "weight_files": ["model.safetensors"],
                    },
                }
            ),
            encoding="utf-8",
        )
    return argparse.Namespace(
        training_plan=plan_path,
        training_status=None,
        checkpoint="/checkpoints/epoch_1",
        checkpoint_epoch=None,
        action_model_path=action_model_path,
        action_model_manifest=action_model_manifest,
        action_validation_annotation=None,
        action_validation_media_root=None,
        validation_annotation=[],
        validation_media_root=[],
        system_prompt=None,
        task_type=None,
        answer_type=None,
        evaluation_batch_size=None,
        evaluation_seed=None,
        evaluation_shard_strategy=None,
        generation_max_tokens=None,
        max_video_pixels=None,
        metric=[],
        results_dir=str(tmp_path / "results"),
        num_gpus=None,
        plan_output=tmp_path / "evaluation-plan.json",
        config_output=tmp_path / "evaluation.toml",
    )


def _resolve(tmp_path: Path, backend: str) -> dict:
    plan_path = tmp_path / f"{backend}.json"
    plan_path.write_text(json.dumps(_plan(backend)), encoding="utf-8")
    return MODULE.resolve(_args(tmp_path, plan_path, backend))


def test_framework_evaluation_inherits_native_torchcodec_profile(tmp_path: Path) -> None:
    result = _resolve(tmp_path, "cosmos-framework")
    assert result["ready"] is True
    assert result["config"]["generation"]["max_tokens"] == 10
    assert result["provenance"]["generation.max_tokens"]["source"] == (
        "framework_bounded_classification_protocol"
    )
    assert result["config"]["vision"] == {
        "num_frames": 8,
        "video_decoder": "torchcodec-cuda-on-demand",
        "video_cache_size": 341,
        "process_threads": 8,
        "decoder_threads": 1,
        "decoder_device": "cuda",
        "dataloader_num_workers": 1,
        "dataloader_prefetch_factor": 2,
        "dataloader_multiprocessing_context": "spawn",
        "dataloader_persistent_workers": True,
        "max_pixels": 81920,
        "min_pixels": 81920,
    }
    assert result["provenance"]["vision.min_pixels"] == {
        "source": "framework_preserve_explicit_max_pixels",
        "value": 81920,
    }


def test_cosmos_rl_evaluation_inherits_its_sealed_pynv_profile(tmp_path: Path) -> None:
    result = _resolve(tmp_path, "cosmos-rl")
    assert result["ready"] is False
    assert result["required_user_inputs"] == [
        {
            "field": "generation.max_tokens",
            "reason": "generation length is not a fine-tuning parameter",
        }
    ]
    assert result["config"]["vision"] == {
        "num_frames": 8,
        "video_decoder": "pynvvideocodec",
        "video_cache_size": 1,
        "decoder_cache_size": 4,
        "frame_transfer": "device_rgbp",
        "max_pixels": 81920,
        "video_override_map": "/data/video_override_map.json",
    }
    assert result["config"]["evaluation"]["shard_strategy"] == "media_balanced"
