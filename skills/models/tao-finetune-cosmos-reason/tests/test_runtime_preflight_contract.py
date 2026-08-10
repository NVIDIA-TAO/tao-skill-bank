# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts" / "cosmos_workflow.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("cosmos_workflow", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _wts_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_family="video_conversation",
        rl_train_batch_per_replica=0,
        rl_mini_batch=1,
        minimum_lr_factor=None,
        container_checkpoint_dir="/checkpoints",
        learning_rate=1.1e-5,
        weight_decay=0.09,
        scheduler="linear",
        optimizer_epsilon=1e-8,
        warmup=0,
        gradient_clip=1.0,
        precision="bfloat16",
        async_checkpoint=False,
        max_checkpoints=2,
        rl_dataloader_num_workers=0,
        rl_dataloader_prefetch_factor=1,
        rl_validation_freq_steps=0,
        validation_batch_size=1,
        seed=42,
        sequence_length=40960,
        nodes=1,
        gpus_per_node=8,
        training_mode="dense",
        experiment_id="wts-smoke",
        frames=8,
        system_prompt="You are a helpful assistant.",
        container_cache_dir="/cache",
        video_override_map="",
        tao_job_id="wts-smoke",
        container_results_dir="/results",
        nccl_debug="INFO",
        cuda_allocator="expandable_segments:True",
    )


def test_wts_spec_and_environment_force_packaged_system_pyav_contract() -> None:
    args = _wts_args()
    spec = MODULE._rl_spec(
        args,
        {"epochs": 1},
        "/models/cosmos3",
        ["/data/train.json"],
        ["/data/train"],
        ["/data/val.json"],
        ["/data/val"],
        {},
    )
    environment = MODULE._env(
        args,
        "cosmos-rl",
        "/models/cosmos3",
        ["/data/train.json"],
        ["/data/train"],
        ["/data/val.json"],
        ["/data/val"],
    )

    assert spec["custom"]["video_decoder"] == "torchvision"
    assert spec["custom"]["vision"]["video_decoder"] == "torchvision"
    assert environment["FORCE_QWENVL_VIDEO_READER"] == "torchvision"
    assert spec["train"]["train_policy"]["dataloader_num_workers"] == 0
    assert "dataloader_prefetch_factor" not in spec["train"]["train_policy"]


def test_cosmos_rl_preflight_rejects_dependency_abi_and_dispatch_regressions() -> None:
    args = SimpleNamespace(
        gpus_per_node=1,
        dataset_family="video_conversation",
        results_dir="/results",
        checkpoint_dir="/checkpoints",
        cache_dir="/cache",
        train_annotation=["/data/train.json"],
        train_media_root=["/data/train"],
        validation_annotation=["/data/val.json"],
        validation_media_root=["/data/val"],
        platform="docker",
        sqsh_path="",
    )
    contract = MODULE._preflight_contract(
        args,
        "cosmos-rl",
        {"tag": "example.invalid/cosmos-rl:test"},
        "/models/cosmos3",
        "/data/train/example.mp4",
    )

    runtime = contract["container_runtime"]
    assert "verify_deepep" in runtime
    assert "verify_vllm_conv3d" in runtime
    assert "h264_cuvid" in runtime
    assert "FORCE_QWENVL_VIDEO_READER" in runtime
    assert "torchvision" in runtime
    assert "_tao_linear_patch_embed" in runtime
    assert "_tao_channels_last_3d" in runtime
    assert "DeepEP Python/extension ABI" in contract["checks"]
    assert "vLLM Qwen3-VL Conv3D dispatch guard" in contract["checks"]
    assert "system PyAV/FFmpeg NVDEC and libnvcuvid" in contract["checks"]
    assert "backward-safe Qwen3-VL PatchEmbed" in contract["checks"]
    assert "384 GiB free result/checkpoint space" in contract["checks"]
