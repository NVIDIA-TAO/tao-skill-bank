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
    assert "_tao_linear_patch_embed" in runtime
    assert "DeepEP Python/extension ABI" in contract["checks"]
    assert "vLLM Qwen3-VL Conv3D dispatch guard" in contract["checks"]
    assert "system PyAV/FFmpeg NVDEC and libnvcuvid" in contract["checks"]
    assert "backward-safe Qwen3-VL PatchEmbed" in contract["checks"]
    assert "384 GiB free result/checkpoint space" in contract["checks"]
