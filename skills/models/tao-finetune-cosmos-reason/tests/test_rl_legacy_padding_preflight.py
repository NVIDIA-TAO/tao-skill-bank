# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts" / "cosmos_workflow.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("cosmos_workflow_legacy_padding", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cosmos_rl_preflight_accepts_safe_legacy_right_padding_contract() -> None:
    args = SimpleNamespace(
        gpus_per_node=8,
        dataset_family="video_conversation",
        results_dir="/results",
        checkpoint_dir="/checkpoints",
        cache_dir="/cache",
        train_annotation=["/data/train.json"],
        train_media_root=["/data/train"],
        validation_annotation=["/data/validation.json"],
        validation_media_root=["/data/validation"],
        platform="docker",
        sqsh_path="",
    )
    runtime = {
        "selected_profile": "system-pyav",
        "video_decoder": "torchvision",
        "frame_transfer": "host_rgb",
        "video_cache_size": 0,
        "decoder_cache_size": 1,
        "sft_batch_threads": 1,
        "dataloader_num_workers": 0,
        "dataloader_prefetch_factor": None,
    }

    contract = MODULE._preflight_contract(
        args,
        "cosmos-rl",
        {"tag": "example.invalid/cosmos-rl:test"},
        "/models/reasoner-8b",
        "/data/train/example.mp4",
        rl_video_runtime=runtime,
    )
    command = contract["container_runtime"]

    assert "TAO_PREFLIGHT_ASSERTION_FAILED:vlm_padding_contract" in command
    assert 'batch["attention_mask"]' in command
    assert 'computed_max_len - len(x["input_ids"])' in command
    assert 'computed_max_len - len(x["logprob_masks"])' in command
    assert "_enforce_visual_gradient_contract" in command
    assert "self.data_packer.sft_collate_fn(" in command
    assert "output = self.forward_model(**batch)" in command
    assert "loss.backward()" in command
