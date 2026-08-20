# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts" / "cosmos_workflow.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("cosmos_workflow_reproducibility", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_rl_spec_preserves_top_level_and_dataloader_seed_contract() -> None:
    args = SimpleNamespace(
        dataset_family="video_conversation",
        rl_dataset_cache_mode="direct",
        rl_train_batch_per_replica=8,
        rl_mini_batch=1,
        minimum_lr_factor=None,
        container_checkpoint_dir="/checkpoints",
        learning_rate=1.6e-5,
        training_mode="dense",
        weight_decay=0.01,
        scheduler="none",
        optimizer_epsilon=1e-8,
        warmup=0.03,
        gradient_clip=1.0,
        precision="bfloat16",
        async_checkpoint=False,
        max_checkpoints=1,
        rl_validation_freq_steps=0,
        rl_validation_shard_strategy="stride",
        validation_batch_size=1,
        seed=42,
        sequence_length=81920,
        frames=0,
        fps=2.0,
        min_frames=None,
        max_frames=128,
        video_start=None,
        video_end=None,
        video_resized_height=None,
        video_resized_width=None,
        video_min_pixels=None,
        video_max_pixels=186625,
        video_total_pixels=None,
        system_prompt="You are a helpful assistant.",
        container_cache_dir="/cache",
        run_mode="full",
        nodes=1,
        gpus_per_node=8,
        experiment_id="reproducibility-contract",
        video_override_map="",
    )
    runtime = {
        "selected_profile": "pynv-device-rgbp",
        "video_decoder": "pynvvideocodec",
        "video_cache_size": 181,
        "decoder_cache_size": 181,
        "dataloader_num_workers": 0,
        "dataloader_prefetch_factor": None,
    }

    spec = MODULE._rl_spec(
        args,
        {"epochs": 8},
        "/models/reasoner-8b",
        ["/data/train.json"],
        ["/data/train"],
        ["/data/validation.json"],
        ["/data/validation"],
        {},
        runtime,
    )

    assert spec["train"]["seed"] == 42
    assert spec["train"]["deterministic"] is True
    assert spec["train"]["train_policy"]["dataloader_seed"] == 42
    assert spec["train"]["train_policy"]["dataloader_num_workers"] == 0
    assert "dataloader_prefetch_factor" not in spec["train"]["train_policy"]
