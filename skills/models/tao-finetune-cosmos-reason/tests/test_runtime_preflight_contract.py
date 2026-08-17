# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
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


def _video_args() -> SimpleNamespace:
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
        rl_dataset_cache_mode="direct",
        rl_validation_freq_steps=0,
        validation_batch_size=1,
        seed=42,
        sequence_length=40960,
        nodes=1,
        gpus_per_node=8,
        training_mode="dense",
        experiment_id="video-smoke",
        frames=8,
        video_max_pixels=81920,
        system_prompt="You are a helpful assistant.",
        container_cache_dir="/cache",
        run_mode="smoke",
        video_override_map="",
        tao_job_id="video-smoke",
        container_results_dir="/results",
        nccl_debug="INFO",
        cuda_allocator="expandable_segments:True",
    )


def _system_video_runtime() -> dict[str, object]:
    return {
        "selected_profile": "system-pyav",
        "video_decoder": "torchvision",
        "frame_transfer": "host_rgb",
        "video_cache_size": 0,
        "decoder_cache_size": 1,
        "sft_batch_threads": 1,
        "dataloader_num_workers": 0,
        "dataloader_prefetch_factor": None,
    }


def test_video_spec_and_environment_force_packaged_system_pyav_contract() -> None:
    args = _video_args()
    video_runtime = _system_video_runtime()
    spec = MODULE._rl_spec(
        args,
        {"epochs": 1},
        "/models/cosmos3",
        ["/data/train.json"],
        ["/data/train"],
        ["/data/val.json"],
        ["/data/val"],
        {},
        video_runtime,
    )
    environment = MODULE._env(
        args,
        "cosmos-rl",
        "/models/cosmos3",
        ["/data/train.json"],
        ["/data/train"],
        ["/data/val.json"],
        ["/data/val"],
        rl_video_runtime=video_runtime,
    )

    assert spec["custom"]["video_decoder"] == "torchvision"
    assert spec["custom"]["vision"]["video_decoder"] == "torchvision"
    assert environment["FORCE_QWENVL_VIDEO_READER"] == "torchvision"
    assert spec["train"]["train_policy"]["dataloader_num_workers"] == 0
    assert spec["train"]["train_policy"]["dataloader_drop_last"] is False
    assert spec["train"]["train_policy"]["enable_dataset_cache"] is False
    assert "dataloader_prefetch_factor" not in spec["train"]["train_policy"]
    assert "dataset_cache_dir" not in spec["train"]["train_policy"]
    assert "cache_dir" not in spec["custom"]["vision"]
    assert "COSMOS_CACHE" not in environment


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
        rl_video_runtime=_system_video_runtime(),
    )

    runtime = contract["container_runtime"]
    assert "verify_deepep" in runtime
    assert "verify_vllm_conv3d" in runtime
    assert "h264_cuvid" not in runtime
    assert "libnvcuvid" not in runtime
    assert "_assert_software_video_decoders" in runtime
    assert "FORCE_QWENVL_VIDEO_READER" in runtime
    assert "torchvision" in runtime
    assert "_tao_linear_patch_embed" in runtime
    assert "_tao_channels_last_3d" in runtime
    assert "DeepEP Python/extension ABI" in contract["checks"]
    assert "vLLM Qwen3-VL Conv3D dispatch guard" in contract["checks"]
    assert "checksum-pinned software System PyAV image capability" in contract["checks"]
    assert "backward-safe Qwen3-VL PatchEmbed" in contract["checks"]
    assert "384 GiB free result/checkpoint space" in contract["checks"]


def test_dataset_manifest_preserves_supplied_shared_filesystem_alias(tmp_path: Path) -> None:
    physical = tmp_path / "fs11" / "dataset"
    physical.mkdir(parents=True)
    (physical / "clip.mp4").write_bytes(b"video")
    annotation = physical / "train.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "id": "sample-1",
                    "video": "clip.mp4",
                    "conversations": [
                        {"from": "human", "value": "What happened?"},
                        {"from": "gpt", "value": "Nothing."},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    alias = tmp_path / "fsw"
    alias.symlink_to(tmp_path / "fs11", target_is_directory=True)

    inspected = MODULE.inspect_dataset(
        dataset_family="video_conversation",
        annotations=[str(alias / "dataset" / "train.json")],
        media_roots=[str(alias / "dataset")],
        verify_media_content=False,
    )

    assert inspected["media_manifest"][0]["path"] == str(
        alias / "dataset" / "clip.mp4"
    )


def test_slurm_renderer_creates_writable_mount_roots_before_pyxis() -> None:
    args = SimpleNamespace(
        platform="slurm",
        partition="polar3",
        account="tao",
        sqsh_path="/images/cosmos.sqsh",
        use_requeue=False,
        timeout="00:10:00",
        container_mount=[
            "/inputs:/data:ro",
            "/runs/results:/results",
            "/runs/checkpoints:/checkpoints",
            "/runs/cache:/cache",
        ],
        nodes=1,
        gpus_per_node=8,
        cpus_per_task=64,
        tao_job_id="cosmos-reason-train-test",
        experiment_id="test",
        time_limit="00:15:00",
        stdout_path="/runs/logs/main.out",
        stderr_path="/runs/logs/main.err",
        qos="",
        reservation="",
        exclusive=True,
        results_dir="/runs/results",
        checkpoint_dir="/runs/checkpoints",
        cache_dir="/runs/cache",
        master_port=29500,
    )
    script = MODULE.render_slurm(
        args,
        {
            "environment": {},
            "command": "true",
            "decoder_artifact": {"required": False, "enabled": False},
        },
    )

    pyxis_offset = script.index("srun ")
    for path in ("/runs/results", "/runs/checkpoints", "/runs/cache"):
        setup = f"mkdir -p -- {path}"
        assert setup in script
        assert script.index(setup) < pyxis_offset
    assert "mkdir -p -- /inputs" not in script
    assert script.count("--container-mounts=") == 1
    assert (
        "--container-mounts=/inputs:/data:ro,/runs/results:/results,"
        "/runs/checkpoints:/checkpoints,/runs/cache:/cache"
    ) in script


def _decoder_args(**overrides: object) -> SimpleNamespace:
    values = {
        "video_override_max_macroblocks": 8192,
        "video_override_workers": 16,
        "video_override_map": "",
        "video_override_manifest": "",
        "video_override_fingerprint": "",
        "video_override_force_video": [],
        "processor_revision": "packaged",
        "cache_dir": "/cache",
        "tao_integration_commit": "a" * 40,
        "train_annotation": ["/data/train.json"],
        "train_media_root": ["/data/train"],
        "validation_annotation": ["/data/val.json"],
        "validation_media_root": ["/data/val"],
        "dataset_family": "task_aware_video_reasoning",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_framework_task_aware_data_uses_native_runtime_without_rl_artifact() -> None:
    artifact = MODULE._decoder_artifact_plan(
        _decoder_args(),
        backend="cosmos-framework",
        model={"fingerprint": "b" * 64},
        model_profile={"frames": 8},
        train_data={"dataset_fingerprint": "c" * 64},
        val_data={"dataset_fingerprint": "d" * 64},
        rl_video_runtime=None,
    )

    assert artifact["required"] is False
    assert artifact["enabled"] is False
    assert artifact["preparation_module"] is None
    assert artifact["validation_module"] is None
    assert artifact["policy"] == {
        "macroblock_scan": False,
        "force_all_validation_media": False,
        "forced_runtime_sources": [],
        "gpu_random_access_validation_required": False,
        "selection_basis": "framework_native_torchcodec_cuda_on_demand",
    }


def test_framework_rejects_cosmos_rl_video_override_artifact() -> None:
    args = _decoder_args(
        video_override_map="/cache/map.json",
        video_override_manifest="/cache/manifest.json",
        video_override_fingerprint="e" * 64,
    )

    try:
        MODULE._decoder_artifact_plan(
            args,
            backend="cosmos-framework",
            model={"fingerprint": "b" * 64},
            model_profile={"frames": 8},
            train_data={"dataset_fingerprint": "c" * 64},
            val_data={"dataset_fingerprint": "d" * 64},
            rl_video_runtime=None,
        )
    except MODULE.WorkflowError as exc:
        assert "owned by the Cosmos-RL backend" in str(exc)
    else:
        raise AssertionError("Framework accepted a Cosmos-RL video override artifact")
