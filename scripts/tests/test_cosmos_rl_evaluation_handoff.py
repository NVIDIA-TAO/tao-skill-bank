# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "models" / "tao-finetune-cosmos-reason"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHECKPOINT = _module(
    "cosmos_rl_checkpoint_action_test",
    SKILL / "scripts" / "cosmos_rl_checkpoint_action.py",
)
RENDERER = _module(
    "render_evaluation_slurm_test",
    SKILL / "scripts" / "render_evaluation_slurm.py",
)


def test_checkpoint_action_supports_pre_is_relative_to_python():
    source = (SKILL / "scripts" / "cosmos_rl_checkpoint_action.py").read_text(
        encoding="utf-8"
    )
    assert ".is_relative_to(" not in source


def _safetensors(path: Path) -> None:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")


def _native_and_export(tmp_path: Path, mode: str, epoch: int = 1) -> tuple[Path, Path]:
    root = tmp_path / "stamp"
    source = root / "checkpoints" / f"epoch_{epoch}" / "policy"
    source.mkdir(parents=True)
    export = root / "safetensors" / f"epoch_{epoch}"
    export.mkdir(parents=True)
    if mode == "dense":
        (export / "config.json").write_text(
            json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8"
        )
        _safetensors(export / "model.safetensors")
    else:
        (export / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 64}), encoding="utf-8"
        )
        _safetensors(export / "adapter_model.safetensors")
    return source, export


def test_native_policy_resolves_to_verified_dense_export(tmp_path: Path) -> None:
    source, export = _native_and_export(tmp_path, "dense")
    result = CHECKPOINT.verify(str(source), "dense", 1)
    assert result["status"] == "VERIFIED"
    assert result["source_checkpoint"] == str(source)
    assert result["action_model_path"] == str(export)
    assert result["checkpoint_kind"] == "hf_dense_safetensors"
    assert {item["path"] for item in result["files"]} == {
        "config.json",
        "model.safetensors",
    }


def test_native_policy_resolves_to_verified_peft_export(tmp_path: Path) -> None:
    source, export = _native_and_export(tmp_path, "peft", epoch=3)
    result = CHECKPOINT.verify(str(source), "peft", 3)
    assert result["action_model_path"] == str(export)
    assert result["checkpoint_kind"] == "hf_peft_adapter_safetensors"


def test_missing_export_fails_before_evaluation(tmp_path: Path) -> None:
    source = tmp_path / "stamp" / "checkpoints" / "epoch_1" / "policy"
    source.mkdir(parents=True)
    try:
        CHECKPOINT.verify(str(source), "dense", 1)
    except CHECKPOINT.CheckpointError as exc:
        assert "epoch export is missing" in str(exc)
    else:
        raise AssertionError("native policy checkpoint was accepted without an HF export")


def test_dense_index_cannot_escape_epoch_export(tmp_path: Path) -> None:
    source, export = _native_and_export(tmp_path, "dense")
    outside = tmp_path / "outside.safetensors"
    _safetensors(outside)
    (export / "model.safetensors").unlink()
    (export / "escape.safetensors").symlink_to(outside)
    (export / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "escape.safetensors"}}), encoding="utf-8"
    )
    try:
        CHECKPOINT.verify(str(source), "dense", 1)
    except CHECKPOINT.CheckpointError as exc:
        assert "escapes its epoch directory" in str(exc)
    else:
        raise AssertionError("checkpoint verifier accepted an escaping weight symlink")


def _ready_plan(
    tmp_path: Path, *, job_id: str, total_gpus: int, backend: str = "cosmos-rl"
) -> tuple[Path, Path, Path]:
    results_dir = Path("/lustre/audit/evaluation") / job_id
    model_path = "/lustre/audit/training/stamp/safetensors/epoch_1"
    config_path = tmp_path / f"{job_id}.toml"
    config_path.write_text("[evaluation]\nseed = 42\n", encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    plan = {
        "schema_version": 1,
        "ready": True,
        "backend": backend,
        "required_user_inputs": [],
        "blockers": [],
        "config_sha256": config_sha,
        "config": {
            "results_dir": str(results_dir),
            "num_gpus": total_gpus,
            "evaluation": {"seed": 42},
            "model": {"model_name": model_path},
        },
        "checkpoint": {
            "action_model_manifest": {
                "schema_version": 1,
                "status": "VERIFIED",
                "backend": backend,
                "action_model_path": model_path,
            }
        },
    }
    if backend == "cosmos-framework":
        plan["config"]["vision"] = {
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
        }
    plan_path = tmp_path / f"{job_id}.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, config_path, results_dir


def _render_args(
    tmp_path: Path,
    *,
    job_id: str,
    total_gpus: int,
    backend: str = "cosmos-rl",
):
    plan_path, config_path, results_dir = _ready_plan(
        tmp_path, job_id=job_id, total_gpus=total_gpus, backend=backend
    )
    return RENDERER.argparse.Namespace(
        evaluation_plan=plan_path,
        config=config_path,
        runtime_config_path=str(results_dir / "specs" / "evaluation.toml"),
        tao_job_id=job_id,
        sqsh_path="/lustre/images/cosmos-rl.sqsh",
        partition="polar3,polar4",
        account="account",
        gpus_per_node=8,
        cpus_per_task=248,
        time_limit="04:00:00",
        child_timeout="03:48:00",
        mount=["/lustre:/lustre"],
        output=tmp_path / f"{job_id}.sbatch",
    )


def test_single_node_renderer_owns_persistent_results_and_child_status(tmp_path: Path) -> None:
    script = RENDERER.render(
        _render_args(tmp_path, job_id="Cosmos3-Nano-evaluate-a1", total_gpus=8)
    )
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --no-requeue" in script
    assert "--nproc-per-node=8" in script
    assert "--standalone" in script
    assert "/lustre/audit/evaluation:/results" in script
    assert "export TAO_API_RESULTS_DIR=/results" in script
    assert "export TAO_API_JOB_ID=Cosmos3-Nano-evaluate-a1" in script
    assert "/results/Cosmos3-Nano-evaluate-a1/status.json" in script
    assert "child_exit_code" in script
    assert "exit \"$child_rc\"" in script


def test_multinode_renderer_derives_torchrun_world_and_rendezvous(tmp_path: Path) -> None:
    script = RENDERER.render(
        _render_args(tmp_path, job_id="Cosmos3-Nano-evaluate-a2", total_gpus=32)
    )
    assert "#SBATCH --nodes=4" in script
    assert "#SBATCH --ntasks=4" in script
    assert "--nodes=4 --ntasks=4 --ntasks-per-node=1" in script
    assert '--nnodes="${SLURM_NNODES:?}"' in script
    assert '--node-rank="${SLURM_PROCID:?}"' in script
    assert "export MASTER_ADDR=" in script
    assert "export MASTER_PORT=" in script


def test_framework_renderer_selects_only_framework_cli_and_baked_runtime_gate(
    tmp_path: Path,
) -> None:
    script = RENDERER.render(
        _render_args(
            tmp_path,
            job_id="Cosmos3-Nano-framework-evaluate-a1",
            total_gpus=8,
            backend="cosmos-framework",
        )
    )
    assert "# evaluation_backend=cosmos-framework" in script
    assert "command -v cosmos-framework-evaluate" in script
    assert "TAO_FRAMEWORK_EVALUATOR_RUNTIME_OK" in script
    assert "FrameworkTorchCodecVideoPreprocessor" in script
    assert "command -v cosmos-rl-evaluate" not in script


def test_renderer_rejects_sbatch_directive_injection(tmp_path: Path) -> None:
    args = _render_args(tmp_path, job_id="Cosmos3-Nano-evaluate-a3", total_gpus=8)
    args.partition = "polar3\n#SBATCH --qos=unreviewed"
    try:
        RENDERER.render(args)
    except RENDERER.RenderError as exc:
        assert "invalid partition" in str(exc)
    else:
        raise AssertionError("renderer accepted an injected SBATCH directive")


def test_renderer_rejects_malformed_wall_time(tmp_path: Path) -> None:
    args = _render_args(tmp_path, job_id="Cosmos3-Nano-evaluate-a4", total_gpus=8)
    args.time_limit = "04:00:00\n#SBATCH --requeue"
    try:
        RENDERER.render(args)
    except RENDERER.RenderError as exc:
        assert "time limit must use HH:MM:SS" in str(exc)
    else:
        raise AssertionError("renderer accepted a malformed wall time")
