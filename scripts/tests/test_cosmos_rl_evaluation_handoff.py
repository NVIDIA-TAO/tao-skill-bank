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
    assert all(len(item["sha256"]) == 64 for item in result["files"])
    for item in result["files"]:
        assert item["sha256"] == hashlib.sha256(
            (export / item["path"]).read_bytes()
        ).hexdigest()


def test_dense_hf_base_model_is_verified_for_automl_baseline(tmp_path: Path) -> None:
    model = tmp_path / "base-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8"
    )
    _safetensors(model / "model.safetensors")

    result = CHECKPOINT.verify(str(model), "dense", None, base_model=True)

    assert result["status"] == "VERIFIED"
    assert result["source_checkpoint"] == str(model)
    assert result["action_model_path"] == str(model)
    assert result["epoch"] is None
    assert result["base_model"] is True
    assert result["checkpoint_kind"] == "hf_dense_base_model_safetensors"


def test_base_model_rejects_epoch_and_peft(tmp_path: Path) -> None:
    model = tmp_path / "base-model"
    model.mkdir()
    for mode, epoch, message in (
        ("dense", 1, "cannot be combined"),
        ("peft", None, "must use dense"),
    ):
        try:
            CHECKPOINT.verify(str(model), mode, epoch, base_model=True)
        except CHECKPOINT.CheckpointError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("invalid baseline model verification was accepted")


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
