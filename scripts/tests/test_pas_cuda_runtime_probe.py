# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the PAS image-level CUDA framework gate."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]
PROBE_PATH = (
    REPO
    / "skills"
    / "applications"
    / "tao-run-deft-pas"
    / "scripts"
    / "check_pas_cuda_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("check_pas_cuda_runtime", PROBE_PATH)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class _Allocation:
    def add_(self, value):
        assert value == 1
        return self


class _Cuda:
    def __init__(self, *, count=1, available=True, failure=None):
        self.count = count
        self.available = available
        self.failure = failure
        self.synchronized = []

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def get_device_properties(self, index):
        if self.failure:
            raise self.failure
        return SimpleNamespace(
            name=f"GPU-{index}", major=8, minor=0, total_memory=80 * 1024**3
        )

    def synchronize(self, index):
        self.synchronized.append(index)


class _Torch:
    __version__ = "test"
    version = SimpleNamespace(cuda="test")

    def __init__(self, cuda):
        self.cuda = cuda
        self.allocations = []

    def empty(self, size, *, device):
        self.allocations.append((size, device))
        return _Allocation()


def test_probe_allocates_and_synchronizes_every_required_gpu():
    cuda = _Cuda(count=2)
    torch = _Torch(cuda)
    result = probe.probe_runtime(
        minimum_gpus=2,
        required_clis=["embedding", "tmm"],
        torch_module=torch,
        which=lambda name: f"/usr/bin/{name}",
    )

    assert result["status"] == "PASS"
    assert result["visible_gpus"] == 2
    assert torch.allocations == [(1, "cuda:0"), (1, "cuda:1")]
    assert cuda.synchronized == [0, 1]


def test_probe_rejects_gpu_enumeration_when_framework_initialization_fails():
    torch = _Torch(_Cuda(failure=RuntimeError("driver too old")))

    with pytest.raises(RuntimeError, match="driver too old"):
        probe.probe_runtime(
            minimum_gpus=1,
            required_clis=["embedding"],
            torch_module=torch,
            which=lambda name: f"/usr/bin/{name}",
        )


def test_probe_rejects_missing_tao_entrypoints():
    with pytest.raises(RuntimeError, match="tmm"):
        probe.probe_runtime(
            minimum_gpus=1,
            required_clis=["embedding", "tmm"],
            torch_module=_Torch(_Cuda()),
            which=lambda name: None if name == "tmm" else f"/usr/bin/{name}",
        )


@pytest.mark.parametrize("minimum", [0, -1])
def test_probe_rejects_invalid_gpu_requirement(minimum):
    with pytest.raises(ValueError, match="at least 1"):
        probe.probe_runtime(
            minimum_gpus=minimum,
            required_clis=[],
            torch_module=_Torch(_Cuda()),
        )


@dataclass
class _Tower:
    mode: str = "frozen"
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    num_last_blocks: int = 3
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05


@dataclass
class _Peft:
    enabled: bool = False
    method: str = "lora"
    train_logit_calibration: bool = True
    vision: _Tower = field(default_factory=_Tower)
    text: _Tower = field(default_factory=_Tower)


@dataclass
class _Config:
    peft: _Peft = field(default_factory=_Peft)


def test_clip_lora_probe_requires_schema_and_checkpoint_symbols():
    runtime = SimpleNamespace(
        LoRALinear=type("LoRALinear", (), {}),
        inject_lora=lambda: None,
        merge_lora=lambda: None,
        _register_lora_checkpoint_compatibility=lambda: None,
    )
    result = probe.probe_clip_lora_contract(_Config, runtime)
    assert result["checkpoint_behavior"] == "register-and-merge"

    del runtime.merge_lora
    with pytest.raises(RuntimeError, match="merge_lora"):
        probe.probe_clip_lora_contract(_Config, runtime)
