# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for launch-preflight GPU architecture handling."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_tao_launch_preflight as preflight  # noqa: E402


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.3", "sm_103"),
        ("10.3a", "sm_103a"),
        ("103a", "sm_103a"),
        ("sm_103a", "sm_103a"),
        ("compute_103a", "sm_103a"),
    ],
)
def test_normalize_gpu_arch_accepts_arch_specific_forms(value, expected):
    assert preflight.normalize_gpu_arch(value) == expected


def test_cosmos_image_allowlist_includes_compute_capability_variant():
    supported = set(preflight.KNOWN_IMAGE_SMS["cosmos-rl"])
    assert {"sm_103", "sm_103a"} <= supported


def test_architecture_specific_target_matches_family_target():
    assert preflight.gpu_arch_is_supported("sm_103", {"sm_103a"})
    assert preflight.gpu_arch_is_supported("sm_103a", {"sm_103"})


def test_different_architecture_families_do_not_match():
    assert not preflight.gpu_arch_is_supported("sm_121", {"sm_120"})


def test_gpu_resources_accept_cumulative_memory_across_devices(capsys):
    assert preflight.check_gpu_resources(None, None, 256, 4, [80], True)
    assert "required_total_memory_gb=256" in capsys.readouterr().out


def test_gpu_resources_accept_nominal_single_device_capacity(capsys):
    # Target memory is supplied in GiB, matching nvidia-smi's MiB-based report.
    assert preflight.check_gpu_resources(None, None, 256, 1, [250.7], True)
    assert "GPU resources OK" in capsys.readouterr().out


def test_gpu_resources_reject_insufficient_cumulative_memory(capsys):
    assert not preflight.check_gpu_resources(None, None, 256, 2, [100], True)
    assert "GPU resource check failed" in capsys.readouterr().out
