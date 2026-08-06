# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for launch-preflight GPU architecture handling."""

import subprocess
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


def test_slurm_preflight_checks_remote_scheduler_pyxis_and_enroot(monkeypatch, tmp_path):
    key = tmp_path / "id_ed25519"
    key.write_text("fixture")
    for name, value in {
        "SLURM_USER": "user",
        "SLURM_HOSTNAME": "login.example",
        "SLURM_PARTITION": "compute",
        "SSH_KEY_PATH": str(key),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(preflight.socket, "getaddrinfo", lambda *_args: [(None,)])
    monkeypatch.setattr(preflight, "check_slurm_runtime", lambda _platform: True)
    commands = []

    def fake_run(command, timeout=30, env=None):
        commands.append(command)
        remote = command[-1]
        if remote == "echo TAO_SSH_OK":
            return subprocess.CompletedProcess(command, 0, stdout="TAO_SSH_OK\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(preflight, "run", fake_run)
    platform = {
        "required_credentials": [
            {"name": "SLURM_USER", "source": "env_var"},
            {"name": "SLURM_HOSTNAME", "source": "env_var"},
            {"name": "SLURM_PARTITION", "source": "env_var"},
        ],
        "credential_groups": [{"require_one_of": ["SSH_KEY_PATH", "SSH_AUTH_SOCK"]}],
    }
    assert preflight.check_slurm(platform, [("data", "/shared/data")], {}, 20, False)
    remote_commands = [command[-1] for command in commands]
    assert any("command -v sbatch" in command for command in remote_commands)
    assert any("command -v enroot" in command for command in remote_commands)
    assert any("--container-image" in command for command in remote_commands)


def test_slurm_preflight_rejects_missing_remote_pyxis(monkeypatch, tmp_path, capsys):
    key = tmp_path / "id_ed25519"
    key.write_text("fixture")
    for name, value in {
        "SLURM_USER": "user",
        "SLURM_HOSTNAME": "login.example",
        "SLURM_PARTITION": "compute",
        "SSH_KEY_PATH": str(key),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(preflight.socket, "getaddrinfo", lambda *_args: [(None,)])
    monkeypatch.setattr(preflight, "check_slurm_runtime", lambda _platform: True)

    def fake_run(command, timeout=30, env=None):
        remote = command[-1]
        if remote == "echo TAO_SSH_OK":
            return subprocess.CompletedProcess(command, 0, stdout="TAO_SSH_OK\n", stderr="")
        if "command -v sbatch" in remote:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="enroot missing")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(preflight, "run", fake_run)
    platform = {
        "required_credentials": [
            {"name": "SLURM_USER", "source": "env_var"},
            {"name": "SLURM_HOSTNAME", "source": "env_var"},
            {"name": "SLURM_PARTITION", "source": "env_var"},
        ],
        "credential_groups": [{"require_one_of": ["SSH_KEY_PATH", "SSH_AUTH_SOCK"]}],
    }
    assert not preflight.check_slurm(platform, [], {}, 20, False)
    assert "Remote SLURM/Pyxis/Enroot preflight failed" in capsys.readouterr().out
