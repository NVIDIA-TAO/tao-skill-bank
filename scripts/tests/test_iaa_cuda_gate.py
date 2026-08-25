# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
IAA_SCRIPTS = REPO / "skills" / "applications" / "tao-run-deft-iaa" / "scripts"
sys.path.insert(0, str(IAA_SCRIPTS))
import run_iaa_cuda_gate as gate  # noqa: E402


def completed(code: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def test_native_success_records_no_environment_value(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "_run", lambda _: completed(0, "IAA_CUDA_PROBE=PASS"))
    output = tmp_path / "receipt.json"
    result = gate.run_gate("example/pyt:fixed", [6, 7], tmp_path / "probe.py", ["clip"], output)
    assert result["compatibility_mode"] == "native"
    assert result["compatibility_path"] is None
    assert "LD_LIBRARY_PATH" not in output.read_text()


def test_driver_failure_uses_verified_bundle_and_typed_argv(monkeypatch, tmp_path):
    calls = []
    responses = iter([
        completed(1, stderr="NVIDIA driver on your system is too old\ntorch.cuda.is_available() is false"),
        completed(0),
        completed(0, "IAA_CUDA_PROBE=PASS"),
    ])

    def fake_run(argv):
        calls.append(argv)
        return next(responses)

    monkeypatch.setattr(gate, "_run", fake_run)
    result = gate.run_gate("example/pyt:fixed", [6, 7], tmp_path / "probe.py", ["clip"], tmp_path / "out.json")
    assert result["compatibility_mode"] == "image_forward_compat"
    assert calls[0][calls[0].index("--gpus") + 1] == '"device=6,7"'
    assert f"LD_LIBRARY_PATH={gate.COMPAT_PATH}" in calls[2]
    rendered = json.dumps(result, sort_keys=True)
    assert "TOKEN" not in rendered and "KEY" not in rendered


def test_missing_bundle_stops(monkeypatch, tmp_path):
    responses = iter([
        completed(1, stderr="NVIDIA driver on your system is too old\ntorch.cuda.is_available() is false"),
        completed(1),
    ])
    monkeypatch.setattr(gate, "_run", lambda _: next(responses))
    with pytest.raises(RuntimeError, match="no verified compatibility bundle"):
        gate.run_gate("example/pyt:fixed", [6], tmp_path / "probe.py", [], tmp_path / "out.json")


def test_compatibility_probe_still_fails(monkeypatch, tmp_path):
    responses = iter([
        completed(1, stderr="NVIDIA driver on your system is too old\ntorch.cuda.is_available() is false"),
        completed(0),
        completed(1),
    ])
    monkeypatch.setattr(gate, "_run", lambda _: next(responses))
    with pytest.raises(RuntimeError, match="still fails"):
        gate.run_gate("example/pyt:fixed", [6], tmp_path / "probe.py", [], tmp_path / "out.json")


def test_unrelated_cuda_error_never_uses_fallback(monkeypatch, tmp_path):
    calls = []

    def fake_run(argv):
        calls.append(argv)
        return completed(1, stderr="CUDA out of memory")

    monkeypatch.setattr(gate, "_run", fake_run)
    with pytest.raises(RuntimeError, match="other than an insufficient driver"):
        gate.run_gate("example/pyt:fixed", [6], tmp_path / "probe.py", [], tmp_path / "out.json")
    assert len(calls) == 1
