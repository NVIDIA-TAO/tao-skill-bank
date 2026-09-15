# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the portable Docker test handoff."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[1] / "test_dinov3_deft_containers.py"
SPEC = importlib.util.spec_from_file_location("deft_container_handoff", MODULE)
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


def args(tmp_path, dry_run=False):
    return argparse.Namespace(skill_bank=tmp_path / "bank", pytorch=tmp_path / "pyt",
                              data_services=tmp_path / "ds", output=tmp_path / "output",
                              pytorch_image="registry/pyt:reviewed", data_services_image="registry/ds:reviewed",
                              suite="source", gpus="device=1", timeout=10, dry_run=dry_run, core=None)


def test_dry_run_needs_no_docker_or_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(handoff, "source_info", lambda _: {"commit": "abc", "base": "def", "changed_files": ["x.py"]})
    monkeypatch.setattr(handoff, "capture", lambda *_: pytest.fail("dry-run invoked Docker"))
    assert handoff.run(args(tmp_path, True)) == 0
    assert not (tmp_path / "output").exists()
    assert json.loads(capsys.readouterr().out)["gpus"] == "device=1"


def test_dirty_checkout_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "capture", lambda *_: " M source.py")
    with pytest.raises(ValueError, match="Commit or stash"):
        handoff.source_info(tmp_path)


def test_regression_selection_includes_changed_pytorch_surfaces():
    command = handoff.regression_commands("tao-pytorch", "python")[0]
    assert "tests/ssl_unit_test/dinov3" in command
    assert "tests/core" in command
    assert "tests/distributed" in command
    assert "tests/ssl_unit_test/nvdinov2/test_model.py" in command
    assert "tests/ssl_unit_test/nvdinov2/test_dataloader.py" in command


@pytest.mark.parametrize("repo", ["tao-pytorch", "tao-data-services", "tao-skill-bank"])
def test_packaged_default_does_not_require_optional_faiss(monkeypatch, repo):
    monkeypatch.setattr(handoff.importlib, "import_module", lambda _: SimpleNamespace(__file__="/installed/module.py"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "faiss", None)
    assert handoff.packaged_check(repo) == 0
    with pytest.raises(ModuleNotFoundError, match="faiss"):
        handoff.packaged_check(repo, require_gpu_faiss=True)


def test_packaged_check_still_rejects_missing_cuda(monkeypatch):
    monkeypatch.setattr(handoff.importlib, "import_module", lambda _: SimpleNamespace(__file__="/installed/module.py"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        handoff.packaged_check("tao-pytorch")


@pytest.mark.parametrize("timeout", [False, True])
@pytest.mark.parametrize("include_core", [False, True])
@pytest.mark.parametrize("require_gpu_faiss", [False, True])
def test_execution_pins_images_preserves_sources_and_cleans_up(tmp_path, monkeypatch, timeout, include_core, require_gpu_faiss):
    monkeypatch.setattr(handoff, "source_info", lambda _: {"commit": "abc", "base": "def", "changed_files": ["x.py"]})
    monkeypatch.setattr(handoff, "capture", lambda argv: "sha256:" + ("1" if "pyt" in argv[-1] else "2") * 64)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "run"] and timeout:
            raise subprocess.TimeoutExpired(argv, 10)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    options = args(tmp_path)
    options.require_gpu_faiss = require_gpu_faiss
    if include_core:
        options.core = tmp_path / "core"
    assert handoff.run(options) == int(timeout)
    runs = [argv for argv in calls if argv[:2] == ["docker", "run"]]
    removals = [argv for argv in calls if argv[:2] == ["docker", "rm"]]
    assert len(runs) == len(removals) == (4 if include_core else 3)
    assert not any("build" in argv or "pull" in argv for argv in calls)
    for argv, cleanup in zip(runs, removals):
        assert "--pull=never" in argv
        assert argv[argv.index("--gpus") + 1] == "device=1"
        assert any(part.startswith("sha256:") for part in argv)
        assert cleanup[-1] == argv[argv.index("--name") + 1]
        assert "CUDA_VISIBLE_DEVICES=0" not in argv
        assert ("--require-gpu-faiss" in argv) == require_gpu_faiss
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert set(summary["exit_codes"].values()) == ({124} if timeout else {0})
    assert ("tao-core" in summary["repos"]) == include_core
    assert summary["require_gpu_faiss"] == require_gpu_faiss
    if include_core:
        assert summary["images"]["tao-core"] == options.pytorch_image
