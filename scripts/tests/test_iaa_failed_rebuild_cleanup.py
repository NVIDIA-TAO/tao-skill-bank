# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skills/applications/tao-run-deft-iaa/scripts/cleanup_failed_dataset_rebuild.py"
SPEC = importlib.util.spec_from_file_location("cleanup_failed_dataset_rebuild", SCRIPT)
assert SPEC and SPEC.loader
CLEANUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLEANUP)


def _fixture(tmp_path):
    workspace = tmp_path / "workspace"
    results = workspace / "results/run"
    dataset = workspace / "data/dataset"
    (results / "dataset_setup").mkdir(parents=True)
    digest = "a" * 64
    state = {
        "workflow": "tao-run-deft-iaa", "results_dir": str(results),
        "active_runtime_sha256": digest,
        "config": {
            "platform": "slurm", "workspace": str(workspace),
            "dataset_root": str(dataset), "iaa_deft_bundle_sha256": digest,
        },
    }
    log = results / "dataset_setup/dataset_rebuild.log"
    log.write_text("outer workload failure\n", encoding="utf-8")
    (results / "dataset_setup/rebuild_verify.log").write_text(
        "OSError: Disk quota exceeded\n", encoding="utf-8"
    )
    status = {
        "workflow": "tao-run-deft-iaa", "name": "dataset_rebuild",
        "status": "error", "backend_state": "ERROR", "backend_exit_code": 1,
        "log_path": str(log),
    }
    (results / "deft_state.json").write_text(json.dumps(state), encoding="utf-8")
    (results / "dataset_setup/dataset_rebuild.status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    return results


def test_cleanup_requires_confirmation(tmp_path):
    results = _fixture(tmp_path)
    with pytest.raises(ValueError, match="--confirm"):
        CLEANUP.cleanup(argparse.Namespace(
            results_dir=results, login="user@login", remote_workspace=pathlib.Path("/lustre/team/workspace"),
            confirm=False,
        ))


def test_cleanup_derives_exact_owned_path_and_writes_receipt(tmp_path, monkeypatch):
    results = _fixture(tmp_path)
    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        captured["command"] = argv[-1]
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(CLEANUP.subprocess, "run", run)
    output = CLEANUP.cleanup(argparse.Namespace(
        results_dir=results, login="user@login", remote_workspace=pathlib.Path("/lustre/team/workspace"),
        confirm=True,
    ))
    expected = "/lustre/team/workspace/data/.dataset.rebuild-aaaaaaaaaaaa"
    assert output["status"] == "removed"
    assert output["remote_staging"] == expected
    assert "rm -rf -- \"$staging\"" in captured["command"]
    assert "chmod -R" not in captured["command"]
    assert 'for child in images captions images_raw' in captured["command"]
    assert 'find -P' not in captured["command"]
    assert expected in captured["command"]
    assert "--confirm" not in " ".join(captured["argv"])
    assert pathlib.Path(output["receipt"]).is_file()
