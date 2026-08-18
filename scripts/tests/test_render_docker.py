#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker's own bundle-rendering conventions.

test_platform_render_contract.py holds every platform to the shared contract.
This holds docker to the conventions documented in its SKILL.md, which the
shared contract deliberately does not dictate: same-absolute-path mounts, the
GPU flag, and what air-gap means for a container run.
"""

from __future__ import annotations

import copy
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
MODULE = REPO / "skills/platform/tao-run-on-docker/references/render.py"

BUNDLE = {
    "network_arch": "visual_changenet",
    "action": "data_merge.pair_prepare",
    "image": "docker.io/library/python:3.11-slim",
    "mode": "args",
    "command": "python /opt/prepare.py",
    "args": ["--input-dir", "/w/ng", "--output", "/r/dataset.csv"],
    "declared_inputs": [
        {"spec_key": "input_dir", "type": "folder", "uri": "/w/ng"},
        {"spec_key": "golden_dir", "type": "folder", "uri": "/w/ok"},
    ],
    "declared_outputs": [{"spec_key": "output_csv", "type": "file"}],
    "compute_shape": {"gpus": 0, "nodes": 1},
}
CTX = {"job_id": "vcn-prep-abc123", "results_dir": "/r", "bank": str(REPO)}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("render_docker", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inputs_mount_at_the_same_absolute_path(mod):
    """A path written into a CSV must resolve on both sides of the mount."""
    argv = mod.render(BUNDLE, CTX)["argv"]
    assert "/w/ng:/w/ng:ro" in argv and "/w/ok:/w/ok:ro" in argv


def test_results_dir_is_mounted_writable(mod):
    argv = mod.render(BUNDLE, CTX)["argv"]
    assert "/r:/r" in argv and "/r:/r:ro" not in argv


def test_results_root_is_exported(mod):
    """The bundle is authored before the job id exists, so it cannot name the
    output path; TAO_RESULTS_ROOT is how the bank passes it."""
    argv = mod.render(BUNDLE, CTX)["argv"]
    assert "TAO_RESULTS_ROOT=/r" in argv


def test_duplicate_inputs_mount_once(mod):
    """Two spec_keys can point at one directory; docker rejects a dup mount."""
    bundle = copy.deepcopy(BUNDLE)
    bundle["declared_inputs"][1]["uri"] = "/w/ng"
    assert mod.render(bundle, CTX)["argv"].count("/w/ng:/w/ng:ro") == 1


def test_glue_stage_requests_no_gpu(mod):
    assert "--gpus" not in mod.render(BUNDLE, CTX)["argv"]


def test_gpu_stage_requests_a_gpu(mod):
    bundle = {**BUNDLE, "compute_shape": {"gpus": 1, "nodes": 1}}
    argv = mod.render(bundle, CTX)["argv"]
    assert argv[argv.index("--gpus") + 1] == "all"


def test_command_and_args_follow_the_image(mod):
    argv = mod.render(BUNDLE, CTX)["argv"]
    at = argv.index(BUNDLE["image"])
    assert argv[at + 1 : at + 3] == ["python", "/opt/prepare.py"]
    assert argv[-2:] == ["--output", "/r/dataset.csv"]


def test_airgap_forbids_an_implicit_pull(mod):
    """Air-gap is a docker convention here, so it belongs in this renderer."""
    argv = mod.render(BUNDLE, {**CTX, "airgap": True})["argv"]
    assert "--pull=never" in argv
    assert "--env=HF_HUB_OFFLINE=1" in argv


def test_networked_mode_adds_no_offline_flags(mod):
    argv = mod.render(BUNDLE, CTX)["argv"]
    assert "--pull=never" not in argv


def test_credentials_pass_by_name_only(mod):
    argv = mod.render(BUNDLE, {**CTX, "env_passthrough": ["HF_TOKEN"]})["argv"]
    assert "HF_TOKEN" in argv
    assert not any(a.startswith("HF_TOKEN=") for a in argv)


@pytest.mark.parametrize("native,code,expected", [
    ("created", 0, "PENDING"), ("running", 0, "RUNNING"), ("paused", 0, "RUNNING"),
])
def test_state_vocabulary(mod, native, code, expected):
    assert mod.STATE_VOCAB[native] == expected


def test_terminal_states_are_resolved_by_exit_code(mod):
    """`exited` is COMPLETE or ERROR depending on the code, so it is not a
    static table entry."""
    assert "exited" not in mod.STATE_VOCAB
