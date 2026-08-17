#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendering a spec-bundle to a local `docker run`.

A DEFT stage that emits raw argv can only ever run on docker. A stage that
emits a spec-bundle describes *what* it needs -- image, command, inputs by URI,
outputs, compute shape -- and each platform renders the *how*. This covers the
docker rendering plus the guards that keep a bundle honest about the frame it
will run in.

The glue case is the one that matters for portability: roughly half of DEFT's
stages are CPU-only host-Python today (`deft_python.sh` selects a *host*
Python), and they copy images and write CSVs. On docker-local the host is the
compute frame so that is invisible; on SLURM or kubernetes it writes to the
wrong machine. Expressed as a bundle with compute_shape.gpus == 0 the same step
runs in the compute frame on every platform.
"""

from __future__ import annotations

import copy
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
MODULE = REPO / "skills/applications/tao-run-deft-aoi/scripts/deft_exec.py"

GLUE = {
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
RESULTS = "/r"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("deft_exec_bundle", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with(**changes):
    bundle = copy.deepcopy(GLUE)
    bundle.update(changes)
    return bundle


def test_inputs_mount_at_the_same_absolute_path(mod):
    """The workspace invariant: a path in a CSV must resolve on both sides."""
    argv = mod.render_docker(GLUE, RESULTS)
    assert "-v" in argv and "/w/ng:/w/ng:ro" in argv
    assert "/w/ok:/w/ok:ro" in argv


def test_results_dir_is_mounted_writable(mod):
    """Outputs must land in the compute frame, not the agent's filesystem."""
    argv = mod.render_docker(GLUE, RESULTS)
    assert f"{RESULTS}:{RESULTS}" in argv
    assert f"{RESULTS}:{RESULTS}:ro" not in argv


def test_glue_stage_requests_no_gpu(mod):
    """compute_shape.gpus == 0 is a real case, not a degenerate one."""
    assert "--gpus" not in mod.render_docker(GLUE, RESULTS)


def test_gpu_stage_requests_a_gpu(mod):
    argv = mod.render_docker(_with(compute_shape={"gpus": 1, "nodes": 1}), RESULTS)
    assert argv[argv.index("--gpus") + 1] == "all"


@pytest.mark.parametrize("uri", ["s3://b/k", "hf://org/model", "ngc://nvidia/tao/x:1"])
def test_unstaged_uris_are_refused(mod, uri):
    """tao-data-io stages these; rendering must not pretend they are paths.

    Silently dropping them would produce a container that starts, finds no
    data, and fails deep inside the workload instead of at submit.
    """
    bundle = copy.deepcopy(GLUE)
    bundle["declared_inputs"][0]["uri"] = uri
    with pytest.raises(ValueError, match="tao-data-io"):
        mod.render_docker(bundle, RESULTS)


def test_relative_input_is_refused(mod):
    """A relative path means something different on each side of the mount."""
    bundle = copy.deepcopy(GLUE)
    bundle["declared_inputs"][0]["uri"] = "data/ng"
    with pytest.raises(ValueError, match="absolute"):
        mod.render_docker(bundle, RESULTS)


def test_duplicate_inputs_mount_once(mod):
    """Two spec_keys can point at one directory; docker rejects a dup mount."""
    bundle = copy.deepcopy(GLUE)
    bundle["declared_inputs"][1]["uri"] = "/w/ng"
    argv = mod.render_docker(bundle, RESULTS)
    assert argv.count("/w/ng:/w/ng:ro") == 1


def test_command_and_args_are_appended_after_the_image(mod):
    argv = mod.render_docker(GLUE, RESULTS)
    image_at = argv.index(GLUE["image"])
    assert argv[image_at + 1 : image_at + 3] == ["python", "/opt/prepare.py"]
    assert argv[-2:] == ["--output", "/r/dataset.csv"]


def test_bundle_is_linted_before_rendering(mod, tmp_path):
    """A malformed bundle must fail on the laptop, not in the allocation."""
    bad = tmp_path / "b.json"
    bad.write_text('{"action": "x"}')
    with pytest.raises(ValueError, match="not a valid spec-bundle"):
        mod.load_bundle(bad)


# ── Air-gap policy on the bundle ────────────────────────────────────────────
# The argv gate reasons about a local docker/podman command line, which is why
# it refuses launchers it cannot inspect. Asking the same question of the
# bundle's data answers it for platforms whose argv this module never sees.

AIRGAP = {"network_mode": "airgap"}
NETWORKED = {"network_mode": "network-enabled"}


@pytest.mark.parametrize(
    "uri",
    ["s3://b/k", "hf://org/m", "ngc://nvidia/tao/x:1",
     "https://example/x.tar", "gs://b/k", "azure://c/k"],
)
def test_airgap_refuses_a_fetchable_input(mod, uri):
    bundle = copy.deepcopy(GLUE)
    bundle["declared_inputs"][0]["uri"] = uri
    with pytest.raises(ValueError, match="air-gap"):
        mod.reject_airgap_bundle(bundle, AIRGAP)


def test_airgap_allows_pre_staged_paths(mod):
    """Air-gap restricts fetching, not running — tier A must still work."""
    mod.reject_airgap_bundle(GLUE, AIRGAP)


def test_networked_mode_allows_fetchable_inputs(mod):
    """network-enabled is the default; the fetch is tao-data-io's job to do."""
    bundle = copy.deepcopy(GLUE)
    bundle["declared_inputs"][0]["uri"] = "s3://b/k"
    mod.reject_airgap_bundle(bundle, NETWORKED)


def test_policy_and_rendering_refuse_for_different_reasons(mod):
    """Two layers, two questions: may you fetch, and can docker mount it.

    Both reject an s3:// input, but only the policy layer is about the network.
    Collapsing them would make the air-gap refusal disappear the moment a
    platform that *can* consume a URI is added.
    """
    bundle = copy.deepcopy(GLUE)
    bundle["declared_inputs"][0]["uri"] = "s3://b/k"
    with pytest.raises(ValueError, match="air-gap"):
        mod.reject_airgap_bundle(bundle, AIRGAP)
    with pytest.raises(ValueError, match="tao-data-io"):
        mod.render_docker(bundle, RESULTS)
