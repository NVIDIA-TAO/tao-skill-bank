#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What deft_exec still owns once rendering moved to the platforms.

Rendering lives with each platform skill (see test_platform_render_contract.py
and test_render_docker.py). What remains here is the workflow-side half: lint
the bundle, apply execution policy to its DATA, and resolve --platform to the
skill that owns the rendering -- by convention, never a table.

The glue case is why this matters: roughly half of DEFT's stages are CPU-only
host-Python today (`deft_python.sh` selects a *host* Python) and they copy
images and write CSVs. On docker-local the host is the compute frame so that is
invisible; on SLURM or kubernetes it writes to the wrong machine. Expressed as
a bundle with compute_shape.gpus == 0, the same step runs in the compute frame
on every platform.
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
        mod.platform_renderer("docker").render(
            bundle, {"job_id": "j-1", "results_dir": RESULTS, "bank": str(REPO)}
        )


# ── Renderer discovery ──────────────────────────────────────────────────────

def test_resolves_a_platform_by_convention(mod):
    """A slug resolves to the skill that owns it; no list lives here."""
    assert mod.platform_renderer("docker").PLATFORM == "docker"
    assert mod.platform_renderer("slurm").PLATFORM == "slurm"


def test_unknown_platform_names_the_contract(mod):
    """An external platform joins by shipping the module, so say which one."""
    with pytest.raises(ValueError, match="bundle-rendering.md"):
        mod.platform_renderer("nonexistent")
