#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every execution platform renders a spec-bundle the same way.

Rendering is platform-owned on purpose: a table of render_docker/render_slurm/
... inside a consumer would have to be edited before any new platform could
run, which is exactly the registry the four-verb contract avoids. `--platform`
is an open validated slug, so a platform joins by shipping the module described
in tao-launch-workflow/references/bundle-rendering.md.

A contract nothing checks is a suggestion. This is the check: discover the
platforms from disk (never a hardcoded list -- a hardcoded list here would
reintroduce the registry through the back door) and hold each to the same
obligations.
"""

from __future__ import annotations

import copy
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
PLATFORM_DIR = REPO / "skills/platform"

# Support skills, not execution backends: they implement no verbs.
NOT_EXECUTION = {"tao-data-io", "tao-setup-nvidia-gpu-host"}

GPU_BUNDLE = {
    "network_arch": "visual_changenet",
    "action": "iter1.train",
    "image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
    "mode": "args",
    "command": "visual_changenet train",
    "args": ["-e", "/w/spec.yaml"],
    "declared_inputs": [{"spec_key": "d", "type": "folder", "uri": "/w/data"}],
    "declared_outputs": [{"spec_key": "o", "type": "folder"}],
    "compute_shape": {"gpus": 1, "nodes": 1},
}
GLUE_BUNDLE = {
    **GPU_BUNDLE,
    "action": "data_merge.pair_prepare",
    "image": "docker.io/library/python:3.11-slim",
    "command": "python /w/prepare.py",
    "args": ["--out", "/r/dataset.csv"],
    "compute_shape": {"gpus": 0, "nodes": 1},
}


def _discovered() -> list[str]:
    found = []
    for entry in sorted(PLATFORM_DIR.iterdir()):
        if not entry.is_dir() or entry.name in NOT_EXECUTION:
            continue
        if (entry / "SKILL.md").is_file():
            found.append(entry.name)
    return found


PLATFORMS = _discovered()


def _load(name: str):
    path = PLATFORM_DIR / name / "references/render.py"
    if not path.is_file():
        pytest.fail(
            f"{name} ships no references/render.py — it cannot run a spec-bundle. "
            "See tao-launch-workflow/references/bundle-rendering.md"
        )
    spec = importlib.util.spec_from_file_location(f"render_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _for(module, bundle):
    """`image` is platform-shaped: a registry URI for containers, the venv
    interpreter for virtualenv. The bundle is portable; that one field is not."""
    if module.PLATFORM == "virtualenv":
        return {**bundle, "image": "/opt/venv/bin/python"}
    return bundle


def _ctx(tmp_path, **extra):
    base = {
        "job_id": "visual_changenet-iter1.train-abc123",
        "results_dir": "/r",
        "bank": str(REPO),
        "login": "user@login",          # slurm
        "instance": "gpu-box",          # brev
        "namespace": "tao",             # kubernetes
        "mount_path": "/",              # kubernetes: accept the fixture paths
    }
    base.update(extra)
    return base


def test_platforms_were_discovered():
    """A hardcoded list here would be the registry we are avoiding."""
    assert len(PLATFORMS) >= 5, f"expected the bank's execution platforms, got {PLATFORMS}"


@pytest.mark.parametrize("name", PLATFORMS)
def test_ships_a_renderer(name):
    _load(name)


@pytest.mark.parametrize("name", PLATFORMS)
def test_platform_constant_matches_the_directory(name):
    """PLATFORM must match the --platform slug, which is derived from the dir."""
    module = _load(name)
    assert module.PLATFORM == name.replace("tao-run-on-", "")


@pytest.mark.parametrize("name", PLATFORMS)
def test_renders_a_glue_bundle(name, tmp_path):
    """gpus == 0 is first-class: CPU-only glue is an ordinary stage."""
    module = _load(name)
    out = module.render(_for(module, GLUE_BUNDLE), _ctx(tmp_path))
    assert set(out) >= {"files", "argv", "backend_ref"}
    assert out["argv"], "render produced no command"


@pytest.mark.parametrize("name", PLATFORMS)
def test_renders_a_gpu_bundle_or_refuses_with_a_reason(name, tmp_path):
    """Only virtualenv legitimately cannot place a GPU, and it must say so."""
    module = _load(name)
    if module.PLATFORM == "virtualenv":
        with pytest.raises(ValueError, match="GPU"):
            module.render(GPU_BUNDLE, _ctx(tmp_path))
        return
    assert module.render(GPU_BUNDLE, _ctx(tmp_path))["argv"]


@pytest.mark.parametrize("name", PLATFORMS)
def test_names_the_backend_object_after_the_job_id(name, tmp_path):
    """The later verbs locate the job by name; anything else is unreachable."""
    module = _load(name)
    ctx = _ctx(tmp_path)
    out = module.render(_for(module, GLUE_BUNDLE), ctx)
    rendered = " ".join(out["argv"]) + " ".join(out["files"].values()) + str(out["backend_ref"])
    assert ctx["job_id"] in rendered, f"{name} does not name the object after job_id"


@pytest.mark.parametrize("name", PLATFORMS)
def test_refuses_an_unstaged_uri(name, tmp_path):
    """Staging is tao-data-io's job; dropping a URI fails deep in the workload."""
    module = _load(name)
    bundle = copy.deepcopy(_for(module, GLUE_BUNDLE))
    bundle["declared_inputs"][0]["uri"] = "s3://bucket/data"
    with pytest.raises(ValueError, match="tao-data-io"):
        module.render(bundle, _ctx(tmp_path))


@pytest.mark.parametrize("name", PLATFORMS)
def test_status_maps_into_the_fixed_vocabulary(name):
    """Native sub-states ride in the record message, never in the state."""
    module = _load(name)
    vocab = {"PENDING", "RUNNING", "COMPLETE", "ERROR", "CANCELED", "UNKNOWN"}
    table = getattr(module, "STATE_VOCAB", None)
    if table is None:
        pytest.skip(f"{name} derives status without a static table")
    assert set(table.values()) <= vocab, f"{name} emits states outside the vocabulary"


@pytest.mark.parametrize("name", PLATFORMS)
def test_no_credential_value_is_rendered(name, tmp_path):
    """Credentials travel by name; a value in argv lands in the process table."""
    module = _load(name)
    ctx = _ctx(tmp_path, env_passthrough=["HF_TOKEN", "NGC_KEY"])
    out = module.render(_for(module, GLUE_BUNDLE), ctx)
    blob = " ".join(out["argv"]) + " ".join(out["files"].values())
    for name_ in ("HF_TOKEN", "NGC_KEY"):
        assert f"{name_}=" not in blob, f"{name} put {name_} on the command line"
