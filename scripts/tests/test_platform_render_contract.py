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
def test_ships_prepare(name):
    """Getting the image into native form is the platform's job too."""
    module = _load(name)
    assert callable(getattr(module, "prepare", None)), (
        f"{name} ships no prepare(); the image fetch would land inside the "
        "metered allocation. See bundle-rendering.md"
    )


@pytest.mark.parametrize("name", PLATFORMS)
def test_prepare_is_idempotent_when_the_image_is_present(name, tmp_path, monkeypatch):
    """The common path must do NO work: conversion and pulls are one-time."""
    module = _load(name)
    if module.PLATFORM == "virtualenv":
        interpreter = tmp_path / "bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("")
        out = module.prepare({"image": str(interpreter)}, _ctx(tmp_path))
        assert out["image"] == str(interpreter)
        return
    if module.PLATFORM == "slurm":
        # An image already given as a .sqsh needs no conversion at all.
        out = module.prepare({"image": "/lustre/images/tao.sqsh"},
                             _ctx(tmp_path, sqsh_dir="/lustre/images"))
        assert out["image"].endswith(".sqsh")
        return
    if module.PLATFORM == "kubernetes":
        out = module.prepare(GPU_BUNDLE, _ctx(tmp_path))
        assert out["image"] == GPU_BUNDLE["image"]
        return
    # docker / brev shell out; assert they report "already present" without pulling.
    calls: list[list[str]] = []

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake(cmd, *a, **k):
        calls.append(cmd)
        return _Done()

    monkeypatch.setattr(module.subprocess, "run", _fake)
    module.prepare(GPU_BUNDLE, _ctx(tmp_path))
    assert not any("pull" in " ".join(c) for c in calls), (
        f"{name} pulled an image that was already present"
    )


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


# ── SLURM sqsh caching ──────────────────────────────────────────────────────
# Conversion is one-time-per-image and expensive, so the decision logic is what
# matters. `test -e` is not enough: a conversion killed by a wall-time cap
# leaves a TRUNCATED file that exists. The SKILL.md claims the sqsh "is
# validated by hsqs magic" -- this is that guard.

class _Done:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


def _slurm_with(monkeypatch, outputs):
    module = _load("tao-run-on-slurm")
    seq, calls = list(outputs), []

    def fake(cmd, *a, **k):
        calls.append(" ".join(cmd))
        return _Done(seq.pop(0) if seq else "")

    monkeypatch.setattr(module.subprocess, "run", fake)
    ctx = {"login": "me@login", "sqsh_dir": "/lustre/img"}
    return module, ctx, calls


def test_sqsh_valid_cache_is_reused_without_converting(monkeypatch):
    module, ctx, calls = _slurm_with(monkeypatch, ["hsqs"])
    out = module.prepare({"image": "nvcr.io/nvidia/tao:1"}, ctx)
    assert out["image"].endswith(".sqsh")
    assert not any("enroot import" in c for c in calls), "reconverted a valid sqsh"


def test_sqsh_missing_is_converted(monkeypatch):
    module, ctx, calls = _slurm_with(monkeypatch, ["", "", "hsqs"])
    module.prepare({"image": "nvcr.io/nvidia/tao:1"}, ctx)
    assert any("enroot import" in c for c in calls)


def test_truncated_sqsh_is_reconverted(monkeypatch):
    """The failure mode the shipped cpu/30min defaults actually produce."""
    module, ctx, calls = _slurm_with(monkeypatch, ["hsq", "", "hsqs"])
    out = module.prepare({"image": "nvcr.io/nvidia/tao:1"}, ctx)
    assert "corrupt or truncated" in out["notes"][0]
    assert any("enroot import" in c for c in calls)


def test_failed_conversion_never_falls_back_to_the_registry(monkeypatch):
    """Falling back puts the pull inside the GPU allocation, exactly when
    something is already wrong."""
    module, ctx, _ = _slurm_with(monkeypatch, ["", "", "hsq"])
    with pytest.raises(ValueError, match="no valid squashfs"):
        module.prepare({"image": "nvcr.io/nvidia/tao:1"}, ctx)


def test_airgap_refuses_to_convert(monkeypatch):
    module, ctx, _ = _slurm_with(monkeypatch, [""])
    with pytest.raises(ValueError, match="air-gap"):
        module.prepare({"image": "nvcr.io/nvidia/tao:1"}, {**ctx, "airgap": True})


def test_conversion_always_passes_an_explicit_time_limit(monkeypatch):
    """The trap is the partition DEFAULT, not its maximum.

    On CS-OCI-ORD every partition has DefaultTime=00:31:00 while cpu allows a
    full day, so a conversion submitted without an explicit -t is capped at 31
    minutes whichever partition it lands on. Changing partition alone fixes
    nothing; the explicit -t is the fix.
    """
    module, ctx, calls = _slurm_with(monkeypatch, ["", "", "hsqs"])
    module.prepare({"image": "nvcr.io/nvidia/tao:1"}, ctx)
    convert = next(c for c in calls if "enroot import" in c)
    assert "-t " in convert, "no explicit time limit: the 31-minute default applies"
    minutes = int(convert.split("-t ")[1].split()[0])
    assert minutes > 31, f"-t {minutes} is under the 31-minute default; it caps tighter"


def test_shipped_conversion_defaults_clear_the_partition_default():
    """skill_info.yaml must not ship a timeout below DefaultTime."""
    import yaml

    info = yaml.safe_load(
        (REPO / "skills/platform/tao-run-on-slurm/references/skill_info.yaml")
        .read_text(encoding="utf-8")
    )
    minutes = info["resource_defaults"]["sqsh_conversion_timeout_minutes"]
    assert minutes > 31, (
        f"sqsh_conversion_timeout_minutes={minutes} is at or under the cluster "
        "DefaultTime of 31 minutes, so it caps the job tighter than passing "
        "nothing would — the truncated-sqsh failure mode"
    )


# ── Scheduling identity reaches every allocation ────────────────────────────
# Found end-to-end on a real cluster: prepare()'s image conversion is a second,
# separate SLURM allocation, and it carried no --account while the job's own
# sbatch did. The scheduler rejected it with "You forgot to specify which
# account you want to use", surfacing as "enroot import failed" — which reads
# as a broken conversion rather than a missing cluster setting.
#
# The general defect is that this renderer emits MORE THAN ONE allocation, so
# any per-cluster scheduling requirement has to reach all of them. These pin
# that without naming a cluster: the values are fixtures, not defaults.

SCHEDULED_CTX = {
    "account": "some-account",
    "qos": "some-qos",
    "partition": "some-partition",
    "login": "user@login",
    "sqsh_dir": "/lustre/img",
}


def _slurm():
    return _load("tao-run-on-slurm")


def test_conversion_allocation_carries_scheduling_identity(monkeypatch):
    """The conversion srun is an allocation like any other."""
    module = _slurm()
    seq, calls = ["", "", "hsqs"], []

    def fake(cmd, *a, **k):
        calls.append(" ".join(cmd))
        return _Done(seq.pop(0) if seq else "")

    monkeypatch.setattr(module.subprocess, "run", fake)
    module.prepare({"image": "nvcr.io/x/y:1"}, dict(SCHEDULED_CTX))
    convert = next(c for c in calls if "enroot import" in c)
    assert "-A some-account" in convert, (
        "the conversion allocation omits the account; the scheduler rejects it "
        "and the failure reads as a broken enroot import"
    )
    assert "--qos some-qos" in convert


def test_job_allocation_carries_the_same_identity(tmp_path):
    """One ctx, both allocations — they cannot drift apart."""
    module = _slurm()
    body = list(module.render(GLUE_BUNDLE, _ctx(tmp_path, **SCHEDULED_CTX))["files"].values())[0]
    assert "#SBATCH --account=some-account" in body
    assert "#SBATCH --qos=some-qos" in body
    assert "#SBATCH --partition=some-partition" in body


def test_no_cluster_values_are_baked_in(tmp_path):
    """A renderer with no scheduling ctx must emit none — not a default site."""
    module = _slurm()
    ctx = _ctx(tmp_path, login="user@login", sqsh_dir="/lustre/img")
    body = list(module.render(GLUE_BUNDLE, ctx)["files"].values())[0]
    assert "--account=" not in body and "--qos=" not in body
    assert module.scheduling_srun_flags({}) == []


def test_free_form_sbatch_extra_still_works(tmp_path):
    """Derived directives must not displace whatever else a caller passes."""
    module = _slurm()
    ctx = _ctx(tmp_path, sbatch_extra="#SBATCH --exclusive", **SCHEDULED_CTX)
    body = list(module.render(GLUE_BUNDLE, ctx)["files"].values())[0]
    assert "#SBATCH --exclusive" in body
    assert "#SBATCH --account=some-account" in body


# ── Image reference translation ─────────────────────────────────────────────
# Found end-to-end: the spec-bundle mandates a fully-qualified registry/path:tag
# and docker consumes that directly, but enroot separates the registry with '#'.
# Passing the canonical form through unchanged made enroot treat the registry
# host as the first path segment, requesting
# /v2/docker.io/library/alpine/manifests/3.20 and failing 401 — which reads as
# a credentials problem, not a syntax one.
#
# The general rule: a platform's native image reference is not necessarily the
# bundle's canonical one, and the renderer owns that translation.

@pytest.mark.parametrize("canonical,expected", [
    ("docker.io/library/alpine:3.20", "docker.io#library/alpine:3.20"),
    ("nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt", "nvcr.io#nvidia/tao/tao-toolkit:7.1.0-pyt"),
    ("localhost:5000/team/img:1", "localhost:5000#team/img:1"),
])
def test_registry_is_separated_for_enroot(canonical, expected):
    assert _slurm().enroot_uri(canonical) == expected


def test_registry_relative_reference_passes_through():
    """No registry component means nothing to separate."""
    assert _slurm().enroot_uri("alpine:3.20") == "alpine:3.20"


def test_already_translated_reference_is_untouched():
    """Idempotent, so a caller that pre-translated is not double-mangled."""
    assert _slurm().enroot_uri("myregistry.io#team/img:1") == "myregistry.io#team/img:1"


def test_conversion_uses_the_translated_reference(monkeypatch):
    """The bug was in the command, so assert on the command."""
    module = _slurm()
    seq, calls = ["", "", "hsqs"], []

    def fake(cmd, *a, **k):
        calls.append(" ".join(cmd))
        return _Done(seq.pop(0) if seq else "")

    monkeypatch.setattr(module.subprocess, "run", fake)
    module.prepare({"image": "docker.io/library/alpine:3.20"},
                   {"login": "u@h", "sqsh_dir": "/lustre/img"})
    convert = next(c for c in calls if "enroot import" in c)
    assert "docker://docker.io#library/alpine:3.20" in convert
    assert "docker://docker.io/library" not in convert, (
        "registry left in the path; enroot requests /v2/<registry>/... and 401s"
    )
