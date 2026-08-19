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
import inspect
import pathlib
import re

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


# A fixture cluster, not this repo's: two CPU queues and a GPU one, so
# partition choice is actually exercised. Fields are sinfo's %R|%l|%L|%a|%G.
FIXTURE_SINFO = (
    "batch*|2-00:00:00|00:31:00|up|(null)\n"
    "short|4:00:00|00:31:00|up|(null)\n"
    "accel|infinite|00:31:00|up|gpu:8\n"
)


def _slurm_with(monkeypatch, magics, sinfo=FIXTURE_SINFO, convert_rc=0):
    """Answer by COMMAND, not by call order.

    The positional version broke four unrelated tests the moment prepare()
    grew one more ssh call -- the fake encoded a call sequence nothing had
    promised to keep. Dispatching on the command says what each response IS,
    so adding a probe cannot silently re-target another command's answer.
    """
    module = _load("tao-run-on-slurm")
    pending, calls = list(magics), []

    def fake(cmd, *a, **k):
        joined = " ".join(cmd)
        calls.append(joined)
        if "sinfo" in joined:
            return _Done(sinfo)
        if "head -c4" in joined:
            return _Done(pending.pop(0) if pending else "")
        if "enroot import" in joined:
            return _Done("", convert_rc)
        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake)
    ctx = {"login": "me@login", "sqsh_dir": "/lustre/img"}
    return module, ctx, calls


def test_sqsh_valid_cache_is_reused_without_converting(monkeypatch):
    module, ctx, calls = _slurm_with(monkeypatch, ["hsqs"])
    out = module.prepare({"image": "nvcr.io/nvidia/tao:1"}, ctx)
    assert out["image"].endswith(".sqsh")
    assert not any("enroot import" in c for c in calls), "reconverted a valid sqsh"


def test_sqsh_missing_is_converted(monkeypatch):
    module, ctx, calls = _slurm_with(monkeypatch, ["", "hsqs"])
    module.prepare({"image": "nvcr.io/nvidia/tao:1"}, ctx)
    assert any("enroot import" in c for c in calls)


def test_truncated_sqsh_is_reconverted(monkeypatch):
    """The failure mode the shipped cpu/30min defaults actually produce."""
    module, ctx, calls = _slurm_with(monkeypatch, ["hsq", "hsqs"])
    out = module.prepare({"image": "nvcr.io/nvidia/tao:1"}, ctx)
    assert "corrupt or truncated" in out["notes"][0]
    assert any("enroot import" in c for c in calls)


def test_failed_conversion_never_falls_back_to_the_registry(monkeypatch):
    """Falling back puts the pull inside the GPU allocation, exactly when
    something is already wrong."""
    module, ctx, _ = _slurm_with(monkeypatch, ["", "hsq"])
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
    module, ctx, calls = _slurm_with(monkeypatch, ["", "hsqs"])
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
    module, ctx, calls = _slurm_with(monkeypatch, ["", "hsqs"])
    module.prepare({"image": "nvcr.io/x/y:1"}, {**ctx, **SCHEDULED_CTX})
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
    module, ctx, calls = _slurm_with(monkeypatch, ["", "hsqs"])
    module.prepare({"image": "docker.io/library/alpine:3.20"}, ctx)
    convert = next(c for c in calls if "enroot import" in c)
    assert "docker://docker.io#library/alpine:3.20" in convert
    assert "docker://docker.io/library" not in convert, (
        "registry left in the path; enroot requests /v2/<registry>/... and 401s"
    )


# ── The skill's documentation is part of the contract ───────────────────────
# The enroot bug is the one worth generalising from. SKILL.md documented the
# correct form -- `docker://<registry>#<image>:<tag>` -- and render.py emitted
# `docker://<image>` anyway. The skill was RIGHT and the code ignored it, and
# because nothing tied a documented command to the rendered one, 66 tests
# passed while the only real cluster run 401'd.
#
# So: assert the documented command shapes and the emitted ones agree.

SLURM_DIR = REPO / "skills/platform/tao-run-on-slurm"


def _slurm_docs() -> str:
    """SKILL.md plus its references -- prose moves between them for size."""
    files = [SLURM_DIR / "SKILL.md", *sorted((SLURM_DIR / "references").glob("*.md"))]
    return "\n".join(f.read_text(encoding="utf-8") for f in files)


def test_documented_enroot_form_separates_the_registry():
    """Guard the documentation itself; the code test below points back here."""
    assert "docker://<registry>#<image>:<tag>" in _slurm_docs(), (
        "the skill no longer documents enroot's registry separator; if this "
        "moved, move test_rendered_enroot_matches_documentation with it"
    )


def test_rendered_enroot_matches_documentation(monkeypatch):
    """What we emit must instantiate what we document."""
    module, ctx, calls = _slurm_with(monkeypatch, ["", "hsqs"])
    module.prepare({"image": "nvcr.io/nvidia/tao/toolkit:7.1.0"}, ctx)
    convert = next(c for c in calls if "enroot import" in c)
    uri = re.search(r"docker://(\S+)", convert).group(1)
    registry, sep, rest = uri.partition("#")
    assert sep, f"documented form is <registry>#<image>:<tag>, emitted {uri!r}"
    assert "/" not in registry, f"registry {registry!r} leaked path components"
    assert rest, "no image path after the separator"


# ── Schedulability is a prerequisite, not a submit-time surprise ────────────
# Preflight verified reachability (ssh, enroot credentials, storage) but never
# schedulability, so both real failures -- a missing account and a partition
# assumption -- were structurally invisible to it.

def test_sinfo_is_parsed_in_the_cluster_s_own_dialect():
    """sinfo prints `31:00` as MM:SS; the requested-limit parser reads HH:MM."""
    module = _slurm()
    assert module.parse_sinfo_minutes("31:00") == 31
    assert module.parse_sinfo_minutes("1-00:00:00") == 1440
    assert module.parse_sinfo_minutes("infinite") is None, (
        "unbounded must be None, not a number that later clamps a ceiling down"
    )
    assert module.parse_sinfo_minutes("garbage") is None


def test_unknown_partition_is_rejected_with_the_real_list():
    """`invalid partition specified` does not say what IS valid."""
    module = _slurm()
    found = {"cpu": {"max_minutes": 1440, "available": True, "gres": ""}}
    with pytest.raises(ValueError, match="available: cpu"):
        module.require_partition(found, "polar3")


def test_conversion_avoids_gpu_partitions():
    """Conversion is CPU work; it should not idle in a GPU queue."""
    module = _slurm()
    found = {
        "gpu": {"max_minutes": 10000, "available": True, "gres": "gpu:8"},
        "cpu": {"max_minutes": 1440, "available": True, "gres": ""},
        "dead": {"max_minutes": None, "available": False, "gres": ""},
    }
    assert module.choose_conversion_partition(found) == "cpu"


def test_no_partition_is_invented_when_discovery_finds_nothing():
    """An ssh blip must not look like a cluster with a partition named 'cpu'."""
    module = _slurm()
    assert module.choose_conversion_partition({}) is None


def test_ceiling_is_clamped_to_what_the_partition_grants():
    """Over-asking is rejected at submit, so it does not fail safe."""
    module = _slurm()
    found = {"short": {"max_minutes": 60, "available": True, "gres": ""}}
    assert module.conversion_minutes(found, "short", 120) == 60
    assert module.conversion_minutes({}, None, 120) == 120


def test_explicit_time_survives_an_unknown_partition(monkeypatch):
    """DefaultTime, not MaxTime, is the trap -- -t must always be emitted."""
    module, ctx, calls = _slurm_with(monkeypatch, ["", "hsqs"], sinfo="")
    module.prepare({"image": "alpine:3.20"}, ctx)
    convert = next(c for c in calls if "enroot import" in c)
    # Only the srun flags: the script body legitimately contains `mkdir -p`.
    flags = convert.split(" bash -c ")[0]
    assert " -p " not in flags, "invented a partition discovery never found"
    assert re.search(r"-t \d+", convert), f"no explicit wall limit in {convert!r}"


# ── The rendered command must satisfy the documented requirements ───────────
# Third instance of one pattern: SKILL.md stated a requirement, render.py did
# not implement it, and the only test looked at the PROSE. Documentation tests
# prove the skill says the right thing; they prove nothing about what runs.
#
# Left unset, enroot unpacks layers into the submit CWD -- a quota'd Lustre or
# home path -- and dies mid-layer with `curl: (23) Failed writing body`, which
# reads as a network fault rather than a placement one.

def _conversion_command(monkeypatch) -> str:
    module, ctx, calls = _slurm_with(monkeypatch, ["", "hsqs"])
    module.prepare({"image": "nvcr.io/nvidia/tao:1"}, ctx)
    return next(c for c in calls if "enroot import" in c)


@pytest.mark.parametrize("required", [
    "TMPDIR=/tmp",                              # not the quota'd submit CWD
    "ENROOT_TEMP_PATH",                         # what direct enroot reads
    "SLURM_ENROOT_TEMP_PATH",                   # what Pyxis may read instead
    "--chdir=/tmp",                             # srun's own CWD
])
def test_conversion_command_places_scratch_on_node_local_disk(monkeypatch, required):
    assert required in _conversion_command(monkeypatch), (
        f"the rendered conversion command omits {required}, which this skill "
        "documents as required; enroot then writes layers to the submit CWD"
    )


def test_conversion_scratch_is_job_unique(monkeypatch):
    """A fixed path is deleted by cleanup from another allocation mid-import."""
    convert = _conversion_command(monkeypatch)
    assert "SLURM_JOB_ID" in convert, (
        "scratch dir is not job-unique; a concurrent allocation's cleanup "
        "removes it and enroot fails whiteout conversion after fetching every "
        "layer -- the expensive way to fail"
    )


def test_conversion_scratch_is_cleaned_up(monkeypatch):
    """Node-local scratch is not reclaimed for us; retries would accumulate."""
    assert "trap" in _conversion_command(monkeypatch)


def test_documented_temp_path_requirements_are_all_rendered(monkeypatch):
    """Couple the two directly: whatever the docs mandate must be emitted.

    This is the guard that was missing. The prose test and the render test
    existed independently, so the skill could document a requirement the
    renderer ignored and both would stay green.
    """
    docs = _slurm_docs()
    convert = _conversion_command(monkeypatch)
    for var in ("ENROOT_TEMP_PATH", "SLURM_ENROOT_TEMP_PATH", "TMPDIR"):
        if var in docs:
            assert var in convert, (
                f"{var} is documented as required but never rendered"
            )


# ── Failure messages must name the measured cause ───────────────────────────
# Measured on enroot 3.4.1 against Docker Hub: auth succeeds, the manifest
# returns 200 with valid JSON, /tmp has 83G free -- and the import still fails
# with "Could not process JSON input". The document is an OCI image index,
# which older enroot cannot parse, and Docker Hub serves it even when the
# client's Accept header offers ONLY the Docker manifest-list type. So there is
# no request-side workaround, and a message that sends the reader to check disk
# space or credentials costs a cluster round trip to disprove.

def _conversion_failure(monkeypatch, stderr: str) -> str:
    module, ctx, _ = _slurm_with(monkeypatch, [""], convert_rc=1)

    def failing(cmd, *a, **k):
        joined = " ".join(cmd)
        if "sinfo" in joined:
            return _Done(FIXTURE_SINFO)
        if "enroot import" in joined:
            done = _Done("")
            done.returncode, done.stderr = 1, stderr
            return done
        return _Done("")

    monkeypatch.setattr(module.subprocess, "run", failing)
    with pytest.raises(ValueError) as excinfo:
        module.prepare({"image": "docker.io/library/alpine:3.20"}, ctx)
    return str(excinfo.value)


@pytest.mark.parametrize("phrase", ["OCI image index", "Accept header", "upgrade enroot"])
def test_manifest_failure_explains_itself(monkeypatch, phrase):
    message = _conversion_failure(monkeypatch, "[ERROR] Could not process JSON input")
    assert phrase in message, f"the diagnosis omits {phrase!r}"


def test_manifest_failure_does_not_blame_disk_space(monkeypatch):
    """It did, and /tmp had 83G free; the advice cost a round trip."""
    message = _conversion_failure(monkeypatch, "[ERROR] Could not process JSON input")
    assert "free space" not in message, (
        "a manifest parse failure is not a disk problem; sending the reader to "
        "check space is a wrong lead they must disprove on the cluster"
    )


def test_write_failure_still_points_at_space_and_credentials(monkeypatch):
    """The genuinely write-shaped error keeps its own diagnosis."""
    message = _conversion_failure(monkeypatch, "curl: (23) Failed writing body")
    assert "free space" in message and "credentials" in message


def test_conversion_requests_memory(monkeypatch):
    """enroot extracts layers in parallel; peak memory tracks concurrency.

    Without --mem the step takes the partition's per-CPU default and is
    OOM-killed at the extract stage -- after every layer has been downloaded,
    so it wastes the whole fetch. enroot's output stops at "Downloading N
    missing layers...", which makes it read as a download failure.
    """
    assert "--mem" in _conversion_command(monkeypatch)


def test_conversion_memory_is_caller_settable(monkeypatch):
    """Not every site grants the same ceiling, so it must not be baked in."""
    module, ctx, calls = _slurm_with(monkeypatch, ["", "hsqs"])
    module.prepare({"image": "nvcr.io/x:1"}, {**ctx, "conversion_memory_gb": 64})
    assert "--mem 64G" in next(c for c in calls if "enroot import" in c)


# ── A long conversion must be observable while it runs ─────────────────────
# A multi-GB image converts for tens of minutes. Its output streams back over
# ssh into the caller's buffer and is printed only on failure, so from outside
# a slow conversion and a hung one are indistinguishable -- the only recourse
# was polling squeue by hand and guessing.

def test_conversion_writes_a_log_beside_the_image(monkeypatch):
    convert = _conversion_command(monkeypatch)
    assert "tee" in convert and ".import.log" in convert, (
        "the conversion leaves no on-disk trace, so progress cannot be tailed"
    )


def test_conversion_log_does_not_swallow_the_exit_code(monkeypatch):
    """A plain pipeline reports tee's status, masking a failed import."""
    convert = _conversion_command(monkeypatch)
    assert "PIPESTATUS" in convert, (
        "piping to tee makes the shell report tee's exit code; a failed enroot "
        "import would then look like a success and a truncated sqsh would only "
        "be caught later by the hsqs check"
    )


def test_successful_conversion_reports_where_the_log_is(monkeypatch):
    module, ctx, _ = _slurm_with(monkeypatch, ["", "hsqs"])
    out = module.prepare({"image": "nvcr.io/x:1"}, ctx)
    assert out["import_log"].endswith(".import.log")


def test_failed_conversion_names_the_log(monkeypatch):
    """The path is most needed exactly when the summary is too short."""
    assert ".import.log" in _conversion_failure(monkeypatch, "boom")


def test_conversion_allocates_cpus_for_enrootss_own_concurrency(monkeypatch):
    """enroot runs mksquashfs with `-processors 8`; -n1 grants one core.

    The conversion still succeeds, so nothing fails and nothing reports the
    mismatch -- it just takes far longer than the tool was asking to go.
    """
    assert "--cpus-per-task" in _conversion_command(monkeypatch)


def test_conversion_cpus_are_caller_settable(monkeypatch):
    module, ctx, calls = _slurm_with(monkeypatch, ["", "hsqs"])
    module.prepare({"image": "nvcr.io/x:1"}, {**ctx, "conversion_cpus": 4})
    assert "--cpus-per-task 4" in next(c for c in calls if "enroot import" in c)


# ── Rendered files must land where the launcher will look ──────────────────
# Found end-to-end: submit wrote the sbatch script to the LOCAL filesystem
# using the cluster path from ctx, i.e. tried to create /lustre on the launch
# host. It surfaced as `[Errno 30] Read-only file system: '/lustre'` only
# because macOS refuses a write at /. On a Linux launch host the same code
# creates a stray local tree and sbatch fails on a path it cannot see -- or,
# where a writable parent exists, silently submits whatever is already on the
# cluster.
#
# The rule is about the launcher, not about slurm: kubernetes writes a
# manifest that a LOCAL kubectl reads, so a local write is right there.

REMOTE_LAUNCHER_TOKENS = ("ssh", "brev")


def test_remote_launchers_own_file_placement():
    """If argv crosses a machine boundary, so must the files."""
    for name in PLATFORMS:
        module = _load(name)
        argv_source = inspect.getsource(module.render)
        launches_remotely = any(
            f'"{token}"' in argv_source for token in REMOTE_LAUNCHER_TOKENS
        )
        writes_files = '"files": {}' not in argv_source
        if launches_remotely and writes_files:
            assert hasattr(module, "place_files"), (
                f"{name} launches remotely and returns rendered files, but "
                "defines no place_files(); a generic caller writes them to the "
                "launch host, where the launcher will never see them"
            )


def test_slurm_places_files_over_ssh(monkeypatch):
    module = _load("tao-run-on-slurm")
    calls = []

    def fake(cmd, *a, **k):
        calls.append((cmd, k.get("input")))
        return _Done("")

    monkeypatch.setattr(module.subprocess, "run", fake)
    module.place_files({"/lustre/work/job.sbatch": "#!/bin/bash\necho hi\n"},
                       {"login": "me@login"})
    cmd, sent = calls[0]
    assert cmd[0] == "ssh" and cmd[1] == "me@login"
    assert "/lustre/work/job.sbatch" in cmd[2] and "mkdir -p /lustre/work" in cmd[2]
    assert sent == "#!/bin/bash\necho hi\n", "content must travel on stdin"


def test_placed_script_content_never_reaches_argv(monkeypatch):
    """A rendered sbatch body can carry credential material; argv is public."""
    module = _load("tao-run-on-slurm")
    calls = []
    monkeypatch.setattr(module.subprocess, "run",
                        lambda cmd, *a, **k: (calls.append(cmd), _Done(""))[1])
    module.place_files({"/w/j.sbatch": "SECRET_SENTINEL_VALUE"}, {"login": "h"})
    assert "SECRET_SENTINEL_VALUE" not in " ".join(calls[0]), (
        "script body appeared in argv, where `ps` exposes it to other users"
    )


def test_placed_script_is_not_world_readable(monkeypatch):
    module = _load("tao-run-on-slurm")
    calls = []
    monkeypatch.setattr(module.subprocess, "run",
                        lambda cmd, *a, **k: (calls.append(cmd), _Done(""))[1])
    module.place_files({"/w/j.sbatch": "body"}, {"login": "h"})
    assert "umask 077" in calls[0][2]


def test_failed_placement_is_reported_not_swallowed(monkeypatch):
    module = _load("tao-run-on-slurm")
    monkeypatch.setattr(module.subprocess, "run",
                        lambda *a, **k: _Done("", 1))
    with pytest.raises(ValueError, match="could not write"):
        module.place_files({"/w/j.sbatch": "body"}, {"login": "h"})
