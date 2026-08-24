#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A DEFT stage must render on every platform from ONE definition.

Each stage used to be a literal `docker run` line in a reference file. That
string names a runtime, a GPU flag spelled docker's way, mount syntax docker's
way, and three flags meaningless elsewhere -- so the workflow could not move
platforms, and each new DEFT workflow forked the same lines again.

These tests hold the property that makes DEFT multi-platform: the same stage
entry produces a valid launch on docker, slurm, and kubernetes, with its
declared inputs reaching the container on all of them.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
AOI = REPO / "skills/applications/tao-run-deft-aoi/scripts"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stage_bundle():
    return _load(AOI / "stage_bundle.py", "deft_stage_bundle")


def _renderer(platform: str):
    return _load(REPO / f"skills/platform/tao-run-on-{platform}/references/render.py",
                 f"render_{platform}")


AMP_PARAMS = {"dataset_dir": "/ws/ag/datasets/nvpcb",
              "defect_spec": "/ws/ag/datasets/nvpcb/defect_spec.jsonl",
              "cosmos_models": "/ws/ag/base_checkpoints"}

PLATFORM_CTX = {
    "docker": {},
    "slurm": {"login": "me@login", "sqsh_dir": "/lustre/img",
              "time_limit": "00:20:00", "account": "some-account",
              "partition": "some-partition"},
    "kubernetes": {"mount_path": "/ws", "namespace": "tao", "pvc_claim": "deft"},
}


def _render(stage_bundle, platform, tmp_path, stage="anomalygen.amp",
            params=None, **build_kw):
    bundle = stage_bundle.build(
        stage, params if params is not None else AMP_PARAMS,
        results_dir="/ws/results/iter1", bank=REPO,
        args=["${ANOMALYGEN_SCRIPTS}/prep_testcase.sh --name iter1"], **build_kw)
    ctx = {"job_id": "ag-1", "results_dir": "/ws/results/iter1",
           "bank": str(REPO), "job_dir": str(tmp_path), **PLATFORM_CTX[platform]}
    return bundle, _renderer(platform).render(bundle, ctx)


@pytest.mark.parametrize("platform", sorted(PLATFORM_CTX))
def test_one_stage_definition_renders_everywhere(stage_bundle, platform, tmp_path):
    _, rendered = _render(stage_bundle, platform, tmp_path)
    assert rendered["argv"], f"{platform} produced no launch command"


@pytest.mark.parametrize("platform", sorted(PLATFORM_CTX))
def test_declared_inputs_reach_the_container(stage_bundle, platform, tmp_path):
    """A missing input does not error at runtime -- it reads empty and exits 0."""
    _, rendered = _render(stage_bundle, platform, tmp_path)
    text = " ".join(rendered["argv"]) + " ".join(rendered.get("files", {}).values())
    for key in ("TAO_INPUT_DATASET_DIR", "TAO_INPUT_COSMOS_MODELS"):
        assert key in text, f"{platform} never exposes {key}"


@pytest.mark.parametrize("platform", sorted(PLATFORM_CTX))
def test_no_docker_isms_survive_translation(stage_bundle, platform, tmp_path):
    """These have no counterpart off docker; they must drop, not translate."""
    _, rendered = _render(stage_bundle, platform, tmp_path)
    text = " ".join(rendered["argv"]) + " ".join(rendered.get("files", {}).values())
    for flag in ("--ipc=host", "--shm-size", "/etc/passwd"):
        if platform == "docker":
            continue
        assert flag not in text, f"{platform} carried the docker-only {flag}"
    assert "--rm" not in text, "--rm deletes the exit code status() reads"


def test_cpu_stage_requests_no_gpu(stage_bundle, tmp_path):
    """AMP's docker recipe says `--gpus all`; it is ~10s of CPU routing.

    On a scheduler an allocation is billed from the moment it starts, so a GPU
    request here idles a card through the whole phase.
    """
    bundle, rendered = _render(stage_bundle, "docker", tmp_path)
    assert bundle["compute_shape"]["gpus"] == 0
    assert "--gpus" not in " ".join(rendered["argv"])


def test_every_stage_emits_a_schema_valid_bundle(stage_bundle, tmp_path):
    """Run the real validator, not a shape check of our own devising."""
    validator = REPO / "scripts/tao_spec_bundle.py"
    spec_yaml = tmp_path / "spec.yaml"
    spec_yaml.write_text("train:\n  num_epochs: 2\n", encoding="utf-8")
    import yaml
    spec = yaml.safe_load(spec_yaml.read_text(encoding="utf-8"))

    for name, entry in stage_bundle.STAGES.items():
        params = {key: f"/ws/{key}" for key in entry["inputs"]}
        is_config = entry["mode"] == "config"
        bundle = stage_bundle.build(
            name, params, results_dir="/ws/results/x", bank=REPO,
            # mode=config reads settings from the spec; the contract forbids
            # carrying args as well, and build() refuses rather than drop them.
            args=None if is_config else ["do-something"],
            spec=spec if is_config else None)
        path = tmp_path / f"{name.replace('.', '_')}.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        done = subprocess.run([sys.executable, str(validator), "validate", str(path)],
                              capture_output=True, text=True)
        assert done.returncode == 0, f"{name}: {done.stdout}{done.stderr}"


def test_missing_input_fails_closed(stage_bundle):
    """Emitting a bundle with an unset input would run and read nothing."""
    with pytest.raises(ValueError, match="needs --param"):
        stage_bundle.build("anomalygen.amp", {"dataset_dir": "/ws/ds"},
                           results_dir="/ws/r", bank=REPO)


def test_config_stage_refuses_to_emit_without_a_spec(stage_bundle):
    """mode=config carries the spec as content; without it the run is a no-op."""
    with pytest.raises(ValueError, match="needs --spec-file"):
        stage_bundle.build("train", {"dataset_dir": "/d", "backbone": "/b"},
                           results_dir="/ws/r", bank=REPO)


def test_config_stage_refuses_args(stage_bundle):
    """Args on a config-mode stage would be dropped by the consumer, not run."""
    with pytest.raises(ValueError, match="cannot take --arg"):
        stage_bundle.build("train", {"dataset_dir": "/d", "backbone": "/b"},
                           results_dir="/ws/r", bank=REPO,
                           spec={"train": {"num_epochs": 2}}, args=["oops"])


def test_config_commands_carry_the_substitution_point(stage_bundle):
    """Without {config_path} the container never reads the spec it was given."""
    for name, entry in stage_bundle.STAGES.items():
        if entry["mode"] == "config":
            assert "{config_path}" in entry["command"], name


def test_images_resolve_through_versions_yaml(stage_bundle):
    """A URI pinned in this table would drift from the bank at the next bump."""
    for name, entry in stage_bundle.STAGES.items():
        assert "/" not in entry["image"], (
            f"{name} pins an image URI instead of a versions.yaml key"
        )
        assert stage_bundle.resolve_image(entry["image"], REPO).startswith("nvcr.io/")


@pytest.mark.parametrize("platform,marker", [
    ("docker", "-w /workspace/paidf-anomalygen"),
    ("slurm", "--container-workdir=/workspace/paidf-anomalygen"),
    ("kubernetes", 'workingDir: "/workspace/paidf-anomalygen"'),
])
def test_workdir_is_spelled_per_platform(stage_bundle, platform, marker, tmp_path):
    """paidf-anomalygen resolves its scripts relative to its own directory.

    Folding this into the command as a `cd` would hide it from the renderer and
    from anyone reading the bundle, so it is a bundle field each platform
    spells its own way.
    """
    _, rendered = _render(stage_bundle, platform, tmp_path)
    text = " ".join(rendered["argv"]) + " ".join(rendered.get("files", {}).values())
    assert marker in text, f"{platform} did not render the workdir"


@pytest.mark.parametrize("platform", sorted(PLATFORM_CTX))
def test_a_bundle_without_workdir_renders_nothing_for_it(stage_bundle, platform, tmp_path):
    """Optional means optional: no empty flag, no broken manifest."""
    bundle = stage_bundle.build(
        "mining.knn",
        {"target_embeddings": "/ws/t", "pool_embeddings": "/ws/p"},
        results_dir="/ws/results/x", bank=REPO, spec={"topn": 5})
    assert "workdir" not in bundle
    ctx = {"job_id": "k-1", "results_dir": "/ws/results/x", "bank": str(REPO),
           "job_dir": str(tmp_path), **PLATFORM_CTX[platform]}
    rendered = _renderer(platform).render(bundle, ctx)
    text = " ".join(rendered["argv"]) + " ".join(rendered.get("files", {}).values())
    assert "--container-workdir= " not in text and 'workingDir: ""' not in text
    if platform == "kubernetes":
        import yaml
        yaml.safe_load(next(iter(rendered["files"].values())))   # must still parse
