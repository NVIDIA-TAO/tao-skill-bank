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
    # job_dir under mount_path: kubernetes now refuses a config written outside
    # the single bound volume, because the pod could not read it.
    job_dir = "/ws/work" if platform == "kubernetes" else str(tmp_path)
    ctx = {"job_id": "k-1", "results_dir": "/ws/results/x", "bank": str(REPO),
           "job_dir": job_dir, **PLATFORM_CTX[platform]}
    rendered = _renderer(platform).render(bundle, ctx)
    text = " ".join(rendered["argv"]) + " ".join(rendered.get("files", {}).values())
    assert "--container-workdir= " not in text and 'workingDir: ""' not in text
    if platform == "kubernetes":
        import yaml
        yaml.safe_load(next(iter(rendered["files"].values())))   # must still parse


# ── The references must not drift back to docker run ───────────────────────
# Converting them is only half the job: the reason each DEFT workflow forked
# the same lines is that a docker recipe is the easiest thing to paste into a
# reference. This keeps the AOI stage docs on the platform contract.

AOI_REFS = REPO / "skills/applications/tao-run-deft-aoi/references"


def test_stage_execution_reference_exists():
    assert (AOI_REFS / "stage-execution.md").is_file(), (
        "the single documented way to launch a stage is missing"
    )


def test_aoi_skill_points_at_it():
    body = (REPO / "skills/applications/tao-run-deft-aoi/SKILL.md").read_text(
        encoding="utf-8")
    assert "references/stage-execution.md" in body


# Two files legitimately contain a docker command:
#   stage-execution.md   quotes the OLD recipe to explain what replaced it
#   prepare-for-inference.md is a HANDOFF -- a self-contained command given to
#                        a customer to run the finished model on their own
#                        machine, with no dependency on this bank
DOCKER_RECIPE_ALLOWED = {"stage-execution.md", "prepare-for-inference.md"}


def test_no_aoi_reference_prescribes_a_docker_run_command():
    """Prose *about* docker is fine; a copyable recipe is the coupling."""
    offenders = []
    for path in sorted(AOI_REFS.glob("*.md")):
        if path.name in DOCKER_RECIPE_ALLOWED:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            # A command, not a mention: starts the line and carries flags.
            if stripped.startswith("docker run") and "-" in stripped:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, (
        "these prescribe a docker run command, which cannot move platforms: "
        + ", ".join(offenders)
        + " -- emit the stage with stage_bundle.py instead"
    )


def test_converted_references_use_the_input_env(stage_bundle):
    """A stage arg naming a host path guesses at a layout it cannot see."""
    body = (AOI_REFS / "tao-generate-anomalies.md").read_text(encoding="utf-8")
    assert "$TAO_INPUT_DATASET_DIR" in body
    assert "$TAO_RESULTS_ROOT" in body


def test_every_documented_stage_name_exists_in_the_table(stage_bundle):
    """A reference naming a stage the emitter does not know fails at runtime."""
    import re as _re
    known = set(stage_bundle.STAGES)
    for path in sorted(AOI_REFS.glob("*.md")):
        body = path.read_text(encoding="utf-8")
        for match in _re.finditer(r'stage_bundle\.py"?\s+([a-z_]+(?:\.[a-z_]+)?)', body):
            name = match.group(1)
            if name in ("--list",):
                continue
            assert name in known, f"{path.name} names unknown stage {name!r}"


def test_image_override_is_available_for_smoke_tests(stage_bundle):
    """A CI job proving the machinery works should not pull a multi-GB image."""
    bundle = stage_bundle.build(
        "anomalygen.amp", AMP_PARAMS, results_dir="/ws/r", bank=REPO,
        args=["true"], image="busybox:1.36")
    assert bundle["image"] == "busybox:1.36"


def test_image_override_does_not_leak_into_the_table(stage_bundle):
    """An override must be per-call, never mutate the shared stage table."""
    stage_bundle.build("anomalygen.amp", AMP_PARAMS, results_dir="/ws/r",
                       bank=REPO, args=["true"], image="busybox:1.36")
    again = stage_bundle.build("anomalygen.amp", AMP_PARAMS,
                               results_dir="/ws/r", bank=REPO, args=["true"])
    assert again["image"].startswith("nvcr.io/")


# ── The platform evals must stay honest ────────────────────────────────────
# An eval that hand-writes `docker run` proves nothing about the contract, and
# an eval that only covers docker leaves the multi-platform claim untested.

EVAL_CONFIG = REPO / "skills/applications/tao-run-deft-aoi/eval.config"


@pytest.fixture(scope="module")
def eval_config():
    return json.loads(EVAL_CONFIG.read_text(encoding="utf-8"))


def test_eval_config_is_valid_json_with_ids(eval_config):
    ids = [e["id"] for e in eval_config["evals"]]
    assert len(ids) == len(set(ids)), f"duplicate eval ids: {ids}"


@pytest.mark.parametrize("platform", ["docker", "kubernetes"])
def test_a_platform_eval_exists(eval_config, platform):
    ids = [e["id"] for e in eval_config["evals"]]
    assert f"deft-aoi-stages-on-{platform}" in ids, (
        f"no DEFT AOI eval covers {platform}; the multi-platform claim is "
        "untested there"
    )


@pytest.mark.parametrize("platform", ["docker", "kubernetes"])
def test_platform_eval_drives_the_contract_not_the_runtime(eval_config, platform):
    """The whole point is that the eval must not bypass deft_exec.py."""
    entry = next(e for e in eval_config["evals"]
                 if e["id"] == f"deft-aoi-stages-on-{platform}")
    body = entry["prompt"] + entry["expected_outcome"]
    assert "stage_bundle.py" in body and "deft_exec.py" in body
    assert "bypass" in body.lower(), (
        "the eval does not forbid hand-writing a native launch, which would "
        "pass while proving nothing"
    )


@pytest.mark.parametrize("platform", ["docker", "kubernetes"])
def test_platform_eval_checks_for_the_silent_empty_read(eval_config, platform):
    """A COMPLETE record over zero inputs is the failure that looks like success."""
    entry = next(e for e in eval_config["evals"]
                 if e["id"] == f"deft-aoi-stages-on-{platform}")
    body = entry["prompt"] + entry["expected_outcome"]
    assert "TAO_INPUT_DATASET_DIR" in body
    assert "zero" in body.lower(), "the empty-read failure is not graded"


@pytest.mark.parametrize("platform", ["docker", "kubernetes"])
def test_platform_eval_names_a_stage_the_emitter_knows(eval_config, platform,
                                                       stage_bundle):
    entry = next(e for e in eval_config["evals"]
                 if e["id"] == f"deft-aoi-stages-on-{platform}")
    assert "anomalygen.amp" in entry["prompt"]
    assert "anomalygen.amp" in stage_bundle.STAGES
    assert stage_bundle.STAGES["anomalygen.amp"]["gpus"] == 0, (
        "the eval claims this stage needs no GPU; the table must agree or CI "
        "will queue for an accelerator it was told it did not need"
    )


def test_the_full_loop_is_covered_on_more_than_one_platform(eval_config):
    """A loop that only ever runs on docker is not platform-neutral.

    The stage-level evals prove the machinery; only a full-loop eval proves the
    nine real stages, with real images, actually run somewhere else.
    """
    full_loop = [e["id"] for e in eval_config["evals"]
                 if "deft-loop" in e["id"]]
    assert len(full_loop) >= 2, (
        f"only {full_loop} runs the whole loop; the multi-platform claim rests "
        "on one platform"
    )


@pytest.mark.parametrize("platform", ["docker", "kubernetes"])
def test_every_full_loop_eval_names_its_platform(eval_config, platform):
    """The converted references read $PLATFORM; an eval that never sets it
    leaves the documented commands with nothing to substitute."""
    entry = next((e for e in eval_config["evals"]
                  if "deft-loop" in e["id"] and platform in e["prompt"]), None)
    assert entry is not None, f"no full-loop eval targets {platform}"
    assert f"PLATFORM={platform}" in entry["prompt"], (
        f"{entry['id']} never exports PLATFORM={platform}"
    )


def test_the_kubernetes_loop_provisions_a_gpu(eval_config):
    """The agent has GPUs; minikube does not inherit them without --gpus.

    Without this the training stages cannot schedule, and the tempting recovery
    -- skip them and report success -- is exactly what must not happen.
    """
    entry = next(e for e in eval_config["evals"]
                 if e["id"] == "deft-loop-ag-mining-kubernetes")
    body = entry["prompt"] + entry["expected_outcome"]
    assert "--gpus all" in body, "minikube started without GPU support"
    assert "nvidia-device-plugin" in body or "gpu-operator" in body, (
        "nothing advertises nvidia.com/gpu, so a GPU request never schedules"
    )
    assert "FAILURE" in body and "no GPU" in body, (
        "skipping the GPU stages on a CPU-only cluster must be graded a failure, "
        "not quietly tolerated"
    )


# ── mode=config bundles must actually reach their spec ─────────────────────
# The contract says the CONSUMER writes the spec file and substitutes its
# compute-frame path into {config_path}. No renderer did. Every mode=config
# stage -- train, evaluate, inference, rca and all three mining stages, 7 of
# the 12 -- therefore reached its container with a literal "{config_path}"
# argument and failed on a file of that name. The k8s full-loop eval could not
# have passed, and neither could the docker one.

CONFIG_MODE_PLATFORMS = ["docker", "slurm", "kubernetes"]


def _config_ctx(platform, tmp_path):
    base = {"job_id": "train-1", "results_dir": "/ws/results/base",
            "bank": str(REPO), "job_dir": "/ws/work",
            "uid": 1000, "gid": 1000, "groups": [1000], "user_name": "ci"}
    return {**base, **PLATFORM_CTX[platform]}


def _train_bundle(stage_bundle):
    return stage_bundle.build(
        "train", {"dataset_dir": "/ws/data", "backbone": "/ws/ptm/b.safetensors"},
        results_dir="/ws/results/base", bank=REPO,
        spec={"train": {"num_epochs": 2}})


@pytest.mark.parametrize("platform", CONFIG_MODE_PLATFORMS)
def test_config_path_is_substituted(stage_bundle, platform, tmp_path):
    rendered = _renderer(platform).render(_train_bundle(stage_bundle),
                                          _config_ctx(platform, tmp_path))
    blob = " ".join(rendered["argv"]) + " ".join(rendered.get("files", {}).values())
    assert "{config_path}" not in blob, (
        f"{platform} left the placeholder in the command; the container would "
        "try to read a file literally named {config_path}"
    )


@pytest.mark.parametrize("platform", CONFIG_MODE_PLATFORMS)
def test_the_spec_file_is_actually_written(stage_bundle, platform, tmp_path):
    """Substituting a path that nothing writes is the same failure, later."""
    import yaml

    rendered = _renderer(platform).render(_train_bundle(stage_bundle),
                                          _config_ctx(platform, tmp_path))
    configs = {p: c for p, c in rendered.get("files", {}).items() if "configs/" in p}
    assert configs, f"{platform} substituted a config path but wrote no file"
    content = yaml.safe_load(next(iter(configs.values())))
    assert content == {"train": {"num_epochs": 2}}, "spec content did not survive"


@pytest.mark.parametrize("platform", CONFIG_MODE_PLATFORMS)
def test_the_substituted_path_is_the_one_written(stage_bundle, platform, tmp_path):
    """A path mismatch fails at runtime, not at render."""
    rendered = _renderer(platform).render(_train_bundle(stage_bundle),
                                          _config_ctx(platform, tmp_path))
    written = [p for p in rendered.get("files", {}) if "configs/" in p]
    blob = " ".join(rendered["argv"]) + " ".join(rendered.get("files", {}).values())
    assert any(p in blob for p in written), (
        f"{platform} wrote {written} but the command references a different path"
    )


def test_kubernetes_refuses_a_config_outside_the_mounted_volume(stage_bundle, tmp_path):
    """A pod sees ONE volume; a spec outside it is unreadable at runtime."""
    module = _renderer("kubernetes")
    ctx = {**_config_ctx("kubernetes", tmp_path), "job_dir": "/somewhere/else"}
    with pytest.raises(ValueError, match="outside the mounted volume"):
        module.render(_train_bundle(stage_bundle), ctx)


@pytest.mark.parametrize("platform", CONFIG_MODE_PLATFORMS)
def test_args_mode_writes_no_config(stage_bundle, platform, tmp_path):
    """Only mode=config materializes a spec."""
    bundle = stage_bundle.build("anomalygen.amp", AMP_PARAMS,
                                results_dir="/ws/results/base", bank=REPO,
                                args=["true"])
    rendered = _renderer(platform).render(bundle, _config_ctx(platform, tmp_path))
    assert not [p for p in rendered.get("files", {}) if "configs/" in p]


# ── The stage table must not restate what the model skill owns ─────────────
# The table carried `visual_changenet classify train`. There is no `classify`
# subcommand -- the subtask lives in the spec (`task: classify`) -- so all three
# VCN stages would have failed on argument parsing. The model skill's
# skill_info.yaml declares the real command, and the table simply disagreed
# with it. Couple them so the copy cannot drift again.

VCN_SKILL_INFO = REPO / "skills/models/tao-train-visual-changenet/references/skill_info.yaml"


@pytest.mark.parametrize("stage", ["train", "evaluate", "inference"])
def test_vcn_stage_commands_match_the_model_skill(stage_bundle, stage):
    import yaml

    owned = yaml.safe_load(VCN_SKILL_INFO.read_text(encoding="utf-8"))
    expected = owned["actions"][stage]["command"]
    assert stage_bundle.STAGES[stage]["command"] == expected, (
        f"the stage table says {stage_bundle.STAGES[stage]['command']!r} but "
        f"tao-train-visual-changenet declares {expected!r}; that skill owns the "
        "command and this table must not disagree with it"
    )


@pytest.mark.parametrize("stage", ["train", "evaluate", "inference"])
def test_vcn_stages_declare_the_backbone(stage_bundle, stage):
    """All three build the model from the spec before loading a checkpoint.

    An unstaged backbone does not always error -- a null path silently degrades
    held-out evaluation quality -- so a missing declaration is worse than a
    crash: the run completes and the number is wrong.
    """
    import yaml

    owned = yaml.safe_load(VCN_SKILL_INFO.read_text(encoding="utf-8"))
    declared = owned["actions"][stage].get("inputs") or []
    needs_backbone = any("pretrained_backbone_path" in k for k in declared)
    assert needs_backbone, f"fixture drift: {stage} no longer declares a backbone"
    assert "backbone" in stage_bundle.STAGES[stage]["inputs"], (
        f"{stage} omits the backbone the model skill declares"
    )


# ── An input can require a fixed in-container path ─────────────────────────
# Identity mounting is the right default -- a path in a CSV or spec resolves the
# same on both sides -- but some images address a FIXED location. Every path in
# a cosmos-rl spec is under /tao-workspace, and its own template warns "NEVER
# over /workspace, where cosmos-rl itself is installed". paidf-anomalygen
# resolves base checkpoints relative to its install dir, which this branch first
# worked around with a symlink inside the stage command.
#
# Getting this wrong does not raise: the input is mounted where the workload
# does not look, so the stage reads nothing and exits 0.

TARGET_BUNDLE_INPUT = {"spec_key": "workspace", "type": "folder",
                       "uri": "/ws/proj", "target": "/tao-workspace"}


def _with_target(stage_bundle):
    bundle = stage_bundle.build("anomalygen.amp", AMP_PARAMS,
                                results_dir="/ws/results/x", bank=REPO, args=["true"])
    bundle["declared_inputs"] = [TARGET_BUNDLE_INPUT]
    return bundle


@pytest.mark.parametrize("platform,marker", [
    ("docker", "/ws/proj:/tao-workspace:ro"),
    ("slurm", "/ws/proj:/tao-workspace:ro"),
])
def test_bind_platforms_honour_the_target(stage_bundle, platform, marker, tmp_path):
    ctx = {"job_id": "t1", "results_dir": "/ws/results/x", "bank": str(REPO),
           "job_dir": "/ws/work", **PLATFORM_CTX[platform]}
    rendered = _renderer(platform).render(_with_target(stage_bundle), ctx)
    blob = " ".join(rendered["argv"]) + " ".join(rendered.get("files", {}).values())
    assert marker in blob, f"{platform} ignored the target and mounted at the uri"


def test_kubernetes_maps_a_target_through_subpath(stage_bundle, tmp_path):
    """A pod sees ONE volume, so a target is the same claim mounted twice."""
    import yaml

    ctx = {"job_id": "t1", "results_dir": "/ws/results/x", "bank": str(REPO),
           "job_dir": "/ws/work", **PLATFORM_CTX["kubernetes"]}
    rendered = _renderer("kubernetes").render(_with_target(stage_bundle), ctx)
    doc = yaml.safe_load(next(iter(rendered["files"].values())))
    mounts = doc["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    extra = [m for m in mounts if m.get("mountPath") == "/tao-workspace"]
    assert extra, "no volumeMount at the requested target"
    assert extra[0]["subPath"] == "proj", (
        "subPath must be the claim-relative part of the uri, or the pod mounts "
        "the wrong directory at the right path"
    )
    assert extra[0]["readOnly"] is True


@pytest.mark.parametrize("platform", ["docker", "slurm", "kubernetes"])
def test_no_target_keeps_identity_mounting(stage_bundle, platform, tmp_path):
    """The default must not change: identity is what makes paths portable."""
    ctx = {"job_id": "t1", "results_dir": "/ws/results/x", "bank": str(REPO),
           "job_dir": "/ws/work", **PLATFORM_CTX[platform]}
    rendered = _renderer(platform).render(
        stage_bundle.build("anomalygen.amp", AMP_PARAMS, results_dir="/ws/results/x",
                           bank=REPO, args=["true"]), ctx)
    blob = " ".join(rendered["argv"]) + " ".join(rendered.get("files", {}).values())
    assert "/ws/ag/datasets/nvpcb" in blob


# ── TOML specs must survive the trip ───────────────────────────────────────
# Cosmos-RL specs are config_format=toml. Python ships tomllib to READ toml and
# nothing to write it, and this bank keeps its dependencies to pyyaml and
# jsonschema, so the renderers carry a minimal writer. Every test here
# round-trips through the stdlib READER rather than comparing strings: the
# failure that matters is a file tomllib parses into something other than the
# spec, not a formatting difference.

TOML_SPECS = [
    pytest.param({"a": 1, "b": "x"}, id="flat"),
    pytest.param({"t": {"a": 1}}, id="one-table"),
    pytest.param({"t": {"u": {"deep": True}}}, id="nested-tables"),
    pytest.param({"t": {"arr": [1, 2, 3], "s": ["a", "b"]}}, id="arrays"),
    pytest.param({"t": {"f": 1e-05, "neg": -3}}, id="numbers"),
    pytest.param({"t": {"q": 'has "quotes" and \\ backslash'}}, id="escapes"),
    pytest.param({"scalar_first": 1, "t": {"x": 2}}, id="scalar-before-table"),
]


@pytest.mark.parametrize("spec", TOML_SPECS)
def test_toml_round_trips(spec):
    import tomllib

    module = _renderer("docker")
    assert tomllib.loads(module.dumps_toml(spec)) == spec


def test_booleans_are_not_written_as_ints():
    """bool subclasses int in Python; checking int first yields `1`, not `true`."""
    import tomllib

    module = _renderer("docker")
    out = tomllib.loads(module.dumps_toml({"t": {"flag": True, "count": 1}}))
    assert out["t"]["flag"] is True and out["t"]["count"] == 1


def test_a_key_after_a_table_would_change_owner():
    """The classic hand-rolled-writer corruption: a parent key emitted after a
    [table] header silently belongs to that table."""
    import tomllib

    module = _renderer("docker")
    spec = {"top": "parent-level", "section": {"inner": 1}}
    assert tomllib.loads(module.dumps_toml(spec)) == spec


def test_unrepresentable_value_fails_loudly():
    """Silently dropping a spec key would run training with wrong settings."""
    module = _renderer("docker")
    with pytest.raises(ValueError, match="no TOML representation"):
        module.dumps_toml({"t": {"when": object()}})


@pytest.mark.parametrize("platform", CONFIG_MODE_PLATFORMS)
def test_every_platform_can_write_toml(platform):
    """cosmos-rl train is toml; a platform without a writer cannot run it."""
    import tomllib

    module = _renderer(platform)
    path, content = module.config_file(
        {"mode": "config", "config_format": "toml", "spec": {"t": {"a": 1}}},
        "job-1", "/ws/results/x")
    assert path.endswith(".toml")
    assert tomllib.loads(content) == {"t": {"a": 1}}
