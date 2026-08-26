#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DEFT Cosmos3 stages must render on every platform from one definition.

Cosmos3 differs from AOI in two ways that these tests pin:

* Most of its loop is HOST-SIDE. Only four container families -- Train,
  Proxy/Benchmark evaluate, AnomalyGen, Mining -- belong in a bundle at all.
* Its commands are RESOLVED from the model skill rather than copied. The AOI
  table copied `visual_changenet classify train`, a subcommand that does not
  exist; resolving removes the copy that can drift.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
C3 = REPO / "skills/applications/tao-run-deft-aoi-cosmos3/scripts"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sb():
    return _load(C3 / "stage_bundle.py", "cosmos3_stage_bundle")


def _renderer(platform: str):
    return _load(REPO / f"skills/platform/tao-run-on-{platform}/references/render.py",
                 f"c3_render_{platform}")


PLATFORM_CTX = {
    "docker": {},
    "slurm": {"login": "me@login", "sqsh_dir": "/lustre/img", "time_limit": "04:00:00",
              "account": "some-account", "partition": "some-partition"},
    "kubernetes": {"mount_path": "/ws", "namespace": "tao", "pvc_claim": "deft"},
}
SPEC = {"custom": {"system_prompt": "Return exactly OK or NG."}, "train": {"epochs": 2}}
TRAIN_PARAMS = {"workspace": "/ws", "annotations": "/ws/annotations/train.json"}


def _ctx(platform):
    return {"job_id": "c3-train-1", "results_dir": "/ws/results/base",
            "bank": str(REPO), "job_dir": "/ws/work", "uid": 1000, "gid": 1000,
            "groups": [1000], "user_name": "ci", **PLATFORM_CTX[platform]}


def _train(sb):
    return sb.build("train", TRAIN_PARAMS, results_dir="/ws/results/base",
                    bank=REPO, spec=SPEC)


@pytest.mark.parametrize("platform", sorted(PLATFORM_CTX))
def test_train_renders_everywhere(sb, platform):
    assert _renderer(platform).render(_train(sb), _ctx(platform))["argv"]


@pytest.mark.parametrize("platform", sorted(PLATFORM_CTX))
def test_the_toml_spec_is_written(sb, platform):
    """cosmos-rl is config_format=toml; nothing in the stdlib writes toml."""
    import tomllib

    rendered = _renderer(platform).render(_train(sb), _ctx(platform))
    configs = {p: c for p, c in rendered.get("files", {}).items() if "configs/" in p}
    assert configs, f"{platform} wrote no spec file"
    path, content = next(iter(configs.items()))
    assert path.endswith(".toml")
    assert tomllib.loads(content) == SPEC


@pytest.mark.parametrize("platform", sorted(PLATFORM_CTX))
def test_the_workspace_lands_at_tao_workspace(sb, platform):
    """Every path in a cosmos-rl spec is under /tao-workspace.

    Mounting the workspace at its host path instead does not fail -- the spec's
    paths simply do not resolve, and the stage fails deep inside the workload.
    """
    rendered = _renderer(platform).render(_train(sb), _ctx(platform))
    blob = " ".join(rendered["argv"]) + " ".join(rendered.get("files", {}).values())
    assert "/tao-workspace" in blob, f"{platform} did not honour the target"


def test_the_command_comes_from_the_model_skill(sb):
    """Not a copy. cosmos-rl train is a multi-line shell hook that computes a
    path inside the container; a transcription would be stale in a release."""
    import yaml

    owned = yaml.safe_load(
        (REPO / "skills/models/tao-finetune-cosmos-reason/references/skill_info.yaml")
        .read_text(encoding="utf-8"))
    assert _train(sb)["command"] == owned["actions"]["train"]["command"]


def test_the_image_comes_from_the_backend_contract(sb):
    import yaml

    owned = yaml.safe_load(
        (REPO / "skills/models/tao-finetune-cosmos-reason/references/skill_info.yaml")
        .read_text(encoding="utf-8"))
    expected = owned["backend_contracts"]["cosmos-rl"]["container_image"]
    assert _train(sb)["image"] == expected


def test_no_stage_pins_an_image_uri(sb):
    """Model-skill stages resolve through the contract; the rest through
    versions.yaml. A URI written into the table drifts at the next bump."""
    for name, entry in sb.STAGES.items():
        if entry["image"]:
            assert "/" not in entry["image"], f"{name} pins {entry['image']!r}"


def test_host_side_stages_are_refused_with_a_reason(sb):
    """Wrapping a local script in a scheduler would be pure overhead."""
    with pytest.raises(ValueError, match="runs on the host"):
        sb.build("proxy_rcca", {}, results_dir="/ws/r", bank=REPO)


def test_every_recorded_stage_is_accounted_for(sb):
    """A stage in neither table is an oversight; this makes that visible."""
    commit = (C3 / "commit_stage.py").read_text(encoding="utf-8")
    block = commit[commit.index("STAGES = ("):commit.index(")", commit.index("STAGES = ("))]
    recorded = {line.strip().strip('",') for line in block.splitlines()[1:] if line.strip()}
    known = set(sb.STAGES) | set(sb.HOST_SIDE_STAGES)
    # Container families expand into sub-stages (anomalygen -> .amp/.sdg).
    families = {name.split(".")[0] for name in known}
    unaccounted = {s for s in recorded if s not in known and s not in families}
    assert not unaccounted, f"stages in neither table: {sorted(unaccounted)}"


def test_every_stage_emits_a_schema_valid_bundle(sb, tmp_path):
    validator = REPO / "scripts/tao_spec_bundle.py"
    for name, entry in sb.STAGES.items():
        params = {key: f"/ws/{key}" for key in entry["inputs"]}
        is_config = entry["action"] is not None or entry["mode"] == "config"
        bundle = sb.build(name, params, results_dir="/ws/results/x", bank=REPO,
                          args=None if is_config else ["true"],
                          spec=SPEC if is_config else None)
        path = tmp_path / f"{name.replace('.', '_')}.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        done = subprocess.run([sys.executable, str(validator), "validate", str(path)],
                              capture_output=True, text=True)
        assert done.returncode == 0, f"{name}: {done.stdout}{done.stderr}"


# ── The cosmos3 references must stay on the contract ───────────────────────
# The reference used to state the train command as
#   cosmos-rl --config {config_path} /opt/cosmos_rl/tao_sft_example.py
# while the model skill computes the hook from cosmos_rl.__file__, landing at
# .../tools/custom_hooks/tao_sft_example.py. Different files: the restated form
# passed cosmos-rl a script that does not exist.

C3_REFS = REPO / "skills/applications/tao-run-deft-aoi-cosmos3/references"


def test_stage_execution_reference_exists():
    assert (C3_REFS / "stage-execution.md").is_file()


def test_the_skill_points_at_it():
    body = (REPO / "skills/applications/tao-run-deft-aoi-cosmos3/SKILL.md").read_text(
        encoding="utf-8")
    assert "references/stage-execution.md" in body


def test_no_reference_restates_the_stale_hook_path():
    """The old literal is wrong AND unresolvable; it must not come back."""
    offenders = [
        path.name for path in sorted(C3_REFS.glob("*"))
        if path.is_file()
        and "/opt/cosmos_rl/tao_sft_example.py" in path.read_text(encoding="utf-8")
        and "does not exist" not in path.read_text(encoding="utf-8")
        and "NOT /opt/cosmos_rl" not in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"{offenders} restate the hardcoded hook path; the model skill computes "
        "it from cosmos_rl.__file__, so the literal points at nothing"
    )


def test_the_workspace_target_is_documented():
    body = (C3_REFS / "stage-execution.md").read_text(encoding="utf-8")
    assert "/tao-workspace" in body and "NEVER mount over `/workspace`" in body


@pytest.mark.parametrize("platform", ["docker", "kubernetes"])
def test_a_full_loop_eval_names_its_platform(platform):
    """The converted references read $PLATFORM; an eval that never sets it
    leaves the documented commands with nothing to substitute."""
    cfg = json.loads(
        (REPO / "skills/applications/tao-run-deft-aoi-cosmos3/eval.config")
        .read_text(encoding="utf-8"))
    entry = next((e for e in cfg["evals"]
                  if "loop" in e["id"] and f"PLATFORM={platform}" in e["prompt"]), None)
    assert entry is not None, f"no cosmos3 loop eval exports PLATFORM={platform}"


def test_the_kubernetes_loop_provisions_a_gpu():
    cfg = json.loads(
        (REPO / "skills/applications/tao-run-deft-aoi-cosmos3/eval.config")
        .read_text(encoding="utf-8"))
    entry = next(e for e in cfg["evals"] if e["id"].endswith("-kubernetes"))
    body = entry["prompt"] + entry["expected_outcome"]
    assert "--gpus all" in body
    assert "nvidia-device-plugin" in body or "gpu-operator" in body
    assert "FAILURE" in body


def test_host_side_stages_are_not_graded_as_platform_violations():
    """RCCA and friends run locally by design; grading them as fallbacks would
    fail every correct run."""
    cfg = json.loads(
        (REPO / "skills/applications/tao-run-deft-aoi-cosmos3/eval.config")
        .read_text(encoding="utf-8"))
    entry = next(e for e in cfg["evals"] if e["id"].endswith("-kubernetes"))
    assert "Host-side stages" in entry["expected_outcome"]


def test_the_documented_test_command_names_its_working_directory():
    """The bundled tests import sibling skills via SKILL_ROOT.parents[1].

    Run from a standalone ~/.claude/skills copy they fail with
    `ModuleNotFoundError: No module named 'filter_mined_history'` -- which names
    the module, not the missing sibling skill, so the documented command has to
    say where to run it from.
    """
    body = (REPO / "skills/applications/tao-run-deft-aoi-cosmos3/SKILL.md").read_text(
        encoding="utf-8")
    assert "unittest tests.test_cosmos3_bare" in body
    assert "TAO_SKILL_BANK_PATH" in body and "cd " in body, (
        "the documented command does not say which directory to run it from"
    )


def test_the_bundled_tests_do_reach_into_sibling_skills():
    """Guard the premise: if the imports become self-contained, the cd is stale."""
    body = (REPO / "skills/applications/tao-run-deft-aoi-cosmos3/tests/test_cosmos3_bare.py"
            ).read_text(encoding="utf-8")
    assert "parents[1]" in body, (
        "the bundled tests no longer reach outside the skill; the working-"
        "directory requirement in SKILL.md can be relaxed"
    )
