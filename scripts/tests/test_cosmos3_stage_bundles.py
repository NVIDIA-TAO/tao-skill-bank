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


# ── The documented verbs must exist ────────────────────────────────────────
# stage-execution.md documented `deft_exec.py --submit/--await-job/--logs` for
# Cosmos3 while that script accepted only `--state` and a trailing command. The
# emitter was useless: an agent following the page hit "unrecognized arguments".
#
# Copying AOI's ~400 lines would have created the fork this work exists to
# remove -- DEFT AOI, IAA and Cosmos3 already ship three commit_stage.py. The
# verbs are workflow-agnostic (no stage name, state file or network arch
# anywhere in them), so they live at bank root and both workflows delegate.

DEFT_EXECS = {
    "aoi": REPO / "skills/applications/tao-run-deft-aoi/scripts/deft_exec.py",
    "cosmos3": REPO / "skills/applications/tao-run-deft-aoi-cosmos3/scripts/deft_exec.py",
}


def test_the_shared_launcher_exists():
    assert (REPO / "scripts/tao_launch.py").is_file(), (
        "the shared four-verb launcher is missing; each workflow would need its "
        "own copy"
    )


@pytest.mark.parametrize("verb", ["submit_bundle", "await_job", "job_logs", "cancel_job"])
def test_the_launcher_implements_every_verb(verb):
    module = _load(REPO / "scripts/tao_launch.py", "tao_launch_probe")
    assert callable(getattr(module, verb, None))


@pytest.mark.parametrize("workflow", sorted(DEFT_EXECS))
@pytest.mark.parametrize("flag", ["--submit", "--await-job", "--logs", "--cancel"])
def test_every_workflow_exposes_the_documented_verbs(workflow, flag):
    """A reference documenting a flag the script rejects is worse than no doc."""
    done = subprocess.run([sys.executable, str(DEFT_EXECS[workflow]), "--help"],
                          capture_output=True, text=True)
    assert flag in done.stdout, f"{workflow} deft_exec has no {flag}"


def test_cosmos3_keeps_its_policy_gate(tmp_path):
    """The verbs are additive: `-- <cmd>` still enforces air-gap policy."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps(
        {"results_dir": str(tmp_path), "execution_policy": {"network_mode": "airgap"}}))
    done = subprocess.run(
        [sys.executable, str(DEFT_EXECS["cosmos3"]), "--state", str(state),
         "--", "pip", "install", "torch"],
        capture_output=True, text=True)
    assert done.returncode != 0 and "air-gap" in done.stdout + done.stderr


def test_a_verb_and_a_trailing_command_are_refused_together(tmp_path):
    """Two different modes; silently preferring one would hide the other."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"results_dir": str(tmp_path)}))
    done = subprocess.run(
        [sys.executable, str(DEFT_EXECS["cosmos3"]), "--state", str(state),
         "--logs", "job-1", "--", "echo", "hi"],
        capture_output=True, text=True)
    assert done.returncode != 0
    assert "different modes" in done.stdout + done.stderr


def test_the_launcher_is_workflow_agnostic():
    """If it starts naming a workflow in CODE, it has stopped being shared.

    The module docstring names the workflows deliberately -- it explains which
    forks this exists to prevent -- so only the code below it is scanned.
    """
    import ast

    source = (REPO / "scripts/tao_launch.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    body = ast.unparse(ast.Module(body=[
        node for node in tree.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ], type_ignores=[])).lower()
    for term in ("deft_state", "changenet", "cosmos3", "iter_label"):
        assert term not in body, f"the shared launcher references {term!r} in code"


def test_nothing_in_the_skill_restates_the_hook_path():
    """The literal points at a file that does not exist, wherever it appears."""
    root = REPO / "skills/applications/tao-run-deft-aoi-cosmos3"
    offenders = []
    for path in sorted(root.rglob("*.md")) + sorted(root.rglob("*.toml")):
        body = path.read_text(encoding="utf-8")
        if "/opt/cosmos_rl/tao_sft_example.py" not in body:
            continue
        # Naming it to say it is WRONG is the point; prescribing it is not.
        if any(k in body for k in ("does not exist", "NOT /opt/cosmos_rl",
                                   "points at a file that does not exist")):
            continue
        offenders.append(str(path.relative_to(root)))
    assert not offenders, (
        f"{offenders} prescribe the hardcoded hook path; the model skill computes "
        "it from cosmos_rl.__file__"
    )


# ── Contracts the cosmos3 scripts actually enforce ─────────────────────────

def test_the_rcca_template_passes_its_own_validator():
    """An error message pointing at a template that fails the check would send
    the reader in a circle. This is the check the branch kept finding: a
    documented artifact that does not exist or does not satisfy its own rule."""
    sys.path.insert(0, str(C3))
    commit = _load(C3 / "commit_stage.py", "cosmos3_commit_stage")
    template = REPO / ("skills/applications/tao-run-deft-aoi-cosmos3/references/"
                       "RCCA_REPORT_TEMPLATE.md")
    assert template.is_file(), "the validator's error names a template that is absent"
    commit._required_rcca_report(template, "--rcca-report")


def test_an_incomplete_rcca_report_names_the_missing_sections(tmp_path):
    sys.path.insert(0, str(C3))
    commit = _load(C3 / "commit_stage.py", "cosmos3_commit_stage2")
    partial = tmp_path / "rcca.md"
    partial.write_text("# RCCA\n\n## Verdict\n\nfine\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        commit._required_rcca_report(partial, "--rcca-report")
    message = str(excinfo.value)
    assert "Per-Defect Analysis" in message and "RCCA_REPORT_TEMPLATE.md" in message


def test_proxy_rcca_requires_the_report():
    body = (C3 / "commit_stage.py").read_text(encoding="utf-8")
    assert "--rcca-report" in body and "_required_rcca_report" in body


def test_the_image_cap_patcher_distinguishes_lifted_from_moved():
    """Absent has two causes. A newer image that lifted the cap must not block
    the loop; a file that moved the cap must not be waved through into
    `ValueError: At most 1 image(s) may be provided in one prompt`."""
    patcher = _load(C3 / "patch_eval_image_cap.py", "cosmos3_cap_patcher")

    _, current = patcher.apply_cap('engine = LLM(model=path)', 2)
    assert current == 2, "a genuinely lifted cap should report no patch needed"

    with pytest.raises(ValueError, match="present but its image cap did not match"):
        patcher.apply_cap('limit_mm_per_prompt=DEFAULT_CAPS', 2)

    _, current = patcher.apply_cap('limit_mm_per_prompt={"image": 1}', 2)
    assert current == 1, "a real cap must still be detected and patched"


def test_the_manifest_key_is_documented_exactly():
    """A flat manifest is rejected by a message naming the nested key; the docs
    have to say which key, or the reader guesses."""
    root = REPO / "skills/applications/tao-run-deft-aoi-cosmos3"
    # SKILL.md is size-capped, so prose migrates into references/. What matters
    # is that the skill documents the key SOMEWHERE, not which file.
    documented = any(
        "evaluation_contract.benchmark.annotations_sha256" in path.read_text(encoding="utf-8")
        for path in [root / "SKILL.md", *sorted((root / "references").glob("*.md"))]
    )
    validator = (C3 / "validate_split_contract.py").read_text(encoding="utf-8")
    assert documented, "the manifest's nested key is documented nowhere"
    assert '["evaluation_contract"]["benchmark"]["annotations_sha256"]' in validator, (
        "the documented key no longer matches what the validator reads"
    )


# ── eval.config is the LIVE-EXECUTION lane ─────────────────────────────────
# docs/skill-requirements.md draws the line: evals/evals.json is the required,
# no-execution routing check; eval.config is the optional layer that "pulls real
# datasets, runs real docker run, measures real" behaviour. A plan-only entry in
# eval.config occupies a Colossus GPU shard (x2 backends) to exercise nothing,
# and duplicates a check the free lane already runs -- more strictly, as
# individually gradable expected_behavior items rather than one prose paragraph.

def test_no_plan_only_eval_sits_in_the_execution_lane():
    cfg = json.loads(
        (REPO / "skills/applications/tao-run-deft-aoi-cosmos3/eval.config")
        .read_text(encoding="utf-8"))
    plan_only = [
        e["id"] for e in cfg["evals"]
        if "do not execute" in e["prompt"].lower()
        or "plan-only" in e["prompt"].lower()
    ]
    assert not plan_only, (
        f"{plan_only} are plan-only but sit in eval.config, which is the "
        "live-execution lane. Routing checks belong in evals/evals.json, which "
        "is required, free, and graded per behaviour"
    )


def test_the_routing_coverage_still_exists():
    """Removing the duplicate must not remove the coverage."""
    entries = json.loads(
        (REPO / "skills/applications/tao-run-deft-aoi-cosmos3/evals/evals.json")
        .read_text(encoding="utf-8"))
    planning = [e for e in entries if "plan" in e["question"].lower()]
    assert planning, "no routing/plan check survives in evals.json"
    # Both fields: expected_behavior lists gradable items, ground_truth carries
    # the narrative the grader compares against. A claim in either is covered.
    behaviours = " ".join(
        str(e.get("expected_behavior", "")) + " " + str(e.get("ground_truth", ""))
        for e in planning
    ).lower()
    # The things the removed eval asserted, still asserted here.
    for claim in ("does not default to a platform", "submit/status/logs/cancel"):
        assert claim in behaviours, f"lost coverage: {claim!r}"
