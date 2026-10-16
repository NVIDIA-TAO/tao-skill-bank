# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensure Skill Bank delegates orchestration to the installed DS package."""

import importlib
import json
import os
from pathlib import Path
import shlex
import sys

import jsonschema
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "applications" / "tao-run-dinov3-ssl-deft"
sys.path.insert(0, str(ROOT / "scripts"))
resolve_tao_image = importlib.import_module("resolve_tao_image")


def test_application_command_and_evals_use_installed_runtime():
    """The declared action must be directly runnable without an unreachable shim."""
    info = yaml.safe_load((SKILL / "references/skill_info.yaml").read_text())
    assert shlex.split(info["actions"]["run"]["command"]) == [
        "python", "-m", "nvidia_tao_ds.mining.dinov3.workflow", "run", "{config_path}",
    ]
    assert all(case.get("expected_script") is None for case in json.loads(
        (SKILL / "evals/evals.json").read_text()))


@pytest.mark.parametrize("name", ["workflow", "metrics", "stage-request"])
def test_all_published_schemas_match_runtime_source(name):
    """Source handoff requires DS parity without importing optional runtimes."""
    source = os.environ.get("TAO_DATA_SERVICES_SOURCE")
    if not source:
        if os.environ.get("TAO_DEFT_REQUIRE_SCHEMA_PARITY") == "1":
            pytest.fail("TAO_DATA_SERVICES_SOURCE is required for schema parity")
        pytest.skip("Cross-repository source handoff supplies TAO_DATA_SERVICES_SOURCE")
    runtime_path = Path(source) / "nvidia_tao_ds/mining/dinov3/workflow/schemas" / f"{name}.schema.yaml"
    runtime = yaml.safe_load(runtime_path.read_text())
    published = json.loads((SKILL / "references" / f"{name}.schema.json").read_text())
    published.pop("$comment", None)  # Copy provenance is not part of the JSON schema contract.
    assert published == runtime


def test_published_schema_accepts_native_lightning_node_contract():
    """The documented external-runner schema accepts DS's canonical request."""
    schema = json.loads(
        (SKILL / "references" / "stage-request.schema.json").read_text(
            encoding="utf-8"
        )
    )
    request = {
        "client_job_id": "d3deft-0123456789abcdefabcd",
        "run_id": "run-1",
        "round_index": 1,
        "stage": "train",
        "command": ["dinov3", "train", "-e", "/results/input.yaml"],
        "workdir": "/workspace",
        "results_dir": "/results",
        "environment": {},
        "resources": {"nodes": 1, "gpus_per_node": 1},
        "execution_contract": {
            "membership": "static",
            "attempt_scope": "process",
            "retry_scope": "process",
            "attempt_id_scope": "backend_attempt",
            "launch_id_environment": None,
            "adapter_managed_resources": {
                "nodes": 2, "gpus_per_node": 1, "world_size": 2,
            },
            "required_capabilities": ["gang_scheduling"],
            "node_environment": {
                "contract": "lightning_node_entrypoint_v1",
                "world_size_semantics": "nodes",
                "required_for_multinode": [
                    "WORLD_SIZE", "NUM_GPU_PER_NODE", "NODE_RANK",
                    "MASTER_ADDR", "MASTER_PORT",
                ],
                "forbidden_at_entrypoint": ["RANK", "LOCAL_RANK"],
            },
        },
    }
    jsonschema.Draft202012Validator(schema).validate(request)


def test_workflow_schema_rejects_controller_owned_training_outputs():
    schema = json.loads(
        (SKILL / "references" / "workflow.schema.json").read_text(
            encoding="utf-8"
        )
    )
    train_schema = schema["properties"]["actions"]["properties"]["train"]
    validator = jsonschema.Draft202012Validator({
        "$defs": schema["$defs"],
        **train_schema,
    })
    validator.validate({"command": ["dinov3", "train"]})
    for field in ("checkpoint", "contract"):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate({
                "command": ["dinov3", "train"],
                field: f"/user/owned/{field}",
            })


def test_workflow_schema_allows_only_base_checkpoint_candidates():
    schema = json.loads(
        (SKILL / "references" / "workflow.schema.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = schema["properties"]["training"]["properties"][
        "checkpoint_policy"
    ]
    assert checkpoint == {"const": "base_checkpoint_each_round"}
    assert "warm_start" not in schema["properties"]["training"]["properties"]


def test_application_image_resolves_to_data_services_runtime():
    resolved = resolve_tao_image.resolve_application_image(
        ROOT, "tao-run-dinov3-ssl-deft", "run"
    )
    versions = yaml.safe_load((ROOT / "versions.yaml").read_text(encoding="utf-8"))
    assert resolved["application"] == "tao-run-dinov3-ssl-deft"
    assert resolved["image"] == versions["images"]["tao_toolkit"][
        "data_services"
    ]
    assert resolved["source"] == "application.container_image"


def test_release_readiness_uses_the_declared_ds_image_contract():
    """Docs and resolver share one release-managed DS image, not a missing key."""
    text = (SKILL / "references/tao-container-integration.md").read_text()
    info = yaml.safe_load((SKILL / "references/skill_info.yaml").read_text())
    assert f"images.{info['container_image']}" in text
    assert "docker image inspect --format '{{index .RepoDigests 0}}'" in text
    assert "deft_dinov3_data_services" not in text
    assert "deft_dinov3_pyt" not in text
    assert "The currently declared general-purpose DS image is not a DEFT release" not in text
    assert "Keep the application PR draft" not in text
    assert "installed" in text and "packaged regression tests" in text
    assert "release team" in text


def test_benchmark_isolation_evaluation_requires_identity_metadata():
    """Keep a behavioral evaluation for renamed copies and disabled evaluation."""
    cases = json.loads((SKILL / "evals/evals.json").read_text())
    case = next(item for item in cases if item["id"].endswith("benchmark-isolation"))
    assert "content_sha256" in case["ground_truth"]
    assert "independent of evaluation" in case["ground_truth"]
    text = (SKILL / "references/adapter-contracts.md").read_text()
    for rule in (
        "data.benchmark_acquisition_units", "data.acquisition_unit_column",
        "content_sha256", "even when evaluation is disabled", "diagnostic_replay",
        "before publishing its artifact seal", "near-duplicates",
        "tao-data-services/pull/57", "omit both",
        '"contracts": {"benchmark_isolation": 1}', "stop before launch",
    ):
        assert rule in text, rule
    skill = (SKILL / "SKILL.md").read_text()
    assert "contracts.benchmark_isolation" in skill
    assert skill.count("adapter-contracts.md#held-out-benchmark-isolation") >= 2
    assert "stock DS image" not in skill


def test_multitask_recipe_is_offered_only_with_a_user_score_adapter():
    """DS ships no multi-task scorer; the skill must not present it as turnkey."""
    skill = (SKILL / "SKILL.md").read_text()
    assert "DS ships no multi-task scorer" in skill
    assert "adapter-contracts.md#task-scoring" in skill
    assert "`implementation_files`" in skill.split("## Approval contract")[1]
    contracts = (SKILL / "references/adapter-contracts.md").read_text()
    assert "/path/to/customer_score_adapter" in contracts
    assert "offer `grit-score` instead" in contracts
    source = os.environ.get("TAO_DATA_SERVICES_SOURCE")
    if not source:
        if os.environ.get("TAO_DEFT_REQUIRE_SCHEMA_PARITY") == "1":
            pytest.fail("TAO_DATA_SERVICES_SOURCE is required for recipe parity")
        return
    recipe = yaml.safe_load((Path(source) / "nvidia_tao_ds/mining/dinov3/workflow/recipes"
                             / "multi_task_round_robin.yaml").read_text())
    assert recipe["actions"]["score"]["command"][0] == "/path/to/customer_score_adapter"


def test_docker_launch_maps_scratch_and_checks_the_allocation_before_approval():
    """The packaged GRIT recipe needs TAO_LOCAL_SCRATCH; preflight must see the config."""
    text = (SKILL / "references/tao-container-integration.md").read_text()
    runtime = next(line for line in text.splitlines() if line.startswith("DEFT_RUNTIME=("))
    assert "TAO_LOCAL_SCRATCH=" in runtime
    assert "preflight /deft/run.yaml --gpu" in text
    assert text.index("preflight /deft/run.yaml --gpu") < text.index("plan /deft/run.yaml")
    assert "resolve_tao_image.py --skill-bank ." in text
    skill = (SKILL / "SKILL.md").read_text()
    assert "`preflight <config> --gpu`" in skill
    assert "TAO_LOCAL_SCRATCH" in skill
    source = os.environ.get("TAO_DATA_SERVICES_SOURCE")
    if source:
        recipe = yaml.safe_load((Path(source) / "nvidia_tao_ds/mining/dinov3/workflow/recipes"
                                 / "grit_score.yaml").read_text())
        scratch = recipe["actions"]["score"]["resources"]["local_scratch"]["path_environment"]
        assert f"{scratch}=" in runtime


def test_application_contract_declares_shared_paths_dynamic_output_and_cancel():
    info = yaml.safe_load(
        (SKILL / "references" / "skill_info.yaml").read_text(encoding="utf-8")
    )
    paths = info["path_contract"]
    assert paths["input_mode"] == "pre_mounted_shared_filesystem"
    assert set(paths["required_input_fields"]) == {
        "model.base_checkpoint",
        "training.base_spec",
        "data.target_manifest",
        "data.source_payload_contract",
    }
    assert {
        "data.source_parts[]",
        "actions.*.implementation_files[]",
        "continuation.previous_run_dir",
        "continuation.previous_release_lock",
    }.issubset(paths["conditional_input_fields"])
    assert paths["output_directory_field"] == "output.run_dir"
    action = info["actions"]["run"]
    assert action["outputs"] == {
        "run_dir": {"type": "folder", "config_path": "output.run_dir"}
    }
    assert action["cancellation"] == {
        "signal": "SIGTERM", "behavior": "workflow_cancel"
    }
