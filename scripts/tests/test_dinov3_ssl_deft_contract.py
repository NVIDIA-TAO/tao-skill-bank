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
