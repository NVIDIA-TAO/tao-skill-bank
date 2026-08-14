# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for producer action requests consumed by Kubernetes."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "skills/platform/tao-run-on-kubernetes/scripts/render_action_job.py"
)
SPEC = importlib.util.spec_from_file_location("render_action_job", SCRIPT)
assert SPEC and SPEC.loader
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


RESULTS = "/localhome/user/workspace/results/run_iaa_k8s"
CONFIG = RESULTS + "/config"
PATCHES = "/opt/tao-skill-bank/skills/applications/tao-run-deft-iaa/patches"
CACHE = "/localhome/user/workspace/cache"


def iaa_pool_embed_request():
    """The five-mount shape emitted for a real IAA pool_embed action."""
    output = RESULTS + "/embeddings/source/embeddings.parquet"
    return {
        "schema_version": "1",
        "platform": "kubernetes",
        "workload_image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
        "spec_bundle": {
            "network_arch": "deft-iaa",
            "action": "deft-iaa-pool_embed-abc123",
            "image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
            "mode": "args",
            "command": "embedding",
            "args": [
                "text_embeddings",
                "-e",
                "/specs/text_embed_spec.yaml",
                "input_parquet=/results/embeddings/source/source_pool.parquet",
                "output_parquet=/results/embeddings/source/embeddings.parquet",
            ],
            "compute_shape": {"gpus": 1, "nodes": 1},
        },
        "mounts": [
            {"source": RESULTS, "target": "/results", "read_only": False},
            {"source": RESULTS, "target": RESULTS, "read_only": False},
            {"source": CONFIG, "target": "/specs", "read_only": True},
            {"source": PATCHES, "target": "/patches", "read_only": True},
            {"source": CACHE, "target": "/cache", "read_only": False},
        ],
        "environment": {
            "HOME": "/tmp",
            "PYTHONPATH": "/patches",
            "HF_HOME": "/cache/huggingface",
            "XDG_CACHE_HOME": "/cache",
        },
        "forward_env": [],
        "fresh_outputs": [output],
    }


def iaa_staging_map():
    return {
        "schema_version": "1",
        "sources": [
            {"source": RESULTS, "sub_path": "jobs/abc123/results"},
            {"source": CONFIG, "sub_path": "jobs/abc123/results/config"},
            {"source": PATCHES, "sub_path": "jobs/abc123/patches"},
            {"source": CACHE, "sub_path": "jobs/abc123/cache"},
        ],
    }


def render(request=None, staging=None, **kwargs):
    return renderer.render_action_job(
        request or iaa_pool_embed_request(),
        staging or iaa_staging_map(),
        job_id="tao-job-abc123",
        namespace="tao-jobs",
        pvc_claim="tao-workspace",
        **kwargs,
    )


def container(manifest):
    job = yaml.safe_load(manifest)
    return job, job["spec"]["template"]["spec"]["containers"][0]


def test_iaa_five_mount_request_preserves_aliases_and_access_modes():
    job, action = container(render())
    mounts = {item["mountPath"]: item for item in action["volumeMounts"]}

    assert job["metadata"]["namespace"] == "tao-jobs"
    assert job["metadata"]["annotations"]["tao.nvidia.com/job-record-id"] == "tao-job-abc123"
    assert set(mounts) == {
        "/dev/shm",
        "/results",
        RESULTS,
        "/specs",
        "/patches",
        "/cache",
    }
    assert mounts["/results"]["subPath"] == "jobs/abc123/results"
    assert mounts[RESULTS]["subPath"] == "jobs/abc123/results"
    assert mounts["/results"]["readOnly"] is False
    assert mounts[RESULTS]["readOnly"] is False
    assert mounts["/specs"]["readOnly"] is True
    assert mounts["/patches"]["readOnly"] is True
    assert mounts["/cache"]["readOnly"] is False
    assert action["command"] == ["embedding"]
    assert action["args"] == iaa_pool_embed_request()["spec_bundle"]["args"]
    assert action["resources"]["limits"]["nvidia.com/gpu"] == "1"


def test_no_credentials_or_registry_secret_means_no_dangling_secret_refs():
    job, action = container(render())
    pod = job["spec"]["template"]["spec"]
    assert pod["imagePullSecrets"] == []
    assert action["envFrom"] == []


def test_forwarded_credentials_are_secret_references_not_inline_values():
    request = iaa_pool_embed_request()
    request["forward_env"] = ["HF_TOKEN"]
    manifest = render(
        request=request,
        credential_secret="tao-creds-abc123",
        image_pull_secret="ngc-pull-secret",
    )
    job, action = container(manifest)
    assert job["spec"]["template"]["spec"]["imagePullSecrets"] == [
        {"name": "ngc-pull-secret"}
    ]
    assert action["envFrom"] == [
        {"secretRef": {"name": "tao-creds-abc123"}}
    ]
    inline_names = {item["name"] for item in action["env"]}
    assert "HF_TOKEN" not in inline_names
    assert "HF_TOKEN" not in manifest


def test_forwarded_credentials_require_a_secret():
    request = iaa_pool_embed_request()
    request["forward_env"] = ["HF_TOKEN"]
    with pytest.raises(renderer.RenderError, match="credential-secret"):
        render(request=request)


def test_inline_and_forwarded_name_collision_is_rejected():
    request = iaa_pool_embed_request()
    request["environment"]["HF_TOKEN"] = "must-not-be-inline"
    request["forward_env"] = ["HF_TOKEN"]
    with pytest.raises(renderer.RenderError, match="must not be present"):
        render(request=request, credential_secret="tao-creds-abc123")


def test_command_tokens_are_not_interpolated_into_a_shell_string():
    request = iaa_pool_embed_request()
    request["spec_bundle"]["args"].append("literal=$(do-not-run); 'quoted'")
    _, action = container(render(request=request))
    assert action["command"] == ["embedding"]
    assert action["args"][-1] == "literal=$(do-not-run); 'quoted'"


def test_empty_argument_and_environment_value_are_preserved():
    request = iaa_pool_embed_request()
    request["spec_bundle"]["args"].append("")
    request["environment"]["OPTIONAL_SETTING"] = ""
    _, action = container(render(request=request))
    assert action["args"][-1] == ""
    environment = {item["name"]: item["value"] for item in action["env"]}
    assert environment["OPTIONAL_SETTING"] == ""


def test_duplicate_source_aliases_need_only_one_staging_entry():
    manifest = render()
    _, action = container(manifest)
    workspace_mounts = [
        item for item in action["volumeMounts"] if item["name"] == "workspace"
    ]
    assert len(workspace_mounts) == 5
    assert sum(item["subPath"] == "jobs/abc123/results" for item in workspace_mounts) == 2


def test_missing_staging_mapping_is_rejected():
    staging = iaa_staging_map()
    staging["sources"] = [
        row for row in staging["sources"] if row["source"] != PATCHES
    ]
    with pytest.raises(renderer.RenderError, match="no staged PVC subPath"):
        render(staging=staging)


def test_undeclared_staging_mapping_is_rejected():
    staging = iaa_staging_map()
    staging["sources"].append(
        {"source": "/undeclared/input", "sub_path": "jobs/abc123/extra"}
    )
    with pytest.raises(renderer.RenderError, match="undeclared mount sources"):
        render(staging=staging)


@pytest.mark.parametrize("sub_path", ["/absolute", "../escape", "a/../escape", ".", "a\\b"])
def test_unsafe_pvc_subpaths_are_rejected(sub_path):
    staging = iaa_staging_map()
    staging["sources"][0]["sub_path"] = sub_path
    with pytest.raises(renderer.RenderError, match="sub_path|subPath|relative|traversal"):
        render(staging=staging)


def test_duplicate_mount_target_is_rejected():
    request = iaa_pool_embed_request()
    request["mounts"][1]["target"] = "/results"
    with pytest.raises(renderer.RenderError, match="repeats Kubernetes mount target"):
        render(request=request)


def test_fresh_outputs_require_a_writable_covering_mount():
    request = iaa_pool_embed_request()
    request["mounts"][0]["read_only"] = True
    request["mounts"][1]["read_only"] = True
    with pytest.raises(renderer.RenderError, match="not covered by a writable"):
        render(request=request)


def test_staging_map_rejects_duplicate_sources():
    staging = iaa_staging_map()
    staging["sources"].append(dict(staging["sources"][0]))
    with pytest.raises(renderer.RenderError, match="repeats source"):
        render(staging=staging)


def test_distinct_sources_cannot_share_one_exact_pvc_subpath():
    staging = iaa_staging_map()
    staging["sources"][1]["sub_path"] = staging["sources"][0]["sub_path"]
    with pytest.raises(renderer.RenderError, match="one PVC subPath"):
        render(staging=staging)


def test_real_iaa_job_record_id_is_normalized_without_losing_identity():
    job_id = (
        "data-services-deft-iaa-pool_embed-"
        "0123456789abcdef-1234567890abcdef-abcdef"
    )
    manifest = renderer.render_action_job(
        iaa_pool_embed_request(),
        iaa_staging_map(),
        job_id=job_id,
        namespace="tao-jobs",
        pvc_claim="tao-workspace",
    )
    job = yaml.safe_load(manifest)
    name = job["metadata"]["name"]
    assert len(name) <= renderer.MAX_JOB_NAME
    assert renderer.DNS_LABEL_RE.fullmatch(name)
    assert "_" not in name
    assert job["metadata"]["annotations"]["tao.nvidia.com/job-record-id"] == job_id
    assert renderer.kubernetes_job_name(job_id) == name


def test_normalized_names_cannot_collide_after_character_replacement():
    underscored = renderer.kubernetes_job_name("action_with_underscore")
    hyphenated = renderer.kubernetes_job_name("action-with-underscore")
    assert underscored != hyphenated


def test_cli_renders_the_same_contract(tmp_path):
    request_path = tmp_path / "action.json"
    staging_path = tmp_path / "staging.json"
    request_path.write_text(json.dumps(iaa_pool_embed_request()), encoding="utf-8")
    staging_path.write_text(json.dumps(iaa_staging_map()), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "render",
            "--request",
            str(request_path),
            "--staging-map",
            str(staging_path),
            "--job-id",
            "tao-job-abc123",
            "--namespace",
            "tao-jobs",
            "--pvc-claim",
            "tao-workspace",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    job = yaml.safe_load(completed.stdout)
    assert job["kind"] == "Job"
    assert job["metadata"]["name"] == "tao-job-abc123"


def test_name_cli_returns_backend_object_name():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "name", "--job-id", "IAA.pool_embed_01"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == renderer.kubernetes_job_name("IAA.pool_embed_01")
