# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for producer action requests consumed by Kubernetes."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
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
IAA_RUNTIME = str(
    REPO / "skills/applications/tao-run-deft-iaa/scripts"
)
IAA_PATCHES = str(
    REPO / "skills/applications/tao-run-deft-iaa/patches"
)


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


def iaa_controller_trees(root):
    root = Path(root)
    controller = root / "skills"
    runtime = controller / "applications/tao-run-deft-iaa/scripts"
    patches = root / "patches"
    shutil.copytree(IAA_RUNTIME, runtime, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(IAA_PATCHES, patches, ignore=shutil.ignore_patterns("__pycache__"))
    artifact = controller / "core/tao-artifacts/references/spec_bundle.schema.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    return controller, runtime, patches


def signed_adapter_request(base, name="gap_analysis", controller=None, patches=None):
    if controller is None or patches is None:
        fixture = Path(tempfile.mkdtemp(dir=base))
        controller, runtime, patches = iaa_controller_trees(fixture)
    else:
        controller = Path(controller)
        patches = Path(patches)
        runtime = controller / "applications/tao-run-deft-iaa/scripts"
    controller = str(controller)
    runtime = str(runtime)
    patches = str(patches)
    runtime_digest = renderer._python_tree_sha256(Path(runtime) / "iaa_deft")
    output = RESULTS + "/iter_1/gaps/kpi_gaps.parquet"
    request = {
        "schema_version": "1", "workflow": "tao-run-deft-iaa",
        "platform": "kubernetes", "name": name, "label": "iter1",
        "runtime_sha256": runtime_digest, "gpu_ids": [],
        "passed_hf_token": False, "forward_env": [],
        "workload_image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
        "spec_bundle": {
            "network_arch": "iaa-adapter", "action": "signed-adapter-action",
            "image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
            "mode": "args", "command": "python3",
            "args": ["/iaa-runtime/run_iaa_compute.py", name,
                     "--results-dir", "/results", "--label", "iter1"],
            "compute_shape": {"gpus": 0, "nodes": 1},
        },
        "mounts": [
            {"source": RESULTS, "target": "/results", "read_only": False},
            {"source": RESULTS, "target": RESULTS, "read_only": False},
            {"source": CONFIG, "target": "/specs", "read_only": True},
            {"source": patches, "target": "/patches", "read_only": True},
            {"source": CACHE, "target": "/cache", "read_only": False},
            {"source": runtime, "target": "/iaa-runtime", "read_only": True},
        ],
        "environment": {
            "HOME": "/tmp", "PYTHONPATH": "/patches",
            "HF_HOME": "/cache/huggingface", "XDG_CACHE_HOME": "/cache",
            "IAA_COMPUTE_FRAME": "kubernetes",
        },
        "controller_snapshot": renderer._snapshot_manifest(Path(controller)),
        "patches_snapshot": renderer._snapshot_manifest(Path(patches)),
        "fresh_outputs": [output],
    }
    request["request_sha256"] = renderer._canonical_request_sha256(request)
    staging = iaa_staging_map()
    next(row for row in staging["sources"] if row["source"] == PATCHES)["source"] = patches
    next(row for row in staging["sources"] if row["source"] == patches).update({
        "sha256": request["patches_snapshot"]["sha256"],
    })
    staging["sources"].append({
        "source": controller, "sub_path": "jobs/abc123/controller",
        "sha256": request["controller_snapshot"]["sha256"],
    })
    return request, staging


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


def test_signed_iaa_adapter_is_cpu_only_and_runtime_digest_bound(tmp_path):
    request, staging = signed_adapter_request(tmp_path)
    job, action = container(render(request=request, staging=staging))
    assert action["resources"] == {}
    assert "nvidia.com/gpu" not in json.dumps(action)
    mounts = {item["mountPath"]: item for item in action["volumeMounts"]}
    assert mounts["/iaa-runtime"]["readOnly"] is True
    assert mounts["/iaa-runtime"]["subPath"].endswith(
        "controller/applications/tao-run-deft-iaa/scripts"
    )
    assert job["metadata"]["annotations"]["tao.nvidia.com/runtime-sha256"] == request["runtime_sha256"]


def test_visualize_finish_accepts_fixed_native_math_thread_caps(tmp_path):
    request, staging = signed_adapter_request(tmp_path, name="visualize_finish")
    request["environment"].update(renderer.IAA_VISUALIZE_THREAD_CAPS)
    request["request_sha256"] = renderer._canonical_request_sha256(request)

    _, action = container(render(request=request, staging=staging))
    environment = {item["name"]: item["value"] for item in action["env"]}

    assert all(
        environment[name] == value
        for name, value in renderer.IAA_VISUALIZE_THREAD_CAPS.items()
    )


def test_zero_gpu_non_adapter_is_rejected():
    request = iaa_pool_embed_request()
    request["spec_bundle"]["compute_shape"]["gpus"] = 0
    with pytest.raises(renderer.RenderError, match="zero-GPU actions require"):
        render(request=request)


def test_adapter_requires_valid_signature_and_read_only_bound_runtime(tmp_path):
    request, staging = signed_adapter_request(tmp_path)
    request["request_sha256"] = "0" * 64
    with pytest.raises(renderer.RenderError, match="signature"):
        render(request=request, staging=staging)

    request, staging = signed_adapter_request(tmp_path)
    next(row for row in request["mounts"] if row["target"] == "/iaa-runtime")["read_only"] = False
    request["request_sha256"] = renderer._canonical_request_sha256(request)
    with pytest.raises(renderer.RenderError, match="read-only"):
        render(request=request, staging=staging)

    request, staging = signed_adapter_request(tmp_path)
    controller_root = request["controller_snapshot"]["root"]
    next(row for row in staging["sources"] if row["source"] == controller_root)["sha256"] = "0" * 64
    with pytest.raises(renderer.RenderError, match="staging receipt"):
        render(request=request, staging=staging)

    request, staging = signed_adapter_request(tmp_path)
    next(
        row for row in request["mounts"] if row["target"] == "/iaa-runtime"
    )["source"] = request["controller_snapshot"]["root"]
    request["request_sha256"] = renderer._canonical_request_sha256(request)
    with pytest.raises(renderer.RenderError, match="derived"):
        render(request=request, staging=staging)


def test_adapter_rejects_credential_like_environment_extra(tmp_path):
    request, staging = signed_adapter_request(tmp_path)
    request["environment"]["AWS_SECRET_ACCESS_KEY"] = "must-not-render"
    request["request_sha256"] = renderer._canonical_request_sha256(request)
    with pytest.raises(renderer.RenderError, match="environment"):
        render(request=request, staging=staging)


@pytest.mark.parametrize(
    ("tree", "operation", "relative"),
    [
        ("controller", "mutate", "run_iaa_compute.py"),
        ("controller", "extra", "unexpected.py"),
        ("controller", "missing", "run_iaa_compute.py"),
        (
            "controller-root", "mutate",
            "core/tao-artifacts/references/spec_bundle.schema.json",
        ),
        ("patches", "mutate", None),
        ("patches", "extra", "unexpected.patch"),
        ("patches", "missing", None),
    ],
)
def test_adapter_rejects_complete_snapshot_tree_changes(
    tmp_path, tree, operation, relative,
):
    controller, runtime, patches = iaa_controller_trees(tmp_path)
    request, staging = signed_adapter_request(
        tmp_path, controller=controller, patches=patches
    )
    target_root = (
        runtime if tree == "controller"
        else controller if tree == "controller-root"
        else patches
    )
    files = sorted(path for path in target_root.rglob("*") if path.is_file())
    target = target_root / relative if relative else files[0]
    if operation == "mutate":
        target.write_bytes(target.read_bytes() + b"\nmutation")
    elif operation == "extra":
        target.write_text("extra", encoding="utf-8")
    else:
        target.unlink()
    snapshot = "controller" if tree.startswith("controller") else "patches"
    with pytest.raises(renderer.RenderError, match=f"{snapshot}_snapshot"):
        render(request=request, staging=staging)


def test_no_credentials_or_registry_secret_means_no_dangling_secret_refs():
    job, action = container(render())
    pod = job["spec"]["template"]["spec"]
    assert pod["imagePullSecrets"] == []
    assert "envFrom" not in action


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
    assert "envFrom" not in action
    environment = {item["name"]: item for item in action["env"]}
    assert environment["HF_TOKEN"] == {
        "name": "HF_TOKEN",
        "valueFrom": {
            "secretKeyRef": {
                "name": "tao-creds-abc123",
                "key": "HF_TOKEN",
            }
        },
    }


def test_secret_projects_only_explicitly_forwarded_names():
    request = iaa_pool_embed_request()
    request["forward_env"] = ["HF_TOKEN", "AWS_ACCESS_KEY_ID"]
    _, action = container(
        render(request=request, credential_secret="shared-credential-secret")
    )

    assert "envFrom" not in action
    forwarded = {
        item["name"]: item["valueFrom"]["secretKeyRef"]
        for item in action["env"]
        if "valueFrom" in item
    }
    assert forwarded == {
        "HF_TOKEN": {"name": "shared-credential-secret", "key": "HF_TOKEN"},
        "AWS_ACCESS_KEY_ID": {
            "name": "shared-credential-secret",
            "key": "AWS_ACCESS_KEY_ID",
        },
    }


def test_forwarded_credentials_require_a_secret():
    request = iaa_pool_embed_request()
    request["forward_env"] = ["HF_TOKEN"]
    with pytest.raises(renderer.RenderError, match="credential-secret"):
        render(request=request)


def test_credential_secret_is_rejected_when_no_names_are_approved():
    with pytest.raises(renderer.RenderError, match="must be omitted"):
        render(credential_secret="unexpected-shared-secret")


def test_inline_and_forwarded_name_collision_is_rejected():
    request = iaa_pool_embed_request()
    request["environment"]["HF_TOKEN"] = "must-not-be-inline"
    request["forward_env"] = ["HF_TOKEN"]
    with pytest.raises(renderer.RenderError, match="must not be present"):
        render(request=request, credential_secret="tao-creds-abc123")


@pytest.mark.parametrize("schema_version", [None, "0", "2", 1])
def test_unsupported_request_schema_version_is_rejected(schema_version):
    request = iaa_pool_embed_request()
    if schema_version is None:
        request.pop("schema_version")
    else:
        request["schema_version"] = schema_version
    with pytest.raises(renderer.RenderError, match="request.schema_version"):
        render(request=request)


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
