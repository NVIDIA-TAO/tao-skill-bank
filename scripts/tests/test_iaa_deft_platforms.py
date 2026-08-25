# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform contract tests for IAA DEFT TAO actions."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]
IAA_SCRIPTS = REPO / "skills" / "applications" / "tao-run-deft-iaa" / "scripts"
sys.path.insert(0, str(IAA_SCRIPTS))
import run_deft_action as action  # noqa: E402
import run_deft_container as docker_action  # noqa: E402
import run_deft_cli as virtualenv_cli  # noqa: E402
import prepare_deft_config as prepare_config  # noqa: E402
import init_deft_state  # noqa: E402
import deft_action_contract as action_contract  # noqa: E402
import commit_stage  # noqa: E402
import audit_deft_run  # noqa: E402
from deft_action_contract import (  # noqa: E402
    platform_evidence_error,
    validate_tao_virtualenv,
)
from command_contract import (  # noqa: E402
    expected_container_command,
    expected_fresh_outputs,
    expected_image_kind,
    expected_stage_directory,
)


PLATFORMS = (
    "docker", "slurm", "kubernetes", "brev", "virtualenv", "airflow",
)
PYT_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt"
DS_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services"


def _managed_sdg_args(platform: str = "docker", *, generation_nodes: int = 1) -> list[str]:
    single_host_brev = platform == "brev" and generation_nodes == 1
    single_host_airflow = platform == "airflow" and generation_nodes == 1
    single_host_six_plus_two = single_host_brev or single_host_airflow
    image_ids = (
        "0,1,2,3" if single_host_six_plus_two
        else "0,1,2,3,4,5,6,7"
        if platform in {"slurm", "kubernetes", "brev", "airflow"}
        else "0"
    )
    return [
        "--visible-gpu-ids",
        "0,1,2,3,4,5,6,7" if single_host_six_plus_two else "0",
        "--image-edit-gpu-ids",
        image_ids,
        "--vlm-gpu-ids",
        "4" if single_host_six_plus_two else "0",
        "--llm-gpu-ids",
        "5" if single_host_six_plus_two else (
            "1" if platform in {"slurm", "kubernetes", "brev", "airflow"} else "0"
        ),
        "--generation-nodes",
        str(generation_nodes),
        *(["--num-gpus", "2", "--gpu-ids", "6,7"] if single_host_six_plus_two else []),
    ]


SPEC_NAMES = (
    "deft_config.yaml",
    "sdg_config.yaml",
    "tao_spec.yaml",
    "text_embed_spec.yaml",
    "image_embed_spec.yaml",
    "mining_spec.yaml",
    "approval.json",
)
ACTION_CASES = (
    ("pool_embed", "baseline"),
    ("evaluate", "baseline"),
    ("evaluate", "iter1"),
    ("target_embed", "iter1"),
    ("knn", "iter1"),
    ("train", "iter1"),
    ("viz_weak_embed", "iter1"),
    ("viz_mined_embed", "iter1"),
    ("viz_previous_embed", "iter1"),
)
DATASET_ACTIONS = {
    "evaluate",
    "train",
    "viz_weak_embed",
    "viz_mined_embed",
    "viz_previous_embed",
    "report",
}


def test_all_action_consumers_bind_the_initializer_run_specs():
    expected = init_deft_state.RUN_SPEC_NAMES

    assert action_contract.RUN_SPEC_NAMES == expected
    assert docker_action.RUN_SPEC_NAMES == expected
    assert audit_deft_run.RUN_SPEC_NAMES == expected


@pytest.fixture(autouse=True)
def _mock_expensive_virtualenv_contract_probes(monkeypatch, tmp_path):
    """Platform tests mock the contract boundary; verifier tests exercise it."""
    monkeypatch.setenv("TAO_STATE_DIR", str(tmp_path / ".tao"))

    def validate(
        path, *, profile, probe_imports, required_cli=None, minimum_gpus=None,
        gpu_ids=None
    ):
        assert profile in {"pyt", "ds"}
        return Path(path).expanduser().resolve()

    def resolve(*, platform, legacy, pyt, ds, probe_imports):
        if platform != "virtualenv":
            if any(value is not None for value in (legacy, pyt, ds)):
                raise ValueError("virtualenv arguments are valid only")
            return None
        if legacy is not None and (pyt is not None or ds is not None):
            raise ValueError("cannot be combined")
        if legacy is not None:
            pyt = ds = legacy
        if pyt is None or ds is None:
            raise ValueError(
                "--pyt-virtualenv and --ds-virtualenv are both required"
            )
        return {"pyt": Path(pyt).resolve(), "ds": Path(ds).resolve()}

    monkeypatch.setattr(action_contract, "validate_tao_virtualenv", validate)
    monkeypatch.setattr(audit_deft_run, "validate_tao_virtualenv", validate)
    monkeypatch.setattr(prepare_config, "resolve_virtualenv_profiles", resolve)
    monkeypatch.setattr(init_deft_state, "resolve_virtualenv_profiles", resolve)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_state_initializer_accepts_every_tao_platform(platform):
    argv = [
        "--results-dir",
        "/workspace/results/run",
        "--workspace",
        "/workspace",
        "--dataset-root",
        "/workspace/data/iaa",
        "--images-archive",
        "/inputs/images_raw.tar",
        "--metadata-archive",
        "/inputs/meta.tar.gz",
        "--max-iterations",
        "1",
        "--platform",
        platform,
        "--pyt-image",
        PYT_IMAGE,
        "--ds-image",
        DS_IMAGE,
        "--deft-config",
        "/workspace/results/run/config/deft_config.yaml",
        "--tao-spec",
        "/workspace/results/run/config/tao_spec.yaml",
        "--sdg-config",
        "/workspace/results/run/config/sdg_config.yaml",
    ]
    if platform == "virtualenv":
        argv.extend(
            [
                "--pyt-virtualenv",
                "/workspace/tao-pyt-venv",
                "--ds-virtualenv",
                "/workspace/tao-ds-venv",
            ]
        )
    args = init_deft_state._build_parser().parse_args(argv)  # noqa: SLF001
    assert args.platform == platform


def _write_fixture(tmp_path: Path, platform: str, *, docker_remote: bool = False):
    if docker_remote and platform != "docker":
        raise ValueError("docker_remote test fixtures require platform=docker")
    workspace = tmp_path / "workspace"
    results = workspace / "results" / f"run_{platform}"
    config_dir = results / "config"
    dataset = workspace / "data" / "iaa_v31_tao_ft"
    (dataset / "images").mkdir(parents=True)
    (dataset / "captions").mkdir()
    config_dir.mkdir(parents=True)
    tao_model = (
        workspace
        / "cache/huggingface/hub/models--google--siglip2-so400m-patch16-256"
    )
    (tao_model / "blobs").mkdir(parents=True)
    (tao_model / "blobs/model").write_bytes(b"siglip-fixture")
    (tao_model / "snapshots/revision").mkdir(parents=True)
    (tao_model / "snapshots/revision/model").symlink_to("../../blobs/model")
    (tao_model / "refs").mkdir()
    (tao_model / "refs/main").write_text("revision", encoding="utf-8")
    payloads = {
        "deft_config.yaml": "experiment:\n  results_path: /results\n",
        "sdg_config.yaml": "schema_version: '1'\n",
        "tao_spec.yaml": "train:\n  num_gpus: 1\n",
        "text_embed_spec.yaml": "model: /results/model\n",
        "image_embed_spec.yaml": "model: /results/model\n",
        "mining_spec.yaml": "topn: 25\n",
        "approval.json": '{"approved": true}\n',
    }
    for name, body in payloads.items():
        (config_dir / name).write_text(body, encoding="utf-8")
    hashes = {
        name: hashlib.sha256((config_dir / name).read_bytes()).hexdigest()
        for name in SPEC_NAMES
    }
    virtualenvs = None
    if platform == "virtualenv":
        virtualenvs = {}
        for profile, clis in (("pyt", ("clip",)), ("ds", ("embedding", "tmm"))):
            venv = workspace / f"tao-{profile}-venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
            os.symlink(sys.executable, venv / "bin" / "python")
            os.symlink(sys.executable, venv / "bin" / "python3")
            for cli in clis:
                entrypoint = venv / "bin" / cli
                entrypoint.write_text(
                    f"#!{venv / 'bin' / 'python'}\n", encoding="utf-8"
                )
                entrypoint.chmod(0o755)
            virtualenvs[profile] = str(venv)
    state = {
        "schema_version": "3",
        "workflow": "tao-run-deft-iaa",
        "results_dir": str(results),
        "max_iterations": 1,
        "config": {
            "iaa_deft_bundle_sha256": action._python_tree_sha256(
                IAA_SCRIPTS / "iaa_deft"
            ),
            "workspace": str(workspace),
            "dataset_root": str(dataset),
            "config_dir": str(config_dir),
            "deft_config": str(config_dir / "deft_config.yaml"),
            "sdg_config": str(config_dir / "sdg_config.yaml"),
            "tao_spec": str(config_dir / "tao_spec.yaml"),
            "spec_sha256": hashes,
            "platform": platform,
            "docker_remote": docker_remote,
            "virtualenvs": virtualenvs,
            "pyt_image": PYT_IMAGE,
            "ds_image": DS_IMAGE,
            "num_gpus": 1,
            "gpu_ids": [0],
            "requires_hf_token": False,
        },
    }
    (results / "deft_state.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    stage = results / "embeddings" / "source"
    output = stage / "embeddings.parquet"
    args = argparse.Namespace(
        results_dir=results,
        image="ds",
        stage_dir=stage,
        name="pool_embed",
        pass_hf_token=False,
        fresh_output=[output],
        command=[
            "embedding",
            "text_embeddings",
            "-e",
            "/specs/text_embed_spec.yaml",
            "input_parquet=/results/embeddings/source/source_pool.parquet",
            "output_parquet=/results/embeddings/source/embeddings.parquet",
        ],
    )
    return workspace, results, stage, output, args


def test_tao_cache_subset_excludes_duplicate_blobs(tmp_path):
    workspace, _, _, _, _ = _write_fixture(tmp_path, "brev")
    manifest = action._tao_cache_subset(workspace / "cache")
    paths = [entry["path"] for entry in manifest["entries"]]
    assert any("/snapshots/" in path for path in paths)
    assert any("/refs/" in path for path in paths)
    assert not any("/blobs/" in path for path in paths)


def test_tao_cache_preflight_reports_signed_inventory(tmp_path):
    workspace, _, _, _, _ = _write_fixture(tmp_path, "brev")
    report = action.tao_cache_preflight(workspace / "cache")
    assert report["status"] == "ok"
    assert report["model"] == "google/siglip2-so400m-patch16-256"
    assert report["entries"] > 0
    assert report["bytes"] > 0
    assert len(report["manifest_sha256"]) == 64


@pytest.mark.parametrize(
    ("mode", "path", "expected"),
    [
        ("native", None, None),
        (
            "image_forward_compat",
            "/usr/local/cuda/compat/lib.real:/usr/local/cuda/lib64",
            "/usr/local/cuda/compat/lib.real:/usr/local/cuda/lib64",
        ),
    ],
)
def test_action_consumes_image_and_gpu_bound_cuda_receipt(tmp_path, mode, path, expected):
    _, results, _, _, args = _write_fixture(tmp_path, "brev")
    receipt = {
        "schema_version": 1,
        "workflow": "tao-run-deft-iaa",
        "image": DS_IMAGE,
        "gpu_ids": [0],
        "status": "PASS",
        "compatibility_mode": mode,
        "compatibility_path": path,
    }
    (results / "config/cuda-runtime-ds.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    request_path, request = action.prepare(args)
    assert request["environment"].get("LD_LIBRARY_PATH") == expected
    _, validated = action._load_request_envelope(request_path)
    assert validated["request_sha256"] == request["request_sha256"]


def test_action_rejects_cuda_receipt_for_another_image(tmp_path):
    _, results, _, _, args = _write_fixture(tmp_path, "brev")
    receipt = {
        "schema_version": 1,
        "workflow": "tao-run-deft-iaa",
        "image": PYT_IMAGE,
        "gpu_ids": [0],
        "status": "PASS",
        "compatibility_mode": "native",
        "compatibility_path": None,
    }
    (results / "config/cuda-runtime-ds.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="image does not match"):
        action.prepare(args)


@pytest.mark.parametrize("kind", ["broken", "escape"])
def test_tao_cache_subset_rejects_unsafe_snapshot(tmp_path, kind):
    workspace, _, _, _, _ = _write_fixture(tmp_path, "brev")
    model = workspace / "cache/huggingface/hub/models--google--siglip2-so400m-patch16-256"
    snapshot = model / "snapshots/revision/model"
    snapshot.unlink()
    if kind == "broken":
        snapshot.symlink_to("../../blobs/missing")
        match = "No such file"
    else:
        outside = workspace / "cache/outside"
        outside.write_bytes(b"escape")
        snapshot.symlink_to("../../../../../outside")
        match = "escapes selected model"
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        action._tao_cache_subset(workspace / "cache")


def _remote_scope(platform: str) -> str:
    return f"/remote/tao-iaa-tests/{platform}/embeddings/source"


def _attest_remote(request_path: Path, request: dict) -> str:
    scope = _remote_scope(request["platform"])
    action.attest_staged(
        argparse.Namespace(
            request=request_path,
            backend_scope=scope,
            absent_path=request["staging_absent_paths"],
        )
    )
    return scope


def _open_job_record(
    request_path: Path,
    request: dict,
    *,
    job_id: str,
    platform: str | None = None,
    results_scope: str | None = None,
    submitted_at: str | None = None,
) -> tuple[Path, dict]:
    platform = request["platform"] if platform is None else platform
    submitted_at = submitted_at or dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"
    )
    if results_scope is None:
        results_scope = (
            _remote_scope(platform)
            if platform in {"slurm", "kubernetes", "brev", "airflow"}
            else request["stage_dir"]
        )
    record = {
        "schema_version": 1,
        "id": job_id,
        "platform": platform,
        "backend_ref": None,
        "image": request["record_image"],
        "network_arch": request["spec_bundle"]["network_arch"],
        "action": request["spec_bundle"]["action"],
        "results_dir": results_scope,
        "storage_tier": "A",
        "upload_excludes": request["spec_bundle"]["upload_excludes"],
        "submitted_at": submitted_at,
        "transitions": [
            {
                "ts": submitted_at,
                "state": "PENDING",
                "message": "opened",
                "source": "agent",
            }
        ],
        "terminal_state": None,
        "terminal_write_by": None,
        "redacted": True,
    }
    jobs = Path(request["job_state_dir"]) / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    record_path = jobs / f"{job_id}.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return record_path, record


def _finish_job_record(record_path: Path, record: dict, *, state: str = "COMPLETE"):
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    record["backend_ref"] = f"native:{record['platform']}:123"
    record["transitions"].extend(
        [
            {"ts": now, "state": "RUNNING", "message": "native", "source": "agent"},
            {"ts": now, "state": state, "message": "terminal", "source": "agent"},
        ]
    )
    record["terminal_state"] = state
    record["terminal_write_by"] = "agent"
    record_path.write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize("platform", PLATFORMS)
def test_prepare_emits_same_valid_action_contract_for_every_platform(tmp_path, platform):
    workspace, results, stage, output, args = _write_fixture(tmp_path, platform)
    request_path, request = action.prepare(args)

    assert request_path == stage / "pool_embed.action.json"
    assert request["platform"] == platform
    assert request["spec_bundle"]["command"] == "embedding"
    assert request["spec_bundle"]["action"].startswith("deft-iaa-pool_embed-")
    legacy_identity = "\0".join(
        (
            str(results),
            stage.relative_to(results).as_posix(),
            "pool_embed",
            "1",
            str(request["started_ns"]),
        )
    )
    assert request["spec_bundle"]["action"] == (
        "deft-iaa-pool_embed-"
        + hashlib.sha256(legacy_identity.encode("utf-8")).hexdigest()[:16]
    )
    assert request["spec_bundle"]["compute_shape"] == {"gpus": 1, "nodes": 1}
    assert request["gpu_ids"] == [0]
    assert request["spec_bundle"]["declared_outputs"] == [
        {"spec_key": "embeddings.parquet", "type": "file"}
    ]
    assert request["fresh_outputs"] == [str(output)]
    runtime_dir = Path(request["platform_runtime_dir"])
    assert runtime_dir == stage / ".tao-runtime" / "pool_embed.attempt-1"
    assert runtime_dir.is_dir()
    mounts = {(item["source"], item["target"]) for item in request["mounts"]}
    assert (str(results), "/results") in mounts
    assert (str(results), str(results)) in mounts
    assert (str(workspace / "data"), "/data") not in mounts
    assert (str(workspace / "data"), str(workspace / "data")) not in mounts
    declared = {
        item["spec_key"] for item in request["spec_bundle"]["declared_inputs"]
    }
    assert "dataset_parent" not in declared
    if platform == "virtualenv":
        assert request["record_image"].endswith("/tao-ds-venv/bin/python")
        assert request["spec_bundle"]["image"] == request["record_image"]
        assert request["workload_image"] == DS_IMAGE
    else:
        assert request["record_image"] == DS_IMAGE
        assert request["spec_bundle"]["image"] == DS_IMAGE


@pytest.mark.parametrize("platform", PLATFORMS)
def test_visualize_finish_request_caps_native_math_threads(tmp_path, platform):
    _, results, _, _, _ = _write_fixture(tmp_path, platform)
    stage = results / "iter_1" / "visualization"
    output = stage / "visualize-finish.host.status.json"
    args = argparse.Namespace(
        results_dir=results,
        image="ds",
        stage_dir=stage,
        name="visualize_finish",
        pass_hf_token=False,
        fresh_output=[output],
        command=[
            "python3",
            "/iaa-runtime/run_iaa_compute.py",
            "visualize_finish",
            "--results-dir",
            "/results",
            "--label",
            "iter1",
        ],
    )

    _, request = action.prepare(args)

    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert request["environment"][name] == "1"


def test_remote_docker_uses_remote_staging_and_binding_contract(tmp_path):
    _, _, stage, output, args = _write_fixture(
        tmp_path, "docker", docker_remote=True
    )
    request_path, request = action.prepare(args)
    assert request["freshness_contract"] == (
        "remote-mirror-with-delete-before-submit"
    )
    assert request["platform_runtime_dir"].startswith(str(stage / ".tao-runtime"))

    results_scope = _attest_remote(request_path, request)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-remote-docker",
        results_scope=results_scope,
    )
    binding_path = action.bind_job(
        argparse.Namespace(request=request_path, job_record=record_path)
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["results_scope"] == results_scope
    assert binding["staging_receipt_sha256"]
    output.write_bytes(b"remote docker output fetched to launcher")
    Path(request["log_path"]).write_text(
        "remote Docker action complete\n", encoding="utf-8"
    )
    _finish_job_record(record_path, record)
    status_path, returncode = action.finalize(
        argparse.Namespace(
            request=request_path,
            job_record=record_path,
            native_exit_code=0,
        )
    )
    evidence = json.loads(status_path.read_text(encoding="utf-8"))
    assert returncode == 0
    assert evidence["freshness_contract"] == (
        "remote-mirror-with-delete-before-submit"
    )
    assert platform_evidence_error(evidence, "docker") is None


def test_docker_remote_flag_is_rejected_for_other_platforms(tmp_path):
    _, results, _, _, args = _write_fixture(tmp_path, "slurm")
    state_path = results / "deft_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["config"]["docker_remote"] = True
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="only for platform=docker"):
        action.prepare(args)


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize(("name", "label"), ACTION_CASES)
def test_every_iaa_tao_action_prepares_on_every_platform(
    tmp_path, platform, name, label
):
    _, results, _, _, _ = _write_fixture(tmp_path, platform)
    config = json.loads((results / "deft_state.json").read_text(encoding="utf-8"))[
        "config"
    ]
    stage = expected_stage_directory(name, label, results)
    command = expected_container_command(name, label, config)
    outputs = expected_fresh_outputs(name, label, results)
    request_path, request = action.prepare(
        argparse.Namespace(
            results_dir=results,
            image=expected_image_kind(name),
            stage_dir=stage,
            name=name,
            pass_hf_token=False,
            fresh_output=outputs,
            command=command,
        )
    )
    if platform == "virtualenv":
        expected_profile = expected_image_kind(name)
        assert request["virtualenv"].endswith(f"/tao-{expected_profile}-venv")

    assert request_path == stage / f"{name}.action.json"
    assert request["platform"] == platform
    assert request["spec_bundle"]["command"] == command[0]
    assert request["spec_bundle"]["args"] == command[1:]
    assert request["fresh_outputs"] == [str(path) for path in outputs]
    assert request["staging_absent_paths"] == [
        *[str(path) for path in outputs],
        str(stage / f"{name}.log"),
    ]
    mounts = {(item["source"], item["target"]) for item in request["mounts"]}
    declared = {
        item["spec_key"] for item in request["spec_bundle"]["declared_inputs"]
    }
    data_parent = results.parents[1] / "data"
    dataset_root = data_parent / "iaa_v31_tao_ft"
    parent_mounts = {
        (str(data_parent), "/data"),
        (str(data_parent), str(data_parent)),
    }
    root_mounts = {
        (str(dataset_root), f"/data/{dataset_root.name}"),
        (str(dataset_root), str(dataset_root)),
    }
    if name in DATASET_ACTIONS:
        assert root_mounts <= mounts
        assert parent_mounts.isdisjoint(mounts)
        assert "dataset_root" in declared
        assert "dataset_parent" not in declared
    else:
        assert parent_mounts.isdisjoint(mounts)
        assert root_mounts.isdisjoint(mounts)
        assert "dataset_root" not in declared
        assert "dataset_parent" not in declared


@pytest.mark.parametrize("platform", PLATFORMS)
def test_public_submit_boundary_revalidates_request_binding_and_job(tmp_path, platform):
    _, _, stage, _, args = _write_fixture(tmp_path, platform)
    request_path, request = action.prepare(args)
    results_scope = (
        _attest_remote(request_path, request)
        if platform in {"slurm", "kubernetes", "brev", "airflow"}
        else str(stage)
    )
    record_path, _ = _open_job_record(
        request_path,
        request,
        job_id=f"data-services-deft-iaa-submit-boundary-{platform}",
        results_scope=results_scope,
    )
    binding_path = action.bind_job(
        argparse.Namespace(request=request_path, job_record=record_path)
    )
    loaded_request, binding, job = action.load_bound_action_for_submit(
        request_path, binding_path, record_path,
    )
    assert loaded_request["request_sha256"] == request["request_sha256"]
    assert binding["job_id"] == job["id"]
    assert job["transitions"][-1]["state"] == "PENDING"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_finalize_binds_native_job_record_and_fresh_output(tmp_path, platform):
    _, _, stage, output, args = _write_fixture(tmp_path, platform)
    request_path, request = action.prepare(args)
    if platform in {"slurm", "kubernetes", "brev", "airflow"}:
        results_scope = _attest_remote(request_path, request)
    else:
        results_scope = str(stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"PAR1 fresh output")
    log = stage / "pool_embed.log"
    log.write_text("native platform action completed\n", encoding="utf-8")
    job_id = f"data-services-deft-iaa-pool-embed-{platform}"
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id=job_id,
        results_scope=results_scope,
    )
    action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))
    _finish_job_record(record_path, record)

    status_path, returncode = action.finalize(
        argparse.Namespace(
            request=request_path,
            job_record=record_path,
            native_exit_code=0,
        )
    )
    evidence = json.loads(status_path.read_text(encoding="utf-8"))
    assert returncode == 0
    assert evidence["schema_version"] == "2"
    assert evidence["kind"] == "platform_action"
    assert evidence["platform"] == platform
    assert evidence["backend_state"] == "COMPLETE"
    assert evidence["backend_exit_code"] == 0
    assert evidence["job_id"] == job_id
    assert bool(evidence["staging_receipt_sha256"]) is (
        platform in {"slurm", "kubernetes", "brev", "airflow"}
    )
    assert platform_evidence_error(evidence, platform) is None
    assert commit_stage._required_command_status(  # noqa: SLF001
        status_path,
        "platform status",
        scope=stage,
        required_output=output,
        required_name="pool_embed",
        required_command=args.command,
        required_image_kind="ds",
        required_image=DS_IMAGE,
        required_hf_forwarding=False,
        required_platform=platform,
    ) == str(status_path)
    errors = []
    assert audit_deft_run._validate_command_status(  # noqa: SLF001
        status_path,
        "platform status",
        errors,
        stage,
        required_name="pool_embed",
        required_command=args.command,
        required_image_kind="ds",
        required_image=DS_IMAGE,
        required_hf_forwarding=False,
        required_platform=platform,
    ) == evidence
    assert errors == []


@pytest.mark.parametrize("platform", ("slurm", "kubernetes", "brev", "airflow"))
def test_remote_finalize_requires_output_absence_attestation(tmp_path, platform):
    _, _, stage, output, args = _write_fixture(tmp_path, platform)
    request_path, request = action.prepare(args)
    output.write_bytes(b"fresh")
    (stage / "pool_embed.log").write_text("log\n", encoding="utf-8")
    record_path, _ = _open_job_record(
        request_path,
        request,
        job_id=f"data-services-deft-iaa-pool-embed-{platform}",
        results_scope=_remote_scope(platform),
    )
    with pytest.raises(ValueError, match="staging receipt"):
        action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))


def test_prepare_and_reconcile_are_crash_safe_across_launch_boundaries(tmp_path):
    _, _, _, output, args = _write_fixture(tmp_path, "docker")
    request_path, first = action.prepare(args)
    output.write_bytes(b"must not be deleted by interrupted re-prepare")

    second_path, second = action.prepare(args)
    assert second_path == request_path
    assert second == first
    assert output.is_file()
    assert action.reconcile_request(argparse.Namespace(request=request_path))["state"] == (
        "NO_JOB_RECORD"
    )

    record_path, record = _open_job_record(
        request_path,
        first,
        job_id="data-services-deft-iaa-pool-embed-crash-safe",
    )
    opened = action.reconcile_request(argparse.Namespace(request=request_path))
    assert opened["state"] == "JOB_OPENED_UNBOUND"
    assert opened["backend_ref_present"] is False
    action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))
    pending_bound = action.reconcile_request(argparse.Namespace(request=request_path))
    assert pending_bound["state"] == "BOUND_BACKEND_RECONCILIATION_REQUIRED"
    binding_before = Path(first["job_binding_path"]).read_bytes()
    action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))
    assert Path(first["job_binding_path"]).read_bytes() == binding_before
    _finish_job_record(record_path, record)
    bound = action.reconcile_request(argparse.Namespace(request=request_path))
    assert bound["state"] == "BOUND"
    assert bound["backend_ref_present"] is True
    assert bound["terminal_state"] == "COMPLETE"


def _cancel_presubmit_record(record_path: Path, record: dict) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    record["transitions"].append(
        {"ts": now, "state": "CANCELED", "message": "abandoned", "source": "agent"}
    )
    record["terminal_state"] = "CANCELED"
    record["terminal_write_by"] = "agent"
    record_path.write_text(json.dumps(record), encoding="utf-8")


def test_reconcile_ignores_only_unbound_abandoned_wrong_ownership_record(tmp_path):
    _, _, _, _, args = _write_fixture(tmp_path, "slurm")
    request_path, request = action.prepare(args)
    bad_path, bad = _open_job_record(
        request_path,
        request,
        job_id="embedding-deft-iaa-pool-embed-abandoned",
        results_scope=_remote_scope("slurm"),
    )
    bad["network_arch"] = "embedding"
    _cancel_presubmit_record(bad_path, bad)
    good_path, good = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-launched",
        results_scope=_remote_scope("slurm"),
    )
    _finish_job_record(good_path, good, state="ERROR")

    matches = action._matching_job_records(request_path, request)  # noqa: SLF001

    assert [(path, job["id"]) for path, job in matches] == [
        (good_path, good["id"])
    ]


def test_reconcile_rejects_bound_or_nonterminal_wrong_ownership_record(tmp_path):
    _, _, _, _, args = _write_fixture(tmp_path, "slurm")
    request_path, request = action.prepare(args)
    bad_path, bad = _open_job_record(
        request_path,
        request,
        job_id="embedding-deft-iaa-pool-embed-poison",
        results_scope=_remote_scope("slurm"),
    )
    bad["network_arch"] = "embedding"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="job-record does not own this action"):
        action._matching_job_records(request_path, request)  # noqa: SLF001

    _cancel_presubmit_record(bad_path, bad)
    Path(request["job_binding_path"]).write_text(
        json.dumps({"job_record_path": str(bad_path)}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="bound job-record cannot be treated"):
        action._matching_job_records(request_path, request)  # noqa: SLF001


def _bound_canceled_slurm_action(tmp_path):
    _, _, _, output, args = _write_fixture(tmp_path, "slurm")
    request_path, request = action.prepare(args)
    scope = _attest_remote(request_path, request)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-preallocation-cancel",
        results_scope=scope,
    )
    action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))
    _finish_job_record(record_path, record, state="CANCELED")
    record["backend_ref"] = "32653488"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return request_path, request, record_path, output, args


def test_slurm_preallocation_cancel_receipt_finalizes_and_allows_retry(
    tmp_path, monkeypatch
):
    request_path, request, record_path, _, args = _bound_canceled_slurm_action(
        tmp_path
    )
    real_run = subprocess.run

    def run(argv, *run_args, **run_kwargs):
        if argv[0] == "sacct":
            return subprocess.CompletedProcess(
                argv,
                0,
                "32653488|CANCELLED by 1000|Unknown|00:00:00|0:0|None assigned\n",
                "",
            )
        return real_run(argv, *run_args, **run_kwargs)

    monkeypatch.setattr(action.subprocess, "run", run)
    log_path = action.capture_preallocation_cancel(
        argparse.Namespace(request=request_path, job_record=record_path)
    )
    receipt = json.loads(log_path.read_text(encoding="utf-8"))
    assert receipt["kind"] == "slurm_preallocation_cancel"
    assert receipt["native_elapsed"] == "00:00:00"
    assert action.capture_preallocation_cancel(
        argparse.Namespace(request=request_path, job_record=record_path)
    ) == log_path

    status_path, rc = action.finalize(
        argparse.Namespace(
            request=request_path, job_record=record_path, native_exit_code=None
        )
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert rc == 3
    assert status["preallocation_cancel_receipt_sha256"] == receipt["receipt_sha256"]
    retry_path, retry = action.prepare(args)
    assert retry_path.name == "pool_embed.attempt-2.action.json"
    assert retry["attempt"] == 2


@pytest.mark.parametrize(
    "row,match",
    (
        (
            "32653488|CANCELLED|2026-08-21T08:00:00|00:00:01|0:0|node-1\n",
            "not canceled before allocation",
        ),
        (
            "99999999|CANCELLED|Unknown|00:00:00|0:0|None assigned\n",
            "one exact parent",
        ),
    ),
)
def test_slurm_preallocation_cancel_receipt_rejects_runtime_or_wrong_job(
    tmp_path, monkeypatch, row, match
):
    request_path, _, record_path, _, _ = _bound_canceled_slurm_action(tmp_path)
    monkeypatch.setattr(
        action.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, row, ""),
    )
    with pytest.raises(ValueError, match=match):
        action.capture_preallocation_cancel(
            argparse.Namespace(request=request_path, job_record=record_path)
        )


def _cache_artifacts(tmp_path: Path, stamp: str) -> Path:
    skill = (
        tmp_path
        / ".codex"
        / "plugins"
        / "cache"
        / "tao-local-plugins"
        / "tao-skill-bank"
        / f"0.1.12+codex.{stamp}"
        / "skills"
        / "applications"
        / "tao-run-deft-iaa"
    )
    patches = skill / "patches"
    patches.mkdir(parents=True)
    (patches / "sitecustomize.py").write_text("# identical\n", encoding="utf-8")
    (skill / "scripts").mkdir()
    (skill / "scripts" / "run_deft_cli.py").write_text(
        "# packaged shim\n", encoding="utf-8"
    )
    return patches


def _relocatable_runtime_state(results: Path, monkeypatch) -> None:
    state_path = results / "deft_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["config"]["iaa_deft_bundle_sha256"] = "a" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(action, "_python_tree_sha256", lambda _: "a" * 64)


@pytest.mark.parametrize("prior_cache_subset", ("missing", "older"))
def test_prepare_rematerializes_only_unlaunched_request_after_cache_refresh(
    tmp_path, monkeypatch, prior_cache_subset
):
    _, results, _, _, args = _write_fixture(tmp_path, "slurm")
    _relocatable_runtime_state(results, monkeypatch)
    old_patches = _cache_artifacts(tmp_path, "old")
    new_patches = _cache_artifacts(tmp_path, "new")
    active = {"patches": old_patches}
    validate = action.validate_action

    def relocated(**kwargs):
        return dataclasses.replace(validate(**kwargs), patches_dir=active["patches"])

    monkeypatch.setattr(action, "validate_action", relocated)
    request_path, old = action.prepare(args)
    if prior_cache_subset == "missing":
        old.pop("cache_subset")
    else:
        old_manifest = old["cache_subset"]
        old_manifest["root"] = str(tmp_path / "previous-cache-layout")
        old_manifest["sha256"] = action._sha256_json(
            {"root": old_manifest["root"], "entries": old_manifest["entries"]}
        )
    old.pop("request_sha256")
    old["request_sha256"] = action._sha256_json(old)
    request_path.write_text(json.dumps(old, indent=2) + "\n", encoding="utf-8")
    old_bytes = request_path.read_bytes()
    shutil.rmtree(old_patches.parents[3])
    active["patches"] = new_patches
    assert new_patches.is_dir()

    migrated_path, migrated = action.prepare(args)

    assert migrated_path == request_path
    assert migrated_path.read_bytes() != old_bytes
    assert migrated["started_at"] == old["started_at"]
    assert migrated["started_ns"] == old["started_ns"]
    assert migrated["spec_bundle"]["action"] == old["spec_bundle"]["action"]
    assert migrated["request_sha256"] != old["request_sha256"]
    assert migrated["cache_subset"]["root"] != str(
        tmp_path / "previous-cache-layout"
    )
    patch_mount = next(item for item in migrated["mounts"] if item["target"] == "/patches")
    assert patch_mount["source"] == migrated["patches_snapshot"]["root"]
    assert Path(patch_mount["source"]).is_dir()
    assert str(new_patches) != patch_mount["source"]
    assert action.reconcile_request(argparse.Namespace(request=request_path))[
        "state"
    ] == "NO_JOB_RECORD"


def test_prepare_never_rematerializes_cache_path_after_job_open(
    tmp_path, monkeypatch
):
    _, results, _, _, args = _write_fixture(tmp_path, "slurm")
    _relocatable_runtime_state(results, monkeypatch)
    old_patches = _cache_artifacts(tmp_path, "old")
    new_patches = _cache_artifacts(tmp_path, "new")
    active = {"patches": old_patches}
    validate = action.validate_action

    def relocated(**kwargs):
        return dataclasses.replace(validate(**kwargs), patches_dir=active["patches"])

    monkeypatch.setattr(action, "validate_action", relocated)
    request_path, request = action.prepare(args)
    _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-cache-refresh",
    )
    before = request_path.read_bytes()
    active["patches"] = new_patches

    repeated_path, repeated = action.prepare(args)
    assert repeated_path == request_path
    assert repeated == request
    assert request_path.read_bytes() == before


def test_reconcile_and_finalize_accept_launched_request_after_cache_refresh(
    tmp_path, monkeypatch
):
    _, results, stage, output, args = _write_fixture(tmp_path, "slurm")
    _relocatable_runtime_state(results, monkeypatch)
    old_patches = _cache_artifacts(tmp_path, "old")
    new_patches = _cache_artifacts(tmp_path, "new")
    active = {"patches": old_patches}
    validate = action.validate_action

    def relocated(**kwargs):
        return dataclasses.replace(validate(**kwargs), patches_dir=active["patches"])

    monkeypatch.setattr(action, "validate_action", relocated)
    request_path, request = action.prepare(args)
    _attest_remote(request_path, request)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-refreshed-finalize",
        results_scope=_remote_scope("slurm"),
    )
    action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))
    request_bytes = request_path.read_bytes()
    output.write_bytes(b"fresh remote embeddings")
    (stage / "pool_embed.log").write_text("native success\n", encoding="utf-8")
    _finish_job_record(record_path, record)
    active["patches"] = new_patches

    reconciled = action.reconcile_request(argparse.Namespace(request=request_path))
    status_path, returncode = action.finalize(
        argparse.Namespace(
            request=request_path,
            job_record=record_path,
            native_exit_code=0,
        )
    )

    assert reconciled["state"] == "BOUND"
    assert reconciled["terminal_state"] == "COMPLETE"
    assert returncode == 0
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "ok"
    assert request_path.read_bytes() == request_bytes


def _finalize_failed_action(
    request_path: Path,
    request: dict,
    *,
    log_text: str = "native failure\n",
    runner_exit_evidence: bool = False,
) -> None:
    Path(request["log_path"]).write_text(log_text, encoding="utf-8")
    if runner_exit_evidence:
        runtime = Path(request["platform_runtime_dir"])
        (runtime / "logs").mkdir(parents=True, exist_ok=True)
        (runtime / "logs" / "job.log").write_text(log_text, encoding="utf-8")
        runner = runtime / ".tao_runner"
        runner.mkdir()
        (runner / "exit_status.json").write_text(
            json.dumps(
                {
                    "canceled": False,
                    "error": None,
                    "finished_at": time.time(),
                    "return_code": 2,
                    "started_at": time.time() - 1,
                }
            ),
            encoding="utf-8",
        )
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id=f"{request['spec_bundle']['network_arch']}-{request['spec_bundle']['action']}",
    )
    action.bind_job(
        argparse.Namespace(request=request_path, job_record=record_path)
    )
    _finish_job_record(record_path, record, state="ERROR")
    _, returncode = action.finalize(
        argparse.Namespace(
            request=request_path,
            job_record=record_path,
            native_exit_code=2 if runner_exit_evidence else 1,
        )
    )
    assert returncode == 3


def _consume_two_predispatch_attempts(tmp_path: Path):
    _, _, stage, output, args = _write_fixture(tmp_path, "virtualenv")
    request_one_path, request_one = action.prepare(args)
    _finalize_failed_action(
        request_one_path,
        request_one,
        log_text=(
            "run_deft_cli: action config must be a regular file below the approved "
            "/specs mount\n"
        ),
        runner_exit_evidence=True,
    )
    request_two_path, request_two = action.prepare(args)
    _finalize_failed_action(
        request_two_path,
        request_two,
        log_text=(
            "run_deft_cli: action request must remain at "
            f"{stage / 'pool_embed.action.json'}\n"
        ),
        runner_exit_evidence=True,
    )
    return stage, output, args, request_one_path, request_two_path


def _train_fixture(tmp_path: Path):
    _, results, _, _, _ = _write_fixture(tmp_path, "slurm")
    state_path = results / "deft_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["config"]["num_gpus"] = 2
    state["config"]["gpu_ids"] = [0, 1]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    config = state["config"]
    stage = expected_stage_directory("train", "iter1", results)
    command = expected_container_command("train", "iter1", config)
    outputs = expected_fresh_outputs("train", "iter1", results)
    args = argparse.Namespace(
        results_dir=results,
        image="pyt",
        stage_dir=stage,
        name="train",
        pass_hf_token=False,
        fresh_output=outputs,
        command=command,
    )
    return results, stage, outputs, args


def _finalize_slurm_train_failure(
    request_path: Path,
    request: dict,
    *,
    log_text: str,
    terminal_state: str,
    native_exit_code: int | None,
) -> None:
    _attest_remote(request_path, request)
    Path(request["log_path"]).write_text(log_text, encoding="utf-8")
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id=f"clip-{request['spec_bundle']['action']}",
    )
    action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))
    _finish_job_record(record_path, record, state=terminal_state)
    _, returncode = action.finalize(
        argparse.Namespace(
            request=request_path,
            job_record=record_path,
            native_exit_code=native_exit_code,
        )
    )
    assert returncode == 3


def _consume_two_slurm_launcher_failures(tmp_path: Path):
    _, stage, outputs, args = _train_fixture(tmp_path)
    request_one_path, request_one = action.prepare(args)
    _finalize_slurm_train_failure(
        request_one_path,
        request_one,
        log_text=(
            "You set `devices=2` in Lightning, but the number of tasks per node "
            "configured in SLURM `--ntasks-per-node=1` does not match.\n"
        ),
        terminal_state="ERROR",
        native_exit_code=1,
    )
    request_two_path, request_two = action.prepare(args)
    _finalize_slurm_train_failure(
        request_two_path,
        request_two,
        log_text="Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2\n",
        terminal_state="CANCELED",
        native_exit_code=None,
    )
    return stage, outputs, args, request_one_path, request_two_path


def test_dispatch_repair_preserves_attempts_and_mints_one_distinct_request(tmp_path):
    stage, _, args, request_one_path, request_two_path = (
        _consume_two_predispatch_attempts(tmp_path)
    )
    one_bytes = request_one_path.read_bytes()
    two_bytes = request_two_path.read_bytes()
    status_two_bytes = (stage / "pool_embed.status.json").read_bytes()

    repair_path, repair = action.dispatch_repair(args)

    assert request_one_path.read_bytes() == one_bytes
    assert request_two_path.read_bytes() == two_bytes
    assert (stage / "pool_embed.attempt-2.status.json").read_bytes() == status_two_bytes
    assert not (stage / "pool_embed.status.json").exists()
    assert repair_path == stage / "pool_embed.dispatch-repair.action.json"
    assert repair["attempt"] == 2
    assert repair["dispatch_repair"] == 1
    assert repair["spec_bundle"]["action"] not in {
        json.loads(one_bytes)["spec_bundle"]["action"],
        json.loads(two_bytes)["spec_bundle"]["action"],
    }
    assert repair["log_path"].endswith("pool_embed.dispatch-repair.log")
    assert repair["platform_runtime_dir"].endswith("pool_embed.dispatch-repair")
    bundle, _ = virtualenv_cli._request_contract(repair_path, repair)  # noqa: SLF001
    assert bundle == repair["spec_bundle"]
    repeated_path, repeated = action.dispatch_repair(args)
    assert repeated_path == repair_path
    assert repeated == repair


def test_dispatch_repair_rejects_when_workload_output_exists(tmp_path):
    _, output, args, _, _ = _consume_two_predispatch_attempts(tmp_path)
    output.write_bytes(b"TAO workload output")

    with pytest.raises(ValueError, match="workload output exists"):
        action.dispatch_repair(args)


@pytest.mark.parametrize(
    "message",
    (
        "run_deft_cli: unknown pre-dispatch failure\n",
        "TAO Toolkit started evaluate\n",
    ),
)
def test_dispatch_repair_rejects_unknown_or_started_workload_log(
    tmp_path, message
):
    stage, _, args, _, _ = _consume_two_predispatch_attempts(tmp_path)
    log = stage / "pool_embed.attempt-2.log"
    native_log = (
        stage
        / ".tao-runtime"
        / "pool_embed.attempt-2"
        / "logs"
        / "job.log"
    )
    log.write_text(message, encoding="utf-8")
    native_log.write_text(message, encoding="utf-8")

    with pytest.raises(ValueError, match="not an allowlisted pre-dispatch classifier"):
        action.dispatch_repair(args)


def test_dispatch_repair_rejects_second_repair_after_terminal_repair(tmp_path):
    _, _, args, _, _ = _consume_two_predispatch_attempts(tmp_path)
    repair_path, repair = action.dispatch_repair(args)
    _finalize_failed_action(repair_path, repair)

    with pytest.raises(ValueError, match="second repair is forbidden"):
        action.dispatch_repair(args)


def test_launcher_repair_preserves_attempts_and_mints_one_distinct_request(tmp_path):
    stage, _, args, request_one_path, request_two_path = (
        _consume_two_slurm_launcher_failures(tmp_path)
    )
    one_bytes = request_one_path.read_bytes()
    two_bytes = request_two_path.read_bytes()
    status_two_bytes = (stage / "train.status.json").read_bytes()

    repair_path, repair = action.launcher_repair(args)

    assert request_one_path.read_bytes() == one_bytes
    assert request_two_path.read_bytes() == two_bytes
    assert (stage / "train.attempt-2.status.json").read_bytes() == status_two_bytes
    assert not (stage / "train.status.json").exists()
    assert repair_path == stage / "train.launcher-repair.action.json"
    assert repair["attempt"] == 2
    assert repair["launcher_repair"] == 1
    assert repair["spec_bundle"]["compute_shape"] == {"gpus": 2, "nodes": 1}
    assert repair["gpu_ids"] == [0, 1]
    assert repair["spec_bundle"]["action"] not in {
        json.loads(one_bytes)["spec_bundle"]["action"],
        json.loads(two_bytes)["spec_bundle"]["action"],
    }
    assert repair["log_path"].endswith("train.launcher-repair.log")
    assert repair["platform_runtime_dir"].endswith("train.launcher-repair")
    assert Path(repair["platform_runtime_dir"]).is_dir()
    repeated_path, repeated = action.launcher_repair(args)
    assert repeated_path == repair_path
    assert repeated == repair


@pytest.mark.parametrize("artifact", ("fresh-output", "checkpoint"))
def test_launcher_repair_rejects_after_workload_artifact(tmp_path, artifact):
    stage, outputs, args, _, _ = _consume_two_slurm_launcher_failures(tmp_path)
    if artifact == "fresh-output":
        outputs[0].parent.mkdir(parents=True, exist_ok=True)
        outputs[0].write_bytes(b"TAO workload output")
        message = "workload output exists"
    else:
        checkpoint = stage / "model_epoch_001.pth"
        checkpoint.write_bytes(b"checkpoint")
        message = "checkpoint exists"

    with pytest.raises(ValueError, match=message):
        action.launcher_repair(args)


@pytest.mark.parametrize(
    "message",
    (
        "unknown SLURM failure\n",
        "Error executing job with overrides: []\n",
        "Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2\n",
        "Epoch 0: 1% complete\n",
    ),
)
def test_launcher_repair_rejects_unknown_or_started_workload_log(tmp_path, message):
    stage, _, args, _, _ = _consume_two_slurm_launcher_failures(tmp_path)
    (stage / "train.log").write_text(message, encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden|not an allowlisted"):
        action.launcher_repair(args)


def test_launcher_repair_rejects_second_repair_after_terminal_repair(tmp_path):
    _, _, args, _, _ = _consume_two_slurm_launcher_failures(tmp_path)
    repair_path, repair = action.launcher_repair(args)
    _finalize_slurm_train_failure(
        repair_path,
        repair,
        log_text="launcher repair terminal infrastructure failure\n",
        terminal_state="ERROR",
        native_exit_code=1,
    )

    with pytest.raises(ValueError, match="second repair is forbidden"):
        action.launcher_repair(args)


def test_virtualenv_shim_accepts_attempt_two_request_path(tmp_path):
    _, _, stage, _, args = _write_fixture(tmp_path, "virtualenv")
    request_one_path, request_one = action.prepare(args)
    _finalize_failed_action(request_one_path, request_one)

    request_two_path, request_two = action.prepare(args)
    bundle, aliases = virtualenv_cli._request_contract(  # noqa: SLF001
        request_two_path, request_two
    )

    assert request_two_path == stage / "pool_embed.attempt-2.action.json"
    assert request_two["attempt"] == 2
    assert bundle == request_two["spec_bundle"]
    assert aliases["/results"] == request_two["results_dir"]


def test_retry_accepts_finalized_request_after_cache_refresh_without_rewriting_it(
    tmp_path, monkeypatch
):
    _, results, stage, _, args = _write_fixture(tmp_path, "docker")
    _relocatable_runtime_state(results, monkeypatch)
    old_patches = _cache_artifacts(tmp_path, "old")
    new_patches = _cache_artifacts(tmp_path, "new")
    active = {"patches": old_patches}
    validate = action.validate_action

    def relocated(**kwargs):
        return dataclasses.replace(validate(**kwargs), patches_dir=active["patches"])

    monkeypatch.setattr(action, "validate_action", relocated)
    request_one_path, request_one = action.prepare(args)
    _finalize_failed_action(request_one_path, request_one)
    request_one_bytes = request_one_path.read_bytes()
    active["patches"] = new_patches

    request_two_path, request_two = action.prepare(args)

    assert request_one_path.read_bytes() == request_one_bytes
    assert request_two_path == stage / "pool_embed.attempt-2.action.json"
    assert request_two["attempt"] == 2
    patch_mount = next(item for item in request_two["mounts"] if item["target"] == "/patches")
    assert patch_mount["source"] == request_two["patches_snapshot"]["root"]
    assert Path(patch_mount["source"]).is_dir()
    assert patch_mount["source"] != str(new_patches)
    archived = json.loads(
        (stage / "pool_embed.attempt-1.status.json").read_text(encoding="utf-8")
    )
    assert archived["request_sha256"] == request_one["request_sha256"]


def _visualize_finish_fixture(tmp_path: Path):
    _, results, _, _, _ = _write_fixture(tmp_path, "docker")
    stage = results / "iter_1" / "visualization"
    output = stage / "visualize-finish.host.status.json"
    args = argparse.Namespace(
        results_dir=results,
        image="ds",
        stage_dir=stage,
        name="visualize_finish",
        pass_hf_token=False,
        fresh_output=[output],
        command=[
            "python3",
            "/iaa-runtime/run_iaa_compute.py",
            "visualize_finish",
            "--results-dir",
            "/results",
            "--label",
            "iter1",
        ],
    )
    return stage, args


def _prepare_legacy_visualize_request(args, monkeypatch):
    current_request = action._request

    def without_thread_caps(*request_args, **request_kwargs):
        payload = current_request(*request_args, **request_kwargs)
        for name in action._VISUALIZE_THREAD_CAPS:
            payload["environment"].pop(name)
        payload["request_sha256"] = action._sha256_json(
            {key: value for key, value in payload.items() if key != "request_sha256"}
        )
        return payload

    monkeypatch.setattr(action, "_request", without_thread_caps)
    request_path, request = action.prepare(args)
    return request_path, request, current_request


def test_retry_adds_visualize_thread_caps_only_after_proven_openblas_sigsegv(
    tmp_path, monkeypatch
):
    stage, args = _visualize_finish_fixture(tmp_path)
    request_path, request, current_request = _prepare_legacy_visualize_request(
        args, monkeypatch
    )
    original_bytes = request_path.read_bytes()
    _finalize_failed_action(
        request_path,
        request,
        log_text=(
            "OpenBLAS warning: precompiled NUM_THREADS exceeded\n"
            "subprocess.CalledProcessError: died with <Signals.SIGSEGV: 11>\n"
        ),
    )
    monkeypatch.setattr(action, "_request", current_request)

    retry_path, retry = action.prepare(args)

    assert request_path.read_bytes() == original_bytes
    assert retry_path == stage / "visualize_finish.attempt-2.action.json"
    assert retry["attempt"] == 2
    for name, value in action._VISUALIZE_THREAD_CAPS.items():
        assert retry["environment"][name] == value


@pytest.mark.parametrize(
    "log_text",
    (
        "OpenBLAS warning: precompiled NUM_THREADS exceeded\nordinary failure\n",
        "subprocess died with SIGSEGV\n",
    ),
)
def test_retry_rejects_visualize_thread_cap_change_without_exact_crash_evidence(
    tmp_path, monkeypatch, log_text
):
    stage, args = _visualize_finish_fixture(tmp_path)
    request_path, request, current_request = _prepare_legacy_visualize_request(
        args, monkeypatch
    )
    _finalize_failed_action(request_path, request, log_text=log_text)
    monkeypatch.setattr(action, "_request", current_request)

    with pytest.raises(ValueError, match="differences exceed a cache relocation"):
        action.prepare(args)

    assert not (stage / "visualize_finish.attempt-2.action.json").exists()


def _rewrite_request_action(request_path: Path, request: dict, action_id: str) -> dict:
    rewritten = json.loads(json.dumps(request))
    rewritten["spec_bundle"]["action"] = action_id
    rewritten.pop("request_sha256")
    rewritten["request_sha256"] = action._sha256_json(rewritten)
    request_path.write_text(json.dumps(rewritten), encoding="utf-8")
    return rewritten


def test_retry_accepts_exact_finalized_regressed_normal_action_id(
    tmp_path, monkeypatch
):
    _, results, stage, _, args = _write_fixture(tmp_path, "docker")
    _relocatable_runtime_state(results, monkeypatch)
    old_patches = _cache_artifacts(tmp_path, "old")
    new_patches = _cache_artifacts(tmp_path, "new")
    active = {"patches": old_patches}
    validate = action.validate_action

    def relocated(**kwargs):
        return dataclasses.replace(validate(**kwargs), patches_dir=active["patches"])

    monkeypatch.setattr(action, "validate_action", relocated)
    current_action_id = action._action_id

    def regressed_action_id(
        context, attempt, started_ns, *, dispatch_repair=0, launcher_repair=0
    ):
        if dispatch_repair or launcher_repair:
            return current_action_id(
                context,
                attempt,
                started_ns,
                dispatch_repair=dispatch_repair,
                launcher_repair=launcher_repair,
            )
        return action._regressed_normal_action_id(context, attempt, started_ns)

    monkeypatch.setattr(action, "_action_id", regressed_action_id)
    request_path, request = action.prepare(args)
    _finalize_failed_action(request_path, request)
    request_one_bytes = request_path.read_bytes()
    monkeypatch.setattr(action, "_action_id", current_action_id)
    active["patches"] = new_patches

    request_two_path, request_two = action.prepare(args)

    assert request_path.read_bytes() == request_one_bytes
    assert request_two_path == stage / "pool_embed.attempt-2.action.json"
    assert request_two["spec_bundle"]["action"] != request["spec_bundle"]["action"]


@pytest.mark.parametrize("extra_drift", (False, True))
def test_retry_rejects_arbitrary_or_drifted_regressed_action_id(
    tmp_path, monkeypatch, extra_drift
):
    _, results, stage, _, args = _write_fixture(tmp_path, "docker")
    _relocatable_runtime_state(results, monkeypatch)
    old_patches = _cache_artifacts(tmp_path, "old")
    new_patches = _cache_artifacts(tmp_path, "new")
    active = {"patches": old_patches}
    validate = action.validate_action

    def relocated(**kwargs):
        return dataclasses.replace(validate(**kwargs), patches_dir=active["patches"])

    monkeypatch.setattr(action, "validate_action", relocated)
    request_path, request = action.prepare(args)
    _finalize_failed_action(request_path, request)
    context = relocated(
        results_dir=args.results_dir,
        image_kind=args.image,
        stage_dir=args.stage_dir,
        name=args.name,
        pass_hf_token=args.pass_hf_token,
        fresh_outputs=args.fresh_output,
        command=args.command,
    )
    action_id = (
        action._regressed_normal_action_id(
            context, request["attempt"], request["started_ns"]
        )
        if extra_drift
        else "deft-iaa-pool_embed-0000000000000000"
    )
    request = _rewrite_request_action(request_path, request, action_id)
    if extra_drift:
        request["environment"]["HOME"] = "/unexpected"
        request.pop("request_sha256")
        request["request_sha256"] = action._sha256_json(request)
        request_path.write_text(json.dumps(request), encoding="utf-8")
    active["patches"] = new_patches

    with pytest.raises(ValueError, match="differences exceed a cache relocation"):
        action.prepare(args)

    assert not (stage / "pool_embed.attempt-2.action.json").exists()


def test_retry_rejects_cache_refresh_when_runtime_hash_changed(tmp_path, monkeypatch):
    _, results, _, _, args = _write_fixture(tmp_path, "docker")
    _relocatable_runtime_state(results, monkeypatch)
    old_patches = _cache_artifacts(tmp_path, "old")
    new_patches = _cache_artifacts(tmp_path, "new")
    active = {"patches": old_patches}
    validate = action.validate_action

    def relocated(**kwargs):
        return dataclasses.replace(validate(**kwargs), patches_dir=active["patches"])

    monkeypatch.setattr(action, "validate_action", relocated)
    request_path, request = action.prepare(args)
    _finalize_failed_action(request_path, request)
    active["patches"] = new_patches
    monkeypatch.setattr(action, "_python_tree_sha256", lambda _: "b" * 64)

    with pytest.raises(ValueError, match="runtime does not match immutable run provenance"):
        action.prepare(args)

    assert not (request_path.parent / "pool_embed.attempt-2.action.json").exists()


def test_retry_snapshots_cache_refresh_from_different_semantic_version(
    tmp_path, monkeypatch
):
    _, results, _, _, args = _write_fixture(tmp_path, "docker")
    _relocatable_runtime_state(results, monkeypatch)
    old_patches = _cache_artifacts(tmp_path, "old")
    new_patches = _cache_artifacts(tmp_path, "new")
    different_version = Path(
        str(new_patches).replace("0.1.12+codex.new", "0.1.13+codex.new")
    )
    different_version.parent.mkdir(parents=True)
    shutil.copytree(new_patches, different_version, dirs_exist_ok=True)
    active = {"patches": old_patches}
    validate = action.validate_action

    def relocated(**kwargs):
        return dataclasses.replace(validate(**kwargs), patches_dir=active["patches"])

    monkeypatch.setattr(action, "validate_action", relocated)
    request_path, request = action.prepare(args)
    _finalize_failed_action(request_path, request)
    active["patches"] = different_version

    request_two_path, request_two = action.prepare(args)
    assert request_two_path == request_path.parent / "pool_embed.attempt-2.action.json"
    assert request_two["patches_snapshot"]["root"] != str(different_version)
    assert Path(request_two["patches_snapshot"]["root"]).is_dir()


def test_retry_snapshots_prior_bundle_paths_outside_plugin_cache(tmp_path, monkeypatch):
    _, results, _, _, args = _write_fixture(tmp_path, "docker")
    _relocatable_runtime_state(results, monkeypatch)
    old_patches = tmp_path / "outside" / "patches"
    old_patches.mkdir(parents=True)
    (old_patches / "sitecustomize.py").write_text("# old\n", encoding="utf-8")
    (old_patches.parent / "scripts").mkdir()
    (old_patches.parent / "scripts" / "run_deft_cli.py").write_text(
        "# old shim\n", encoding="utf-8"
    )
    new_patches = _cache_artifacts(tmp_path, "new")
    active = {"patches": old_patches}
    validate = action.validate_action

    def relocated(**kwargs):
        return dataclasses.replace(validate(**kwargs), patches_dir=active["patches"])

    monkeypatch.setattr(action, "validate_action", relocated)
    request_path, request = action.prepare(args)
    _finalize_failed_action(request_path, request)
    active["patches"] = new_patches

    request_two_path, request_two = action.prepare(args)
    assert request_two_path == request_path.parent / "pool_embed.attempt-2.action.json"
    assert request_two["patches_snapshot"]["root"] != str(new_patches)
    assert Path(request_two["patches_snapshot"]["root"]).is_dir()


def test_shallow_error_status_cannot_retry_or_destroy_live_attempt(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    record_path, _ = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-live",
    )
    binding_path = action.bind_job(
        argparse.Namespace(request=request_path, job_record=record_path)
    )
    binding_before = binding_path.read_bytes()
    output.write_bytes(b"partial native output")
    status_path = stage / "pool_embed.status.json"
    status_path.write_text(
        json.dumps({"attempt": 1, "status": "error"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="schema_version"):
        action.prepare(args)

    assert binding_path.read_bytes() == binding_before
    assert output.read_bytes() == b"partial native output"
    assert not (stage / "pool_embed.attempt-2.action.json").exists()


def test_retry_requires_finalized_lineage_is_idempotent_and_bounded(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_one_path, request_one = action.prepare(args)
    log_one = Path(request_one["log_path"])
    log_one.write_text("native attempt one failed\n", encoding="utf-8")
    record_one_path, record_one = _open_job_record(
        request_one_path,
        request_one,
        job_id="data-services-deft-iaa-pool-embed-attempt-1",
    )
    binding_one = action.bind_job(
        argparse.Namespace(request=request_one_path, job_record=record_one_path)
    )
    _finish_job_record(record_one_path, record_one, state="ERROR")
    status_path, returncode = action.finalize(
        argparse.Namespace(
            request=request_one_path,
            job_record=record_one_path,
            native_exit_code=1,
        )
    )
    assert returncode == 3
    status_one = json.loads(status_path.read_text(encoding="utf-8"))

    request_two_path, request_two = action.prepare(args)
    assert request_two_path == stage / "pool_embed.attempt-2.action.json"
    assert request_two["attempt"] == 2
    assert request_two["log_path"] == str(stage / "pool_embed.attempt-2.log")
    assert request_two["job_binding_path"] == str(
        stage / "pool_embed.attempt-2.job-binding.json"
    )
    assert not status_path.exists()
    assert json.loads(
        (stage / "pool_embed.attempt-1.status.json").read_text(encoding="utf-8")
    ) == status_one
    assert request_one_path.is_file()
    assert binding_one.is_file()
    assert log_one.is_file()

    repeated_path, repeated_request = action.prepare(args)
    assert repeated_path == request_two_path
    assert repeated_request == request_two

    Path(request_two["log_path"]).write_text(
        "native attempt two failed\n", encoding="utf-8"
    )
    record_two_path, record_two = _open_job_record(
        request_two_path,
        request_two,
        job_id="data-services-deft-iaa-pool-embed-attempt-2",
    )
    action.bind_job(
        argparse.Namespace(request=request_two_path, job_record=record_two_path)
    )
    _finish_job_record(record_two_path, record_two, state="ERROR")
    _, returncode = action.finalize(
        argparse.Namespace(
            request=request_two_path,
            job_record=record_two_path,
            native_exit_code=1,
        )
    )
    assert returncode == 3
    with pytest.raises(ValueError, match="attempt budget exhausted"):
        action.prepare(args)
    assert not output.exists()


def test_retry_rejects_status_that_changes_prepared_command(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    Path(request["log_path"]).write_text("native failure\n", encoding="utf-8")
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-tampered-status",
    )
    action.bind_job(
        argparse.Namespace(request=request_path, job_record=record_path)
    )
    _finish_job_record(record_path, record, state="ERROR")
    status_path, _ = action.finalize(
        argparse.Namespace(
            request=request_path,
            job_record=record_path,
            native_exit_code=1,
        )
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["command"] = ["embedding", "different-action"]
    status_path.write_text(json.dumps(status), encoding="utf-8")
    output.write_bytes(b"preserve until lineage validates")

    with pytest.raises(ValueError, match="command does not match"):
        action.prepare(args)

    assert output.read_bytes() == b"preserve until lineage validates"
    assert not (stage / "pool_embed.attempt-2.action.json").exists()


def test_retry_revalidates_terminal_job_transition_lineage(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    Path(request["log_path"]).write_text("native failure\n", encoding="utf-8")
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-mutated-lineage",
    )
    action.bind_job(
        argparse.Namespace(request=request_path, job_record=record_path)
    )
    _finish_job_record(record_path, record, state="ERROR")
    action.finalize(
        argparse.Namespace(
            request=request_path,
            job_record=record_path,
            native_exit_code=1,
        )
    )
    record["transitions"] = [record["transitions"][0], record["transitions"][-1]]
    record_path.write_text(json.dumps(record), encoding="utf-8")
    output.write_bytes(b"preserve until terminal lineage validates")

    with pytest.raises(ValueError, match="PENDING, RUNNING, and terminal"):
        action.prepare(args)

    assert output.read_bytes() == b"preserve until terminal lineage validates"
    assert not (stage / "pool_embed.attempt-2.action.json").exists()


def test_bind_job_rejects_record_after_native_submission(tmp_path):
    _, _, _, _, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-too-late",
    )
    _finish_job_record(record_path, record)

    with pytest.raises(ValueError, match="before native submit"):
        action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))


def test_bind_job_rejects_local_record_with_different_results_scope(tmp_path):
    _, _, _, _, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    record_path, _ = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-wrong-local-scope",
        results_scope=str(tmp_path / "other-results"),
    )

    with pytest.raises(ValueError, match="results_dir must equal the action stage"):
        action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))


def test_bind_job_rejects_upload_exclusion_drift(tmp_path):
    _, _, _, _, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-upload-excludes",
    )
    record["upload_excludes"] = []
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="upload_excludes"):
        action.bind_job(
            argparse.Namespace(request=request_path, job_record=record_path)
        )
    assert not Path(request["job_binding_path"]).exists()


def test_concurrent_distinct_job_cannot_replace_first_binding(tmp_path):
    _, _, _, _, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    first_path, _ = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-first",
    )
    second_path, _ = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-second",
    )
    binding_path = action.bind_job(
        argparse.Namespace(request=request_path, job_record=first_path)
    )
    first_binding = binding_path.read_bytes()

    with pytest.raises(ValueError, match="does not match"):
        action.bind_job(
            argparse.Namespace(request=request_path, job_record=second_path)
        )
    assert binding_path.read_bytes() == first_binding


def test_bind_job_rejects_remote_record_outside_attested_scope(tmp_path):
    _, _, _, _, args = _write_fixture(tmp_path, "slurm")
    request_path, request = action.prepare(args)
    _attest_remote(request_path, request)
    record_path, _ = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-wrong-remote-scope",
        results_scope="/remote/tao-iaa-tests/slurm/other",
    )

    with pytest.raises(ValueError, match="must equal the attested backend scope"):
        action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))


@pytest.mark.parametrize(
    "scope",
    (
        "/remote/tao/../escape",
        "s3://bucket/tao/../escape",
        "s3://bucket/tao/%2e%2e/escape",
        "s3://user:password@bucket/tao/results",
    ),
)
def test_remote_scope_rejects_traversal_and_credentials(tmp_path, scope):
    _, _, _, _, args = _write_fixture(tmp_path, "slurm")
    request_path, request = action.prepare(args)

    with pytest.raises(ValueError, match="remote backend scope"):
        action.attest_staged(
            argparse.Namespace(
                request=request_path,
                backend_scope=scope,
                absent_path=request["staging_absent_paths"],
            )
        )


def test_finalize_requires_pre_submit_job_binding(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-unbound",
    )
    _finish_job_record(record_path, record)
    output.write_bytes(b"fresh")
    (stage / "pool_embed.log").write_text("log\n", encoding="utf-8")

    with pytest.raises(ValueError, match="job binding"):
        action.finalize(
            argparse.Namespace(
                request=request_path,
                job_record=record_path,
                native_exit_code=0,
            )
        )


def _metric_parse_args(results: Path) -> argparse.Namespace:
    config = json.loads((results / "deft_state.json").read_text(encoding="utf-8"))[
        "config"
    ]
    name = "metric_parse"
    label = "iter1"
    return argparse.Namespace(
        results_dir=results,
        image=expected_image_kind(name),
        stage_dir=expected_stage_directory(name, label, results),
        name=name,
        pass_hf_token=False,
        fresh_output=expected_fresh_outputs(name, label, results),
        command=expected_container_command(name, label, config),
    )


def test_bound_presubmit_recovery_archives_binding_and_restores_safe_open(
    tmp_path, monkeypatch,
):
    _, results, _, _, args = _write_fixture(tmp_path, "slurm")
    request_path, request = action.prepare(args)
    scope = _attest_remote(request_path, request)
    job_id = "data-services-deft-iaa-pool-bound-presubmit-a1b2c3"
    record_path, record = _open_job_record(
        request_path, request, job_id=job_id, results_scope=scope,
    )
    binding_path = action.bind_job(
        argparse.Namespace(request=request_path, job_record=record_path)
    )
    _cancel_presubmit_record(record_path, record)
    monkeypatch.setattr(
        action.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    recovery_args = argparse.Namespace(
        request=request_path, job_record=record_path, login="user@login",
        confirm=True,
    )
    evidence_path = action.recover_bound_presubmit(recovery_args)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["kind"] == "slurm_bound_presubmit_recovery"
    assert evidence["job_id"] == job_id
    assert evidence["scheduler_login"] == "user@login"
    assert not binding_path.exists()
    assert Path(evidence["binding_archive"]).is_file()
    assert action.reconcile_request(argparse.Namespace(request=request_path))["state"] == (
        "NO_JOB_RECORD"
    )
    assert action.recover_bound_presubmit(recovery_args) == evidence_path

    replacement_path, _ = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-bound-presubmit-d4e5f6",
        results_scope=scope,
    )
    replacement_binding = action.bind_job(
        argparse.Namespace(request=request_path, job_record=replacement_path)
    )
    assert replacement_binding.is_file()
    replacement = json.loads(replacement_path.read_text(encoding="utf-8"))
    _cancel_presubmit_record(replacement_path, replacement)
    second_evidence = action.recover_bound_presubmit(argparse.Namespace(
        request=request_path, job_record=replacement_path, login="user@login",
        confirm=True,
    ))
    assert second_evidence != evidence_path
    assert Path(second_evidence).is_file()


def test_bound_presubmit_recovery_rejects_existing_native_job(tmp_path, monkeypatch):
    _, results, _, _, _ = _write_fixture(tmp_path, "slurm")
    args = _metric_parse_args(results)
    request_path, request = action.prepare(args)
    scope = _attest_remote(request_path, request)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="iaa-adapter-deft-iaa-metric-native-exists-a1b2c3",
        results_scope=scope,
    )
    binding_path = action.bind_job(
        argparse.Namespace(request=request_path, job_record=record_path)
    )
    _cancel_presubmit_record(record_path, record)
    monkeypatch.setattr(
        action.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="32670000|iaa-adapter-deft-iaa-metric-native-exists-a1b2c3\n",
            stderr="",
        ),
    )
    with pytest.raises(ValueError, match="existing native SLURM job"):
        action.recover_bound_presubmit(argparse.Namespace(
            request=request_path, job_record=record_path, confirm=True,
        ))
    assert binding_path.is_file()


def _terminal_unbound_metric_action(tmp_path: Path):
    _, results, _, _, _ = _write_fixture(tmp_path, "slurm")
    args = _metric_parse_args(results)
    request_path, request = action.prepare(args)
    scope = _attest_remote(request_path, request)
    job_id = "iaa-adapter-deft-iaa-metric-parse-unbound-a1b2c3"
    record_path, record = _open_job_record(
        request_path, request, job_id=job_id, results_scope=scope
    )
    _finish_job_record(record_path, record)
    for output in args.fresh_output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"metric": 0.2}\n', encoding="utf-8")
    Path(request["log_path"]).write_text("metric parse complete\n", encoding="utf-8")
    return args, request_path, request, record_path


def test_unbound_replay_quarantines_attempt1_and_finalizes_distinct_attempt2(
    tmp_path,
):
    args, request_one_path, request_one, record_one_path = (
        _terminal_unbound_metric_action(tmp_path)
    )
    request_two_path, request_two = action.unbound_replay(args)
    assert request_two_path == args.stage_dir / "metric_parse.attempt-2.action.json"
    assert request_two["attempt"] == 2
    assert request_two["unbound_replay"] == 1
    assert request_two["spec_bundle"]["action"] != request_one["spec_bundle"]["action"]
    assert not any(path.exists() for path in args.fresh_output)
    evidence_path = args.stage_dir / "metric_parse.unbound-replay.evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["prior_request_path"] == str(request_one_path)
    assert evidence["prior_job_record_path"] == str(record_one_path)
    assert evidence["prior_backend_state"] == "COMPLETE"
    assert evidence["evidence_sha256"] == request_two[
        "unbound_replay_evidence_sha256"
    ]
    assert all(Path(row["archive"]).is_file() for row in evidence["quarantined_outputs"])

    repeated_path, repeated = action.unbound_replay(args)
    assert repeated_path == request_two_path
    assert repeated == request_two

    scope = _attest_remote(request_two_path, request_two)
    job_two = "iaa-adapter-deft-iaa-metric-parse-unbound-replay-d4e5f6"
    record_two_path, record_two = _open_job_record(
        request_two_path, request_two, job_id=job_two, results_scope=scope
    )
    action.bind_job(
        argparse.Namespace(request=request_two_path, job_record=record_two_path)
    )
    _finish_job_record(record_two_path, record_two)
    for output in args.fresh_output:
        output.write_text('{"metric": 0.21}\n', encoding="utf-8")
    Path(request_two["log_path"]).write_text(
        "metric replay complete\n", encoding="utf-8"
    )
    status_path, returncode = action.finalize(
        argparse.Namespace(
            request=request_two_path,
            job_record=record_two_path,
            native_exit_code=0,
        )
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert returncode == 0
    assert status["status"] == "ok"
    assert status["unbound_replay"] == 1
    assert status["unbound_replay_evidence_sha256"] == evidence["evidence_sha256"]


def test_unbound_replay_rejects_existing_binding(tmp_path):
    _, results, _, _, _ = _write_fixture(tmp_path, "slurm")
    args = _metric_parse_args(results)
    request_path, request = action.prepare(args)
    scope = _attest_remote(request_path, request)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="iaa-adapter-deft-iaa-metric-bound-a1b2c3",
        results_scope=scope,
    )
    action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))
    _finish_job_record(record_path, record)
    for output in args.fresh_output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"metric": 0.2}\n', encoding="utf-8")
    Path(request["log_path"]).write_text("metric parse complete\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden when a job binding exists"):
        action.unbound_replay(args)


def test_unbound_replay_rejects_non_allowlisted_gpu_action(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "slurm")
    request_path, request = action.prepare(args)
    scope = _attest_remote(request_path, request)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-unbound-a1b2c3",
        results_scope=scope,
    )
    _finish_job_record(record_path, record)
    output.write_bytes(b"PAR1")
    (stage / "pool_embed.log").write_text("pool complete\n", encoding="utf-8")
    with pytest.raises(ValueError, match="allowlisted deterministic SLURM adapters"):
        action.unbound_replay(args)


def test_finalize_rejects_unredacted_platform_log(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-secret-log",
    )
    action.bind_job(
        argparse.Namespace(request=request_path, job_record=record_path)
    )
    _finish_job_record(record_path, record)
    output.write_bytes(b"fresh")
    (stage / "pool_embed.log").write_text(
        "HF_TOKEN=hf_nvapi-SUPERSECRET123\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="credential lint"):
        action.finalize(
            argparse.Namespace(
                request=request_path,
                job_record=record_path,
                native_exit_code=0,
            )
        )


def test_finalize_rejects_invalid_terminal_transition_lineage(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-terminal-lineage",
    )
    action.bind_job(
        argparse.Namespace(request=request_path, job_record=record_path)
    )
    _finish_job_record(record_path, record)
    record["transitions"].insert(
        1,
        {
            "ts": record["transitions"][0]["ts"],
            "state": "ERROR",
            "message": "impossible early terminal",
            "source": "agent",
        },
    )
    record_path.write_text(json.dumps(record), encoding="utf-8")
    output.write_bytes(b"fresh")
    (stage / "pool_embed.log").write_text("log\n", encoding="utf-8")

    with pytest.raises(ValueError, match="after an earlier terminal"):
        action.finalize(
            argparse.Namespace(
                request=request_path,
                job_record=record_path,
                native_exit_code=0,
            )
        )


def test_platform_evidence_must_match_initialized_platform(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-evidence-platform",
    )
    action.bind_job(argparse.Namespace(request=request_path, job_record=record_path))
    _finish_job_record(record_path, record)
    output.write_bytes(b"fresh")
    (stage / "pool_embed.log").write_text("log\n", encoding="utf-8")
    status_path, returncode = action.finalize(
        argparse.Namespace(
            request=request_path,
            job_record=record_path,
            native_exit_code=0,
        )
    )
    assert returncode == 0
    evidence = json.loads(status_path.read_text(encoding="utf-8"))
    assert platform_evidence_error(evidence, "docker") is None
    assert "does not match" in platform_evidence_error(evidence, "slurm")

    legacy = {
        "schema_version": "1",
        "kind": "container",
        "docker_exit_code": 0,
        "artifact_error": None,
    }
    assert platform_evidence_error(legacy, "docker") is None
    assert "only for Docker" in platform_evidence_error(legacy, "virtualenv")


def test_finalize_rejects_job_record_from_another_platform(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "slurm")
    request_path, request = action.prepare(args)
    output.write_bytes(b"fresh")
    (stage / "pool_embed.log").write_text("log\n", encoding="utf-8")
    record_path, _ = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-abcdef",
        platform="docker",
        results_scope=str(stage),
    )
    with pytest.raises(ValueError, match="does not own this action"):
        action.finalize(
            argparse.Namespace(
                request=request_path,
                job_record=record_path,
                native_exit_code=0,
            )
        )


def test_prepare_invalidates_stale_output_and_log(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    stage.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"stale")
    log = stage / "pool_embed.log"
    log.write_text("stale log\n", encoding="utf-8")

    _, request = action.prepare(args)

    assert not output.exists()
    assert not log.exists()
    assert request["staging_absent_paths"] == [str(output), str(log)]


def test_prepare_rejects_workspace_cache_symlink_before_any_action_write(tmp_path):
    workspace, _, stage, _, args = _write_fixture(tmp_path, "docker")
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    marker = outside / "preserve.txt"
    marker.write_text("unchanged\n", encoding="utf-8")
    shutil.rmtree(workspace / "cache")
    os.symlink(outside, workspace / "cache")

    with pytest.raises(ValueError, match="workspace cache.*symlink"):
        action.prepare(args)

    assert marker.read_text(encoding="utf-8") == "unchanged\n"
    assert sorted(path.name for path in outside.iterdir()) == ["preserve.txt"]
    assert not stage.exists()


def test_prepare_rejects_symlinked_stage_parent_without_writing_outside(tmp_path):
    _, results, _, _, args = _write_fixture(tmp_path, "docker")
    outside = tmp_path / "outside-stage"
    outside.mkdir()
    marker = outside / "preserve.txt"
    marker.write_text("unchanged\n", encoding="utf-8")
    os.symlink(outside, results / "embeddings")

    with pytest.raises(ValueError, match="symlink"):
        action.prepare(args)

    assert marker.read_text(encoding="utf-8") == "unchanged\n"
    assert sorted(path.name for path in outside.iterdir()) == ["preserve.txt"]


def test_prepare_rejects_symlinked_existing_status_without_following_it(tmp_path):
    _, _, stage, _, args = _write_fixture(tmp_path, "docker")
    stage.mkdir(parents=True)
    outside = tmp_path / "outside-status.json"
    outside.write_text('{"attempt": 1, "status": "error"}\n', encoding="utf-8")
    os.symlink(outside, stage / "pool_embed.status.json")

    with pytest.raises(ValueError, match="existing command status.*unsafe"):
        action.prepare(args)

    assert json.loads(outside.read_text(encoding="utf-8"))["status"] == "error"
    assert not (stage / "pool_embed.action.json").exists()


def test_config_materialization_rejects_existing_config_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    results = workspace / "results" / "run_symlink"
    results.mkdir(parents=True)
    dataset = workspace / "data" / "iaa"
    archives = tmp_path / "archives"
    archives.mkdir()
    images = archives / "images_raw.tar"
    metadata = archives / "meta.tar.gz"
    images.write_bytes(b"images")
    metadata.write_bytes(b"metadata")
    outside = tmp_path / "outside-config"
    outside.mkdir()
    marker = outside / "preserve.txt"
    marker.write_text("unchanged\n", encoding="utf-8")
    os.symlink(outside, results / "config")
    args = prepare_config._parser().parse_args(  # noqa: SLF001
        [
            "--workspace",
            str(workspace),
            "--results-dir",
            str(results),
            "--dataset-root",
            str(dataset),
            "--images-archive",
            str(images),
            "--metadata-archive",
            str(metadata),
            "--platform",
            "docker",
            "--max-iterations",
            "1",
            *_managed_sdg_args(),
        ]
    )

    with pytest.raises(ValueError, match="run config directory.*symlink"):
        prepare_config.materialize(args)

    assert marker.read_text(encoding="utf-8") == "unchanged\n"
    assert sorted(path.name for path in outside.iterdir()) == ["preserve.txt"]


def test_finalize_rejects_job_record_that_predates_request(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    output.write_bytes(b"fresh")
    (stage / "pool_embed.log").write_text("fresh log\n", encoding="utf-8")
    submitted = (
        dt.datetime.fromisoformat(request["started_at"]) - dt.timedelta(seconds=1)
    ).isoformat(timespec="seconds")
    record_path, _ = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-iaa-pool-embed-old",
        submitted_at=submitted,
    )

    with pytest.raises(ValueError, match="predates"):
        action.finalize(
            argparse.Namespace(
                request=request_path,
                job_record=record_path,
                native_exit_code=0,
            )
        )


def test_virtualenv_shim_executes_cli_with_translated_paths(tmp_path):
    workspace, results, stage, output, args = _write_fixture(tmp_path, "virtualenv")
    venv_bin = workspace / "tao-ds-venv" / "bin"
    executable = venv_bin / "embedding"
    executable.write_text(
        f"#!{venv_bin / 'python'}\n"
        "import json, os, pathlib, sys, yaml\n"
        "spec_path = pathlib.Path(sys.argv[sys.argv.index('-e') + 1])\n"
        "output = pathlib.Path(next(v for v in sys.argv if v.startswith('output_parquet=')).split('=', 1)[1])\n"
        "output.write_text(json.dumps({\n"
        "  'argv': sys.argv[1:], 'spec': yaml.safe_load(spec_path.read_text()),\n"
        "  'cache': os.environ.get('XDG_CACHE_HOME'),\n"
        "  'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "  'home': os.environ.get('HOME'), 'path': os.environ.get('PATH'),\n"
        "  'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "  'omp': os.environ.get('OMP_NUM_THREADS'),\n"
        "  'hf': os.environ.get('HF_TOKEN'),\n"
        "  'ngc': os.environ.get('NGC_KEY'),\n"
        "  'brev': os.environ.get('BREV_API_TOKEN'),\n"
        "  'aws': os.environ.get('AWS_SECRET_ACCESS_KEY'),\n"
        "  'sentinel': os.environ.get('ARBITRARY_SENTINEL'),\n"
        "  'pythonhome': os.environ.get('PYTHONHOME'),\n"
        "  'ld_preload': os.environ.get('LD_PRELOAD'),\n"
        "}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    request_path, request = action.prepare(args)
    env = {
        **os.environ,
        "VIRTUAL_ENV": request["virtualenv"],
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "7",
        "HF_TOKEN": "must-not-leak-without-forwarding",
        "NGC_KEY": "must-not-leak",
        "BREV_API_TOKEN": "must-not-leak",
        "AWS_SECRET_ACCESS_KEY": "must-not-leak",
        "ARBITRARY_SENTINEL": "must-not-leak",
        "LD_PRELOAD": "",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(IAA_SCRIPTS / "run_deft_cli.py"),
            "--request",
            str(request_path),
            "--",
            request["spec_bundle"]["command"],
            *request["spec_bundle"]["args"],
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(output.read_text(encoding="utf-8"))
    assert f"input_parquet={results}/embeddings/source/source_pool.parquet" in observed["argv"]
    assert f"output_parquet={results}/embeddings/source/embeddings.parquet" in observed["argv"]
    translated_spec = Path(observed["argv"][observed["argv"].index("-e") + 1])
    assert translated_spec.parent == stage
    assert translated_spec.name == "pool_embed.virtualenv.yaml"
    assert observed["spec"]["model"] == f"{results}/model"
    assert observed["cache"] == str(workspace / "cache")
    assert observed["pythonpath"] == request["patches_snapshot"]["root"]
    assert observed["home"] == "/tmp"
    assert observed["path"].startswith(f"{request['virtualenv']}/bin:")
    assert observed["cuda"] == "0"
    assert observed["omp"] == "7"
    assert observed["hf"] is None
    assert observed["ngc"] is None
    assert observed["brev"] is None
    assert observed["aws"] is None
    assert observed["sentinel"] is None
    assert observed["pythonhome"] is None
    assert observed["ld_preload"] is None


def test_virtualenv_child_environment_forwards_only_approved_names(tmp_path, monkeypatch):
    _, _, _, _, args = _write_fixture(tmp_path, "virtualenv")
    monkeypatch.setenv("HF_TOKEN", "approved-test-token")
    monkeypatch.setenv("NGC_KEY", "must-not-leak")
    monkeypatch.setenv("BREV_API_TOKEN", "must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    monkeypatch.setenv("ARBITRARY_SENTINEL", "must-not-leak")
    monkeypatch.setenv("PYTHONHOME", "/must/not/leak")
    monkeypatch.setenv("LD_PRELOAD", "/must/not/leak.so")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    request_path, request = action.prepare(args)
    del request_path
    request["forward_env"] = ["HF_TOKEN"]
    aliases = {item["target"]: item["source"] for item in request["mounts"]}

    environment = virtualenv_cli._execution_environment(  # noqa: SLF001
        request, aliases, request["virtualenv"]
    )

    assert environment["HF_TOKEN"] == "approved-test-token"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["OPENBLAS_NUM_THREADS"] == "3"
    for name in (
        "NGC_KEY",
        "BREV_API_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "ARBITRARY_SENTINEL",
        "PYTHONHOME",
        "LD_PRELOAD",
    ):
        assert name not in environment


def test_virtualenv_typed_adapter_uses_pinned_profile_python_with_zero_gpus(
    tmp_path, monkeypatch
):
    workspace, results, _, _, args = _write_fixture(tmp_path, "virtualenv")
    state_path = results / "deft_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    archives = workspace / "inputs"
    archives.mkdir()
    (archives / "images.tar").write_bytes(b"images")
    (archives / "meta.tar.gz").write_bytes(b"metadata")
    state["config"].update(
        {
            "images_archive": str(archives / "images.tar"),
            "metadata_archive": str(archives / "meta.tar.gz"),
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    args.name = "dataset_rebuild"
    args.stage_dir = results / "dataset_setup"
    args.image = "ds"
    args.command = expected_container_command(args.name, "baseline", state["config"])
    args.fresh_output = expected_fresh_outputs(args.name, "baseline", results)

    request_path, request = action.prepare(args)
    bundle, aliases = virtualenv_cli._request_contract(request_path, request)  # noqa: SLF001
    assert bundle["command"] == "python3"
    assert request["gpu_ids"] == []
    assert bundle["compute_shape"]["gpus"] == 0
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    environment = virtualenv_cli._execution_environment(  # noqa: SLF001
        request, aliases, request["virtualenv"]
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    executable = Path(request["virtualenv"]) / "bin" / "python3"
    approved = Path(request["virtualenv"]) / "bin" / "python"
    fd, pinned = virtualenv_cli._open_verified_interpreter(  # noqa: SLF001
        executable, request["virtualenv_entrypoint_sha256"], approved
    )
    try:
        assert pinned == f"/proc/self/fd/{fd}"
    finally:
        os.close(fd)


def test_dataset_rebuild_prepare_materializes_only_the_missing_parent(tmp_path):
    workspace, results, _, _, args = _write_fixture(tmp_path, "airflow")
    state_path = results / "deft_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    dataset = Path(state["config"]["dataset_root"])
    shutil.rmtree(dataset.parent)
    archives = workspace / "inputs"
    archives.mkdir()
    (archives / "images.tar").write_bytes(b"images")
    (archives / "meta.tar.gz").write_bytes(b"metadata")
    state["config"].update({
        "images_archive": str(archives / "images.tar"),
        "metadata_archive": str(archives / "meta.tar.gz"),
    })
    state_path.write_text(json.dumps(state), encoding="utf-8")
    args.name = "dataset_rebuild"
    args.stage_dir = results / "dataset_setup"
    args.image = "ds"
    args.command = expected_container_command(args.name, "baseline", state["config"])
    args.fresh_output = expected_fresh_outputs(args.name, "baseline", results)

    _, request = action.prepare(args)

    assert dataset.parent.is_dir()
    assert not dataset.exists()
    assert any(
        row["source"] == str(dataset.parent) and not row["read_only"]
        for row in request["mounts"]
    )


def test_virtualenv_shim_accepts_only_regular_configs_in_declared_mounts(tmp_path):
    specs = tmp_path / "config"
    results = tmp_path / "results"
    specs.mkdir()
    (results / "zs" / "specs").mkdir(parents=True)
    (results / "iter_1" / "specs").mkdir(parents=True)
    aliases = {"/specs": str(specs), "/results": str(results)}
    shared = specs / "shared.yaml"
    phase = results / "zs" / "specs" / "eval_config.yaml"
    train = results / "iter_1" / "specs" / "train_config.yaml"
    shared.write_text("model: test\n", encoding="utf-8")
    phase.write_text("evaluate: {}\n", encoding="utf-8")
    train.write_text("train: {}\n", encoding="utf-8")

    assert virtualenv_cli._approved_config_path(  # noqa: SLF001
        "/specs/shared.yaml", shared, aliases
    )
    assert virtualenv_cli._approved_config_path(  # noqa: SLF001
        "/results/zs/specs/eval_config.yaml", phase, aliases
    )
    assert virtualenv_cli._approved_config_path(  # noqa: SLF001
        "/results/iter_1/specs/train_config.yaml", train, aliases
    )
    assert not virtualenv_cli._approved_config_path(  # noqa: SLF001
        "/results/zs/specs/eval_config.yaml", tmp_path / "missing.yaml", aliases
    )
    assert not virtualenv_cli._approved_config_path(  # noqa: SLF001
        "/outside/eval_config.yaml", phase, aliases
    )

    link = results / "zs" / "specs" / "linked.yaml"
    link.symlink_to(shared)
    assert not virtualenv_cli._approved_config_path(  # noqa: SLF001
        "/results/zs/specs/linked.yaml", link, aliases
    )


def test_virtualenv_shim_rejects_runner_gpu_selection_mismatch(tmp_path):
    _, _, _, _, args = _write_fixture(tmp_path, "virtualenv")
    request_path, request = action.prepare(args)
    aliases = {item["target"]: item["source"] for item in request["mounts"]}

    with pytest.raises(ValueError, match="does not match the approved gpu_ids"):
        virtualenv_cli._execution_environment(  # noqa: SLF001
            request, aliases, request["virtualenv"]
        )


def test_virtualenv_shim_rejects_runner_environment_from_wrong_profile(tmp_path):
    workspace, _, _, _, args = _write_fixture(tmp_path, "virtualenv")
    request_path, request = action.prepare(args)
    completed = subprocess.run(
        [
            sys.executable,
            str(IAA_SCRIPTS / "run_deft_cli.py"),
            "--request",
            str(request_path),
            "--",
            request["spec_bundle"]["command"],
            *request["spec_bundle"]["args"],
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "VIRTUAL_ENV": str(workspace / "tao-pyt-venv")},
        check=False,
    )
    assert completed.returncode == 2
    assert "does not match the action's approved profile" in completed.stderr


def _rewrite_action_request(path: Path, mutation) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("request_sha256")
    mutation(payload)
    payload["request_sha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


@pytest.mark.parametrize(
    "case",
    (
        "schema",
        "workflow",
        "name",
        "image-kind",
        "bundle-args",
        "duplicate-target",
        "relative-source",
        "relative-target",
        "missing-base-target",
    ),
)
def test_virtualenv_shim_rejects_forged_request_structures(tmp_path, case):
    _, _, _, _, args = _write_fixture(tmp_path, "virtualenv")
    request_path, request = action.prepare(args)

    def mutation(payload):
        if case == "schema":
            payload["schema_version"] = "future"
        elif case == "workflow":
            payload["workflow"] = "forged-workflow"
        elif case == "name":
            payload["name"] = "../escape"
        elif case == "image-kind":
            payload["image_kind"] = "pyt"
        elif case == "bundle-args":
            payload["spec_bundle"]["args"] = "not-an-array"
        elif case == "duplicate-target":
            payload["mounts"].append(dict(payload["mounts"][0]))
        elif case == "relative-source":
            payload["mounts"][0]["source"] = "relative/results"
        elif case == "relative-target":
            payload["mounts"][0]["target"] = "relative-results"
        elif case == "missing-base-target":
            payload["mounts"] = [
                item for item in payload["mounts"] if item["target"] != "/specs"
            ]

    request = _rewrite_action_request(request_path, mutation)
    completed = subprocess.run(
        [
            sys.executable,
            str(IAA_SCRIPTS / "run_deft_cli.py"),
            "--request",
            str(request_path),
            "--",
            request["spec_bundle"]["command"],
            *request["spec_bundle"]["args"],
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "VIRTUAL_ENV": request["virtualenv"]},
        check=False,
    )
    assert completed.returncode == 2
    assert "run_deft_cli:" in completed.stderr


def test_virtualenv_shim_rejects_same_digest_symlink_replacement(tmp_path):
    workspace, _, _, _, args = _write_fixture(tmp_path, "virtualenv")
    venv_bin = workspace / "tao-ds-venv" / "bin"
    executable = venv_bin / "embedding"
    marker = tmp_path / "must-not-run"
    executable.write_text(
        f"#!{venv_bin / 'python'}\n"
        "import pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    request_path, request = action.prepare(args)

    outside = tmp_path / "outside-entrypoint"
    outside.write_bytes(executable.read_bytes())
    outside.chmod(0o755)
    executable.unlink()
    os.symlink(outside, executable)

    completed = subprocess.run(
        [
            sys.executable,
            str(IAA_SCRIPTS / "run_deft_cli.py"),
            "--request",
            str(request_path),
            "--",
            request["spec_bundle"]["command"],
            *request["spec_bundle"]["args"],
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "VIRTUAL_ENV": request["virtualenv"]},
        check=False,
    )
    assert completed.returncode == 127
    assert "without symlink traversal" in completed.stderr
    assert not marker.exists()


def test_virtualenv_shim_rejects_request_moved_outside_approved_stage(tmp_path):
    _, _, _, _, args = _write_fixture(tmp_path, "virtualenv")
    request_path, request = action.prepare(args)
    moved = tmp_path / request_path.name
    moved.write_bytes(request_path.read_bytes())
    completed = subprocess.run(
        [
            sys.executable,
            str(IAA_SCRIPTS / "run_deft_cli.py"),
            "--request",
            str(moved),
            "--",
            request["spec_bundle"]["command"],
            *request["spec_bundle"]["args"],
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "VIRTUAL_ENV": request["virtualenv"]},
        check=False,
    )
    assert completed.returncode == 2
    assert "must remain at" in completed.stderr


def test_virtualenv_shim_rejects_non_object_request(tmp_path):
    request = tmp_path / "bad.action.json"
    request.write_text("[]\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(IAA_SCRIPTS / "run_deft_cli.py"),
            "--request",
            str(request),
            "--",
            "embedding",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "VIRTUAL_ENV": str(tmp_path / "venv")},
        check=False,
    )
    assert completed.returncode == 2
    assert "invalid action request" in completed.stderr


def test_virtualenv_four_verb_submit_status_logs_executes_iaa_action(tmp_path):
    workspace, results, _, output, args = _write_fixture(tmp_path, "virtualenv")
    state_path = results / "deft_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["config"]["num_gpus"] = 2
    state["config"]["gpu_ids"] = [6, 7]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    venv_bin = workspace / "tao-ds-venv" / "bin"
    executable = venv_bin / "embedding"
    executable.write_text(
        f"#!{venv_bin / 'python'}\n"
        "import pathlib, sys\n"
        "token = next(v for v in sys.argv if v.startswith('output_parquet='))\n"
        "path = pathlib.Path(token.split('=', 1)[1])\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "import os\n"
        "path.write_bytes(b'PAR1 ' + os.environ['CUDA_VISIBLE_DEVICES'].encode())\n"
        "print('IAA virtualenv action complete')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    request_path, request = action.prepare(args)
    runtime_dir = Path(request["platform_runtime_dir"])
    runner = (
        REPO
        / "skills"
        / "platform"
        / "tao-run-on-virtualenv"
        / "references"
        / "virtualenv_runner.py"
    )
    command = [
        sys.executable,
        str(runner),
        "submit",
        "--job-dir",
        str(runtime_dir),
        "--venv",
        request["virtualenv"],
        "--script",
        request["virtualenv_shim"],
        "--job-id",
        "iaa-virtualenv-smoke",
        "--gpu-ids",
        "6,7",
        "--gpus",
        "2",
        "--arg=--request",
        f"--arg={request_path}",
        "--arg=--",
        *[
            f"--arg={token}"
            for token in [
                request["spec_bundle"]["command"],
                *request["spec_bundle"]["args"],
            ]
        ],
    ]
    submitted = subprocess.run(command, capture_output=True, text=True, check=False)
    assert submitted.returncode == 0, submitted.stdout + submitted.stderr
    submission = json.loads(submitted.stdout)
    assert submission["status"] == "RUNNING"

    deadline = time.monotonic() + 15
    status = {}
    while time.monotonic() < deadline:
        polled = subprocess.run(
            [sys.executable, str(runner), "status", "--job-dir", str(runtime_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        status = json.loads(polled.stdout)
        if status.get("status") in {"COMPLETE", "ERROR", "CANCELED"}:
            break
        time.sleep(0.05)
    assert status.get("status") == "COMPLETE", status
    assert output.read_bytes() == b"PAR1 6,7"
    submit_meta = json.loads(
        (runtime_dir / ".tao_runner" / "submit_meta.json").read_text(encoding="utf-8")
    )
    assert submit_meta["gpu_ids"] == [6, 7]
    logs = subprocess.run(
        [
            sys.executable,
            str(runner),
            "logs",
            "--job-dir",
            str(runtime_dir),
            "--tail",
            "20",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert logs.returncode == 0
    assert "IAA virtualenv action complete" in logs.stdout


@pytest.mark.parametrize(
    "platform,generation_nodes",
    [(platform, 1) for platform in PLATFORMS]
    + [("slurm", 3), ("brev", 3), ("airflow", 3)],
)
def test_config_approval_immutably_binds_selected_platform(
    tmp_path, platform, generation_nodes
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    images = archive_root / "images_raw.tar"
    metadata = archive_root / "meta.tar.gz"
    images.write_bytes(b"images")
    metadata.write_bytes(b"metadata")
    virtualenvs = None
    if platform == "virtualenv":
        virtualenvs = {}
        for profile, clis in (("pyt", ("clip",)), ("ds", ("embedding", "tmm"))):
            virtualenv = workspace / f"tao-{profile}-venv"
            (virtualenv / "bin").mkdir(parents=True)
            (virtualenv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
            os.symlink(sys.executable, virtualenv / "bin" / "python")
            for cli in clis:
                entrypoint = virtualenv / "bin" / cli
                entrypoint.write_text(
                    f"#!{virtualenv / 'bin' / 'python'}\n", encoding="utf-8"
                )
                entrypoint.chmod(0o755)
            virtualenvs[profile] = virtualenv
    args = prepare_config._parser().parse_args(  # noqa: SLF001
        [
            "--workspace",
            str(workspace),
            "--results-dir",
            str(workspace / "results" / f"run_{platform}_{generation_nodes}"),
            "--dataset-root",
            str(workspace / "data" / "iaa_v31_tao_ft"),
            "--images-archive",
            str(images),
            "--metadata-archive",
            str(metadata),
            "--platform",
            platform,
            "--max-iterations",
            "1",
            *_managed_sdg_args(platform, generation_nodes=generation_nodes),
            *(
                [
                    "--pyt-virtualenv",
                    str(virtualenvs["pyt"]),
                    "--ds-virtualenv",
                    str(virtualenvs["ds"]),
                ]
                if virtualenvs is not None
                else []
            ),
        ]
    )
    report = prepare_config.materialize(args)
    approval = json.loads(Path(report["approval_manifest"]).read_text(encoding="utf-8"))
    assert approval["schema_version"] == "3"
    assert approval["platform"] == platform
    assert approval["docker_remote"] is False
    assert approval["sdg"]["generation_nodes"] == generation_nodes
    assert approval["sdg"]["gpus_per_generation_node"] == (
        4 if platform in {"brev", "airflow"} and generation_nodes == 1
        else 8 if platform in {"slurm", "kubernetes", "brev", "airflow"}
        else 1
    )
    assert approval["virtualenvs"] == (
        {name: str(path.resolve()) for name, path in virtualenvs.items()}
        if virtualenvs is not None
        else None
    )


def test_config_approval_immutably_binds_remote_docker_mode(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    archives = tmp_path / "archives"
    archives.mkdir()
    images = archives / "images_raw.tar"
    metadata = archives / "meta.tar.gz"
    images.write_bytes(b"images")
    metadata.write_bytes(b"metadata")
    args = prepare_config._parser().parse_args(  # noqa: SLF001
        [
            "--workspace",
            str(workspace),
            "--results-dir",
            str(workspace / "results" / "run_remote_docker"),
            "--dataset-root",
            str(workspace / "data" / "iaa_v31_tao_ft"),
            "--images-archive",
            str(images),
            "--metadata-archive",
            str(metadata),
            "--platform",
            "docker",
            "--docker-remote",
            "--max-iterations",
            "1",
            *_managed_sdg_args(),
        ]
    )
    report = prepare_config.materialize(args)
    approval = json.loads(
        Path(report["approval_manifest"]).read_text(encoding="utf-8")
    )
    assert approval["platform"] == "docker"
    assert approval["docker_remote"] is True
    assert report["docker_remote"] is True


def test_airflow_orchestration_keeps_docker_as_compute_platform(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    archives = tmp_path / "archives"
    archives.mkdir()
    images = archives / "images_raw.tar"
    metadata = archives / "meta.tar.gz"
    images.write_bytes(b"images")
    metadata.write_bytes(b"metadata")
    args = prepare_config._parser().parse_args(  # noqa: SLF001
        [
            "--workspace", str(workspace),
            "--results-dir", str(workspace / "results" / "run_airflow_docker"),
            "--dataset-root", str(workspace / "data" / "iaa_v31_tao_ft"),
            "--images-archive", str(images),
            "--metadata-archive", str(metadata),
            "--platform", "docker",
            "--orchestrator", "airflow",
            "--max-iterations", "1",
            "--visible-gpu-ids", "0,1,2,3,4,5,6,7",
            "--image-edit-gpu-ids", "0,1,2,3",
            "--vlm-gpu-ids", "4",
            "--llm-gpu-ids", "5",
            "--num-gpus", "2",
            "--gpu-ids", "6,7",
        ]
    )

    report = prepare_config.materialize(args)
    approval = json.loads(Path(report["approval_manifest"]).read_text())

    assert report["platform"] == approval["platform"] == "docker"
    assert report["orchestrator"] == approval["orchestrator"] == "airflow"
    assert approval["sdg"]["gpus_per_generation_node"] == 4
    assert init_deft_state.main(
        [
            "--results-dir", str(workspace / "results" / "run_airflow_docker"),
            "--workspace", str(workspace),
            "--dataset-root", str(workspace / "data" / "iaa_v31_tao_ft"),
            "--images-archive", str(images),
            "--metadata-archive", str(metadata),
            "--max-iterations", "1",
            "--platform", "docker",
            "--orchestrator", "airflow",
            "--pyt-image", PYT_IMAGE,
            "--ds-image", DS_IMAGE,
            "--deft-config", report["deft_config"],
            "--sdg-config", report["sdg_config"],
            "--tao-spec", report["tao_spec"],
        ]
    ) == 0
    state = json.loads(
        (workspace / "results" / "run_airflow_docker" / "deft_state.json").read_text()
    )
    assert state["config"]["platform"] == "docker"
    assert state["config"]["orchestrator"] == "airflow"
    audit = audit_deft_run.audit(workspace / "results" / "run_airflow_docker")
    assert "state immutable approval fields disagree with approval.json" not in audit["errors"]


def test_new_airflow_orchestration_rejects_overloaded_airflow_platform(tmp_path):
    parser = prepare_config._parser()  # noqa: SLF001
    args = parser.parse_args(
        [
            "--workspace", str(tmp_path),
            "--results-dir", str(tmp_path / "results" / "run_bad"),
            "--dataset-root", str(tmp_path / "data" / "iaa"),
            "--images-archive", str(tmp_path / "images_raw.tar"),
            "--metadata-archive", str(tmp_path / "meta.tar.gz"),
            "--platform", "airflow",
            "--orchestrator", "airflow",
            "--max-iterations", "1",
            *_managed_sdg_args("airflow"),
        ]
    )
    with pytest.raises(ValueError, match="reserved for resuming"):
        prepare_config.materialize(args)


def test_virtualenv_selection_requires_a_real_virtualenv(tmp_path):
    parser = prepare_config._parser()  # noqa: SLF001
    args = parser.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "results" / "run_bad"),
            "--dataset-root",
            str(tmp_path / "data" / "iaa"),
            "--images-archive",
            str(tmp_path / "images_raw.tar"),
            "--metadata-archive",
            str(tmp_path / "meta.tar.gz"),
            "--platform",
            "virtualenv",
            "--max-iterations",
            "1",
            *_managed_sdg_args(),
        ]
    )
    with pytest.raises(ValueError, match="--pyt-virtualenv and --ds-virtualenv"):
        prepare_config.materialize(args)


def test_virtualenv_execution_profiles_cannot_reuse_workspace_control_env(tmp_path):
    control = tmp_path / ".venv"
    args = prepare_config._parser().parse_args(  # noqa: SLF001
        [
            "--workspace",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "results" / "run_bad"),
            "--dataset-root",
            str(tmp_path / "data" / "iaa"),
            "--images-archive",
            str(tmp_path / "images_raw.tar"),
            "--metadata-archive",
            str(tmp_path / "meta.tar.gz"),
            "--platform",
            "virtualenv",
            "--pyt-virtualenv",
            str(control),
            "--ds-virtualenv",
            str(tmp_path / "tao-ds-venv"),
            "--max-iterations",
            "1",
            *_managed_sdg_args(),
        ]
    )
    with pytest.raises(ValueError, match="separate from the workspace control"):
        prepare_config.materialize(args)


def test_virtualenv_contract_rejects_executable_without_distribution_metadata(
    tmp_path, monkeypatch
):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    os.symlink(sys.executable, venv / "bin" / "python")
    clip = venv / "bin" / "clip"
    clip.write_text(f"#!{venv / 'bin' / 'python'}\n", encoding="utf-8")
    clip.chmod(0o755)
    import virtualenv_runtime

    monkeypatch.setattr(
        virtualenv_runtime,
        "lock_status",
        lambda profile: {"ready_to_install": True, "blocker": None},
    )
    with pytest.raises(ValueError, match="runtime contract"):
        validate_tao_virtualenv(venv, profile="pyt", probe_imports=False)
