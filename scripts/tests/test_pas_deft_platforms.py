# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform contract tests for PAS DEFT TAO actions."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
PAS_SCRIPTS = REPO / "skills" / "applications" / "tao-run-deft-pas" / "scripts"
sys.path.insert(0, str(PAS_SCRIPTS))
import run_deft_action as action  # noqa: E402
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


PLATFORMS = ("docker", "slurm", "kubernetes", "brev", "virtualenv")
PYT_IMAGE = action_contract.PINNED_IMAGES["pyt"]
DS_IMAGE = action_contract.PINNED_IMAGES["ds"]
DUMMY_SHA256 = "0" * 64
SPEC_NAMES = (
    "deft_config.yaml",
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
}


def _archive_digest_args(images: Path, metadata: Path) -> list[str]:
    return [
        "--images-archive-sha256",
        hashlib.sha256(images.read_bytes()).hexdigest(),
        "--metadata-archive-sha256",
        hashlib.sha256(metadata.read_bytes()).hexdigest(),
    ]


@pytest.fixture(autouse=True)
def _mock_expensive_virtualenv_contract_probes(monkeypatch, tmp_path):
    """Platform tests mock the contract boundary; verifier tests exercise it."""
    monkeypatch.setenv("TAO_STATE_DIR", str(tmp_path / ".tao"))

    def validate(path, *, profile, probe_imports, required_cli=None, minimum_gpus=None):
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
        "/workspace/data/pas",
        "--images-archive",
        "/inputs/images_raw.tar",
        "--images-archive-sha256",
        DUMMY_SHA256,
        "--metadata-archive",
        "/inputs/meta.tar.gz",
        "--metadata-archive-sha256",
        DUMMY_SHA256,
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


def test_platform_cli_options_default_to_local_docker_for_compatibility():
    prepare_args = prepare_config._parser().parse_args(  # noqa: SLF001
        [
            "--workspace", "/workspace",
            "--results-dir", "/workspace/results/run",
            "--dataset-root", "/workspace/data/pas",
            "--images-archive", "/inputs/images_raw.tar",
            "--images-archive-sha256", DUMMY_SHA256,
            "--metadata-archive", "/inputs/meta.tar.gz",
            "--metadata-archive-sha256", DUMMY_SHA256,
            "--max-iterations", "1",
        ]
    )
    state_args = init_deft_state._build_parser().parse_args(  # noqa: SLF001
        [
            "--results-dir", "/workspace/results/run",
            "--workspace", "/workspace",
            "--dataset-root", "/workspace/data/pas",
            "--images-archive", "/inputs/images_raw.tar",
            "--images-archive-sha256", DUMMY_SHA256,
            "--metadata-archive", "/inputs/meta.tar.gz",
            "--metadata-archive-sha256", DUMMY_SHA256,
            "--max-iterations", "1",
            "--pyt-image", PYT_IMAGE,
            "--ds-image", DS_IMAGE,
            "--deft-config", "/workspace/results/run/config/deft_config.yaml",
            "--tao-spec", "/workspace/results/run/config/tao_spec.yaml",
        ]
    )
    assert prepare_args.platform == "docker"
    assert state_args.platform == "docker"


def _write_fixture(tmp_path: Path, platform: str, *, docker_remote: bool = False):
    if docker_remote and platform != "docker":
        raise ValueError("docker_remote test fixtures require platform=docker")
    workspace = tmp_path / "workspace"
    results = workspace / "results" / f"run_{platform}"
    config_dir = results / "config"
    dataset = workspace / "data" / "pas_v31_tao_ft"
    (dataset / "images").mkdir(parents=True)
    (dataset / "captions").mkdir()
    config_dir.mkdir(parents=True)
    payloads = {
        "deft_config.yaml": "experiment:\n  results_path: /results\n",
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
            for cli in clis:
                entrypoint = venv / "bin" / cli
                entrypoint.write_text(
                    f"#!{venv / 'bin' / 'python'}\n", encoding="utf-8"
                )
                entrypoint.chmod(0o755)
            virtualenvs[profile] = str(venv)
    state = {
        "schema_version": "3",
        "workflow": "tao-run-deft-pas",
        "results_dir": str(results),
        "max_iterations": 1,
        "config": {
            "workspace": str(workspace),
            "dataset_root": str(dataset),
            "config_dir": str(config_dir),
            "deft_config": str(config_dir / "deft_config.yaml"),
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


def _remote_scope(platform: str) -> str:
    return f"/remote/tao-pas-tests/{platform}/embeddings/source"


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
            if platform in {"slurm", "kubernetes", "brev"}
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
    assert request["spec_bundle"]["action"].startswith("deft-pas-pool_embed-")
    assert request["spec_bundle"]["compute_shape"] == {"gpus": 1, "nodes": 1}
    assert request["spec_bundle"]["declared_outputs"] == [
        {"spec_key": "embeddings.parquet", "type": "file"}
    ]
    assert request["fresh_outputs"] == [str(output)]
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
        job_id="data-services-deft-pas-pool-embed-remote-docker",
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
def test_every_pas_tao_action_prepares_on_every_platform(
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
    data_mounts = {
        (str(results.parents[1] / "data"), "/data"),
        (str(results.parents[1] / "data"), str(results.parents[1] / "data")),
    }
    if name in DATASET_ACTIONS:
        assert data_mounts <= mounts
        assert "dataset_parent" in declared
    else:
        assert data_mounts.isdisjoint(mounts)
        assert "dataset_parent" not in declared


@pytest.mark.parametrize("platform", PLATFORMS)
def test_finalize_binds_native_job_record_and_fresh_output(tmp_path, platform):
    _, _, stage, output, args = _write_fixture(tmp_path, platform)
    request_path, request = action.prepare(args)
    if platform in {"slurm", "kubernetes", "brev"}:
        results_scope = _attest_remote(request_path, request)
    else:
        results_scope = str(stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"PAR1 fresh output")
    log = stage / "pool_embed.log"
    log.write_text("native platform action completed\n", encoding="utf-8")
    job_id = f"data-services-deft-pas-pool-embed-{platform}"
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
        platform in {"slurm", "kubernetes", "brev"}
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


def test_successful_action_status_prevents_relaunch(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"PAR1 completed output")
    Path(request["log_path"]).write_text("action completed\n", encoding="utf-8")
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-pas-pool-embed-no-relaunch",
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
    assert returncode == 0
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "ok"

    with pytest.raises(ValueError, match="action already completed successfully"):
        action.prepare(args)


@pytest.mark.parametrize("platform", ("slurm", "kubernetes", "brev"))
def test_remote_finalize_requires_output_absence_attestation(tmp_path, platform):
    _, _, stage, output, args = _write_fixture(tmp_path, platform)
    request_path, request = action.prepare(args)
    output.write_bytes(b"fresh")
    (stage / "pool_embed.log").write_text("log\n", encoding="utf-8")
    record_path, _ = _open_job_record(
        request_path,
        request,
        job_id=f"data-services-deft-pas-pool-embed-{platform}",
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
        job_id="data-services-deft-pas-pool-embed-crash-safe",
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


def test_reconcile_ignores_unrelated_malformed_shared_job_records(tmp_path):
    _, _, _, _, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    jobs = Path(request["job_state_dir"]) / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "invalid-json.json").write_text("{", encoding="utf-8")
    (jobs / "old-shape.json").write_text(
        json.dumps({"action": "some-other-action", "legacy": True}),
        encoding="utf-8",
    )

    reconciled = action.reconcile_request(argparse.Namespace(request=request_path))

    assert reconciled["state"] == "NO_JOB_RECORD"


def test_reconcile_rejects_malformed_matching_shared_job_record(tmp_path):
    _, _, _, _, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    jobs = Path(request["job_state_dir"]) / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "matching-but-invalid.json").write_text(
        json.dumps({"action": request["spec_bundle"]["action"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed matching job-record"):
        action.reconcile_request(argparse.Namespace(request=request_path))


def test_shallow_error_status_cannot_retry_or_destroy_live_attempt(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    record_path, _ = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-pas-pool-embed-live",
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
        job_id="data-services-deft-pas-pool-embed-attempt-1",
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
        job_id="data-services-deft-pas-pool-embed-attempt-2",
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


def test_legacy_docker_error_status_can_migrate_to_attempt_two(tmp_path):
    _, _, stage, _, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    Path(request["log_path"]).write_text("legacy Docker failure\n", encoding="utf-8")
    finished_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    legacy_status = {
        "schema_version": "1",
        "workflow": "tao-run-deft-pas",
        "kind": "container",
        "name": request["name"],
        "attempt": 1,
        "image_kind": request["image_kind"],
        "image": request["workload_image"],
        "command": [
            request["spec_bundle"]["command"],
            *request["spec_bundle"]["args"],
        ],
        "command_sha256": action.command_sha256(
            [
                request["spec_bundle"]["command"],
                *request["spec_bundle"]["args"],
            ]
        ),
        "passed_hf_token": request["passed_hf_token"],
        "started_at": request["started_at"],
        "started_ns": request["started_ns"],
        "finished_at": finished_at,
        "status": "error",
        "exit_code": 1,
        "log_path": request["log_path"],
        "fresh_outputs": request["fresh_outputs"],
    }
    (stage / "pool_embed.status.json").write_text(
        json.dumps(legacy_status), encoding="utf-8"
    )

    retry_path, retry = action.prepare(args)

    assert retry_path == stage / "pool_embed.attempt-2.action.json"
    assert retry["attempt"] == 2
    assert json.loads(
        (stage / "pool_embed.attempt-1.status.json").read_text(encoding="utf-8")
    ) == legacy_status


def test_retry_rejects_status_that_changes_prepared_command(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    Path(request["log_path"]).write_text("native failure\n", encoding="utf-8")
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-pas-pool-embed-tampered-status",
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
        job_id="data-services-deft-pas-pool-embed-mutated-lineage",
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
        job_id="data-services-deft-pas-pool-embed-too-late",
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
        job_id="data-services-deft-pas-pool-embed-wrong-local-scope",
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
        job_id="data-services-deft-pas-pool-embed-upload-excludes",
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
        job_id="data-services-deft-pas-pool-embed-first",
    )
    second_path, _ = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-pas-pool-embed-second",
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
        job_id="data-services-deft-pas-pool-embed-wrong-remote-scope",
        results_scope="/remote/tao-pas-tests/slurm/other",
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
        job_id="data-services-deft-pas-pool-embed-unbound",
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


def test_finalize_rejects_unredacted_platform_log(tmp_path):
    _, _, stage, output, args = _write_fixture(tmp_path, "docker")
    request_path, request = action.prepare(args)
    record_path, record = _open_job_record(
        request_path,
        request,
        job_id="data-services-deft-pas-pool-embed-secret-log",
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
        job_id="data-services-deft-pas-pool-embed-terminal-lineage",
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
        job_id="data-services-deft-pas-pool-embed-evidence-platform",
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
        job_id="data-services-deft-pas-pool-embed-abcdef",
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
    dataset = workspace / "data" / "pas"
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
            *_archive_digest_args(images, metadata),
            "--platform",
            "docker",
            "--max-iterations",
            "1",
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
        job_id="data-services-deft-pas-pool-embed-old",
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
        "CUDA_VISIBLE_DEVICES": "3",
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
            str(PAS_SCRIPTS / "run_deft_cli.py"),
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
    assert observed["pythonpath"] == str(PAS_SCRIPTS.parent / "patches")
    assert observed["home"] == "/tmp"
    assert observed["path"].startswith(f"{request['virtualenv']}/bin:")
    assert observed["cuda"] == "3"
    assert observed["omp"] == "7"
    assert observed["hf"] is None
    assert observed["ngc"] is None
    assert observed["brev"] is None
    assert observed["aws"] is None
    assert observed["sentinel"] is None
    assert observed["pythonhome"] is None
    assert observed["ld_preload"] is None


def test_virtualenv_shim_executes_attempt_two_request(tmp_path):
    workspace, _, _, output, args = _write_fixture(tmp_path, "virtualenv")
    venv_bin = workspace / "tao-ds-venv" / "bin"
    executable = venv_bin / "embedding"
    executable.write_text(
        f"#!{venv_bin / 'python'}\n"
        "import pathlib, sys\n"
        "output = pathlib.Path(next(v for v in sys.argv if v.startswith('output_parquet=')).split('=', 1)[1])\n"
        "output.write_bytes(b'PAR1 attempt two')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    first_path, first = action.prepare(args)
    Path(first["log_path"]).write_text("attempt one failed\n", encoding="utf-8")
    record_path, record = _open_job_record(
        first_path,
        first,
        job_id="data-services-deft-pas-pool-embed-virtualenv-attempt-1",
    )
    action.bind_job(argparse.Namespace(request=first_path, job_record=record_path))
    _finish_job_record(record_path, record, state="ERROR")
    action.finalize(
        argparse.Namespace(
            request=first_path,
            job_record=record_path,
            native_exit_code=1,
        )
    )
    second_path, second = action.prepare(args)

    completed = subprocess.run(
        [
            sys.executable,
            str(PAS_SCRIPTS / "run_deft_cli.py"),
            "--request",
            str(second_path),
            "--",
            second["spec_bundle"]["command"],
            *second["spec_bundle"]["args"],
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "VIRTUAL_ENV": second["virtualenv"]},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert second_path.name == "pool_embed.attempt-2.action.json"
    assert output.read_bytes() == b"PAR1 attempt two"


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
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    request_path, request = action.prepare(args)
    del request_path
    request["forward_env"] = ["HF_TOKEN"]
    aliases = {item["target"]: item["source"] for item in request["mounts"]}

    environment = virtualenv_cli._execution_environment(  # noqa: SLF001
        request, aliases, request["virtualenv"]
    )

    assert environment["HF_TOKEN"] == "approved-test-token"
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


def test_virtualenv_shim_rejects_runner_environment_from_wrong_profile(tmp_path):
    workspace, _, _, _, args = _write_fixture(tmp_path, "virtualenv")
    request_path, request = action.prepare(args)
    completed = subprocess.run(
        [
            sys.executable,
            str(PAS_SCRIPTS / "run_deft_cli.py"),
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
            str(PAS_SCRIPTS / "run_deft_cli.py"),
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
            str(PAS_SCRIPTS / "run_deft_cli.py"),
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
            str(PAS_SCRIPTS / "run_deft_cli.py"),
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
            str(PAS_SCRIPTS / "run_deft_cli.py"),
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


def test_virtualenv_four_verb_submit_status_logs_executes_pas_action(tmp_path):
    workspace, _, _, output, args = _write_fixture(tmp_path, "virtualenv")
    venv_bin = workspace / "tao-ds-venv" / "bin"
    executable = venv_bin / "embedding"
    executable.write_text(
        f"#!{venv_bin / 'python'}\n"
        "import pathlib, sys\n"
        "token = next(v for v in sys.argv if v.startswith('output_parquet='))\n"
        "path = pathlib.Path(token.split('=', 1)[1])\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_bytes(b'PAR1 virtualenv action')\n"
        "print('PAS virtualenv action complete')\n",
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
        "pas-virtualenv-smoke",
        "--gpu-ids",
        "0",
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
    assert output.read_bytes() == b"PAR1 virtualenv action"
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
    assert "PAS virtualenv action complete" in logs.stdout


@pytest.mark.parametrize("platform", PLATFORMS)
def test_config_approval_immutably_binds_selected_platform(tmp_path, platform):
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
            str(workspace / "results" / f"run_{platform}"),
            "--dataset-root",
            str(workspace / "data" / "pas_v31_tao_ft"),
            "--images-archive",
            str(images),
            "--metadata-archive",
            str(metadata),
            *_archive_digest_args(images, metadata),
            "--platform",
            platform,
            "--max-iterations",
            "1",
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
    assert approval["schema_version"] == "4"
    assert approval["platform"] == platform
    assert approval["docker_remote"] is False
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
            str(workspace / "data" / "pas_v31_tao_ft"),
            "--images-archive",
            str(images),
            "--metadata-archive",
            str(metadata),
            *_archive_digest_args(images, metadata),
            "--platform",
            "docker",
            "--docker-remote",
            "--max-iterations",
            "1",
        ]
    )
    report = prepare_config.materialize(args)
    approval = json.loads(
        Path(report["approval_manifest"]).read_text(encoding="utf-8")
    )
    assert approval["platform"] == "docker"
    assert approval["docker_remote"] is True
    assert report["docker_remote"] is True


def test_virtualenv_selection_requires_a_real_virtualenv(tmp_path):
    parser = prepare_config._parser()  # noqa: SLF001
    args = parser.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "results" / "run_bad"),
            "--dataset-root",
            str(tmp_path / "data" / "pas"),
            "--images-archive",
            str(tmp_path / "images_raw.tar"),
            "--metadata-archive",
            str(tmp_path / "meta.tar.gz"),
            "--images-archive-sha256",
            DUMMY_SHA256,
            "--metadata-archive-sha256",
            DUMMY_SHA256,
            "--platform",
            "virtualenv",
            "--max-iterations",
            "1",
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
            str(tmp_path / "data" / "pas"),
            "--images-archive",
            str(tmp_path / "images_raw.tar"),
            "--metadata-archive",
            str(tmp_path / "meta.tar.gz"),
            "--images-archive-sha256",
            DUMMY_SHA256,
            "--metadata-archive-sha256",
            DUMMY_SHA256,
            "--platform",
            "virtualenv",
            "--pyt-virtualenv",
            str(control),
            "--ds-virtualenv",
            str(tmp_path / "tao-ds-venv"),
            "--max-iterations",
            "1",
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
