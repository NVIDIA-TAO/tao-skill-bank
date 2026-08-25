# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
IAA = ROOT / "skills/applications/tao-run-deft-iaa/scripts"
SLURM = ROOT / "skills/platform/tao-run-on-slurm/scripts"
sys.path.insert(0, str(IAA))
sys.path.insert(0, str(SLURM))


def _module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BRIDGE = _module("iaa_airflow_slurm_bridge", IAA / "airflow_slurm_action.py")
SDG_BRIDGE = _module(
    "iaa_airflow_slurm_sdg_bridge", IAA / "airflow_slurm_sdg_action.py"
)
SUBMIT = _module("iaa_slurm_submit", SLURM / "slurm_submit_action.py")


def _request(*, adapter: bool) -> dict:
    environment = {
        "HF_HOME": "/cache/huggingface",
        "HOME": "/tmp",
        "IAA_COMPUTE_FRAME": "slurm",
        "PYTHONPATH": "/patches",
        "XDG_CACHE_HOME": "/cache",
    } if adapter else {
        "HF_HOME": "/cache/huggingface",
        "HOME": "/tmp",
        "PYTHONPATH": "/patches",
        "XDG_CACHE_HOME": "/cache",
    }
    return {
        "name": "dataset_rebuild" if adapter else "evaluate",
        "environment": environment,
        "gpu_ids": [] if adapter else [0, 1],
        "mounts": [
            {"source": "/local/results", "target": "/results", "read_only": False},
            {"source": "/local/config", "target": "/specs", "read_only": True},
            {"source": "/local/cache", "target": "/cache", "read_only": False},
        ],
        "spec_bundle": {
            "network_arch": "iaa-adapter" if adapter else "clip",
            "compute_shape": {"gpus": 0 if adapter else 2, "nodes": 1},
            "command": "python3" if adapter else "clip",
            "args": ["/iaa-runtime/run_iaa_compute.py", "dataset_rebuild"]
            if adapter else ["evaluate", "-e", "/results/zs/specs/eval_config.yaml"],
        },
    }


@pytest.mark.parametrize("adapter", [True, False])
def test_render_preserves_receipt_bound_mounts_and_gpu_shape(tmp_path, adapter):
    request = _request(adapter=adapter)
    backend = {
        "/local/results": "/lustre/run/results",
        "/local/config": "/lustre/run/results/config",
        "/local/cache": "/lustre/run/cache",
    }
    text = BRIDGE._render(  # noqa: SLF001
        request=request, mount_map=backend,
        job_id=("iaa-adapter-deft-iaa-dataset_rebuild-test" if adapter
                else "clip-deft-iaa-evaluate-test"),
        image=pathlib.Path("/lustre/images/tao.sqsh"),
        log_dir=pathlib.Path("/lustre/run/logs"), account="approved-account",
        cpu_partition="cpu_short", gpu_partition="polar3",
        cpu_time_minutes=30, gpu_time_minutes=60,
    )
    rendered = tmp_path / "job.sbatch"
    rendered.write_text(text, encoding="utf-8")

    serialized_request = json.loads(json.dumps(request, sort_keys=True))
    SUBMIT._validate_iaa_rendered_mounts(  # noqa: SLF001
        serialized_request, rendered, backend
    )
    SUBMIT._validate_iaa_rendered_compute(  # noqa: SLF001
        serialized_request, rendered
    )
    assert "--gpus all" not in text
    assert f"#SBATCH --time={'00:30:00' if adapter else '01:00:00'}" in text
    assert ("#SBATCH --gres" not in text) is adapter
    if not adapter:
        assert "export NCCL_IB_DISABLE=1" in text
        assert "export NCCL_NET=Socket" in text
        assert "export NCCL_P2P_DISABLE=1" in text
        assert "#SBATCH --time=01:00:00" in text


def test_workspace_mapping_is_explicit_and_cannot_escape(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    source = local / "results/run"
    source.mkdir(parents=True)
    remote = pathlib.Path("/lustre/approved/workspace")
    assert BRIDGE._workspace_mapping(source, local, remote) == (  # noqa: SLF001
        remote / "results/run"
    )
    with pytest.raises(BRIDGE.BridgeError, match="outside"):
        BRIDGE._workspace_mapping(tmp_path / "other", local, remote)  # noqa: SLF001


def test_visualization_side_outputs_follow_immutable_flags(tmp_path):
    results = tmp_path / "workspace/results/run_1"
    state = {"config": {
        "visualize": True, "visualize_embeddings": True,
        "continual_dataset": False,
    }}
    status, outputs = BRIDGE._visualization_side_outputs(  # noqa: SLF001
        results, "iter1", state, "visualize_prepare"
    )
    assert status == results / "iter_1/visualization/visualize-prepare.host.status.json"
    assert outputs == (
        results / "iter_1/embeddings/viz_weak/input.parquet",
        results / "iter_1/mining/mined_unique_images.parquet",
        results / "iter_1/visualization/samples",
    )
    finish_status, finish_outputs = BRIDGE._visualization_side_outputs(  # noqa: SLF001
        results, "iter1", state, "visualize_finish"
    )
    assert finish_status == results / "iter_1/visualization/visualize-finish.host.status.json"
    assert finish_outputs == (results / "iter_1/visualization/tsne_plot.png",)


def test_visualization_sync_fetches_files_and_directory_with_digest_receipt(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "shared/workspace"
    results = workspace / "results/run_1"
    phase = results / "iter_1"
    status = phase / "visualization/visualize-prepare.host.status.json"
    host_log = phase / "visualization/visualize-prepare.host.log"
    status.parent.mkdir(parents=True)
    expected = (
        phase / "embeddings/viz_weak/input.parquet",
        phase / "mining/mined_unique_images.parquet",
        phase / "visualization/samples",
    )
    status.write_text(json.dumps({
        "workflow": "tao-run-deft-iaa", "name": "visualize-prepare",
        "status": "ok", "exit_code": 0,
        "log_path": str(host_log),
        "fresh_outputs": [str(path) for path in expected],
    }))
    state = {"config": {
        "visualize": True, "visualize_embeddings": True,
        "continual_dataset": False,
    }}
    file_body = b"verified parquet"
    host_log_body = b"verified adapter log"
    file_digest = hashlib.sha256(file_body).hexdigest()
    host_log_digest = hashlib.sha256(host_log_body).hexdigest()
    tree_rows = [{"relative": "sheet.png", "size": 5, "sha256": hashlib.sha256(b"sheet").hexdigest()}]
    def file_evidence(_login, remote):
        if remote.name.endswith("host.log"):
            return len(host_log_body), host_log_digest
        return len(file_body), file_digest

    monkeypatch.setattr(BRIDGE, "_remote_file_evidence", file_evidence)

    def copy_file(_login, _remote, local):
        local.write_bytes(host_log_body if local.name.endswith("host.log") else file_body)

    def copy_tree(_login, _remote, local, evidence):
        assert evidence == tree_rows
        if local.exists():
            return "reused"
        local.mkdir()
        (local / "sheet.png").write_bytes(b"sheet")
        return "fetched"

    monkeypatch.setattr(BRIDGE, "_copy_remote_file", copy_file)
    monkeypatch.setattr(BRIDGE, "_remote_tree_evidence", lambda *_args: tree_rows)
    monkeypatch.setattr(BRIDGE, "_copy_remote_tree", copy_tree)
    receipt = BRIDGE._synchronize_visualization_outputs(  # noqa: SLF001
        login="user@login", results=results, local_workspace=workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        state=state, name="visualize_prepare", recovered=False,
    )
    payload = json.loads(receipt.read_text())
    assert [row["kind"] for row in payload["outputs"]] == [
        "file", "file", "file", "directory"
    ]
    assert payload["outputs"][0]["role"] == "host_log"
    assert payload["receipt_sha256"] == BRIDGE._canonical_sha256(  # noqa: SLF001
        payload, "receipt_sha256"
    )
    assert BRIDGE._synchronize_visualization_outputs(  # noqa: SLF001
        login="user@login", results=results, local_workspace=workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        state=state, name="visualize_prepare", recovered=True,
    ) == receipt


def test_remote_train_output_evidence_is_safe_and_requires_checkpoint(monkeypatch):
    checkpoint = b"checkpoint"
    event = b"tensorboard"
    rows = [
        (
            hashlib.sha256(checkpoint).hexdigest(), len(checkpoint),
            "weights/model.pth",
        ),
        (
            hashlib.sha256(event).hexdigest(), len(event),
            "events.out.tfevents.1",
        ),
    ]
    encoded = "\n".join(
        f"{digest}|{size}|{__import__('base64').b64encode(path.encode()).decode()}"
        for digest, size, path in rows
    )
    monkeypatch.setattr(BRIDGE, "_ssh", lambda *_args, **_kwargs: encoded)
    evidence = BRIDGE._remote_train_output_evidence(  # noqa: SLF001
        "user@login", pathlib.Path("/lustre/run/iter_1/train")
    )
    assert [row["relative"] for row in evidence] == [
        "weights/model.pth", "events.out.tfevents.1"
    ]
    monkeypatch.setattr(BRIDGE, "_ssh", lambda *_args, **_kwargs: "")
    with pytest.raises(BRIDGE.BridgeError, match="no publishable checkpoint"):
        BRIDGE._remote_train_output_evidence(  # noqa: SLF001
            "user@login", pathlib.Path("/lustre/run/iter_1/train")
        )
    assert BRIDGE._remote_train_output_evidence(  # noqa: SLF001
        "user@login", pathlib.Path("/lustre/run/iter_1/train"), allow_empty=True
    ) == []


def test_train_output_sync_fetches_checkpoint_and_is_idempotent(tmp_path, monkeypatch):
    workspace = tmp_path / "shared/workspace"
    results = workspace / "results/run_1"
    phase = results / "iter_1/train"
    phase.mkdir(parents=True)
    (phase / "status.json").write_text("Train finished successfully.\n")
    bodies = {
        "weights/model.pth": b"checkpoint",
        "events.out.tfevents.1": b"tensorboard",
    }
    evidence = [
        {
            "relative": relative, "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for relative, body in bodies.items()
    ]
    monkeypatch.setattr(
        BRIDGE, "_remote_train_output_evidence", lambda *_args, **_kwargs: evidence
    )

    def copy(_login, _remote, local):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(bodies[local.relative_to(phase).as_posix()])

    monkeypatch.setattr(BRIDGE, "_copy_remote_file", copy)
    request = {"request_sha256": "a" * 64}
    receipt = BRIDGE._synchronize_train_outputs(  # noqa: SLF001
        login="user@login", results=results, local_workspace=workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        request=request,
    )
    payload = json.loads(receipt.read_text())
    assert payload["kind"] == "airflow_slurm_train_output_sync"
    assert [row["relative"] for row in payload["outputs"]] == list(bodies)
    assert all(row["disposition"] == "fetched" for row in payload["outputs"])
    assert BRIDGE._synchronize_train_outputs(  # noqa: SLF001
        login="user@login", results=results, local_workspace=workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        request=request,
    ) == receipt


def test_publish_checkpoint_sync_fetches_nested_outputs_and_log(tmp_path, monkeypatch):
    workspace = tmp_path / "shared/workspace"
    results = workspace / "results/run_1"
    phase = results / "iter_1"
    stage = phase / "train"
    stage.mkdir(parents=True)
    host_log = stage / "publish-checkpoint.host.log"
    outputs = (
        phase / "pretrained/model_state.pth",
        stage / "best/clip_best_val_t2i_mAP.json",
        stage / "best/clip_best_val_t2i_mAP.pth",
    )
    raw = stage / "model_epoch_000_step_001.pth"
    raw.write_bytes(b"raw checkpoint")
    status = stage / "publish-checkpoint.host.status.json"
    status.write_text(json.dumps({
        "workflow": "tao-run-deft-iaa", "name": "publish-checkpoint",
        "status": "ok", "exit_code": 0, "log_path": str(host_log),
        "fresh_outputs": [str(path) for path in outputs],
    }))
    bodies = {
        host_log: b"published\n", outputs[0]: b"normalized",
        outputs[1]: json.dumps({
            "selected_checkpoint": str(raw),
            "published_checkpoint": str(outputs[2]),
            "publish_mode": "symlink",
        }).encode(),
    }

    def evidence(_login, remote):
        local = workspace / remote.relative_to("/lustre/workspace")
        body = bodies[local]
        return len(body), hashlib.sha256(body).hexdigest()

    def copy(_login, remote, local):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(bodies[local])

    monkeypatch.setattr(BRIDGE, "_remote_file_evidence", evidence)
    monkeypatch.setattr(BRIDGE, "_copy_remote_file", copy)
    monkeypatch.setattr(
        BRIDGE, "_remote_symlink_target_evidence",
        lambda _login, _remote: (
            "../model_epoch_000_step_001.pth",
            pathlib.Path("/lustre/workspace")
            / raw.relative_to(workspace),
            raw.stat().st_size,
            hashlib.sha256(raw.read_bytes()).hexdigest(),
        ),
    )
    request = {"request_sha256": "b" * 64}
    receipt = BRIDGE._synchronize_publish_checkpoint_outputs(  # noqa: SLF001
        login="user@login", results=results, local_workspace=workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        request=request, recovered=True,
    )
    payload = json.loads(receipt.read_text())
    assert payload["kind"] == "airflow_slurm_publish_checkpoint_output_sync"
    assert payload["recovered_after_terminal"] is True
    assert [row["role"] for row in payload["outputs"]] == [
        "host_log", "output", "output", "output"
    ]
    assert outputs[2].is_symlink()
    assert outputs[2].resolve() == raw
    assert BRIDGE._synchronize_publish_checkpoint_outputs(  # noqa: SLF001
        login="user@login", results=results, local_workspace=workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        request=request, recovered=False,
    ) == receipt


def test_bridge_rejects_wrong_action_state_before_results_transfer(
    tmp_path, monkeypatch
):
    shared = tmp_path / "shared"
    results = shared / "workspace/results/run"
    results.mkdir(parents=True)
    request_path = results / "zs/specs/eval_config.action.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text("{}", encoding="utf-8")
    request = {
        "job_state_dir": str(tmp_path / "wrong-state"),
        "platform_runtime_dir": str(request_path.parent / ".tao-runtime/eval_config.attempt-1"),
    }
    state = {
        "workflow": "deft-iaa", "config": {
            "platform": "slurm", "orchestrator": "airflow",
            "workspace": str(shared / "workspace"),
        },
    }
    transferred = []
    monkeypatch.setenv("TAO_STATE_DIR", str(shared / ".tao"))
    monkeypatch.setattr(BRIDGE, "_prepare_action", lambda _args: (request_path, request))
    monkeypatch.setattr(BRIDGE, "_json", lambda *_args: state)
    monkeypatch.setattr(BRIDGE, "_stage_tree", lambda **_kwargs: transferred.append(True))
    args = SimpleNamespace(
        login="user@login", account="account", cpu_partition="cpu_short",
        gpu_partition="polar3", shared_root=shared, results_dir=results,
        remote_workspace=pathlib.Path("/remote/workspace"),
        backend_dataset_root=None,
    )
    with pytest.raises(BRIDGE.BridgeError, match="job_state_dir does not match"):
        BRIDGE.run_action(args)
    assert transferred == []


def test_airflow_slurm_sdg_outputs_are_exact_and_controller_local(tmp_path):
    results = tmp_path / "shared/workspace/results/run_1"
    outputs = SDG_BRIDGE._local_outputs(results, 2)  # noqa: SLF001
    assert outputs == [
        results / "iter_2/datagen/dataset/sdg_manifest.json",
        results / "iter_2/datagen/dataset/sdg_pairs.json",
        results / "iter_2/datagen/dataset/sdg_image_list.txt",
        results / "iter_2/datagen/sdg_execution_manifest.json",
        results / "iter_2/datagen/endpoint_pool.json",
        results / "iter_2/datagen/endpoint_manifest.json",
        results / "iter_2/datagen/status/sdg-normalize.slurm.status.json",
    ]
    assert all(path.is_relative_to(results) for path in outputs)


def test_airflow_slurm_sdg_stages_digest_scoped_canonical_consumer(tmp_path):
    consumer = SDG_BRIDGE._stage_consumer(tmp_path / "shared")  # noqa: SLF001
    assert consumer.name == "slurm_sdg_action.py"
    assert consumer.is_file()
    assert consumer.read_bytes() == (
        SLURM / "slurm_sdg_action.py"
    ).read_bytes()
    repeated = SDG_BRIDGE._stage_consumer(tmp_path / "shared")  # noqa: SLF001
    assert repeated == consumer
    assert (consumer.parents[4] / "scripts/redact_secrets.py").is_file()
    assert (
        consumer.parents[4]
        / "skills/applications/tao-run-deft-iaa/scripts/run_sdg_stage.py"
    ).is_file()


def test_airflow_slurm_sdg_error_transition_is_retry_classified(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        SDG_BRIDGE.action_bridge, "_run",
        lambda argv, **kwargs: calls.append((argv, kwargs)),
    )
    SDG_BRIDGE._mark(  # noqa: SLF001
        "iaa-sdg-job", "ERROR", tmp_path, backend_ref="12345",
        message="bounded failure", err_class="ERR_INFRA",
    )
    command, kwargs = calls[0]
    assert command[-2:] == ["--err-class", "ERR_INFRA"]
    assert command[command.index("--state") + 1] == "ERROR"
    assert command[command.index("--backend-ref") + 1] == "12345"
    assert kwargs["env"]["TAO_STATE_DIR"] == str(tmp_path)
    with pytest.raises(SDG_BRIDGE.BridgeError, match="valid only"):
        SDG_BRIDGE._mark(  # noqa: SLF001
            "iaa-sdg-job", "COMPLETE", tmp_path, err_class="ERR_INFRA"
        )


def test_airflow_slurm_sdg_preflights_before_mutation(monkeypatch):
    calls = []

    class Completed:
        stdout = '{"status":"PASS"}\n'

    monkeypatch.setenv("AIRFLOW_IAA_COORDINATOR_POOL", "iaa-coordinator")
    monkeypatch.setattr(
        SDG_BRIDGE.action_bridge, "_run",
        lambda argv, **kwargs: calls.append((argv, kwargs)) or Completed(),
    )
    SDG_BRIDGE._preflight_airflow()  # noqa: SLF001
    command, kwargs = calls[0]
    assert command[-3:] == ["preflight", "--pool", "iaa-coordinator:1"]
    assert kwargs["operation"] == "preflight Airflow SLURM SDG orchestration"


def test_airflow_slurm_sdg_pre_submit_failure_requires_exact_absence(monkeypatch):
    calls = []

    class Completed:
        def __init__(self, stdout):
            self.stdout = stdout

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        name = kwargs["operation"].removeprefix("prove no native SLURM job for ")
        return Completed(f"{name}|\n")

    monkeypatch.setattr(SDG_BRIDGE.action_bridge, "_run", run)
    assert SDG_BRIDGE._native_job_names_absent(  # noqa: SLF001
        login="user@login", job_id="iaa-sdg-job", generation_nodes=3,
    ) is True
    assert len(calls) == 4
    assert all(call[0][:3] == ["ssh", "-o", "BatchMode=yes"] for call in calls)
    assert all(call[1]["timeout"] == 30 for call in calls)

    def present(argv, **kwargs):
        name = kwargs["operation"].removeprefix("prove no native SLURM job for ")
        suffix = "12345" if name.endswith("img-001") else ""
        return Completed(f"{name}|{suffix}\n")

    monkeypatch.setattr(SDG_BRIDGE.action_bridge, "_run", present)
    assert SDG_BRIDGE._native_job_names_absent(  # noqa: SLF001
        login="user@login", job_id="iaa-sdg-job", generation_nodes=3,
    ) is False


def test_airflow_slurm_sdg_retry_job_record_binds_attempt1(tmp_path, monkeypatch):
    calls = []

    class Completed:
        stdout = "iaa-sdg-attempt-2\n"

    monkeypatch.setattr(
        SDG_BRIDGE.action_bridge, "_run",
        lambda argv, **kwargs: calls.append((argv, kwargs)) or Completed(),
    )
    request = {
        "action_id": "deft-iaa-sdg-attempt2",
        "stage_dir": "/lustre/results/run/iter_1/datagen",
    }
    job_id, record = SDG_BRIDGE._open_job(  # noqa: SLF001
        request=request, state_dir=tmp_path, image="image@sha256:" + "a" * 64,
        retry_of="iaa-sdg-attempt-1",
    )
    command, _ = calls[0]
    assert command[command.index("--retry-of") + 1] == "iaa-sdg-attempt-1"
    assert command[command.index("--action") + 1] == request["action_id"]
    assert job_id == "iaa-sdg-attempt-2"
    assert record == tmp_path / "jobs/iaa-sdg-attempt-2.json"


def test_airflow_slurm_sdg_parser_accepts_only_explicit_retry_evidence():
    parser = SDG_BRIDGE._parser()  # noqa: SLF001
    args = parser.parse_args([
        "--results-dir", "/shared/results/run", "--iteration", "1",
        "--login", "user@login", "--remote-workspace", "/lustre/workspace",
        "--shared-root", "/shared", "--backend-dataset-root", "/lustre/data",
        "--cache-dir", "/lustre/cache", "--augmentation-sqsh", "/images/a.sqsh",
        "--auto-labeling-sqsh", "/images/l.sqsh", "--image-edit-sqsh", "/images/i.sqsh",
        "--text-serving-sqsh", "/images/t.sqsh", "--account", "account",
        "--retry-from-request", "/shared/sdg.action.json",
        "--retry-from-job-record", "/shared/jobs/attempt1.json",
    ])
    assert args.retry_from_request == pathlib.Path("/shared/sdg.action.json")
    assert args.retry_from_job_record == pathlib.Path("/shared/jobs/attempt1.json")

    repair_args = parser.parse_args([
        "--results-dir", "/shared/results/run", "--iteration", "1",
        "--login", "user@login", "--remote-workspace", "/lustre/workspace",
        "--shared-root", "/shared", "--backend-dataset-root", "/lustre/data",
        "--cache-dir", "/lustre/cache", "--augmentation-sqsh", "/images/a.sqsh",
        "--auto-labeling-sqsh", "/images/l.sqsh",
        "--image-edit-sqsh", "/images/i.sqsh",
        "--text-serving-sqsh", "/images/t.sqsh", "--account", "account",
        "--repair-from-request", "/shared/sdg.attempt-2.action.json",
        "--repair-from-job-record", "/shared/jobs/attempt2.json",
    ])
    assert repair_args.repair_from_request == pathlib.Path(
        "/shared/sdg.attempt-2.action.json"
    )
    assert repair_args.repair_from_job_record == pathlib.Path(
        "/shared/jobs/attempt2.json"
    )

    reschedule_args = parser.parse_args([
        "--results-dir", "/shared/results/run", "--iteration", "1",
        "--login", "user@login", "--remote-workspace", "/lustre/workspace",
        "--shared-root", "/shared", "--backend-dataset-root", "/lustre/data",
        "--cache-dir", "/lustre/cache", "--augmentation-sqsh", "/images/a.sqsh",
        "--auto-labeling-sqsh", "/images/l.sqsh",
        "--image-edit-sqsh", "/images/i.sqsh",
        "--text-serving-sqsh", "/images/t.sqsh", "--account", "account",
        "--time-minutes", "60",
        "--reschedule-from-request", "/shared/sdg.attempt-2-repair.action.json",
        "--reschedule-from-job-record", "/shared/jobs/attempt2-repair.json",
    ])
    assert reschedule_args.time_minutes == 60
    assert reschedule_args.reschedule_from_request == pathlib.Path(
        "/shared/sdg.attempt-2-repair.action.json"
    )
    assert reschedule_args.reschedule_from_job_record == pathlib.Path(
        "/shared/jobs/attempt2-repair.json"
    )

    launch_repair_args = parser.parse_args([
        "--results-dir", "/shared/results/run", "--iteration", "1",
        "--login", "user@login", "--remote-workspace", "/lustre/workspace",
        "--shared-root", "/shared", "--backend-dataset-root", "/lustre/data",
        "--cache-dir", "/lustre/cache", "--augmentation-sqsh", "/images/a.sqsh",
        "--auto-labeling-sqsh", "/images/l.sqsh",
        "--image-edit-sqsh", "/images/i.sqsh",
        "--text-serving-sqsh", "/images/t.sqsh", "--account", "account",
        "--time-minutes", "60",
        "--launch-repair-from-request", "/shared/rescheduled.action.json",
        "--launch-repair-from-job-record", "/shared/jobs/rescheduled.json",
    ])
    assert launch_repair_args.launch_repair_from_request == pathlib.Path(
        "/shared/rescheduled.action.json"
    )
    assert launch_repair_args.launch_repair_from_job_record == pathlib.Path(
        "/shared/jobs/rescheduled.json"
    )


def test_bounded_integer_has_concise_finite_validation():
    parser = BRIDGE._parser()  # noqa: SLF001
    help_text = parser.format_help()
    assert len(help_text) < 4000
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--results-dir", "/shared/results/run", "--name", "evaluate",
            "--label", "baseline", "--login", "user@login",
            "--remote-workspace", "/lustre/workspace", "--shared-root", "/shared",
            "--pyt-sqsh", "/lustre/pyt.sqsh", "--ds-sqsh", "/lustre/ds.sqsh",
            "--account", "account", "--deadline", "59",
        ])


def test_backend_dataset_override_replaces_only_exact_dataset_source(tmp_path):
    dataset = tmp_path / "workspace/data/dataset"
    dataset.mkdir(parents=True)
    request = _request(adapter=False)
    request["mounts"].append({
        "source": str(dataset), "target": "/data/dataset", "read_only": True,
    })
    mapped = {
        row["source"]: (
            "/lustre/shared/dataset" if row["source"] == str(dataset)
            else "/lustre/mapped/" + pathlib.Path(row["source"]).name
        )
        for row in request["mounts"]
    }
    text = BRIDGE._render(  # noqa: SLF001
        request=request, mount_map=mapped, job_id="clip-deft-iaa-evaluate-test",
        image=pathlib.Path("/lustre/images/tao.sqsh"),
        log_dir=pathlib.Path("/lustre/run/logs"), account="approved-account",
        cpu_partition="cpu_short", gpu_partition="polar3",
    )
    assert "/lustre/shared/dataset:/data/dataset:ro" in text


def test_dataset_materialization_sync_fetches_exact_nested_outputs(
    tmp_path, monkeypatch
):
    local_workspace = tmp_path / "workspace"
    results = local_workspace / "results" / "run"
    status_path = results / "dataset_setup" / "dataset-materialize.host.status.json"
    status_path.parent.mkdir(parents=True)
    expected = BRIDGE._dataset_materialization_outputs(results)  # noqa: SLF001
    payloads = {
        path.name: f"verified-{index}-{path.name}\n".encode()
        for index, path in enumerate(expected, start=1)
    }
    status_path.write_text(json.dumps({
        "workflow": "tao-run-deft-iaa",
        "name": "dataset-materialize",
        "status": "ok",
        "exit_code": 0,
        "fresh_outputs": [str(path) for path in expected],
    }), encoding="utf-8")

    def evidence(_login, remote):
        body = payloads[remote.name]
        return len(body), hashlib.sha256(body).hexdigest()

    def copy(_login, remote, local):
        local.write_bytes(payloads[remote.name])

    monkeypatch.setattr(BRIDGE, "_remote_file_evidence", evidence)
    monkeypatch.setattr(BRIDGE, "_copy_remote_file", copy)
    receipt = BRIDGE._synchronize_dataset_materialization(  # noqa: SLF001
        login="user@login", results=results, local_workspace=local_workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), recovered=False,
    )

    assert all(path.read_bytes() == payloads[path.name] for path in expected)
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    assert [row["disposition"] for row in recorded["outputs"]] == [
        "fetched"
    ] * len(expected)
    assert recorded["receipt_sha256"] == BRIDGE._canonical_sha256(  # noqa: SLF001
        recorded, "receipt_sha256"
    )

    repeated = BRIDGE._synchronize_dataset_materialization(  # noqa: SLF001
        login="user@login", results=results, local_workspace=local_workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), recovered=True,
    )
    assert repeated == receipt


def test_dataset_materialization_sync_rejects_unbound_output_list(tmp_path):
    local_workspace = tmp_path / "workspace"
    results = local_workspace / "results" / "run"
    status_path = results / "dataset_setup" / "dataset-materialize.host.status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(json.dumps({
        "workflow": "tao-run-deft-iaa",
        "name": "dataset-materialize",
        "status": "ok",
        "exit_code": 0,
        "fresh_outputs": [str(results / "unexpected.txt")],
    }), encoding="utf-8")

    with pytest.raises(BRIDGE.BridgeError, match="bind exact outputs"):
        BRIDGE._synchronize_dataset_materialization(  # noqa: SLF001
            login="user@login", results=results,
            local_workspace=local_workspace,
            remote_workspace=pathlib.Path("/lustre/workspace"), recovered=True,
        )


def _gap_sync_fixture(tmp_path):
    local_workspace = tmp_path / "workspace"
    results = local_workspace / "results" / "run"
    gap = results / "iter_1" / "gaps" / "kpi_gaps.parquet"
    gap.parent.mkdir(parents=True)
    gap.write_bytes(b"bounded-gap-output")
    request = {
        "workflow": "tao-run-deft-iaa", "name": "gap_analysis",
        "label": "iter1", "request_sha256": "a" * 64,
    }
    return local_workspace, results, request


def test_gap_history_sync_fetches_and_reuses_exact_ledger(tmp_path, monkeypatch):
    local_workspace, results, request = _gap_sync_fixture(tmp_path)
    body = json.dumps({"iterations": {"1": ["caption-a"]}}).encode()

    monkeypatch.setattr(
        BRIDGE, "_remote_file_evidence",
        lambda _login, _remote: (len(body), hashlib.sha256(body).hexdigest()),
    )
    monkeypatch.setattr(
        BRIDGE, "_copy_remote_file",
        lambda _login, _remote, local: local.write_bytes(body),
    )
    receipt = BRIDGE._synchronize_gap_history(  # noqa: SLF001
        login="user@login", results=results, local_workspace=local_workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        request=request, recovered=True,
    )
    ledger = results / "caption_selection_history.json"
    assert ledger.read_bytes() == body
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    assert recorded["current"]["disposition"] == "fetched"
    assert recorded["receipt_sha256"] == BRIDGE._canonical_sha256(  # noqa: SLF001
        recorded, "receipt_sha256"
    )
    assert BRIDGE._synchronize_gap_history(  # noqa: SLF001
        login="user@login", results=results, local_workspace=local_workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        request=request, recovered=False,
    ) == receipt


def test_gap_history_sync_archives_prior_iteration_before_replace(
    tmp_path, monkeypatch
):
    local_workspace, results, request = _gap_sync_fixture(tmp_path)
    old = json.dumps({"iterations": {"1": ["caption-a"]}}).encode()
    new = json.dumps({"iterations": {"1": ["caption-a"], "2": ["caption-b"]}}).encode()
    ledger = results / "caption_selection_history.json"
    ledger.write_bytes(old)

    monkeypatch.setattr(
        BRIDGE, "_remote_file_evidence",
        lambda _login, _remote: (len(new), hashlib.sha256(new).hexdigest()),
    )
    monkeypatch.setattr(
        BRIDGE, "_copy_remote_file",
        lambda _login, _remote, local: local.write_bytes(new),
    )
    receipt = BRIDGE._synchronize_gap_history(  # noqa: SLF001
        login="user@login", results=results, local_workspace=local_workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        request=request, recovered=False,
    )
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    archive = pathlib.Path(recorded["previous"]["path"])
    assert ledger.read_bytes() == new
    assert archive.read_bytes() == old
    assert recorded["current"]["disposition"] == "replaced_after_archive"


def test_mining_candidate_sync_fetches_complete_history_input_contract(
    tmp_path, monkeypatch
):
    local_workspace = tmp_path / "workspace"
    results = local_workspace / "results" / "run"
    candidates = results / "iter_1/mining/history_candidates"
    candidates.mkdir(parents=True)
    expected = BRIDGE._mining_candidate_outputs(results, "iter1")  # noqa: SLF001
    payloads = {
        expected[0].name: b"/data/image-a.jpg\n",
        expected[1].name: b"[]\n",
        expected[2].name: b'{"image_dir": "/data"}\n',
    }
    request = {
        "workflow": "tao-run-deft-iaa", "name": "mining_postprocess",
        "label": "iter1", "request_sha256": "b" * 64,
        "fresh_outputs": [str(expected[1])],
    }
    expected[1].write_bytes(payloads[expected[1].name])

    monkeypatch.setattr(
        BRIDGE, "_remote_file_evidence",
        lambda _login, remote: (
            len(payloads[remote.name]), hashlib.sha256(payloads[remote.name]).hexdigest()
        ),
    )
    monkeypatch.setattr(
        BRIDGE, "_copy_remote_file",
        lambda _login, remote, local: local.write_bytes(payloads[remote.name]),
    )
    receipt = BRIDGE._synchronize_mining_candidates(  # noqa: SLF001
        login="user@login", results=results, local_workspace=local_workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        request=request, recovered=True,
    )
    assert all(path.read_bytes() == payloads[path.name] for path in expected)
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    assert [row["disposition"] for row in recorded["outputs"]] == [
        "fetched", "reused", "fetched",
    ]
    assert recorded["receipt_sha256"] == BRIDGE._canonical_sha256(  # noqa: SLF001
        recorded, "receipt_sha256"
    )


def test_history_host_status_sync_fetches_resume_evidence(tmp_path, monkeypatch):
    local_workspace = tmp_path / "workspace"
    results = local_workspace / "results/run"
    stage = results / "iter_1/mining"
    stage.mkdir(parents=True)
    body = json.dumps({
        "workflow": "tao-run-deft-iaa", "name": "history-select",
        "status": "ok", "exit_code": 0, "resume": False,
    }).encode()
    request = {
        "workflow": "tao-run-deft-iaa", "name": "history_select",
        "label": "iter1", "request_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        BRIDGE, "_remote_file_evidence",
        lambda _login, _remote: (len(body), hashlib.sha256(body).hexdigest()),
    )
    monkeypatch.setattr(
        BRIDGE, "_copy_remote_file",
        lambda _login, _remote, local: local.write_bytes(body),
    )
    receipt = BRIDGE._synchronize_history_host_status(  # noqa: SLF001
        login="user@login", results=results, local_workspace=local_workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        request=request, recovered=True,
    )
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    assert recorded["output"]["disposition"] == "fetched"
    assert recorded["receipt_sha256"] == BRIDGE._canonical_sha256(  # noqa: SLF001
        recorded, "receipt_sha256"
    )


def test_mining_history_sync_fetches_current_iteration_ledger(tmp_path, monkeypatch):
    local_workspace = tmp_path / "workspace"
    results = local_workspace / "results/run"
    (results / "iter_1/mining").mkdir(parents=True)
    body = json.dumps({"iterations": [{"iteration": 1, "selected": 10}]}).encode()
    request = {
        "workflow": "tao-run-deft-iaa", "name": "history_select",
        "label": "iter1", "request_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        BRIDGE, "_remote_file_evidence",
        lambda _login, _remote: (len(body), hashlib.sha256(body).hexdigest()),
    )
    monkeypatch.setattr(
        BRIDGE, "_copy_remote_file",
        lambda _login, _remote, local: local.write_bytes(body),
    )
    receipt = BRIDGE._synchronize_mining_history(  # noqa: SLF001
        login="user@login", results=results, local_workspace=local_workspace,
        remote_workspace=pathlib.Path("/lustre/workspace"), label="iter1",
        request=request, recovered=True,
    )
    assert (results / "mining_selection_history.json").read_bytes() == body
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    assert recorded["current"]["disposition"] == "fetched"


def test_presubmit_absence_proof_rejects_an_exact_native_match(monkeypatch):
    monkeypatch.setattr(BRIDGE, "_ssh", lambda *args, **kwargs: "")
    BRIDGE._prove_remote_job_absent(  # noqa: SLF001
        "user@login", "data-services-deft-iaa-pool-test"
    )

    monkeypatch.setattr(
        BRIDGE, "_ssh", lambda *args, **kwargs:
        "32790000|data-services-deft-iaa-pool-test\n",
    )
    with pytest.raises(BRIDGE.BridgeError, match="exact-name SLURM job exists"):
        BRIDGE._prove_remote_job_absent(  # noqa: SLF001
            "user@login", "data-services-deft-iaa-pool-test"
        )


def test_orchestration_evidence_paths_are_job_scoped(tmp_path):
    first = BRIDGE._orchestration_paths(tmp_path, "iaa-action-first")  # noqa: SLF001
    second = BRIDGE._orchestration_paths(tmp_path, "iaa-action-second")  # noqa: SLF001
    assert set(first).isdisjoint(second)
    assert all(path.parent == tmp_path for path in (*first, *second))
    with pytest.raises(BRIDGE.BridgeError, match="unsafe"):
        BRIDGE._orchestration_paths(tmp_path, "unsafe/job")  # noqa: SLF001
