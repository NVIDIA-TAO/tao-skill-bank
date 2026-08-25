from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import sys
import time
import contextlib
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "skills/applications/tao-run-deft-iaa/scripts"
sys.path.insert(0, str(SCRIPTS))


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_adapter_commands_are_exact_and_cpu_only_contract():
    contract = _module("command_contract")
    config = {"metric_name": "Rank-1"}
    command = contract.expected_container_command("gap_analysis", "iter2", config)
    assert command == [
        "python3", "/iaa-runtime/run_iaa_compute.py", "gap_analysis",
        "--results-dir", "/results", "--label", "iter2",
    ]
    assert contract.expected_image_kind("gap_analysis") == "ds"
    assert contract.expected_image_kind("publish_checkpoint") == "pyt"
    assert contract.expected_container_command("sdg_normalize_repair", "iter2", config) == [
        "python3", "/iaa-runtime/repair_sdg_normalize_freshness.py", "recompute",
        "--results-dir", "/results", "--iteration", "2",
    ]
    assert contract.expected_image_kind("sdg_normalize_repair") == "ds"


def test_adapter_stage_and_outputs_are_bounded(tmp_path: pathlib.Path):
    contract = _module("command_contract")
    assert contract.expected_stage_directory("dataset_rebuild", "baseline", tmp_path) == tmp_path / "dataset_setup"
    assert contract.expected_stage_directory("metric_parse", "baseline", tmp_path) == tmp_path / "zs/evaluate"
    assert contract.expected_fresh_outputs("evaluate", "baseline", tmp_path) == [
        tmp_path / "zs/evaluate/nvidia_pas_metrics.csv",
        tmp_path / "zs/evaluate/nvidia_pas_metrics_aggregate.csv",
        tmp_path / "zs/evaluate/nvidia_pas_metrics_weighted_aggregate.csv",
        tmp_path / "zs/evaluate/status.json",
    ]
    assert contract.expected_fresh_outputs("report", "terminal", tmp_path) == [tmp_path / "DEFT_Loop_Report.html"]
    assert contract.expected_fresh_outputs("train_config", "iter3", tmp_path) == [tmp_path / "iter_3/specs/train_config.yaml"]
    assert contract.expected_fresh_outputs("history_select", "iter2", tmp_path) == [
        tmp_path / "iter_2/mining/mined_image_list.txt",
        tmp_path / "iter_2/mining/mined_pairs.json",
        tmp_path / "iter_2/mining/mined_dataset.json",
        tmp_path / "iter_2/mining/cumulative_mined_unique_names.json",
    ]
    assert contract.expected_stage_directory("sdg_normalize_repair", "iter2", tmp_path) == tmp_path / "iter_2/datagen"
    assert contract.expected_fresh_outputs("sdg_normalize_repair", "iter2", tmp_path) == [
        tmp_path / "iter_2/datagen/dataset/sdg_manifest.json",
        tmp_path / "iter_2/datagen/dataset/sdg_pairs.json",
        tmp_path / "iter_2/datagen/dataset/sdg_image_list.txt",
    ]


def test_history_attempt_two_repairs_only_proven_omitted_candidates(
    tmp_path: pathlib.Path, monkeypatch
):
    compute = _module("run_iaa_compute")
    results = tmp_path / "results/run"
    mining = results / "iter_1/mining"
    candidates = mining / "history_candidates"
    candidates.mkdir(parents=True)
    pairs = candidates / "mined_pairs.json"
    pairs.write_text("[]\n", encoding="utf-8")
    log = mining / "history_select.log"
    missing_manifest = candidates / "mined_dataset.json"
    log.write_text(f"No such file or directory: '{missing_manifest}'\n", encoding="utf-8")
    (mining / "history_select.attempt-1.status.json").write_text(json.dumps({
        "workflow": "tao-run-deft-iaa", "name": "history_select",
        "attempt": 1, "status": "error", "backend_state": "ERROR",
        "exit_code": 1, "log_path": str(log),
    }), encoding="utf-8")
    calls = []

    def repair(argv):
        calls.append(argv)
        (candidates / "mined_image_list.txt").write_text("/data/a.jpg\n")
        missing_manifest.write_text('{"image_dir": "/data"}\n')

    monkeypatch.setattr(compute, "_run", repair)
    compute._repair_history_candidates_after_proven_failure(  # noqa: SLF001
        results, {"config": {"history_aware": True}}, "iter1"
    )
    assert len(calls) == 1
    assert calls[0][calls[0].index("--iter-num") + 1] == "1"


def test_history_candidate_repair_rejects_missing_declared_pairs(tmp_path):
    compute = _module("run_iaa_compute")
    results = tmp_path / "results/run"
    (results / "iter_1/mining/history_candidates").mkdir(parents=True)
    with pytest.raises(ValueError, match="pairs are missing"):
        compute._repair_history_candidates_after_proven_failure(  # noqa: SLF001
            results, {"config": {"history_aware": True}}, "iter1"
        )


def test_sdg_normalize_repair_bundle_is_zero_gpu_with_exact_outputs(tmp_path: pathlib.Path):
    action = _module("run_deft_action")
    results = tmp_path / "workspace/results/run"
    dataset = tmp_path / "workspace/data/iaa"
    config = results / "config"
    cache = tmp_path / "workspace/cache"
    for path in (results / "iter_2/datagen/dataset", dataset, config, cache):
        path.mkdir(parents=True, exist_ok=True)
    outputs = [
        results / "iter_2/datagen/dataset/sdg_manifest.json",
        results / "iter_2/datagen/dataset/sdg_pairs.json",
        results / "iter_2/datagen/dataset/sdg_image_list.txt",
    ]
    context = SimpleNamespace(
        name="sdg_normalize_repair", label="iter2", results_dir=results,
        dataset_root=dataset, config_dir=config, cache_dir=cache,
        image_kind="ds", config={"num_gpus": 8},
        stage_dir=results / "iter_2/datagen", fresh_outputs=outputs,
        command=["python3", "/iaa-runtime/repair_sdg_normalize_freshness.py", "recompute"],
    )
    controller = {"root": str(tmp_path / "snapshot/skills")}
    patches = {"root": str(tmp_path / "snapshot/patches")}
    bundle = action._bundle(context, "nvcr.io/test/ds:1", 1, 1, controller, patches)  # noqa: SLF001
    assert bundle["compute_shape"] == {"gpus": 0, "nodes": 1}
    assert [row["spec_key"] for row in bundle["declared_outputs"]] == [
        "dataset/sdg_manifest.json", "dataset/sdg_pairs.json", "dataset/sdg_image_list.txt",
    ]
    assert any(
        row == {
            "spec_key": "dataset_root", "type": "folder", "uri": str(dataset)
        }
        for row in bundle["declared_inputs"]
    )


def test_sdg_normalize_repair_rejects_broader_output_contract(tmp_path: pathlib.Path):
    contract = _module("command_contract")
    exact = contract.expected_fresh_outputs("sdg_normalize_repair", "iter1", tmp_path)
    assert exact != [*exact, tmp_path / "iter_1/datagen/dataset/images/extra.jpg"]


def test_slurm_cpu_template_never_requests_gpus():
    template = (ROOT / "templates/slurm/cpu.sbatch.tmpl").read_text()
    assert "--gres" not in template
    assert "@@NUM_GPUS@@" not in template
    assert "--container-image=@@IMAGE@@" in template
    assert "--container-mounts=@@CONTAINER_MOUNTS@@" in template
    assert (
        "--container-env=HF_HOME,HOME,IAA_COMPUTE_FRAME,PYTHONPATH,XDG_CACHE_HOME"
        in template
    )


def test_metric_adapter_reads_top_level_metric_contract(tmp_path: pathlib.Path):
    compute = _module("run_iaa_compute")
    state = {
        "metric_contract": {
            "metric_name": "Rank-1", "query_type": "medium", "op": ">=", "target": 0.4,
        },
        "config": {"metric_name": "wrong-location"},
    }
    argv = compute.metric_argv(tmp_path, state, "iter2")
    assert argv[argv.index("--metric-name") + 1] == "Rank-1"
    assert argv[argv.index("--query-type") + 1] == "medium"
    assert argv[argv.index("--target") + 1] == "0.4"
    assert str(tmp_path / "iter_2/evaluate/metric_result.json") in argv


def test_adapter_dependency_contract_is_typed():
    compute = _module("run_iaa_compute")
    assert "pyarrow" in compute.ADAPTER_IMPORTS["dataset_materialize"]
    assert "matplotlib" in compute.ADAPTER_IMPORTS["report"]
    assert compute.ADAPTER_IMPORTS["metric_parse"] == ()


def test_terminal_report_declares_and_mounts_dataset_read_only(tmp_path: pathlib.Path):
    action = _module("run_deft_action")
    results = tmp_path / "workspace/results/run"
    dataset = tmp_path / "workspace/data/iaa"
    config = results / "config"
    cache = tmp_path / "workspace/cache"
    for path in (results, dataset, config, cache):
        path.mkdir(parents=True, exist_ok=True)
    context = SimpleNamespace(
        name="report", results_dir=results, dataset_root=dataset,
        config_dir=config, cache_dir=cache, image_kind="ds",
        config={
            "num_gpus": 2,
            "images_archive": str(tmp_path / "inputs/images.tar"),
            "metadata_archive": str(tmp_path / "metadata/meta.tar.gz"),
        }, stage_dir=results,
        fresh_outputs=[results / "DEFT_Loop_Report.html"],
        command=["python3", "/iaa-runtime/run_iaa_compute.py", "report"],
    )
    controller = {"root": str(tmp_path / "snapshot/skills")}
    patches = {"root": str(tmp_path / "snapshot/patches")}
    mounts = action._mounts(context, controller, patches)  # noqa: SLF001
    assert {tuple((row["source"], row["target"], row["read_only"])) for row in mounts} >= {
        (str(dataset), f"/data/{dataset.name}", True),
        (str(dataset), str(dataset), True),
    }
    bundle = action._bundle(context, "nvcr.io/test/ds:1", 1, 1, controller, patches)  # noqa: SLF001
    dataset_inputs = [
        row for row in bundle["declared_inputs"] if row["spec_key"] == "dataset_root"
    ]
    assert dataset_inputs == [{
        "spec_key": "dataset_root", "type": "folder", "uri": str(dataset)
    }]
    archive_inputs = {
        row["uri"] for row in bundle["declared_inputs"]
        if row["spec_key"].startswith("archive_parent_")
    }
    assert archive_inputs == {str(tmp_path / "inputs"), str(tmp_path / "metadata")}
    mount_targets = {row["target"] for row in mounts}
    assert {str(tmp_path / "inputs"), str(tmp_path / "metadata")} <= mount_targets


def test_history_adapter_enables_resume_only_from_unique_iteration_evidence(
    monkeypatch, tmp_path: pathlib.Path
):
    compute = _module("run_iaa_compute")
    (tmp_path / "config").mkdir()
    (tmp_path / "mining_selection_history.json").write_text(
        json.dumps({"version": 1, "iterations": [{"iteration": 2}]})
    )
    captured = []
    monkeypatch.setattr(compute, "_run", lambda argv: captured.append(argv))
    compute.stage_adapter(tmp_path, {}, "history_select", "iter2")
    assert captured[0][-1] == "--resume"

    (tmp_path / "mining_selection_history.json").write_text(
        json.dumps({"version": 1, "iterations": [{"iteration": 2}, {"iteration": 2}]})
    )
    with pytest.raises(ValueError, match="duplicates iteration 2"):
        compute.stage_adapter(tmp_path, {}, "history_select", "iter2")


def test_adapter_commit_binds_nested_host_outputs(monkeypatch, tmp_path: pathlib.Path):
    commit = _module("commit_stage")
    host_status = tmp_path / "iter_1/visualization/visualize-prepare.host.status.json"
    samples = tmp_path / "iter_1/visualization/samples"
    calls = []
    monkeypatch.setattr(commit, "expected_fresh_outputs", lambda *_args: [host_status])
    monkeypatch.setattr(commit, "expected_container_command", lambda *_args: ["command"])
    monkeypatch.setattr(commit, "expected_image_kind", lambda *_args: "ds")
    monkeypatch.setattr(
        commit,
        "_required_command_status",
        lambda path, name, **kwargs: calls.append((path, name, kwargs)) or str(path),
    )
    result = commit._required_adapter_status(
        tmp_path / "platform.status.json", "--status", scope=tmp_path / "iter_1",
        outputs=[samples], adapter="visualize_prepare", label="iter1",
        config={"platform": "docker", "ds_image": "ds", "pyt_image": "pyt"},
    )
    assert result.endswith("platform.status.json")
    assert calls[0][2]["required_outputs"] == [host_status]
    assert calls[1][0] == host_status
    assert calls[1][2]["required_outputs"] == [samples]
    assert calls[1][2]["required_name"] == "visualize-prepare"


def test_command_status_freshness_uses_publication_symlink_not_target(
    tmp_path: pathlib.Path,
):
    commit = _module("commit_stage")
    results = tmp_path / "run"
    results.mkdir()
    (results / "deft_state.json").write_text("{}")
    target = results / "model.pth"
    target.write_bytes(b"checkpoint")
    old_ns = target.stat().st_mtime_ns
    time.sleep(0.01)
    published = results / "best.pth"
    published.symlink_to(target.name)
    started_ns = max(old_ns + 1, published.lstat().st_mtime_ns - 1)
    log = results / "publish.log"
    log.write_text("published\n")
    status = results / "publish.status.json"
    status.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "workflow": "tao-run-deft-iaa",
                "name": "publish-checkpoint",
                "attempt": 1,
                "status": "ok",
                "exit_code": 0,
                "started_ns": started_ns,
                "finished_at": "2026-08-20T00:00:01+00:00",
                "log_path": str(log),
                "fresh_outputs": [str(published)],
            }
        )
    )

    assert commit._required_command_status(  # noqa: SLF001
        status,
        "published checkpoint",
        scope=results,
        required_output=published,
        required_name="publish-checkpoint",
    ) == str(status)


def test_nested_publication_accepts_only_subsecond_remote_mtime_rounding(
    tmp_path: pathlib.Path,
):
    commit = _module("commit_stage")
    results = tmp_path / "run"
    results.mkdir()
    (results / "deft_state.json").write_text("{}")
    output = results / "best.pth"
    output.write_bytes(b"checkpoint")
    log = results / "publish.log"
    log.write_text("published\n")
    started_ns = time.time_ns()
    rounded_ns = started_ns - 500_000_000
    os.utime(output, ns=(rounded_ns, rounded_ns))
    status = results / "publish.status.json"
    status.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "workflow": "tao-run-deft-iaa",
                "name": "publish-checkpoint",
                "attempt": 1,
                "status": "ok",
                "exit_code": 0,
                "started_ns": started_ns,
                "finished_at": "2026-08-21T00:00:01+00:00",
                "log_path": str(log),
                "fresh_outputs": [str(output)],
            }
        )
    )

    with pytest.raises(ValueError, match="older than the command"):
        commit._required_command_status(  # noqa: SLF001
            status,
            "published checkpoint",
            scope=results,
            required_output=output,
            required_name="publish-checkpoint",
        )
    assert commit._required_command_status(  # noqa: SLF001
        status,
        "published checkpoint",
        scope=results,
        required_output=output,
        required_name="publish-checkpoint",
        allow_remote_coarse_mtime=True,
    ) == str(status)

    too_old_ns = started_ns - 1_000_000_000
    os.utime(output, ns=(too_old_ns, too_old_ns))
    with pytest.raises(ValueError, match="older than the command"):
        commit._required_command_status(  # noqa: SLF001
            status,
            "published checkpoint",
            scope=results,
            required_output=output,
            required_name="publish-checkpoint",
            allow_remote_coarse_mtime=True,
        )


@pytest.mark.parametrize("attested", (False, True))
def test_publish_adapter_coarse_mtime_requires_outer_remote_attestation(
    monkeypatch, tmp_path: pathlib.Path, attested: bool,
):
    commit = _module("commit_stage")
    platform_status = tmp_path / "publish-checkpoint.status.json"
    platform_status.write_text(
        json.dumps(
            {
                "schema_version": "2" if attested else "1",
                "platform": "slurm",
                "freshness_contract": (
                    "remote-mirror-with-delete-before-submit" if attested else "local"
                ),
                "staging_receipt_sha256": "a" * 64 if attested else None,
            }
        )
    )
    host_status = tmp_path / "publish-checkpoint.host.status.json"
    output = tmp_path / "best.pth"
    calls = []
    monkeypatch.setattr(commit, "expected_fresh_outputs", lambda *_args: [host_status])
    monkeypatch.setattr(commit, "expected_container_command", lambda *_args: ["command"])
    monkeypatch.setattr(commit, "expected_image_kind", lambda *_args: "pyt")
    monkeypatch.setattr(
        commit,
        "_required_command_status",
        lambda path, name, **kwargs: calls.append((path, name, kwargs)) or str(path),
    )

    commit._required_adapter_status(  # noqa: SLF001
        platform_status,
        "--publish-checkpoint-status",
        scope=tmp_path,
        outputs=[output],
        adapter="publish_checkpoint",
        label="iter2",
        config={"platform": "slurm", "ds_image": "ds", "pyt_image": "pyt"},
    )
    assert calls[1][2]["allow_remote_coarse_mtime"] is attested


@pytest.mark.parametrize("platform", ["docker", "virtualenv", "slurm", "brev", "kubernetes"])
def test_adapter_mutator_requires_the_selected_compute_frame(
    monkeypatch, tmp_path: pathlib.Path, platform: str
):
    compute = _module("run_iaa_compute")
    monkeypatch.setattr(
        compute, "_state", lambda _results: ({"config": {"platform": platform}}, tmp_path)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_iaa_compute.py", "eval_config", "--results-dir", str(tmp_path), "--label", "baseline"],
    )
    monkeypatch.delenv("IAA_COMPUTE_FRAME", raising=False)
    with pytest.raises(ValueError, match=f"selected {platform} compute frame"):
        compute.main()


def test_successful_adapter_emits_deterministic_nonempty_native_log(
    monkeypatch, tmp_path: pathlib.Path, capsys
):
    compute = _module("run_iaa_compute")
    monkeypatch.setattr(
        compute,
        "_state",
        lambda _results: ({"config": {"platform": "virtualenv"}}, tmp_path),
    )
    monkeypatch.setattr(compute, "_require_adapter_imports", lambda _operation: None)
    monkeypatch.setattr(compute, "metric_parse", lambda *_args: None)
    monkeypatch.setenv("IAA_COMPUTE_FRAME", "virtualenv")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_iaa_compute.py", "metric_parse", "--results-dir", str(tmp_path),
            "--label", "iter1",
        ],
    )

    assert compute.main() == 0
    assert capsys.readouterr().out == (
        "IAA_ADAPTER_COMPLETE operation=metric_parse label=iter1\n"
    )


def test_action_owned_snapshot_survives_source_cache_replacement(tmp_path):
    action = _module("run_deft_action")
    source = tmp_path / "plugin-cache" / "scripts"
    (source / "iaa_deft").mkdir(parents=True)
    (source / "run_iaa_compute.py").write_text("VALUE = 1\n")
    (source / "iaa_deft" / "stage.py").write_text("VALUE = 2\n")
    destination = tmp_path / "run" / ".tao-runtime" / "input-snapshots" / "iaa-runtime"

    original = action._materialize_snapshot(source, destination, python_only=True)  # noqa: SLF001
    shutil.rmtree(tmp_path / "plugin-cache")
    recovered = action._materialize_snapshot(source, destination, python_only=True)  # noqa: SLF001

    assert recovered == original
    assert recovered["root"] == str(destination)
    assert {entry["path"] for entry in recovered["entries"]} == {
        "run_iaa_compute.py", "iaa_deft/stage.py",
    }


def test_action_owned_snapshot_detects_post_prepare_mutation(tmp_path):
    action = _module("run_deft_action")
    source = tmp_path / "scripts"
    source.mkdir()
    (source / "run_iaa_compute.py").write_text("VALUE = 1\n")
    destination = tmp_path / "run" / ".tao-runtime" / "snapshot"
    approved = action._materialize_snapshot(source, destination, python_only=True)  # noqa: SLF001
    target = destination / "run_iaa_compute.py"
    target.chmod(0o600)
    target.write_text("VALUE = 9\n")
    actual = action._snapshot_manifest(destination)  # noqa: SLF001
    assert actual["sha256"] != approved["sha256"]


def test_presubmit_cancellation_is_the_only_ignored_job_record_shape():
    action = _module("run_deft_action")
    job = {
        "terminal_state": "CANCELED",
        "terminal_write_by": "agent",
        "backend_ref": None,
        "transitions": [
            {"state": "PENDING"},
            {"state": "CANCELED"},
        ],
    }
    assert action._is_abandoned_presubmit_job(job)  # noqa: SLF001

    for field, value in (
        ("backend_ref", "32620000"),
        ("terminal_write_by", "poller"),
        ("terminal_state", "ERROR"),
    ):
        candidate = dict(job)
        candidate[field] = value
        assert not action._is_abandoned_presubmit_job(candidate)  # noqa: SLF001

    malformed = dict(job)
    malformed["transitions"] = [
        {"state": "PENDING"}, {"state": "RUNNING"}, {"state": "CANCELED"}
    ]
    assert not action._is_abandoned_presubmit_job(malformed)  # noqa: SLF001


def test_airflow_state_rebind_archives_only_unlaunched_request(
    tmp_path: pathlib.Path, monkeypatch
):
    action = _module("run_deft_action")
    stage = tmp_path / "results/run/zs/specs"
    stage.mkdir(parents=True)
    request_path = stage / "eval_config.action.json"
    old_state = tmp_path / "old-state"
    new_state = tmp_path / "shared/.tao"
    old_state.mkdir()
    new_state.mkdir(parents=True)
    request = {
        "attempt": 1,
        "request_sha256": "a" * 64,
        "job_state_dir": str(old_state),
        "job_binding_path": str(stage / "eval_config.job-binding.json"),
        "log_path": str(stage / "eval_config.log"),
        "staging_receipt_path": str(stage / "eval_config.staged.json"),
        "platform_runtime_dir": str(stage / ".tao-runtime/eval_config.attempt-1"),
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    context = SimpleNamespace(
        platform="slurm", config={"orchestrator": "airflow"}, name="eval_config",
        stage_dir=stage, status_path=stage / "eval_config.status.json",
        fresh_outputs=[],
    )
    replacement = dict(request)
    replacement.update(
        request_sha256="b" * 64,
        job_state_dir=str(new_state),
        started_ns=123,
    )
    monkeypatch.setattr(action, "_load_request_envelope", lambda _path: (request_path, request))
    monkeypatch.setattr(action, "_request_context", lambda _request: context)
    monkeypatch.setattr(action, "_job_state_dir", lambda: new_state)
    monkeypatch.setattr(action, "_request_lock", lambda _request: contextlib.nullcontext())
    monkeypatch.setattr(action, "_matching_job_records", lambda *_args: [])
    monkeypatch.setattr(action, "_request", lambda *_args, **_kwargs: replacement)
    monkeypatch.setattr(action, "_load_request", lambda _path: (_path, replacement))

    evidence = action.rebind_airflow_state(
        SimpleNamespace(request=request_path, confirm=True)
    )

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    proof = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["job_state_dir"] == str(new_state)
    assert proof["prior_job_state_dir"] == str(old_state)
    assert proof["replacement_job_state_dir"] == str(new_state)
    assert pathlib.Path(proof["prior_request_path"]).is_file()


def test_airflow_state_rebind_rejects_live_binding(tmp_path: pathlib.Path, monkeypatch):
    action = _module("run_deft_action")
    stage = tmp_path / "results/run/zs/specs"
    stage.mkdir(parents=True)
    old_state = tmp_path / "old-state"
    new_state = tmp_path / "shared/.tao"
    old_state.mkdir()
    new_state.mkdir(parents=True)
    request_path = stage / "eval_config.action.json"
    binding = stage / "eval_config.job-binding.json"
    binding.write_text("{}", encoding="utf-8")
    request = {
        "attempt": 1, "request_sha256": "a" * 64,
        "job_state_dir": str(old_state), "job_binding_path": str(binding),
        "log_path": str(stage / "eval_config.log"),
        "staging_receipt_path": str(stage / "eval_config.staged.json"),
        "platform_runtime_dir": str(stage / ".tao-runtime/eval_config.attempt-1"),
    }
    context = SimpleNamespace(
        platform="slurm", config={"orchestrator": "airflow"}, name="eval_config",
        stage_dir=stage, status_path=stage / "eval_config.status.json",
        fresh_outputs=[],
    )
    monkeypatch.setattr(action, "_load_request_envelope", lambda _path: (request_path, request))
    monkeypatch.setattr(action, "_request_context", lambda _request: context)
    monkeypatch.setattr(action, "_job_state_dir", lambda: new_state)
    monkeypatch.setattr(action, "_request_lock", lambda _request: contextlib.nullcontext())
    with pytest.raises(ValueError, match="bound pre-submit recovery"):
        action.rebind_airflow_state(SimpleNamespace(request=request_path, confirm=True))


def test_legacy_dataset_bridge_is_local_pre_rebind_only(tmp_path):
    audit = _module("audit_deft_run")
    status = tmp_path / "dataset_setup" / "dataset-materialize.host.status.json"
    state = {
        "config": {"platform": "docker", "docker_remote": False},
        "runtime_lineage": [{"rebound_at": "2026-08-19T22:57:59+00:00"}],
        "iterations": {"baseline": {"dataset_materialize_status": str(status)}},
    }
    entries = [{
        "iteration": "baseline", "stage": "dataset_setup", "status": "ok",
        "ts": "2026-08-19T22:10:36Z",
    }]
    assert audit._legacy_local_dataset_candidate(state, entries, tmp_path) == (  # noqa: SLF001
        state["iterations"]["baseline"], status
    )
    state["config"]["platform"] = "slurm"
    assert audit._legacy_local_dataset_candidate(state, entries, tmp_path) is None  # noqa: SLF001
    state["config"]["platform"] = "docker"
    entries[0]["ts"] = "2026-08-19T23:00:00Z"
    assert audit._legacy_local_dataset_candidate(state, entries, tmp_path) is None  # noqa: SLF001


def test_gap_status_audit_uses_feed_iteration_label():
    audit = _module("audit_deft_run")
    assert audit._status_command_label("gap_analysis_status", "baseline") == "iter1"  # noqa: SLF001
    assert audit._status_command_label("gap_analysis_status", "iter2") == "iter3"  # noqa: SLF001
    assert audit._status_command_label("eval_config_status", "baseline") == "baseline"  # noqa: SLF001


def test_platform_status_schema_types_history_resume_marker():
    schema = json.loads(
        (ROOT / "skills/applications/tao-run-deft-iaa/references/platform-action-status.schema.json").read_text()
    )
    assert schema["properties"]["resume"] == {"type": "boolean"}
