#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for runtime-only Cosmos backend orchestration."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import shlex
import subprocess
import sys
import tomllib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "models" / "tao-finetune-cosmos-reason"
sys.path.insert(0, str(SKILL / "scripts"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load_module("cosmos_common", SKILL / "scripts" / "cosmos_common.py")
workflow = load_module("cosmos_workflow_test", SKILL / "scripts" / "cosmos_workflow.py")
metric = load_module("cosmos_metrics_test", SKILL / "scripts" / "extract_cosmos_metrics.py")
framework_action = load_module(
    "framework_checkpoint_action_test", SKILL / "scripts" / "framework_checkpoint_action.py"
)
action_bundle = load_module(
    "stage_action_bundle_test", SKILL / "scripts" / "stage_action_bundle.py"
)
framework_image_preflight = load_module(
    "framework_evaluation_image_preflight_test",
    SKILL / "scripts" / "framework_evaluation_image_preflight.py",
)


def make_model(tmp_path: Path, model_type: str = "qwen3_vl") -> Path:
    model = tmp_path / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": model_type, "architectures": ["Qwen3VLForConditionalGeneration"]}))
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "tokenizer.json").write_text("{}")
    (model / "processor_config.json").write_text("{}")
    return model


def make_video_conversation(tmp_path: Path, split: str, count: int = 16) -> tuple[Path, Path]:
    root = tmp_path / split
    media = root / "media"
    media.mkdir(parents=True)
    records = []
    for index in range(count):
        name = f"{split}-{index}.mp4"
        (media / name).write_bytes(f"video-{split}-{index}".encode())
        records.append({"id": f"{split}-{index}", "video": name, "width": 960, "height": 540, "fps": 24, "duration_seconds": 12, "conversations": [{"from": "human", "value": "<video> question"}, {"from": "gpt", "value": "Yes"}]})
    annotation = root / "manifest.json"
    annotation.write_text(json.dumps(records))
    return annotation, media


def make_task_aware_video(tmp_path: Path, split: str) -> tuple[list[Path], Path]:
    media = tmp_path / split / "media"
    media.mkdir(parents=True)
    annotations = []
    for task in ("bcq", "mcq", "scene_description"):
        items = []
        for index in range(8):
            name = f"{split}-{task}-{index}.mp4"
            (media / name).write_bytes(name.encode())
            answer = "Yes" if task == "bcq" else "A" if task == "mcq" else "A road scene"
            items.append({"id": f"{split}-{task}-{index}", "video_id": name, "task": task, "conversations": [{"from": "human", "value": "question"}, {"from": "gpt", "value": answer}]})
        path = tmp_path / split / f"{task}.json"
        path.write_text(json.dumps({"format": "tao-vl-reason-v1.0", "metadata": {"task": task}, "items": items}))
        annotations.append(path)
    return annotations, media


def args_for(tmp_path: Path, *, backend: str = "cosmos-framework", dataset_family: str = "video_conversation", run_mode: str = "full", training_mode: str = "dense", model_name: str = "nvidia/Cosmos3-Nano"):
    model = make_model(tmp_path, "cosmos3_edge" if "Edge" in model_name else "qwen3_vl")
    if dataset_family == "video_conversation":
        train_annotations, train_media = [make_video_conversation(tmp_path, "train")[0]], [tmp_path / "train" / "media"]
        val_annotations, val_media = [make_video_conversation(tmp_path, "validation")[0]], [tmp_path / "validation" / "media"]
    else:
        train_annotations, train_root = make_task_aware_video(tmp_path, "train")
        val_annotations, val_root = make_task_aware_video(tmp_path, "validation")
        train_media, val_media = [train_root], [val_root]
    for name in ("results", "checkpoints", "cache", "sqsh-cache", "integration", "framework", "rl", "daft", "tao-core"):
        (tmp_path / name).mkdir(exist_ok=True)
    ssh_key = tmp_path / "id_ed25519"; ssh_key.write_text("fixture")
    sqsh = tmp_path / "sqsh-cache" / "image.sqsh"; sqsh.write_bytes(b"sqsh")
    values = [
        "plan", "--model", model_name, "--backend", backend, "--action", "train",
        "--workload", "training", "--dataset-family", dataset_family, "--platform", "docker", "--run-mode", run_mode,
        "--training-mode", training_mode, "--base-model-path-or-uri", str(model),
        "--results-dir", str(tmp_path / "results"), "--checkpoint-dir", str(tmp_path / "checkpoints"),
        "--cache-dir", str(tmp_path / "cache"), "--sqsh-cache-dir", str(tmp_path / "sqsh-cache"),
        "--ssh-key-path", str(ssh_key), "--tao-integration-repo", str(tmp_path / "integration"),
        "--cosmos-framework-repo", str(tmp_path / "framework"), "--cosmos-rl-repo", str(tmp_path / "rl"),
        "--daft-repo", str(tmp_path / "daft"), "--tao-core-repo", str(tmp_path / "tao-core"),
        "--build-context", str(tmp_path), "--image-tag", f"example/{backend}:test",
        "--sqsh-path", str(sqsh), "--cosmos-framework-commit", "f" * 40,
        "--cosmos-rl-commit", "r" * 40, "--tao-integration-commit", "i" * 40,
        "--daft-commit", "d" * 40, "--tao-core-commit", "c" * 40,
        "--cosmos-framework-base-image", "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04",
        "--cosmos-framework-source-repository", "https://github.com/example/cosmos-framework.git",
        "--cosmos-framework-source-branch", "dev/test-framework",
        "--cosmos-rl-base-image", "example/cosmos-rl-runtime:test",
        "--cosmos-rl-source-repository", "ssh://git@gitlab.example.com:12051/group/cosmos-reason",
        "--cosmos-rl-source-branch", "feature/enhanced-hooks-and-custom-loggers",
        "--native-tree", "n" * 40, "--integration-tree", "t" * 40,
        "--daft-tree", "d" * 40, "--tao-core-tree", "c" * 40,
        "--build-timestamp", "2026-08-05T00:00:00Z", "--write-spec", str(tmp_path / "spec.toml"),
        "--nodes", "1", "--gpus-per-node", "8", "--effective-global-batch", "8",
    ]
    for annotation in train_annotations:
        values += ["--train-annotation", str(annotation)]
    for root in train_media:
        values += ["--train-media-root", str(root)]
    for annotation in val_annotations:
        values += ["--validation-annotation", str(annotation)]
    for root in val_media:
        values += ["--validation-media-root", str(root)]
    if training_mode == "peft":
        values += ["--lora-rank", "16", "--lora-alpha", "32", "--lora-dropout", "0.05", "--lora-target-modules", "q_proj", "--lora-target-modules", "v_proj", "--lora-use-rslora"]
    return workflow.parse_args(values)


def attach_decoder_artifact(args, tmp_path: Path) -> None:
    args.video_override_map = str(tmp_path / "override-map.json")
    args.video_override_manifest = str(tmp_path / "override-manifest.json")
    args.video_override_fingerprint = "a" * 64
    Path(args.video_override_map).write_text("{}")
    Path(args.video_override_manifest).write_text("{}")


def test_framework_action_bundle_is_import_closed_and_idempotent(tmp_path: Path) -> None:
    destination = tmp_path / "framework-checkpoint-input"
    first = action_bundle.stage("framework-checkpoint", destination)
    second = action_bundle.stage("framework-checkpoint", destination)
    assert first["state"] == "staged"
    assert second["state"] == "reused"
    assert {item["path"] for item in first["files"]} == {
        "framework_checkpoint_action.py",
        "cosmos_common.py",
    }
    imported = subprocess.run(
        [sys.executable, str(destination / "framework_checkpoint_action.py"), "--help"],
        cwd=destination,
        text=True,
        capture_output=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr


def test_framework_image_preflight_rejects_old_sqsh_and_accepts_baked_runtime() -> None:
    old = framework_image_preflight.check_listing(
        "squashfs-root/workspace/.venv/lib/python3.13/site-packages/cosmos_rl/evaluation/base.py\n"
        "squashfs-root/workspace/.venv/lib/python3.13/site-packages/cosmos_rl/framework/runtime.py\n",
        "/lustre/old.sqsh",
    )
    assert old["compatible"] is False
    assert old["missing_baked_paths"] == [
        "/cosmos_rl/utils/framework_torchcodec_video.py"
    ]
    current = framework_image_preflight.check_listing(
        "\n".join(
            f"squashfs-root/workspace/.venv/lib/python3.13/site-packages{suffix}"
            for suffix in framework_image_preflight.REQUIRED_SUFFIXES
        ),
        "/lustre/current.sqsh",
    )
    assert current["compatible"] is True
    listing = "\n".join(
        f"squashfs-root/workspace/.venv/lib/python3.13/site-packages{suffix}"
        for suffix in framework_image_preflight.REQUIRED_SUFFIXES
    )
    stale = framework_image_preflight.check_listing(
        listing,
        "/lustre/stale.sqsh",
        sources={suffix: "# old implementation" for suffix in framework_image_preflight.REQUIRED_SUFFIXES},
    )
    assert stale["compatible"] is False
    assert stale["source_attestation_performed"] is True
    assert set(stale["missing_source_capabilities"]) == set(
        framework_image_preflight.REQUIRED_SUFFIXES
    )
    baked = framework_image_preflight.check_listing(
        listing,
        "/lustre/baked.sqsh",
        sources={
            suffix: "\n".join(framework_image_preflight.REQUIRED_SOURCE_TOKENS[suffix])
            for suffix in framework_image_preflight.REQUIRED_SUFFIXES
        },
    )
    assert baked["compatible"] is True
    assert baked["missing_source_capabilities"] == {}


def test_node_exclusions_are_live_filtered_and_rendered_without_hand_patch(
    tmp_path: Path,
) -> None:
    args = args_for(tmp_path)
    inventory = tmp_path / "nodes.txt"
    inventory.write_text(
        "NodeName=batch-block5-00001 State=IDLE\n"
        "NodeName=batch-block5-00002 State=DOWN\n"
        "NodeName=batch-block5-00003 State=IDLE+PLANNED Comment=Run network diagnostics\n"
        "NodeName=batch-block5-00004 State=IDLE Comment=Consult runbook for pyxis-enroot mount failure\n"
        "NodeName=batch-block5-00005 State=IDLE Comment=Consult https://scheduler/runbook#io-error\n"
        "NodeName=batch-block5-00006 State=IDLE Comment=Check FACT dashboard for this node\n",
        encoding="utf-8",
    )
    args.exclude_node = ["batch-block5-00002", "retired-node-99999"]
    args.exclude_unhealthy_inventory_nodes = True
    args.slurm_node_inventory_file = str(inventory)
    plan = workflow.build_plan(args)
    assert plan["slurm_node_exclusions"]["validated"] == [
        "batch-block5-00002",
        "batch-block5-00003",
        "batch-block5-00004",
        "batch-block5-00005",
    ]
    assert plan["slurm_node_exclusions"]["auto_excluded"] == [
        "batch-block5-00002",
        "batch-block5-00003",
        "batch-block5-00004",
        "batch-block5-00005",
    ]
    assert plan["slurm_node_exclusions"]["auto_exclusion_reasons"] == {
        "batch-block5-00002": ["scheduler_state=DOWN"],
        "batch-block5-00003": ["scheduler_diagnostic_comment"],
        "batch-block5-00004": ["scheduler_diagnostic_comment"],
        "batch-block5-00005": ["scheduler_diagnostic_comment"],
    }
    assert plan["slurm_node_exclusions"]["retired_or_missing"] == [
        "retired-node-99999"
    ]

    args.platform = "slurm"
    args.partition = "polar3,polar4"
    args.account = "account"
    args.container_mount = [f"{tmp_path}:{tmp_path}"]
    args.stdout_path = str(tmp_path / "%x-%j.out")
    args.stderr_path = str(tmp_path / "%x-%j.err")
    args.timeout = "03:48:00"
    args.time_limit = "04:00:00"
    args.exclusive = True
    script = workflow.render_slurm(args, plan)
    assert (
        "#SBATCH --exclude=batch-block5-00002,batch-block5-00003,batch-block5-00004,batch-block5-00005"
        in script
    )
    assert "batch-block5-00006" not in script
    assert "retired-node-99999" not in script

    workflow.write_spec(args, plan)
    artifact = tmp_path / "sealed-plan.json"
    args.plan_artifact = str(artifact)
    plan["initial_metadata"] = workflow.initial_metadata(args, plan)
    workflow.save_plan_artifact(args, plan, str(artifact))
    rendered_path = tmp_path / "job.sbatch"
    rendered = subprocess.run(
        [
            sys.executable,
            str(SKILL / "scripts" / "cosmos_workflow.py"),
            "render-slurm",
            "--plan-artifact",
            str(artifact),
            "--render-output",
            str(rendered_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert rendered.returncode == 0, rendered.stderr
    assert "#SBATCH --exclude=batch-block5-00002" in rendered_path.read_text()


def test_node_auto_exclusions_ignore_non_target_and_non_gpu_inventory(
    tmp_path: Path,
) -> None:
    args = args_for(tmp_path)
    inventory = tmp_path / "partitioned-nodes.txt"
    inventory.write_text(
        "NodeName=polar-node State=IDLE Gres=gpu:8 Partitions=polar3 Comment=Run network diagnostics\n"
        "NodeName=other-gpu State=DOWN Gres=gpu:8 Partitions=grizzly\n"
        "NodeName=cpu-node State=DOWN Gres=(null) Partitions=cpu_long\n",
        encoding="utf-8",
    )
    args.partition = "polar3,polar4"
    args.exclude_unhealthy_inventory_nodes = True
    args.slurm_node_inventory_file = str(inventory)
    plan = workflow.build_plan(args)
    assert plan["slurm_node_exclusions"]["auto_excluded"] == ["polar-node"]


def test_retry_helper_reuses_sealed_inspection_and_refreshes_job_identity(
    tmp_path: Path,
) -> None:
    args = args_for(tmp_path)
    prior_output = tmp_path / "prior-plan.json"
    args.plan_artifact = str(prior_output)
    plan = workflow.build_plan(args)
    workflow.write_spec(args, plan)
    plan["input_frame"] = {
        "kind": "slurm_remote",
        "verified_host": "login.example.invalid",
        "inspection_transport": "repository_helper_streamed_over_ssh",
    }
    plan["initial_metadata"] = workflow.initial_metadata(args, plan)
    workflow.save_plan_artifact(args, plan, str(prior_output))

    inventory = tmp_path / "nodes.txt"
    inventory.write_text("batch-block5-00002\nbatch-block5-00003\n", encoding="utf-8")
    retry_output = tmp_path / "retry-plan.json"
    retry_root = tmp_path / "training" / "cosmos-reason-train-retry01"
    retry_spec = retry_root / "config" / "train.toml"
    retry_spec.parent.mkdir(parents=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(SKILL / "scripts" / "cosmos_retry_plan.py"),
            "--prior-plan",
            str(prior_output),
            "--job-id",
            "cosmos-reason-train-retry01",
            "--write-spec",
            str(retry_spec),
            "--exclude-node",
            "batch-block5-00002",
            "--exclude-node",
            "retired-node-99999",
            "--slurm-node-inventory",
            str(inventory),
            "--output",
            str(retry_output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    retry = json.loads(retry_output.read_text(encoding="utf-8"))
    assert retry["experiment_id"] == "cosmos-reason-train-retry01"
    assert retry["retry_preparation"]["inspection_reused"] is True
    assert retry["slurm_node_exclusions"]["validated"] == ["batch-block5-00002"]
    assert retry["slurm_node_exclusions"]["retired_or_missing"] == [
        "retired-node-99999"
    ]
    assert retry["config"]["original"] == str(retry_spec)
    request = retry["planner_request"]
    assert request["results_dir"] == str(retry_root / "results")
    assert request["checkpoint_dir"] == str(retry_root / "checkpoints")
    assert request["cache_dir"] == str(retry_root / "cache")
    assert request["container_results_dir"] == str(retry_root / "results")
    assert request["container_checkpoint_dir"] == str(retry_root / "checkpoints")
    assert request["container_cache_dir"] == str(retry_root / "cache")
    assert request["stdout_path"] == str(retry_root / "logs" / "%x-%j.out")
    assert request["stderr_path"] == str(retry_root / "logs" / "%x-%j.err")
    assert retry["retry_preparation"]["attempt_root"] == str(retry_root)
    assert retry["model"] == plan["model"]
    assert retry["datasets"] == plan["datasets"]

    render_request = deepcopy(request)
    render_request.update(
        {
            "platform": "slurm",
            "partition": "polar3,polar4",
            "account": "account",
            "container_mount": [f"{tmp_path}:{tmp_path}"],
            "timeout": "03:48:00",
            "time_limit": "04:00:00",
            "exclusive": True,
        }
    )
    rendered = workflow.render_slurm(SimpleNamespace(**render_request), retry)
    assert str(retry_root / "results") in rendered
    assert str(retry_root / "checkpoints") in rendered
    assert str(retry_root / "cache") in rendered
    assert plan["planner_request"]["results_dir"] not in rendered
    assert plan["planner_request"]["checkpoint_dir"] not in rendered
    assert plan["planner_request"]["cache_dir"] not in rendered


def test_model_backend_resolution_and_comparative_explicitness():
    assert workflow.select_backend(model="Cosmos3-Nano", action="train", workload="training")[0] == "cosmos-rl"
    assert workflow.select_backend(model="Cosmos3-Nano", action="evaluate", workload="training")[0] == "cosmos-rl"
    assert workflow.select_backend(model="Cosmos3-Nano", action="export", workload="training")[0] == "cosmos-framework"
    assert workflow.select_backend(model="Cosmos3-Edge", action="evaluate", workload="training")[0] == "cosmos-framework"
    assert workflow.select_backend(model="Cosmos3-Edge", action="inference_microservice", workload="training")[0] == "cosmos-framework"
    with pytest.raises(common.WorkflowError, match="backend selection"):
        workflow.select_backend(model="Cosmos3-Nano", action="train", backend="auto", comparative=True)


def test_toml_bytes_survive_sorted_sealed_plan_round_trip() -> None:
    spec = {
        "z_table": {"b": 2, "a": 1},
        "root_b": "b",
        "a_table": {"nested": {"z": False, "a": True}, "value": 3},
        "root_a": "a",
    }
    before = workflow.dump_toml(spec)
    after = workflow.dump_toml(json.loads(json.dumps(spec, sort_keys=True)))
    assert before == after


def make_framework_dcp(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "framework-run"
    checkpoint = run / "checkpoints" / "epoch_1"
    model_dcp = checkpoint / "model"
    model_dcp.mkdir(parents=True)
    (model_dcp / ".metadata").write_bytes(b"dcp-metadata")
    (model_dcp / "__0_0.distcp").write_bytes(b"dcp-shard")
    config = run / "config.yaml"
    config.write_text("model:\n  _target_: cosmos_framework.model.generator.vlm_model.VLMModel\n")
    base_model = make_model(tmp_path / "base-model")
    return checkpoint, config, base_model


def framework_action_args(
    checkpoint: Path,
    config: Path,
    base_model: Path,
    *,
    verb: str = "plan",
    export_dir: Path | None = None,
):
    values = [
        verb,
        "--action", "evaluate",
        "--checkpoint-path", str(checkpoint),
        "--config-file", str(config),
        "--base-model-path-or-uri", str(base_model),
        "--base-model-revision", "immutable-test-revision",
        "--python-executable", sys.executable,
    ]
    if export_dir:
        values += ["--export-dir", str(export_dir)]
    return framework_action.parse_args(values)


def write_framework_export(output: Path, checkpoint: Path, config: Path, base_model: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    (output / "model.safetensors").write_bytes(b"exported-weights")
    metadata = checkpoint / "model" / ".metadata"
    manifest = {
        "format": "cosmos-framework-vlm-dcp",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_metadata_sha256": common.sha256_file(metadata),
        "config": str(config.resolve()),
        "config_sha256": common.sha256_file(config),
        "base_model_path_or_uri": str(base_model.resolve()),
        "base_model_revision": "immutable-test-revision",
        "base_model_fingerprint": {
            "kind": "local",
            "source": str(base_model),
            "sha256": framework_action._base_model_fingerprint(base_model),
        },
        "tensor_count": 1,
        "lora": {"enabled": False},
        "merged_adapters": 0,
    }
    (output / "export_manifest.json").write_text(json.dumps(manifest))
    (output / "checkpoint.json").write_text(json.dumps({
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_type": "vlm_dcp",
    }))


def test_framework_action_plan_exports_dcp_and_preserves_runtime_paths(tmp_path):
    checkpoint, config, base_model = make_framework_dcp(tmp_path)
    supplied = str(checkpoint.parent / "." / checkpoint.name)
    args = framework_action_args(checkpoint, config, base_model)
    args.checkpoint_path = supplied
    plan = framework_action.build_plan(args)
    assert plan["checkpoint_kind"] == "framework_dcp"
    assert plan["checkpoint"]["original"] == supplied
    assert plan["checkpoint"]["resolved"] == str(checkpoint.resolve())
    assert plan["export_required"] is True
    assert plan["export_state"] == "missing"
    assert plan["action_model_path"].startswith(str(checkpoint.parent.parent / "hf_exports"))
    assert framework_action.EXPORTER_MODULE in plan["pre_action"]["argv"]
    assert plan["pre_action"]["argv"][plan["pre_action"]["argv"].index("--base-model-revision") + 1] == "immutable-test-revision"


def test_framework_action_verified_export_is_reused_and_stale_export_is_rejected(tmp_path):
    checkpoint, config, base_model = make_framework_dcp(tmp_path)
    export = tmp_path / "exports" / "epoch_1"
    write_framework_export(export, checkpoint, config, base_model)
    args = framework_action_args(checkpoint, config, base_model, export_dir=export)
    plan = framework_action.build_plan(args)
    assert plan["export_required"] is False
    assert plan["export_state"] == "verified_complete"
    verified = framework_action.verify_export(
        checkpoint_path=str(checkpoint), config_file=str(config), export_dir=str(export),
        base_model_path_or_uri=str(base_model), base_model_revision="immutable-test-revision",
    )
    assert verified["ok"]
    config.write_text(config.read_text() + "trainer: {}\n")
    stale = framework_action.build_plan(args)
    assert stale["export_required"] is True
    assert stale["export_state"] == "stale_or_incomplete"
    assert "fingerprint is stale" in stale["export_validation_error"]


def test_framework_prepare_runs_export_once_then_reuses_it(tmp_path, monkeypatch):
    checkpoint, config, base_model = make_framework_dcp(tmp_path)
    export = tmp_path / "exports" / "epoch_1"
    args = framework_action_args(checkpoint, config, base_model, verb="prepare", export_dir=export)
    calls = []

    def fake_run(command, check=False):
        calls.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        write_framework_export(output, checkpoint, config, base_model)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(framework_action.subprocess, "run", fake_run)
    first = framework_action.prepare_export(args)
    second = framework_action.prepare_export(args)
    assert first["pre_action_result"] == "exported"
    assert second["pre_action_result"] == "reused"
    assert len(calls) == 1
    assert (export / ".tao_export_complete").is_file()


def test_framework_action_model_uri_requires_immutable_revision(tmp_path):
    args = framework_action.parse_args([
        "plan", "--action", "inference", "--checkpoint-path", "vendor/model-name",
    ])
    with pytest.raises(common.WorkflowError, match="immutable revision"):
        framework_action.build_plan(args)


def test_framework_action_contract_is_packaged_and_dataset_agnostic():
    contract = workflow.load_yaml(workflow.BACKEND_FILES["cosmos-framework"])
    pre_action = contract["checkpoint"]["action_preparation"]
    assert pre_action["orchestrator"] == "scripts/framework_checkpoint_action.py"
    assert set(pre_action["applies_before"]) == {"evaluate", "inference", "inference_microservice"}
    assert contract["actions"]["evaluate"]["pre_action"] == "export_if_framework_dcp"
    assert contract["actions"]["inference"]["command"].startswith("cosmos-framework-inference")
    source = (SKILL / "scripts" / "framework_checkpoint_action.py").read_text(encoding="utf-8")
    assert "cosmos_framework.scripts.export_vlm_dcp" in source
    for forbidden in ("/lustre/", "rarunachalam", "wts", "aetc"):
        assert forbidden not in source.casefold()


def test_model_input_required_and_uri_revision_required(tmp_path):
    with pytest.raises(common.WorkflowError, match="required"):
        common.inspect_model("")
    with pytest.raises(common.WorkflowError, match="revision"):
        common.inspect_model("nvidia/Cosmos3-Nano")
    identity = common.inspect_model("nvidia/Cosmos3-Nano", "0123456789abcdef")
    assert identity["revision"] == "0123456789abcdef"


def test_indexed_model_weights_are_validated_and_fingerprinted(tmp_path):
    model = tmp_path / "indexed-model"
    weights = model / "weights"
    weights.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": "cosmos3_edge"}))
    (weights / "model-00001-of-00001.safetensors").write_bytes(b"edge-weights")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer.weight": "weights/model-00001-of-00001.safetensors"}}))
    inspected = common.inspect_model(str(model))
    assert "weights/model-00001-of-00001.safetensors" in {item["path"] for item in inspected["files"]}
    (weights / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(common.WorkflowError, match="missing weight file"):
        common.inspect_model(str(model))


def test_fast_model_fingerprint_does_not_read_weight_bytes(tmp_path, monkeypatch):
    model = make_model(tmp_path)
    weight = model / "model.safetensors"
    original_sha256_file = common.sha256_file

    def guarded_sha256_file(path):
        if Path(path) == weight:
            raise AssertionError("weight bytes were hashed")
        return original_sha256_file(path)

    monkeypatch.setattr(common, "sha256_file", guarded_sha256_file)
    inspected = common.inspect_model(str(model), fast_weight_fingerprint=True)
    weight_entry = next(item for item in inspected["files"] if item["path"] == weight.name)
    assert weight_entry == {"path": weight.name, "size": weight.stat().st_size}
    assert inspected["fingerprint_mode"] == "metadata_content_and_weight_sizes"


def test_runtime_paths_are_preserved_and_resolved(tmp_path):
    path = tmp_path / "somewhere"
    path.mkdir()
    supplied = str(tmp_path / "." / "somewhere")
    identity = common.path_identity(supplied)
    assert identity["original"] == supplied
    assert identity["resolved"] == str(path.resolve())


def test_video_conversation_framework_dense_spec_and_no_historical_paths(tmp_path):
    args = args_for(tmp_path)
    args.optimizer_epsilon = 1e-6
    plan = workflow.build_plan(args)
    workflow.write_spec(args, plan)
    assert plan["backend"] == "cosmos-framework"
    assert plan["training"]["training_mode"] == "dense"
    assert plan["spec"]["model"]["parallelism"]["data_parallel_shard_degree"] == 8
    assert plan["spec"]["trainer"]["grad_accum_iter"] == 1
    assert plan["training"]["per_forward_batch"] == 1
    assert plan["training"]["gradient_accumulation"] == 1
    assert plan["spec"]["trainer"]["max_iter"] == 2
    assert plan["training"]["optimizer_epsilon"] == 1e-6
    assert plan["spec"]["optimizer"]["eps"] == 1e-6
    assert "lora_enabled" not in plan["spec"]["model"]
    assert plan["decoder_artifact"]["required"] is False
    assert plan["decoder_artifact"]["enabled"] is False
    assert plan["framework_video_runtime"] == {
        "selected_profile": "torchcodec-cuda-on-demand",
        "selection_reason": "Framework video supervision uses its source-baked CUDA TorchCodec profile",
        "video_decoder": "torchcodec",
        "implementation": "torchcodec_cuda_indexed",
        "decoder_device": "cuda",
        "decoder_device_binding": "explicit_local_rank",
        "frame_transfer": "cuda_uint8_to_host_pil",
        "video_cache_size": 16,
        "video_cache_scope": "rank_local_decoded_pil_frames",
        "video_cache_population": "on_demand_during_training",
        "video_cache_persists_to_disk": False,
        "sft_process_threads": 8,
        "decoder_threads": 1,
        "dataloader_num_workers": 1,
        "dataloader_prefetch_factor": 2,
        "dataloader_multiprocessing_context": "spawn",
        "dataloader_persistent_workers": True,
        "unique_media_capacity_basis": 16,
        "dataset_prewarm": False,
        "actual_device_attestation": "first_successful_decode_per_rank",
        "actual_device_requirement": "resolved_device_equals_cuda_local_rank",
    }
    assert plan["environment"]["TAO_VIDEO_CACHE_SIZE"] == "16"
    assert plan["environment"]["TAO_FRAMEWORK_SFT_PROCESS_THREADS"] == "8"
    assert plan["environment"]["TAO_FRAMEWORK_DATALOADER_NUM_WORKERS"] == "1"
    assert plan["environment"]["TAO_FRAMEWORK_DATALOADER_PREFETCH_FACTOR"] == "2"
    assert plan["environment"]["TAO_VIDEO_DECODER_DEVICE"] == "cuda"
    assert plan["environment"]["TAO_VIDEO_DECODER_THREADS"] == "1"
    preflight = plan["preflight"]["container_runtime"]
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:contiguous_batcher_max_tokens" in preflight
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:contiguous_batcher_source_order" in preflight
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:cross_epoch_resume_cursor" in preflight
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:framework_spawn_prefetch" in preflight
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:framework_spawn_pickle" in preflight
    assert plan["datasets"]["train"]["annotations"][0]["original"] == args.train_annotation[0]
    source = Path(workflow.__file__).read_text(encoding="utf-8")
    assert "/lustre/" not in source and "rarunachalam" not in source
    with Path(args.write_spec).open("rb") as stream:
        assert tomllib.load(stream)["trainer"]["max_iter"] == 2


def test_framework_packed_forward_preserves_effective_global_batch(tmp_path):
    args = args_for(tmp_path)
    args.effective_global_batch = 64
    args.framework_per_forward_batch = 8

    plan = workflow.build_plan(args)

    assert plan["spec"]["dataloader_train"]["max_samples_per_batch"] == 8
    assert plan["spec"]["trainer"]["grad_accum_iter"] == 1
    assert plan["training"]["effective_global_batch"] == 64
    assert plan["training"]["per_forward_batch"] == 8
    assert plan["training"]["gradient_accumulation"] == 1


def test_cosmos_rl_peft_spec_defaults_to_direct_processing(tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl", training_mode="peft")
    args.optimizer_epsilon = 1e-6
    plan = workflow.build_plan(args)
    assert plan["training"]["optimizer_epsilon"] == 1e-6
    assert plan["spec"]["train"]["epsilon"] == 1e-6
    assert plan["spec"]["train"]["optm_impl"] == "foreach"
    lora = plan["spec"]["policy"]["lora"]
    assert lora == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "use_rslora": True,
        "modules_to_save": [],
        "adapter_dtype": "bfloat16",
    }
    native_contract = workflow.load_yaml(workflow.BACKEND_FILES["cosmos-rl"])
    schema = native_contract["configuration"]["peft_schema"]
    assert set(lora) == set(schema["fields"])
    assert not set(lora) & set(schema["forbidden_legacy_fields"])
    assert plan["cache_prewarm"]["mode"] == "direct"
    assert plan["cache_prewarm"]["required"] is False
    assert plan["spec"]["train"]["train_policy"]["enable_dataset_cache"] is False
    train_policy = plan["spec"]["train"]["train_policy"]
    assert train_policy["dataloader_drop_last"] is False
    for key in (
        "dataset_cache_dir",
        "dataset_cache_fingerprint",
        "validation_dataset_cache_fingerprint",
        "require_complete_dataset_cache",
    ):
        assert key not in train_policy
    assert "cache_dir" not in plan["spec"]["custom"]["vision"]
    assert "COSMOS_CACHE" not in plan["environment"]
    runtime = plan["rl_video_runtime"]
    assert runtime == {
        "requested_profile": "auto",
        "selected_profile": "pynv-device-rgbp",
        "selection_reason": "video_conversation defaults to the source-baked device-RGBP throughput profile",
        "video_decoder": "pynvvideocodec",
        "implementation": "pynv_device_rgbp_dlpack",
        "frame_transfer": "device_rgbp",
        "video_cache_size": 16,
        "video_cache_scope": "rank_local_processed_fetch_video_memory",
        "video_cache_population": "on_demand_during_training",
        "video_cache_persists_to_disk": False,
        "decoder_cache_size": 16,
        "decoder_cache_scope": "rank_local_pynv_native_sessions",
        "sft_batch_threads": 4,
        "dataloader_num_workers": 1,
        "dataloader_prefetch_factor": 2,
        "unique_media_capacity_basis": 16,
        "dataset_prewarm": False,
        "capability_fallback": "tao_system_pyav_sparse",
        "capability_fallback_scope": "nvdec_incompatible_stream_only",
    }
    assert plan["decoder_artifact"]["required"] is False
    assert plan["decoder_artifact"]["enabled"] is False
    assert plan["decoder_artifact"]["policy"]["force_all_validation_media"] is False
    assert plan["decoder_artifact"]["policy"]["selection_basis"] == (
        "optional_resolved_hardware_decoder_compatibility"
    )
    assert plan["decoder_artifact"]["preparation_arguments"].count(
        "--force-annotation"
    ) == 0
    assert plan["spec"]["custom"]["video_decoder"] == "pynvvideocodec"
    assert plan["spec"]["custom"]["video_cache_size"] == 16
    assert plan["spec"]["custom"]["video_decoder_cache_size"] == 16
    assert train_policy["dataloader_num_workers"] == 1
    assert train_policy["dataloader_prefetch_factor"] == 2
    assert plan["spec"]["validation"]["dataloader_num_workers"] == 1
    assert plan["spec"]["validation"]["dataloader_prefetch_factor"] == 2
    assert plan["environment"]["FORCE_QWENVL_VIDEO_READER"] == "pynvvideocodec"
    assert plan["environment"]["TAO_PYNV_FRAME_TRANSFER"] == "device_rgbp"
    assert plan["environment"]["TAO_SFT_BATCH_THREADS"] == "4"
    assert plan["environment"]["TAO_PYNV_VIDEO_CACHE_SIZE"] == "16"
    assert plan["environment"]["TAO_PYNV_DECODER_CACHE_SIZE"] == "16"
    preflight = plan["preflight"]["container_runtime"]
    assert "inspect.getsource" in preflight
    assert "TAO_PYNV_DECODER_CACHE_SIZE" in preflight
    assert "persistent_workers" in preflight
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:worker_decoder_cache_forwarding" in preflight
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:worker_pixel_bound_normalization" in preflight
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:processed_video_cache_binding" in preflight
    assert "packer_module.qwen_vl_process_vision_info" in preflight
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:persistent_workers" in preflight
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:pixel_bound_visibility" in preflight
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:pixel_bound_type" in preflight
    assert "pixel_probe=normalize_video_pixel_bounds" in preflight
    assert "TAO_PREFLIGHT_ASSERTION_FAILED:all_child_failures_propagate" in preflight


def test_cosmos_rl_dense_optimizer_is_dataset_independent(tmp_path):
    for dataset_family in ("video_conversation", "task_aware_video_reasoning"):
        args = args_for(
            tmp_path / dataset_family,
            backend="cosmos-rl",
            dataset_family=dataset_family,
            training_mode="dense",
        )
        plan = workflow.build_plan(args)
        assert plan["spec"]["train"]["optm_impl"] == "fused"


def test_cosmos_rl_system_pyav_profile_is_explicit_and_worker_zero_safe(tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl", training_mode="peft")
    args.rl_video_profile = "system-pyav"

    plan = workflow.build_plan(args)

    runtime = plan["rl_video_runtime"]
    assert runtime["selected_profile"] == "system-pyav"
    assert runtime["video_decoder"] == "torchvision"
    assert runtime["frame_transfer"] == "host_rgb"
    assert runtime["video_cache_size"] == 0
    assert runtime["decoder_cache_size"] == 1
    assert runtime["sft_batch_threads"] == 1
    assert runtime["dataloader_num_workers"] == 0
    assert runtime["dataloader_prefetch_factor"] is None
    assert plan["spec"]["custom"]["video_decoder"] == "torchvision"
    assert plan["environment"]["FORCE_QWENVL_VIDEO_READER"] == "torchvision"
    assert "TAO_PYNV_FRAME_TRANSFER" not in plan["environment"]
    assert "dataloader_prefetch_factor" not in plan["spec"]["train"]["train_policy"]
    assert "dataloader_prefetch_factor" not in plan["spec"]["validation"]


def test_cosmos_rl_explicit_zero_disables_processed_video_cache(tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl", training_mode="peft")
    args.rl_video_profile = "pynv-device-rgbp"
    args.rl_video_cache_size = 0
    args.rl_video_decoder_cache_size = 4

    plan = workflow.build_plan(args)

    runtime = plan["rl_video_runtime"]
    assert runtime["video_cache_size"] == 0
    assert runtime["decoder_cache_size"] == 4
    assert plan["spec"]["custom"]["video_cache_size"] == 0
    assert plan["spec"]["custom"]["video_decoder_cache_size"] == 4
    assert plan["environment"]["TAO_PYNV_VIDEO_CACHE_SIZE"] == "0"
    assert plan["environment"]["TAO_PYNV_DECODER_CACHE_SIZE"] == "4"


def test_cosmos_nano_video_pixel_budget_is_shared_by_framework_and_rl(tmp_path):
    framework_args = args_for(tmp_path / "framework", backend="cosmos-framework")
    framework_plan = workflow.build_plan(framework_args)

    rl_args = args_for(tmp_path / "rl", backend="cosmos-rl", training_mode="peft")
    rl_plan = workflow.build_plan(rl_args)

    assert framework_plan["processor_profile"]["max_video_pixels"] == 81920
    assert framework_plan["environment"]["TAO_VIDEO_MAX_PIXELS"] == "81920"
    assert rl_plan["processor_profile"]["max_video_pixels"] == 81920
    assert rl_plan["spec"]["custom"]["vision"]["max_pixels"] == 81920


def test_cosmos_rl_prewarm_remains_an_explicit_opt_in(tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl")
    args.rl_dataset_cache_mode = "prewarm"

    plan = workflow.build_plan(args)

    assert plan["cache_prewarm"]["mode"] == "prewarm"
    assert plan["cache_prewarm"]["required"] is True
    train_policy = plan["spec"]["train"]["train_policy"]
    assert train_policy["enable_dataset_cache"] is True
    assert train_policy["dataset_cache_dir"] == "/cache"
    assert (
        train_policy["dataset_cache_fingerprint"]
        == plan["cache_prewarm"]["keys"]["train"]
    )
    assert (
        train_policy["validation_dataset_cache_fingerprint"]
        == plan["cache_prewarm"]["keys"]["validation"]
    )
    assert train_policy["require_complete_dataset_cache"] is True
    assert plan["spec"]["custom"]["vision"]["cache_dir"] == "/cache"
    assert plan["environment"]["COSMOS_CACHE"] == "/cache"


def test_cosmos_rl_maps_common_constant_scheduler_to_native_none(tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl")
    args.scheduler = "constant"

    plan = workflow.build_plan(args)

    assert plan["training"]["scheduler"] == "constant"
    assert plan["spec"]["train"]["optm_decay_type"] == "none"

    args.minimum_lr_factor = 0.0
    plan = workflow.build_plan(args)
    assert plan["spec"]["train"]["optm_min_lr_factor"] == 0.0


def test_cosmos_rl_resolves_sft_hook_from_installed_native_package(tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl")
    attach_decoder_artifact(args, tmp_path)
    plan = workflow.build_plan(args)

    assert "Path(cosmos_rl.__file__).parent" in plan["command"]
    assert "tools/custom_hooks" not in plan["command"]  # assembled with pathlib
    assert "/opt/cosmos_rl/tao_sft_example.py" not in plan["command"]
    assert 'test -f "$hook"' in plan["command"]
    args.platform = "slurm"; args.partition = "compute"; args.account = "project"
    args.slurm_user = "user"; args.slurm_host = ["login.example"]
    args.stdout_path = str(tmp_path / "stdout.log"); args.stderr_path = str(tmp_path / "stderr.log")
    args.container_mount = [f"{tmp_path}:{tmp_path}"]
    script = workflow.render_slurm(args, plan)
    assert "--container-env=" in script
    assert "FORCE_QWENVL_VIDEO_READER" in script
    assert "TAO_PYNV_DECODER_CACHE_SIZE" in script
    child_argv = shlex.split(script.split("set +e\n", 1)[1].split("\nchild_rc=", 1)[0])
    nested = subprocess.run(
        ["bash", "-n", "-c", child_argv[-1]], capture_output=True, text=True
    )
    assert nested.returncode == 0, nested.stderr


def test_cosmos_rl_static_train_contract_resolves_installed_hook():
    metadata = workflow.load_yaml(SKILL / "references" / "skill_info.yaml")
    command = metadata["actions"]["train"]["command"]

    assert "Path(cosmos_rl.__file__).parent" in command
    assert 'test -f "$hook"' in command
    assert 'cosmos-rl --config {config_path} "$hook"' in command
    assert "package://" not in command


def test_task_aware_hybrid_expansion_has_paired_optimizer_updates(tmp_path):
    framework_args = args_for(
        tmp_path / "framework", backend="cosmos-framework",
        dataset_family="task_aware_video_reasoning",
    )
    rl_args = args_for(
        tmp_path / "rl", backend="cosmos-rl",
        dataset_family="task_aware_video_reasoning",
    )
    framework = workflow.build_plan(framework_args)
    rl = workflow.build_plan(rl_args)

    for plan in (framework, rl):
        assert plan["training"]["logical_train_records"] == 24
        assert plan["training"]["train_response_mode"] == "hybrid"
        assert plan["training"]["train_sample_multiplier"] == 2
        assert plan["training"]["exposed_train_samples"] == 48
        assert plan["training"]["optimizer_updates"] == 6
    assert framework["spec"]["trainer"]["max_iter"] == 6
    assert rl["cache_prewarm"]["required"] is False
    assert rl["spec"]["train"]["train_policy"]["enable_dataset_cache"] is False
    assert rl["spec"]["train"]["train_policy"]["dataloader_drop_last"] is False
    assert "enable_dataset_cache" not in rl["spec"]["validation"]
    assert "require_complete_dataset_cache" not in rl["spec"]["train"]["train_policy"]
    assert "cache_dir" not in rl["spec"]["custom"]["vision"]
    assert rl["spec"]["train"]["optm_impl"] == "fused"
    assert framework["decoder_artifact"]["required"] is False
    assert framework["decoder_artifact"]["enabled"] is False
    assert framework["decoder_artifact"]["policy"]["gpu_random_access_validation_required"] is False
    assert framework["decoder_artifact"]["policy"]["selection_basis"] == "framework_native_torchcodec_cuda_on_demand"
    assert rl["decoder_artifact"]["required"] is False
    assert rl["decoder_artifact"]["enabled"] is False
    assert rl["decoder_artifact"]["policy"]["gpu_random_access_validation_required"] is False
    assert rl["spec"]["custom"]["video_decoder"] == "torchvision"
    assert rl["environment"]["FORCE_QWENVL_VIDEO_READER"] == "torchvision"


def test_framework_rejects_cosmos_rl_decoder_artifact(tmp_path):
    args = args_for(
        tmp_path, backend="cosmos-framework",
        dataset_family="task_aware_video_reasoning",
    )
    args.video_override_map = str(tmp_path / "override-map.json")
    args.video_override_manifest = str(tmp_path / "override-manifest.json")
    args.video_override_fingerprint = "a" * 64
    args.video_override_force_video = [
        str(tmp_path / "train" / "media" / "train-bcq-0.mp4")
    ]

    with pytest.raises(common.WorkflowError, match="owned by the Cosmos-RL backend"):
        workflow.build_plan(args)


def test_decoder_artifact_requires_map_manifest_and_fingerprint(tmp_path):
    args = args_for(tmp_path)
    args.video_override_map = str(tmp_path / "override-map.json")

    with pytest.raises(common.WorkflowError, match="must be supplied together"):
        workflow.build_plan(args)


def test_framework_task_aware_slurm_uses_native_runtime_without_decoder_artifact(tmp_path):
    args = args_for(
        tmp_path, dataset_family="task_aware_video_reasoning"
    )
    args.platform = "slurm"
    args.partition = "compute"
    args.account = "project"
    args.container_mount = [f"{tmp_path}:{tmp_path}"]
    plan = workflow.build_plan(args)

    script = workflow.render_slurm(args, plan)
    assert "--container-image=" in script
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


def test_task_aware_constant_schedule_keeps_lr_factor_at_one(tmp_path):
    framework_args = args_for(
        tmp_path / "framework", backend="cosmos-framework",
        dataset_family="task_aware_video_reasoning",
    )
    rl_args = args_for(
        tmp_path / "rl", backend="cosmos-rl",
        dataset_family="task_aware_video_reasoning",
    )
    framework_args.scheduler = "constant"
    rl_args.scheduler = "constant"

    framework = workflow.build_plan(framework_args)
    rl = workflow.build_plan(rl_args)

    assert framework["spec"]["scheduler"]["f_min"] == [1.0]
    assert rl["spec"]["train"]["optm_decay_type"] == "none"
    assert rl["spec"]["train"]["optm_min_lr_factor"] == 1.0


def test_framework_warmup_epochs_translate_to_optimizer_steps(tmp_path):
    args = args_for(
        tmp_path,
        backend="cosmos-framework",
        dataset_family="task_aware_video_reasoning",
    )
    args.epochs = 3
    args.warmup = 1
    args.scheduler = "constant"

    plan = workflow.build_plan(args)

    assert plan["training"]["optimizer_updates"] == 18
    assert plan["spec"]["trainer"]["steps_per_epoch"] == 6
    assert plan["spec"]["scheduler"]["cycle_lengths"] == [18]
    assert plan["spec"]["scheduler"]["warm_up_steps"] == [6]
    assert plan["spec"]["scheduler"]["f_start"] == [0.0]
    assert plan["spec"]["scheduler"]["f_min"] == [1.0]


def test_materialization_text_result_is_not_misclassified_as_preflight():
    rendered = workflow._text(
        {
            "ok": True,
            "config": {
                "original": "/results/specs/train.toml",
                "resolved": "/results/specs/train.toml",
                "sha256": "a" * 64,
            },
            "generated_artifacts": [],
        }
    )

    assert rendered.startswith("Cosmos materialization: PASS")
    assert "config sha256" in rendered


def test_task_aware_smoke_limit_counts_logical_records_before_expansion(tmp_path):
    args = args_for(
        tmp_path, backend="cosmos-framework",
        dataset_family="task_aware_video_reasoning", run_mode="smoke",
    )
    plan = workflow.build_plan(args)
    assert plan["training"]["logical_train_records"] == 16
    assert plan["training"]["exposed_train_samples"] == 32
    assert plan["training"]["optimizer_updates"] == 4
    assert plan["environment"]["TAO_VIDEO_TRAIN_LIMIT"] == "32"
    assert plan["spec"]["trainer"]["max_iter"] == 4


def test_framework_peft_spec_is_native_not_rl_schema(tmp_path):
    args = args_for(tmp_path, training_mode="peft")
    plan = workflow.build_plan(args)
    assert plan["spec"]["model"]["lora_enabled"] is True
    assert plan["spec"]["model"]["lora_target_modules"] == "q_proj,v_proj"
    assert plan["spec"]["optimizer"]["keys_to_select"] == ["lora_"]
    assert "policy" not in plan["spec"]


def test_task_aware_paths_tasks_and_accuracy_coverage(tmp_path):
    args = args_for(tmp_path, dataset_family="task_aware_video_reasoning")
    plan = workflow.build_plan(args)
    assert plan["datasets"]["train"]["tasks"] == {"bcq": 8, "mcq": 8, "scene_description": 8}
    coverage = plan["datasets"]["validation"]["metric_coverage"]
    assert coverage["accuracy_tasks"] == ["bcq", "mcq"]
    assert coverage["excluded_tasks"] == ["scene_description"]
    assert json.loads(plan["environment"]["TAO_VIDEO_TRAIN_ANNOTATIONS"]) == args.train_annotation
    assert plan["spec"]["job"]["experiment"] == "tao_task_aware_video_reasoning"
    args_rl = args_for(tmp_path / "rl", dataset_family="task_aware_video_reasoning", backend="cosmos-rl")
    plan_rl = workflow.build_plan(args_rl)
    assert "tao_vl_reason_daft_sft_example.py" in plan_rl["command"]


def test_task_aware_question_answer_schema_is_supported_and_validated(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.mp4").write_bytes(b"video")
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps({
        "format": "tao-vl-reason-v1.0",
        "metadata": {"task": "bcq"},
        "items": [{
            "video_id": "clip.mp4",
            "question": "Did an event occur?",
            "answer": "Yes",
            "reasoning": "The event is visible.",
        }],
    }))
    inspected = common.inspect_dataset(
        dataset_family="auto", annotations=[str(annotation)], media_roots=[str(media)]
    )
    assert inspected["dataset_family"] == "task_aware_video_reasoning"
    assert inspected["tasks"] == {"bcq": 1}

    payload = json.loads(annotation.read_text())
    del payload["items"][0]["answer"]
    annotation.write_text(json.dumps(payload))
    with pytest.raises(common.WorkflowError, match="question/answer"):
        common.inspect_dataset(
            dataset_family="auto", annotations=[str(annotation)], media_roots=[str(media)]
        )


def test_streamable_input_inspector_preserves_paths_and_planned_outputs(tmp_path):
    model = make_model(tmp_path)
    train_annotation, train_media = make_video_conversation(tmp_path, "train")
    val_annotation, val_media = make_video_conversation(tmp_path, "validation")
    planned = tmp_path / "new-results" / "job"
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL / "scripts" / "cosmos_common.py"),
            "inspect-inputs",
            "--base-model-path-or-uri", str(model),
            "--dataset-family", "auto",
            "--train-annotation", str(train_annotation),
            "--train-media-root", str(train_media),
            "--validation-annotation", str(val_annotation),
            "--validation-media-root", str(val_media),
            "--runtime-path", f"results_dir={planned}",
            "--fast-media-fingerprint",
            "--fast-model-fingerprint",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["frame"] == "target_compute"
    assert payload["model"]["supplied"]["original"] == str(model)
    assert payload["model"]["fingerprint_mode"] == "metadata_content_and_weight_sizes"
    assert payload["runtime_paths"]["results_dir"]["original"] == str(planned)
    assert payload["runtime_paths"]["results_dir"]["exists"] is False
    assert payload["runtime_paths"]["results_dir"]["parent_writable"] is True


def test_materialize_dataset_filters_tasks_limits_and_never_overwrites_source(tmp_path):
    annotations, _ = make_task_aware_video(tmp_path, "train")
    output = tmp_path / "generated" / "smoke.json"
    result = common.materialize_dataset(
        dataset_family="task_aware_video_reasoning",
        annotations=[str(path) for path in annotations],
        output_path=str(output),
        selected_tasks=["mcq"],
        sample_limit=3,
    )
    payload = json.loads(output.read_text())
    assert result["record_count"] == 3
    assert result["sample_limit"] == 3
    assert {item["task"] for item in payload["items"]} == {"mcq"}
    assert result["sha256"] == common.sha256_file(output)
    with pytest.raises(common.WorkflowError, match="must not overwrite"):
        common.materialize_dataset(
            dataset_family="auto",
            annotations=[str(annotations[0])],
            output_path=str(annotations[0]),
        )


@pytest.mark.parametrize("dataset_family,experiment", [("video_conversation", "tao_video_conversation_edge"), ("task_aware_video_reasoning", "tao_task_aware_video_reasoning_edge")])
def test_public_edge_checkpoint_uses_skill_runtime_profile(tmp_path, dataset_family, experiment):
    args = args_for(tmp_path, dataset_family=dataset_family, model_name="nvidia/Cosmos3-Edge")
    plan = workflow.build_plan(args)

    assert plan["backend"] == "cosmos-framework"
    assert plan["model_preparation"]["required"] is False
    assert "no processor overlay" in plan["model_preparation"]["reason"]
    assert plan["prepared_model_container_path"] == str((tmp_path / "model").resolve())
    assert plan["spec"]["job"]["experiment"] == experiment
    assert plan["processor_profile"] == {
        "model_tier": "edge",
        "source": "dataset_metadata" if dataset_family == "video_conversation" else "model_safe_default",
        "frames": 6,
        "sequence_length": 16000,
        "attention_implementation": "flash_attention_2",
        "frame_width": 960 if dataset_family == "video_conversation" else 1280,
        "frame_height": 540 if dataset_family == "video_conversation" else 720,
        "max_video_pixels": 3110400 if dataset_family == "video_conversation" else 5529600,
        "checkpoint_mutation": False,
        "dataset_profile_fingerprints": plan["processor_profile"]["dataset_profile_fingerprints"],
        "selection_basis": ["model_tier", "dataset_resolution_metadata", "record_count", "media_reuse", "explicit_overrides"],
    }
    assert plan["environment"]["TAO_VIDEO_MAX_PIXELS"] == str(plan["processor_profile"]["max_video_pixels"])


def test_public_edge_uri_is_snapshotted_without_alternate_checkpoint(tmp_path):
    args = args_for(tmp_path, model_name="nvidia/Cosmos3-Edge")
    args.base_model_path_or_uri = "nvidia/Cosmos3-Edge"
    args.base_model_revision = "0123456789abcdef"
    plan = workflow.build_plan(args)

    assert plan["model_preparation"]["kind"] == "immutable_public_checkpoint_snapshot"
    assert plan["model_preparation"]["required"] is True
    assert "processor overlay" not in plan["model_preparation"]["command"]
    assert plan["processor_profile"]["checkpoint_mutation"] is False


def test_model_tier_is_inferred_from_public_checkpoint_identity(tmp_path):
    args = args_for(tmp_path, model_name="nvidia/Cosmos3-Edge")
    args.model = "auto"
    args.base_model_path_or_uri = "nvidia/Cosmos3-Edge"
    args.base_model_revision = "0123456789abcdef"
    plan = workflow.build_plan(args)
    assert plan["model_name"] == "nvidia/Cosmos3-Edge"
    assert plan["backend"] == "cosmos-framework"


def test_edge_profile_explicit_override_is_recorded(tmp_path):
    args = args_for(tmp_path, model_name="nvidia/Cosmos3-Edge")
    args.frames = 4
    args.video_max_pixels = 3686400
    args.sequence_length = 12000
    plan = workflow.build_plan(args)
    assert plan["processor_profile"]["source"] == "user"
    assert plan["processor_profile"]["frames"] == 4
    assert plan["processor_profile"]["max_video_pixels"] == 3686400
    assert plan["training"]["sequence_length"] == 12000


def test_dataset_overlap_and_missing_media_fail(tmp_path):
    annotation, media = make_video_conversation(tmp_path, "same")
    inspected = common.inspect_dataset(dataset_family="auto", annotations=[str(annotation)], media_roots=[str(media)])
    with pytest.raises(common.WorkflowError, match="overlap"):
        common.assert_no_overlap(inspected, inspected)
    records = json.loads(annotation.read_text()); (media / records[0]["video"]).unlink()
    with pytest.raises(common.WorkflowError, match="missing"):
        common.inspect_dataset(dataset_family="auto", annotations=[str(annotation)], media_roots=[str(media)])


def test_customer_dataset_family_and_profile_are_inferred_from_structure(tmp_path):
    annotation, media = make_video_conversation(tmp_path / "customer-project", "split-alpha")
    inspected = common.inspect_dataset(
        dataset_family="auto", annotations=[str(annotation)], media_roots=[str(media)]
    )
    assert inspected["dataset_family"] == "video_conversation"
    assert inspected["profile"]["quantity_class"] == "small"
    assert inspected["profile"]["resolution"]["class"] == "up_to_720p"
    assert inspected["profile"]["resolution"]["median_width"] == 960
    assert inspected["profile"]["video"]["median_duration_seconds"] == 12
    assert inspected["metric_coverage"]["accuracy_tasks"] == ["default"]
    assert inspected["metric_coverage"]["task_metrics"] == {"default": "accuracy"}
    assert inspected["metric_coverage"]["inferred_metrics"] == {
        "default": "all conversation targets are deterministic classification labels"
    }


def test_free_form_video_conversation_does_not_invent_accuracy(tmp_path):
    annotation, media = make_video_conversation(tmp_path, "free-form")
    records = json.loads(annotation.read_text())
    for record in records:
        record["conversations"][-1]["value"] = "A detailed description of the road scene."
    annotation.write_text(json.dumps(records))

    inspected = common.inspect_dataset(
        dataset_family="auto", annotations=[str(annotation)], media_roots=[str(media)]
    )
    assert inspected["metric_coverage"]["accuracy_tasks"] == []
    assert inspected["metric_coverage"]["excluded_tasks"] == ["default"]
    assert inspected["metric_coverage"]["inferred_metrics"] == {}


def test_arbitrary_task_uses_declared_metric_instead_of_dataset_name(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.mp4").write_bytes(b"video")
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps({
        "metadata": {"task": "customer_hazard_decision", "metric": "accuracy"},
        "items": [{
            "id": "item-1", "video": "clip.mp4",
            "conversations": [{"from": "human", "value": "question"}, {"from": "gpt", "value": "safe"}],
        }],
    }))
    inspected = common.inspect_dataset(
        dataset_family="auto", annotations=[str(annotation)], media_roots=[str(media)]
    )
    assert inspected["dataset_family"] == "task_aware_video_reasoning"
    assert inspected["metric_coverage"]["accuracy_tasks"] == ["customer_hazard_decision"]
    assert inspected["metric_coverage"]["task_metrics"] == {"customer_hazard_decision": "accuracy"}


def test_smoke_limit_never_leaks_to_full(tmp_path):
    args = args_for(tmp_path, run_mode="full")
    args.train_sample_limit = 4
    with pytest.raises(common.WorkflowError, match="full runs"):
        workflow.build_plan(args)
    args = args_for(tmp_path / "smoke", run_mode="smoke")
    plan = workflow.build_plan(args)
    assert plan["training"]["epochs"] == 1
    assert plan["spec"]["trainer"]["max_iter"] == 2
    full = workflow.build_plan(args_for(tmp_path / "full-again", run_mode="full"))
    assert not any(key.endswith("_LIMIT") for key in full["environment"])


def test_slurm_script_is_bash_sqsh_no_requeue_and_preserves_failure(tmp_path):
    args = args_for(tmp_path)
    args.platform = "slurm"; args.partition = "compute"; args.account = "project"
    args.slurm_user = "user"; args.slurm_host = ["login.example"]
    args.stdout_path = str(tmp_path / "stdout.log"); args.stderr_path = str(tmp_path / "stderr.log")
    args.container_mount = [f"{tmp_path}:{tmp_path}"]
    args.timeout = "03:48:00"
    plan = workflow.build_plan(args); workflow.write_spec(args, plan)
    assert "--no-container-remap-root" in plan["preflight"]["container_runtime"]
    assert "--no-container-mount-home" in plan["preflight"]["container_runtime"]
    script = workflow.render_slurm(args, plan)
    assert script.startswith("#!/usr/bin/env bash\n#SBATCH --job-name=")
    assert f"#SBATCH --job-name={args.tao_job_id}" in script
    assert script.index("#SBATCH --account=") < script.index("set -Eeuo pipefail")
    assert "#SBATCH --no-requeue" in script and "--container-image=" in script
    assert "--no-container-remap-root" in script
    assert "--no-container-mount-home" in script
    assert 'export HOME="/tmp/tao-${TAO_JOB_ID:?TAO_JOB_ID must be set}-${SLURM_PROCID:-0}"' in script
    assert 'mkdir -p -m 700 "$HOME"' in script
    assert "timeout --signal=TERM --kill-after=30s 13680s srun" in script
    assert "TAO_COSMOS_PACKAGED_RUNTIME_STARTUP_OK" in script
    assert plan["preflight"]["container_runtime"] not in script
    assert script.count("--container-image=") == 1
    assert "export SLURM_EXPORT_ENV=ALL" in script
    assert "--container-env=" in script
    assert "TAO_STATUS_FILE" in script
    assert 'exit "$child_rc"' in script
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    child_argv = shlex.split(script.split("set +e\n", 1)[1].split("\nchild_rc=", 1)[0])
    assert child_argv[-2] == "-lc"
    assert plan["preflight"]["container_startup"] in child_argv[-1]
    assert subprocess.run(
        ["bash", "-n", "-c", child_argv[-1]], capture_output=True, text=True
    ).returncode == 0
    # Controlled child failure uses the same capture idiom as the generated job.
    result = subprocess.run(["bash", "-c", "set -Eeuo pipefail; rc=0; set +e; bash -c 'exit 17'; rc=$?; set -e; exit $rc"])
    assert result.returncode == 17
    assert subprocess.run(["sh", "-n"], input=script, text=True).returncode != 0 or "#!/usr/bin/env bash" in script


def test_single_node_exclusive_slurm_step_uses_allocated_cpus(tmp_path):
    args = args_for(tmp_path)
    args.platform = "slurm"; args.partition = "compute"; args.account = "project"
    args.slurm_user = "user"; args.slurm_host = ["login.example"]
    args.stdout_path = str(tmp_path / "stdout.log"); args.stderr_path = str(tmp_path / "stderr.log")
    args.container_mount = [f"{tmp_path}:{tmp_path}"]
    args.cpus_per_task = 16
    args.exclusive = True
    plan = workflow.build_plan(args); workflow.write_spec(args, plan)

    script = workflow.render_slurm(args, plan)

    assert "#SBATCH --cpus-per-task=16" in script
    assert 'slurm_job_record="$(scontrol show job -o "${SLURM_JOB_ID:?SLURM_JOB_ID must be set}")"' in script
    assert 'step_cpus_per_task="${BASH_REMATCH[1]}"' in script
    assert 'policy=allocated-exclusive-single-node' in script
    assert '--cpus-per-task="$step_cpus_per_task"' in script
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


def test_slurm_script_rejects_invalid_child_timeout(tmp_path):
    args = args_for(tmp_path)
    args.platform = "slurm"; args.partition = "compute"; args.account = "project"
    args.slurm_user = "user"; args.slurm_host = ["login.example"]
    args.stdout_path = str(tmp_path / "stdout.log"); args.stderr_path = str(tmp_path / "stderr.log")
    args.container_mount = [f"{tmp_path}:{tmp_path}"]
    plan = workflow.build_plan(args); workflow.write_spec(args, plan)
    args.timeout = "03:99:00"
    with pytest.raises(common.WorkflowError, match="child timeout"):
        workflow.render_slurm(args, plan)


def test_framework_spec_uses_only_current_strict_sft_schema_keys(tmp_path):
    args = args_for(tmp_path)
    plan = workflow.build_plan(args)
    assert "keys_to_exclude" not in plan["spec"]["optimizer"]
    assert plan["spec"]["checkpoint"]["dcp_async_mode_enabled"] is False

    args.async_checkpoint = True
    plan = workflow.build_plan(args)
    assert plan["spec"]["checkpoint"]["dcp_async_mode_enabled"] is True


def test_rl_sft_batch_is_per_dp_worker_and_multinode_launchers_are_packaged(tmp_path):
    args = args_for(tmp_path)
    args.backend = "cosmos-rl"
    args.rl_mini_batch = 1
    args.effective_global_batch = 8
    plan = workflow.build_plan(args)
    assert plan["spec"]["train"]["train_batch_per_replica"] == 1

    args.rl_train_batch_per_replica = 8
    args.rl_dataloader_num_workers = 1
    args.rl_dataloader_prefetch_factor = 1
    plan = workflow.build_plan(args)
    assert plan["spec"]["train"]["train_batch_per_replica"] == 8
    assert plan["spec"]["train"]["train_policy"]["mini_batch"] == 1
    assert plan["spec"]["train"]["train_policy"]["dataloader_num_workers"] == 1
    assert plan["spec"]["train"]["train_policy"]["dataloader_drop_last"] is False
    assert plan["spec"]["train"]["train_policy"]["dataloader_prefetch_factor"] == 1
    assert plan["spec"]["validation"]["dataloader_num_workers"] == 1
    assert plan["spec"]["validation"]["dataloader_prefetch_factor"] == 1
    assert plan["spec"]["validation"]["freq_in_epoch"] == 1

    args.rl_validation_freq_steps = 54
    plan = workflow.build_plan(args)
    assert plan["spec"]["validation"]["freq"] == 54

    args.nodes = 2
    args.effective_global_batch = 16
    plan = workflow.build_plan(args)
    assert "Path(cosmos_rl.__file__).parent" in plan["command"]
    assert 'bash "$launcher_dir/launch_controller.sh"' in plan["command"]
    assert 'bash "$launcher_dir/launch_replica.sh"' in plan["command"]


def test_framework_expands_one_shared_media_root_per_annotation(tmp_path):
    args = args_for(tmp_path)
    environment = workflow._env(
        args,
        "cosmos-framework",
        "/model",
        ["/train-a.json", "/train-b.json"],
        ["/train-media"],
        ["/val-a.json", "/val-b.json"],
        ["/val-media"],
        framework_video_runtime={
            "video_cache_size": 8,
            "sft_process_threads": 4,
            "decoder_device": "cuda",
            "decoder_threads": 1,
        },
    )
    assert len(json.loads(environment["TAO_VIDEO_TRAIN_MEDIA_ROOTS"])) == 2
    assert len(json.loads(environment["TAO_VIDEO_VAL_MEDIA_ROOTS"])) == 2
    assert environment["IMAGINAIRE_OUTPUT_ROOT"] == args.container_checkpoint_dir
    assert environment["TAO_RESULTS_ROOT"] == args.container_results_dir


def test_requeue_rejected(tmp_path):
    args = args_for(tmp_path); args.platform = "slurm"; args.partition = "p"; args.account = "a"; args.use_requeue = True
    args.container_mount = [f"{tmp_path}:{tmp_path}"]
    plan = workflow.build_plan(args); workflow.write_spec(args, plan)
    with pytest.raises(common.WorkflowError, match="requeue"):
        workflow.render_slurm(args, plan)


def test_image_provenance_source_equivalence_and_dirty_rejected():
    expected = {"cosmos-framework": "a" * 40, "cosmos-rl": "b" * 40}
    trees = {"cosmos-framework": "c" * 40, "cosmos-rl": "d" * 40}
    common.validate_provenance({"repositories": {name: {"commit": commit, "tree": trees[name], "dirty": False} for name, commit in expected.items()}}, expected, trees)
    with pytest.raises(common.WorkflowError, match="source mismatch"):
        common.validate_provenance({"repositories": {"cosmos-framework": {"commit": "c" * 40}}}, {"cosmos-framework": "a" * 40})
    with pytest.raises(common.WorkflowError, match="dirty"):
        common.validate_provenance({"repositories": {"cosmos-framework": {"commit": "a" * 40, "dirty": True}}}, {"cosmos-framework": "a" * 40})
    with pytest.raises(common.WorkflowError, match="tree mismatch"):
        common.validate_provenance({"repositories": {"cosmos-framework": {"commit": "a" * 40, "tree": "x", "dirty": False}}}, {"cosmos-framework": "a" * 40}, {"cosmos-framework": "y"})


def test_clean_build_plan_requires_new_sqsh_and_provenance(tmp_path):
    plan = workflow.build_plan(args_for(tmp_path))
    assert plan["image"]["dockerfile"] == "Dockerfile"
    assert plan["image"]["build_arguments"]["COSMOS_BACKEND"] == "cosmos-framework"
    assert plan["image"]["build_arguments"]["COSMOS_FRAMEWORK_BASE_IMAGE"] == "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04"
    assert plan["image"]["build_arguments"]["COSMOS_FRAMEWORK_REPO"] == "https://github.com/example/cosmos-framework.git"
    assert plan["image"]["build_arguments"]["COSMOS_FRAMEWORK_BRANCH"] == "dev/test-framework"
    assert plan["image"]["build_arguments"]["EXPECTED_FRAMEWORK_COMMIT"] == "f" * 40
    assert plan["image"]["build_arguments"]["EXPECTED_FRAMEWORK_TREE"] == "n" * 40
    assert len(plan["image"]["clean_build_commands"]) == 1
    assert plan["image"]["must_rebuild_after_source_change"] is True
    assert plan["image"]["sqsh"]["reuse_allowed"] is False
    assert plan["image"]["provenance_path"] == "/opt/tao/image-provenance.json"
    assert plan["image"]["required_commits"]["cosmos-framework"] == "f" * 40

    rl_plan = workflow.build_plan(args_for(tmp_path / "rl", backend="cosmos-rl"))
    assert rl_plan["image"]["dockerfile"] == "Dockerfile"
    assert rl_plan["image"]["build_arguments"]["COSMOS_BACKEND"] == "cosmos-rl"
    assert rl_plan["image"]["build_arguments"]["COSMOS_RL_GITHUB_REPO"] == "ssh://git@gitlab.example.com:12051/group/cosmos-reason"
    assert rl_plan["image"]["build_arguments"]["COSMOS_RL_GITHUB_BRANCH"] == "feature/enhanced-hooks-and-custom-loggers"
    assert rl_plan["image"]["build_arguments"]["COSMOS_RL_COMMIT"] == "r" * 40
    assert rl_plan["image"]["build_arguments"]["COSMOS_RL_TREE"] == "n" * 40
    assert "USE_LOCAL_COSMOS_RL_GITHUB" not in rl_plan["image"]["build_arguments"]
    assert "--ssh" in rl_plan["image"]["clean_build_commands"][0]
    assert rl_plan["image"]["build_arguments"]["PYAV_WHEEL_SHA256"] == "f9a65d1f48b818323fb411e80358f89d77dec340b01d27c6b2dfbb9cbf4b779f"


def test_slurm_preflight_refreshes_sqsh_existence_without_replanning(monkeypatch, tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl")
    plan = workflow.build_plan(args)
    args.platform = "slurm"
    args.partition = "polar3,polar4"
    args.account = "account"
    args.slurm_user = "user"
    args.slurm_host = ["login"]
    args.container_mount = [f"{tmp_path}:{tmp_path}"]
    plan["input_frame"] = {
        "kind": "slurm_remote",
        "verified_host": "login",
    }
    plan["sqsh"] = {"exists": False, "kind": "missing"}
    monkeypatch.setattr(workflow, "_remote_file_exists", lambda *_args, **_kwargs: True)
    result = workflow.local_preflight(args, plan)
    assert not any("new SQSH" in error for error in result["errors"])
    assert not any("repository" in error for error in result["errors"])


def test_remote_sqsh_existence_uses_portable_test_syntax(monkeypatch, tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl")
    captured = {}

    def fake_ssh(_args, host, remote_command):
        captured.update(host=host, remote_command=remote_command)
        return ["ssh", host, remote_command]

    monkeypatch.setattr(workflow, "_ssh_command", fake_ssh)
    monkeypatch.setattr(
        workflow.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    assert workflow._remote_file_exists(args, path="/shared/image.sqsh", host="login")
    assert captured == {
        "host": "login",
        "remote_command": "test -f /shared/image.sqsh",
    }


def test_container_mount_translation_preserves_original_paths(tmp_path):
    args = args_for(tmp_path)
    args.platform = "slurm"
    args.container_mount = [f"{tmp_path}:/runtime"]
    args.partition = "p"; args.account = "a"; args.slurm_user = "u"; args.slurm_host = ["h"]
    plan = workflow.build_plan(args)
    assert plan["datasets"]["train"]["annotations"][0]["original"] == args.train_annotation[0]
    assert plan["environment"]["TAO_VIDEO_TRAIN_ANNOTATION"].startswith("/runtime/")
    assert plan["prepared_model_container_path"].startswith("/runtime/")
    assert plan["config_container_path"] == "/runtime/spec.toml"
    assert args.container_results_dir == "/runtime/results"
    assert args.container_checkpoint_dir == "/runtime/checkpoints"
    assert args.container_cache_dir == "/runtime/cache"


def test_remote_slurm_plan_streams_inspection_without_local_lustre(monkeypatch, tmp_path):
    local_args = args_for(tmp_path / "fixtures")
    local_plan = workflow.build_plan(local_args)
    local_prefix = str((tmp_path / "fixtures").resolve())
    remote_prefix = "/cluster/runtime"

    def remotize(value):
        return json.loads(json.dumps(value).replace(local_prefix, remote_prefix))

    remote_runtime_paths = {}
    for label in ("results_dir", "checkpoint_dir", "cache_dir", "sqsh_cache_dir", "sqsh_path"):
        path = f"{remote_prefix}/{label}"
        remote_runtime_paths[label] = {
            "original": path,
            "expanded": path,
            "resolved": None,
            "exists": False,
            "kind": "missing",
            "nearest_existing_parent": remote_prefix,
            "parent_writable": True,
        }
    inspection = {
        "schema_version": 1,
        "frame": "target_compute",
        "verified_host": "login.example",
        "model": remotize(local_plan["model"]),
        "datasets": remotize(local_plan["datasets"]),
        "runtime_paths": remote_runtime_paths,
    }

    args = args_for(tmp_path / "request")
    args.platform = "slurm"
    args.slurm_user = "user"
    args.slurm_host = ["login.example"]
    args.partition = "compute"
    args.account = "project"
    args.container_mount = [f"{remote_prefix}:{remote_prefix}"]
    args.base_model_path_or_uri = f"{remote_prefix}/model"
    args.train_annotation = [f"{remote_prefix}/train/manifest.json"]
    args.train_media_root = [f"{remote_prefix}/train/media"]
    args.validation_annotation = [f"{remote_prefix}/validation/manifest.json"]
    args.validation_media_root = [f"{remote_prefix}/validation/media"]
    args.results_dir = remote_runtime_paths["results_dir"]["original"]
    args.checkpoint_dir = remote_runtime_paths["checkpoint_dir"]["original"]
    args.cache_dir = remote_runtime_paths["cache_dir"]["original"]
    args.sqsh_cache_dir = remote_runtime_paths["sqsh_cache_dir"]["original"]
    args.sqsh_path = remote_runtime_paths["sqsh_path"]["original"] + ".sqsh"
    args.write_spec = f"{remote_prefix}/generated/spec.toml"
    inspection["runtime_paths"]["sqsh_path"]["original"] = args.sqsh_path
    inspection["runtime_paths"]["sqsh_path"]["expanded"] = args.sqsh_path
    monkeypatch.setattr(workflow, "_remote_inspection", lambda _args: inspection)

    plan = workflow.build_plan(args)
    assert plan["input_frame"] == {
        "kind": "slurm_remote",
        "verified_host": "login.example",
        "inspection_transport": "repository_helper_streamed_over_ssh",
    }
    assert plan["model"]["supplied"]["original"] == f"{remote_prefix}/model"
    assert plan["paths"]["results_dir"]["original"] == args.results_dir
    assert plan["preflight"]["submission_host"] == "command -v ssh >/dev/null"
    workflow.write_spec(args, plan)
    assert plan["config"]["materialized"] is False
    assert plan["config"]["resolved"] == args.write_spec
    assert plan["config"]["container"] == args.write_spec


def test_remote_slurm_materializes_config_and_rl_smoke_manifests(monkeypatch, tmp_path):
    local_args = args_for(tmp_path / "fixtures", backend="cosmos-rl", run_mode="smoke")
    local_plan = workflow.build_plan(local_args)
    local_prefix = str((tmp_path / "fixtures").resolve())
    remote_prefix = "/cluster/runtime"

    def remotize(value):
        return json.loads(json.dumps(value).replace(local_prefix, remote_prefix))

    args = args_for(tmp_path / "request", backend="cosmos-rl", run_mode="smoke")
    args.platform = "slurm"
    args.slurm_user = "user"
    args.slurm_host = ["login.example"]
    args.partition = "compute"
    args.account = "project"
    args.container_mount = [f"{remote_prefix}:{remote_prefix}"]
    args.base_model_path_or_uri = f"{remote_prefix}/model"
    args.train_annotation = [f"{remote_prefix}/train/manifest.json"]
    args.train_media_root = [f"{remote_prefix}/train/media"]
    args.validation_annotation = [f"{remote_prefix}/validation/manifest.json"]
    args.validation_media_root = [f"{remote_prefix}/validation/media"]
    args.results_dir = f"{remote_prefix}/results"
    args.checkpoint_dir = f"{remote_prefix}/checkpoints"
    args.cache_dir = f"{remote_prefix}/cache"
    args.sqsh_cache_dir = f"{remote_prefix}/sqsh-cache"
    args.sqsh_path = f"{remote_prefix}/sqsh-cache/image.sqsh"
    args.write_spec = f"{remote_prefix}/generated/spec.toml"
    runtime_paths = {
        label: {
            "original": value,
            "expanded": value,
            "resolved": None,
            "exists": False,
            "kind": "missing",
            "nearest_existing_parent": remote_prefix,
            "parent_writable": True,
        }
        for label, value in {
            "results_dir": args.results_dir,
            "checkpoint_dir": args.checkpoint_dir,
            "cache_dir": args.cache_dir,
            "sqsh_cache_dir": args.sqsh_cache_dir,
            "sqsh_path": args.sqsh_path,
        }.items()
    }
    inspection = {
        "schema_version": 1,
        "frame": "target_compute",
        "verified_host": "login.example",
        "model": remotize(local_plan["model"]),
        "datasets": remotize(local_plan["datasets"]),
        "runtime_paths": runtime_paths,
    }
    monkeypatch.setattr(workflow, "_remote_inspection", lambda _args: inspection)
    generated: list[tuple[str, str, int]] = []

    def fake_materialize(_args, *, split, output_path, sample_limit, host):
        generated.append((split, output_path, sample_limit))
        return {"sha256": split[0] * 64, "record_count": sample_limit}

    written = {}

    def fake_write(_args, *, output_path, content, host):
        written.update({"output": output_path, "content": content, "host": host})
        return hashlib.sha256(content.encode()).hexdigest()

    monkeypatch.setattr(workflow, "_remote_materialize_dataset", fake_materialize)
    monkeypatch.setattr(workflow, "_remote_write_text", fake_write)
    plan = workflow.build_plan(args)
    workflow.write_spec(args, plan, allow_remote_write=True)

    assert generated == [
        ("train", f"{remote_prefix}/generated/train_smoke.json", 16),
        ("validation", f"{remote_prefix}/generated/validation_smoke.json", 8),
    ]
    assert written["output"] == args.write_spec
    assert written["host"] == "login.example"
    assert plan["config"]["materialized"] is True
    assert all(item["materialized"] for item in plan["generated_artifacts"])
    assert plan["spec"]["custom"]["train_dataset"]["annotation_path"].endswith("train_smoke.json")


def test_sealed_plan_artifact_is_reused_without_reinspection(monkeypatch, tmp_path, capsys):
    args = args_for(tmp_path / "request", backend="cosmos-rl")
    plan = workflow.build_plan(args)
    workflow.write_spec(args, plan)
    metadata = workflow.initial_metadata(args, plan)
    workflow.validate_metadata(metadata)
    plan["initial_metadata"] = metadata
    artifact = tmp_path / "approved-plan.json"
    workflow.save_plan_artifact(args, plan, str(artifact))

    current = workflow.parse_args(["materialize", "--plan-artifact", str(artifact)])
    restored_args, restored_plan = workflow.load_plan_artifact(current, str(artifact))
    assert restored_args.model == "nvidia/Cosmos3-Nano"
    assert restored_args.dataset_family == "video_conversation"
    assert restored_args.write_spec == args.write_spec
    assert restored_plan["datasets"] == plan["datasets"]

    def unexpected_reinspection(_args):
        raise AssertionError("build_plan must not run after the approved plan is sealed")

    monkeypatch.setattr(workflow, "build_plan", unexpected_reinspection)
    assert workflow.main(["materialize", "--plan-artifact", str(artifact)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"]
    assert result["approved_plan"]["sha256"]


def test_sealed_plan_artifact_rejects_tampering(tmp_path):
    args = args_for(tmp_path / "request", backend="cosmos-rl")
    plan = workflow.build_plan(args)
    workflow.write_spec(args, plan)
    artifact = tmp_path / "approved-plan.json"
    workflow.save_plan_artifact(args, plan, str(artifact))

    changed = json.loads(artifact.read_text())
    changed["training"]["epochs"] = 99
    artifact.write_text(json.dumps(changed))
    current = workflow.parse_args(["render-slurm", "--plan-artifact", str(artifact)])
    with pytest.raises(common.WorkflowError, match="checksum mismatch"):
        workflow.load_plan_artifact(current, str(artifact))


def test_pairwise_parity_blocks_model_dataset_and_optimization_mismatch(tmp_path):
    left = workflow.build_plan(args_for(tmp_path / "left", backend="cosmos-framework"))
    right_args = args_for(tmp_path / "right", backend="cosmos-rl")
    # Use the exact same logical inputs for a valid pair.
    right_args.base_model_path_or_uri = left["model"]["supplied"]["resolved"]
    right_args.train_annotation = [item["resolved"] for item in left["datasets"]["train"]["annotations"]]
    right_args.train_media_root = [item["resolved"] for item in left["datasets"]["train"]["media_roots"]]
    right_args.validation_annotation = [item["resolved"] for item in left["datasets"]["validation"]["annotations"]]
    right_args.validation_media_root = [item["resolved"] for item in left["datasets"]["validation"]["media_roots"]]
    right = workflow.build_plan(right_args)
    report = workflow.parity_report(left, right)
    assert report["launch_allowed"], report
    changed = deepcopy(right); changed["training"]["learning_rate"] *= 2
    report = workflow.parity_report(left, changed)
    assert not report["launch_allowed"] and "optimization" in report["invalid_mismatches"]
    changed = deepcopy(right); changed["model"]["fingerprint"] = "different"
    assert "model" in workflow.parity_report(left, changed)["invalid_mismatches"]


def _status_records():
    return [
        {"status": "STARTED", "message": "Cosmos Framework"},
        {"status": "RUNNING", "phase": "train_complete", "kpi": {"train/avg_loss": 0.5, "train/loss_numerator": 50.0, "train/valid_label_count": 100}},
        {"status": "RUNNING", "phase": "validation_batch_complete", "kpi": {"val/batch_loss": 9.9}},
        {"status": "RUNNING", "phase": "validation_complete", "epoch": 1, "kpi": {"val/avg_loss": 0.1, "val/loss_numerator": 10.0, "val/valid_label_count": 100}},
        {"status": "RUNNING", "phase": "checkpoint_saved", "checkpoint_path": "/results/epoch_1"},
        {"status": "SUCCESS"},
    ]


def test_metric_extraction_requires_weighted_losses_and_accuracy():
    evaluation = {"average_validation_accuracy": 0.9, "numerator": 90, "denominator": 100, "per_task": {}, "excluded_tasks": [], "aggregation": "example_weighted", "coverage": {}}
    summary = metric.summarize_records(_status_records(), evaluation)
    assert summary["average_training_loss"]["average"] == 0.5
    assert summary["average_validation_loss"]["average"] == 0.1
    assert summary["evaluation"]["average_validation_accuracy"] == 0.9
    incomplete = copy = _status_records(); copy[1] = {"status": "RUNNING", "kpi": {"train/loss": 0.2}}
    with pytest.raises(metric.MetricError, match="training loss"):
        metric.summarize_records(copy, evaluation)
    with pytest.raises(metric.MetricError, match="accuracy"):
        metric.summarize_records(_status_records())


def test_metric_extraction_accepts_pretty_json_array_and_nested_phase(tmp_path):
    records = deepcopy(_status_records())
    for record in records:
        if "phase" in record:
            record.setdefault("data", {})["phase"] = record.pop("phase")
    path = tmp_path / "status.json"
    path.write_text(json.dumps(records, indent=2))
    loaded = metric.records_from_jsonl(path)
    evaluation = {"average_validation_accuracy": 0.9, "numerator": 90, "denominator": 100}
    summary = metric.summarize_records(loaded, evaluation)
    assert summary["average_validation_loss"]["average"] == 0.1


def test_metadata_schema_and_child_failure_guard(tmp_path):
    args = args_for(tmp_path); args.partition = "p"; args.account = "a"; args.stdout_path = "out"; args.stderr_path = "err"
    plan = workflow.build_plan(args); workflow.write_spec(args, plan)
    metadata = workflow.initial_metadata(args, plan)
    common.validate_metadata(metadata)
    metadata["child_process"]["exit_code"] = 7; metadata["terminal_tao_status"] = "SUCCESS"
    with pytest.raises(common.WorkflowError, match="nonzero"):
        common.validate_metadata(metadata)
    del metadata["image"]
    with pytest.raises(common.WorkflowError, match="incomplete"):
        common.validate_metadata(metadata)


def test_metadata_finalization_requires_child_and_tao_terminal_status(tmp_path):
    args = args_for(tmp_path); args.partition = "p"; args.account = "a"; args.stdout_path = "out"; args.stderr_path = "err"
    plan = workflow.build_plan(args); workflow.write_spec(args, plan)
    metadata = workflow.initial_metadata(args, plan)
    child = tmp_path / "child"; child.write_text("0\n")
    status = tmp_path / "status.json"; status.write_text(json.dumps([{"status": "SUCCESS"}]))
    finalized = workflow.finalize_metadata(
        metadata, child_exit_file=child, status_file=status, scheduler_state="COMPLETED",
        scheduler_reason=None, scheduler_exit_code="0:0", allocated_nodes=["node-a"], job_id="123",
    )
    assert finalized["terminal_tao_status"] == "SUCCESS"
    jsonl = tmp_path / "status-jsonl.json"
    jsonl.write_text('{"status":"RUNNING"}\n{"status":"SUCCESS"}\n')
    jsonl_finalized = workflow.finalize_metadata(
        workflow.initial_metadata(args, plan), child_exit_file=child, status_file=jsonl,
        scheduler_state="COMPLETED", scheduler_reason=None, scheduler_exit_code="0:0",
    )
    assert jsonl_finalized["terminal_tao_status"] == "SUCCESS"
    child.write_text("9\n")
    failed = workflow.finalize_metadata(
        workflow.initial_metadata(args, plan), child_exit_file=child, status_file=status,
        scheduler_state="COMPLETED", scheduler_reason=None, scheduler_exit_code="0:0",
    )
    assert failed["terminal_tao_status"] == "FAILURE"
    child.unlink()
    with pytest.raises(common.WorkflowError, match="exit-code file"):
        workflow.finalize_metadata(
            workflow.initial_metadata(args, plan), child_exit_file=child, status_file=status,
            scheduler_state="COMPLETED", scheduler_reason=None, scheduler_exit_code="0:0",
        )


def test_request_and_metadata_schemas_and_no_environment_history():
    json.loads((SKILL / "schemas" / "train.schema.json").read_text())
    json.loads((SKILL / "schemas" / "cosmos-job-metadata.schema.json").read_text())
    forbidden = ("/lustre", "/localhome", "rarunachalam")
    for path in SKILL.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".yml", ".json"}:
            text = path.read_text(encoding="utf-8")
            assert not any(value in text for value in forbidden), path
            development_dataset_names = ("w" + "ts", "ae" + "tc")
            assert not any(value in text.casefold() for value in development_dataset_names), path
