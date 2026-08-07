#!/usr/bin/env python3
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
        "--cosmos-framework-base-tag", "example/framework-base:test",
        "--cosmos-rl-base-image", "example/cosmos-rl-runtime:test",
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


def test_model_backend_resolution_and_comparative_explicitness():
    assert workflow.select_backend(model="Cosmos3-Nano", action="train", workload="training")[0] == "cosmos-framework"
    assert workflow.select_backend(model="Cosmos3-Nano", action="evaluate", workload="training")[0] == "cosmos-rl"
    assert workflow.select_backend(model="Cosmos3-Nano", action="export", workload="training")[0] == "cosmos-framework"
    assert workflow.select_backend(model="Cosmos3-Edge", action="evaluate", workload="training")[0] == "cosmos-framework"
    assert workflow.select_backend(model="Cosmos3-Edge", action="inference_microservice", workload="training")[0] == "cosmos-framework"
    with pytest.raises(common.WorkflowError, match="backend selection"):
        workflow.select_backend(model="Cosmos3-Nano", action="train", backend="auto", comparative=True)


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
    assert plan["spec"]["trainer"]["max_iter"] == 2
    assert plan["training"]["optimizer_epsilon"] == 1e-6
    assert plan["spec"]["optimizer"]["eps"] == 1e-6
    assert "lora_enabled" not in plan["spec"]["model"]
    assert plan["datasets"]["train"]["annotations"][0]["original"] == args.train_annotation[0]
    source = Path(workflow.__file__).read_text(encoding="utf-8")
    assert "/lustre/" not in source and "rarunachalam" not in source
    with Path(args.write_spec).open("rb") as stream:
        assert tomllib.load(stream)["trainer"]["max_iter"] == 2


def test_cosmos_rl_peft_spec_has_equivalent_lora_and_cache(tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl", training_mode="peft")
    args.optimizer_epsilon = 1e-6
    plan = workflow.build_plan(args)
    assert plan["training"]["optimizer_epsilon"] == 1e-6
    assert plan["spec"]["train"]["epsilon"] == 1e-6
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
    assert plan["cache_prewarm"]["required"]
    assert plan["spec"]["train"]["train_policy"]["require_complete_dataset_cache"]
    assert "dataloader_prefetch_factor" not in plan["spec"]["train"]["train_policy"]


def test_cosmos_rl_maps_common_constant_scheduler_to_native_none(tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl")
    args.scheduler = "constant"

    plan = workflow.build_plan(args)

    assert plan["training"]["scheduler"] == "constant"
    assert plan["spec"]["train"]["optm_decay_type"] == "none"


def test_cosmos_rl_resolves_sft_hook_from_installed_native_package(tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl")
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
    child_argv = shlex.split(script.split("set +e\n", 1)[1].split("\nchild_rc=", 1)[0])
    nested = subprocess.run(
        ["bash", "-n", "-c", child_argv[-1]], capture_output=True, text=True
    )
    assert nested.returncode == 0, nested.stderr


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
    assert rl["spec"]["validation"]["enable_dataset_cache"] is False
    assert "require_complete_dataset_cache" not in rl["spec"]["train"]["train_policy"]
    assert "cache_dir" not in rl["spec"]["custom"]["vision"]
    assert rl["spec"]["train"]["optm_impl"] == "fused"


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
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["frame"] == "target_compute"
    assert payload["model"]["supplied"]["original"] == str(model)
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
    assert 'exit "$child_rc"' in script
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    child_argv = shlex.split(script.split("set +e\n", 1)[1].split("\nchild_rc=", 1)[0])
    assert child_argv[-2] == "-lc"
    assert subprocess.run(
        ["bash", "-n", "-c", child_argv[-1]], capture_output=True, text=True
    ).returncode == 0
    # Controlled child failure uses the same capture idiom as the generated job.
    result = subprocess.run(["bash", "-c", "set -Eeuo pipefail; rc=0; set +e; bash -c 'exit 17'; rc=$?; set -e; exit $rc"])
    assert result.returncode == 17
    assert subprocess.run(["sh", "-n"], input=script, text=True).returncode != 0 or "#!/usr/bin/env bash" in script


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
    assert "dcp_async_mode_enabled" not in plan["spec"]["checkpoint"]


def test_rl_sft_batch_is_per_dp_worker_and_multinode_launchers_are_packaged(tmp_path):
    args = args_for(tmp_path)
    args.backend = "cosmos-rl"
    args.rl_mini_batch = 1
    args.effective_global_batch = 8
    plan = workflow.build_plan(args)
    assert plan["spec"]["train"]["train_batch_per_replica"] == 1

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
    assert plan["image"]["must_rebuild_after_source_change"] is True
    assert plan["image"]["sqsh"]["reuse_allowed"] is False
    assert plan["image"]["provenance_path"] == "/opt/tao/image-provenance.json"
    assert plan["image"]["required_commits"]["cosmos-framework"] == "f" * 40


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
    assert report["launch_allowed"]
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
