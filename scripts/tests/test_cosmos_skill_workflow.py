#!/usr/bin/env python3
"""Regression tests for runtime-only Cosmos backend orchestration."""

from __future__ import annotations

import importlib.util
import json
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


def make_model(tmp_path: Path, model_type: str = "qwen3_vl") -> Path:
    model = tmp_path / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": model_type, "architectures": ["Qwen3VLForConditionalGeneration"]}))
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "tokenizer.json").write_text("{}")
    (model / "processor_config.json").write_text("{}")
    return model


def make_wts(tmp_path: Path, split: str, count: int = 16) -> tuple[Path, Path]:
    root = tmp_path / split
    media = root / "media"
    media.mkdir(parents=True)
    records = []
    for index in range(count):
        name = f"{split}-{index}.mp4"
        (media / name).write_bytes(f"video-{split}-{index}".encode())
        records.append({"id": f"{split}-{index}", "video": name, "conversations": [{"from": "human", "value": "<video> question"}, {"from": "gpt", "value": "Yes"}]})
    annotation = root / "manifest.json"
    annotation.write_text(json.dumps(records))
    return annotation, media


def make_aetc(tmp_path: Path, split: str) -> tuple[list[Path], Path]:
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


def args_for(tmp_path: Path, *, backend: str = "cosmos-framework", workload: str = "wts", run_mode: str = "full", training_mode: str = "dense", model_name: str = "nvidia/Cosmos3-Nano"):
    model = make_model(tmp_path, "cosmos3_edge" if "Edge" in model_name else "qwen3_vl")
    if workload == "wts":
        train_annotations, train_media = [make_wts(tmp_path, "train")[0]], [tmp_path / "train" / "media"]
        val_annotations, val_media = [make_wts(tmp_path, "validation")[0]], [tmp_path / "validation" / "media"]
    else:
        train_annotations, train_root = make_aetc(tmp_path, "train")
        val_annotations, val_root = make_aetc(tmp_path, "validation")
        train_media, val_media = [train_root], [val_root]
    for name in ("results", "checkpoints", "cache", "sqsh-cache", "integration", "framework", "rl", "daft", "tao-core"):
        (tmp_path / name).mkdir(exist_ok=True)
    ssh_key = tmp_path / "id_ed25519"; ssh_key.write_text("fixture")
    sqsh = tmp_path / "sqsh-cache" / "image.sqsh"; sqsh.write_bytes(b"sqsh")
    values = [
        "plan", "--model", model_name, "--backend", backend, "--action", "train",
        "--workload", workload, "--platform", "docker", "--run-mode", run_mode,
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
    assert workflow.select_backend(model="Cosmos3-Nano", action="train", workload="wts")[0] == "cosmos-framework"
    assert workflow.select_backend(model="Cosmos3-Nano", action="evaluate", workload="wts")[0] == "cosmos-rl"
    with pytest.raises(common.WorkflowError, match="backend selection"):
        workflow.select_backend(model="Cosmos3-Nano", action="train", backend="auto", comparative=True)


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


def test_wts_framework_dense_spec_and_no_historical_paths(tmp_path):
    args = args_for(tmp_path)
    plan = workflow.build_plan(args)
    workflow.write_spec(args, plan)
    assert plan["backend"] == "cosmos-framework"
    assert plan["training"]["training_mode"] == "dense"
    assert plan["spec"]["model"]["parallelism"]["data_parallel_shard_degree"] == 8
    assert plan["spec"]["trainer"]["grad_accum_iter"] == 1
    assert plan["spec"]["trainer"]["max_iter"] == 2
    assert "lora_enabled" not in plan["spec"]["model"]
    assert plan["datasets"]["train"]["annotations"][0]["original"] == args.train_annotation[0]
    source = Path(workflow.__file__).read_text(encoding="utf-8")
    assert "/lustre/" not in source and "rarunachalam" not in source
    with Path(args.write_spec).open("rb") as stream:
        assert tomllib.load(stream)["trainer"]["max_iter"] == 2


def test_cosmos_rl_peft_spec_has_equivalent_lora_and_cache(tmp_path):
    args = args_for(tmp_path, backend="cosmos-rl", training_mode="peft")
    plan = workflow.build_plan(args)
    lora = plan["spec"]["policy"]["lora"]
    assert lora == {"dim": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"], "bias": "none", "use_rslora": True, "modules_to_save": [], "adapter_dtype": "bfloat16"}
    assert plan["cache_prewarm"]["required"]
    assert plan["spec"]["train"]["train_policy"]["require_complete_dataset_cache"]
    assert "dataloader_prefetch_factor" not in plan["spec"]["train"]["train_policy"]


def test_framework_peft_spec_is_native_not_rl_schema(tmp_path):
    args = args_for(tmp_path, training_mode="peft")
    plan = workflow.build_plan(args)
    assert plan["spec"]["model"]["lora_enabled"] is True
    assert plan["spec"]["model"]["lora_target_modules"] == "q_proj,v_proj"
    assert plan["spec"]["optimizer"]["keys_to_select"] == ["lora_"]
    assert "policy" not in plan["spec"]


def test_aetc_paths_tasks_and_accuracy_coverage(tmp_path):
    args = args_for(tmp_path, workload="aetc")
    plan = workflow.build_plan(args)
    assert plan["datasets"]["train"]["tasks"] == {"bcq": 8, "mcq": 8, "scene_description": 8}
    coverage = plan["datasets"]["validation"]["metric_coverage"]
    assert coverage["accuracy_tasks"] == ["bcq", "mcq"]
    assert coverage["excluded_tasks"] == ["scene_description"]
    assert json.loads(plan["environment"]["AETC_TRAIN_ANNOTATIONS"]) == args.train_annotation
    assert plan["spec"]["job"]["experiment"] == "aetc_daft_vlm"
    args_rl = args_for(tmp_path / "rl", workload="aetc", backend="cosmos-rl")
    plan_rl = workflow.build_plan(args_rl)
    assert "tao_vl_reason_daft_sft_example.py" in plan_rl["command"]


@pytest.mark.parametrize("workload,experiment", [("wts", "wts_vlm_edge"), ("aetc", "aetc_daft_vlm_edge")])
def test_public_edge_checkpoint_uses_skill_runtime_profile(tmp_path, workload, experiment):
    args = args_for(tmp_path, workload=workload, model_name="nvidia/Cosmos3-Edge")
    plan = workflow.build_plan(args)

    assert plan["backend"] == "cosmos-framework"
    assert plan["model_preparation"]["required"] is False
    assert "no processor overlay" in plan["model_preparation"]["reason"]
    assert plan["prepared_model_container_path"] == str((tmp_path / "model").resolve())
    assert plan["spec"]["job"]["experiment"] == experiment
    assert plan["processor_profile"] == {
        "model_tier": "edge",
        "source": "tao_skill_default",
        "frames": 6,
        "sequence_length": 16000,
        "attention_implementation": "flash_attention_2",
        "frame_width": 1280,
        "frame_height": 720,
        "max_video_pixels": 5529600,
        "checkpoint_mutation": False,
    }
    assert plan["environment"]["WTS_VIDEO_MAX_PIXELS"] == "5529600"
    assert plan["environment"]["AETC_VIDEO_MAX_PIXELS"] == "5529600"


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
    annotation, media = make_wts(tmp_path, "same")
    inspected = common.inspect_dataset(workload="wts", annotations=[str(annotation)], media_roots=[str(media)])
    with pytest.raises(common.WorkflowError, match="overlap"):
        common.assert_no_overlap(inspected, inspected)
    records = json.loads(annotation.read_text()); (media / records[0]["video"]).unlink()
    with pytest.raises(common.WorkflowError, match="missing"):
        common.inspect_dataset(workload="wts", annotations=[str(annotation)], media_roots=[str(media)])


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
    plan = workflow.build_plan(args); workflow.write_spec(args, plan)
    script = workflow.render_slurm(args, plan)
    assert script.startswith("#!/usr/bin/env bash\nset -Eeuo pipefail")
    assert "#SBATCH --no-requeue" in script and "--container-image=" in script
    assert 'exit "$child_rc"' in script
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    # Controlled child failure uses the same capture idiom as the generated job.
    result = subprocess.run(["bash", "-c", "set -Eeuo pipefail; rc=0; set +e; bash -c 'exit 17'; rc=$?; set -e; exit $rc"])
    assert result.returncode == 17
    assert subprocess.run(["sh", "-n"], input=script, text=True).returncode != 0 or "#!/usr/bin/env bash" in script


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
    assert plan["environment"]["WTS_TRAIN_ANNOTATION"].startswith("/runtime/")
    assert plan["prepared_model_container_path"].startswith("/runtime/")


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
    forbidden = ("/lustre", "/localhome", "rarunachalam", "wts_train", "wts_eval")
    for path in SKILL.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".yml", ".json"}:
            text = path.read_text(encoding="utf-8")
            assert not any(value in text for value in forbidden), path
