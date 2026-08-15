#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "models" / "tao-finetune-cosmos-reason"
SCRIPT = SKILL / "scripts" / "evaluation_workflow.py"
sys.path.insert(0, str(SKILL / "scripts"))

from cosmos_common import inspect_dataset, stable_hash  # noqa: E402


def _sealed_plan(
    tmp_path: Path,
    *,
    backend: str = "cosmos-rl",
    mode: str = "dense",
    prompt: str = "training prompt",
    max_video_pixels: int | None = 4096,
) -> Path:
    plan = {
        "schema_version": 2,
        "experiment_id": "training-job",
        "action": "train",
        "backend": backend,
        "training": {
            "training_mode": mode,
            "precision": "bfloat16",
            "seed": 17,
            "frames": 8,
            "system_prompt": prompt,
        },
        "datasets": {
            "train": {"dataset_fingerprint": "train-fingerprint"},
            "validation": {
                "dataset_fingerprint": "validation-fingerprint",
                "annotations": [{"original": "/runtime/validation.json"}],
                "media_roots": [{"original": "/runtime/validation-media"}],
                "evaluation_profile": {
                    "inferred_task_type": "binary",
                    "answer_type": "freeform",
                    "metric_names": [],
                    "requires_user_input": [],
                    "unresolved_accuracy_tasks": [],
                },
            },
        },
        "model": {
            "fingerprint": "model-fingerprint",
            "supplied": {"original": "/runtime/base-model"},
        },
        "model_preparation": {
            "output": {"original": "/runtime/prepared-base-model"}
        },
        "prepared_model_container_path": "/runtime/prepared-base-model",
        "processor_profile": {"frames": 8, "max_video_pixels": max_video_pixels},
        "compute": {"total_gpus": 8},
        "decoder_artifact": {"enabled": False},
        "evaluation_contract": {
            "schema_version": 1,
            "validation_dataset_fingerprint": "validation-fingerprint",
            "validation_annotations": ["/runtime/validation.json"],
            "validation_media_roots": ["/runtime/validation-media"],
            "system_prompt": prompt,
            "frames": 8,
            "max_video_pixels": max_video_pixels,
            "precision": "bfloat16",
            "seed": 17,
            "batch_size": 1,
            "task_profile": {
                "inferred_task_type": "binary",
                "answer_type": "freeform",
                "metric_names": [],
                "requires_user_input": [],
                "unresolved_accuracy_tasks": [],
            },
            "generation": {
                "max_tokens": None,
                "temperature": 0.0,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
            "checkpoint_selection": None,
        },
    }
    plan["plan_artifact"] = {
        "schema_version": 1,
        "path": str(tmp_path / "training-plan.json"),
        "sha256": stable_hash(plan),
    }
    path = tmp_path / "training-plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _status(tmp_path: Path, *, multiple: bool = False) -> Path:
    records = [
        {
            "status": "RUNNING",
            "phase": "checkpoint_saved",
            "checkpoint_path": "/runtime/checkpoints/epoch_1",
            "kpi": {"epoch": 1},
        }
    ]
    if multiple:
        records.append(
            {
                "status": "RUNNING",
                "phase": "checkpoint_saved",
                "checkpoint_path": "/runtime/checkpoints/epoch_2",
                "kpi": {"epoch": 2},
            }
        )
    records.append({"status": "SUCCESS", "message": "training complete"})
    path = tmp_path / "status.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _run(
    tmp_path: Path,
    *,
    backend: str = "cosmos-rl",
    mode: str = "dense",
    multiple: bool = False,
    prompt: str = "training prompt",
    max_video_pixels: int | None = 4096,
    extra: list[str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    plan_output = tmp_path / "evaluation-plan.json"
    config_output = tmp_path / "evaluation.toml"
    command = [
        sys.executable,
        str(SCRIPT),
        "--training-plan",
        str(
            _sealed_plan(
                tmp_path,
                backend=backend,
                mode=mode,
                prompt=prompt,
                max_video_pixels=max_video_pixels,
            )
        ),
        "--training-status",
        str(_status(tmp_path, multiple=multiple)),
        "--results-dir",
        "/runtime/evaluation-results",
        "--generation-max-tokens",
        "16",
        "--plan-output",
        str(plan_output),
        "--config-output",
        str(config_output),
        *(extra or []),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False), plan_output, config_output


def test_dense_evaluation_inherits_training_parity_fields(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    resolved = json.loads(plan_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert resolved["ready"] is True
    assert resolved["required_user_inputs"] == []
    assert config["dataset"] == {
        "annotation_path": "/runtime/validation.json",
        "media_dir": "/runtime/validation-media",
        "system_prompt": "training prompt",
    }
    assert config["task"]["type"] == "binary"
    assert config["metrics"]["names"] == []
    assert config["vision"]["num_frames"] == 8
    assert config["vision"]["max_pixels"] == 4096
    assert "nframes" not in config["vision"]
    assert config["model"]["model_name"] == "/runtime/checkpoints/epoch_1"
    assert config["model"]["enable_lora"] is False
    assert config["evaluation"]["batch_size"] == 1
    assert config["num_gpus"] == 8
    assert resolved["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()


def test_missing_pixel_budget_accepts_explicit_evaluation_input(tmp_path: Path) -> None:
    unresolved, plan_path, config_path = _run(tmp_path, max_video_pixels=None)
    assert unresolved.returncode == 3, unresolved.stderr
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [item["field"] for item in plan["required_user_inputs"]] == ["vision.max_pixels"]
    assert not config_path.exists()

    resolved, resolved_plan_path, resolved_config_path = _run(
        tmp_path,
        max_video_pixels=None,
        extra=["--max-video-pixels", "3136000"],
    )
    assert resolved.returncode == 0, resolved.stderr
    resolved_plan = json.loads(resolved_plan_path.read_text(encoding="utf-8"))
    config = tomllib.loads(resolved_config_path.read_text(encoding="utf-8"))
    assert config["vision"]["max_pixels"] == 3136000
    assert resolved_plan["provenance"]["vision.max_pixels"] == {
        "source": "user",
        "value": 3136000,
    }


def test_cosmos_rl_peft_recovers_base_model_without_user_reentry(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(tmp_path, mode="peft")
    assert result.returncode == 0, result.stderr
    resolved = json.loads(plan_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert resolved["required_user_inputs"] == []
    assert config["model"]["enable_lora"] is True
    assert config["model"]["base_model_path"] == "/runtime/prepared-base-model"
    assert resolved["provenance"]["model.base_model_path"]["source"] == "sealed_training_plan.model_preparation"


def test_recorded_empty_system_prompt_is_inherited_not_reported_missing(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(tmp_path, prompt="")
    assert result.returncode == 0, result.stderr
    resolved = json.loads(plan_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert resolved["required_user_inputs"] == []
    assert config["dataset"]["system_prompt"] == ""
    assert resolved["provenance"]["dataset.system_prompt"]["source"] == "sealed_training_plan.evaluation_contract"


def test_framework_export_is_automated_not_user_intake(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(tmp_path, backend="cosmos-framework", mode="peft")
    assert result.returncode == 3, result.stderr
    unresolved = json.loads(plan_path.read_text(encoding="utf-8"))

    assert unresolved["required_user_inputs"] == []
    assert unresolved["automated_actions"][0]["action"] == "framework_checkpoint_pre_action"
    assert not config_path.exists()

    rerun, resolved_path, resolved_config = _run(
        tmp_path,
        backend="cosmos-framework",
        mode="peft",
        extra=["--action-model-path", "/runtime/exported-checkpoint"],
    )
    assert rerun.returncode == 0, rerun.stderr
    config = tomllib.loads(resolved_config.read_text(encoding="utf-8"))
    assert config["model"]["model_name"] == "/runtime/exported-checkpoint"
    assert config["model"]["enable_lora"] is False
    assert json.loads(resolved_path.read_text())["ready"] is True


def test_multiple_checkpoints_require_exact_selection(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(tmp_path, multiple=True)
    assert result.returncode == 3, result.stderr
    unresolved = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [item["field"] for item in unresolved["required_user_inputs"]] == ["checkpoint_selection"]
    assert not config_path.exists()

    selected, _, selected_config = _run(tmp_path, multiple=True, extra=["--checkpoint-epoch", "2"])
    assert selected.returncode == 0, selected.stderr
    config = tomllib.loads(selected_config.read_text(encoding="utf-8"))
    assert config["model"]["model_name"] == "/runtime/checkpoints/epoch_2"


def test_multiple_recorded_validation_manifests_are_automated_not_user_selection(tmp_path: Path) -> None:
    training_plan = _sealed_plan(tmp_path)
    plan = json.loads(training_plan.read_text(encoding="utf-8"))
    plan.pop("plan_artifact")
    plan["evaluation_contract"]["validation_annotations"] = [
        "/runtime/validation-a.json",
        "/runtime/validation-b.json",
    ]
    plan["evaluation_contract"]["validation_media_roots"] = [
        "/runtime/validation-media",
    ]
    plan["plan_artifact"] = {
        "schema_version": 1,
        "path": str(training_plan),
        "sha256": stable_hash(plan),
    }
    training_plan.write_text(json.dumps(plan), encoding="utf-8")
    evaluation_plan = tmp_path / "evaluation-plan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--training-plan",
            str(training_plan),
            "--training-status",
            str(_status(tmp_path)),
            "--results-dir",
            "/runtime/evaluation-results",
            "--generation-max-tokens",
            "16",
            "--plan-output",
            str(evaluation_plan),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 3, result.stderr
    unresolved = json.loads(evaluation_plan.read_text(encoding="utf-8"))
    assert unresolved["required_user_inputs"] == []
    assert unresolved["automated_actions"][0]["action"] == "materialize_exact_validation_manifest"
    assert unresolved["blockers"][0]["user_input"] is False


def test_dataset_inspection_records_binary_and_ambiguous_profiles(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    for name in ("yes.mp4", "no.mp4"):
        (media / name).write_bytes(name.encode())
    binary_manifest = tmp_path / "binary.json"
    binary_manifest.write_text(
        json.dumps(
            [
                {"id": "yes", "video": "yes.mp4", "conversations": [{"value": "q"}, {"value": "Yes"}]},
                {"id": "no", "video": "no.mp4", "conversations": [{"value": "q"}, {"value": "No"}]},
            ]
        ),
        encoding="utf-8",
    )
    binary = inspect_dataset(
        dataset_family="auto",
        annotations=[str(binary_manifest)],
        media_roots=[str(media)],
        verify_media_content=False,
    )
    assert binary["evaluation_profile"]["inferred_task_type"] == "binary"
    assert binary["evaluation_profile"]["requires_user_input"] == []

    ambiguous_manifest = tmp_path / "ambiguous.json"
    ambiguous_manifest.write_text(
        json.dumps(
            [
                {"id": "a", "video": "yes.mp4", "conversations": [{"value": "q"}, {"value": "A"}]},
                {"id": "b", "video": "no.mp4", "conversations": [{"value": "q"}, {"value": "B"}]},
            ]
        ),
        encoding="utf-8",
    )
    ambiguous = inspect_dataset(
        dataset_family="auto",
        annotations=[str(ambiguous_manifest)],
        media_roots=[str(media)],
        verify_media_content=False,
    )
    assert ambiguous["evaluation_profile"]["unresolved_accuracy_tasks"] == ["default"]
    assert "task.type" in ambiguous["evaluation_profile"]["requires_user_input"]


def test_template_is_dataset_neutral_and_schema_preserves_automl_dimensions() -> None:
    template = yaml.safe_load((SKILL / "references" / "spec_template_evaluate.yaml").read_text())
    schema = json.loads((SKILL / "schemas" / "evaluate.schema.json").read_text())
    text = json.dumps({"template": template, "schema": schema}).casefold()

    assert template == schema["default"]
    assert template["dataset"]["system_prompt"] == ""
    assert template["metrics"]["names"] == []
    assert template["vision"]["num_frames"] == 0
    assert "nframes" not in template["vision"]
    assert "street-view" not in text
    assert "cr1_1_zero_shot" not in text
    assert schema["automl_default_parameters"] == [
        "dataset.system_prompt",
        "vision.num_frames",
        "generation.max_tokens",
        "generation.temperature",
        "generation.repetition_penalty",
        "generation.presence_penalty",
        "generation.frequency_penalty",
    ]
    assert schema["x_tao_resolution"]["template_launchable"] is False


def test_schema_generator_cannot_reintroduce_dataset_prompt_or_legacy_frame_key() -> None:
    generator_path = ROOT / "scripts" / "generate_dataclass_schemas.py"
    module_spec = importlib.util.spec_from_file_location("cosmos_schema_generator_test", generator_path)
    assert module_spec and module_spec.loader
    generator = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(generator)
    generated = generator.apply_cosmos_evaluate_resolution_metadata(
        {
            "default": {},
            "properties": {
                "dataset": {
                    "properties": {
                        "system_prompt": {
                            "default": "legacy dataset prompt",
                            "enum": ["legacy dataset prompt"],
                            "type": "categorical",
                        }
                    }
                },
                "vision": {
                    "properties": {
                        "nframes": {"default": 8, "enum": [4, 8, 16]}
                    }
                },
                "generation": {
                    "properties": {
                        "max_tokens": {"default": 1024, "enum": [256, 1024]},
                        "temperature": {"default": 0.0},
                        "max_retries": {"default": 10},
                    }
                },
                "model": {
                    "properties": {
                        "model_name": {"default": ""},
                        "save_folder": {"default": "legacy-output-name"},
                        "tokenizer_model_name": {"default": "legacy-tokenizer"},
                    }
                },
                "metrics": {
                    "properties": {
                        "names": {"default": ["bleu", "rouge"]}
                    }
                },
            },
        }
    )

    prompt = generated["properties"]["dataset"]["properties"]["system_prompt"]
    vision = generated["properties"]["vision"]["properties"]
    assert prompt["default"] == ""
    assert "enum" not in prompt
    assert "nframes" not in vision
    assert vision["num_frames"]["default"] == 0
    assert "save_folder" not in generated["properties"]["model"]["properties"]
    assert generated["properties"]["metrics"]["properties"]["names"]["default"] == []
    assert generated["default"]["dataset"]["system_prompt"] == ""
    assert generated["automl_default_parameters"][1] == "vision.num_frames"
    assert generated["x_tao_resolution"]["template_launchable"] is False
