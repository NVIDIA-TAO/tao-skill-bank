# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-contract guards for the Video-CLIP skill."""

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "skills" / "models" / "tao-finetune-video-clip"


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def test_decode_contract_requires_pyav_without_decord_install_workaround():
    skill = (MODEL_ROOT / "SKILL.md").read_text()
    eval_config = (MODEL_ROOT / "eval.config").read_text()
    combined = skill + eval_config

    assert "PyAV is the primary Video-CLIP decoder" in skill
    assert "FFmpeg CLI and OpenCV" in skill
    assert "PyAV 17.1.0" in skill
    assert "ONNXScript 0.7.1" in skill
    assert "decord absent" in skill
    assert 'python -c \\"import av; print(av.__version__)\\"' in eval_config
    assert "pip install decord" not in combined.lower()
    assert "decord==" not in combined.lower()


def test_eval_data_staging_uses_the_provisioned_smoke_pack():
    eval_config = (MODEL_ROOT / "eval.config").read_text()

    assert "--profile seanlin" in eval_config
    assert "s3://computex/skill-eval-ci/iv2clip/*" in eval_config
    assert "team-tao" not in eval_config
    assert "s3://tao-skill-eval-ci/iv2clip/" not in eval_config
    assert "s3://skill_eval/test_data/tao-finetune-video-clip/" not in eval_config
    assert "s3://bucket/skill_eval/test_data/tao-finetune-video-clip/" not in eval_config


def test_deploy_actions_and_templates_match_video_clip_contract():
    base_info = _load_yaml(MODEL_ROOT / "references" / "skill_info.yaml")
    deploy_info = _load_yaml(
        MODEL_ROOT / "references" / "tao-deploy-video-clip.skill_info.yaml"
    )

    assert "gen_trt_engine" in base_info["actions"]
    assert set(deploy_info["actions"]) == {"gen_trt_engine", "evaluate", "inference"}
    assert set(deploy_info["spec_templates"]) == {
        "gen_trt_engine",
        "evaluate",
        "inference",
    }
    assert {
        "dataset.val.gt_queries",
        "dataset.val.metadata",
    } <= set(deploy_info["actions"]["evaluate"]["inputs"])

    inference_inputs = deploy_info["actions"]["inference"]["inputs"]
    for optional_input in (
        "inference.text_file",
        "dataset.val.gt_queries",
        "dataset.val.metadata",
    ):
        assert inference_inputs[optional_input]["optional"] is True

    templates = {
        action: _load_yaml(MODEL_ROOT / "references" / template)
        for action, template in deploy_info["spec_templates"].items()
    }
    assert templates["gen_trt_engine"]["gen_trt_engine"]["onnx_file"]
    assert templates["gen_trt_engine"]["gen_trt_engine"]["trt_engine"]
    assert templates["evaluate"]["dataset"]["val"]["gt_queries"]
    assert templates["evaluate"]["dataset"]["val"]["metadata"]
    assert templates["evaluate"]["evaluate"]["trt_engine"]
    assert templates["inference"]["inference"]["trt_engine"]
    assert templates["inference"]["inference"]["text_file"]

    deploy_docs = (
        MODEL_ROOT / "references" / "tao-deploy-video-clip.md"
    ).read_text()
    for required in ("*_config.yaml", "*_tokenizer/", "results.json"):
        assert required in deploy_docs

    export_spec = _load_yaml(MODEL_ROOT / "references" / "spec_template_export.yaml")
    assert export_spec["export"]["encoder_type"] == "combined"
    assert export_spec["export"]["batch_size"] == -1


def test_deploy_eval_case_is_registered():
    evals = json.loads((MODEL_ROOT / "evals" / "evals.json").read_text())
    ids = {case["id"] for case in evals}
    assert "tao-finetune-video-clip-tensorrt-deploy" in ids
