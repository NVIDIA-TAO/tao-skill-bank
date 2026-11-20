# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-contract guards for the Cosmos-Embed1 skill."""

import json
import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "skills" / "models" / "tao-finetune-cosmos-embed"
REFERENCES = MODEL_ROOT / "references"

IMAGE_TAG = "7.3.0-cosmos-embed"
SKILL_MD_CHAR_CAP = 18000
FORBIDDEN_PATH_SHAPES = (
    re.compile(r"/Users/[^/\s]+/"),
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"(?i)\bdevpod\b"),
    re.compile(r"(?i)\blepton\b"),
    re.compile(r"(?i)\bamlfs\b"),
)


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def _lora_template():
    return _load_yaml(REFERENCES / "spec_template_train_lora.yaml")


def test_lora_template_trains_explicit_adapters_with_plain_attention():
    spec = _lora_template()
    lora = spec["model"]["lora"]

    assert lora["enabled"] is True
    assert lora["target_modules"], "the LoRA recipe must name its adapters"
    vit_targets = [m for m in lora["target_modules"] if m.startswith("visual_encoder.blocks.")]
    assert vit_targets, "the recipe must adapt ViT attention blocks"
    assert all(m.endswith((".attn.qkv", ".attn.proj")) for m in vit_targets)
    assert spec["model"]["network"]["visual_encoder"]["transformer_engine"] is False
    assert spec["train"]["freeze_visual_encoder"] is True


def test_lora_template_uses_chunk_manifest_caption_pool():
    spec = _lora_template()
    train_ds = spec["dataset"]["train_dataset"]

    assert train_ds["dataset_type"] == "vad_r1_chunks"
    assert isinstance(train_ds["caption_field"], list) and len(train_ds["caption_field"]) >= 2
    assert train_ds["caption_mode"] == "all"
    assert train_ds["label_mode"] == "caption"
    assert train_ds["path_prefix_mapping"], "labels.json paths are rewritten only via path_prefix_mapping"
    assert spec["train"]["grad_accum_iter"] >= 1
    assert spec["train"]["validation_iter"] == spec["train"]["checkpoint_iter"]


def test_lora_template_benchmark_validation_is_opt_in_and_paired():
    spec = _lora_template()
    text = (REFERENCES / "spec_template_train_lora.yaml").read_text()

    # The public dataset ships no benchmark release root, so the template must train without one ...
    assert spec["dataset"]["val_dataset"]["dataset_type"] == "mock"
    assert "retrieval_benchmark" not in spec["train"]["callbacks"]
    # ... and document how to turn the benchmark on (dataset and callback together).
    assert "# retrieval_benchmark:" in text
    assert "# dataset_type: retrieval_benchmark" in text


def test_downstream_templates_carry_a_lora_mirror_block():
    for name in ("spec_template_export_onnx.yaml", "spec_template_evaluate.yaml", "spec_template_inference.yaml"):
        spec = _load_yaml(REFERENCES / name)
        assert spec["model"]["lora"]["enabled"] is False, name
        assert "target_modules" in (REFERENCES / name).read_text(), name


def test_onnx_export_template_is_per_tower_with_dynamic_batch():
    spec = _load_yaml(REFERENCES / "spec_template_export_onnx.yaml")

    assert spec["export"]["mode"] in {"video", "text"}
    assert spec["export"]["batch_size"] == -1
    assert spec["export"]["on_cpu"] is True
    assert spec["export"]["opset_version"] == 17


def test_skill_md_documents_the_image_and_adapter_contract():
    skill = (MODEL_ROOT / "SKILL.md").read_text()

    # Built by concatenation so the versions stamper does not treat this line as a stamped pin.
    stamped_pin = f"tao-toolkit:{IMAGE_TAG}  # versions-key: " + "images.tao_toolkit.cosmos_embed"
    assert skill.count(stamped_pin) == 2
    assert "list exactly the modules that were trained" in skill
    assert "export.batch_size=-1" in skill
    assert "selection.json" in skill
    assert "lora_B" in skill
    assert len(skill) <= SKILL_MD_CHAR_CAP


def test_skill_info_matches_the_templates():
    info = _load_yaml(REFERENCES / "skill_info.yaml")

    assert IMAGE_TAG in info["container_image"]
    assert {"nvidia/Cosmos-Embed1-224p", "nvidia/Cosmos-Embed1-336p", "nvidia/Cosmos-Embed1-448p"} <= set(
        info["huggingface_model_ids"]
    )
    assert info["spec_shorthand_keys"]["grad_accum_iter"] == "train.grad_accum_iter"
    assert info["spec_shorthand_keys"]["lora_rank"] == "model.lora.lora_rank"


def test_eval_cases_cover_the_lora_recipe_and_two_tower_export():
    evals = json.loads((MODEL_ROOT / "evals" / "evals.json").read_text())
    ids = {case["id"] for case in evals}

    assert {
        "tao-finetune-cosmos-embed-basic",
        "tao-finetune-cosmos-embed-lora-physicalai",
        "tao-finetune-cosmos-embed-onnx-two-tower-export",
    } <= ids
    for case in evals:
        assert case["expected_skill"] == "tao-finetune-cosmos-embed"
        assert "Do NOT run" in case["question"]
        assert case["expected_behavior"]


def test_skill_card_lists_every_reference_file():
    card = (MODEL_ROOT / "skill-card.md").read_text()
    for path in sorted(REFERENCES.iterdir()):
        if path.suffix in {".yaml", ".md"}:
            assert f"references/{path.name}" in card, path.name


def test_no_private_paths_or_internal_hosts_in_the_skill():
    for path in MODEL_ROOT.rglob("*"):
        if not path.is_file() or path.suffix == ".sig":
            continue
        text = path.read_text(errors="ignore")
        for pattern in FORBIDDEN_PATH_SHAPES:
            assert not pattern.search(text), f"{path.relative_to(REPO_ROOT)} matches {pattern.pattern}"
