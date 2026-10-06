# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Sparse4D specifications, artifact contracts, and the depth repair CLI.

Source parity tests run when the public TAO Core/Data Services/PyTorch packages
are importable. The remaining tests require only the skill-bank test environment.
No model jobs, downloads, or GPU allocations are performed.
"""

import copy
import importlib
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills/models/tao-train-sparse4d"
INFO = yaml.safe_load((SKILL / "references/skill_info.yaml").read_text())
ACTIONS = tuple(INFO["actions"])
NORMALIZER = SKILL / "scripts/normalize_depth_paths.py"


def schema(action):
    return json.loads((SKILL / "schemas" / f"{action}.schema.json").read_text())


def merge(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) else value
    return out


def normalize_metadata(value):
    if isinstance(value, dict):
        return {
            key: sorted(item)
            if key in {"popular", "automl_disabled_parameters"}
            and isinstance(item, list)
            and all(isinstance(entry, str) for entry in item)
            else normalize_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_metadata(item) for item in value]
    return value


@pytest.mark.parametrize("action", ACTIONS)
def test_templates_match_packaged_schema_defaults(action):
    template = yaml.safe_load((SKILL / "references" / f"spec_template_{action}.yaml").read_text())
    assert template == schema(action)["default"]


@pytest.mark.parametrize("action", ACTIONS)
def test_declared_action_inputs_exist_in_schema(action):
    for field in INFO["actions"][action]["inputs"]:
        node = schema(action)
        for part in field.split("."):
            assert part in node["properties"], (action, field, part)
            node = node["properties"][part]


def test_evaluation_stages_the_selected_eval_dataset():
    sources = INFO["data_sources"]["evaluate"]
    inputs = INFO["actions"]["evaluate"]["inputs"]
    assert sources["dataset.test_dataset.ann_file"]["source"] == "eval_dataset"
    assert "dataset.test_dataset.ann_file" in inputs
    assert "dataset.train_dataset.ann_file" not in inputs
    assert "dataset.val_dataset.ann_file" not in inputs
    assert "model.head.instance_bank.anchor" not in sources
    assert all(item["source"] != "train_datasets" for item in sources.values())


def test_training_artifacts_are_not_required_for_export_or_inference():
    assert INFO["data_sources"]["export"] == {}
    assert set(INFO["actions"]["export"]["inputs"]) == {
        "model.head.instance_bank.anchor", "export.checkpoint"
    }
    assert "dataset.train_dataset.ann_file" not in INFO["actions"]["inference"]["inputs"]
    assert "dataset.test_dataset.ann_file" not in INFO["actions"]["train"]["inputs"]
    assert INFO["actions"]["train"]["inputs"]["model.head.loose_to_tight.mlp_ckpt"]["optional"]
    assert INFO["actions"]["quantize"]["inputs"]["dataset.test_dataset.ann_file"]["optional"]


@pytest.mark.parametrize("action", ACTIONS)
def test_schema_regenerates_from_public_sources(action):
    pytest.importorskip("nvidia_tao_core")
    if action == "dataset_convert":
        pytest.importorskip("nvidia_tao_ds")
    spec = importlib.util.spec_from_file_location("sparse4d_schema_generator", REPO / "scripts/generate_dataclass_schemas.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generated, _, _ = generator.generate_schema_for_action(INFO, "sparse4d", action)
    assert normalize_metadata(generated) == normalize_metadata(schema(action))


@pytest.mark.parametrize("backbone", ["resnet_50", "resnet_101"])
def test_geometric_fragment_preserves_selected_backbone_and_artifacts(backbone):
    base = schema("train")["default"]
    base["model"]["backbone"]["type"] = backbone
    base["model"]["head"]["instance_bank"]["anchor"] = "/models/selected/anchor.npy"
    base["train"]["pretrained_model_path"] = "/models/selected/model.pth"
    base["dataset"]["val_dataset"]["ann_file"] = "/data/heldout.pkl"
    fragment = yaml.safe_load((SKILL / "references/geometric_distillation_train.yaml").read_text())
    cfg = merge(base, fragment)
    assert cfg["model"]["backbone"]["type"] == backbone
    assert cfg["model"]["head"]["instance_bank"]["anchor"] == "/models/selected/anchor.npy"
    assert cfg["train"]["pretrained_model_path"] == "/models/selected/model.pth"
    assert cfg["dataset"]["val_dataset"]["ann_file"] == "/data/heldout.pkl"
    ltt = cfg["model"]["head"]["loose_to_tight"]
    assert ltt["enable"] and ltt["pseudo_enable"]
    assert ltt["mlp_ckpt"].endswith("sparse4d_rn50_vtrainable_v3.0/_loose_to_tight_mlp.pth")


@pytest.mark.parametrize("package", ["nvidia_tao_core", "nvidia_tao_pytorch"])
@pytest.mark.parametrize("backbone", ["resnet_50", "resnet_101"])
def test_geometric_fragment_merges_into_runtime_dataclass(package, backbone):
    pytest.importorskip(package)
    omega = pytest.importorskip("omegaconf").OmegaConf
    module = importlib.import_module(f"{package}.config.sparse4d.default_config")
    cfg = omega.structured(module.ExperimentConfig())
    cfg.model.backbone.type = backbone
    merged = omega.merge(cfg, omega.load(SKILL / "references/geometric_distillation_train.yaml"))
    assert merged.model.backbone.type == backbone
    assert merged.model.head.loose_to_tight.enable
    assert merged.dataset.sync_route


def test_core_validator_accepts_distillation_and_rejects_unknown_field():
    pytest.importorskip("nvidia_tao_core")
    from nvidia_tao_core.api_utils.json_schema_validation import validate_jsonschema

    fragment = yaml.safe_load((SKILL / "references/geometric_distillation_train.yaml").read_text())
    properties = schema("train")["properties"]
    assert validate_jsonschema(fragment, properties) is None
    invalid = copy.deepcopy(fragment)
    invalid["model"]["head"]["loose_to_tight"]["missing_runtime_option"] = True
    assert "missing_runtime_option" in validate_jsonschema(invalid, properties)


def test_annotation_free_conversion_spec_matches_runtime():
    pytest.importorskip("nvidia_tao_core")
    ds = pytest.importorskip("nvidia_tao_ds")
    omega = pytest.importorskip("omegaconf").OmegaConf
    from nvidia_tao_ds.config.annotations.default_config import ExperimentConfig
    from nvidia_tao_core.api_utils.json_schema_validation import validate_jsonschema

    path = Path(ds.__file__).parent / "annotations/experiment_specs/aicity2ovpkl_unlabeled.yaml"
    spec = yaml.safe_load(path.read_text())
    spec["aicity"].update(root="/data/aicity_root", split="train", fps=25)
    merged = omega.merge(omega.structured(ExperimentConfig()), spec)
    assert merged.aicity.load_annotations is False
    assert merged.aicity.recentering is False
    assert merged.aicity.fps == 25
    assert validate_jsonschema(spec, schema("dataset_convert")["properties"]) is None


@pytest.mark.parametrize("payload_as_dict", [False, True])
def test_depth_repair_cli_dry_run_and_idempotence(tmp_path, payload_as_dict):
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "Scene/depth_maps/Camera1.h5"
    h5_path.parent.mkdir(parents=True)
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("frame.png", data=[[1, 2], [3, 4]])
    infos = [{"cams": {"Camera1": {"depth_map_path": ("Scene/Camera1", "depth/frame.png")}}}]
    payload = {"infos": infos, "metadata": {"version": "fixture"}} if payload_as_dict else infos
    annotation = tmp_path / "scene_infos_train.pkl"
    annotation.write_bytes(pickle.dumps(payload))
    original = annotation.read_bytes()
    command = [sys.executable, str(NORMALIZER), "--data-root", str(tmp_path), str(annotation)]
    dry = subprocess.run(command + ["--dry-run"], check=True, capture_output=True, text=True)
    assert "total_depth_paths_normalized=1" in dry.stdout
    assert annotation.read_bytes() == original
    subprocess.run(command, check=True, capture_output=True)
    updated = pickle.loads(annotation.read_bytes())
    rows = updated["infos"] if payload_as_dict else updated
    relative, key = rows[0]["cams"]["Camera1"]["depth_map_path"]
    with h5py.File(tmp_path / relative) as handle:
        assert handle[key][:].tolist() == [[1, 2], [3, 4]]
    normalized = annotation.read_bytes()
    again = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "total_depth_paths_normalized=0" in again.stdout
    assert annotation.read_bytes() == normalized


def test_depth_repair_leaves_unlabeled_samples_unchanged(tmp_path):
    annotation = tmp_path / "unlabeled_infos_train.pkl"
    annotation.write_bytes(pickle.dumps({"infos": [{"gt_boxes": None, "cams": {"Camera1": {"data_path": "image.jpg"}}}]}))
    original = annotation.read_bytes()
    subprocess.run([sys.executable, str(NORMALIZER), "--data-root", str(tmp_path), str(annotation)], check=True, capture_output=True)
    assert annotation.read_bytes() == original


def test_depth_repair_refuses_non_annotation_pickle_without_writing(tmp_path):
    annotation = tmp_path / "invalid.pkl"
    annotation.write_bytes(pickle.dumps({"frame_index": []}))
    original = annotation.read_bytes()
    result = subprocess.run([sys.executable, str(NORMALIZER), "--data-root", str(tmp_path), str(annotation)], capture_output=True)
    assert result.returncode != 0
    assert annotation.read_bytes() == original
