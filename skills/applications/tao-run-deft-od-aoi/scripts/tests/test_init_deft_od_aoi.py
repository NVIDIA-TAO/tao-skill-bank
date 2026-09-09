# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "init_deft_od_aoi.py"
SPEC = importlib.util.spec_from_file_location("init_deft_od_aoi", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _role(root: Path, name: str, boxed: bool) -> dict:
    images = root / name / "images"
    images.mkdir(parents=True)
    image = images / f"{name}.png"
    image.write_bytes(b"image")
    annotations = ([{"id": 1, "image_id": 1, "category_id": 1,
                     "bbox": [1, 1, 4, 4], "area": 16}] if boxed else [])
    coco = root / name / "coco.json"
    coco.write_text(json.dumps({"images": [{"id": 1, "file_name": image.name}],
                                "annotations": annotations,
                                "categories": [{"id": 1, "name": "defect"}]}))
    return {"images": str(images), "coco": str(coco)}


def _config(root: Path) -> Path:
    checkpoint = root / "base.pth"
    checkpoint.write_bytes(b"model")
    path = root / "policy.yaml"
    path.write_text(yaml.safe_dump({"platform": "slurm", "max_iterations": 2,
                                    "base_checkpoint": str(checkpoint),
                                    "sources": {name: _role(root, name, name != "clean")
                                                for name in MODULE.ROLES}}))
    return path


def _append_boxless_image(role: dict, name: str) -> None:
    images = Path(role["images"])
    image = images / f"{name}_boxless.png"
    image.write_bytes(b"boxless-image")
    coco = Path(role["coco"])
    data = json.loads(coco.read_text())
    data["images"].append({"id": 2, "file_name": image.name})
    coco.write_text(json.dumps(data))


def test_initialize_freezes_real_only_disjoint_contract(tmp_path: Path) -> None:
    state = MODULE.initialize(_config(tmp_path), tmp_path / "results")
    assert state["mode"] == "rtdetr_real_only"
    assert state["next_stage"] == "candidate_cache"
    assert Path(state["policy"]).is_file()
    assert Path(state["classmap"]).read_text() == "background\ndefect\n"
    assert state["roles"]["clean"]["annotation_count"] == 0


def test_initialize_rejects_role_overlap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    kpi = Path(value["sources"]["kpi"]["coco"])
    clean = Path(value["sources"]["clean"]["coco"])
    clean_data = json.loads(clean.read_text())
    kpi_image = Path(value["sources"]["kpi"]["images"]) / "kpi.png"
    clean_data["images"][0]["source_path"] = str(kpi_image)
    clean.write_text(json.dumps(clean_data))
    with pytest.raises(ValueError, match="overlaps"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_rejects_boxed_clean_role(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    clean = Path(value["sources"]["clean"]["coco"])
    data = json.loads(clean.read_text())
    data["annotations"] = [{"id": 1, "image_id": 1, "category_id": 1,
                            "bbox": [0, 0, 2, 2]}]
    clean.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="clean role"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_accepts_mixed_boxed_and_boxless_heldout_roles(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    for name in ("kpi", "test"):
        _append_boxless_image(value["sources"][name], name)

    state = MODULE.initialize(config, tmp_path / "results")

    for name in ("kpi", "test"):
        assert state["roles"][name]["image_count"] == 2
        assert state["roles"][name]["annotation_count"] == 1


def test_initialize_rejects_boxless_defective_real_role(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _append_boxless_image(value["sources"]["real"], "real")

    with pytest.raises(ValueError, match="real role contains boxless images"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_routes_missing_synthesis_weights_to_bootstrap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    pool, dataset, base, nn = tmp_path / "pool", tmp_path / "ft_dataset", tmp_path / "ft_base", tmp_path / "nn"
    for path in (pool, dataset, base, nn):
        path.mkdir()
    defect, validation, vae = tmp_path / "defect.jsonl", tmp_path / "validation.jsonl", tmp_path / "vae.pth"
    defect.write_text("{}\n")
    validation.write_text("{}\n")
    vae.write_bytes(b"vae")
    value["synthesis"] = {"enabled": True, "pool_dataset_root": str(pool),
                          "defect_spec": str(defect), "routes": {"route": {"finetune": {
                              "dataset_root": str(dataset), "validation_testcase": str(validation),
                              "base_checkpoint": str(base), "vae_path": str(vae),
                              "nn_backbone": str(nn),
                              "result_handoff": str(tmp_path / "future/handoff.json")}}}}
    config.write_text(yaml.safe_dump(value))
    state = MODULE.initialize(config, tmp_path / "results")
    assert state["synthesis_bootstrap_required"] is True
    assert state["next_stage"] == "synthesis_bootstrap"
