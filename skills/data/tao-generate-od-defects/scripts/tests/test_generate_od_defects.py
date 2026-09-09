# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "generate_od_defects.py"
SPEC = importlib.util.spec_from_file_location("generate_od_defects", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _inputs(root: Path) -> argparse.Namespace:
    clean, mask = root / "clean.png", root / "mask.png"
    clean.write_bytes(b"image")
    mask.write_bytes(b"mask")
    testcase = root / "testcase.jsonl"
    testcase.write_text(json.dumps({"image_filename": str(clean), "mask_filename": str(mask),
                                    "anomaly_type": "texture+defect"}) + "\n")
    checkpoint, recipe = root / "adapter.pt", root / "recipe.yaml"
    checkpoint.write_bytes(b"adapter")
    recipe.write_text(yaml.safe_dump({"anomaly_types": [["texture", "defect"]]}))
    return argparse.Namespace(inputs_dir=None, input_data_path=testcase, checkpoint=checkpoint,
                              recipe=recipe, anomaly_types="", dataset_id="default", datasets=None)


def test_native_contract_infers_and_validates_types(tmp_path: Path) -> None:
    selected = MODULE.groups(_inputs(tmp_path))
    assert selected[0]["anomaly_types"] == ["texture+defect"]
    assert selected[0]["requested_rows"] == 1


def test_contract_rejects_type_absent_from_recipe(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    args.recipe.write_text(yaml.safe_dump({"anomaly_types": [["texture", "other"]]}))
    with pytest.raises(ValueError, match="absent from the recipe"):
        MODULE.groups(args)


def test_merge_validates_boxes_and_writes_binary_projection(tmp_path: Path) -> None:
    image = tmp_path / "generated.png"
    image.write_bytes(b"image")
    result = {"group": {"dataset_id": "d", "requested_rows": 1,
                        "anomaly_types": ["texture+defect"]},
              "generated": 1, "blocked": 0, "image_root": tmp_path,
              "coco": {"images": [{"id": 7, "file_name": str(image),
                                      "width": 20, "height": 10}],
                       "annotations": [{"id": 9, "image_id": 7, "category_id": 4,
                                        "bbox": [1, 2, 5, 6], "area": 30}],
                       "categories": [{"id": 4, "name": "texture+defect"}]}}
    report = MODULE._merge([result], tmp_path / "out")
    assert report["status"] == "COMPLETE" and report["training_pool_mutated"] is False
    binary = json.loads((tmp_path / "out/pseudo_labels/coco_annotations_od_defect.json").read_text())
    assert binary["categories"] == [{"id": 1, "name": "defect"}]
    assert binary["annotations"][0]["category_id"] == 1
