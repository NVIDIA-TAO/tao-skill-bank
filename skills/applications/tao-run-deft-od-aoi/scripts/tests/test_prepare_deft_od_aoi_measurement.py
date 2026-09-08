# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import yaml


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_measurement.py"
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_measurement", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_measurement_freezes_binary_inference_and_dual_gap_specs(tmp_path: Path) -> None:
    sources = {}
    for role in ("kpi", "test"):
        images = tmp_path / role
        images.mkdir()
        (images / f"{role}.png").write_bytes(b"image")
        coco = tmp_path / f"{role}.json"
        coco.write_text(json.dumps({"images": [{"id": 1, "file_name": f"{role}.png"}],
                                    "annotations": [{"id": 1, "image_id": 1,
                                                     "category_id": 1, "bbox": [1, 2, 3, 4]}],
                                    "categories": [{"id": 1, "name": "defect"}]}))
        sources[role] = {"images": str(images), "coco": str(coco)}
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"sources": sources,
                                      "gap": {"inference_confidence": 0.001,
                                              "loose_confidence": 0.3,
                                              "strict_confidence": 0.8,
                                              "match_iou": 0.5}}))
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"model")
    report = MODULE.prepare(policy, checkpoint, tmp_path / "measure/kpi/inference/labels",
                            tmp_path / "measure", tmp_path / "specs")
    assert report["status"] == "COMPLETE"
    inference = yaml.safe_load((tmp_path / "specs/kpi_inference.yaml").read_text())
    assert inference["dataset"]["num_classes"] == 2
    assert Path(inference["dataset"]["infer_data_sources"]["classmap"]).read_text() == (
        "background\ndefect\n"
    )
    loose = yaml.safe_load((tmp_path / "specs/gap_loose.yaml").read_text())
    strict = yaml.safe_load((tmp_path / "specs/gap_strict.yaml").read_text())
    assert loose["conf_threshold"] == 0.3 and strict["conf_threshold"] == 0.8
    label = tmp_path / "specs/kpi_ground_truth_kitti/kpi.txt"
    assert label.read_text().startswith("defect 0.0 0 0.0 1.000000 2.000000 4.000000 6.000000")
