# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_synthesis.py"
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_synthesis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_synthesis_normalizes_exact_kpi_false_negative(tmp_path: Path) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    image, mask = images / "image.png", tmp_path / "mask.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [{"id": 7, "file_name": image.name, "dataset_id": "route",
                    "texture_id": "texture"}],
        "annotations": [{"id": 3, "image_id": 7, "category_id": 1,
                         "bbox": [4, 5, 10, 12], "defect_class": "crack",
                         "fn_mask_source": str(mask)}],
        "categories": [{"id": 1, "name": "defect"}]}))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec, checkpoint, recipe = tmp_path / "defect.jsonl", tmp_path / "adapter.pt", tmp_path / "recipe.yaml"
    defect_spec.write_text(json.dumps({"defect_type": "texture+crack"}) + "\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"sources": {"kpi": {"images": str(images),
                                                             "coco": str(coco)}},
                                      "retrieval": {"model": "SigLIP", "model_path": "siglip",
                                                    "candidate_overfetch": 15},
                                      "synthesis": {"enabled": True,
                                                    "pool_dataset_root": str(pool),
                                                    "defect_spec": str(defect_spec),
                                                    "routes": {"route": {"checkpoint": str(checkpoint),
                                                                           "recipe": str(recipe)}},
                                                    "max_neighbors_per_fn": 5,
                                                    "min_similarity": 0.9,
                                                    "amp_model_id": "nvidia/Cosmos3-Nano"}}))
    gaps = tmp_path / "strict.parquet"
    pd.DataFrame([{"image_id": 7, "filepath": str(image), "gap_type": "FN",
                   "bbox": [4, 5, 14, 17], "class": "defect"}]).to_parquet(gaps)
    report = MODULE.prepare(policy, gaps, tmp_path / "out")
    assert report["fn_count"] == 1
    normalized = pd.read_parquet(tmp_path / "out/normalized_fn_gaps.parquet").iloc[0]
    assert normalized.anomaly_type == "texture+crack"
    config = yaml.safe_load((tmp_path / "out/anomalygen_filtering.yaml").read_text())
    assert config["datasets"]["route"]["checkpoint"] == str(checkpoint.resolve())
