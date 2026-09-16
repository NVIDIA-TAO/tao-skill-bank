# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import pytest
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


def test_synthesis_resolves_gap_filename_stem_to_coco_id(tmp_path: Path) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    image, mask = images / "stable-image-key.png", tmp_path / "mask.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [{"id": 17, "file_name": image.name, "dataset_id": "route",
                    "texture_id": "texture"}],
        "annotations": [{"id": 3, "image_id": 17, "category_id": 1,
                         "bbox": [4, 5, 10, 12], "defect_class": "crack",
                         "fn_mask_source": str(mask)}],
        "categories": [{"id": 1, "name": "defect"}]}))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec = tmp_path / "defect.jsonl"
    checkpoint = tmp_path / "adapter.pt"
    recipe = tmp_path / "recipe.yaml"
    defect_spec.write_text(json.dumps({"defect_type": "texture+crack"}) + "\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": {"kpi": {"images": str(images), "coco": str(coco)}},
        "retrieval": {"model": "SigLIP", "model_path": "siglip",
                      "candidate_overfetch": 15},
        "synthesis": {"enabled": True, "pool_dataset_root": str(pool),
                      "defect_spec": str(defect_spec),
                      "routes": {"route": {"checkpoint": str(checkpoint),
                                             "recipe": str(recipe)}},
                      "max_neighbors_per_fn": 5, "min_similarity": 0.9,
                      "amp_model_id": "nvidia/Cosmos3-Nano"}}))
    gaps = tmp_path / "strict.parquet"
    pd.DataFrame([{"image_id": image.stem, "filepath": str(image), "gap_type": "FN",
                   "bbox": [4, 5, 14, 17], "class": "defect"}]).to_parquet(gaps)

    report = MODULE.prepare(policy, gaps, tmp_path / "out")

    assert report["fn_count"] == 1
    normalized = pd.read_parquet(tmp_path / "out/normalized_fn_gaps.parquet").iloc[0]
    assert normalized.image_id == image.stem


def test_synthesis_accepts_hardlinked_normalized_kpi_view(tmp_path: Path) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    original = tmp_path / "original.png"
    normalized = images / "stable-image-key.png"
    mask = tmp_path / "mask.png"
    original.write_bytes(b"image")
    os.link(original, normalized)
    mask.write_bytes(b"mask")
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [{"id": 17, "file_name": normalized.name,
                    "source_path": str(original), "dataset_id": "route",
                    "texture_id": "texture"}],
        "annotations": [{"id": 3, "image_id": 17, "category_id": 1,
                         "bbox": [4, 5, 10, 12], "defect_class": "crack",
                         "fn_mask_source": str(mask)}],
        "categories": [{"id": 1, "name": "defect"}],
    }))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec = tmp_path / "defect.jsonl"
    checkpoint = tmp_path / "adapter.pt"
    recipe = tmp_path / "recipe.yaml"
    defect_spec.write_text(json.dumps({"defect_type": "texture+crack"}) + "\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": {"kpi": {"images": str(images), "coco": str(coco)}},
        "retrieval": {"model": "SigLIP", "model_path": "siglip",
                      "candidate_overfetch": 15},
        "synthesis": {"enabled": True, "pool_dataset_root": str(pool),
                      "defect_spec": str(defect_spec),
                      "routes": {"route": {"checkpoint": str(checkpoint),
                                             "recipe": str(recipe)}},
                      "max_neighbors_per_fn": 5, "min_similarity": 0.9,
                      "amp_model_id": "nvidia/Cosmos3-Nano"},
    }))
    gaps = tmp_path / "strict.parquet"
    pd.DataFrame([{"image_id": normalized.stem, "filepath": str(normalized),
                   "gap_type": "FN", "bbox": [4, 5, 14, 17],
                   "class": "defect"}]).to_parquet(gaps)

    report = MODULE.prepare(policy, gaps, tmp_path / "out")

    assert report["fn_count"] == 1
    output = pd.read_parquet(tmp_path / "out/normalized_fn_gaps.parquet").iloc[0]
    assert Path(output.filepath) == original.resolve()

    unrelated = tmp_path / "unrelated.png"
    unrelated.write_bytes(b"different")
    pd.DataFrame([{"image_id": normalized.stem, "filepath": str(unrelated),
                   "gap_type": "FN", "bbox": [4, 5, 14, 17],
                   "class": "defect"}]).to_parquet(gaps)
    with pytest.raises(ValueError, match="filepath does not match"):
        MODULE.prepare(policy, gaps, tmp_path / "out-unrelated")


def test_synthesis_skips_unrouted_dataset_without_weakening_routed_masks(
    tmp_path: Path,
) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    routed_image = images / "routed.png"
    unrouted_image = images / "unrouted.png"
    mask = tmp_path / "mask.png"
    for path in (routed_image, unrouted_image, mask):
        path.write_bytes(b"input")
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": routed_image.name, "dataset_id": "route",
             "texture_id": "texture"},
            {"id": 2, "file_name": unrouted_image.name, "dataset_id": "boxes_only",
             "texture_id": "board"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4],
             "defect_class": "crack", "fn_mask_source": str(mask)},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [5, 6, 7, 8],
             "defect_class": "open"},
        ],
        "categories": [{"id": 1, "name": "defect"}],
    }))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec = tmp_path / "defect.jsonl"
    checkpoint = tmp_path / "adapter.pt"
    recipe = tmp_path / "recipe.yaml"
    defect_spec.write_text(json.dumps({"defect_type": "texture+crack"}) + "\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": {"kpi": {"images": str(images), "coco": str(coco)}},
        "retrieval": {"model": "SigLIP", "model_path": "siglip",
                      "candidate_overfetch": 15},
        "synthesis": {"enabled": True, "pool_dataset_root": str(pool),
                      "defect_spec": str(defect_spec),
                      "routes": {"route": {"checkpoint": str(checkpoint),
                                             "recipe": str(recipe)}},
                      "max_neighbors_per_fn": 5, "min_similarity": 0.9,
                      "amp_model_id": "nvidia/Cosmos3-Nano"},
    }))
    gaps = tmp_path / "strict.parquet"
    pd.DataFrame([
        {"image_id": 1, "filepath": str(routed_image), "gap_type": "FN",
         "bbox": [1, 2, 4, 6], "class": "defect"},
        {"image_id": 2, "filepath": str(unrouted_image), "gap_type": "FN",
         "bbox": [5, 6, 12, 14], "class": "defect"},
    ]).to_parquet(gaps)

    report = MODULE.prepare(policy, gaps, tmp_path / "out")

    assert report["fn_count"] == 1
    assert report["skipped_unrouted_by_dataset"] == {"boxes_only": 1}
    normalized = pd.read_parquet(tmp_path / "out/normalized_fn_gaps.parquet")
    assert list(normalized.dataset_id) == ["route"]

    coco_value = json.loads(coco.read_text())
    coco_value["annotations"][0].pop("fn_mask_source")
    coco.write_text(json.dumps(coco_value))
    with pytest.raises(ValueError, match="fn_mask_source"):
        MODULE.prepare(policy, gaps, tmp_path / "out-missing-routed-mask")


def test_image_index_rejects_duplicate_filename_stems() -> None:
    with pytest.raises(ValueError, match="duplicate KPI image identity: shared"):
        MODULE._image_index([
            {"id": 1, "file_name": "line-a/shared.png"},
            {"id": 2, "file_name": "line-b/shared.jpg"},
        ])
