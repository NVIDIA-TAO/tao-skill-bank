# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject YAML values that Data Services spec validators previously coerced."""

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


NEAREST = _load(
    "strict_nearest",
    "skills/data/tao-mine-nearest-neighbors/scripts/prepare_nearest_neighbors_spec.py",
)
IMAGE = _load(
    "strict_image",
    "skills/data/tao-generate-image-embeddings/scripts/verify_image_embeddings_spec.py",
)
GAPS = _load(
    "strict_gaps",
    "skills/data/tao-analyze-gaps-od-map/scripts/verify_object_detection_spec.py",
)
KPI = _load(
    "strict_kpi",
    "skills/data/tao-analyze-detection-kpi/scripts/verify_kpi_analyze_spec.py",
)
UNIQUE = _load(
    "strict_unique",
    "skills/data/tao-mine-od-images/scripts/verify_unique_neighbor_matching_spec.py",
)


@pytest.fixture
def paths(tmp_path):
    source = tmp_path / "source.parquet"
    target = tmp_path / "target.parquet"
    mapping = tmp_path / "mapping.yaml"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    mapping.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    return source, target, mapping, output


def test_nearest_neighbor_validator_rejects_coerced_values(paths):
    source, target, _, output = paths
    base = {
        "source_parquet": str(source),
        "target_parquet": str(target),
        "output_parquet": str(output / "mined.parquet"),
        "topn": 5,
        "knn_metric": "cosine",
        "filter_by_label": "false",
    }
    NEAREST.validate_config(dict(base))
    for key, value in (
        ("topn", True),
        ("topn", 1.5),
        ("filter_by_label", False),
        ("source_embed_column_name", None),
        ("distance_threshold", True),
    ):
        with pytest.raises(ValueError, match=key):
            NEAREST.validate_config({**base, key: value})


def test_image_embedding_validator_rejects_coerced_values(paths):
    source, _, _, output = paths
    base = {
        "input_parquet": str(source),
        "output_parquet": str(output / "embeddings.parquet"),
        "model": "SigLIP",
        "model_path": "google/siglip-base-patch16-224",
        "batch_size": 64,
    }
    IMAGE.validate_config(dict(base))
    for key, value in (
        ("batch_size", True),
        ("batch_size", 1.5),
        ("model_path", -1),
        ("model_config_path", -1),
    ):
        with pytest.raises(ValueError, match=key):
            IMAGE.validate_config({**base, key: value})


def test_object_detection_gap_validator_rejects_coerced_values(paths):
    source, target, _, output = paths
    base = {
        "input_format": "coco",
        "ground_truth_ann_path": str(source),
        "inference_ann_path": str(target),
        "images_dir": str(output),
        "results_dir": str(output),
        "kpi": "vehicle",
        "iou_threshold": 0.5,
        "conf_threshold": 0.0,
        "weak_thresholds": {},
    }
    GAPS.validate_config(dict(base))
    for key, value in (
        ("iou_threshold", True),
        ("conf_threshold", False),
        ("kpi", True),
        ("min_area", True),
        ("min_area", float("nan")),
        ("default_ap50_threshold", "0.5"),
    ):
        with pytest.raises(ValueError, match=key):
            GAPS.validate_config({**base, key: value})

    with pytest.raises(ValueError, match="weak_thresholds"):
        GAPS.validate_config(
            {**base, "weak_thresholds": {"vehicle": {"ap50": float("inf")}}}
        )


def test_detection_kpi_validator_rejects_coerced_values(paths):
    source, target, mapping, output = paths
    base = {
        "data": {
            "input_format": "COCO",
            "kpi_sources": [
                {
                    "image_dir": str(output),
                    "ground_truth_ann_path": str(source),
                    "inference_ann_path": str(target),
                }
            ],
            "mapping": str(mapping),
        },
        "visualize": {"platform": "local"},
        "kpi": {
            "iou_threshold": 0.5,
            "conf_threshold": 0.0,
            "num_recall_points": 11,
            "filter": False,
            "is_internal": False,
            "ignore_sqwidth": 0,
        },
        "results_dir": str(output),
    }
    KPI.validate_config(base)
    invalid = {
        "iou_threshold": True,
        "conf_threshold": False,
        "num_recall_points": 1.5,
        "filter": "false",
        "is_internal": 0,
        "ignore_sqwidth": True,
    }
    for key, value in invalid.items():
        payload = {**base, "kpi": {**base["kpi"], key: value}}
        with pytest.raises(ValueError, match=key):
            KPI.validate_config(payload)


def test_unique_neighbor_validator_rejects_coerced_values(paths):
    source, target, _, output = paths
    base = {
        "source_path": str(source),
        "target_path": str(target),
        "output_dir": str(output),
        "desired_unique_count": 10,
        "allocation_policy": "global",
        "distance_metric": "cosine",
        "save_embeddings": False,
        "visualize": False,
    }
    UNIQUE.validate_config(dict(base))
    for key, value in (
        ("desired_unique_count", True),
        ("desired_unique_count", 1.5),
        ("candidate_expansion_factor", True),
        ("save_embeddings", "false"),
        ("visualize", 0),
    ):
        with pytest.raises(ValueError, match=key):
            UNIQUE.validate_config({**base, key: value})
