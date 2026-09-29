# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_retrieval.py"
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_retrieval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _image(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def _oriented_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[274] = 8
    Image.fromarray(np.full((20, 40, 3), 80, dtype=np.uint8)).save(path, exif=exif)


def _policy(root: Path) -> Path:
    sources = {}
    for role in ("real", "clean"):
        images = root / role
        source = images / f"{role}.png"
        _image(source)
        annotations = ([{"id": 9, "image_id": 1, "category_id": 1,
                         "bbox": [8, 8, 12, 12]}] if role == "real" else [])
        coco = root / f"{role}.json"
        coco.write_text(json.dumps({"images": [{"id": 1, "file_name": source.name}],
                                    "annotations": annotations,
                                    "categories": [{"id": 1, "name": "defect"}]}))
        sources[role] = {"images": str(images), "coco": str(coco)}
    policy = root / "policy.yaml"
    policy.write_text(yaml.safe_dump({"sources": sources,
                                      "gap": {"background_iou_upper": 0.05,
                                              "near_miss_iou_upper": 0.5},
                                      "retrieval": {"model": "SigLIP", "model_path": "siglip",
                                                    "defect_context_scale": 1.5,
                                                    "clean_grids": [1, 2],
                                                    "candidate_overfetch": 15},
                                      "routing": {"real_mine_factor_min": 1,
                                                  "real_mine_factor_max": 6,
                                                  "near_miss_real_factor": 2,
                                                  "near_miss_real_cap_per_pocket": 20,
                                                  "clean_factor": 2}}))
    return policy


def test_candidate_cache_uses_defect_crops_and_clean_grid(tmp_path: Path) -> None:
    report = MODULE.candidates(_policy(tmp_path), tmp_path / "candidates")
    assert report == {"real": 1, "clean": 5}
    real = pd.read_parquet(tmp_path / "candidates/real_candidates.parquet")
    clean = pd.read_parquet(tmp_path / "candidates/clean_candidates.parquet")
    assert real.source_filepath.nunique() == clean.source_filepath.nunique() == 1
    assert all(Path(path).is_file() for path in list(real.filepath) + list(clean.filepath))


def test_candidate_cache_applies_exif_orientation_before_cropping(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    source = tmp_path / "real/real.png"
    _oriented_image(source)
    coco = tmp_path / "real.json"
    payload = json.loads(coco.read_text())
    payload["images"][0].update(width=20, height=40)
    payload["annotations"][0]["bbox"] = [2, 25, 5, 5]
    coco.write_text(json.dumps(payload))

    report = MODULE.candidates(policy, tmp_path / "candidates")

    assert report["real"] == 1
    frame = pd.read_parquet(tmp_path / "candidates/real_candidates.parquet")
    with Image.open(frame.iloc[0].filepath) as crop:
        assert crop.size == (9, 9)


def test_queries_route_fn_near_miss_and_background_fp(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    query_image = tmp_path / "query.png"
    _image(query_image)
    document = json.loads(policy.read_text()) if policy.suffix == ".json" else yaml.safe_load(policy.read_text())
    kpi = tmp_path / "kpi.json"
    kpi.write_text(json.dumps({"images": [{"id": 1, "file_name": query_image.name,
                                            "source_path": str(query_image),
                                            "deft_od_aoi": {"benchmark": "visa",
                                                            "texture": "pcb1",
                                                            "defect_type": "bad"}}],
                                "annotations": [], "categories": [{"id": 1, "name": "defect"}]}))
    document["sources"]["kpi"] = {"images": str(tmp_path), "coco": str(kpi)}
    policy.write_text(yaml.safe_dump(document))
    MODULE.candidates(policy, tmp_path / "candidates")
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.1}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [2, 2, 10, 10], "best_iou": value}
                  for value in (0.01, 0.2, 0.8)]).to_parquet(loose)
    report = MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                            tmp_path / "candidates", None)
    assert report["query_counts"] == {"real": 2, "clean": 1}
    assert report["admission_targets"] == {
        "real": {"fn": 1, "near_miss_fp": 2},
        "clean": {"background_fp": 2},
    }
    real = pd.read_parquet(tmp_path / "queries/real_queries.parquet")
    assert set(real.reason) == {"fn", "near_miss_fp"}
    real_mining = yaml.safe_load((tmp_path / "queries/mine_real.yaml").read_text())
    clean_mining = yaml.safe_load((tmp_path / "queries/mine_clean.yaml").read_text())
    assert real_mining["desired_unique_count"] == 1
    assert clean_mining["desired_unique_count"] == 5
    assert real_mining["candidate_expansion_factor"] == 15


def test_queries_route_boxless_background_without_defect_pocket(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    query_image = tmp_path / "boxless.png"
    _image(query_image)
    document = yaml.safe_load(policy.read_text())
    kpi = tmp_path / "kpi.json"
    kpi.write_text(json.dumps({
        "images": [{"id": 1, "file_name": query_image.name,
                    "source_path": str(query_image),
                    "deft_od_aoi": {"benchmark": "visa", "texture": "pcb1"}}],
        "annotations": [], "categories": [{"id": 1, "name": "defect"}],
    }))
    document["sources"]["kpi"] = {"images": str(tmp_path), "coco": str(kpi)}
    policy.write_text(yaml.safe_dump(document))
    MODULE.candidates(policy, tmp_path / "candidates")
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame(columns=["filepath", "gap_type", "bbox", "best_iou"]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [4, 4, 12, 12], "best_iou": 0.0}]).to_parquet(loose)

    report = MODULE.queries(
        policy, strict, loose, 1, tmp_path / "queries", tmp_path / "candidates", None
    )

    assert report["query_counts"] == {"real": 0, "clean": 1}
    clean = pd.read_parquet(tmp_path / "queries/clean_queries.parquet")
    assert clean.reason.tolist() == ["background_fp"]
    assert clean.defect.tolist() == ["unknown"]


def test_queries_reject_real_gap_without_defect_pocket(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    query_image = tmp_path / "defect.png"
    _image(query_image)
    document = yaml.safe_load(policy.read_text())
    kpi = tmp_path / "kpi.json"
    kpi.write_text(json.dumps({
        "images": [{"id": 1, "file_name": query_image.name,
                    "source_path": str(query_image),
                    "deft_od_aoi": {"benchmark": "visa", "texture": "pcb1"}}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [4, 4, 12, 12]}],
        "categories": [{"id": 1, "name": "defect"}],
    }))
    document["sources"]["kpi"] = {"images": str(tmp_path), "coco": str(kpi)}
    policy.write_text(yaml.safe_dump(document))
    MODULE.candidates(policy, tmp_path / "candidates")
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 12, 12], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame(columns=["filepath", "gap_type", "bbox", "best_iou"]).to_parquet(loose)

    with pytest.raises(ValueError, match="lacks frozen pocket metadata"):
        MODULE.queries(
            policy, strict, loose, 1, tmp_path / "queries", tmp_path / "candidates", None
        )


def test_queries_exclude_crops_from_previously_admitted_sources(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    query_image = tmp_path / "query.png"
    _image(query_image)
    document = yaml.safe_load(policy.read_text())
    kpi = tmp_path / "kpi.json"
    kpi.write_text(json.dumps({
        "images": [{"id": 1, "file_name": query_image.name,
                    "source_path": str(query_image),
                    "deft_od_aoi": {"benchmark": "visa", "texture": "pcb1",
                                    "defect_type": "bad"}}],
        "annotations": [], "categories": [{"id": 1, "name": "defect"}],
    }))
    document["sources"]["kpi"] = {"images": str(tmp_path), "coco": str(kpi)}
    policy.write_text(yaml.safe_dump(document))
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.1}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [2, 2, 10, 10], "best_iou": 0.01}]).to_parquet(loose)

    candidates = tmp_path / "candidates"
    candidates.mkdir()
    prior_real = tmp_path / "prior_real.png"
    novel_real = tmp_path / "novel_real.png"
    prior_clean = tmp_path / "prior_clean.png"
    novel_clean = tmp_path / "novel_clean.png"
    for path in (prior_real, novel_real, prior_clean, novel_clean):
        _image(path)
    pd.DataFrame([
        {"filepath": str(candidates / "real-prior-a.png"),
         "source_filepath": str(prior_real)},
        {"filepath": str(candidates / "real-prior-b.png"),
         "source_filepath": str(prior_real)},
        {"filepath": str(candidates / "real-novel.png"),
         "source_filepath": str(novel_real)},
    ]).to_parquet(candidates / "real_candidates.parquet", index=False)
    pd.DataFrame([
        {"filepath": str(candidates / "clean-prior.png"),
         "source_filepath": str(prior_clean)},
        {"filepath": str(candidates / "clean-novel.png"),
         "source_filepath": str(novel_clean)},
    ]).to_parquet(candidates / "clean_candidates.parquet", index=False)
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"images": [
        {"deft_kind": "real_defect", "original_source_path": str(prior_real)},
        {"deft_kind": "clean_negative", "source_path": str(prior_clean)},
        {"deft_kind": "synthetic_defect", "source_path": str(tmp_path / "synthetic.png")},
    ]}))

    report = MODULE.queries(policy, strict, loose, 4, tmp_path / "queries",
                            candidates, None, previous)

    assert report["excluded_candidate_crops"] == {"real": 2, "clean": 1}
    assert report["excluded_source_images"] == {"real": 1, "clean": 1}
    real_spec = yaml.safe_load((tmp_path / "queries/mine_real.yaml").read_text())
    clean_spec = yaml.safe_load((tmp_path / "queries/mine_clean.yaml").read_text())
    real_excluded = pd.read_parquet(real_spec["exclude_path"])
    clean_excluded = pd.read_parquet(clean_spec["exclude_path"])
    assert set(real_excluded.filepath) == {
        str(candidates / "real-prior-a.png"), str(candidates / "real-prior-b.png")
    }
    assert set(clean_excluded.filepath) == {str(candidates / "clean-prior.png")}
