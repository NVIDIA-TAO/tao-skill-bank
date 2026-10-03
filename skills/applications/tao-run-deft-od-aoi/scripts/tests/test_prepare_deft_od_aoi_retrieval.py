# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image, ImageOps


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
    y, x = np.indices((20, 40), dtype=np.uint8)
    values = np.stack((x, y, x + y), axis=-1)
    Image.fromarray(values).save(path, exif=exif)


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
                                                  "clean_factor": 2}}))
    return policy


def _empty_role(policy: Path, role: str) -> None:
    value = yaml.safe_load(policy.read_text())
    coco = Path(value["sources"][role]["coco"])
    document = json.loads(coco.read_text())
    document["images"] = []
    document["annotations"] = []
    coco.write_text(json.dumps(document))


def test_candidate_cache_uses_defect_crops_and_clean_grid(tmp_path: Path) -> None:
    report = MODULE.candidates(_policy(tmp_path), tmp_path / "candidates")
    assert report["counts"] == {"real": 1, "clean": 5}
    real = pd.read_parquet(tmp_path / "candidates/real_candidates.parquet")
    clean = pd.read_parquet(tmp_path / "candidates/clean_candidates.parquet")
    assert real.source_filepath.nunique() == clean.source_filepath.nunique() == 1
    assert all(Path(path).is_file() for path in list(real.filepath) + list(clean.filepath))


def test_candidate_cache_records_empty_clean_role_without_embedding_spec(
        tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    empty_role = "clean"
    _empty_role(policy, empty_role)

    report = MODULE.candidates(policy, tmp_path / "candidates")

    assert report["counts"][empty_role] == 0
    assert report["role_status"][empty_role] == {
        "status": "EXHAUSTED", "candidate_count": 0, "reason": "empty_source_role"}
    assert not (tmp_path / f"candidates/{empty_role}_candidates.parquet").exists()
    assert not (tmp_path / f"candidates/embed_{empty_role}_candidates.yaml").exists()


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

    assert report["counts"]["real"] == 1
    frame = pd.read_parquet(tmp_path / "candidates/real_candidates.parquet")
    with Image.open(frame.iloc[0].filepath) as crop:
        assert crop.size == (9, 9)


def test_tiny_crop_uses_displayed_geometry_before_minimum_expansion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oriented.png"
    output = tmp_path / "crop.png"
    _oriented_image(source)

    MODULE._crop(source, (1, 30, 3, 32), output)

    with Image.open(source) as opened:
        displayed = ImageOps.exif_transpose(opened).convert("RGB")
    expected = displayed.crop((0, 27, 8, 35))
    with Image.open(output) as crop:
        assert crop.size == (8, 8)
        assert np.array_equal(np.asarray(crop), np.asarray(expected))


def test_queries_route_fn_near_miss_and_background_fp(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    MODULE.candidates(policy, tmp_path / "candidates")
    query_image = tmp_path / "query.png"
    _image(query_image)
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
    real = pd.read_parquet(tmp_path / "queries/real_queries.parquet")
    assert set(real.reason) == {"fn", "near_miss_fp"}
    real_mining = yaml.safe_load((tmp_path / "queries/mine_real.yaml").read_text())
    clean_mining = yaml.safe_load((tmp_path / "queries/mine_clean.yaml").read_text())
    assert real_mining["desired_unique_count"] == 1
    assert clean_mining["desired_unique_count"] == 2


def test_queries_skip_empty_clean_role_while_real_continues(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    empty_role = "clean"
    _empty_role(policy, empty_role)
    MODULE.candidates(policy, tmp_path / "candidates")
    query_image = tmp_path / "query.png"
    _image(query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.01}]).to_parquet(loose)

    report = MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                            tmp_path / "candidates", None)

    continuing_role = "real"
    assert report["enabled_roles"] == [continuing_role]
    assert report["role_status"][empty_role]["status"] == "EXHAUSTED"
    assert {key: report["role_status"][empty_role][key] for key in
            ("candidate_count", "excluded_count", "remaining_candidate_count")} == {
                "candidate_count": 0, "excluded_count": 0, "remaining_candidate_count": 0}
    assert report["role_status"][continuing_role]["status"] == "READY"
    assert not (tmp_path / f"queries/embed_{empty_role}_queries.yaml").exists()
    assert not (tmp_path / f"queries/mine_{empty_role}.yaml").exists()
    assert (tmp_path / f"queries/mine_{continuing_role}.yaml").is_file()


def test_exhausted_role_is_skipped_while_other_role_continues(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    MODULE.candidates(policy, tmp_path / "candidates")
    real_candidates = pd.read_parquet(tmp_path / "candidates/real_candidates.parquet")
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "real.png",
                    "source_path": real_candidates.iloc[0].source_filepath}],
        "annotations": [], "categories": [{"id": 1, "name": "defect"}],
    }))
    query_image = tmp_path / "query.png"
    _image(query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.01}]).to_parquet(loose)

    report = MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                            tmp_path / "candidates", None, previous)

    assert report["enabled_roles"] == ["clean"]
    assert report["role_status"]["real"]["status"] == "EXHAUSTED"
    assert report["role_status"]["clean"]["status"] == "READY"
    assert not (tmp_path / "queries/mine_real.yaml").exists()
    assert (tmp_path / "queries/mine_clean.yaml").is_file()


def test_all_roles_exhausted_emit_convergence_without_mining_specs(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    MODULE.candidates(policy, tmp_path / "candidates")
    sources = []
    for role in ("real", "clean"):
        frame = pd.read_parquet(tmp_path / f"candidates/{role}_candidates.parquet")
        sources.extend(sorted(set(frame.source_filepath.astype(str))))
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"images": [
        {"id": index, "file_name": Path(source).name, "source_path": source}
        for index, source in enumerate(sources, start=1)
    ], "annotations": [], "categories": [{"id": 1, "name": "defect"}]}))
    query_image = tmp_path / "query.png"
    _image(query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.01}]).to_parquet(loose)

    report = MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                            tmp_path / "candidates", None, previous)

    assert report["enabled_roles"] == [] and report["converged"] is True
    assert {value["status"] for value in report["role_status"].values()} == {"EXHAUSTED"}
    assert not list((tmp_path / "queries").glob("mine_*.yaml"))


def test_all_roles_exhausted_do_not_preempt_pending_synthesis(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    value = yaml.safe_load(policy.read_text())
    value["synthesis"] = {"enabled": True}
    policy.write_text(yaml.safe_dump(value))
    MODULE.candidates(policy, tmp_path / "candidates")
    sources = []
    for role in ("real", "clean"):
        frame = pd.read_parquet(tmp_path / f"candidates/{role}_candidates.parquet")
        sources.extend(sorted(set(frame.source_filepath.astype(str))))
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"images": [
        {"id": index, "file_name": Path(source).name, "source_path": source}
        for index, source in enumerate(sources, start=1)
    ]}))
    query_image = tmp_path / "query.png"
    _image(query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame(columns=["filepath", "gap_type", "bbox", "best_iou"]).to_parquet(loose)

    report = MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                            tmp_path / "candidates", None, previous)

    assert report["enabled_roles"] == []
    assert report["synthesis_pending"] is True and report["converged"] is False


def test_queries_reject_missing_candidate_manifest(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    query_image = tmp_path / "query.png"
    _image(query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [4, 4, 20, 20], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame(columns=["filepath", "gap_type", "bbox", "best_iou"]).to_parquet(loose)

    with pytest.raises(FileNotFoundError, match="candidate manifest"):
        MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                       tmp_path / "missing-candidates", None)


@pytest.mark.parametrize(
    ("size", "box", "expected_box", "expected_padding"),
    [
        ((32, 32), (0, 0, 2, 2), (0, 0, 8, 8), (0, 0, 0, 0)),
        ((32, 32), (30, 30, 32, 32), (24, 24, 32, 32), (0, 0, 0, 0)),
        ((5, 3), (0, 0, 1, 1), (0, 0, 5, 3), (3, 4, 0, 1)),
        ((32, 32), (4, 5, 20, 24), (4, 5, 20, 24), (0, 0, 0, 0)),
    ],
)
def test_minimum_crop_geometry_expands_then_pads_only_when_required(
    size: tuple[int, int], box: tuple[int, int, int, int],
    expected_box: tuple[int, int, int, int], expected_padding: tuple[int, int, int, int],
) -> None:
    assert MODULE._minimum_crop_geometry(box, *size) == (expected_box, expected_padding)


def test_tiny_gap_crops_are_embedding_safe_at_boundaries(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    MODULE.candidates(policy, tmp_path / "candidates")
    query_image = tmp_path / "narrow.png"
    Image.fromarray(np.full((3, 5, 3), 80, dtype=np.uint8)).save(query_image)
    strict = tmp_path / "strict.parquet"
    loose = tmp_path / "loose.parquet"
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FN",
                   "bbox": [0, 0, 1, 1], "best_iou": 0.0}]).to_parquet(strict)
    pd.DataFrame([{"filepath": str(query_image), "gap_type": "FP",
                   "bbox": [4, 2, 5, 3], "best_iou": 0.01}]).to_parquet(loose)

    MODULE.queries(policy, strict, loose, 1, tmp_path / "queries",
                   tmp_path / "candidates", None)

    crops = list((tmp_path / "queries/crops").rglob("*.png"))
    assert len(crops) == 2
    assert all(Image.open(path).size == (8, 8) for path in crops)
    assert all(Image.open(path).getextrema() == ((80, 80),) * 3 for path in crops)
