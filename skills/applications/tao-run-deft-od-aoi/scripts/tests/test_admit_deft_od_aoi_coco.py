# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "admit_deft_od_aoi_coco.py"
SPEC = importlib.util.spec_from_file_location("admit_deft_od_aoi_coco", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _fixture(root: Path, similarity: float = 1.0) -> tuple[Path, Path, Path]:
    candidate_root, retrieval_root = root / "candidates", root / "retrieval"
    candidate_root.mkdir()
    retrieval_root.mkdir()
    sources = {}
    for role in ("real", "clean"):
        image = root / f"{role}.png"
        image.write_bytes(role.encode())
        annotations = ([{"id": 5, "image_id": 1, "category_id": 1,
                         "bbox": [1, 1, 4, 4], "area": 16}] if role == "real" else [])
        coco = root / f"{role}.json"
        coco.write_text(json.dumps({"images": [{"id": 1, "file_name": image.name,
                                                 "source_path": str(image)}],
                                    "annotations": annotations,
                                    "categories": [{"id": 1, "name": "defect"}]}))
        sources[role] = {"images": str(root), "coco": str(coco)}
        crop = f"/{role}-crop.png"
        pd.DataFrame([{"filepath": crop, "source_filepath": str(image),
                       "source_image_id": 1, "embedding": [similarity, 1.0 - similarity]}]).to_parquet(
            candidate_root / f"{role}_candidate_embeddings.parquet"
        )
        pd.DataFrame([{"filepath": f"/{role}-query.png", "embedding": [1.0, 0.0]}]).to_parquet(
            retrieval_root / f"{role}_query_embeddings.parquet"
        )
        mine = retrieval_root / f"mine_{role}"
        mine.mkdir()
        pd.DataFrame([{"filepath": crop}]).to_parquet(mine / "final_unique_files.parquet")
    policy = root / "policy.yaml"
    policy.write_text(yaml.safe_dump({"sources": sources,
                                      "retrieval": {"minimum_similarity": 0.5},
                                      "routing": {"clean_cumulative_cap_per_real": 1.0},
                                      "admission": {"minimum_box_area_px": 4,
                                                    "maximum_box_aspect": 25.0},
                                      "synthesis": {"cumulative_fraction_of_real_defects": 1.0}}))
    (retrieval_root / "query_manifest.json").write_text(
        json.dumps({"iteration": 1, "enabled_roles": ["real", "clean"]})
    )
    return policy, candidate_root, retrieval_root


def test_admission_deduplicates_sources_and_preserves_explicit_clean(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")
    assert report["admitted"] == {"real": 1, "clean": 1, "synthetic": 0}
    assert report["by_kind"] == {"real_defect": 1, "clean_negative": 1,
                                 "synthetic_defect": 0}
    coco = json.loads((tmp_path / "out/train.json").read_text())
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 1
    clean_id = next(row["id"] for row in coco["images"] if row["deft_kind"] == "clean_negative")
    assert all(row["image_id"] != clean_id for row in coco["annotations"])


def test_admission_rejects_empty_enabled_result_after_similarity_gate(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path, similarity=0.0)
    with pytest.raises(ValueError, match="admitted no source images"):
        MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")


def test_admission_folds_capped_synthetic_categories_to_defect(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    image = generated / "synthetic.png"
    image.write_bytes(b"synthetic")
    coco = tmp_path / "synthetic.json"
    coco.write_text(json.dumps({"images": [{"id": 2, "file_name": image.name,
                                             "width": 16, "height": 16}],
                                "annotations": [{"id": 8, "image_id": 2, "category_id": 4,
                                                 "bbox": [1, 1, 3, 3]}],
                                "categories": [{"id": 4, "name": "texture+defect"}]}))
    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy",
                          coco, generated)
    assert report["admitted"]["synthetic"] == 1
    output = json.loads((tmp_path / "out/train.json").read_text())
    assert {row["category_id"] for row in output["annotations"]} == {1}


def test_synthetic_quality_filter_and_proportional_allocation(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    extra = tmp_path / "extra-real.png"
    extra.write_bytes(b"extra")
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": "real.png", "source_path": str(tmp_path / "real.png"),
             "width": 16, "height": 16, "deft_kind": "real_defect"},
            {"id": 2, "file_name": extra.name, "source_path": str(extra),
             "width": 16, "height": 16, "deft_kind": "real_defect"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 4, 4]},
        ],
        "categories": [{"id": 1, "name": "defect"}],
    }))
    generated = tmp_path / "generated"
    generated.mkdir()
    images, annotations = [], []
    for index in range(6):
        name = f"synthetic-{index}.png"
        (generated / name).write_bytes(name.encode())
        images.append({"id": index + 1, "file_name": name, "width": 20, "height": 20,
                       "dataset_id": "line-a" if index < 3 else "line-b"})
        bbox = ([0, 0, 20, 20] if index == 0 else
                [1, 1, 1, 1] if index == 3 else [2, 2, 8, 8])
        annotations.append({"id": index + 1, "image_id": index + 1,
                            "category_id": 7, "bbox": bbox})
    synthetic = tmp_path / "synthetic.json"
    synthetic.write_text(json.dumps({"images": images, "annotations": annotations,
                                     "categories": [{"id": 7, "name": "defect-variant"}]}))

    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", previous, "copy",
                          synthetic, generated)

    admission = report["synthetic_admission"]
    assert admission["quality_filter"]["rejected_annotations_full_frame"] == 1
    assert admission["quality_filter"]["rejected_annotations_small"] == 1
    assert admission["requested_new"] == 4
    assert admission["admitted_new"] == 2
    assert admission["admitted_by_stratum"] == {"line-a": 1, "line-b": 1}


def test_synthetic_cap_selection_is_independent_of_coco_order(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    images, annotations = [], []
    for index, name in enumerate(("z.png", "a.png", "m.png"), start=1):
        (generated / name).write_bytes(name.encode())
        images.append({"id": index, "file_name": name, "width": 16, "height": 16})
        annotations.append({"id": index, "image_id": index, "category_id": 1,
                            "bbox": [2, 2, 8, 8]})
    synthetic = tmp_path / "synthetic.json"

    def run(order: list[int], output: Path) -> str:
        synthetic.write_text(json.dumps({
            "images": [images[index] for index in order],
            "annotations": annotations,
            "categories": [{"id": 1, "name": "defect"}],
        }))
        MODULE.admit(policy, candidates, retrieval, output, None, "copy", synthetic, generated)
        coco = json.loads((output / "train.json").read_text())
        return next(row["source_path"] for row in coco["images"]
                    if row["deft_kind"] == "synthetic_defect")

    assert run([0, 1, 2], tmp_path / "out-a") == run([2, 0, 1], tmp_path / "out-b")
