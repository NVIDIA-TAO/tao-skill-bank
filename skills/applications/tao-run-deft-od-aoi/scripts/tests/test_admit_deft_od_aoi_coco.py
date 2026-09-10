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
                                      "routing": {"clean_cumulative_cap_per_real": 1.0}}))
    (retrieval_root / "query_manifest.json").write_text(
        json.dumps({"iteration": 1, "enabled_roles": ["real", "clean"]})
    )
    return policy, candidate_root, retrieval_root


def test_admission_deduplicates_sources_and_preserves_explicit_clean(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")
    assert report["admitted"] == {"real": 1, "clean": 1}
    assert report["by_kind"] == {"real_defect": 1, "clean_negative": 1}
    coco = json.loads((tmp_path / "out/train.json").read_text())
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 1
    clean_id = next(row["id"] for row in coco["images"] if row["deft_kind"] == "clean_negative")
    assert all(row["image_id"] != clean_id for row in coco["annotations"])


def test_admission_rejects_empty_enabled_result_after_similarity_gate(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path, similarity=0.0)
    with pytest.raises(ValueError, match="admitted no source images"):
        MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")
