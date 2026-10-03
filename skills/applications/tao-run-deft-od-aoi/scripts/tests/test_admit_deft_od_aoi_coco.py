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
        reason = "fn" if role == "real" else "background_fp"
        pd.DataFrame([{"filepath": f"/{role}-query.png", "embedding": [1.0, 0.0],
                       "reason": reason}]).to_parquet(
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
        json.dumps({"iteration": 1, "enabled_roles": ["real", "clean"],
                    "admission_targets": {"real": {"fn": 1, "near_miss_fp": 0},
                                          "clean": {"background_fp": 1}},
                    "requested_crop_counts": {"real": 15, "clean": 15}})
    )
    return policy, candidate_root, retrieval_root


def test_admission_deduplicates_sources_and_preserves_explicit_clean(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")
    assert report["admitted"] == {"real": 1, "clean": 1, "synthetic": 0}
    assert report["by_kind"] == {"real_defect": 1, "clean_negative": 1,
                                 "synthetic_defect": 0}
    preview = json.loads((tmp_path / "out/admission_preview.json").read_text())
    assert preview["roles"]["real"]["branches"]["fn"] == {
        "desired_parents": 1, "mined_crops": 1, "unique_parents": 1,
        "novel_parents": 1, "selected_parents": 1, "shortfall_parents": 0,
        "quota_met": True,
    }
    assert preview["roles"]["real"]["per_dataset"] == {"unknown": 1}
    coco = json.loads((tmp_path / "out/train.json").read_text())
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 1
    clean_id = next(row["id"] for row in coco["images"] if row["deft_kind"] == "clean_negative")
    assert all(row["image_id"] != clean_id for row in coco["annotations"])


def test_clean_admission_uses_cumulative_real_capacity_after_real_mining_exhausts(
        tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    extra_clean = tmp_path / "clean-2.png"
    extra_clean.write_bytes(b"clean-2")
    clean_coco = tmp_path / "clean.json"
    clean_document = json.loads(clean_coco.read_text())
    clean_document["images"].append({
        "id": 2, "file_name": extra_clean.name, "source_path": str(extra_clean)})
    clean_coco.write_text(json.dumps(clean_document))
    clean_embeddings = candidates / "clean_candidate_embeddings.parquet"
    embedded = pd.read_parquet(clean_embeddings)
    embedded.loc[len(embedded)] = {
        "filepath": "/clean-crop-2.png", "source_filepath": str(extra_clean),
        "source_image_id": 2, "embedding": [1.0, 0.0],
    }
    embedded.to_parquet(clean_embeddings, index=False)
    mined = retrieval / "mine_clean/final_unique_files.parquet"
    pd.DataFrame({"filepath": ["/clean-crop.png", "/clean-crop-2.png"]}).to_parquet(
        mined, index=False)
    (retrieval / "query_manifest.json").write_text(
        json.dumps({"iteration": 2, "enabled_roles": ["clean"]})
    )
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "real.png",
                    "source_path": str(tmp_path / "real.png"),
                    "width": 16, "height": 16, "deft_kind": "real_defect"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [1, 1, 4, 4], "area": 16}],
        "categories": [{"id": 1, "name": "defect"}],
    }))

    report = MODULE.admit(
        policy, candidates, retrieval, tmp_path / "out", previous, "copy")

    assert len(pd.read_parquet(mined)) == 2
    assert report["admitted"] == {"real": 0, "clean": 1, "synthetic": 0}
    assert report["by_kind"] == {"real_defect": 1, "clean_negative": 1,
                                 "synthetic_defect": 0}


def test_admission_rejects_empty_enabled_result_after_similarity_gate(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path, similarity=0.0)
    with pytest.raises(ValueError, match="mining admitted no source images"):
        MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")


def test_admission_reports_parent_shortfall_without_failing(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    manifest = json.loads((retrieval / "query_manifest.json").read_text())
    manifest["admission_targets"]["real"]["fn"] = 2
    (retrieval / "query_manifest.json").write_text(json.dumps(manifest))

    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")

    assert report["admitted"]["real"] == 1
    preview = json.loads((tmp_path / "out/admission_preview.json").read_text())
    assert preview["roles"]["real"]["branches"]["fn"] == {
        "desired_parents": 2, "mined_crops": 1, "unique_parents": 1,
        "novel_parents": 1, "selected_parents": 1, "shortfall_parents": 1,
        "quota_met": False,
    }


def test_admission_uses_overfetch_to_replace_a_previously_used_parent(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    first, second = tmp_path / "real.png", tmp_path / "real-second.png"
    second.write_bytes(b"second")
    document = json.loads((tmp_path / "real.json").read_text())
    document["images"].append({"id": 2, "file_name": second.name, "source_path": str(second),
                               "deft_od_aoi": {
                                   "dataset_id": "canonical-dataset-b",
                                    "benchmark": "source-dataset-b",
                               }})
    document["annotations"].append({"id": 6, "image_id": 2, "category_id": 1,
                                    "bbox": [1, 1, 4, 4], "area": 16})
    (tmp_path / "real.json").write_text(json.dumps(document))
    pd.DataFrame([
        {"filepath": "/real-crop-a.png", "source_filepath": str(first),
         "source_image_id": 1, "embedding": [1.0, 0.0]},
        {"filepath": "/real-crop-b.png", "source_filepath": str(second),
         "source_image_id": 2, "embedding": [0.9, 0.1]},
    ]).to_parquet(candidates / "real_candidate_embeddings.parquet")
    pd.DataFrame([{"filepath": "/real-crop-a.png"},
                  {"filepath": "/real-crop-b.png"}]).to_parquet(
        retrieval / "mine_real/final_unique_files.parquet"
    )
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": [{"id": 1, "file_name": first.name, "source_path": str(first),
                    "deft_kind": "real_defect"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [1, 1, 4, 4]}],
        "categories": [{"id": 1, "name": "defect"}],
    }))

    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", previous, "copy")

    assert report["admitted"]["real"] == 1
    preview = json.loads((tmp_path / "out/admission_preview.json").read_text())
    assert preview["roles"]["real"]["branches"]["fn"]["unique_parents"] == 2
    assert preview["roles"]["real"]["branches"]["fn"]["novel_parents"] == 1
    assert preview["roles"]["real"]["per_dataset"] == {"canonical-dataset-b": 1}


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


def test_admission_resolves_binary_coco_from_declared_generation_output(tmp_path: Path) -> None:
    root = tmp_path / "generation"
    relative = "pseudo_labels/coco_annotations_od_defect.json"
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_text('{"images": [], "annotations": [], "categories": []}\n')

    assert MODULE._generation_output(root, "binary_coco") == target.resolve()

    target.unlink()
    with pytest.raises(FileNotFoundError, match="declared generation output binary_coco"):
        MODULE._generation_output(root, "binary_coco")


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
                          synthetic, generated, synthetic_only=True)

    admission = report["synthetic_admission"]
    assert admission["quality_filter"]["rejected_annotations_full_frame"] == 1
    assert admission["quality_filter"]["rejected_annotations_small"] == 1
    assert admission["requested_new"] == 4
    assert admission["admitted_new"] == 2
    assert admission["admitted_by_stratum"] == {"line-a": 1, "line-b": 1}
    preview = json.loads((tmp_path / "out/admission_preview.json").read_text())
    assert preview["mining_admission"] == "skipped" and preview["roles"] == {}


def test_synthetic_inputs_do_not_implicitly_disable_mining(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "old.png", "source_path": str(tmp_path / "old.png"),
                    "deft_kind": "synthetic_defect"}],
        "annotations": [], "categories": [{"id": 1, "name": "defect"}],
    }))
    (tmp_path / "old.png").write_bytes(b"old")
    generated = tmp_path / "generated"
    generated.mkdir()
    image = generated / "synthetic.png"
    image.write_bytes(b"synthetic")
    synthetic = tmp_path / "synthetic.json"
    synthetic.write_text(json.dumps({
        "images": [{"id": 2, "file_name": image.name, "width": 16, "height": 16}],
        "annotations": [{"id": 2, "image_id": 2, "category_id": 1,
                         "bbox": [1, 1, 3, 3]}],
        "categories": [{"id": 1, "name": "defect"}],
    }))

    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", previous, "copy",
                          synthetic, generated)

    assert report["admitted"]["real"] == 1
    preview = json.loads((tmp_path / "out/admission_preview.json").read_text())
    assert preview["mining_admission"] == "evaluated"


def test_admission_reports_missing_selected_source_cleanly(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    frame = pd.read_parquet(candidates / "real_candidate_embeddings.parquet")
    frame["source_filepath"] = str(tmp_path / "not-in-frozen-coco.png")
    frame.to_parquet(candidates / "real_candidate_embeddings.parquet", index=False)

    with pytest.raises(ValueError, match="selected real source is absent from its frozen COCO"):
        MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")


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


def test_total_fraction_limit_matches_historical_definition() -> None:
    assert MODULE._synthetic_limit(
        2828, {"cumulative_fraction_of_total_defects": 0.25}
    ) == (942, "fraction_of_total", 0.25)
    assert MODULE._synthetic_limit(
        2828, {"cumulative_fraction_of_real_defects": 0.25}
    ) == (707, "fraction_of_real", 0.25)
