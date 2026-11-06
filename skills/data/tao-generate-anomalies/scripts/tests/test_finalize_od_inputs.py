# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import pytest
from PIL import Image

SCRIPT = Path(__file__).parents[1] / "finalize_od_inputs.py"
SPEC = importlib.util.spec_from_file_location("finalize_od_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _image(path: Path, full: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.startswith("clean"):
        value = np.full((16, 16, 3), 80, dtype=np.uint8)
    else:
        value = np.full((16, 16), 255 if full else 0, dtype=np.uint8)
        if not full:
            value[3:9, 4:10] = 255
    Image.fromarray(value).save(path)


def _fixture(root: Path, reject_first: bool = False) -> None:
    manifests = root / "manifests"
    prepared = root / "prepared_anomalygen_inputs"
    manifests.mkdir(parents=True)
    prepared.mkdir()
    checkpoint, recipe = root / "model.pt", root / "recipe.yaml"
    checkpoint.write_bytes(b"model")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    config = {
        "source_tag": "test",
        "pool_dataset_root": str(root / "pool"),
        "datasets": {"dataset": {"checkpoint": str(checkpoint), "recipe": str(recipe)}},
        "retrieval": {"max_neighbors_per_fn": 1},
    }
    (prepared / "filtering_config.yaml").write_text(yaml.safe_dump(config))
    (prepared / "input_contract.json").write_text('{"status":"COMPLETE"}\n')
    clean_a, clean_b = root / "clean-a.png", root / "clean-b.png"
    _image(clean_a)
    _image(clean_b)
    rows = []
    for order, fn_id in enumerate(("fn-1", "fn-2")):
        for rank, clean in enumerate((clean_a, clean_b), start=1):
            rows.append(
                {
                    "candidate_id": f"{fn_id}-pair-{rank}",
                    "fn_id": fn_id,
                    "query_order": order,
                    "dataset_id": "dataset",
                    "anomaly_type": "texture+crack",
                    "od_category": "defect",
                    "fn_filepath": str(root / "defect.png"),
                    "clean_filepath": str(clean),
                    "neighbor_rank": rank,
                    "cosine_similarity": 1.0 - rank / 10,
                    "eligible_for_amp": True,
                    "gate_reason": "",
                }
            )
    pd.DataFrame(rows).to_parquet(manifests / "knn_candidates.parquet")
    mask_rows, amp_rows = [], []
    for fn_id in ("fn-1", "fn-2"):
        for branch in sorted(MODULE.BRANCHES):
            source = root / "source" / f"{fn_id}__{branch}.png"
            _image(source)
            mask_rows.append(
                {"fn_id": fn_id, "branch": branch, "mask_path": str(source),
                 "mask_sha256": MODULE._sha256(source), "source_mask": str(source),
                 "source_mask_sha256": MODULE._sha256(source)}
            )
            for clean in (clean_a, clean_b):
                aligned = root / "amp" / clean.stem / f"{source.stem}__seed0.png"
                _image(
                    aligned, full=reject_first and fn_id == "fn-1" and clean == clean_a
                )
                amp_rows.append(
                    {
                        "image_filename": str(clean),
                        "mask_filename": str(aligned),
                        "anomaly_type": "texture+crack",
                    }
                )
    pd.DataFrame(mask_rows).to_parquet(manifests / "mask_selection.parquet")
    (root / "amp" / "testcase.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in amp_rows)
    )


def test_finalize_preserves_same_clean_across_false_negatives(tmp_path: Path) -> None:
    _fixture(tmp_path)
    report = MODULE.finalize(tmp_path)
    assert report["selected_fn_count"] == 2
    assert report["selected_pair_count"] == 2
    assert report["generator_row_count"] == 4
    selected = pd.read_parquet(tmp_path / "manifests" / "selected_pairs.parquet")
    assert selected.clean_filepath.nunique() == 1
    assert report["training_pool_mutated"] is False
    assert all(Path(row["path"]).is_file() for row in report["artifacts"])
    assert all(
        MODULE._sha256(Path(row["path"])) == row["sha256"]
        for row in report["artifacts"]
    )


def test_finalize_rejects_full_image_mask_and_uses_next_neighbor(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path, reject_first=True)
    MODULE.finalize(tmp_path)
    selected = pd.read_parquet(tmp_path / "manifests" / "selected_pairs.parquet")
    first = selected[selected.fn_id == "fn-1"].iloc[0]
    assert Path(first.clean_filepath).name == "clean-b.png"
    status = pd.read_parquet(tmp_path / "manifests" / "knn_roi_status.parquet")
    rejected = status[(status.fn_id == "fn-1") & (status.neighbor_rank == 1)].iloc[0]
    assert rejected.selection_reason.endswith(":full_image")


def test_finalize_rejects_excluded_stem_reintroduced_in_candidates(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    frozen = tmp_path / "prepared_anomalygen_inputs/filtering_config.yaml"
    config = yaml.safe_load(frozen.read_text())
    config["excluded_stems"] = ["clean-a"]
    frozen.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="excluded"):
        MODULE.finalize(tmp_path)


def _real_prepare_with_test_only_placement(root):
    """Real producer inputs/outputs; copy masks as TEST_ONLY placement, no AMP."""
    import shutil
    spec = importlib.util.spec_from_file_location("od_producer_fixture", Path(__file__).with_name("test_prepare_od_inputs.py"))
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    config, _ = fixture.fixture(root)
    prepared = root / "prepared"
    fixture.MODULE.prepare(config, prepared)
    queries = pd.read_parquet(prepared / "manifests/selected_fn_queries.parquet")
    masks = pd.read_parquet(prepared / "manifests/mask_selection.parquet")
    clean = root / "pool/texture_1/clean_image/clean1.png"
    candidates, placement = [], []
    for i, query in enumerate(queries.to_dict("records")):
        candidates.append({"candidate_id": f"pair-{i}", "fn_id": query["fn_id"], "query_order": i, "dataset_id": query["dataset_id"], "anomaly_type": query["anomaly_type"], "od_category": query["od_category"], "fn_filepath": query["filepath"], "clean_filepath": str(clean), "neighbor_rank": 1, "cosine_similarity": 1.0, "eligible_for_amp": True, "gate_reason": ""})
        for mask in masks[masks.fn_id == query["fn_id"]].to_dict("records"):
            source = Path(mask["mask_path"])
            aligned = prepared / "amp" / f"{source.stem}__seed0.png"
            aligned.parent.mkdir(exist_ok=True)
            shutil.copy2(source, aligned)
            placement.append({"image_filename": str(clean), "mask_filename": str(aligned), "anomaly_type": query["anomaly_type"]})
    pd.DataFrame(candidates).to_parquet(prepared / "manifests/knn_candidates.parquet")
    (prepared / "amp/testcase.jsonl").write_text("".join(json.dumps(row) + "\n" for row in placement))
    return prepared


@pytest.mark.parametrize("original_stem", ["defect_mask", "mask1", "mask2"])
def test_real_producer_chain_preserves_original_donor_exclusions(tmp_path, original_stem):
    from anomalygen_consumer import validate_prepared_ancestry
    prepared = _real_prepare_with_test_only_placement(tmp_path)
    report = MODULE.finalize(prepared)
    manifest = prepared / "prepared_anomalygen_inputs/prepared_inputs_manifest.json"
    binding = {"manifest": str(manifest), "manifest_sha256": MODULE._sha256(manifest), "dataset_id": "example"}
    validate_prepared_ancestry(binding)
    excluded = tmp_path / "excluded.txt"; excluded.write_text(original_stem + "\n")
    with pytest.raises(ValueError, match="excluded"):
        validate_prepared_ancestry(binding, exclude_stems=excluded)
    provenance = [json.loads(line) for line in Path(report["generator_groups"][0]["provenance"]).read_text().splitlines()]
    assert {Path(row["source_mask"]).stem for row in provenance} == {"defect_mask", "mask1", "mask2"}
    for row in provenance:
        assert len({row["source_mask"], row["copied_mask"], row["aligned_mask"]}) == 3
        for key in ("source_mask", "copied_mask", "aligned_mask"):
            assert MODULE._sha256(Path(row[key])) == row[key + "_sha256"]


@pytest.mark.parametrize("source", ["original", "copied"])
def test_real_finalizer_rejects_mask_mutation_after_preparation(tmp_path, source):
    prepared = _real_prepare_with_test_only_placement(tmp_path)
    path = tmp_path / "defect_mask.png" if source == "original" else Path(pd.read_parquet(prepared / "manifests/mask_selection.parquet").iloc[0].mask_path)
    path.write_bytes(b"TEST_ONLY_CHANGED_MASK")
    with pytest.raises(ValueError, match="mask.*changed"):
        MODULE.finalize(prepared)


def test_real_consumer_rejects_original_donor_mutation_after_finalization(tmp_path):
    from anomalygen_consumer import validate_prepared_ancestry
    prepared = _real_prepare_with_test_only_placement(tmp_path)
    MODULE.finalize(prepared)
    manifest = prepared / "prepared_anomalygen_inputs/prepared_inputs_manifest.json"
    binding = {"manifest": str(manifest), "manifest_sha256": MODULE._sha256(manifest), "dataset_id": "example"}
    (tmp_path / "defect_mask.png").write_bytes(b"TEST_ONLY_CHANGED_MASK")
    with pytest.raises(ValueError, match="mask changed"):
        validate_prepared_ancestry(binding)
