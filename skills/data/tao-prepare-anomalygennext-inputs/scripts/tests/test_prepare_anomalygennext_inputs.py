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


SCRIPT = Path(__file__).parents[1] / "prepare_anomalygennext_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_anomalygennext_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def image(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def mask(path: Path, offset: int = 0) -> None:
    value = np.zeros((32, 32), dtype=np.uint8)
    value[4 + offset:20 + offset, 4:20] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    defective = tmp_path / "defect.png"
    source_mask = tmp_path / "defect_mask.png"
    image(defective)
    mask(source_mask)
    pool = tmp_path / "pool" / "texture_1"
    image(pool / "clean_image" / "clean1.png", 10)
    image(pool / "clean_image" / "clean2.png", 20)
    mask(pool / "mask" / "crack" / "mask1.png")
    mask(pool / "mask" / "crack" / "mask2.png", 1)
    gaps = tmp_path / "gaps.parquet"
    pd.DataFrame([
        {"image_id": 1, "filepath": str(defective), "gap_type": "FN",
         "bbox": [4, 4, 20, 20], "class": "defect", "split": "kpi",
         "dataset_id": "example", "texture_id": "texture_1",
         "defect_class": "crack", "anomaly_type": "texture_1+crack",
         "fn_mask_source": str(source_mask)},
        {"image_id": 1, "filepath": str(defective), "gap_type": "FN",
         "bbox": [6, 6, 18, 18], "class": "defect", "split": "kpi",
         "dataset_id": "example", "texture_id": "texture_1",
         "defect_class": "crack", "anomaly_type": "texture_1+crack",
         "fn_mask_source": str(source_mask)},
    ]).to_parquet(gaps)
    checkpoint = tmp_path / "adapter.pt"
    checkpoint.write_bytes(b"checkpoint")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"anomaly_types": [["texture_1", "crack"]]}))
    defect_spec = tmp_path / "defect_spec.jsonl"
    defect_spec.write_text(json.dumps({"defect_type": "texture_1+crack",
                                       "spatial_dependency": "free"}) + "\n")
    config = tmp_path / "filtering.yaml"
    config.write_text(yaml.safe_dump({
        "source_tag": "test", "gap_parquet": str(gaps),
        "pool_dataset_root": str(tmp_path / "pool"), "defect_spec": str(defect_spec),
        "datasets": {"example": {"checkpoint": str(checkpoint), "recipe": str(recipe)}},
        "selection": {"mode": "all_eligible", "datasets": ["example"],
                      "mask_sample_seed": 7},
        "embedding": {"model": "SigLIP", "model_path": "local/siglip", "batch_size": 8},
    }, sort_keys=False))
    return config, gaps


def test_prepare_preserves_box_level_queries_and_unique_embedding_input(tmp_path: Path) -> None:
    config, _ = fixture(tmp_path)
    output = tmp_path / "output"
    report = MODULE.prepare(config, output)
    assert report["status"] == "COMPLETE"
    assert report["source_tag"] == "test"
    assert report["selected_fn_count"] == 2
    assert report["clean_image_count"] == 2
    assert report["source_mask_count"] == 4
    assert report["training_pool_mutated"] is False
    queries = pd.read_parquet(output / "manifests" / "selected_fn_queries.parquet")
    embedding = pd.read_parquet(output / "manifests" / "fn_embedding_inputs.parquet")
    masks = pd.read_parquet(output / "manifests" / "mask_selection.parquet")
    assert len(queries) == 2 and queries.fn_id.nunique() == 2
    assert len(embedding) == 1
    assert masks.groupby("fn_id").branch.nunique().eq(2).all()
    assert all(Path(path).is_file() for path in masks.mask_path)
    clean_spec = yaml.safe_load((output / "specs" / "clean_embeddings.yaml").read_text())
    fn_spec = yaml.safe_load((output / "specs" / "fn_embeddings.yaml").read_text())
    assert clean_spec["model_path"] == fn_spec["model_path"] == "local/siglip"


def test_typed_pool_root_is_frozen_for_both_embedding_specs(tmp_path: Path) -> None:
    config, _ = fixture(tmp_path)
    configured_pool = tmp_path / "pool"
    staged_pool = tmp_path / "staged/pool"
    staged_pool.parent.mkdir()
    configured_pool.rename(staged_pool)
    output = tmp_path / "output"

    MODULE.prepare(config, output, staged_pool)

    frozen = yaml.safe_load(
        (output / "prepared_anomalygennext_inputs/filtering_config.yaml").read_text()
    )
    assert frozen["pool_dataset_root"] == str(staged_pool.resolve())
    clean = pd.read_parquet(output / "manifests/clean_pool.parquet")
    assert all(Path(path).is_relative_to(staged_pool.resolve()) for path in clean.filepath)
    contract = json.loads(
        (output / "prepared_anomalygennext_inputs/input_contract.json").read_text()
    )
    expected_mount = {
        "type": "folder", "compute_path": str(staged_pool.resolve()), "read_only": True
    }
    assert contract["downstream_inputs"] == {
        "clean_embeddings": {"pool_dataset_root": expected_mount},
        "run_amp": {"pool_dataset_root": expected_mount},
    }
    for name in ("clean", "fn"):
        spec = yaml.safe_load((output / f"specs/{name}_embeddings.yaml").read_text())
        assert "pool_dataset_root" not in spec
        assert Path(spec["input_parquet"]).is_relative_to(output.resolve())
        assert Path(spec["output_parquet"]).is_relative_to(output.resolve())
        inputs = pd.read_parquet(spec["input_parquet"])
        assert all(Path(path).is_file() for path in inputs.filepath)


def test_action_contract_declares_pool_folder_for_prepare_and_amp() -> None:
    info = yaml.safe_load(
        (Path(__file__).parents[2] / "references/skill_info.yaml").read_text()
    )
    for action in ("prepare_plan", "run_amp"):
        assert info["actions"][action]["inputs"]["pool_dataset_root"] == {"type": "folder"}
        assert "{pool_dataset_root}" in info["actions"][action]["args"]["pool_dataset_root"]
    assert "config" not in info["actions"]["run_amp"]["inputs"]
    assert "config" not in info["actions"]["run_amp"]["args"]


def test_prepare_requires_normalized_identity(tmp_path: Path) -> None:
    config, gaps = fixture(tmp_path)
    frame = pd.read_parquet(gaps).drop(columns=["fn_mask_source"])
    frame.to_parquet(gaps)
    with pytest.raises(ValueError, match="normalized columns"):
        MODULE.prepare(config, tmp_path / "output")


def test_prepare_refuses_output_reuse(tmp_path: Path) -> None:
    config, _ = fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        MODULE.prepare(config, output)
