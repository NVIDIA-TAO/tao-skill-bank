# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "run_anomalygennext_amp.py"
SPEC = importlib.util.spec_from_file_location("run_anomalygennext_amp", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def inputs(root: Path) -> dict:
    (root / "manifests").mkdir(parents=True)
    (root / "embeddings").mkdir()
    pd.DataFrame([
        {"filepath": "/clean/a.png", "pool_key": "texture_1", "embedding": [1.0, 0.0]},
        {"filepath": "/clean/b.png", "pool_key": "texture_1", "embedding": [0.8, 0.2]},
    ]).to_parquet(root / "embeddings" / "clean_embeddings.parquet")
    pd.DataFrame([
        {"filepath": "/defect/shared.png", "embedding": [1.0, 0.0]},
    ]).to_parquet(root / "embeddings" / "fn_embeddings.parquet")
    pd.DataFrame([
        {"filepath": "/defect/shared.png", "fn_id": "fn-1", "query_order": 0,
         "dataset_id": "d", "pool_key": "texture_1",
         "anomaly_type": "texture_1+crack", "od_category": "defect"},
        {"filepath": "/defect/shared.png", "fn_id": "fn-2", "query_order": 1,
         "dataset_id": "d", "pool_key": "texture_1",
         "anomaly_type": "texture_1+crack", "od_category": "defect"},
    ]).to_parquet(root / "manifests" / "selected_fn_queries.parquet")
    pd.DataFrame([
        {"fn_id": fn, "branch": branch, "mask_path": f"/masks/{fn}-{branch}.png"}
        for fn in ("fn-1", "fn-2")
        for branch in sorted(MODULE.BRANCHES)
    ]).to_parquet(root / "manifests" / "mask_selection.parquet")
    return {"retrieval": {"metric": "cosine", "candidate_topn": 2,
                           "min_similarity": -1.0,
                           "prior_clean_exclusion_manifest": ""}}


def test_plan_preserves_pairs_for_same_source_image(tmp_path: Path) -> None:
    report = MODULE.plan(tmp_path, inputs(tmp_path))
    assert report == {"candidates": 4, "amp_rows": 8, "embedding_dim": 2}
    candidates = pd.read_parquet(tmp_path / "manifests" / "knn_candidates.parquet")
    assert candidates.fn_id.nunique() == 2
    assert candidates.groupby("fn_id").clean_filepath.nunique().eq(2).all()
    requests = json.loads((tmp_path / "amp" / "amp_samples.json").read_text())
    assert len({row["name"] for row in requests}) == 8


def test_plan_rejects_zero_norm_embeddings(tmp_path: Path) -> None:
    config = inputs(tmp_path)
    frame = pd.read_parquet(tmp_path / "embeddings" / "fn_embeddings.parquet")
    frame["embedding"] = pd.Series([[0.0, 0.0]], dtype=object)
    frame.to_parquet(tmp_path / "embeddings" / "fn_embeddings.parquet")
    with pytest.raises(ValueError, match="zero-norm"):
        MODULE.plan(tmp_path, config)
