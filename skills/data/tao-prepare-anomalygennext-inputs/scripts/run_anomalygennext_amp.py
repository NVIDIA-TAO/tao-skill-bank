#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build pair-preserving KNN requests and run AnomalyGenNext AMP."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


BRANCHES = {"fn_mask", "same_type_sampled_mask"}


def _matrix(values: pd.Series) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    if not rows or len({row.size for row in rows}) != 1:
        raise ValueError("embeddings are empty or have inconsistent widths")
    matrix = np.stack(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(norms == 0):
        raise ValueError("embeddings contain non-finite or zero-norm rows")
    return matrix / norms


def _pair_id(fn_id: str, clean: str) -> str:
    return "pair-" + hashlib.sha256(f"{fn_id}\0{clean}".encode()).hexdigest()[:16]


def plan(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifests, embeddings = root / "manifests", root / "embeddings"
    clean = pd.read_parquet(embeddings / "clean_embeddings.parquet").reset_index(drop=True)
    embedded = pd.read_parquet(embeddings / "fn_embeddings.parquet")
    queries = pd.read_parquet(manifests / "selected_fn_queries.parquet")
    masks = pd.read_parquet(manifests / "mask_selection.parquet")
    for label, frame in (("clean", clean), ("FN", embedded)):
        if frame.empty or not {"filepath", "embedding"}.issubset(frame.columns):
            raise ValueError(f"{label} embeddings lack filepath/embedding")
    if embedded.filepath.astype(str).duplicated().any():
        raise ValueError("FN embedding rows must be unique by filepath")
    queries = queries.merge(embedded[["filepath", "embedding"]], on="filepath",
                            how="left", validate="many_to_one")
    if queries.embedding.isna().any():
        raise ValueError("one or more FN queries have no embedding")
    clean_vectors, query_vectors = _matrix(clean.embedding), _matrix(queries.embedding)
    retrieval = config["retrieval"]
    if retrieval.get("metric", "cosine") != "cosine":
        raise ValueError("only cosine retrieval is supported")
    topn, floor = int(retrieval["candidate_topn"]), float(retrieval["min_similarity"])
    excluded: set[str] = set()
    exclusion = str(retrieval.get("prior_clean_exclusion_manifest") or "")
    if exclusion:
        excluded = set(pd.read_parquet(exclusion).filepath.astype(str))

    candidates, requests = [], []
    for position, query in queries.reset_index(drop=True).iterrows():
        pool = clean.index[clean.pool_key.astype(str) == str(query.pool_key)].to_numpy()
        if not len(pool):
            raise ValueError(f"no clean embeddings for pool_key={query.pool_key}")
        scores = clean_vectors[pool] @ query_vectors[position]
        mask_rows = masks[masks.fn_id == query.fn_id]
        if set(mask_rows.branch.astype(str)) != BRANCHES or len(mask_rows) != 2:
            raise ValueError(f"FN {query.fn_id} needs exactly two mask branches")
        for rank, local in enumerate(np.argsort(-scores, kind="stable")[:topn], start=1):
            clean_path = str(clean.loc[int(pool[local]), "filepath"])
            score = float(scores[local])
            reason = "below_similarity_floor" if score < floor else (
                "prior_iteration_exclusion" if clean_path in excluded else ""
            )
            pair = _pair_id(str(query.fn_id), clean_path)
            candidate = {
                "candidate_id": pair, "fn_id": str(query.fn_id),
                "query_order": int(query.query_order), "dataset_id": str(query.dataset_id),
                "anomaly_type": str(query.anomaly_type), "od_category": str(query.od_category),
                "pool_key": str(query.pool_key), "fn_filepath": str(query.filepath),
                "clean_filepath": clean_path, "neighbor_rank": rank,
                "cosine_similarity": score, "eligible_for_amp": not reason,
                "gate_reason": reason,
            }
            candidates.append(candidate)
            if not reason:
                for mask in mask_rows.sort_values("branch").itertuples():
                    requests.append({
                        "clean_image": clean_path, "defect_type": str(query.anomaly_type),
                        "submask": str(mask.mask_path), "name": f"{pair}__{mask.branch}",
                        "cad_mask": None, "cad_mask_label": None, "n_seeds": 1,
                        "submask_split_largest": False,
                    })
    if not requests:
        raise ValueError("no candidates passed the AMP gate")
    pd.DataFrame(candidates).to_parquet(manifests / "knn_candidates.parquet", index=False)
    amp = root / "amp"
    amp.mkdir(parents=True, exist_ok=True)
    (amp / "amp_samples.json").write_text(json.dumps(requests, indent=2) + "\n")
    return {"candidates": len(candidates), "amp_rows": len(requests),
            "embedding_dim": int(clean_vectors.shape[1])}


def run(config_path: Path, root: Path) -> dict[str, Any]:
    frozen = root / "prepared_anomalygennext_inputs" / "filtering_config.yaml"
    if config_path.read_bytes() != frozen.read_bytes():
        raise ValueError("config differs from the frozen preparation snapshot")
    config = yaml.safe_load(frozen.read_text())
    report = plan(root, config)
    amp = config.get("amp") or {}
    command = [sys.executable, "-m", "anomalygen.scripts.auto_mask_placement.roi_place",
               "--input_pair_path", str(root / "amp" / "amp_samples.json"),
               "--defect_desc", str(Path(config["defect_spec"]).resolve()),
               "--output_dir", str(root / "amp"), "--n_seeds", "1",
               "--seed", str(int(amp.get("seed", 43))),
               "--model_id", str(amp.get("model_id", "nvidia/Cosmos3-Nano"))]
    subprocess.run(command, check=True)
    testcase = root / "amp" / "testcase.jsonl"
    if not testcase.is_file() or not testcase.read_text().strip():
        raise ValueError("AMP produced no testcase rows")
    report["testcase"] = str(testcase)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root", required=True)
    args = parser.parse_args()
    result = run(Path(args.config).resolve(), Path(args.prepared_root).resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
