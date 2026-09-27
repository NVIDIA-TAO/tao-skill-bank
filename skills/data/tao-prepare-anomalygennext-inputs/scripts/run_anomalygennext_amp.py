#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build pair-preserving KNN requests and run AnomalyGenNext AMP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image


BRANCHES = {"fn_mask", "same_type_sampled_mask"}
AMP_HF_REPOS = ("Qwen/Qwen3-VL-8B-Instruct",)


def _hf_hub(root: Path) -> Path:
    return root / "hub" if (root / "hub").is_dir() else root


def _validate_checkpoint_root(root: Path, repo: Path) -> Path:
    resolved = root.expanduser().resolve()
    image_root = (repo / "checkpoints").resolve()
    if resolved != image_root:
        raise ValueError(
            f"checkpoint root must be mounted at {repo}/checkpoints: {resolved}"
        )
    hf_home = resolved / "hf"
    hub = _hf_hub(hf_home)
    missing = []
    for name in AMP_HF_REPOS:
        directory = hub / f"models--{name.replace('/', '--')}"
        if not (directory / "blobs").is_dir() or not (directory / "snapshots").is_dir():
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            "checkpoint root lacks required AMP Hugging Face repositories: "
            + ", ".join(missing)
        )
    sam2 = (
        resolved / "facebook" / "sam2.1-hiera-large" / "sam2.1_hiera_large.pt"
    )
    if not sam2.is_file():
        raise FileNotFoundError(f"checkpoint root lacks SAM2.1 checkpoint: {sam2}")
    return hf_home


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


def _validate_amp_mask(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"AMP mask is missing: {path}")
    with Image.open(path) as image:
        mask = np.asarray(image)
        canvas = image.size
    if mask.ndim != 2 or set(np.unique(mask).tolist()) != {0, 255}:
        raise ValueError(f"AMP mask must contain exactly binary values 0 and 255: {path}")
    foreground = mask == 255
    count = int(np.count_nonzero(foreground))
    if count == 0 or count == foreground.size:
        raise ValueError(f"AMP mask foreground must be nonempty and non-full: {path}")
    ys, xs = np.nonzero(foreground)
    tight = (int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1)
    if tight == canvas:
        raise ValueError(f"AMP mask tight extent fills the canvas: {path}")


def plan(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifests, embeddings = root / "manifests", root / "embeddings"
    clean = pd.read_parquet(embeddings / "clean_embeddings.parquet").reset_index(drop=True)
    embedded = pd.read_parquet(embeddings / "fn_embeddings.parquet")
    queries = pd.read_parquet(manifests / "selected_fn_queries.parquet")
    masks = pd.read_parquet(manifests / "mask_selection.parquet")
    for mask_path in sorted(set(masks.mask_path.astype(str))):
        _validate_amp_mask(Path(mask_path))
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


def _publish_paths(amp_dir: Path, runtime_root: Path, published_root: Path) -> None:
    source, destination = str(runtime_root.resolve()), str(published_root.resolve())
    if source == destination:
        return
    for path in sorted(amp_dir.rglob("*.json")) + sorted(amp_dir.rglob("*.jsonl")):
        value = path.read_text(encoding="utf-8")
        if source in value:
            path.write_text(value.replace(source, destination), encoding="utf-8")


def run(config_path: Path, root: Path, checkpoint_root: Path,
        published_root: Path | None = None,
        repo: Path = Path("/workspace/paidf-anomalygen")) -> dict[str, Any]:
    hf_home = _validate_checkpoint_root(checkpoint_root, repo)
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
    env = os.environ.copy()
    env.update(HF_HOME=str(hf_home), HF_HUB_CACHE=str(_hf_hub(hf_home)),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run(command, check=True, stdout=sys.stderr, env=env)
    published_root = (published_root or root).resolve()
    _publish_paths(root / "amp", root, published_root)
    testcase = root / "amp" / "testcase.jsonl"
    if not testcase.is_file() or not testcase.read_text().strip():
        raise ValueError("AMP produced no testcase rows")
    report["testcase"] = str(published_root / "amp" / "testcase.jsonl")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--published-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/workspace/paidf-anomalygen"))
    args = parser.parse_args()
    result = run(Path(args.config).resolve(), Path(args.prepared_root).resolve(),
                 args.checkpoint_root.resolve(), args.published_root, args.repo.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
