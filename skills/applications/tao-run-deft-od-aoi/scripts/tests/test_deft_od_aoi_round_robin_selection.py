# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "deft_od_aoi_round_robin_selection.py"
SPEC = importlib.util.spec_from_file_location("deft_od_aoi_round_robin_selection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
FIXTURE = Path(__file__).parent / "fixtures/round_robin_similarity_reference.json"
COMMIT_SCRIPT = Path(__file__).parents[1] / "commit_deft_od_aoi_stage.py"
COMMIT_SPEC = importlib.util.spec_from_file_location(
    "commit_deft_od_aoi_stage_for_round_robin", COMMIT_SCRIPT
)
COMMIT_MODULE = importlib.util.module_from_spec(COMMIT_SPEC)
assert COMMIT_SPEC.loader
COMMIT_SPEC.loader.exec_module(COMMIT_MODULE)


def test_frozen_fixture_matches_7ebdfbb_semantic_selection() -> None:
    fixture = json.loads(FIXTURE.read_text())
    queries = fixture["queries"]
    for query in queries:
        query.update(dict(zip(
            ("benchmark", "texture", "defect_type"),
            fixture["source_metadata"][query["query_id"]],
        )))
    retrieval = fixture["policy"]["retrieval"]
    selected = MODULE.round_robin_rank(
        pd.DataFrame(fixture["candidates"]), pd.DataFrame(queries),
        excluded=set(fixture["ledger"]), quota=4,
        minimum=retrieval["minimum_similarity"],
        overfetch=retrieval["candidate_overfetch"], audit_top_k=20,
    )
    semantic = [
        [row["query_id"], row["candidate_id"], row["source_filepath"]]
        for row in selected
    ]
    assert fixture["source_commit"] == "7ebdfbb"
    assert semantic == fixture["reference_semantic_selection"]


def test_rank_preserves_rounds_ties_exclusions_threshold_and_parent_dedup() -> None:
    candidates = pd.DataFrame([
        {"candidate_id": "excluded", "source_filepath": "/excluded", "embedding": [1, 0]},
        {"candidate_id": "first", "source_filepath": "/same", "embedding": [1, 0]},
        {"candidate_id": "same-parent", "source_filepath": "/same", "embedding": [1, 0]},
        {"candidate_id": "tie-a", "source_filepath": "/a", "embedding": [0, 1]},
        {"candidate_id": "tie-b", "source_filepath": "/b", "embedding": [0, 1]},
        {"candidate_id": "below", "source_filepath": "/below", "embedding": [-1, 0]},
    ])
    queries = pd.DataFrame([
        {"query_id": "q-first", "embedding": [1, 0]},
        {"query_id": "q-second", "embedding": [0, 1]},
    ])

    selected = MODULE.round_robin_rank(
        candidates, queries, excluded={"/excluded"}, minimum=0.01,
        quota=4, overfetch=2, audit_top_k=20,
    )

    assert [(row["query_id"], row["candidate_id"]) for row in selected] == [
        ("q-first", "first"), ("q-second", "tie-a"), ("q-second", "tie-b"),
    ]
    assert len({row["source_filepath"] for row in selected}) == len(selected)


def test_select_isolates_pockets_and_caps_clean() -> None:
    real_candidates = pd.DataFrame([
        {"candidate_id": f"r-{index}", "source_filepath": f"/real/{index}",
         "embedding": embedding}
        for index, embedding in enumerate((
            [1, 0], [.99, .1], [0, 1], [.1, .99], [.8, .6], [.6, .8],
        ))
    ])
    clean_candidates = pd.DataFrame([
        {"candidate_id": f"c-{index}", "source_filepath": f"/clean/{index}",
         "embedding": embedding}
        for index, embedding in enumerate(([1, 0], [0, 1], [.7, .7]))
    ])
    real_queries = pd.DataFrame([
        {"query_id": "strict-a", "reason": "fn", "benchmark": "b1",
         "texture": "t1", "defect_type": "d1", "real_factor": 2,
         "embedding": [1, 0]},
        {"query_id": "strict-b", "reason": "fn", "benchmark": "b2",
         "texture": "t2", "defect_type": "d2", "real_factor": 1,
         "embedding": [0, 1]},
    ])
    clean_queries = pd.DataFrame([
        {"query_id": "clean-a", "reason": "background_fp", "embedding": [1, 0]},
        {"query_id": "clean-b", "reason": "background_fp", "embedding": [0, 1]},
    ])
    policy = {
        "retrieval": {"minimum_similarity": -1, "candidate_overfetch": 3,
                      "audit_top_k_per_query": 20},
        "routing": {"real_mine_factor_min": 1, "near_miss_real_factor": 2,
                    "near_miss_real_cap_per_pocket": 2, "clean_factor": 2,
                    "clean_cumulative_cap_per_real": 1.0},
    }

    selected, audit = MODULE.select(
        {"real": real_candidates, "clean": clean_candidates},
        {"real": real_queries, "clean": clean_queries}, policy,
        {"real": set(), "clean": set()},
    )

    assert [row["query_id"] for row in selected["real"]] == [
        "strict-a", "strict-a", "strict-b",
    ]
    assert len(selected["clean"]) == 3
    assert [row["requested"] for row in audit["branches"]] == [2, 1, 3]
    assert all(row["shortfall"] == 0 for row in audit["branches"])


def test_materialized_outputs_satisfy_retrieval_stage_contract(tmp_path: Path) -> None:
    candidate_root, retrieval_root = tmp_path / "candidates", tmp_path / "retrieval"
    candidate_root.mkdir()
    retrieval_root.mkdir()
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "retrieval": {
            "selection": {"strategy": "round_robin_similarity"},
            "minimum_similarity": 0.0,
            "candidate_overfetch": 2,
            "audit_top_k_per_query": 20,
        },
        "routing": {
            "real_mine_factor_min": 1, "near_miss_real_factor": 2,
            "near_miss_real_cap_per_pocket": 20, "clean_factor": 1,
            "clean_cumulative_cap_per_real": 1.0,
        },
    }))
    query_counts = {"real": 1, "clean": 1}
    (retrieval_root / "query_manifest.json").write_text(json.dumps({
        "status": "COMPLETE", "iteration": 1, "query_counts": query_counts,
        "enabled_roles": list(query_counts),
        "selection_strategy": "round_robin_similarity",
        "role_status": {
            "real": {"excluded_count": 1, "exclusion_manifest": str(
                retrieval_root / "exclude_real_candidates.parquet"
            )},
            "clean": {"excluded_count": 0, "exclusion_manifest": str(
                retrieval_root / "exclude_clean_candidates.parquet"
            )},
        },
    }))
    for role in query_counts:
        candidates = [{
            "filepath": f"/{role}-candidate.png", "candidate_id": f"{role}-candidate",
            "source_filepath": f"/{role}-source.png", "source_image_id": 1,
            "embedding": [1.0, 0.0],
        }]
        if role == "real":
            candidates.append({
                "filepath": "/real-fallback.png", "candidate_id": "real-fallback",
                "source_filepath": "/real-fallback-source.png", "source_image_id": 2,
                "embedding": [0.8, 0.6],
            })
        pd.DataFrame(candidates).to_parquet(
            candidate_root / f"{role}_candidate_embeddings.parquet", index=False
        )
        excluded = [f"/{role}-candidate.png"] if role == "real" else []
        pd.DataFrame({"filepath": excluded}).to_parquet(
            retrieval_root / f"exclude_{role}_candidates.parquet", index=False
        )
        query = {
            "filepath": f"/{role}-query.png", "query_id": f"{role}-query",
            "reason": "fn" if role == "real" else "background_fp",
            "embedding": [1.0, 0.0],
        }
        if role == "real":
            query.update(benchmark="line", texture="board", defect_type="bridge",
                         real_factor=1)
        frame = pd.DataFrame([query])
        frame.drop(columns="embedding").to_parquet(
            retrieval_root / f"{role}_queries.parquet", index=False
        )
        frame.to_parquet(retrieval_root / f"{role}_query_embeddings.parquet", index=False)

    report = MODULE.materialize(policy, candidate_root, retrieval_root)
    artifacts = {
        "query_manifest": {"path": retrieval_root / "query_manifest.json"},
        "selection_report": {
            "path": retrieval_root / "round_robin_selection_report.json"
        },
    }
    for role in query_counts:
        artifacts[f"{role}_queries"] = {"path": retrieval_root / f"{role}_queries.parquet"}
        artifacts[f"{role}_query_embeddings"] = {
            "path": retrieval_root / f"{role}_query_embeddings.parquet"
        }
        artifacts[f"{role}_mined"] = {
            "path": retrieval_root / f"mine_{role}/final_unique_files.parquet"
        }

    COMMIT_MODULE._validate_retrieval(1, artifacts)
    assert report["selected_counts"] == {"real": 1, "clean": 1}
    selected_real = pd.read_parquet(
        retrieval_root / "mine_real/final_unique_files.parquet"
    )
    assert selected_real.filepath.tolist() == ["/real-fallback.png"]
    with pytest.raises(FileExistsError):
        MODULE.materialize(policy, candidate_root, retrieval_root)


def test_materialize_rejects_unknown_candidate_exclusion(tmp_path: Path) -> None:
    candidates, retrieval = tmp_path / "candidates", tmp_path / "retrieval"
    candidates.mkdir()
    retrieval.mkdir()
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "retrieval": {"selection": {"strategy": "round_robin_similarity"}},
        "routing": {},
    }))
    (retrieval / "query_manifest.json").write_text(json.dumps({
        "status": "COMPLETE", "iteration": 1, "enabled_roles": ["real"],
        "role_status": {"real": {"excluded_count": 1}},
    }))
    pd.DataFrame([{"filepath": "/known", "candidate_id": "known",
                   "source_filepath": "/source", "embedding": [1, 0]}]).to_parquet(
        candidates / "real_candidate_embeddings.parquet", index=False
    )
    pd.DataFrame([{"filepath": "/query", "query_id": "query",
                   "embedding": [1, 0]}]).to_parquet(
        retrieval / "real_query_embeddings.parquet", index=False
    )
    pd.DataFrame({"filepath": ["/unknown"]}).to_parquet(
        retrieval / "exclude_real_candidates.parquet", index=False
    )

    with pytest.raises(ValueError, match="unknown candidates"):
        MODULE.materialize(policy, candidates, retrieval)
