# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "deft_od_aoi_round_robin_selection.py"
SPEC = importlib.util.spec_from_file_location("deft_od_aoi_round_robin_selection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
FIXTURE = Path(__file__).parent / "fixtures/round_robin_similarity_reference.json"


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
                    "near_miss_real_cap": 2, "clean_factor": 2,
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
