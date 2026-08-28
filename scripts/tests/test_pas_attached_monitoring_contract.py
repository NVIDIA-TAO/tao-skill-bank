# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract regression tests for unattended PAS loop liveness."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAS = ROOT / "skills" / "applications" / "tao-run-deft-pas"


def test_skill_forbids_final_response_while_run_is_nonterminal():
    skill = (PAS / "SKILL.md").read_text()
    pipeline = (PAS / "references" / "pipeline-and-state.md").read_text()
    compact_skill = " ".join(skill.split())
    compact_pipeline = " ".join(pipeline.split())
    assert "Never send a final response while an approved run is nonterminal" in compact_skill
    assert "30 seconds" in skill
    assert "continuing through finalize, commit, audit, and the next stage" in compact_skill
    assert "Do not use open-ended polling" not in skill
    assert "Do not convert the progress line into a final response" in compact_pipeline
    assert "wait for a human `continue` message" in pipeline


def test_behavioral_eval_covers_ended_turn_liveness():
    cases = json.loads((PAS / "evals" / "evals.json").read_text())
    case = next(
        item for item in cases if item["id"] == "tao-run-deft-pas-attached-loop-liveness"
    )
    assert "future poll" in case["question"]
    assert "no future poll can wake an ended turn" in case["ground_truth"]
    assert any("final responses" in item for item in case["expected_behavior"])
