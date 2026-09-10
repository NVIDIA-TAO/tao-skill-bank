# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "commit_deft_od_aoi_stage.py"
SPEC = importlib.util.spec_from_file_location("commit_deft_od_aoi_stage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _state(root: Path) -> tuple[Path, Path]:
    state = root / "deft_state.json"
    state.write_text(json.dumps({"status": "READY", "next_stage": "candidate_cache",
                                 "current_iteration": 0, "max_iterations": 1}))
    artifact = root / "done.json"
    artifact.write_text("{}")
    return state, artifact


def test_commit_enforces_order_and_completes_after_final_gaps(tmp_path: Path) -> None:
    state, artifact = _state(tmp_path)
    stages = [("candidate_cache", 0), ("baseline_measurement", 0),
              ("baseline_gaps", 0), ("iteration_retrieval", 1),
              ("iteration_admission", 1), ("iteration_training", 1),
              ("iteration_measurement", 1), ("iteration_gaps", 1)]
    value = None
    for stage, iteration in stages:
        value = MODULE.commit(state, stage, iteration, [f"done={artifact}"])
    assert value["status"] == "COMPLETE" and value["next_stage"] is None
    assert len(value["events"]) == len(stages)
    assert len((tmp_path / "loop_log.jsonl").read_text().splitlines()) == len(stages)


def test_commit_rejects_out_of_order_stage(tmp_path: Path) -> None:
    state, artifact = _state(tmp_path)
    with pytest.raises(ValueError, match="expected stage candidate_cache"):
        MODULE.commit(state, "baseline_measurement", 0, [f"done={artifact}"])
