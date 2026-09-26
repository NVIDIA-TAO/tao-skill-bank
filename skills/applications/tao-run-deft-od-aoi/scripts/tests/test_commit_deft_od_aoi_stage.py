# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pandas as pd
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


def _retrieval(root: Path, sparse: bool = False) -> list[str]:
    manifest = root / "query_manifest.json"
    counts = {"real": 1} if sparse else {"real": 1, "clean": 1}
    manifest.write_text(json.dumps({"status": "COMPLETE", "iteration": 1,
                                    "query_counts": counts,
                                    "enabled_roles": list(counts)}))
    artifacts = [f"query_manifest={manifest}"]
    for role, count in counts.items():
        queries = root / f"{role}_queries.parquet"
        embeddings = root / f"{role}_query_embeddings.parquet"
        mined = root / f"{role}_mined.parquet"
        pd.DataFrame({"filepath": [f"/{role}-query"] * count}).to_parquet(queries)
        pd.DataFrame({"filepath": [f"/{role}-query"] * count,
                      "embedding": [[1.0, 0.0]] * count}).to_parquet(embeddings)
        pd.DataFrame({"filepath": [f"/{role}-candidate"]}).to_parquet(mined)
        artifacts.extend((f"{role}_queries={queries}",
                          f"{role}_query_embeddings={embeddings}",
                          f"{role}_mined={mined}"))
    return artifacts


def _iteration_artifacts(root: Path) -> dict[str, list[str]]:
    admission = root / "admission_report.json"
    admission.write_text(json.dumps({"status": "COMPLETE", "iteration": 1,
                                     "admitted": {"synthetic": 0}}))
    checkpoint = root / "model_epoch_001.pth"
    checkpoint.write_bytes(b"checkpoint")
    status = root / "train_status.json"
    status.write_text('{"epoch": 1, "kpi": {"val_mAP50": 0.75}}\n')
    selection = root / "checkpoint_selection.json"
    selection.write_text(json.dumps({
        "status": "COMPLETE", "action": "select", "best_epoch": 1,
        "best_kpi_mAP50": 0.75, "selected_checkpoint": str(checkpoint.resolve()),
        "status_files": [str(status.resolve())],
    }))
    roles = {}
    for role in ("kpi", "test"):
        predictions = root / role / "labels"
        predictions.mkdir(parents=True)
        (predictions / "image.txt").write_text("")
        roles[role] = {"expected_images": 1, "predictions": str(predictions)}
    measurement = root / "measurement_manifest.json"
    measurement.write_text(json.dumps({
        "status": "COMPLETE", "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": MODULE._sha(checkpoint), "inference_roles": roles,
    }))
    gap_artifacts = []
    for kind, confidence in (("loose", 0.3), ("strict", 0.8)):
        report = root / f"{kind}_gap_report.json"
        report.write_text(json.dumps({
            "kpi": f"kpi_{kind}", "counts_by_type": {"FN": 1},
            "counts_by_class": {"defect": {"FN": 1}},
            "settings": {"iou_threshold": 0.5, "conf_threshold": confidence,
                         "min_area": 0},
        }))
        boxes = root / f"{kind}_box_gaps.parquet"
        pd.DataFrame({"gap_type": ["FN"]}).to_parquet(boxes)
        gap_artifacts.extend((f"{kind}_gap_report={report}", f"{kind}_box_gaps={boxes}"))
    return {
        "iteration_admission": [f"admission_report={admission}"],
        "iteration_training": [f"checkpoint_selection={selection}"],
        "iteration_measurement": [f"measurement_manifest={measurement}"],
        "iteration_gaps": gap_artifacts,
    }


def test_commit_enforces_semantic_evidence_and_completes(tmp_path: Path) -> None:
    state, artifact = _state(tmp_path)
    iteration = _iteration_artifacts(tmp_path)
    stages = [("candidate_cache", 0), ("baseline_measurement", 0),
              ("baseline_gaps", 0), ("iteration_retrieval", 1),
              ("iteration_admission", 1), ("iteration_training", 1),
              ("iteration_measurement", 1), ("iteration_gaps", 1)]
    value = None
    for stage, number in stages:
        artifacts = (_retrieval(tmp_path) if stage == "iteration_retrieval" else
                     iteration.get(stage, [f"done={artifact}"]))
        value = MODULE.commit(state, stage, number, artifacts)
    assert value["status"] == "COMPLETE" and value["next_stage"] is None


def test_retrieval_accepts_omitted_zero_count_role(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_retrieval",
                 current_iteration=0, last_stage="baseline_gaps")
    state.write_text(json.dumps(value))
    result = MODULE.commit(state, "iteration_retrieval", 1, _retrieval(tmp_path, sparse=True))
    assert result["next_stage"] == "iteration_admission"


def test_measurement_rejects_incomplete_inference(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    artifacts = _iteration_artifacts(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_training", current_iteration=1,
                 last_stage="iteration_admission", events=[])
    state.write_text(json.dumps(value))
    MODULE.commit(state, "iteration_training", 1, artifacts["iteration_training"])
    manifest = tmp_path / "measurement_manifest.json"
    report = json.loads(manifest.read_text())
    report["inference_roles"]["kpi"]["expected_images"] = 2
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="inference cardinality mismatch"):
        MODULE.commit(state, "iteration_measurement", 1, artifacts["iteration_measurement"])


def test_training_rejects_checkpoint_that_does_not_match_epoch(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    artifacts = _iteration_artifacts(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_training", current_iteration=1,
                 last_stage="iteration_admission")
    state.write_text(json.dumps(value))
    selection = tmp_path / "checkpoint_selection.json"
    report = json.loads(selection.read_text())
    report["best_epoch"] = 2
    selection.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="KPI-best checkpoint"):
        MODULE.commit(state, "iteration_training", 1, artifacts["iteration_training"])


def test_gaps_reject_report_parquet_count_disagreement(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    artifacts = _iteration_artifacts(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_gaps", current_iteration=1,
                 last_stage="iteration_measurement")
    state.write_text(json.dumps(value))
    pd.DataFrame({"gap_type": []}).to_parquet(tmp_path / "strict_box_gaps.parquet")
    with pytest.raises(ValueError, match="parquet and report counts disagree"):
        MODULE.commit(state, "iteration_gaps", 1, artifacts["iteration_gaps"])


def test_synthesis_reconciles_generated_and_blocked_counts(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", synthesis_enabled=True, next_stage="iteration_synthesis",
                 current_iteration=1, last_stage="iteration_admission")
    state.write_text(json.dumps(value))
    generation = tmp_path / "generation_report.json"
    generation.write_text(json.dumps({
        "status": "COMPLETE", "generated": 2,
        "groups": [{"requested": 3, "generated": 2, "guardrail_blocked": 1}],
    }))
    admission = tmp_path / "admission_report.json"
    admission.write_text(json.dumps({"status": "COMPLETE", "iteration": 1,
                                     "admitted": {"synthetic": 2}}))
    result = MODULE.commit(state, "iteration_synthesis", 1,
                           [f"generation_report={generation}",
                            f"admission_report={admission}"])
    assert result["next_stage"] == "iteration_training"


def test_synthesis_accepts_typed_no_eligible_skip(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", synthesis_enabled=True, next_stage="iteration_synthesis",
                 current_iteration=1, last_stage="iteration_admission")
    state.write_text(json.dumps(value))
    preparation = tmp_path / "input_contract.json"
    preparation.write_text(json.dumps({
        "status": "SKIPPED", "reason": "no_eligible_false_negatives",
        "selection_candidate_fn_count": 2, "eligible_fn_count": 0,
        "skipped_fn_count": 2,
        "skipped_fns": [{"fn_id": "fn-1"}, {"fn_id": "fn-2"}],
        "skip_counts": {"empty_fn_mask": 2},
    }))

    result = MODULE.commit(
        state, "iteration_synthesis", 1,
        [f"synthesis_preparation={preparation}"],
    )

    assert result["next_stage"] == "iteration_training"


def test_synthesis_accepts_typed_no_routed_skip(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", synthesis_enabled=True, next_stage="iteration_synthesis",
                 current_iteration=1, last_stage="iteration_admission")
    state.write_text(json.dumps(value))
    request = tmp_path / "synthesis_request.json"
    request.write_text(json.dumps({
        "status": "SKIPPED", "reason": "no_routed_false_negatives", "fn_count": 0,
        "skipped_unrouted_fn_count": 2,
        "skipped_unrouted_by_dataset": {"boxes_only": 2},
        "observed_dataset_ids": ["boxes_only"],
        "configured_route_keys": ["route"],
        "message": "Skipped synthesis because every FN is unrouted.",
    }))

    result = MODULE.commit(
        state, "iteration_synthesis", 1, [f"synthesis_request={request}"],
    )

    assert result["next_stage"] == "iteration_training"


def test_commit_rejects_out_of_order_stage(tmp_path: Path) -> None:
    state, artifact = _state(tmp_path)
    with pytest.raises(ValueError, match="expected stage candidate_cache"):
        MODULE.commit(state, "baseline_measurement", 0, [f"done={artifact}"])
