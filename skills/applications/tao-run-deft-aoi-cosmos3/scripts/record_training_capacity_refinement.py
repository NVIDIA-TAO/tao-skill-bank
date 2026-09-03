#!/usr/bin/env python3
"""Atomically record a probe-backed training micro-batch capacity refinement."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(value: Any, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty file path")
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{label} must be an existing non-empty file: {path}")
    return path


def _within(path: pathlib.Path, root: pathlib.Path, label: str) -> pathlib.Path:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be under {root}: {path}") from exc
    return path.resolve()


def _json(path: pathlib.Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _rows(path: pathlib.Path) -> int:
    with path.open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


def _write_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def apply(
    state_path: pathlib.Path, iteration: str, evidence_path: pathlib.Path
) -> dict[str, Any]:
    state_path = state_path.expanduser().resolve()
    evidence_path = evidence_path.expanduser().resolve()
    state = _json(_file(str(state_path), "state"), "state")
    evidence = _json(_file(str(evidence_path), "evidence"), "evidence")
    if state.get("version") != 7 or state.get("status") != "in_progress":
        raise ValueError("capacity refinement requires an in-progress v7 DEFT state")
    if not re.fullmatch(r"iter[1-9][0-9]*", iteration):
        raise ValueError("iteration must be iterN")
    phase = state.get("iterations", {}).get(iteration)
    if not isinstance(phase, dict) or phase.get("stage_completed") != "validate_data":
        raise ValueError("capacity refinement is only valid immediately before train")
    if evidence.get("schema") != "training_capacity_refinement_v1":
        raise ValueError("unsupported training capacity evidence schema")

    training = state.get("config", {}).get("training")
    if not isinstance(training, dict):
        raise ValueError("state is missing config.training")
    if training.get("gradient_accumulation") != 1:
        raise ValueError("capacity refinement preserves gradient_accumulation=1")
    if training.get("learning_rate_scaling") != "fixed":
        raise ValueError("capacity refinement preserves fixed learning-rate policy")
    optimizer = training.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("learning_rate") != 1e-6:
        raise ValueError("capacity refinement preserves learning_rate=1e-6")
    num_gpus = training.get("num_gpus")
    if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus <= 0:
        raise ValueError("config.training.num_gpus must be a positive integer")
    previous = evidence.get("previous_micro_batch_per_rank")
    refined = evidence.get("refined_micro_batch_per_rank")
    if previous != training.get("micro_batch_per_rank"):
        raise ValueError("capacity evidence previous micro-batch does not match state")
    if not isinstance(refined, int) or isinstance(refined, bool) or not 0 < refined < previous:
        raise ValueError("refined micro-batch must be a smaller positive integer")
    global_batch = refined * num_gpus

    results_root = state_path.parent.resolve()
    phase_root = results_root / iteration
    artifact_root = _within(evidence_path.parent, phase_root, "evidence directory")
    train_jsonl = _within(_file(str(artifact_root / "train.jsonl"), "train JSONL"), phase_root, "train JSONL")
    assemble_path = _within(_file(str(artifact_root / "assemble_summary.json"), "assemble summary"), phase_root, "assemble summary")
    validation_path = _within(_file(str(artifact_root / "validation_report.json"), "validation report"), phase_root, "validation report")
    row_count = _rows(train_jsonl)
    if row_count <= 0 or row_count % global_batch:
        raise ValueError(
            f"refined training rows must be a positive multiple of global batch {global_batch}"
        )
    assemble = _json(assemble_path, "assemble summary")
    validation = _json(validation_path, "validation report")
    if assemble.get("output_records") != row_count or assemble.get("row_multiple") != global_batch:
        raise ValueError("assemble summary does not match refined training rows/global batch")
    if (
        validation.get("state") != "COMPLETE"
        or validation.get("materialized_rows") != row_count
        or validation.get("row_multiple") != global_batch
    ):
        raise ValueError("validation report does not match refined training rows/global batch")

    failure_record_path = _file(evidence.get("failed_job_record"), "failed job record")
    failure_log_path = _within(_file(evidence.get("failure_log"), "failure log"), results_root, "failure log")
    probe_record_path = _file(evidence.get("probe_job_record"), "probe job record")
    probe_status_path = _within(_file(evidence.get("probe_status"), "probe status"), results_root, "probe status")
    failure_record = _json(failure_record_path, "failed job record")
    probe_record = _json(probe_record_path, "probe job record")
    probe_status = _json(probe_status_path, "probe status")
    if failure_record.get("terminal_state") != "ERROR":
        raise ValueError("failed job record must be terminal ERROR")
    if "CUDA out of memory" not in failure_log_path.read_text(errors="replace"):
        raise ValueError("failure log does not contain CUDA out-of-memory evidence")
    if probe_record.get("terminal_state") != "COMPLETE":
        raise ValueError("probe job record must be terminal COMPLETE")
    if (
        probe_status.get("state") != "COMPLETE"
        or probe_status.get("micro_batch_per_rank") != refined
        or probe_status.get("gradient_accumulation") != 1
        or probe_status.get("effective_global_batch") != global_batch
        or probe_status.get("learning_rate") != 1e-6
        or int(probe_status.get("optimizer_updates_observed", 0)) <= 0
    ):
        raise ValueError("probe status does not prove the refined training contract")

    audit = {
        "schema": "training_capacity_refinement_audit_v1",
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "iteration": iteration,
        "previous_micro_batch_per_rank": previous,
        "refined_micro_batch_per_rank": refined,
        "num_gpus": num_gpus,
        "gradient_accumulation": 1,
        "previous_global_batch": previous * num_gpus,
        "refined_global_batch": global_batch,
        "learning_rate": 1e-6,
        "learning_rate_scaling": "fixed",
        "failed_job_id": failure_record.get("id"),
        "failed_backend_ref": failure_record.get("backend_ref"),
        "failed_job_record": str(failure_record_path),
        "failure_log": str(failure_log_path),
        "failure_log_sha256": _sha256(failure_log_path),
        "probe_job_id": probe_record.get("id"),
        "probe_backend_ref": probe_record.get("backend_ref"),
        "probe_job_record": str(probe_record_path),
        "probe_status": str(probe_status_path),
        "probe_status_sha256": _sha256(probe_status_path),
        "probe_optimizer_updates": probe_status["optimizer_updates_observed"],
        "combined_training_jsonl": str(train_jsonl),
        "combined_training_sha256": _sha256(train_jsonl),
        "materialized_rows": row_count,
        "assemble_summary": str(assemble_path),
        "validation_report": str(validation_path),
        "evidence": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
    }
    refinements = training.setdefault("capacity_refinements", [])
    if not isinstance(refinements, list):
        raise ValueError("config.training.capacity_refinements must be an array")
    if any(item.get("evidence_sha256") == audit["evidence_sha256"] for item in refinements if isinstance(item, dict)):
        return state
    training["micro_batch_per_rank"] = refined
    training["global_batch"] = global_batch
    refinements.append(audit)
    phase["combined_training_jsonl"] = str(train_jsonl)
    phase["assemble_summary"] = str(assemble_path)
    phase["validation_report"] = str(validation_path)
    phase["training_capacity_refinement"] = audit
    events = state.setdefault("events", [])
    sequence = max((item.get("seq", 0) for item in events if isinstance(item, dict)), default=0) + 1
    events.append(
        {
            "seq": sequence,
            "ts": audit["recorded_at"],
            "iter": iteration,
            "stage": "training_capacity_refinement",
            "status": "ok",
            "summary": (
                f"Refined per-GPU micro-batch from {previous} to {refined} after "
                f"an actual OOM and a successful {audit['probe_optimizer_updates']}-update probe."
            ),
            "duration_sec": max(1, int(evidence.get("duration_sec", 1))),
            "context_tokens": 0,
        }
    )
    _write_atomic(state_path, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=pathlib.Path)
    parser.add_argument("--iteration", required=True)
    parser.add_argument("--evidence", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        state = apply(args.state, args.iteration, args.evidence)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"record_training_capacity_refinement: {exc}", file=sys.stderr)
        return 2
    training = state["config"]["training"]
    print(
        "record_training_capacity_refinement: OK "
        f"micro_batch_per_rank={training['micro_batch_per_rank']} "
        f"global_batch={training['global_batch']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
