# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit DEFT disk state and report the only safe resume/completion status.

This is deliberately read-only. It cross-checks ``deft_state.json``,
``loop_log.jsonl``, and every path recorded in iteration state. Agents run it
on startup, after compaction, before each stage, and before claiming that the
loop or an iteration completed.

Exit codes:
  0: structurally valid (IN_PROGRESS, FAILED, or COMPLETE)
  1: inconsistent/invalid, non-terminal with --require-terminal, or
     unsuccessful with --require-complete
  2: input file could not be loaded
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

from metric_contract import (
    contract_from_state,
    pick_best,
    render_target,
    result_from_iteration,
    result_passes,
)


VALID_ITERATION_STATUSES = {"pending", "in_progress", "complete", "failed"}
VALID_LOG_STATUSES = {"ok", "error"}
VALID_STAGES = {
    "train",
    "evaluate",
    "rca",
    "routing",
    "anomalygen_finetune",
    "anomalygen",
    "data_mining",
    "data_merge",
    "loop_stop",
}
VALID_COMPLETED_STAGES = VALID_STAGES - {"loop_stop"}
PATH_FIELDS = {
    "best_ckpt_path",
    "inference_csv",
    "rca_gaps_parquet",
    "routing_mining_parquet",
    "routing_anomalygen_parquet",
    "anomalygen_sdg_csv",
    "anomalygen_allocation_json",
    "mining_mined_parquet",
    "mining_parquet",  # legacy field: accepted but still checked
    "mining_summary",
    "mining_target_embeddings",
    "mining_source_embeddings",
    "mining_target_log",
    "mining_source_log",
    "mining_knn_log",
    "training_spec",
    "combined_training_csv",
    "provenance_csv",
    "merge_validation_report",
    "training_csv",
    "validation_csv",
    "kpi_test_csv",
    "specs_file",
}
FIELD_STAGE = {
    "inference_csv": "evaluate",
    "far_pct": "evaluate",
    "metric_result": "evaluate",
    "threshold": "evaluate",
    "rca_gaps_parquet": "rca",
    "routing_mining_parquet": "routing",
    "routing_anomalygen_parquet": "routing",
    "anomalygen_sdg_csv": "anomalygen",
    "anomalygen_allocation_json": "anomalygen",
    "anomalygen_amp_allocated": "anomalygen",
    "mining_mined_parquet": "data_mining",
    "mining_parquet": "data_mining",
    "mining_summary": "data_mining",
    "mining_target_embeddings": "data_mining",
    "mining_source_embeddings": "data_mining",
    "mining_target_log": "data_mining",
    "mining_source_log": "data_mining",
    "mining_knn_log": "data_mining",
    "combined_training_csv": "data_merge",
    "provenance_csv": "data_merge",
    "merge_validation_report": "data_merge",
}
STAGE_REQUIRED_FIELD_SETS = {
    "train": (("best_ckpt_path", "training_spec"),),
    "evaluate": (("best_ckpt_path", "inference_csv"),),
    "rca": (("rca_gaps_parquet",),),
    "routing": (("routing_mining_parquet", "routing_anomalygen_parquet"),),
    "anomalygen": (("anomalygen_sdg_csv",), ("anomalygen_skipped",)),
    "data_mining": (
        (
            "mining_mined_parquet",
            "mining_summary",
            "mining_target_embeddings",
            "mining_source_embeddings",
            "mining_target_log",
            "mining_source_log",
            "mining_knn_log",
            "mining_mined_count",
        ),
        ("data_mining_skipped",),
    ),
    "data_merge": (
        ("combined_training_csv", "provenance_csv", "merge_validation_report"),
    ),
}


def _load_state(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("deft_state.json root must be an object")
    return data


def _load_log(path: pathlib.Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"loop_log.jsonl:{lineno}: invalid JSON: {exc}")
            continue
        if not isinstance(entry, dict):
            errors.append(f"loop_log.jsonl:{lineno}: entry must be an object")
            continue
        entries.append(entry)
    return entries


def _iteration_sort_key(label: str) -> tuple[int, int]:
    if label == "baseline":
        return (0, 0)
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    return (1, int(match.group(1))) if match else (2, 0)


def _parquet_proof(
    path: pathlib.Path,
    field: str,
    required_columns: set[str],
    errors: list[str],
) -> int | None:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        columns = set(parquet.schema_arrow.names)
    except ModuleNotFoundError as exc:
        if exc.name != "pyarrow":
            errors.append(f"{field} is not a readable parquet: {exc}")
            return None
        helper = pathlib.Path(__file__).with_name("deft_python.sh")
        payload = (
            "import json, sys; import pyarrow.parquet as pq; "
            "p = pq.ParquetFile(sys.argv[1]); "
            "print(json.dumps({'rows': p.metadata.num_rows, "
            "'columns': p.schema_arrow.names}))"
        )
        try:
            proof = json.loads(
                subprocess.check_output(
                    [str(helper), "-c", payload, str(path)], text=True
                )
            )
            rows = int(proof["rows"])
            columns = set(proof["columns"])
        except Exception as fallback_exc:
            errors.append(
                f"{field} is not a readable parquet: {exc}; "
                f"fallback via deft_python.sh failed: {fallback_exc}"
            )
            return None
    except Exception as exc:
        errors.append(f"{field} is not a readable parquet: {exc}")
        return None
    missing = sorted(required_columns - columns)
    if missing:
        errors.append(f"{field} is missing parquet columns {missing}")
    return rows


def _tao_pass_log(path: pathlib.Path, field: str, errors: list[str]) -> None:
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        errors.append(f"{field} cannot be read: {exc}")
        return
    statuses = re.findall(r"Execution status:\s*(PASS|FAIL)", text)
    if not statuses:
        errors.append(f"{field} has no TAO 'Execution status' marker")
    elif statuses[-1] != "PASS" or "FAIL" in statuses:
        errors.append(f"{field} does not prove one clean TAO PASS: {statuses}")


def _mining_summary_proof(
    path: pathlib.Path,
    mined_rows: int | None,
    recorded_count: Any,
    field: str,
    errors: list[str],
) -> None:
    required = {
        "candidate_count",
        "kept_count",
        "rejected_count",
        "similarity_threshold",
    }
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        errors.append(f"{field} cannot be read as CSV: {exc}")
        return
    missing = sorted(required - columns)
    if missing:
        errors.append(f"{field} is missing columns {missing}")
        return
    if len(rows) != 1:
        errors.append(f"{field} must contain exactly one data row, got {len(rows)}")
        return
    row = rows[0]
    try:
        candidate = int(row["candidate_count"])
        kept = int(row["kept_count"])
        rejected = int(row["rejected_count"])
        threshold = float(row["similarity_threshold"])
    except (TypeError, ValueError) as exc:
        errors.append(f"{field} contains non-numeric counts/threshold: {exc}")
        return
    if min(candidate, kept, rejected) < 0 or candidate != kept + rejected:
        errors.append(
            f"{field} counts disagree: candidate={candidate}, kept={kept}, "
            f"rejected={rejected}"
        )
    if not math.isfinite(threshold):
        errors.append(f"{field}.similarity_threshold must be finite")
    if not isinstance(recorded_count, int) or isinstance(recorded_count, bool):
        errors.append(f"{field} has no integer mining_mined_count in state")
    elif kept != recorded_count:
        errors.append(
            f"{field}.kept_count={kept} disagrees with mining_mined_count="
            f"{recorded_count}"
        )
    if mined_rows is not None and kept != mined_rows:
        errors.append(
            f"{field}.kept_count={kept} disagrees with mined parquet rows={mined_rows}"
        )


def _allocation_proof(
    path: pathlib.Path,
    recorded_count: Any,
    field: str,
    errors: list[str],
) -> None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{field} is invalid JSON: {exc}")
        return
    if not isinstance(payload, dict) or not payload:
        errors.append(f"{field} must be a non-empty defect-to-count object")
        return
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        for value in payload.values()
    ):
        errors.append(f"{field} contains invalid allocation counts")
        return
    allocated = sum(payload.values())
    if allocated <= 0:
        errors.append(f"{field} must allocate at least one AMP")
    if not isinstance(recorded_count, int) or isinstance(recorded_count, bool):
        errors.append(f"{field} has no integer anomalygen_amp_allocated in state")
    elif recorded_count != allocated:
        errors.append(
            f"{field} total={allocated} disagrees with "
            f"anomalygen_amp_allocated={recorded_count}"
        )


def _expected_next(entry: dict[str, Any]) -> set[tuple[str, str]]:
    label = str(entry.get("iter"))
    stage = str(entry.get("stage"))
    if entry.get("status") == "error":
        return {(label, "loop_stop")}
    if stage == "loop_stop":
        return set()
    if label == "baseline":
        if stage == "train":
            return {("baseline", "evaluate")}
        if stage == "evaluate":
            return {("baseline", "rca"), ("baseline", "loop_stop")}
        if stage == "rca":
            return {("iter1", "routing")}
        return set()
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    if not match:
        return set()
    number = int(match.group(1))
    if stage == "routing":
        return {(label, "anomalygen"), (label, "anomalygen_finetune")}
    if stage == "anomalygen_finetune":
        return {(label, "anomalygen")}
    if stage == "anomalygen":
        return {(label, "data_mining")}
    if stage == "data_mining":
        return {(label, "data_merge")}
    if stage == "data_merge":
        return {(label, "train")}
    if stage == "train":
        return {(label, "evaluate")}
    if stage == "evaluate":
        return {(label, "rca"), (label, "loop_stop")}
    if stage == "rca":
        return {(f"iter{number + 1}", "routing")}
    return set()


def _next_action(
    state: dict[str, Any],
    entries: list[dict[str, Any]],
    status: str,
    contract: dict[str, Any] | None,
) -> tuple[str, str | None]:
    if status == "INVALID":
        return "repair disk-state inconsistencies before running another stage", None
    if status == "COMPLETE":
        return "run prepare_inference_spec.py if handoff artifacts are absent", "references/prepare-for-inference.md"
    if status == "FAILED":
        return "surface the logged hard stop; do not retry automatically", "references/pipeline-and-state.md"
    if not entries:
        return "baseline train", "references/visual-changenet.md"

    last = entries[-1]
    label = str(last.get("iter", ""))
    stage = str(last.get("stage", ""))
    if stage == "train":
        return f"{label} inference + KPI evaluate", "references/visual-changenet.md"
    if stage == "evaluate":
        iterations = state.get("iterations", {})
        info = iterations.get(label, {}) if isinstance(iterations, dict) else {}
        metric_met = False
        if isinstance(info, dict) and contract is not None:
            try:
                result = result_from_iteration(info, contract)
                metric_met = bool(result and result_passes(contract, result)[0])
            except ValueError:
                metric_met = False
        iter_match = re.fullmatch(r"iter([1-9][0-9]*)", label)
        max_iterations = state.get("max_iterations")
        reached_max = bool(
            iter_match
            and isinstance(max_iterations, int)
            and not isinstance(max_iterations, bool)
            and int(iter_match.group(1)) >= max_iterations
        )
        if metric_met or reached_max:
            return "append loop_stop and run the loop-end sequence", "references/pipeline-and-state.md"
        return f"{label} RCA", "references/tao-analyze-gaps-visual-changenet.md"
    if stage == "rca":
        next_label = "iter1" if label == "baseline" else f"iter{int(label[4:]) + 1}"
        return f"{next_label} routing", "references/tao-route-visual-changenet-samples.md"
    if stage == "routing":
        return f"{label} AnomalyGen", "references/paidf-anomalygen.md"
    if stage == "anomalygen":
        return f"{label} data mining", "references/tao-mine-aoi-images.md"
    if stage == "data_mining":
        return f"{label} assemble + validate training CSV", "references/pipeline-and-state.md"
    if stage == "data_merge":
        return f"{label} train", "references/visual-changenet.md"
    if stage == "loop_stop":
        return "run the remaining loop-end sequence", "references/pipeline-and-state.md"
    return "inspect pipeline-and-state.md before continuing", "references/pipeline-and-state.md"


def audit(results_dir: pathlib.Path) -> dict[str, Any]:
    results_dir = results_dir.expanduser().resolve()
    state_path = results_dir / "deft_state.json"
    log_path = results_dir / "loop_log.jsonl"
    state = _load_state(state_path)
    errors: list[str] = []
    warnings: list[str] = []
    entries = _load_log(log_path, errors)

    contract: dict[str, Any] | None
    try:
        contract = contract_from_state(state)
    except ValueError as exc:
        contract = None
        errors.append(f"invalid metric contract: {exc}")
    if contract is not None:
        evaluator = contract["evaluator"]
        if evaluator["type"] == "builtin" and evaluator.get("id") != "far_at_recall":
            errors.append(
                f"unsupported builtin metric evaluator {evaluator.get('id')!r}"
            )
        if evaluator["type"] == "command":
            evaluator_path = pathlib.Path(str(evaluator.get("path", ""))).expanduser()
            if not evaluator_path.is_absolute():
                errors.append("metric command evaluator path must be absolute")
            elif not evaluator_path.is_file():
                errors.append(
                    f"metric command evaluator does not exist: {evaluator_path}"
                )
            elif not os.access(evaluator_path, os.X_OK):
                errors.append(
                    f"metric command evaluator is not executable: {evaluator_path}"
                )

    recorded_results = pathlib.Path(str(state.get("results_dir", ""))).expanduser()
    if not recorded_results.is_absolute():
        errors.append("state.results_dir must be an absolute path")
    elif recorded_results.resolve() != results_dir:
        errors.append(
            f"state.results_dir={recorded_results.resolve()} does not match {results_dir}"
        )

    for name in ("max_iterations", "current_iteration"):
        value = state.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"state.{name} must be an integer")
    if isinstance(state.get("max_iterations"), int) and state["max_iterations"] <= 0:
        errors.append("state.max_iterations must be > 0")
    if isinstance(state.get("current_iteration"), int) and state["current_iteration"] < 0:
        errors.append("state.current_iteration must be >= 0")
    if (
        isinstance(state.get("current_iteration"), int)
        and isinstance(state.get("max_iterations"), int)
        and state["current_iteration"] > state["max_iterations"]
    ):
        errors.append("state.current_iteration must not exceed max_iterations")
    if "baseline" in state:
        errors.append(
            "legacy top-level state.baseline is forbidden; move it to state.iterations.baseline"
        )
    iterations = state.get("iterations")
    if not isinstance(iterations, dict):
        errors.append("state.iterations must be an object")
        iterations = {}
    iteration_numbers = [
        int(match.group(1))
        for label in iterations
        if (match := re.fullmatch(r"iter([1-9][0-9]*)", label))
    ]
    expected_current = max(iteration_numbers, default=0)
    if (
        isinstance(state.get("current_iteration"), int)
        and state["current_iteration"] != expected_current
    ):
        errors.append(
            f"state.current_iteration={state['current_iteration']} does not match "
            f"highest iteration key ({expected_current})"
        )

    log_keys: set[tuple[str, str]] = set()
    last_successful_stage: dict[str, str] = {}
    expected_seq = 1
    for index, entry in enumerate(entries, 1):
        seq = entry.get("seq")
        if seq != expected_seq:
            errors.append(
                f"loop_log entry {index} has seq={seq!r}; expected {expected_seq}"
            )
            expected_seq = seq + 1 if isinstance(seq, int) else expected_seq + 1
        else:
            expected_seq += 1
        label = entry.get("iter")
        stage = entry.get("stage")
        log_status = entry.get("status")
        if label != "baseline" and not (
            isinstance(label, str) and re.fullmatch(r"iter[1-9][0-9]*", label)
        ):
            errors.append(f"loop_log seq={seq}: invalid iter label {label!r}")
        if stage not in VALID_STAGES:
            errors.append(f"loop_log seq={seq}: invalid stage {stage!r}")
        if log_status not in VALID_LOG_STATUSES:
            errors.append(f"loop_log seq={seq}: invalid status {log_status!r}")
        if not isinstance(entry.get("summary"), str) or not entry["summary"].strip():
            errors.append(f"loop_log seq={seq}: summary must be non-empty")
        duration = entry.get("duration_sec")
        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
            or duration < 0
        ):
            errors.append(
                f"loop_log seq={seq}: duration_sec must be a non-negative integer"
            )
        key = (str(label), str(stage))
        if key in log_keys:
            errors.append(
                f"loop_log contains duplicate stage event {key}; one event per stage is required"
            )
        log_keys.add(key)
        if log_status == "ok" and stage != "loop_stop":
            last_successful_stage[str(label)] = str(stage)
        if index == 1 and key not in {("baseline", "train"), ("baseline", "evaluate")}:
            errors.append(
                f"loop_log seq={seq}: first stage must be baseline/train or "
                f"baseline/evaluate (preseed), got {label}/{stage}"
            )
        elif index > 1:
            previous = entries[index - 2]
            allowed = _expected_next(previous)
            if key not in allowed:
                rendered = ", ".join(f"{i}/{s}" for i, s in sorted(allowed))
                errors.append(
                    f"loop_log seq={seq}: illegal transition "
                    f"{previous.get('iter')}/{previous.get('stage')} -> {label}/{stage}; "
                    f"expected one of [{rendered or 'end-of-log'}]"
                )

    config = state.get("config")
    if not isinstance(config, dict):
        errors.append("state.config must be an object")
    else:
        for field in (
            "specs_file",
            "training_csv",
            "validation_csv",
            "kpi_test_csv",
            "mining_pool_csv",
        ):
            value = config.get(field)
            if not value:
                errors.append(f"state.config.{field} is required")
            else:
                path = pathlib.Path(str(value)).expanduser()
                if not path.is_absolute():
                    errors.append(f"state.config.{field} must be absolute: {value}")
                elif not path.exists():
                    errors.append(f"state.config.{field} does not exist: {value}")
                elif not path.is_file():
                    errors.append(f"state.config.{field} is not a file: {value}")
                elif path.stat().st_size == 0:
                    errors.append(f"state.config.{field} is empty: {value}")
                elif field == "mining_pool_csv":
                    try:
                        with path.open(newline="") as handle:
                            rows = csv.reader(handle)
                            header = next(rows, None)
                            has_data_row = any(
                                any(cell.strip() for cell in row) for row in rows
                            )
                    except (OSError, csv.Error) as exc:
                        errors.append(
                            f"state.config.{field} cannot be read as CSV: {value}: {exc}"
                        )
                    else:
                        if not header or not any(cell.strip() for cell in header):
                            errors.append(
                                f"state.config.{field} has no CSV header: {value}"
                            )
                        elif not has_data_row:
                            errors.append(
                                f"state.config.{field} has no data rows: {value}"
                            )

    metric_candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    iteration_results: dict[str, dict[str, Any]] = {}
    completed_iteration_numbers: list[int] = []
    error_labels = {
        str(entry.get("iter"))
        for entry in entries
        if entry.get("status") == "error"
    }
    for label in sorted(iterations, key=_iteration_sort_key):
        info = iterations[label]
        if label != "baseline" and not re.fullmatch(r"iter[1-9][0-9]*", label):
            errors.append(f"state.iterations has invalid key {label!r}")
        if not isinstance(info, dict):
            errors.append(f"state.iterations.{label} must be an object")
            continue
        iter_status = info.get("status")
        if iter_status not in VALID_ITERATION_STATUSES:
            errors.append(
                f"state.iterations.{label}.status={iter_status!r} is invalid"
            )
        elif iter_status == "failed" and label not in error_labels:
            errors.append(
                f"state.iterations.{label}.status is 'failed' but loop_log has no error event"
            )
        elif iter_status == "pending" and any(key[0] == label for key in log_keys):
            errors.append(
                f"state.iterations.{label}.status is 'pending' despite committed log events"
            )
        completed = info.get("stage_completed")
        if completed is not None and completed not in VALID_COMPLETED_STAGES:
            errors.append(
                f"state.iterations.{label}.stage_completed={completed!r} is invalid; "
                f"use one of {sorted(VALID_COMPLETED_STAGES)}"
            )
        elif completed is not None and (label, str(completed)) not in log_keys:
            errors.append(
                f"state says {label}/{completed} completed but loop_log has no matching event"
            )
        expected_completed = last_successful_stage.get(label)
        if expected_completed is not None and completed != expected_completed:
            errors.append(
                f"state.iterations.{label}.stage_completed={completed!r} is stale; "
                f"last successful log stage is {expected_completed!r}"
            )

        for field, value in info.items():
            if field in PATH_FIELDS and value:
                path = pathlib.Path(str(value)).expanduser()
                if not path.is_absolute():
                    errors.append(f"state.iterations.{label}.{field} must be absolute")
                elif not path.exists():
                    errors.append(
                        f"state.iterations.{label}.{field} does not exist: {value}"
                    )
                elif not path.is_file():
                    errors.append(
                        f"state.iterations.{label}.{field} is not a file: {value}"
                    )
                elif path.stat().st_size == 0:
                    errors.append(
                        f"state.iterations.{label}.{field} is empty: {value}"
                    )
            stage = FIELD_STAGE.get(field)
            if stage and value is not None and (label, stage) not in log_keys:
                errors.append(
                    f"state.iterations.{label}.{field} is set but loop_log lacks {label}/{stage}"
                )

        allocation_value = info.get("anomalygen_allocation_json")
        if allocation_value:
            allocation_path = pathlib.Path(str(allocation_value)).expanduser()
            if allocation_path.is_file() and allocation_path.stat().st_size > 0:
                _allocation_proof(
                    allocation_path,
                    info.get("anomalygen_amp_allocated"),
                    f"state.iterations.{label}.anomalygen_allocation_json",
                    errors,
                )
        elif info.get("anomalygen_amp_allocated") is not None:
            errors.append(
                f"state.iterations.{label}.anomalygen_amp_allocated is set "
                "without anomalygen_allocation_json"
            )

        merge_report_value = info.get("merge_validation_report")
        if merge_report_value:
            merge_report_path = pathlib.Path(str(merge_report_value)).expanduser()
            if merge_report_path.is_file() and merge_report_path.stat().st_size > 0:
                try:
                    merge_report = json.loads(merge_report_path.read_text())
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(
                        f"state.iterations.{label}.merge_validation_report is invalid: {exc}"
                    )
                else:
                    rows_checked = merge_report.get("rows_checked")
                    if not isinstance(rows_checked, int) or rows_checked <= 0:
                        errors.append(
                            f"state.iterations.{label}.merge_validation_report "
                            "must record rows_checked > 0"
                        )
                    for count_field in (
                        "missing_file_count",
                        "train_val_leakage_overlap_count",
                    ):
                        if merge_report.get(count_field) != 0:
                            errors.append(
                                f"state.iterations.{label}.merge_validation_report "
                                f"must record {count_field}=0"
                            )

        if (label, "train") in log_keys:
            checkpoint_value = info.get("best_ckpt_path")
            if checkpoint_value:
                checkpoint_path = pathlib.Path(str(checkpoint_value)).resolve()
                expected_train_dir = (results_dir / label / "train").resolve()
                try:
                    checkpoint_path.relative_to(expected_train_dir)
                except ValueError:
                    errors.append(
                        f"state.iterations.{label}.best_ckpt_path must be a "
                        f"checkpoint emitted under {expected_train_dir}; got "
                        f"{checkpoint_path}"
                    )

        if (label, "data_mining") in log_keys and info.get("data_mining_skipped"):
            routing_value = info.get("routing_mining_parquet")
            if not routing_value:
                errors.append(
                    f"state.iterations.{label}.data_mining_skipped has no "
                    "routing_mining_parquet proof"
                )
            else:
                routing_path = pathlib.Path(str(routing_value)).expanduser()
                if routing_path.is_file() and routing_path.stat().st_size > 0:
                    routing_rows = _parquet_proof(
                        routing_path,
                        f"state.iterations.{label}.routing_mining_parquet",
                        {"filepath"},
                        errors,
                    )
                    if routing_rows is not None and routing_rows != 0:
                        errors.append(
                            f"state.iterations.{label}.data_mining_skipped is legal "
                            f"only for zero routed rows; found {routing_rows}"
                        )

        mining_fields = {
            "mining_mined_parquet": ({"filepath"}, False),
            "mining_target_embeddings": ({"filepath", "embedding"}, True),
            "mining_source_embeddings": ({"filepath", "embedding"}, True),
        }
        mined_rows: int | None = None
        if (label, "data_mining") in log_keys and not info.get(
            "data_mining_skipped"
        ):
            for field, (required_columns, require_rows) in mining_fields.items():
                value = info.get(field)
                if not value:
                    continue
                path = pathlib.Path(str(value)).expanduser()
                if not path.is_file() or path.stat().st_size == 0:
                    continue
                expected_phase_dir = (results_dir / label).resolve()
                try:
                    path.resolve().relative_to(expected_phase_dir)
                except ValueError:
                    errors.append(
                        f"state.iterations.{label}.{field} must be under "
                        f"{expected_phase_dir}"
                    )
                rows = _parquet_proof(
                    path,
                    f"state.iterations.{label}.{field}",
                    required_columns,
                    errors,
                )
                if field == "mining_mined_parquet":
                    mined_rows = rows
                if require_rows and rows is not None and rows <= 0:
                    errors.append(
                        f"state.iterations.{label}.{field} must contain rows"
                    )
            for field in (
                "mining_target_log",
                "mining_source_log",
                "mining_knn_log",
            ):
                value = info.get(field)
                if not value:
                    continue
                path = pathlib.Path(str(value)).expanduser()
                if path.is_file() and path.stat().st_size > 0:
                    expected_phase_dir = (results_dir / label).resolve()
                    try:
                        path.resolve().relative_to(expected_phase_dir)
                    except ValueError:
                        errors.append(
                            f"state.iterations.{label}.{field} must be under "
                            f"{expected_phase_dir}"
                        )
                    _tao_pass_log(
                        path, f"state.iterations.{label}.{field}", errors
                    )
            summary_value = info.get("mining_summary")
            if summary_value:
                summary_path = pathlib.Path(str(summary_value)).expanduser()
                if summary_path.is_file() and summary_path.stat().st_size > 0:
                    expected_phase_dir = (results_dir / label).resolve()
                    try:
                        summary_path.resolve().relative_to(expected_phase_dir)
                    except ValueError:
                        errors.append(
                            f"state.iterations.{label}.mining_summary must be under "
                            f"{expected_phase_dir}"
                        )
                    _mining_summary_proof(
                        summary_path,
                        mined_rows,
                        info.get("mining_mined_count"),
                        f"state.iterations.{label}.mining_summary",
                        errors,
                    )

        result: dict[str, Any] | None = None
        if contract is not None:
            try:
                result = result_from_iteration(info, contract)
                if result is not None:
                    iteration_results[label] = result
                    metric_candidates.append((label, info, result))
                    if result.get("unit", "") != contract.get("unit", ""):
                        errors.append(
                            f"state.iterations.{label}.metric_result.unit does not "
                            f"match metric_contract.unit"
                        )
                    computed_passed, failures = result_passes(contract, result)
                    if any(failure.endswith(":missing") for failure in failures):
                        errors.append(
                            f"state.iterations.{label}.metric_result is missing "
                            f"constraint values: {failures}"
                        )
                    recorded_passed = result.get("passed")
                    if recorded_passed is not None and (
                        not isinstance(recorded_passed, bool)
                        or recorded_passed != computed_passed
                    ):
                        errors.append(
                            f"state.iterations.{label}.metric_result.passed does not "
                            f"match the metric contract comparison"
                        )
                    if "metric_result" in info:
                        evidence = result.get("evidence_path")
                        if not evidence:
                            errors.append(
                                f"state.iterations.{label}.metric_result.evidence_path "
                                f"is required"
                            )
                        else:
                            evidence_path = pathlib.Path(str(evidence)).expanduser()
                            if not evidence_path.is_absolute():
                                errors.append(
                                    f"state.iterations.{label}.metric_result.evidence_path "
                                    f"must be absolute"
                                )
                            elif not evidence_path.is_file():
                                errors.append(
                                    f"state.iterations.{label}.metric_result.evidence_path "
                                    f"does not exist: {evidence}"
                                )
                            elif evidence_path.stat().st_size == 0:
                                errors.append(
                                    f"state.iterations.{label}.metric_result.evidence_path "
                                    f"is empty: {evidence}"
                                )
                            else:
                                if contract["evaluator"]["type"] == "artifact":
                                    expected_evidence = pathlib.Path(
                                        contract["evaluator"]["path_template"].replace(
                                            "{iter_label}", label
                                        )
                                    ).expanduser().resolve()
                                    if evidence_path.resolve() != expected_evidence:
                                        errors.append(
                                            f"state.iterations.{label}.metric_result."
                                            "evidence_path does not match the configured "
                                            f"artifact path: {expected_evidence}"
                                        )
                                try:
                                    evidence_payload = json.loads(
                                        evidence_path.read_text()
                                    )
                                    evidence_result = result_from_iteration(
                                        {"metric_result": evidence_payload}, contract
                                    )
                                except (OSError, ValueError, json.JSONDecodeError) as exc:
                                    errors.append(
                                        f"state.iterations.{label}.metric_result evidence "
                                        f"is invalid: {exc}"
                                    )
                                else:
                                    if evidence_result is None or any(
                                        evidence_result.get(field) != result.get(field)
                                        for field in ("name", "value", "unit", "constraints")
                                    ):
                                        errors.append(
                                            f"state.iterations.{label}.metric_result does not "
                                            "match its evidence JSON"
                                        )
                    if info.get("far_pct") is not None and contract["name"] == "far_pct":
                        if float(info["far_pct"]) != float(result["value"]):
                            errors.append(
                                f"state.iterations.{label}.far_pct disagrees with "
                                f"metric_result.value"
                            )
            except ValueError as exc:
                errors.append(f"state.iterations.{label} metric result is invalid: {exc}")

        if iter_status == "complete":
            for field in ("best_ckpt_path", "inference_csv"):
                if info.get(field) is None:
                    errors.append(
                        f"complete iteration {label} is missing required field {field}"
                    )
            if result is None:
                errors.append(
                    f"complete iteration {label} is missing metric_result for "
                    f"{contract['name'] if contract else 'the configured metric'}"
                )
            if (
                contract is not None
                and contract["evaluator"].get("id") == "far_at_recall"
                and info.get("threshold") is None
            ):
                errors.append(
                    f"complete metric iteration {label} is missing required field threshold"
                )
            match = re.fullmatch(r"iter([1-9][0-9]*)", label)
            if match:
                completed_iteration_numbers.append(int(match.group(1)))
            if completed not in {"evaluate", "rca"}:
                errors.append(
                    f"complete iteration {label} must end at evaluate or rca, got {completed!r}"
                )

    best: tuple[str, dict[str, Any], dict[str, Any]] | None = None
    if contract is not None and metric_candidates:
        best = pick_best(metric_candidates, contract)

    for entry in entries:
        label = str(entry.get("iter"))
        info = iterations.get(label)
        if not isinstance(info, dict):
            errors.append(
                f"loop_log commits {label}/{entry.get('stage')} but "
                f"state.iterations.{label} is missing"
            )
            continue
        if entry.get("status") == "error" and info.get("status") != "failed":
            errors.append(
                f"loop_log records {label}/{entry.get('stage')} error but "
                f"state.iterations.{label}.status is not 'failed'"
            )
        if entry.get("status") != "ok":
            continue
        stage = str(entry.get("stage"))
        alternatives = STAGE_REQUIRED_FIELD_SETS.get(stage)
        if not alternatives:
            continue

        def _alternative_is_set(fields: tuple[str, ...]) -> bool:
            for field in fields:
                value = info.get(field)
                if field.endswith("_skipped"):
                    if value is not True:
                        return False
                elif value is None or value == "":
                    return False
            return True

        if not any(_alternative_is_set(fields) for fields in alternatives):
            rendered = " or ".join("+".join(fields) for fields in alternatives)
            errors.append(
                f"loop_log commits {label}/{stage} but state lacks required proof: {rendered}"
            )
        if stage == "evaluate" and label not in iteration_results:
            errors.append(
                f"loop_log commits {label}/evaluate but state lacks a valid result "
                f"for the configured primary metric"
            )

    last = entries[-1] if entries else None
    terminal = bool(last and last.get("stage") == "loop_stop")
    error_entries = [entry for entry in entries if entry.get("status") == "error"]
    if terminal and not error_entries:
        target_met = False
        if contract is not None:
            for label, result in iteration_results.items():
                info = iterations.get(label)
                if isinstance(info, dict) and info.get("status") == "complete":
                    try:
                        if result_passes(contract, result)[0]:
                            target_met = True
                            break
                    except ValueError:
                        pass
        max_iterations = state.get("max_iterations")
        reached_max = bool(
            completed_iteration_numbers
            and isinstance(max_iterations, int)
            and not isinstance(max_iterations, bool)
            and max(completed_iteration_numbers) >= max_iterations
        )
        if not target_met and not reached_max:
            errors.append(
                "loop_stop has no successful stop proof: KPI is not met and "
                "no completed iteration reaches max_iterations"
            )
    if errors:
        status = "INVALID"
    elif terminal and error_entries:
        status = "FAILED"
        warnings.append("run ended after a hard stop; do not claim KPI completion")
    elif terminal and last and last.get("status") == "ok":
        status = "COMPLETE"
    elif last and last.get("status") == "error":
        status = "FAILED"
        warnings.append("last committed stage is a hard stop; do not auto-retry")
    else:
        status = "IN_PROGRESS"

    next_action, required_reference = _next_action(state, entries, status, contract)
    best_result = best[2] if best else None
    return {
        "status": status,
        "terminal": terminal,
        "results_dir": str(results_dir),
        "max_iterations": state.get("max_iterations"),
        "current_iteration": state.get("current_iteration"),
        "kpi_target": state.get("kpi_target"),
        "metric_contract": contract,
        "log_entries": len(entries),
        "last_committed": last,
        "best_iteration": best[0] if best else None,
        "best_metric_result": best_result,
        "best_far_pct": (
            best_result["value"]
            if best_result and contract and contract["name"] == "far_pct"
            else None
        ),
        "next_action": next_action,
        "required_reference": required_reference,
        "errors": errors,
        "warnings": warnings,
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"DEFT_RUN_STATUS={report['status']}")
    print(f"results_dir={report['results_dir']}")
    print(
        f"iteration={report['current_iteration']}/{report['max_iterations']} "
        f"log_entries={report['log_entries']}"
    )
    if report["last_committed"]:
        last = report["last_committed"]
        print(
            "last_committed="
            f"seq:{last.get('seq')} {last.get('iter')}/{last.get('stage')} "
            f"status:{last.get('status')}"
        )
    else:
        print("last_committed=none")
    if report["best_iteration"] is not None:
        contract = report["metric_contract"]
        result = report["best_metric_result"]
        unit = result.get("unit", "")
        suffix = unit if unit == "%" else (f" {unit}" if unit else "")
        print(
            f"best={report['best_iteration']} metric={contract['name']} "
            f"value={result['value']:.6g}{suffix} target={render_target(contract)}"
        )
    print(f"next_action={report['next_action']}")
    if report["required_reference"]:
        print(f"read_before_action={report['required_reference']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}")


def _completion_report_error(results_dir: pathlib.Path) -> str | None:
    """Return why the deterministic final HTML is missing/stale/invalid."""
    results_dir = results_dir.expanduser().resolve()
    report_path = results_dir / "DEFT_Loop_Report.html"
    if not report_path.is_file() or report_path.stat().st_size == 0:
        return f"final HTML report is missing or empty: {report_path}"
    evidence = [results_dir / "deft_state.json", results_dir / "loop_log.jsonl"]
    newest_evidence = max(
        (path.stat().st_mtime_ns for path in evidence if path.exists()),
        default=0,
    )
    if report_path.stat().st_mtime_ns < newest_evidence:
        return "final HTML report is older than canonical state/log; rerun render_report.py"
    text = report_path.read_text(encoding="utf-8")
    required = ("DEFT Loop Final Report", "Final Status", "--nvidia-green: #76b900")
    missing = [token for token in required if token not in text]
    if missing:
        return "final HTML report is missing required content: " + ", ".join(missing)
    if re.search(r"\{\{\s+[A-Z0-9_]+\s+\}\}", text):
        return "final HTML report contains unfilled placeholders"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--require-terminal",
        action="store_true",
        help="fail unless the last committed event is loop_stop (failed runs allowed)",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless a successful loop_stop has KPI/max-iteration proof",
    )
    args = parser.parse_args(argv)
    try:
        report = audit(args.results_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"audit_deft_run: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text(report)
    if report["status"] == "INVALID":
        return 1
    if args.require_terminal and not report["terminal"]:
        return 1
    if args.require_complete:
        if report["status"] != "COMPLETE":
            return 1
        report_error = _completion_report_error(args.results_dir)
        if report_error:
            print(f"ERROR: {report_error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
