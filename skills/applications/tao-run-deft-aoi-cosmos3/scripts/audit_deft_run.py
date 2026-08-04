#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Read-only audit and resume oracle for Cosmos3 DEFT AOI runs."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

from metric_contract import (
    contract_from_state,
    pick_best,
    render_target,
    result_from_iteration,
    result_passes,
)
from validate_sharegpt import load_records, validate_records
from validate_split_contract import validate as validate_splits


VALID_STAGES = {
    "train",
    "evaluate_proxy",
    "proxy_rcca",
    "evaluate_benchmark",
    "benchmark_metrics",
    "routing",
    "anomalygen",
    "data_mining",
    "assemble_data",
    "validate_data",
    "loop_stop",
}
VALID_ITERATION_STATUSES = {"pending", "in_progress", "complete", "failed"}
VALID_LOG_STATUSES = {"ok", "error"}
PATH_FIELDS = {
    "best_ckpt_path",
    "training_spec",
    "proxy_results_json",
    "proxy_gaps_summary",
    "false_accepts_json",
    "false_rejects_json",
    "benchmark_results_json",
    "benchmark_metrics_summary",
    "mining_targets_json",
    "anomalygen_sdg_csv",
    "anomalygen_sharegpt_json",
    "mining_mined_parquet",
    "mining_summary",
    "mining_target_embeddings",
    "mining_source_embeddings",
    "mined_sharegpt_json",
    "combined_training_json",
    "assemble_summary",
    "validation_report",
}
STAGE_REQUIRED_FIELDS = {
    "train": ("best_ckpt_path", "training_spec"),
    "evaluate_proxy": ("proxy_results_json",),
    "proxy_rcca": (
        "proxy_gaps_summary",
        "false_accepts_json",
        "false_rejects_json",
    ),
    "evaluate_benchmark": ("benchmark_results_json",),
    "benchmark_metrics": (
        "benchmark_metrics_summary",
        "metric_result",
    ),
    "routing": ("mining_targets_json",),
    "data_mining": (
        "mining_mined_parquet",
        "mining_summary",
        "mining_target_embeddings",
        "mining_source_embeddings",
        "mining_mined_count",
    ),
    "anomalygen": ("anomalygen_sdg_csv", "anomalygen_sharegpt_json"),
    "assemble_data": (
        "mined_sharegpt_json",
        "combined_training_json",
        "assemble_summary",
    ),
    "validate_data": ("validation_report",),
}
# A stage may record a documented branch skip instead of its artifacts. The
# skip itself still has to be justified against disk evidence below.
SKIP_FLAGS = {"anomalygen": "anomalygen_skipped"}
FIELD_STAGE = {
    field: stage
    for stage, fields in STAGE_REQUIRED_FIELDS.items()
    for field in fields
}


def _driving_label(label: str) -> str | None:
    """Return the label whose Proxy RCCA drives this iteration's augmentation."""
    number = _iteration_number(label)
    if number < 1:
        return None
    return "baseline" if number == 1 else f"iter{number - 1}"


def _load_state(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("deft_state.json root must be an object")
    return payload


def _load_log(path: pathlib.Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"loop_log.jsonl:{line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(entry, dict):
            errors.append(f"loop_log.jsonl:{line_number}: entry must be an object")
            continue
        entries.append(entry)
    return entries


def _iteration_number(label: str) -> int:
    if label == "baseline":
        return 0
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    return int(match.group(1)) if match else -1


def _expected_next(entry: dict[str, Any]) -> set[tuple[str, str]]:
    label = str(entry.get("iter", ""))
    stage = str(entry.get("stage", ""))
    if entry.get("status") == "error":
        return {(label, "loop_stop")}
    if stage == "loop_stop":
        return set()
    # Gate first: the frozen Benchmark decides whether the run continues, so it
    # is evaluated before any Proxy work. Proxy evaluate/RCCA only run when the
    # gate is unmet, and exist solely to seed the NEXT iteration's mining.
    if stage == "train":
        return {(label, "evaluate_benchmark")}
    if stage == "evaluate_benchmark":
        return {(label, "benchmark_metrics")}
    if stage == "benchmark_metrics":
        return {(label, "loop_stop"), (label, "evaluate_proxy")}
    if stage == "evaluate_proxy":
        return {(label, "proxy_rcca")}
    if stage == "proxy_rcca":
        number = _iteration_number(label)
        next_label = "iter1" if number == 0 else f"iter{number + 1}"
        return {(label, "loop_stop"), (next_label, "routing")}
    if stage == "routing":
        return {(label, "anomalygen")}
    if stage == "anomalygen":
        return {(label, "data_mining")}
    if stage == "data_mining":
        return {(label, "assemble_data")}
    if stage == "assemble_data":
        return {(label, "validate_data")}
    if stage == "validate_data":
        return {(label, "train")}
    return set()


def _next_action(
    state: dict[str, Any],
    entries: list[dict[str, Any]],
    status: str,
    contract: dict[str, Any] | None,
) -> tuple[str, str | None]:
    if status == "INVALID":
        return "repair disk-state inconsistencies before another stage", None
    if status == "FAILED":
        return (
            "surface the committed hard stop; do not retry automatically",
            "references/pipeline-and-state.md",
        )
    if status == "COMPLETE":
        return (
            "hand off the best evaluated model and auto-rendered final report",
            None,
        )
    if not entries:
        return (
            "baseline frozen Benchmark evaluate on the base model",
            "references/cosmos-reason.md",
        )
    last = entries[-1]
    label = str(last.get("iter"))
    stage = str(last.get("stage"))
    mapping = {
        "train": (
            f"{label} frozen Benchmark evaluate",
            "references/cosmos-reason.md",
        ),
        "evaluate_benchmark": (
            f"{label} Benchmark KPI analysis",
            "references/gap-analysis.md",
        ),
        "evaluate_proxy": (
            f"{label} Proxy RCCA",
            "references/gap-analysis.md",
        ),
        "routing": (
            f"{label} AnomalyGen synthetic defect generation",
            "references/paidf-anomalygen.md",
        ),
        "anomalygen": (
            f"{label} real-pair data mining",
            "references/tao-mine-aoi-images.md",
        ),
        "data_mining": (
            f"{label} mined-pair alignment and bare OK/NG assembly",
            "references/aoi-annotation.md",
        ),
        "assemble_data": (
            f"{label} bare OK/NG validation",
            "references/aoi-annotation.md",
        ),
        "validate_data": (
            f"{label} Cosmos3 retrain",
            "references/cosmos-reason.md",
        ),
        "loop_stop": (
            "run the remaining loop-end sequence",
            "references/pipeline-and-state.md",
        ),
    }
    if stage == "benchmark_metrics":
        info = state.get("iterations", {}).get(label, {})
        metric_met = False
        if contract is not None and isinstance(info, dict):
            try:
                result = result_from_iteration(info, contract)
                metric_met = bool(result and result_passes(contract, result)[0])
            except ValueError:
                pass
        reached_max = (
            _iteration_number(label) >= int(state.get("max_iterations", 0))
            and label != "baseline"
        )
        if metric_met or reached_max:
            return (
                "commit loop_stop and run the loop-end sequence",
                "references/pipeline-and-state.md",
            )
        # Gate unmet: only now is Proxy worth evaluating, to seed next mining.
        return (
            f"{label} Proxy evaluate",
            "references/cosmos-reason.md",
        )
    if stage == "proxy_rcca":
        next_label = (
            "iter1"
            if label == "baseline"
            else f"iter{_iteration_number(label) + 1}"
        )
        return (
            f"{next_label} route Proxy gaps into mining targets",
            "references/gap-analysis.md",
        )
    return mapping.get(
        stage,
        (
            "inspect pipeline-and-state.md before continuing",
            "references/pipeline-and-state.md",
        ),
    )


def _path_proof(
    value: Any,
    *,
    field: str,
    phase_root: pathlib.Path | None,
    allow_directory: bool,
    errors: list[str],
) -> pathlib.Path | None:
    path = pathlib.Path(str(value)).expanduser()
    if not path.is_absolute():
        errors.append(f"{field} must be absolute")
        return None
    if not path.exists():
        errors.append(f"{field} does not exist: {path}")
        return None
    if allow_directory:
        if path.is_dir() and not any(path.iterdir()):
            errors.append(f"{field} directory is empty: {path}")
        elif path.is_file() and path.stat().st_size == 0:
            errors.append(f"{field} file is empty: {path}")
    elif not path.is_file() or path.stat().st_size == 0:
        errors.append(f"{field} must be a non-empty file: {path}")
    if phase_root is not None:
        try:
            path.resolve().relative_to(phase_root.resolve())
        except ValueError:
            errors.append(f"{field} must be under {phase_root}: {path}")
    return path.resolve()


def _json_object(
    path: pathlib.Path | None, field: str, errors: list[str]
) -> dict[str, Any] | None:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{field} is invalid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{field} root must be an object")
        return None
    return payload


def _json_list(
    path: pathlib.Path | None, field: str, errors: list[str]
) -> list[Any] | None:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{field} is invalid JSON: {exc}")
        return None
    if not isinstance(payload, list):
        errors.append(f"{field} root must be a list")
        return None
    return payload


def _parquet_rows(
    path: pathlib.Path | None,
    field: str,
    required_columns: set[str],
    errors: list[str],
) -> int | None:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        columns = set(parquet.schema_arrow.names)
    except Exception as exc:
        errors.append(f"{field} is not a readable parquet: {exc}")
        return None
    missing = sorted(required_columns - columns)
    if missing:
        errors.append(f"{field} is missing parquet columns {missing}")
    return rows


def audit(results_dir: pathlib.Path) -> dict[str, Any]:
    results_dir = results_dir.expanduser().resolve()
    state = _load_state(results_dir / "deft_state.json")
    errors: list[str] = []
    warnings: list[str] = []
    entries = _load_log(results_dir / "loop_log.jsonl", errors)

    if state.get("workflow") != "tao-run-deft-aoi-cosmos3":
        errors.append("state.workflow must be tao-run-deft-aoi-cosmos3")
    if state.get("version") != 3:
        errors.append("state.version must be 3")
    recorded_results = pathlib.Path(str(state.get("results_dir", ""))).expanduser()
    if not recorded_results.is_absolute() or recorded_results.resolve() != results_dir:
        errors.append("state.results_dir must match the audited absolute directory")
    for field in ("max_iterations", "current_iteration"):
        value = state.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"state.{field} must be an integer")
    if isinstance(state.get("max_iterations"), int) and state["max_iterations"] <= 0:
        errors.append("state.max_iterations must be > 0")

    try:
        contract = contract_from_state(state)
    except ValueError as exc:
        contract = None
        errors.append(f"invalid metric contract: {exc}")

    config = state.get("config")
    if not isinstance(config, dict):
        errors.append("state.config must be an object")
        config = {}
    if config.get("annotation_mode") != "bare_okng":
        errors.append("state.config.annotation_mode must be bare_okng")
    if config.get("model_skill") != "tao-finetune-cosmos-reason":
        errors.append("state.config.model_skill must be tao-finetune-cosmos-reason")
    if config.get("automl_policy") != "off":
        errors.append("Cosmos3 DEFT must record automl_policy=off")
    if (
        config.get("training", {}).get("annotation_source")
        != "generated_from_mining_and_anomalygen"
    ):
        errors.append(
            "state.config.training.annotation_source must be "
            "generated_from_mining_and_anomalygen"
        )
    if not isinstance(config.get("platform"), str) or not config.get("platform"):
        errors.append("state.config.platform is required")
    for name in ("cosmos_rl", "data_services"):
        value = config.get("containers", {}).get(name)
        if not isinstance(value, str) or not value:
            errors.append(f"state.config.containers.{name} is required")

    media_root = pathlib.Path(str(config.get("media_root", ""))).expanduser()
    if not media_root.is_absolute() or not media_root.is_dir():
        errors.append("state.config.media_root must be an existing absolute directory")
    annotation_values = config.get("annotations")
    role_paths: dict[str, pathlib.Path] = {}
    if not isinstance(annotation_values, dict):
        errors.append("state.config.annotations must be an object")
    else:
        for role in ("proxy", "benchmark", "mining"):
            value = annotation_values.get(role)
            if not value:
                errors.append(f"state.config.annotations.{role} is required")
                continue
            path = _path_proof(
                value,
                field=f"state.config.annotations.{role}",
                phase_root=None,
                allow_directory=False,
                errors=errors,
            )
            if path is not None:
                role_paths[role] = path
                try:
                    validate_records(
                        load_records(path),
                        media_root=media_root,
                        require_files=False,
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(
                        f"state.config.annotations.{role} is invalid: {exc}"
                    )
    if len(role_paths) == 3 and media_root.is_dir():
        expected_hash = (
            config.get("evaluation", {})
            .get("benchmark", {})
            .get("sha256")
        )
        try:
            validate_splits(
                role_paths,
                media_root=media_root,
                expected_benchmark_sha256=expected_hash,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"split contract is invalid: {exc}")

    specs = config.get("specs")
    if not isinstance(specs, dict):
        errors.append("state.config.specs must be an object")
    else:
        for role in ("train", "proxy", "benchmark"):
            _path_proof(
                specs.get(role),
                field=f"state.config.specs.{role}",
                phase_root=None,
                allow_directory=False,
                errors=errors,
            )

    log_keys: set[tuple[str, str]] = set()
    last_successful: dict[str, str] = {}
    expected_seq = 1
    for index, entry in enumerate(entries):
        seq = entry.get("seq")
        if seq != expected_seq:
            errors.append(f"loop_log seq={seq!r}; expected {expected_seq}")
        expected_seq += 1
        label = entry.get("iter")
        stage = entry.get("stage")
        status = entry.get("status")
        if label != "baseline" and not (
            isinstance(label, str) and re.fullmatch(r"iter[1-9][0-9]*", label)
        ):
            errors.append(f"loop_log seq={seq}: invalid iter {label!r}")
        if stage not in VALID_STAGES:
            errors.append(f"loop_log seq={seq}: invalid stage {stage!r}")
        if status not in VALID_LOG_STATUSES:
            errors.append(f"loop_log seq={seq}: invalid status {status!r}")
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
            errors.append(f"duplicate stage event: {key}")
        log_keys.add(key)
        if status == "ok" and stage != "loop_stop":
            last_successful[str(label)] = str(stage)
        if index == 0 and key != ("baseline", "evaluate_benchmark"):
            errors.append(
                "the first log event must be baseline/evaluate_benchmark"
            )
        elif index:
            previous = entries[index - 1]
            allowed = _expected_next(previous)
            if key not in allowed:
                rendered = ", ".join(
                    f"{label}/{name}" for label, name in sorted(allowed)
                )
                errors.append(
                    f"illegal transition {previous.get('iter')}/"
                    f"{previous.get('stage')} -> {key[0]}/{key[1]}; "
                    f"expected [{rendered or 'end-of-log'}]"
                )

    iterations = state.get("iterations")
    if not isinstance(iterations, dict):
        errors.append("state.iterations must be an object")
        iterations = {}
    expected_current = max(
        (_iteration_number(label) for label in iterations),
        default=0,
    )
    expected_current = max(expected_current, 0)
    if state.get("current_iteration") != expected_current:
        errors.append(
            f"state.current_iteration={state.get('current_iteration')} does not "
            f"match highest iteration key {expected_current}"
        )

    metric_candidates: list[
        tuple[str, dict[str, Any], dict[str, Any]]
    ] = []
    complete_numbers: list[int] = []
    error_labels = {
        str(entry.get("iter"))
        for entry in entries
        if entry.get("status") == "error"
    }
    for label, phase in sorted(
        iterations.items(), key=lambda item: _iteration_number(item[0])
    ):
        if label != "baseline" and _iteration_number(label) < 1:
            errors.append(f"invalid iteration key: {label!r}")
        if not isinstance(phase, dict):
            errors.append(f"state.iterations.{label} must be an object")
            continue
        phase_status = phase.get("status")
        if phase_status not in VALID_ITERATION_STATUSES:
            errors.append(
                f"state.iterations.{label}.status={phase_status!r} is invalid"
            )
        if phase_status == "failed" and label not in error_labels:
            errors.append(
                f"state.iterations.{label} is failed without an error log event"
            )
        completed = phase.get("stage_completed")
        if completed is not None and completed not in VALID_STAGES - {"loop_stop"}:
            errors.append(
                f"state.iterations.{label}.stage_completed={completed!r} is invalid"
            )
        expected_completed = last_successful.get(label)
        if expected_completed is not None and completed != expected_completed:
            errors.append(
                f"state.iterations.{label}.stage_completed={completed!r}; "
                f"last successful log stage is {expected_completed!r}"
            )

        phase_root = results_dir / label
        for field in PATH_FIELDS:
            value = phase.get(field)
            if not value:
                continue
            path = _path_proof(
                value,
                field=f"state.iterations.{label}.{field}",
                phase_root=None if field == "training_spec" else phase_root,
                allow_directory=field == "best_ckpt_path",
                errors=errors,
            )
            stage = FIELD_STAGE.get(field)
            if stage and (label, stage) not in log_keys:
                errors.append(
                    f"state.iterations.{label}.{field} is set without "
                    f"{label}/{stage} in loop_log"
                )
            if field == "proxy_gaps_summary":
                payload = _json_object(path, field, errors)
                if payload and (
                    payload.get("evaluation_role") != "proxy"
                    or payload.get("aggregate_only") is not False
                ):
                    errors.append(
                        f"{field} must be Proxy-only, non-aggregate RCCA output"
                    )
            if field in {
                "proxy_results_json",
                "benchmark_results_json",
                "mining_targets_json",
                "mined_sharegpt_json",
                "combined_training_json",
            }:
                payload = _json_list(path, field, errors)
                if field in {
                    "proxy_results_json",
                    "benchmark_results_json",
                    "mining_targets_json",
                    "mined_sharegpt_json",
                    "combined_training_json",
                } and payload is not None and not payload:
                    errors.append(f"{field} must contain at least one record")
                if (
                    field == "combined_training_json"
                    and path is not None
                    and len(role_paths) == 3
                    and media_root.is_dir()
                ):
                    train_roles = {**role_paths, "train": path}
                    # SKILL.md: a generated Train file's targets must come from
                    # Mining "or, with --synthetic, from the iteration's
                    # AnomalyGen output". Supply the recorded synthetic role so
                    # an iteration that actually ran AnomalyGen can be audited;
                    # without it, only the `anomalygen --skip` branch is ever
                    # committable.
                    synthetic_value = phase.get("anomalygen_sharegpt_json")
                    if synthetic_value:
                        train_roles["synthetic"] = pathlib.Path(synthetic_value)
                    try:
                        validate_splits(
                            train_roles,
                            media_root=media_root,
                            expected_benchmark_sha256=(
                                config.get("evaluation", {})
                                .get("benchmark", {})
                                .get("sha256")
                            ),
                        )
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        errors.append(
                            f"{field} violates generated Train lineage: {exc}"
                        )
            if field == "benchmark_metrics_summary":
                payload = _json_object(path, field, errors)
                if payload and (
                    payload.get("evaluation_role") != "benchmark"
                    or payload.get("kpi", {}).get("gate_eligible") is not True
                ):
                    errors.append(
                        f"{field} must be frozen Benchmark gate output"
                    )
            if field == "assemble_summary":
                payload = _json_object(path, field, errors)
                if payload and (
                    payload.get("mode") != "bare_okng"
                    or not isinstance(payload.get("output_records"), int)
                    or payload["output_records"] <= 0
                ):
                    errors.append(
                        f"{field} must prove non-empty bare_okng assembly"
                    )
                if payload:
                    expected_exclusions = {
                        str(role_paths.get("proxy")),
                        str(role_paths.get("benchmark")),
                    }
                    recorded_exclusions = {
                        str(value)
                        for value in payload.get("validation_jsons", [])
                    }
                    if recorded_exclusions != expected_exclusions:
                        errors.append(
                            f"{field} must record exact Proxy and Benchmark "
                            "leakage exclusions"
                        )
            if field == "validation_report":
                payload = _json_object(path, field, errors)
                if payload and (
                    payload.get("mode") != "bare_okng"
                    or not isinstance(payload.get("records"), int)
                    or payload["records"] <= 0
                ):
                    errors.append(
                        f"{field} must prove non-empty bare_okng validation; "
                        "it is the validate_sharegpt.py --summary output, not "
                        "validate_split_contract.py --summary"
                    )

        if (label, "data_mining") in log_keys:
            mined_rows = _parquet_rows(
                pathlib.Path(str(phase.get("mining_mined_parquet", ""))),
                f"state.iterations.{label}.mining_mined_parquet",
                {"filepath"},
                errors,
            )
            for field in (
                "mining_target_embeddings",
                "mining_source_embeddings",
            ):
                _parquet_rows(
                    pathlib.Path(str(phase.get(field, ""))),
                    f"state.iterations.{label}.{field}",
                    {"filepath", "embedding"},
                    errors,
                )
            recorded_count = phase.get("mining_mined_count")
            if (
                not isinstance(recorded_count, int)
                or isinstance(recorded_count, bool)
                or recorded_count <= 0
            ):
                errors.append(
                    f"state.iterations.{label}.mining_mined_count must be > 0"
                )
            elif mined_rows is not None and mined_rows != recorded_count:
                errors.append(
                    f"state.iterations.{label}.mining_mined_count="
                    f"{recorded_count} disagrees with parquet rows={mined_rows}"
                )
            summary_path = pathlib.Path(
                str(phase.get("mining_summary", ""))
            )
            summary = _json_object(
                summary_path,
                f"state.iterations.{label}.mining_summary",
                errors,
            )
            if summary is not None and summary.get("kept_rows") != recorded_count:
                errors.append(
                    f"state.iterations.{label}.mining_summary.kept_rows "
                    "must match mining_mined_count"
                )

        for stage, fields in STAGE_REQUIRED_FIELDS.items():
            if (label, stage) not in log_keys:
                continue
            # A hard stop is usually a job that died BEFORE writing its
            # artifact, and SKILL.md requires committing it as an error stage.
            # Demanding the artifacts anyway would make that commit impossible,
            # so a failed phase only has to carry its summary.
            if phase_status == "failed":
                continue
            skip_flag = SKIP_FLAGS.get(stage)
            if skip_flag and phase.get(skip_flag):
                present = [field for field in fields if phase.get(field)]
                if present:
                    errors.append(
                        f"state.iterations.{label}.{skip_flag} cannot be set "
                        f"alongside {present}"
                    )
                continue
            missing = [
                field
                for field in fields
                if phase.get(field) is None or phase.get(field) == ""
            ]
            if missing:
                errors.append(
                    f"loop_log commits {label}/{stage} but state lacks {missing}"
                )

        if (label, "anomalygen") in log_keys and phase.get("anomalygen_skipped"):
            # A skip is legal only when the driving Proxy RCCA found no false
            # accepts: with nothing being under-detected, synthesizing more
            # defects has no gap to close.
            driver = _driving_label(label)
            driving_phase = iterations.get(driver) if driver else None
            evidence = (
                driving_phase.get("false_accepts_json")
                if isinstance(driving_phase, dict)
                else None
            )
            if not evidence:
                errors.append(
                    f"state.iterations.{label}.anomalygen_skipped has no "
                    f"{driver}.false_accepts_json proof"
                )
            else:
                evidence_path = pathlib.Path(str(evidence)).expanduser()
                if evidence_path.is_file():
                    try:
                        payload = json.loads(evidence_path.read_text())
                    except (OSError, json.JSONDecodeError) as exc:
                        errors.append(
                            f"state.iterations.{label}.anomalygen_skipped "
                            f"proof is unreadable: {exc}"
                        )
                    else:
                        count = len(payload) if isinstance(payload, list) else None
                        if count:
                            errors.append(
                                f"state.iterations.{label}.anomalygen_skipped is "
                                f"legal only for zero {driver} false accepts; "
                                f"found {count}"
                            )
        result: dict[str, Any] | None = None
        if contract is not None:
            try:
                result = result_from_iteration(phase, contract)
            except ValueError as exc:
                errors.append(
                    f"state.iterations.{label}.metric_result is invalid: {exc}"
                )
            if result is not None:
                metric_candidates.append((label, phase, result))
                try:
                    passed, failures = result_passes(contract, result)
                except ValueError as exc:
                    errors.append(
                        f"state.iterations.{label}.metric_result is invalid: {exc}"
                    )
                else:
                    if result.get("passed") is not passed:
                        errors.append(
                            f"state.iterations.{label}.metric_result.passed "
                            f"disagrees with contract; failures={failures}"
                        )
                evidence = result.get("evidence_path")
                if not evidence:
                    errors.append(
                        f"state.iterations.{label}.metric_result.evidence_path "
                        "is required"
                    )
                else:
                    evidence_path = _path_proof(
                        evidence,
                        field=(
                            f"state.iterations.{label}."
                            "metric_result.evidence_path"
                        ),
                        phase_root=phase_root,
                        allow_directory=False,
                        errors=errors,
                    )
                    payload = _json_object(
                        evidence_path, "metric_result.evidence_path", errors
                    )
                    if payload and any(
                        payload.get(field) != result.get(field)
                        for field in ("name", "value", "unit", "constraints")
                    ):
                        errors.append(
                            f"state.iterations.{label}.metric_result does not "
                            "match evidence JSON"
                        )
        if phase_status == "complete":
            required = [
                "evaluated_model",
                "benchmark_results_json",
                "benchmark_metrics_summary",
                "metric_result",
            ]
            # Proxy only runs when the gate was unmet and the loop continued,
            # so its artifacts are required exactly for a continuing iteration.
            if completed == "proxy_rcca":
                required.append("proxy_results_json")
            if label != "baseline":
                required.extend(
                    (
                        "best_ckpt_path",
                        "training_spec",
                        "combined_training_json",
                        "validation_report",
                    )
                )
            missing = [field for field in required if not phase.get(field)]
            if missing:
                errors.append(
                    f"complete iteration {label} is missing {missing}"
                )
            if completed not in ("benchmark_metrics", "proxy_rcca"):
                errors.append(
                    f"complete iteration {label} must end at "
                    "benchmark_metrics (gate stop) or proxy_rcca (continuing)"
                )
            number = _iteration_number(label)
            if number > 0:
                complete_numbers.append(number)

    best = (
        pick_best(metric_candidates, contract)
        if metric_candidates and contract is not None
        else None
    )
    if contract is not None:
        # Any step past a decided gate is a continuation: first the Proxy work
        # that seeds the next round, then that round's routing. Both are
        # illegal once the gate passed or max_iterations was reached.
        continuations = {
            ("benchmark_metrics", "evaluate_proxy"),
            ("proxy_rcca", "routing"),
        }
        for previous, current in zip(entries, entries[1:]):
            if (
                str(previous.get("stage")),
                str(current.get("stage")),
            ) not in continuations:
                continue
            previous_label = str(previous.get("iter"))
            previous_phase = iterations.get(previous_label)
            if not isinstance(previous_phase, dict):
                continue
            try:
                previous_result = result_from_iteration(
                    previous_phase, contract
                )
                passed = bool(
                    previous_result
                    and result_passes(contract, previous_result)[0]
                )
            except ValueError:
                passed = False
            reached_max = (
                previous_label != "baseline"
                and _iteration_number(previous_label)
                >= int(state.get("max_iterations", 0))
            )
            if passed:
                errors.append(
                    f"loop continued to {current.get('iter')}/"
                    f"{current.get('stage')} after "
                    f"{previous_label} passed the frozen Benchmark gate"
                )
            if reached_max:
                errors.append(
                    f"loop continued to {current.get('iter')}/"
                    f"{current.get('stage')} after "
                    f"{previous_label} reached max_iterations"
                )
    for entry in entries:
        label = str(entry.get("iter"))
        phase = iterations.get(label)
        if not isinstance(phase, dict):
            errors.append(
                f"loop_log commits {label}/{entry.get('stage')} but "
                f"state.iterations.{label} is missing"
            )
            continue
        if entry.get("status") == "error" and phase.get("status") != "failed":
            errors.append(
                f"{label}/{entry.get('stage')} is error but phase is not failed"
            )

    last = entries[-1] if entries else None
    terminal = bool(last and last.get("stage") == "loop_stop")
    error_entries = [entry for entry in entries if entry.get("status") == "error"]
    if terminal and not error_entries:
        target_met = False
        if contract is not None:
            for _, phase, result in metric_candidates:
                if phase.get("status") == "complete":
                    try:
                        if result_passes(contract, result)[0]:
                            target_met = True
                    except ValueError:
                        pass
        reached_max = bool(
            complete_numbers
            and isinstance(state.get("max_iterations"), int)
            and max(complete_numbers) >= state["max_iterations"]
        )
        if not target_met and not reached_max:
            errors.append(
                "loop_stop has no proof: Benchmark KPI is unmet and no "
                "completed iteration reaches max_iterations"
            )

    if errors:
        status = "INVALID"
    elif terminal and error_entries:
        status = "FAILED"
        warnings.append("run ended after a hard stop; do not claim KPI completion")
    elif terminal:
        status = "COMPLETE"
    elif last and last.get("status") == "error":
        status = "FAILED"
        warnings.append("last stage is a hard stop; append loop_stop only")
    else:
        status = "IN_PROGRESS"

    next_action, required_reference = _next_action(
        state, entries, status, contract
    )
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
        "best_metric_result": best[2] if best else None,
        "next_action": next_action,
        "required_reference": required_reference,
        "errors": errors,
        "warnings": warnings,
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"DEFT_RUN_STATUS={report['status']}")
    print(f"results_dir={report['results_dir']}")
    print(
        f"iteration={report['current_iteration']}/"
        f"{report['max_iterations']} log_entries={report['log_entries']}"
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
        result = report["best_metric_result"]
        print(
            f"best={report['best_iteration']} "
            f"metric={result['name']} value={result['value']:.6g} "
            f"target={render_target(report['metric_contract'])}"
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
    required = (
        "NVIDIA TAO · DEFT AOI",
        "Dataset Isolation",
        "Prompt Examples",
        "Hard Stops / Warnings",
        "--nvidia-green: #76b900",
    )
    missing = [token for token in required if token not in text]
    if missing:
        return "final HTML report is missing required content: " + ", ".join(missing)
    if re.search(r"\{\{\s+[A-Z0-9_]+\s+\}\}", text):
        return "final HTML report contains unfilled placeholders"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-terminal", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
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
