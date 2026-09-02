#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Apply an authorized, auditable DEFT operator contract change atomically."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: pathlib.Path) -> int:
    with path.open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


def _required_file(value: Any, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty absolute file path")
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{label} must be an existing non-empty absolute file: {path}")
    return path.resolve()


def _verify_jsonl(
    payload: dict[str, Any],
    *,
    path_key: str,
    rows_key: str,
    sha_key: str,
    label: str,
) -> tuple[pathlib.Path, int, str]:
    path = _required_file(payload.get(path_key), f"{label}.{path_key}")
    expected_rows = payload.get(rows_key)
    if not isinstance(expected_rows, int) or isinstance(expected_rows, bool) or expected_rows <= 0:
        raise ValueError(f"{label}.{rows_key} must be a positive integer")
    actual_rows = _rows(path)
    if actual_rows != expected_rows:
        raise ValueError(
            f"{label} row count mismatch: expected {expected_rows}, got {actual_rows}"
        )
    expected_sha = payload.get(sha_key)
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"{label}.{sha_key} must be a SHA-256 hex digest")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
    return path, actual_rows, actual_sha


def _write_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def apply(state_path: pathlib.Path, exception_path: pathlib.Path) -> dict[str, Any]:
    state_path = state_path.expanduser().resolve()
    exception_path = exception_path.expanduser().resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    exception = json.loads(exception_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("version") != 7:
        raise ValueError("state schema is not the Cosmos Framework v7 contract")
    if state.get("status") != "in_progress":
        raise ValueError("operator contract changes require an in-progress run")
    if not isinstance(exception, dict) or exception.get("schema") != "deft_operator_contract_exception_v1":
        raise ValueError("unsupported operator contract exception schema")
    for key in ("authorized_by", "authorization_date", "scope"):
        if not isinstance(exception.get(key), str) or not exception[key]:
            raise ValueError(f"operator exception requires {key}")

    replacement = exception.get("benchmark_replacement")
    proxy = exception.get("proxy")
    overlap = exception.get("authorized_overlap_exception")
    if not isinstance(replacement, dict) or not isinstance(proxy, dict) or not isinstance(overlap, dict):
        raise ValueError("operator exception is missing benchmark, proxy, or overlap evidence")
    if overlap.get("cohort_mutation_allowed") is not False:
        raise ValueError("authorized overlap exception must prohibit cohort mutation")
    for field in (
        "physical_target_overlap",
        "benchmark_rows_on_overlapping_targets",
        "proxy_rows_on_overlapping_targets",
    ):
        value = overlap.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"authorized_overlap_exception.{field} must be positive")

    active_path, active_rows, active_sha = _verify_jsonl(
        replacement,
        path_key="active_path",
        rows_key="active_rows",
        sha_key="active_sha256",
        label="active Benchmark",
    )
    retired_path, retired_rows, retired_sha = _verify_jsonl(
        replacement,
        path_key="retired_path",
        rows_key="retired_rows",
        sha_key="retired_sha256",
        label="retired Benchmark",
    )
    provenance_path = _required_file(
        replacement.get("provenance_path"), "benchmark_replacement.provenance_path"
    )
    if _sha256(provenance_path) != active_sha or _rows(provenance_path) != active_rows:
        raise ValueError("active Benchmark does not match its provenance copy")
    proxy_path, proxy_rows, proxy_sha = _verify_jsonl(
        proxy,
        path_key="path",
        rows_key="rows",
        sha_key="sha256",
        label="Proxy",
    )

    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError("state.config must be an object")
    annotations = config.get("annotations")
    seals = config.get("annotation_sha256")
    evaluation = config.get("evaluation")
    benchmark_eval = evaluation.get("benchmark") if isinstance(evaluation, dict) else None
    if not isinstance(annotations, dict) or not isinstance(seals, dict) or not isinstance(benchmark_eval, dict):
        raise ValueError("state is missing annotation/evaluation seals")
    if pathlib.Path(str(annotations.get("benchmark"))).resolve() != active_path:
        raise ValueError("active Benchmark path does not match state.config.annotations.benchmark")
    if pathlib.Path(str(benchmark_eval.get("annotations"))).resolve() != active_path:
        raise ValueError("active Benchmark path does not match state.config.evaluation.benchmark")
    if pathlib.Path(str(annotations.get("proxy"))).resolve() != proxy_path:
        raise ValueError("Proxy path does not match state.config.annotations.proxy")
    if seals.get("proxy") != proxy_sha:
        raise ValueError("Proxy seal changed; this operator exception does not authorize it")

    exception_sha = _sha256(exception_path)
    changes = state.setdefault("operator_contract_changes", [])
    if not isinstance(changes, list):
        raise ValueError("state.operator_contract_changes must be an array")
    matching = [item for item in changes if isinstance(item, dict) and item.get("exception_sha256") == exception_sha]
    if matching:
        if seals.get("benchmark") != active_sha or benchmark_eval.get("sha256") != active_sha:
            raise ValueError("recorded operator change does not match the current Benchmark seal")
        return state
    if seals.get("benchmark") != retired_sha or benchmark_eval.get("sha256") != retired_sha:
        raise ValueError(
            "prior frozen Benchmark seal does not match the authorized retired Benchmark"
        )

    seals["benchmark"] = active_sha
    benchmark_eval["sha256"] = active_sha
    changes.append(
        {
            "schema": "deft_operator_contract_change_audit_v1",
            "applied_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "exception_path": str(exception_path),
            "exception_sha256": exception_sha,
            "authorized_by": exception["authorized_by"],
            "authorization_date": exception["authorization_date"],
            "scope": exception["scope"],
            "effective_iteration": replacement.get("effective_iteration"),
            "previous_benchmark_path": str(retired_path),
            "previous_benchmark_rows": retired_rows,
            "previous_benchmark_sha256": retired_sha,
            "active_benchmark_path": str(active_path),
            "active_benchmark_rows": active_rows,
            "active_benchmark_sha256": active_sha,
            "provenance_path": str(provenance_path),
            "proxy_path": str(proxy_path),
            "proxy_rows": proxy_rows,
            "proxy_sha256": proxy_sha,
            "authorized_overlap_exception": overlap,
        }
    )
    _write_atomic(state_path, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=pathlib.Path)
    parser.add_argument("--exception", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        state = apply(args.state, args.exception)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"apply_operator_contract_change: {exc}", file=sys.stderr)
        return 2
    change = state["operator_contract_changes"][-1]
    print(
        "apply_operator_contract_change: OK "
        f"benchmark_rows={change['active_benchmark_rows']} "
        f"benchmark_sha256={change['active_benchmark_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
