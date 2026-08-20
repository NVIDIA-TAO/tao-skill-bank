#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a fresh sealed Cosmos retry plan from a prior inspected plan.

This helper is deliberately preparation-only: the platform consumer still
opens the retry job-record first and uses its normal submit/status/logs/cancel
verbs.  It reuses immutable model/dataset inspection evidence, refreshes the
job identity and config path, and filters evidence-backed node exclusions
against a live scheduler inventory before rendering.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cosmos_workflow
from cosmos_common import WorkflowError


def _identity(value: str, kind: str) -> dict[str, object]:
    return {
        "original": value,
        "expanded": value,
        "resolved": value,
        "exists": True,
        "kind": kind,
        "nearest_existing_parent": value,
        "parent_writable": kind == "directory",
    }


def _attempt_root(spec_path: Path, *, label: str) -> Path:
    """Resolve the record-owned action root from ``<root>/config/<spec>``."""

    expanded = spec_path.expanduser().resolve()
    if expanded.parent.name != "config":
        raise WorkflowError(
            f"{label} must use the record-owned <action-root>/config/<spec> layout; "
            f"found {expanded}"
        )
    return expanded.parent.parent


def build_retry(args: argparse.Namespace) -> dict[str, object]:
    prior_path = args.prior_plan.expanduser().resolve()
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    artifact = prior.get("plan_artifact", {})
    expected = str(artifact.get("sha256") or "")
    actual = cosmos_workflow._plan_artifact_sha256(prior)
    if artifact.get("schema_version") != cosmos_workflow.PLAN_ARTIFACT_SCHEMA_VERSION:
        raise WorkflowError("prior plan has an unsupported artifact schema")
    if not expected or expected != actual:
        raise WorkflowError(
            f"prior plan checksum mismatch: expected {expected or '<missing>'}, found {actual}"
        )
    if prior.get("action") != "train" or prior.get("backend") not in {
        "cosmos-framework",
        "cosmos-rl",
    }:
        raise WorkflowError("retry preparation requires a sealed Cosmos training plan")

    request = copy.deepcopy(prior.get("planner_request"))
    if not isinstance(request, dict) or not request:
        raise WorkflowError("prior plan has no sealed planner_request")
    write_spec = args.write_spec.expanduser().resolve()
    action_root = _attempt_root(write_spec, label="--write-spec")
    container_spec = Path(args.container_spec_path or write_spec)
    container_action_root = _attempt_root(
        container_spec, label="--container-spec-path"
    )
    inherited_exclusions = (
        []
        if args.replace_node_exclusions
        else list(prior.get("slurm_node_exclusions", {}).get("validated", []))
    )
    requested_exclusions = list(
        dict.fromkeys([*inherited_exclusions, *args.exclude_node])
    )
    request.update(
        {
            "experiment_id": args.job_id,
            "tao_job_id": args.job_id,
            "write_spec": str(write_spec),
            "container_spec_path": str(container_spec.expanduser().resolve()),
            "results_dir": str(action_root / "results"),
            "checkpoint_dir": str(action_root / "checkpoints"),
            "cache_dir": str(action_root / "cache"),
            "stdout_path": str(action_root / "logs" / "%x-%j.out"),
            "stderr_path": str(action_root / "logs" / "%x-%j.err"),
            "container_results_dir": str(container_action_root / "results"),
            "container_checkpoint_dir": str(container_action_root / "checkpoints"),
            "container_cache_dir": str(container_action_root / "cache"),
            "exclude_node": requested_exclusions,
            "exclude_unhealthy_inventory_nodes": True,
            "slurm_node_inventory_file": str(args.slurm_node_inventory.expanduser().resolve()),
        }
    )
    planned_args = argparse.Namespace(**request)
    planned_args.verb = "plan"
    planned_args.format = "json"
    planned_args.plan_artifact = str(args.output.expanduser().resolve())

    verified_host = prior.get("input_frame", {}).get("verified_host")
    if not verified_host:
        raise WorkflowError("prior plan has no verified SLURM inspection host")
    remote_inspection = {
        "frame": "target_compute",
        "verified_host": verified_host,
        "model": prior["model"],
        "datasets": prior["datasets"],
        "runtime_paths": {
            "results_dir": _identity(planned_args.results_dir, "directory"),
            "checkpoint_dir": _identity(planned_args.checkpoint_dir, "directory"),
            "cache_dir": _identity(planned_args.cache_dir, "directory"),
            "sqsh_cache_dir": _identity(planned_args.sqsh_cache_dir, "directory"),
            "sqsh_path": _identity(planned_args.sqsh_path, "file"),
        },
    }
    plan = cosmos_workflow.build_plan(
        planned_args, remote_inspection_override=remote_inspection
    )
    cosmos_workflow.write_spec(planned_args, plan, allow_remote_write=False)
    metadata = cosmos_workflow.initial_metadata(planned_args, plan)
    cosmos_workflow.validate_metadata(metadata)
    plan["initial_metadata"] = metadata
    plan["retry_preparation"] = {
        "retry_of_plan": str(prior_path),
        "retry_of_plan_sha256": expected,
        "inspection_reused": True,
        "attempt_root": str(action_root),
        "container_attempt_root": str(container_action_root),
        "node_exclusions": plan["slurm_node_exclusions"],
        "inherited_node_exclusions": inherited_exclusions,
    }
    cosmos_workflow.save_plan_artifact(planned_args, plan, str(args.output))
    return {
        "schema_version": 1,
        "job_id": args.job_id,
        "backend": plan["backend"],
        "output": str(args.output.expanduser().resolve()),
        "config": plan["config"],
        "node_exclusions": plan["slurm_node_exclusions"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-plan", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--write-spec",
        type=Path,
        required=True,
        help=(
            "Fresh record-owned <action-root>/config/<spec> path. The retry "
            "rebases results, checkpoints, cache, and logs under action-root."
        ),
    )
    parser.add_argument(
        "--container-spec-path",
        default="",
        help=(
            "Container-visible <action-root>/config/<spec> path; defaults to "
            "--write-spec and owns the rebased container output roots."
        ),
    )
    parser.add_argument("--exclude-node", action="append", default=[])
    parser.add_argument(
        "--replace-node-exclusions",
        action="store_true",
        help="Do not inherit the prior plan's validated exclusion set.",
    )
    parser.add_argument("--slurm-node-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        print(json.dumps(build_retry(parse_args()), indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, ValueError, WorkflowError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
