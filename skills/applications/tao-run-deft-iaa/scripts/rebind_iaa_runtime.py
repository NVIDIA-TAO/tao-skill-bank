# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicitly rebind a clean IAA DEFT run to a validated refreshed runtime."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
from typing import Any

import yaml

from audit_deft_run import audit
from iaa_deft.sdg import ROLES, container_name
from runtime_binding import MAX_RUNTIME_REBINDS, active_runtime_sha256, python_tree_sha256, validate_runtime_lineage


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise


def _plugin_versions(script_dir: pathlib.Path) -> tuple[str, str]:
    root = script_dir.parents[3]
    plugin = json.loads((root / ".codex-plugin" / "plugin.json").read_text())
    full = plugin.get("version")
    if not isinstance(full, str) or not full:
        raise ValueError("plugin version is missing")
    base = full.split("+", 1)[0]
    frontmatter = (script_dir.parent / "SKILL.md").read_text().split("---", 2)[1]
    skill = yaml.safe_load(frontmatter).get("metadata", {}).get("version")
    if not isinstance(skill, str) or not skill:
        raise ValueError("IAA skill version is missing")
    return base, skill


def _clean_actions(results_dir: pathlib.Path) -> None:
    for request_path in results_dir.rglob("*.action.json"):
        request = json.loads(request_path.read_text())
        if request.get("kind") == "slurm_sdg_action":
            stage_dir = pathlib.Path(str(request.get("stage_dir", ""))).resolve()
            backend_results = pathlib.Path(str(request.get("results_dir", "")))
            iteration = request.get("iteration")
            claimed_digest = request.get("request_sha256")
            unsigned = dict(request)
            unsigned.pop("request_sha256", None)
            actual_digest = hashlib.sha256(
                json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if (
                not backend_results.is_absolute()
                or backend_results == pathlib.Path(backend_results.anchor)
                or ".." in backend_results.parts
                or backend_results.name != results_dir.name
                or not isinstance(iteration, int)
                or isinstance(iteration, bool)
                or iteration < 1
                or stage_dir != backend_results / f"iter_{iteration}" / "datagen"
            ):
                raise ValueError(
                    f"SLURM SDG preparation request has an invalid backend run mapping: "
                    f"{request_path}"
                )
            if (
                request.get("schema_version") != "1"
                or request.get("workflow") != "tao-run-deft-iaa"
                or request.get("platform") != "slurm"
                or not isinstance(claimed_digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", claimed_digest)
                or claimed_digest != actual_digest
                or "status_path" in request
                or "log_path" in request
            ):
                raise ValueError(f"SLURM SDG preparation request is malformed: {request_path}")
            state_root = pathlib.Path(
                os.environ.get("TAO_STATE_DIR", str(pathlib.Path.home() / ".tao"))
            )
            matching_records = []
            for record_path in (state_root / "jobs").glob("*.json"):
                try:
                    record = json.loads(record_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(record, dict) or record.get("action") != request.get("action_id"):
                    continue
                matching_records.append(record)
                if (
                    record_path.name != f"{record.get('id')}.json"
                    or record.get("platform") != "slurm"
                    or record.get("results_dir") != str(stage_dir)
                    or record.get("terminal_state") not in {"COMPLETE", "ERROR", "CANCELED"}
                ):
                    raise ValueError(
                        f"SLURM SDG request has an active or mismatched job record: {record_path}"
                    )
            for group_path in request_path.parent.glob("slurm-job-group.*.json"):
                group = json.loads(group_path.read_text())
                if group.get("request_sha256") != claimed_digest:
                    continue
                job_id = group.get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    raise ValueError(f"SLURM SDG job group is malformed: {group_path}")
                record_path = state_root / "jobs" / f"{job_id}.json"
                if not record_path.is_file():
                    raise ValueError(f"SLURM SDG job group lacks a terminal job record: {group_path}")
                record = json.loads(record_path.read_text())
                if (
                    record.get("id") != job_id
                    or record.get("action") != request.get("action_id")
                    or record.get("terminal_state") not in {"COMPLETE", "ERROR", "CANCELED"}
                ):
                    raise ValueError(f"SLURM SDG job group is not terminal: {group_path}")
            # This is the immutable output of prepare-request, not a submitted
            # generic action. Launched copies are job-scoped and terminalized
            # through the platform job-record/four-verb contract.
            continue
        status_path = pathlib.Path(str(request.get("status_path", "")))
        if request.get("attempt") == 1:
            attempt_two = request_path.with_name(
                request_path.name.removesuffix(".action.json") + ".attempt-2.action.json"
            )
            if attempt_two.is_file():
                status_path = request_path.with_name(
                    request_path.name.removesuffix(".action.json") + ".attempt-1.status.json"
                )
        if not status_path.is_file():
            raise ValueError(f"unfinalized action request blocks runtime rebind: {request_path}")
        status = json.loads(status_path.read_text())
        if status.get("status") not in {"ok", "error"} or status.get("request_sha256") != request.get("request_sha256"):
            raise ValueError(f"active or mismatched action blocks runtime rebind: {request_path}")


def _clean_endpoints(state: dict[str, Any], inspect_command=subprocess.run) -> None:
    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError("state.config must be an object")
    platform = config.get("platform")
    if platform not in {
        "docker", "virtualenv", "brev", "slurm", "kubernetes", "airflow",
    }:
        raise ValueError("state.config.platform is unsupported for runtime rebind")
    if platform in {"brev", "slurm", "kubernetes"}:
        return
    run_id = pathlib.Path(str(state["results_dir"])).name
    for role in ROLES:
        name = container_name(run_id, role)
        result = inspect_command(["docker", "inspect", name], capture_output=True, text=True)
        if result.returncode != 0:
            continue
        payload = json.loads(result.stdout)
        if not isinstance(payload, list) or len(payload) != 1:
            raise ValueError(f"cannot validate endpoint state for {name}")
        status = payload[0].get("State", {}).get("Status")
        if status in {"running", "created", "restarting", "paused"}:
            raise ValueError(f"active or created endpoint blocks runtime rebind: {name} ({status})")


def _validate_current_tree(results_dir: pathlib.Path, sequence: int) -> pathlib.Path:
    script_dir = pathlib.Path(__file__).resolve().parent
    skill_dir = script_dir.parent
    commands = [
        [os.sys.executable, "-m", "unittest", "discover", "-s", str(skill_dir / "tests"), "-p", "test_*.py"],
        [os.sys.executable, "-m", "py_compile", *[str(path) for path in sorted(script_dir.rglob("*.py"))]],
    ]
    records = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"refreshed runtime validation failed: {' '.join(command[:4])}")
        records.append({"command": command, "exit_code": result.returncode})
    evidence = results_dir / "runtime_rebind" / f"validation-{sequence}.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(evidence, {
        "schema_version": "1", "workflow": "tao-run-deft-iaa",
        "kind": "runtime_rebind_validation", "result": "PASS", "commands": records,
        "runtime_sha256": python_tree_sha256(script_dir / "iaa_deft"),
    })
    return evidence


def rebind(results_dir: pathlib.Path, reason: str, *, inspect_command=subprocess.run) -> dict[str, Any]:
    results_dir = results_dir.resolve()
    state_path = results_dir / "deft_state.json"
    if not reason.strip():
        raise ValueError("--reason must be non-empty")
    state = json.loads(state_path.read_text())
    if state.get("workflow") != "tao-run-deft-iaa" or state.get("schema_version") != "3":
        raise ValueError("runtime rebind supports only IAA DEFT schema v3")
    report = audit(results_dir)
    if report.get("errors") != ["bundled IAA runtime changed after initialization"]:
        raise ValueError("run must be otherwise valid at a committed stage boundary")
    if (results_dir / ".deft_commit_transaction.json").exists():
        raise ValueError("unfinished stage transaction blocks runtime rebind")
    _clean_actions(results_dir)
    _clean_endpoints(state, inspect_command)
    lineage = validate_runtime_lineage(state, results_dir)
    if len(lineage) >= MAX_RUNTIME_REBINDS:
        raise ValueError("runtime rebind budget exhausted")
    old = active_runtime_sha256(state)
    new = python_tree_sha256(pathlib.Path(__file__).resolve().parent / "iaa_deft")
    if new == old:
        raise ValueError("runtime rebind is a no-op")
    base, skill = _plugin_versions(pathlib.Path(__file__).resolve().parent)
    if lineage and (lineage[-1]["plugin_base_version"] != base or lineage[-1]["skill_version"] != skill):
        raise ValueError("refreshed runtime changes the compatible plugin or skill base")
    evidence = _validate_current_tree(results_dir, len(lineage) + 1)
    record = {
        "schema_version": "1", "sequence": len(lineage) + 1,
        "old_sha256": old, "new_sha256": new,
        "rebound_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": reason.strip(), "evidence_path": str(evidence),
        "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "plugin_base_version": base, "skill_version": skill,
    }
    state["runtime_lineage"] = [*lineage, record]
    state["active_runtime_sha256"] = new
    _atomic_json(state_path, state)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    try:
        record = rebind(args.results_dir, args.reason)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"rebind_iaa_runtime: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
