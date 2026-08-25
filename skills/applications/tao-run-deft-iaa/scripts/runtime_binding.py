# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated active-runtime lineage for resumable IAA DEFT runs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
from typing import Any

MAX_RUNTIME_REBINDS = 3


def python_tree_sha256(root: pathlib.Path) -> str:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise ValueError(f"bundled IAA runtime contains no Python files: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def active_runtime_sha256(state: dict[str, Any]) -> str:
    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError("state.config must be an object")
    initial = config.get("iaa_deft_bundle_sha256")
    active = state.get("active_runtime_sha256", initial)
    if not isinstance(active, str) or len(active) != 64:
        raise ValueError("active IAA runtime digest is invalid")
    return active


def validate_runtime_lineage(state: dict[str, Any], results_dir: pathlib.Path) -> list[dict[str, Any]]:
    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError("state.config must be an object")
    initial = config.get("iaa_deft_bundle_sha256")
    if not isinstance(initial, str) or len(initial) != 64:
        raise ValueError("initial IAA runtime digest is invalid")
    lineage = state.get("runtime_lineage", [])
    if not isinstance(lineage, list) or len(lineage) > MAX_RUNTIME_REBINDS:
        raise ValueError(f"runtime_lineage must contain at most {MAX_RUNTIME_REBINDS} records")
    expected_old = initial
    seen = {initial}
    expected_base = None
    expected_skill = None
    for sequence, record in enumerate(lineage, 1):
        if not isinstance(record, dict) or set(record) != {
            "schema_version", "sequence", "old_sha256", "new_sha256", "rebound_at",
            "reason", "evidence_path", "evidence_sha256", "plugin_base_version", "skill_version",
        }:
            raise ValueError(f"runtime_lineage[{sequence}] has an invalid shape")
        if record["schema_version"] != "1" or record["sequence"] != sequence:
            raise ValueError(f"runtime_lineage[{sequence}] sequence/schema is invalid")
        if record["old_sha256"] != expected_old:
            raise ValueError(f"runtime_lineage[{sequence}] does not continue the digest chain")
        new = record["new_sha256"]
        if not isinstance(new, str) or len(new) != 64 or new in seen:
            raise ValueError(f"runtime_lineage[{sequence}] is a no-op or digest downgrade")
        reason = record["reason"]
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
            raise ValueError(f"runtime_lineage[{sequence}] reason is invalid")
        try:
            dt.datetime.fromisoformat(record["rebound_at"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"runtime_lineage[{sequence}] timestamp is invalid") from exc
        evidence = pathlib.Path(str(record["evidence_path"]))
        try:
            evidence.relative_to(results_dir)
        except ValueError as exc:
            raise ValueError(f"runtime_lineage[{sequence}] evidence is outside results_dir") from exc
        if not evidence.is_file() or evidence.is_symlink():
            raise ValueError(f"runtime_lineage[{sequence}] evidence is missing or unsafe")
        digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
        if digest != record["evidence_sha256"]:
            raise ValueError(f"runtime_lineage[{sequence}] evidence digest mismatch")
        payload = json.loads(evidence.read_text())
        if not isinstance(payload, dict) or payload.get("result") != "PASS":
            raise ValueError(f"runtime_lineage[{sequence}] evidence does not record PASS")
        base = record["plugin_base_version"]
        skill = record["skill_version"]
        if not isinstance(base, str) or not base or not isinstance(skill, str) or not skill:
            raise ValueError(f"runtime_lineage[{sequence}] compatibility versions are invalid")
        if expected_base is not None and (base != expected_base or skill != expected_skill):
            raise ValueError(f"runtime_lineage[{sequence}] changes the compatible skill base")
        expected_base, expected_skill = base, skill
        expected_old = new
        seen.add(new)
    active = state.get("active_runtime_sha256", initial)
    if active != expected_old:
        raise ValueError("active_runtime_sha256 does not match runtime_lineage")
    return lineage
