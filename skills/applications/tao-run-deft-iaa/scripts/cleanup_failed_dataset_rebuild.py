#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove one proven run-owned SLURM rebuild staging tree after quota failure."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
from typing import Any, Sequence


SAFE_LOGIN = re.compile(r"^[A-Za-z0-9_.@-]+$")
SAFE_REMOTE = re.compile(r"^/[A-Za-z0-9_./-]+$")


def _json(path: pathlib.Path, label: str) -> dict[str, Any]:
    path = pathlib.Path(os.path.abspath(path.expanduser()))
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise ValueError(f"{label} is missing or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _remote(value: pathlib.Path, label: str) -> pathlib.Path:
    if (
        not value.is_absolute()
        or value == pathlib.Path(value.anchor)
        or ".." in value.parts
        or SAFE_REMOTE.fullmatch(str(value)) is None
    ):
        raise ValueError(f"{label} must be one safe non-root absolute path")
    return value


def _mapping(
    local: pathlib.Path, local_workspace: pathlib.Path, remote_workspace: pathlib.Path
) -> pathlib.Path:
    local = pathlib.Path(os.path.abspath(local.expanduser()))
    try:
        relative = local.relative_to(local_workspace)
    except ValueError as exc:
        raise ValueError("cleanup target escapes the approved local workspace") from exc
    return _remote(remote_workspace / relative, "mapped cleanup target")


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
    finally:
        try:
            os.unlink(raw)
        except FileNotFoundError:
            pass


def cleanup(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm:
        raise ValueError("cleanup requires --confirm")
    if SAFE_LOGIN.fullmatch(args.login) is None:
        raise ValueError("--login contains unsupported characters")
    results = pathlib.Path(os.path.abspath(args.results_dir.expanduser()))
    state = _json(results / "deft_state.json", "DEFT state")
    status = _json(
        results / "dataset_setup/dataset_rebuild.status.json",
        "dataset rebuild failure status",
    )
    if (
        state.get("workflow") != "tao-run-deft-iaa"
        or state.get("config", {}).get("platform") != "slurm"
        or pathlib.Path(str(state.get("results_dir", ""))) != results
    ):
        raise ValueError("results do not identify one canonical SLURM IAA run")
    if (
        status.get("workflow") != "tao-run-deft-iaa"
        or status.get("name") != "dataset_rebuild"
        or status.get("status") != "error"
        or status.get("backend_state") != "ERROR"
        or status.get("backend_exit_code") != 1
    ):
        raise ValueError("dataset rebuild lacks one finalized workload failure")
    diagnostic = results / "dataset_setup/rebuild_verify.log"
    if (
        diagnostic.is_symlink()
        or not diagnostic.is_file()
        or diagnostic.resolve() != diagnostic
        or "Disk quota exceeded" not in diagnostic.read_text(
            encoding="utf-8", errors="replace"
        )
    ):
        raise ValueError("dataset rebuild failure is not a proven quota failure")
    config = state["config"]
    workspace = pathlib.Path(str(config["workspace"]))
    dataset = pathlib.Path(str(config["dataset_root"]))
    digest = state.get("active_runtime_sha256") or config.get("iaa_deft_bundle_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("DEFT runtime digest is invalid")
    staging = dataset.parent / f".{dataset.name}.rebuild-{digest[:12]}"
    remote_workspace = _remote(args.remote_workspace, "--remote-workspace")
    remote_staging = _mapping(staging, workspace, remote_workspace)
    remote_dataset = _mapping(dataset, workspace, remote_workspace)
    remote_results = _mapping(results, workspace, remote_workspace)
    receipt = results / "dataset_setup/dataset_rebuild.quota-cleanup.json"
    if receipt.is_file() and not receipt.is_symlink():
        existing = _json(receipt, "quota cleanup receipt")
        if existing.get("remote_staging") == str(remote_staging):
            return {"status": "reused", "receipt": str(receipt), **existing}
        raise ValueError("quota cleanup receipt conflicts with the requested target")
    command = f"""set -Eeuo pipefail
staging={shlex.quote(str(remote_staging))}
dataset={shlex.quote(str(remote_dataset))}
evidence={shlex.quote(str(remote_results / 'dataset_setup/rebuild_verify.log'))}
test ! -e "$dataset"
test -d "$staging"
test ! -L "$staging"
test -f "$evidence"
grep -Fq 'Disk quota exceeded' "$evidence"
chmod u+rwx -- "$staging"
for child in images captions images_raw; do
  if test -d "$staging/$child"; then chmod u+rwx -- "$staging/$child"; fi
done
rm -rf -- "$staging"
test ! -e "$staging"
"""
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.login, command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        timeout=7200,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no diagnostic output"
        raise RuntimeError(f"remote quota cleanup failed: {detail[-2000:]}")
    payload = {
        "schema_version": "1",
        "workflow": "tao-run-deft-iaa",
        "kind": "failed_dataset_rebuild_quota_cleanup",
        "results_dir": str(results),
        "failure_status_sha256": hashlib.sha256(
            (results / "dataset_setup/dataset_rebuild.status.json").read_bytes()
        ).hexdigest(),
        "remote_staging": str(remote_staging),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "recoverable": False,
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(receipt, payload)
    return {"status": "removed", "receipt": str(receipt), **payload}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--login", required=True)
    parser.add_argument("--remote-workspace", required=True, type=pathlib.Path)
    parser.add_argument("--confirm", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(json.dumps(cleanup(_parser().parse_args(argv)), sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"cleanup_failed_dataset_rebuild: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
