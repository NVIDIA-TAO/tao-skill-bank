#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and finalize one platform-dispatched IAA DEFT TAO action.

``prepare`` validates the exact immutable action, writes a schema-valid
platform-neutral spec bundle plus a concrete staging/mount contract, and
invalidates stale outputs. The selected TAO platform skill then owns the
native submit/status/logs/cancel lifecycle and job-record. ``finalize`` binds
that terminal job-record and native exit status to fresh workflow outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.parse
from copy import deepcopy
from typing import Any

import jsonschema

from command_contract import command_sha256
from deft_action_contract import (
    ADAPTER_ACTIONS,
    WORKFLOW,
    ActionContext,
    atomic_json,
    load_existing_status,
    safe_absolute_path,
    validate_action,
)
from runtime_binding import active_runtime_sha256, validate_runtime_lineage


REQUEST_SCHEMA_VERSION = "1"
STATUS_SCHEMA_VERSION = "2"
REMOTE_PLATFORMS = frozenset({"slurm", "kubernetes", "brev", "airflow"})
JOB_IDENTITY_FIELDS = (
    "schema_version",
    "id",
    "platform",
    "image",
    "network_arch",
    "action",
    "results_dir",
    "storage_tier",
    "upload_excludes",
    "submitted_at",
)

# Only actions that dereference image paths need the dataset tree.  Text-only
# embedding and k-NN actions operate entirely on run-owned parquet artifacts.
# Keeping this decision in the producer makes every remote consumer stage the
# same minimal, truthful input set instead of relying on platform-specific
# exclusions.
DATASET_ACTIONS = frozenset(
    {
        "evaluate",
        "train",
        "viz_weak_embed",
        "viz_mined_embed",
        "viz_previous_embed",
        "dataset_rebuild",
        "dataset_materialize",
        "gap_analysis",
        "visualize_prepare",
        # The terminal renderer runs the strict audit, which validates the
        # committed dataset tree in addition to run-owned results.
        "report",
        "sdg_normalize_repair",
    }
)
UNBOUND_REPLAY_ACTIONS = frozenset(
    {
        "gap_analysis",
        "mining_postprocess",
        "history_select",
        "eval_config",
        "train_config",
        "iteration_summary",
        "metric_parse",
    }
)

TAO_MODEL_CACHE_RELATIVE_ROOT = pathlib.PurePosixPath(
    "huggingface/hub/models--google--siglip2-so400m-patch16-256"
)


def _snapshot_manifest(root: pathlib.Path) -> dict[str, Any]:
    """Inventory one immutable request-owned controller input tree."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"action snapshot is missing or unsafe: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"action snapshot contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"action snapshot contains a non-regular file: {path}")
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    if not entries:
        raise ValueError(f"action snapshot contains no files: {root}")
    return {
        "root": str(root),
        "entries": entries,
        "sha256": _sha256_json({"entries": entries}),
    }


def _materialize_snapshot(
    source: pathlib.Path, destination: pathlib.Path, *, python_only: bool
) -> dict[str, Any]:
    """Atomically create or validate one durable action-owned snapshot."""
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = pathlib.Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.", dir=destination.parent
        ))
        try:
            for path in sorted(source.rglob("*")):
                if "__pycache__" in path.parts or (python_only and path.suffix != ".py"):
                    continue
                relative = path.relative_to(source)
                if path.is_symlink():
                    raise ValueError(f"snapshot source contains a symlink: {path}")
                if path.is_dir():
                    continue
                if not path.is_file():
                    raise ValueError(f"snapshot source contains a non-regular file: {path}")
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                os.chmod(target, 0o500 if os.access(path, os.X_OK) else 0o400)
            _snapshot_manifest(temporary)
            for directory in sorted(
                (path for path in temporary.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                os.chmod(directory, 0o500)
            os.chmod(temporary, 0o500)
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return _snapshot_manifest(destination)


def _materialize_controller_snapshot(destination: pathlib.Path) -> dict[str, Any]:
    """Snapshot the minimal skill-bank topology needed by action controllers."""
    bank_root = pathlib.Path(__file__).resolve().parents[4]
    application_root = pathlib.Path(__file__).resolve().parent.parent
    selected = [
        *sorted((application_root / "scripts").rglob("*.py")),
        *sorted((application_root / "references").glob("*.json")),
        *sorted(
            path for path in (application_root / "patches").rglob("*")
            if path.is_file()
        ),
        *sorted(
            (
                bank_root
                / "skills"
                / "core"
                / "tao-artifacts"
                / "references"
            ).glob("*.json")
        ),
    ]
    if not selected:
        raise ValueError("controller snapshot source selection is empty")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = pathlib.Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.", dir=destination.parent
        ))
        try:
            for source in selected:
                if source.is_symlink() or not source.is_file():
                    raise ValueError(f"controller snapshot source is unsafe: {source}")
                relative = source.relative_to(bank_root)
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                os.chmod(target, 0o400)
            _snapshot_manifest(temporary)
            for directory in sorted(
                (path for path in temporary.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                os.chmod(directory, 0o500)
            os.chmod(temporary, 0o500)
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return _snapshot_manifest(destination)


def _controller_scripts(snapshot: dict[str, Any]) -> pathlib.Path:
    return (
        pathlib.Path(snapshot["root"])
        / "skills"
        / "applications"
        / "tao-run-deft-iaa"
        / "scripts"
    )


def _tao_cache_subset(cache_dir: pathlib.Path) -> dict[str, Any]:
    """Bind only the TAO SigLIP cache required by dispatched TAO actions."""
    root = cache_dir / pathlib.Path(TAO_MODEL_CACHE_RELATIVE_ROOT)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"required TAO model cache is missing or unsafe: {root}")
    entries: list[dict[str, Any]] = []
    cache_resolved = cache_dir.resolve(strict=True)
    model_resolved = root.resolve(strict=True)
    snapshot_entries = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        files.sort()
        base = pathlib.Path(directory)
        if any((base / name).is_symlink() for name in dirs):
            raise ValueError("TAO model cache must not contain symlink directories")
        for name in files:
            path = base / name
            model_relative = path.relative_to(root)
            if model_relative.parts[0] == "blobs":
                if path.is_symlink():
                    raise ValueError(f"TAO cache blob entry must not be a symlink: {path}")
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise ValueError(f"TAO cache blob entry is not regular: {path}")
                continue
            if not path.is_symlink() and not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"TAO cache entry is not a regular file: {path}")
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(cache_resolved)
            except ValueError as exc:
                raise ValueError(f"TAO cache entry escapes cache root: {path}") from exc
            if not resolved.is_file():
                raise ValueError(f"TAO cache entry is not a regular file: {path}")
            if path.is_symlink():
                try:
                    target_relative = resolved.relative_to(model_resolved)
                except ValueError as exc:
                    raise ValueError(f"TAO cache snapshot escapes selected model: {path}") from exc
                if not target_relative.parts or target_relative.parts[0] != "blobs":
                    raise ValueError(f"TAO cache snapshot does not resolve to model blobs: {path}")
                snapshot_entries += 1
            relative = path.relative_to(cache_dir).as_posix()
            entries.append(
                {"path": relative, "size": resolved.stat().st_size, "sha256": _sha256_file(resolved)}
            )
    if not entries:
        raise ValueError(f"required TAO model cache is empty: {root}")
    if snapshot_entries == 0:
        raise ValueError(f"required TAO model cache has no validated snapshots: {root}")
    manifest = {"root": str(cache_dir), "entries": entries}
    manifest["sha256"] = _sha256_json(manifest)
    return manifest


def tao_cache_preflight(cache_dir: pathlib.Path) -> dict[str, Any]:
    """Validate the exact public SigLIP cache before any workflow mutation."""
    resolved = safe_absolute_path(
        cache_dir.expanduser(), "TAO model cache root", require_exists=True
    )
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"TAO model cache root is missing or unsafe: {resolved}")
    manifest = _tao_cache_subset(resolved)
    return {
        "status": "ok",
        "cache_dir": str(resolved),
        "model": "google/siglip2-so400m-patch16-256",
        "entries": len(manifest["entries"]),
        "bytes": sum(entry["size"] for entry in manifest["entries"]),
        "manifest_sha256": manifest["sha256"],
    }


def _artifact_schema(name: str) -> dict[str, Any]:
    path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "core"
        / "tao-artifacts"
        / "references"
        / name
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_schema(name: str) -> dict[str, Any]:
    path = pathlib.Path(__file__).resolve().parent.parent / "references" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_tree_sha256(root: pathlib.Path) -> str:
    files = sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )
    if not files:
        raise ValueError(f"bundled IAA runtime contains no Python files: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _aware_timestamp(value: Any, name: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty timezone-aware timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset")
    return parsed.astimezone(dt.timezone.utc)


def _relative_output(path: pathlib.Path, root: pathlib.Path) -> str:
    return path.relative_to(root).as_posix()


def _attempt_path(
    stage_dir: pathlib.Path,
    name: str,
    attempt: int,
    suffix: str,
    *,
    dispatch_repair: int = 0,
    launcher_repair: int = 0,
    train_output_replay: int = 0,
) -> pathlib.Path:
    if attempt not in {1, 2}:
        raise ValueError(f"action attempt must be 1 or 2, got {attempt!r}")
    if dispatch_repair not in {0, 1}:
        raise ValueError("dispatch repair must be 0 or 1")
    if launcher_repair not in {0, 1}:
        raise ValueError("launcher repair must be 0 or 1")
    if train_output_replay not in {0, 1}:
        raise ValueError("train output replay must be 0 or 1")
    if sum((dispatch_repair, launcher_repair, train_output_replay)) > 1:
        raise ValueError("action recovery modes are mutually exclusive")
    if dispatch_repair:
        if attempt != 2:
            raise ValueError("dispatch repair is valid only after attempt 2")
        return stage_dir / f"{name}.dispatch-repair.{suffix}"
    if launcher_repair:
        if attempt != 2:
            raise ValueError("launcher repair is valid only after attempt 2")
        return stage_dir / f"{name}.launcher-repair.{suffix}"
    if train_output_replay:
        if attempt != 2 or name != "train":
            raise ValueError("train output replay is valid only for train attempt 2")
        return stage_dir / f"{name}.output-replay.{suffix}"
    infix = "" if attempt == 1 else ".attempt-2"
    return stage_dir / f"{name}{infix}.{suffix}"


def _request_path_for(context: ActionContext, attempt: int) -> pathlib.Path:
    return _attempt_path(context.stage_dir, context.name, attempt, "action.json")


def _request_path_from_payload(request: dict[str, Any]) -> pathlib.Path:
    return _attempt_path(
        pathlib.Path(request["stage_dir"]),
        request["name"],
        request["attempt"],
        "action.json",
        dispatch_repair=request.get("dispatch_repair", 0),
        launcher_repair=request.get("launcher_repair", 0),
        train_output_replay=request.get("train_output_replay", 0),
    )


def _runtime_dir_for(
    context: ActionContext, attempt: int, *, dispatch_repair: int = 0,
    launcher_repair: int = 0, train_output_replay: int = 0,
) -> pathlib.Path:
    name = (
        f"{context.name}.dispatch-repair"
        if dispatch_repair
        else (
            f"{context.name}.launcher-repair"
            if launcher_repair
            else (
                f"{context.name}.output-replay"
                if train_output_replay
                else f"{context.name}.attempt-{attempt}"
            )
        )
    )
    return (
        context.stage_dir
        / ".tao-runtime"
        / name
    )


def _job_state_dir() -> pathlib.Path:
    raw = os.environ.get("TAO_STATE_DIR")
    if raw is not None and not raw.strip():
        raise ValueError("TAO_STATE_DIR is set but empty")
    path = pathlib.Path(raw) if raw is not None else pathlib.Path.home() / ".tao"
    return safe_absolute_path(path, "TAO job state directory")


@contextlib.contextmanager
def _exclusive_lock(path: pathlib.Path):
    path = safe_absolute_path(path, "action launch lock")
    try:
        fd = os.open(
            path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise ValueError(f"cannot open safe action launch lock {path}: {exc}") from exc
    with os.fdopen(fd, "a+", encoding="utf-8") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError(f"action launch lock is not a regular file: {path}")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another process owns this action launch: {path}") from exc
        yield


def _request_lock(request: dict[str, Any]):
    return _exclusive_lock(
        pathlib.Path(request["stage_dir"]) / f"{request['name']}.launch.lock"
    )


def _action_id(
    context: ActionContext,
    attempt: int,
    started_ns: int,
    *,
    dispatch_repair: int = 0,
    launcher_repair: int = 0,
    unbound_replay: int = 0,
    unbound_replay_evidence_sha256: str | None = None,
    train_output_replay: int = 0,
    train_output_replay_evidence_sha256: str | None = None,
) -> str:
    """Bind the job-record action to one prepared run/stage attempt."""
    identity_fields = [
        str(context.results_dir),
        context.stage_dir.relative_to(context.results_dir).as_posix(),
        context.name,
        str(attempt),
    ]
    if dispatch_repair:
        identity_fields.append("dispatch-repair-1")
    if launcher_repair:
        identity_fields.append("launcher-repair-1")
    if unbound_replay:
        if (
            not isinstance(unbound_replay_evidence_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", unbound_replay_evidence_sha256) is None
        ):
            raise ValueError("unbound replay requires its evidence digest")
        identity_fields.extend(
            ("unbound-replay-1", unbound_replay_evidence_sha256)
        )
    if train_output_replay:
        if (
            context.name != "train"
            or attempt != 2
            or not isinstance(train_output_replay_evidence_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", train_output_replay_evidence_sha256) is None
        ):
            raise ValueError("train output replay requires exact train evidence")
        identity_fields.extend(
            ("train-output-replay-1", train_output_replay_evidence_sha256)
        )
    identity_fields.append(str(started_ns))
    identity = "\0".join(identity_fields)
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"deft-iaa-{context.name}-{suffix}"


def _regressed_normal_action_id(
    context: ActionContext, attempt: int, started_ns: int
) -> str:
    """Return the exact short-lived action id emitted for normal actions.

    One development cache included the inactive dispatch-repair marker in
    ordinary action identities.  Finalized evidence from that cache remains
    immutable, but a bounded retry may recognize only this exact derivation.
    """
    identity = "\0".join(
        (
            str(context.results_dir),
            context.stage_dir.relative_to(context.results_dir).as_posix(),
            context.name,
            str(attempt),
            "dispatch-repair-0",
            str(started_ns),
        )
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"deft-iaa-{context.name}-{suffix}"


def _bundle(
    context: ActionContext,
    execution_image: str,
    attempt: int,
    started_ns: int,
    controller_snapshot: dict[str, Any],
    patches_snapshot: dict[str, Any],
    *,
    dispatch_repair: int = 0,
    launcher_repair: int = 0,
    unbound_replay: int = 0,
    unbound_replay_evidence_sha256: str | None = None,
    train_output_replay: int = 0,
    train_output_replay_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    adapter = context.name in ADAPTER_ACTIONS
    network = "iaa-adapter" if adapter else ("clip" if context.image_kind == "pyt" else "data-services")
    declared_inputs = [
        {
            "spec_key": "workflow_results",
            "type": "folder",
            "uri": str(context.results_dir),
        },
        {
            "spec_key": "run_config",
            "type": "folder",
            "uri": str(context.config_dir),
        },
        {
            "spec_key": "compatibility_patches",
            "type": "folder",
            "uri": patches_snapshot["root"],
        },
        {
            "spec_key": "model_cache",
            "type": "folder",
            "uri": str(context.cache_dir),
        },
    ]
    if adapter:
        declared_inputs.insert(1, {
            "spec_key": "iaa_runtime",
            "type": "folder",
            "uri": controller_snapshot["root"],
        })
    if context.name in DATASET_ACTIONS:
        dataset_exists = (
            context.dataset_root.is_dir() and not context.dataset_root.is_symlink()
        )
        declared_inputs.insert(
            1,
            {
                "spec_key": "dataset_root" if dataset_exists else "dataset_parent",
                "type": "folder",
                "uri": str(context.dataset_root if dataset_exists else context.dataset_root.parent),
            },
        )
    if context.name in {"dataset_rebuild", "report"}:
        archive_parents = {
            pathlib.Path(context.config[key]).parent
            for key in ("images_archive", "metadata_archive")
        }
        for index, parent in enumerate(sorted(archive_parents), start=1):
            declared_inputs.insert(
                1,
                {
                    "spec_key": f"archive_parent_{index}",
                    "type": "folder",
                    "uri": str(parent),
                },
            )

    action_identity_options = {
        "dispatch_repair": dispatch_repair,
        "launcher_repair": launcher_repair,
    }
    if unbound_replay:
        action_identity_options.update(
            {
                "unbound_replay": unbound_replay,
                "unbound_replay_evidence_sha256": (
                    unbound_replay_evidence_sha256
                ),
            }
        )
    if train_output_replay:
        action_identity_options.update(
            {
                "train_output_replay": train_output_replay,
                "train_output_replay_evidence_sha256": (
                    train_output_replay_evidence_sha256
                ),
            }
        )
    bundle = {
        "network_arch": network,
        "action": _action_id(
            context,
            attempt,
            started_ns,
            **action_identity_options,
        ),
        "image": execution_image,
        "mode": "args",
        "command": context.command[0],
        "args": context.command[1:],
        "declared_inputs": declared_inputs,
        "declared_outputs": [
            {
                # The job-record binds results_dir to this action's stage, so
                # output keys are relative to that exact job root rather than
                # redundantly repeating the workflow-level path.
                "spec_key": _relative_output(output, context.stage_dir),
                "type": "file",
            }
            for output in context.fresh_outputs
        ],
        "upload_excludes": [".tao-runtime/", "*.launch.lock"],
        "compute_shape": {
            "gpus": 0 if adapter else int(context.config["num_gpus"]),
            "nodes": 1,
        },
    }
    jsonschema.validate(bundle, _artifact_schema("spec_bundle.schema.json"))
    return bundle


def _mounts(
    context: ActionContext,
    controller_snapshot: dict[str, Any],
    patches_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return aliases required by existing specs and embedded parquet paths."""
    data_parent = context.dataset_root.parent
    mounts = [
        {"source": str(context.results_dir), "target": "/results", "read_only": False},
        {
            "source": str(context.results_dir),
            "target": str(context.results_dir),
            "read_only": False,
        },
        {"source": str(context.config_dir), "target": "/specs", "read_only": True},
        {"source": patches_snapshot["root"], "target": "/patches", "read_only": True},
        {"source": str(context.cache_dir), "target": "/cache", "read_only": False},
    ]
    if context.name in ADAPTER_ACTIONS:
        mounts.append({
            "source": str(_controller_scripts(controller_snapshot)),
            "target": "/iaa-runtime",
            "read_only": True,
        })
    if context.name in DATASET_ACTIONS:
        if context.dataset_root.is_dir() and not context.dataset_root.is_symlink():
            mounts[2:2] = [
                {
                    "source": str(context.dataset_root),
                    "target": f"/data/{context.dataset_root.name}",
                    "read_only": True,
                },
                {
                    "source": str(context.dataset_root),
                    "target": str(context.dataset_root),
                    "read_only": True,
                },
            ]
        else:
            mounts[2:2] = [
                {"source": str(data_parent), "target": "/data", "read_only": True},
                {
                    "source": str(data_parent),
                    "target": str(data_parent),
                    "read_only": True,
                },
            ]
    if context.name in {"dataset_rebuild", "report"}:
        existing_targets = {item["target"] for item in mounts}
        archive_parents = {
            pathlib.Path(context.config[key]).parent
            for key in ("images_archive", "metadata_archive")
        }
        mounts.extend(
            {
                "source": str(parent),
                "target": str(parent),
                "read_only": True,
            }
            for parent in sorted(archive_parents)
            if str(parent) not in existing_targets
        )
    if context.name == "dataset_rebuild":
        for item in mounts:
            if item["source"] == str(data_parent):
                item["read_only"] = False
    return mounts


def _request(
    context: ActionContext,
    attempt: int,
    started_ns: int,
    *,
    started_at: str | None = None,
    job_state_dir: pathlib.Path | None = None,
    dispatch_repair: int = 0,
    launcher_repair: int = 0,
    unbound_replay: int = 0,
    unbound_replay_evidence_sha256: str | None = None,
    train_output_replay: int = 0,
    train_output_replay_evidence_sha256: str | None = None,
    materialize_snapshots: bool = False,
) -> dict[str, Any]:
    if unbound_replay not in {0, 1}:
        raise ValueError("unbound replay must be 0 or 1")
    if unbound_replay:
        if attempt != 2 or dispatch_repair or launcher_repair:
            raise ValueError(
                "unbound replay is an exclusive attempt-2 recovery"
            )
        if (
            not isinstance(unbound_replay_evidence_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", unbound_replay_evidence_sha256) is None
        ):
            raise ValueError("unbound replay requires a valid evidence digest")
    elif unbound_replay_evidence_sha256 is not None:
        raise ValueError("normal action cannot declare unbound replay evidence")
    if train_output_replay not in {0, 1}:
        raise ValueError("train output replay must be 0 or 1")
    if train_output_replay:
        if (
            context.platform != "slurm"
            or context.name != "train"
            or attempt != 2
            or dispatch_repair
            or launcher_repair
            or unbound_replay
        ):
            raise ValueError("train output replay is an exclusive SLURM train recovery")
        if (
            not isinstance(train_output_replay_evidence_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", train_output_replay_evidence_sha256) is None
        ):
            raise ValueError("train output replay requires a valid evidence digest")
    elif train_output_replay_evidence_sha256 is not None:
        raise ValueError("normal action cannot declare train output replay evidence")
    if context.platform == "virtualenv":
        if context.virtualenv is None:
            raise ValueError("virtualenv platform has no validated action profile")
        virtualenv = str(context.virtualenv)
        record_image = str(context.virtualenv / "bin" / "python")
        entrypoint_sha256 = _sha256_file(
            context.virtualenv / "bin" / context.command[0]
        )
    else:
        virtualenv = None
        record_image = context.image
        entrypoint_sha256 = None
    platform_runtime_dir = _runtime_dir_for(
        context,
        attempt,
        dispatch_repair=dispatch_repair,
        launcher_repair=launcher_repair,
        train_output_replay=train_output_replay,
    )
    controller_root = platform_runtime_dir / "input-snapshots" / "skill-bank"
    patches_root = platform_runtime_dir / "input-snapshots" / "patches"
    state = json.loads((context.results_dir / "deft_state.json").read_text())
    validate_runtime_lineage(state, context.results_dir)
    if materialize_snapshots:
        current_runtime = pathlib.Path(__file__).resolve().parent / "iaa_deft"
        if _python_tree_sha256(current_runtime) != active_runtime_sha256(state):
            raise ValueError(
                "cannot prepare action: bundled IAA runtime does not match immutable run provenance"
            )
    if materialize_snapshots:
        controller_snapshot = _materialize_controller_snapshot(controller_root)
        patches_snapshot = _materialize_snapshot(
            context.patches_dir, patches_root, python_only=False
        )
    else:
        controller_snapshot = _snapshot_manifest(controller_root)
        patches_snapshot = _snapshot_manifest(patches_root)
    bundle = _bundle(
        context,
        record_image,
        attempt,
        started_ns,
        controller_snapshot,
        patches_snapshot,
        dispatch_repair=dispatch_repair,
        launcher_repair=launcher_repair,
        unbound_replay=unbound_replay,
        unbound_replay_evidence_sha256=unbound_replay_evidence_sha256,
        train_output_replay=train_output_replay,
        train_output_replay_evidence_sha256=train_output_replay_evidence_sha256,
    )
    log_path = _attempt_path(
        context.stage_dir,
        context.name,
        attempt,
        "log",
        dispatch_repair=dispatch_repair,
        launcher_repair=launcher_repair,
        train_output_replay=train_output_replay,
    )
    payload = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "runtime_sha256": active_runtime_sha256(state),
        "controller_snapshot": controller_snapshot,
        "patches_snapshot": patches_snapshot,
        "platform": context.platform,
        "name": context.name,
        "attempt": attempt,
        "label": context.label,
        "image_kind": context.image_kind,
        "record_image": record_image,
        "workload_image": context.image,
        "gpu_ids": [] if context.name in ADAPTER_ACTIONS else list(context.config["gpu_ids"]),
        "passed_hf_token": context.pass_hf_token,
        "forward_env": ["HF_TOKEN"] if context.pass_hf_token else [],
        "spec_bundle": bundle,
        "mounts": _mounts(context, controller_snapshot, patches_snapshot),
        "environment": {
            "HOME": "/tmp",
            "PYTHONPATH": "/patches",
            "HF_HOME": "/cache/huggingface",
            "XDG_CACHE_HOME": "/cache",
        },
        "virtualenv": virtualenv,
        "virtualenv_entrypoint_sha256": entrypoint_sha256,
        "virtualenv_shim": str(_controller_scripts(controller_snapshot) / "run_deft_cli.py"),
        "results_dir": str(context.results_dir),
        "stage_dir": str(context.stage_dir),
        "platform_runtime_dir": str(platform_runtime_dir),
        "status_path": str(context.status_path),
        "log_path": str(log_path),
        "fresh_outputs": [str(path) for path in context.fresh_outputs],
        "staging_absent_paths": [
            *[str(path) for path in context.fresh_outputs],
            str(log_path),
        ],
        "freshness_contract": (
            "local-mtime-after-prepare"
            if context.platform == "virtualenv"
            or (
                context.platform == "docker"
                and context.config.get("docker_remote") is not True
            )
            else "remote-mirror-with-delete-before-submit"
        ),
        "staging_receipt_path": str(
            _attempt_path(
                context.stage_dir,
                context.name,
                attempt,
                "staged.json",
                dispatch_repair=dispatch_repair,
                launcher_repair=launcher_repair,
                train_output_replay=train_output_replay,
            )
        ),
        "job_binding_path": str(
            _attempt_path(
                context.stage_dir,
                context.name,
                attempt,
                "job-binding.json",
                dispatch_repair=dispatch_repair,
                launcher_repair=launcher_repair,
                train_output_replay=train_output_replay,
            )
        ),
        "job_state_dir": str(job_state_dir or _job_state_dir()),
        "started_at": started_at
        or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "started_ns": started_ns,
    }
    cuda_receipt = context.config_dir / f"cuda-runtime-{context.image_kind}.json"
    if context.name not in ADAPTER_ACTIONS and cuda_receipt.is_file():
        cuda_gate = json.loads(cuda_receipt.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "workflow": WORKFLOW,
            "image": context.image,
            "gpu_ids": list(context.config["gpu_ids"]),
            "status": "PASS",
        }
        for field, value in expected.items():
            if cuda_gate.get(field) != value:
                raise ValueError(f"CUDA runtime receipt {field} does not match the action")
        mode = cuda_gate.get("compatibility_mode")
        path = cuda_gate.get("compatibility_path")
        if mode == "native" and path is not None:
            raise ValueError("native CUDA runtime receipt must not declare a compatibility path")
        if mode == "image_forward_compat":
            allowed = "/usr/local/cuda/compat/lib.real:/usr/local/cuda/lib64"
            if path != allowed:
                raise ValueError("CUDA runtime receipt has an unapproved compatibility path")
            payload["environment"]["LD_LIBRARY_PATH"] = allowed
        elif mode != "native":
            raise ValueError("CUDA runtime receipt has an unsupported compatibility mode")
    if context.name in ADAPTER_ACTIONS:
        payload["environment"]["IAA_COMPUTE_FRAME"] = context.platform
    if context.name == "visualize_finish":
        # scikit-learn/OpenBLAS can otherwise inherit a host-sized thread
        # count that exceeds the data-services image's compiled thread table
        # and segfault before t-SNE can report an ordinary Python error.
        payload["environment"].update(
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
    if context.name not in ADAPTER_ACTIONS:
        payload["cache_subset"] = _tao_cache_subset(context.cache_dir)
    if dispatch_repair:
        payload["dispatch_repair"] = dispatch_repair
    if launcher_repair:
        payload["launcher_repair"] = launcher_repair
    if unbound_replay:
        payload["unbound_replay"] = unbound_replay
        payload["unbound_replay_evidence_sha256"] = (
            unbound_replay_evidence_sha256
        )
    if train_output_replay:
        payload["train_output_replay"] = train_output_replay
        payload["train_output_replay_evidence_sha256"] = (
            train_output_replay_evidence_sha256
        )
    payload["request_sha256"] = _sha256_json(payload)
    return payload


def _plugin_cache_instance(
    path: pathlib.Path, suffix: tuple[str, ...], name: str
) -> tuple[pathlib.Path, str]:
    """Validate one packaged path and return its cache instance and version."""
    absolute = safe_absolute_path(path, name)
    if tuple(absolute.parts[-len(suffix) :]) != suffix:
        raise ValueError(f"{name} is not the expected packaged artifact path")
    instance = absolute
    for _ in suffix:
        instance = instance.parent
    if instance.parent.name != "tao-skill-bank":
        raise ValueError(f"{name} is not under the TAO plugin cache")
    version = instance.name.split("+codex.", 1)[0]
    if not version:
        raise ValueError(f"{name} has an invalid TAO plugin cache version")
    return instance, version


def _load_request_envelope(path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    """Load request integrity/schema without binding it to the current cache path."""
    resolved = safe_absolute_path(path, "action request", require_exists=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"action request is missing or unsafe: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid action request: {resolved}")
    expected_hash = payload.pop("request_sha256", None)
    actual_hash = _sha256_json(payload)
    payload["request_sha256"] = expected_hash
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ValueError(f"action request digest mismatch: {resolved}")
    try:
        jsonschema.validate(payload, _reference_schema("action-request.schema.json"))
        jsonschema.validate(
            payload["spec_bundle"], _artifact_schema("spec_bundle.schema.json")
        )
    except jsonschema.ValidationError as exc:
        raise ValueError(f"action request schema violation: {exc.message}") from exc
    _aware_timestamp(payload["started_at"], "action request started_at")
    expected = safe_absolute_path(
        _request_path_from_payload(payload), "action request path"
    )
    if resolved != expected:
        raise ValueError(f"action request must remain at {expected}, got {resolved}")
    state = json.loads((pathlib.Path(payload["results_dir"]) / "deft_state.json").read_text())
    lineage = validate_runtime_lineage(state, pathlib.Path(payload["results_dir"]))
    allowed_runtime_digests = {
        state["config"]["iaa_deft_bundle_sha256"],
        *(record["new_sha256"] for record in lineage),
    }
    request_runtime = payload.get("runtime_sha256")
    if request_runtime is not None and request_runtime not in allowed_runtime_digests:
        raise ValueError("action request runtime digest is outside approved lineage")
    return resolved, payload


def _cache_relocation_expected(
    context: ActionContext,
    request: dict[str, Any],
    *,
    allow_regressed_normal_action_id: bool = False,
    allow_visualize_thread_cap_repair: bool = False,
    allow_report_input_mount_repair: bool = False,
) -> dict[str, Any]:
    """Validate a refresh-only cache relocation and return the current request."""
    runtime = pathlib.Path(__file__).resolve().parent / "iaa_deft"
    state = json.loads((context.results_dir / "deft_state.json").read_text())
    validate_runtime_lineage(state, context.results_dir)
    approved_runtime = active_runtime_sha256(state)
    if (
        not isinstance(approved_runtime, str)
        or len(approved_runtime) != 64
        or _python_tree_sha256(runtime) != approved_runtime
    ):
        raise ValueError(
            "cannot rematerialize action request: bundled IAA runtime does not "
            "match immutable run provenance"
        )

    expected = _request(
        context,
        request["attempt"],
        request["started_ns"],
        started_at=request["started_at"],
        job_state_dir=safe_absolute_path(
            pathlib.Path(request["job_state_dir"]), "action request job_state_dir"
        ),
        dispatch_repair=request.get("dispatch_repair", 0),
        launcher_repair=request.get("launcher_repair", 0),
        unbound_replay=request.get("unbound_replay", 0),
        unbound_replay_evidence_sha256=request.get(
            "unbound_replay_evidence_sha256"
        ),
        train_output_replay=request.get("train_output_replay", 0),
        train_output_replay_evidence_sha256=request.get(
            "train_output_replay_evidence_sha256"
        ),
    )
    # Current requests bind controller and patch inputs to durable action-owned
    # snapshots. A plugin cache replacement therefore has no path to relocate;
    # only the independently bound TAO model-cache subset may be refreshed on
    # an unlaunched request.
    if "controller_snapshot" in request and "patches_snapshot" in request:
        normalized = deepcopy(request)
        if "cache_subset" in expected:
            normalized["cache_subset"] = expected["cache_subset"]
        else:
            normalized.pop("cache_subset", None)
        if allow_regressed_normal_action_id and not request.get("dispatch_repair", 0):
            actual_action = normalized["spec_bundle"]["action"]
            if actual_action == _regressed_normal_action_id(
                context, request["attempt"], request["started_ns"]
            ):
                normalized["spec_bundle"]["action"] = expected["spec_bundle"]["action"]
        if allow_visualize_thread_cap_repair:
            _normalize_visualize_thread_cap_repair(normalized, expected)
        if allow_report_input_mount_repair:
            _normalize_report_input_mount_repair(normalized, expected)
        normalized["request_sha256"] = expected["request_sha256"]
        if normalized != expected:
            raise ValueError("action request differences exceed a cache relocation")
        return expected
    patch_suffix = (
        "skills",
        "applications",
        "tao-run-deft-iaa",
        "patches",
    )
    shim_suffix = (
        "skills",
        "applications",
        "tao-run-deft-iaa",
        "scripts",
        "run_deft_cli.py",
    )
    old_patch_mounts = [
        item for item in request["mounts"] if item.get("target") == "/patches"
    ]
    new_patch_mounts = [
        item for item in expected["mounts"] if item.get("target") == "/patches"
    ]
    old_patch_inputs = [
        item
        for item in request["spec_bundle"]["declared_inputs"]
        if item.get("spec_key") == "compatibility_patches"
    ]
    new_patch_inputs = [
        item
        for item in expected["spec_bundle"]["declared_inputs"]
        if item.get("spec_key") == "compatibility_patches"
    ]
    if not all(
        len(items) == 1
        for items in (
            old_patch_mounts,
            new_patch_mounts,
            old_patch_inputs,
            new_patch_inputs,
        )
    ):
        raise ValueError(
            "cannot rematerialize action request: compatibility patch contract is malformed"
        )
    old_patch = pathlib.Path(old_patch_mounts[0]["source"])
    new_patch = pathlib.Path(new_patch_mounts[0]["source"])
    if old_patch_inputs[0].get("uri") != str(old_patch):
        raise ValueError(
            "cannot rematerialize action request: prior patch mount/input disagree"
        )
    if new_patch_inputs[0].get("uri") != str(new_patch):
        raise ValueError(
            "cannot rematerialize action request: current patch mount/input disagree"
        )
    old_instance, old_version = _plugin_cache_instance(
        old_patch, patch_suffix, "prior compatibility patches"
    )
    new_instance, new_version = _plugin_cache_instance(
        new_patch, patch_suffix, "current compatibility patches"
    )
    old_shim_instance, old_shim_version = _plugin_cache_instance(
        pathlib.Path(request["virtualenv_shim"]),
        shim_suffix,
        "prior virtualenv shim",
    )
    new_shim_instance, new_shim_version = _plugin_cache_instance(
        pathlib.Path(expected["virtualenv_shim"]),
        shim_suffix,
        "current virtualenv shim",
    )
    if (
        old_instance != old_shim_instance
        or new_instance != new_shim_instance
        or len({old_version, new_version, old_shim_version, new_shim_version}) != 1
        or old_instance == new_instance
    ):
        raise ValueError(
            "cannot rematerialize action request: cache instances or plugin versions "
            "do not form one valid refresh"
        )

    normalized = deepcopy(request)
    normalized["mounts"][normalized["mounts"].index(old_patch_mounts[0])][
        "source"
    ] = str(new_patch)
    normalized_input = normalized["spec_bundle"]["declared_inputs"][
        normalized["spec_bundle"]["declared_inputs"].index(old_patch_inputs[0])
    ]
    normalized_input["uri"] = str(new_patch)
    normalized["virtualenv_shim"] = expected["virtualenv_shim"]
    # Cache-subset contracts may change between plugin refreshes. Replace the
    # prior manifest only in this already-proven, unlaunched relocation path;
    # the caller rejects status, staging, output, and job-record evidence.
    if "cache_subset" in expected:
        normalized["cache_subset"] = expected["cache_subset"]
    else:
        normalized.pop("cache_subset", None)
    normalized["request_sha256"] = expected["request_sha256"]
    if allow_regressed_normal_action_id and not request.get("dispatch_repair", 0):
        actual_action = normalized["spec_bundle"]["action"]
        regressed_action = _regressed_normal_action_id(
            context, request["attempt"], request["started_ns"]
        )
        if actual_action == regressed_action:
            normalized["spec_bundle"]["action"] = expected["spec_bundle"]["action"]
    if allow_visualize_thread_cap_repair:
        _normalize_visualize_thread_cap_repair(normalized, expected)
    if allow_report_input_mount_repair:
        _normalize_report_input_mount_repair(normalized, expected)
    if normalized != expected:
        raise ValueError(
            "action request differences exceed a cache relocation"
        )
    return expected


_VISUALIZE_THREAD_CAPS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _normalize_visualize_thread_cap_repair(
    normalized: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Apply only the proven OpenBLAS crash correction during retry comparison."""
    prior_environment = normalized.get("environment")
    expected_environment = expected.get("environment")
    if not isinstance(prior_environment, dict) or not isinstance(
        expected_environment, dict
    ):
        raise ValueError("visualize thread-cap repair requires request environments")
    if any(name in prior_environment for name in _VISUALIZE_THREAD_CAPS):
        raise ValueError(
            "visualize thread-cap repair requires all prior caps to be absent"
        )
    if any(
        expected_environment.get(name) != value
        for name, value in _VISUALIZE_THREAD_CAPS.items()
    ):
        raise ValueError("visualize thread-cap repair does not match current caps")
    prior_environment.update(_VISUALIZE_THREAD_CAPS)


def _visualize_thread_cap_repair_evidence(
    context: ActionContext, request: dict[str, Any]
) -> bool:
    """Prove attempt 1 hit the exact OpenBLAS thread-table SIGSEGV."""
    if context.name != "visualize_finish" or request.get("attempt") != 1:
        return False
    if not context.status_path.is_file() or context.status_path.is_symlink():
        return False
    try:
        status = json.loads(context.status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        status.get("status") != "error"
        or status.get("attempt") != 1
        or status.get("request_sha256") != request.get("request_sha256")
    ):
        return False
    try:
        log_path = safe_absolute_path(
            pathlib.Path(request["log_path"]),
            "prior visualize action log",
            require_exists=True,
        )
        if not log_path.is_file() or log_path.is_symlink():
            return False
        log_text = log_path.read_text(encoding="utf-8")
    except (KeyError, OSError, ValueError):
        return False
    return (
        "OpenBLAS warning: precompiled NUM_THREADS exceeded" in log_text
        and "SIGSEGV" in log_text
    )


def _normalize_report_input_mount_repair(
    normalized: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Add only dataset/archive inputs omitted from a failed terminal report."""
    prior_mounts = normalized.get("mounts")
    expected_mounts = expected.get("mounts")
    prior_inputs = normalized.get("spec_bundle", {}).get("declared_inputs")
    expected_inputs = expected.get("spec_bundle", {}).get("declared_inputs")
    if not all(isinstance(rows, list) for rows in (
        prior_mounts, expected_mounts, prior_inputs, expected_inputs
    )):
        raise ValueError("report dataset-mount repair requires typed request inputs")
    additions = [row for row in expected_mounts if row not in prior_mounts]
    input_additions = [row for row in expected_inputs if row not in prior_inputs]
    allowed_inputs = {
        row.get("uri") for row in input_additions
        if row.get("spec_key") == "dataset_parent"
        or str(row.get("spec_key", "")).startswith("archive_parent_")
    }
    allowed_targets = {"/data", *allowed_inputs}
    if (
        not additions
        or not input_additions
        or len(allowed_inputs) != len(input_additions)
        or any(row.get("target") not in allowed_targets for row in additions)
        or any(row.get("read_only") is not True for row in additions)
        or any(row not in expected_mounts for row in prior_mounts)
    ):
        raise ValueError("report input-mount repair exceeds the approved correction")
    normalized["mounts"] = deepcopy(expected_mounts)
    normalized["spec_bundle"]["declared_inputs"] = deepcopy(expected_inputs)


def _report_input_mount_repair_evidence(
    context: ActionContext, request: dict[str, Any]
) -> bool:
    """Prove a report attempt failed only on an omitted read-only input mount."""
    attempt = request.get("attempt")
    if context.name != "report" or attempt not in {1, 2}:
        return False
    status_candidates = (
        context.status_path,
        context.stage_dir / f"{context.name}.attempt-{attempt}.status.json",
    )
    status = None
    for candidate in status_candidates:
        if not candidate.is_file() or candidate.is_symlink():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("attempt") == attempt
            and payload.get("request_sha256") == request.get("request_sha256")
        ):
            status = payload
            break
    if status is None:
        return False
    try:
        log_path = safe_absolute_path(
            pathlib.Path(request["log_path"]),
            "prior report action log",
            require_exists=True,
        )
        log_text = log_path.read_text(encoding="utf-8")
    except (KeyError, OSError, ValueError):
        return False
    markers = {
        1: (
            "committed dataset root is missing images/:",
            "committed dataset root is missing captions/:",
            "committed dataset root is missing non-empty train_pairs.json:",
        ),
        2: (
            "state.config.images_archive must be an existing absolute non-empty file:",
            "state.config.metadata_archive must be an existing absolute non-empty file:",
        ),
    }[attempt]
    return (
        status.get("status") == "error"
        and status.get("attempt") == attempt
        and status.get("request_sha256") == request.get("request_sha256")
        and log_path.is_file()
        and not log_path.is_symlink()
        and all(marker in log_text for marker in markers)
    )


def _rematerialize_unlaunched_cache_request(
    context: ActionContext,
    request_path: pathlib.Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Move an unlaunched request across cache instances without changing work."""
    expected = _cache_relocation_expected(context, request)

    if context.status_path.exists():
        raise ValueError("cannot rematerialize action request with action status evidence")
    for key, label in (
        ("staging_receipt_path", "staging receipt"),
        ("job_binding_path", "job binding"),
        ("log_path", "action log"),
    ):
        if pathlib.Path(request[key]).exists():
            raise ValueError(f"cannot rematerialize action request with {label} evidence")
    if any(path.exists() for path in context.fresh_outputs):
        raise ValueError("cannot rematerialize action request with fresh outputs present")
    if _matching_job_records(request_path, request):
        raise ValueError("cannot rematerialize action request after a job-record was opened")

    atomic_json(request_path, expected)
    return expected


def _load_finalized_request_for_retry(
    context: ActionContext, path: pathlib.Path
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Validate a finalized prior request across a plugin-cache refresh.

    Finalized evidence stays bound to the original request digest and bytes.
    Cache relocation is accepted only when the immutable runtime provenance,
    semantic plugin version, and every non-cache request field still match.
    """
    resolved, request = _load_request_envelope(path)
    try:
        _, current = _load_request(resolved)
        return resolved, current
    except ValueError as exc:
        if str(exc) != "action request fields do not match immutable workflow state and paths":
            raise
    _cache_relocation_expected(
        context,
        request,
        allow_regressed_normal_action_id=True,
        allow_visualize_thread_cap_repair=(
            _visualize_thread_cap_repair_evidence(context, request)
        ),
        allow_report_input_mount_repair=(
            _report_input_mount_repair_evidence(context, request)
        ),
    )
    return resolved, request


def _load_or_rematerialize_unlaunched_request(
    context: ActionContext, path: pathlib.Path
) -> tuple[pathlib.Path, dict[str, Any]]:
    resolved, request = _load_request_envelope(path)
    try:
        _, current = _load_request(resolved)
        return resolved, current
    except ValueError as exc:
        if str(exc) != "action request fields do not match immutable workflow state and paths":
            raise
    return resolved, _rematerialize_unlaunched_cache_request(
        context, resolved, request
    )


def _rebind_virtualenv_previous_pool(context: ActionContext) -> dict[str, int]:
    """Rebind prior-image paths into the approved virtualenv host frame.

    The previous-data pool is derived from container-authored train configs, so
    its ``filepath`` column may contain ``/results`` or ``/data`` paths.  A
    virtualenv action dereferences the parquet on the host and therefore needs
    the corresponding immutable host aliases.  Rewrite only this one
    action-owned input, reject traversal or paths outside the two approved
    mount roots, and leave an already rebound parquet byte-for-byte unchanged.
    """
    if context.platform != "virtualenv" or context.name != "viz_previous_embed":
        return {"rows": 0, "rewritten": 0}

    import pandas as pd

    pool = safe_absolute_path(
        context.stage_dir / "prev_pool.parquet",
        "virtualenv previous-data pool",
    )
    # Preserve the producer's existing behavior for synthetic/preparation-only
    # callers.  The real visualization branch creates this input first, while
    # the embedding command remains responsible for reporting a missing input.
    if not pool.exists():
        return {"rows": 0, "rewritten": 0}
    if not pool.is_file() or pool.is_symlink():
        raise ValueError(f"virtualenv previous-data pool is missing or unsafe: {pool}")

    approved_roots = (
        safe_absolute_path(context.results_dir, "approved results root", require_exists=True),
        safe_absolute_path(
            context.dataset_root.parent,
            "approved dataset parent",
            require_exists=True,
        ),
    )
    aliases = {
        "/results": approved_roots[0],
        "/data": approved_roots[1],
    }

    def under_approved(path: pathlib.Path) -> bool:
        return any(path == root or root in path.parents for root in approved_roots)

    def validate_image_path(candidate: pathlib.Path, value: str, row: int) -> pathlib.Path:
        """Accept an in-root alias only when its resolved file also stays in-root."""
        if not candidate.is_absolute():
            raise ValueError(
                f"previous-data filepath row {row} must be an absolute path: {value}"
            )
        if any(part == ".." for part in candidate.parts):
            raise ValueError(
                f"previous-data filepath row {row} contains traversal: {value}"
            )
        lexical = pathlib.Path(os.path.abspath(candidate))
        if not under_approved(lexical):
            raise ValueError(
                f"previous-data filepath row {row} is outside approved roots: {value}"
            )
        try:
            resolved = lexical.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as exc:
            raise ValueError(
                f"previous-data filepath row {row} is missing or has an invalid symlink: "
                f"{value}"
            ) from exc
        if not under_approved(resolved) or not resolved.is_file():
            raise ValueError(
                f"previous-data filepath row {row} does not resolve to an approved "
                f"regular file: {value}"
            )
        return lexical

    def rebind(raw: Any, row: int) -> str:
        if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
            raise ValueError(f"previous-data filepath row {row} must be a canonical string")
        value = raw
        for alias, host_root in aliases.items():
            if value == alias or value.startswith(alias + "/"):
                suffix = pathlib.PurePosixPath(value).relative_to(alias)
                if any(part in {".", ".."} for part in suffix.parts):
                    raise ValueError(
                        f"previous-data filepath row {row} contains traversal: {value}"
                    )
                candidate = host_root.joinpath(*suffix.parts)
                break
        else:
            candidate = pathlib.Path(value)
        candidate = validate_image_path(candidate, value, row)
        return str(candidate)

    frame = pd.read_parquet(pool)
    if "filepath" not in frame.columns or frame.empty:
        raise ValueError("virtualenv previous-data pool must contain non-empty filepath rows")
    rebound = [rebind(value, row) for row, value in enumerate(frame["filepath"])]
    original = frame["filepath"].tolist()
    rewritten = sum(before != after for before, after in zip(original, rebound, strict=True))
    if rewritten:
        frame = frame.copy()
        frame["filepath"] = rebound
        fd, temporary = tempfile.mkstemp(
            prefix=pool.name + ".", suffix=".tmp", dir=str(pool.parent)
        )
        os.close(fd)
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, pool)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    return {"rows": len(frame), "rewritten": rewritten}


def _unbound_replay_evidence_path(context: ActionContext) -> pathlib.Path:
    return context.stage_dir / f"{context.name}.unbound-replay.evidence.json"


def _load_unbound_replay_evidence(
    context: ActionContext, expected_digest: str
) -> dict[str, Any]:
    path = safe_absolute_path(
        _unbound_replay_evidence_path(context),
        "unbound replay evidence",
        require_exists=True,
    )
    if not path.is_file() or path.is_symlink():
        raise ValueError("unbound replay evidence is missing or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("unbound replay evidence root must be an object")
    body = dict(payload)
    digest = body.pop("evidence_sha256", None)
    if digest != expected_digest or digest != _sha256_json(body):
        raise ValueError("unbound replay evidence digest mismatch")
    if (
        payload.get("schema_version") != "1"
        or payload.get("workflow") != WORKFLOW
        or payload.get("kind") != "terminal_unbound_replay"
        or payload.get("platform") != "slurm"
        or payload.get("name") != context.name
        or payload.get("attempt") != 1
    ):
        raise ValueError("unbound replay evidence identity is invalid")
    rows = payload.get("quarantined_outputs")
    if not isinstance(rows, list) or len(rows) != len(context.fresh_outputs):
        raise ValueError("unbound replay evidence output inventory is invalid")
    expected_outputs = {str(path) for path in context.fresh_outputs}
    actual_outputs: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("unbound replay output evidence row is invalid")
        original = row.get("original")
        if not isinstance(original, str):
            raise ValueError("unbound replay output original path is invalid")
        actual_outputs.add(original)
        archive = safe_absolute_path(
            pathlib.Path(str(row.get("archive", ""))),
            "quarantined unbound output",
            require_exists=True,
        )
        if (
            not archive.is_file()
            or archive.is_symlink()
            or archive.stat().st_size != row.get("size")
            or _sha256_file(archive) != row.get("sha256")
        ):
            raise ValueError("quarantined unbound output digest/size is invalid")
    if actual_outputs != expected_outputs:
        raise ValueError("unbound replay evidence does not bind exact action outputs")
    for path_key, hash_key, label in (
        ("prior_request_path", "prior_request_file_sha256", "prior action request"),
        ("prior_job_record_path", "prior_job_record_sha256", "prior job record"),
        ("prior_log_path", "prior_log_sha256", "prior action log"),
    ):
        artifact = safe_absolute_path(
            pathlib.Path(str(payload.get(path_key, ""))), label, require_exists=True
        )
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or _sha256_file(artifact) != payload.get(hash_key)
        ):
            raise ValueError(f"{label} changed after unbound replay evidence")
    return payload


def unbound_replay(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, Any]]:
    """Replay one safe SLURM adapter after a terminal job lacked its binding."""

    context = validate_action(
        results_dir=args.results_dir,
        image_kind=args.image,
        stage_dir=args.stage_dir,
        name=args.name,
        pass_hf_token=args.pass_hf_token,
        fresh_outputs=args.fresh_output,
        command=args.command,
    )
    if context.platform != "slurm" or context.name not in UNBOUND_REPLAY_ACTIONS:
        raise ValueError(
            "unbound replay is restricted to allowlisted deterministic SLURM adapters"
        )
    request_one_path = _request_path_for(context, 1)
    request_two_path = _request_path_for(context, 2)
    evidence_path = _unbound_replay_evidence_path(context)
    with _exclusive_lock(context.lock_path):
        if context.status_path.exists():
            raise ValueError("unbound replay is forbidden after platform status exists")
        if request_two_path.exists():
            _, request = _load_request(request_two_path)
            if request.get("unbound_replay") != 1:
                raise ValueError("attempt-2 request exists outside unbound replay lineage")
            _load_unbound_replay_evidence(
                context, request["unbound_replay_evidence_sha256"]
            )
            return request_two_path, request
        if not request_one_path.is_file():
            raise ValueError("unbound replay requires the immutable attempt-1 request")
        request_one_path, request_one = _load_launched_request(request_one_path)
        if request_one.get("attempt") != 1:
            raise ValueError("unbound replay requires attempt 1")
        binding_path = pathlib.Path(request_one["job_binding_path"])
        if binding_path.exists():
            raise ValueError("unbound replay is forbidden when a job binding exists")
        matches = _matching_job_records(request_one_path, request_one)
        if len(matches) != 1:
            raise ValueError("unbound replay requires exactly one request-owned job record")
        job_path, job = matches[0]
        if job.get("terminal_state") != "COMPLETE":
            raise ValueError("unbound replay requires a terminal COMPLETE native job")
        _validate_terminal_job(job)
        if not isinstance(job.get("backend_ref"), str) or not job["backend_ref"].strip():
            raise ValueError("unbound replay requires the prior native backend reference")
        log_path = safe_absolute_path(
            pathlib.Path(request_one["log_path"]),
            "prior unbound action log",
            require_exists=True,
        )
        if not log_path.is_file() or log_path.is_symlink() or log_path.stat().st_size == 0:
            raise ValueError("unbound replay requires one captured non-empty action log")

        if evidence_path.exists():
            existing = json.loads(evidence_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError("unbound replay evidence root must be an object")
            evidence_digest = existing.get("evidence_sha256")
            evidence = _load_unbound_replay_evidence(context, evidence_digest)
        else:
            quarantine_root = safe_absolute_path(
                context.stage_dir
                / ".tao-runtime"
                / f"{context.name}.unbound-replay-evidence"
                / "outputs",
                "unbound replay quarantine",
            )
            quarantine_root.mkdir(parents=True, exist_ok=True)
            if quarantine_root.is_symlink() or quarantine_root.resolve() != quarantine_root:
                raise ValueError("unbound replay quarantine is unsafe")
            rows = []
            for output in context.fresh_outputs:
                output = safe_absolute_path(output, "unbound action output")
                relative = output.relative_to(context.stage_dir)
                archive = safe_absolute_path(
                    quarantine_root / relative,
                    "quarantined unbound action output",
                )
                if output.exists() and archive.exists():
                    raise ValueError("both live and quarantined unbound outputs exist")
                if output.exists():
                    if not output.is_file() or output.is_symlink() or output.stat().st_size == 0:
                        raise ValueError("unbound replay supports non-empty regular outputs only")
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(output, archive)
                if not archive.is_file() or archive.is_symlink() or archive.stat().st_size == 0:
                    raise ValueError("unbound action output is missing from quarantine")
                rows.append(
                    {
                        "original": str(output),
                        "archive": str(archive),
                        "size": archive.stat().st_size,
                        "sha256": _sha256_file(archive),
                    }
                )
            evidence = {
                "schema_version": "1",
                "workflow": WORKFLOW,
                "kind": "terminal_unbound_replay",
                "platform": "slurm",
                "name": context.name,
                "attempt": 1,
                "prior_request_path": str(request_one_path),
                "prior_request_sha256": request_one["request_sha256"],
                "prior_request_file_sha256": _sha256_file(request_one_path),
                "prior_job_record_path": str(job_path),
                "prior_job_record_sha256": _sha256_file(job_path),
                "prior_job_id": job["id"],
                "prior_backend_ref": job["backend_ref"],
                "prior_backend_state": job["terminal_state"],
                "prior_log_path": str(log_path),
                "prior_log_sha256": _sha256_file(log_path),
                "quarantined_outputs": rows,
                "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
            evidence["evidence_sha256"] = _sha256_json(evidence)
            atomic_json(evidence_path, evidence)
            evidence_digest = evidence["evidence_sha256"]
            _load_unbound_replay_evidence(context, evidence_digest)

        if any(path.exists() for path in context.fresh_outputs):
            raise ValueError("unbound replay outputs were not fully quarantined")
        started_ns = time.time_ns()
        payload = _request(
            context,
            2,
            started_ns,
            unbound_replay=1,
            unbound_replay_evidence_sha256=evidence["evidence_sha256"],
            materialize_snapshots=True,
        )
        for evidence_key in ("log_path", "staging_receipt_path", "job_binding_path"):
            if pathlib.Path(payload[evidence_key]).exists():
                raise ValueError("unbound replay attempt-2 evidence already exists")
        atomic_json(request_two_path, payload)
        runtime_dir = safe_absolute_path(
            pathlib.Path(payload["platform_runtime_dir"]),
            "platform runtime directory",
        )
        runtime_dir.relative_to(context.stage_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        if runtime_dir.is_symlink() or runtime_dir.resolve() != runtime_dir:
            raise ValueError("unbound replay platform runtime directory is unsafe")
        return request_two_path, payload


def prepare(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, Any]]:
    context = validate_action(
        results_dir=args.results_dir,
        image_kind=args.image,
        stage_dir=args.stage_dir,
        name=args.name,
        pass_hf_token=args.pass_hf_token,
        fresh_outputs=args.fresh_output,
        command=args.command,
    )
    with _exclusive_lock(context.lock_path):
        existing = load_existing_status(context.status_path)
        attempt_one_path = _request_path_for(context, 1)
        attempt_two_path = _request_path_for(context, 2)
        archived_status_path = (
            context.stage_dir / f"{context.name}.attempt-1.status.json"
        )
        safe_absolute_path(archived_status_path, "archived attempt-1 status")

        if existing is None:
            if attempt_two_path.exists():
                if not archived_status_path.is_file():
                    raise ValueError(
                        "attempt-2 request exists without archived terminal attempt-1 status"
                    )
                _, request = _load_or_rematerialize_unlaunched_request(
                    context, attempt_two_path
                )
                return attempt_two_path, request
            if attempt_one_path.exists():
                if archived_status_path.exists():
                    raise ValueError(
                        "archived attempt-1 status exists without its attempt-2 request"
                    )
                # Crash-safe initial prepare: return the same immutable attempt.
                _, request = _load_or_rematerialize_unlaunched_request(
                    context, attempt_one_path
                )
                return attempt_one_path, request
            prior_attempt = 0
        else:
            raw_attempt = existing.get("attempt")
            if (
                not isinstance(raw_attempt, int)
                or isinstance(raw_attempt, bool)
                or raw_attempt not in {1, 2}
            ):
                raise ValueError(
                    f"existing command status has invalid attempt: {context.status_path}"
                )
            prior_attempt = raw_attempt
            if existing.get("status") == "ok":
                raise ValueError(
                    f"action already completed successfully: {context.status_path}"
                )
            prior_request_path = _request_path_for(context, prior_attempt)
            if not prior_request_path.exists():
                raise ValueError(
                    "cannot retry without the prior immutable action request"
                )
            prior_request_path, prior_request = _load_finalized_request_for_retry(
                context, prior_request_path
            )
            _validate_retry_lineage(
                context,
                prior_request_path,
                prior_request,
                existing,
            )
            if prior_attempt >= 2:
                raise ValueError(
                    f"attempt budget exhausted for {context.name} (attempt={prior_attempt})"
                )
            if archived_status_path.exists():
                archived = load_existing_status(archived_status_path)
                if archived != existing:
                    raise ValueError(
                        "archived attempt-1 status conflicts with active terminal status"
                    )
            else:
                atomic_json(archived_status_path, existing)
            if attempt_two_path.exists():
                # Recovery after the attempt-2 request became durable but before
                # the fixed active-status path was cleared.
                _, request = _load_request(attempt_two_path)
                context.status_path.unlink()
                return attempt_two_path, request

        _rebind_virtualenv_previous_pool(context)

        if context.name == "dataset_rebuild":
            data_parent = safe_absolute_path(
                context.dataset_root.parent, "dataset rebuild parent"
            )
            data_parent.mkdir(parents=True, exist_ok=True)
            if data_parent.is_symlink() or data_parent.resolve() != data_parent:
                raise ValueError("dataset rebuild parent is unsafe")

        for output in context.fresh_outputs:
            try:
                output.unlink()
            except FileNotFoundError:
                pass
        attempt = prior_attempt + 1
        log_path = _attempt_path(context.stage_dir, context.name, attempt, "log")
        try:
            log_path.unlink()
        except FileNotFoundError:
            pass
        for evidence in (
            _attempt_path(context.stage_dir, context.name, attempt, "staged.json"),
            _attempt_path(context.stage_dir, context.name, attempt, "job-binding.json"),
        ):
            safe_absolute_path(evidence, "prior attempt evidence")
            try:
                evidence.unlink()
            except FileNotFoundError:
                pass
        started_ns = time.time_ns()
        payload = _request(
            context, attempt, started_ns, materialize_snapshots=True
        )
        request_path = _request_path_for(context, attempt)
        atomic_json(request_path, payload)
        runtime_dir = safe_absolute_path(
            pathlib.Path(payload["platform_runtime_dir"]),
            "platform runtime directory",
        )
        runtime_dir.relative_to(context.stage_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        if runtime_dir.is_symlink() or runtime_dir.resolve() != runtime_dir:
            raise ValueError(
                f"platform runtime directory is unsafe: {runtime_dir}"
            )
        if existing is not None:
            # The prior terminal status is durable at its attempt-specific
            # archive before the fixed active path is cleared.  A repeated
            # prepare now returns the already-written attempt-2 request.
            context.status_path.unlink()
    return request_path, payload


def _dispatch_repair_classification(
    context: ActionContext, request: dict[str, Any], status: dict[str, Any]
) -> str:
    """Classify one exact pre-workload dispatch-contract failure."""
    if request.get("name") == "report":
        if (
            status.get("backend_state") != "ERROR"
            or status.get("backend_exit_code") != 1
            or status.get("artifact_error") is not None
            or status.get("status") != "error"
        ):
            raise ValueError("report repair requires an exact terminal adapter error")
        log_path = safe_absolute_path(
            pathlib.Path(request["log_path"]),
            "report repair action log",
            require_exists=True,
        )
        if not log_path.is_file() or log_path.is_symlink():
            raise ValueError("report repair requires a safe captured action log")
        log_text = log_path.read_text(encoding="utf-8")
        if request.get("attempt") == 1 and all(marker in log_text for marker in (
            "committed dataset root is missing images/:",
            "committed dataset root is missing captions/:",
            "committed dataset root is missing non-empty train_pairs.json:",
        )):
            return "report-dataset-mount-contract"
        if request.get("attempt") == 2 and all(marker in log_text for marker in (
            "state.config.images_archive must be an existing absolute non-empty file:",
            "state.config.metadata_archive must be an existing absolute non-empty file:",
        )):
            return "report-archive-mount-contract"
        raise ValueError("report repair failure is not an allowlisted mount classifier")
    if request["platform"] != "virtualenv":
        raise ValueError("dispatch repair is supported only for virtualenv pre-dispatch failures")
    if (
        status.get("backend_state") != "ERROR"
        or status.get("backend_exit_code") != 2
        or status.get("artifact_error") is not None
    ):
        raise ValueError("dispatch repair requires an exact terminal pre-dispatch status")
    log_path = safe_absolute_path(
        pathlib.Path(request["log_path"]), "pre-dispatch action log", require_exists=True
    )
    if not log_path.is_file() or log_path.is_symlink():
        raise ValueError("dispatch repair requires a safe captured action log")
    expected_messages = {
        "config-mount-contract": (
            "run_deft_cli: action config must be a regular file below the approved "
            "/specs mount\n"
        ),
        "attempt-request-path-contract": (
            "run_deft_cli: action request must remain at "
            f"{context.stage_dir / (context.name + '.action.json')}\n"
        ),
    }
    try:
        log_text = log_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("dispatch repair action log is not UTF-8") from exc
    classifications = [
        name for name, message in expected_messages.items() if log_text == message
    ]
    if len(classifications) != 1:
        raise ValueError("dispatch repair failure is not an allowlisted pre-dispatch classifier")

    runner_status_path = safe_absolute_path(
        pathlib.Path(request["platform_runtime_dir"])
        / ".tao_runner"
        / "exit_status.json",
        "virtualenv runner exit status",
        require_exists=True,
    )
    if not runner_status_path.is_file() or runner_status_path.is_symlink():
        raise ValueError("dispatch repair requires safe virtualenv runner exit evidence")
    runner_status = json.loads(runner_status_path.read_text(encoding="utf-8"))
    if (
        not isinstance(runner_status, dict)
        or runner_status.get("return_code") != 2
        or runner_status.get("canceled") is not False
        or runner_status.get("error") is not None
        or not isinstance(runner_status.get("finished_at"), (int, float))
    ):
        raise ValueError("dispatch repair runner evidence is not an exact clean exit-code-2")
    native_log = safe_absolute_path(
        pathlib.Path(request["platform_runtime_dir"]) / "logs" / "job.log",
        "virtualenv runner native log",
        require_exists=True,
    )
    if (
        not native_log.is_file()
        or native_log.is_symlink()
        or _sha256_file(native_log) != _sha256_file(log_path)
    ):
        raise ValueError("dispatch repair captured and native logs do not match")
    return classifications[0]


def dispatch_repair(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, Any]]:
    """Mint one non-workload repair after two proven shim pre-dispatch failures."""
    context = validate_action(
        results_dir=args.results_dir,
        image_kind=args.image,
        stage_dir=args.stage_dir,
        name=args.name,
        pass_hf_token=args.pass_hf_token,
        fresh_outputs=args.fresh_output,
        command=args.command,
    )
    repair_path = _attempt_path(
        context.stage_dir,
        context.name,
        2,
        "action.json",
        dispatch_repair=1,
    )
    archived_two_path = context.stage_dir / f"{context.name}.attempt-2.status.json"
    with _exclusive_lock(context.lock_path):
        if repair_path.exists():
            active = load_existing_status(context.status_path)
            if active is not None:
                raise ValueError("dispatch repair already finalized; a second repair is forbidden")
            return _load_or_rematerialize_unlaunched_request(context, repair_path)

        request_paths = (
            _request_path_for(context, 1),
            _request_path_for(context, 2),
        )
        status_paths = (
            context.stage_dir / f"{context.name}.attempt-1.status.json",
            archived_two_path if archived_two_path.exists() else context.status_path,
        )
        if not all(path.is_file() for path in (*request_paths, *status_paths)):
            raise ValueError("dispatch repair requires both finalized action attempts")

        requests: list[dict[str, Any]] = []
        classifications: list[str] = []
        for request_path, status_path in zip(request_paths, status_paths, strict=True):
            resolved_request, request = _load_finalized_request_for_retry(
                context, request_path
            )
            status = load_existing_status(status_path)
            if status is None:
                raise ValueError("dispatch repair terminal status is missing")
            _validate_retry_lineage(
                context, resolved_request, request, status
            )
            matches = _matching_job_records(resolved_request, request)
            if (
                len(matches) != 1
                or matches[0][1].get("terminal_state") != "ERROR"
            ):
                raise ValueError("dispatch repair requires one inactive terminal ERROR job")
            classifications.append(
                _dispatch_repair_classification(context, request, status)
            )
            requests.append(request)

        expected_classifications = (
            {
                "report-dataset-mount-contract",
                "report-archive-mount-contract",
            }
            if context.name == "report"
            else {"config-mount-contract", "attempt-request-path-contract"}
        )
        if set(classifications) != expected_classifications:
            raise ValueError("dispatch repair requires both known dispatch defects")
        if any(path.exists() for path in context.fresh_outputs):
            raise ValueError("dispatch repair is forbidden after any workload output exists")

        if context.status_path.exists():
            if archived_two_path.exists():
                raise ValueError("dispatch repair attempt-2 status archive already exists")
            os.replace(context.status_path, archived_two_path)

        started_ns = time.time_ns()
        payload = _request(
            context,
            2,
            started_ns,
            dispatch_repair=1,
            materialize_snapshots=True,
        )
        for evidence_key in ("log_path", "staging_receipt_path", "job_binding_path"):
            if pathlib.Path(payload[evidence_key]).exists():
                raise ValueError("dispatch repair evidence already exists without its request")
        atomic_json(repair_path, payload)
        return repair_path, payload


def _launcher_repair_classification(
    context: ActionContext, request: dict[str, Any], status: dict[str, Any]
) -> str:
    """Prove one SLURM CLIP train stopped in an allowlisted launcher defect."""
    bundle = request.get("spec_bundle", {})
    if (
        request.get("platform") != "slurm"
        or request.get("name") != "train"
        or bundle.get("command") != "clip"
        or bundle.get("args", [])[:1] != ["train"]
    ):
        raise ValueError("launcher repair supports only SLURM IAA clip train")
    if (
        status.get("backend_state") not in {"ERROR", "CANCELED"}
        or status.get("artifact_error") is not None
        or status.get("status") != "error"
    ):
        raise ValueError("launcher repair requires an exact terminal topology status")
    log_path = safe_absolute_path(
        pathlib.Path(request["log_path"]), "launcher repair action log", require_exists=True
    )
    if not log_path.is_file() or log_path.is_symlink():
        raise ValueError("launcher repair requires a safe captured action log")
    try:
        log_text = log_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("launcher repair action log is not UTF-8") from exc

    first_batch_markers = (
        "Epoch 0:",
        "Training DataLoader 0:",
        "Train finished successfully.",
        "GLOBAL_RANK: 1",
        "MEMBER: 2/2",
    )
    if any(marker in log_text for marker in first_batch_markers):
        raise ValueError("launcher repair is forbidden after rank 1 or workload start")

    mismatch = (
        "You set `devices=2` in Lightning, but the number of tasks per node "
        "configured in SLURM `--ntasks-per-node=1` does not match."
    )
    hung_init = "Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2"
    if (
        request.get("attempt") == 1
        and status.get("backend_state") == "ERROR"
        and status.get("backend_exit_code") == 1
        and mismatch in log_text
        and hung_init not in log_text
    ):
        return "single-task-device-mismatch"
    if (
        request.get("attempt") == 2
        and status.get("backend_state") == "CANCELED"
        and status.get("backend_exit_code") is None
        and hung_init in log_text
        and mismatch not in log_text
        and "Traceback (most recent call last)" not in log_text
        and "Error executing job" not in log_text
    ):
        return "single-parent-rank0-hang"
    raise ValueError("launcher repair failure is not an allowlisted topology classifier")


def launcher_repair(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, Any]]:
    """Mint one distinct SLURM train repair after two proven topology failures."""
    context = validate_action(
        results_dir=args.results_dir,
        image_kind=args.image,
        stage_dir=args.stage_dir,
        name=args.name,
        pass_hf_token=args.pass_hf_token,
        fresh_outputs=args.fresh_output,
        command=args.command,
    )
    repair_path = _attempt_path(
        context.stage_dir, context.name, 2, "action.json", launcher_repair=1
    )
    archived_two_path = context.stage_dir / f"{context.name}.attempt-2.status.json"
    with _exclusive_lock(context.lock_path):
        if repair_path.exists():
            active = load_existing_status(context.status_path)
            if active is not None:
                raise ValueError("launcher repair already finalized; a second repair is forbidden")
            return _load_or_rematerialize_unlaunched_request(context, repair_path)

        request_paths = (_request_path_for(context, 1), _request_path_for(context, 2))
        status_paths = (
            context.stage_dir / f"{context.name}.attempt-1.status.json",
            archived_two_path if archived_two_path.exists() else context.status_path,
        )
        if not all(path.is_file() for path in (*request_paths, *status_paths)):
            raise ValueError("launcher repair requires both finalized action attempts")

        classifications: list[str] = []
        for request_path, status_path in zip(request_paths, status_paths, strict=True):
            resolved_request, request = _load_finalized_request_for_retry(context, request_path)
            status = load_existing_status(status_path)
            if status is None:
                raise ValueError("launcher repair terminal status is missing")
            _validate_retry_lineage(context, resolved_request, request, status)
            matches = _matching_job_records(resolved_request, request)
            if (
                len(matches) != 1
                or matches[0][1].get("terminal_state") not in {"ERROR", "CANCELED"}
            ):
                raise ValueError("launcher repair requires one inactive terminal job per attempt")
            classifications.append(
                _launcher_repair_classification(context, request, status)
            )

        if classifications != [
            "single-task-device-mismatch",
            "single-parent-rank0-hang",
        ]:
            raise ValueError("launcher repair requires both known topology failures in order")
        if any(path.exists() for path in context.fresh_outputs):
            raise ValueError("launcher repair is forbidden after any workload output exists")
        if any(context.stage_dir.rglob("*.pth")) or any(context.stage_dir.rglob("*.ckpt")):
            raise ValueError("launcher repair is forbidden after any checkpoint exists")

        if context.status_path.exists():
            if archived_two_path.exists():
                raise ValueError("launcher repair attempt-2 status archive already exists")
            os.replace(context.status_path, archived_two_path)

        started_ns = time.time_ns()
        payload = _request(
            context, 2, started_ns, launcher_repair=1,
            materialize_snapshots=True,
        )
        for evidence_key in ("log_path", "staging_receipt_path", "job_binding_path"):
            if pathlib.Path(payload[evidence_key]).exists():
                raise ValueError("launcher repair evidence already exists without its request")
        atomic_json(repair_path, payload)
        runtime_dir = safe_absolute_path(
            pathlib.Path(payload["platform_runtime_dir"]),
            "platform runtime directory",
        )
        runtime_dir.relative_to(context.stage_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        if runtime_dir.is_symlink() or runtime_dir.resolve() != runtime_dir:
            raise ValueError(
                f"platform runtime directory is unsafe: {runtime_dir}"
            )
        return repair_path, payload


def _train_checkpoint_candidates(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Return publishable checkpoints without following or accepting symlinks."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("train output replay requires one safe train directory")
    candidates: list[pathlib.Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            if path.suffix.lower() in {".pth", ".ckpt", ".safetensors"}:
                raise ValueError("train checkpoint inventory contains a symlink")
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        lowered = path.name.lower()
        if (
            path.suffix.lower() in {".pth", ".ckpt", ".safetensors"}
            and "best" not in relative.parts
            and "latest" not in lowered
            and not lowered.endswith("_pretrained.pth")
            and not lowered.endswith(".tmp")
        ):
            candidates.append(path)
    return tuple(candidates)


def _train_output_replay_evidence_path(context: ActionContext) -> pathlib.Path:
    return context.stage_dir / ".tao-runtime" / "train.output-replay.evidence.json"


def _load_train_output_replay_evidence(
    context: ActionContext, expected_digest: str | None
) -> dict[str, Any]:
    path = safe_absolute_path(
        _train_output_replay_evidence_path(context),
        "train output replay evidence", require_exists=True,
    )
    if not path.is_file() or path.is_symlink():
        raise ValueError("train output replay evidence is missing or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("train output replay evidence root must be an object")
    body = dict(payload)
    digest = body.pop("evidence_sha256", None)
    if (
        not isinstance(expected_digest, str)
        or digest != expected_digest
        or digest != _sha256_json(body)
    ):
        raise ValueError("train output replay evidence digest mismatch")
    expected = {
        "schema_version": "1",
        "workflow": WORKFLOW,
        "kind": "airflow_slurm_train_output_loss",
        "platform": "slurm",
        "name": "train",
        "label": context.label,
        "results_dir": str(context.results_dir),
        "stage_dir": str(context.stage_dir),
        "prior_train_attempt": 2,
        "publisher_attempt": 1,
        "controller_checkpoint_count": 0,
        "remote_checkpoint_count": 0,
        "classifier": "successful-train-checkpoint-not-synchronized",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"train output replay evidence {field} is invalid")
    if payload.get("remote_checkpoint_inventory_sha256") != hashlib.sha256(b"").hexdigest():
        raise ValueError("train output replay remote absence digest is invalid")
    rows = payload.get("artifacts")
    required_roles = {
        "prior_train_request", "prior_train_job_record", "prior_train_log",
        "prior_train_platform_status", "prior_train_workload_status",
        "publisher_request", "publisher_job_record", "publisher_log",
        "publisher_platform_status",
    }
    if not isinstance(rows, list) or len(rows) != len(required_roles):
        raise ValueError("train output replay artifact inventory is invalid")
    roles: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"role", "path", "size", "sha256"}:
            raise ValueError("train output replay artifact row is invalid")
        role = row.get("role")
        if not isinstance(role, str) or role in roles:
            raise ValueError("train output replay artifact role is invalid")
        roles.add(role)
        artifact = safe_absolute_path(
            pathlib.Path(str(row.get("path", ""))),
            f"train output replay {role}", require_exists=True,
        )
        if (
            not artifact.is_file() or artifact.is_symlink()
            or artifact.stat().st_size != row.get("size")
            or _sha256_file(artifact) != row.get("sha256")
        ):
            raise ValueError(f"train output replay {role} changed after classification")
    if roles != required_roles:
        raise ValueError("train output replay artifact roles are incomplete")
    return payload


def train_output_replay(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, Any]]:
    """Replay one successful SLURM train after proven bridge checkpoint loss."""
    context = validate_action(
        results_dir=args.results_dir,
        image_kind=args.image,
        stage_dir=args.stage_dir,
        name=args.name,
        pass_hf_token=args.pass_hf_token,
        fresh_outputs=args.fresh_output,
        command=args.command,
    )
    if (
        context.platform != "slurm"
        or context.config.get("orchestrator") != "airflow"
        or context.name != "train"
        or context.command[:2] != ["clip", "train"]
        or not re.fullmatch(r"iter[1-9][0-9]*", context.label)
    ):
        raise ValueError("train output replay is restricted to Airflow-over-SLURM CLIP train")
    evidence_path = safe_absolute_path(
        args.recovery_evidence, "train output replay evidence", require_exists=True
    )
    if evidence_path != _train_output_replay_evidence_path(context):
        raise ValueError("train output replay evidence path is noncanonical")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("train output replay evidence root must be an object")
    evidence_digest = evidence.get("evidence_sha256")
    replay_path = _attempt_path(
        context.stage_dir, context.name, 2, "action.json", train_output_replay=1
    )
    archive_root = context.stage_dir / ".tao-runtime" / "train.output-replay.prior"
    platform_status = context.status_path
    workload_status = context.fresh_outputs[0]
    archived_platform = archive_root / "train.status.json"
    archived_workload = archive_root / "status.json"

    with _exclusive_lock(context.lock_path):
        if replay_path.exists():
            if context.status_path.exists():
                raise ValueError("train output replay already finalized; a second replay is forbidden")
            return _load_or_rematerialize_unlaunched_request(context, replay_path)
        if _train_checkpoint_candidates(context.stage_dir):
            raise ValueError("train output replay is forbidden while a checkpoint exists")
        if (context.stage_dir / "train.output-sync.json").exists():
            raise ValueError("train output replay is forbidden after checkpoint synchronization")
        if any((context.stage_dir / name).exists() for name in (
            "train.output-replay.log", "train.output-replay.staged.json",
            "train.output-replay.job-binding.json",
        )):
            raise ValueError("train output replay evidence already exists without its request")
        if not platform_status.is_file() or platform_status.is_symlink():
            raise ValueError("train output replay requires the successful train status")
        if not workload_status.is_file() or workload_status.is_symlink():
            raise ValueError("train output replay requires the successful workload status")

        rows = evidence.get("artifacts") if isinstance(evidence, dict) else None
        if not isinstance(rows, list):
            raise ValueError("train output replay artifact inventory is invalid")
        indexed = {
            row.get("role"): row for row in rows if isinstance(row, dict)
        }
        for role, source, archive in (
            ("prior_train_platform_status", platform_status, archived_platform),
            ("prior_train_workload_status", workload_status, archived_workload),
        ):
            row = indexed.get(role)
            if not isinstance(row, dict) or pathlib.Path(str(row.get("path", ""))) != archive:
                raise ValueError(f"train output replay {role} archive path is invalid")
            if (
                source.stat().st_size != row.get("size")
                or _sha256_file(source) != row.get("sha256")
            ):
                raise ValueError(f"train output replay {role} source changed")
            if archive.exists() or archive.is_symlink():
                raise ValueError(f"train output replay {role} archive already exists")

        train_status = json.loads(platform_status.read_text(encoding="utf-8"))
        publisher_status_path = context.stage_dir / "publish_checkpoint.status.json"
        publisher_status = json.loads(publisher_status_path.read_text(encoding="utf-8"))
        publisher_log = context.stage_dir / "publish_checkpoint.log"
        artifact_paths = {
            role: pathlib.Path(str(row["path"]))
            for role, row in indexed.items()
            if role not in {
                "prior_train_platform_status", "prior_train_workload_status"
            }
        }
        prior_request = json.loads(
            artifact_paths["prior_train_request"].read_text(encoding="utf-8")
        )
        prior_job = json.loads(
            artifact_paths["prior_train_job_record"].read_text(encoding="utf-8")
        )
        publisher_request = json.loads(
            artifact_paths["publisher_request"].read_text(encoding="utf-8")
        )
        publisher_job = json.loads(
            artifact_paths["publisher_job_record"].read_text(encoding="utf-8")
        )
        marker = f"No checkpoints found under {context.stage_dir}"
        if (
            train_status.get("workflow") != WORKFLOW
            or train_status.get("name") != "train"
            or train_status.get("attempt") != 2
            or train_status.get("platform") != "slurm"
            or train_status.get("backend_state") != "COMPLETE"
            or train_status.get("status") != "ok"
            or train_status.get("exit_code") != 0
            or train_status.get("command", [])[:2] != ["clip", "train"]
        ):
            raise ValueError("train output replay lacks exact successful attempt-2 status")
        if (
            prior_request.get("workflow") != WORKFLOW
            or prior_request.get("name") != "train"
            or prior_request.get("attempt") != 2
            or prior_request.get("request_sha256") != train_status.get("request_sha256")
            or prior_request.get("spec_bundle", {}).get("command") != "clip"
            or prior_request.get("spec_bundle", {}).get("args", [])[:1] != ["train"]
            or prior_job.get("id") != train_status.get("job_id")
            or prior_job.get("action") != prior_request.get("spec_bundle", {}).get("action")
            or prior_job.get("terminal_state") != "COMPLETE"
            or prior_job.get("backend_ref") != train_status.get("backend_ref")
        ):
            raise ValueError("train output replay prior request/job lineage is invalid")
        if (
            publisher_status.get("workflow") != WORKFLOW
            or publisher_status.get("name") != "publish_checkpoint"
            or publisher_status.get("attempt") != 1
            or publisher_status.get("platform") != "slurm"
            or publisher_status.get("backend_state") != "ERROR"
            or publisher_status.get("status") != "error"
            or marker not in publisher_log.read_text(encoding="utf-8")
        ):
            raise ValueError("train output replay lacks the exact checkpoint-loss publisher failure")
        if (
            publisher_request.get("workflow") != WORKFLOW
            or publisher_request.get("name") != "publish_checkpoint"
            or publisher_request.get("attempt") != 1
            or publisher_request.get("request_sha256")
            != publisher_status.get("request_sha256")
            or publisher_job.get("id") != publisher_status.get("job_id")
            or publisher_job.get("action")
            != publisher_request.get("spec_bundle", {}).get("action")
            or publisher_job.get("terminal_state") != "ERROR"
            or publisher_job.get("backend_ref") != publisher_status.get("backend_ref")
        ):
            raise ValueError("train output replay publisher request/job lineage is invalid")
        if (context.stage_dir / "publish_checkpoint.attempt-2.action.json").exists():
            raise ValueError("train output replay is forbidden after publisher retry")

        archive_root.mkdir(parents=True)
        if archive_root.is_symlink() or archive_root.resolve() != archive_root:
            raise ValueError("train output replay archive is unsafe")
        os.replace(platform_status, archived_platform)
        os.replace(workload_status, archived_workload)
        _load_train_output_replay_evidence(context, evidence_digest)

        started_ns = time.time_ns()
        payload = _request(
            context, 2, started_ns,
            train_output_replay=1,
            train_output_replay_evidence_sha256=evidence_digest,
            materialize_snapshots=True,
        )
        atomic_json(replay_path, payload)
        runtime_dir = safe_absolute_path(
            pathlib.Path(payload["platform_runtime_dir"]),
            "train output replay runtime directory",
        )
        runtime_dir.relative_to(context.stage_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        if runtime_dir.is_symlink() or runtime_dir.resolve() != runtime_dir:
            raise ValueError("train output replay runtime directory is unsafe")
        return replay_path, payload


def _request_context(payload: dict[str, Any]) -> ActionContext:
    """Reconstruct the immutable action context from a validated envelope."""
    bundle = payload["spec_bundle"]
    return validate_action(
        results_dir=pathlib.Path(payload["results_dir"]),
        image_kind=payload["image_kind"],
        stage_dir=pathlib.Path(payload["stage_dir"]),
        name=payload["name"],
        pass_hf_token=payload["passed_hf_token"],
        fresh_outputs=[pathlib.Path(item) for item in payload["fresh_outputs"]],
        command=[bundle["command"], *bundle["args"]],
        mutate=False,
        require_forwarded_credentials=False,
        verify_virtualenv_runtime=False,
    )


def _load_request(path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    resolved, payload = _load_request_envelope(path)
    context = _request_context(payload)
    if payload.get("unbound_replay") == 1:
        _load_unbound_replay_evidence(
            context, payload.get("unbound_replay_evidence_sha256")
        )
    if payload.get("train_output_replay") == 1:
        _load_train_output_replay_evidence(
            context, payload.get("train_output_replay_evidence_sha256")
        )
    expected_payload = _request(
        context,
        payload["attempt"],
        payload["started_ns"],
        started_at=payload["started_at"],
        job_state_dir=safe_absolute_path(
            pathlib.Path(payload["job_state_dir"]), "action request job_state_dir"
        ),
        dispatch_repair=payload.get("dispatch_repair", 0),
        launcher_repair=payload.get("launcher_repair", 0),
        unbound_replay=payload.get("unbound_replay", 0),
        unbound_replay_evidence_sha256=payload.get(
            "unbound_replay_evidence_sha256"
        ),
        train_output_replay=payload.get("train_output_replay", 0),
        train_output_replay_evidence_sha256=payload.get(
            "train_output_replay_evidence_sha256"
        ),
    )
    if payload != expected_payload:
        raise ValueError(
            "action request fields do not match immutable workflow state and paths"
        )
    return resolved, payload


def _load_job_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = safe_absolute_path(path, "job-record", require_exists=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"job-record is missing or unsafe: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    jsonschema.validate(payload, _artifact_schema("job_record.schema.json"))
    return payload


def _job_record_path(request: dict[str, Any], job: dict[str, Any]) -> pathlib.Path:
    return safe_absolute_path(
        pathlib.Path(request["job_state_dir"]) / "jobs" / f"{job['id']}.json",
        "bound job-record path",
    )


def _job_identity_sha256(job: dict[str, Any]) -> str:
    return _sha256_json({field: job.get(field) for field in JOB_IDENTITY_FIELDS})


def _ownership_mismatches(
    request: dict[str, Any], job: dict[str, Any]
) -> list[str]:
    bundle = request["spec_bundle"]
    mismatches = []
    for field, expected in (
        ("platform", request["platform"]),
        ("image", request["record_image"]),
        ("network_arch", bundle["network_arch"]),
        ("action", bundle["action"]),
        ("upload_excludes", bundle["upload_excludes"]),
    ):
        if job.get(field) != expected:
            mismatches.append(f"{field}={job.get(field)!r}, expected {expected!r}")
    return mismatches


def _canonical_remote_scope(value: Any) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("remote backend scope must be a non-empty canonical string")
    if any(ord(character) < 32 for character in value) or "\\" in value:
        raise ValueError(
            "remote backend scope must not contain control characters or backslashes"
        )
    if value.startswith("/"):
        path = pathlib.PurePosixPath(value)
        if (
            value == "/"
            or path.as_posix() != value
            or ".." in path.parts
            or "." in path.parts
        ):
            raise ValueError("remote backend scope path must be normalized and non-root")
        return value
    parsed = urllib.parse.urlsplit(value)
    remote_path = pathlib.PurePosixPath(parsed.path)
    decoded_path = urllib.parse.unquote(parsed.path)
    decoded_parts = pathlib.PurePosixPath(decoded_path).parts
    if (
        not parsed.scheme
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path
        or remote_path.as_posix() != parsed.path
        or ".." in remote_path.parts
        or "." in remote_path.parts
        or ".." in decoded_parts
        or "." in decoded_parts
        or "\\" in decoded_path
    ):
        raise ValueError(
            "remote backend scope must be an absolute compute path or a "
            "credential-free normalized persistence URI"
        )
    return value


def _validate_job_ownership(
    request_path: pathlib.Path,
    request: dict[str, Any],
    job_path: pathlib.Path,
    job: dict[str, Any],
) -> None:
    expected_path = _job_record_path(request, job)
    if job_path != expected_path:
        raise ValueError(
            f"job-record must be the request-bound state record {expected_path}, got {job_path}"
        )
    mismatches = _ownership_mismatches(request, job)
    if mismatches:
        raise ValueError("job-record does not own this action: " + "; ".join(mismatches))
    request_started = _aware_timestamp(request["started_at"], "action request started_at")
    job_submitted = _aware_timestamp(job["submitted_at"], "job-record submitted_at")
    if job_submitted < request_started:
        raise ValueError("job-record predates the prepared action request")
    if request_path != _request_path_from_payload(request):
        raise ValueError("action request path is inconsistent during job binding")


def attest_staged(args: argparse.Namespace) -> pathlib.Path:
    """Record the platform consumer's post-sync absence check for remote jobs."""
    request_path, request = _load_request(args.request)
    if request["freshness_contract"] != "remote-mirror-with-delete-before-submit":
        raise ValueError("staging attestation is valid only for remote platforms")
    checked = [str(pathlib.Path(raw)) for raw in args.absent_path]
    if checked != request["staging_absent_paths"]:
        raise ValueError(
            "--absent-path must repeat every request staging_absent_paths entry in order"
        )
    backend_scope = _canonical_remote_scope(args.backend_scope)
    mount_pairs = getattr(args, "mount_map", None) or []
    mount_map = []
    if mount_pairs:
        expected_sources = list(dict.fromkeys(
            str(row["source"]) for row in request["mounts"]
        ))
        supplied_sources: list[str] = []
        for pair in mount_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("--mount-map requires LOCAL_SOURCE BACKEND_SOURCE")
            source, backend_source = map(str, pair)
            if source in supplied_sources:
                raise ValueError("--mount-map local sources must be unique")
            backend_path = pathlib.PurePosixPath(backend_source)
            if (
                not source.startswith("/")
                or not backend_source.startswith("/")
                or backend_path == pathlib.PurePosixPath("/")
                or ".." in backend_path.parts
                or any(character in backend_source for character in ("\\", ":", ",", "\x00"))
            ):
                raise ValueError("--mount-map paths must be safe non-root absolute paths")
            supplied_sources.append(source)
            mount_map.append({"source": source, "backend_source": backend_source})
        if supplied_sources != expected_sources:
            raise ValueError(
                "--mount-map must repeat every unique request mount source in first-use order"
            )
    payload = {
        "schema_version": "1",
        "workflow": WORKFLOW,
        "platform": request["platform"],
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "backend_scope": backend_scope,
        "checked_paths_absent": checked,
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if mount_map:
        payload["mount_map"] = mount_map
    payload["receipt_sha256"] = _sha256_json(payload)
    jsonschema.validate(payload, _reference_schema("staging-receipt.schema.json"))
    path = safe_absolute_path(
        pathlib.Path(request["staging_receipt_path"]), "staging receipt"
    )
    atomic_json(path, payload)
    return path


def _validate_staging_receipt(
    request: dict[str, Any], job: dict[str, Any]
) -> tuple[str | None, str]:
    if request["freshness_contract"] != "remote-mirror-with-delete-before-submit":
        expected = str(safe_absolute_path(pathlib.Path(request["stage_dir"]), "action stage"))
        if job.get("results_dir") != expected:
            raise ValueError(
                f"local job-record results_dir must equal the action stage {expected}"
            )
        return None, expected
    path = safe_absolute_path(
        pathlib.Path(str(request["staging_receipt_path"])),
        "staging receipt",
        require_exists=True,
    )
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise ValueError(f"remote action lacks a safe staging receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(receipt, _reference_schema("staging-receipt.schema.json"))
    body = dict(receipt)
    digest = body.pop("receipt_sha256", None)
    actual = _sha256_json(body)
    if digest != actual:
        raise ValueError(f"staging receipt digest mismatch: {path}")
    expected = {
        "platform": request["platform"],
        "request_path": str(_request_path_from_payload(request)),
        "request_sha256": request["request_sha256"],
        "checked_paths_absent": request["staging_absent_paths"],
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"staging receipt {field} does not match the action request")
    checked_at = _aware_timestamp(receipt["checked_at"], "staging receipt checked_at")
    submitted_at = _aware_timestamp(job["submitted_at"], "job-record submitted_at")
    if submitted_at < checked_at:
        raise ValueError("job-record predates the remote output-absence attestation")
    backend_scope = _canonical_remote_scope(receipt.get("backend_scope"))
    if job.get("results_dir") != backend_scope:
        raise ValueError(
            "remote job-record results_dir must equal the attested backend scope"
        )
    return digest, backend_scope


def _binding_payload(
    *,
    request_path: pathlib.Path,
    request: dict[str, Any],
    job_path: pathlib.Path,
    job: dict[str, Any],
    staging_receipt_sha256: str | None,
    results_scope: str,
    bound_at: str,
) -> dict[str, Any]:
    _aware_timestamp(bound_at, "job binding bound_at")
    payload = {
        "schema_version": "1",
        "workflow": WORKFLOW,
        "platform": request["platform"],
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "job_record_path": str(job_path),
        "job_id": job["id"],
        "job_identity_sha256": _job_identity_sha256(job),
        "results_scope": results_scope,
        "staging_receipt_sha256": staging_receipt_sha256,
        "bound_at": bound_at,
    }
    payload["binding_sha256"] = _sha256_json(payload)
    jsonschema.validate(payload, _reference_schema("job-binding.schema.json"))
    return payload


def _load_job_binding(
    request_path: pathlib.Path,
    request: dict[str, Any],
    job_path: pathlib.Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    path = safe_absolute_path(
        pathlib.Path(request["job_binding_path"]),
        "job binding",
        require_exists=True,
    )
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"job binding is missing or unsafe: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, _reference_schema("job-binding.schema.json"))
    body = dict(payload)
    digest = body.pop("binding_sha256", None)
    if digest != _sha256_json(body):
        raise ValueError(f"job binding digest mismatch: {path}")
    staging_digest, results_scope = _validate_staging_receipt(request, job)
    expected = _binding_payload(
        request_path=request_path,
        request=request,
        job_path=job_path,
        job=job,
        staging_receipt_sha256=staging_digest,
        results_scope=results_scope,
        bound_at=payload["bound_at"],
    )
    if payload != expected:
        raise ValueError("job binding does not match the immutable request and job")
    return payload


def load_bound_action_for_submit(
    request_path: pathlib.Path,
    binding_path: pathlib.Path,
    job_path: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Public fail-closed boundary for application-local platform consumers.

    It revalidates the full action/schema/state/cache lineage plus the exact
    request-owned PENDING job and immutable pre-submit binding.  A consumer may
    call a native submit only after this function returns.
    """
    resolved_request, request = _load_request(request_path)
    resolved_job = safe_absolute_path(job_path, "job-record", require_exists=True)
    job = _load_job_record(resolved_job)
    _validate_job_ownership(resolved_request, request, resolved_job, job)
    resolved_binding = safe_absolute_path(
        binding_path, "job binding", require_exists=True,
    )
    if resolved_binding != safe_absolute_path(
        pathlib.Path(request["job_binding_path"]),
        "request-owned job binding",
        require_exists=True,
    ):
        raise ValueError("supplied job binding is not the request-owned binding")
    binding = _load_job_binding(resolved_request, request, resolved_job, job)
    if (
        job.get("backend_ref") is not None
        or job.get("terminal_state") is not None
        or len(job.get("transitions", [])) != 1
        or job["transitions"][0].get("state") != "PENDING"
    ):
        raise ValueError("native submit requires one bound PENDING job")
    return request, binding, job


def _validate_retry_lineage(
    context: ActionContext,
    request_path: pathlib.Path,
    request: dict[str, Any],
    status: dict[str, Any],
) -> None:
    """Require one fully finalized terminal attempt before minting a retry."""
    try:
        jsonschema.validate(
            status, _reference_schema("platform-action-status.schema.json")
        )
    except jsonschema.ValidationError as exc:
        raise ValueError(
            f"existing command status schema violation: {exc.message}"
        ) from exc
    expected = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "kind": "platform_action",
        "name": context.name,
        "attempt": request["attempt"],
        "platform": request["platform"],
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "status": "error",
        "image_kind": request["image_kind"],
        "image": request["workload_image"],
        "command": [
            request["spec_bundle"]["command"],
            *request["spec_bundle"]["args"],
        ],
        "command_sha256": command_sha256(
            [
                request["spec_bundle"]["command"],
                *request["spec_bundle"]["args"],
            ]
        ),
        "passed_hf_token": request["passed_hf_token"],
        "started_at": request["started_at"],
        "started_ns": request["started_ns"],
        "log_path": request["log_path"],
        "fresh_outputs": request["fresh_outputs"],
        "freshness_contract": request["freshness_contract"],
    }
    for field, value in expected.items():
        if status.get(field) != value:
            raise ValueError(
                f"existing command status {field} does not match prior action lineage"
            )
    if (
        not isinstance(status.get("exit_code"), int)
        or isinstance(status["exit_code"], bool)
        or status["exit_code"] == 0
    ):
        raise ValueError("existing error status must record a nonzero exit_code")
    finished_at = _aware_timestamp(
        status.get("finished_at"), "existing status finished_at"
    )
    if finished_at < _aware_timestamp(
        request["started_at"], "prior action request started_at"
    ):
        raise ValueError("existing status finished_at predates the prior action")
    log_path = safe_absolute_path(
        pathlib.Path(request["log_path"]), "prior action log", require_exists=True
    )
    if not log_path.is_file() or log_path.is_symlink() or log_path.stat().st_size == 0:
        raise ValueError("cannot retry without the immutable prior action log")
    matches = _matching_job_records(request_path, request)
    if len(matches) != 1:
        raise ValueError(
            "existing error status must have exactly one request-owned job-record"
        )
    job_path, job = matches[0]
    binding = _load_job_binding(request_path, request, job_path, job)
    if job.get("terminal_state") not in {"COMPLETE", "ERROR", "CANCELED"}:
        raise ValueError("cannot retry while the prior native job is nonterminal")
    _validate_terminal_job(job)
    if not isinstance(job.get("backend_ref"), str) or not job["backend_ref"].strip():
        raise ValueError("cannot retry without the prior native backend reference")
    for field, expected_value in (
        ("job_id", job["id"]),
        ("backend_ref", job["backend_ref"]),
        ("storage_tier", job["storage_tier"]),
        ("backend_state", job["terminal_state"]),
        ("job_binding_sha256", binding["binding_sha256"]),
        ("results_scope", binding["results_scope"]),
        ("staging_receipt_sha256", binding["staging_receipt_sha256"]),
    ):
        if status.get(field) != expected_value:
            raise ValueError(
                f"existing command status {field} does not match prior job evidence"
            )
    expected_exit = (
        status["backend_exit_code"] if status["backend_exit_code"] else 3
    )
    if status["exit_code"] != expected_exit or status["exit_code"] == 0:
        raise ValueError(
            "existing command status exit_code is inconsistent with terminal evidence"
        )


def bind_job(args: argparse.Namespace) -> pathlib.Path:
    """Bind one freshly opened job-record to a request before native submit."""
    request_path, request = _load_launched_request(args.request)
    with _request_lock(request):
        job_path = safe_absolute_path(args.job_record, "job-record", require_exists=True)
        job = _load_job_record(job_path)
        _validate_job_ownership(request_path, request, job_path, job)
        if (
            job.get("backend_ref") is not None
            or job.get("terminal_state") is not None
            or len(job.get("transitions", [])) != 1
            or job["transitions"][0].get("state") != "PENDING"
        ):
            raise ValueError("job binding must occur after open and before native submit")
        staging_digest, results_scope = _validate_staging_receipt(request, job)
        path = safe_absolute_path(pathlib.Path(request["job_binding_path"]), "job binding")
        if path.exists():
            # Binding is immutable and idempotent for the same record.  A
            # second concurrently opened record fails validation here and can
            # never overwrite the first winner's binding.
            _load_job_binding(request_path, request, job_path, job)
            return path
        payload = _binding_payload(
            request_path=request_path,
            request=request,
            job_path=job_path,
            job=job,
            staging_receipt_sha256=staging_digest,
            results_scope=results_scope,
            bound_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        )
        atomic_json(path, payload)
    return path


def _is_abandoned_presubmit_job(job: dict[str, Any]) -> bool:
    """Return true only for a typed, never-submitted controller cancellation."""
    transitions = job.get("transitions")
    return bool(
        job.get("terminal_state") == "CANCELED"
        and job.get("backend_ref") is None
        and isinstance(transitions, list)
        and len(transitions) == 2
        and [item.get("state") for item in transitions if isinstance(item, dict)]
        == ["PENDING", "CANCELED"]
        and job.get("terminal_write_by") == "agent"
    )


def _validate_abandoned_presubmit_job(
    request_path: pathlib.Path,
    request: dict[str, Any],
    job_path: pathlib.Path,
    job: dict[str, Any],
) -> None:
    """Prove a typed cancellation never acquired a binding or backend object."""

    expected_path = _job_record_path(request, job)
    if job_path != expected_path:
        raise ValueError(
            f"abandoned job-record must be the request state record {expected_path}"
        )
    if request_path != _request_path_from_payload(request):
        raise ValueError("action request path is inconsistent during reconciliation")
    request_started = _aware_timestamp(request["started_at"], "action request started_at")
    job_submitted = _aware_timestamp(job["submitted_at"], "job-record submitted_at")
    if job_submitted < request_started:
        raise ValueError("abandoned job-record predates the prepared action request")
    binding_path = safe_absolute_path(
        pathlib.Path(request["job_binding_path"]), "job binding"
    )
    if not binding_path.exists():
        return
    if not binding_path.is_file() or binding_path.is_symlink():
        raise ValueError("existing job binding is missing or unsafe during reconciliation")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if not isinstance(binding, dict):
        raise ValueError("existing job binding root must be an object")
    if binding.get("job_record_path") == str(job_path):
        raise ValueError("a bound job-record cannot be treated as abandoned pre-submit")


BOUND_PRESUBMIT_RECOVERY_LIMIT = 3


def _bound_presubmit_recovery_path(
    request: dict[str, Any], job_id: str,
) -> pathlib.Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id) is None:
        raise ValueError("bound pre-submit recovery job ID is unsafe")
    return safe_absolute_path(
        pathlib.Path(request["stage_dir"])
        / f"{request['name']}.{job_id}.bound-presubmit-recovery.evidence.json",
        "bound pre-submit recovery evidence",
    )


def _query_absent_slurm_job(
    job_id: str, *, login: str | None = None,
) -> dict[str, str]:
    """Prove that an exact deterministic SLURM name has no native allocation."""

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id) is None:
        raise ValueError("bound pre-submit recovery job ID is not a safe SLURM name")
    commands = {
        "squeue_stdout_sha256": [
            "squeue", "-h", "--name", job_id, "-o", "%i|%j",
        ],
        "sacct_stdout_sha256": [
            "sacct", "-X", "-n", "--name", job_id,
            "-o", "JobIDRaw,JobName", "-P",
        ],
    }
    if login is not None:
        if re.fullmatch(r"[A-Za-z0-9_.@-]+", login) is None:
            raise ValueError("SLURM recovery login contains unsupported characters")
        commands = {
            field: [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                login, " ".join(shlex.quote(part) for part in command),
            ]
            for field, command in commands.items()
        }
    evidence: dict[str, str] = {}
    for field, command in commands.items():
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError("SLURM native-absence query failed") from exc
        if result.returncode != 0:
            raise ValueError("SLURM native-absence query returned a nonzero status")
        if result.stdout.strip():
            raise ValueError("bound pre-submit recovery found an existing native SLURM job")
        evidence[field] = hashlib.sha256(result.stdout.strip().encode()).hexdigest()
    return evidence


def _load_bound_presubmit_recovery(
    request_path: pathlib.Path,
    request: dict[str, Any],
    job_path: pathlib.Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    path = _bound_presubmit_recovery_path(request, job["id"])
    if not path.is_file() or path.is_symlink():
        raise ValueError("bound pre-submit recovery evidence is missing or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bound pre-submit recovery evidence root must be an object")
    body = dict(payload)
    digest = body.pop("evidence_sha256", None)
    if digest != _sha256_json(body):
        raise ValueError("bound pre-submit recovery evidence digest mismatch")
    archive = safe_absolute_path(
        pathlib.Path(payload.get("binding_archive", "")),
        "archived bound pre-submit binding",
        require_exists=True,
    )
    staging = safe_absolute_path(
        pathlib.Path(request["staging_receipt_path"]),
        "staging receipt",
        require_exists=True,
    )
    expected = {
        "schema_version": "1",
        "workflow": WORKFLOW,
        "kind": "slurm_bound_presubmit_recovery",
        "platform": "slurm",
        "name": request["name"],
        "action_id": request["spec_bundle"]["action"],
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "job_record_path": str(job_path),
        "job_record_sha256": _sha256_file(job_path),
        "job_id": job["id"],
        "binding_original": request["job_binding_path"],
        "binding_archive": str(archive),
        "binding_sha256": _sha256_file(archive),
        "staging_receipt_sha256": _sha256_file(staging),
        "squeue_stdout_sha256": hashlib.sha256(b"").hexdigest(),
        "sacct_stdout_sha256": hashlib.sha256(b"").hexdigest(),
    }
    scheduler_login = payload.get("scheduler_login")
    if scheduler_login is not None and not isinstance(scheduler_login, str):
        raise ValueError("bound pre-submit recovery scheduler login is invalid")
    if scheduler_login is not None:
        expected["scheduler_login"] = scheduler_login
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"bound pre-submit recovery {field} is invalid")
    if set(payload) != {*expected, "recorded_at", "evidence_sha256"}:
        raise ValueError("bound pre-submit recovery evidence has extra fields")
    _aware_timestamp(payload.get("recorded_at"), "bound pre-submit recovery recorded_at")
    _query_absent_slurm_job(job["id"], login=scheduler_login)
    return payload


def recover_bound_presubmit(args: argparse.Namespace) -> pathlib.Path:
    """Archive one canceled binding after proving that native submit never occurred."""

    if not args.confirm:
        raise ValueError("bound pre-submit recovery requires --confirm")
    request_path, request = _load_request_envelope(args.request)
    context = _request_context(request)
    if (
        request["platform"] != "slurm"
        or request.get("attempt") != 1
        or any(request.get(key) for key in (
            "dispatch_repair", "launcher_repair", "unbound_replay",
            "train_output_replay",
        ))
    ):
        raise ValueError(
            "bound pre-submit recovery is restricted to original SLURM actions"
        )
    job_path = safe_absolute_path(args.job_record, "job-record", require_exists=True)
    job = _load_job_record(job_path)
    _validate_job_ownership(request_path, request, job_path, job)
    if not _is_abandoned_presubmit_job(job):
        raise ValueError("bound pre-submit recovery requires an agent-canceled no-backend record")
    jobs_dir = safe_absolute_path(
        pathlib.Path(request["job_state_dir"]) / "jobs", "job-record directory",
        require_exists=True,
    )
    abandoned = 0
    for candidate in sorted(jobs_dir.glob("*.json")):
        candidate_job = _load_job_record(candidate)
        if (
            candidate_job.get("action") == request["spec_bundle"]["action"]
            and _is_abandoned_presubmit_job(candidate_job)
        ):
            abandoned += 1
    if abandoned > BOUND_PRESUBMIT_RECOVERY_LIMIT:
        raise ValueError(
            "bound pre-submit recovery budget exhausted "
            f"({BOUND_PRESUBMIT_RECOVERY_LIMIT})"
        )
    if context.status_path.exists() or pathlib.Path(request["log_path"]).exists():
        raise ValueError("bound pre-submit recovery is forbidden after status or log evidence exists")
    if any(path.exists() for path in context.fresh_outputs):
        raise ValueError("bound pre-submit recovery is forbidden after action output exists")
    binding_path = safe_absolute_path(
        pathlib.Path(request["job_binding_path"]), "job binding",
    )
    archive = safe_absolute_path(
        context.stage_dir / ".tao-runtime"
        / f"{context.name}.bound-presubmit-recovery" / job["id"] / "binding.json",
        "archived bound pre-submit binding",
    )
    evidence_path = _bound_presubmit_recovery_path(request, job["id"])
    with _request_lock(request):
        if evidence_path.exists():
            evidence = _load_bound_presubmit_recovery(
                request_path, request, job_path, job,
            )
            if binding_path.exists():
                raise ValueError("recovered bound pre-submit binding reappeared")
            return evidence_path
        if binding_path.exists() and archive.exists():
            raise ValueError("live and archived bound pre-submit bindings both exist")
        source = binding_path if binding_path.exists() else archive
        if not source.is_file() or source.is_symlink():
            raise ValueError("bound pre-submit recovery requires one safe binding")
        binding = json.loads(source.read_text(encoding="utf-8"))
        jsonschema.validate(binding, _reference_schema("job-binding.schema.json"))
        unsigned = dict(binding)
        binding_digest = unsigned.pop("binding_sha256", None)
        if binding_digest != _sha256_json(unsigned):
            raise ValueError("bound pre-submit binding digest mismatch")
        expected_binding = _binding_payload(
            request_path=request_path,
            request=request,
            job_path=job_path,
            job=job,
            staging_receipt_sha256=binding["staging_receipt_sha256"],
            results_scope=binding["results_scope"],
            bound_at=binding["bound_at"],
        )
        if binding != expected_binding:
            raise ValueError("bound pre-submit binding does not match the request and job")
        staging_path = safe_absolute_path(
            pathlib.Path(request["staging_receipt_path"]),
            "staging receipt",
            require_exists=True,
        )
        if not staging_path.is_file() or staging_path.is_symlink():
            raise ValueError("bound pre-submit staging receipt is unsafe")
        staging = json.loads(staging_path.read_text(encoding="utf-8"))
        jsonschema.validate(staging, _reference_schema("staging-receipt.schema.json"))
        staging_body = dict(staging)
        staging_digest = staging_body.pop("receipt_sha256", None)
        if (
            staging_digest != _sha256_json(staging_body)
            or staging.get("request_path") != str(request_path)
            or staging.get("request_sha256") != request["request_sha256"]
            or staging.get("checked_paths_absent") != request["staging_absent_paths"]
            or _canonical_remote_scope(staging.get("backend_scope"))
                != binding["results_scope"]
        ):
            raise ValueError("bound pre-submit staging receipt is invalid")
        scheduler_login = getattr(args, "login", None)
        scheduler_evidence = _query_absent_slurm_job(
            job["id"], login=scheduler_login,
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        if binding_path.exists():
            os.replace(binding_path, archive)
        payload = {
            "schema_version": "1", "workflow": WORKFLOW,
            "kind": "slurm_bound_presubmit_recovery", "platform": "slurm",
            "name": request["name"], "action_id": request["spec_bundle"]["action"],
            "request_path": str(request_path), "request_sha256": request["request_sha256"],
            "job_record_path": str(job_path), "job_record_sha256": _sha256_file(job_path),
            "job_id": job["id"], "binding_original": str(binding_path),
            "binding_archive": str(archive), "binding_sha256": _sha256_file(archive),
            "staging_receipt_sha256": _sha256_file(staging_path),
            **scheduler_evidence,
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        if scheduler_login is not None:
            payload["scheduler_login"] = scheduler_login
        payload["evidence_sha256"] = _sha256_json(payload)
        atomic_json(evidence_path, payload)
        _load_bound_presubmit_recovery(request_path, request, job_path, job)
    reconciled = reconcile_request(argparse.Namespace(request=request_path))
    if reconciled["state"] != "NO_JOB_RECORD":
        raise ValueError("bound pre-submit recovery did not restore a safe open boundary")
    return evidence_path


def rebind_airflow_state(args: argparse.Namespace) -> pathlib.Path:
    """Rebind one never-submitted SLURM request to Airflow shared state.

    This recovery is intentionally narrower than an action retry.  It changes
    only ``job_state_dir`` after proving that no native backend handle, live
    job record, platform status, log, binding, or workload output exists.  The
    prior immutable request and any staging/runtime evidence are archived so
    the correction remains auditable.
    """

    if not args.confirm:
        raise ValueError("Airflow state rebind requires --confirm")
    request_path, request = _load_request_envelope(args.request)
    context = _request_context(request)
    if (
        context.platform != "slurm"
        or context.config.get("orchestrator") != "airflow"
        or request.get("attempt") != 1
        or any(request.get(key) for key in (
            "dispatch_repair", "launcher_repair", "unbound_replay",
            "train_output_replay",
        ))
    ):
        raise ValueError(
            "Airflow state rebind is restricted to original Airflow-orchestrated "
            "SLURM actions"
        )
    target_state_dir = _job_state_dir()
    prior_state_dir = safe_absolute_path(
        pathlib.Path(request["job_state_dir"]), "prior action job_state_dir"
    )
    if target_state_dir == prior_state_dir:
        raise ValueError("action request already uses the selected Airflow state directory")
    if not target_state_dir.is_dir() or target_state_dir.is_symlink():
        raise ValueError("selected Airflow state directory is missing or unsafe")

    archive_root = safe_absolute_path(
        context.stage_dir / ".tao-runtime"
        / f"{context.name}.airflow-state-rebind-{request['request_sha256'][:12]}",
        "Airflow state rebind archive",
    )
    archived_request = archive_root / "prior.action.json"
    archived_staging = archive_root / "prior.staged.json"
    archived_runtime = archive_root / "prior-runtime"
    evidence_path = archive_root / "evidence.json"

    with _request_lock(request):
        # Re-read under the action lock so another controller cannot bind or
        # launch between the proof and the atomic request replacement.
        request_path, request = _load_request_envelope(request_path)
        if pathlib.Path(request["job_state_dir"]) != prior_state_dir:
            raise ValueError("action request state directory changed during recovery")
        if context.status_path.exists() or pathlib.Path(request["log_path"]).exists():
            raise ValueError("Airflow state rebind is forbidden after status or log evidence")
        if pathlib.Path(request["job_binding_path"]).exists():
            raise ValueError(
                "Airflow state rebind requires bound pre-submit recovery first"
            )
        if any(path.exists() for path in context.fresh_outputs):
            raise ValueError("Airflow state rebind is forbidden after action output exists")
        if _matching_job_records(request_path, request):
            raise ValueError("Airflow state rebind requires NO_JOB_RECORD reconciliation")
        if evidence_path.exists():
            raise ValueError("Airflow state rebind evidence already exists")
        if archive_root.exists():
            raise ValueError("partial Airflow state rebind archive already exists")

        staging_path = pathlib.Path(request["staging_receipt_path"])
        if staging_path.exists():
            staging = json.loads(staging_path.read_text(encoding="utf-8"))
            jsonschema.validate(staging, _reference_schema("staging-receipt.schema.json"))
            staging_body = dict(staging)
            staging_digest = staging_body.pop("receipt_sha256", None)
            if (
                staging_digest != _sha256_json(staging_body)
                or staging.get("request_path") != str(request_path)
                or staging.get("request_sha256") != request["request_sha256"]
            ):
                raise ValueError("Airflow state rebind staging receipt is invalid")

        archive_root.mkdir(parents=True)
        shutil.copy2(request_path, archived_request)
        if _sha256_file(archived_request) != _sha256_file(request_path):
            raise ValueError("archived Airflow state rebind request digest mismatch")
        if staging_path.exists():
            os.replace(staging_path, archived_staging)
        runtime_dir = pathlib.Path(request["platform_runtime_dir"])
        if runtime_dir.exists():
            if not runtime_dir.is_dir() or runtime_dir.is_symlink():
                raise ValueError("prior action runtime directory is unsafe")
            os.replace(runtime_dir, archived_runtime)

        replacement = _request(
            context,
            1,
            time.time_ns(),
            job_state_dir=target_state_dir,
            materialize_snapshots=True,
        )
        atomic_json(request_path, replacement)
        _load_request(request_path)
        evidence = {
            "schema_version": "1",
            "workflow": WORKFLOW,
            "kind": "airflow_state_rebind",
            "platform": "slurm",
            "name": context.name,
            "prior_request_path": str(archived_request),
            "prior_request_sha256": request["request_sha256"],
            "prior_request_file_sha256": _sha256_file(archived_request),
            "prior_job_state_dir": str(prior_state_dir),
            "replacement_request_path": str(request_path),
            "replacement_request_sha256": replacement["request_sha256"],
            "replacement_request_file_sha256": _sha256_file(request_path),
            "replacement_job_state_dir": str(target_state_dir),
            "archived_staging": str(archived_staging) if archived_staging.exists() else None,
            "archived_runtime": str(archived_runtime) if archived_runtime.exists() else None,
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        evidence["evidence_sha256"] = _sha256_json(evidence)
        atomic_json(evidence_path, evidence)
    return evidence_path


def _matching_job_records(
    request_path: pathlib.Path, request: dict[str, Any]
) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    jobs = safe_absolute_path(
        pathlib.Path(request["job_state_dir"]) / "jobs", "job-record directory"
    )
    if not jobs.exists():
        return []
    if not jobs.is_dir():
        raise ValueError(f"job-record directory is not a directory: {jobs}")
    matches: list[tuple[pathlib.Path, dict[str, Any]]] = []
    action_id = request["spec_bundle"]["action"]
    for path in sorted(jobs.glob("*.json")):
        try:
            job = _load_job_record(path)
        except (OSError, ValueError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
            raise ValueError(f"cannot reconcile malformed job-record {path}: {exc}") from exc
        if job.get("action") != action_id:
            continue
        # A controller can be interrupted after opening a record but before
        # binding/submission, or two controllers can race at that boundary.
        # The job-record writer's explicit PENDING -> CANCELED transition is
        # the typed abandonment receipt for such a record.  Ignore only that
        # exact, never-submitted and never-bound shape. Its rejected pre-submit
        # fields need not own the request because ownership failure is why the
        # record was canceled. Launched, bound, malformed, stale, and
        # nonterminal duplicates continue to fail closed below.
        if _is_abandoned_presubmit_job(job):
            _validate_abandoned_presubmit_job(request_path, request, path, job)
            continue
        _validate_job_ownership(request_path, request, path, job)
        matches.append((path, job))
    if len(matches) > 1:
        raise ValueError("multiple job-records claim the same prepared action")
    return matches


def _load_launched_request(
    path: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Load an immutable opened request across a proven plugin-cache refresh."""
    resolved, request = _load_request_envelope(path)
    try:
        return _load_request(resolved)
    except ValueError as exc:
        if str(exc) != "action request fields do not match immutable workflow state and paths":
            raise
    _cache_relocation_expected(
        _request_context(request), request, allow_regressed_normal_action_id=True
    )
    if len(_matching_job_records(resolved, request)) != 1:
        raise ValueError(
            "cannot accept a refreshed launched request without exactly one matching job-record"
        )
    return resolved, request


def reconcile_request(args: argparse.Namespace) -> dict[str, Any]:
    request_path, request = _load_launched_request(args.request)
    matches = _matching_job_records(request_path, request)
    binding_path = pathlib.Path(request["job_binding_path"])
    if not matches:
        if binding_path.exists():
            raise ValueError("job binding exists without its request-owned job-record")
        return {
            "state": "NO_JOB_RECORD",
            "request": str(request_path),
            "action": request["spec_bundle"]["action"],
        }
    job_path, job = matches[0]
    if binding_path.exists():
        _load_job_binding(request_path, request, job_path, job)
        state = (
            "BOUND"
            if job.get("backend_ref")
            else "BOUND_BACKEND_RECONCILIATION_REQUIRED"
        )
    else:
        state = "JOB_OPENED_UNBOUND"
    return {
        "state": state,
        "request": str(request_path),
        "action": request["spec_bundle"]["action"],
        "job_id": job["id"],
        "job_record": str(job_path),
        "backend_ref_present": bool(job.get("backend_ref")),
        "terminal_state": job.get("terminal_state"),
    }


def _validate_terminal_job(job: dict[str, Any]) -> None:
    terminal = job.get("terminal_state")
    transitions = job.get("transitions")
    if not isinstance(transitions, list) or len(transitions) < 3:
        raise ValueError(
            "terminal job-record must preserve PENDING, RUNNING, and terminal transitions"
        )
    states = [item.get("state") if isinstance(item, dict) else None for item in transitions]
    if states[0] != "PENDING" or states[-1] != terminal or "RUNNING" not in states[1:-1]:
        raise ValueError(
            "terminal job-record transition lineage must be PENDING -> RUNNING -> terminal"
        )
    terminal_states = {"COMPLETE", "ERROR", "CANCELED"}
    if any(state in terminal_states for state in states[:-1]):
        raise ValueError("job-record has a transition after an earlier terminal state")
    timestamps = [
        _aware_timestamp(item.get("ts"), "job-record transition timestamp")
        for item in transitions
    ]
    if timestamps != sorted(timestamps):
        raise ValueError("job-record transitions are not timestamp ordered")


def _preallocation_cancel_receipt(
    log_path: pathlib.Path,
    request: dict[str, Any],
    job: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate the packaged SLURM accounting receipt when the job never ran."""

    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("kind") != "slurm_preallocation_cancel":
        return None
    body = dict(payload)
    digest = body.pop("receipt_sha256", None)
    if digest != _sha256_json(body):
        raise ValueError("SLURM preallocation cancellation receipt digest mismatch")
    expected = {
        "schema_version": "1",
        "workflow": WORKFLOW,
        "kind": "slurm_preallocation_cancel",
        "platform": "slurm",
        "request_sha256": request["request_sha256"],
        "job_binding_sha256": binding["binding_sha256"],
        "job_id": job["id"],
        "backend_ref": job["backend_ref"],
        "native_state": "CANCELLED",
        "native_start": "None",
        "native_elapsed": "00:00:00",
        "native_exit_code": "0:0",
        "native_nodelist": "None assigned",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"SLURM preallocation cancellation receipt {field} is invalid"
            )
    _aware_timestamp(payload.get("recorded_at"), "cancellation receipt recorded_at")
    if set(payload) != {*expected, "recorded_at", "receipt_sha256"}:
        raise ValueError("SLURM preallocation cancellation receipt has extra fields")
    return payload


def capture_preallocation_cancel(args: argparse.Namespace) -> pathlib.Path:
    """Capture native proof for one bound SLURM job canceled before allocation."""

    request_path, request = _load_launched_request(args.request)
    if request.get("platform") != "slurm":
        raise ValueError("preallocation cancellation capture is SLURM-only")
    job_path = safe_absolute_path(args.job_record, "job-record", require_exists=True)
    job = _load_job_record(job_path)
    _validate_job_ownership(request_path, request, job_path, job)
    binding = _load_job_binding(request_path, request, job_path, job)
    if job.get("terminal_state") != "CANCELED" or job.get("terminal_write_by") != "agent":
        raise ValueError("preallocation cancellation requires an agent-owned CANCELED job")
    _validate_terminal_job(job)
    backend_ref = job.get("backend_ref")
    if not isinstance(backend_ref, str) or not backend_ref.isdigit():
        raise ValueError("preallocation cancellation requires a numeric SLURM backend ref")
    if any(pathlib.Path(raw).exists() for raw in request["fresh_outputs"]):
        raise ValueError("preallocation cancellation is forbidden after action output exists")
    log_path = safe_absolute_path(pathlib.Path(request["log_path"]), "action log")
    if log_path.exists():
        if not log_path.is_file() or log_path.is_symlink() or log_path.stat().st_size == 0:
            raise ValueError("existing action log is unsafe during cancellation capture")
        receipt = _preallocation_cancel_receipt(log_path, request, job, binding)
        if receipt is None:
            raise ValueError("native action log already exists; cancellation capture is forbidden")
        return log_path
    result = subprocess.run(
        [
            "sacct", "-j", backend_ref, "-X", "-n", "-P", "-o",
            "JobIDRaw,State,Start,Elapsed,ExitCode,NodeList",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("SLURM accounting query failed during cancellation capture")
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) == 6 and fields[0] == backend_ref:
            rows.append(fields)
    if len(rows) != 1:
        raise ValueError("SLURM accounting lacks one exact parent cancellation row")
    _, state, start, elapsed, exit_code, nodelist = rows[0]
    normalized_state = state.split(None, 1)[0].rstrip("+")
    normalized_start = "None" if start in {"", "Unknown", "None", "N/A"} else start
    normalized_nodelist = (
        "None assigned"
        if nodelist in {"", "Unknown", "None", "None assigned", "N/A"}
        else nodelist
    )
    if (
        normalized_state != "CANCELLED"
        or normalized_start != "None"
        or elapsed != "00:00:00"
        or exit_code != "0:0"
        or normalized_nodelist != "None assigned"
    ):
        raise ValueError("SLURM job was not canceled before allocation/runtime")
    payload = {
        "schema_version": "1",
        "workflow": WORKFLOW,
        "kind": "slurm_preallocation_cancel",
        "platform": "slurm",
        "request_sha256": request["request_sha256"],
        "job_binding_sha256": binding["binding_sha256"],
        "job_id": job["id"],
        "backend_ref": backend_ref,
        "native_state": normalized_state,
        "native_start": normalized_start,
        "native_elapsed": elapsed,
        "native_exit_code": exit_code,
        "native_nodelist": normalized_nodelist,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    atomic_json(log_path, payload)
    _preallocation_cancel_receipt(log_path, request, job, binding)
    return log_path


def finalize(args: argparse.Namespace) -> tuple[pathlib.Path, int]:
    request_path, request = _load_launched_request(args.request)
    job_path = safe_absolute_path(args.job_record, "job-record", require_exists=True)
    job = _load_job_record(job_path)
    _validate_job_ownership(request_path, request, job_path, job)
    binding = _load_job_binding(request_path, request, job_path, job)
    bundle = request["spec_bundle"]
    if not isinstance(job.get("backend_ref"), str) or not job["backend_ref"].strip():
        raise ValueError("job-record lacks the native backend reference")
    staging_receipt_sha256 = binding["staging_receipt_sha256"]
    terminal = job.get("terminal_state")
    if terminal not in {"COMPLETE", "ERROR", "CANCELED"}:
        raise ValueError("job-record is not terminal; poll the selected platform first")
    _validate_terminal_job(job)
    log_path = pathlib.Path(str(request["log_path"]))
    if (
        not log_path.is_absolute()
        or not log_path.is_file()
        or log_path.stat().st_size == 0
        or log_path.is_symlink()
        or log_path.resolve() != log_path
    ):
        raise ValueError(
            f"platform logs must be captured at the immutable action log path: {log_path}"
        )
    stage_dir = pathlib.Path(str(request["stage_dir"])).resolve()
    log_path.relative_to(stage_dir)
    secret_linter = pathlib.Path(__file__).resolve().parents[4] / "scripts" / "redact_secrets.py"
    linted = subprocess.run(
        [sys.executable, str(secret_linter), "lint", str(log_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if linted.returncode != 0:
        raise ValueError(
            "platform log failed credential lint; redact the captured log before finalization"
        )
    cancel_receipt = (
        _preallocation_cancel_receipt(log_path, request, job, binding)
        if terminal == "CANCELED"
        else None
    )
    started_ns = request.get("started_ns")
    if not isinstance(started_ns, int) or isinstance(started_ns, bool) or started_ns < 1:
        raise ValueError("action request started_ns is invalid")

    artifact_error = None
    if terminal == "COMPLETE":
        if args.native_exit_code != 0:
            raise ValueError("a COMPLETE backend must report --native-exit-code 0")
        for raw in request["fresh_outputs"]:
            output = pathlib.Path(raw)
            if (
                output.is_symlink()
                or not output.is_file()
                or output.stat().st_size == 0
            ):
                artifact_error = f"fresh output is missing, empty, or a symlink: {output}"
                break
            if (
                request["freshness_contract"] == "local-mtime-after-prepare"
                and output.stat().st_mtime_ns < started_ns
            ):
                artifact_error = f"fresh output predates this action: {output}"
                break
    history_resume = False
    if terminal == "COMPLETE" and request["name"] == "history_select":
        host_status_path = stage_dir / "history-select.host.status.json"
        if host_status_path.is_file() and not host_status_path.is_symlink():
            try:
                host_status = json.loads(host_status_path.read_text())
            except (OSError, json.JSONDecodeError):
                host_status = None
            if (
                isinstance(host_status, dict)
                and host_status.get("name") == "history-select"
                and host_status.get("status") == "ok"
                and host_status.get("exit_code") == 0
                and isinstance(host_status.get("resume"), bool)
            ):
                history_resume = host_status["resume"]
            else:
                artifact_error = "history-select host status is missing valid resume evidence"
        else:
            artifact_error = "history-select host status is missing valid resume evidence"
    backend_exit_code = args.native_exit_code
    success = terminal == "COMPLETE" and backend_exit_code == 0 and artifact_error is None
    payload = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "kind": "platform_action",
        "name": request["name"],
        "attempt": request["attempt"],
        "platform": request["platform"],
        "job_id": job["id"],
        "backend_ref": job["backend_ref"],
        "storage_tier": job["storage_tier"],
        "backend_state": terminal,
        "backend_exit_code": backend_exit_code,
        "image_kind": request["image_kind"],
        "image": request["workload_image"],
        "command": [bundle["command"], *bundle["args"]],
        "command_sha256": command_sha256([bundle["command"], *bundle["args"]]),
        "passed_hf_token": request["passed_hf_token"],
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "job_binding_sha256": binding["binding_sha256"],
        "results_scope": binding["results_scope"],
        "staging_receipt_sha256": staging_receipt_sha256,
        "started_at": request["started_at"],
        "started_ns": started_ns,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "ok" if success else "error",
        "exit_code": 0 if success else (backend_exit_code if backend_exit_code else 3),
        "log_path": str(log_path),
        "fresh_outputs": request["fresh_outputs"],
        "freshness_contract": request["freshness_contract"],
        "artifact_error": artifact_error,
    }
    if request.get("dispatch_repair") == 1:
        payload["dispatch_repair"] = 1
    if request.get("launcher_repair") == 1:
        payload["launcher_repair"] = 1
    if request.get("unbound_replay") == 1:
        payload["unbound_replay"] = 1
        payload["unbound_replay_evidence_sha256"] = request[
            "unbound_replay_evidence_sha256"
        ]
    if request.get("train_output_replay") == 1:
        payload["train_output_replay"] = 1
        payload["train_output_replay_evidence_sha256"] = request[
            "train_output_replay_evidence_sha256"
        ]
    if cancel_receipt is not None:
        payload["preallocation_cancel_receipt_sha256"] = cancel_receipt[
            "receipt_sha256"
        ]
    if request["name"] == "history_select":
        payload["resume"] = history_resume
    jsonschema.validate(
        payload, _reference_schema("platform-action-status.schema.json")
    )
    status_path = pathlib.Path(str(request["status_path"]))
    atomic_json(status_path, payload)
    return status_path, 0 if success else 3


def _common_action_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--image", required=True, choices=("pyt", "ds"))
    parser.add_argument("--stage-dir", required=True, type=pathlib.Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pass-hf-token", action="store_true")
    parser.add_argument(
        "--fresh-output", action="append", default=[], type=pathlib.Path
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    prepare_parser = sub.add_parser("prepare")
    _common_action_args(prepare_parser)
    repair_parser = sub.add_parser("dispatch-repair")
    _common_action_args(repair_parser)
    launcher_repair_parser = sub.add_parser("launcher-repair")
    _common_action_args(launcher_repair_parser)
    unbound_replay_parser = sub.add_parser("unbound-replay")
    _common_action_args(unbound_replay_parser)
    train_output_replay_parser = sub.add_parser("train-output-replay")
    train_output_replay_parser.add_argument(
        "--recovery-evidence", required=True, type=pathlib.Path
    )
    _common_action_args(train_output_replay_parser)
    cache_parser = sub.add_parser("cache-preflight")
    cache_parser.add_argument("--cache-dir", required=True, type=pathlib.Path)
    staged_parser = sub.add_parser("attest-staged")
    staged_parser.add_argument("--request", required=True, type=pathlib.Path)
    staged_parser.add_argument("--backend-scope", required=True)
    staged_parser.add_argument(
        "--absent-path", action="append", default=[], required=True
    )
    staged_parser.add_argument(
        "--mount-map", action="append", nargs=2, metavar=("LOCAL_SOURCE", "BACKEND_SOURCE")
    )
    reconcile_parser = sub.add_parser("reconcile")
    reconcile_parser.add_argument("--request", required=True, type=pathlib.Path)
    bind_parser = sub.add_parser("bind-job")
    bind_parser.add_argument("--request", required=True, type=pathlib.Path)
    bind_parser.add_argument("--job-record", required=True, type=pathlib.Path)
    bound_recovery_parser = sub.add_parser("recover-bound-presubmit")
    bound_recovery_parser.add_argument("--request", required=True, type=pathlib.Path)
    bound_recovery_parser.add_argument("--job-record", required=True, type=pathlib.Path)
    bound_recovery_parser.add_argument("--login")
    bound_recovery_parser.add_argument("--confirm", action="store_true")
    state_rebind_parser = sub.add_parser("rebind-airflow-state")
    state_rebind_parser.add_argument("--request", required=True, type=pathlib.Path)
    state_rebind_parser.add_argument("--confirm", action="store_true")
    cancel_parser = sub.add_parser("capture-preallocation-cancel")
    cancel_parser.add_argument("--request", required=True, type=pathlib.Path)
    cancel_parser.add_argument("--job-record", required=True, type=pathlib.Path)
    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--request", required=True, type=pathlib.Path)
    finalize_parser.add_argument("--job-record", required=True, type=pathlib.Path)
    finalize_parser.add_argument("--native-exit-code", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verb in {
        "prepare", "dispatch-repair", "launcher-repair", "unbound-replay",
        "train-output-replay",
    }:
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            print("run_deft_action: command after -- is required", file=sys.stderr)
            return 2
    try:
        if args.verb == "cache-preflight":
            print(json.dumps(tao_cache_preflight(args.cache_dir), sort_keys=True))
            return 0
        if args.verb in {
            "prepare", "dispatch-repair", "launcher-repair", "unbound-replay",
            "train-output-replay",
        }:
            path, payload = (
                prepare(args)
                if args.verb == "prepare"
                else (
                    dispatch_repair(args)
                    if args.verb == "dispatch-repair"
                    else (
                        launcher_repair(args)
                        if args.verb == "launcher-repair"
                        else (
                            unbound_replay(args)
                            if args.verb == "unbound-replay"
                            else train_output_replay(args)
                        )
                    )
                )
            )
            reconciliation = reconcile_request(
                argparse.Namespace(request=path)
            )
            print(
                json.dumps(
                    {
                        "request": str(path),
                        "platform": payload["platform"],
                        "action": payload["spec_bundle"]["action"],
                        "attempt": payload["attempt"],
                        "dispatch_repair": payload.get("dispatch_repair", 0),
                        "launcher_repair": payload.get("launcher_repair", 0),
                        "unbound_replay": payload.get("unbound_replay", 0),
                        "train_output_replay": payload.get(
                            "train_output_replay", 0
                        ),
                        "reconciliation": reconciliation,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.verb == "attest-staged":
            path = attest_staged(args)
            print(json.dumps({"staging_receipt": str(path)}, sort_keys=True))
            return 0
        if args.verb == "reconcile":
            print(json.dumps(reconcile_request(args), sort_keys=True))
            return 0
        if args.verb == "bind-job":
            path = bind_job(args)
            print(json.dumps({"job_binding": str(path)}, sort_keys=True))
            return 0
        if args.verb == "recover-bound-presubmit":
            path = recover_bound_presubmit(args)
            print(json.dumps({"recovery_evidence": str(path)}, sort_keys=True))
            return 0
        if args.verb == "rebind-airflow-state":
            path = rebind_airflow_state(args)
            print(json.dumps({"state_rebind_evidence": str(path)}, sort_keys=True))
            return 0
        if args.verb == "capture-preallocation-cancel":
            path = capture_preallocation_cancel(args)
            print(json.dumps({"cancellation_receipt": str(path)}, sort_keys=True))
            return 0
        path, returncode = finalize(args)
        print(json.dumps({"status": str(path), "exit_code": returncode}, sort_keys=True))
        return returncode
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        jsonschema.ValidationError,
    ) as exc:
        print(f"run_deft_action: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
