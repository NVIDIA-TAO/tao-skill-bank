# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Journal and commit one IAA DEFT stage to state and loop_log.

Use this instead of inline Python, jq, or hand-authored JSON. For evaluate,
this command validates and records the metric result before adding the ordered
log event, then runs the sibling audit script and rolls both files back if the
audit reports INVALID.

Stage machine (per iteration label):

    baseline: dataset_setup -> pool_embed -> evaluate -> gap_analysis (optional
        terminal; feeds iter1)
    iterN:    data_mining -> history_select -> sdg -> visualize (or --skip) -> train
        -> evaluate -> gap_analysis (optional terminal; feeds iterN+1)
    loop_stop: committed once at run level with --reason
        (kpi_met | max_iterations | hard_stop)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from checkpoint_contract import validate_best_checkpoint
from command_contract import (
    command_sha256,
    expected_container_command,
    expected_fresh_outputs,
    expected_hf_forwarding,
    expected_image_kind,
)
from deft_action_contract import platform_evidence_error, remote_freshness_attested
from log_stage import append_stage, next_seq
from iaa_deft.sdg import validate_image_edit_endpoint_pool, validate_normalized_dataset

try:
    from record_metric_result import commit as commit_metric_result
except ImportError:  # the sibling recorder ships with the skill
    commit_metric_result = None

STAGES = (
    "dataset_setup",
    "pool_embed",
    "evaluate",
    "gap_analysis",
    "data_mining",
    "history_select",
    "sdg",
    "visualize",
    "train",
    "loop_stop",
)
SKIPPABLE_STAGES = ("visualize",)
_LOOP_STOP_REASONS = ("kpi_met", "max_iterations", "hard_stop")
_EVAL_SUCCESS_MARKER = "Evaluate finished successfully"
_TRAIN_SUCCESS_MARKER = "Train finished successfully."
_VERIFY_MARKER = "VERIFY: PASS"
_CHECKSUM_MARKER = "CHECKSUM_VERIFY: PASS"


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_text(path: pathlib.Path, text: str) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _required_file(path: pathlib.Path | None, name: str) -> str:
    if path is None:
        raise ValueError(f"{name} is required")
    expanded = pathlib.Path(os.path.abspath(path.expanduser()))
    if not expanded.is_absolute():
        raise ValueError(f"{name} must be absolute: {path}")
    if not expanded.is_file() or expanded.stat().st_size == 0:
        raise ValueError(f"{name} must be an existing non-empty file: {path}")
    if expanded.resolve() != expanded:
        raise ValueError(f"{name} must not traverse a symlink: {expanded}")
    return str(expanded)


def _required_dir(path: pathlib.Path | None, name: str) -> str:
    if path is None:
        raise ValueError(f"{name} is required")
    expanded = pathlib.Path(os.path.abspath(path.expanduser()))
    if not expanded.is_absolute():
        raise ValueError(f"{name} must be absolute: {path}")
    if not expanded.is_dir() or not any(expanded.iterdir()):
        raise ValueError(f"{name} must be an existing non-empty directory: {path}")
    if expanded.resolve() != expanded:
        raise ValueError(f"{name} must not traverse a symlink: {expanded}")
    return str(expanded)


def _required_png(path: pathlib.Path | None, name: str) -> str:
    resolved = pathlib.Path(_required_file(path, name))
    with resolved.open("rb") as handle:
        header = handle.read(8)
    if header != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{name} must be a PNG image: {resolved}")
    return str(resolved.resolve())


def _required_image_dir(path: pathlib.Path | None, name: str) -> str:
    resolved = pathlib.Path(_required_dir(path, name))
    links = [item for item in resolved.rglob("*") if item.is_symlink()]
    if links:
        raise ValueError(f"{name} must not contain symlinks: {links[0]}")
    images = sorted(
        item
        for item in resolved.rglob("*")
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    valid = False
    for image in images:
        with image.open("rb") as handle:
            header = handle.read(8)
        if header == b"\x89PNG\r\n\x1a\n" or header[:3] == b"\xff\xd8\xff":
            valid = True
            break
    if not valid:
        raise ValueError(
            f"{name} must contain at least one non-empty PNG or JPEG image: {resolved}"
        )
    return str(resolved.resolve())


def _require_exact(path: str, expected: pathlib.Path, name: str) -> str:
    absolute = pathlib.Path(os.path.abspath(path))
    expected_absolute = pathlib.Path(os.path.abspath(expected))
    if absolute != expected_absolute:
        raise ValueError(f"{name} must be {expected_absolute}, got {absolute}")
    if absolute.resolve() != absolute:
        raise ValueError(f"{name} must not traverse a symlink: {absolute}")
    return str(absolute)


def _required_json(
    path: pathlib.Path | None,
    name: str,
    *,
    root_type: type,
    nonempty: bool = True,
) -> tuple[str, Any]:
    resolved = pathlib.Path(_required_file(path, name))
    try:
        payload = json.loads(resolved.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON: {resolved}: {exc}") from exc
    if not isinstance(payload, root_type):
        raise ValueError(f"{name} JSON root must be {root_type.__name__}: {resolved}")
    if nonempty and not payload:
        raise ValueError(f"{name} JSON must not be empty: {resolved}")
    return str(resolved), payload


def _required_parquet(
    path: pathlib.Path | None,
    name: str,
    *,
    required_columns: set[str],
) -> str:
    resolved = pathlib.Path(_required_file(path, name))
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            f"{name} validation requires pyarrow; run commit_stage.py with the "
            "approved runtime interpreter"
        ) from exc
    try:
        parquet = pq.ParquetFile(resolved)
        rows = parquet.metadata.num_rows
        columns = set(parquet.schema_arrow.names)
    except Exception as exc:
        raise ValueError(f"{name} is not a readable parquet: {resolved}: {exc}") from exc
    if rows < 1:
        raise ValueError(f"{name} must contain at least one row: {resolved}")
    missing = sorted(required_columns - columns)
    if missing:
        raise ValueError(f"{name} is missing parquet columns {missing}: {resolved}")
    return str(resolved)


def _validated_sdg_endpoint_evidence(
    pool_payload: dict[str, Any], auxiliary: dict[str, Any],
    sdg_config: dict[str, Any], execution_platform: str,
) -> None:
    """Bind the runtime pool and auxiliary services to immutable SDG intent."""
    pool = validate_image_edit_endpoint_pool(pool_payload)
    expected_capacity = (
        sdg_config["generation"]["generation_nodes"]
        * sdg_config["generation"]["gpus_per_generation_node"]
    )
    expected_model = sdg_config["models"]["image_edit"]
    if pool["platform"] != execution_platform:
        raise ValueError("--endpoint-pool platform disagrees with workflow platform")
    active_capacity = pool["required_capacity"]
    services_per_node = sdg_config["generation"]["gpus_per_generation_node"]
    if execution_platform == "slurm":
        if not services_per_node <= active_capacity <= expected_capacity or active_capacity % services_per_node:
            raise ValueError(
                "--endpoint-pool active capacity is not a complete-worker subset of "
                "the immutable SLURM generation topology"
            )
        topology = auxiliary.get("image_edit_pool")
        if not isinstance(topology, dict) or topology.get("requested_capacity") != expected_capacity:
            raise ValueError("--endpoint-manifest requested capacity disagrees with immutable topology")
        if topology.get("requested_nodes") != sdg_config["generation"]["generation_nodes"]:
            raise ValueError("--endpoint-manifest requested nodes disagree with immutable topology")
        if (
            topology.get("required_capacity") != active_capacity
            or topology.get("active_nodes") != active_capacity // services_per_node
        ):
            raise ValueError("--endpoint-manifest active topology disagrees with endpoint pool")
    elif active_capacity != expected_capacity:
        raise ValueError("--endpoint-pool capacity disagrees with immutable generation topology")
    if pool["model"] != {"id": expected_model["id"], "revision": expected_model["revision"]}:
        raise ValueError("--endpoint-pool model disagrees with immutable config")
    request_sha256 = auxiliary.get("request_sha256")
    if request_sha256 is not None and request_sha256 != pool["request_sha256"]:
        raise ValueError("--endpoint-manifest request digest disagrees with endpoint pool")

    records = auxiliary.get("containers") or auxiliary.get("probes") or auxiliary.get("roles")
    if not isinstance(records, dict):
        raise ValueError("--endpoint-manifest lacks auxiliary role evidence")
    for role in ("vlm", "llm"):
        record = records.get(role)
        if not isinstance(record, dict) or record.get("model") != sdg_config["models"][role]["id"]:
            raise ValueError(f"--endpoint-manifest {role} model disagrees with immutable config")
        if execution_platform in {"docker", "virtualenv"}:
            ownership = sdg_config["endpoints"]["ownership"]
            if ownership == "managed" and record.get("owned") is not True:
                raise ValueError(f"--endpoint-manifest {role} is not run-owned")
        elif record.get("ready") is not True and record.get("probe", {}).get("models_ok") is not True:
            raise ValueError(f"--endpoint-manifest {role} lacks readiness evidence")

    component_evidence = auxiliary.get("component_images") or auxiliary.get("components")
    if not isinstance(component_evidence, dict):
        raise ValueError("--endpoint-manifest lacks workflow-component provenance")
    for component in ("augmentation", "auto_labeling"):
        record = component_evidence.get(component)
        image = record.get("image") if isinstance(record, dict) else record
        if image != sdg_config["images"][component]:
            raise ValueError(f"--endpoint-manifest {component} image disagrees with immutable config")
        if isinstance(record, dict) and record.get("present") is False:
            raise ValueError(f"--endpoint-manifest lacks local {component} image evidence")


def _endpoint_pool_binding(path: pathlib.Path, payload: dict[str, Any]) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "request_sha256": payload["request_sha256"],
        "required_capacity": payload["required_capacity"],
    }


def _endpoint_pool_binding_matches(
    recorded: Any, canonical: dict[str, Any], execution_platform: str,
) -> bool:
    """Validate a pool binding after an evidence-preserving remote sync.

    Remote execution records the backend-absolute pool path.  Airflow/SLURM
    then synchronizes that exact artifact into the controller results tree,
    where its absolute prefix necessarily differs.  Retain strict path
    equality for local execution.  For remote platforms, accept only an
    identical content/request/capacity binding whose run/iteration-relative
    suffix is unchanged.
    """
    if not isinstance(recorded, dict):
        return False
    for field in ("sha256", "request_sha256", "required_capacity"):
        if recorded.get(field) != canonical.get(field):
            return False
    recorded_path = recorded.get("path")
    canonical_path = canonical.get("path")
    if not isinstance(recorded_path, str) or not isinstance(canonical_path, str):
        return False
    if pathlib.Path(recorded_path) == pathlib.Path(canonical_path):
        return True
    if execution_platform not in {"slurm", "brev"}:
        return False
    # run_<id>/iter_<N>/datagen/endpoint_pool.json must survive the sync.
    return pathlib.PurePosixPath(recorded_path).parts[-4:] == pathlib.Path(
        canonical_path
    ).parts[-4:]


def _synced_remote_path_matches(
    recorded: str, local: pathlib.Path, results_root: pathlib.Path,
) -> bool:
    """Match one backend-absolute path to its synchronized controller path."""
    if not isinstance(recorded, str):
        return False
    remote = pathlib.PurePosixPath(recorded)
    if not remote.is_absolute() or ".." in remote.parts:
        return False
    local_resolved = local.resolve()
    root_resolved = results_root.resolve()
    try:
        relative = local_resolved.relative_to(root_resolved)
    except ValueError:
        return False
    suffix = (root_resolved.name, *relative.parts)
    return len(remote.parts) >= len(suffix) and remote.parts[-len(suffix):] == suffix


def _coarse_slurm_sdg_freshness_attested(
    payload: dict[str, Any], outputs: list[pathlib.Path], *,
    scope: pathlib.Path, status_path: pathlib.Path,
) -> bool:
    """Accept whole-second Lustre mtimes only with independently bound evidence."""
    try:
        started_ns = payload["started_ns"]
        finished_ns = payload["finished_ns"]
        attempt = payload["attempt"]
        if (
            payload.get("schema_version") != "1"
            or payload.get("workflow") != "tao-run-deft-iaa"
            or payload.get("name") != "sdg-normalize"
            or payload.get("execution_platform") != "slurm"
            or not isinstance(started_ns, int) or isinstance(started_ns, bool)
            or not isinstance(finished_ns, int) or isinstance(finished_ns, bool)
            or not 1 <= started_ns <= finished_ns
            or not isinstance(attempt, int) or isinstance(attempt, bool)
            or not 1 <= attempt <= 2
        ):
            return False
        datagen = status_path.parent.parent.resolve()
        if status_path.resolve() != datagen / "status" / "sdg-normalize.slurm.status.json":
            return False
        datagen.relative_to(scope.resolve())

        expected_paths = [str(path.resolve()) for path in outputs]
        pre_binding = payload.get("pre_action")
        if not isinstance(pre_binding, dict) or set(pre_binding) != {"path", "sha256"}:
            return False
        pre_path = datagen / "status" / "sdg-normalize.slurm.pre-action.json"
        if (
            not _synced_remote_path_matches(
                pre_binding["path"], pre_path, _results_root_for_scope(scope)
            )
            or not pre_path.is_file() or pre_path.is_symlink()
            or hashlib.sha256(pre_path.read_bytes()).hexdigest() != pre_binding["sha256"]
        ):
            return False
        pre = json.loads(pre_path.read_text())
        if (
            set(pre) != {
                "schema_version", "workflow", "name", "execution_platform",
                "attempt", "started_ns", "inputs", "outputs",
            }
            or pre.get("schema_version") != "1"
            or pre.get("workflow") != "tao-run-deft-iaa"
            or pre.get("name") != "sdg-normalize"
            or pre.get("execution_platform") != "slurm"
            or pre.get("attempt") != attempt
            or pre.get("started_ns") != started_ns
            or pre.get("inputs") != payload.get("inputs")
        ):
            return False
        pre_outputs = pre.get("outputs")
        if not isinstance(pre_outputs, list) or len(pre_outputs) != len(outputs):
            return False
        results_root = _results_root_for_scope(scope)
        for output, record in zip(outputs, pre_outputs):
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "absent"}
                or record.get("absent") is not True
                or not _synced_remote_path_matches(
                    record.get("path"), output, results_root
                )
            ):
                return False

        evidence = payload.get("output_evidence")
        if not isinstance(evidence, list) or len(evidence) != len(outputs):
            return False
        for path, record in zip(outputs, evidence):
            stat = path.stat()
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "sha256", "size", "mtime_ns"}
                or not _synced_remote_path_matches(
                    record.get("path"), path, results_root
                )
                or record.get("size") != stat.st_size or stat.st_size <= 0
                or record.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or not isinstance(record.get("mtime_ns"), int)
                or isinstance(record.get("mtime_ns"), bool)
                or record["mtime_ns"] % 1_000_000_000 != 0
                or record["mtime_ns"] // 1_000_000_000 != started_ns // 1_000_000_000
                or record["mtime_ns"] > finished_ns
            ):
                return False

        endpoint_path = datagen / "endpoint_manifest.json"
        endpoint = json.loads(endpoint_path.read_text())
        job_id = endpoint.get("job_id")
        request_sha256 = endpoint.get("request_sha256")
        action_id = endpoint.get("action_id")
        request_attempt = endpoint.get("attempt")
        if (
            not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9.-]+", job_id)
            or not isinstance(request_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None
            or not isinstance(action_id, str) or not action_id
            or not isinstance(request_attempt, int) or isinstance(request_attempt, bool)
        ):
            return False
        request_path = datagen / ".tao-runtime" / f"sdg.action.{job_id}.json"
        terminal_path = datagen / f"slurm_sdg_terminal.{job_id}.json"
        if any(not path.is_file() or path.is_symlink() for path in (endpoint_path, request_path, terminal_path)):
            return False
        request = json.loads(request_path.read_text())
        unsigned = dict(request)
        unsigned.pop("request_sha256", None)
        actual_request_sha256 = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        terminal = json.loads(terminal_path.read_text())
        if (
            actual_request_sha256 != request_sha256
            or request.get("request_sha256") != request_sha256
            or request.get("action_id") != action_id
            or request.get("attempt") != request_attempt
            or len(request.get("expected_outputs", [])) < len(outputs)
            or terminal.get("status") != "ok"
            or terminal.get("job_id") != job_id
            or terminal.get("request_sha256") != request_sha256
            or terminal.get("action_id") != action_id
            or terminal.get("attempt") != request_attempt
        ):
            return False
        for source in (request["expected_outputs"], terminal.get("expected_outputs", [])):
            if len(source) < len(outputs) or any(
                not _synced_remote_path_matches(recorded, output, results_root)
                for recorded, output in zip(source[:len(outputs)], outputs)
            ):
                return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def _required_command_status(
    path: pathlib.Path | None,
    name: str,
    *,
    scope: pathlib.Path,
    required_output: pathlib.Path | None = None,
    required_outputs: list[pathlib.Path] | None = None,
    required_name: str | None = None,
    allow_stale_resume: bool = False,
    required_command: list[str] | None = None,
    required_image_kind: str | None = None,
    required_image: str | None = None,
    required_hf_forwarding: bool | None = None,
    required_platform: str | None = None,
    allow_coarse_slurm_sdg: bool = False,
    allow_remote_coarse_mtime: bool = False,
) -> str:
    resolved_text, payload = _required_json(path, name, root_type=dict)
    resolved = pathlib.Path(resolved_text)
    _require_within(str(resolved), scope, name)
    if (
        payload.get("schema_version") not in {"1", "2"}
        or payload.get("workflow") != "tao-run-deft-iaa"
        or payload.get("status") != "ok"
        or payload.get("exit_code") != 0
        or not isinstance(payload.get("name"), str)
        or not payload.get("name", "").strip()
        or not isinstance(payload.get("finished_at"), str)
        or not payload.get("finished_at", "").strip()
    ):
        raise ValueError(f"{name} must record a successful IAA DEFT command: {resolved}")
    if required_name is not None and payload.get("name") != required_name:
        raise ValueError(
            f"{name}.name must be {required_name!r}, got {payload.get('name')!r}"
        )
    if required_command is not None:
        if required_platform is None:
            raise ValueError(f"{name} requires an initialized workflow platform")
        evidence_error = platform_evidence_error(payload, required_platform)
        if evidence_error is not None:
            raise ValueError(f"{name} does not prove native action success: {evidence_error}")
        if payload.get("command") != required_command:
            raise ValueError(f"{name} does not record the approved command argv")
        if payload.get("command_sha256") != command_sha256(required_command):
            raise ValueError(f"{name}.command_sha256 does not match its approved argv")
        if payload.get("image_kind") != required_image_kind:
            raise ValueError(
                f"{name}.image_kind must be {required_image_kind!r}, "
                f"got {payload.get('image_kind')!r}"
            )
        if payload.get("image") != required_image:
            raise ValueError(
                f"{name}.image must be {required_image!r}, got {payload.get('image')!r}"
            )
        if payload.get("passed_hf_token") is not required_hf_forwarding:
            raise ValueError(
                f"{name}.passed_hf_token must be {required_hf_forwarding!r}"
            )
    attempt = payload.get("attempt")
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or not 1 <= attempt <= 2
    ):
        raise ValueError(f"{name}.attempt must be an integer in [1, 2]")
    outputs = list(required_outputs or [])
    if required_output is not None:
        outputs.append(required_output)
    recorded_log_path = str(payload.get("log_path", ""))
    log_path = pathlib.Path(recorded_log_path).expanduser()
    if allow_coarse_slurm_sdg:
        local_log = resolved.parent.parent / "logs" / "sdg-normalize.slurm.log"
        if _synced_remote_path_matches(
            recorded_log_path, local_log, _results_root_for_scope(scope)
        ):
            log_path = local_log
    if (
        not log_path.is_absolute()
        or not log_path.is_file()
        or log_path.stat().st_size == 0
        or log_path.is_symlink()
        or log_path.resolve() != log_path
    ):
        raise ValueError(f"{name} references a missing, empty, or unsafe log: {log_path}")
    _require_within(str(log_path.resolve()), scope, f"{name}.log_path")
    if not isinstance(payload.get("fresh_outputs"), list) or not payload["fresh_outputs"]:
        raise ValueError(f"{name}.fresh_outputs must be a non-empty list")
    results_root = _results_root_for_scope(scope)
    coarse_fresh: set[str] = set()
    for item in payload["fresh_outputs"]:
        if not isinstance(item, str):
            raise ValueError(f"{name}.fresh_outputs entries must be absolute paths")
        raw_output = pathlib.Path(item).expanduser()
        absolute_output = pathlib.Path(os.path.abspath(raw_output))
        if not raw_output.is_absolute() or raw_output != absolute_output:
            raise ValueError(
                f"{name}.fresh_outputs entry must be a normalized absolute path: {item}"
            )
        if allow_coarse_slurm_sdg and outputs:
            matches = [
                output for output in outputs
                if _synced_remote_path_matches(item, output, results_root)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{name}.fresh_outputs does not map to one synchronized output: {item}"
                )
            coarse_fresh.add(str(matches[0].resolve()))
        else:
            _require_within(
                str(absolute_output.resolve()), results_root, f"{name}.fresh_outputs"
            )
    if outputs:
        fresh = coarse_fresh or {
            str(pathlib.Path(str(item)).resolve())
            for item in payload.get("fresh_outputs", [])
        }
        started_ns = payload.get("started_ns")
        if not isinstance(started_ns, int) or isinstance(started_ns, bool) or started_ns < 1:
            raise ValueError(f"{name}.started_ns must be a positive integer")
        for output in outputs:
            if str(output.resolve()) not in fresh:
                raise ValueError(
                    f"{name} does not bind required fresh output {output.resolve()}"
                )
            mtime_ns = (output.lstat() if output.is_symlink() else output.stat()).st_mtime_ns
            coarse_sdg = (
                allow_coarse_slurm_sdg
                and _coarse_slurm_sdg_freshness_attested(
                    payload, outputs, scope=scope, status_path=resolved,
                )
            )
            # Some remote filesystems expose only whole-second mtimes. A file
            # created after a command starts can therefore appear up to (but
            # never a full) second older than started_ns. This exception is
            # enabled only by the adapter bridge after its outer finalized
            # platform status proves the remote staged-absence contract.
            coarse_remote_mtime = (
                allow_remote_coarse_mtime
                and mtime_ns < started_ns
                and started_ns - mtime_ns < 1_000_000_000
            )
            if (
                mtime_ns < started_ns
                and not remote_freshness_attested(payload)
                and not (allow_stale_resume and payload.get("resume") is True)
                and not coarse_sdg
                and not coarse_remote_mtime
            ):
                raise ValueError(
                    f"{output} is older than the command recorded by {name}"
                )
    return str(resolved)


def _required_adapter_status(path, argument: str, *, scope: pathlib.Path,
                             outputs: list[pathlib.Path], adapter: str,
                             label: str, config: dict,
                             allow_stale_resume: bool = False) -> str:
    """Require finalized platform evidence for one deterministic adapter."""
    results_root = _results_root_for_scope(scope)
    action_outputs = expected_fresh_outputs(adapter, label, results_root)
    platform_status = _required_command_status(
        path, argument, scope=scope, required_outputs=action_outputs,
        required_name=adapter, allow_stale_resume=allow_stale_resume,
        required_command=expected_container_command(adapter, label, config),
        required_image_kind=expected_image_kind(adapter),
        required_image=config["pyt_image" if expected_image_kind(adapter) == "pyt" else "ds_image"],
        required_hf_forwarding=False, required_platform=config["platform"],
    )
    required = {str(item.resolve()) for item in outputs}
    directly_bound = {str(item.resolve()) for item in action_outputs}
    if required.issubset(directly_bound):
        return platform_status
    host_statuses = [
        item for item in action_outputs if item.name.endswith(".host.status.json")
    ]
    if len(host_statuses) != 1:
        missing = sorted(required - directly_bound)
        raise ValueError(
            f"{argument} does not bind required adapter outputs: {missing}"
        )
    allow_remote_coarse_mtime = False
    if adapter == "publish_checkpoint":
        platform_payload = json.loads(pathlib.Path(platform_status).read_text())
        allow_remote_coarse_mtime = remote_freshness_attested(platform_payload)
    _required_command_status(
        host_statuses[0], f"{argument} nested host status", scope=scope,
        required_outputs=outputs, required_name=adapter.replace("_", "-"),
        allow_stale_resume=allow_stale_resume,
        allow_remote_coarse_mtime=allow_remote_coarse_mtime,
    )
    return platform_status


def _require_within(path: str, root: pathlib.Path, name: str) -> str:
    resolved = pathlib.Path(path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} must be under {root}: {resolved}") from exc
    return str(resolved)


def _results_root_for_scope(scope: pathlib.Path) -> pathlib.Path:
    """Find the canonical run root while retaining phase-scoped status checks."""
    resolved = scope.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "deft_state.json").is_file():
            return candidate
    return resolved


def _require_marker(path: str, marker: str, name: str) -> str:
    try:
        text = pathlib.Path(path).read_text(errors="replace")
    except OSError as exc:
        raise ValueError(f"{name} cannot be read: {path}: {exc}") from exc
    if marker not in text:
        raise ValueError(f"{name} must contain {marker!r}: {path}")
    return path


def _phase_dir(results_dir: pathlib.Path, iter_label: str) -> pathlib.Path:
    """Map an iteration label to its iaa_deft results directory.

    baseline artifacts live under zs/ (zero-shot); iterN artifacts live under
    iter_<N>/ — note the underscore.
    """
    if iter_label == "baseline":
        return results_dir / "zs"
    match = re.fullmatch(r"iter([1-9][0-9]*)", iter_label)
    if not match:
        raise ValueError(f"unrecognized iteration label: {iter_label}")
    return results_dir / f"iter_{match.group(1)}"


def _load_log(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"loop_log line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(entry, dict):
            raise ValueError(f"loop_log line {line_number} must be an object")
        entries.append(entry)
    return entries


def _expected_next(
    entry: dict[str, Any], state: dict[str, Any]
) -> set[tuple[str, str]]:
    label = str(entry.get("iteration"))
    stage = str(entry.get("stage"))
    if entry.get("status") == "error":
        return {(label, "loop_stop")}
    if stage == "loop_stop":
        return set()
    if label == "baseline":
        if stage == "dataset_setup":
            return {("baseline", "pool_embed")}
        if stage == "pool_embed":
            return {("baseline", "evaluate")}
        if stage == "evaluate":
            info = state.get("iterations", {}).get("baseline", {})
            result = info.get("metric_result") if isinstance(info, dict) else None
            return (
                {("baseline", "loop_stop")}
                if isinstance(result, dict) and result.get("passed") is True
                else {("baseline", "gap_analysis")}
            )
        if stage == "gap_analysis":
            return {("iter1", "data_mining")}
        return set()
    match = re.fullmatch(r"iter([1-9][0-9]*)", label)
    if not match:
        return set()
    number = int(match.group(1))
    if stage == "data_mining":
        return {(label, "history_select")}
    if stage == "history_select":
        return {(label, "sdg")}
    if stage == "sdg":
        return {(label, "visualize")}
    if stage == "visualize":
        return {(label, "train")}
    if stage == "train":
        return {(label, "evaluate")}
    if stage == "evaluate":
        info = state.get("iterations", {}).get(label, {})
        result = info.get("metric_result") if isinstance(info, dict) else None
        passed = isinstance(result, dict) and result.get("passed") is True
        maximum = int(state.get("max_iterations", 0))
        return (
            {(label, "loop_stop")}
            if passed or number >= maximum
            else {(label, "gap_analysis")}
        )
    if stage == "gap_analysis":
        maximum = int(state.get("max_iterations", 0))
        return (
            {(f"iter{number + 1}", "data_mining")}
            if number < maximum
            else set()
        )
    return set()


def _validate_transition(
    entries: list[dict[str, Any]], state: dict[str, Any], iter_label: str, stage: str
) -> None:
    key = (iter_label, stage)
    if any((entry.get("iteration"), entry.get("stage")) == key for entry in entries):
        raise ValueError(f"stage already committed: {iter_label}/{stage}")
    if not entries:
        if key != ("baseline", "dataset_setup"):
            raise ValueError("first stage must be baseline/dataset_setup")
        return
    allowed = _expected_next(entries[-1], state)
    if key not in allowed:
        rendered = ", ".join(f"{label}/{name}" for label, name in sorted(allowed))
        previous = entries[-1]
        raise ValueError(
            f"illegal transition {previous.get('iteration')}/{previous.get('stage')} -> "
            f"{iter_label}/{stage}; expected one of [{rendered or 'end-of-log'}]"
        )


def _validate_loop_stop(state: dict[str, Any], args: argparse.Namespace) -> None:
    if args.reason is None:
        raise ValueError(
            "--reason is required for loop_stop "
            f"(one of {', '.join(_LOOP_STOP_REASONS)})"
        )
    if args.reason == "kpi_met" and state.get("gate_met") is not True:
        raise ValueError(
            "loop_stop reason kpi_met requires gate_met=true in state; no "
            "recorded evaluate result has passed the metric gate"
        )
    if args.reason == "max_iterations":
        current = int(state.get("current_iteration", 0))
        maximum = int(state.get("max_iterations", 0))
        if current < maximum:
            raise ValueError(
                "loop_stop reason max_iterations requires "
                f"current_iteration >= max_iterations ({current} < {maximum})"
            )


def _apply_success(
    state: dict[str, Any],
    phase: dict[str, Any],
    stage: str,
    args: argparse.Namespace,
    results_dir: pathlib.Path,
    iter_label: str,
) -> None:
    phase_root = _phase_dir(results_dir, iter_label)
    config = state.get("config", {})
    if stage == "visualize":
        configured = bool(config.get("visualize") or config.get("visualize_embeddings"))
        if args.skip and configured:
            raise ValueError(
                "visualize may be skipped only when both approved visualization flags are false"
            )
        if not args.skip and not configured:
            raise ValueError(
                "visualize must use --skip when both approved visualization flags are false"
            )
    if stage == "dataset_setup":
        splits = pathlib.Path(
            _require_exact(
                _required_dir(args.iaa_splits_dir, "--iaa-splits-dir"),
                results_dir / "iaa_splits",
                "--iaa-splits-dir",
            )
        )
        split_outputs: list[pathlib.Path] = []
        for name, root_type in (
            ("eval_pairs.json", list),
            ("aug_pool_pairs.json", list),
        ):
            _required_json(splits / name, f"--iaa-splits-dir/{name}", root_type=root_type)
            split_outputs.append(splits / name)
        for name in ("eval_list.txt", "val_list.txt", "aug_pool_list.txt"):
            _required_file(splits / name, f"--iaa-splits-dir/{name}")
            split_outputs.append(splits / name)
        source_pool = pathlib.Path(
            _required_parquet(
                args.source_pool_parquet,
                "--source-pool-parquet",
                required_columns={"filepath", "text"},
            )
        )
        _require_exact(
            str(source_pool),
            results_dir / "embeddings" / "source" / "source_pool.parquet",
            "--source-pool-parquet",
        )
        verify_log = _require_marker(
            _required_file(args.verify_log, "--verify-log"),
            _VERIFY_MARKER,
            "--verify-log",
        )
        _require_exact(
            verify_log,
            results_dir / "dataset_setup" / "rebuild_verify.log",
            "--verify-log",
        )
        phase["dataset_rebuild_status"] = _require_exact(
            _required_adapter_status(
                args.dataset_rebuild_status, "--dataset-rebuild-status",
                scope=results_dir, outputs=[pathlib.Path(verify_log)],
                adapter="dataset_rebuild", label="baseline", config=config,
            ),
            results_dir / "dataset_setup" / "dataset_rebuild.status.json",
            "--dataset-rebuild-status",
        )
        dataset_status = _required_adapter_status(
            args.dataset_materialize_status,
            "--dataset-materialize-status",
            scope=results_dir,
            outputs=[results_dir / "dataset_setup" / "dataset-materialize.host.status.json"], adapter="dataset_materialize",
            label="baseline", config=config,
        )
        phase["dataset_materialize_status"] = _require_exact(
            dataset_status,
            results_dir / "dataset_setup" / "dataset_materialize.status.json",
            "--dataset-materialize-status",
        )
        if config.get("checksums_file"):
            checksum_log = _require_marker(
                _required_file(args.checksum_verify_log, "--checksum-verify-log"),
                _CHECKSUM_MARKER,
                "--checksum-verify-log",
            )
            phase["checksum_verify_log"] = _require_exact(
                checksum_log,
                results_dir / "dataset_setup" / "checksum_verify.log",
                "--checksum-verify-log",
            )
        dataset_root = pathlib.Path(str(config.get("dataset_root", "")))
        for name in ("images", "captions"):
            if not (dataset_root / name).is_dir():
                raise ValueError(f"dataset root is missing {name}/ after dataset_setup")
        for name in ("train_pairs.json", "val_pairs.json"):
            _required_json(dataset_root / name, f"dataset_root/{name}", root_type=list)
        phase["iaa_splits_dir"] = str(splits)
        phase["source_pool_parquet"] = str(source_pool)
        phase["verify_log"] = verify_log
    elif stage == "pool_embed":
        output = pathlib.Path(
            _required_parquet(
                args.pool_embeddings_parquet,
                "--pool-embeddings-parquet",
                required_columns={"filepath", "embedding"},
            )
        )
        _require_exact(
            str(output),
            results_dir / "embeddings" / "source" / "embeddings.parquet",
            "--pool-embeddings-parquet",
        )
        phase["pool_embed_command_status"] = _require_exact(
            _required_command_status(
                args.pool_embed_command_status,
                "--pool-embed-command-status",
                scope=results_dir,
                required_output=output,
                required_name="pool_embed",
                required_command=expected_container_command(
                    "pool_embed", iter_label, config
                ),
                required_image_kind=expected_image_kind("pool_embed"),
                required_image=config["ds_image"],
                required_hf_forwarding=expected_hf_forwarding("pool_embed", config),
                required_platform=config["platform"],
            ),
            results_dir / "embeddings" / "source" / "pool_embed.status.json",
            "--pool-embed-command-status",
        )
        phase["pool_embeddings_parquet"] = str(output)
    elif stage == "evaluate":
        evaluate_dir = phase_root / "evaluate"
        eval_config = pathlib.Path(
            _require_exact(
                _required_file(args.eval_config, "--eval-config"),
                phase_root / "specs" / "eval_config.yaml",
                "--eval-config",
            )
        )
        eval_config_status = _required_adapter_status(
            args.eval_config_status,
            "--eval-config-status",
            scope=phase_root,
            outputs=[eval_config], adapter="eval_config", label=iter_label, config=config,
        )
        phase["eval_config"] = str(eval_config)
        phase["eval_config_status"] = _require_exact(
            eval_config_status,
            phase_root / "specs" / "eval_config.status.json",
            "--eval-config-status",
        )
        metrics = _require_exact(
            _required_file(args.metrics_aggregate_csv, "--metrics-aggregate-csv"),
            evaluate_dir / "nvidia_pas_metrics_aggregate.csv",
            "--metrics-aggregate-csv",
        )
        detailed_metrics = _required_file(
            evaluate_dir / "nvidia_pas_metrics.csv",
            "detailed evaluation metrics",
        )
        weighted_metrics = _required_file(
            evaluate_dir / "nvidia_pas_metrics_weighted_aggregate.csv",
            "weighted evaluation metrics",
        )
        eval_status = _require_exact(
            _require_marker(
                _required_file(args.eval_status_json, "--eval-status-json"),
                _EVAL_SUCCESS_MARKER,
                "--eval-status-json",
            ),
            evaluate_dir / "status.json",
            "--eval-status-json",
        )
        phase["eval_command_status"] = _require_exact(
            _required_command_status(
                args.eval_command_status,
                "--eval-command-status",
                scope=phase_root,
                required_outputs=[
                    pathlib.Path(detailed_metrics),
                    pathlib.Path(metrics),
                    pathlib.Path(weighted_metrics),
                    pathlib.Path(eval_status),
                ],
                required_name="evaluate",
                required_command=expected_container_command(
                    "evaluate", iter_label, config
                ),
                required_image_kind=expected_image_kind("evaluate"),
                required_image=config["pyt_image"],
                required_hf_forwarding=expected_hf_forwarding("evaluate", config),
                required_platform=config["platform"],
            ),
            evaluate_dir / "evaluate.status.json",
            "--eval-command-status",
        )
        phase["metrics_detailed_csv"] = str(detailed_metrics)
        phase["metrics_aggregate_csv"] = metrics
        phase["metrics_weighted_aggregate_csv"] = str(weighted_metrics)
        phase["eval_status_json"] = eval_status
        phase["metric_parse_status"] = _require_exact(
            _required_adapter_status(
                args.metric_parse_status, "--metric-parse-status",
                scope=phase_root, outputs=[evaluate_dir / "metric_result.json"],
                adapter="metric_parse", label=iter_label, config=config,
            ),
            evaluate_dir / "metric_parse.status.json", "--metric-parse-status",
        )
        required = ("metric_result",)
        missing = [field for field in required if not phase.get(field)]
        if missing or phase.get("status") != "complete":
            raise ValueError(
                "evaluate metric commit is incomplete; missing "
                f"{missing or ['status=complete']}"
            )
        if iter_label != "baseline":
            summary = _require_exact(
                _required_json(
                    args.iteration_summary,
                    "--iteration-summary",
                    root_type=dict,
                )[0],
                phase_root / "iteration_summary.json",
                "--iteration-summary",
            )
            phase["iteration_summary"] = summary
            summary_status = _required_adapter_status(
                args.iteration_summary_status,
                "--iteration-summary-status",
                scope=phase_root,
                outputs=[pathlib.Path(summary)], adapter="iteration_summary",
                label=iter_label, config=config,
            )
            phase["iteration_summary_status"] = _require_exact(
                summary_status,
                phase_root / "iteration_summary.status.json",
                "--iteration-summary-status",
            )
    elif stage == "gap_analysis":
        feed_number = 1 if iter_label == "baseline" else int(iter_label[4:]) + 1
        gaps = _required_parquet(
            args.gaps_parquet,
            "--gaps-parquet",
            required_columns={"filepath", "text", "weak_attribute"},
        )
        phase["gaps_parquet"] = _require_exact(
            gaps,
            results_dir / f"iter_{feed_number}" / "gaps" / "kpi_gaps.parquet",
            "--gaps-parquet",
        )
        history_path, history_payload = _required_json(
            args.caption_history, "--caption-history", root_type=dict
        )
        phase["caption_history"] = _require_exact(
            history_path,
            results_dir / "caption_selection_history.json",
            "--caption-history",
        )
        gap_status = _required_adapter_status(
            args.gap_analysis_status,
            "--gap-analysis-status",
            scope=results_dir / f"iter_{feed_number}",
            outputs=[pathlib.Path(phase["gaps_parquet"])], adapter="gap_analysis",
            label=f"iter{feed_number}", config=config,
        )
        phase["gap_analysis_status"] = _require_exact(
            gap_status,
            results_dir
            / f"iter_{feed_number}"
            / "gaps"
            / "gap_analysis.status.json",
            "--gap-analysis-status",
        )
        entries = history_payload.get("entries")
        if not isinstance(entries, list) or not any(
            isinstance(item, dict) and item.get("iter") == feed_number
            for item in entries
        ):
            raise ValueError(
                f"--caption-history has no selection entry for feed iteration {feed_number}"
            )
    elif stage == "data_mining":
        target = pathlib.Path(
            _required_parquet(
                args.target_embeddings_parquet,
                "--target-embeddings-parquet",
                required_columns={"filepath", "embedding"},
            )
        )
        mined = pathlib.Path(
            _required_parquet(
                args.mined_parquet,
                "--mined-parquet",
                required_columns={"filepath"},
            )
        )
        candidate_dir = (
            phase_root / "mining" / "history_candidates"
            if config.get("history_aware")
            else phase_root / "mining"
        )
        candidate = _required_json(
            args.candidate_pairs, "--candidate-pairs", root_type=list
        )[0]
        phase["target_embeddings_parquet"] = _require_exact(
            str(target),
            phase_root / "embeddings" / "target" / "embeddings.parquet",
            "--target-embeddings-parquet",
        )
        phase["mined_parquet"] = _require_exact(
            str(mined),
            phase_root / "mining" / "mined_samples.parquet",
            "--mined-parquet",
        )
        phase["candidate_pairs"] = _require_exact(
            candidate,
            candidate_dir / "mined_pairs.json",
            "--candidate-pairs",
        )
        phase["target_embed_command_status"] = _require_exact(
            _required_command_status(
                args.target_embed_command_status,
                "--target-embed-command-status",
                scope=phase_root,
                required_output=target,
                required_name="target_embed",
                required_command=expected_container_command(
                    "target_embed", iter_label, config
                ),
                required_image_kind=expected_image_kind("target_embed"),
                required_image=config["ds_image"],
                required_hf_forwarding=expected_hf_forwarding("target_embed", config),
                required_platform=config["platform"],
            ),
            phase_root / "embeddings" / "target" / "target_embed.status.json",
            "--target-embed-command-status",
        )
        phase["knn_command_status"] = _require_exact(
            _required_command_status(
                args.knn_command_status,
                "--knn-command-status",
                scope=phase_root,
                required_output=mined,
                required_name="knn",
                required_command=expected_container_command("knn", iter_label, config),
                required_image_kind=expected_image_kind("knn"),
                required_image=config["ds_image"],
                required_hf_forwarding=expected_hf_forwarding("knn", config),
                required_platform=config["platform"],
            ),
            phase_root / "mining" / "knn.status.json",
            "--knn-command-status",
        )
        postprocess_status = _required_adapter_status(
            args.mining_postprocess_status,
            "--mining-postprocess-status",
            scope=phase_root,
            outputs=[pathlib.Path(phase["candidate_pairs"])], adapter="mining_postprocess",
            label=iter_label, config=config,
        )
        phase["mining_postprocess_status"] = _require_exact(
            postprocess_status,
            phase_root / "mining" / "mining_postprocess.status.json",
            "--mining-postprocess-status",
        )
    elif stage == "history_select":
        phase["mined_image_list"] = _require_exact(
            _required_file(args.mined_image_list, "--mined-image-list"),
            phase_root / "mining" / "mined_image_list.txt",
            "--mined-image-list",
        )
        phase["mined_pairs"] = _require_exact(
            _required_json(args.mined_pairs, "--mined-pairs", root_type=list)[0],
            phase_root / "mining" / "mined_pairs.json",
            "--mined-pairs",
        )
        phase["mined_manifest"] = _require_exact(
            _required_json(args.mined_manifest, "--mined-manifest", root_type=dict)[0],
            phase_root / "mining" / "mined_dataset.json",
            "--mined-manifest",
        )
        phase["cumulative_names"] = _require_exact(
            _required_json(args.cumulative_names, "--cumulative-names", root_type=list)[0],
            phase_root / "mining" / "cumulative_mined_unique_names.json",
            "--cumulative-names",
        )
        if config.get("history_aware"):
            history_path, history_payload = _required_json(
                args.mining_history, "--mining-history", root_type=dict
            )
            _require_exact(
                history_path,
                results_dir / "mining_selection_history.json",
                "--mining-history",
            )
            iteration_number = int(iter_label[4:])
            raw_iterations = history_payload.get("iterations")
            if isinstance(raw_iterations, dict):
                present = str(iteration_number) in raw_iterations or iteration_number in raw_iterations
            elif isinstance(raw_iterations, list):
                present = any(
                    isinstance(item, dict) and item.get("iteration") == iteration_number
                    for item in raw_iterations
                )
            else:
                present = False
            if not present:
                raise ValueError(
                    f"--mining-history has no committed entry for iteration {iteration_number}"
                )
            phase["mining_history"] = history_path
        history_outputs = [
            pathlib.Path(phase[field])
            for field in (
                "mined_image_list",
                "mined_pairs",
                "mined_manifest",
                "cumulative_names",
            )
        ]
        history_status = _required_adapter_status(
            args.history_select_status,
            "--history-select-status",
            scope=phase_root,
            outputs=history_outputs, adapter="history_select", label=iter_label, config=config,
            allow_stale_resume=True,
        )
        phase["history_select_status"] = _require_exact(
            history_status,
            phase_root / "mining" / "history_select.status.json",
            "--history-select-status",
        )
    elif stage == "sdg":
        datagen = phase_root / "datagen"
        dataset = datagen / "dataset"
        manifest_path, manifest = _required_json(
            args.sdg_manifest, "--sdg-manifest", root_type=dict
        )
        phase["sdg_manifest"] = _require_exact(
            manifest_path, dataset / "sdg_manifest.json", "--sdg-manifest"
        )
        if manifest.get("schema_version") != "1" or manifest.get("dataset_format_version") != 3:
            raise ValueError("--sdg-manifest must use schema_version=1 and dataset_format_version=3")
        if manifest.get("rejected_samples_included") != 0:
            raise ValueError("--sdg-manifest includes rejected samples")
        if not isinstance(manifest.get("accepted_provenance"), list) or not manifest["accepted_provenance"]:
            raise ValueError("--sdg-manifest lacks accepted provenance")
        phase["sdg_pairs"] = _require_exact(
            _required_json(args.sdg_pairs, "--sdg-pairs", root_type=list)[0],
            dataset / "sdg_pairs.json", "--sdg-pairs",
        )
        phase["sdg_image_list"] = _require_exact(
            _required_file(args.sdg_image_list, "--sdg-image-list"),
            dataset / "sdg_image_list.txt", "--sdg-image-list",
        )
        validate_normalized_dataset(
            pathlib.Path(phase["sdg_manifest"]),
            pathlib.Path(phase["sdg_pairs"]),
            pathlib.Path(phase["sdg_image_list"]),
        )
        execution_path, execution = _required_json(
            args.sdg_execution_manifest, "--sdg-execution-manifest", root_type=dict
        )
        phase["sdg_execution_manifest"] = _require_exact(
            execution_path, datagen / "sdg_execution_manifest.json", "--sdg-execution-manifest"
        )
        if execution.get("accepted_crops", 0) < 1 or execution.get("accepted_sources", 0) < 1:
            raise ValueError("--sdg-execution-manifest records no accepted generated data")
        sdg_config_path = pathlib.Path(str(config.get("sdg_config", "")))
        sdg_config = yaml.safe_load(
            pathlib.Path(
                _required_file(sdg_config_path, "state.config.sdg_config")
            ).read_text()
        )
        pool_path, pool_payload = _required_json(
            args.endpoint_pool, "--endpoint-pool", root_type=dict
        )
        phase["endpoint_pool"] = _require_exact(
            pool_path, datagen / "endpoint_pool.json", "--endpoint-pool"
        )
        endpoint_path, endpoint_payload = _required_json(
            args.endpoint_manifest, "--endpoint-manifest", root_type=dict
        )
        phase["endpoint_manifest"] = _require_exact(
            endpoint_path, datagen / "endpoint_manifest.json", "--endpoint-manifest"
        )
        _validated_sdg_endpoint_evidence(
            pool_payload, endpoint_payload, sdg_config, config["platform"]
        )
        execution_pool = execution.get("endpoint_pool")
        expected_pool_binding = _endpoint_pool_binding(
            pathlib.Path(phase["endpoint_pool"]), pool_payload
        )
        if not _endpoint_pool_binding_matches(
            execution_pool, expected_pool_binding, config["platform"]
        ):
            raise ValueError(
                "--sdg-execution-manifest endpoint_pool does not bind the canonical pool"
            )
        status = _required_command_status(
            args.sdg_status, "--sdg-status", scope=phase_root,
            required_outputs=[
                pathlib.Path(phase["sdg_manifest"]),
                pathlib.Path(phase["sdg_pairs"]),
                pathlib.Path(phase["sdg_image_list"]),
            ],
            required_name="sdg-normalize",
            allow_coarse_slurm_sdg=config["platform"] == "slurm",
        )
        status_platforms = {config["platform"]}
        if config["platform"] in {"docker", "virtualenv"}:
            status_platforms.add("host")
        expected_statuses = {
            (datagen / "status" / f"sdg-normalize.{platform}.status.json").resolve()
            for platform in status_platforms
        }
        if pathlib.Path(status).resolve() not in expected_statuses:
            raise ValueError(
                "--sdg-status must be the canonical platform-specific sdg-normalize status"
            )
        phase["sdg_status"] = status
    elif stage == "visualize":
        if args.skip:
            # Documented branch skip: the loop config disabled visualization,
            # so there is no t-SNE plot to record. The skip still occupies its
            # slot in the ordered log so train remains the next legal stage.
            phase["visualize_skipped"] = True
        else:
            prepare_outputs: list[pathlib.Path] = []
            if config.get("visualize"):
                phase["samples_dir"] = _require_exact(
                    _required_image_dir(args.samples_dir, "--samples-dir"),
                    phase_root / "visualization" / "samples",
                    "--samples-dir",
                )
                prepare_outputs.append(pathlib.Path(phase["samples_dir"]))
            if config.get("visualize_embeddings"):
                weak_input = phase_root / "embeddings" / "viz_weak" / "input.parquet"
                mined_input = phase_root / "mining" / "mined_unique_images.parquet"
                _required_parquet(
                    weak_input,
                    "visualization weak input",
                    required_columns={"filepath"},
                )
                _required_parquet(
                    mined_input,
                    "visualization mined input",
                    required_columns={"filepath"},
                )
                prepare_outputs.extend([weak_input, mined_input])
                previous_input = (
                    phase_root / "embeddings" / "previous" / "prev_pool.parquet"
                )
                if previous_input.is_file() and previous_input.stat().st_size > 0:
                    _required_parquet(
                        previous_input,
                        "visualization previous input",
                        required_columns={"filepath"},
                    )
                    prepare_outputs.append(previous_input)
                phase["tsne_plot"] = _require_exact(
                    _required_png(args.tsne_plot, "--tsne-plot"),
                    phase_root / "visualization" / "tsne_plot.png",
                    "--tsne-plot",
                )
                statuses = args.visualize_command_status or []
                expected_outputs = [
                    phase_root / "embeddings" / "viz_weak" / "embeddings.parquet",
                    phase_root / "embeddings" / "augmented" / "mined_embeddings.parquet",
                ]
                previous_input = (
                    phase_root / "embeddings" / "previous" / "prev_pool.parquet"
                )
                if previous_input.is_file() and previous_input.stat().st_size > 0:
                    expected_outputs.append(
                        phase_root / "embeddings" / "previous" / "embeddings.parquet"
                    )
                if len(statuses) != len(expected_outputs):
                    raise ValueError(
                        "--visualize-command-status must be supplied in weak, mined, "
                        "and optional previous-data order (expected "
                        f"{len(expected_outputs)}, got {len(statuses)})"
                    )
                checked_statuses: list[str] = []
                for status, output in zip(statuses, expected_outputs):
                    _required_parquet(
                        output,
                        "visualization embedding output",
                        required_columns={"filepath", "embedding"},
                    )
                    command_name = (
                        "viz_weak_embed"
                        if len(checked_statuses) == 0
                        else "viz_mined_embed"
                        if len(checked_statuses) == 1
                        else "viz_previous_embed"
                    )
                    checked_statuses.append(
                        _require_exact(
                            _required_command_status(
                                status,
                                "--visualize-command-status",
                                scope=phase_root,
                                required_output=output,
                                required_name=command_name,
                                required_command=expected_container_command(
                                    command_name, iter_label, config
                                ),
                                required_image_kind=expected_image_kind(command_name),
                                required_image=config["ds_image"],
                                required_hf_forwarding=expected_hf_forwarding(
                                    command_name, config
                                ),
                                required_platform=config["platform"],
                            ),
                            output.parent / f"{command_name}.status.json",
                            "--visualize-command-status",
                        )
                    )
                phase["visualize_command_statuses"] = checked_statuses
            prepare_status = _required_adapter_status(
                args.visualize_prepare_status,
                "--visualize-prepare-status",
                scope=phase_root,
                outputs=prepare_outputs, adapter="visualize_prepare",
                label=iter_label, config=config,
            )
            phase["visualize_prepare_status"] = _require_exact(
                prepare_status,
                phase_root / "visualization" / "visualize_prepare.status.json",
                "--visualize-prepare-status",
            )
            if config.get("visualize_embeddings"):
                finish_status = _required_adapter_status(
                    args.visualize_finish_status,
                    "--visualize-finish-status",
                    scope=phase_root,
                    outputs=[pathlib.Path(phase["tsne_plot"])], adapter="visualize_finish",
                    label=iter_label, config=config,
                )
                phase["visualize_finish_status"] = _require_exact(
                    finish_status,
                    phase_root / "visualization" / "visualize_finish.status.json",
                    "--visualize-finish-status",
                )
    elif stage == "train":
        train_dir = phase_root / "train"
        if args.best_ckpt is None:
            raise ValueError("--best-ckpt is required")
        phase["pretrained_state"] = _require_exact(
            _required_file(args.pretrained_state, "--pretrained-state"),
            phase_root / "pretrained" / "model_state.pth",
            "--pretrained-state",
        )
        phase["train_config"] = _require_exact(
            _required_file(args.train_config, "--train-config"),
            phase_root / "specs" / "train_config.yaml",
            "--train-config",
        )
        train_config_status = _required_adapter_status(
            args.train_config_status,
            "--train-config-status",
            scope=phase_root,
            outputs=[pathlib.Path(phase["train_config"])], adapter="train_config",
            label=iter_label, config=config,
        )
        phase["train_config_status"] = _require_exact(
            train_config_status,
            phase_root / "specs" / "train_config.status.json",
            "--train-config-status",
        )
        train_tao_status = _require_exact(
            _require_marker(
                _required_file(args.train_tao_status_json, "--train-tao-status-json"),
                _TRAIN_SUCCESS_MARKER,
                "--train-tao-status-json",
            ),
            train_dir / "status.json",
            "--train-tao-status-json",
        )
        train_command_status = _required_command_status(
            args.train_command_status,
            "--train-command-status",
            scope=phase_root,
            required_output=pathlib.Path(train_tao_status),
            required_name="train",
            required_command=expected_container_command("train", iter_label, config),
            required_image_kind=expected_image_kind("train"),
            required_image=config["pyt_image"],
            required_hf_forwarding=expected_hf_forwarding("train", config),
            required_platform=config["platform"],
        )
        phase["train_command_status"] = _require_exact(
            train_command_status,
            train_dir / "train.status.json",
            "--train-command-status",
        )
        phase["train_tao_status_json"] = train_tao_status
        train_payload = json.loads(pathlib.Path(train_command_status).read_text())
        provenance = validate_best_checkpoint(
            args.best_ckpt,
            train_dir,
            started_ns=train_payload.get(
                "lineage_started_ns", train_payload["started_ns"]
            ),
        )
        phase.update(provenance)
        publish_host_status = phase_root / "train" / "publish-checkpoint.host.status.json"
        publish_status = _required_adapter_status(
            args.publish_checkpoint_status,
            "--publish-checkpoint-status",
            scope=phase_root,
            outputs=[
                pathlib.Path(phase["best_ckpt_path"]),
                pathlib.Path(phase["best_ckpt_metadata"]),
                pathlib.Path(phase["pretrained_state"]),
            ],
            adapter="publish_checkpoint", label=iter_label, config=config,
        )
        phase["publish_checkpoint_status"] = _require_exact(
            publish_status,
            phase_root / "train" / "publish_checkpoint.status.json",
            "--publish-checkpoint-status",
        )
        # The finalized platform status binds the adapter's host-status file;
        # that nested status binds the checkpoints produced by the adapter.
        # Inspecting the outer status here would incorrectly require it to
        # flatten outputs that belong to the signed nested command.
        publish_payload = json.loads(publish_host_status.read_text())
        publish_fresh = {
            str(pathlib.Path(os.path.abspath(pathlib.Path(str(item)).expanduser())))
            for item in publish_payload.get("fresh_outputs", [])
        }
        if phase["best_ckpt_path"] not in publish_fresh:
            raise ValueError(
                "--publish-checkpoint-status does not bind the canonical best checkpoint"
            )
    elif stage != "loop_stop":
        raise ValueError(f"unsupported stage: {stage}")

    if stage != "loop_stop":
        phase["stage_completed"] = stage
    if stage != "evaluate" and phase.get("status") != "complete":
        phase["status"] = "in_progress"


def _run_audit(results_dir: pathlib.Path) -> tuple[bool, str, str]:
    """Run the sibling audit script against the results dir.

    Returns (ok, run_status, output). Missing or unrecognized audit evidence
    fails closed and rolls the transaction back.
    """
    audit_script = pathlib.Path(__file__).resolve().parent / "audit_deft_run.py"
    if not audit_script.is_file():
        return (
            False,
            "MISSING",
            f"audit script not found: {audit_script}",
        )
    completed = subprocess.run(
        [sys.executable, str(audit_script), "--results-dir", str(results_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    match = re.search(r"DEFT_RUN_STATUS=(\S+)", output)
    run_status = match.group(1) if match else "UNKNOWN"
    if completed.returncode != 0 or run_status not in {
        "IN_PROGRESS",
        "FAILED",
        "COMPLETE",
    }:
        return False, run_status, output
    return True, run_status, output


def commit(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"baseline|iter[1-9][0-9]*", args.iter_label):
        raise ValueError("--iter-label must be baseline or iterN (N >= 1)")
    if args.skip and args.stage not in SKIPPABLE_STAGES:
        raise ValueError(
            f"--skip is valid only for: {', '.join(SKIPPABLE_STAGES)}"
        )
    if args.skip and args.status == "error":
        raise ValueError("--skip and --status error are mutually exclusive")
    if not args.summary.strip():
        raise ValueError("--summary must be a non-empty outcome description")

    results_dir = args.results_dir.expanduser().resolve()
    state_path = results_dir / "deft_state.json"
    log_path = results_dir / "loop_log.jsonl"
    transaction_path = results_dir / ".deft_commit_transaction.json"
    if not state_path.is_file():
        raise ValueError(f"state file not found: {state_path}")
    original_state_text = state_path.read_text()
    original_log = log_path.read_text() if log_path.exists() else None
    if transaction_path.exists():
        raise ValueError(
            f"unfinished stage transaction found: {transaction_path}; run "
            "recover_commit.py before committing another stage"
        )
    state = json.loads(original_state_text)
    if (
        not isinstance(state, dict)
        or state.get("workflow") != "tao-run-deft-iaa"
        or state.get("schema_version") != "3"
    ):
        raise ValueError("deft_state.json must be an IAA DEFT schema-v3 object")
    if pathlib.Path(str(state.get("results_dir", ""))).resolve() != results_dir:
        raise ValueError("state.results_dir does not match --results-dir")

    entries = _load_log(log_path)
    _validate_transition(entries, state, args.iter_label, args.stage)
    log_status = "skip" if args.skip else args.status

    if args.status == "ok" and args.stage == "loop_stop":
        _validate_loop_stop(state, args)
    match = re.fullmatch(r"iter([1-9][0-9]*)", args.iter_label)
    if (
        args.status == "ok"
        and args.stage == "data_mining"
        and match
        and int(match.group(1)) > int(state.get("max_iterations", 0))
    ):
        raise ValueError(
            f"{args.iter_label} exceeds max_iterations="
            f"{state.get('max_iterations')}; commit loop_stop instead"
        )

    _atomic_json(
        transaction_path,
        {
            "schema_version": "1",
            "workflow": "tao-run-deft-iaa",
            "results_dir": str(results_dir),
            "iteration": args.iter_label,
            "stage": args.stage,
            "original_state_text": original_state_text,
            "original_log_text": original_log,
        },
    )
    try:
        if args.stage == "evaluate" and args.status == "ok":
            if commit_metric_result is None:
                raise ValueError(
                    "record_metric_result.py was not found next to "
                    "commit_stage.py; evaluate commits require the metric "
                    "recorder"
                )
            commit_metric_result(
                argparse.Namespace(
                    results_dir=results_dir,
                    iter_label=args.iter_label,
                    metric_result=args.metric_result,
                    metrics_csv=args.metrics_aggregate_csv,
                )
            )
            state = json.loads(state_path.read_text())

        iterations = state.get("iterations")
        if not isinstance(iterations, dict):
            raise ValueError("state.iterations must be an object")
        if args.stage == "loop_stop":
            if args.status == "ok":
                state["loop_stop_reason"] = args.reason
        else:
            existing = iterations.setdefault(
                args.iter_label, {"status": "in_progress"}
            )
            if not isinstance(existing, dict):
                raise ValueError(
                    f"state.iterations.{args.iter_label} must be an object"
                )
            if args.status == "error":
                existing["status"] = "failed"
            else:
                _apply_success(
                    state,
                    existing,
                    args.stage,
                    args,
                    results_dir,
                    args.iter_label,
                )
                if args.stage == "evaluate":
                    result = existing.get("metric_result")
                    if isinstance(result, dict) and result.get("passed") is True:
                        state["gate_met"] = True

        if match:
            state["current_iteration"] = max(
                int(match.group(1)), int(state.get("current_iteration", 0))
            )

        _atomic_json(state_path, state)
        append_stage(
            log_path,
            iteration=args.iter_label,
            stage=args.stage,
            status=log_status,
            summary=args.summary,
            duration_s=args.duration_s,
        )
        # State and log are now both durable. Remove the rollback journal
        # before auditing; a crash before this point is recovered by
        # recover_commit.py, while a crash after it leaves a complete pair.
        transaction_path.unlink()
        audit_ok, run_status, audit_output = _run_audit(results_dir)
        if not audit_ok:
            raise ValueError(
                "post-commit audit failed; state and log rolled back:\n"
                + (audit_output or f"audit exited with status {run_status}")
            )
    except Exception:
        _atomic_text(state_path, original_state_text)
        if original_log is None:
            try:
                log_path.unlink()
            except FileNotFoundError:
                pass
        else:
            _atomic_text(log_path, original_log)
        try:
            transaction_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "status": run_status,
        "last_committed": {
            "seq": next_seq(log_path) - 1,
            "iteration": args.iter_label,
            "stage": args.stage,
            "status": log_status,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--iter-label", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--status", choices=("ok", "error"), default="ok")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument(
        "--skip",
        action="store_true",
        help=(
            "Record a documented skip instead of artifacts; logs status=skip "
            f"and marks the stage completed. Valid only for: "
            f"{', '.join(SKIPPABLE_STAGES)}."
        ),
    )
    parser.add_argument("--iaa-splits-dir", type=pathlib.Path)
    parser.add_argument("--dataset-rebuild-status", type=pathlib.Path)
    parser.add_argument("--dataset-materialize-status", type=pathlib.Path)
    parser.add_argument("--source-pool-parquet", type=pathlib.Path)
    parser.add_argument(
        "--verify-log",
        type=pathlib.Path,
        help=f'dataset_setup verification log; must contain "{_VERIFY_MARKER}".',
    )
    parser.add_argument("--checksum-verify-log", type=pathlib.Path)
    parser.add_argument("--pool-embeddings-parquet", type=pathlib.Path)
    parser.add_argument("--pool-embed-command-status", type=pathlib.Path)
    parser.add_argument("--metrics-aggregate-csv", type=pathlib.Path)
    parser.add_argument(
        "--eval-status-json",
        type=pathlib.Path,
        help=f'evaluate status artifact; must contain "{_EVAL_SUCCESS_MARKER}".',
    )
    parser.add_argument("--metric-result", type=pathlib.Path)
    parser.add_argument("--metric-parse-status", type=pathlib.Path)
    parser.add_argument("--eval-command-status", type=pathlib.Path)
    parser.add_argument("--eval-config", type=pathlib.Path)
    parser.add_argument("--eval-config-status", type=pathlib.Path)
    parser.add_argument("--iteration-summary", type=pathlib.Path)
    parser.add_argument("--iteration-summary-status", type=pathlib.Path)
    parser.add_argument("--gaps-parquet", type=pathlib.Path)
    parser.add_argument("--caption-history", type=pathlib.Path)
    parser.add_argument("--gap-analysis-status", type=pathlib.Path)
    parser.add_argument("--target-embeddings-parquet", type=pathlib.Path)
    parser.add_argument("--target-embed-command-status", type=pathlib.Path)
    parser.add_argument("--mined-parquet", type=pathlib.Path)
    parser.add_argument("--knn-command-status", type=pathlib.Path)
    parser.add_argument("--candidate-pairs", type=pathlib.Path)
    parser.add_argument("--mining-postprocess-status", type=pathlib.Path)
    parser.add_argument("--mined-image-list", type=pathlib.Path)
    parser.add_argument("--mined-pairs", type=pathlib.Path)
    parser.add_argument("--mined-manifest", type=pathlib.Path)
    parser.add_argument("--cumulative-names", type=pathlib.Path)
    parser.add_argument("--mining-history", type=pathlib.Path)
    parser.add_argument("--history-select-status", type=pathlib.Path)
    parser.add_argument("--sdg-manifest", type=pathlib.Path)
    parser.add_argument("--sdg-pairs", type=pathlib.Path)
    parser.add_argument("--sdg-image-list", type=pathlib.Path)
    parser.add_argument("--sdg-execution-manifest", type=pathlib.Path)
    parser.add_argument("--endpoint-pool", type=pathlib.Path)
    parser.add_argument("--endpoint-manifest", type=pathlib.Path)
    parser.add_argument("--sdg-status", type=pathlib.Path)
    parser.add_argument("--samples-dir", type=pathlib.Path)
    parser.add_argument("--tsne-plot", type=pathlib.Path)
    parser.add_argument("--visualize-prepare-status", type=pathlib.Path)
    parser.add_argument("--visualize-finish-status", type=pathlib.Path)
    parser.add_argument(
        "--visualize-command-status", action="append", type=pathlib.Path
    )
    parser.add_argument("--best-ckpt", type=pathlib.Path)
    parser.add_argument("--pretrained-state", type=pathlib.Path)
    parser.add_argument("--train-config", type=pathlib.Path)
    parser.add_argument("--train-command-status", type=pathlib.Path)
    parser.add_argument(
        "--train-tao-status-json",
        type=pathlib.Path,
        help=f'train status artifact; must contain "{_TRAIN_SUCCESS_MARKER}".',
    )
    parser.add_argument("--train-config-status", type=pathlib.Path)
    parser.add_argument("--publish-checkpoint-status", type=pathlib.Path)
    parser.add_argument("--reason", choices=_LOOP_STOP_REASONS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = commit(args)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"commit_stage: {exc}", file=sys.stderr)
        return 2
    last = report["last_committed"]
    print(
        f"committed seq={last['seq']} {last['iteration']}/{last['stage']} "
        f"status={last['status']} run={report['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
