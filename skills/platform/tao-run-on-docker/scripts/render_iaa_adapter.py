#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render one signed IAA CPU-adapter request as a Docker argv vector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
from typing import Any


WORKFLOW = "tao-run-deft-iaa"
ADAPTER_ACTIONS = frozenset({
    "dataset_rebuild", "dataset_materialize", "gap_analysis",
    "mining_postprocess", "history_select", "visualize_prepare",
    "visualize_finish", "eval_config", "train_config",
    "publish_checkpoint", "iteration_summary", "metric_parse", "report",
})
SHA256 = re.compile(r"[0-9a-f]{64}")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
EXPECTED_ENVIRONMENT = {
    "HOME": "/tmp",
    "PYTHONPATH": "/patches",
    "HF_HOME": "/cache/huggingface",
    "XDG_CACHE_HOME": "/cache",
    "IAA_COMPUTE_FRAME": "docker",
}
VISUALIZE_THREAD_CAPS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
CONTROLLER_RUNTIME_RELATIVE = pathlib.Path(
    "skills/applications/tao-run-deft-iaa/scripts"
)


class RenderError(ValueError):
    pass


def _canonical_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("request_sha256", None)
    return hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _python_tree_sha256(root: pathlib.Path) -> str:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise RenderError(f"IAA runtime has no Python files: {root}")
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise RenderError(f"IAA runtime contains an unsafe Python path: {path}")
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _snapshot_manifest(root: pathlib.Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RenderError(f"snapshot root is missing or unsafe: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RenderError(f"snapshot contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RenderError(f"snapshot contains a non-regular file: {path}")
        content = path.read_bytes()
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    if not entries:
        raise RenderError(f"snapshot contains no files: {root}")
    digest = hashlib.sha256(
        json.dumps(
            {"entries": entries}, ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {"root": str(root), "entries": entries, "sha256": digest}


def _validate_snapshot(request: dict[str, Any], field: str, root: pathlib.Path) -> str:
    approved = request.get(field)
    actual = _snapshot_manifest(root)
    if not isinstance(approved, dict) or approved != actual:
        raise RenderError(f"{field} does not match the complete local snapshot")
    return actual["sha256"]


def _absolute(value: Any, field: str) -> pathlib.Path:
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise RenderError(f"{field} must be an absolute path")
    path = pathlib.Path(value)
    if path == pathlib.Path("/") or pathlib.Path(os.path.abspath(path)) != path:
        raise RenderError(f"{field} must be normalized, non-root, and traversal-free")
    if "," in value:
        raise RenderError(f"{field} cannot contain a comma in a Docker --mount value")
    return path


def validate_adapter_request(request: Any) -> tuple[dict[str, Any], pathlib.Path]:
    if not isinstance(request, dict) or request.get("schema_version") != "1":
        raise RenderError("adapter request must be a schema-v1 object")
    if request.get("workflow") != WORKFLOW or request.get("platform") != "docker":
        raise RenderError("adapter request must bind tao-run-deft-iaa on Docker")
    name = request.get("name")
    if name not in ADAPTER_ACTIONS:
        raise RenderError("request name is not an allowlisted IAA adapter")
    request_sha256 = request.get("request_sha256")
    if (not isinstance(request_sha256, str) or SHA256.fullmatch(request_sha256) is None
            or request_sha256 != _canonical_sha256(request)):
        raise RenderError("adapter request signature is missing or invalid")
    runtime_sha256 = request.get("runtime_sha256")
    if not isinstance(runtime_sha256, str) or SHA256.fullmatch(runtime_sha256) is None:
        raise RenderError("adapter request requires runtime_sha256")
    if request.get("gpu_ids") != [] or request.get("passed_hf_token") is not False:
        raise RenderError("IAA adapters must bind gpu_ids=[] and passed_hf_token=false")
    if request.get("forward_env") != []:
        raise RenderError("IAA adapters cannot forward model credentials")
    expected_environment = dict(EXPECTED_ENVIRONMENT)
    if name == "visualize_finish":
        expected_environment.update(VISUALIZE_THREAD_CAPS)
    if request.get("environment") != expected_environment:
        raise RenderError("IAA adapter environment must match the exact Docker allowlist")
    bundle = request.get("spec_bundle")
    expected_args = [
        "/iaa-runtime/run_iaa_compute.py", name,
        "--results-dir", "/results", "--label", request.get("label"),
    ]
    if (not isinstance(bundle, dict) or bundle.get("network_arch") != "iaa-adapter"
            or bundle.get("mode") != "args" or bundle.get("command") != "python3"
            or bundle.get("args") != expected_args
            or bundle.get("compute_shape") != {"gpus": 0, "nodes": 1}):
        raise RenderError("IAA adapter bundle is outside the signed CPU allowlist")
    image = bundle.get("image")
    if not isinstance(image, str) or not image or request.get("workload_image") != image:
        raise RenderError("adapter workload image is missing or inconsistent")
    mounts = request.get("mounts")
    if not isinstance(mounts, list) or not mounts:
        raise RenderError("adapter request mounts must be a non-empty array")
    runtime_mounts = [
        row for row in mounts
        if isinstance(row, dict) and row.get("target") == "/iaa-runtime"
    ]
    if len(runtime_mounts) != 1 or runtime_mounts[0].get("read_only") is not True:
        raise RenderError("adapter requires one read-only /iaa-runtime mount")
    patches_mounts = [
        row for row in mounts
        if isinstance(row, dict) and row.get("target") == "/patches"
    ]
    if len(patches_mounts) != 1 or patches_mounts[0].get("read_only") is not True:
        raise RenderError("adapter requires one read-only /patches mount")
    runtime_source = _absolute(runtime_mounts[0].get("source"), "IAA runtime source")
    patches_source = _absolute(patches_mounts[0].get("source"), "patches source")
    controller = request.get("controller_snapshot")
    patches = request.get("patches_snapshot")
    if not isinstance(controller, dict):
        raise RenderError("controller_snapshot must be an object")
    if not isinstance(patches, dict):
        raise RenderError("patches_snapshot must be an object")
    controller_root = _absolute(controller.get("root"), "controller_snapshot.root")
    patches_root = _absolute(patches.get("root"), "patches_snapshot.root")
    if runtime_source != controller_root / CONTROLLER_RUNTIME_RELATIVE:
        raise RenderError("/iaa-runtime source must be derived from controller_snapshot.root")
    if patches_source != patches_root:
        raise RenderError("/patches source must equal patches_snapshot.root")
    _validate_snapshot(request, "controller_snapshot", controller_root)
    _validate_snapshot(request, "patches_snapshot", patches_root)
    if _python_tree_sha256(runtime_source / "iaa_deft") != runtime_sha256:
        raise RenderError("IAA runtime source does not match request.runtime_sha256")
    return request, runtime_source


def render_argv(request: dict[str, Any], job_id: str) -> list[str]:
    request, _ = validate_adapter_request(request)
    if not isinstance(job_id, str) or SAFE_NAME.fullmatch(job_id) is None:
        raise RenderError("job id cannot form a safe Docker container name")
    bundle = request["spec_bundle"]
    argv = [
        "docker", "run", "-d", "--name", job_id,
        "--label", f"tao-job={job_id}",
        "--label", f"tao-action={request['name']}",
        "--label", f"tao-request-sha256={request['request_sha256']}",
        "--label", f"tao-runtime-sha256={request['runtime_sha256']}",
        "--user", f"{os.getuid()}:{os.getgid()}",
    ]
    targets: set[str] = set()
    for index, row in enumerate(request["mounts"]):
        if not isinstance(row, dict) or not isinstance(row.get("read_only"), bool):
            raise RenderError(f"mounts[{index}] is invalid")
        source = _absolute(row.get("source"), f"mounts[{index}].source")
        target = _absolute(row.get("target"), f"mounts[{index}].target")
        if str(target) in targets:
            raise RenderError(f"duplicate mount target: {target}")
        targets.add(str(target))
        mount = f"type=bind,src={source},dst={target}"
        if row["read_only"]:
            mount += ",readonly"
        argv += ["--mount", mount]
    for name, value in sorted(request["environment"].items()):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise RenderError(f"invalid environment name: {name}")
        if not isinstance(value, str) or "\x00" in value:
            raise RenderError(f"invalid environment value: {name}")
        argv += ["-e", f"{name}={value}"]
    argv += [bundle["image"], bundle["command"], *bundle["args"]]
    if "--gpus" in argv or "NVIDIA_VISIBLE_DEVICES" in argv:
        raise AssertionError("CPU adapter renderer emitted a GPU selector")
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=pathlib.Path)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        argv = render_argv(request, args.job_id)
    except (OSError, json.JSONDecodeError, RenderError) as exc:
        print(f"render_iaa_adapter: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"backend_name": args.job_id, "argv": argv}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
