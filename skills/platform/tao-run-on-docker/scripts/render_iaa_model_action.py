#!/usr/bin/env python3
"""Render one signed GPU IAA model-action request as a Docker argv vector."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import pathlib
import sys
from typing import Any

from render_iaa_adapter import (
    CONTROLLER_RUNTIME_RELATIVE,
    RenderError,
    SAFE_NAME,
    WORKFLOW,
    _absolute,
    _canonical_sha256,
    _python_tree_sha256,
    _snapshot_manifest,
    _validate_snapshot,
)


MODEL_ACTIONS = {
    "pool_embed": ("embedding", "data-services"),
    "target_embed": ("embedding", "data-services"),
    "knn": ("tmm", "data-services"),
    "viz_weak_embed": ("embedding", "data-services"),
    "viz_mined_embed": ("embedding", "data-services"),
    "viz_previous_embed": ("embedding", "data-services"),
    "train": ("clip", "clip"),
    "evaluate": ("clip", "clip"),
}
EXPECTED_ENVIRONMENT = {
    "HOME": "/tmp",
    "PYTHONPATH": "/patches",
    "HF_HOME": "/cache/huggingface",
    "XDG_CACHE_HOME": "/cache",
}
RUNTIME_ENVIRONMENT = {
    "TORCH_HOME": "/cache/torch",
    "TRITON_CACHE_DIR": "/cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
    "MPLCONFIGDIR": "/cache/matplotlib",
}


def validate_model_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("schema_version") != "1":
        raise RenderError("model request must be a schema-v1 object")
    if request.get("workflow") != WORKFLOW or request.get("platform") != "docker":
        raise RenderError("model request must bind tao-run-deft-iaa on Docker")
    name = request.get("name")
    if name not in MODEL_ACTIONS:
        raise RenderError("request name is not an allowlisted IAA model action")
    if request.get("request_sha256") != _canonical_sha256(request):
        raise RenderError("model request signature is missing or invalid")
    gpu_ids = request.get("gpu_ids")
    if (
        not isinstance(gpu_ids, list)
        or not gpu_ids
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise RenderError("model request requires unique explicit nonnegative gpu_ids")
    if request.get("environment") != EXPECTED_ENVIRONMENT:
        raise RenderError("model request environment must match the exact Docker allowlist")
    passed_hf = request.get("passed_hf_token")
    expected_forward = ["HF_TOKEN"] if passed_hf is True else []
    if request.get("forward_env") != expected_forward:
        raise RenderError("model request credential forwarding is inconsistent")

    bundle = request.get("spec_bundle")
    command, network = MODEL_ACTIONS[name]
    if (
        not isinstance(bundle, dict)
        or bundle.get("mode") != "args"
        or bundle.get("command") != command
        or bundle.get("network_arch") != network
        or bundle.get("compute_shape") != {"gpus": len(gpu_ids), "nodes": 1}
        or bundle.get("image") != request.get("workload_image")
    ):
        raise RenderError("IAA model bundle is outside the signed GPU allowlist")
    action = bundle.get("action")
    if not isinstance(action, str) or not action.startswith(f"deft-iaa-{name}-"):
        raise RenderError("IAA model bundle action is inconsistent")

    mounts = request.get("mounts")
    if not isinstance(mounts, list) or not mounts:
        raise RenderError("model request mounts must be a non-empty array")
    targets: set[str] = set()
    for index, row in enumerate(mounts):
        if not isinstance(row, dict) or not isinstance(row.get("read_only"), bool):
            raise RenderError(f"mounts[{index}] is invalid")
        _absolute(row.get("source"), f"mounts[{index}].source")
        target = _absolute(row.get("target"), f"mounts[{index}].target")
        if str(target) in targets:
            raise RenderError(f"duplicate mount target: {target}")
        targets.add(str(target))
    patches_rows = [row for row in mounts if row.get("target") == "/patches"]
    cache_rows = [row for row in mounts if row.get("target") == "/cache"]
    if len(patches_rows) != 1 or patches_rows[0].get("read_only") is not True:
        raise RenderError("model action requires one read-only /patches mount")
    if len(cache_rows) != 1 or cache_rows[0].get("read_only") is not False:
        raise RenderError("model action requires one writable /cache mount")

    controller = request.get("controller_snapshot")
    patches = request.get("patches_snapshot")
    if not isinstance(controller, dict) or not isinstance(patches, dict):
        raise RenderError("model action requires controller and patches snapshots")
    controller_root = _absolute(controller.get("root"), "controller_snapshot.root")
    patches_root = _absolute(patches.get("root"), "patches_snapshot.root")
    if _absolute(patches_rows[0].get("source"), "patches source") != patches_root:
        raise RenderError("/patches source must equal patches_snapshot.root")
    _validate_snapshot(request, "controller_snapshot", controller_root)
    _validate_snapshot(request, "patches_snapshot", patches_root)
    runtime = controller_root / CONTROLLER_RUNTIME_RELATIVE / "iaa_deft"
    if _python_tree_sha256(runtime) != request.get("runtime_sha256"):
        raise RenderError("controller runtime does not match request.runtime_sha256")
    return request


def render_argv(request: dict[str, Any], job_id: str) -> list[str]:
    request = validate_model_request(request)
    if not isinstance(job_id, str) or SAFE_NAME.fullmatch(job_id) is None:
        raise RenderError("job id cannot form a safe Docker container name")
    bundle = request["spec_bundle"]
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise RenderError("refusing writable Docker launch as UID 0")
    gpu_ids = ",".join(str(item) for item in request["gpu_ids"])
    argv = [
        "docker", "run", "-d", "--name", job_id,
        "--label", f"tao-job={job_id}",
        "--label", f"tao-action={request['name']}",
        "--label", f"tao-request-sha256={request['request_sha256']}",
        "--gpus", f'"device={gpu_ids}"', "--ipc=host", "--shm-size=8g",
        "--user", f"{uid}:{gid}",
    ]
    for group_id in os.getgroups():
        if group_id != gid:
            argv += ["--group-add", str(group_id)]
    for row in request["mounts"]:
        mount = f"type=bind,src={row['source']},dst={row['target']}"
        if row["read_only"]:
            mount += ",readonly"
        argv += ["--mount", mount]
    environment = dict(request["environment"])
    environment.update(RUNTIME_ENVIRONMENT)
    user = getpass.getuser()
    environment.update({"USER": user, "LOGNAME": user})
    for name, value in sorted(environment.items()):
        argv += ["-e", f"{name}={value}"]
    for name in request["forward_env"]:
        argv += ["-e", name]
    argv += [bundle["image"], bundle["command"], *bundle["args"]]
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
        print(f"render_iaa_model_action: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"backend_name": args.job_id, "argv": argv}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
