#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Brev consumer for a platform-neutral TAO action request."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import re
import shlex
import sys
from typing import Any, Sequence

from brev_transport import run_remote


SAFE_JOB_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
SAFE_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
ITERATION_LABEL = re.compile(r"iter[1-9][0-9]*")
IDENTITY_MARKER = "TAO_BREV_IDENTITY="
SNAPSHOT_MARKER = "TAO_IAA_SNAPSHOT="
IAA_ADAPTER_RUNTIME = "/iaa-runtime"
IAA_ADAPTER_SCRIPT = f"{IAA_ADAPTER_RUNTIME}/run_iaa_compute.py"
IAA_CONTROLLER_RUNTIME_RELATIVE = pathlib.PurePosixPath(
    "skills/applications/tao-run-deft-iaa/scripts"
)
IAA_ADAPTER_ACTIONS = frozenset(
    {
        "dataset_rebuild",
        "dataset_materialize",
        "gap_analysis",
        "mining_postprocess",
        "history_select",
        "visualize_prepare",
        "visualize_finish",
        "eval_config",
        "train_config",
        "publish_checkpoint",
        "iteration_summary",
        "metric_parse",
        "report",
    }
)
IAA_BASELINE_ADAPTERS = frozenset({"dataset_rebuild", "dataset_materialize"})
IAA_TERMINAL_ADAPTERS = frozenset({"report"})
IAA_MIXED_LABEL_ADAPTERS = frozenset({"eval_config", "metric_parse"})
IAA_ADAPTER_ENVIRONMENT = {
    "HOME": "/tmp",
    "PYTHONPATH": "/patches",
    "HF_HOME": "/cache/huggingface",
    "XDG_CACHE_HOME": "/cache",
    "IAA_COMPUTE_FRAME": "brev",
}
IAA_VISUALIZE_THREAD_CAPS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
REMOTE_SNAPSHOT_CODE = """\
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
if root.is_symlink() or not root.is_dir():
    raise SystemExit("IAA snapshot root is missing or unsafe")
entries = []
for path in sorted(root.rglob("*")):
    if path.is_symlink() or not path.is_file():
        if path.is_dir() and not path.is_symlink():
            continue
        raise SystemExit("IAA snapshot contains an unsafe path")
    content = path.read_bytes()
    entries.append({
        "path": path.relative_to(root).as_posix(),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    })
if not entries:
    raise SystemExit("IAA snapshot contains no files")
payload = {"entries": entries}
payload["sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
if sys.argv[2] == "controller":
    runtime = root / "skills/applications/tao-run-deft-iaa/scripts/iaa_deft"
    files = sorted(
        path for path in runtime.rglob("*.py") if "__pycache__" not in path.parts
    )
    if not files:
        raise SystemExit("IAA runtime has no Python files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(runtime).as_posix().encode("utf-8"))
        digest.update(b"\\0")
        digest.update(path.read_bytes())
        digest.update(b"\\0")
    payload["runtime_sha256"] = digest.hexdigest()
print("TAO_IAA_SNAPSHOT=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""


def _remote_python_command(code: str, *args: str) -> str:
    """Render a one-line transport-safe Python command without shell data interpolation."""
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    loader = f"import base64;exec(base64.b64decode({encoded!r}))"
    command = shlex.join(["python3", "-c", loader, *args])
    if "\n" in command or "\x00" in command:
        raise ValueError("rendered remote Python command is not transport-safe")
    return command


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _absolute_path(value: Any, field: str) -> pathlib.PurePosixPath:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"adapter {field} must be an absolute path")
    path = pathlib.PurePosixPath(value)
    if path == pathlib.PurePosixPath("/") or str(path) != value or ".." in path.parts:
        raise ValueError(f"adapter {field} must be normalized, non-root, and traversal-free")
    return path


def _under(child: pathlib.PurePosixPath, parent: pathlib.PurePosixPath) -> bool:
    return child == parent or parent in child.parents


def _remove_suffix(
    path: pathlib.PurePosixPath, suffix: pathlib.PurePosixPath, field: str
) -> pathlib.PurePosixPath:
    if path.parts[-len(suffix.parts) :] != suffix.parts:
        raise ValueError(f"adapter {field} does not end with {suffix}")
    root = path
    for _ in suffix.parts:
        root = root.parent
    if root == pathlib.PurePosixPath("/"):
        raise ValueError(f"adapter {field} has an unsafe root")
    return root


def _adapter_label_is_valid(name: str, label: Any) -> bool:
    if not isinstance(label, str):
        return False
    if name in IAA_BASELINE_ADAPTERS:
        return label == "baseline"
    if name in IAA_TERMINAL_ADAPTERS:
        return label == "terminal"
    if name in IAA_MIXED_LABEL_ADAPTERS:
        return label == "baseline" or bool(ITERATION_LABEL.fullmatch(label))
    return bool(ITERATION_LABEL.fullmatch(label))


def _validate_snapshot_contract(
    snapshot: Any, expected_root: str, field: str
) -> dict[str, Any]:
    if not isinstance(snapshot, dict) or set(snapshot) != {"root", "entries", "sha256"}:
        raise ValueError(f"adapter {field} has an invalid shape")
    if snapshot.get("root") != expected_root:
        raise ValueError(f"adapter {field} root does not match its mount source")
    entries = snapshot.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"adapter {field} must contain files")
    paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError(f"adapter {field} contains an invalid entry")
        relative = entry.get("path")
        if not isinstance(relative, str):
            raise ValueError(f"adapter {field} contains an invalid path")
        path = pathlib.PurePosixPath(relative)
        if (
            path.is_absolute()
            or str(path) != relative
            or relative in {"", "."}
            or ".." in path.parts
        ):
            raise ValueError(f"adapter {field} contains an unsafe path")
        size = entry.get("size")
        digest = entry.get("sha256")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or not SHA256.fullmatch(digest)
        ):
            raise ValueError(f"adapter {field} contains invalid file evidence")
        paths.append(relative)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"adapter {field} paths must be sorted and unique")
    if snapshot.get("sha256") != _sha256_json({"entries": entries}):
        raise ValueError(f"adapter {field} aggregate digest is invalid")
    return {"entries": entries, "sha256": snapshot["sha256"]}


def _validate_adapter_evidence(payload: dict[str, Any]) -> None:
    results = _absolute_path(payload.get("results_dir"), "results_dir")
    stage = _absolute_path(payload.get("stage_dir"), "stage_dir")
    log = _absolute_path(payload.get("log_path"), "log_path")
    receipt = _absolute_path(payload.get("staging_receipt_path"), "staging_receipt_path")
    binding = _absolute_path(payload.get("job_binding_path"), "job_binding_path")
    if not _under(stage, results):
        raise ValueError("adapter stage_dir must be within results_dir")
    if not all(_under(path, stage) for path in (log, receipt, binding)):
        raise ValueError("adapter log, staging receipt, and job binding must be within stage_dir")
    fresh = payload.get("fresh_outputs")
    absent = payload.get("staging_absent_paths")
    if (
        not isinstance(fresh, list)
        or not fresh
        or any(not _under(_absolute_path(item, "fresh_outputs"), results) for item in fresh)
        or absent != [*fresh, str(log)]
    ):
        raise ValueError("adapter freshness evidence is invalid")
    if payload.get("freshness_contract") != "remote-mirror-with-delete-before-submit":
        raise ValueError("adapter freshness contract is invalid")


def _validate_adapter_request(payload: dict[str, Any], bundle: dict[str, Any]) -> None:
    name = payload.get("name")
    label = payload.get("label")
    if payload.get("schema_version") != "1" or payload.get("workflow") != "tao-run-deft-iaa":
        raise ValueError("zero-GPU execution is restricted to schema-v1 tao-run-deft-iaa adapters")
    if name not in IAA_ADAPTER_ACTIONS or bundle.get("network_arch") != "iaa-adapter":
        raise ValueError("zero-GPU execution is restricted to allowlisted IAA adapters")
    if not _adapter_label_is_valid(name, label):
        raise ValueError("IAA adapter label is invalid for the requested action")
    expected_args = [
        IAA_ADAPTER_SCRIPT,
        name,
        "--results-dir",
        "/results",
        "--label",
        label,
    ]
    if (
        bundle.get("mode") != "args"
        or bundle.get("compute_shape") != {"gpus": 0, "nodes": 1}
        or bundle.get("command") != "python3"
        or bundle.get("args") != expected_args
    ):
        raise ValueError("IAA adapter command does not match the allowlisted command contract")
    expected_image = bundle.get("image")
    if (
        not isinstance(expected_image, str)
        or not expected_image
        or payload.get("record_image") != expected_image
        or payload.get("workload_image") != expected_image
    ):
        raise ValueError("IAA adapter image fields do not match the signed bundle image")
    if payload.get("forward_env") != [] or payload.get("passed_hf_token", False) is not False:
        raise ValueError("IAA adapters must not forward credentials")
    if "cache_subset" in payload:
        raise ValueError("IAA adapters must not declare a model cache subset")
    environment = payload.get("environment")
    expected_environment = dict(IAA_ADAPTER_ENVIRONMENT)
    if name == "visualize_finish":
        expected_environment.update(IAA_VISUALIZE_THREAD_CAPS)
    if environment != expected_environment:
        raise ValueError("IAA adapter environment must match the exact producer contract")
    digest = payload.get("runtime_sha256")
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ValueError("IAA adapter runtime_sha256 must be a lowercase SHA-256 digest")
    mounts = payload.get("mounts")
    if not isinstance(mounts, list):
        raise ValueError("IAA adapter mounts must be a list")
    runtime_mounts = [
        item
        for item in mounts
        if isinstance(item, dict) and item.get("target") == IAA_ADAPTER_RUNTIME
    ]
    results_mounts = [
        item
        for item in mounts
        if isinstance(item, dict) and item.get("target") == "/results"
    ]
    patches_mounts = [
        item
        for item in mounts
        if isinstance(item, dict) and item.get("target") == "/patches"
    ]
    if len(runtime_mounts) != 1 or runtime_mounts[0].get("read_only") is not True:
        raise ValueError("IAA adapter requires one read-only /iaa-runtime mount")
    if len(results_mounts) != 1 or results_mounts[0].get("read_only") is not False:
        raise ValueError("IAA adapter requires one writable /results mount")
    if len(patches_mounts) != 1 or patches_mounts[0].get("read_only") is not True:
        raise ValueError("IAA adapter requires one read-only /patches mount")
    runtime_source = str(
        _absolute_path(runtime_mounts[0].get("source"), "runtime mount source")
    )
    patches_source = str(
        _absolute_path(patches_mounts[0].get("source"), "patches mount source")
    )
    controller_root = _remove_suffix(
        pathlib.PurePosixPath(runtime_source),
        IAA_CONTROLLER_RUNTIME_RELATIVE,
        "runtime mount source",
    )
    controller = _validate_snapshot_contract(
        payload.get("controller_snapshot"), str(controller_root), "controller_snapshot"
    )
    _validate_snapshot_contract(
        payload.get("patches_snapshot"), patches_source, "patches_snapshot"
    )
    controller_entrypoint = (
        IAA_CONTROLLER_RUNTIME_RELATIVE / "run_iaa_compute.py"
    ).as_posix()
    controller_paths = {entry["path"] for entry in controller["entries"]}
    if controller_entrypoint not in controller_paths:
        raise ValueError("adapter controller_snapshot lacks run_iaa_compute.py")
    if not any(
        path.startswith("skills/applications/tao-run-deft-iaa/references/")
        for path in controller_paths
    ):
        raise ValueError("adapter controller_snapshot lacks application references")
    if not any(
        path.startswith("skills/core/tao-artifacts/references/")
        and path.endswith(".schema.json")
        for path in controller_paths
    ):
        raise ValueError("adapter controller_snapshot lacks core artifact schemas")
    declared = bundle.get("declared_inputs")
    runtime_inputs = (
        [
            item
            for item in declared
            if isinstance(item, dict) and item.get("spec_key") == "iaa_runtime"
        ]
        if isinstance(declared, list)
        else []
    )
    if len(runtime_inputs) != 1 or runtime_inputs[0] != {
        "spec_key": "iaa_runtime",
        "type": "folder",
        "uri": str(controller_root),
    }:
        raise ValueError("IAA adapter declared input must bind the exact /iaa-runtime source")
    patches_inputs = (
        [
            item
            for item in declared
            if isinstance(item, dict)
            and item.get("spec_key") == "compatibility_patches"
        ]
        if isinstance(declared, list)
        else []
    )
    if len(patches_inputs) != 1 or patches_inputs[0] != {
        "spec_key": "compatibility_patches",
        "type": "folder",
        "uri": patches_source,
    }:
        raise ValueError("IAA adapter declared input must bind the exact /patches source")
    if not isinstance(bundle.get("action"), str) or not bundle["action"].startswith(
        f"deft-iaa-{name}-"
    ):
        raise ValueError("IAA adapter action identity is invalid")
    _validate_adapter_evidence(payload)


def _is_adapter_request(payload: dict[str, Any], bundle: dict[str, Any]) -> bool:
    return payload.get("name") in IAA_ADAPTER_ACTIONS or bundle.get("network_arch") == "iaa-adapter"


def load_request(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("--request must be an existing absolute regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("platform") != "brev":
        raise ValueError("action request must be an object for platform=brev")
    expected_digest = payload.get("request_sha256")
    unsigned = dict(payload)
    unsigned.pop("request_sha256", None)
    if not isinstance(expected_digest, str) or _sha256_json(unsigned) != expected_digest:
        raise ValueError("action request digest does not match its content")
    bundle = payload.get("spec_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("action request spec_bundle must be an object")
    if bundle.get("nodes") not in (None, 1):
        raise ValueError("Brev supports only one compute node")
    shape = bundle.get("compute_shape")
    gpu_ids = payload.get("gpu_ids")
    if (
        not isinstance(shape, dict)
        or not isinstance(shape.get("gpus"), int)
        or isinstance(shape.get("gpus"), bool)
        or not isinstance(gpu_ids, list)
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in gpu_ids
        )
        or len(set(gpu_ids)) != len(gpu_ids)
        or len(gpu_ids) != shape["gpus"]
    ):
        raise ValueError("request gpu_ids must be unique and match compute_shape.gpus")
    if shape["gpus"] == 0:
        _validate_adapter_request(payload, bundle)
    elif not gpu_ids:
        raise ValueError("TAO actions require at least one explicit GPU id")
    elif _is_adapter_request(payload, bundle):
        raise ValueError("IAA adapter actions must use compute_shape.gpus=0 and gpu_ids=[]")
    elif bundle.get("command") == "python3":
        raise ValueError("arbitrary Python is not an accepted Brev TAO action")
    if not isinstance(payload.get("record_image"), str) or not payload["record_image"]:
        raise ValueError("request record_image must be non-empty")
    for key in ("command", "args"):
        value = bundle.get(key)
        if key == "command":
            valid = isinstance(value, str) and bool(value)
        else:
            valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
        if not valid:
            raise ValueError(f"request spec_bundle.{key} is invalid")
    return payload


def _mount_mapping(raw: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in raw:
        target, separator, source = item.partition("=")
        if not separator or not target.startswith("/") or not source.startswith("/"):
            raise ValueError("--mount must be COMPUTE_TARGET=/absolute/remote/source")
        if target in mapping:
            raise ValueError(f"duplicate remote mount target: {target}")
        for name, value in (("target", target), ("source", source)):
            if any(character in value for character in ("\x00", "\n", "\r", ",")):
                raise ValueError(f"remote mount {name} contains an unsupported character")
            if ".." in pathlib.PurePosixPath(value).parts:
                raise ValueError(f"remote mount {name} must not traverse '..'")
        mapping[target] = source
    return mapping


def validate_mounts(request: dict[str, Any], raw: Sequence[str]) -> list[dict[str, Any]]:
    mapping = _mount_mapping(raw)
    declared = request.get("mounts")
    if not isinstance(declared, list) or not declared:
        raise ValueError("request mounts must be a non-empty list")
    targets = [item.get("target") for item in declared if isinstance(item, dict)]
    if len(targets) != len(declared) or any(not isinstance(item, str) for item in targets):
        raise ValueError("request contains an invalid mount")
    if len(set(targets)) != len(targets):
        raise ValueError("request contains duplicate mount targets")
    if set(mapping) != set(targets):
        missing = sorted(set(targets) - set(mapping))
        extra = sorted(set(mapping) - set(targets))
        raise ValueError(f"remote mount mapping mismatch; missing={missing}, extra={extra}")
    return [
        {
            "source": mapping[item["target"]],
            "target": item["target"],
            "read_only": item.get("read_only") is True,
        }
        for item in declared
    ]


def verify_remote_adapter_snapshots(
    instance: str, request: dict[str, Any], mounts: Sequence[dict[str, Any]]
) -> None:
    """Fail closed unless both staged executable trees match signed manifests."""
    if not _is_adapter_request(request, request["spec_bundle"]):
        return
    checks = (
        (IAA_ADAPTER_RUNTIME, "controller", "controller_snapshot"),
        ("/patches", "patches", "patches_snapshot"),
    )
    for target, kind, snapshot_field in checks:
        selected = [item for item in mounts if item.get("target") == target]
        if len(selected) != 1 or selected[0].get("read_only") is not True:
            raise ValueError(f"staged IAA {kind} mount is missing or writable")
        source = str(
            _absolute_path(selected[0].get("source"), f"remote {kind} source")
        )
        probe_source = source
        if kind == "controller":
            probe_source = str(
                _remove_suffix(
                    pathlib.PurePosixPath(source),
                    IAA_CONTROLLER_RUNTIME_RELATIVE,
                    "remote runtime source",
                )
            )
        completed = run_remote(
            instance,
            _remote_python_command(REMOTE_SNAPSHOT_CODE, probe_source, kind),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Brev IAA {kind} snapshot probe failed: "
                + _stderr(completed).strip()
            )
        markers = [
            line[len(SNAPSHOT_MARKER) :]
            for line in _stdout(completed).splitlines()
            if line.startswith(SNAPSHOT_MARKER)
        ]
        try:
            observed = json.loads(markers[0]) if len(markers) == 1 else None
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Brev IAA {kind} snapshot probe returned invalid JSON") from exc
        expected = request[snapshot_field]
        expected_manifest = {
            "entries": expected["entries"],
            "sha256": expected["sha256"],
        }
        if kind == "controller":
            expected_manifest["runtime_sha256"] = request["runtime_sha256"]
        if observed != expected_manifest:
            raise RuntimeError(
                f"staged Brev IAA {kind} tree does not match request.{snapshot_field}"
            )


def _validate_job_id(job_id: str) -> str:
    if not SAFE_JOB_ID.fullmatch(job_id):
        raise ValueError("job id contains unsupported characters")
    return job_id


def _identity_from_output(output: str) -> tuple[int, int, str, list[int]]:
    lines = [line for line in output.splitlines() if line.startswith(IDENTITY_MARKER)]
    if len(lines) != 1:
        raise RuntimeError("Brev identity probe did not return one marker line")
    fields = lines[0][len(IDENTITY_MARKER) :].split(":")
    if len(fields) != 4:
        raise RuntimeError("Brev identity probe returned malformed output")
    uid, gid = int(fields[0]), int(fields[1])
    username = fields[2]
    groups = [int(item) for item in fields[3].split(",") if item]
    if uid == 0:
        raise RuntimeError("refusing a writable Brev container launch as remote UID 0")
    if not username or any(character.isspace() for character in username):
        raise RuntimeError("Brev identity probe returned an invalid username")
    return uid, gid, username, groups


def identity(instance: str) -> tuple[int, int, str, list[int]]:
    command = (
        "printf '" + IDENTITY_MARKER + "%s:%s:%s:' \"$(id -u)\" \"$(id -g)\" "
        "\"$(id -un)\"; id -G | tr ' ' ','; printf '\\n'"
    )
    completed = run_remote(instance, command)
    if completed.returncode != 0:
        raise RuntimeError("Brev identity probe failed")
    return _identity_from_output(_stdout(completed))


def build_submit_command(
    request: dict[str, Any],
    job_id: str,
    mounts: list[dict[str, Any]],
    *,
    uid: int,
    gid: int,
    username: str,
    groups: Sequence[int],
) -> str:
    job_id = _validate_job_id(job_id)
    if uid <= 0 or gid < 0:
        raise ValueError("remote uid must be non-root and gid must be non-negative")
    environment = request.get("environment")
    forward = request.get("forward_env")
    if not isinstance(environment, dict) or not isinstance(forward, list):
        raise ValueError("request environment/forward_env is invalid")
    argv = [
        "docker",
        "run",
        "-d",
        "--name",
        job_id,
        "--label",
        f"tao-job={job_id}",
        "--label",
        f"tao-request-sha256={request['request_sha256']}",
    ]
    if _is_adapter_request(request, request["spec_bundle"]):
        argv.extend(
            [
                "--label",
                f"tao-action={request['name']}",
                "--label",
                f"tao-runtime-sha256={request['runtime_sha256']}",
            ]
        )
    if request["gpu_ids"]:
        argv.extend(
            ["--gpus", '"device=' + ",".join(str(item) for item in request["gpu_ids"]) + '"']
        )
    argv.extend(["--ipc=host", "--user", f"{uid}:{gid}"])
    for group in sorted(set(groups) - {gid}):
        if group < 0:
            raise ValueError("remote supplementary groups must be non-negative")
        argv.extend(["--group-add", str(group)])
    for mount in mounts:
        option = f"type=bind,src={mount['source']},dst={mount['target']}"
        if mount["read_only"]:
            option += ",readonly"
        argv.extend(["--mount", option])
    combined_environment = {**environment, "USER": username, "LOGNAME": username}
    for name, value in sorted(combined_environment.items()):
        if not SAFE_ENV_NAME.fullmatch(name) or not isinstance(value, str):
            raise ValueError("request contains an invalid non-secret environment entry")
        argv.extend(["-e", f"{name}={value}"])
    for name in forward:
        if not isinstance(name, str) or not SAFE_ENV_NAME.fullmatch(name):
            raise ValueError("request forward_env contains an invalid variable name")
        argv.extend(["-e", name])
    bundle = request["spec_bundle"]
    argv.extend([request["record_image"], bundle["command"], *bundle["args"]])
    return shlex.join(argv)


def _stdout(completed: Any) -> str:
    value = completed.stdout or b""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _stderr(completed: Any) -> str:
    value = completed.stderr or b""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _inspect(
    instance: str, job_id: str, request_sha256: str | None = None
) -> dict[str, Any] | None:
    completed = run_remote(instance, shlex.join(["docker", "inspect", job_id]))
    if completed.returncode != 0:
        error = _stderr(completed).lower()
        if "no such object" in error or "no such container" in error:
            return None
        raise RuntimeError("Brev docker inspect failed: " + _stderr(completed).strip())
    payload = json.loads(_stdout(completed))
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeError("Brev docker inspect returned malformed JSON")
    labels = payload[0].get("Config", {}).get("Labels", {})
    if not isinstance(labels, dict) or labels.get("tao-job") != job_id:
        raise RuntimeError("remote container exists but is not owned by this TAO job")
    if request_sha256 is not None and labels.get("tao-request-sha256") != request_sha256:
        raise RuntimeError("remote container exists but belongs to a different action request")
    return payload[0]


def submit(args: argparse.Namespace) -> int:
    request = load_request(args.request)
    job_id = _validate_job_id(args.job_id)
    existing = (
        _inspect(args.instance, job_id, request["request_sha256"])
        if args.reconcile
        else _inspect(args.instance, job_id)
    )
    if existing is not None:
        if not args.reconcile:
            raise RuntimeError(
                "owned remote container already exists; reconcile instead of resubmitting"
            )
        state = existing.get("State") if isinstance(existing.get("State"), dict) else {}
        native = state.get("Status")
        mapped = (
            "RUNNING" if native in {"created", "restarting", "running", "paused"}
            else "COMPLETE" if native == "exited" and state.get("ExitCode") == 0
            else "ERROR" if native == "exited"
            else "UNKNOWN"
        )
        print(json.dumps({
            "backend_ref": f"{args.instance}/{job_id}", "status": mapped,
            "native_state": native, "reconciled": True,
        }, sort_keys=True))
        return 0
    mounts = validate_mounts(request, args.mount)
    verify_remote_adapter_snapshots(args.instance, request, mounts)
    uid, gid, username, groups = identity(args.instance)
    command = build_submit_command(
        request, job_id, mounts, uid=uid, gid=gid, username=username, groups=groups
    )
    forwarded: dict[str, str] = {}
    for name in request["forward_env"]:
        value = os.environ.get(name)
        if value is None or not value:
            raise RuntimeError(f"approved forwarded variable is unset: {name}")
        forwarded[name] = value
    completed = run_remote(args.instance, command, environment=forwarded, timeout=600)
    if completed.returncode != 0:
        sys.stderr.write(_stderr(completed))
        return int(completed.returncode)
    if not args.json:
        sys.stdout.write(_stdout(completed))
        sys.stderr.write(_stderr(completed))
        return 0
    native_id = _stdout(completed).strip().splitlines()[-1] if _stdout(completed).strip() else None
    print(json.dumps({
        "backend_ref": f"{args.instance}/{job_id}", "status": "RUNNING",
        "native_id": native_id, "reconciled": False,
    }, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    job_id = _validate_job_id(args.job_id)
    payload = _inspect(args.instance, job_id)
    if payload is None:
        print(json.dumps({"status": "UNKNOWN", "native_state": "missing"}))
        return 0
    state = payload.get("State")
    if not isinstance(state, dict):
        raise RuntimeError("remote container State is missing")
    native = state.get("Status")
    exit_code = state.get("ExitCode")
    if native in {"created", "restarting"}:
        mapped = "PENDING"
    elif native in {"running", "paused"}:
        mapped = "RUNNING"
    elif native == "exited":
        mapped = "COMPLETE" if exit_code == 0 else "ERROR"
    else:
        mapped = "UNKNOWN"
    print(json.dumps({"status": mapped, "native_state": native, "exit_code": exit_code}))
    return 0


def logs(args: argparse.Namespace) -> int:
    job_id = _validate_job_id(args.job_id)
    if _inspect(args.instance, job_id) is None:
        raise RuntimeError("remote TAO container is missing")
    completed = run_remote(
        args.instance,
        shlex.join(["docker", "logs", "--tail", str(args.tail), job_id]),
    )
    sys.stdout.write(_stdout(completed))
    sys.stderr.write(_stderr(completed))
    return int(completed.returncode)


def cancel(args: argparse.Namespace) -> int:
    if not args.confirm:
        raise ValueError("cancel requires --confirm after user approval")
    job_id = _validate_job_id(args.job_id)
    if _inspect(args.instance, job_id) is None:
        raise RuntimeError("remote TAO container is missing")
    completed = run_remote(args.instance, shlex.join(["docker", "rm", "-f", job_id]))
    if completed.returncode != 0:
        sys.stderr.write(_stderr(completed))
        return int(completed.returncode)
    if args.json:
        print(json.dumps({
            "backend_ref": f"{args.instance}/{job_id}", "status": "CANCELED",
            "native_state": "removed",
        }, sort_keys=True))
    else:
        sys.stdout.write(_stdout(completed))
        sys.stderr.write(_stderr(completed))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="verb", required=True)
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--request", type=pathlib.Path, required=True)
    submit_parser.add_argument("--instance", required=True)
    submit_parser.add_argument("--job-id", required=True)
    submit_parser.add_argument("--mount", action="append", default=[])
    submit_parser.add_argument("--json", action="store_true")
    submit_parser.add_argument(
        "--reconcile",
        action="store_true",
        help="adopt only an exact job/request-owned container after a lost response",
    )
    for verb in ("status", "logs", "cancel"):
        child = subparsers.add_parser(verb)
        child.add_argument("--instance", required=True)
        child.add_argument("--job-id", required=True)
        if verb == "logs":
            child.add_argument("--tail", type=int, default=200)
        if verb == "cancel":
            child.add_argument("--confirm", action="store_true")
            child.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return {"submit": submit, "status": status, "logs": logs, "cancel": cancel}[
            args.verb
        ](args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"brev action failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
