#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render one producer action request as a single-pod Kubernetes Job.

The data mover first mirrors each distinct request ``mounts[].source`` into a
job-owned directory on one PVC.  Its receipt records the relative PVC path for
each source.  This renderer preserves every producer-declared target and access
mode, including multiple target aliases for the same source, by mounting that
PVC subPath once per target.

Credentials are deliberately not accepted as values.  Each ``forward_env``
name is projected from the same key of an optional, pre-created Secret; keys
that the producer did not approve are never imported.  Registry authentication
is an independent optional image-pull Secret.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import posixpath
import re
import sys
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_TEMPLATE = REPO_ROOT / "templates/k8s/action-job.yaml.tmpl"
MARKER_RE = re.compile(r"@@[A-Z][A-Z0-9_]*@@")
ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DNS_LABEL_RE = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")
DNS_SUBDOMAIN_RE = re.compile(
    r"[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?"
)
QUANTITY_RE = re.compile(r"[1-9][0-9]*(?:Ki|Mi|Gi|Ti|Pi|Ei)")
MAX_JOB_NAME = 52
REQUEST_SCHEMA_VERSION = "1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
IAA_WORKFLOW = "tao-run-deft-iaa"
IAA_ADAPTER_ACTIONS = frozenset({
    "dataset_rebuild", "dataset_materialize", "gap_analysis",
    "mining_postprocess", "history_select", "visualize_prepare",
    "visualize_finish", "eval_config", "train_config",
    "publish_checkpoint", "iteration_summary", "metric_parse", "report",
})
IAA_ADAPTER_ENVIRONMENT = {
    "HOME": "/tmp",
    "PYTHONPATH": "/patches",
    "HF_HOME": "/cache/huggingface",
    "XDG_CACHE_HOME": "/cache",
    "IAA_COMPUTE_FRAME": "kubernetes",
}
IAA_VISUALIZE_THREAD_CAPS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
IAA_CONTROLLER_RUNTIME_RELATIVE = pathlib.PurePosixPath(
    "applications/tao-run-deft-iaa/scripts"
)


class RenderError(ValueError):
    """The action request or staging receipt cannot be rendered safely."""


def _load_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RenderError(f"{label} JSON root must be an object")
    return payload


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RenderError(f"{label} must be a non-empty string without NUL")
    return value


def _string_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise RenderError(f"{label} must be a string without NUL")
    return value


def _dns_name(value: str, label: str, *, label_only: bool = False) -> str:
    pattern = DNS_LABEL_RE if label_only else DNS_SUBDOMAIN_RE
    maximum = 63 if label_only else 253
    if len(value) > maximum or pattern.fullmatch(value) is None:
        kind = "DNS label" if label_only else "DNS subdomain"
        raise RenderError(f"{label} must be a valid Kubernetes {kind}")
    return value


def kubernetes_job_name(job_id: str) -> str:
    """Map an immutable job-record id to a stable Kubernetes-safe name.

    Job-record components may contain underscores, uppercase letters, dots, or
    enough text to exceed Kubernetes' practical Job-name length. Preserve an
    already-safe short id; otherwise append a digest so normalization and
    truncation cannot merge two record ids into one backend object.
    """
    original = _nonempty_string(job_id, "job id")
    if len(original) <= MAX_JOB_NAME and DNS_LABEL_RE.fullmatch(original):
        return original
    slug = re.sub(r"[^a-z0-9]+", "-", original.lower()).strip("-")
    if not slug:
        slug = "tao-job"
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:10]
    prefix = slug[: MAX_JOB_NAME - len(digest) - 1].rstrip("-") or "tao-job"
    return f"{prefix}-{digest}"


def _absolute_clean_path(value: Any, label: str) -> str:
    path = _nonempty_string(value, label)
    if (
        not path.startswith("/")
        or path == "/"
        or "\\" in path
        or any(ord(ch) < 32 for ch in path)
    ):
        raise RenderError(f"{label} must be an absolute non-root POSIX path")
    normalized = posixpath.normpath(path)
    if normalized != path or "//" in path:
        raise RenderError(f"{label} must be normalized and contain no traversal")
    return path


def _safe_sub_path(value: Any, label: str) -> str:
    path = _nonempty_string(value, label)
    if path.startswith("/") or "\\" in path or any(ord(ch) < 32 for ch in path):
        raise RenderError(f"{label} must be a safe relative POSIX path")
    normalized = posixpath.normpath(path)
    if normalized != path or path in {".", ".."} or path.startswith("../"):
        raise RenderError(f"{label} must be normalized and contain no traversal")
    return path


def _command_bundle(request: dict[str, Any]) -> tuple[str, list[str], str, int]:
    bundle = request.get("spec_bundle")
    if not isinstance(bundle, dict):
        raise RenderError("request.spec_bundle must be an object")
    if bundle.get("mode") != "args":
        raise RenderError("Kubernetes action rendering requires spec_bundle.mode=args")
    command = _nonempty_string(bundle.get("command"), "spec_bundle.command")
    raw_args = bundle.get("args")
    if not isinstance(raw_args, list):
        raise RenderError("spec_bundle.args must be an array")
    args = [
        _string_value(value, f"spec_bundle.args[{index}]")
        for index, value in enumerate(raw_args)
    ]
    image = _nonempty_string(bundle.get("image"), "spec_bundle.image")
    workload_image = request.get("workload_image")
    if workload_image is not None and workload_image != image:
        raise RenderError("request.workload_image must equal spec_bundle.image")
    shape = bundle.get("compute_shape")
    if not isinstance(shape, dict):
        raise RenderError("spec_bundle.compute_shape must be an object")
    gpus = shape.get("gpus")
    nodes = shape.get("nodes")
    if not isinstance(gpus, int) or isinstance(gpus, bool) or gpus < 0:
        raise RenderError("spec_bundle.compute_shape.gpus must be a non-negative integer")
    if nodes != 1:
        raise RenderError("single-pod action rendering requires compute_shape.nodes=1")
    return command, args, image, gpus


def _staged_sources(payload: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    if payload.get("schema_version") != "1":
        raise RenderError("staging map schema_version must be '1'")
    rows = payload.get("sources")
    if not isinstance(rows, list) or not rows:
        raise RenderError("staging map sources must be a non-empty array")
    result: dict[str, str] = {}
    digests: dict[str, str] = {}
    sub_paths: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RenderError(f"staging map sources[{index}] must be an object")
        source = _absolute_clean_path(row.get("source"), f"staging sources[{index}].source")
        sub_path = _safe_sub_path(row.get("sub_path"), f"staging sources[{index}].sub_path")
        if source in result:
            raise RenderError(f"staging map repeats source: {source}")
        if sub_path in sub_paths:
            raise RenderError(
                "staging map assigns one PVC subPath to distinct sources: "
                f"{sub_paths[sub_path]} and {source}"
            )
        result[source] = sub_path
        sub_paths[sub_path] = source
        digest = row.get("sha256")
        if digest is not None:
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                raise RenderError(f"staging sources[{index}].sha256 must be lowercase SHA-256")
            digests[source] = digest
    return result, digests


def _canonical_request_sha256(request: dict[str, Any]) -> str:
    unsigned = dict(request)
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


def _validate_snapshot(
    request: dict[str, Any], field: str, root: pathlib.Path,
    staged_digests: dict[str, str],
) -> str:
    approved = request.get(field)
    actual = _snapshot_manifest(root)
    if not isinstance(approved, dict) or approved != actual:
        raise RenderError(f"{field} does not match the complete local snapshot")
    digest = actual["sha256"]
    if staged_digests.get(str(root)) != digest:
        raise RenderError(f"staging receipt does not bind {field} SHA-256")
    return digest


def _validate_zero_gpu_adapter(
    request: dict[str, Any], staged: dict[str, str], staged_digests: dict[str, str],
) -> str:
    """Validate the only signed request class allowed to render with zero GPUs."""
    if request.get("workflow") != IAA_WORKFLOW:
        raise RenderError("zero-GPU actions require the signed IAA workflow contract")
    name = request.get("name")
    if name not in IAA_ADAPTER_ACTIONS:
        raise RenderError("zero-GPU action is not an allowlisted IAA adapter")
    request_digest = request.get("request_sha256")
    if (not isinstance(request_digest, str) or SHA256_RE.fullmatch(request_digest) is None
            or request_digest != _canonical_request_sha256(request)):
        raise RenderError("zero-GPU adapter request signature is missing or invalid")
    runtime_digest = request.get("runtime_sha256")
    if not isinstance(runtime_digest, str) or SHA256_RE.fullmatch(runtime_digest) is None:
        raise RenderError("zero-GPU adapter requires a runtime_sha256 binding")
    if request.get("gpu_ids") != []:
        raise RenderError("zero-GPU adapter request must bind gpu_ids=[]")
    if request.get("passed_hf_token") is not False or request.get("forward_env") != []:
        raise RenderError("zero-GPU adapters cannot receive model credentials")
    expected_environment = dict(IAA_ADAPTER_ENVIRONMENT)
    if name == "visualize_finish":
        expected_environment.update(IAA_VISUALIZE_THREAD_CAPS)
    if request.get("environment") != expected_environment:
        raise RenderError(
            "zero-GPU adapter environment must match the exact Kubernetes allowlist"
        )
    bundle = request.get("spec_bundle")
    expected_args = [
        "/iaa-runtime/run_iaa_compute.py", name,
        "--results-dir", "/results", "--label", request.get("label"),
    ]
    if (not isinstance(bundle, dict) or bundle.get("network_arch") != "iaa-adapter"
            or bundle.get("command") != "python3" or bundle.get("args") != expected_args):
        raise RenderError("zero-GPU adapter argv is outside the signed allowlist")
    runtime_mounts = [
        row for row in request.get("mounts", [])
        if isinstance(row, dict) and row.get("target") == "/iaa-runtime"
    ]
    if len(runtime_mounts) != 1 or runtime_mounts[0].get("read_only") is not True:
        raise RenderError("zero-GPU adapter requires one read-only /iaa-runtime mount")
    patches_mounts = [
        row for row in request.get("mounts", [])
        if isinstance(row, dict) and row.get("target") == "/patches"
    ]
    if len(patches_mounts) != 1 or patches_mounts[0].get("read_only") is not True:
        raise RenderError("zero-GPU adapter requires one read-only /patches mount")
    runtime_source = pathlib.Path(
        _absolute_clean_path(runtime_mounts[0].get("source"), "IAA runtime source")
    )
    patches_source = pathlib.Path(
        _absolute_clean_path(patches_mounts[0].get("source"), "patches source")
    )
    controller = request.get("controller_snapshot")
    patches = request.get("patches_snapshot")
    if not isinstance(controller, dict):
        raise RenderError("controller_snapshot must be an object")
    if not isinstance(patches, dict):
        raise RenderError("patches_snapshot must be an object")
    controller_root = pathlib.Path(
        _absolute_clean_path(controller.get("root"), "controller_snapshot.root")
    )
    patches_root = pathlib.Path(
        _absolute_clean_path(patches.get("root"), "patches_snapshot.root")
    )
    if runtime_source != controller_root / IAA_CONTROLLER_RUNTIME_RELATIVE:
        raise RenderError("/iaa-runtime source must be derived from controller_snapshot.root")
    if patches_source != patches_root:
        raise RenderError("/patches source must equal patches_snapshot.root")
    _validate_snapshot(
        request, "controller_snapshot", controller_root, staged_digests
    )
    _validate_snapshot(request, "patches_snapshot", patches_root, staged_digests)
    if str(controller_root) not in staged:
        raise RenderError("staging map does not contain controller_snapshot.root")
    if str(runtime_source) in staged:
        raise RenderError(
            "staging map must derive /iaa-runtime from controller_snapshot.root"
        )
    controller_sub_path = staged.pop(str(controller_root))
    staged[str(runtime_source)] = posixpath.join(
        controller_sub_path, IAA_CONTROLLER_RUNTIME_RELATIVE.as_posix()
    )
    if _python_tree_sha256(runtime_source / "iaa_deft") != runtime_digest:
        raise RenderError("IAA runtime source does not match request.runtime_sha256")
    return runtime_digest


def _mounts(
    request: dict[str, Any], staged: dict[str, str]
) -> tuple[list[dict[str, Any]], list[tuple[pathlib.PurePosixPath, bool]]]:
    rows = request.get("mounts")
    if not isinstance(rows, list) or not rows:
        raise RenderError("request.mounts must be a non-empty array")
    rendered: list[dict[str, Any]] = [
        {"name": "dshm", "mountPath": "/dev/shm", "readOnly": False}
    ]
    used_sources: set[str] = set()
    targets: set[str] = {"/dev/shm"}
    source_modes: list[tuple[pathlib.PurePosixPath, bool]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RenderError(f"request.mounts[{index}] must be an object")
        source = _absolute_clean_path(row.get("source"), f"mounts[{index}].source")
        target = _absolute_clean_path(row.get("target"), f"mounts[{index}].target")
        read_only = row.get("read_only")
        if not isinstance(read_only, bool):
            raise RenderError(f"mounts[{index}].read_only must be boolean")
        if source not in staged:
            raise RenderError(f"mount source has no staged PVC subPath: {source}")
        if target in targets:
            raise RenderError(f"request repeats Kubernetes mount target: {target}")
        targets.add(target)
        used_sources.add(source)
        source_modes.append((pathlib.PurePosixPath(source), read_only))
        rendered.append(
            {
                "name": "workspace",
                "mountPath": target,
                "subPath": staged[source],
                "readOnly": read_only,
            }
        )
    extras = sorted(set(staged) - used_sources)
    if extras:
        raise RenderError("staging map contains undeclared mount sources: " + ", ".join(extras))
    return rendered, source_modes


def _require_writable_outputs(
    request: dict[str, Any], source_modes: list[tuple[pathlib.PurePosixPath, bool]]
) -> None:
    outputs = request.get("fresh_outputs", [])
    if not isinstance(outputs, list):
        raise RenderError("request.fresh_outputs must be an array")
    for index, raw in enumerate(outputs):
        output = pathlib.PurePosixPath(
            _absolute_clean_path(raw, f"fresh_outputs[{index}]")
        )
        writable = False
        for source, read_only in source_modes:
            try:
                output.relative_to(source)
            except ValueError:
                continue
            if not read_only:
                writable = True
                break
        if not writable:
            raise RenderError(
                f"fresh output is not covered by a writable declared mount: {output}"
            )


def _environment(
    request: dict[str, Any], credential_secret: str | None
) -> list[dict[str, Any]]:
    raw_environment = request.get("environment", {})
    if not isinstance(raw_environment, dict):
        raise RenderError("request.environment must be an object")
    environment: list[dict[str, Any]] = []
    for name in sorted(raw_environment):
        if ENV_NAME_RE.fullmatch(name) is None:
            raise RenderError(f"invalid environment variable name: {name!r}")
        value = _string_value(raw_environment[name], f"environment.{name}")
        environment.append({"name": name, "value": value})

    raw_forward = request.get("forward_env", [])
    if not isinstance(raw_forward, list):
        raise RenderError("request.forward_env must be an array")
    forward: list[str] = []
    for index, raw in enumerate(raw_forward):
        name = _nonempty_string(raw, f"forward_env[{index}]")
        if ENV_NAME_RE.fullmatch(name) is None:
            raise RenderError(f"invalid forwarded environment variable name: {name!r}")
        if name in forward:
            raise RenderError(f"request.forward_env repeats {name}")
        if name in raw_environment:
            raise RenderError(f"credential {name} must not be present in request.environment")
        forward.append(name)
    if forward and credential_secret is None:
        raise RenderError(
            "request.forward_env is non-empty but no --credential-secret was supplied"
        )
    if not forward and credential_secret is not None:
        raise RenderError(
            "--credential-secret must be omitted when request.forward_env is empty"
        )
    if credential_secret is not None:
        environment.extend(
            {
                "name": name,
                "valueFrom": {
                    "secretKeyRef": {"name": credential_secret, "key": name}
                },
            }
            for name in forward
        )
    return environment


def render_action_job(
    request: dict[str, Any],
    staging_map: dict[str, Any],
    *,
    job_id: str,
    namespace: str,
    pvc_claim: str,
    credential_secret: str | None = None,
    image_pull_secret: str | None = None,
    ttl_seconds: int = 3600,
    shm_size: str = "16Gi",
    template_path: pathlib.Path = DEFAULT_TEMPLATE,
) -> str:
    """Validate inputs and return a complete Kubernetes Job manifest."""
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise RenderError(
            f"request.schema_version must be {REQUEST_SCHEMA_VERSION!r}"
        )
    if request.get("platform") != "kubernetes":
        raise RenderError("request.platform must be 'kubernetes'")
    job_id = _nonempty_string(job_id, "job id")
    job_name = kubernetes_job_name(job_id)
    namespace = _dns_name(namespace, "namespace", label_only=True)
    pvc_claim = _dns_name(pvc_claim, "PVC claim")
    if credential_secret is not None:
        credential_secret = _dns_name(credential_secret, "credential secret")
    if image_pull_secret is not None:
        image_pull_secret = _dns_name(image_pull_secret, "image pull secret")
    if (
        not isinstance(ttl_seconds, int)
        or isinstance(ttl_seconds, bool)
        or not 0 <= ttl_seconds <= 604800
    ):
        raise RenderError("ttl_seconds must be an integer from 0 through 604800")
    if QUANTITY_RE.fullmatch(shm_size) is None:
        raise RenderError("shm_size must be a positive binary Kubernetes quantity such as 16Gi")

    command, args, image, gpus = _command_bundle(request)
    staged, staged_digests = _staged_sources(staging_map)
    adapter_runtime_sha256: str | None = None
    if gpus == 0:
        adapter_runtime_sha256 = _validate_zero_gpu_adapter(
            request, staged, staged_digests
        )
    elif request.get("workflow") == IAA_WORKFLOW and request.get("name") in IAA_ADAPTER_ACTIONS:
        raise RenderError("allowlisted IAA adapters must request exactly zero GPUs")
    volume_mounts, source_modes = _mounts(request, staged)
    _require_writable_outputs(request, source_modes)
    environment = _environment(request, credential_secret)
    image_pull_secrets = (
        [{"name": image_pull_secret}] if image_pull_secret is not None else []
    )

    values: dict[str, Any] = {
        "JOB_NAME_JSON": job_name,
        "ANNOTATIONS_JSON": {
            "tao.nvidia.com/job-record-id": job_id,
            **({"tao.nvidia.com/runtime-sha256": adapter_runtime_sha256}
               if adapter_runtime_sha256 is not None else {}),
        },
        "NAMESPACE_JSON": namespace,
        "TTL_SECONDS_JSON": ttl_seconds,
        "IMAGE_PULL_SECRETS_JSON": image_pull_secrets,
        "IMAGE_JSON": image,
        "COMMAND_JSON": [command],
        "ARGS_JSON": args,
        "RESOURCES_JSON": (
            {"limits": {"nvidia.com/gpu": str(gpus)}} if gpus else {}
        ),
        "ENV_JSON": environment,
        "VOLUME_MOUNTS_JSON": volume_mounts,
        "SHM_SIZE_JSON": shm_size,
        "PVC_CLAIM_JSON": pvc_claim,
    }
    try:
        rendered = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RenderError(f"cannot read Kubernetes action template: {exc}") from exc
    for marker, value in values.items():
        rendered = rendered.replace(f"@@{marker}@@", json.dumps(value, separators=(",", ":")))
    remaining = MARKER_RE.findall(rendered)
    if remaining:
        raise RenderError(f"unresolved template markers: {sorted(set(remaining))}")
    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    name = commands.add_parser(
        "name", help="print the deterministic Kubernetes name for a job-record id"
    )
    name.add_argument("--job-id", required=True)

    render = commands.add_parser("render", help="render a complete Job manifest")
    render.add_argument("--request", required=True, type=pathlib.Path)
    render.add_argument("--staging-map", required=True, type=pathlib.Path)
    render.add_argument("--job-id", required=True)
    render.add_argument(
        "--namespace", default=os.environ.get("TAO_K8S_NAMESPACE", "default")
    )
    render.add_argument("--pvc-claim", required=True)
    render.add_argument("--credential-secret")
    render.add_argument("--image-pull-secret")
    render.add_argument("--ttl-seconds", type=int, default=3600)
    render.add_argument("--shm-size", default="16Gi")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.operation == "name":
            print(kubernetes_job_name(args.job_id))
            return 0
        request = _load_object(args.request, "action request")
        staging_map = _load_object(args.staging_map, "staging map")
        manifest = render_action_job(
            request,
            staging_map,
            job_id=args.job_id,
            namespace=args.namespace,
            pvc_claim=args.pvc_claim,
            credential_secret=args.credential_secret,
            image_pull_secret=args.image_pull_secret,
            ttl_seconds=args.ttl_seconds,
            shm_size=args.shm_size,
        )
    except RenderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
