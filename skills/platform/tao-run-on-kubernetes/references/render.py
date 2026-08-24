#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a spec-bundle into a Job manifest, and map Job state.

Contract: `skills/core/tao-launch-workflow/references/bundle-rendering.md`.
Conventions are this skill's own: the packaged single-pod template, a bound
PVC for tier A, and credentials by `envFrom.secretRef` so no value is ever in
the manifest.

Unlike docker and slurm the handle is known before launch — `kubectl apply`
prints a confirmation, not an id — so `backend_ref` is returned directly as
`namespace/job`.
"""

from __future__ import annotations

import pathlib
import re
import shlex
import subprocess
from typing import Any

PLATFORM = "kubernetes"

TEMPLATE = "templates/k8s/single-pod-job.yaml.tmpl"


def prepare(bundle: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """No agent-side pull is possible here — the kubelet pulls on the node.

    Docker and SLURM can hoist the image fetch off the metered resource. On
    kubernetes there is no equivalent: a pod reserves `nvidia.com/gpu` for its
    whole lifetime, so a first-time multi-GB pull is billed GPU-idle time and
    the only real mitigations are cluster-side (pre-warmed nodes, a local
    registry mirror). Pretending to "prepare" here would be theatre.

    What is checkable from the agent is that the reference is well-formed and,
    when a registry is reachable, that it resolves — which turns a typo or a
    missing pull secret into a submit-time error instead of ImagePullBackOff
    discovered after the pod has been scheduled onto a GPU.
    """
    image = bundle["image"]
    notes = ["kubelet pulls on the node; no agent-side pull is possible"]
    if ctx.get("airgap"):
        notes.append("air-gap: the image must already be on the nodes or a local mirror")
        return {"image": image, "notes": notes}
    if ctx.get("verify_image_resolves"):
        probe = subprocess.run(
            ["docker", "manifest", "inspect", image],
            capture_output=True, text=True, check=False,
        )
        if probe.returncode != 0:
            raise ValueError(
                f"{image} does not resolve in the registry: {probe.stderr.strip()}"
            )
        notes.append("manifest resolves")
    return {"image": image, "notes": notes}


def input_env(bundle: dict[str, Any]) -> dict[str, str]:
    """Declared inputs as TAO_INPUT_<SPEC_KEY>, mirroring TAO_RESULTS_ROOT.

    A bundle declares inputs by spec_key; the path the WORKLOAD sees is chosen
    by the platform. Without this a stage command has to name a path directly
    and guess the layout, and a wrong guess does not fail -- the directory is
    simply absent, so the stage reads nothing, writes empty output and exits 0.
    """
    env: dict[str, str] = {}
    for item in bundle.get("declared_inputs") or []:
        key = re.sub(r"[^A-Za-z0-9]+", "_", str(item["spec_key"])).strip("_").upper()
        if key:
            env[f"TAO_INPUT_{key}"] = str(item["uri"])
    return env


def render(bundle: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Bundle -> a rendered Job manifest plus `kubectl apply`."""
    job_id = ctx["job_id"]
    results_dir = ctx["results_dir"]
    namespace = ctx.get("namespace", "default")
    bank = pathlib.Path(ctx["bank"])
    template = (bank / TEMPLATE).read_text(encoding="utf-8")

    shape = bundle["compute_shape"]
    if int(shape["nodes"]) > 1:
        raise ValueError(
            "multi-node bundles need templates/k8s/indexed-job.yaml.tmpl; "
            "render() covers the single-pod template only"
        )

    # A pod sees one bound volume, so every declared input must already live
    # under the mounted claim — there is no per-input bind mount on kubernetes.
    mount_path = ctx.get("mount_path") or results_dir
    for item in bundle.get("declared_inputs") or []:
        uri = str(item["uri"])
        if "://" in uri:
            raise ValueError(
                f"declared_input {item['spec_key']} is {uri!r}; stage it onto the "
                "PVC with tao-data-io (tier A) or use tier C, then declare the "
                "in-pod path"
            )
        if not uri.startswith(mount_path.rstrip("/")):
            raise ValueError(
                f"declared_input {item['spec_key']} is {uri!r}, which is outside "
                f"the mounted volume {mount_path!r}; a pod cannot bind it"
            )

    command = " ".join(
        shlex.quote(token)
        for token in [*shlex.split(bundle["command"]), *(bundle.get("args") or [])]
    )
    substitutions = {
        "JOB_NAME": job_id,
        "IMAGE": bundle["image"],
        "NUM_GPUS": str(int(shape["gpus"])),
        "COMMAND": command,
        "MOUNT_PATH": mount_path,
        "WORKING_DIR": (
            f'\n          workingDir: "{bundle["workdir"]}"'
            if bundle.get("workdir") else ""
        ),
        "RESULTS_DIR": results_dir,
        # Rendered as additional `env:` list entries at the same indentation as
        # TAO_RESULTS_ROOT, so a bundle finds its inputs here exactly as it
        # does on docker and slurm.
        "INPUT_ENV": "".join(
            f'\n            - name: {name}\n              value: "{value}"'
            for name, value in input_env(bundle).items()
        ),
        "PVC_CLAIM": str(ctx.get("pvc_claim", "")),
        "CRED_SECRET": str(ctx.get("cred_secret", f"tao-creds-{job_id}")),
        "IMAGE_PULL_SECRET": str(ctx.get("image_pull_secret", "")),
        "SHM_SIZE": str(ctx.get("shm_size", "8Gi")),
        "TTL_SECONDS": str(ctx.get("ttl_seconds", 86400)),
    }
    body = template
    for name, value in substitutions.items():
        body = body.replace(f"@@{name}@@", value)

    # Only @@UPPER_CASE@@ are real slots; the template header documents the
    # convention with a literal "@@<NAME>@@", which must not trip this check.
    # An empty pull secret must not render as `- name: ""`. Kubernetes rejects
    # a nameless LocalObjectReference, and the whole block is optional, so drop
    # it rather than emit a manifest that fails at apply time.
    if not substitutions["IMAGE_PULL_SECRET"]:
        body = re.sub(
            r"\n\s*imagePullSecrets:\n\s*- name: \"\"\n", "\n", body, count=1
        )

    remaining = sorted(set(re.findall(r"@@[A-Z][A-Z0-9_]*@@", body)))
    if remaining:
        raise ValueError(f"unsubstituted template placeholders: {', '.join(remaining)}")

    manifest = f"{ctx.get('job_dir') or results_dir}/manifests/job_{job_id}.yaml"
    return {
        "files": {manifest: body},
        "argv": ["kubectl", "apply", "-n", namespace, "-f", manifest],
        "backend_ref": f"{namespace}/{job_id}",
    }


def status(backend_ref: str, ctx: dict[str, Any]) -> tuple[str, int]:
    """Map Job conditions into the fixed vocabulary."""
    namespace, _, job = backend_ref.partition("/")
    probe = subprocess.run(
        ["kubectl", "get", "job", job, "-n", namespace, "-o",
         "jsonpath={.status.succeeded} {.status.failed} {.status.active}"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        return "UNKNOWN", 0
    succeeded, failed, active = (probe.stdout.split() + ["", "", ""])[:3]
    if succeeded and int(succeeded or 0) > 0:
        return "COMPLETE", 0
    if failed and int(failed or 0) > 0:
        return "ERROR", 1
    if active and int(active or 0) > 0:
        return "RUNNING", 0
    return "PENDING", 0


def logs(backend_ref: str, ctx: dict[str, Any], tail: int = 200) -> str:
    """Tail the Job's pod logs."""
    namespace, _, job = backend_ref.partition("/")
    probe = subprocess.run(
        ["kubectl", "logs", f"job/{job}", "-n", namespace,
         "--tail", str(int(tail)), "--all-containers"],
        capture_output=True, text=True, check=False,
    )
    return (probe.stdout + probe.stderr).strip()


def cancel(backend_ref: str, ctx: dict[str, Any]) -> bool:
    """Delete the Job, which is the only stop Kubernetes offers.

    Unlike docker stop, this destroys the object status() reads, so afterwards
    status() returns UNKNOWN rather than CANCELED. The caller must mark the
    record CANCELED itself; polling will not converge on its own.
    """
    namespace, _, job = backend_ref.partition("/")
    deleted = subprocess.run(
        ["kubectl", "delete", "job", job, "-n", namespace],
        capture_output=True, text=True, check=False,
    )
    return deleted.returncode == 0
