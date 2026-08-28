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

import json
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



def _toml_value(value: Any) -> str:
    """Serialize one TOML scalar or array. Booleans BEFORE ints on purpose:
    bool is a subclass of int in Python, so the order matters."""
    import json as _json

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return _json.dumps(value)          # TOML basic strings match JSON escaping
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"no TOML representation for {type(value).__name__}")


def dumps_toml(spec: dict[str, Any], _prefix: str = "") -> str:
    """Minimal TOML writer for a nested spec dict.

    Python ships tomllib to READ toml and nothing to write it, and this bank
    keeps its dependency set to pyyaml/jsonschema. Cosmos-RL specs are
    config_format=toml, so without this every cosmos train stage would fail at
    render. The supported shape is exactly what those specs contain: scalars,
    homogeneous arrays, and nested tables.

    Scalars are emitted before sub-tables at each level -- a key written after a
    [table] header would silently belong to that table instead of its parent,
    which is the classic way a hand-rolled TOML writer corrupts a config.
    """
    scalars, tables = [], []
    for key, value in spec.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            scalars.append(f"{key} = {_toml_value(value)}")
    out = "\n".join(scalars)
    for key, value in tables:
        name = f"{_prefix}{key}"
        body = dumps_toml(value, f"{name}.")
        out += f"\n\n[{name}]\n{body}" if out else f"[{name}]\n{body}"
    return out.strip() + "\n"


# Commands that are SHELL SCRIPTS, not argv ---------------------------------
# A model skill may own a command that is a script rather than a program plus
# arguments -- cosmos-rl's train computes a hook path from cosmos_rl.__file__,
# tests it, then runs it. Splitting that on whitespace and re-quoting each token
# produces `exec "hook=$(...)"`, i.e. an attempt to run a binary named after the
# whole first line. It must go to a shell intact.
SHELL_META = ("\n", ";", "&&", "||", "$(", "`", "|", ">", "<")


def is_shell_script(command: str) -> bool:
    return any(token in command for token in SHELL_META)


def _claim_relative(uri: str, mount_path: str) -> str:
    """The part of `uri` inside the claim. Empty when the uri IS the mount root.

    An empty subPath is not the same as no subPath: kubernetes treats "" as the
    volume root, which happens to be right when the input is the whole claim
    and silently wrong the moment it is not. Emit the key only when there is a
    sub-path to name.
    """
    return str(uri)[len(str(mount_path).rstrip("/")):].lstrip("/")


def config_file(bundle: dict[str, Any], job_id: str, config_root: str) -> tuple[str, str]:
    """Serialize a mode=config spec and return (path, content).

    The contract says the CONSUMER writes the spec file and substitutes its
    compute-frame path into `{config_path}`. No renderer did, so every
    mode=config bundle -- train, evaluate, inference, rca and all three mining
    stages -- reached its container with a literal `{config_path}` argument and
    failed on a file of that name.

    The file goes into the rendered `files` map, so it is placed by the same
    mechanism as every other rendered file: locally for docker and kubernetes,
    over ssh for slurm. That keeps ONE placement path rather than a per-platform
    write.
    """
    import json as _json

    fmt = str(bundle.get("config_format") or "yaml").lower()
    spec = bundle.get("spec") or {}
    if fmt == "json":
        return f"{config_root.rstrip('/')}/configs/{job_id}.json", _json.dumps(spec, indent=2)
    if fmt == "toml":
        return f"{config_root.rstrip('/')}/configs/{job_id}.toml", dumps_toml(spec)
    import yaml as _yaml

    return (f"{config_root.rstrip('/')}/configs/{job_id}.yaml",
            _yaml.safe_dump(spec, sort_keys=False))


def substitute_config_path(tokens: list[str], config_path: str) -> list[str]:
    return [t.replace("{config_path}", config_path) for t in tokens]



def render(bundle: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Bundle -> a rendered Job manifest plus `kubectl apply`."""
    # A Job name must be an RFC1123 subdomain: lowercase alphanumerics, '-'
    # and '.' only. Stage actions carry underscores (mining.embed_pool), which
    # the API server rejects at apply time with a validation error that names
    # the field rather than the offending character.
    job_id = re.sub(r"[^a-z0-9.-]+", "-", str(ctx["job_id"]).lower()).strip("-.")
    if not job_id:
        raise ValueError(f"job_id {ctx['job_id']!r} has no RFC1123-safe characters")
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

    command_text = str(bundle["command"])
    config_path_for_command = ""
    tokens = [*shlex.split(command_text), *(bundle.get("args") or [])]
    extra_files: dict[str, str] = {}
    if bundle.get("mode") == "config":
        # A pod sees ONE bound volume, so the spec has to land under it or the
        # container cannot read the path we substitute.
        config_path, content = config_file(
            bundle, job_id, str(ctx.get("job_dir") or results_dir))
        if not config_path.startswith(str(mount_path).rstrip("/")):
            raise ValueError(
                f"config path {config_path} is outside the mounted volume "
                f"{mount_path!r}; set --ctx job_dir to a path under it, or the "
                "pod cannot read its spec"
            )
        extra_files[config_path] = content
        config_path_for_command = config_path
        tokens = substitute_config_path(tokens, config_path)

    # Create the per-job results dir before the workload writes. docker's
    # prepare() and slurm's sbatch prologue both do this host/cluster-side; on
    # kubernetes the PVC is provisioned empty and NOTHING creates
    # results_dir/<job_id>, so the first output write fails with a
    # FileNotFoundError that names the file, not the missing directory. The
    # template runs /bin/sh -c, so a prefix is safe for both command forms.
    mkdir_prefix = f"mkdir -p {shlex.quote(str(results_dir))} && "
    if is_shell_script(command_text):
        # The template runs this through `/bin/sh -c`, so the script goes in
        # verbatim. Splitting and re-quoting each token would rewrite the
        # script's own quoting and change what it means.
        command = command_text.replace("{config_path}", config_path_for_command)
        command += "".join(" " + shlex.quote(a) for a in (bundle.get("args") or []))
        command = mkdir_prefix + command
    else:
        command = mkdir_prefix + " ".join(shlex.quote(token) for token in tokens)
    substitutions = {
        "JOB_NAME": job_id,
        "IMAGE": bundle["image"],
        "NUM_GPUS": str(int(shape["gpus"])),
        # The template places this inside command: ["/bin/sh", "-c", "@@…@@"],
        # a double-quoted JSON scalar. A quote or backslash in the command --
        # ordinary in a `bash -lc "…"` stage -- would terminate the string and
        # produce an unparseable manifest. json.dumps escapes it; the slice
        # drops the quotes the template already supplies.
        "COMMAND": json.dumps(command)[1:-1],
        "MOUNT_PATH": mount_path,
        # A pod sees ONE bound volume, so an input needing a fixed in-container
        # path becomes an extra volumeMount of the SAME claim with a subPath --
        # the claim-relative portion of its uri. That is how a single PVC can
        # appear at both its natural location and, say, /tao-workspace.
        "EXTRA_MOUNTS": "".join(
            f'\n            - name: data'
            f'\n              mountPath: "{item["target"]}"'
            + (f'\n              subPath: "{_claim_relative(item["uri"], mount_path)}"'
               if _claim_relative(item["uri"], mount_path) else "")
            + ("" if item.get("writable") else '\n              readOnly: true')
            for item in (bundle.get("declared_inputs") or [])
            if item.get("target") and str(item["target"]) != str(item["uri"])
        ),
        # Parity with docker's --user. A TAO image running as its own non-root
        # user cannot write a PVC owned by someone else; fsGroup makes the
        # mounted volume group-writable for the pod.
        "RUN_AS_USER": (
            f'\n        runAsUser: {int(ctx["uid"])}'
            f'\n        runAsGroup: {int(ctx.get("gid", ctx["uid"]))}'
            f'\n        fsGroup: {int(ctx.get("fs_group", ctx.get("gid", ctx["uid"])))}'
            if ctx.get("uid") is not None else ""
        ),
        # Same default as docker's -w: the image WORKDIR is often root-owned
        # (/workspace in cosmos-rl), and runAsUser alone does not fix a
        # relative-path write into it.
        "WORKING_DIR": (
            f'\n          workingDir: "{bundle.get("workdir") or results_dir}"'
        ),
        "RESULTS_DIR": results_dir,
        # Rendered as additional `env:` list entries at the same indentation as
        # TAO_RESULTS_ROOT, so a bundle finds its inputs here exactly as it
        # does on docker and slurm.
        # Per-key credential projection, adopting #141's fail-closed model: a
        # Secret's EXISTENCE is not authorization for every key it holds. Only
        # names the caller forwarded are projected; nothing forwarded means no
        # secret reference at all, so the old failure -- envFrom importing the
        # whole Secret once it exists -- cannot recur.
        "CRED_ENV": "".join(
            f'\n            - name: {name}'
            f'\n              valueFrom:'
            f'\n                secretKeyRef:'
            f'\n                  name: "{ctx.get("cred_secret", f"tao-creds-{job_id}")}"'
            f'\n                  key: {name}'
            for name in (ctx.get("env_passthrough") or [])
        ),
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
        "files": {manifest: body, **extra_files},
        "argv": ["kubectl", "apply", "-n", namespace, "-f", manifest],
        "backend_ref": f"{namespace}/{job_id}",
    }


def status(backend_ref: str, ctx: dict[str, Any]) -> tuple[str, int]:
    """Map Job conditions into the fixed vocabulary."""
    namespace, _, job = backend_ref.partition("/")
    probe = subprocess.run(
        ["kubectl", "get", "job", job, "-n", namespace, "-o",
         # '|' delimited, NOT space. jsonpath renders an unset counter as an
         # empty string, so a failed Job prints " 1 " and .split() collapses it
         # to ["1"] -- which lands in `succeeded` and reports COMPLETE for a
         # job that FAILED. An explicit delimiter keeps the fields positional.
         "jsonpath={.status.succeeded}|{.status.failed}|{.status.active}"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        return "UNKNOWN", 0
    succeeded, failed, active = (probe.stdout.split("|") + ["", "", ""])[:3]
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
