#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four-verb launcher, shared by every DEFT workflow.

`submit`, `status`, `logs` and `cancel` over a spec-bundle are not
workflow-specific: nothing in them mentions a stage name, a state file, or a
network architecture. They were implemented inside tao-run-deft-aoi's
deft_exec.py, which meant a second workflow could only get them by copying ~400
lines -- and copying is exactly what this bank keeps paying for. DEFT IAA,
Cosmos3 and AOI each ship their own commit_stage.py for the same reason.

So this lives at bank root beside tao_job_record.py and tao_spec_bundle.py, and
each workflow's deft_exec.py delegates to it. A workflow still owns its air-gap
POLICY (which commands it refuses) and its state file; only the launch
mechanism is shared.

Which backend to ask always comes from the job RECORD, never a flag: `open`
wrote the platform and `mark --backend-ref` wrote the handle, so a job can be
awaited, read or cancelled from a session that never launched it. That is the
point of the record-then-launch invariant.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any

OFFLINE_ENV = {
    "AIR_GAPPED": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PIP_NO_INDEX": "1",
}

DOCKER_STATE_VOCAB = {
    "created": "PENDING",
    "restarting": "PENDING",
    "running": "RUNNING",
    "paused": "RUNNING",
}

FETCH_SCHEMES = ("s3://", "hf://", "ngc://", "http://", "https://", "gs://", "azure://")

def _bank() -> pathlib.Path:
    env = os.environ.get("TAO_SKILL_BANK_PATH")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    # This file lives at <bank>/scripts/, not inside a skill: one level up,
    # not four. Getting this wrong resolves a bank that does not exist and the
    # failure surfaces as a missing helper rather than a bad root.
    return pathlib.Path(__file__).resolve().parents[1]

def _bank_module(name: str):
    """Load a bank helper by path, preferring the checkout this file lives in.

    `_bank()` honours TAO_SKILL_BANK_PATH, which points at the *installed* bank
    and may be a different checkout entirely — running from a clone with the env
    var set elsewhere would silently load that other copy of the helper, or fail
    when it does not have it. For a helper this skill is calling directly, the
    code shipped alongside it is the correct one. Loading by path also avoids
    mutating sys.path as a side effect of a library call.
    """
    own = pathlib.Path(__file__).resolve().parent / f"{name}.py"
    candidate = own if own.is_file() else _bank() / "scripts" / f"{name}.py"
    if not candidate.is_file():
        raise ValueError(f"bank helper {name}.py not found (looked in {candidate.parent})")
    spec = importlib.util.spec_from_file_location(name, candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _record(*args: str) -> str:
    script = _bank() / "scripts" / "tao_job_record.py"
    if not script.is_file():
        raise ValueError(
            f"job record helper not found at {script}; set TAO_SKILL_BANK_PATH"
        )
    result = subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise ValueError(f"tao_job_record.py {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()

def _run_index(command: list[str]) -> int | None:
    """Index of the `run` subcommand of a local container runtime, if any."""
    tokens = [_basename(token) for token in command]
    runtimes = [i for i, token in enumerate(tokens) if token in {"docker", "podman"}]
    if not runtimes or "run" not in tokens[runtimes[0] + 1 :]:
        return None
    return tokens.index("run", runtimes[0] + 1)

def platform_renderer(platform: str):
    """Load the chosen platform skill's renderer, by convention not by table.

    A `render_docker`/`render_slurm`/... table here would have to be edited
    before any new platform could run a bundle, which is the registry the
    four-verb contract avoids. `--platform` is an open validated slug, so this
    resolves the slug to the skill that owns it and loads the module it ships.
    An external platform skill conforms by shipping the same file; nothing in
    this workflow changes. Contract:
    skills/core/tao-launch-workflow/references/bundle-rendering.md
    """
    bank = _bank()
    path = bank / "skills/platform" / f"tao-run-on-{platform}" / "references/render.py"
    if not path.is_file():
        raise ValueError(
            f"platform {platform!r} ships no renderer at {path}; see "
            "tao-launch-workflow/references/bundle-rendering.md"
        )
    spec = importlib.util.spec_from_file_location(f"tao_render_{platform}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_bundle(path: pathlib.Path, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read and lint a spec-bundle, then apply execution policy, before rendering."""
    bundle = json.loads(path.expanduser().read_text(encoding="utf-8"))
    problems = _bank_module("tao_spec_bundle").validate(bundle)
    if problems:
        raise ValueError(
            f"{path} is not a valid spec-bundle:\n  - " + "\n  - ".join(problems)
        )
    if policy is not None:
        reject_airgap_bundle(bundle, policy)
    return bundle

def reject_airgap_bundle(bundle: dict[str, Any], policy: dict[str, Any]) -> None:
    """Apply the air-gap policy to the bundle's DATA, not to a rendered argv.

    `_reject_airgap` reasons about a local docker/podman command line, which is
    why it refuses launchers it cannot inspect. A bundle is platform-agnostic,
    so the same policy question — does running this reach the network? — is
    answerable before any platform renders it, and stays answerable for
    platforms whose argv this module will never see (srun, kubectl, brev).
    """
    if policy.get("network_mode") != "airgap":
        return
    for item in bundle.get("declared_inputs") or []:
        uri = str(item.get("uri", ""))
        if uri.startswith(FETCH_SCHEMES):
            raise ValueError(
                f"declared_input {item.get('spec_key')} is {uri!r}: resolving it "
                "is a network fetch, forbidden by air-gap policy. Pre-stage the "
                "asset and declare its compute-frame path instead."
            )

def verify_inputs_staged(bundle: dict[str, Any], platform: str) -> None:
    """Every declared_input must be readable from the compute frame already.

    `tao-data-io` owns getting data there — it picks the storage tier and, for
    tier A, the answer is usually "it is already on the Lustre/PVC mount, do
    nothing". It has no general staging CLI (its packaged script is the narrow
    annotation-selective downloader), so staging is driven by the agent reading
    that skill. What this does is refuse to launch against inputs that are not
    there yet, and say which ones — the alternative is a container that starts,
    finds nothing, and fails deep inside the workload.

    Only locally-checkable paths are verified: on a remote platform the compute
    frame is another machine, so presence is that platform's problem and the
    renderer's URI refusal is the backstop.
    """
    remote = platform in {"slurm", "kubernetes", "brev"}
    missing: list[str] = []
    unstaged: list[str] = []
    for item in bundle.get("declared_inputs") or []:
        uri = str(item.get("uri", ""))
        if "://" in uri:
            unstaged.append(f"{item.get('spec_key')} -> {uri}")
        elif not remote and not pathlib.Path(uri).exists():
            missing.append(f"{item.get('spec_key')} -> {uri}")
    if unstaged:
        raise ValueError(
            "these inputs are not staged into the compute frame: "
            + "; ".join(unstaged)
            + ". Stage them with the tao-data-io skill (it picks the storage "
            "tier), then declare the compute-frame path in the bundle."
        )
    if missing:
        raise ValueError(
            "these declared inputs do not exist: " + "; ".join(missing)
        )

def _prepared(command: list[str], policy: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Apply the air-gap gate and return the launch command plus its env."""
    _reject_airgap(command, policy)
    command = _with_no_pull(command, policy)
    command = _with_offline_container_env(command, policy)
    environment = os.environ.copy()
    if policy.get("network_mode") == "airgap":
        environment.update(OFFLINE_ENV)
    return command, environment

def _with_detach(command: list[str], job_id: str) -> list[str]:
    """Name the container after the job id and detach, per the launch contract."""
    index = _run_index(command)
    if index is None:
        raise ValueError("--submit currently supports a local `docker run` command only")
    tokens = [_basename(token) for token in command]
    for flag in ("-d", "--detach", "--name"):
        if flag in tokens:
            raise ValueError(f"{flag} is set by --submit; remove it from the command")
    if "--rm" in tokens:
        raise ValueError(
            "--rm cannot be combined with --submit: it deletes the container on "
            "exit, so the exit code is unreadable and the job can never reach a "
            "terminal state. Tear down with the cancel verb after --await-job."
        )
    return [*command[: index + 1], "--name", job_id, "-d", *command[index + 1 :]]

def submit_bundle(
    state_path: pathlib.Path,
    bundle: dict[str, Any],
    *,
    storage_tier: str,
    parent_job: str | None,
    platform: str,
    ctx_extra: dict[str, Any] | None = None,
) -> str:
    """Open a record, let the platform render the bundle, launch, record the handle.

    The rendered argv is NOT re-checked by `_reject_airgap`. That gate reasons
    about a local docker/podman command line and refuses launchers it cannot
    inspect, so re-running it here would reject a legitimate `ssh … sbatch` or
    `kubectl apply` that this module generated itself. Policy for a bundle is
    applied to the bundle, in `reject_airgap_bundle`, before anything renders —
    which is the whole reason that check moved to the data.
    """
    policy = _policy(state_path)
    state = json.loads(state_path.expanduser().read_text())
    results_dir = state.get("results_dir")
    if not results_dir:
        raise ValueError(f"{state_path} has no results_dir")

    open_args = [
        "open", "--platform", platform, "--image", bundle["image"],
        "--network-arch", bundle["network_arch"], "--action", bundle["action"],
        "--storage-tier", storage_tier, "--results-root", str(results_dir),
    ]
    if parent_job:
        open_args += ["--parent-job", parent_job]
    job_id = _record(*open_args)

    ctx: dict[str, Any] = {
        "job_id": job_id,
        "results_dir": os.path.join(str(results_dir), job_id),
        "bank": str(_bank()),
        "airgap": policy.get("network_mode") == "airgap",
    }
    ctx.update(ctx_extra or {})

    renderer = platform_renderer(platform)

    # Inputs must exist in the COMPUTE frame before anything renders. Staging is
    # tao-data-io's job and it is agent-driven (the skill has no general CLI), so
    # this verifies and names the exact gap rather than guessing a fetch.
    try:
        verify_inputs_staged(bundle, platform)
    except Exception:
        _record("mark", job_id, "--state", "ERROR", "--err-class", "ERR_INFRA",
                "--message", "inputs not staged")
        raise

    # Make the image available in the platform's native form: pull-if-missing on
    # docker/brev, reuse-or-convert the Lustre .sqsh on slurm, nothing on k8s.
    # Idempotent by design — the common path does no work.
    try:
        prepared = renderer.prepare(bundle, ctx)
        bundle = {**bundle, "image": prepared["image"]}
    except Exception:
        _record("mark", job_id, "--state", "ERROR", "--err-class", "ERR_INFRA",
                "--message", "image prepare failed")
        raise

    try:
        rendered = renderer.render(bundle, ctx)
    except Exception:
        _record("mark", job_id, "--state", "ERROR", "--err-class", "ERR_INFRA",
                "--message", "render failed")
        raise

    # Where the rendered files belong depends on where the launcher runs.
    # kubernetes writes a manifest a local kubectl reads; slurm writes an
    # sbatch script that must exist on the CLUSTER, since its argv is ssh. A
    # platform that launches remotely owns placement and says so with
    # place_files(); everything else is local.
    try:
        place = getattr(renderer, "place_files", None)
        if place is not None:
            place(rendered.get("files") or {}, ctx)
        else:
            for path, content in (rendered.get("files") or {}).items():
                target = pathlib.Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
    except Exception:
        # Without this the record stays PENDING forever: it is already open,
        # and nothing downstream runs to close it.
        _record("mark", job_id, "--state", "ERROR", "--err-class", "ERR_INFRA",
                "--message", "could not place rendered files")
        raise

    # Last gate before argv runs. A platform may refuse a launch its renderer
    # could legitimately produce -- docker refuses UID 0 for a writable bind,
    # which must be checked HERE and not at render time, since rendering
    # creates nothing.
    guard = getattr(renderer, "preflight_launch", None)
    if guard is not None:
        try:
            guard(bundle, ctx)
        except Exception:
            _record("mark", job_id, "--state", "ERROR", "--err-class", "ERR_INFRA",
                    "--message", "launch refused by the platform")
            raise

    environment = os.environ.copy()
    if ctx["airgap"]:
        environment.update(OFFLINE_ENV)
    launched = subprocess.run(
        rendered["argv"], env=environment,
        capture_output=True, text=True, check=False,
    )
    if launched.returncode != 0:
        _record("mark", job_id, "--state", "ERROR", "--err-class", "ERR_INFRA",
                "--message", "launch failed")
        raise ValueError(f"launch failed: {launched.stderr.strip()}")

    backend_ref = rendered.get("backend_ref") or launched.stdout.strip()
    _record("mark", job_id, "--state", "RUNNING", "--backend-ref", backend_ref)
    return job_id

def _backend(job_id: str, ctx_extra: dict[str, Any] | None = None):
    """Resolve a job id to (renderer, backend_ref, ctx) from its record.

    Same rule as await_job: which backend to ask comes from the record, not a
    flag, so a job can be inspected or cancelled from a session that never
    launched it.
    """
    record = json.loads(_record("show", job_id)) or {}
    platform = record.get("platform") or "docker"
    ctx = {"job_id": job_id, "results_dir": record.get("results_dir") or "",
           "bank": str(_bank())}
    ctx.update(ctx_extra or {})
    return platform_renderer(platform), record.get("backend_ref") or job_id, ctx

def job_logs(job_id: str, tail: int, ctx_extra: dict[str, Any] | None = None) -> int:
    """The `logs` verb. Diagnosing a failure should not need a hand-written ssh."""
    renderer, backend_ref, ctx = _backend(job_id, ctx_extra)
    output = renderer.logs(backend_ref, ctx, tail=tail)
    if output:
        print(output)
        return 0
    # Distinguish "ran and said nothing" from "we looked in the wrong place":
    # an empty tail otherwise reads as a silent job.
    print(f"no logs for {job_id} (backend_ref {backend_ref})", file=sys.stderr)
    return 1

def cancel_job(job_id: str, ctx_extra: dict[str, Any] | None = None) -> int:
    """The `cancel` verb, then close the record.

    The record has to be marked here rather than left to polling: on some
    platforms cancelling destroys the object status() reads, so a later poll
    returns UNKNOWN forever and the job never reaches a terminal state.

    A job that has already finished is not an error. The caller's intent --
    "make sure this is not running" -- is already satisfied, and the record
    refuses transitions out of a terminal state (rightly: a COMPLETE run whose
    results exist must not be relabelled CANCELED). So check first and say so,
    rather than attempting the illegal transition and surfacing the guard as a
    failure.
    """
    record = json.loads(_record("show", job_id)) or {}
    terminal = record.get("terminal_state")
    if terminal:
        print(f"{job_id} already finished ({terminal}); nothing to cancel")
        return 0

    renderer, backend_ref, ctx = _backend(job_id, ctx_extra)
    stopped = renderer.cancel(backend_ref, ctx)
    try:
        _record("mark", job_id, "--state", "CANCELED",
                "--message", "canceled by request")
    except ValueError:
        # It reached a terminal state between the check above and this mark.
        # The record is already closed and correct; the cancel was just late,
        # and re-raising would report a failure for a job that ended fine.
        closed = json.loads(_record("show", job_id)) or {}
        print(f"{job_id} finished as {closed.get('terminal_state')} while being "
              "cancelled; leaving its record at that state")
        return 0
    if not stopped:
        print(f"backend refused to cancel {backend_ref}; record marked CANCELED",
              file=sys.stderr)
    return 0 if stopped else 1

def await_job(
    job_id: str,
    *,
    poll_seconds: float,
    timeout_seconds: float,
    ctx_extra: dict[str, Any] | None = None,
) -> int:
    """Poll to a terminal state, close the record, return the backend's code.

    Which backend to ask comes from the record, not from a flag: `open` wrote
    the platform and `mark --backend-ref` wrote the handle, so a job can be
    awaited from a session that never launched it. That is the point of the
    record-then-launch invariant.
    """
    record = json.loads(_record("show", job_id)) or {}
    platform = record.get("platform") or "docker"
    backend_ref = record.get("backend_ref") or job_id
    ctx = {"job_id": job_id, "results_dir": record.get("results_dir") or "",
           "bank": str(_bank())}
    ctx.update(ctx_extra or {})
    status = platform_renderer(platform).status

    waited = 0.0
    while True:
        state, exit_code = status(backend_ref, ctx)
        if state in {"COMPLETE", "ERROR"}:
            break
        if timeout_seconds and waited >= timeout_seconds:
            raise ValueError(f"{job_id} still {state} after {timeout_seconds}s")
        time.sleep(poll_seconds)
        waited += poll_seconds

    tier = record.get("storage_tier")
    if state == "ERROR":
        _record("mark", job_id, "--state", "ERROR", "--err-class", "ERR_PROGRAM",
                "--message", f"container exited {exit_code}")
    elif tier == "A":
        # Tier A results are already readable on a local mount, so the container
        # exiting 0 is the same event as the results surviving.
        _record("mark", job_id, "--state", "COMPLETE")
    else:
        # Tier B/C must upload and verify BEFORE the terminal mark — a terminal
        # record refuses later transitions, so a failed upload recorded after
        # COMPLETE would be unrepresentable.
        print(
            f"{job_id}: container exited 0; upload results, then "
            f"`tao_job_record.py mark {job_id} --state COMPLETE`",
            file=sys.stderr,
        )
    return exit_code

