#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one canonical composite IAA SDG action through Airflow and SLURM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import sys
import time
from typing import Any, Sequence


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
BANK = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

import airflow_orchestrator as orchestrator  # noqa: E402
import airflow_slurm_action as action_bridge  # noqa: E402


WORKFLOW = "tao-run-deft-iaa"
SAFE_LOGIN = re.compile(r"^[A-Za-z0-9_.@-]+$")
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")
TERMINAL = frozenset({"COMPLETE", "ERROR", "CANCELED"})


class BridgeError(RuntimeError):
    pass


def _stage_consumer(shared_root: pathlib.Path) -> pathlib.Path:
    source_skill = BANK / "skills/platform/tao-run-on-slurm"
    source_app = BANK / "skills/applications/tao-run-deft-iaa/scripts"
    sources = [
        source_skill / "scripts/slurm_sdg_action.py",
        source_skill / "templates/slurm_sdg.sbatch.tmpl",
        source_skill / "templates/slurm_sdg_image.sbatch.tmpl",
        BANK / "scripts/redact_secrets.py",
        *sorted(path for path in source_app.rglob("*.py") if "__pycache__" not in path.parts),
    ]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source.relative_to(BANK)).encode())
        digest.update(source.read_bytes())
    runtime = shared_root / "runtime" / f"slurm-sdg-{digest.hexdigest()[:16]}"
    for source in sources:
        target = runtime / source.relative_to(BANK)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise BridgeError(f"staged SDG consumer dependency differs: {target}")
        else:
            shutil.copy2(source, target)
    return runtime / "skills/platform/tao-run-on-slurm/scripts/slurm_sdg_action.py"


def _open_job(
    *, request: dict[str, Any], state_dir: pathlib.Path, image: str,
    retry_of: str | None = None,
) -> tuple[str, pathlib.Path]:
    env = dict(os.environ)
    env["TAO_STATE_DIR"] = str(state_dir)
    command = [
        sys.executable, str(BANK / "scripts/tao_job_record.py"), "open",
        "--platform", "slurm", "--image", image,
        "--network-arch", "iaa-sdg", "--action", request["action_id"],
        "--storage-tier", "A", "--results-dir", request["stage_dir"],
    ]
    if retry_of is not None:
        command.extend(["--retry-of", retry_of])
    completed = action_bridge._run(  # noqa: SLF001
        command, operation="open SLURM SDG job record", env=env
    )
    job_id = completed.stdout.strip()
    if not job_id or SAFE_TOKEN.fullmatch(job_id) is None:
        raise BridgeError("job-record writer returned an invalid SDG job id")
    return job_id, state_dir / "jobs" / f"{job_id}.json"


def _mark(
    job_id: str, state: str, state_dir: pathlib.Path, *, backend_ref: str | None = None,
    message: str = "", err_class: str | None = None,
) -> None:
    env = dict(os.environ)
    env["TAO_STATE_DIR"] = str(state_dir)
    command = [
        sys.executable, str(BANK / "scripts/tao_job_record.py"), "mark", job_id,
        "--state", state, "--source", "poller",
    ]
    if backend_ref is not None:
        command.extend(["--backend-ref", backend_ref])
    if message:
        command.extend(["--message", message])
    if err_class is not None:
        if state != "ERROR" or err_class not in {"ERR_INFRA", "ERR_PROGRAM"}:
            raise BridgeError("err_class is valid only for an ERROR job transition")
        command.extend(["--err-class", err_class])
    action_bridge._run(command, operation=f"mark SDG job {state}", env=env)  # noqa: SLF001


def _local_outputs(results: pathlib.Path, iteration: int) -> list[pathlib.Path]:
    stage = results / f"iter_{iteration}" / "datagen"
    return [
        stage / "dataset/sdg_manifest.json",
        stage / "dataset/sdg_pairs.json",
        stage / "dataset/sdg_image_list.txt",
        stage / "sdg_execution_manifest.json",
        stage / "endpoint_pool.json",
        stage / "endpoint_manifest.json",
        stage / "status/sdg-normalize.slurm.status.json",
    ]


def _native_job_names_absent(
    *, login: str, job_id: str, generation_nodes: int,
) -> bool:
    """Prove a failed Airflow submit created none of the deterministic jobs."""
    if (
        SAFE_LOGIN.fullmatch(login) is None
        or SAFE_TOKEN.fullmatch(job_id) is None
        or not 1 <= generation_nodes <= 64
    ):
        raise BridgeError("native absence proof received invalid job identity")
    names = [
        *(f"{job_id}-img-{index:03d}" for index in range(generation_nodes)),
        f"{job_id}-coord",
    ]
    rows = []
    for name in names:
        quoted = shlex.quote(name)
        command = (
            "set -Eeuo pipefail; "
            f"ids=$(squeue -h --name {quoted} -o '%i'; "
            f"sacct -X -n --name {quoted} -o JobIDRaw 2>/dev/null || true); "
            "ids=$(printf '%s\\n' \"$ids\" | awk 'NF {split($1,a,\".\"); "
            "if (a[1] ~ /^[0-9]+$/) print a[1]}' | sort -u | paste -sd, -); "
            f"printf '%s|%s\\n' {quoted} \"$ids\""
        )
        completed = action_bridge._run(  # noqa: SLF001
            ["ssh", "-o", "BatchMode=yes", login, command],
            timeout=30, operation=f"prove no native SLURM job for {name}",
        )
        rows.append(completed.stdout.strip())
    expected = {f"{name}|" for name in names}
    return set(rows) == expected and len(rows) == len(expected)


def _preflight_airflow() -> None:
    """Validate Airflow before creating an SDG request or native job record."""
    pool = os.environ.get("AIRFLOW_IAA_COORDINATOR_POOL", "iaa-coordinator")
    if SAFE_TOKEN.fullmatch(pool) is None:
        raise BridgeError("AIRFLOW_IAA_COORDINATOR_POOL contains unsupported characters")
    action_bridge._run(  # noqa: SLF001
        [
            sys.executable, str(SCRIPT_DIR / "airflow_action.py"), "preflight",
            "--pool", f"{pool}:1",
        ],
        operation="preflight Airflow SLURM SDG orchestration",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if SAFE_LOGIN.fullmatch(args.login) is None:
        raise BridgeError("--login contains unsupported characters")
    for flag, value in (("--account", args.account), ("--partition", args.partition)):
        if SAFE_TOKEN.fullmatch(value) is None:
            raise BridgeError(f"{flag} contains unsupported characters")
    _preflight_airflow()
    results = args.results_dir.resolve()
    shared_root = args.shared_root.resolve()
    state = action_bridge._json(results / "deft_state.json", "DEFT state")  # noqa: SLF001
    if (
        state.get("workflow") != WORKFLOW
        or state.get("config", {}).get("platform") != "slurm"
        or state["config"].get("orchestrator") != "airflow"
    ):
        raise BridgeError("run is not an Airflow-orchestrated SLURM IAA workflow")
    phase = state.get("iterations", {}).get(f"iter{args.iteration}")
    if (
        not isinstance(phase, dict)
        or phase.get("status") != "in_progress"
        or phase.get("stage_completed") != "history_select"
    ):
        raise BridgeError("SDG bridge requires committed history_select as the next boundary")
    local_workspace = pathlib.Path(state["config"]["workspace"]).resolve()
    try:
        results.relative_to(shared_root)
    except ValueError as exc:
        raise BridgeError("results_dir must be visible under the Airflow shared root") from exc
    remote_workspace = action_bridge._remote_path(  # noqa: SLF001
        args.remote_workspace, "--remote-workspace"
    )
    remote_results = action_bridge._workspace_mapping(  # noqa: SLF001
        results, local_workspace, remote_workspace
    )
    backend_dataset = action_bridge._remote_path(  # noqa: SLF001
        args.backend_dataset_root, "--backend-dataset-root"
    )
    action_bridge._verify_backend_dataset(  # noqa: SLF001
        login=args.login, local=pathlib.Path(state["config"]["dataset_root"]),
        remote=backend_dataset,
    )
    remote_cache = action_bridge._remote_path(args.cache_dir, "--cache-dir")  # noqa: SLF001
    action_bridge._ssh(  # noqa: SLF001
        args.login,
        "set -Eeuo pipefail; "
        f"test -d {shlex.quote(str(remote_cache))}; "
        f"test ! -L {shlex.quote(str(remote_cache))}",
        operation="verify remote SDG cache",
    )

    retry_values = (args.retry_from_request, args.retry_from_job_record)
    repair_values = (args.repair_from_request, args.repair_from_job_record)
    reschedule_values = (
        args.reschedule_from_request, args.reschedule_from_job_record,
    )
    launch_repair_values = (
        args.launch_repair_from_request, args.launch_repair_from_job_record,
    )
    if sum(
        any(value is not None for value in group)
        for group in (
            retry_values, repair_values, reschedule_values, launch_repair_values,
        )
    ) > 1:
        raise BridgeError(
            "bounded retry, pool-rebind repair, scheduler reschedule, and launch "
            "repair are mutually exclusive"
        )
    if any(value is not None for value in retry_values) and not all(
        value is not None for value in retry_values
    ):
        raise BridgeError(
            "bounded retry requires both --retry-from-request and --retry-from-job-record"
        )
    if any(value is not None for value in repair_values) and not all(
        value is not None for value in repair_values
    ):
        raise BridgeError(
            "pool-rebind repair requires both --repair-from-request and "
            "--repair-from-job-record"
        )
    if any(value is not None for value in reschedule_values) and not all(
        value is not None for value in reschedule_values
    ):
        raise BridgeError(
            "scheduler reschedule requires both --reschedule-from-request and "
            "--reschedule-from-job-record"
        )
    if any(value is not None for value in launch_repair_values) and not all(
        value is not None for value in launch_repair_values
    ):
        raise BridgeError(
            "image master-port repair requires both --launch-repair-from-request "
            "and --launch-repair-from-job-record"
        )
    parent_job_id: str | None = None
    parent_label = "attempt-1"
    parent_record_path = args.retry_from_job_record
    if args.repair_from_job_record is not None:
        parent_record_path = args.repair_from_job_record
        parent_label = "attempt-2 repair-source"
    if args.reschedule_from_job_record is not None:
        parent_record_path = args.reschedule_from_job_record
        parent_label = "attempt-2 reschedule-source"
    if args.launch_repair_from_job_record is not None:
        parent_record_path = args.launch_repair_from_job_record
        parent_label = "attempt-2 launch-repair-source"
    if parent_record_path is not None:
        prior_record = action_bridge._json(  # noqa: SLF001
            parent_record_path.resolve(), f"{parent_label} SLURM SDG job record"
        )
        parent_job_id = prior_record.get("id")
        if not isinstance(parent_job_id, str) or SAFE_TOKEN.fullmatch(parent_job_id) is None:
            raise BridgeError(f"{parent_label} job record has an invalid id")

    stage = results / f"iter_{args.iteration}" / "datagen"
    runtime = stage / ".tao-runtime" / "controller"
    runtime.mkdir(parents=True, exist_ok=True)
    request_path = runtime / (
        "sdg.attempt-2-launch-repair.action.json"
        if args.launch_repair_from_job_record is not None
        else (
            f"sdg.attempt-2-reschedule-{args.time_minutes}.action.json"
            if args.reschedule_from_job_record is not None
            else (
                "sdg.attempt-2-repair.action.json"
                if args.repair_from_job_record is not None
                else (
                    "sdg.attempt-2.action.json"
                    if parent_job_id is not None else "sdg.action.json"
                )
            )
        )
    )
    slurm_consumer_source = BANK / "skills/platform/tao-run-on-slurm/scripts/slurm_sdg_action.py"
    prepare_command = [
        sys.executable, str(slurm_consumer_source), "prepare-request",
        "--deft-state", str(results / "deft_state.json"),
        "--sdg-config", str(results / "config/sdg_config.yaml"),
        "--iteration", str(args.iteration),
        "--runtime-root", str(remote_results / f"iter_{args.iteration}/datagen/.tao-runtime/runtime"),
        "--cache-dir", str(remote_cache),
        "--backend-results-dir", str(remote_results),
        "--backend-dataset-root", str(backend_dataset),
        "--augmentation-image", str(args.augmentation_sqsh),
        "--auto-labeling-image", str(args.auto_labeling_sqsh),
        "--image-edit-image", str(args.image_edit_sqsh),
        "--text-serving-image", str(args.text_serving_sqsh),
        "--account", args.account, "--partition", args.partition,
        "--image-worker-cpus-per-task", str(args.image_worker_cpus),
        "--coordinator-cpus-per-task", str(args.coordinator_cpus),
        "--time-minutes", str(args.time_minutes),
        "--output", str(request_path),
    ]
    if args.retry_from_job_record is not None:
        prepare_command.extend([
            "--retry-from-request", str(args.retry_from_request.resolve()),
            "--retry-from-job-record", str(args.retry_from_job_record.resolve()),
            "--retry-login", args.login,
        ])
    if args.repair_from_job_record is not None:
        prepare_command.extend([
            "--repair-from-request", str(args.repair_from_request.resolve()),
            "--repair-from-job-record", str(args.repair_from_job_record.resolve()),
            "--repair-login", args.login,
        ])
    if args.reschedule_from_job_record is not None:
        prepare_command.extend([
            "--reschedule-from-request", str(args.reschedule_from_request.resolve()),
            "--reschedule-from-job-record",
            str(args.reschedule_from_job_record.resolve()),
            "--reschedule-login", args.login,
        ])
    if args.launch_repair_from_job_record is not None:
        prepare_command.extend([
            "--launch-repair-from-request",
            str(args.launch_repair_from_request.resolve()),
            "--launch-repair-from-job-record",
            str(args.launch_repair_from_job_record.resolve()),
            "--launch-repair-login", args.login,
        ])
    prepare = action_bridge._json_result(action_bridge._run(  # noqa: SLF001
        prepare_command, operation="prepare signed SLURM SDG request"
    ), "SLURM SDG request preparation")
    request = prepare.get("request")
    if not isinstance(request, dict):
        request = action_bridge._json(request_path, "SLURM SDG request")  # noqa: SLF001

    if parent_job_id is None:
        action_bridge._stage_tree(  # noqa: SLF001
            source=results, login=args.login, target=remote_results,
            receipt=runtime / "slurm-results-tree.staged.json",
            incremental_existing=True,
        )

    state_dir_raw = os.environ.get("TAO_STATE_DIR")
    if not state_dir_raw:
        raise BridgeError("TAO_STATE_DIR must identify the Airflow shared state root")
    state_dir = pathlib.Path(state_dir_raw)
    job_id, job_record = _open_job(
        request=request, state_dir=state_dir,
        image=str(request["component_sources"]["augmentation"]),
        retry_of=parent_job_id,
    )
    consumer = _stage_consumer(shared_root)
    remote_script = remote_results / f"iter_{args.iteration}/datagen/.tao-runtime/job_{job_id}"
    interpreter = "/usr/bin/python3"
    common = [
        "--request", str(request_path), "--login", args.login,
        "--job-id", job_id, "--job-record", str(job_record),
    ]
    commands = {
        "submit": [
            interpreter, str(consumer), "submit", *common,
            "--remote-script", str(remote_script),
            "--account", args.account, "--partition", args.partition,
        ],
        "status": [
            interpreter, str(consumer), "status", *common,
            "--backend-ref", "{backend_ref}",
            "--local-results-dir", str(results),
        ],
        "logs": [
            interpreter, str(consumer), "logs", *common,
            "--backend-ref", "{backend_ref}", "--tail", "500",
        ],
        "cancel": [
            interpreter, str(consumer), "cancel", *common,
            "--backend-ref", "{backend_ref}", "--confirm",
        ],
    }
    plan_path = runtime / f"airflow-slurm-sdg-plan.{job_id}.json"
    envelope_path = runtime / f"airflow-slurm-sdg-orchestration.{job_id}.json"
    action_bridge._atomic_json(plan_path, {  # noqa: SLF001
        "commands": commands,
        "expected_outputs": [str(path) for path in _local_outputs(results, args.iteration)],
        "poll_interval_s": args.compute_poll_interval,
        "deadline_s": args.deadline, "unknown_status_limit": 3,
        "retain_on_failure": True, "forward_env": request["forward_env"],
    })
    if orchestrator.prepare(argparse.Namespace(
        compute_platform="slurm", compute_kind="sdg",
        compute_request=request_path, job_record=job_record,
        job_binding=None, consumer_plan=plan_path, output=envelope_path,
    )) != 0:
        raise BridgeError("Airflow SDG envelope preparation failed")
    submitted = action_bridge._json_result(action_bridge._run([  # noqa: SLF001
        sys.executable, str(SCRIPT_DIR / "airflow_orchestrator.py"), "submit",
        "--envelope", str(envelope_path),
    ], operation="submit Airflow SLURM SDG orchestration"), "Airflow SDG submit")
    airflow_ref = submitted.get("backend_ref")
    if not isinstance(airflow_ref, str) or not airflow_ref:
        raise BridgeError("Airflow SDG submit returned no backend reference")
    envelope = action_bridge._json(envelope_path, "Airflow SDG envelope")  # noqa: SLF001
    receipt_path = pathlib.Path(envelope["receipt_path"])
    backend_ref: str | None = None
    marked = False
    deadline = time.monotonic() + args.deadline + 300
    last = "PENDING"
    while time.monotonic() < deadline:
        status = action_bridge._json_result(action_bridge._run([  # noqa: SLF001
            sys.executable, str(SCRIPT_DIR / "airflow_orchestrator.py"), "status",
            "--envelope", str(envelope_path), "--backend-ref", airflow_ref,
        ], operation="poll Airflow SLURM SDG orchestration"), "Airflow SDG status")
        last = str(status.get("status", "UNKNOWN")).upper()
        if receipt_path.is_file():
            receipt = action_bridge._json(receipt_path, "Airflow SDG receipt")  # noqa: SLF001
            candidate = receipt.get("compute_backend_ref")
            if isinstance(candidate, str) and candidate:
                backend_ref = candidate
                if not marked:
                    _mark(
                        job_id, "RUNNING", state_dir, backend_ref=backend_ref,
                        message="composite SLURM SDG submitted by Airflow",
                    )
                    marked = True
        if last in TERMINAL:
            break
        time.sleep(args.controller_poll_interval)
    if last != "COMPLETE" or backend_ref is None:
        if backend_ref is None and last in {"ERROR", "CANCELED"} and _native_job_names_absent(
            login=args.login, job_id=job_id,
            generation_nodes=request["generation_nodes"],
        ):
            _mark(
                job_id, "CANCELED", state_dir,
                message=(
                    "Airflow ended before composite SLURM submission; exact native "
                    "job names are absent"
                ),
            )
        elif backend_ref is not None and last == "ERROR":
            _mark(
                job_id, "ERROR", state_dir, backend_ref=backend_ref,
                err_class="ERR_INFRA",
                message="Airflow-observed composite SLURM SDG failure; owned evidence retained",
            )
        elif backend_ref is not None and last == "CANCELED":
            _mark(
                job_id, "CANCELED", state_dir, backend_ref=backend_ref,
                message="Airflow-observed composite SLURM SDG cancellation",
            )
        raise BridgeError(
            f"Airflow SLURM SDG ended as {last}; owned resources and job record retained"
        )
    _mark(
        job_id, "COMPLETE", state_dir, backend_ref=backend_ref,
        message="Airflow and composite SLURM SDG completed with synchronized outputs",
    )
    return {
        "status": "COMPLETE", "job_id": job_id,
        "slurm_backend_ref": backend_ref, "airflow_backend_ref": airflow_ref,
        "request": str(request_path), "job_record": str(job_record),
        "outputs": [str(path) for path in _local_outputs(results, args.iteration)],
    }


def _bounded(minimum: int, maximum: int, flag: str):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{flag} must be an integer") from exc
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(f"{flag} must be in [{minimum}, {maximum}]")
        return number
    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--login", required=True)
    parser.add_argument("--remote-workspace", required=True, type=pathlib.Path)
    parser.add_argument("--shared-root", required=True, type=pathlib.Path)
    parser.add_argument("--backend-dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--cache-dir", required=True, type=pathlib.Path)
    parser.add_argument("--augmentation-sqsh", required=True, type=pathlib.Path)
    parser.add_argument("--auto-labeling-sqsh", required=True, type=pathlib.Path)
    parser.add_argument("--image-edit-sqsh", required=True, type=pathlib.Path)
    parser.add_argument("--text-serving-sqsh", required=True, type=pathlib.Path)
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", default="polar3")
    parser.add_argument("--retry-from-request", type=pathlib.Path)
    parser.add_argument("--retry-from-job-record", type=pathlib.Path)
    parser.add_argument("--repair-from-request", type=pathlib.Path)
    parser.add_argument("--repair-from-job-record", type=pathlib.Path)
    parser.add_argument("--reschedule-from-request", type=pathlib.Path)
    parser.add_argument("--reschedule-from-job-record", type=pathlib.Path)
    parser.add_argument("--launch-repair-from-request", type=pathlib.Path)
    parser.add_argument("--launch-repair-from-job-record", type=pathlib.Path)
    parser.add_argument("--image-worker-cpus", type=int, default=64)
    parser.add_argument("--coordinator-cpus", type=int, default=60)
    parser.add_argument("--time-minutes", type=int, default=240)
    parser.add_argument(
        "--compute-poll-interval", type=_bounded(5, 300, "--compute-poll-interval"),
        default=15,
    )
    parser.add_argument(
        "--controller-poll-interval",
        type=_bounded(1, 300, "--controller-poll-interval"), default=10,
    )
    parser.add_argument(
        "--deadline", type=_bounded(60, 604800, "--deadline"), default=21600,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(run(args), sort_keys=True))
        return 0
    except (BridgeError, OSError, ValueError, RuntimeError) as exc:
        print(f"airflow_slurm_sdg_action: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
