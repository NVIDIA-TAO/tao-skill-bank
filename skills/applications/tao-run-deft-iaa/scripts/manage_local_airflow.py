#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install and manage a run-scoped local Airflow development service for IAA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any


AIRFLOW_VERSION = "3.3.1"
PYTHON_VERSION = "3.12"
CONSTRAINT_URL = (
    "https://raw.githubusercontent.com/apache/airflow/"
    f"constraints-{AIRFLOW_VERSION}/constraints-{PYTHON_VERSION}.txt"
)
DAG_ID = "tao_deft_iaa_action_v1"
POOL_SLOTS = {
    "iaa-cpu": 2,
    "iaa-tao-gpu": 1,
    "iaa-image-workers": 1,
    "iaa-coordinator": 1,
}


class LocalAirflowError(ValueError):
    pass


def _normalized_target(path: pathlib.Path, field: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(path.expanduser()))
    if lexical == pathlib.Path("/") or lexical.resolve(strict=False) != lexical:
        raise LocalAirflowError(f"{field} must be a normalized non-root non-symlink path")
    return lexical


def _safe_dir(path: pathlib.Path, field: str, *, create: bool = False) -> pathlib.Path:
    lexical = _normalized_target(path, field)
    if create:
        lexical.mkdir(parents=True, exist_ok=True)
    if not lexical.is_dir() or lexical.is_symlink():
        raise LocalAirflowError(f"{field} must be an existing directory")
    return lexical


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(root: pathlib.Path) -> dict[str, pathlib.Path]:
    return {
        "venv": root / "venv",
        "airflow_home": root / "home",
        "dags": root / "home" / "dags",
        "runtime": root / "runtime",
        "receipt": root / "local-airflow.json",
        "service_log": root / "airflow-standalone.log",
    }


def _python312() -> pathlib.Path:
    found = shutil.which("python3.12")
    if not found:
        raise LocalAirflowError("python3.12 is required for the pinned local Airflow runtime")
    return pathlib.Path(found).resolve()


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            handle.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _environment(root: pathlib.Path, shared_root: pathlib.Path, port: int) -> dict[str, str]:
    paths = _paths(root)
    env = dict(os.environ)
    env["PATH"] = str(paths["venv"] / "bin") + os.pathsep + env.get("PATH", "")
    env.update({
        "AIRFLOW_HOME": str(paths["airflow_home"]),
        "AIRFLOW__CORE__DAGS_FOLDER": str(paths["dags"]),
        "AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION": "False",
        "AIRFLOW__CORE__LOAD_EXAMPLES": "False",
        "AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS": "True",
        "AIRFLOW__CORE__EXECUTOR": "LocalExecutor",
        "AIRFLOW__CORE__EXECUTION_API_SERVER_URL": f"http://127.0.0.1:{port}/execution/",
        "AIRFLOW__API__HOST": "127.0.0.1",
        "AIRFLOW__API__PORT": str(port),
        "TAO_IAA_AIRFLOW_RUNTIME_ROOT": str(paths["runtime"]),
        "TAO_IAA_AIRFLOW_SHARED_ROOT": str(shared_root),
        "TAO_IAA_AIRFLOW_LOCAL_ALL_ADMINS": "1",
        "AIRFLOW_IAA_COORDINATOR_POOL": "iaa-coordinator",
    })
    return env


def plan(args: argparse.Namespace) -> dict[str, Any]:
    root = _normalized_target(args.root, "--root")
    shared = _normalized_target(args.shared_root, "--shared-root")
    paths = _paths(root)
    installed = paths["venv"].joinpath("bin", "airflow").is_file()
    deployed = paths["dags"].joinpath("tao_deft_iaa_action_v1.py").is_file()
    return {
        "status": "ok", "action": "plan", "airflow_version": AIRFLOW_VERSION,
        "python": str(_python312()), "root": str(root), "shared_root": str(shared),
        "base_url": f"http://127.0.0.1:{args.port}",
        "port_available": _port_available(args.port), "installed": installed,
        "deployed": deployed, "dag_id": DAG_ID, "pools": POOL_SLOTS,
        "root_exists": root.is_dir(), "shared_root_exists": shared.is_dir(),
        "authentication": "loopback-only simple-auth all-admin token",
    }


def install(args: argparse.Namespace) -> dict[str, Any]:
    root = _safe_dir(args.root, "--root", create=True)
    paths = _paths(root)
    airflow = paths["venv"] / "bin" / "airflow"
    if not airflow.is_file():
        # Some minimal Debian hosts provide python3.12 and pip but omit
        # ensurepip/python3.12-venv.  A no-pip environment plus pip's supported
        # --python target remains isolated and avoids requiring root packages.
        subprocess.run([
            str(_python312()), "-m", "venv", "--without-pip", str(paths["venv"]),
        ], check=True)
        pip = [str(_python312()), "-m", "pip", "--python", str(paths["venv"])]
        subprocess.run([
            *pip, "install", f"apache-airflow=={AIRFLOW_VERSION}",
            "--constraint", CONSTRAINT_URL,
        ], check=True)
        subprocess.run([
            *pip, "install", "pandas", "pyarrow", "PyYAML", "jsonschema",
            "--constraint", CONSTRAINT_URL,
        ], check=True)
    completed = subprocess.run([str(airflow), "version"], capture_output=True, text=True, check=True)
    version = completed.stdout.strip().splitlines()[-1]
    if version != AIRFLOW_VERSION:
        raise LocalAirflowError(f"installed Airflow version is {version}, expected {AIRFLOW_VERSION}")
    return {"status": "ok", "action": "install", "airflow_version": version, "venv": str(paths["venv"])}


def deploy(args: argparse.Namespace) -> dict[str, Any]:
    root = _safe_dir(args.root, "--root", create=True)
    paths = _paths(root)
    paths["dags"].mkdir(parents=True, exist_ok=True)
    paths["runtime"].mkdir(parents=True, exist_ok=True)
    skill_root = pathlib.Path(__file__).resolve().parent.parent
    sources = {
        skill_root / "assets" / "airflow" / "tao_deft_iaa_action_v1.py":
            paths["dags"] / "tao_deft_iaa_action_v1.py",
        pathlib.Path(__file__).resolve().parent / "airflow_dag_runtime.py":
            paths["runtime"] / "airflow_dag_runtime.py",
        pathlib.Path(__file__).resolve().parent / "airflow_orchestrator.py":
            paths["runtime"] / "airflow_orchestrator.py",
    }
    evidence = {}
    for source, destination in sources.items():
        if not source.is_file() or source.is_symlink():
            raise LocalAirflowError(f"packaged Airflow source is missing or unsafe: {source}")
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        evidence[destination.name] = _sha256(destination)
    return {"status": "ok", "action": "deploy", "dag_id": DAG_ID, "files": evidence}


def _token(base_url: str, timeout: int = 10) -> str:
    request = urllib.request.Request(base_url + "/auth/token", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise LocalAirflowError("local Airflow token endpoint is not ready") from exc
    token = payload.get("access_token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        raise LocalAirflowError("local Airflow token response is invalid")
    return token


def _api(base_url: str, path: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url + path, headers={"Authorization": f"Bearer {token}"}, method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise LocalAirflowError(f"local Airflow API request failed: {path}") from exc
    if not isinstance(payload, dict):
        raise LocalAirflowError(f"local Airflow API returned invalid JSON: {path}")
    return payload


def _configure_pools(airflow: pathlib.Path, env: dict[str, str]) -> None:
    for name, slots in POOL_SLOTS.items():
        subprocess.run(
            [str(airflow), "pools", "set", name, str(slots), "TAO IAA local Airflow"],
            check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def _scheduler_healthy(airflow: pathlib.Path, env: dict[str, str]) -> bool:
    completed = subprocess.run(
        [str(airflow), "jobs", "check", "--job-type", "SchedulerJob", "--local"],
        check=False,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def start(args: argparse.Namespace) -> dict[str, Any]:
    root = _safe_dir(args.root, "--root", create=True)
    shared = _safe_dir(args.shared_root, "--shared-root", create=True)
    paths = _paths(root)
    airflow = paths["venv"] / "bin" / "airflow"
    dag_path = paths["dags"] / "tao_deft_iaa_action_v1.py"
    runtime_path = paths["runtime"] / "airflow_dag_runtime.py"
    orchestrator_path = paths["runtime"] / "airflow_orchestrator.py"
    if not all(path.is_file() for path in (airflow, dag_path, runtime_path, orchestrator_path)):
        raise LocalAirflowError("run install and deploy before start")
    if paths["receipt"].is_file():
        prior = json.loads(paths["receipt"].read_text())
        pid = prior.get("pid")
        if isinstance(pid, int) and pathlib.Path(f"/proc/{pid}").exists():
            return status(args)
    if not _port_available(args.port):
        raise LocalAirflowError(f"loopback port {args.port} is already in use")
    env = _environment(root, shared, args.port)
    paths["airflow_home"].mkdir(parents=True, exist_ok=True)
    with paths["service_log"].open("ab") as service_log:
        process = subprocess.Popen(
            [str(airflow), "standalone"], env=env,
            stdin=subprocess.DEVNULL, stdout=service_log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = {
        "schema_version": "1", "service": "tao-iaa-local-airflow",
        "pid": process.pid, "process_group": process.pid, "port": args.port,
        "base_url": f"http://127.0.0.1:{args.port}", "airflow_version": AIRFLOW_VERSION,
        "shared_root": str(shared), "dag_id": DAG_ID,
        "dag_sha256": _sha256(dag_path), "runtime_sha256": _sha256(runtime_path),
        "orchestrator_sha256": _sha256(orchestrator_path),
        "service_log": str(paths["service_log"]),
    }
    _atomic_json(paths["receipt"], receipt)
    deadline = time.monotonic() + args.timeout
    token = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LocalAirflowError(f"local Airflow exited during startup with code {process.returncode}")
        try:
            token = _token(receipt["base_url"])
            break
        except LocalAirflowError:
            time.sleep(2)
    if token is None:
        raise LocalAirflowError(f"local Airflow readiness timed out after {args.timeout}s")
    scheduler_deadline = time.monotonic() + args.timeout
    while time.monotonic() < scheduler_deadline:
        if process.poll() is not None:
            raise LocalAirflowError(
                f"local Airflow exited while waiting for scheduler health; inspect {paths['service_log']}"
            )
        if _scheduler_healthy(airflow, env):
            break
        time.sleep(2)
    else:
        raise LocalAirflowError(
            f"local Airflow scheduler health timed out; inspect {paths['service_log']}"
        )
    _configure_pools(airflow, env)
    dag_deadline = time.monotonic() + args.timeout
    dag_payload = None
    while time.monotonic() < dag_deadline:
        try:
            dag_payload = _api(receipt["base_url"], f"/api/v2/dags/{DAG_ID}", token)
            if dag_payload.get("dag_id") == DAG_ID:
                break
        except LocalAirflowError:
            pass
        time.sleep(2)
    if not isinstance(dag_payload, dict) or dag_payload.get("dag_id") != DAG_ID:
        raise LocalAirflowError(f"packaged DAG did not load within {args.timeout}s")
    return status(args)


def status(args: argparse.Namespace) -> dict[str, Any]:
    root = _safe_dir(args.root, "--root")
    paths = _paths(root)
    if not paths["receipt"].is_file():
        raise LocalAirflowError("local Airflow has no service receipt")
    receipt = json.loads(paths["receipt"].read_text())
    pid = receipt.get("pid")
    running = isinstance(pid, int) and pathlib.Path(f"/proc/{pid}").exists()
    if not running:
        return {**receipt, "status": "stopped"}
    token = _token(receipt["base_url"])
    dag = _api(receipt["base_url"], f"/api/v2/dags/{DAG_ID}", token)
    shared = _safe_dir(pathlib.Path(receipt["shared_root"]), "receipt shared_root")
    env = _environment(root, shared, int(receipt["port"]))
    scheduler_healthy = _scheduler_healthy(paths["venv"] / "bin" / "airflow", env)
    return {
        **receipt, "status": "running" if scheduler_healthy else "degraded",
        "scheduler_healthy": scheduler_healthy,
        "dag_loaded": dag.get("dag_id") == DAG_ID,
        "dag_paused": bool(dag.get("is_paused")), "authentication": "loopback all-admin",
    }


def stop(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm:
        raise LocalAirflowError("stop requires --confirm")
    root = _safe_dir(args.root, "--root")
    paths = _paths(root)
    if not paths["receipt"].is_file():
        raise LocalAirflowError("local Airflow has no service receipt")
    receipt = json.loads(paths["receipt"].read_text())
    pid = receipt.get("pid")
    if not isinstance(pid, int) or not pathlib.Path(f"/proc/{pid}").exists():
        return {**receipt, "status": "stopped"}
    cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
    if "airflow" not in cmdline or "standalone" not in cmdline:
        raise LocalAirflowError("service receipt PID no longer identifies Airflow standalone")
    os.killpg(int(receipt["process_group"]), signal.SIGTERM)
    deadline = time.monotonic() + 30
    while pathlib.Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.5)
    return {**receipt, "status": "stopped"}


def _bounded_int(minimum: int, maximum: int, label: str):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must be an integer") from exc
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{label} must be in [{minimum}, {maximum}]"
            )
        return parsed

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "install", "deploy", "start", "status", "stop"):
        child = sub.add_parser(action)
        child.add_argument("--root", required=True, type=pathlib.Path)
        child.add_argument("--shared-root", type=pathlib.Path)
        child.add_argument(
            "--port", type=_bounded_int(1024, 65535, "port"), default=8081
        )
        if action == "start":
            child.add_argument(
                "--timeout", type=_bounded_int(30, 1800, "timeout"), default=300
            )
        if action == "stop":
            child.add_argument("--confirm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action in {"plan", "start"} and args.shared_root is None:
            raise LocalAirflowError(f"{args.action} requires --shared-root")
        result = {
            "plan": plan, "install": install, "deploy": deploy,
            "start": start, "status": status, "stop": stop,
        }[args.action](args)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (LocalAirflowError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"manage_local_airflow[{args.action}]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
