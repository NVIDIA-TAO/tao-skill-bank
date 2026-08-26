#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run one external command under the immutable DEFT execution policy."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any


PACKAGE_TOOLS = {
    "apt", "apt-get", "conda", "dnf", "mamba", "micromamba", "pip",
    "pip3", "uv", "yum",
}
NETWORK_TOOLS = {
    "aria2c",
    "aws",
    "curl",
    "git-lfs",
    "http",
    "httpie",
    "ngc",
    "rsync",
    "s3cmd",
    "scp",
    "sftp",
    "ssh",
    "wget",
}
# Launchers that hand the real command to another scheduler, host, or runtime.
# The air-gap checks below can only reason about a local docker/podman argv, so
# these are refused rather than waved through unexamined.
REMOTE_LAUNCHERS = {
    "apptainer",
    "enroot",
    "kubectl",
    "nerdctl",
    "sbatch",
    "singularity",
    "srun",
}
SHELLS = {"bash", "dash", "ksh", "sh", "zsh"}
OFFLINE_ENV = {
    "AIR_GAPPED": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PIP_NO_INDEX": "1",
}


def _policy(state_path: pathlib.Path) -> dict[str, Any]:
    state = json.loads(state_path.expanduser().read_text())
    policy = state.get("execution_policy")
    if isinstance(policy, dict):
        return policy
    airgap = os.environ.get("AIR_GAPPED") == "1"
    return {
        "network_mode": "airgap" if airgap else "network-enabled",
        "allow_package_install": not airgap,
        "allow_remote_fetch": not airgap,
        "allow_container_pull": not airgap,
        "allow_registry_login": not airgap,
    }


def _basename(value: str) -> str:
    return pathlib.PurePath(value).name.lower()


def _reject_airgap(command: list[str], policy: dict[str, Any]) -> None:
    if policy.get("network_mode") != "airgap":
        return
    if not command:
        raise ValueError("command is empty")
    tokens = [_basename(token) for token in command]
    if any(token in REMOTE_LAUNCHERS for token in tokens):
        raise ValueError(
            "remote launchers are not supported by this execution path; it can "
            "enforce air-gap policy for local docker/podman commands only"
        )
    if any(token in PACKAGE_TOOLS for token in tokens):
        raise ValueError("package installation is forbidden by air-gap policy")
    if any(token in NETWORK_TOOLS for token in tokens):
        raise ValueError("network/download command is forbidden by air-gap policy")
    for index, token in enumerate(tokens[:-1]):
        if token.startswith("python") and tokens[index + 1 : index + 3] == ["-m", "pip"]:
            raise ValueError("python -m pip is forbidden by air-gap policy")
    if "git" in tokens and any(
        token in {"clone", "fetch", "pull"}
        for token in tokens[tokens.index("git") + 1 :]
    ):
        raise ValueError("remote git operation is forbidden by air-gap policy")
    runtime_indexes = [index for index, token in enumerate(tokens) if token in {"docker", "podman"}]
    if runtime_indexes:
        runtime_index = runtime_indexes[0]
        runtime_tokens = tokens[runtime_index + 1 :]
        if any(token in {"login", "pull"} for token in runtime_tokens):
            raise ValueError("container registry access is forbidden by air-gap policy")
        if "manifest" in runtime_tokens:
            raise ValueError("container manifest lookup is forbidden by air-gap policy")
        pull_values = [token for token in runtime_tokens if token.startswith("--pull=")]
        if any(token != "--pull=never" for token in pull_values):
            raise ValueError("air-gap container runs require --pull=never")
        if "--pull" in runtime_tokens:
            index = runtime_tokens.index("--pull")
            if index + 1 >= len(runtime_tokens) or runtime_tokens[index + 1] != "never":
                raise ValueError("air-gap container runs require --pull never")
        for token in runtime_tokens:
            normalized = token.removeprefix("--env=").removeprefix("-e")
            if "=" not in normalized:
                continue
            name, value = normalized.split("=", 1)
            name = name.upper()
            if name in OFFLINE_ENV and value != OFFLINE_ENV[name]:
                raise ValueError(f"air-gap container cannot override {name}={value}")
        for index, token in enumerate(runtime_tokens[:-1]):
            if token not in {"-e", "--env"} or "=" not in runtime_tokens[index + 1]:
                continue
            name, value = runtime_tokens[index + 1].split("=", 1)
            name = name.upper()
            if name in OFFLINE_ENV and value != OFFLINE_ENV[name]:
                raise ValueError(f"air-gap container cannot override {name}={value}")
    # Recurse into an inline shell payload wherever the shell appears, not only
    # at argv[0]: `<launcher> bash -c '<payload>'` hides the real command.
    for index, token in enumerate(tokens):
        if token not in SHELLS:
            continue
        for offset in range(index + 1, len(tokens) - 1):
            flag = tokens[offset]
            if not flag.startswith("-"):
                break
            if "c" not in flag[1:]:
                continue
            try:
                nested = shlex.split(command[offset + 1])
            except ValueError as err:
                raise ValueError(
                    f"air-gap policy cannot parse the inline shell payload: {err}"
                ) from err
            _reject_airgap(nested, policy)
            break


def _with_no_pull(command: list[str], policy: dict[str, Any]) -> list[str]:
    if policy.get("network_mode") != "airgap" or not command:
        return command
    tokens = [_basename(token) for token in command]
    runtime_indexes = [index for index, token in enumerate(tokens) if token in {"docker", "podman"}]
    if not runtime_indexes or "run" not in tokens[runtime_indexes[0] + 1 :]:
        return command
    run_index = tokens.index("run", runtime_indexes[0] + 1)
    if any(token == "--pull" or token.startswith("--pull=") for token in tokens):
        return command
    return [*command[: run_index + 1], "--pull=never", *command[run_index + 1 :]]


def _with_offline_container_env(
    command: list[str], policy: dict[str, Any]
) -> list[str]:
    if policy.get("network_mode") != "airgap" or not command:
        return command
    tokens = [_basename(token) for token in command]
    runtime_indexes = [index for index, token in enumerate(tokens) if token in {"docker", "podman"}]
    if not runtime_indexes or "run" not in tokens[runtime_indexes[0] + 1 :]:
        return command
    run_index = tokens.index("run", runtime_indexes[0] + 1)
    options = [f"--env={name}={value}" for name, value in OFFLINE_ENV.items()]
    return [*command[: run_index + 1], *options, *command[run_index + 1 :]]


def run(state_path: pathlib.Path, command: list[str]) -> int:
    policy = _policy(state_path)
    _reject_airgap(command, policy)
    command = _with_no_pull(command, policy)
    command = _with_offline_container_env(command, policy)
    environment = os.environ.copy()
    if policy.get("network_mode") == "airgap":
        environment.update(OFFLINE_ENV)
    return subprocess.run(command, env=environment, check=False).returncode


def _launcher():
    """The bank's shared four-verb launcher.

    submit/status/logs/cancel over a spec-bundle are identical for every DEFT
    workflow -- nothing in them mentions a stage, a state file or a network
    architecture -- so they live at bank root beside tao_job_record.py rather
    than being copied into each workflow. This module keeps what IS specific to
    Cosmos3: its air-gap policy and its state file.
    """
    import importlib.util

    path = _bank() / "scripts" / "tao_launch.py"
    if not path.is_file():
        raise ValueError(
            f"the shared launcher is missing at {path}; set TAO_SKILL_BANK_PATH "
            "to a bank that ships scripts/tao_launch.py"
        )
    spec = importlib.util.spec_from_file_location("tao_launch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tao_launch"] = module
    spec.loader.exec_module(module)
    return module


def _bank() -> pathlib.Path:
    env = os.environ.get("TAO_SKILL_BANK_PATH")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    return pathlib.Path(__file__).resolve().parents[4]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=pathlib.Path)
    # The four verbs, delegated to the shared launcher. Which backend to ask
    # comes from the job RECORD, not a flag, so a job can be awaited, read or
    # cancelled from a session that never launched it.
    parser.add_argument("--submit", action="store_true",
                        help="launch a spec-bundle and print the job id")
    parser.add_argument("--bundle", type=pathlib.Path)
    parser.add_argument("--await-job", metavar="JOB_ID",
                        help="poll to a terminal state and close the record")
    parser.add_argument("--logs", metavar="JOB_ID")
    parser.add_argument("--cancel", metavar="JOB_ID")
    parser.add_argument("--platform", default="docker")
    parser.add_argument("--ctx", action="append", metavar="KEY=VALUE", default=[])
    parser.add_argument("--tail", type=int, default=200)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command

    try:
        verbs = [name for name in ("submit", "await_job", "logs", "cancel")
                 if getattr(args, name)]
        if len(verbs) > 1:
            raise ValueError(
                f"{', '.join('--' + v.replace('_', '-') for v in verbs)} are "
                "separate steps"
            )
        if verbs and command:
            raise ValueError(
                "a verb and a trailing command are different modes: the verbs "
                "launch a bundle, `-- <cmd>` runs one command under the policy"
            )
        if verbs:
            launcher = _launcher()
            ctx_extra = dict(item.split("=", 1) for item in args.ctx)
            if args.logs:
                return launcher.job_logs(args.logs, args.tail, ctx_extra)
            if args.cancel:
                return launcher.cancel_job(args.cancel, ctx_extra)
            if args.await_job:
                return launcher.await_job(
                    args.await_job, poll_seconds=args.poll_seconds,
                    timeout_seconds=args.timeout_seconds, ctx_extra=ctx_extra)
            if not args.bundle:
                raise ValueError("--submit needs --bundle")
            bundle = launcher.load_bundle(args.bundle, _policy(args.state))
            state = json.loads(args.state.expanduser().read_text())
            results_dir = state.get("results_dir")
            if not results_dir:
                raise ValueError(f"{args.state} has no results_dir")
            print(launcher.submit_bundle(
                args.state, bundle, results_dir=results_dir,
                platform=args.platform, ctx_extra=ctx_extra))
            return 0
        return run(args.state, command)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"deft_exec: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
