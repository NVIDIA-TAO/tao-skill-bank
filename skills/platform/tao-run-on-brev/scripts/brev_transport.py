#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe transport helpers for current Brev CLI output and stdin semantics."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from typing import BinaryIO, Callable, Sequence


READY_MARKER = "TAO_BREV_READY"
SAFE_INSTANCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SAFE_REGISTRY = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:-]*")
SAFE_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _strip_instance_footer(stdout: str | bytes, instance: str) -> str | bytes:
    """Remove only Brev's exact trailing instance-name line from stdout."""
    encoded = isinstance(stdout, bytes)
    marker = instance.encode("utf-8") if encoded else instance
    newlines = (b"\r\n", b"\n") if encoded else ("\r\n", "\n")
    for newline in newlines:
        footer = marker + newline
        if stdout.endswith(footer):
            prefix = stdout[: -len(footer)]
            if any(prefix.endswith(item) for item in newlines):
                return prefix
    return stdout


def _validate_remote_command(command: str) -> str:
    if not command or "\x00" in command or "\n" in command or "\r" in command:
        raise ValueError("remote command must be one non-empty line without NUL bytes")
    return command


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"required executable is missing: {name}")
    return executable


def _validate_instance(instance: str) -> str:
    if not SAFE_INSTANCE.fullmatch(instance):
        raise ValueError("instance must contain only letters, digits, dot, underscore, or hyphen")
    return instance


def check_ready(
    instance: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    brev_executable: str | None = None,
) -> bool:
    """Return true only when the remote command succeeds and emits the marker."""
    executable = brev_executable or _require_executable("brev")
    completed = runner(
        [executable, "exec", _validate_instance(instance), f"printf '{READY_MARKER}\\n'"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        return False
    # Current Brev releases may append an instance-name footer.  Match one
    # complete marker line instead of requiring byte-identical stdout.
    return READY_MARKER in completed.stdout.splitlines()


def run_remote(
    instance: str,
    command: str,
    *,
    environment: dict[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    brev_executable: str | None = None,
    ssh_executable: str | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Run one remote command, forwarding secrets through stdin when needed.

    Brev requires the complete remote command to be the final, single argument.
    It also consumes stdin itself, so explicitly forwarded environment values use
    the Brev-managed SSH alias and a small remote Python bootstrap.  Values are
    JSON on stdin and never appear in either the local or remote process argv.
    """
    instance = _validate_instance(instance)
    command = _validate_remote_command(command)
    environment = environment or {}
    if not isinstance(environment, dict):
        raise ValueError("environment must be a name-to-value object")
    for name, value in environment.items():
        if not SAFE_ENV_NAME.fullmatch(name):
            raise ValueError(f"invalid environment variable name: {name!r}")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"environment variable {name} must be a NUL-free string")

    if not environment:
        executable = brev_executable or _require_executable("brev")
        completed = runner(
            [executable, "exec", instance, command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        completed.stdout = _strip_instance_footer(completed.stdout or "", instance)
        return completed

    executable = ssh_executable or _require_executable("ssh")
    names = sorted(environment)
    bootstrap = (
        "import json,os,sys;"
        "p=json.load(sys.stdin);n=sys.argv[2:];"
        "set(p)==set(n) or (_ for _ in ()).throw(ValueError('environment names'));"
        "os.environ.update({k:p[k] for k in n});"
        "os.execvp('sh',['sh','-lc',sys.argv[1]])"
    )
    remote_command = shlex.join(["python3", "-c", bootstrap, command, *names])
    return runner(
        [executable, "-o", "BatchMode=yes", instance, remote_command],
        input=json.dumps(environment, separators=(",", ":")).encode("utf-8"),
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def registry_login(
    instance: str,
    registry: str,
    username: str,
    *,
    password_stream: BinaryIO,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ssh_executable: str | None = None,
) -> int:
    """Forward password stdin over Brev's SSH alias without putting it on argv."""
    if not SAFE_REGISTRY.fullmatch(registry):
        raise ValueError("registry contains unsupported characters")
    if not username:
        raise ValueError("username must not be empty")
    executable = ssh_executable or _require_executable("ssh")
    remote_command = shlex.join(
        [
            "docker",
            "login",
            registry,
            "--username",
            username,
            "--password-stdin",
        ]
    )
    completed = runner(
        [
            executable,
            "-o",
            "BatchMode=yes",
            _validate_instance(instance),
            remote_command,
        ],
        stdin=password_stream,
        check=False,
    )
    return int(completed.returncode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="verb", required=True)
    ready = subparsers.add_parser("ready")
    ready.add_argument("--instance", required=True)
    login = subparsers.add_parser("registry-login")
    login.add_argument("--instance", required=True)
    login.add_argument("--registry", default="nvcr.io")
    login.add_argument("--username", default="$oauthtoken")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verb == "ready":
            if check_ready(args.instance):
                print(READY_MARKER)
                return 0
            print("Brev instance is not exec-ready", file=sys.stderr)
            return 1
        if sys.stdin.isatty():
            raise RuntimeError("registry password must be supplied on stdin")
        return registry_login(
            args.instance,
            args.registry,
            args.username,
            password_stream=sys.stdin.buffer,
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"brev transport failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
