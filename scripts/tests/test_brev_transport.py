# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Brev readiness output and secure stdin forwarding."""

from __future__ import annotations

import importlib.util
import io
import shlex
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "skills"
    / "platform"
    / "tao-run-on-brev"
    / "scripts"
    / "brev_transport.py"
)
SPEC = importlib.util.spec_from_file_location("brev_transport", SCRIPT)
assert SPEC and SPEC.loader
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


@pytest.mark.parametrize(
    "stdout",
    [
        transport.READY_MARKER + "\n",
        transport.READY_MARKER + "\ntao-iaa-brev-smoke\n",
        "banner\n" + transport.READY_MARKER + "\ninstance\n",
    ],
)
def test_ready_accepts_marker_line_and_tolerates_cli_footer(stdout):
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    assert transport.check_ready(
        "tao-iaa-brev-smoke", runner=runner, brev_executable="/usr/bin/brev"
    )
    argv, kwargs = calls[0]
    assert argv == [
        "/usr/bin/brev",
        "exec",
        "tao-iaa-brev-smoke",
        f"printf '{transport.READY_MARKER}\\n'",
    ]
    assert kwargs["timeout"] == 600


@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [(1, transport.READY_MARKER + "\n"), (0, "tao-iaa-brev-smoke\n")],
)
def test_ready_requires_success_and_exact_marker_line(returncode, stdout):
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")

    assert not transport.check_ready(
        "tao-iaa-brev-smoke", runner=runner, brev_executable="brev"
    )


def test_registry_login_forwards_stdin_through_ssh_without_secret_argv():
    password = io.BytesIO(b"secret-material-never-on-argv")
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    assert (
        transport.registry_login(
            "tao-iaa-brev-smoke",
            "nvcr.io",
            "$oauthtoken",
            password_stream=password,
            runner=runner,
            ssh_executable="/usr/bin/ssh",
        )
        == 0
    )
    argv, kwargs = calls[0]
    assert argv[:5] == [
        "/usr/bin/ssh",
        "-o",
        "BatchMode=yes",
        "tao-iaa-brev-smoke",
        shlex.join(
            [
                "docker",
                "login",
                "nvcr.io",
                "--username",
                "$oauthtoken",
                "--password-stdin",
            ]
        ),
    ]
    assert kwargs["stdin"] is password
    assert b"secret-material" not in " ".join(argv).encode()


@pytest.mark.parametrize("instance", ["-option", "bad name", "name;command"])
def test_transport_rejects_unsafe_instance_names(instance):
    with pytest.raises(ValueError):
        transport.check_ready(instance, brev_executable="brev")
