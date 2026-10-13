# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security regressions for PAS discovery-time credential checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "skills"
    / "applications"
    / "tao-run-deft-pas"
    / "scripts"
    / "check_pas_credentials.py"
)


def test_presence_report_never_emits_credential_values():
    sentinel = "must-not-appear-6871079"
    environment = {
        **os.environ,
        "NGC_KEY": sentinel,
        "HF_TOKEN": "",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            "--ngc-key-requirement",
            "required",
            "--optional",
            "HF_TOKEN",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert sentinel not in completed.stdout
    assert sentinel not in completed.stderr
    report = json.loads(completed.stdout)
    assert report == {
        "schema_version": "1",
        "ngc_key_requirement": "required",
        "credentials": [
            {"name": "NGC_KEY", "requirement": "required", "status": "set"},
            {"name": "HF_TOKEN", "requirement": "optional", "status": "missing"},
        ],
        "missing_required": [],
    }


def test_missing_required_credential_is_a_value_free_failure():
    environment = os.environ.copy()
    environment.pop("PAS_TEST_REQUIRED_TOKEN", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            "--ngc-key-requirement",
            "not-required",
            "--required",
            "PAS_TEST_REQUIRED_TOKEN",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["missing_required"] == ["PAS_TEST_REQUIRED_TOKEN"]
    assert report["credentials"][0]["status"] == "missing"
    assert report["credentials"][1] == {
        "name": "NGC_KEY",
        "requirement": "optional",
        "status": "missing",
    }


def test_image_acquisition_is_the_only_ngc_requirement_authority():
    environment = os.environ.copy()
    environment.pop("NGC_KEY", None)

    required = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            "--ngc-key-requirement",
            "required",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert required.returncode == 2
    assert json.loads(required.stdout)["missing_required"] == ["NGC_KEY"]

    duplicate = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            "--ngc-key-requirement",
            "not-required",
            "--required",
            "NGC_KEY",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert duplicate.returncode == 2
    assert "classified only by --ngc-key-requirement" in duplicate.stderr
