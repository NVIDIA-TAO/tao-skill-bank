# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for deterministic PAS image-acquisition decisions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "skills"
    / "applications"
    / "tao-run-deft-pas"
    / "scripts"
    / "classify_pas_image_acquisition.py"
)
SPEC = importlib.util.spec_from_file_location("classify_pas_image_acquisition", SCRIPT)
assert SPEC and SPEC.loader
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _args(*values: str):
    return classifier._parser().parse_args(values)  # noqa: SLF001


def test_docker_requires_acquisition_when_either_pinned_image_is_missing(
    monkeypatch,
):
    commands = []

    def present(command):
        commands.append(command)
        return command[-1] == "pyt:present"

    monkeypatch.setattr(classifier, "_present", present)
    report = classifier.classify(
        _args(
            "--platform",
            "docker",
            "--image",
            "pyt:present",
            "--image",
            "ds:missing",
        )
    )

    assert report["image_acquisition"] == "required"
    assert report["ngc_key_requirement"] == "required"
    assert commands == [
        ["docker", "image", "inspect", "pyt:present"],
        ["docker", "image", "inspect", "ds:missing"],
        ["docker", "manifest", "inspect", "ds:missing"],
    ]


def test_brev_inspects_each_image_with_one_remote_command_argument(monkeypatch):
    commands = []

    def present(command):
        commands.append(command)
        return True

    monkeypatch.setattr(classifier, "_present", present)
    report = classifier.classify(
        _args(
            "--platform",
            "brev",
            "--brev-instance",
            "pas-gpu",
            "--image",
            "nvcr.io/pyt:tag",
            "--image",
            "nvcr.io/ds:tag",
        )
    )

    assert report["image_acquisition"] == "not-required"
    assert report["ngc_key_requirement"] == "not-required"
    assert commands == [
        ["brev", "exec", "pas-gpu", "docker image inspect nvcr.io/pyt:tag"],
        ["brev", "exec", "pas-gpu", "docker image inspect nvcr.io/ds:tag"],
    ]


def test_slurm_checks_planned_sqsh_files_not_registry_uris(monkeypatch):
    commands = []
    monkeypatch.setattr(
        classifier,
        "_present",
        lambda command: commands.append(command) is None or True,
    )

    report = classifier.classify(
        _args(
            "--platform",
            "slurm",
            "--slurm-host",
            "login",
            "--platform-image",
            "/lustre/images/pyt.sqsh",
            "--platform-image",
            "/lustre/images/ds.sqsh",
        )
    )

    assert report["image_acquisition"] == "not-required"
    assert report["ngc_key_requirement"] == "not-required"
    assert commands[0] == [
        "ssh",
        "login",
        "test",
        "-r",
        "/lustre/images/pyt.sqsh",
    ]


def test_existing_slurm_registry_credentials_avoid_requiring_process_ngc_key(
    monkeypatch,
):
    def present(command):
        return command[-1] == "test -s ~/.config/enroot/.credentials"

    monkeypatch.setattr(classifier, "_present", present)
    report = classifier.classify(
        _args(
            "--platform",
            "slurm",
            "--slurm-host",
            "login",
            "--platform-image",
            "/lustre/images/missing.sqsh",
        )
    )

    assert report["image_acquisition"] == "required"
    assert report["ngc_key_requirement"] == "not-required"
    assert report["authentication_evidence"] == [
        {"resource": "enroot-credentials", "status": "present"}
    ]


def test_kubernetes_without_a_verified_pull_secret_requires_acquisition():
    report = classifier.classify(_args("--platform", "kubernetes"))

    assert report["image_acquisition"] == "required"
    assert report["ngc_key_requirement"] == "required"
    assert report["evidence"] == [
        {"resource": "imagePullSecret", "status": "missing"}
    ]


def test_kubernetes_pull_secret_decouples_acquisition_from_local_ngc_key(
    monkeypatch,
):
    monkeypatch.setattr(classifier, "_present", lambda command: True)
    report = classifier.classify(
        _args(
            "--platform",
            "kubernetes",
            "--kubernetes-image-pull-secret",
            "ngc-pull",
        )
    )

    assert report["image_acquisition"] == "required"
    assert report["ngc_key_requirement"] == "not-required"


def test_virtualenv_never_requires_container_image_acquisition():
    report = classifier.classify(_args("--platform", "virtualenv"))

    assert report["image_acquisition"] == "not-required"
    assert report["ngc_key_requirement"] == "not-required"
    assert report["evidence"] == []


def test_platform_specific_inputs_are_mandatory():
    with pytest.raises(ValueError, match="brev requires"):
        classifier.classify(_args("--platform", "brev", "--image", "pyt"))
    with pytest.raises(ValueError, match="slurm requires"):
        classifier.classify(_args("--platform", "slurm"))
