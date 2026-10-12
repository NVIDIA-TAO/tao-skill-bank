# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for deterministic PAS image-acquisition decisions."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
    monkeypatch.setenv("SLURM_USER", "cluster-user")
    commands = []
    monkeypatch.setattr(
        classifier,
        "_probe",
        lambda command, **kwargs: (
            commands.append(command)
            or subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        ),
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
    assert commands[0] == classifier.ssh_command(
        "login", "test -r /lustre/images/pyt.sqsh",
    )


def test_existing_slurm_registry_credentials_avoid_requiring_process_ngc_key(
    monkeypatch,
):
    monkeypatch.setenv("SLURM_USER", "cluster-user")
    monkeypatch.setenv("SSH_KEY_PATH", "/keys/cluster")
    commands = []

    def probe(command, **kwargs):
        commands.append(command)
        code = 0 if command[-1] == "test -s ~/.config/enroot/.credentials" else 1
        return subprocess.CompletedProcess(command, code, stdout="", stderr="")

    monkeypatch.setattr(classifier, "_probe", probe)
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
    assert commands[0][:-1] == commands[1][:-1]
    assert "cluster-user@login" in commands[1]
    assert "/keys/cluster" in commands[1]


def test_kubernetes_without_a_verified_pull_secret_requires_acquisition():
    report = classifier.classify(_args("--platform", "kubernetes"))

    assert report["image_acquisition"] == "required"
    assert report["ngc_key_requirement"] == "required"
    assert report["evidence"] == [
        {"resource": "imagePullSecret", "status": "missing"}
    ]


@pytest.mark.parametrize("secret_type", [
    "kubernetes.io/dockerconfigjson", "kubernetes.io/dockercfg",
])
def test_kubernetes_pull_secret_decouples_acquisition_from_local_ngc_key(
    monkeypatch, secret_type,
):
    monkeypatch.setattr(classifier, "_probe", lambda command:
                        subprocess.CompletedProcess(command, 0, stdout=secret_type))
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


def test_slurm_probes_use_platform_identity_and_quoted_remote_paths(monkeypatch):
    monkeypatch.setenv("SLURM_USER", "cluster-user")
    monkeypatch.setenv("SSH_KEY_PATH", "/keys/cluster key")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(classifier.subprocess, "run", run)
    classifier.classify(_args(
        "--platform", "slurm", "--slurm-host", "login",
        "--platform-image", "/lustre/image with spaces.sqsh",
    ))
    assert "cluster-user@login" in commands[0]
    assert commands[0][commands[0].index("-i") + 1] == "/keys/cluster key"
    assert "BatchMode=yes" in commands[0]
    assert commands[0][-1] == "test -r '/lustre/image with spaces.sqsh'"


def test_slurm_connection_failure_is_not_missing_image_or_credentials(monkeypatch):
    monkeypatch.setenv("SLURM_USER", "cluster-user")
    monkeypatch.setattr(classifier.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 255, stdout="", stderr=""))
    with pytest.raises(ValueError, match="probe.*failed"):
        classifier.classify(_args(
            "--platform", "slurm", "--slurm-host", "login",
            "--platform-image", "/lustre/pyt.sqsh",
        ))


def test_kubernetes_opaque_secret_is_not_registry_authentication(monkeypatch):
    monkeypatch.setattr(classifier.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, stdout="Opaque", stderr=""))
    report = classifier.classify(_args(
        "--platform", "kubernetes", "--kubernetes-image-pull-secret", "ordinary-secret",
    ))
    assert report["ngc_key_requirement"] == "required"
    assert report["evidence"][0]["status"] == "unsupported-type"


@pytest.mark.parametrize("code", [2, 127, 255])
def test_slurm_credential_probe_errors_block_classification(monkeypatch, code):
    monkeypatch.setenv("SLURM_USER", "cluster-user")
    def run(command, **kwargs):
        status = code if command[-1].startswith("test -s") else 1
        return subprocess.CompletedProcess(command, status, stdout="")
    monkeypatch.setattr(classifier.subprocess, "run", run)
    with pytest.raises(ValueError, match="probe.*failed"):
        classifier.classify(_args("--platform", "slurm", "--slurm-host", "login",
                                  "--platform-image", "/lustre/missing.sqsh"))


@pytest.mark.parametrize("failure", [OSError("missing CLI"), subprocess.TimeoutExpired("ssh", 20)])
def test_probe_invocation_errors_do_not_request_ngc_key(monkeypatch, failure, capsys):
    monkeypatch.setenv("SLURM_USER", "cluster-user")
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(classifier.subprocess, "run", run)
    assert classifier.main(["--platform", "slurm", "--slurm-host", "login",
                            "--platform-image", "/lustre/pyt.sqsh"]) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "probe failed" in output.err


@pytest.mark.parametrize("code,secret_type,expected", [
    (0, "", "required"),
    (0, "kubernetes.io/dockerconfigjson", "not-required"),
    (0, "kubernetes.io/dockercfg", "not-required"),
    (0, "Opaque", "required"),
    (1, "", None),
])
def test_kubernetes_probe_queries_type_only_and_preserves_context(
    monkeypatch, code, secret_type, expected,
):
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, code, stdout=secret_type)
    monkeypatch.setattr(classifier.subprocess, "run", run)
    args = _args("--platform", "kubernetes", "--kubernetes-image-pull-secret", "pull",
                 "--kubernetes-context", "cluster", "--kubernetes-namespace", "team")
    if expected is None:
        with pytest.raises(ValueError, match="kubectl probe failed"):
            classifier.classify(args)
    else:
        assert classifier.classify(args)["ngc_key_requirement"] == expected
    assert commands == [["kubectl", "--context", "cluster", "--namespace", "team",
                         "get", "secret", "pull", "--ignore-not-found", "-o", "jsonpath={.type}"]]


def test_slurm_ssh_builder_is_the_platform_preflight_builder(monkeypatch):
    import check_tao_launch_preflight as preflight
    assert classifier.ssh_command is preflight.ssh_command
    monkeypatch.setenv("SLURM_USER", "cluster-user")
    monkeypatch.delenv("SSH_KEY_PATH", raising=False)
    command = classifier.ssh_command("configured-alias", "true")
    assert "-i" not in command
    assert command[-2:] == ["cluster-user@configured-alias", "true"]
    assert "StrictHostKeyChecking=yes" in command
    monkeypatch.delenv("SLURM_USER")
    with pytest.raises(ValueError, match="SLURM_USER"):
        classifier.ssh_command("configured-alias", "true")


@pytest.mark.parametrize("platform,code,secret_type,expected", [
    ("slurm", 0, "", "not-required"),
    ("slurm", 1, "", "required"),
    ("slurm", 255, "", None),
    ("kubernetes", 0, "Opaque", "required"),
    ("kubernetes", 0, "kubernetes.io/dockerconfigjson", "not-required"),
    ("kubernetes", 0, "kubernetes.io/dockercfg", "not-required"),
    ("kubernetes", 0, "", "required"),
    ("kubernetes", 1, "", None),
])
def test_fresh_classifier_cli_with_controlled_native_command_responses(
    tmp_path, platform, code, secret_type, expected,
):
    """Run the actual public CLI; stand-ins model native exit/output contracts."""
    executable = tmp_path / ("ssh" if platform == "slurm" else "kubectl")
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "if os.path.basename(sys.argv[0]) == 'ssh':\n"
        "    assert args[-2] == 'cluster-user@login'\n"
        "    assert args[args.index('-i')+1] == '/keys/key with spaces'\n"
        "    assert 'BatchMode=yes' in args and 'StrictHostKeyChecking=yes' in args\n"
        "    assert args[-1] in [\"test -r '/lustre/image with spaces.sqsh'\",\n"
        "                         'test -s ~/.config/enroot/.credentials']\n"
        "else:\n"
        "    assert args == ['--context', 'cluster', '--namespace', 'team',\n"
        "                    'get', 'secret', 'pull', '--ignore-not-found',\n"
        "                    '-o', 'jsonpath={.type}']\n"
        "    print(os.environ['TEST_SECRET_TYPE'])\n"
        "print('diagnostic-must-not-leak', file=sys.stderr)\n"
        "sys.exit(int(os.environ['TEST_EXIT']))\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    environment = {
        **os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
        "SLURM_USER": "cluster-user", "SSH_KEY_PATH": "/keys/key with spaces",
        "TEST_EXIT": str(code), "TEST_SECRET_TYPE": secret_type,
    }
    options = (
        ["--slurm-host", "login", "--platform-image", "/lustre/image with spaces.sqsh"]
        if platform == "slurm" else
        ["--kubernetes-context", "cluster", "--kubernetes-namespace", "team",
         "--kubernetes-image-pull-secret", "pull"]
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--platform", platform, *options],
        env=environment, cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert "diagnostic-must-not-leak" not in result.stdout + result.stderr
    if expected is None:
        assert result.returncode == 2
        assert not result.stdout
        assert "probe failed" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["ngc_key_requirement"] == expected
