# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Brev readiness output and secure stdin forwarding."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import shlex
import subprocess
import sys
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

sys.path.insert(0, str(SCRIPT.parent))
ACTION_SCRIPT = SCRIPT.parent / "brev_action.py"
ACTION_SPEC = importlib.util.spec_from_file_location("brev_action", ACTION_SCRIPT)
assert ACTION_SPEC and ACTION_SPEC.loader
action = importlib.util.module_from_spec(ACTION_SPEC)
ACTION_SPEC.loader.exec_module(action)


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


def test_remote_command_is_one_brev_argument():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    result = transport.run_remote(
        "tao-iaa-brev-smoke",
        "docker inspect job-id",
        runner=runner,
        brev_executable="/usr/bin/brev",
    )

    assert result.returncode == 0
    argv, kwargs = calls[0]
    assert argv == [
        "/usr/bin/brev",
        "exec",
        "tao-iaa-brev-smoke",
        "docker inspect job-id",
    ]
    assert kwargs["timeout"] == 600


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("[{}]\ntao-iaa-brev-smoke\n", "[{}]\n"),
        ("log line\r\ntao-iaa-brev-smoke\r\n", "log line\r\n"),
        (b"binary log\ntao-iaa-brev-smoke\n", b"binary log\n"),
        ("tao-iaa-brev-smoke\npayload\n", "tao-iaa-brev-smoke\npayload\n"),
        ("payload\ntao-iaa-brev-smoke-extra\n", "payload\ntao-iaa-brev-smoke-extra\n"),
        ("tao-iaa-brev-smoke\n", "tao-iaa-brev-smoke\n"),
    ],
)
def test_remote_strips_only_exact_appended_instance_footer(stdout, expected):
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    result = transport.run_remote(
        "tao-iaa-brev-smoke",
        "docker inspect job-id",
        runner=runner,
        brev_executable="/usr/bin/brev",
    )

    assert result.stdout == expected


def test_remote_forwarded_environment_uses_ssh_stdin_not_argv():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok\n", stderr=b"")

    secret = "hf_secret-material-never-on-argv"
    result = transport.run_remote(
        "tao-iaa-brev-smoke",
        "docker run -e HF_TOKEN image command",
        environment={"HF_TOKEN": secret},
        runner=runner,
        ssh_executable="/usr/bin/ssh",
    )

    assert result.returncode == 0
    argv, kwargs = calls[0]
    assert argv[:4] == [
        "/usr/bin/ssh",
        "-o",
        "BatchMode=yes",
        "tao-iaa-brev-smoke",
    ]
    assert secret not in " ".join(argv)
    assert json.loads(kwargs["input"])["HF_TOKEN"] == secret
    assert kwargs["timeout"] == 600


@pytest.mark.parametrize("instance", ["-option", "bad name", "name;command"])
def test_transport_rejects_unsafe_instance_names(instance):
    with pytest.raises(ValueError):
        transport.check_ready(instance, brev_executable="brev")


def _signed_request(tmp_path: Path, *, gpu_ids=(6, 7), forward_env=()):
    results = "/remote/iaa/run"
    payload = {
        "platform": "brev",
        "record_image": "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
        "gpu_ids": list(gpu_ids),
        "environment": {
            "HOME": "/tmp",
            "PYTHONPATH": "/patches",
            "HF_HOME": "/cache/huggingface",
        },
        "forward_env": list(forward_env),
        "mounts": [
            {"source": "/launcher/results", "target": "/results", "read_only": False},
            {"source": "/launcher/specs", "target": "/specs", "read_only": True},
            {"source": "/launcher/cache", "target": "/cache", "read_only": False},
        ],
        "spec_bundle": {
            "compute_shape": {"gpus": len(gpu_ids), "nodes": 1},
            "command": "clip",
            "args": ["train", "-e", "/specs/train.yaml", f"results_dir={results}"],
        },
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["request_sha256"] = hashlib.sha256(encoded).hexdigest()
    path = tmp_path / "train.action.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _remote_mounts():
    return [
        "/results=/mnt/iaa/run",
        "/specs=/mnt/iaa/run/config",
        "/cache=/mnt/iaa/cache",
    ]


def _signed_adapter_request(tmp_path: Path, *, name="gap_analysis", label="iter1"):
    results = "/launcher/results"
    stage = f"{results}/iter_1/gaps"
    fresh = [f"{stage}/kpi_gaps.parquet"]
    controller_root = "/launcher/runtime/input-snapshots/skills"
    runtime = (
        f"{controller_root}/"
        "skills/applications/tao-run-deft-iaa/scripts"
    )
    patches = "/launcher/runtime/input-snapshots/patches"
    controller_entries = [
        {
            "path": "skills/applications/tao-run-deft-iaa/references/pipeline-and-state.md",
            "size": 5,
            "sha256": "0" * 64,
        },
        {
            "path": "skills/applications/tao-run-deft-iaa/scripts/iaa_deft/controller.py",
            "size": 10,
            "sha256": "1" * 64,
        },
        {
            "path": "skills/applications/tao-run-deft-iaa/scripts/run_iaa_compute.py",
            "size": 20,
            "sha256": "2" * 64,
        },
        {
            "path": "skills/core/tao-artifacts/references/spec_bundle.schema.json",
            "size": 25,
            "sha256": "4" * 64,
        },
    ]
    patches_entries = [
        {"path": "sitecustomize.py", "size": 30, "sha256": "3" * 64},
    ]
    controller_snapshot = {
        "root": controller_root,
        "entries": controller_entries,
        "sha256": action._sha256_json({"entries": controller_entries}),  # noqa: SLF001
    }
    patches_snapshot = {
        "root": patches,
        "entries": patches_entries,
        "sha256": action._sha256_json({"entries": patches_entries}),  # noqa: SLF001
    }
    image = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services"
    payload = {
        "schema_version": "1",
        "workflow": "tao-run-deft-iaa",
        "runtime_sha256": "a" * 64,
        "controller_snapshot": controller_snapshot,
        "patches_snapshot": patches_snapshot,
        "platform": "brev",
        "name": name,
        "attempt": 1,
        "label": label,
        "record_image": image,
        "workload_image": image,
        "gpu_ids": [],
        "passed_hf_token": False,
        "forward_env": [],
        "spec_bundle": {
            "network_arch": "iaa-adapter",
            "action": f"deft-iaa-{name}-0123456789abcdef",
            "image": image,
            "mode": "args",
            "compute_shape": {"gpus": 0, "nodes": 1},
            "command": "python3",
            "args": [
                action.IAA_ADAPTER_SCRIPT,
                name,
                "--results-dir",
                "/results",
                "--label",
                label,
            ],
            "declared_inputs": [
                {"spec_key": "workflow_results", "type": "folder", "uri": results},
                {
                    "spec_key": "iaa_runtime",
                    "type": "folder",
                    "uri": controller_root,
                },
                {
                    "spec_key": "compatibility_patches",
                    "type": "folder",
                    "uri": patches,
                },
            ],
        },
        "mounts": [
            {"source": results, "target": "/results", "read_only": False},
            {"source": runtime, "target": "/iaa-runtime", "read_only": True},
            {"source": patches, "target": "/patches", "read_only": True},
        ],
        "environment": dict(action.IAA_ADAPTER_ENVIRONMENT),
        "results_dir": results,
        "stage_dir": stage,
        "log_path": f"{stage}/{name}.log",
        "fresh_outputs": fresh,
        "staging_absent_paths": [*fresh, f"{stage}/{name}.log"],
        "freshness_contract": "remote-mirror-with-delete-before-submit",
        "staging_receipt_path": f"{stage}/{name}.staged.json",
        "job_binding_path": f"{stage}/{name}.job-binding.json",
    }
    payload["request_sha256"] = action._sha256_json(payload)  # noqa: SLF001
    path = tmp_path / f"{name}.action.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _write_signed(path: Path, payload: dict):
    payload.pop("request_sha256", None)
    payload["request_sha256"] = action._sha256_json(payload)  # noqa: SLF001
    path.write_text(json.dumps(payload), encoding="utf-8")


def _adapter_remote_mounts():
    return [
        "/results=/mnt/iaa/results",
        (
            "/iaa-runtime=/mnt/iaa/staged/skills-aabbccdd/"
            "skills/applications/tao-run-deft-iaa/scripts"
        ),
        "/patches=/mnt/iaa/staged/patches-aabbccdd",
    ]


def _remote_snapshot_output(request, field, *, entries=None):
    snapshot = request[field]
    observed_entries = snapshot["entries"] if entries is None else entries
    observed = {
        "entries": observed_entries,
        "sha256": action._sha256_json({"entries": observed_entries}),  # noqa: SLF001
    }
    if field == "controller_snapshot":
        observed["runtime_sha256"] = request["runtime_sha256"]
    return action.SNAPSHOT_MARKER + json.dumps(observed, sort_keys=True) + "\n"


def test_brev_action_preserves_explicit_gpu_ids_and_mounts(tmp_path):
    path, _ = _signed_request(tmp_path)
    request = action.load_request(path)
    mounts = action.validate_mounts(request, _remote_mounts())
    command = action.build_submit_command(
        request,
        "clip-deft-iaa-train-abc123",
        mounts,
        uid=1000,
        gid=1000,
        username="ubuntu",
        groups=[1000, 44],
    )
    argv = shlex.split(command)

    assert argv[argv.index("--gpus") + 1] == '"device=6,7"'
    assert "--gpus all" not in command
    assert "type=bind,src=/mnt/iaa/run,dst=/results" in argv
    assert "type=bind,src=/mnt/iaa/run/config,dst=/specs,readonly" in argv
    assert argv[-5:] == [
        "clip",
        "train",
        "-e",
        "/specs/train.yaml",
        "results_dir=/remote/iaa/run",
    ]


def test_brev_action_accepts_typed_cpu_adapter_and_omits_gpu_flag(tmp_path):
    path, _ = _signed_adapter_request(tmp_path)
    request = action.load_request(path)
    mounts = action.validate_mounts(request, _adapter_remote_mounts())
    command = action.build_submit_command(
        request,
        "deft-iaa-gap-analysis-abc123",
        mounts,
        uid=1000,
        gid=1000,
        username="ubuntu",
        groups=[1000],
    )
    argv = shlex.split(command)

    assert "--gpus" not in argv
    assert f"tao-action=gap_analysis" in argv
    assert f"tao-request-sha256={request['request_sha256']}" in argv
    assert f"tao-runtime-sha256={request['runtime_sha256']}" in argv
    assert (
        "type=bind,src=/mnt/iaa/staged/skills-aabbccdd/"
        "skills/applications/tao-run-deft-iaa/scripts,dst=/iaa-runtime,readonly"
        in argv
    )
    assert argv[-8:] == [
        "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services",
        "python3",
        action.IAA_ADAPTER_SCRIPT,
        "gap_analysis",
        "--results-dir",
        "/results",
        "--label",
        "iter1",
    ]


def test_brev_action_accepts_visualize_finish_thread_caps(tmp_path):
    path, payload = _signed_adapter_request(tmp_path, name="visualize_finish")
    payload["environment"].update(action.IAA_VISUALIZE_THREAD_CAPS)
    _write_signed(path, payload)

    request = action.load_request(path)

    assert all(
        request["environment"][name] == value
        for name, value in action.IAA_VISUALIZE_THREAD_CAPS.items()
    )


def test_brev_action_verifies_staged_executable_trees_on_instance(tmp_path, monkeypatch):
    path, _ = _signed_adapter_request(tmp_path)
    request = action.load_request(path)
    mounts = action.validate_mounts(request, _adapter_remote_mounts())
    calls = []

    def run_remote(instance, command, **kwargs):
        calls.append((instance, command, kwargs))
        kind = shlex.split(command)[-1]
        field = "controller_snapshot" if kind == "controller" else "patches_snapshot"
        output = _remote_snapshot_output(request, field)
        return subprocess.CompletedProcess([], 0, stdout=output, stderr="")

    monkeypatch.setattr(action, "run_remote", run_remote)
    action.verify_remote_adapter_snapshots("tao-iaa-brev-smoke", request, mounts)

    assert calls[0][0] == "tao-iaa-brev-smoke"
    assert shlex.split(calls[0][1])[-2:] == [
        "/mnt/iaa/staged/skills-aabbccdd",
        "controller",
    ]
    assert shlex.split(calls[1][1])[-2:] == [
        "/mnt/iaa/staged/patches-aabbccdd",
        "patches",
    ]
    assert "base64.b64decode" in shlex.split(calls[0][1])[-3]
    assert transport._validate_remote_command(calls[0][1]) == calls[0][1]  # noqa: SLF001


def test_brev_snapshot_probe_is_accepted_by_real_transport_validator():
    command = action._remote_python_command(  # noqa: SLF001
        action.REMOTE_SNAPSHOT_CODE,
        "/remote/snapshot",
        "controller",
    )
    assert transport._validate_remote_command(command) == command  # noqa: SLF001
    assert "\n" not in command


@pytest.mark.parametrize(
    ("field", "change"),
    [
        ("controller_snapshot", "mutation"),
        ("controller_snapshot", "extra"),
        ("controller_snapshot", "missing"),
        ("patches_snapshot", "mutation"),
        ("patches_snapshot", "extra"),
        ("patches_snapshot", "missing"),
    ],
)
def test_brev_action_rejects_staged_executable_tree_drift(
    tmp_path, monkeypatch, field, change
):
    path, _ = _signed_adapter_request(tmp_path)
    request = action.load_request(path)
    mounts = action.validate_mounts(request, _adapter_remote_mounts())

    altered = json.loads(json.dumps(request[field]["entries"]))
    if change == "mutation":
        altered[0]["sha256"] = "f" * 64
    elif change == "extra":
        altered.append({"path": "unexpected.py", "size": 1, "sha256": "e" * 64})
    else:
        altered.pop()

    def run_remote(instance, command, **kwargs):
        kind = shlex.split(command)[-1]
        current = "controller_snapshot" if kind == "controller" else "patches_snapshot"
        entries = altered if current == field else None
        return subprocess.CompletedProcess(
            [], 0, stdout=_remote_snapshot_output(request, current, entries=entries), stderr=""
        )

    monkeypatch.setattr(action, "run_remote", run_remote)
    expected_kind = "controller" if field == "controller_snapshot" else "patches"
    with pytest.raises(RuntimeError, match=f"staged Brev IAA {expected_kind} tree"):
        action.verify_remote_adapter_snapshots("tao-iaa-brev-smoke", request, mounts)


def test_brev_action_rejects_zero_gpu_tao_or_arbitrary_python(tmp_path):
    path, payload = _signed_request(tmp_path, gpu_ids=())
    _write_signed(path, payload)
    with pytest.raises(ValueError, match="zero-GPU execution"):
        action.load_request(path)

    path, payload = _signed_adapter_request(tmp_path)
    payload["name"] = "run_python"
    payload["spec_bundle"]["network_arch"] = "python"
    payload["spec_bundle"]["args"][1] = "run_python"
    _write_signed(path, payload)
    with pytest.raises(ValueError, match="zero-GPU execution"):
        action.load_request(path)


def test_brev_action_rejects_gpu_backed_adapter(tmp_path):
    path, payload = _signed_adapter_request(tmp_path)
    payload["gpu_ids"] = [3]
    payload["spec_bundle"]["compute_shape"]["gpus"] = 1
    _write_signed(path, payload)

    with pytest.raises(ValueError, match="must use compute_shape.gpus=0"):
        action.load_request(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: next(
                item for item in payload["mounts"] if item["target"] == "/iaa-runtime"
            ).update(read_only=False),
            "read-only /iaa-runtime",
        ),
        (
            lambda payload: payload.update(runtime_sha256="not-a-digest"),
            "runtime_sha256",
        ),
        (
            lambda payload: next(
                item for item in payload["mounts"] if item["target"] == "/iaa-runtime"
            ).update(source=payload["controller_snapshot"]["root"]),
            "does not end with",
        ),
        (
            lambda payload: payload["controller_snapshot"].update(
                root=next(
                    item
                    for item in payload["mounts"]
                    if item["target"] == "/iaa-runtime"
                )["source"]
            ),
            "root does not match",
        ),
        (
            lambda payload: next(
                item
                for item in payload["spec_bundle"]["declared_inputs"]
                if item["spec_key"] == "iaa_runtime"
            ).update(uri="/different/runtime"),
            "declared input",
        ),
    ],
)
def test_brev_action_rejects_unbound_adapter_runtime(tmp_path, mutation, message):
    path, payload = _signed_adapter_request(tmp_path)
    mutation(payload)
    _write_signed(path, payload)

    with pytest.raises(ValueError, match=message):
        action.load_request(path)


def test_brev_action_rejects_adapter_freshness_evidence_drift(tmp_path):
    path, payload = _signed_adapter_request(tmp_path)
    payload["staging_absent_paths"] = payload["fresh_outputs"]
    _write_signed(path, payload)

    with pytest.raises(ValueError, match="freshness evidence"):
        action.load_request(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda environment: environment.update(HOME="/root"),
        lambda environment: environment.update(EXTRA_SETTING="unexpected"),
        lambda environment: environment.pop("PYTHONPATH"),
    ],
)
def test_brev_action_rejects_adapter_environment_drift(tmp_path, mutation):
    path, payload = _signed_adapter_request(tmp_path)
    mutation(payload["environment"])
    _write_signed(path, payload)

    with pytest.raises(ValueError, match="exact producer contract"):
        action.load_request(path)


def test_brev_action_requires_every_declared_staged_mount(tmp_path):
    path, _ = _signed_request(tmp_path)
    request = action.load_request(path)

    with pytest.raises(ValueError, match=r"missing=\['/cache'\]"):
        action.validate_mounts(request, _remote_mounts()[:-1])


def test_brev_action_rejects_gpu_count_or_request_digest_drift(tmp_path):
    path, payload = _signed_request(tmp_path)
    payload["gpu_ids"] = [6]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        action.load_request(path)

    path, payload = _signed_request(tmp_path)
    payload["spec_bundle"]["compute_shape"]["gpus"] = 1
    unsigned = dict(payload)
    unsigned.pop("request_sha256")
    payload["request_sha256"] = action._sha256_json(unsigned)  # noqa: SLF001
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="match compute_shape"):
        action.load_request(path)


def test_brev_submit_forwards_only_approved_secret_names_via_stdin(
    tmp_path, monkeypatch
):
    path, _ = _signed_request(tmp_path, gpu_ids=(3,), forward_env=("HF_TOKEN",))
    monkeypatch.setenv("HF_TOKEN", "hf_secret-never-in-command")
    monkeypatch.setattr(action, "_inspect", lambda instance, job_id: None)
    monkeypatch.setattr(action, "identity", lambda instance: (1000, 1000, "ubuntu", [1000]))
    calls = []

    def run_remote(instance, command, **kwargs):
        calls.append((instance, command, kwargs))
        return subprocess.CompletedProcess([], 0, stdout="container-id\n", stderr="")

    monkeypatch.setattr(action, "run_remote", run_remote)
    args = action._parser().parse_args(  # noqa: SLF001
        [
            "submit",
            "--request",
            str(path),
            "--instance",
            "tao-iaa-brev-smoke",
            "--job-id",
            "clip-deft-iaa-train-abc123",
            *sum((["--mount", value] for value in _remote_mounts()), []),
        ]
    )

    assert action.submit(args) == 0
    _, command, kwargs = calls[0]
    assert "hf_secret-never-in-command" not in command
    assert "-e HF_TOKEN" in command
    assert kwargs["environment"] == {"HF_TOKEN": "hf_secret-never-in-command"}
    assert kwargs["timeout"] == 600


def test_brev_submit_refuses_duplicate_owned_container_on_resume(tmp_path, monkeypatch):
    path, _ = _signed_request(tmp_path)
    monkeypatch.setattr(action, "_inspect", lambda instance, job_id: {"Id": "existing"})
    args = action._parser().parse_args(  # noqa: SLF001
        [
            "submit",
            "--request",
            str(path),
            "--instance",
            "tao-iaa-brev-smoke",
            "--job-id",
            "clip-deft-iaa-train-abc123",
            *sum((["--mount", value] for value in _remote_mounts()), []),
        ]
    )

    with pytest.raises(RuntimeError, match="reconcile instead of resubmitting"):
        action.submit(args)


@pytest.mark.parametrize(
    ("native", "exit_code", "expected"),
    [
        ("created", 0, "PENDING"),
        ("running", 0, "RUNNING"),
        ("exited", 0, "COMPLETE"),
        ("exited", 2, "ERROR"),
        ("dead", 2, "UNKNOWN"),
    ],
)
def test_brev_status_maps_native_state(native, exit_code, expected, monkeypatch, capsys):
    monkeypatch.setattr(
        action,
        "_inspect",
        lambda instance, job_id: {"State": {"Status": native, "ExitCode": exit_code}},
    )
    args = action._parser().parse_args(  # noqa: SLF001
        ["status", "--instance", "tao-iaa-brev-smoke", "--job-id", "clip-job"]
    )
    assert action.status(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == expected


def test_brev_cancel_requires_confirmation_and_owned_container(monkeypatch):
    args = action._parser().parse_args(  # noqa: SLF001
        ["cancel", "--instance", "tao-iaa-brev-smoke", "--job-id", "clip-job"]
    )
    with pytest.raises(ValueError, match="--confirm"):
        action.cancel(args)


def test_brev_logs_and_confirmed_cancel_use_owned_container(monkeypatch, capsys):
    monkeypatch.setattr(action, "_inspect", lambda instance, job_id: {"Id": "owned"})
    commands = []

    def run_remote(instance, command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess([], 0, stdout="native output\n", stderr="")

    monkeypatch.setattr(action, "run_remote", run_remote)
    logs_args = action._parser().parse_args(  # noqa: SLF001
        [
            "logs",
            "--instance",
            "tao-iaa-brev-smoke",
            "--job-id",
            "clip-job",
            "--tail",
            "40",
        ]
    )
    cancel_args = action._parser().parse_args(  # noqa: SLF001
        [
            "cancel",
            "--instance",
            "tao-iaa-brev-smoke",
            "--job-id",
            "clip-job",
            "--confirm",
        ]
    )

    assert action.logs(logs_args) == 0
    assert action.cancel(cancel_args) == 0
    assert commands == ["docker logs --tail 40 clip-job", "docker rm -f clip-job"]
    assert capsys.readouterr().out == "native output\nnative output\n"
