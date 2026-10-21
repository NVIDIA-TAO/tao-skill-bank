# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for launch-preflight GPU architecture handling.

The Windows-container cases at the bottom run on LINUX, like the rest of this
file: they drive the branch with fake daemon and driver replies rather than
with a real Windows daemon. A test that needed a Windows host would only run
where the bug cannot be introduced — the branch is written and reviewed on
POSIX machines, so that is where it has to be covered.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_tao_launch_preflight as preflight  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.3", "sm_103"),
        ("10.3a", "sm_103a"),
        ("103a", "sm_103a"),
        ("sm_103a", "sm_103a"),
        ("compute_103a", "sm_103a"),
    ],
)
def test_normalize_gpu_arch_accepts_arch_specific_forms(value, expected):
    assert preflight.normalize_gpu_arch(value) == expected


def test_cosmos_image_allowlist_includes_compute_capability_variant():
    supported = set(preflight.KNOWN_IMAGE_SMS["cosmos-rl"])
    assert {"sm_103", "sm_103a"} <= supported


def test_pas_image_families_have_a_shared_checker_architecture_gate():
    assert preflight.known_supported_sms(
        "nvcr.io/nvstaging/tao/tao-toolkit-pyt:test", []
    )
    assert preflight.known_supported_sms(
        "nvcr.io/nvstaging/tao/tao-toolkit-ds:test", []
    )


def test_docker_container_probe_is_deferred_when_preapproval_requests_it(
    monkeypatch, capsys
):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        preflight,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                '{"nvidia": {}}\n'
                if command[:2] == ["docker", "info"] and "--format" in command
                else ""
            ),
            stderr="",
        ),
    )
    monkeypatch.setattr(preflight, "query_host_gpus", lambda **_kwargs: (True, _host_gpus()[:1]))
    monkeypatch.setattr(
        preflight,
        "check_docker_gpu_smoke",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pre-approval check must not start a container")
        ),
    )
    assert preflight.check_local_docker(
        [], {}, 20, False,
        "nvcr.io/nvstaging/tao/tao-toolkit-pyt:test",
        "ubuntu:22.04", False, [], None, 50.0, False, [], True,
    )
    assert "deferred until approval" in capsys.readouterr().out


def test_architecture_specific_target_matches_family_target():
    assert preflight.gpu_arch_is_supported("sm_103", {"sm_103a"})
    assert preflight.gpu_arch_is_supported("sm_103a", {"sm_103"})


def test_different_architecture_families_do_not_match():
    assert not preflight.gpu_arch_is_supported("sm_121", {"sm_120"})


def test_gpu_resources_accept_cumulative_memory_across_devices(capsys):
    assert preflight.check_gpu_resources(None, None, 256, 4, [80], True)
    assert "required_total_memory_gb=256" in capsys.readouterr().out


def test_gpu_resources_accept_nominal_single_device_capacity(capsys):
    # Target memory is supplied in GiB, matching nvidia-smi's MiB-based report.
    assert preflight.check_gpu_resources(None, None, 256, 1, [250.7], True)
    assert "GPU resources OK" in capsys.readouterr().out


def test_gpu_resources_reject_insufficient_cumulative_memory(capsys):
    assert not preflight.check_gpu_resources(None, None, 256, 2, [100], True)
    assert "GPU resource check failed" in capsys.readouterr().out


def test_target_gpu_selection_excludes_unallocated_display_gpu(capsys):
    gpus = [
        {"index": "0", "name": "A100"},
        {"index": "1", "name": "A100"},
        {"index": "3", "name": "Display"},
        {"index": "4", "name": "A100"},
    ]
    ok, selected = preflight.filter_target_gpus(gpus, ["0", "1", "4"])
    assert ok
    assert [gpu["index"] for gpu in selected] == ["0", "1", "4"]
    assert "indices=0,1,4" in capsys.readouterr().out


def test_target_count_without_memory_uses_detected_hardware(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "detect_local_gpu_memory_gb", lambda: [80.0, 80.0])
    assert preflight.check_gpu_resources(2, 40, None, 2, [], False)
    assert "detected=0.0GiB" not in capsys.readouterr().out


def test_target_gpu_selection_rejects_missing_index(capsys):
    ok, selected = preflight.filter_target_gpus([{"index": "0"}], ["0", "2"])
    assert not ok and selected == []
    assert "missing index(es)=2" in capsys.readouterr().out


def _host_gpus():
    return [
        {
            "index": str(index),
            "uuid": f"GPU-0000000{index}",
            "name": "A100",
            "driver_version": "600.0",
            "memory_mib": 81920.0,
            "sm": "sm_80",
        }
        for index in range(4)
    ]


def test_cuda_visible_devices_filters_and_renumbers_host_inventory(capsys):
    ok, visible = preflight.filter_cuda_visible_gpus(_host_gpus(), "2,0")
    assert ok
    assert [gpu["index"] for gpu in visible] == ["2", "0"]
    assert [gpu["visible_index"] for gpu in visible] == ["0", "1"]
    assert "physical=4, visible=2" in capsys.readouterr().out


def test_cuda_visible_devices_accepts_unique_gpu_uuid_prefix():
    ok, visible = preflight.filter_cuda_visible_gpus(_host_gpus(), "GPU-00000002")
    assert ok
    assert [gpu["index"] for gpu in visible] == ["2"]


@pytest.mark.parametrize("visibility", ["", "-1"])
def test_cuda_visible_devices_rejects_empty_allocation(visibility):
    ok, visible = preflight.filter_cuda_visible_gpus(_host_gpus(), visibility)
    assert not ok
    assert visible == []


@pytest.mark.parametrize("visibility", ["0,,1", "7", "GPU-unknown", "0,0"])
def test_cuda_visible_devices_fails_closed_on_invalid_selection(visibility):
    ok, visible = preflight.filter_cuda_visible_gpus(_host_gpus(), visibility)
    assert not ok
    assert visible == []


def test_gpu_resource_count_uses_cuda_visible_inventory(monkeypatch, capsys):
    monkeypatch.setattr(
        preflight,
        "query_host_gpus",
        lambda **_kwargs: (True, _host_gpus()[:2]),
    )
    assert not preflight.check_gpu_resources(4, None, None, None, [], False)
    output = capsys.readouterr().out
    assert "qualifying_gpus=2 < required=4" in output


def test_multi_device_docker_gpu_request_preserves_csv_quotes():
    assert preflight.docker_gpu_request(["0", "1", "2", "4"]) == '"device=0,1,2,4"'
    assert preflight.docker_gpu_request([]) == "all"


def test_free_disk_requirement_passes_for_existing_parent(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _path: preflight.shutil._ntuple_diskusage(500 * 1024**3, 100, 400 * 1024**3),
    )
    output = tmp_path / "not-created-yet" / "results"
    assert preflight.check_free_disk_space(
        [("results", str(output))], {"results": 256}, skip_access=False
    )
    assert "Free-disk OK" in capsys.readouterr().out


def test_local_docker_accepts_missing_approved_creation_target(tmp_path, capsys):
    target = tmp_path / "new" / "workspace"
    assert preflight.check_local_docker(
        [("workspace", str(target))],
        {},
        20,
        True,
        None,
        "ubuntu:22.04",
        False,
        [],
        None,
        50.0,
        False,
        [],
        True,
        {"workspace"},
    )
    assert not target.exists()
    assert "Local creation target OK" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("docker_host", "is_local"),
    [
        (None, True),
        ("unix:///var/run/docker.sock", True),
        ("/var/run/docker.sock", True),
        ("npipe:////./pipe/docker_engine", True),
        ("ssh://gpu.example", False),
        ("tcp://gpu.example:2376", False),
    ],
)
def test_creation_target_cli_uses_docker_endpoint_scope(
    tmp_path, docker_host, is_local
):
    target = tmp_path / "new" / "workspace"
    environment = os.environ.copy()
    if docker_host is None:
        environment.pop("DOCKER_HOST", None)
    else:
        environment["DOCKER_HOST"] = docker_host

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(preflight.__file__)),
            "--skill-bank",
            str(Path(__file__).resolve().parents[2]),
            "--platform",
            "docker",
            "--path",
            f"workspace={target}",
            "--allow-missing-path",
            "workspace",
            "--skip-platform-access",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    if is_local:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "Local creation target OK" in completed.stdout
    else:
        assert completed.returncode != 0
        assert "supported only for local Docker paths" in completed.stderr
    assert not target.exists()


def test_missing_creation_target_rejects_symlinked_parent(tmp_path, capsys):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    assert not preflight.check_local_creation_target(
        "workspace", str(linked_parent / "workspace")
    )
    assert "symlink component" in capsys.readouterr().out


def test_free_disk_requirement_rejects_insufficient_capacity(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _path: preflight.shutil._ntuple_diskusage(500 * 1024**3, 450, 50 * 1024**3),
    )
    assert not preflight.check_free_disk_space(
        [("results", str(tmp_path))], {"results": 256}, skip_access=False
    )
    assert "free=50.0GiB < required=256GiB" in capsys.readouterr().out


def test_slurm_preflight_checks_remote_scheduler_pyxis_and_enroot(monkeypatch, tmp_path):
    key = tmp_path / "id_ed25519"
    key.write_text("fixture")
    for name, value in {
        "SLURM_USER": "user",
        "SLURM_HOSTNAME": "login.example",
        "SLURM_PARTITION": "compute",
        "SSH_KEY_PATH": str(key),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(preflight.socket, "getaddrinfo", lambda *_args: [(None,)])
    monkeypatch.setattr(preflight, "check_slurm_runtime", lambda _platform: True)
    commands = []

    def fake_run(command, timeout=30, env=None):
        commands.append(command)
        remote = command[-1]
        if remote == "echo TAO_SSH_OK":
            return subprocess.CompletedProcess(command, 0, stdout="TAO_SSH_OK\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(preflight, "run", fake_run)
    platform = {
        "required_credentials": [
            {"name": "SLURM_USER", "source": "env_var"},
            {"name": "SLURM_HOSTNAME", "source": "env_var"},
            {"name": "SLURM_PARTITION", "source": "env_var"},
        ],
        "credential_groups": [{"require_one_of": ["SSH_KEY_PATH", "SSH_AUTH_SOCK"]}],
    }
    assert preflight.check_slurm(platform, [("data", "/shared/data")], {}, 20, False)
    remote_commands = [command[-1] for command in commands]
    assert any("command -v sbatch" in command for command in remote_commands)
    assert any("command -v enroot" in command for command in remote_commands)
    assert any("--container-image" in command for command in remote_commands)


def test_slurm_preflight_rejects_missing_remote_pyxis(monkeypatch, tmp_path, capsys):
    key = tmp_path / "id_ed25519"
    key.write_text("fixture")
    for name, value in {
        "SLURM_USER": "user",
        "SLURM_HOSTNAME": "login.example",
        "SLURM_PARTITION": "compute",
        "SSH_KEY_PATH": str(key),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(preflight.socket, "getaddrinfo", lambda *_args: [(None,)])
    monkeypatch.setattr(preflight, "check_slurm_runtime", lambda _platform: True)

    def fake_run(command, timeout=30, env=None):
        remote = command[-1]
        if remote == "echo TAO_SSH_OK":
            return subprocess.CompletedProcess(command, 0, stdout="TAO_SSH_OK\n", stderr="")
        if "command -v sbatch" in remote:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="enroot missing")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(preflight, "run", fake_run)
    platform = {
        "required_credentials": [
            {"name": "SLURM_USER", "source": "env_var"},
            {"name": "SLURM_HOSTNAME", "source": "env_var"},
            {"name": "SLURM_PARTITION", "source": "env_var"},
        ],
        "credential_groups": [{"require_one_of": ["SSH_KEY_PATH", "SSH_AUTH_SOCK"]}],
    }
    assert not preflight.check_slurm(platform, [], {}, 20, False)
    assert "Remote SLURM/Pyxis/Enroot preflight failed" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Windows containers.
#
# All of these run on Linux. The daemon reply, the driver query and the GPU
# smoke are faked, so the branch is exercised everywhere the suite runs — a
# Windows-host-only test would be a test that almost never runs.
# --------------------------------------------------------------------------- #

def _fake_docker(server_os="windows", smoke_rc=0, calls=None, image_present=True):
    """A stand-in for preflight.run covering every command the branch issues."""
    def fake_run(command, timeout=30, env=None):
        if calls is not None:
            calls.append(list(command))
        joined = " ".join(command)
        if command[:3] == ["docker", "image", "inspect"]:
            # The presence guard. `docker run` PULLS an absent image, so the
            # smoke step has to ask first, exactly as the Linux twin does.
            return subprocess.CompletedProcess(
                command, 0 if image_present else 1, stdout="", stderr="No such image")
        if command[:2] == ["docker", "version"]:
            return subprocess.CompletedProcess(command, 0, stdout=f"{server_os}\n", stderr="")
        if command[:2] == ["docker", "info"]:
            # Linux-only probe. Reaching it on a Windows daemon is the defect
            # under test, so answer as a daemon with NO nvidia runtime: if the
            # branch consults it, the check must visibly fail.
            return subprocess.CompletedProcess(
                command, 0, stdout='{"runc":{}}', stderr="")
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                command, 0,
                stdout="0, NVIDIA RTX PRO 6000 Blackwell, 581.15, 98304, 12.0\n",
                stderr="")
        if command[:2] == ["docker", "run"]:
            if smoke_rc == 0:
                return subprocess.CompletedProcess(
                    command, 0, stdout="GPU 0: NVIDIA RTX PRO 6000 Blackwell (UUID: GPU-x)\n",
                    stderr="")
            return subprocess.CompletedProcess(
                command, smoke_rc, stdout="",
                stderr="hcsshim::CreateComputeSystem: device class not found")
        raise AssertionError(f"unexpected command in the Windows branch: {joined}")
    return fake_run


@pytest.fixture
def _windows_daemon(monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(preflight.shutil, "which", lambda tool: f"/usr/bin/{tool}")


def test_windows_daemon_uses_the_device_class_and_never_demands_the_nvidia_runtime(
    _windows_daemon, monkeypatch, capsys
):
    """A Windows daemon must not be measured with the Linux GPU contract.

    There is no NVIDIA Container Toolkit for Windows containers and therefore
    no `nvidia` entry in `docker info` runtimes — the GPU arrives through a
    device class. Routing Windows through the Linux branch fails preflight with
    "install/configure NVIDIA Container Toolkit", which is unactionable advice
    on a host where the toolkit does not exist.
    """
    calls = []
    monkeypatch.setattr(preflight, "run", _fake_docker(server_os="windows", calls=calls))

    assert preflight.check_windows_docker(
        [], {}, 20, False, "registry.example/tao/cosmos-framework-windows:tag",  # unpinned: test fixture, not a real image
        False, [], None, 50.0, [],
    )
    out = capsys.readouterr().out
    assert "Docker daemon OK: Windows-container mode" in out
    assert "NVIDIA runtime" not in out, (
        "the Windows branch consulted the Linux runtime check; there is no "
        "nvidia runtime to find on a Windows daemon")
    assert not any("Runtimes" in " ".join(call) for call in calls), (
        "docker info --format {{json .Runtimes}} was queried on a Windows daemon")

    smoke = next(call for call in calls if call[:2] == ["docker", "run"])
    assert "--isolation=process" in smoke, (
        "the daemon default is hyperv isolation, which has NO GPU path")
    assert preflight.WINDOWS_GPU_DEVICE_CLASS in smoke
    assert "--gpus" not in smoke, "--gpus does not exist for Windows containers"


def test_linux_mode_daemon_is_rejected_before_the_image_is_blamed(
    _windows_daemon, monkeypatch, capsys
):
    """`Server.Os=linux` must be named as the cause.

    A daemon left in Linux-container mode answers every later step with "no
    matching manifest for linux/amd64", which reads as a bad image tag and
    sends the user to fix the wrong thing.
    """
    monkeypatch.setattr(preflight, "run", _fake_docker(server_os="linux"))
    assert not preflight.check_windows_docker(
        [], {}, 20, False, "nvcr.io/x/y:tag", False, [], None, 50.0, [],  # unpinned: test fixture, not a real image
    )
    out = capsys.readouterr().out
    assert "not in Windows-container mode" in out and "SwitchWindowsEngine" in out


@pytest.mark.parametrize(
    ("docker_host", "is_remote"),
    [
        ("npipe:////./pipe/docker_engine", False),  # the Windows local daemon
        ("unix:///var/run/docker.sock", False),     # its Linux twin
        ("/var/run/docker.sock", False),
        ("", False),
        ("ssh://user@gpu-host", True),
        ("tcp://10.0.0.5:2375", True),
    ],
)
def test_local_daemon_endpoints_are_not_classified_as_remote(docker_host, is_remote):
    assert preflight.docker_host_is_remote(docker_host) is is_remote


@pytest.mark.parametrize(
    "docker_host",
    ["npipe:////./pipe/docker_engine", "unix:///var/run/docker.sock"],
)
def test_setting_a_local_docker_host_does_not_demand_a_remote_daemon(
    monkeypatch, capsys, docker_host
):
    """DOCKER_HOST set to the LOCAL endpoint is the normal case, not a request.

    The requirement used to key off `bool(DOCKER_HOST)`, so exporting the
    idiomatic local endpoint — `npipe:////./pipe/docker_engine` on Windows,
    `unix:///var/run/docker.sock` on Linux — made preflight demand "a remote
    Docker daemon URL such as ssh://user@gpu-host" and refuse to run against
    the daemon that was right there. The unix:// case is the regression guard:
    this was never a Windows-only bug.
    """
    monkeypatch.setenv("DOCKER_HOST", docker_host)
    monkeypatch.setattr(
        sys, "argv",
        ["check_tao_launch_preflight.py", "--skill-bank", str(REPO),
         "--platform", "docker", "--skip-platform-access"],
    )
    assert preflight.main() == 0
    assert "Missing remote Docker requirement" not in capsys.readouterr().out


def test_docker_aliases_normalize_before_the_dispatch_reads_the_name():
    """`remote-docker` reaches main() already normalized to `docker`.

    Recorded because the remote-daemon requirement in main() is spelled
    ``name == "remote-docker" or docker_host_is_remote(...)`` and this makes the
    first operand unreachable — after the alias map runs, `name` is "docker".
    The requirement is therefore decided entirely by the endpoint, and asking
    for `--platform remote-docker` does not by itself demand a remote daemon.
    If that clause is ever meant to bite, it has to test the RAW `--platform`
    value; this test is what will fail when someone changes the alias map to
    make it reachable.

    The two Windows slugs are here as the contrast: they are NOT aliased, so
    the dispatch sees them verbatim and routes each to its own branch.
    """
    resolved = {
        requested: preflight.resolve_platform(REPO, requested)["name"]
        for requested in ("docker", "local-docker", "remote-docker",
                          "windows-docker", "windows-venv")
    }
    assert resolved == {
        "docker": "docker",
        "local-docker": "docker",
        "remote-docker": "docker",
        "windows-docker": "windows-docker",
        "windows-venv": "windows-venv",
    }


def test_windows_branch_accepts_a_drive_letter_host_path(tmp_path, monkeypatch, capsys):
    """`C:\\data\\train` must pass the Windows path check.

    The Linux bind-path check requires a leading "/" and probes the path from
    inside a container with `test -e`. Both are wrong here: a Windows host path
    starts with a drive letter, and there is no `test` in a Windows image. The
    Windows branch checks the host filesystem directly, because preflight runs
    on the same host the bind mount comes from.
    """
    if os.name == "nt":
        target = tmp_path / "data" / "train"
        target.mkdir(parents=True)
        windows_path = str(target)              # already C:\...\data\train
    else:
        # A backslash is an ordinary filename character on Linux, so the literal
        # Windows string can be made to exist and drive the same code path.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "C:\\data\\train").mkdir()
        windows_path = "C:\\data\\train"
    assert windows_path[1:2] == ":" and "\\" in windows_path

    def no_docker(*_args, **_kwargs):
        raise AssertionError("the host path check must not shell out to docker")

    monkeypatch.setattr(preflight, "run", no_docker)
    assert preflight.check_windows_docker(
        [("data", windows_path)], {}, 20, True, None, False, [], None, 50.0, [],
    )
    assert f"Host path OK: data={windows_path}" in capsys.readouterr().out


def test_linux_bind_path_check_still_rejects_a_windows_path(capsys):
    """The POSIX twin must keep rejecting it — that is why the branch exists."""
    assert not preflight.check_docker_bind_path(
        "data", "C:\\data\\train", "ubuntu:22.04", False)
    assert "not absolute" in capsys.readouterr().out


def test_windows_container_smoke_refuses_to_pull_an_image_nobody_approved(
    _windows_daemon, monkeypatch, capsys
):
    """`docker run` on an absent image PULLS it. The smoke step must ask first.

    check_docker_gpu_smoke() has enforced this since it was written; the
    Windows twin shipped without it, so on a host that had not already pulled
    the multi-GB Windows image, preflight silently started a download nobody
    approved, blew the 180s timeout, and then blamed the device class, the
    isolation mode and the Windows Server build — three things that were never
    consulted, because the container never started.
    """
    calls = []
    monkeypatch.setattr(
        preflight, "run",
        _fake_docker(calls=calls, image_present=False))

    assert not preflight.check_windows_container_gpu_smoke("nvcr.io/x/y:tag", False)  # unpinned: test fixture, not a real image

    assert not any(call[:2] == ["docker", "run"] for call in calls), (
        "`docker run` was issued for an image that is not on the host; that "
        "starts an unapproved multi-GB pull")
    assert ["docker", "image", "inspect", "nvcr.io/x/y:tag"] in calls  # unpinned: test fixture, not a real image
    out = capsys.readouterr().out
    assert "is not present on the Docker host" in out
    assert "--pull-smoke-image" in out
    # The diagnosis must not be the smoke-failure one: nothing was executed, so
    # the device class / isolation / host-build advice would misdirect.
    assert "Windows container GPU smoke failed" not in out
    assert "Check the device class" not in out
    assert "matches the host" not in out


def test_pull_smoke_image_is_the_approval_that_lets_the_windows_smoke_pull(
    _windows_daemon, monkeypatch, capsys
):
    """--pull-smoke-image means "you may pull", exactly as it does on Linux.

    With approval given, the presence probe is skipped entirely (the Linux twin
    short-circuits the same way) and `docker run` is allowed to fetch the image.
    """
    calls = []
    monkeypatch.setattr(
        preflight, "run",
        _fake_docker(calls=calls, image_present=False))

    assert preflight.check_windows_container_gpu_smoke("nvcr.io/x/y:tag", True)  # unpinned: test fixture, not a real image

    assert not any(call[:3] == ["docker", "image", "inspect"] for call in calls), (
        "approval was given, so the presence probe is unnecessary")
    smoke = next(call for call in calls if call[:2] == ["docker", "run"])
    assert "--isolation=process" in smoke
    assert "Windows container GPU smoke OK" in capsys.readouterr().out


def test_present_image_that_fails_the_smoke_still_gets_the_smoke_diagnosis(
    _windows_daemon, monkeypatch, capsys
):
    """The two failures must stay distinguishable.

    A pull refusal and a genuine GPU/isolation failure send the user to
    completely different fixes, so the presence guard must not swallow the
    real diagnostic.
    """
    monkeypatch.setattr(
        preflight, "run",
        _fake_docker(smoke_rc=125, image_present=True))

    assert not preflight.check_windows_container_gpu_smoke("nvcr.io/x/y:tag", False)  # unpinned: test fixture, not a real image
    out = capsys.readouterr().out
    assert "Windows container GPU smoke failed" in out
    assert "isolation=process" in out
    assert "is not present on the Docker host" not in out


# --------------------------------------------------------------------------- #
# windows-venv: a LOCAL platform, so its paths are checkable and must be checked
# --------------------------------------------------------------------------- #

def _venv_argv(path_arg=None, extra=()):
    argv = ["check_tao_launch_preflight.py", "--skill-bank", str(REPO),
            "--platform", "windows-venv"]
    if path_arg:
        argv += ["--path", path_arg]
    return argv + list(extra)


def test_windows_venv_rejects_a_path_that_does_not_exist(tmp_path, monkeypatch, capsys):
    """The regression guard for the whole defect.

    windows-venv runs the job in a venv on the machine preflight runs on, so a
    non-remote path is a path on THIS filesystem. Routing it to the env-only
    check AND exempting it from the mounted-volume-proof gate left it with no
    existence check at all: `--path train=C:\\does\\not\\exist` printed "TAO
    launch preflight passed" and exited 0, where origin/main exited 2.
    """
    missing = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(sys, "argv", _venv_argv(f"train={missing}"))

    assert preflight.main() == 2

    out = capsys.readouterr().out
    assert f"Local path missing: train={missing}" in out
    assert "TAO launch preflight failed" in out
    assert "verify it is mounted in the job" not in out, (
        "windows-venv is not a remote platform; there is no mount to verify")


def test_windows_venv_still_rejects_a_missing_path_under_skip_platform_access(
    tmp_path, monkeypatch, capsys
):
    """--skip-platform-access waives DAEMON/credential probes, not the filesystem.

    windows-docker checks its bind paths regardless of the flag for the same
    reason: the path is on this host either way. Without this case the exact
    same fail->pass hole reopens behind one extra flag.
    """
    missing = tmp_path / "nope"
    monkeypatch.setattr(
        sys, "argv", _venv_argv(f"train={missing}", ["--skip-platform-access"]))
    assert preflight.main() == 2
    assert "Local path missing" in capsys.readouterr().out


def test_windows_venv_accepts_a_path_that_exists(tmp_path, monkeypatch, capsys):
    """The other half of the contract: a real local path must still pass."""
    present = tmp_path / "data"
    present.mkdir()
    monkeypatch.setattr(sys, "argv", _venv_argv(f"train={present}"))

    assert preflight.main() == 0
    out = capsys.readouterr().out
    assert f"Local path OK: train={present}" in out
    assert "TAO launch preflight passed" in out


def test_windows_venv_is_not_routed_to_the_env_only_check(tmp_path, monkeypatch):
    """check_env_only() cannot see these paths, so it must not be the route.

    It exists for platforms whose paths live on another machine and can only
    print "verify it is mounted in the job". Sending a local platform there is
    what produced the fail->pass regression, so the routing itself is asserted
    rather than only its output.
    """
    def unreachable(*_args, **_kwargs):
        raise AssertionError(
            "windows-venv reached check_env_only(); that branch cannot verify "
            "a local path and passes for one that does not exist")

    monkeypatch.setattr(preflight, "check_env_only", unreachable)
    present = tmp_path / "data"
    present.mkdir()
    monkeypatch.setattr(sys, "argv", _venv_argv(f"train={present}"))
    assert preflight.main() == 0


def test_windows_venv_mount_proof_exemption_is_backed_by_a_real_check(tmp_path):
    """The exemption is only legitimate because check_windows_venv() verifies.

    has_unverified_remote_mounts() exists for paths on ANOTHER machine, which
    preflight cannot inspect. Exempting a slug is a promise that its own branch
    checked the paths for real; this pins both halves together so the exemption
    can never again be the thing that removes the last check.
    """
    missing = str(tmp_path / "absent")
    assert not preflight.has_unverified_remote_mounts(
        "windows-venv", [("train", missing)], False)
    assert not preflight.check_windows_venv({}, [("train", missing)], {}, 20)


def test_windows_venv_leaves_remote_uris_to_the_storage_check(capsys):
    """s3:// and friends are not local paths and must not be existence-checked."""
    assert preflight.check_windows_venv(
        {}, [("train", "s3://bucket/key")], {}, 20)
    assert "Remote path deferred to the storage check" in capsys.readouterr().out


def test_windows_venv_credential_gate_survives_the_new_branch(capsys):
    """The env/credential gate check_env_only() applied has to still apply."""
    platform = {"required_credentials": [
        {"name": "TAO_DEFINITELY_UNSET_FOR_THIS_TEST", "source": "env_var"}]}
    assert not preflight.check_windows_venv(platform, [], {}, 20)
    assert "Missing requirement(s)" in capsys.readouterr().out


def test_windows_docker_bind_paths_are_still_verified(tmp_path, monkeypatch, capsys):
    """The sibling platform's existence check must not have been lost too."""
    monkeypatch.setattr(
        preflight, "run",
        lambda *_a, **_k: pytest.fail("the host path check must not shell out"))
    assert not preflight.check_windows_docker(
        [("data", str(tmp_path / "absent"))], {}, 20, True, None, False, [],
        None, 50.0, [],
    )
    assert "Host path missing" in capsys.readouterr().out
