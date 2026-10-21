# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle tests for the vendored Windows job runner.

These run REAL detached processes through the four verbs. The cases mirror the
POSIX suite (durable launcher record, start gate, PID-reuse identity, tree
cleanup, cancel ordering, bounded log tail) plus the four that only exist here:

  * status polling is a PURE QUERY — the property the POSIX runner's
    ``os.kill(pid, 0)`` liveness probe silently violates on Windows, where
    ``os.kill`` is ``TerminateProcess``. Polling many times must not change the
    outcome of a job that is still working.
  * the whole process TREE dies on cancel, including a grandchild the target
    spawned, because Job Object membership is inherited at creation.
  * an unsigned-DWORD crash code survives into the record and widens the ERROR
    message instead of being read as a signal.
  * the runner refuses to run at all on a POSIX host, through the JSON contract.

Everything except the last is skipped off Windows.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import windows_job_runner as wr  # noqa: E402

windows_only = pytest.mark.skipif(
    os.name != "nt", reason="native Windows process primitives (Job Objects, GetProcessTimes)"
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def venv(tmp_path_factory):
    """A real venv, built by `python -m venv`.

    Hand-assembling one by copying an interpreter does not work on Windows: the
    Scripts entry is a venv redirector that resolves its base install through
    pyvenv.cfg, and a bare copy of a base python.exe cannot find python3xx.dll
    from a temp directory. --without-pip keeps it to about a second; nothing
    here needs site-packages.
    """
    root = tmp_path_factory.mktemp("env") / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root)],
                   check=True, capture_output=True)
    assert (root / "Scripts" / "python.exe").is_file()
    return root


@pytest.fixture()
def job_dir(tmp_path):
    d = tmp_path / "results" / "job-0001"
    d.mkdir(parents=True)
    return d


def write_script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def run_verb(capsys, *argv):
    rc = wr.main(list(argv))
    out = capsys.readouterr().out.strip()
    payload = json.loads(out.splitlines()[-1]) if out else {}
    return rc, payload


def wait_for(predicate, timeout=60.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _evidence(job_dir):
    """Everything durable about a job dir — embedded in failure messages."""
    paths = wr.RunnerPaths(Path(job_dir))
    return {
        "exit_status": wr._read_json(paths.exit_status),
        "launcher_status": wr._read_json(paths.launcher_status),
        "cancel_marker": paths.cancel_marker.exists(),
        "log": wr._read_tail(paths.log_path, 20) if paths.log_path.exists() else None,
    }


def wait_terminal(capsys, job_dir, timeout=60.0):
    result = {}

    def is_terminal():
        _, payload = run_verb(capsys, "status", "--job-dir", str(job_dir))
        result.update(payload)
        return payload.get("status") in ("COMPLETE", "ERROR", "CANCELED")

    assert wait_for(is_terminal, timeout=timeout), \
        f"never terminal: {result} evidence={_evidence(job_dir)}"
    result["evidence"] = _evidence(job_dir)
    return result


def _alive(pid):
    return wr._process_liveness(pid) == "alive"


# ---------------------------------------------------------------------------
# Refusal on the wrong operating system — the only test that runs everywhere
# ---------------------------------------------------------------------------

def test_posix_host_is_refused_through_the_json_contract(capsys, monkeypatch, job_dir):
    """A POSIX host must get a JSON refusal, never a traceback.

    The Job Object / GetProcessTimes primitives have no meaning there, and the
    mirror-image mistake (running the POSIX runner on Windows) is the one that
    kills jobs with a status poll. Both directions must refuse in-band.
    """
    monkeypatch.setattr(os, "name", "posix")
    for verb in ("status", "cancel"):
        rc, payload = run_verb(capsys, verb, "--job-dir", str(job_dir))
        assert rc == 1
        assert payload["result"] == "error"
        assert "native Windows" in payload["message"]
        assert "tao-run-on-virtualenv" in payload["message"]
    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir),
                           "--venv", str(job_dir), "--module", "json.tool")
    assert rc == 1 and payload["result"] == "error"


# ---------------------------------------------------------------------------
# Happy path + artifacts
# ---------------------------------------------------------------------------

@windows_only
def test_submit_runs_to_complete_and_writes_results(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "train.py", (
        "import os, pathlib, sys\n"
        "out = pathlib.Path(os.environ['TAO_RESULTS_ROOT']) / 'metrics.json'\n"
        "out.write_text('{\"metric\": 0.9}')\n"
        "print('trained', sys.argv[1:])\n"
    ))
    rc, sub = run_verb(
        capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
        "--script", str(script), "--job-id", "job-0001",
        "--arg", "train", "--arg=--results={results_dir}",
    )
    assert rc == 0 and sub["result"] == "submitted" and sub["status"] == "RUNNING"
    assert sub["backend_ref"].startswith("pid:")
    assert sub["job_object"].startswith("Local\\tao-win-")
    assert sub["process_start_marker"]  # GetProcessTimes creation FILETIME

    final = wait_terminal(capsys, job_dir)
    assert final["status"] == "COMPLETE" and final["return_code"] == 0, final
    assert json.loads((job_dir / "metrics.json").read_text())["metric"] == 0.9

    rc = wr.main(["logs", "--job-dir", str(job_dir), "--tail", "50"])
    assert rc == 0
    assert "trained" in capsys.readouterr().out


@windows_only
def test_placeholders_render_and_env_lands_in_the_target(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "dump.py", (
        "import json, os, sys, pathlib\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({\n"
        "  'argv': sys.argv[1:],\n"
        "  'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "  'venv': os.environ.get('VIRTUAL_ENV'),\n"
        "  'job_id': os.environ.get('TAO_JOB_ID'),\n"
        "  'passthrough': os.environ.get('TAO_TEST_TOKEN'),\n"
        "}))\n"
    ))
    os.environ["TAO_TEST_TOKEN"] = "present"
    try:
        rc, _ = run_verb(
            capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
            "--script", str(script), "--job-id", "jid-42",
            "--config-path", "C:\\spec.yaml",
            "--arg", "{results_dir}\\dump.json", "--arg=-e", "--arg", "{config_path}",
            "--gpu-ids", "0", "-e", "TAO_TEST_TOKEN",
        )
        assert rc == 0
        final = wait_terminal(capsys, job_dir)
        assert final["status"] == "COMPLETE", final
    finally:
        os.environ.pop("TAO_TEST_TOKEN", None)
    dumped = json.loads((job_dir / "dump.json").read_text())
    assert dumped["argv"] == [str(job_dir / "dump.json"), "-e", "C:\\spec.yaml"]
    assert dumped["cuda"] == "0"
    assert dumped["venv"] == str(venv.resolve())
    assert dumped["job_id"] == "jid-42"
    assert dumped["passthrough"] == "present"


@windows_only
def test_tao_windows_env_sets_both_api_variables(capsys, venv, job_dir, tmp_path):
    """TAO_API_JOB_ID alone is fatal for training; the actions need it for eval.

    One env block has to satisfy both, so --tao-windows-env must always set
    TAO_API_RESULTS_DIR as well.
    """
    script = write_script(tmp_path, "env.py", (
        "import json, os, pathlib\n"
        "pathlib.Path(os.environ['TAO_RESULTS_ROOT'], 'env.json').write_text(json.dumps({\n"
        "  k: os.environ.get(k) for k in\n"
        "  ('TAO_API_JOB_ID','TAO_API_RESULTS_DIR','USE_LIBUV','MASTER_ADDR',\n"
        "   'WORLD_SIZE','IMAGINAIRE_OUTPUT_ROOT','PYTHONIOENCODING')}))\n"
    ))
    rc, _ = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                     "--script", str(script), "--job-id", "jid-9", "--tao-windows-env")
    assert rc == 0
    assert wait_terminal(capsys, job_dir)["status"] == "COMPLETE"
    seen = json.loads((job_dir / "env.json").read_text())
    assert seen["TAO_API_JOB_ID"] == "jid-9"
    assert seen["TAO_API_RESULTS_DIR"] == str(job_dir)
    assert seen["USE_LIBUV"] == "0"
    assert seen["MASTER_ADDR"] == "127.0.0.1"
    assert seen["WORLD_SIZE"] == "1"
    assert seen["PYTHONIOENCODING"] == "utf-8"
    assert seen["IMAGINAIRE_OUTPUT_ROOT"] == str(job_dir)


@windows_only
def test_module_target_runs(capsys, venv, job_dir, tmp_path):
    payload = tmp_path / "in.json"
    payload.write_text('{"a": 1}', encoding="utf-8")
    rc, sub = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                       "--module", "json.tool", "--arg", str(payload))
    assert rc == 0 and sub["target"] == "module:json.tool"
    assert wait_terminal(capsys, job_dir)["status"] == "COMPLETE"


# ---------------------------------------------------------------------------
# The property the POSIX runner violates on Windows
# ---------------------------------------------------------------------------

@windows_only
def test_status_polling_does_not_kill_the_job(capsys, venv, job_dir, tmp_path):
    """Poll hard, then let the job finish on its own.

    os.kill(pid, 0) on Windows is OpenProcess + TerminateProcess, so a runner
    written that way turns this test into a COMPLETE with return_code 0 and no
    artifact. The artifact is the assertion that catches it.
    """
    script = write_script(tmp_path, "slow.py", (
        "import os, pathlib, time\n"
        "time.sleep(6)\n"
        "pathlib.Path(os.environ['TAO_RESULTS_ROOT'], 'done.txt').write_text('finished')\n"
    ))
    rc, sub = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                       "--script", str(script))
    assert rc == 0
    pid = sub["pid"]
    for _ in range(25):
        _, status = run_verb(capsys, "status", "--job-dir", str(job_dir))
        assert status["status"] == "RUNNING", status
        time.sleep(0.05)
    final = wait_terminal(capsys, job_dir)
    assert final["status"] == "COMPLETE" and final["return_code"] == 0, final
    assert (job_dir / "done.txt").read_text() == "finished", (
        "the job was killed by a status poll — the liveness probe signalled it"
    )
    assert not _alive(pid)


@windows_only
def test_liveness_probe_never_signals(capsys, venv, job_dir, tmp_path):
    """Direct unit check on the primitive, independent of the verbs."""
    proc = subprocess.Popen(
        [str(venv / "Scripts" / "python.exe"), "-c", "import time; time.sleep(20)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(20):
            assert wr._process_liveness(proc.pid) == "alive"
        assert proc.poll() is None, "the liveness probe terminated the process"
        marker = wr._process_creation_marker(proc.pid)
        assert marker and marker.isdigit()
        assert wr._process_matches(proc.pid, marker)
        assert not wr._process_matches(proc.pid, "1")  # wrong creation time
    finally:
        proc.kill()
        proc.wait(timeout=10)
    assert wait_for(lambda: wr._process_liveness(proc.pid) != "alive", timeout=10)


# ---------------------------------------------------------------------------
# Cancel + tree teardown
# ---------------------------------------------------------------------------

@windows_only
def test_cancel_kills_the_whole_tree_including_a_grandchild(capsys, venv, job_dir, tmp_path):
    """Job Object membership is inherited at creation, so nothing escapes.

    This is the case a "kill the pid we launched" cancel gets wrong: the
    grandchild is not a direct child of anything the runner recorded.
    """
    script = write_script(tmp_path, "spawner.py", (
        "import os, pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        "pathlib.Path(os.environ['TAO_RESULTS_ROOT'], 'child.pid').write_text(str(child.pid))\n"
        "time.sleep(300)\n"
    ))
    rc, sub = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                       "--script", str(script))
    assert rc == 0
    assert wait_for(lambda: (job_dir / "child.pid").is_file(), timeout=45)
    grandchild = int((job_dir / "child.pid").read_text())
    assert _alive(grandchild)

    rc, payload = run_verb(capsys, "cancel", "--job-dir", str(job_dir))
    assert rc == 0 and payload["result"] == "canceled"
    assert payload["status"] == "CANCELED", payload
    assert wait_for(lambda: not _alive(grandchild), timeout=20), \
        "the grandchild outlived TerminateJobObject"
    assert wait_for(lambda: not _alive(sub["pid"]), timeout=20)

    _, again = run_verb(capsys, "cancel", "--job-dir", str(job_dir))
    assert again["result"] == "already_terminal" and again["status"] == "CANCELED"


@windows_only
def test_cancel_after_natural_exit_reports_already_terminal(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "quick.py", "print('ok')\n")
    rc, _ = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                     "--script", str(script))
    assert rc == 0
    assert wait_terminal(capsys, job_dir)["status"] == "COMPLETE"
    _, payload = run_verb(capsys, "cancel", "--job-dir", str(job_dir))
    assert payload["result"] == "already_terminal"
    assert payload["status"] == "COMPLETE", payload
    # The cancel claim must be withdrawn: this job exited on its own.
    assert not wr.RunnerPaths(job_dir).cancel_marker.exists()


@windows_only
def test_cancel_before_the_gate_self_cancels(capsys, venv, job_dir, tmp_path):
    """The wrapper honours the marker while still blocked on the start gate."""
    paths = wr.RunnerPaths(job_dir)
    paths.runner_dir.mkdir(parents=True)
    wrapper = paths.wrapper
    wrapper.write_text(wr.JOB_WRAPPER_SOURCE, encoding="utf-8")
    paths.cancel_marker.touch()
    proc = subprocess.Popen(
        [str(venv / "Scripts" / "python.exe"), str(wrapper), str(paths.exit_status),
         str(paths.launcher_status), str(paths.start_gate), str(paths.cancel_marker),
         "Local\\tao-win-test-selfcancel",
         str(venv / "Scripts" / "python.exe"), "-c", "print('never runs')"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert proc.wait(timeout=60) == 130
    record = json.loads(paths.exit_status.read_text())
    assert record["error"] == wr.CANCELED_BEFORE_START
    status, _, _ = wr._derive_status(paths)
    assert status == "CANCELED"


@windows_only
def test_wrapper_gives_up_when_the_gate_is_never_opened(capsys, venv, job_dir):
    """A submit that dies before authorizing start must not leave a phantom."""
    paths = wr.RunnerPaths(job_dir)
    paths.runner_dir.mkdir(parents=True)
    paths.wrapper.write_text(wr.JOB_WRAPPER_SOURCE, encoding="utf-8")
    env = {**os.environ, "TAO_RUNNER_GATE_TIMEOUT": "1"}
    proc = subprocess.Popen(
        [str(venv / "Scripts" / "python.exe"), str(paths.wrapper), str(paths.exit_status),
         str(paths.launcher_status), str(paths.start_gate), str(paths.cancel_marker),
         "Local\\tao-win-test-nogate",
         str(venv / "Scripts" / "python.exe"), "-c", "print('never runs')"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert proc.wait(timeout=60) == 125
    record = json.loads(paths.exit_status.read_text())
    assert record["return_code"] == 125
    assert "never authorized" in record["error"]


# ---------------------------------------------------------------------------
# Unsigned DWORD exit codes
# ---------------------------------------------------------------------------

@windows_only
def test_access_violation_survives_as_an_unsigned_dword(capsys, venv, job_dir, tmp_path):
    """0xC0000005 must reach the record intact and widen the ERROR message.

    On POSIX this would be a negative return_code naming SIGSEGV. Here the
    DWORD is all there is, so losing it loses the diagnosis entirely.
    """
    script = write_script(tmp_path, "crash.py", (
        "import ctypes\n"
        "from ctypes import wintypes\n"
        "k = ctypes.WinDLL('kernel32')\n"
        "k.ExitProcess.argtypes = (wintypes.UINT,)\n"
        "k.ExitProcess(0xC0000005)\n"
    ))
    rc, _ = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                     "--script", str(script))
    assert rc == 0
    final = wait_terminal(capsys, job_dir)
    assert final["status"] == "ERROR", final
    assert final["return_code"] == 0xC0000005
    assert "0xC0000005" in final["message"]
    assert "STATUS_ACCESS_VIOLATION" in final["message"]
    # And the branch that can never fire on Windows stays unreachable.
    assert final["return_code"] > 0


@windows_only
def test_plain_nonzero_exit_is_an_error(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "fail.py", "import sys; sys.exit(3)\n")
    rc, _ = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                     "--script", str(script))
    assert rc == 0
    final = wait_terminal(capsys, job_dir)
    assert final["status"] == "ERROR" and final["return_code"] == 3
    assert final["message"] == "Process exited with code 3"


def test_describe_code_widens_only_the_message():
    assert wr._describe_code(3) == "code 3"
    assert "STATUS_ACCESS_VIOLATION" in wr._describe_code(0xC0000005)
    assert "STATUS_DLL_NOT_FOUND" in wr._describe_code(0xC0000135)
    assert "NTSTATUS" in wr._describe_code(0xC0000123)
    # The vocabulary itself is fixed — no new state is ever invented.
    assert {wr.VOCAB_PENDING, wr.VOCAB_RUNNING, wr.VOCAB_COMPLETE,
            wr.VOCAB_ERROR, wr.VOCAB_CANCELED, wr.VOCAB_UNKNOWN} == {
        "PENDING", "RUNNING", "COMPLETE", "ERROR", "CANCELED", "UNKNOWN"}


# ---------------------------------------------------------------------------
# Submission lock, validation, degraded input
# ---------------------------------------------------------------------------

@windows_only
def test_one_submit_per_job_dir(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "quick.py", "print('ok')\n")
    rc, _ = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                     "--script", str(script))
    assert rc == 0
    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                           "--script", str(script))
    assert rc == 1 and "already has runner state" in payload["message"]


@windows_only
def test_planted_wrapper_path_is_refused(capsys, venv, job_dir, tmp_path):
    paths = wr.RunnerPaths(job_dir)
    paths.runner_dir.mkdir(parents=True)
    paths.wrapper.write_text("print('attacker')\n", encoding="utf-8")
    script = write_script(tmp_path, "quick.py", "print('ok')\n")
    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir),
                           "--venv", str(venv), "--script", str(script))
    assert rc == 1 and "wrapper path" in payload["message"]
    assert paths.wrapper.read_text() == "print('attacker')\n"  # never clobbered
    assert not paths.submit_meta.exists()  # the lock was released


@windows_only
def test_bin_python_venv_is_rejected_with_a_useful_message(capsys, tmp_path, job_dir):
    root = tmp_path / "posixvenv"
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    (root / "bin" / "python").write_text("", encoding="utf-8")
    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir),
                           "--venv", str(root), "--module", "json.tool")
    assert rc == 1
    assert "Scripts\\python.exe" in payload["message"]


@windows_only
def test_exactly_one_target_selector(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "quick.py", "print('ok')\n")
    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv))
    assert rc == 1 and "exactly one of" in payload["message"]
    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                           "--script", str(script), "--module", "json.tool")
    assert rc == 1 and "exactly one of" in payload["message"]


@windows_only
def test_missing_entrypoint_names_the_scripts_dir(capsys, venv, job_dir):
    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                           "--entrypoint", "cosmos-framework-evaluate")
    assert rc == 1
    assert "cosmos-framework-evaluate.exe" in payload["message"]


@windows_only
@pytest.mark.parametrize("flags", [
    ["--gpus", "2"],
    ["--gpu-ids", "0,1"],
    ["--gpu-ids", "0,1,2,3"],
])
def test_multi_gpu_is_refused_at_submit(capsys, venv, job_dir, flags):
    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                           "--module", "json.tool", *flags)
    assert rc == 1
    assert "0xC0000005" in payload["message"]
    assert "gloo" in payload["message"]


def test_unknown_placeholder_is_rejected(capsys, tmp_path, job_dir):
    if os.name != "nt":
        pytest.skip("submit refuses before argument validation off Windows")
    root = tmp_path / "venv"
    (root / "Scripts").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = C:\\python\n", encoding="utf-8")
    (root / "Scripts" / "python.exe").write_bytes(b"MZ")
    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(root),
                           "--module", "json.tool", "--arg", "{config_path}")
    assert rc == 1 and "Unknown placeholder" in payload["message"]


def test_corrupted_launcher_pid_yields_json_not_crash(capsys, job_dir):
    if os.name != "nt":
        pytest.skip("covered by the POSIX-refusal test off Windows")
    paths = wr.RunnerPaths(job_dir)
    paths.runner_dir.mkdir(parents=True)
    wr._atomic_write_json(paths.submit_meta, {"launch_started_at": time.time() - 600})
    wr._atomic_write_json(paths.launcher_status,
                          {"pid": "garbage", "process_start_marker": "x",
                           "job_object": "Local\\tao-win-does-not-exist"})
    _, status = run_verb(capsys, "status", "--job-dir", str(job_dir))
    assert status["status"] in ("ERROR", "PENDING", "UNKNOWN")
    rc, _ = run_verb(capsys, "cancel", "--job-dir", str(job_dir))
    assert rc in (0, 1)  # valid JSON reply either way — no traceback


def test_status_of_an_unknown_job_dir(capsys, tmp_path):
    if os.name != "nt":
        pytest.skip("covered by the POSIX-refusal test off Windows")
    _, payload = run_verb(capsys, "status", "--job-dir", str(tmp_path / "nope"))
    assert payload["status"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# The submit JSON contract survives a job dir the filesystem refuses
#
# These run EVERYWHERE. The failure is an ordinary OSError from mkdir/os.open,
# so the platform gate is monkeypatched away and the venv is a stub — nothing
# here launches a process, and the contract being tested (stdout is always
# parseable JSON) is the same on both operating systems.
# ---------------------------------------------------------------------------

@pytest.fixture()
def stub_venv(tmp_path, monkeypatch):
    """Enough of a venv for _validate_venv, on any OS, plus the platform gate off.

    _validate_venv only asks for a directory, pyvenv.cfg and
    Scripts\\python.exe, so submit can be driven to the filesystem work that
    follows without a real interpreter — and without needing a Windows host to
    cover a contract violation that a reviewer on Linux has to be able to see.
    """
    monkeypatch.setattr(wr, "_windows_only", lambda: None)
    root = tmp_path / "venv"
    (root / "Scripts").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = C:\\python\n", encoding="utf-8")
    (root / "Scripts" / "python.exe").write_bytes(b"MZ")
    return root


def test_submit_reports_an_unusable_job_dir_as_json_not_a_traceback(
    capsys, tmp_path, stub_venv
):
    """`--job-dir` pointing at a FILE must come back as the error contract.

    The runner-dir and log-dir mkdirs sat outside the guarded region, so every
    OSError but FileExistsError-on-submit_meta escaped as a raw traceback with
    an EMPTY stdout. An agent that runs this verb does json.loads(stdout) and
    was left with nothing to parse — no result, no message, no way to tell a
    bad --job-dir from a crashed runner. Reproduced on Windows with a
    --job-dir that is a file and with an unmapped drive (Z:\\jobs\\run1).
    """
    not_a_dir = tmp_path / "job-dir-is-actually-a-file"
    not_a_dir.write_text("regular file", encoding="utf-8")

    rc, payload = run_verb(capsys, "submit", "--job-dir", str(not_a_dir),
                           "--venv", str(stub_venv), "--module", "json.tool")

    assert rc == 1, "a failed submit has to exit non-zero"
    assert payload["result"] == "error"
    assert "Cannot create the runner directories" in payload["message"]
    assert str(not_a_dir) in payload["message"]
    # The OS's own words are the actionable part; they must not be swallowed.
    # [WinError 183] on Windows, [Errno 20] on Linux — same contract either way.
    assert re.search(r"\[(WinError|Errno) \d+\]", payload["message"]), payload


def test_submit_lock_failure_is_not_reported_as_a_duplicate_submit(
    capsys, tmp_path, stub_venv, monkeypatch
):
    """An O_EXCL create can fail for reasons other than "already claimed".

    FileExistsError is the submission lock doing its job and has to keep its
    own message. Any other OSError there (no permission, read-only volume) is
    a different problem with a different fix, and it too has to arrive as JSON
    rather than as a traceback.
    """
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    real_open = os.open

    def refuse_the_lock(path, flags, *args, **kwargs):
        if str(path).endswith("submit_meta.json"):
            raise PermissionError(13, "Access is denied")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(wr.os, "open", refuse_the_lock)

    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir),
                           "--venv", str(stub_venv), "--module", "json.tool")

    assert rc == 1 and payload["result"] == "error"
    assert "Cannot create the submission lock" in payload["message"]
    assert "Access is denied" in payload["message"]
    assert "one submit per job dir" not in payload["message"], (
        "a permission failure was reported as a duplicate submit, which sends "
        "the user to open a new job record instead of fixing the permission")


# ---------------------------------------------------------------------------
# Bounded log tail
# ---------------------------------------------------------------------------

def test_read_tail_is_bounded(tmp_path):
    log = tmp_path / "job.log"
    log.write_bytes(b"".join(b"line %d\n" % i for i in range(50000)))
    tail = wr._read_tail(log, 5)
    assert tail.splitlines() == [f"line {i}" for i in range(49995, 50000)]
    assert wr._read_tail(log, 0) == ""
