#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Four-verb process lifecycle for NATIVE Windows venv TAO jobs (no WSL, no container).

This is the windows-venv platform's "native CLI" — the role `docker` plays for
the docker skill. It launches a target as an argv vector whose first element is
``<venv>\\Scripts\\python.exe`` (or a console entrypoint ``.exe`` from the same
``Scripts`` directory), with no shell and no activation, detached in its own
process group and its own named Job Object, with a durable on-disk lifecycle
that survives the launching process:

    submit  --job-dir D --venv V (--script S | --module M | --entrypoint E)
            [--arg TOKEN]...            -> prints {pid, job_object, ...}
    status  --job-dir D            -> {status: PENDING|RUNNING|COMPLETE|ERROR|CANCELED|UNKNOWN}
    logs    --job-dir D [--tail N] -> bounded tail of the job log
    cancel  --job-dir D            -> TerminateJobObject on the whole tree

Job records are NOT written here — the agent owns them via tao_job_record.py
(open binds results_dir BEFORE submit; mark records the states this CLI
reports). The runner's own durable truth lives under ``<job-dir>\\.tao_runner\\``:
``submit_meta.json``, ``launcher_status.json`` (written by the wrapper the moment
it starts: pid + creation-time marker + Job Object name), ``exit_status.json``
(fsync'd atomic write on exit), a ``start_authorized`` gate, and a
``cancel_requested`` marker.

WHY THIS IS NOT THE POSIX RUNNER WITH PATHS SWAPPED
---------------------------------------------------
None of the POSIX primitives exist here, and one of them is actively destructive:

* ``os.kill(pid, 0)`` is NOT a liveness probe on Windows CPython. It is
  ``OpenProcess`` + ``TerminateProcess`` with no signal-0 special case, so using
  it to poll status KILLS the job — and ``TerminateProcess(h, 0)`` makes the
  corpse look like a clean exit 0. Liveness here is
  ``OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION)`` followed by
  ``WaitForSingleObject(h, 0)``: ``WAIT_TIMEOUT`` means alive, ``WAIT_OBJECT_0``
  means exited. Nothing in this file ever calls ``os.kill``.
* There is no ``setsid`` / ``killpg``. The process-group analogue is a **named
  Job Object** created by the wrapper (``CREATE_NEW_PROCESS_GROUP`` on the
  launch is for console signal isolation only). Every descendant inherits job
  membership automatically, ``QueryInformationJobObject`` enumerates the tree,
  and ``TerminateJobObject`` tears it down atomically from outside.

  The job is created with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, and that is
  load-bearing rather than tidy-up. A named job's NAME is released from the
  object namespace when the last **handle** closes, and the wrapper is the only
  handle holder — job membership does not itself hold a handle. Without the
  limit, killing the wrapper leaves the target and every descendant running
  under a job no one can open: ``status`` cannot see them and ``cancel`` cannot
  terminate them. With it, "the name no longer resolves" genuinely means "no
  members remain", which is the invariant ``_derive_status`` and ``cmd_cancel``
  are written against. Note the stock ``python -m venv`` layout gets this
  behaviour by accident through the CPython ``Scripts\\python.exe`` redirector's
  own kill-on-close job; a venv from PyPA ``virtualenv``, ``uv venv`` or
  ``python -m venv --symlinks`` has no redirector, which is why this is set
  explicitly here instead of being relied on.
* There is no ``/proc``. Durable identity is two facts recorded by the wrapper
  itself: (1) the **Job Object name**, a uuid4-derived kernel object name that
  cannot be recycled while any member process lives, and (2) the launcher's
  **process creation time** (``GetProcessTimes`` ``ftCreationTime``, a 64-bit
  FILETIME), which is the canonical PID-reuse disambiguator on Windows and plays
  exactly the role of ``/proc/<pid>/stat`` field 22.
* There is no ``O_NOFOLLOW``. A pre-planted wrapper path is refused by an
  explicit ``os.path.lexists`` check plus ``O_EXCL``.

TWO THINGS WINDOWS CANNOT REPRODUCE (see SKILL.md — do not imply parity)
-----------------------------------------------------------------------
(a) No graceful-then-forceful escalation. ``TerminateJobObject`` is
    unconditional and there is no Windows analogue of SIGTERM-then-SIGKILL, so
    cancel is always immediate and a job never gets to flush or checkpoint.
(b) ``GetExitCodeProcess`` returns an unsigned DWORD. There is no signal number
    to recover, the POSIX runner's ``code < 0`` branch is dead here, and a
    ``0xC0000005`` access violation collapses to a plain ``1`` once the wrapper's
    own exit code is normalized to 0..255. The raw DWORD is preserved in
    ``exit_status.json`` and rendered in hex in the ERROR message, which is the
    only compensation available. The six-state vocabulary is unchanged; only the
    ERROR message widens. Windows job records carry strictly less failure detail
    than Linux ones.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

VOCAB_PENDING = "PENDING"
VOCAB_RUNNING = "RUNNING"
VOCAB_COMPLETE = "COMPLETE"
VOCAB_ERROR = "ERROR"
VOCAB_CANCELED = "CANCELED"
VOCAB_UNKNOWN = "UNKNOWN"

RUNNER_DIR_NAME = ".tao_runner"
LOG_TAIL_MAX_BYTES = 2 * 1024 * 1024
LAUNCHER_RECORD_TIMEOUT_SECONDS = 15.0
# STATUS may never declare a launch dead while submit could still legitimately
# be waiting for the launcher record — the grace strictly exceeds that wait.
# Windows process creation is slower than fork/exec (no COW, full image load),
# and Defender real-time scanning of a fresh .py can add seconds on first touch.
PENDING_LAUNCH_GRACE_SECONDS = LAUNCHER_RECORD_TIMEOUT_SECONDS + 10.0
GATE_WAIT_TIMEOUT_SECONDS = 30.0  # wrapper gives up if submit dies pre-gate
REPLACE_RETRY_SECONDS = 5.0       # AV/indexer sharing violations on os.replace

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_ENTRYPOINT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

WRAPPER_EXIT_NOT_AUTHORIZED = 125
WRAPPER_EXIT_STRAYS = 126
WRAPPER_EXIT_LAUNCH_FAILED = 127
WRAPPER_EXIT_CANCELED = 130
CANCELED_BEFORE_START = "Canceled before the script started"

# The verified single-GPU Windows training environment. Constants, not secrets —
# safe to place in the child environment directly. An operator-set value in the
# parent environment always wins, so MASTER_PORT can be moved off a Hyper-V
# reserved range (see `netsh interface ipv4 show excludedportrange protocol=tcp`)
# without editing this file.
TAO_WINDOWS_ENV = {
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "29500",
    "RANK": "0",
    "WORLD_SIZE": "1",
    "LOCAL_RANK": "0",
    "USE_LIBUV": "0",
    "PYTHONIOENCODING": "utf-8",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_MODE": "disabled",
}

# ---------------------------------------------------------------------------
# Win32 bindings. Created lazily/guarded so this module still IMPORTS on POSIX
# (the verbs refuse there through the JSON contract, they do not traceback).
# ---------------------------------------------------------------------------
IS_WINDOWS = sys.platform == "win32"

PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
JOB_OBJECT_ALL_ACCESS = 0x1F001F
JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
ERROR_ALREADY_EXISTS = 183
ERROR_INVALID_PARAMETER = 87
ERROR_MORE_DATA = 234
ERROR_FILE_NOT_FOUND = 2

_kernel32 = None


def _k32():
    """kernel32 with argtypes pinned — HANDLE must not truncate on x64."""
    global _kernel32
    if _kernel32 is not None:
        return _kernel32
    from ctypes import wintypes

    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    lib.OpenProcess.restype = wintypes.HANDLE
    lib.CloseHandle.argtypes = (wintypes.HANDLE,)
    lib.CloseHandle.restype = wintypes.BOOL
    lib.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    lib.WaitForSingleObject.restype = wintypes.DWORD
    lib.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    lib.TerminateProcess.restype = wintypes.BOOL
    lib.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
    )
    lib.GetProcessTimes.restype = wintypes.BOOL
    lib.OpenJobObjectW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    lib.OpenJobObjectW.restype = wintypes.HANDLE
    lib.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    lib.TerminateJobObject.restype = wintypes.BOOL
    lib.QueryInformationJobObject.argtypes = (
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    lib.QueryInformationJobObject.restype = wintypes.BOOL
    _kernel32 = lib
    return lib


def _pid_list_struct(capacity: int):
    from ctypes import wintypes

    class _PidList(ctypes.Structure):
        _fields_ = [
            ("NumberOfAssignedProcesses", wintypes.DWORD),
            ("NumberOfProcessIdsInList", wintypes.DWORD),
            ("ProcessIdList", ctypes.c_size_t * capacity),
        ]

    return _PidList


def _open_process(pid: int, access: int):
    """(handle, last_error). handle is None when the open failed."""
    lib = _k32()
    handle = lib.OpenProcess(access, False, pid)
    if not handle:
        return None, ctypes.get_last_error()
    return handle, 0


def _process_liveness(pid: int) -> str:
    """'alive' | 'gone' | 'unknown' — WITHOUT ever signalling the process.

    ``os.kill(pid, 0)`` is TerminateProcess on Windows. This is the replacement:
    a wait with a zero timeout on the process handle, which is a pure query.
    ``unknown`` (e.g. the pid belongs to another user, so OpenProcess is denied)
    is an indeterminate probe and must never be read as evidence of death.
    """
    if pid <= 0:
        return "gone"
    handle, err = _open_process(pid, SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION)
    if handle is None:
        return "gone" if err == ERROR_INVALID_PARAMETER else "unknown"
    lib = _k32()
    try:
        waited = lib.WaitForSingleObject(handle, 0)
    finally:
        lib.CloseHandle(handle)
    if waited == WAIT_TIMEOUT:
        return "alive"
    if waited == WAIT_OBJECT_0:
        return "gone"
    return "unknown"


def _process_creation_marker(pid: int) -> str | None:
    """The process's 64-bit creation FILETIME, as a decimal string.

    This is the Windows PID-reuse disambiguator — the direct analogue of
    ``/proc/<pid>/stat`` field 22. Two processes can share a pid over time but
    never a (pid, creation-time) pair.
    """
    if pid <= 0:
        return None
    from ctypes import wintypes

    handle, _ = _open_process(pid, PROCESS_QUERY_LIMITED_INFORMATION)
    if handle is None:
        return None
    lib = _k32()
    creation = wintypes.FILETIME()
    exited = wintypes.FILETIME()
    kernel_time = wintypes.FILETIME()
    user_time = wintypes.FILETIME()
    try:
        ok = lib.GetProcessTimes(
            handle, ctypes.byref(creation), ctypes.byref(exited),
            ctypes.byref(kernel_time), ctypes.byref(user_time),
        )
    finally:
        lib.CloseHandle(handle)
    if not ok:
        return None
    return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)


def _open_job(name: str, access: int = JOB_OBJECT_ALL_ACCESS):
    if not name:
        return None
    lib = _k32()
    handle = lib.OpenJobObjectW(access, False, name)
    return handle or None


def _job_process_ids(name: str) -> list[int] | None:
    """Members of the named Job Object, or None when the name cannot be opened.

    None is AMBIGUOUS and must never be read as "no members left". A named Job
    Object is unregistered from the object namespace as soon as the last
    *handle* closes, and the wrapper is the only handle holder — job membership
    does not itself hold a handle. So once the wrapper dies the name is
    unresolvable even while the target and its descendants are still running.

    Callers must therefore distinguish three cases:
      []    the job opened and is genuinely empty  -> everything exited
      [...] the job opened and has live members    -> still running
      None  the name could not be opened           -> liveness is UNKNOWABLE
    """
    handle = _open_job(name)
    if handle is None:
        return None
    lib = _k32()
    from ctypes import wintypes

    try:
        capacity = 256
        for _ in range(6):
            buffer = _pid_list_struct(capacity)()
            returned = wintypes.DWORD(0)
            ok = lib.QueryInformationJobObject(
                handle, JOB_OBJECT_BASIC_PROCESS_ID_LIST, ctypes.byref(buffer),
                ctypes.sizeof(buffer), ctypes.byref(returned),
            )
            if ok:
                count = buffer.NumberOfProcessIdsInList
                return [int(buffer.ProcessIdList[i]) for i in range(count)]
            if ctypes.get_last_error() != ERROR_MORE_DATA:
                return []
            capacity *= 4
        return []
    finally:
        lib.CloseHandle(handle)


def _terminate_job(name: str, exit_code: int = 1) -> bool:
    """Unconditional teardown of the whole tree. There is NO graceful phase."""
    handle = _open_job(name)
    if handle is None:
        return False
    lib = _k32()
    try:
        return bool(lib.TerminateJobObject(handle, exit_code))
    finally:
        lib.CloseHandle(handle)


def _terminate_tree_by_pid(pid: int, timeout: float = 5.0) -> bool:
    """taskkill /T /F fallback for when the Job Object name is unreachable.

    Named kernel objects live in the caller's session namespace, so a cancel
    issued from a DIFFERENT Windows session (a service, another RDP login)
    cannot open the job by name. taskkill walks the parent-pid chain instead
    and works across sessions given the rights.
    """
    if pid <= 0:
        return False
    try:
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, text=True, timeout=max(5.0, timeout),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


# ---------------------------------------------------------------------------
# The wrapper written into the job dir and executed as
#   <venv>\Scripts\python.exe launch_job.py <exit> <launcher> <gate> <cancel>
#                             <job_object_name> <command...>
# Self-contained (stdlib + ctypes only).
# ---------------------------------------------------------------------------
JOB_WRAPPER_SOURCE = r'''
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from ctypes import wintypes

PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
ERROR_ALREADY_EXISTS = 183
ERROR_MORE_DATA = 234


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimitInformation(ctypes.Structure):
    """JOBOBJECT_EXTENDED_LIMIT_INFORMATION.

    Laid out in full even though only LimitFlags is set:
    SetInformationJobObject validates cbJobObjectInformationLength against the
    exact struct size and fails with ERROR_BAD_LENGTH on a truncated one.
    """

    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
k32.CreateJobObjectW.restype = wintypes.HANDLE
k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
k32.AssignProcessToJobObject.restype = wintypes.BOOL
k32.GetCurrentProcess.restype = wintypes.HANDLE
k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
k32.OpenProcess.restype = wintypes.HANDLE
k32.CloseHandle.argtypes = (wintypes.HANDLE,)
k32.CloseHandle.restype = wintypes.BOOL
k32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
k32.TerminateProcess.restype = wintypes.BOOL
k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
k32.WaitForSingleObject.restype = wintypes.DWORD
k32.GetProcessTimes.argtypes = (
    wintypes.HANDLE,
    ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
    ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME))
k32.GetProcessTimes.restype = wintypes.BOOL
k32.QueryInformationJobObject.argtypes = (
    wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD))
k32.QueryInformationJobObject.restype = wintypes.BOOL
k32.SetInformationJobObject.argtypes = (
    wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
k32.SetInformationJobObject.restype = wintypes.BOOL

# Module-global so the job handle is never garbage collected: closing it would
# drop the kernel object name that cancel resolves the tree through.
JOB_HANDLE = None


def _write_status(path, payload):
    """Atomic + fsync'd, with a retry window.

    os.replace is MoveFileEx(REPLACE_EXISTING) here, and on Windows that loses
    to any open handle on the destination — an antivirus real-time scan or the
    search indexer touching the file produces a transient WinError 5/32. A
    single attempt is exactly how a completed job ends up with no exit record.
    """
    temp_path = "%s.%d.tmp" % (path, os.getpid())
    with open(temp_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    deadline = time.monotonic() + 5.0
    while True:
        try:
            os.replace(temp_path, path)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def _creation_marker(pid):
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    created = wintypes.FILETIME()
    a, b, c = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    try:
        ok = k32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(a),
                                 ctypes.byref(b), ctypes.byref(c))
    finally:
        k32.CloseHandle(handle)
    if not ok:
        return None
    return str((created.dwHighDateTime << 32) | created.dwLowDateTime)


def _create_job(name):
    """Create the named job and put OURSELVES in it.

    Every descendant inherits membership at creation, so there is no window in
    which a spawned DataLoader worker is outside the job — the race that makes
    "assign the child after Popen" wrong. Nested jobs are supported since
    Windows 8, so this succeeds even when the agent's shell is already jobbed.
    """
    global JOB_HANDLE
    handle = k32.CreateJobObjectW(None, name)
    if not handle:
        return None, "CreateJobObject failed (error %d)" % ctypes.get_last_error()
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        k32.CloseHandle(handle)
        return None, "Job object name %s already exists" % name
    if not k32.AssignProcessToJobObject(handle, k32.GetCurrentProcess()):
        err = ctypes.get_last_error()
        k32.CloseHandle(handle)
        return None, "AssignProcessToJobObject failed (error %d)" % err
    # KILL_ON_JOB_CLOSE, set only AFTER we are a member so the failure paths
    # above can still CloseHandle without killing anything.
    #
    # This is what makes "the job name no longer resolves" actually MEAN "no
    # members remain". A named job's name is released when the last HANDLE
    # closes, and this wrapper is the only handle holder -- so without this
    # limit, killing the wrapper leaves the target and its descendants running
    # under a job nobody can open, i.e. an unreapable orphan that `status`
    # cannot see and `cancel` cannot terminate.
    #
    # The stock `python -m venv` layout gets this behaviour by accident, via
    # the CPython Scripts\\python.exe redirector's own kill-on-close job
    # (LimitFlags 0x3000). A venv from PyPA `virtualenv`, `uv venv`, or
    # `python -m venv --symlinks` has no redirector and no such protection.
    # Setting it here makes the guarantee explicit and layout-independent.
    #
    # Non-fatal on failure: the job still works for tracking and teardown, and
    # _derive_status reports UNKNOWN rather than a false terminal state when a
    # name turns out to be unresolvable.
    try:
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(
            handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info), ctypes.sizeof(info),
        )
    except Exception:
        pass
    JOB_HANDLE = handle
    return handle, None


def _job_members(handle, own_pid):
    capacity = 256
    for _ in range(6):
        class _PidList(ctypes.Structure):
            _fields_ = [("NumberOfAssignedProcesses", wintypes.DWORD),
                        ("NumberOfProcessIdsInList", wintypes.DWORD),
                        ("ProcessIdList", ctypes.c_size_t * capacity)]
        buffer = _PidList()
        returned = wintypes.DWORD(0)
        ok = k32.QueryInformationJobObject(
            handle, JOB_OBJECT_BASIC_PROCESS_ID_LIST, ctypes.byref(buffer),
            ctypes.sizeof(buffer), ctypes.byref(returned))
        if ok:
            n = buffer.NumberOfProcessIdsInList
            return [int(buffer.ProcessIdList[i]) for i in range(n)
                    if int(buffer.ProcessIdList[i]) != own_pid]
        if ctypes.get_last_error() != ERROR_MORE_DATA:
            return []
        capacity *= 4
    return []


def _stop_strays(handle, timeout=2.0):
    """Kill anything left in the job after the target exits.

    TerminateJobObject is not usable here: the wrapper is itself a member, so it
    would kill the writer before the exit record lands. Strays are terminated
    one at a time instead. There is NO graceful phase — Windows has no SIGTERM.
    """
    own_pid = os.getpid()
    members = _job_members(handle, own_pid)
    for pid in members:
        child = k32.OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, False, pid)
        if not child:
            continue
        try:
            k32.TerminateProcess(child, 1)
            k32.WaitForSingleObject(child, int(timeout * 1000))
        finally:
            k32.CloseHandle(child)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        members = _job_members(handle, own_pid)
        if not members:
            return
        time.sleep(0.05)
    members = _job_members(handle, own_pid)
    if members:
        raise RuntimeError("job object still has live members: %s" % members)


def main():
    status_path = sys.argv[1]
    launch_path = sys.argv[2]
    start_gate_path = sys.argv[3]
    cancel_path = sys.argv[4]
    job_name = sys.argv[5]
    command = sys.argv[6:]
    started_at = time.time()

    handle, job_error = _create_job(job_name)
    if handle is None:
        _write_status(status_path, {
            "return_code": 125,
            "error": "Job object could not be created: %s" % job_error,
            "started_at": started_at,
            "finished_at": time.time(),
        })
        return 125

    _write_status(launch_path, {
        "pid": os.getpid(),
        "process_start_marker": _creation_marker(os.getpid()),
        "job_object": job_name,
        "started_at": started_at,
    })

    # A healthy submit opens the gate within one poll of reading our record; if
    # it died first, give up with a durable record instead of spinning as a
    # phantom RUNNING job forever.
    gate_timeout = float(os.environ.get("TAO_RUNNER_GATE_TIMEOUT", "30"))
    gate_deadline = time.monotonic() + gate_timeout
    while not os.path.exists(start_gate_path):
        if time.monotonic() > gate_deadline:
            _write_status(status_path, {
                "return_code": 125,
                "error": "Start was never authorized (submit died before opening the gate)",
                "started_at": started_at,
                "finished_at": time.time(),
            })
            return 125
        if os.path.exists(cancel_path):
            _write_status(status_path, {
                "return_code": 130,
                "error": "Canceled before the script started",
                "started_at": started_at,
                "finished_at": time.time(),
            })
            return 130
        time.sleep(0.02)
    if os.path.exists(cancel_path):
        _write_status(status_path, {
            "return_code": 130,
            "error": "Canceled before the script started",
            "started_at": started_at,
            "finished_at": time.time(),
        })
        return 130

    # Route the target's output EXPLICITLY at this wrapper's own stdout/stderr,
    # which submit set to the job log.
    #
    # `subprocess.call(command)` with all three streams None is the bug it
    # replaces: CPython's Windows _get_handles short-circuits that case to
    # "don't set STARTF_USESTDHANDLES" and leaves the child to pick up handles
    # by implicit inheritance. Under DETACHED_PROCESS (no console) that is not
    # reliable, and whether it works depends on the venv LAYOUT: a stock
    # `python -m venv` works only because Scripts\python.exe is CPython's
    # redirector, which re-spawns the real interpreter passing handles on.
    # A venv from PyPA `virtualenv`, `uv venv`, or `python -m venv --symlinks`
    # has a real interpreter there and no redirector -- and _validate_venv
    # accepts all of them. On those layouts every byte of the job's output was
    # silently discarded: job.log stayed 0 bytes and `logs` returned nothing,
    # while `status` still reported COMPLETE with return_code 0. Losing the
    # training log of a completed multi-hour run is not recoverable after the
    # fact, so pass the descriptors rather than hoping they propagate.
    stdout_target = 1
    try:
        os.fstat(stdout_target)
    except OSError:
        stdout_target = None  # No usable fd: fall back to inheritance.
    error = None
    try:
        return_code = subprocess.call(
            command, stdout=stdout_target,
            stderr=subprocess.STDOUT if stdout_target is not None else None,
        )
    except BaseException as exc:  # Preserve a durable failure record.
        traceback.print_exc()
        return_code = 127
        error = "%s: %s" % (type(exc).__name__, exc)
    try:
        _stop_strays(handle)
    except BaseException as exc:
        traceback.print_exc()
        cleanup_error = "%s: %s" % (type(exc).__name__, exc)
        error = ("%s; %s" % (error, cleanup_error)) if error else cleanup_error
        if return_code == 0:
            return_code = 126
    _write_status(status_path, {
        "return_code": return_code,
        "error": error,
        "started_at": started_at,
        "finished_at": time.time(),
    })
    # GetExitCodeProcess is an unsigned DWORD, so a crash code such as
    # 0xC0000005 (3221225477) cannot survive this clamp. The RAW value is in the
    # record above; only the wrapper's own process exit code is lossy.
    return return_code if 0 <= return_code <= 255 else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _replace_with_retry(temp_path: Path, path: Path,
                        timeout: float = REPLACE_RETRY_SECONDS) -> None:
    """os.replace, retried. See the wrapper's _write_status for the why."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.replace(temp_path, path)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _replace_with_retry(temp_path, path)


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _coerce_pid(value) -> int:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return 0
    return pid if pid > 0 else 0


def _launcher_state(pid: int, expected_marker: str | None, job_name: str | None) -> str:
    """'running' | 'gone' for STATUS derivation.

    Conservative toward 'running': a dead launcher self-heals on the next poll
    (its exit record, or a definitive probe), but a live job misread as gone
    would be recorded terminal. So 'gone' requires POSITIVE evidence — the wait
    handle signalled, or the creation-time marker no longer matches (a recycled
    pid). Indeterminate probes stay 'running'.
    """
    if pid <= 0:
        return "gone"
    liveness = _process_liveness(pid)
    if liveness == "gone":
        return "gone"
    if expected_marker:
        marker_now = _process_creation_marker(pid)
        if marker_now is not None and marker_now != str(expected_marker):
            return "gone"  # PID reuse: a different process now holds this pid.
    if job_name and liveness == "unknown" and _job_process_ids(job_name) is None:
        # Both probes indeterminate/negative and the kernel object name is gone:
        # nothing of this job remains.
        return "gone"
    return "running"


def _process_matches(pid: int, expected_marker: str | None) -> bool:
    """True only if ``pid`` is POSITIVELY this job's launcher (PID-reuse safe).

    The bias is the inverse of _launcher_state: this gates the taskkill
    fallback, so an indeterminate probe means "do not kill".
    """
    if pid <= 0:
        return False
    if _process_liveness(pid) != "alive":
        return False
    if expected_marker:
        marker_now = _process_creation_marker(pid)
        if marker_now is None or marker_now != str(expected_marker):
            return False
    return True


def _read_tail(path: Path, line_count: int, max_bytes: int = LOG_TAIL_MAX_BYTES) -> str:
    """Read at most ``line_count`` lines without scanning the whole log."""
    if line_count <= 0:
        return ""
    block_size = 64 * 1024
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        remaining = stream.tell()
        chunks: list[bytes] = []
        newline_count = 0
        bytes_read = 0
        while remaining > 0 and newline_count <= line_count and bytes_read < max_bytes:
            size = min(block_size, remaining, max_bytes - bytes_read)
            remaining -= size
            stream.seek(remaining)
            chunk = stream.read(size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
            bytes_read += len(chunk)
    data = b"".join(reversed(chunks))
    return b"".join(data.splitlines(keepends=True)[-line_count:]).decode(
        "utf-8", errors="replace"
    )


class RunnerPaths:
    """All durable control-file paths for one job dir."""

    def __init__(self, job_dir: Path):
        self.job_dir = job_dir
        self.runner_dir = job_dir / RUNNER_DIR_NAME
        self.wrapper = self.runner_dir / "launch_job.py"
        self.submit_meta = self.runner_dir / "submit_meta.json"
        self.launcher_status = self.runner_dir / "launcher_status.json"
        self.exit_status = self.runner_dir / "exit_status.json"
        self.start_gate = self.runner_dir / "start_authorized"
        self.cancel_marker = self.runner_dir / "cancel_requested"
        self.log_path = job_dir / "logs" / "job.log"


def _fail(message: str, **extra) -> int:
    print(json.dumps({"result": "error", "message": message, **extra}))
    return 1


def _emit(**payload) -> int:
    print(json.dumps(payload))
    return 0


def _windows_only() -> str | None:
    if os.name == "nt":
        return None
    return (
        "windows_job_runner.py runs only on native Windows (os.name == 'nt'); "
        f"this host reports os.name={os.name!r}. Use tao-run-on-virtualenv for "
        "POSIX hosts — its process-group primitives have no meaning here, and "
        "this runner's Job Object and GetProcessTimes primitives have none there."
    )


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def _validate_venv(venv: str) -> Path:
    path = Path(venv).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Virtual environment does not exist: {path}")
    if not (path / "pyvenv.cfg").is_file():
        raise ValueError(f"Not a Python virtual environment (pyvenv.cfg is missing): {path}")
    python_path = path / "Scripts" / "python.exe"
    if not python_path.is_file():
        raise ValueError(
            f"Virtual environment interpreter is missing: {python_path} "
            "(Windows venvs put it in Scripts\\python.exe, not bin/python)"
        )
    return path


def _validate_gpus(gpus: int | None, gpu_ids: str | None) -> list[int] | None:
    if gpus is not None and gpus < 0:
        raise ValueError("--gpus must be a non-negative integer")
    if gpu_ids is None:
        if gpus is not None and gpus > 1:
            raise ValueError(
                "--gpus greater than 1 is refused: on native Windows gloo cannot run any "
                "collective on a CUDA tensor and NCCL has no Windows build, so a "
                "multi-GPU run dies with 0xC0000005 and no traceback. Multi-node is "
                "not supported either. Request a single device."
            )
        return None
    tokens = [] if not gpu_ids.strip() else gpu_ids.split(",")
    normalized: list[int] = []
    for token in tokens:
        token = token.strip()
        if not re.fullmatch(r"\d+", token):
            raise ValueError("--gpu-ids must be a comma-separated list of non-negative integers")
        normalized.append(int(token))
    if len(set(normalized)) != len(normalized):
        raise ValueError("--gpu-ids must not contain duplicates")
    if gpus is not None and len(normalized) != gpus:
        raise ValueError("--gpus must match the number of --gpu-ids")
    if len(normalized) > 1:
        raise ValueError(
            "more than one --gpu-ids entry is refused: on native Windows gloo cannot run "
            "any collective on a CUDA tensor and NCCL has no Windows build, so a "
            "multi-GPU run dies with 0xC0000005 and no traceback. Pass exactly one id."
        )
    return normalized


def _render_arg(token: str, placeholders: dict[str, str]) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in placeholders:
            raise ValueError(
                f"Unknown placeholder {{{name}}} in --arg {token!r}; "
                f"supported: {', '.join(sorted(placeholders))}"
            )
        return placeholders[name]

    return _PLACEHOLDER_RE.sub(replace, token)


def _resolve_target(args: argparse.Namespace, venv_path: Path, cwd: Path) -> tuple[list[str], str]:
    """(prefix argv, human-readable target). Exactly one selector is allowed."""
    chosen = [name for name in ("script", "module", "entrypoint") if getattr(args, name)]
    if len(chosen) != 1:
        raise ValueError(
            "exactly one of --script / --module / --entrypoint is required "
            f"(got: {chosen or 'none'})"
        )
    python_path = venv_path / "Scripts" / "python.exe"
    if args.script:
        script_path = Path(args.script).expanduser()
        if not script_path.is_absolute():
            script_path = cwd / script_path
        script_path = script_path.resolve()
        if not script_path.is_file():
            raise ValueError(f"Python script does not exist: {script_path}")
        return [str(python_path), str(script_path)], f"script:{script_path}"
    if args.module:
        if not _MODULE_RE.fullmatch(args.module):
            raise ValueError(f"--module is not a dotted Python module name: {args.module!r}")
        return [str(python_path), "-m", args.module], f"module:{args.module}"
    name = args.entrypoint
    if not _ENTRYPOINT_RE.fullmatch(name):
        raise ValueError(
            f"--entrypoint must be a bare console-script name, not a path: {name!r}"
        )
    exe_path = venv_path / "Scripts" / (name if name.endswith(".exe") else f"{name}.exe")
    if not exe_path.is_file():
        raise ValueError(
            f"Console entrypoint not installed in this venv: {exe_path}. "
            "Install the selected model's console entrypoint into this venv and "
            "read that model's native-Windows setup reference."
        )
    return [str(exe_path)], f"entrypoint:{name}"


def cmd_submit(args: argparse.Namespace) -> int:
    refusal = _windows_only()
    if refusal:
        return _fail(refusal)
    try:
        venv_path = _validate_venv(args.venv)
        cwd = Path(args.cwd).expanduser().resolve() if args.cwd else Path.cwd()
        if not cwd.is_dir():
            raise ValueError(f"Working directory does not exist: {cwd}")
        target_argv, target_label = _resolve_target(args, venv_path, cwd)
        gpu_ids = _validate_gpus(args.gpus, args.gpu_ids)
        job_dir = Path(args.job_dir).expanduser().resolve()
        paths = RunnerPaths(job_dir)
        for existing in (paths.launcher_status, paths.exit_status):
            if existing.exists():
                raise ValueError(
                    f"Job dir already has runner state ({existing.name}); "
                    "one submit per job dir — open a new job record for a retry"
                )
        placeholders = {
            "results_dir": str(job_dir),
            "job_id": args.job_id or job_dir.name,
        }
        if args.config_path:
            placeholders["config_path"] = args.config_path
        rendered_args = [_render_arg(token, placeholders) for token in (args.arg or [])]
    except ValueError as exc:
        return _fail(str(exc))

    # Creating the control-file tree is a filesystem operation on an
    # operator-supplied --job-dir, so it fails for all the ordinary reasons: the
    # path is a file, the drive is unmapped, the directory is read-only. Those
    # have to come back as the JSON error contract like every other submit
    # failure — an agent parses stdout, and a raw OSError traceback leaves it
    # with an empty stdout and nothing to json.loads().
    try:
        paths.runner_dir.mkdir(parents=True, exist_ok=True)
        paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _fail(f"Cannot create the runner directories under {job_dir}: {exc}")
    # O_EXCL on submit_meta is the SUBMISSION LOCK: exactly one submit can claim
    # a job dir, even racing concurrently (check-then-act is not enough).
    try:
        meta_fd = os.open(paths.submit_meta, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return _fail(
            "Job dir already has runner state (submit_meta.json); "
            "one submit per job dir — open a new job record for a retry"
        )
    except OSError as exc:
        # Anything but "already claimed": no permission, bad path, read-only
        # volume. Distinct message, because this is NOT a duplicate submit.
        return _fail(f"Cannot create the submission lock {paths.submit_meta}: {exc}")
    # Windows has no O_NOFOLLOW, so the symlink/junction refusal is explicit:
    # lexists sees the link itself, and O_EXCL then refuses any regular file.
    if os.path.lexists(paths.wrapper):
        os.close(meta_fd)
        paths.submit_meta.unlink(missing_ok=True)
        return _fail(
            f"Refusing pre-existing wrapper path {paths.wrapper}: it already exists "
            "(a planted file, symlink or junction is never followed or executed)"
        )
    try:
        wrapper_fd = os.open(paths.wrapper, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except OSError as exc:
        os.close(meta_fd)
        paths.submit_meta.unlink(missing_ok=True)
        return _fail(f"Refusing pre-existing wrapper path {paths.wrapper}: {exc}")
    with os.fdopen(wrapper_fd, "w", encoding="utf-8") as stream:
        stream.write(JOB_WRAPPER_SOURCE)

    # The Job Object name IS the durable, PID-reuse-proof handle on the tree.
    # Local\ scopes it to this Windows session; see _terminate_tree_by_pid.
    job_object_name = f"Local\\tao-win-{uuid.uuid4().hex}"
    script_argv = [*target_argv, *rendered_args]
    launcher_argv = [
        str(venv_path / "Scripts" / "python.exe"),
        str(paths.wrapper),
        str(paths.exit_status),
        str(paths.launcher_status),
        str(paths.start_gate),
        str(paths.cancel_marker),
        job_object_name,
        *script_argv,
    ]

    process_env = os.environ.copy()
    process_env.pop("PYTHONHOME", None)
    for name in args.env or []:
        # Pass credentials by NAME only; the value comes from this process's
        # environment and never lands on argv or in runner files.
        if name in os.environ:
            process_env[name] = os.environ[name]
    process_env["VIRTUAL_ENV"] = str(venv_path)
    current_path = process_env.get("PATH", "")
    process_env["PATH"] = str(venv_path / "Scripts") + (
        os.pathsep + current_path if current_path else ""
    )
    process_env["TAO_JOB_ID"] = placeholders["job_id"]
    process_env["TAO_RESULTS_ROOT"] = str(job_dir)
    process_env.setdefault("PYTHONUNBUFFERED", "1")
    # cp1252 is still the default console encoding on many Windows hosts; a
    # single non-ASCII byte in a progress bar otherwise aborts the run with
    # UnicodeEncodeError long after the useful work started.
    process_env.setdefault("PYTHONIOENCODING", "utf-8")
    if args.tao_windows_env:
        for name, value in TAO_WINDOWS_ENV.items():
            process_env.setdefault(name, value)  # operator-set value wins
        process_env.setdefault("IMAGINAIRE_OUTPUT_ROOT", str(job_dir))
        # Both, always. TAO_API_JOB_ID alone is fatal at training iteration 0
        # with "TAO_API_RESULTS_DIR is required when TAO_API_JOB_ID is set",
        # while eval/inference/export need TAO_API_JOB_ID or they refuse to
        # start. One env block has to satisfy both.
        process_env["TAO_API_JOB_ID"] = placeholders["job_id"]
        process_env["TAO_API_RESULTS_DIR"] = str(job_dir)
    if gpu_ids is not None:
        process_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
    elif args.gpus == 0:
        process_env["CUDA_VISIBLE_DEVICES"] = ""

    with os.fdopen(meta_fd, "w", encoding="utf-8") as stream:
        json.dump({
            "job_id": placeholders["job_id"],
            "launch_started_at": time.time(),
            "argv": script_argv,
            "target": target_label,
            "venv_path": str(venv_path),
            "wrapper_path": str(paths.wrapper),
            "job_object": job_object_name,
            "cwd": str(cwd),
            "log_path": str(paths.log_path),
            "gpu_ids": gpu_ids,
            "env_passthrough": list(args.env or []),
            "tao_windows_env": bool(args.tao_windows_env),
        }, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    # DETACHED_PROCESS: no console, so closing the launching terminal cannot
    # take the job with it. CREATE_NEW_PROCESS_GROUP: a Ctrl+C in the agent's
    # console is not delivered to the job. Neither is start_new_session, which
    # is POSIX-only and would raise here.
    creation_flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        log_file = paths.log_path.open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                launcher_argv,
                cwd=str(cwd),
                env=process_env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                shell=False,
                close_fds=True,
                creationflags=creation_flags,
            )
        finally:
            log_file.close()  # Popen duplicated the handle for the child.
    except OSError as exc:
        return _fail(f"Failed to launch the wrapper: {exc}")

    # Wait for the wrapper's durable identity record, then open the gate. A
    # wrapper that never writes it means the interpreter/env is broken — tear
    # the tree down and report, before the target has had any chance to run.
    deadline = time.monotonic() + LAUNCHER_RECORD_TIMEOUT_SECONDS
    launcher = None
    while time.monotonic() < deadline:
        launcher = _read_json(paths.launcher_status)
        if launcher and launcher.get("pid"):
            break
        if process.poll() is not None:
            break
        time.sleep(0.05)
    if not launcher or not launcher.get("pid"):
        if not _terminate_job(job_object_name):
            _terminate_tree_by_pid(process.pid)
        tail = _read_tail(paths.log_path, 40) if paths.log_path.exists() else ""
        return _fail(
            "Launcher never wrote its durable identity record "
            f"(wrapper exit code: {process.poll()})",
            log_tail=tail,
        )

    paths.start_gate.touch()
    return _emit(
        result="submitted",
        status=VOCAB_RUNNING,
        job_dir=str(job_dir),
        pid=launcher["pid"],
        process_start_marker=launcher.get("process_start_marker"),
        job_object=job_object_name,
        target=target_label,
        log_path=str(paths.log_path),
        backend_ref=f"pid:{launcher['pid']}",
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _describe_code(code: int) -> str:
    """Widen the ERROR message with what the unsigned DWORD actually was.

    This is the only compensation for the missing signal number. 0xC0000005 is
    the one every Windows GPU operator meets: an access violation, produced here
    by a CUDA collective under gloo, by a DCP stage-through-host-memory step
    with too little RAM, or by a mismatched native DLL.
    """
    known = {
        0xC0000005: "STATUS_ACCESS_VIOLATION",
        0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
        0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
        0xC00000FD: "STATUS_STACK_OVERFLOW",
        0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
        0xC0000374: "STATUS_HEAP_CORRUPTION",
        0xC0000135: "STATUS_DLL_NOT_FOUND",
        0xC0000139: "STATUS_ENTRYPOINT_NOT_FOUND",
    }
    unsigned = code & 0xFFFFFFFF
    if unsigned in known:
        return f"code {code} (0x{unsigned:08X} {known[unsigned]})"
    if unsigned >= 0xC0000000:
        return f"code {code} (0x{unsigned:08X}, an NTSTATUS failure)"
    return f"code {code}"


def _status_from_exit_record(paths: RunnerPaths) -> tuple[str, str, dict] | None:
    exit_record = _read_json(paths.exit_status)
    if exit_record is None:
        return None
    code = exit_record.get("return_code")
    if not isinstance(code, int) or isinstance(code, bool):
        return None
    if paths.cancel_marker.exists():
        return VOCAB_CANCELED, "Job object was terminated by cancel", {"return_code": code}
    if code == 0:
        return VOCAB_COMPLETE, "Process exited successfully", {"return_code": 0}
    # There is no `code < 0` branch: GetExitCodeProcess is an unsigned DWORD, so
    # a Windows job record can never name the signal that killed it.
    return (
        VOCAB_ERROR,
        f"Process exited with {_describe_code(code)}",
        {"return_code": code, "error": exit_record.get("error")},
    )


def _derive_status(paths: RunnerPaths) -> tuple[str, str, dict]:
    """Stateless status derivation from the job dir's durable files."""
    from_record = _status_from_exit_record(paths)
    if from_record is not None:
        return from_record

    canceled = paths.cancel_marker.exists()
    submit_meta = _read_json(paths.submit_meta)
    launcher = _read_json(paths.launcher_status)
    job_name = (launcher or {}).get("job_object") or (submit_meta or {}).get("job_object")

    if launcher and _coerce_pid(launcher.get("pid")):
        pid = _coerce_pid(launcher.get("pid"))
        marker = launcher.get("process_start_marker")
        if _launcher_state(pid, marker, job_name) == "running":
            message = f"Process {pid} is running"
            if canceled:
                message += " (cancel requested)"
            return VOCAB_RUNNING, message, {"pid": pid, "job_object": job_name}
        # TOCTOU guard: the wrapper may have finished BETWEEN our exit-record
        # read and the liveness probe — its fsync'd exit_status write strictly
        # precedes its exit, so with the process now gone one re-read is
        # definitive: the record either landed or never will.
        from_record = _status_from_exit_record(paths)
        if from_record is not None:
            return from_record
        # A dead launcher does NOT mean a dead job: the target is a job member
        # that outlives a wrapper-only kill. The Job Object answers this
        # exactly — no cmdline heuristics, no recycled-pid ambiguity, because
        # the name is unique to this submit.
        members = _job_process_ids(job_name) if job_name else []
        survivors = [p for p in members if p != pid] if members else []
        if survivors:
            message = (
                f"Launcher {pid} died but the job object still has live members "
                f"(pids {survivors})"
            )
            if canceled:
                message += " (cancel requested)"
            return VOCAB_RUNNING, message, {"pid": pid, "orphaned": True,
                                            "job_object": job_name}
        # Ordered deliberately: an explicit cancel is DURABLE INTENT and
        # outranks the ambiguity below. cancel calls TerminateJobObject, which
        # kills the wrapper too, so the job name is always unresolvable
        # afterwards — letting the UNKNOWN branch run first would report every
        # successful cancel as UNKNOWN.
        if canceled:
            return (
                VOCAB_CANCELED,
                "Canceled before a durable exit status was recorded",
                {"pid": pid},
            )
        if members is None:
            # No cancel was requested and the name is gone. That is ambiguous,
            # not terminal: a named job's name is released when the last HANDLE
            # closes, and the wrapper is the only handle holder, so this is
            # equally consistent with "everything exited" and with "the wrapper
            # was killed and members may still be running". ERROR would declare
            # a possibly-live training job failed; COMPLETE would invent an exit
            # code. UNKNOWN is the vocabulary entry for exactly this.
            return (
                VOCAB_UNKNOWN,
                f"Launcher {pid} ended without a durable exit status and the job "
                f"object name is no longer resolvable, so job liveness cannot be "
                f"determined. Members may still be running; check the log and any "
                f"surviving processes before resubmitting",
                {"pid": pid, "job_object": job_name},
            )
        return (
            VOCAB_ERROR,
            "Launcher ended before a durable exit status was recorded",
            {"pid": pid},
        )

    if submit_meta is not None:
        try:
            launch_age = time.time() - float(submit_meta.get("launch_started_at"))
        except (TypeError, ValueError):
            launch_age = PENDING_LAUNCH_GRACE_SECONDS
        if canceled:
            return VOCAB_CANCELED, "Canceled before the launcher started", {}
        if launch_age < PENDING_LAUNCH_GRACE_SECONDS:
            return VOCAB_PENDING, "Waiting for the durable launcher record", {}
        return VOCAB_ERROR, "Job host ended before the launcher started", {}

    return VOCAB_UNKNOWN, f"No runner state under {paths.runner_dir}", {}


def cmd_status(args: argparse.Namespace) -> int:
    refusal = _windows_only()
    if refusal:
        return _fail(refusal, status=VOCAB_UNKNOWN)
    paths = RunnerPaths(Path(args.job_dir).expanduser().resolve())
    status, message, extra = _derive_status(paths)
    return _emit(status=status, message=message, **extra)


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------

def cmd_logs(args: argparse.Namespace) -> int:
    paths = RunnerPaths(Path(args.job_dir).expanduser().resolve())
    try:
        tail = _read_tail(paths.log_path, args.tail)
    except OSError as exc:
        return _fail(f"No readable log at {paths.log_path}: {exc}")
    # Write BYTES, not str. sys.stdout is the ANSI codepage (cp1252 here)
    # whenever stdout is redirected or captured -- which is exactly how an
    # agent invokes this verb. submit forces PYTHONIOENCODING=utf-8 into the
    # child, so the log is always UTF-8, and _read_tail's errors="replace"
    # yields U+FFFD; neither a tqdm block (U+2588) nor U+FFFD is encodable in
    # cp1252, so sys.stdout.write raised UnicodeEncodeError -- not an OSError,
    # so the handler above missed it and the verb died with a traceback and
    # zero bytes on stdout, with the log intact on disk the whole time.
    try:
        sys.stdout.flush()
        sys.stdout.buffer.write(tail.encode("utf-8", "replace"))
        sys.stdout.buffer.flush()
    except AttributeError:
        # No .buffer (pytest capsys, a StringIO): fall back to a str write the
        # active encoding can survive.
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.write(tail.encode(encoding, "replace").decode(encoding, "replace"))
    return 0


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

def cmd_cancel(args: argparse.Namespace) -> int:
    refusal = _windows_only()
    if refusal:
        return _fail(refusal, status=VOCAB_UNKNOWN)
    paths = RunnerPaths(Path(args.job_dir).expanduser().resolve())
    status, message, extra = _derive_status(paths)
    if status in (VOCAB_COMPLETE, VOCAB_ERROR, VOCAB_CANCELED):
        return _emit(result="already_terminal", status=status, message=message, **extra)
    if status == VOCAB_UNKNOWN:
        return _fail(message)

    # Mark first so a wrapper still waiting on the gate self-cancels, then
    # verify identity before terminating anything.
    paths.cancel_marker.touch()
    launcher = _read_json(paths.launcher_status)
    submit_meta = _read_json(paths.submit_meta)
    pid = _coerce_pid((launcher or {}).get("pid"))
    marker = (launcher or {}).get("process_start_marker")
    job_name = (launcher or {}).get("job_object") or (submit_meta or {}).get("job_object")

    signaled = False
    if job_name and _job_process_ids(job_name) is not None:
        # The name is unique to this submit, so opening it IS the identity
        # check. Unconditional: TerminateJobObject has no graceful phase and no
        # SIGTERM analogue, so the tree dies without flushing or checkpointing.
        signaled = _terminate_job(job_name, exit_code=1)
    if not signaled and pid and _process_matches(pid, marker):
        # Cross-session fallback: Local\ names are not visible from another
        # Windows session. The creation-time marker makes this pid-reuse safe.
        signaled = _terminate_tree_by_pid(pid, timeout=args.timeout)

    if signaled:
        deadline = time.monotonic() + max(1.0, args.timeout)
        while time.monotonic() < deadline:
            members = _job_process_ids(job_name) if job_name else None
            if not members and _process_liveness(pid) != "alive":
                break
            time.sleep(0.05)
        else:
            members = _job_process_ids(job_name) if job_name else None
            if members:
                return _fail(
                    f"job object {job_name} survived TerminateJobObject (pids {members})",
                    status=VOCAB_UNKNOWN,
                )

    if not signaled:
        # Nothing was terminated by this cancel: any landed exit record is a
        # NATURAL exit that beat the cancel — undo the cancel claim rather than
        # mislabel it (the one exception: the wrapper's gate self-cancel record,
        # which only exists because of OUR marker).
        deadline = time.monotonic() + 2.0
        exit_record = _read_json(paths.exit_status)
        while exit_record is None and time.monotonic() < deadline:
            time.sleep(0.05)
            exit_record = _read_json(paths.exit_status)
        code = (exit_record or {}).get("return_code")
        if isinstance(code, int) and not isinstance(code, bool):
            self_canceled = (exit_record or {}).get("error") == CANCELED_BEFORE_START
            if not self_canceled:
                paths.cancel_marker.unlink(missing_ok=True)
                status, message, extra = _derive_status(paths)
                return _emit(
                    result="already_terminal", status=status, message=message, **extra
                )

    status, message, extra = _derive_status(paths)
    result = "canceled"
    if not signaled and status in (VOCAB_PENDING, VOCAB_RUNNING):
        # Honest reply: the marker is set (a gated wrapper will self-cancel) but
        # nothing live was terminated — the caller should re-run cancel.
        result = "cancel_requested"
    return _emit(result=result, status=status, message=message, **extra)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)

    submit = sub.add_parser("submit", help="Launch a target in a Windows venv, detached.")
    submit.add_argument("--job-dir", required=True,
                        help="The job's results_dir (bound by tao_job_record open BEFORE submit).")
    submit.add_argument("--venv", required=True,
                        help="Virtualenv root (pyvenv.cfg + Scripts\\python.exe).")
    submit.add_argument("--script", default=None, help="Python script (.py) to run.")
    submit.add_argument("--module", default=None,
                        help="Dotted module to run as `python -m` (e.g. cosmos_framework.scripts.train).")
    submit.add_argument("--entrypoint", default=None,
                        help="Console script installed in the venv's Scripts dir "
                             "(e.g. cosmos-framework-evaluate).")
    submit.add_argument("--arg", action="append", default=[],
                        help="Target argv token; may use {config_path} {results_dir} {job_id}.")
    submit.add_argument("--job-id", default=None, help="Job-record id (sets TAO_JOB_ID).")
    submit.add_argument("--config-path", default=None,
                        help="Spec file the agent authored; fills {config_path}.")
    submit.add_argument("--cwd", default=None, help="Working directory (default: current).")
    submit.add_argument("--gpus", type=int, default=None,
                        help="GPU count; 0 hides CUDA devices. Values above 1 are refused.")
    submit.add_argument("--gpu-ids", default=None,
                        help="ONE CUDA device id -> CUDA_VISIBLE_DEVICES.")
    submit.add_argument("-e", "--env", action="append", default=[],
                        help="Environment variable NAME to pass through (value from this env).")
    submit.add_argument("--tao-windows-env", action="store_true",
                        help="Inject the verified single-GPU Windows env block plus "
                             "TAO_API_JOB_ID and TAO_API_RESULTS_DIR (both, always).")
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="Derive job status from durable state.")
    status.add_argument("--job-dir", required=True)
    status.set_defaults(func=cmd_status)

    logs = sub.add_parser("logs", help="Bounded tail of the job log.")
    logs.add_argument("--job-dir", required=True)
    logs.add_argument("--tail", type=int, default=200)
    logs.set_defaults(func=cmd_logs)

    cancel = sub.add_parser("cancel", help="Terminate the job's whole Job Object tree.")
    cancel.add_argument("--job-dir", required=True)
    cancel.add_argument("--timeout", type=float, default=5.0)
    cancel.set_defaults(func=cmd_cancel)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
