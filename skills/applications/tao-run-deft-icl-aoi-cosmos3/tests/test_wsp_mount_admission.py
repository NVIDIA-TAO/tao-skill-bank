# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise generated mounts through the actual submission validator, no Docker."""
import json
from types import SimpleNamespace
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import wsp_runtime as runtime


class Backend:
    image = 'synthetic-pinned-image'
    def __init__(self, root): self.root, self.calls = root, []
    def record(self, *args):
        self.calls.append(args)
        if args[0] == 'open': return 'synthetic-mount-job'
        if args[0] == 'show': return json.dumps({'results_dir': str(self.root / 'job')})
        return ''


def fake_submission(tmp_path, monkeypatch):
    backend = Backend(tmp_path)
    monkeypatch.setattr(runtime.os, 'getuid', lambda: 1000)
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(runtime.subprocess, 'check_output', lambda *a, **k: 'synthetic-container')
    return backend


def test_generated_stop_mounts_pass_submission_validator(tmp_path, monkeypatch):
    backend = fake_submission(tmp_path, monkeypatch)
    development = tmp_path / 'development'; development.mkdir()
    for name in ('contract', 'capability', 'control', 'provider', 'auth'):
        (tmp_path / name).write_text('synthetic fixture')
    binary = tmp_path / 'bin'; binary.mkdir()
    for name in ('codex', 'codex-code-mode-host', 'codex-exec-server'):
        (binary / name).write_text('synthetic binary fixture')
    mounts = runtime.sandbox_mounts(development, tmp_path / 'contract', tmp_path / 'capability',
        control_socket=tmp_path / 'control', provider_socket=tmp_path / 'provider',
        codex_binary=binary / 'codex', auth_file=tmp_path / 'auth', model_context=development)
    opened = []
    result = runtime.submit_sandbox(backend, 'synthetic-probe', ['probe'], mounts, on_open=opened.append)
    assert result['job_id'] == 'synthetic-mount-job' and opened == ['synthetic-mount-job']
    command = json.loads((tmp_path / 'job/launch.json').read_text())['argv']
    assert any('dst=/runtime/wsp_stop.py,readonly' in arg for arg in command)
    assert any('dst=/channel/control.sock,readonly' in arg for arg in command)


def test_rejected_mount_still_records_minted_job_before_failure(tmp_path, monkeypatch):
    backend = fake_submission(tmp_path, monkeypatch)
    opened = []
    with pytest.raises(ValueError, match='sandbox mount rejected'):
        runtime.submit_sandbox(backend, 'synthetic-rejected', ['probe'],
            [(tmp_path, '/forbidden-target', True)], on_open=opened.append)
    assert opened == ['synthetic-mount-job']
    assert any(call[0] == 'mark' and 'ERROR' in call for call in backend.calls)
