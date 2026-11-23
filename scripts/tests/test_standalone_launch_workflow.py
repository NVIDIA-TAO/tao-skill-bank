# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for shared helpers in standalone skill installs."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "skills/core/tao-setup/SKILL.md"
LAUNCH = ROOT / "skills/core/tao-launch-workflow/SKILL.md"
PREFLIGHT = ROOT / "skills/core/tao-launch-workflow/references/platform-preflight.md"


def _bank_check(path):
    text = path.read_text(encoding="utf-8")
    return re.search(r"```bash\n(export TAO_SKILL_BANK_PATH=.*?\ndone)", text, re.S).group(1)


def _make_bank(root, snippet):
    items = re.search(r"for item in (.*); do", snippet).group(1).split()
    for item in items:
        path = root / item
        if item in {"skills/models", "skills/platform"}:
            path.mkdir(parents=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


@pytest.mark.parametrize("path", [SETUP, PREFLIGHT])
def test_standalone_check_accepts_real_checkout(path):
    """The documented prerequisite paths must exist in the shipped bank."""
    result = subprocess.run(
        ["bash", "-c", _bank_check(path)],
        env={**os.environ, "TAO_SKILL_BANK_PATH": str(ROOT)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", [SETUP, PREFLIGHT])
def test_standalone_check_accepts_complete_bank(path, tmp_path):
    """Both entry points accept a checkout containing the shared helpers."""
    snippet = _bank_check(path)
    bank = tmp_path / "tao-skill-bank"
    _make_bank(bank, snippet)
    env = {**os.environ, "HOME": str(tmp_path)}
    env.pop("TAO_SKILL_BANK_PATH", None)

    result = subprocess.run(["bash", "-c", snippet], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", [SETUP, PREFLIGHT])
def test_standalone_check_rejects_incomplete_bank(path, tmp_path):
    """An individual skill copy cannot masquerade as the full bank."""
    snippet = _bank_check(path)
    bank = tmp_path / "individual-skill"
    _make_bank(bank, snippet)
    (bank / "scripts/tao_job_record.py").unlink()

    result = subprocess.run(
        ["bash", "-c", snippet],
        env={**os.environ, "TAO_SKILL_BANK_PATH": str(bank)},
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "missing scripts/tao_job_record.py" in result.stderr


@pytest.mark.parametrize("path", [SETUP, PREFLIGHT])
def test_standalone_check_rejects_missing_checkout(path, tmp_path):
    """An unset bank path cannot silently proceed without a checkout."""
    env = {**os.environ, "HOME": str(tmp_path)}
    env.pop("TAO_SKILL_BANK_PATH", None)
    result = subprocess.run(
        ["bash", "-c", _bank_check(path)], env=env,
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Incomplete TAO skill bank" in result.stderr


def _bash_block(path, first_line):
    """The whole fenced bash block of ``path`` that starts with ``first_line``."""
    text = path.read_text(encoding="utf-8")
    return re.search(r"```bash\n(" + re.escape(first_line) + r".*?)\n```", text, re.S).group(1)


def _python3_stub(folder, importable):
    """A ``python3`` on PATH that can, or cannot, import PyYAML."""
    folder.mkdir()
    flags = "" if importable else " -I -S"   # -S: no site-packages, so no PyYAML
    stub = folder / "python3"
    stub.write_text(f'#!/bin/sh\nexec "{sys.executable}"{flags} "$@"\n')
    stub.chmod(0o755)
    probe = subprocess.run([str(stub), "-c", "import yaml"], capture_output=True)
    assert (probe.returncode == 0) == importable, probe.stderr
    return {**os.environ, "PATH": f"{folder}{os.pathsep}{os.environ['PATH']}"}


PYYAML_INSTALL = "python3 -m pip install pyyaml"


@pytest.mark.parametrize("importable", [True, False])
def test_setup_check_reports_whether_python3_can_import_pyyaml(importable, tmp_path):
    """Seven shared helpers import yaml, so setup must not call a host ready without it."""
    block = _bash_block(SETUP, "export TAO_SKILL_BANK_PATH=")
    lines = [line for line in block.splitlines() if "python3 -c 'import yaml'" in line]
    assert len(lines) == 1, lines
    result = subprocess.run(["bash", "-c", lines[0]], env=_python3_stub(tmp_path / "bin", importable),
                            capture_output=True, text=True)
    report = result.stdout.splitlines()
    if importable:
        assert report == ["OK: PyYAML"], result
    else:
        assert len(report) == 1 and report[0].startswith("MISSING: PyYAML"), result
        assert PYYAML_INSTALL in report[0]
    setup = SETUP.read_text(encoding="utf-8")
    compatibility = re.search(r"(?m)^compatibility: (.*)$", setup).group(1)
    assert "PyYAML" in compatibility
    assert "`MISSING: PyYAML`" in setup


@pytest.mark.parametrize("importable", [True, False])
def test_preflight_bank_check_stops_when_python3_cannot_import_pyyaml(importable, tmp_path):
    """The launch preflight helper imports yaml; the check before it must say so first."""
    block = _bash_block(PREFLIGHT, "export TAO_SKILL_BANK_PATH=")
    assert block.count("python3 -c 'import yaml'") == 1
    result = subprocess.run(
        ["bash", "-c", block + "\necho REACHED"],
        env={**_python3_stub(tmp_path / "bin", importable), "TAO_SKILL_BANK_PATH": str(ROOT)},
        capture_output=True, text=True,
    )
    if importable:
        assert result.returncode == 0 and result.stdout == "REACHED\n", result
        assert result.stderr == ""
    else:
        assert result.returncode != 0 and "REACHED" not in result.stdout, result
        message = result.stderr.splitlines()
        assert len(message) == 1 and message[0].startswith("MISSING: PyYAML"), result
        assert PYYAML_INSTALL in message[0]


@pytest.mark.parametrize("path", [SETUP, LAUNCH])
def test_card_pack_route_needs_the_pi_harness_and_a_request(path):
    """Pack drivers stop without pi, so no session is told to prefer one unasked."""
    text = " ".join(path.read_text(encoding="utf-8").split())
    assert "the `pi` harness is installed" in text
    assert "the user asked for the card-pack route" in text
    assert not re.search(r"\bprefer (?:its `driver\.sh`|the pack's|the token-efficient path)", text)


def test_packaged_clip_image_resolves_from_skill_info():
    """Image lookup uses the packaged model contract, not config.json."""
    launch = LAUNCH.read_text(encoding="utf-8")
    assert "references/skill_info.yaml" in launch
    assert "skills/models/<network>/config.json" not in launch
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/resolve_tao_image.py"),
         "--skill-bank", str(ROOT), "--model", "tao-finetune-clip",
         "--action", "inference", "--format", "json"],
        capture_output=True, text=True, check=True,
    )
    metadata = yaml.safe_load((ROOT / "skills/models/tao-finetune-clip/references/skill_info.yaml").read_text())
    assert json.loads(result.stdout)["image"] == metadata["container_image"]


@pytest.mark.parametrize("output,code,launches", [
    ("job-123", 0, True), ("", 0, False), ("", 2, False),
])
def test_job_record_must_precede_launch(tmp_path, output, code, launches):
    """The documented submit guard stops on failure or an empty ID."""
    launch = LAUNCH.read_text(encoding="utf-8")
    start = launch.index("  if ! JOB_ID=$(")
    end = launch.index("  # <native launch", start)
    guard = re.sub(r"<[^>]+>", "value", launch[start:end])
    bank = tmp_path / "bank"
    helper = bank / "scripts/tao_job_record.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(f"#!/bin/sh\nprintf '%s' '{output}'\nexit {code}\n")
    helper.chmod(0o755)
    script = f'BANK="{bank}"\n{guard}\nprintf "LAUNCHED:%s\\n" "$JOB_ID"\n'

    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert ("LAUNCHED:job-123" in result.stdout) == launches
    assert (result.returncode == 0) == launches
