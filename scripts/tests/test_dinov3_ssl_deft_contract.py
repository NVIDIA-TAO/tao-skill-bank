# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensure Skill Bank delegates orchestration to the installed DS package."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "applications" / "tao-run-dinov3-ssl-deft"


def test_skill_delegates_without_host_dependencies(monkeypatch):
    path = SKILL / "scripts" / "run_workflow.py"
    spec = importlib.util.spec_from_file_location("deft_shim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module.runpy, "run_module", lambda *a, **k: calls.append((a, k)))
    module.main()
    assert calls == [(("nvidia_tao_ds.mining.dinov3.workflow",), {"run_name": "__main__"})]


def test_controller_is_not_bundled_in_skill_bank():
    assert not list((SKILL / "scripts" / "dinov3_ssl_deft").glob("*.py"))
    assert not list((SKILL / "recipes").glob("*.yaml"))
