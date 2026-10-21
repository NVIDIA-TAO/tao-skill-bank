#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the native Windows DLL bootstrap generator."""

from __future__ import annotations

import importlib.util
import pathlib
import sys


SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "configure_windows_arm64_dll_paths",
        SCRIPTS / "configure_windows_arm64_dll_paths.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_configure_writes_stable_one_line_pth(tmp_path: pathlib.Path, monkeypatch) -> None:
    module = _load_module()
    venv = tmp_path / "venv"
    torch_lib = venv / "Lib" / "site-packages" / "torch" / "lib"
    gstreamer = tmp_path / "gstreamer" / "bin"
    torch_lib.mkdir(parents=True)
    gstreamer.mkdir(parents=True)

    monkeypatch.setattr(
        module,
        "candidate_directories",
        lambda configured_venv, extra: module._unique_existing([torch_lib, gstreamer, *extra]),
    )
    output, directories = module.configure(
        venv=venv, extra=[], force=False, dry_run=False
    )

    assert directories == [torch_lib.resolve(), gstreamer.resolve()]
    content = output.read_text(encoding="utf-8")
    assert content.count("\n") == 1
    assert "os.add_dll_directory" in content
    assert "_win_tao_handles" in content
    assert repr(str(torch_lib.resolve())) in content
    assert repr(str(gstreamer.resolve())) in content


def test_configure_refuses_different_existing_pth(tmp_path: pathlib.Path, monkeypatch) -> None:
    module = _load_module()
    venv = tmp_path / "venv"
    torch_lib = venv / "Lib" / "site-packages" / "torch" / "lib"
    gstreamer = tmp_path / "gstreamer" / "bin"
    torch_lib.mkdir(parents=True)
    gstreamer.mkdir(parents=True)
    output = venv / "Lib" / "site-packages" / module.PTH_FILENAME
    output.write_text("different\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "candidate_directories",
        lambda configured_venv, extra: [torch_lib.resolve(), gstreamer.resolve()],
    )

    try:
        module.configure(venv=venv, extra=[], force=False, dry_run=False)
    except ValueError as exc:
        assert "refusing to replace" in str(exc)
    else:
        raise AssertionError("different bootstrap must not be overwritten")
