# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "skills" / "applications" / "tao-run-deft-iaa" / "scripts"
    / "manage_local_airflow.py"
)
SPEC = importlib.util.spec_from_file_location("iaa_manage_local_airflow", SCRIPT)
assert SPEC and SPEC.loader
AIRFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AIRFLOW)


def test_local_service_binds_execution_api_to_selected_port(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "service"
    shared = tmp_path / "shared"
    environment = AIRFLOW._environment(root, shared, 18081)
    assert environment["AIRFLOW__API__HOST"] == "127.0.0.1"
    assert environment["AIRFLOW__API__PORT"] == "18081"
    assert environment["AIRFLOW__CORE__EXECUTION_API_SERVER_URL"] == (
        "http://127.0.0.1:18081/execution/"
    )


def test_deploy_digest_binds_dag_and_orchestration_runtimes(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "service"
    root.mkdir()

    result = AIRFLOW.deploy(SimpleNamespace(root=root))

    assert result["status"] == "ok"
    assert set(result["files"]) == {
        "tao_deft_iaa_action_v1.py",
        "airflow_dag_runtime.py",
        "airflow_orchestrator.py",
    }
    for digest in result["files"].values():
        assert len(digest) == 64


def test_local_service_parser_has_concise_bounded_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as help_exit:
        AIRFLOW._parser().parse_args(["status", "--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert len(help_text) < 2_000
    assert "65534" not in help_text

    with pytest.raises(SystemExit) as invalid_exit:
        AIRFLOW._parser().parse_args(["status", "--root", "/tmp/service", "--port", "1"])
    assert invalid_exit.value.code == 2
    assert "port must be in [1024, 65535]" in capsys.readouterr().err
