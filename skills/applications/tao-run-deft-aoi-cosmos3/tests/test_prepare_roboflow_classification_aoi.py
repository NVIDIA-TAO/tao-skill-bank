#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract test for the public Roboflow PCB classification converter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"


def _load_converter():
    module_path = SCRIPTS / "prepare_roboflow_classification_aoi.py"
    spec = importlib.util.spec_from_file_location("prepare_roboflow", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_source_export(root: pathlib.Path) -> None:
    for split in ("train", "valid", "test"):
        for label in ("good", "defect"):
            for index in (1, 2):
                image = root / split / label / f"{split}-{label}-{index}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"{split}:{label}:{index}".encode("ascii"))


def test_prepare_classification_export(tmp_path: pathlib.Path) -> None:
    converter = _load_converter()
    source = tmp_path / "expanded-export"
    workspace = tmp_path / "workspace"
    _make_source_export(source)

    result = converter.prepare(
        dataset_root=source,
        workspace=workspace,
        prompt="Return exactly OK or NG.",
        provenance_url="https://example.invalid/poc-ongfq/dataset/2",
    )

    assert result["summary"]["reference_policy"] == "demo_pseudo_reference"
    for role, filename in (
        ("mining", "mining_pool.json"),
        ("proxy", "proxy_kpi.json"),
        ("benchmark", "benchmark_kpi.json"),
    ):
        records = json.loads((workspace / "annotations" / filename).read_text())
        assert len(records) == 4
        assert {record["conversations"][-1]["value"] for record in records} == {"OK", "NG"}
        assert all(len(record["images"]) == 2 for record in records)
        assert all(record["images"][0] != record["images"][1] for record in records)
        if role == "mining":
            assert all("id" not in record for record in records)
        else:
            assert len({record["id"] for record in records}) == len(records)

    manifest_path = pathlib.Path(result["manifest"])
    manifest = json.loads(manifest_path.read_text())
    benchmark = workspace / "annotations" / "benchmark_kpi.json"
    assert manifest["reference_policy"]["production_eligible"] is False
    assert manifest["evaluation_contract"]["benchmark"]["annotations_sha256"] == hashlib.sha256(
        benchmark.read_bytes()
    ).hexdigest()

    for script, extra_args in (
        ("check_annotations.py", ["--require-files"]),
        ("validate_split_contract.py", ["--manifest", str(manifest_path)]),
    ):
        completed = subprocess.run(
            [sys.executable, str(SCRIPTS / script), "--workspace", str(workspace), *extra_args],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_prepare_refuses_annotation_overwrite(tmp_path: pathlib.Path) -> None:
    converter = _load_converter()
    source = tmp_path / "expanded-export"
    workspace = tmp_path / "workspace"
    _make_source_export(source)
    (workspace / "annotations").mkdir(parents=True)
    existing = workspace / "annotations" / "mining_pool.json"
    existing.write_text("[]\n", encoding="utf-8")

    try:
        converter.prepare(
            dataset_root=source,
            workspace=workspace,
            prompt="Return exactly OK or NG.",
            provenance_url="https://example.invalid/poc-ongfq/dataset/2",
        )
    except ValueError as exc:
        assert "refusing to overwrite existing annotations" in str(exc)
    else:
        raise AssertionError("existing annotations must prevent conversion")
