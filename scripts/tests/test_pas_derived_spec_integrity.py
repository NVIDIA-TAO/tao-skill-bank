# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for content-bound PAS derived TAO specs."""

import json
import sys
from pathlib import Path

import pytest
import yaml

PAS_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "applications"
    / "tao-run-deft-pas"
    / "scripts"
)
sys.path.insert(0, str(PAS_SCRIPTS))

import audit_deft_run as audit  # noqa: E402
import init_deft_state as state  # noqa: E402
import prepare_deft_config as prepare  # noqa: E402
import run_deft_container as container  # noqa: E402
import run_pas_stage as stage  # noqa: E402


def _initialized_run(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    results = workspace / "results" / "run"
    dataset = workspace / "data" / "pas"
    images_archive = tmp_path / "images_raw.tar"
    metadata_archive = tmp_path / "meta.tar.gz"
    images_archive.write_bytes(b"images")
    metadata_archive.write_bytes(b"metadata")
    common = [
        "--workspace",
        str(workspace),
        "--results-dir",
        str(results),
        "--dataset-root",
        str(dataset),
        "--images-archive",
        str(images_archive),
        "--metadata-archive",
        str(metadata_archive),
        "--max-iterations",
        "1",
    ]
    assert prepare.main(common) == 0
    assert state.main(
        [
            *common,
            "--platform",
            "docker",
            "--pyt-image",
            prepare.PINNED_PYT_IMAGE,
            "--ds-image",
            prepare.PINNED_DS_IMAGE,
            "--deft-config",
            str(results / "config" / "deft_config.yaml"),
            "--tao-spec",
            str(results / "config" / "tao_spec.yaml"),
        ]
    ) == 0
    assert stage.main(
        [
            "eval-config",
            "--results-dir",
            str(results),
            "--deft-config",
            str(results / "config" / "deft_config.yaml"),
            "--iter-label",
            "baseline",
        ]
    ) == 0
    assert stage.main(
        [
            "train-config",
            "--results-dir",
            str(results),
            "--deft-config",
            str(results / "config" / "deft_config.yaml"),
            "--iter-num",
            "1",
        ]
    ) == 0
    return results


def test_generated_eval_spec_is_bound_before_launch_and_during_audit(tmp_path):
    results = _initialized_run(tmp_path)
    spec = results / "zs" / "specs" / "eval_config.yaml"
    status = results / "zs" / "specs" / "eval-config.host.status.json"
    evidence = json.loads(status.read_text())

    assert evidence["fresh_output_sha256"][str(spec)]
    assert container._validate_derived_spec_input(  # noqa: SLF001
        "evaluate", "baseline", results
    ) == evidence["fresh_output_sha256"]
    assert audit.audit(results)["status"] == "IN_PROGRESS"

    payload = yaml.safe_load(spec.read_text())
    payload["evaluate"]["batch_size"] += 1
    spec.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="content hash mismatch"):
        container._validate_derived_spec_input(  # noqa: SLF001
            "evaluate", "baseline", results
        )
    report = audit.audit(results)
    assert report["status"] == "INVALID"
    assert any("content hash mismatch" in error for error in report["errors"])


def test_generated_spec_requires_successful_producer_evidence(tmp_path):
    results = _initialized_run(tmp_path)
    status = results / "zs" / "specs" / "eval-config.host.status.json"
    payload = json.loads(status.read_text())
    payload["status"] = "running"
    payload["exit_code"] = None
    status.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="successful content-bound eval-config"):
        container._validate_derived_spec_input(  # noqa: SLF001
            "evaluate", "baseline", results
        )


def test_generated_train_spec_uses_the_same_content_binding(tmp_path):
    results = _initialized_run(tmp_path)
    spec = results / "iter_1" / "specs" / "train_config.yaml"
    status = results / "iter_1" / "specs" / "train-config.host.status.json"
    evidence = json.loads(status.read_text())

    assert container._validate_derived_spec_input(  # noqa: SLF001
        "train", "iter1", results
    ) == evidence["fresh_output_sha256"]

    spec.write_text(spec.read_text() + "\n# unapproved edit\n")
    with pytest.raises(ValueError, match="content hash mismatch"):
        container._validate_derived_spec_input(  # noqa: SLF001
            "train", "iter1", results
        )
