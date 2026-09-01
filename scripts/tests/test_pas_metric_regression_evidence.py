# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for durable PAS per-round metric comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PAS_ROOT = REPO_ROOT / "skills" / "applications" / "tao-run-deft-pas"
PAS_SCRIPTS = PAS_ROOT / "scripts"
sys.path.insert(0, str(PAS_SCRIPTS))

import audit_deft_run  # noqa: E402
import commit_stage  # noqa: E402
import init_deft_state  # noqa: E402
import metric_contract  # noqa: E402
import parse_pas_metrics  # noqa: E402
import prepare_deft_config  # noqa: E402
import record_metric_result  # noqa: E402
import render_deft_report  # noqa: E402
import run_pas_stage  # noqa: E402


def _result(label: str, value: float, metric_name: str = "Rank-1") -> dict:
    return {
        "iter_label": label,
        "metric_name": metric_name,
        "query_type": "medium",
        "value": value,
    }


@pytest.mark.parametrize(
    ("metric_name", "operator", "baseline", "current", "outcome"),
    [
        ("Rank-1", ">=", 0.1472306583790525, 0.1349624854096809, "regressed"),
        ("Rank-1", ">=", 0.25, 0.30, "improved"),
        ("Zero@5", "<=", 0.10, 0.20, "regressed"),
        ("Zero@5", "<=", 0.10, 0.05, "improved"),
        ("Rank-1", ">=", 0.25, 0.25, "unchanged"),
    ],
)
def test_relative_metric_summary_follows_approved_operator(
    metric_name, operator, baseline, current, outcome
):
    state = {
        "metric_contract": {
            "metric_name": metric_name,
            "query_type": "medium",
            "op": operator,
            "target": None,
        },
        "iterations": {
            "baseline": {
                "metric_result": _result("baseline", baseline, metric_name)
            },
            "iter1": {"metric_result": _result("iter1", current, metric_name)},
        },
    }

    summary = metric_contract.relative_metric_summary(state, "iter1")

    assert summary["value"] == pytest.approx(current)
    assert summary["baseline_value"] == pytest.approx(baseline)
    assert summary["previous_label"] == "baseline"
    assert summary["previous_value"] == pytest.approx(baseline)
    assert summary["delta_from_baseline"] == pytest.approx(current - baseline)
    assert summary["delta_from_previous"] == pytest.approx(current - baseline)
    assert summary["comparison_to_baseline"] == outcome
    assert summary["comparison_to_previous"] == outcome


def test_relative_metric_summary_distinguishes_baseline_and_previous_round():
    state = {
        "metric_contract": {
            "metric_name": "Rank-1",
            "query_type": "medium",
            "op": ">=",
            "target": None,
        },
        "iterations": {
            "baseline": {"metric_result": _result("baseline", 0.2)},
            "iter1": {"metric_result": _result("iter1", 0.1)},
            "iter2": {"metric_result": _result("iter2", 0.15)},
        },
    }

    summary = metric_contract.relative_metric_summary(state, "iter2")

    assert summary["delta_from_baseline"] == pytest.approx(-0.05)
    assert summary["comparison_to_baseline"] == "regressed"
    assert summary["previous_label"] == "iter1"
    assert summary["delta_from_previous"] == pytest.approx(0.05)
    assert summary["comparison_to_previous"] == "improved"


def test_relative_metric_summary_treats_float_noise_as_unchanged():
    baseline = 0.25
    current = baseline + 1e-14
    state = {
        "metric_contract": {
            "metric_name": "Rank-1",
            "query_type": "medium",
            "op": ">=",
            "target": None,
        },
        "iterations": {
            "baseline": {"metric_result": _result("baseline", baseline)},
            "iter1": {"metric_result": _result("iter1", current)},
        },
    }

    summary = metric_contract.relative_metric_summary(state, "iter1")

    assert summary["delta_from_baseline"] == pytest.approx(current - baseline)
    assert summary["comparison_to_baseline"] == "unchanged"
    assert summary["comparison_to_previous"] == "unchanged"


def test_baseline_summary_has_no_self_comparison():
    state = {
        "metric_contract": {
            "metric_name": "Rank-1",
            "query_type": "medium",
            "op": ">=",
            "target": None,
        },
        "iterations": {
            "baseline": {"metric_result": _result("baseline", 0.25)},
        },
    }

    summary = metric_contract.relative_metric_summary(state, "baseline")

    assert summary["value"] == pytest.approx(0.25)
    for field in (
        "baseline_value",
        "delta_from_baseline",
        "comparison_to_baseline",
        "previous_label",
        "previous_value",
        "delta_from_previous",
        "comparison_to_previous",
    ):
        assert summary[field] is None


def test_metric_evidence_versions_fail_closed():
    assert metric_contract.relative_evidence_required({}) is False
    assert (
        metric_contract.relative_evidence_required(
            {"metric_evidence_version": "1"}
        )
        is True
    )
    with pytest.raises(ValueError, match="unsupported metric_evidence_version"):
        metric_contract.relative_evidence_required(
            {"metric_evidence_version": "2"}
        )


def test_report_rows_tolerate_missing_previous_metric():
    state = {
        "metric_contract": {
            "metric_name": "Rank-1",
            "query_type": "medium",
            "op": ">=",
            "target": None,
        },
        "iterations": {
            "baseline": {"metric_result": _result("baseline", 0.2)},
            "iter2": {"metric_result": _result("iter2", 0.15)},
        },
    }
    entries = [
        {"iteration": "baseline", "stage": "evaluate", "status": "ok"},
        {"iteration": "iter2", "stage": "evaluate", "status": "ok"},
    ]

    rows, best = render_deft_report._metric_rows(  # noqa: SLF001
        state,
        entries,
        state["metric_contract"],
    )

    assert best == "baseline"
    assert [row["label"] for row in rows] == ["baseline", "iter2"]
    assert rows[1]["value"] == pytest.approx(0.15)
    assert rows[1]["delta_from_baseline"] is None
    assert rows[1]["delta_from_previous"] is None


def test_audit_text_surfaces_the_round_regression_without_html(capsys):
    state = {
        "metric_contract": {
            "metric_name": "Rank-1",
            "query_type": "medium",
            "op": ">=",
            "target": None,
        },
        "iterations": {
            "baseline": {
                "metric_result": _result("baseline", 0.1472306583790525)
            },
            "iter1": {"metric_result": _result("iter1", 0.1349624854096809)},
        },
    }
    baseline = metric_contract.relative_metric_summary(state, "baseline")
    iteration = metric_contract.relative_metric_summary(state, "iter1")
    audit_deft_run._print_text(  # noqa: SLF001
        {
            "status": "COMPLETE",
            "results_dir": "/results/run",
            "current_iteration": 1,
            "max_iterations": 1,
            "log_entries": 9,
            "gate_met": False,
            "last_committed": {
                "seq": 9,
                "iteration": "iter1",
                "stage": "loop_stop",
                "status": "ok",
            },
            "metric_gate": state["metric_contract"],
            "best_iteration": "baseline",
            "best_metric_result": state["iterations"]["baseline"][
                "metric_result"
            ],
            "metric_results": [baseline, iteration],
            "next_action": "render report",
            "required_reference": None,
            "warnings": [],
            "errors": [],
        }
    )

    output = capsys.readouterr().out
    assert "metric_result=iter1 value=0.134962" in output
    assert "delta_baseline=-0.0122682 (regressed)" in output
    assert "delta_previous=-0.0122682 (regressed)" in output


def _write_metric_result(
    results: Path,
    label: str,
    value: float,
) -> tuple[Path, Path]:
    phase = results / ("zs" if label == "baseline" else f"iter_{label[4:]}")
    evaluate = phase / "evaluate"
    evaluate.mkdir(parents=True, exist_ok=True)
    csv_path = evaluate / "nvidia_pas_metrics_aggregate.csv"
    csv_path.write_text(f"QueryType,Rank-1\nmedium,{value}\n", encoding="utf-8")
    result = parse_pas_metrics.build_result(
        argparse.Namespace(
            metrics_csv=csv_path,
            metric_name="Rank-1",
            query_type="medium",
            op=">=",
            target=None,
            iter_label=label,
        )
    )
    result_path = evaluate / "metric_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return csv_path, result_path


def _materialize_run(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    results = workspace / "results" / "run"
    dataset = workspace / "data" / "pas_v31_tao_ft"
    dataset.mkdir(parents=True)
    images = tmp_path / "images_raw.tar"
    metadata = tmp_path / "meta.tar.gz"
    images.write_bytes(b"images")
    metadata.write_bytes(b"metadata")
    prepare_args = prepare_deft_config._parser().parse_args(  # noqa: SLF001
        [
            "--workspace",
            str(workspace),
            "--results-dir",
            str(results),
            "--dataset-root",
            str(dataset),
            "--images-archive",
            str(images),
            "--metadata-archive",
            str(metadata),
            "--platform",
            "docker",
            "--max-iterations",
            "1",
        ]
    )
    prepare_deft_config.materialize(prepare_args)
    approval = json.loads((results / "config" / "approval.json").read_text())
    init_argv = [
        "--results-dir",
        str(results),
        "--workspace",
        str(workspace),
        "--dataset-root",
        str(dataset),
        "--images-archive",
        str(images),
        "--metadata-archive",
        str(metadata),
        "--max-iterations",
        "1",
        "--metric-name",
        "Rank-1",
        "--metric-query-type",
        "medium",
        "--metric-op",
        ">=",
        "--platform",
        "docker",
        "--pyt-image",
        approval["pyt_image"],
        "--ds-image",
        approval["ds_image"],
        "--deft-config",
        str(results / "config" / "deft_config.yaml"),
        "--tao-spec",
        str(results / "config" / "tao_spec.yaml"),
    ]
    assert init_deft_state.main(init_argv) == 0
    return results, results / "config" / "deft_config.yaml"


def test_user_facing_iteration_summary_records_regression_without_html(
    tmp_path: Path,
):
    results, config = _materialize_run(tmp_path)
    baseline_csv, baseline_result = _write_metric_result(
        results, "baseline", 0.1472306583790525
    )
    record_metric_result.commit(
        argparse.Namespace(
            results_dir=results,
            iter_label="baseline",
            metric_result=baseline_result,
            metrics_csv=baseline_csv,
        )
    )
    _write_metric_result(results, "iter1", 0.1349624854096809)

    exit_code = run_pas_stage.main(
        [
            "iteration-summary",
            "--results-dir",
            str(results),
            "--deft-config",
            str(config),
            "--iter-num",
            "1",
        ]
    )

    assert exit_code == 0
    assert not (results / "DEFT_Loop_Report.html").exists()
    summary = json.loads((results / "iter_1" / "iteration_summary.json").read_text())
    metric = summary["metric"]
    assert metric["value"] == pytest.approx(0.1349624854096809)
    assert metric["delta_from_baseline"] == pytest.approx(
        0.1349624854096809 - 0.1472306583790525
    )
    assert metric["comparison_to_baseline"] == "regressed"
    assert metric["comparison_to_previous"] == "regressed"


def test_metric_commit_persists_canonical_relative_change(tmp_path: Path):
    results, _ = _materialize_run(tmp_path)
    baseline_csv, baseline_result = _write_metric_result(results, "baseline", 0.2)
    record_metric_result.commit(
        argparse.Namespace(
            results_dir=results,
            iter_label="baseline",
            metric_result=baseline_result,
            metrics_csv=baseline_csv,
        )
    )
    current_csv, current_result = _write_metric_result(results, "iter1", 0.1)
    committed = record_metric_result.commit(
        argparse.Namespace(
            results_dir=results,
            iter_label="iter1",
            metric_result=current_result,
            metrics_csv=current_csv,
        )
    )

    relative = committed["relative_change"]
    assert relative["comparison_to_baseline"] == "regressed"
    assert relative["delta_from_baseline"] == pytest.approx(-0.1)
    state = json.loads((results / "deft_state.json").read_text())
    assert (
        state["iterations"]["iter1"]["metric_result"]["relative_change"]
        == relative
    )

    summary = {"metric": dict(relative)}
    commit_stage._validate_iteration_summary_metric(  # noqa: SLF001
        summary, state, "iter1"
    )
    summary["metric"]["comparison_to_baseline"] = "improved"
    with pytest.raises(ValueError, match="does not match"):
        commit_stage._validate_iteration_summary_metric(  # noqa: SLF001
            summary, state, "iter1"
        )
