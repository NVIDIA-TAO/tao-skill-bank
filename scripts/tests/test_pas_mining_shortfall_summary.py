# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PAS_SCRIPTS = (
    REPO_ROOT / "skills" / "applications" / "tao-run-deft-pas" / "scripts"
)
sys.path.insert(0, str(PAS_SCRIPTS))

import render_deft_report  # noqa: E402
import run_pas_stage  # noqa: E402
from pas_deft import data_mining  # noqa: E402


def _write_iteration_summary(
    tmp_path: Path,
    stats: dict[str, object],
) -> tuple[dict[str, object], Path]:
    iteration = tmp_path / "iter_5"
    mining = iteration / "mining"
    mining.mkdir(parents=True)
    stats_path = mining / "mined_stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    output = Path(
        data_mining.write_iteration_summary(
            experiment_dir=str(iteration),
            iter_num=5,
            gaps_parquet=str(iteration / "gaps" / "kpi_gaps.parquet"),
            mined_parquet=str(mining / "mined_samples.parquet"),
            mined_pairs_file=str(mining / "mined_pairs.json"),
            training_checkpoint="/results/iter_4/train/best/model.pth",
            next_checkpoint_path="/results/iter_5/train/best/model.pth",
            metric={
                "schema_version": "1",
                "iter_label": "iter5",
                "metric_name": "Rank-1",
                "query_type": "medium",
                "op": ">=",
                "target": None,
                "value": 0.5,
            },
            mining_stats_file=str(stats_path),
        )
    )
    return json.loads(output.read_text(encoding="utf-8")), stats_path


def test_iteration_summary_records_history_aware_shortfall(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    summary, stats_path = _write_iteration_summary(
        tmp_path,
        {
            "target_query_count": 10_000,
            "selected_count": 5_419,
            "selection_shortfall": 4_581,
            "mode": "novel_then_fill",
        },
    )

    assert summary["mining"] == {
        "target_query_count": 10_000,
        "selected_count": 5_419,
        "shortfall": 4_581,
        "mode": "novel_then_fill",
        "stats_file": str(stats_path),
    }
    assert (
        "WARNING: iteration 5 mined 5419 of 10000 requested samples "
        "(4581 short)"
    ) in capsys.readouterr().out


def test_iteration_summary_supports_non_history_mining_stats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    summary, stats_path = _write_iteration_summary(
        tmp_path,
        {
            "target_query_count": 128,
            "final_unique_basenames": 128,
            "target_pair_shortfall": 0,
        },
    )

    assert summary["mining"] == {
        "target_query_count": 128,
        "selected_count": 128,
        "shortfall": 0,
        "mode": "",
        "stats_file": str(stats_path),
    }
    assert "requested samples" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "stats, missing_field",
    [
        ({"selected_count": 5, "selection_shortfall": 0}, "target_query_count"),
        ({"target_query_count": 5, "selection_shortfall": 0}, "selected_count"),
        ({"target_query_count": 5, "selected_count": 5}, "selection_shortfall"),
        (
            {
                "target_query_count": 5,
                "selected_count": True,
                "selection_shortfall": 0,
            },
            "selected_count",
        ),
    ],
)
def test_iteration_summary_rejects_incomplete_or_invalid_mining_stats(
    tmp_path: Path,
    stats: dict[str, object],
    missing_field: str,
):
    with pytest.raises(ValueError, match=missing_field):
        _write_iteration_summary(tmp_path, stats)


def test_iteration_summary_stage_passes_canonical_stats_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    iteration = tmp_path / "iter_3"
    iteration.mkdir()
    evaluate = iteration / "evaluate"
    evaluate.mkdir()
    (evaluate / "metric_result.json").write_text(
        json.dumps({"iter_label": "iter3", "value": 0.5}),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    def fake_write_iteration_summary(**kwargs):
        observed.update(kwargs)
        output = iteration / "iteration_summary.json"
        output.write_text("{}\n", encoding="utf-8")
        return str(output)

    monkeypatch.setattr(run_pas_stage, "_results", lambda _: tmp_path)
    monkeypatch.setattr(
        run_pas_stage,
        "_config",
        lambda *_: SimpleNamespace(),
    )
    monkeypatch.setattr(run_pas_stage, "_iter_dir", lambda *_: iteration)
    monkeypatch.setattr(run_pas_stage, "_state", lambda *_: {})
    monkeypatch.setattr(
        run_pas_stage,
        "relative_metric_summary",
        lambda *_args, **_kwargs: {"iter_label": "iter3", "value": 0.5},
    )
    monkeypatch.setattr(
        run_pas_stage,
        "_training_checkpoint",
        lambda *_: "/results/iter_2/train/best/model.pth",
    )
    monkeypatch.setattr(
        data_mining,
        "write_iteration_summary",
        fake_write_iteration_summary,
    )

    run_pas_stage.iteration_summary(
        argparse.Namespace(
            results_dir=tmp_path,
            deft_config=tmp_path / "config" / "deft_config.yaml",
            iter_num=3,
        )
    )

    assert observed["mining_stats_file"] == str(
        iteration / "mining" / "mined_stats.json"
    )
    assert observed["metric"] == {"iter_label": "iter3", "value": 0.5}


def test_loop_report_surfaces_mining_shortfall(tmp_path: Path):
    iteration = tmp_path / "iter_5"
    iteration.mkdir()
    (iteration / "iteration_summary.json").write_text(
        json.dumps(
            {
                "mining": {
                    "target_query_count": 10_000,
                    "selected_count": 5_419,
                    "shortfall": 4_581,
                    "mode": "novel_then_fill",
                }
            }
        ),
        encoding="utf-8",
    )
    state = {
        "workflow": "tao-run-deft-pas",
        "started_at": "2026-10-08T00:00:00+00:00",
        "max_iterations": 5,
        "current_iteration": 5,
        "loop_stop_reason": None,
        "metric_contract": {
            "metric_name": "Rank-1",
            "query_type": "medium",
            "op": ">=",
            "target": None,
        },
        "config": {},
        "iterations": {
            "iter5": {
                "status": "complete",
                "stage_completed": "evaluate",
            }
        },
    }

    document, _, _ = render_deft_report._render_html(  # noqa: SLF001
        results_dir=tmp_path,
        trigger="iteration-complete",
        state=state,
        entries=[],
        audit_report={
            "status": "IN_PROGRESS",
            "terminal": False,
            "next_action": "iter5/evaluate",
            "warnings": [],
        },
    )

    assert "Mining selected / requested" in document
    assert "5419 / 10000" in document
    assert "4581 short" in document
    assert "iter5 mined 5419 of 10000 requested samples (4581 short)." in document
    assert "novel_then_fill" in document
