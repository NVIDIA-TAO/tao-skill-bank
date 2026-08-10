# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rca.tools — the deterministic tools under the RCA agent."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fixture import RCA_THRESHOLD, build_rca_run  # noqa: E402

pytest.importorskip("pyarrow", reason="parquet fixtures need pyarrow")
pytest.importorskip("PIL", reason="view_images needs pillow")

from indexer.run_index import RunIndex  # noqa: E402
from rca.tools import RcaChatTools  # noqa: E402

# Point ids are positional and stable for this fixture: KPI rows come first in
# RCA_KPI order, then the seed pool, then the unused candidates.
C0, U0, N1 = 0, 6, 13


@pytest.fixture
def tools(tmp_path):
    _, rd = build_rca_run(tmp_path)
    return RcaChatTools(RunIndex(str(rd)))


def _copy_inference(rd, src_iter, dst_iter):
    """Give another iteration the same inference output (RCA needs a threshold
    AND per-image scores from the same iteration)."""
    src = rd / src_iter / "inference" / "best_val" / "inference.csv"
    dst = rd / dst_iter / "inference" / "best_val"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "inference.csv").write_text(src.read_text())


# --------------------------------------------------------------------------- #
# threshold resolution — every outcome label depends on it
# --------------------------------------------------------------------------- #

def test_threshold_comes_from_the_best_iteration(tools):
    assert tools.thr == RCA_THRESHOLD
    assert tools.best == "iter1"


def test_falls_back_to_an_iteration_with_both_threshold_and_inference(tmp_path):
    _, rd = build_rca_run(tmp_path, iterations={
        "baseline": {"far_pct": 90.0, "threshold": 0.25},
        "iter1": {"far_pct": 50.0},                       # evaluated, no threshold
    })
    _copy_inference(rd, "iter1", "baseline")
    t = RcaChatTools(RunIndex(str(rd)))
    assert (t.thr, t.best) == (0.25, "baseline")


def test_an_iteration_with_a_threshold_but_no_inference_is_skipped(tmp_path):
    _, rd = build_rca_run(tmp_path, iterations={
        "baseline": {"far_pct": 90.0, "threshold": 0.25},  # threshold, no inference/
        "iter1": {"far_pct": 50.0},                        # inference/, no threshold
    })
    with pytest.raises(RuntimeError, match="threshold and inference"):
        RcaChatTools(RunIndex(str(rd)))


def test_no_threshold_anywhere_fails_loudly(tmp_path):
    _, rd = build_rca_run(tmp_path, iterations={
        "baseline": {"far_pct": 90.0}, "iter1": {"far_pct": 50.0}})
    with pytest.raises(RuntimeError, match="threshold"):
        RcaChatTools(RunIndex(str(rd)))


# --------------------------------------------------------------------------- #
# _load_inference — the derivation the other six tools sit on
# --------------------------------------------------------------------------- #

def test_outcome_classifies_both_failure_modes(tools):
    by_obj = tools.inf.set_index("object_name")["outcome"]
    assert by_obj["C0"] == "false_alarm"      # PASS scored 0.90 > 0.40
    assert by_obj["C5"] == "correct"          # PASS scored 0.10
    assert by_obj["N0"] == "correct"          # defect scored 0.95 -> caught
    assert by_obj["N1"] == "missed_defect"    # defect scored 0.30 -> missed


def test_a_score_exactly_at_the_threshold_is_not_a_failure(tmp_path):
    _, rd = build_rca_run(tmp_path)
    inf = rd / "iter1" / "inference" / "best_val" / "inference.csv"
    import pandas as pd
    df = pd.read_csv(inf)
    df.loc[df["object_name"] == "C5", "siamese_score"] = RCA_THRESHOLD
    df.to_csv(inf, index=False)

    t = RcaChatTools(RunIndex(str(rd)))
    assert t.inf.set_index("object_name")["outcome"]["C5"] == "correct"


def test_margin_sign_convention_flips_by_class(tools):
    m = tools.inf.set_index("object_name")["margin"]
    # PASS wants a LOW score: margin = thr - score (negative once it fails)
    assert m["C0"] == pytest.approx(RCA_THRESHOLD - 0.90)
    assert m["C5"] == pytest.approx(RCA_THRESHOLD - 0.10)
    # a defect wants a HIGH score: margin = score - thr
    assert m["N0"] == pytest.approx(0.95 - RCA_THRESHOLD)
    assert m["N1"] == pytest.approx(0.30 - RCA_THRESHOLD)


def test_rows_are_joined_onto_map_point_ids(tools):
    by_obj = tools.inf.set_index("object_name")["point_id"]
    assert by_obj["C0"] == C0
    assert by_obj["N1"] == N1


def test_metadata_columns_survive_the_join(tools):
    row = tools.inf.set_index("object_name").loc["C0"]
    assert row["comp_type_2"] == "C"
    assert row["boardname"] == "brd"


# --------------------------------------------------------------------------- #
# reconciliation — derived numbers must agree with the run's own record
# --------------------------------------------------------------------------- #

def test_computed_false_alarm_rate_matches_the_recorded_far(tmp_path):
    _, rd = build_rca_run(tmp_path)
    t = RcaChatTools(RunIndex(str(rd)))
    recorded = json.loads((rd / "deft_state.json").read_text())["iterations"]["iter1"]["far_pct"]

    pass_rows = t.inf[t.inf["is_pass"]]
    fp = int((pass_rows["outcome"] == "false_alarm").sum())
    computed = 100.0 * fp / len(pass_rows)
    assert computed == pytest.approx(recorded, abs=100.0 / len(pass_rows))  # within one image


def test_overview_counts_match_the_derived_outcomes(tools):
    o, _ = tools.run_overview()
    assert o["false_alarms"] == int((tools.inf["outcome"] == "false_alarm").sum()) == 6
    assert o["missed_defects"] == int((tools.inf["outcome"] == "missed_defect").sum()) == 2


def test_overview_echoes_state_fields_verbatim(tools):
    o, _ = tools.run_overview()
    assert o["best_iteration"] == "iter1"
    assert o["threshold"] == RCA_THRESHOLD
    assert o["best_far_pct"] == 50.0 and o["kpi_met"] is False
    assert [t["iter"] for t in o["trajectory"]] == ["baseline", "iter1"]


# --------------------------------------------------------------------------- #
# list_failures
# --------------------------------------------------------------------------- #

def test_kind_selects_the_failure_mode(tools):
    fa, _ = tools.list_failures(kind="false_alarm", limit=99)
    md, _ = tools.list_failures(kind="missed_defect", limit=99)
    al, _ = tools.list_failures(kind="all", limit=99)
    assert (fa["total_of_kind"], md["total_of_kind"]) == (6, 2)
    assert al["total_of_kind"] == 8
    assert {r["outcome"] for r in fa["failures"]} == {"false_alarm"}


def test_sort_worst_leads_with_the_most_confident_error(tools):
    r, _ = tools.list_failures(kind="false_alarm", sort="worst", limit=3)
    assert [f["score"] for f in r["failures"]] == [0.90, 0.80, 0.70]


def test_sort_best_leads_with_the_borderline_case(tools):
    # `sort` is advertised in the tool schema, so it has to actually invert.
    r, _ = tools.list_failures(kind="false_alarm", sort="best", limit=3)
    assert [f["score"] for f in r["failures"]] == [0.45, 0.50, 0.60]


def test_sort_inverts_for_missed_defects_too(tools):
    # "worst" flips meaning by kind: a missed defect is worse the LOWER it scored
    worst, _ = tools.list_failures(kind="missed_defect", sort="worst", limit=2)
    best, _ = tools.list_failures(kind="missed_defect", sort="best", limit=2)
    assert [f["score"] for f in worst["failures"]] == [0.20, 0.30]
    assert [f["score"] for f in best["failures"]] == [0.30, 0.20]


def test_limit_caps_rows_without_hiding_the_total(tools):
    r, _ = tools.list_failures(kind="false_alarm", limit=2)
    assert r["shown"] == 2 and r["total_of_kind"] == 6


def test_refs_carry_map_point_ids_for_highlighting(tools):
    r, refs = tools.list_failures(kind="false_alarm", sort="worst", limit=3)
    assert refs == [f["point_id"] for f in r["failures"]]
    assert C0 in refs and all(isinstance(i, int) for i in refs)


def test_each_failure_reports_score_threshold_and_margin(tools):
    r, _ = tools.list_failures(kind="false_alarm", sort="worst", limit=1)
    row = r["failures"][0]
    assert row["threshold"] == round(RCA_THRESHOLD, 4)
    assert row["margin"] == pytest.approx(RCA_THRESHOLD - row["score"], abs=1e-4)


# --------------------------------------------------------------------------- #
# defect_breakdown / failure_by
# --------------------------------------------------------------------------- #

def test_defect_breakdown_counts_caught_versus_missed(tools):
    r, _ = tools.defect_breakdown()
    by = {d["defect_type"]: d for d in r["defects"]}
    assert (by["Missing"]["kpi_count"], by["Missing"]["caught"], by["Missing"]["missed"]) == (2, 1, 1)
    assert (by["Shift"]["kpi_count"], by["Shift"]["missed"]) == (1, 1)


def test_defect_breakdown_reports_the_tightest_margin(tools):
    r, _ = tools.defect_breakdown()
    by = {d["defect_type"]: d for d in r["defects"]}
    # Missing spans 0.95 and 0.30; the tightest is the missed one at 0.30-0.40
    assert by["Missing"]["min_margin"] == pytest.approx(0.30 - RCA_THRESHOLD, abs=1e-4)


def test_failure_by_reports_rate_beside_sample_size(tools):
    r, _ = tools.failure_by(column="comp_type_2", kind="false_alarm")
    groups = {g["comp_type_2"]: g for g in r["groups"]}
    assert groups["C"]["n"] == 6 and groups["C"]["failures"] == 5
    assert groups["C"]["rate_pct"] == 83.3
    assert groups["U"]["rate_pct"] == 16.7


def test_failure_by_orders_worst_first(tools):
    r, _ = tools.failure_by(column="comp_type_2", kind="false_alarm")
    rates = [g["rate_pct"] for g in r["groups"]]
    assert rates == sorted(rates, reverse=True)


def test_failure_by_rejects_a_column_outside_the_metadata_set(tools):
    r, _ = tools.failure_by(column="siamese_score")
    assert "error" in r


# --------------------------------------------------------------------------- #
# coverage_census — the data-gap vs hard-case decision
# --------------------------------------------------------------------------- #

def test_many_similar_images_already_in_training_means_tune_not_collect(tools):
    # C0 sits on the same vector as all six seed pool images.
    r, refs = tools.coverage_census(point_id=C0)
    assert r["in_training"] == 6 and r["unused_pool"] == 0
    assert r["route_hint"] == "hard_case_or_tune"
    assert refs


def test_similar_images_sitting_unused_in_the_pool_means_mine(tools):
    # N1 shares a vector with the three never-kept mining candidates.
    r, _ = tools.coverage_census(point_id=N1)
    assert r["in_training"] == 0 and r["unused_pool"] == 3
    assert r["route_hint"] == "mine_unused_pool"


def test_nothing_similar_anywhere_means_generate_or_collect(tools):
    r, _ = tools.coverage_census(point_id=U0)
    assert (r["in_training"], r["unused_pool"]) == (0, 0)
    assert r["route_hint"] == "generate_or_collect"


def test_census_can_be_asked_about_a_whole_defect_class(tools):
    r, _ = tools.coverage_census(defect_type="Missing")
    assert r["n_query_points"] == 2 and "route_hint" in r


def test_census_on_an_unknown_defect_class_reports_an_error(tools):
    r, _ = tools.coverage_census(defect_type="NoSuchDefect")
    assert "error" in r


def test_neighbours_are_reported_with_their_cosine_and_provenance(tools):
    r, _ = tools.coverage_census(point_id=C0)
    n = r["training_neighbors"][0]
    assert {"point_id", "cosine", "provenance", "object_name"} <= set(n)
    assert n["cosine"] == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# view_images
# --------------------------------------------------------------------------- #

def test_returns_base64_crops_for_valid_point_ids(tools):
    r, _ = tools.view_images(images=[C0, U0])
    assert len(r["images"]) == 2
    assert all(im["b64"] for im in r["images"])


def test_out_of_range_and_unparseable_ids_are_skipped(tools):
    r, _ = tools.view_images(images=[C0, 99999, "not-an-id", None])
    assert len(r["images"]) == 1


def test_at_most_six_images_are_returned(tools):
    r, _ = tools.view_images(images=list(range(12)))
    assert len(r["images"]) <= 6


def test_inference_joins_on_the_indexs_resolved_images_dir(tmp_path):
    _, rd = build_rca_run(tmp_path)
    idx = RunIndex(str(rd))
    t = RcaChatTools(idx)
    assert t.inf["point_id"].notna().all()
    assert len(t.inf) == 15                       # every KPI row joined
