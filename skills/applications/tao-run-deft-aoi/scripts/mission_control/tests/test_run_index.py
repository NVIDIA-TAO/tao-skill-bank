# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for indexer.run_index — the point model the server builds at startup."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fixture import add_inference, add_mining, build_run  # noqa: E402
from indexer import run_index as ri  # noqa: E402

pytest.importorskip("pyarrow", reason="parquet fixtures need pyarrow")


@pytest.fixture
def run(tmp_path):
    ws, rd = build_run(tmp_path)
    add_mining(ws, rd)
    return ws, rd


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """PATH_MAP is module-level; keep tests independent of each other."""
    ri.PATH_MAP.clear()
    yield
    ri.PATH_MAP.clear()


# --------------------------------------------------------------------------- #
# construction — the hard requirements
# --------------------------------------------------------------------------- #

def test_builds_from_a_complete_run(run):
    _, rd = run
    idx = ri.RunIndex(str(rd))
    assert len(idx.points) == 5
    assert idx.iter_order == ["baseline", "iter1"]


def test_missing_deft_state_is_refused(tmp_path):
    (tmp_path / "not_a_run").mkdir()
    with pytest.raises(RuntimeError, match="deft_state.json"):
        ri.RunIndex(str(tmp_path / "not_a_run"))


def test_missing_embeddings_names_the_build_step(run):
    _, rd = run
    (rd / "mission_control" / "embeddings.parquet").unlink()
    with pytest.raises(RuntimeError, match=r"prepare\.py"):
        ri.RunIndex(str(rd))


def test_iterations_order_numerically_not_lexically(tmp_path):
    ws, rd = build_run(tmp_path, iterations={
        "baseline": {}, "iter1": {}, "iter2": {}, "iter10": {}})
    assert ri.RunIndex(str(rd)).iter_order == ["baseline", "iter1", "iter2", "iter10"]


def test_a_run_moved_out_of_its_workspace_still_resolves(tmp_path, capsys):
    ws, rd = build_run(tmp_path)
    add_mining(ws, rd)
    moved = tmp_path / "elsewhere" / "run_X"
    moved.parent.mkdir()
    rd.rename(moved)

    idx = ri.RunIndex(str(moved))
    assert idx.images_dir == ws / "kpi" / "images"
    assert idx.summary()["counts"]["kpi"] == 2           # KPI scores intact
    assert "no KPI images" not in capsys.readouterr().out


def test_warns_when_neither_the_record_nor_the_layout_resolves(tmp_path, capsys):
    ws, rd = build_run(tmp_path)
    add_mining(ws, rd)
    state = json.loads((rd / "deft_state.json").read_text())
    state["config"].update({k: f"/gone/{k}" for k in
                            ("images_dir", "kpi_test_csv", "training_csv", "validation_csv")})
    moved = tmp_path / "elsewhere" / "run_X"
    moved.parent.mkdir()
    rd.rename(moved)
    (moved / "deft_state.json").write_text(json.dumps(state))

    try:
        ri.RunIndex(str(moved))
    except Exception:                                    # noqa: BLE001
        pass
    assert "no KPI images at" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# _mining_dir — repeated stages resolve by recency
# --------------------------------------------------------------------------- #

def test_single_attempt_is_selected(run):
    _, rd = run
    assert ri.RunIndex(str(rd))._mining_dir("iter1").name == "2026-01-01_120000"


def test_latest_attempt_supersedes_an_earlier_one(run):
    ws, rd = run
    add_mining(ws, rd, ts="2026-01-02_090000")           # a later re-run
    assert ri.RunIndex(str(rd))._mining_dir("iter1").name == "2026-01-02_090000"


def test_latest_wins_even_when_it_has_no_mining_spec(run):
    ws, rd = run
    add_mining(ws, rd, ts="2026-01-03_090000", spec=False)
    assert ri.RunIndex(str(rd))._mining_dir("iter1").name == "2026-01-03_090000"


def test_no_mining_results_is_none_not_an_error(run):
    _, rd = run
    assert ri.RunIndex(str(rd))._mining_dir("baseline") is None


# --------------------------------------------------------------------------- #
# mining_spec — the run's own recipe, with a config fallback
# --------------------------------------------------------------------------- #

def test_spec_is_read_from_the_iterations_own_yaml(run):
    _, rd = run
    spec = ri.RunIndex(str(rd)).mining_spec("iter1")
    assert spec["topn"] == 5
    assert spec["knn_metric"] == "cosine"
    assert spec["filter_by_label"] is False
    assert spec["min_similarity"] == 0.9      # not in the YAML — from config


def test_config_spellings_are_translated_to_the_canonical_keys(tmp_path):
    # the config says top_k_per_target/metric; the YAML and this dict say
    # topn/knn_metric. A rename on either side must not pass silently.
    ws, rd = build_run(tmp_path, config={
        "mining_filter": {"top_k_per_target": 11, "metric": "euclidean",
                          "min_similarity": 0.55}})
    add_mining(ws, rd, spec=False)
    spec = ri.RunIndex(str(rd)).mining_spec("iter1")
    assert spec["topn"] == 11
    assert spec["knn_metric"] == "euclidean"
    assert "top_k_per_target" not in spec and "metric" not in spec


def test_absent_yaml_falls_back_to_run_level_config(tmp_path):
    # Real runs vary: an iteration can carry embeddings but no mining_spec.yaml.
    ws, rd = build_run(tmp_path, config={
        "mining_filter": {"top_k_per_target": 7, "metric": "cosine", "min_similarity": 0.75}})
    add_mining(ws, rd, spec=False)
    spec = ri.RunIndex(str(rd)).mining_spec("iter1")
    assert (spec["topn"], spec["min_similarity"]) == (7, 0.75)


def _partial(tmp_path, yaml_text, **cfg):
    """A run whose YAML and config deliberately disagree, so blending is visible."""
    ws, rd = build_run(tmp_path, config={"mining_filter": dict(
        {"top_k_per_target": 7, "metric": "euclidean", "min_similarity": 0.75}, **cfg)})
    md, _, _ = add_mining(ws, rd)
    (md / "mining_spec.yaml").write_text(yaml_text)
    return ri.RunIndex(str(rd)).mining_spec("iter1")


def test_a_yaml_without_min_similarity_takes_it_from_config(tmp_path):
    # min_similarity is never written to the YAML — the loop filters in Python
    spec = _partial(tmp_path, 'topn: 3\nknn_metric: cosine\n')
    assert spec["min_similarity"] == 0.75          # config, not the 0.9 default
    assert spec["topn"] == 3                       # YAML still wins where it speaks


def test_a_yaml_without_topn_takes_it_from_config(tmp_path):
    spec = _partial(tmp_path, 'knn_metric: cosine\n')
    assert spec["topn"] == 7
    assert spec["knn_metric"] == "cosine"          # YAML overrides config euclidean


def test_a_yaml_min_similarity_overrides_config_when_present(tmp_path):
    spec = _partial(tmp_path, 'topn: 3\nmin_similarity: 0.42\n')
    assert spec["min_similarity"] == 0.42


def test_an_empty_yaml_leaves_every_config_value_standing(tmp_path):
    spec = _partial(tmp_path, "\n")
    assert (spec["topn"], spec["knn_metric"], spec["min_similarity"]) == (7, "euclidean", 0.75)


def test_a_yaml_without_filter_by_label_reads_false_not_a_fallback(tmp_path):
    # filter_by_label has no config counterpart, so absence means False
    spec = _partial(tmp_path, 'topn: 3\n', filter_by_label=True)
    assert spec["filter_by_label"] is False


def test_yaml_filter_by_label_true_is_parsed_from_its_string_form(run):
    ws, rd = run
    md = rd / "iter1" / "mining_results" / "2026-01-01_120000"
    (md / "mining_spec.yaml").write_text(
        'topn: 3\nknn_metric: cosine\nfilter_by_label: "true"\n')
    spec = ri.RunIndex(str(rd)).mining_spec("iter1")
    assert spec["filter_by_label"] is True and spec["topn"] == 3


# --------------------------------------------------------------------------- #
# _kept_keys — every tier must reproduce knn_summary.csv
# --------------------------------------------------------------------------- #

def _expected_kept(rd, iteration="iter1"):
    row = pd.read_csv(rd / iteration / "mining_filter" / "knn_summary.csv").iloc[0]
    return int(row["kept_count"])


def test_recompute_tier_matches_knn_summary(run):
    # No mined.parquet -> recompute from the run's own embeddings at its own
    # recipe. Two of three sources sit exactly on a target (cos 1.0 >= 0.9).
    _, rd = run
    idx = ri.RunIndex(str(rd))
    assert len(idx._kept_keys("iter1")) == _expected_kept(rd) == 2


def test_mined_parquet_tier_is_preferred_when_present(tmp_path):
    ws, rd = build_run(tmp_path)
    md, srcs, _ = add_mining(ws, rd, mined=None, kept_count=1)
    pd.DataFrame({"filepath": srcs[:1]}).to_parquet(md / "mined.parquet", index=False)
    idx = ri.RunIndex(str(rd))
    assert len(idx._kept_keys("iter1")) == _expected_kept(rd) == 1


def test_threshold_is_honoured_when_nothing_clears_the_gate(tmp_path):
    ws, rd = build_run(tmp_path, config={
        "mining_filter": {"top_k_per_target": 5, "metric": "cosine", "min_similarity": 1.01}})
    add_mining(ws, rd, spec=False)                      # force the config threshold
    assert ri.RunIndex(str(rd))._kept_keys("iter1") == set()


def test_iteration_without_mining_artifacts_keeps_nothing(run):
    _, rd = run
    assert ri.RunIndex(str(rd))._kept_keys("baseline") == set()


# --------------------------------------------------------------------------- #
# summary — the payload the UI reads first
# --------------------------------------------------------------------------- #

def test_best_is_the_lowest_far_iteration(run):
    _, rd = run
    s = ri.RunIndex(str(rd)).summary()
    assert s["best"]["label"] == "iter1" and s["best"]["far_pct"] == 10.0


def test_best_keeps_the_full_iteration_shape_when_nothing_is_evaluated(tmp_path):
    ws, rd = build_run(tmp_path, iterations={
        "baseline": {"threshold": 0.3}, "iter1": {"threshold": 0.4}})
    add_mining(ws, rd)
    s = ri.RunIndex(str(rd)).summary()
    assert s["best"]["far_pct"] is None
    assert set(s["best"]) == set(s["iterations"][0])


def test_summary_reports_the_siglip_space_only(run):
    _, rd = run
    s = ri.RunIndex(str(rd)).summary()
    assert s["spaces"] == ["siglip"] and s["mining_encoder"] == "siglip"


def test_summary_counts_split_pool_from_kpi(run):
    _, rd = run
    assert ri.RunIndex(str(rd)).summary()["counts"] == {"pool": 3, "kpi": 2}


# --------------------------------------------------------------------------- #
# serve cache — must round-trip, and must invalidate
# --------------------------------------------------------------------------- #

def test_second_load_comes_from_the_serve_cache(run):
    _, rd = run
    assert ri.RunIndex(str(rd))._lazy is False           # cold: full build
    assert ri.RunIndex(str(rd))._lazy is True            # warm: cached


def test_cached_load_reproduces_the_full_build(run):
    _, rd = run
    cold = ri.RunIndex(str(rd))
    warm = ri.RunIndex(str(rd))
    assert warm.summary()["counts"] == cold.summary()["counts"]
    assert [p["key"] for p in warm.points] == [p["key"] for p in cold.points]


def test_a_new_iteration_invalidates_the_cache(run):
    _, rd = run
    ri.RunIndex(str(rd))
    assert ri.RunIndex(str(rd))._lazy is True
    time.sleep(0.01)
    (rd / "iter2").mkdir()
    assert ri.RunIndex(str(rd))._lazy is False
    assert ri.RunIndex(str(rd))._lazy is True           # and re-arms afterwards


def test_re_embedding_invalidates_the_cache(run):
    _, rd = run
    ri.RunIndex(str(rd))
    assert ri.RunIndex(str(rd))._lazy is True
    time.sleep(0.01)
    os.utime(rd / "mission_control" / "embeddings.parquet", None)
    assert ri.RunIndex(str(rd))._lazy is False


# --------------------------------------------------------------------------- #
# module-level state — must not leak between runs in one process
# --------------------------------------------------------------------------- #

def test_path_map_from_a_previous_run_is_cleared(run):
    _, rd = run
    ri.PATH_MAP.extend([("/origin/machine/ws", "/local/ws")])
    ri.RunIndex(str(rd))
    assert ("/origin/machine/ws", "/local/ws") not in ri.PATH_MAP


# --------------------------------------------------------------------------- #
# _json_safe — every API response passes through it
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value, expected", [
    (float("nan"), None),
    (float("inf"), None),
    (np.float32("nan"), None),
    (1.5, 1.5),
    (np.float64(2.5), 2.5),
    (np.int64(3), 3),
    (np.bool_(True), True),
    ("text", "text"),
    (None, None),
])
def test_scalars_are_made_json_safe(value, expected):
    assert ri._json_safe(value) == expected


def test_nan_nested_in_containers_is_neutralized():
    # A bare NaN inside a list is not valid JSON; without recursion the response
    # serializes to something no JSON parser will accept.
    assert ri._json_safe([1.0, float("nan")]) == [1.0, None]
    assert ri._json_safe((1.0, float("nan"))) == [1.0, None]
    assert ri._json_safe(np.array([1.0, np.nan])) == [1.0, None]
    assert ri._json_safe({"coords": [float("nan"), 2.0]}) == {"coords": [None, 2.0]}


def test_json_safe_output_actually_serializes():
    json.dumps(ri._json_safe({"a": [float("nan")], "b": np.array([np.inf])}))


# --------------------------------------------------------------------------- #
# lighting — a component is one point however many captures it has
# --------------------------------------------------------------------------- #

def test_multi_light_run_reports_every_condition_in_channel_order(tmp_path):
    ws, rd = build_run(tmp_path, lights=("SolderLight", "WhiteLight", "UVLight"))
    add_mining(ws, rd)
    idx = ri.RunIndex(str(rd))
    assert idx.lights == ["SolderLight", "WhiteLight", "UVLight"]
    assert idx.light == "SolderLight"          # channel 0 keys every point
    assert idx.summary()["lights"] == idx.lights


def test_extra_lights_do_not_add_map_points(tmp_path):
    _, rd1 = build_run(tmp_path / "one", lights=("SolderLight",))
    _, rd3 = build_run(tmp_path / "three", lights=("SolderLight", "WhiteLight", "UVLight"))
    assert len(ri.RunIndex(str(rd1)).points) == len(ri.RunIndex(str(rd3)).points)


def test_image_path_defaults_to_the_channel_zero_capture(tmp_path):
    ws, rd = build_run(tmp_path, lights=("SolderLight", "WhiteLight"))
    add_mining(ws, rd)
    idx = ri.RunIndex(str(rd))
    assert idx.image_path(0).endswith("_SolderLight.jpg")
    assert idx.image_path(0, "SolderLight") == idx.image_path(0)


def test_image_path_reaches_a_sibling_lighting_capture(tmp_path):
    ws, rd = build_run(tmp_path, lights=("SolderLight", "WhiteLight"))
    add_mining(ws, rd)
    idx = ri.RunIndex(str(rd))
    other = idx.image_path(0, "WhiteLight")
    assert other.endswith("_WhiteLight.jpg")
    assert Path(other).is_file()               # the fixture captured it
    # same component, different capture
    assert Path(other).parent == Path(idx.image_path(0)).parent


def test_uncaptured_lighting_yields_a_path_that_does_not_exist(tmp_path):
    ws, rd = build_run(tmp_path, lights=("SolderLight",))
    add_mining(ws, rd)
    idx = ri.RunIndex(str(rd))
    assert not Path(idx.image_path(0, "WhiteLight")).is_file()


# --------------------------------------------------------------------------- #
# workspace inputs — recorded paths preferred over the positional guess
# --------------------------------------------------------------------------- #

def test_inputs_come_from_the_runs_own_record(tmp_path):
    ws, rd = build_run(tmp_path)
    add_mining(ws, rd)
    idx = ri.RunIndex(str(rd))
    assert set(idx.paths["sources"].values()) == {"config"}
    assert idx.images_dir == ws / "kpi" / "images"


# --------------------------------------------------------------------------- #
# added images vs added rows — the two must not be conflated
# --------------------------------------------------------------------------- #

def test_added_counts_images_not_csv_rows(tmp_path):
    ws, rd = build_run(tmp_path)
    add_mining(ws, rd)
    idx = ri.RunIndex(str(rd))
    it1 = next(i for i in idx.summary()["iterations"] if i["label"] == "iter1")
    assert it1["added_rows"] == sum(1 for v in idx.membership.values() if v == "iter1")


def test_added_images_are_split_by_how_they_arrived(tmp_path):
    ws, rd = build_run(tmp_path)
    add_mining(ws, rd)
    idx = ri.RunIndex(str(rd))
    for it in idx.summary()["iterations"]:
        if it["label"] == "baseline":
            assert it["added_by_provenance"] is None
            continue
        by = it["added_by_provenance"]
        assert isinstance(by, dict)
        assert sum(by.values()) == it["added_rows"]      # the split accounts for all


def test_the_row_delta_reconciles_consecutive_train_row_counts(tmp_path):
    ws, rd = build_run(tmp_path)
    add_mining(ws, rd)
    idx = ri.RunIndex(str(rd))
    its = idx.summary()["iterations"]
    for prev, cur in zip(its, its[1:]):
        if cur["appended_rows"] is None or cur["train_rows"] is None:
            continue
        assert prev["train_rows"] + cur["appended_rows"] == cur["train_rows"]


def test_appended_rows_is_never_smaller_than_new_images(tmp_path):
    ws, rd = build_run(tmp_path)
    add_mining(ws, rd)
    for it in ri.RunIndex(str(rd)).summary()["iterations"]:
        if it["appended_rows"] is None:
            continue
        assert it["appended_rows"] >= it["added_rows"]


def test_points_sharing_an_object_name_are_distinguishable(tmp_path):
    ws, rd = build_run(tmp_path)
    add_mining(ws, rd)
    pts = ri.RunIndex(str(rd)).points_payload()

    ids = [p["id"] for p in pts]
    assert len(set(ids)) == len(ids)                       # ids are unique
    assert all("location" in p for p in pts)

    by_name = {}
    for p in pts:
        by_name.setdefault(p["object_name"], []).append(p)
    for name, group in by_name.items():
        if len(group) > 1:                                  # same designator
            locs = [p["location"] for p in group]
            assert len(set(locs)) == len(locs), f"{name} points share a location"


def test_location_is_relative_not_an_absolute_host_path(tmp_path):
    ws, rd = build_run(tmp_path)
    add_mining(ws, rd)
    for p in ri.RunIndex(str(rd)).points_payload():
        assert not p["location"].startswith("/")


# --------------------------------------------------------------------------- #
# defect_margin_table — per-defect distance to each iteration's threshold
# --------------------------------------------------------------------------- #

# Four KPI defects plus a PASS, scored so every margin is hand-checkable
# against the iter1 threshold of 0.40:
#   Missing:  0.50 -> +0.10   0.41 -> +0.01   0.395 -> -0.005
#   Bridge:   0.60 -> +0.20
MARGIN_KPI = [("b1", "K0", "PASS"), ("b1", "K1", "Missing"),
              ("b1", "K2", "Missing"), ("b1", "K3", "Missing"),
              ("b1", "K4", "Bridge")]
MARGIN_SCORES = [("b1", "K0", "PASS", 0.20), ("b1", "K1", "Missing", 0.50),
                 ("b1", "K2", "Missing", 0.41), ("b1", "K3", "Missing", 0.395),
                 ("b1", "K4", "Bridge", 0.60)]


@pytest.fixture
def margins(tmp_path):
    """A run whose iter1 KPI scores straddle the threshold in both directions."""
    ws, rd = build_run(tmp_path, kpi=MARGIN_KPI)
    add_inference(rd, MARGIN_SCORES, iteration="iter1")
    return {r["kpi_defect_type"]: r
            for r in ri.RunIndex(str(rd)).defect_margin_table()
            if r["iter"] == "iter1"}


def test_one_row_per_defect_type_with_the_documented_columns(margins):
    assert set(margins) == {"Missing", "Bridge"}
    assert set(margins["Missing"]) == {"iter", "kpi_defect_type", "n",
                                       "min_margin", "median_margin", "at_risk"}


def test_n_counts_the_scored_samples_of_that_defect(margins):
    assert margins["Missing"]["n"] == 3
    assert margins["Bridge"]["n"] == 1


def test_min_margin_is_the_worst_sample_and_goes_negative_below_threshold(margins):
    # 0.395 - 0.40; a negative margin is a sample the threshold would miss
    assert margins["Missing"]["min_margin"] == -0.005


def test_median_margin_is_the_middle_sample_not_the_mean(margins):
    # margins are -0.005, 0.01, 0.10 — mean would be ~0.035
    assert margins["Missing"]["median_margin"] == 0.01


def test_median_of_a_single_sample_equals_its_margin(margins):
    assert margins["Bridge"]["median_margin"] == margins["Bridge"]["min_margin"] == 0.20


def test_at_risk_counts_margins_inside_the_002_band(margins):
    # 0.01 and -0.005 are within 0.02 of the threshold; 0.10 is not
    assert margins["Missing"]["at_risk"] == 2
    assert margins["Bridge"]["at_risk"] == 0


def test_margins_are_measured_against_that_iterations_own_threshold(tmp_path):
    # the same scores against baseline's 0.30 must shift every margin by +0.10
    ws, rd = build_run(tmp_path, kpi=MARGIN_KPI)
    add_inference(rd, MARGIN_SCORES, iteration="baseline")
    add_inference(rd, MARGIN_SCORES, iteration="iter1")
    rows = {(r["iter"], r["kpi_defect_type"]): r
            for r in ri.RunIndex(str(rd)).defect_margin_table()}
    assert rows[("baseline", "Missing")]["min_margin"] == 0.095
    assert rows[("iter1", "Missing")]["min_margin"] == -0.005
    # a looser threshold pulls every sample clear of the at-risk band
    assert rows[("baseline", "Missing")]["at_risk"] == 0
    assert rows[("iter1", "Missing")]["at_risk"] == 2


def test_pass_samples_are_excluded(tmp_path):
    # K0 is PASS with a score; it must not create a row or inflate any n
    ws, rd = build_run(tmp_path, kpi=MARGIN_KPI)
    add_inference(rd, MARGIN_SCORES, iteration="iter1")
    rows = ri.RunIndex(str(rd)).defect_margin_table()
    assert "PASS" not in {r["kpi_defect_type"] for r in rows}
    assert sum(r["n"] for r in rows if r["iter"] == "iter1") == 4


def test_an_iteration_with_no_scores_yields_no_rows(tmp_path):
    # baseline has a threshold recorded but no inference.csv on disk
    ws, rd = build_run(tmp_path, kpi=MARGIN_KPI)
    add_inference(rd, MARGIN_SCORES, iteration="iter1")
    iters = {r["iter"] for r in ri.RunIndex(str(rd)).defect_margin_table()}
    assert iters == {"iter1"}


def test_an_iteration_without_a_threshold_is_skipped(tmp_path):
    ws, rd = build_run(tmp_path, kpi=MARGIN_KPI, iterations={
        "baseline": {"far_pct": 50.0, "threshold": None},
        "iter1": {"far_pct": 10.0, "threshold": 0.40}})
    add_inference(rd, MARGIN_SCORES, iteration="baseline")
    add_inference(rd, MARGIN_SCORES, iteration="iter1")
    assert {r["iter"] for r in ri.RunIndex(str(rd)).defect_margin_table()} == {"iter1"}


def test_rows_are_ordered_by_defect_type_within_an_iteration(tmp_path):
    ws, rd = build_run(tmp_path, kpi=MARGIN_KPI)
    add_inference(rd, MARGIN_SCORES, iteration="iter1")
    types = [r["kpi_defect_type"] for r in ri.RunIndex(str(rd)).defect_margin_table()]
    assert types == sorted(types)


def test_a_run_with_no_inference_returns_no_rows(run):
    ws, rd = run
    assert ri.RunIndex(str(rd)).defect_margin_table() == []


# --------------------------------------------------------------------------- #
# inference_csv — the loop has written this two ways
# --------------------------------------------------------------------------- #

def test_an_undocumented_flat_layout_is_not_accepted(tmp_path):
    # only references/data-layout.md's per-checkpoint layout counts
    ws, rd = build_run(tmp_path, kpi=MARGIN_KPI)
    add_inference(rd, MARGIN_SCORES, iteration="iter1", kind=None)
    assert ri.RunIndex(str(rd)).inference_csv("iter1") is None


def test_best_ckpt_kind_is_honoured_over_best_val(tmp_path):
    ws, rd = build_run(tmp_path, kpi=MARGIN_KPI, iterations={
        "iter1": {"far_pct": 10.0, "threshold": 0.40, "best_ckpt_kind": "latest"}})
    add_inference(rd, MARGIN_SCORES, iteration="iter1", kind="best_val")
    add_inference(rd, MARGIN_SCORES, iteration="iter1", kind="latest")
    assert ri.RunIndex(str(rd)).inference_csv("iter1").parent.name == "latest"


def test_no_inference_anywhere_returns_none(run):
    ws, rd = run
    assert ri.RunIndex(str(rd)).inference_csv("iter1") is None


# --------------------------------------------------------------------------- #
# projection — t-SNE by default, UMAP opt-in, caches keyed by method
# --------------------------------------------------------------------------- #

def test_tsne_is_the_default(run):
    ws, rd = run
    assert ri.RunIndex(str(rd)).projection == "tsne"


def test_an_unknown_projection_fails_loudly(run):
    ws, rd = run
    with pytest.raises(RuntimeError, match="unknown projection"):
        ri.RunIndex(str(rd), projection="pca")


def test_the_coords_cache_is_keyed_by_method(run):
    ws, rd = run
    ri.RunIndex(str(rd))
    assert (rd / "mission_control" / "coords_tsne.npy").is_file()
    assert not (rd / "mission_control" / "coords_umap.npy").is_file()


def test_the_serve_cache_records_the_projection(run):
    ws, rd = run
    ri.RunIndex(str(rd))
    meta = json.loads((rd / "mission_control" / "serve_meta.json").read_text())
    assert meta["projection"] == "tsne"


def test_serving_reproduces_the_built_projection_without_a_flag(run):
    # prepare.py records it; server.py passes nothing and must still agree
    ws, rd = run
    meta_f = rd / "mission_control" / "serve_meta.json"
    ri.RunIndex(str(rd))
    meta = json.loads(meta_f.read_text())
    meta["projection"] = "umap"
    meta_f.write_text(json.dumps(meta))
    assert ri.RunIndex(str(rd)).projection == "umap"


def test_switching_projection_invalidates_the_serve_cache(run):
    # coords carry no method of their own — reusing them would show the old
    # layout and look merely different, not broken
    ws, rd = run
    ri.RunIndex(str(rd))
    ix = ri.RunIndex(str(rd), projection="tsne")
    assert ix._load_serve() is True
    ix.projection = "umap"
    assert ix._load_serve() is False
