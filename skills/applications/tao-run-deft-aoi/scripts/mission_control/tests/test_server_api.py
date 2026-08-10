# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Mission Control HTTP API, driven in-process by TestClient."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fixture import add_inference, add_mining, add_routing, build_run  # noqa: E402

pytest.importorskip("pyarrow", reason="parquet fixtures need pyarrow")
TestClient = pytest.importorskip("fastapi.testclient", reason="fastapi not installed").TestClient

import server  # noqa: E402
from indexer.run_index import RunIndex  # noqa: E402


@pytest.fixture
def client(tmp_path):
    """A TestClient with a synthetic run loaded into the server's INDEX global."""
    ws, rd = build_run(tmp_path)
    _, _, tgts = add_mining(ws, rd)
    add_routing(rd, tgts)
    server.INDEX = RunIndex(str(rd))
    server.AGENTS.clear()
    with TestClient(server.app) as c:
        yield c
    server.INDEX = None


def _strict_json(resp):
    """Parse with the strict grammar — json.loads accepts NaN/Infinity by
    default, exactly the tokens a browser's JSON.parse rejects."""
    return json.loads(resp.text, parse_constant=lambda c: pytest.fail(
        f"response contains bare {c}, which is not valid JSON"))


# --------------------------------------------------------------------------- #
# the payloads app.js loads on boot
# --------------------------------------------------------------------------- #

def test_summary_carries_the_fields_the_header_and_timeline_read(client):
    body = _strict_json(client.get("/api/summary"))
    assert {"run_id", "iterations", "best", "counts", "spaces",
            "kpi_target", "stages"} <= set(body)
    assert body["best"]["label"] == "iter1"
    assert body["counts"] == {"pool": 3, "kpi": 2}
    assert body["spaces"] == ["siglip"]


def test_points_is_a_list_of_map_points(client):
    body = _strict_json(client.get("/api/points"))
    assert isinstance(body, list) and len(body) == 5
    # `key` is the internal join key and is deliberately withheld; the frontend
    # addresses points by the positional `id` it gets here.
    assert {"id", "kind", "label", "image_url"} <= set(body[0])
    assert "key" not in body[0]
    assert [p["id"] for p in body] == list(range(5))


def test_defect_margins_returns_rows(client):
    body = _strict_json(client.get("/api/defect_margins"))
    assert isinstance(body, list)


def test_defect_margins_serves_the_table_the_frontend_renders(tmp_path):
    # the margins panel indexes these six keys by name; a rename breaks it
    # silently, and the shared fixture has no inference.csv so it serves []
    ws, rd = build_run(tmp_path, kpi=[("b1", "K0", "PASS"), ("b1", "K1", "Missing"),
                                      ("b1", "K2", "Missing")])
    add_inference(rd, [("b1", "K0", "PASS", 0.20), ("b1", "K1", "Missing", 0.50),
                       ("b1", "K2", "Missing", 0.41)], iteration="iter1")
    server.INDEX = RunIndex(str(rd))
    with TestClient(server.app) as c:
        body = _strict_json(c.get("/api/defect_margins"))

    (row,) = [r for r in body if r["iter"] == "iter1"]
    assert row == {"iter": "iter1", "kpi_defect_type": "Missing", "n": 2,
                   "min_margin": 0.01, "median_margin": 0.055, "at_risk": 1}


def test_boot_payloads_are_all_served(client):
    for route in ("/api/summary", "/api/points", "/api/defect_margins"):
        assert client.get(route).status_code == 200


# --------------------------------------------------------------------------- #
# per-iteration routes
# --------------------------------------------------------------------------- #

def test_mining_edges_reports_the_iterations_own_recipe(client):
    body = _strict_json(client.get("/api/mining_edges/iter1"))
    assert body["iteration"] == "iter1"
    assert body["encoder"] == "siglip"
    assert body["min_similarity"] == 0.9
    assert isinstance(body["edges"], list)


def test_weak_targets_are_ordered_by_weakness(client):
    body = _strict_json(client.get("/api/weak_targets/iter1"))
    weaknesses = [t.get("weakness") for t in body["targets"]]
    assert weaknesses == sorted(weaknesses, reverse=True)


def test_iteration_without_artifacts_is_empty_not_an_error(client):
    assert client.get("/api/mining_edges/baseline").status_code == 200
    assert _strict_json(client.get("/api/weak_targets/baseline"))["targets"] == []


def test_unknown_iteration_does_not_500(client):
    for route in ("/api/mining_edges/iter99", "/api/weak_targets/iter99"):
        assert client.get(route).status_code in (200, 404)


# --------------------------------------------------------------------------- #
# point-scoped routes
# --------------------------------------------------------------------------- #

def test_neighbors_replays_in_the_siglip_space(client):
    body = _strict_json(client.get("/api/neighbors/0", params={"iteration": "iter1"}))
    assert body.get("space", "siglip") == "siglip"


def test_neighbors_out_of_range_index_is_404_not_a_crash(client):
    r = client.get("/api/neighbors/99999", params={"iteration": "iter1"})
    assert r.status_code == 404


def test_image_and_neighbors_agree_on_a_bad_index(client):
    assert client.get("/api/image/99999").status_code == 404
    assert client.get("/api/neighbors/99999",
                      params={"iteration": "iter1"}).status_code == 404


def test_image_returns_the_crop_bytes(client):
    r = client.get("/api/image/0")
    assert r.status_code == 200
    assert r.content


def test_image_out_of_range_index_is_404(client):
    assert client.get("/api/image/99999").status_code == 404


# --------------------------------------------------------------------------- #
# run switching
# --------------------------------------------------------------------------- #

def test_runs_lists_the_loaded_run(client):
    body = _strict_json(client.get("/api/runs"))
    loaded = [r for r in body if r["loaded"]]
    assert len(loaded) == 1
    assert loaded[0]["name"] == "run_X" and loaded[0]["standard"] is True


def test_reload_rereads_the_same_run(client):
    r = client.post("/api/reload")
    assert r.status_code == 200
    assert r.json()["reloaded"] == "run_X"


def test_loading_a_non_deft_directory_is_rejected(client):
    (server.INDEX.rd.parent / "run_bogus").mkdir()
    assert client.post("/api/load/run_bogus").status_code == 400


# --------------------------------------------------------------------------- #
# cross-cutting
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("route", [
    "/api/summary", "/api/points", "/api/defect_margins",
    "/api/mining_edges/iter1", "/api/weak_targets/iter1", "/api/runs",
])
def test_every_json_route_is_strictly_parseable(client, route):
    _strict_json(client.get(route))


def test_a_nan_in_the_source_data_never_reaches_the_wire(tmp_path):
    ws, rd = build_run(tmp_path)
    _, _, tgts = add_mining(ws, rd)
    rr = add_routing(rd, tgts)
    df = pd.read_parquet(rr / "mining_gaps.parquet")
    df.loc[0, "weakness"] = float("nan")
    df.to_parquet(rr / "mining_gaps.parquet", index=False)

    server.INDEX = RunIndex(str(rd))
    with TestClient(server.app) as c:
        resp = c.get("/api/weak_targets/iter1")
    server.INDEX = None

    assert "NaN" not in resp.text
    body = _strict_json(resp)
    assert any(t["weakness"] is None for t in body["targets"])


@pytest.mark.parametrize("route", ["/", "/app.js", "/style.css", "/api/summary"])
def test_responses_are_marked_no_store(client, route):
    assert client.get(route).headers["cache-control"] == "no-store"


def test_frontend_assets_are_served(client):
    assert client.get("/").headers["content-type"].startswith("text/html")
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_agent_config_reports_key_state_without_calling_an_llm(client):
    body = _strict_json(client.get("/api/agent/config"))
    assert {"provider", "model", "key_set"} <= set(body)
    assert isinstance(body["key_set"], bool)


# --------------------------------------------------------------------------- #
# lighting — one component, one point, N viewable captures
# --------------------------------------------------------------------------- #

def test_summary_advertises_the_runs_lighting_conditions(client):
    assert _strict_json(client.get("/api/summary"))["lights"] == ["SolderLight"]


def test_image_serves_a_named_lighting_capture(tmp_path):
    ws, rd = build_run(tmp_path, lights=("SolderLight", "WhiteLight"))
    add_mining(ws, rd)
    server.INDEX = RunIndex(str(rd))
    with TestClient(server.app) as c:
        assert _strict_json(c.get("/api/summary"))["lights"] == ["SolderLight", "WhiteLight"]
        assert c.get("/api/image/0").status_code == 200                      # channel 0
        assert c.get("/api/image/0", params={"light": "WhiteLight"}).status_code == 200
    server.INDEX = None


def test_a_lighting_that_was_not_captured_is_404_not_a_silent_fallback(client):
    r = client.get("/api/image/0", params={"light": "UVLight"})
    assert r.status_code == 404
