# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SDK-free AutoML step engine."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
import automl_step as step  # noqa: E402


SCHEMA = {
    "automl_default_parameters": [
        "train.lr",
        "train.batch_size",
        "train.num_epochs",
        "model.variant",
        "augment.flip",
    ],
    "default": {
        "train": {"lr": 0.001, "batch_size": 4, "num_epochs": 10},
        "model": {"variant": "small"},
        "augment": {"flip": True},
    },
    "properties": {
        "train": {
            "type": "object",
            "properties": {
                "lr": {
                    "type": "float",
                    "default": 0.001,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "automl_enabled": True,
                },
                "batch_size": {
                    "type": "int",
                    "default": 4,
                    "minimum": 1,
                    "maximum": float("inf"),
                    "automl_enabled": True,
                },
                "num_epochs": {
                    "type": "int",
                    "default": 10,
                    "minimum": 2,
                    "maximum": 20,
                    "automl_enabled": True,
                },
            },
        },
        "model": {
            "type": "object",
            "properties": {
                "variant": {
                    "type": "string",
                    "default": "small",
                    "enum": ["small", "large"],
                    "automl_enabled": True,
                }
            },
        },
        "augment": {
            "type": "object",
            "properties": {
                "flip": {
                    "type": "boolean",
                    "default": True,
                    "automl_enabled": True,
                }
            },
        },
    },
}


@pytest.fixture
def schema_path(tmp_path):
    path = tmp_path / "train.schema.json"
    path.write_text(json.dumps(SCHEMA))
    return path


@pytest.fixture
def llm_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            envelope = json.loads(self.rfile.read(length))
            prompt = json.loads(envelope["messages"][1]["content"])
            requests.append({"path": self.path, "prompt": prompt, "headers": self.headers})
            if prompt["purpose"].startswith("hybrid_"):
                content = {"algorithm": "bfbo", "reason": "start with batch exploration"}
            else:
                content = {
                    "normalized_vector": [0.25] * len(prompt["parameters"]),
                    "reason": "bounded proposal from prior outcomes",
                    "evolvable_text": {
                        "model.variant": "generated architecture instruction"
                    },
                }
            body = json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content)}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        thread.join()


def invoke(capsys, *args):
    assert step.main(list(args)) == 0
    return json.loads(capsys.readouterr().out)


def init(capsys, state, schema_path, *, max_recommendations=3, seed=17):
    return invoke(
        capsys,
        "init",
        "--state",
        str(state),
        "--schema",
        str(schema_path),
        "--experiment-id",
        "automl-test",
        "--network-arch",
        "dino",
        "--metric",
        "val_loss",
        "--direction",
        "minimize",
        "--max-recommendations",
        str(max_recommendations),
        "--candidate-count",
        "64",
        "--seed",
        str(seed),
    )


def recommend(capsys, state):
    return invoke(capsys, "recommend", "--state", str(state))["recommendation"]


def bind(capsys, state, rec_id, job_id):
    return invoke(
        capsys,
        "bind-job",
        "--state",
        str(state),
        "--rec-id",
        rec_id,
        "--job-id",
        job_id,
    )


def report_success(capsys, state, rec_id, job_id, metric, checkpoint):
    return invoke(
        capsys,
        "report",
        "--state",
        str(state),
        "--rec-id",
        rec_id,
        "--outcome",
        "SUCCESS",
        "--job-id",
        job_id,
        "--metric",
        str(metric),
        "--checkpoint-uri",
        checkpoint,
    )


def test_init_snapshots_schema_and_derives_safe_ranges(capsys, tmp_path, schema_path):
    state = tmp_path / "experiment.json"
    output = init(capsys, state, schema_path)
    saved = json.loads(state.read_text())

    jsonschema.validate(saved, json.loads(step.EXPERIMENT_SCHEMA.read_text()))
    assert output["experiment_id"] == "automl-test"
    assert len(saved["search_parameters"]) == 5
    by_name = {param["name"]: param for param in saved["search_parameters"]}
    assert by_name["train.lr"]["scale"] == "log"
    assert by_name["train.lr"]["minimum"] == pytest.approx(1e-5)
    assert by_name["train.lr"]["maximum"] == pytest.approx(0.1)
    assert by_name["train.batch_size"]["maximum"] == 8
    assert by_name["model.variant"]["options"] == ["small", "large"]


def test_recommend_is_deterministic_and_idempotent(capsys, tmp_path, schema_path):
    first_state = tmp_path / "first.json"
    second_state = tmp_path / "second.json"
    init(capsys, first_state, schema_path, seed=44)
    init(capsys, second_state, schema_path, seed=44)

    first = recommend(capsys, first_state)
    second = recommend(capsys, second_state)
    repeated = invoke(capsys, "recommend", "--state", str(first_state))

    assert first["vector"] == second["vector"]
    assert first["parameters"] == second["parameters"]
    assert repeated["created"] is False
    assert repeated["recommendation"] == first
    assert all("." not in key for key in first["spec"])
    assert first["spec"]["train"]["lr"] == first["parameters"]["train.lr"]


def test_parameter_values_apply_schema_and_dino_constraints():
    parameters = [
        {
            "name": "model.num_queries",
            "kind": "int",
            "default": 100,
            "minimum": 100,
            "maximum": 200,
            "scale": "linear",
            "parent_param": "TRUE",
        },
        {
            "name": "model.num_select",
            "kind": "int",
            "default": 100,
            "minimum": 1,
            "maximum": 300,
            "scale": "linear",
            "depends_on": "model.num_queries",
        },
        {
            "name": "crop.min",
            "kind": "int",
            "default": 4,
            "minimum": 2,
            "maximum": 8,
            "scale": "linear",
            "parent_param": "TRUE",
        },
        {
            "name": "crop.max",
            "kind": "int",
            "default": 8,
            "minimum": 3,
            "maximum": 12,
            "scale": "linear",
            "math_cond": "> depends_on",
            "depends_on": "crop.min",
        },
        {
            "name": "model.width",
            "kind": "int",
            "default": 8,
            "minimum": 1,
            "maximum": 16,
            "scale": "linear",
            "math_cond": "^ 2",
        },
        {
            "name": "model.hidden",
            "kind": "int",
            "default": 8,
            "minimum": 1,
            "maximum": 17,
            "scale": "linear",
            "math_cond": "/ 8",
        },
        {
            "name": "train.warmup",
            "kind": "int",
            "default": 1,
            "minimum": 0,
            "maximum": 10,
            "scale": "linear",
            "math_cond": "/ 2",
            "depends_on": "crop.min",
        },
    ]

    values = step._parameter_values(
        parameters,
        [0.0, 0.99, 0.99, 0.0, 0.6, 0.99, 0.5],
        base_spec={},
        network_arch="dino",
    )

    assert values["model.num_select"] <= values["model.num_queries"]
    assert values["crop.max"] > values["crop.min"]
    assert values["model.width"] in {2, 4, 8, 16}
    assert values["model.hidden"] in {8, 16}
    assert values["train.warmup"] == values["crop.min"] // 2


def test_real_dino_recommendation_honors_relational_constraints(capsys, tmp_path):
    schema = step.SKILLS_DIR / "models" / "tao-train-dino" / "schemas" / "train.schema.json"
    state = tmp_path / "dino.json"
    init(capsys, state, schema, max_recommendations=1, seed=23)
    rec = recommend(capsys, state)

    assert (
        rec["parameters"]["dataset.augmentation.train_random_crop_max"]
        > rec["parameters"]["dataset.augmentation.train_random_crop_min"]
    )
    assert rec["parameters"]["model.num_select"] <= rec["parameters"][
        "model.num_queries"
    ]


def test_bind_is_idempotent_and_blocks_new_recommendation(capsys, tmp_path, schema_path):
    state = tmp_path / "experiment.json"
    init(capsys, state, schema_path)
    rec = recommend(capsys, state)

    first = bind(capsys, state, rec["id"], "dino-train-a1")
    second = bind(capsys, state, rec["id"], "dino-train-a1")
    blocked = invoke(capsys, "recommend", "--state", str(state))

    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert blocked["ready"] is False
    assert blocked["reason"] == "recommendation_in_flight"


def test_infrastructure_error_retries_same_rec_without_budget(capsys, tmp_path, schema_path):
    state = tmp_path / "experiment.json"
    init(capsys, state, schema_path, max_recommendations=1)
    rec = recommend(capsys, state)
    bind(capsys, state, rec["id"], "dino-train-old")

    result = invoke(
        capsys,
        "report",
        "--state",
        str(state),
        "--rec-id",
        rec["id"],
        "--outcome",
        "ERR_INFRA",
        "--job-id",
        "dino-train-old",
        "--message",
        "NODE_FAIL",
    )
    repeated = invoke(
        capsys,
        "report",
        "--state",
        str(state),
        "--rec-id",
        rec["id"],
        "--outcome",
        "ERR_INFRA",
        "--job-id",
        "dino-train-old",
        "--message",
        "NODE_FAIL",
    )
    retried = recommend(capsys, state)
    bind(capsys, state, rec["id"], "dino-train-new")
    with pytest.raises(SystemExit, match="does not match active job"):
        step.main(
            [
                "report",
                "--state",
                str(state),
                "--rec-id",
                rec["id"],
                "--outcome",
                "ERR_INFRA",
                "--job-id",
                "dino-train-old",
            ]
        )
    saved = json.loads(state.read_text())

    assert result["status"] == "ACTIVE"
    assert result["recommendation"]["state"] == "READY"
    assert repeated["idempotent"] is True
    assert retried["id"] == rec["id"]
    assert len(saved["recommendations"]) == 1
    assert [attempt["job_id"] for attempt in saved["recommendations"][0]["attempts"]] == [
        "dino-train-old",
        "dino-train-new",
    ]


def test_report_replay_rejects_conflicting_terminal_fact(capsys, tmp_path, schema_path):
    state = tmp_path / "experiment.json"
    init(capsys, state, schema_path, max_recommendations=1)
    rec = recommend(capsys, state)
    bind(capsys, state, rec["id"], "dino-train-a1")
    report_success(
        capsys,
        state,
        rec["id"],
        "dino-train-a1",
        0.23,
        "/results/a1/model.pth",
    )

    repeated = report_success(
        capsys,
        state,
        rec["id"],
        "dino-train-a1",
        0.23,
        "/results/a1/model.pth",
    )
    assert repeated["idempotent"] is True

    with pytest.raises(SystemExit, match="Conflicting repeated report.*metric"):
        step.main(
            [
                "report",
                "--state",
                str(state),
                "--rec-id",
                rec["id"],
                "--outcome",
                "SUCCESS",
                "--job-id",
                "dino-train-a1",
                "--metric",
                "0.99",
                "--checkpoint-uri",
                "/results/a1/model.pth",
            ]
        )


def test_success_report_requires_checkpoint(capsys, tmp_path, schema_path):
    state = tmp_path / "experiment.json"
    init(capsys, state, schema_path, max_recommendations=1)
    rec = recommend(capsys, state)
    bind(capsys, state, rec["id"], "dino-train-a1")

    with pytest.raises(SystemExit, match="requires --metric and --checkpoint-uri"):
        step.main(
            [
                "report",
                "--state",
                str(state),
                "--rec-id",
                rec["id"],
                "--outcome",
                "SUCCESS",
                "--job-id",
                "dino-train-a1",
                "--metric",
                "0.23",
            ]
        )


def test_sdk_free_init_accepts_non_train_action(capsys, tmp_path, schema_path):
    state = tmp_path / "experiment.json"
    output = invoke(
        capsys,
        "init",
        "--state",
        str(state),
        "--schema",
        str(schema_path),
        "--network-arch",
        "dino",
        "--action",
        "evaluate",
        "--metric",
        "val_loss",
        "--direction",
        "minimize",
    )
    assert output["algorithm"] == "bayesian"
    assert json.loads(state.read_text())["action"] == "evaluate"


def test_metric_record_drives_report_and_rejects_wrong_identity(
    capsys, tmp_path, schema_path
):
    state = tmp_path / "experiment.json"
    init(capsys, state, schema_path, max_recommendations=1)
    rec = recommend(capsys, state)
    bind(capsys, state, rec["id"], "dino-train-a1")
    metric_record = {
        "schema_version": 1,
        "experiment_id": "automl-test",
        "rec_id": rec["id"],
        "job_id": "dino-train-a1",
        "status": "COMPLETE",
        "primary_metric": "val_loss",
        "direction": "minimize",
        "metrics": {"val_loss": 0.23},
        "artifacts": {"checkpoint_uri": "/results/a1/model.pth"},
        "failure": None,
        "measured_at": "2026-07-21T12:00:00+00:00",
    }
    metric_path = tmp_path / "metrics.json"
    metric_path.write_text(json.dumps(metric_record))

    output = invoke(
        capsys,
        "report",
        "--state",
        str(state),
        "--rec-id",
        rec["id"],
        "--metric-record",
        str(metric_path),
    )
    assert output["recommendation"]["state"] == "SUCCEEDED"
    assert output["recommendation"]["metric"] == pytest.approx(0.23)
    assert output["status"] == "COMPLETE"

    other_state = tmp_path / "other.json"
    init(capsys, other_state, schema_path, max_recommendations=1)
    other_rec = recommend(capsys, other_state)
    bind(capsys, other_state, other_rec["id"], "dino-train-b1")
    metric_record["job_id"] = "wrong-job"
    metric_path.write_text(json.dumps(metric_record))
    with pytest.raises(SystemExit, match="job_id"):
        step.main(
            [
                "report",
                "--state",
                str(other_state),
                "--rec-id",
                other_rec["id"],
                "--metric-record",
                str(metric_path),
            ]
        )


def test_bayesian_loop_selects_best_and_finalizes_contract(capsys, tmp_path, schema_path):
    state = tmp_path / "experiment.json"
    best_path = tmp_path / "best_rec.json"
    init(capsys, state, schema_path, max_recommendations=3, seed=9)

    metrics = [0.8, 0.3, 0.5]
    recommendations = []
    for index, metric in enumerate(metrics):
        rec = recommend(capsys, state)
        recommendations.append(rec)
        job_id = f"dino-train-{index}"
        bind(capsys, state, rec["id"], job_id)
        report_success(
            capsys,
            state,
            rec["id"],
            job_id,
            metric,
            f"/results/{index}/model.pth",
        )

    assert len({_canonical(rec["parameters"]) for rec in recommendations}) == 3
    status = invoke(capsys, "status", "--state", str(state), "--full")
    assert status["status"] == "COMPLETE"
    assert status["best"]["rec_id"] == "rec-0001"

    finalized = invoke(
        capsys,
        "finalize",
        "--state",
        str(state),
        "--out",
        str(best_path),
    )
    best = json.loads(best_path.read_text())
    jsonschema.validate(best, json.loads(step.BEST_REC_SCHEMA.read_text()))
    assert finalized["payload"] == best
    assert best["best"]["rec_id"] == "rec-0001"
    assert best["best"]["score"] == pytest.approx(0.3)
    assert "num_epochs" not in best["best"]["specs"]["train"]
    assert best["best"]["observed_budget"]["num_epochs"] == recommendations[1][
        "parameters"
    ]["train.num_epochs"]


def test_finalize_rejects_active_experiment(capsys, tmp_path, schema_path):
    state = tmp_path / "experiment.json"
    init(capsys, state, schema_path, max_recommendations=2)
    rec = recommend(capsys, state)
    bind(capsys, state, rec["id"], "dino-train-a1")
    report_success(
        capsys,
        state,
        rec["id"],
        "dino-train-a1",
        0.23,
        "/results/a1/model.pth",
    )

    with pytest.raises(SystemExit, match="Cannot finalize an active experiment"):
        step.main(["finalize", "--state", str(state)])


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_program_error_consumes_recommendation_budget(capsys, tmp_path, schema_path):
    state = tmp_path / "experiment.json"
    init(capsys, state, schema_path, max_recommendations=1)
    rec = recommend(capsys, state)
    bind(capsys, state, rec["id"], "dino-train-bad")
    output = invoke(
        capsys,
        "report",
        "--state",
        str(state),
        "--rec-id",
        rec["id"],
        "--outcome",
        "ERR_PROGRAM",
        "--job-id",
        "dino-train-bad",
        "--message",
        "invalid batch size",
    )

    assert output["status"] == "COMPLETE"
    assert output["counts"]["failed"] == 1
    with pytest.raises(SystemExit, match="no successful"):
        step.main(["finalize", "--state", str(state)])


def test_init_refuses_to_overwrite_resume_state(capsys, tmp_path, schema_path):
    state = tmp_path / "experiment.json"
    init(capsys, state, schema_path)
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        init(capsys, state, schema_path)


def init_algorithm(capsys, state, schema_path, algorithm, *extra):
    return invoke(
        capsys,
        "init",
        "--state",
        str(state),
        "--schema",
        str(schema_path),
        "--experiment-id",
        f"automl-{algorithm}",
        "--network-arch",
        "cosmos-rl",
        "--algorithm",
        algorithm,
        "--metric",
        "val_loss",
        "--direction",
        "minimize",
        "--candidate-count",
        "64",
        *extra,
    )


@pytest.mark.parametrize(
    "algorithm",
    ["hyperband", "bohb", "asha", "dehb", "hyperband_es"],
)
def test_budgeted_algorithms_promote_best_and_resume(
    capsys, tmp_path, schema_path, algorithm
):
    state = tmp_path / f"{algorithm}.json"
    extra = [
        "--max-epochs",
        "2",
        "--reduction-factor",
        "2",
        "--epoch-multiplier",
        "1",
        "--max-recommendations",
        "2",
        "--max-concurrent",
        "2",
    ]
    if algorithm == "asha":
        extra.extend(["--max-trials", "2", "--min-top-configs", "1"])
    init_algorithm(capsys, state, schema_path, algorithm, *extra)

    first = invoke(capsys, "recommend", "--state", str(state))["recommendations"]
    assert len(first) == 2
    assert all(rec["budget"] == 1 for rec in first)
    assert all(rec["spec"]["train"]["num_epochs"] == 1 for rec in first)
    for index, rec in enumerate(first):
        job_id = f"{algorithm}-rung0-{index}"
        bind(capsys, state, rec["id"], job_id)
        report_success(
            capsys,
            state,
            rec["id"],
            job_id,
            [0.9, 0.4][index],
            f"/results/{job_id}/model.pth",
        )

    promoted = recommend(capsys, state)
    assert promoted["budget"] == 2
    assert promoted["parent_rec_id"] == first[1]["id"]
    assert promoted["config_id"] == first[1]["config_id"]
    assert promoted["resume_from_job_id"] == f"{algorithm}-rung0-1"
    assert promoted["parameters"] == first[1]["parameters"]
    job_id = f"{algorithm}-rung1"
    bind(capsys, state, promoted["id"], job_id)
    report_success(
        capsys,
        state,
        promoted["id"],
        job_id,
        0.6,
        f"/results/{job_id}/model.pth",
    )

    terminal = invoke(capsys, "recommend", "--state", str(state))
    assert terminal["ready"] is False
    assert terminal["status"] == "COMPLETE"
    assert terminal["best"]["rec_id"] == promoted["id"]
    assert terminal["best"]["budget"] == 2


def test_pbt_generations_exploit_perturb_and_resume(capsys, tmp_path, schema_path):
    state = tmp_path / "pbt.json"
    init_algorithm(
        capsys,
        state,
        schema_path,
        "pbt",
        "--population-size",
        "2",
        "--max-generations",
        "2",
        "--eval-interval",
        "1",
        "--max-concurrent",
        "2",
    )
    population = invoke(capsys, "recommend", "--state", str(state))["recommendations"]
    assert len(population) == 2
    assert all(rec["generation"] == 0 and rec["budget"] == 1 for rec in population)
    for index, rec in enumerate(population):
        job_id = f"pbt-member-{index}-g0"
        bind(capsys, state, rec["id"], job_id)
        report_success(
            capsys,
            state,
            rec["id"],
            job_id,
            [0.9, 0.4][index],
            f"/results/{job_id}/model.pth",
        )

    resumed = invoke(capsys, "recommend", "--state", str(state))["recommendations"]
    assert len(resumed) == 2
    assert all(rec["generation"] == 1 and rec["budget"] == 2 for rec in resumed)
    assert all(rec["resume_from_job_id"] for rec in resumed)
    assert any(rec["metadata"]["exploit"] == population[1]["id"] for rec in resumed)
    for index, rec in enumerate(resumed):
        job_id = f"pbt-member-{index}-g1"
        bind(capsys, state, rec["id"], job_id)
        report_success(
            capsys,
            state,
            rec["id"],
            job_id,
            [0.7, 0.3][index],
            f"/results/{job_id}/model.pth",
        )
    terminal = invoke(capsys, "recommend", "--state", str(state))
    assert terminal["status"] == "COMPLETE"
    assert terminal["best"]["budget"] == 2


def test_multi_objective_and_non_train_artifact(capsys, tmp_path, schema_path):
    state = tmp_path / "multi.json"
    invoke(
        capsys,
        "init",
        "--state",
        str(state),
        "--schema",
        str(schema_path),
        "--experiment-id",
        "automl-multi",
        "--network-arch",
        "dino",
        "--action",
        "evaluate",
        "--metric",
        "accuracy",
        "--direction",
        "maximize",
        "--objective",
        "accuracy:maximize:2:1",
        "--objective",
        "latency:minimize:1:10",
        "--max-recommendations",
        "1",
    )
    rec = recommend(capsys, state)
    bind(capsys, state, rec["id"], "evaluate-0")
    result = invoke(
        capsys,
        "report",
        "--state",
        str(state),
        "--rec-id",
        rec["id"],
        "--outcome",
        "SUCCESS",
        "--job-id",
        "evaluate-0",
        "--metric-value",
        "accuracy=0.8",
        "--metric-value",
        "latency=20",
        "--artifact-uri",
        "/results/evaluate-0/metrics.json",
    )
    assert result["recommendation"]["objective_values"] == {
        "accuracy": 0.8,
        "latency": 20.0,
    }
    assert result["recommendation"]["objective_score"] == pytest.approx(-0.4)
    finalized = invoke(capsys, "finalize", "--state", str(state))
    assert "checkpoint_uri" not in finalized["best"]
    assert finalized["best"]["artifact_uri"].endswith("metrics.json")


def test_multi_objective_status_exposes_pareto_front(
    capsys, tmp_path, schema_path
):
    state = tmp_path / "pareto.json"
    invoke(
        capsys,
        "init",
        "--state",
        str(state),
        "--schema",
        str(schema_path),
        "--experiment-id",
        "automl-pareto",
        "--network-arch",
        "dino",
        "--action",
        "evaluate",
        "--metric",
        "accuracy",
        "--direction",
        "maximize",
        "--objective",
        "accuracy:maximize:1:1",
        "--objective",
        "latency:minimize:1:100",
        "--max-recommendations",
        "4",
    )
    values = [(0.90, 20.0), (0.88, 10.0), (0.92, 30.0), (0.85, 25.0)]
    for index, (accuracy, latency) in enumerate(values):
        rec = recommend(capsys, state)
        job_id = f"pareto-{index}"
        bind(capsys, state, rec["id"], job_id)
        invoke(
            capsys,
            "report",
            "--state",
            str(state),
            "--rec-id",
            rec["id"],
            "--outcome",
            "SUCCESS",
            "--job-id",
            job_id,
            "--metric-value",
            f"accuracy={accuracy}",
            "--metric-value",
            f"latency={latency}",
            "--artifact-uri",
            f"/results/{job_id}/metrics.json",
        )
    status = invoke(capsys, "status", "--state", str(state))
    assert {row["rec_id"] for row in status["pareto_front"]} == {
        "rec-0000",
        "rec-0001",
        "rec-0002",
    }


@pytest.mark.parametrize("algorithm", ["llm", "hybrid", "autoresearch"])
def test_llm_algorithms_use_skill_owned_openai_compatible_client(
    capsys, tmp_path, schema_path, llm_server, monkeypatch, algorithm
):
    endpoint, requests = llm_server
    monkeypatch.setenv("AUTOML_LLM_API_KEY", "secret-that-must-not-be-persisted")
    state = tmp_path / f"{algorithm}.json"
    extra = [
        "--llm-endpoint",
        endpoint,
        "--llm-model",
        "local-test-model",
        "--llm-max-retries",
        "1",
        "--max-recommendations",
        "1",
        "--max-experiments",
        "1",
    ]
    if algorithm == "autoresearch":
        extra.extend(["--evolvable-text-parameter", "model.variant"])
    init_algorithm(capsys, state, schema_path, algorithm, *extra)
    rec = recommend(capsys, state)

    assert rec["metadata"]["llm"]["fallback"] is False
    assert rec["metadata"]["llm"]["usage"]["total_tokens"] == 15
    assert requests[0]["path"] == "/v1/chat/completions"
    assert "secret-that-must-not-be-persisted" not in state.read_text()
    if algorithm == "hybrid":
        assert rec["metadata"]["sub_algorithm"] == "bfbo"
    if algorithm == "autoresearch":
        assert rec["parameters"]["model.variant"] == "generated architecture instruction"


def test_llm_failure_is_explicit_and_uses_deterministic_fallback(
    capsys, tmp_path, schema_path
):
    state = tmp_path / "llm-fallback.json"
    init_algorithm(
        capsys,
        state,
        schema_path,
        "llm",
        "--llm-endpoint",
        "http://127.0.0.1:1/v1",
        "--llm-model",
        "unavailable",
        "--llm-timeout",
        "0.01",
        "--llm-max-retries",
        "1",
        "--max-recommendations",
        "1",
    )
    rec = recommend(capsys, state)
    assert rec["metadata"]["llm"]["fallback"] is True
    assert rec["metadata"]["proposer"] == "llm_proposal_fallback"
    assert rec["metadata"]["llm"]["error"]


def test_wandb_offline_tracking_logs_result_without_sdk_state_dependency(
    capsys, tmp_path, schema_path, monkeypatch
):
    calls = []

    class Run:
        def __init__(self):
            self.summary = {}

        def log(self, values, step):
            calls.append(("log", values, step))

        def finish(self):
            calls.append(("finish",))

    def wandb_init(**kwargs):
        calls.append(("init", kwargs))
        return Run()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=wandb_init))
    state = tmp_path / "wandb.json"
    invoke(
        capsys,
        "init",
        "--state",
        str(state),
        "--schema",
        str(schema_path),
        "--experiment-id",
        "automl-wandb",
        "--network-arch",
        "dino",
        "--metric",
        "val_loss",
        "--direction",
        "minimize",
        "--max-recommendations",
        "1",
        "--wandb",
        "--wandb-mode",
        "offline",
        "--wandb-project",
        "automl-regression",
    )
    rec = recommend(capsys, state)
    bind(capsys, state, rec["id"], "wandb-job")
    output = report_success(
        capsys,
        state,
        rec["id"],
        "wandb-job",
        0.25,
        "/results/wandb/model.pth",
    )

    assert calls[0][0] == "init"
    assert calls[0][1]["mode"] == "offline"
    assert calls[1][1]["objective/val_loss"] == pytest.approx(0.25)
    assert calls[-1] == ("finish",)
    assert output["recommendation"]["metadata"]["wandb"]["logged"] is True


def test_autoresearch_receives_sanitized_evaluation_feedback(
    capsys, tmp_path, schema_path, llm_server
):
    endpoint, requests = llm_server
    state = tmp_path / "autoresearch-feedback.json"
    init_algorithm(
        capsys,
        state,
        schema_path,
        "autoresearch",
        "--llm-endpoint",
        endpoint,
        "--llm-model",
        "local-test-model",
        "--llm-max-retries",
        "1",
        "--max-experiments",
        "2",
        "--evolvable-text-parameter",
        "model.variant",
    )
    first = recommend(capsys, state)
    bind(capsys, state, first["id"], "autoresearch-0")
    feedback = tmp_path / "feedback.json"
    feedback.write_text(
        json.dumps(
            {
                "failures": [
                    {
                        "id": "private-item",
                        "video": "/private/video.mp4",
                        "gold": "Yes",
                        "generated_output": "No",
                        "comment": "The response ignored temporal order.",
                    }
                ]
            }
        )
    )
    invoke(
        capsys,
        "report",
        "--state",
        str(state),
        "--rec-id",
        first["id"],
        "--outcome",
        "SUCCESS",
        "--job-id",
        "autoresearch-0",
        "--metric",
        "0.5",
        "--checkpoint-uri",
        "/results/autoresearch-0/model.pth",
        "--feedback",
        str(feedback),
    )
    recommend(capsys, state)

    history = requests[-1]["prompt"]["history"]
    serialized = json.dumps(history)
    assert "ignored temporal order" in serialized
    assert "generated_output" in serialized
    assert "private-item" not in serialized
    assert "/private/video.mp4" not in serialized
    assert '"gold"' not in serialized


def test_asha_refills_capacity_and_promotes_asynchronously(
    capsys, tmp_path, schema_path
):
    state = tmp_path / "asha-async.json"
    init_algorithm(
        capsys,
        state,
        schema_path,
        "asha",
        "--max-epochs",
        "2",
        "--reduction-factor",
        "2",
        "--max-trials",
        "3",
        "--max-concurrent",
        "2",
    )
    initial = invoke(capsys, "recommend", "--state", str(state))["recommendations"]
    for index, rec in enumerate(initial):
        bind(capsys, state, rec["id"], f"asha-initial-{index}")

    report_success(
        capsys,
        state,
        initial[0]["id"],
        "asha-initial-0",
        0.8,
        "/results/asha-0/model.pth",
    )
    replacement = recommend(capsys, state)
    assert replacement["rung"] == 0
    assert replacement["metadata"]["asynchronous_trial"] is True
    bind(capsys, state, replacement["id"], "asha-replacement")

    report_success(
        capsys,
        state,
        initial[1]["id"],
        "asha-initial-1",
        0.4,
        "/results/asha-1/model.pth",
    )
    promoted = recommend(capsys, state)
    assert promoted["rung"] == 1
    assert promoted["parent_rec_id"] == initial[1]["id"]
    assert promoted["metadata"]["asynchronous_promotion"] is True


def test_hyperband_es_observations_request_platform_cancellation(
    capsys, tmp_path, schema_path
):
    state = tmp_path / "hes-observe.json"
    init_algorithm(
        capsys,
        state,
        schema_path,
        "hyperband_es",
        "--max-epochs",
        "8",
        "--reduction-factor",
        "2",
        "--max-recommendations",
        "2",
        "--max-concurrent",
        "2",
        "--min-early-stop-epochs",
        "3",
        "--early-stop-threshold",
        "0",
    )
    recs = invoke(capsys, "recommend", "--state", str(state))["recommendations"]
    assert len(recs) == 2
    for index, rec in enumerate(recs):
        bind(capsys, state, rec["id"], f"hes-{index}")
    invoke(
        capsys,
        "observe",
        "--state",
        str(state),
        "--rec-id",
        recs[0]["id"],
        "--job-id",
        "hes-0",
        "--step",
        "1",
        "--metric-value",
        "val_loss=0.1",
    )
    outputs = []
    for step_number, value in ((1, 1.0), (2, 2.0), (3, 3.0)):
        outputs.append(
            invoke(
                capsys,
                "observe",
                "--state",
                str(state),
                "--rec-id",
                recs[1]["id"],
                "--job-id",
                "hes-1",
                "--step",
                str(step_number),
                "--metric-value",
                f"val_loss={value}",
            )
        )
    assert outputs[-1]["should_cancel"] is True
    assert outputs[-1]["recommendation"]["decision"] == "EARLY_STOP"


@pytest.mark.parametrize("algorithm", ["hyperband", "bohb", "dehb", "hyperband_es"])
def test_hyperband_family_executes_all_brackets(
    capsys, tmp_path, schema_path, algorithm
):
    state = tmp_path / f"{algorithm}-brackets.json"
    init_algorithm(
        capsys,
        state,
        schema_path,
        algorithm,
        "--max-epochs",
        "4",
        "--reduction-factor",
        "2",
        "--max-concurrent",
        "4",
        "--min-points-in-model",
        "2",
        "--kde-samples",
        "16",
    )
    launched = []
    while True:
        result = invoke(capsys, "recommend", "--state", str(state))
        if not result["ready"]:
            assert result["status"] == "COMPLETE"
            break
        recs = result["recommendations"]
        launched.extend(recs)
        for rec in recs:
            job_id = f"{algorithm}-{rec['id']}"
            bind(capsys, state, rec["id"], job_id)
            report_success(
                capsys,
                state,
                rec["id"],
                job_id,
                1.0 / (rec["index"] + 1),
                f"/results/{job_id}/model.pth",
            )

    assert {rec["bracket"] for rec in launched} == {0, 1}
    assert max(rec["budget"] for rec in launched) == 4
    assert len(launched) == 11
    if algorithm == "bohb":
        assert any(
            rec["metadata"].get("proposer") == "bohb_tpe_density_ratio"
            for rec in launched
        )
    if algorithm == "dehb":
        assert any(
            rec["metadata"].get("proposer") == "differential_evolution"
            for rec in launched
        )


def test_asha_all_failed_trials_complete_without_hanging(
    capsys, tmp_path, schema_path
):
    state = tmp_path / "asha-failed.json"
    init_algorithm(
        capsys,
        state,
        schema_path,
        "asha",
        "--max-epochs",
        "2",
        "--reduction-factor",
        "2",
        "--max-trials",
        "2",
        "--min-top-configs",
        "1",
        "--max-concurrent",
        "2",
    )
    recs = invoke(capsys, "recommend", "--state", str(state))["recommendations"]
    for index, rec in enumerate(recs):
        job_id = f"asha-failed-{index}"
        bind(capsys, state, rec["id"], job_id)
        invoke(
            capsys,
            "report",
            "--state",
            str(state),
            "--rec-id",
            rec["id"],
            "--outcome",
            "ERR_PROGRAM",
            "--job-id",
            job_id,
            "--message",
            "intentional failure",
        )
    terminal = invoke(capsys, "recommend", "--state", str(state))
    assert terminal["status"] == "COMPLETE"
    assert terminal["best"] is None


def test_asha_without_trial_cap_runs_until_final_rung_target(
    capsys, tmp_path, schema_path
):
    state = tmp_path / "asha-open-ended.json"
    init_algorithm(
        capsys,
        state,
        schema_path,
        "asha",
        "--max-recommendations",
        "1",
        "--max-epochs",
        "2",
        "--reduction-factor",
        "2",
        "--min-top-configs",
        "2",
        "--max-concurrent",
        "2",
    )
    while True:
        result = invoke(capsys, "recommend", "--state", str(state))
        if not result["ready"]:
            assert result["status"] == "COMPLETE"
            break
        for rec in result["recommendations"]:
            job_id = f"asha-open-{rec['id']}"
            bind(capsys, state, rec["id"], job_id)
            report_success(
                capsys,
                state,
                rec["id"],
                job_id,
                1.0 / (rec["index"] + 1),
                f"/results/{job_id}/model.pth",
            )
    payload = json.loads(state.read_text())
    final_rung = [
        rec
        for rec in payload["recommendations"]
        if rec["rung"] == 1 and rec["state"] == "SUCCEEDED"
    ]
    assert len(final_rung) >= 2
    assert len(payload["recommendations"]) > payload["max_recommendations"]


def test_hyperband_es_maximize_direction_promotes_larger_metric(
    capsys, tmp_path, schema_path
):
    state = tmp_path / "hes-maximize.json"
    invoke(
        capsys,
        "init",
        "--state",
        str(state),
        "--schema",
        str(schema_path),
        "--experiment-id",
        "hes-maximize",
        "--network-arch",
        "cosmos-rl",
        "--algorithm",
        "hyperband_es",
        "--metric",
        "val_accuracy",
        "--direction",
        "maximize",
        "--candidate-count",
        "64",
        "--max-epochs",
        "2",
        "--reduction-factor",
        "2",
        "--max-concurrent",
        "2",
    )
    recs = invoke(capsys, "recommend", "--state", str(state))["recommendations"]
    for index, rec in enumerate(recs):
        job_id = f"hes-maximize-{index}"
        bind(capsys, state, rec["id"], job_id)
        report_success(
            capsys,
            state,
            rec["id"],
            job_id,
            [0.1, 0.9][index],
            f"/results/{job_id}/model.pth",
        )
    promoted = recommend(capsys, state)
    assert promoted["parent_rec_id"] == recs[1]["id"]


def test_dehb_accepts_zero_as_successful_population_metric(
    capsys, tmp_path, schema_path
):
    state = tmp_path / "dehb-zero.json"
    init_algorithm(
        capsys,
        state,
        schema_path,
        "dehb",
        "--max-epochs",
        "4",
        "--reduction-factor",
        "2",
        "--max-concurrent",
        "4",
    )
    recs = invoke(capsys, "recommend", "--state", str(state))["recommendations"]
    assert len(recs) == 4
    for rec in recs:
        job_id = f"dehb-zero-{rec['id']}"
        bind(capsys, state, rec["id"], job_id)
        report_success(
            capsys,
            state,
            rec["id"],
            job_id,
            0.0,
            f"/results/{job_id}/model.pth",
        )
    promoted = invoke(capsys, "recommend", "--state", str(state))["recommendations"]
    assert len(promoted) == 2
    assert all(rec["parent_rec_id"] for rec in promoted)
