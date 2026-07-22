# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from gepa_step import (  # noqa: E402
    GEPAutoPrompter,
    GEPAReflectionLM,
    RoutedTAOActionBatchRunner,
    TAOActionBatchRunner,
    TAOGEPAAdapter,
)


def test_reflection_lm_preserves_multimodal_messages_and_system_prompt():
    calls = []

    class Client:
        def chat(self, messages, json_mode=False):
            calls.append((messages, json_mode))
            return SimpleNamespace(ok=True, content="proposal", error=None)

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Review the failure."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
            ],
        }
    ]
    result = GEPAReflectionLM(Client(), system_prompt="Stay general.")(prompt)
    assert result == "proposal"
    assert calls[0][0] == [
        {"role": "system", "content": "Stay general."},
        *prompt,
    ]


def test_reflection_lm_surfaces_client_failure():
    client = SimpleNamespace(
        chat=lambda messages, json_mode=False: SimpleNamespace(
            ok=False,
            content="",
            error="endpoint unavailable",
        )
    )
    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        GEPAReflectionLM(client)("prompt")


def test_action_and_routed_batch_runners_apply_candidates_without_mutation():
    base = {"dataset": {"system_prompt": "seed"}, "vision": {"nframes": 8}}
    calls = []

    def evaluate(specs, items):
        calls.append((specs, items))
        return [item["id"] for item in items]

    runner = TAOActionBatchRunner(
        base,
        evaluate,
        value_coercers={"vision.nframes": int},
    )
    assert runner.run_batch(
        {"dataset.system_prompt": "new", "vision.nframes": "16"},
        [{"id": "a"}],
    ) == ["a"]
    assert calls[0][0]["vision"]["nframes"] == 16
    assert base["vision"]["nframes"] == 8

    routed = RoutedTAOActionBatchRunner(
        SimpleNamespace(
            run_batch=lambda candidate, items: [
                f"{item['id']}:{candidate['vision.nframes']}" for item in items
            ]
        ),
        route_fn=lambda item: item["route"],
        route_candidates={"hard": {"vision.nframes": 16}},
    )
    assert routed.run_batch(
        {"vision.nframes": 8},
        [
            {"id": "a", "route": "easy"},
            {"id": "b", "route": "hard"},
            {"id": "c", "route": "easy"},
        ],
    ) == ["a:8", "b:16", "c:8"]

    unaligned = RoutedTAOActionBatchRunner(
        SimpleNamespace(run_batch=lambda candidate, items: []),
        route_fn=lambda item: "all",
        route_candidates={},
    )
    with pytest.raises(ValueError, match="returned 0 outputs for 1 input items"):
        unaligned.run_batch({}, [{"id": "a"}])


def test_adapter_builds_aligned_leak_free_reflection_records():
    adapter = TAOGEPAAdapter(
        SimpleNamespace(run_batch=lambda candidate, items: ["Yes", "No"]),
        lambda output, gold: (
            float(output == gold),
            {
                "comment": "Re-check temporal order.",
                "video_id": "private-video",
                "gold": gold,
            },
            None,
        ),
    )
    items = [
        {"id": "private-a", "video": "/secret/a.mp4", "query": "Stop?", "gold": "Yes"},
        {"id": "private-b", "video": "/secret/b.mp4", "query": "Turn?", "gold": "Yes"},
    ]
    evaluated = adapter.evaluate(
        items,
        {"dataset.system_prompt": "seed"},
        capture_traces=True,
    )
    records = adapter.make_reflective_dataset(
        {"dataset.system_prompt": "seed"},
        evaluated,
        ["dataset.system_prompt"],
    )["dataset.system_prompt"]
    assert evaluated.scores == [1.0, 0.0]
    assert records[0]["Inputs"] == {"query": "Stop?"}
    serialized = repr(records)
    assert "private-a" not in serialized
    assert "private-video" not in serialized
    assert "/secret" not in serialized
    assert "'gold'" not in serialized


def test_adapter_scopes_visual_evidence_and_validates_alignment():
    evidence_calls = []

    def evidence(item, candidate):
        evidence_calls.append((item["id"], dict(candidate)))
        return {"t=1.0s": "frame-one"}

    adapter = TAOGEPAAdapter(
        SimpleNamespace(run_batch=lambda candidate, items: ["A", "B"]),
        lambda output, gold: (float(output == gold), "review", None),
        fixed_candidate={"vision.nframes": "8"},
        reflection_evidence_fn=evidence,
        vision_components=["dataset.system_prompt"],
    )
    items = [
        {"id": "private-a", "query": "First?", "gold": "A"},
        {"id": "private-b", "query": "Second?", "gold": "A"},
    ]
    evaluated = adapter.evaluate(
        items,
        {"dataset.system_prompt": "prompt"},
        capture_traces=True,
    )
    records = adapter.make_reflective_dataset(
        {"dataset.system_prompt": "prompt"},
        evaluated,
        ["dataset.system_prompt", "text.summary_prompt"],
    )
    assert evidence_calls == [
        (
            "private-b",
            {"vision.nframes": "8", "dataset.system_prompt": "prompt"},
        )
    ]
    assert "Visual Evidence" not in records["dataset.system_prompt"][0]
    assert records["dataset.system_prompt"][1]["Visual Evidence"] == {
        "t=1.0s": "frame-one"
    }
    assert "Visual Evidence" not in records["text.summary_prompt"][1]

    unaligned = TAOGEPAAdapter(
        SimpleNamespace(run_batch=lambda candidate, rows: ["one"]),
        lambda output, gold: 1.0,
    )
    with pytest.raises(ValueError, match="1 outputs for 2 input items"):
        unaligned.evaluate(items, {"prompt": "seed"})


def test_joint_config_choices_are_bounded_and_history_aware():
    adapter = TAOGEPAAdapter(
        SimpleNamespace(run_batch=lambda candidate, items: [candidate["vision.nframes"]]),
        lambda output, gold: float(output == gold),
        config_choices={"vision.nframes": [4, 8, 16]},
    )
    item = {"query": "Frames?", "gold": "8"}
    adapter.evaluate([item], {"vision.nframes": "8"})
    first = adapter.propose_new_texts(
        {"vision.nframes": "8"}, {"vision.nframes": []}, ["vision.nframes"]
    )
    adapter.evaluate([item], {"vision.nframes": first["vision.nframes"]})
    second = adapter.propose_new_texts(
        {"vision.nframes": first["vision.nframes"]},
        {"vision.nframes": []},
        ["vision.nframes"],
    )
    assert first == {"vision.nframes": "4"}
    assert second == {"vision.nframes": "16"}


def test_joint_config_validation_skips_unneeded_visual_evidence():
    evidence_calls = []
    adapter = TAOGEPAAdapter(
        SimpleNamespace(run_batch=lambda candidate, items: ["B"]),
        lambda output, gold: (0.0, "review", None),
        reflection_evidence_fn=lambda item, candidate: evidence_calls.append(item)
        or {"t=0": "frame"},
        vision_components=["system_prompt"],
        config_choices={"vision.nframes": [4, 8, 16]},
    )
    item = {"id": "private", "query": "Question?", "gold": "A"}
    evaluated = adapter.evaluate(
        [item],
        {"system_prompt": "seed", "vision.nframes": "8"},
        capture_traces=True,
    )
    records = adapter.make_reflective_dataset(
        {"system_prompt": "seed", "vision.nframes": "8"},
        evaluated,
        ["vision.nframes"],
    )
    assert evidence_calls == []
    assert "Visual Evidence" not in records["vision.nframes"][0]
    with pytest.raises(ValueError, match="missing config component"):
        adapter.validate_seed_candidate({"system_prompt": "seed"})
    with pytest.raises(ValueError, match="is not in"):
        adapter.validate_seed_candidate(
            {"system_prompt": "seed", "vision.nframes": 32}
        )


def test_gepa_validation_reranking_freezes_winner_before_test(monkeypatch):
    class Runner:
        def __init__(self):
            self.calls = []

        def run_batch(self, candidate, items):
            self.calls.append((dict(candidate), [item["id"] for item in items]))
            return [candidate["prompt"] for _ in items]

    runner = Runner()
    adapter = TAOGEPAAdapter(
        runner,
        lambda output, gold: (float(output == gold), "feedback", None),
        fixed_candidate={"vision.nframes": 8},
    )
    fake_result = SimpleNamespace(
        candidates=[{"prompt": "proxy"}, {"prompt": "official"}],
        val_aggregate_scores=[0.9, 0.8],
        best_idx=0,
    )

    def optimize(**kwargs):
        assert kwargs["max_metric_calls"] == 20
        return fake_result

    monkeypatch.setitem(sys.modules, "gepa", SimpleNamespace(optimize=optimize))

    def aggregate(outputs, _golds):
        score = 0.8 if outputs[0] == "official" else 0.6
        return {"macro_f1": score, "accuracy": score + 0.05}

    result = GEPAutoPrompter(
        adapter,
        reflection_lm=object(),
        aggregate_metric_fn=aggregate,
    ).optimize(
        {"prompt": "seed"},
        [{"id": "train", "gold": "seed"}],
        [{"id": "val", "gold": "Yes"}],
        budget=20,
        testset=[{"id": "test", "gold": "Yes"}],
    )
    assert result.gepa_candidate_index == 0
    assert result.selected_candidate_index == 1
    assert result.best_full_candidate == {"vision.nframes": 8, "prompt": "official"}
    assert result.test_metrics["macro_f1"] == pytest.approx(0.8)
    assert runner.calls[-1][1] == ["test"]


def test_gepa_aggregate_contract_and_cost_tiebreak(monkeypatch):
    fake_result = SimpleNamespace(
        candidates=[
            {"prompt": "seed", "vision.nframes": "16"},
            {"prompt": "seed", "vision.nframes": "8"},
        ],
        val_aggregate_scores=[0.9, 0.8],
        best_idx=0,
    )
    monkeypatch.setitem(
        sys.modules,
        "gepa",
        SimpleNamespace(optimize=lambda **kwargs: fake_result),
    )
    runner = SimpleNamespace(
        run_batch=lambda candidate, items: ["Yes"] * len(items)
    )
    adapter = TAOGEPAAdapter(runner, lambda output, gold: (1.0, "ok", None))
    missing_key = GEPAutoPrompter(
        adapter,
        reflection_lm=object(),
        aggregate_metric_fn=lambda outputs, golds: {"accuracy": 1.0},
        aggregate_metric_key="macro_f1",
    )
    with pytest.raises(KeyError, match="macro_f1"):
        missing_key.optimize(
            {"prompt": "seed", "vision.nframes": "8"},
            [{"query": "train", "gold": "Yes"}],
            [{"query": "val", "gold": "Yes"}],
            budget=1,
        )

    result = GEPAutoPrompter(
        adapter,
        reflection_lm=object(),
        aggregate_metric_fn=lambda outputs, golds: {"accuracy": 1.0},
        aggregate_metric_key="accuracy",
        candidate_cost_fn=lambda candidate: float(candidate["vision.nframes"]),
    ).optimize(
        {"prompt": "seed", "vision.nframes": "8"},
        [{"query": "train", "gold": "Yes"}],
        [{"query": "val", "gold": "Yes"}],
        budget=1,
    )
    assert result.selected_candidate_index == 1
    assert result.candidate_validation_metrics[0]["cost"] == 16.0
    assert result.candidate_validation_metrics[1]["utility"] == 1.0


def test_real_gepa_helper_improves_a_candidate_when_installed():
    pytest.importorskip("gepa")

    class Runner:
        def run_batch(self, candidate, items):
            return [candidate["prompt"] for _ in items]

    class ReflectionLM:
        def __call__(self, _prompt):
            return "good"

    adapter = TAOGEPAAdapter(
        Runner(),
        lambda output, gold: (
            float(output == gold),
            "Use the expected concise answer.",
            None,
        ),
    )
    result = GEPAutoPrompter(
        adapter,
        reflection_lm=ReflectionLM(),
        display_progress_bar=False,
    ).optimize(
        {"prompt": "bad"},
        [{"query": "train", "gold": "good"}],
        [{"query": "validation", "gold": "good"}],
        budget=8,
    )
    assert result.best_candidate == {"prompt": "good"}
    assert result.validation_score == pytest.approx(1.0)
