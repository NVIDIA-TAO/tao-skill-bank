#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DEFT CR ITS mining workflow helper scripts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from assemble_train_annotations import assemble_annotations
from build_llava_from_mining import build_llava_records
from compute_bcq_accuracy_metrics import compute_metrics, compute_metrics_file, extract_yes_no
from cosmos_embed_outputs_to_parquet import consolidate_embeddings, raw_embeddings_dataframe
from inspect_gap_analysis import write_status
from log_stage import append_stage, next_seq, read_valid_events
from prepare_gap_analysis_predictions import annotation_video_lookup, prepare_predictions, write_predictions
from record_mined_paths import record_mined_paths
from resume_position import INITIAL_STAGES, ITERATION_STAGES, resume_position
from setup_cosmos_reason_stage import generate_evaluate_toml, generate_train_toml, latest_safetensors_checkpoint
from setup_for_cosmos_embed import collect_embedding_inputs, stage_provided_embedding_parquet
from setup_iteration_mining import (
    build_text_target,
    build_video_target,
    filter_source_pool,
    setup_iteration_mining,
    text_target_dataframe,
)
from summarize_bcq_accuracy_metrics import write_report
from verify_workflow_yaml import validate_cosmos_embed_template
from workflow_common import dataset_modalities, optional_embedding_parquets


def write_json(path: Path, payload: object) -> None:
    """Write compact JSON for tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL rows for tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_yaml(path: Path, text: str) -> None:
    """Write a YAML/TOML fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_mining_targets_and_specs(tmp: Path) -> None:
    """Text targets keep failed questions; video targets dedupe failed videos."""
    workspace = tmp / "workspace"
    run_dir = workspace / "results" / "run1"
    specs = workspace / "specs"
    media = workspace / "data" / "kpi" / "media"
    train_media = workspace / "data" / "train" / "media"
    media.mkdir(parents=True)
    train_media.mkdir(parents=True)
    video_a = media / "a.mp4"
    video_b = media / "b.mp4"
    video_a.write_text("a", encoding="utf-8")
    video_b.write_text("b", encoding="utf-8")
    q0 = run_dir / "cosmos_embed_output" / "kpi" / "results" / "text" / "questions" / "q_00000.txt"
    q1 = run_dir / "cosmos_embed_output" / "kpi" / "results" / "text" / "questions" / "q_00001.txt"
    q0.parent.mkdir(parents=True)
    q0.write_text("Is there a collision?", encoding="utf-8")
    q1.write_text("Is the road empty?", encoding="utf-8")
    kpi_lookup = run_dir / "cosmos_embed_output" / "kpi" / "lookup.parquet"
    pd.DataFrame(
        [
            {"filepath": str(q0), "annotation_id": "kpi-collision", "video_path": str(video_a), "item_index": 0, "question": "Is there a collision?", "answer": "yes"},
            {"filepath": str(q1), "annotation_id": "kpi-empty", "video_path": str(video_a), "item_index": 1, "question": "Is the road empty?", "answer": "no"},
        ]
    ).to_parquet(kpi_lookup, index=False)
    kpi_embeddings = run_dir / "embedding_parquets" / "kpi" / "embeddings.parquet"
    train_embeddings = run_dir / "embedding_parquets" / "train" / "embeddings.parquet"
    train_q = tmp / "train_q.txt"
    train_video_path = train_media / "source.mp4"
    for path, rows in (
        (
            kpi_embeddings,
            [
                {"filepath": str(q0), "embedding": [1.0, 0.0], "modality": "text"},
                {"filepath": str(q1), "embedding": [0.0, 1.0], "modality": "text"},
                {"filepath": str(video_a), "embedding": [1.0, 0.0], "modality": "video"},
                {"filepath": str(video_b), "embedding": [0.0, 1.0], "modality": "video"},
            ],
        ),
        (
            train_embeddings,
            [
                {"filepath": str(train_q), "embedding": [1.0, 0.0], "modality": "text"},
                {"filepath": str(train_video_path), "embedding": [1.0, 0.0], "modality": "video"},
            ],
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    gaps = run_dir / "iter_1" / "gaps" / "kpi_gaps.jsonl"
    write_jsonl(
        gaps,
        [
            {"video_id": str(video_a), "question": "<video>\nIs there a collision? Answer with yes or no.", "ground_truth": "yes", "response": "no"},
            {"video_id": str(video_a), "question": "Is the road empty?", "ground_truth": "no", "response": "yes"},
        ],
    )
    text_target = run_dir / "iter_1" / "text_target.parquet"
    video_target = run_dir / "iter_1" / "video_target.parquet"
    assert build_text_target(gaps, kpi_embeddings, kpi_lookup, text_target) == 2
    assert build_video_target(gaps, kpi_embeddings, video_target) == 1
    assert set(pd.read_parquet(text_target)["filepath"]) == {str(q0), str(q1)}
    assert pd.read_parquet(video_target)["filepath"].tolist() == [str(video_a)]

    template = specs / "nearest_neighbors.yaml"
    write_yaml(template, "source_parquet: null\ntarget_parquet: null\noutput_parquet: null\ntopn: 3\nknn_metric: cosine\n")
    workflow = specs / "workflow.yaml"
    write_yaml(
        workflow,
        f"""
run:
  name: run1
  max_iterations: 1
kpi_dataset:
  annotations_path: {workspace}/data/kpi/annotations.json
  media_dir: {media}
train_dataset:
  annotations_path: {workspace}/data/train/annotations.json
  media_dir: {train_media}
cosmos_reason:
  baseline_model_path: {workspace}/model/baseline
  base_evaluate_toml: {workspace}/specs/cr_base_evaluate.toml
  base_train_toml: {workspace}/specs/cr_base_train.toml
mining:
  embeddings_spec_template: {workspace}/specs/cosmos_embed.yaml
  embeddings_modality: both
  mining_spec_template: {template}
""",
    )
    (workspace / "data" / "kpi").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "train").mkdir(parents=True, exist_ok=True)
    write_json(workspace / "data" / "kpi" / "annotations.json", [])
    write_json(workspace / "data" / "train" / "annotations.json", [])
    (workspace / "model" / "baseline").mkdir(parents=True)
    for name in ("cr_base_evaluate.toml", "cr_base_train.toml", "cosmos_embed.yaml"):
        write_yaml(workspace / "specs" / name, "")
    generated_specs = setup_iteration_mining(workspace, workflow, run_dir, 1, gaps)
    assert [path.name for path in generated_specs] == ["nearest_neighbors.yaml"]
    assert all(path.is_file() for path in generated_specs)
    mining_dir = run_dir / "iter_1" / "mining"
    filtered_source = pd.read_parquet(mining_dir / "filtered_source.parquet")
    assert set(filtered_source["filepath"]) == {str(train_q), str(train_video_path)}
    combined_target = pd.read_parquet(mining_dir / "target.parquet")
    assert combined_target["modality"].value_counts().to_dict() == {"text": 2, "video": 1}

    mined = mining_dir / "mined_neighbors.parquet"
    pd.DataFrame([{"filepath": str(train_q)}, {"filepath": str(train_video_path)}]).to_parquet(
        mined,
        index=False,
    )
    mined_log = run_dir / "mining" / "mined_paths_log.parquet"
    assert record_mined_paths([mined], mined_log) == 2
    filtered = run_dir / "iter_2" / "mining" / "filtered_source.parquet"
    assert filter_source_pool(train_embeddings, mined_log, filtered) == 0
    assert pd.read_parquet(filtered).empty


def test_build_llava_from_mining_and_assemble(tmp: Path) -> None:
    """Mined video joins expand to all source annotations and final merge dedupes ids."""
    train_lookup = tmp / "train_lookup.parquet"
    video = str(tmp / "source.mp4")
    qfile = str(tmp / "q0.txt")
    pd.DataFrame(
        [
            {"filepath": qfile, "annotation_id": "source-collision", "video_path": video, "item_index": 0, "question": "Is there a collision?", "answer": "yes"},
            {"filepath": str(tmp / "q1.txt"), "annotation_id": "source-raining", "video_path": video, "item_index": 1, "question": "Is it raining?", "answer": "no"},
        ]
    ).to_parquet(train_lookup, index=False)
    train_embeddings = tmp / "train_embeddings.parquet"
    pd.DataFrame(
        [
            {"filepath": qfile, "embedding": [1.0, 0.0], "modality": "text"},
            {"filepath": video, "embedding": [1.0, 0.0], "modality": "video"},
        ]
    ).to_parquet(train_embeddings, index=False)
    neighbors = tmp / "neighbors.parquet"
    pd.DataFrame([{"filepath": qfile}, {"filepath": video}]).to_parquet(neighbors, index=False)

    mined_records = build_llava_records(neighbors, train_embeddings, train_lookup)
    assert [record["id"] for record in mined_records] == ["source-collision", "source-raining"]

    previous = tmp / "previous.json"
    mined_annotations = tmp / "mined_annotations.json"
    write_json(previous, [{"id": "seed", "video": "seed.mp4", "conversations": []}])
    write_json(mined_annotations, mined_records)
    first_iter_merged = assemble_annotations(None, [mined_annotations])
    assert [record["id"] for record in first_iter_merged] == ["source-collision", "source-raining"]

    merged = assemble_annotations(previous, [mined_annotations])
    assert [record["id"] for record in merged] == ["seed", "source-collision", "source-raining"]


def test_consolidated_embedding_parquets(tmp: Path) -> None:
    """Generated modality outputs consolidate with row-level modality provenance."""
    output_dir = tmp / "cosmos_embed_output" / "train"
    text_inference = output_dir / "results" / "text" / "inference"
    video_inference = output_dir / "results" / "video" / "inference"
    text_inference.mkdir(parents=True)
    video_inference.mkdir(parents=True)
    pd.DataFrame([{"filepath": "q0.txt", "question": "Question zero"}]).to_parquet(
        output_dir / "lookup.parquet", index=False
    )
    np.save(text_inference / "text.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    write_json(
        text_inference / "text_embeddings.json",
        {"npy_file": "text.npy", "results": [{"text": "Question zero", "npy_row": 0}]},
    )
    np.save(video_inference / "video.npy", np.asarray([[0.0, 1.0]], dtype=np.float32))
    write_json(
        video_inference / "video_embeddings.json",
        {"npy_file": "video.npy", "results": [{"video_path": "v0.mp4", "npy_row": 0}]},
    )
    output_path = consolidate_embeddings(output_dir, tmp / "embedding_parquets" / "train", "both")
    combined = pd.read_parquet(output_path)
    assert combined[["filepath", "modality"]].to_dict("records") == [
        {"filepath": "q0.txt", "modality": "text"},
        {"filepath": "v0.mp4", "modality": "video"},
    ]
    assert dataset_modalities("kpi", "text") == ["text"]
    assert dataset_modalities("kpi", "both") == ["text", "video"]
    assert dataset_modalities("train", "text") == ["text", "video"]


def test_combined_embedding_parquet_inputs(tmp: Path) -> None:
    """Workflow inputs accept complete combined dataset parquets and reject stale fields."""
    workspace = tmp / "workspace"
    inputs = workspace / "data" / "embeddings"
    inputs.mkdir(parents=True)
    kpi = inputs / "kpi.parquet"
    train = inputs / "train.parquet"
    pd.DataFrame(
        [{"filepath": "kpi-q.txt", "embedding": [1.0, 0.0], "modality": "text"}]
    ).to_parquet(kpi, index=False)
    original_question = inputs / "train-q.txt"
    original_question.write_text("Training question", encoding="utf-8")
    pd.DataFrame(
        [
            {"filepath": str(original_question), "embedding": [1.0, 0.0], "modality": "text"},
            {"filepath": "train-v.mp4", "embedding": [0.0, 1.0], "modality": "video"},
        ]
    ).to_parquet(train, index=False)

    provided = optional_embedding_parquets(
        {"embedding_parquets": {"kpi": str(kpi), "train": str(train)}},
        workspace,
        "text",
    )
    assert provided == {"kpi": kpi, "train": train}
    current_question = workspace / "results" / "run" / "cosmos_embed_output" / "train" / "questions" / "q_00000.txt"
    lookup = workspace / "results" / "run" / "cosmos_embed_output" / "train" / "lookup.parquet"
    current_question.parent.mkdir(parents=True)
    current_question.write_text("Training question", encoding="utf-8")
    pd.DataFrame(
        [{"filepath": str(current_question), "question": "Training question"}]
    ).to_parquet(lookup, index=False)
    staged = stage_provided_embedding_parquet(
        train,
        workspace / "results" / "run" / "embedding_parquets" / "train",
        lookup,
    )
    staged_rows = pd.read_parquet(staged)
    assert not staged.is_symlink()
    assert staged_rows["filepath"].tolist() == [str(current_question), "train-v.mp4"]

    try:
        optional_embedding_parquets({"text_embeddings": {}}, workspace, "text")
    except ValueError as exc:
        assert "removed per-modality mining fields" in str(exc)
    else:
        raise AssertionError("stale per-modality embedding fields must fail")

    try:
        optional_embedding_parquets(
            {"embedding_parquets": {"kpi": str(kpi), "train": str(kpi)}},
            workspace,
            "text",
        )
    except ValueError as exc:
        assert "must contain exactly modalities ['text', 'video']" in str(exc)
    else:
        raise AssertionError("train embedding parquet without video rows must fail")


def test_text_embeddings_join_by_content(tmp: Path) -> None:
    """Reordered and duplicate text results map by content, not NPY row order."""
    output_dir = tmp / "cosmos_embed_output" / "kpi"
    inference_dir = output_dir / "results" / "text" / "inference"
    inference_dir.mkdir(parents=True)
    q0 = str(tmp / "q_00000.txt")
    q1 = str(tmp / "q_00001.txt")
    q2 = str(tmp / "q_00002.txt")
    collision_question = "Is there a collision?"
    rain_question = "Is it raining?"
    pd.DataFrame(
        [
            {"filepath": q0, "question": collision_question},
            {"filepath": q1, "question": rain_question},
            {"filepath": q2, "question": collision_question},
        ]
    ).to_parquet(output_dir / "lookup.parquet", index=False)
    np.save(
        inference_dir / "text_embeddings.npy",
        np.asarray([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float32),
    )
    write_json(
        inference_dir / "text_embeddings.json",
        {
            "npy_file": "text_embeddings.npy",
            "results": [
                {"text": rain_question, "npy_row": 2},
                {"text": collision_question, "npy_row": 0},
                {"text": collision_question, "npy_row": 1},
            ],
        },
    )

    frame = raw_embeddings_dataframe("text", output_dir)
    assert frame["filepath"].tolist() == [q1, q0, q2]
    assert frame["embedding"].tolist() == [[30.0, 0.0], [10.0, 0.0], [20.0, 0.0]]

    metadata_path = inference_dir / "text_embeddings.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["results"].pop()
    write_json(metadata_path, metadata)
    try:
        raw_embeddings_dataframe("text", output_dir)
    except ValueError as exc:
        assert "text lookup rows were not returned" in str(exc)
    else:
        raise AssertionError("missing Cosmos Embed text results must fail")


def test_embedding_gpu_template_and_required_run_dir(tmp: Path) -> None:
    """Preflight exposes GPU count and embedding setup requires the initialized run."""
    template = tmp / "cosmos_embed.yaml"
    write_yaml(template, "inference:\n  num_gpus: 8\n")
    assert validate_cosmos_embed_template(template) == 8
    write_yaml(template, "inference:\n  num_gpus: 0\n")
    try:
        validate_cosmos_embed_template(template)
    except ValueError as exc:
        assert "inference.num_gpus must be a positive integer" in str(exc)
    else:
        raise AssertionError("non-positive Cosmos Embed GPU count must fail")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "setup_for_cosmos_embed.py"),
            "--workspace",
            str(tmp),
            "--workflow-yaml",
            str(tmp / "workflow.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--run-dir" in result.stderr


def test_cosmos_reason_stage_configs(tmp: Path) -> None:
    """Train/evaluate TOMLs patch only run-specific paths and checkpoints."""
    workspace = tmp / "workspace"
    run_dir = workspace / "results" / "run1"
    specs = workspace / "specs"
    kpi_media = workspace / "data" / "kpi" / "media"
    train_media = workspace / "data" / "train" / "media"
    model = workspace / "model" / "baseline"
    for path in (kpi_media, train_media, model, specs):
        path.mkdir(parents=True, exist_ok=True)
    kpi_ann = workspace / "data" / "kpi" / "annotations.json"
    train_ann = workspace / "data" / "train" / "annotations.json"
    iter_train_ann = run_dir / "iter_1" / "train" / "train_annotations.json"
    write_json(kpi_ann, [])
    write_json(train_ann, [])
    write_json(iter_train_ann, [])
    eval_toml = specs / "cr_base_evaluate.toml"
    train_toml = specs / "cr_base_train.toml"
    write_yaml(eval_toml, 'results_dir = ""\n[dataset]\nannotation_path = ""\nmedia_dir = ""\n[model]\nmodel_name = ""\n')
    write_yaml(
        train_toml,
        'results_dir = ""\n[train]\noutput_dir = ""\n[policy]\nmodel_name_or_path = ""\n[custom.train_dataset]\nannotation_path = ""\nmedia_path = ""\n[custom.val_dataset]\nannotation_path = ""\nmedia_path = ""\n',
    )
    workflow = specs / "workflow.yaml"
    write_yaml(
        workflow,
        f"""
run:
  name: run1
  max_iterations: 1
kpi_dataset:
  annotations_path: {kpi_ann}
  media_dir: {kpi_media}
train_dataset:
  annotations_path: {train_ann}
  media_dir: {train_media}
cosmos_reason:
  baseline_model_path: {model}
  base_evaluate_toml: {eval_toml}
  base_train_toml: {train_toml}
mining:
  embeddings_spec_template: {specs}/cosmos_embed.yaml
  embeddings_modality: text
  mining_spec_template: {specs}/nearest_neighbors.yaml
""",
    )
    out_eval = generate_evaluate_toml(
        workspace,
        workflow,
        run_dir,
        output_dir=run_dir / "baseline" / "evaluate",
        checkpoint_path=str(model),
    )
    out_train = generate_train_toml(
        workspace,
        workflow,
        run_dir,
        iteration=1,
        train_annotations=iter_train_ann,
        checkpoint_path=str(model),
    )
    assert "model_name" in out_eval.read_text(encoding="utf-8")
    train_text = out_train.read_text(encoding="utf-8")
    assert "[custom.train_dataset]" in train_text
    assert str(iter_train_ann) in train_text

    ckpt = run_dir / "iter_1" / "train" / "20260101000000" / "safetensors" / "epoch_2"
    ckpt.mkdir(parents=True)
    assert latest_safetensors_checkpoint(run_dir / "iter_1" / "train") == ckpt


def test_gap_status_and_explicit_run_dir(tmp: Path) -> None:
    """Gap status handles absent/filled output and init honors an explicit run directory."""
    missing_gaps = tmp / "missing" / "kpi_gaps.jsonl"
    status_path = tmp / "missing" / "gap_status.json"
    assert write_status(missing_gaps, status_path)["weak_sample_count"] == 0
    gaps = tmp / "gaps" / "kpi_gaps.jsonl"
    write_jsonl(gaps, [{"video_id": "/data/a.mp4"}, {"video_id": "/data/b.mp4"}])
    status = write_status(gaps, tmp / "gaps" / "gap_status.json")
    assert status["has_gaps"] is True
    assert status["weak_sample_count"] == 2

    workspace = tmp / "workspace"
    workflow = workspace / "specs" / "workflow.yaml"
    explicit_run_dir = workspace / "results" / "kfp-selected-run"
    write_yaml(
        workflow,
        """
run:
  name: ignored-by-explicit-run-dir
  max_iterations: 1
mining:
  embeddings_modality: text
cosmos_reason:
  continual_model: false
""",
    )
    command = [
        sys.executable,
        str(SCRIPT_DIR / "init_deft_cr_mining_state.py"),
        "--workspace",
        str(workspace),
        "--workflow-yaml",
        str(workflow),
        "--run-dir",
        str(explicit_run_dir),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    state = json.loads((explicit_run_dir / "deft_state.json").read_text(encoding="utf-8"))
    assert state["run_dir"] == str(explicit_run_dir)


def test_prepare_gap_analysis_predictions(tmp: Path) -> None:
    """Prediction annotation ids resolve to media paths without deduping questions."""
    media_dir = tmp / "kpi"
    videos_dir = media_dir / "videos"
    videos_dir.mkdir(parents=True)
    video_a = videos_dir / "a.mp4"
    video_b = videos_dir / "b.mp4"
    video_a.write_text("a", encoding="utf-8")
    video_b.write_text("b", encoding="utf-8")
    annotations_path = tmp / "annotations.json"
    annotations = [
        {"id": "a-collision", "video": "videos/a.mp4", "conversations": []},
        {"id": "a-weather", "video": "videos/a.mp4", "conversations": []},
        {"id": "b-collision", "video": str(video_b), "conversations": []},
    ]
    write_json(annotations_path, annotations)
    results_path = tmp / "results.json"
    predictions = [
        {"video_id": "a-collision", "question": "Collision?", "response": "no", "gt": "yes"},
        {"video_id": "a-weather", "question": "Raining?", "response": "yes", "gt": "no"},
        {"video_id": "videos/b.mp4", "question": "Collision?", "response": "no", "gt": "yes"},
    ]
    write_json(results_path, predictions)
    output_path = tmp / "gaps" / "predictions.json"
    prepared = write_predictions(results_path, annotations_path, media_dir, output_path)
    assert [row["video_id"] for row in prepared] == [str(video_a), str(video_a), str(video_b)]
    assert [row["question"] for row in prepared] == ["Collision?", "Raining?", "Collision?"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == prepared

    conflicting = [
        {"id": "duplicate", "video": "videos/a.mp4"},
        {"id": "duplicate", "video": "videos/b.mp4"},
    ]
    try:
        annotation_video_lookup(conflicting, annotations_path, media_dir)
    except ValueError as exc:
        assert "conflicting video paths" in str(exc)
    else:
        raise AssertionError("conflicting annotation ids must fail")

    lookup, video_paths = annotation_video_lookup(annotations, annotations_path, media_dir)
    try:
        prepare_predictions(
            [{"video_id": "unknown-annotation"}],
            results_path,
            lookup,
            video_paths,
            media_dir,
        )
    except ValueError as exc:
        assert "does not match an annotation id or annotation video path" in str(exc)
    else:
        raise AssertionError("unknown prediction annotation ids must fail")


def test_multi_question_predictions_to_mining_target(tmp: Path) -> None:
    """Two questions for one video survive prediction adaptation and target selection."""
    media_dir = tmp / "kpi"
    media_dir.mkdir()
    video = media_dir / "shared.mp4"
    video.write_text("video", encoding="utf-8")
    annotations_path = tmp / "annotations.json"
    annotations = [
        {
            "id": "shared-collision",
            "video": "shared.mp4",
            "conversations": [
                {"from": "human", "value": "<video>\nIs there a collision?"},
                {"from": "gpt", "value": "yes"},
            ],
        },
        {
            "id": "shared-weather",
            "video": "shared.mp4",
            "conversations": [
                {"from": "human", "value": "<video>\nIs it raining?"},
                {"from": "gpt", "value": "no"},
            ],
        },
    ]
    write_json(annotations_path, annotations)
    output_dir = tmp / "cosmos_embed_output" / "kpi"
    collect_embedding_inputs(annotations_path, media_dir, output_dir)
    lookup_path = output_dir / "lookup.parquet"
    lookup = pd.read_parquet(lookup_path)
    assert lookup["annotation_id"].tolist() == ["shared-collision", "shared-weather"]
    assert lookup["video_path"].tolist() == [str(video), str(video)]

    results_path = tmp / "results.json"
    write_json(
        results_path,
        [
            {"video_id": "shared-collision", "question": "Is there a collision?", "response": "no", "gt": "yes"},
            {"video_id": "shared-weather", "question": "Is it raining?", "response": "yes", "gt": "no"},
        ],
    )
    predictions_path = tmp / "gaps" / "predictions.json"
    prepared = write_predictions(results_path, annotations_path, media_dir, predictions_path)
    gaps_path = tmp / "gaps" / "kpi_gaps.jsonl"
    write_jsonl(gaps_path, prepared)
    embeddings_path = tmp / "embedding_parquets" / "kpi" / "embeddings.parquet"
    embeddings_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"filepath": filepath, "embedding": [float(index), 0.0], "modality": "text"}
            for index, filepath in enumerate(lookup["filepath"].tolist(), start=1)
        ]
    ).to_parquet(embeddings_path, index=False)
    target = text_target_dataframe(gaps_path, embeddings_path, lookup_path)
    assert set(target["filepath"]) == set(lookup["filepath"])


def test_resilient_log_state_and_two_iteration_resume(tmp: Path) -> None:
    """Malformed log lines do not block state refresh or two-iteration resume."""
    run_dir = tmp / "run"
    run_dir.mkdir()
    state_path = run_dir / "deft_state.json"
    log_path = run_dir / "loop_log.jsonl"
    write_json(
        state_path,
        {
            "version": 1,
            "workflow": "tao-run-deft-cr-its-mining",
            "run_dir": str(run_dir),
            "max_iterations": 2,
            "current_iteration": 0,
            "status": "initialized",
            "mine_unique_only": True,
            "baseline_results_json": None,
            "iterations": {},
        },
    )

    append_stage(
        log_path,
        iter_label="initialization",
        stage="baseline_evaluate",
        status="ok",
        summary="baseline complete",
        duration_sec=1,
        artifacts=[str(run_dir / "baseline" / "evaluate" / "results.json")],
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq":')
    assert next_seq(log_path) == 2
    for stage in ("setup_embeddings", "cosmos_embed", "convert_embeddings"):
        append_stage(
            log_path,
            iter_label="initialization",
            stage=stage,
            status="ok",
            summary=f"{stage} complete",
            duration_sec=1,
            artifacts=[],
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    position = resume_position(state, read_valid_events(log_path, warn=False))
    assert position["next_iteration"] == 1
    assert position["next_stage"] == "gap_analysis"

    for iteration in (1, 2):
        for stage in ITERATION_STAGES:
            append_stage(
                log_path,
                iter_label=f"iter_{iteration}",
                stage=stage,
                status="ok",
                summary=f"{stage} complete",
                duration_sec=1,
                artifacts=[],
            )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["current_iteration"] == 2
    assert state["status"] == "running"
    position = resume_position(state, read_valid_events(log_path, warn=False))
    assert position["next_stage"] == "loop_stop"

    append_stage(
        log_path,
        iter_label="workflow",
        stage="loop_stop",
        status="ok",
        summary="max_iterations",
        duration_sec=0,
        artifacts=[],
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert len(read_valid_events(log_path, warn=False)) == 21
    assert resume_position(state, read_valid_events(log_path, warn=False))["next_stage"] is None


def test_resume_stops_after_zero_gap_iteration(tmp: Path) -> None:
    """A crash before loop_stop must not resume zero-gap output into mining."""
    run_dir = tmp / "run"
    run_dir.mkdir()
    state_path = run_dir / "deft_state.json"
    log_path = run_dir / "loop_log.jsonl"
    write_json(
        state_path,
        {
            "status": "initialized",
            "current_iteration": 0,
            "max_iterations": 2,
            "mine_unique_only": True,
            "iterations": {},
        },
    )
    for stage in INITIAL_STAGES:
        append_stage(
            log_path,
            iter_label="initialization",
            stage=stage,
            status="ok",
            summary=f"{stage} complete",
            duration_sec=1,
            artifacts=[],
        )
    append_stage(
        log_path,
        iter_label="iter_1",
        stage="gap_analysis",
        status="ok",
        summary="no weak samples",
        duration_sec=1,
        artifacts=[],
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    missing_status = resume_position(state, read_valid_events(log_path, warn=False), run_dir)
    assert missing_status["next_stage"] == "gap_analysis"
    assert "no gap_status.json" in missing_status["reason"]

    write_json(
        run_dir / "iter_1" / "gaps" / "gap_status.json",
        {"has_gaps": False, "weak_sample_count": 0},
    )
    position = resume_position(state, read_valid_events(log_path, warn=False), run_dir)
    assert position["next_iteration"] is None
    assert position["next_stage"] == "loop_stop"
    assert position["reason"] == "iteration 1 has no weak samples"


def test_bcq_accuracy_metrics_and_report(tmp: Path) -> None:
    """Binary parsing, confusion counts, and the run report share one contract."""
    assert extract_yes_no("YES.") == "yes"
    assert extract_yes_no("No, there is no collision.") == "no"
    assert extract_yes_no("Yes, there is no visible collision between those vehicles.") == "yes"
    assert extract_yes_no("The final answer is no, not yes.") == "no"
    assert extract_yes_no("It may be yes or no.") is None
    assert extract_yes_no("Yes or no") is None

    records = [
        {"response": "yes.", "gt": "yes"},
        {"response": "No collision is visible.", "gt": "no."},
        {"answer": "The final answer is yes.", "ground_truth": "no"},
        {"response": "no", "gt": "yes"},
        {"response": "Yes, there is no collision between those vehicles.", "gt": "yes"},
        {"response": "unclear", "gt": "no"},
    ]
    source = tmp / "results.json"
    write_json(source, records)
    metrics = compute_metrics(records, source)
    assert metrics["true_positives"] == 2
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["unparseable_predictions"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5

    run_dir = tmp / "run"
    baseline_metrics = run_dir / "baseline" / "evaluate" / "bcq_accuracy_metrics.json"
    iteration_metrics = run_dir / "iter_1" / "evaluate" / "bcq_accuracy_metrics.json"
    compute_metrics_file(source, baseline_metrics)
    compute_metrics_file(source, iteration_metrics)
    report_path = run_dir / "bcq_accuracy_report.md"
    summary_path = run_dir / "bcq_accuracy_summary.json"
    evaluations = write_report(run_dir, report_path, summary_path)
    assert [item["evaluation"] for item in evaluations] == ["Baseline", "Iteration 1"]
    report = report_path.read_text(encoding="utf-8")
    assert "| Baseline | 50.00% | 50.00% | 1 | 1 | 1 | 6 |" in report
    assert "| Iteration 1 | 50.00% | 50.00% | 1 | 1 | 1 | 6 |" in report
    assert len(json.loads(summary_path.read_text(encoding="utf-8"))["evaluations"]) == 2

    try:
        compute_metrics([{"response": "yes", "gt": "unknown"}], source)
    except ValueError as exc:
        assert "ground truth does not contain one clear yes/no label" in str(exc)
    else:
        raise AssertionError("unparseable ground truth must fail")


def main() -> int:
    """Run tests without requiring pytest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        test_mining_targets_and_specs(tmp)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_build_llava_from_mining_and_assemble(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_consolidated_embedding_parquets(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_combined_embedding_parquet_inputs(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_text_embeddings_join_by_content(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_embedding_gpu_template_and_required_run_dir(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_cosmos_reason_stage_configs(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_gap_status_and_explicit_run_dir(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_prepare_gap_analysis_predictions(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_multi_question_predictions_to_mining_target(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_resilient_log_state_and_two_iteration_resume(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_resume_stops_after_zero_gap_iteration(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_bcq_accuracy_metrics_and_report(Path(tmpdir))
    print("test_deft_cr_mining_helpers.py: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
