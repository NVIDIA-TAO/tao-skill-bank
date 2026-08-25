# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "skills"
    / "platform"
    / "tao-run-on-slurm"
    / "scripts"
    / "slurm_stage_tree.py"
)
SPEC = importlib.util.spec_from_file_location("slurm_stage_tree", SCRIPT)
assert SPEC and SPEC.loader
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)

SUBSET_SCRIPT = (
    REPO
    / "skills"
    / "applications"
    / "tao-run-deft-iaa"
    / "scripts"
    / "prepare_slurm_dataset_subset.py"
)
SUBSET_SPEC = importlib.util.spec_from_file_location("prepare_slurm_dataset_subset", SUBSET_SCRIPT)
assert SUBSET_SPEC and SUBSET_SPEC.loader
subset = importlib.util.module_from_spec(SUBSET_SPEC)
SUBSET_SPEC.loader.exec_module(subset)


def test_inventory_is_deterministic_and_tracks_regular_and_symlink_content(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    payload = source / "nested" / "sample.bin"
    payload.write_bytes(b"abc")
    (source / "link").symlink_to("nested/sample.bin")

    first = stage.inventory_tree(source)
    assert first == stage.inventory_tree(source)
    assert first["counts"] == {"directories": 1, "regular_files": 1, "symlinks": 1}
    assert first["regular_bytes"] == 3

    payload.write_bytes(b"abd")
    assert stage.inventory_tree(source)["sha256"] != first["sha256"]
    payload.write_bytes(b"abc")
    (source / "link").unlink()
    (source / "link").symlink_to("nested/missing.bin")
    assert stage.inventory_tree(source)["sha256"] != first["sha256"]


def test_incremental_symlink_gate_accepts_internal_and_rejects_escape_or_dangling(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    target = source / "target.bin"
    target.write_bytes(b"target")
    link = source / "link.bin"
    link.symlink_to("target.bin")
    assert stage._internal_symlinks_only(source) is True
    link.unlink()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link.symlink_to(outside)
    assert stage._internal_symlinks_only(source) is False
    link.unlink()
    link.symlink_to("missing.bin")
    assert stage._internal_symlinks_only(source) is False


@pytest.mark.parametrize("target", ["relative/path", "/", "/home", "/lustre", "/a/b/../c"])
def test_remote_target_rejects_broad_or_ambiguous_paths(target):
    with pytest.raises(ValueError):
        stage._safe_remote_target(target)


def test_receipt_reuse_requires_exact_core_and_archive_digest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    inventory = {"sha256": "a" * 64, "counts": {}, "regular_bytes": 0}
    core = stage._receipt_core(
        source.resolve(),
        stage._safe_remote_target("/lustre/team/user/run/data/dataset"),
        inventory,
    )
    receipt = {**core, "archive_sha256": "b" * 64, "completed_at": "now"}
    assert stage.receipt_reusable(receipt, core)
    assert not stage.receipt_reusable({**receipt, "target": "/lustre/team/other"}, core)
    assert not stage.receipt_reusable({key: value for key, value in receipt.items() if key != "archive_sha256"}, core)


def test_receipt_reuse_tolerates_relocated_action_manifest_with_same_inventory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    inventory = {"sha256": "a" * 64, "counts": {}, "regular_bytes": 0}
    core = stage._receipt_core(
        source.resolve(),
        stage._safe_remote_target("/lustre/team/user/run/data/dataset"),
        inventory,
    )
    receipt = {
        **core,
        "manifest_sha256": "b" * 64,
        "archive_sha256": "c" * 64,
        "completed_at": "then",
    }
    assert stage.receipt_reusable(receipt, core)
    assert not stage.receipt_reusable(
        {**receipt, "inventory": {**inventory, "sha256": "d" * 64}}, core
    )


def test_stage_reuses_remote_receipt_without_streaming(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "item").write_text("value", encoding="utf-8")
    target = "/lustre/team/user/run/data/dataset"
    receipt_path = tmp_path / "control" / "stage.json"
    inventory = stage.inventory_tree(source)
    core = stage._receipt_core(source.resolve(), stage._safe_remote_target(target), inventory)
    remote = {**core, "archive_sha256": "c" * 64, "completed_at": "then"}
    monkeypatch.setattr(stage, "_read_remote_receipt", lambda *_: remote)
    monkeypatch.setattr(
        stage,
        "_stream_archive",
        lambda *_: pytest.fail("a matching remote receipt must skip transfer"),
    )

    result = stage.stage_tree(
        source_raw=str(source),
        login_raw="user@login.example",
        target_raw=target,
        receipt_raw=str(receipt_path),
    )
    assert result["status"] == "reused"
    assert json.loads(receipt_path.read_text()) == remote


def test_stage_promotes_only_after_stream_and_confirms_receipt(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "item").write_bytes(b"value")
    receipt_path = tmp_path / "stage.json"
    calls = []
    confirmed = {"value": None}

    def read_remote(*_):
        return confirmed["value"]

    def stream(*args):
        calls.append(("stream", args[2]))
        return "d" * 64

    def promote(login, target, temp, backup, receipt):
        calls.append(("promote", target, temp, backup))
        confirmed["value"] = receipt

    monkeypatch.setattr(stage, "_read_remote_receipt", read_remote)
    monkeypatch.setattr(stage, "_stream_archive", stream)
    monkeypatch.setattr(stage, "_promote", promote)

    result = stage.stage_tree(
        source_raw=str(source),
        login_raw="user@login.example",
        target_raw="/lustre/team/user/run/data/dataset",
        receipt_raw=str(receipt_path),
    )
    assert result["status"] == "staged"
    assert [item[0] for item in calls] == ["stream", "promote"]
    assert ".staging-" in str(calls[0][1])
    assert json.loads(receipt_path.read_text())["archive_sha256"] == "d" * 64


def test_stage_uses_copy_on_write_incremental_transport_for_changed_existing_tree(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "item").write_bytes(b"new")
    target = stage._safe_remote_target("/lustre/team/user/run/data/dataset")
    receipt_path = tmp_path / "stage.json"
    remote = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "target": str(target),
        "inventory": {
            "sha256": "a" * 64,
            "counts": {"directories": 0, "regular_files": 1, "symlinks": 0},
            "regular_bytes": 3,
        },
        "archive_sha256": "b" * 64,
        "completed_at": "then",
    }
    confirmed = {"value": remote}
    calls = []
    monkeypatch.setattr(stage, "_read_remote_receipt", lambda *_: confirmed["value"])
    monkeypatch.setattr(
        stage, "_stream_archive",
        lambda *_: pytest.fail("valid existing tree must not use the full tar stream"),
    )

    def incremental(login, actual_source, actual_target, temp, inventory_sha256):
        calls.append((login, actual_source, actual_target, temp, inventory_sha256))
        return "c" * 64

    def promote(_login, _target, _temp, _backup, receipt):
        confirmed["value"] = receipt

    monkeypatch.setattr(stage, "_seed_and_rsync", incremental)
    monkeypatch.setattr(stage, "_promote", promote)
    result = stage.stage_tree(
        source_raw=str(source), login_raw="user@login.example",
        target_raw=str(target), receipt_raw=str(receipt_path),
        incremental_existing=True,
    )
    assert result["status"] == "staged"
    assert len(calls) == 1
    receipt = json.loads(receipt_path.read_text())
    assert receipt["transport"] == "rsync-cow-v1"
    assert receipt["archive_sha256"] == "c" * 64


def test_stage_incremental_flag_falls_back_to_tar_without_valid_seed(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "item").write_bytes(b"value")
    confirmed = {"value": None}
    calls = []
    monkeypatch.setattr(stage, "_read_remote_receipt", lambda *_: confirmed["value"])
    monkeypatch.setattr(
        stage, "_seed_and_rsync",
        lambda *_: pytest.fail("missing remote receipt cannot seed incrementally"),
    )
    monkeypatch.setattr(
        stage, "_stream_archive", lambda *_args: calls.append("tar") or "d" * 64
    )
    monkeypatch.setattr(
        stage, "_promote",
        lambda _login, _target, _temp, _backup, receipt: confirmed.update(value=receipt),
    )
    stage.stage_tree(
        source_raw=str(source), login_raw="user@login.example",
        target_raw="/lustre/team/user/run/data/dataset",
        receipt_raw=str(tmp_path / "stage.json"), incremental_existing=True,
    )
    assert calls == ["tar"]
    assert confirmed["value"]["transport"] == "tar-stream-v1"


def test_incremental_transport_uses_one_receiver_acknowledged_checksum_pass(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "item").write_bytes(b"value")
    remote_calls = []
    rsync_calls = []
    monkeypatch.setattr(
        stage, "_ssh_capture",
        lambda _login, command: remote_calls.append(command)
        or stage.subprocess.CompletedProcess([], 0, "", ""),
    )

    def run(argv, **_kwargs):
        rsync_calls.append(argv)
        return stage.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(stage.subprocess, "run", run)
    digest = stage._seed_and_rsync(
        "user@login.example", source,
        stage._safe_remote_target("/lustre/team/user/run/data/dataset"),
        stage._safe_remote_target("/lustre/team/user/run/data/.dataset.staging"),
        "a" * 64,
    )
    assert len(remote_calls) == 1
    assert len(rsync_calls) == 1
    assert "--checksum" in rsync_calls[0]
    assert "--delete" in rsync_calls[0]
    assert "--dry-run" not in rsync_calls[0]
    assert digest == hashlib.sha256(b"rsync-cow-v1\0" + b"a" * 64).hexdigest()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evaluate_fixture(tmp_path):
    data_parent = tmp_path / "data"
    dataset = data_parent / "dataset"
    results = tmp_path / "results" / "run_1"
    (dataset / "images").mkdir(parents=True)
    (dataset / "captions").mkdir()
    (dataset / "images_raw" / "eval").mkdir(parents=True)
    (results / "iaa_splits").mkdir(parents=True)
    (results / "zs" / "specs").mkdir(parents=True)
    (dataset / "images_raw" / "eval" / "source.jpg").write_bytes(b"image")
    (dataset / "images" / "eval_000.jpg").symlink_to("../images_raw/eval/source.jpg")
    (dataset / "captions" / "eval_000.txt").write_text("caption", encoding="utf-8")
    (results / "iaa_splits" / "eval_list.txt").write_text("eval_000.jpg\n", encoding="utf-8")
    mapping = {
        "image_dir": str(dataset / "images"),
        "caption_dir": str(dataset / "captions"),
        "image_list_file": "/results/iaa_splits/eval_list.txt",
        "caption_file_suffix": ".txt",
    }
    config = {
        "dataset": {
            "train": {"datasets": [mapping]},
            "val": {"datasets": [mapping]},
        },
        "evaluate": {"datasets": [mapping]},
    }
    import yaml

    (results / "zs" / "specs" / "eval_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    request = {
        "platform": "slurm",
        "name": "evaluate",
        "results_dir": str(results),
        "mounts": [
            {"source": str(results), "target": "/results", "read_only": False},
            {"source": str(data_parent), "target": "/data", "read_only": True},
        ],
        "spec_bundle": {
            "command": "clip",
            "args": ["evaluate", "-e", "/results/zs/specs/eval_config.yaml"],
        },
    }
    request["request_sha256"] = _digest(request)
    request_path = results / "zs" / "evaluate" / "evaluate.action.json"
    request_path.parent.mkdir()
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return request_path, data_parent, dataset, results


def test_evaluate_manifest_selects_only_action_required_leaves(tmp_path):
    request_path, data_parent, dataset, _ = _evaluate_fixture(tmp_path)
    manifest = subset.build_evaluate_manifest(request_path)
    assert manifest["image_list_count"] == 1
    assert manifest["entries"] == [
        "dataset/captions/eval_000.txt",
        "dataset/images/eval_000.jpg",
        "dataset/images_raw/eval/source.jpg",
    ]
    assert manifest["entry_count"] == 3
    entries, digest = stage._manifest_entries(
        _write_manifest(tmp_path / "manifest.json", manifest), data_parent.resolve()
    )
    inventory = stage.inventory_paths(data_parent.resolve(), entries)
    assert digest == manifest["manifest_sha256"]
    assert inventory["counts"] == {"directories": 0, "regular_files": 2, "symlinks": 1}
    assert inventory["regular_bytes"] == len(b"image") + len(b"caption")


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _snapshot_request(tmp_path, field):
    source = tmp_path / field
    source.mkdir()
    (source / "nested").mkdir()
    (source / "entry.py").write_bytes(b"one")
    (source / "nested" / "entry.py").write_bytes(b"two")
    entries = [
        {
            "path": path.relative_to(source).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(source.rglob("*.py"))
    ]
    manifest = {"root": str(source.resolve()), "entries": entries}
    manifest["sha256"] = _digest({"entries": entries})
    request = {
        "workflow": "tao-run-deft-iaa",
        "platform": "slurm",
        field: manifest,
    }
    request["request_sha256"] = _digest(request)
    path = tmp_path / f"{field}.action.json"
    path.write_text(json.dumps(request))
    return source.resolve(), path, manifest


@pytest.mark.parametrize("field", ("controller_snapshot", "patches_snapshot"))
def test_slurm_action_snapshot_validates_complete_per_file_manifest(tmp_path, field):
    source, request, manifest = _snapshot_request(tmp_path, field)
    entries, digest = stage._action_snapshot_entries(request, source, field)
    assert entries == [entry["path"] for entry in manifest["entries"]]
    assert digest == manifest["sha256"]


@pytest.mark.parametrize("field", ("controller_snapshot", "patches_snapshot"))
@pytest.mark.parametrize("mutation", ("changed", "extra", "missing"))
def test_slurm_action_snapshot_rejects_tree_mutation(tmp_path, field, mutation):
    source, request, _ = _snapshot_request(tmp_path, field)
    if mutation == "changed":
        (source / "entry.py").write_bytes(b"changed")
    elif mutation == "extra":
        (source / "extra.py").write_bytes(b"extra")
    else:
        (source / "entry.py").unlink()
    with pytest.raises(ValueError, match="changed|digest|missing|unrelated"):
        stage._action_snapshot_entries(request, source, field)


def test_evaluate_manifest_fails_closed_on_unexpected_dataset_mapping(tmp_path):
    request_path, _, dataset, results = _evaluate_fixture(tmp_path)
    config_path = results / "zs" / "specs" / "eval_config.yaml"
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace(str(dataset / "captions"), str(dataset / "other"), 1))
    with pytest.raises(ValueError, match="unexpected names|one exact dataset mapping"):
        subset.build_evaluate_manifest(request_path)


@pytest.mark.parametrize("name,relative", [
    ("viz_weak_embed", "iter_1/embeddings/viz_weak/input.parquet"),
    ("viz_mined_embed", "iter_1/mining/mined_unique_images.parquet"),
    ("viz_previous_embed", "iter_1/embeddings/previous/prev_pool.parquet"),
])
def test_visualization_manifest_selects_only_requested_images_and_targets(tmp_path, name, relative):
    import pyarrow as pa
    import pyarrow.parquet as pq

    results = tmp_path / "results"
    data = tmp_path / "data"
    dataset = data / "dataset"
    (dataset / "images_raw").mkdir(parents=True)
    (dataset / "images").mkdir()
    selected = dataset / "images_raw" / "selected.jpg"
    unrelated = dataset / "images_raw" / "unrelated.jpg"
    selected.write_bytes(b"selected")
    unrelated.write_bytes(b"unrelated")
    link = dataset / "images" / "selected.jpg"
    link.symlink_to("../images_raw/selected.jpg")
    input_path = results / relative
    input_path.parent.mkdir(parents=True)
    filepaths = [str(link)]
    if name == "viz_previous_embed":
        generated = results / "iter_1" / "datagen" / "dataset" / "images" / "generated.jpg"
        generated.parent.mkdir(parents=True)
        generated.write_bytes(b"generated")
        filepaths.append("/results/iter_1/datagen/dataset/images/generated.jpg")
    pq.write_table(pa.table({"filepath": filepaths}), input_path)
    request = {
        "platform": "slurm", "name": name,
        "mounts": [
            {"source": str(results), "target": "/results", "read_only": False},
            {"source": str(data), "target": "/data", "read_only": True},
        ],
        "spec_bundle": {"command": "embedding", "args": [
            "image_embeddings", "-e", "/specs/image_embed_spec.yaml",
            f"input_parquet=/results/{relative}", "output_parquet=/results/out.parquet",
        ]},
    }
    request["request_sha256"] = _digest(request)
    request_path = results / f"{name}.action.json"
    request_path.write_text(json.dumps(request))
    manifest = subset.build_image_embedding_manifest(request_path)
    assert manifest["image_count"] == len(filepaths)
    assert manifest["entries"] == [
        "dataset/images/selected.jpg", "dataset/images_raw/selected.jpg",
    ]
    assert "dataset/images_raw/unrelated.jpg" not in manifest["entries"]


def test_visualization_manifest_rejects_outside_and_noncanonical_inputs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    results = tmp_path / "results"
    data = tmp_path / "data"
    input_path = results / "iter_1/embeddings/viz_weak/input.parquet"
    input_path.parent.mkdir(parents=True)
    outside = tmp_path / "eval-leak.jpg"
    outside.write_bytes(b"outside")
    pq.write_table(pa.table({"filepath": [str(outside)]}), input_path)
    request = {
        "platform": "slurm", "name": "viz_weak_embed",
        "mounts": [{"source": str(results), "target": "/results"}, {"source": str(data), "target": "/data"}],
        "spec_bundle": {"command": "embedding", "args": ["image_embeddings", "-e", "/specs/x.yaml", "input_parquet=/results/iter_1/embeddings/viz_weak/input.parquet"]},
    }
    data.mkdir()
    request["request_sha256"] = _digest(request)
    request_path = results / "request.json"
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="outside the approved dataset parent"):
        subset.build_image_embedding_manifest(request_path)


def _train_fixture(tmp_path):
    import yaml

    data_parent = tmp_path / "data"
    dataset = data_parent / "dataset"
    results = tmp_path / "results" / "run_1"
    for directory in (
        dataset / "images",
        dataset / "captions",
        dataset / "images_raw" / "train",
        dataset / "images_raw" / "val",
        results / "iter_1" / "mining",
        results / "iter_1" / "datagen" / "dataset" / "images",
        results / "iter_1" / "datagen" / "dataset" / "captions",
        results / "iter_1" / "specs",
        results / "iaa_splits",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (dataset / "images_raw" / "train" / "source.jpg").write_bytes(b"train")
    (dataset / "images_raw" / "val" / "source.jpg").write_bytes(b"val")
    (dataset / "images" / "train_0001.jpg").symlink_to("../images_raw/train/source.jpg")
    (dataset / "images" / "val_0001.jpg").symlink_to("../images_raw/val/source.jpg")
    (dataset / "captions" / "train_0001.txt").write_text("train caption")
    (dataset / "captions" / "val_0001.txt").write_text("val caption")
    (results / "iter_1" / "mining" / "mined_image_list.txt").write_text(
        "train_0001.jpg\n", encoding="utf-8"
    )
    (results / "iter_1" / "mining" / "mined_pairs.json").write_text(
        "[]\n", encoding="utf-8"
    )
    generated = results / "iter_1" / "datagen" / "dataset"
    (generated / "sdg_image_list.txt").write_text("sdg_0001.jpg\n")
    (generated / "sdg_pairs.json").write_text("[]\n")
    (results / "iaa_splits" / "val_list.txt").write_text("val_0001.jpg\n")
    # This must never be selected merely because it is another split file.
    (results / "iaa_splits" / "eval_list.txt").write_text("eval_9999.jpg\n")
    original = {
        "image_dir": str(dataset / "images"),
        "caption_dir": str(dataset / "captions"),
        "image_list_file": "/results/iter_1/mining/mined_image_list.txt",
        "caption_file_suffix": ".txt",
        "train_pairs_file": "/results/iter_1/mining/mined_pairs.json",
    }
    generated_mapping = {
        "image_dir": "/results/iter_1/datagen/dataset/images",
        "caption_dir": "/results/iter_1/datagen/dataset/captions",
        "image_list_file": "/results/iter_1/datagen/dataset/sdg_image_list.txt",
        "caption_file_suffix": ".txt",
        "train_pairs_file": "/results/iter_1/datagen/dataset/sdg_pairs.json",
    }
    validation = {
        "image_dir": str(dataset / "images"),
        "caption_dir": str(dataset / "captions"),
        "image_list_file": "/results/iaa_splits/val_list.txt",
        "caption_file_suffix": ".txt",
    }
    config = {"dataset": {"train": {"datasets": [original, generated_mapping]},
                           "val": {"datasets": [validation]}}}
    config_path = results / "iter_1" / "specs" / "train_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    request = {
        "platform": "slurm",
        "name": "train",
        "mounts": [
            {"source": str(results), "target": "/results", "read_only": False},
            {"source": str(data_parent), "target": "/data", "read_only": True},
        ],
        "spec_bundle": {
            "command": "clip",
            "args": ["train", "-e", "/results/iter_1/specs/train_config.yaml"],
        },
    }
    request["request_sha256"] = _digest(request)
    request_path = results / "iter_1" / "train" / "train.action.json"
    request_path.parent.mkdir()
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return request_path, data_parent, dataset, results


def test_train_manifest_selects_mined_and_validation_leaves_only(tmp_path):
    request_path, data_parent, _, _ = _train_fixture(tmp_path)
    manifest = subset.build_train_manifest(request_path)
    assert manifest["training_image_count"] == 1
    assert manifest["validation_image_count"] == 1
    assert manifest["entries"] == [
        "dataset/captions/train_0001.txt",
        "dataset/captions/val_0001.txt",
        "dataset/images/train_0001.jpg",
        "dataset/images/val_0001.jpg",
        "dataset/images_raw/train/source.jpg",
        "dataset/images_raw/val/source.jpg",
    ]
    assert all("eval_9999" not in item for item in manifest["entries"])
    entries, _ = stage._manifest_entries(
        _write_manifest(tmp_path / "train-manifest.json", manifest), data_parent.resolve()
    )
    inventory = stage.inventory_paths(data_parent.resolve(), entries)
    assert inventory["counts"] == {"directories": 0, "regular_files": 4, "symlinks": 2}


def test_train_manifest_deduplicates_continual_original_inputs(tmp_path):
    request_path, _, _, results = _train_fixture(tmp_path)
    config_path = results / "iter_1" / "specs" / "train_config.yaml"
    import yaml
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["train"]["datasets"].insert(
        1, dict(config["dataset"]["train"]["datasets"][0])
    )
    config_path.write_text(yaml.safe_dump(config))
    manifest = subset.build_train_manifest(request_path)
    assert manifest["training_image_count"] == 2
    assert manifest["entry_count"] == 6


def test_train_manifest_rejects_eval_split_and_outside_leaf(tmp_path):
    request_path, _, dataset, results = _train_fixture(tmp_path)
    config_path = results / "iter_1" / "specs" / "train_config.yaml"
    import yaml
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["val"]["datasets"][0]["image_list_file"] = "/results/iaa_splits/eval_list.txt"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="validation image list is not a canonical"):
        subset.build_train_manifest(request_path)

    config["dataset"]["val"]["datasets"][0]["image_list_file"] = "/results/iaa_splits/val_list.txt"
    config_path.write_text(yaml.safe_dump(config))
    (dataset / "images" / "train_0001.jpg").unlink()
    (dataset / "images" / "train_0001.jpg").symlink_to(tmp_path / "outside.jpg")
    (tmp_path / "outside.jpg").write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes the approved dataset parent"):
        subset.build_train_manifest(request_path)


def test_manifest_digest_and_paths_fail_closed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "item").write_text("value", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "entries": ["item"],
        "entry_count": 1,
    }
    payload["manifest_sha256"] = _digest(payload)
    path = _write_manifest(tmp_path / "manifest.json", payload)
    assert stage._manifest_entries(path, source.resolve())[0] == ["item"]
    payload["entries"] = ["../item"]
    _write_manifest(path, payload)
    with pytest.raises(ValueError, match="digest mismatch"):
        stage._manifest_entries(path, source.resolve())


def test_manifest_inventory_requires_each_symlink_target(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "target").write_text("value", encoding="utf-8")
    (source / "link").symlink_to("target")
    with pytest.raises(ValueError, match="omits the in-source target"):
        stage.inventory_paths(source.resolve(), ["link"])


def test_manifest_archive_stream_contains_only_selected_leaves(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "selected").write_text("selected", encoding="utf-8")
    (source / "unrelated").write_text("unrelated", encoding="utf-8")
    remote_parent = tmp_path / "remote"
    remote_parent.mkdir()
    remote_temp = stage._safe_remote_target(str(remote_parent / ".dataset.staging-deadbeef"))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text("#!/bin/sh\nexec bash -c \"$4\"\n", encoding="utf-8")
    fake_ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    stale = Path(str(remote_temp))
    (stale / "locked").mkdir(parents=True)
    (stale / "locked" / "old").write_text("old", encoding="utf-8")
    (stale / "locked" / "old").chmod(0o400)
    (stale / "locked").chmod(0o500)

    digest = stage._stream_archive(
        "test@login", source, remote_temp, ["selected"]
    )
    assert len(digest) == 64
    assert (Path(str(remote_temp)) / "selected").read_text() == "selected"
    assert not (Path(str(remote_temp)) / "unrelated").exists()


def test_promote_removes_read_only_stale_and_replaced_trees(monkeypatch, tmp_path):
    remote_parent = tmp_path / "remote"
    remote_parent.mkdir()
    target = stage._safe_remote_target(str(remote_parent / "dataset"))
    temporary = stage._safe_remote_target(str(remote_parent / ".dataset.staging-deadbeef"))
    backup = stage._safe_remote_target(str(remote_parent / ".dataset.previous-deadbeef"))
    for root, value in ((Path(str(target)), "old"), (Path(str(backup)), "stale")):
        (root / "locked").mkdir(parents=True)
        (root / "locked" / "item").write_text(value, encoding="utf-8")
        (root / "locked" / "item").chmod(0o400)
        (root / "locked").chmod(0o500)
    Path(str(temporary)).mkdir()
    (Path(str(temporary)) / "new").write_text("new", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text("#!/bin/sh\nexec bash -c \"$4\"\n", encoding="utf-8")
    fake_ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    stage._promote(
        "test@login", target, temporary, backup,
        {"schema_version": 1, "archive_sha256": "a" * 64},
    )

    assert (Path(str(target)) / "new").read_text(encoding="utf-8") == "new"
    assert not Path(str(backup)).exists()
