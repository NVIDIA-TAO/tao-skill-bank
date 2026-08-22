from pathlib import Path

import numpy as np

from build_deft_snapshots import materialize_snapshot, select_deft_samples


def test_selection_preserves_hard_anchors_and_propagates_neighbors():
    names = ["a.png", "b.png", "c.png", "d.png"]
    embeddings = np.asarray(
        [[1.0, 0.0], [0.99, 0.10], [0.0, 1.0], [0.10, 0.99]],
        dtype=np.float64,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    difficulties = np.asarray([1.0, 0.1, 0.8, 0.2])
    rows, metadata = select_deft_samples(
        names,
        embeddings,
        difficulties,
        anchor_fraction=0.25,
        duplicate_fraction=0.50,
        neighbors_per_anchor=1,
    )
    by_name = {row["filename"]: row for row in rows}
    assert by_name["a.png"]["selection_reason"] == "hard_oof_anchor"
    assert by_name["b.png"]["selection_reason"] == "feature_neighbor_propagation"
    assert metadata == {
        "sample_count": 4,
        "anchor_count": 1,
        "duplicate_count": 2,
        "resulting_train_count": 6,
        "anchor_fraction": 0.25,
        "duplicate_fraction": 0.50,
        "neighbors_per_anchor": 1,
        "priority_weights": {"oof_difficulty_rank": 0.70, "neighbor_vote_rank": 0.30},
    }


def test_materialization_pairs_image_mask_duplicates_and_excludes_val_selection(tmp_path: Path):
    source = tmp_path / "source"
    for category, split in (("images", "train"), ("masks", "train"), ("images", "val"), ("masks", "val"), ("images", "test")):
        (source / category / split).mkdir(parents=True)
    for name in ("a.png", "b.png"):
        (source / "images/train" / name).write_bytes(b"image")
        (source / "masks/train" / name).write_bytes(b"mask")
    rows = [
        {"filename": "a.png", "selected": True},
        {"filename": "b.png", "selected": False},
    ]
    metadata = {"resulting_train_count": 3}
    output = tmp_path / "snapshot"
    manifest = materialize_snapshot(
        "fan_base", rows, metadata, {}, source, output
    )
    assert sorted(p.name for p in (output / "images/train").iterdir()) == [
        "a.png", "a__deft001.png", "b.png"
    ]
    assert sorted(p.name for p in (output / "masks/train").iterdir()) == [
        "a.png", "a__deft001.png", "b.png"
    ]
    assert (output / "images/val").is_symlink()
    assert manifest["validation_used_for_selection"] is False
