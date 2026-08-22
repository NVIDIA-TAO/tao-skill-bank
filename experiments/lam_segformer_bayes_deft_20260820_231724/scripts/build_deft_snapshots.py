#!/usr/bin/env python3
"""Build train-only DEFT reweighting snapshots from OOF errors and embeddings."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


BACKBONES = ("fan_base", "fan_large", "mit_b5")


def rank01(values: np.ndarray, names: list[str]) -> np.ndarray:
    """Return deterministic ascending percentile ranks in [0, 1]."""
    if values.ndim != 1 or len(values) != len(names):
        raise ValueError("rank inputs must be aligned one-dimensional arrays")
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), names[i]))
    denominator = max(len(values) - 1, 1)
    result = np.zeros(len(values), dtype=np.float64)
    for rank, index in enumerate(order):
        result[index] = rank / denominator
    return result


def select_deft_samples(
    names: list[str],
    embeddings: np.ndarray,
    difficulties: np.ndarray,
    *,
    anchor_fraction: float,
    duplicate_fraction: float,
    neighbors_per_anchor: int,
) -> tuple[list[dict], dict]:
    """Select hard OOF anchors and propagate their gaps through feature neighbors."""
    count = len(names)
    if count < 2 or embeddings.shape[0] != count or difficulties.shape != (count,):
        raise ValueError("names, embeddings, and difficulties are not aligned")
    if len(set(names)) != count:
        raise ValueError("sample names are not unique")
    if not 0 < anchor_fraction <= duplicate_fraction <= 1:
        raise ValueError("require 0 < anchor_fraction <= duplicate_fraction <= 1")
    if not 1 <= neighbors_per_anchor < count:
        raise ValueError("neighbors_per_anchor must be between 1 and sample_count - 1")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("embeddings must be L2-normalized")

    anchor_count = max(1, math.ceil(count * anchor_fraction))
    duplicate_count = max(anchor_count, round(count * duplicate_fraction))
    hard_order = sorted(range(count), key=lambda i: (-float(difficulties[i]), names[i]))
    anchors = hard_order[:anchor_count]
    anchor_set = set(anchors)

    similarities = embeddings @ embeddings.T
    votes = np.zeros(count, dtype=np.float64)
    vote_sources: list[list[str]] = [[] for _ in range(count)]
    for anchor in anchors:
        neighbor_order = sorted(
            (i for i in range(count) if i != anchor),
            key=lambda i: (-float(similarities[anchor, i]), names[i]),
        )[:neighbors_per_anchor]
        for neighbor in neighbor_order:
            # Cosine similarity is shifted into [0, 1] so every anchor casts a
            # non-negative vote while closer feature neighbors receive more.
            affinity = max(0.0, min(1.0, (float(similarities[anchor, neighbor]) + 1.0) / 2.0))
            votes[neighbor] += float(difficulties[anchor]) * affinity
            vote_sources[neighbor].append(names[anchor])

    difficulty_rank = rank01(difficulties, names)
    vote_rank = rank01(votes, names)
    combined = 0.70 * difficulty_rank + 0.30 * vote_rank
    propagated_order = sorted(
        (i for i in range(count) if i not in anchor_set),
        # The non-anchor quota exists specifically to expand model gaps into
        # their feature neighborhoods, so neighbor votes lead this ordering.
        # Combined difficulty remains the deterministic tie-breaker.
        key=lambda i: (-float(votes[i]), -float(combined[i]), names[i]),
    )
    selected = anchors + propagated_order[: duplicate_count - anchor_count]
    selected_set = set(selected)

    rows = []
    for index in sorted(range(count), key=lambda i: (-float(combined[i]), names[i])):
        rows.append(
            {
                "filename": names[index],
                "selected": index in selected_set,
                "selection_reason": "hard_oof_anchor" if index in anchor_set else (
                    "feature_neighbor_propagation" if index in selected_set else "not_duplicated"
                ),
                "oof_difficulty": float(difficulties[index]),
                "oof_difficulty_rank": float(difficulty_rank[index]),
                "neighbor_vote": float(votes[index]),
                "neighbor_vote_rank": float(vote_rank[index]),
                "combined_priority": float(combined[index]),
                "voting_anchors": sorted(vote_sources[index]),
            }
        )
    metadata = {
        "sample_count": count,
        "anchor_count": anchor_count,
        "duplicate_count": duplicate_count,
        "resulting_train_count": count + duplicate_count,
        "anchor_fraction": anchor_fraction,
        "duplicate_fraction": duplicate_fraction,
        "neighbors_per_anchor": neighbors_per_anchor,
        "priority_weights": {"oof_difficulty_rank": 0.70, "neighbor_vote_rank": 0.30},
    }
    return rows, metadata


def load_inputs(
    backbone: str,
    scores_root: Path,
    embeddings_root: Path,
    expected_count: int,
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    scores_payload = json.loads((scores_root / f"{backbone}_oof_scores.json").read_text())
    if scores_payload.get("validation_used") is not False:
        raise RuntimeError(f"{backbone}: OOF scores do not guarantee validation exclusion")
    score_by_name = {
        row["filename"]: float(row["difficulty"])
        for row in scores_payload["scores"]
    }
    embedding_dir = embeddings_root / backbone
    embedding_meta = json.loads((embedding_dir / "metadata.json").read_text())
    if (
        embedding_meta.get("validation_used") is not False
        or embedding_meta.get("normalized") is not True
    ):
        raise RuntimeError(f"{backbone}: invalid embedding provenance")
    payload = np.load(embedding_dir / "embeddings.npz")
    names = [str(name) for name in payload["names"].tolist()]
    embeddings = payload["embeddings"].astype(np.float64, copy=False)
    if len(names) != expected_count or set(names) != set(score_by_name):
        raise RuntimeError(
            f"{backbone}: expected {expected_count} aligned samples, "
            f"found embeddings={len(names)} scores={len(score_by_name)}"
        )
    difficulties = np.asarray([score_by_name[name] for name in names], dtype=np.float64)
    provenance = {
        "oof_scores": str(scores_root / f"{backbone}_oof_scores.json"),
        "embeddings": str(embedding_dir / "embeddings.npz"),
        "mean_oof_miou": scores_payload.get("mean_oof_miou"),
        "embedding_dimension": embedding_meta.get("dimension"),
    }
    return names, embeddings, difficulties, provenance


def symlink_checked(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"missing source file: {source}")
    destination.symlink_to(source)


def materialize_snapshot(
    backbone: str,
    selection_rows: list[dict],
    metadata: dict,
    provenance: dict,
    source_root: Path,
    output_root: Path,
) -> dict:
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing snapshot: {output_root}")
    image_train = output_root / "images/train"
    mask_train = output_root / "masks/train"
    image_train.mkdir(parents=True)
    mask_train.mkdir(parents=True)

    ordered_names = sorted(row["filename"] for row in selection_rows)
    selected_names = {
        row["filename"] for row in selection_rows if row["selected"]
    }
    entries = []
    for filename in ordered_names:
        source_image = source_root / "images/train" / filename
        source_mask = source_root / "masks/train" / filename
        symlink_checked(source_image, image_train / filename)
        symlink_checked(source_mask, mask_train / filename)
        entries.append({"filename": filename, "source": filename, "copy": 0})
        if filename in selected_names:
            source_name = Path(filename)
            duplicate_name = f"{source_name.stem}__deft001{source_name.suffix}"
            symlink_checked(source_image, image_train / duplicate_name)
            symlink_checked(source_mask, mask_train / duplicate_name)
            entries.append({"filename": duplicate_name, "source": filename, "copy": 1})

    for category, split in (("images", "val"), ("masks", "val"), ("images", "test")):
        source = source_root / category / split
        destination = output_root / category / split
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)

    observed_images = len(list(image_train.iterdir()))
    observed_masks = len(list(mask_train.iterdir()))
    if observed_images != metadata["resulting_train_count"] or observed_masks != observed_images:
        raise RuntimeError(
            f"materialized count mismatch: images={observed_images} masks={observed_masks}"
        )
    manifest = {
        "schema_version": 1,
        "kind": "model_driven_deft_reweighting",
        "backbone": backbone,
        "source": str(source_root),
        "validation_used_for_selection": False,
        "method": {
            "gap_signal": "four-fold out-of-fold SegFormer prediction error",
            "propagation": "cosine k-nearest neighbors in trained-backbone feature space",
            **metadata,
        },
        "provenance": provenance,
        "selection": selection_rows,
        "entries": entries,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-fraction", type=float, default=0.20)
    parser.add_argument("--duplicate-fraction", type=float, default=0.25)
    parser.add_argument("--neighbors-per-anchor", type=int, default=3)
    parser.add_argument("--expected-count", type=int, default=316)
    args = parser.parse_args()

    for backbone in BACKBONES:
        names, embeddings, difficulties, provenance = load_inputs(
            backbone,
            args.scores_root,
            args.embeddings_root,
            args.expected_count,
        )
        rows, metadata = select_deft_samples(
            names,
            embeddings,
            difficulties,
            anchor_fraction=args.anchor_fraction,
            duplicate_fraction=args.duplicate_fraction,
            neighbors_per_anchor=args.neighbors_per_anchor,
        )
        manifest = materialize_snapshot(
            backbone,
            rows,
            metadata,
            provenance,
            args.source_root,
            args.output_root / backbone,
        )
        print(
            f"DEFT_SNAPSHOT backbone={backbone} "
            f"anchors={manifest['method']['anchor_count']} "
            f"duplicates={manifest['method']['duplicate_count']} "
            f"train={manifest['method']['resulting_train_count']} "
            "validation_used=false",
            flush=True,
        )


if __name__ == "__main__":
    main()
