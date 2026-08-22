#!/usr/bin/env python3
"""Create deterministic, train-only folds for SegFormer DEFT error mining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FOLD_COUNT = 4
EXPECTED_SAMPLES = 316


def build_manifest(score_manifest: Path) -> dict:
    payload = json.loads(score_manifest.read_text())
    scores = payload["scores"]
    if len(scores) != EXPECTED_SAMPLES:
        raise RuntimeError(f"expected {EXPECTED_SAMPLES} train samples, found {len(scores)}")
    filenames = [row["filename"] for row in scores]
    if len(set(filenames)) != EXPECTED_SAMPLES:
        raise RuntimeError("training filenames are not unique")

    # Round-robin a descending difficulty ordering so every held-out fold has a
    # comparable mixture of rare-class content, boundaries, and mask entropy.
    ordered = sorted(scores, key=lambda row: (-row["difficulty"], row["filename"]))
    folds = []
    for fold_index in range(FOLD_COUNT):
        held_out = sorted(
            row["filename"]
            for index, row in enumerate(ordered)
            if index % FOLD_COUNT == fold_index
        )
        train = sorted(set(filenames) - set(held_out))
        folds.append(
            {
                "fold": fold_index,
                "train": train,
                "held_out": held_out,
                "train_count": len(train),
                "held_out_count": len(held_out),
            }
        )

    held_out_union = set().union(*(set(fold["held_out"]) for fold in folds))
    if held_out_union != set(filenames):
        raise RuntimeError("held-out folds do not cover the training set exactly")
    for left in range(FOLD_COUNT):
        for right in range(left + 1, FOLD_COUNT):
            if set(folds[left]["held_out"]) & set(folds[right]["held_out"]):
                raise RuntimeError(f"held-out folds {left} and {right} overlap")
    if any(fold["train_count"] != 237 or fold["held_out_count"] != 79 for fold in folds):
        raise RuntimeError("unexpected 4-fold cardinality")

    return {
        "schema_version": 1,
        "method": "difficulty_balanced_round_robin",
        "source": "training split only",
        "validation_used": False,
        "fold_count": FOLD_COUNT,
        "sample_count": EXPECTED_SAMPLES,
        "folds": folds,
    }


def link(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"conflicting destination: {destination}")
        return
    destination.symlink_to(source)


def materialize(manifest: dict, source_root: Path, destination_root: Path) -> None:
    for fold in manifest["folds"]:
        fold_root = destination_root / f"fold{fold['fold']}"
        for filename in fold["train"]:
            for kind in ("images", "masks"):
                link(
                    source_root / kind / "train" / filename,
                    fold_root / kind / "train" / filename,
                )
        for filename in fold["held_out"]:
            for kind in ("images", "masks"):
                link(
                    source_root / kind / "train" / filename,
                    fold_root / kind / "val" / filename,
                )
            link(
                source_root / "images" / "train" / filename,
                fold_root / "images" / "test" / filename,
            )

        observed = {
            "images/train": len(list((fold_root / "images/train").iterdir())),
            "masks/train": len(list((fold_root / "masks/train").iterdir())),
            "images/val": len(list((fold_root / "images/val").iterdir())),
            "masks/val": len(list((fold_root / "masks/val").iterdir())),
            "images/test": len(list((fold_root / "images/test").iterdir())),
        }
        expected = {
            "images/train": 237,
            "masks/train": 237,
            "images/val": 79,
            "masks/val": 79,
            "images/test": 79,
        }
        if observed != expected:
            raise RuntimeError(
                f"fold{fold['fold']} materialization mismatch: {observed} != {expected}"
            )
        print(f"fold{fold['fold']} train=237 held_out=79", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-manifest", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--destination-root", type=Path)
    args = parser.parse_args()

    if args.write_manifest:
        if not args.score_manifest:
            parser.error("--score-manifest is required with --write-manifest")
        payload = build_manifest(args.score_manifest)
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"manifest={args.write_manifest} folds=4 samples=316")
        return

    if args.manifest:
        if not args.source_root or not args.destination_root:
            parser.error("--source-root and --destination-root are required with --manifest")
        materialize(
            json.loads(args.manifest.read_text()),
            args.source_root,
            args.destination_root,
        )
        return

    parser.error("choose --write-manifest or --manifest")


if __name__ == "__main__":
    main()
