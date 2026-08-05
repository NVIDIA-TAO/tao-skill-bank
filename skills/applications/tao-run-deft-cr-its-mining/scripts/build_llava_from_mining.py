#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert mined nearest-neighbor rows into LLaVA training annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from workflow_common import absolute_path, write_json_array


def build_llava_records(
    mined_neighbors_parquet: Path,
    train_embeddings_parquet: Path,
    train_lookup_parquet: Path,
) -> list[dict]:
    """Recover mined source modalities and return their LLaVA records."""
    neighbors = pd.read_parquet(mined_neighbors_parquet)
    source_col = "source_filepath" if "source_filepath" in neighbors.columns else "filepath"
    if source_col not in neighbors.columns:
        raise ValueError(f"{mined_neighbors_parquet}: missing source filepath column")

    embeddings = pd.read_parquet(train_embeddings_parquet)
    required_embedding_columns = {"filepath", "modality"}
    missing = required_embedding_columns - set(embeddings.columns)
    if missing:
        raise ValueError(f"{train_embeddings_parquet}: missing required columns: {sorted(missing)}")
    source_metadata = embeddings[["filepath", "modality"]].drop_duplicates()
    conflicts = source_metadata.groupby("filepath")["modality"].nunique()
    conflicting_paths = conflicts[conflicts > 1].index.astype(str).tolist()
    if conflicting_paths:
        raise ValueError(
            f"{train_embeddings_parquet}: source filepaths map to multiple modalities: "
            f"{conflicting_paths[:5]}"
        )
    mined = neighbors[[source_col]].drop_duplicates().rename(columns={source_col: "source_filepath"})
    mined = mined.merge(
        source_metadata,
        left_on="source_filepath",
        right_on="filepath",
        how="left",
        validate="one_to_one",
    )
    missing_metadata = mined[mined["modality"].isna()]["source_filepath"].astype(str).tolist()
    if missing_metadata:
        raise ValueError(
            "mined source filepaths are absent from the train embeddings parquet: "
            f"{missing_metadata[:5]}"
        )
    unsupported = sorted(set(mined["modality"].astype(str)) - {"text", "video"})
    if unsupported:
        raise ValueError(f"unsupported source modalities in {train_embeddings_parquet}: {unsupported}")

    lookup = pd.read_parquet(train_lookup_parquet)
    required = {"filepath", "annotation_id", "video_path", "question", "answer"}
    missing = required - set(lookup.columns)
    if missing:
        raise ValueError(f"{train_lookup_parquet}: missing required columns: {sorted(missing)}")

    text_sources = mined[mined["modality"] == "text"]
    video_sources = mined[mined["modality"] == "video"]
    matched: list[pd.DataFrame] = []
    if not text_sources.empty:
        matched.append(
            text_sources.merge(lookup, left_on="source_filepath", right_on="filepath", how="inner")
        )
    if not video_sources.empty:
        matched.append(
            video_sources.merge(lookup, left_on="source_filepath", right_on="video_path", how="inner")
        )
    merged = pd.concat(matched, ignore_index=True) if matched else pd.DataFrame()
    if merged.empty:
        raise RuntimeError("no mined neighbors joined to train lookup rows")
    unmatched = sorted(
        set(mined["source_filepath"].astype(str)) - set(merged["source_filepath"].astype(str))
    )
    if unmatched:
        raise ValueError(f"mined source filepaths did not match the train lookup: {unmatched[:5]}")

    merged = merged.drop_duplicates(subset=["annotation_id"]).reset_index(drop=True)
    records: list[dict] = []
    for _, row in merged.iterrows():
        annotation_id = str(row["annotation_id"])
        video = str(row["video_path"])
        question = str(row["question"])
        answer = str(row["answer"])
        records.append(
            {
                "id": annotation_id,
                "video": video,
                "conversations": [
                    {"from": "human", "value": f"<video>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mined-neighbors-parquet", required=True, type=Path)
    parser.add_argument("--train-embeddings-parquet", required=True, type=Path)
    parser.add_argument("--train-lookup-parquet", required=True, type=Path)
    parser.add_argument("--output-llava-json", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    records = build_llava_records(
        absolute_path(args.mined_neighbors_parquet),
        absolute_path(args.train_embeddings_parquet),
        absolute_path(args.train_lookup_parquet),
    )
    output_path = absolute_path(args.output_llava_json)
    write_json_array(output_path, records)
    print(f"Wrote {len(records)} LLaVA records -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
