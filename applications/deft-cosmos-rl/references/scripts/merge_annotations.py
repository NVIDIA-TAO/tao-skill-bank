#!/usr/bin/env python3
"""Annotation utilities: create LLaVA training JSON or merge annotation files.

Two modes:
  --mode create-llava: Convert a parquet to LLaVA-format training JSON.
  --mode merge:        Concatenate two JSON annotation lists into one.

Supports both local paths and S3 URIs (s3://) via fsspec.
"""
import argparse
import json

import fsspec
import pandas as pd


def create_llava_training_json(cosmos_data_parquet: str, output_json_path: str) -> str:
    df = pd.read_parquet(cosmos_data_parquet)

    training_data = []
    for _, row in df.iterrows():
        video_path = row["generated_video_path"]
        training_data.append({
            "id": video_path,
            "video": video_path,
            "conversations": [
                {"from": "human", "value": row["question"]},
                {"from": "gpt", "value": row["ground_truth"]},
            ],
        })

    with fsspec.open(output_json_path, "w") as f:
        json.dump(training_data, f, indent=4)

    return output_json_path


def merge_annotations(prev_annotations_path: str, curr_annotations_path: str,
                      result_path: str) -> str:
    with fsspec.open(curr_annotations_path, "r") as f:
        curr_annotations = json.load(f)

    if prev_annotations_path:
        try:
            with fsspec.open(prev_annotations_path, "r") as f:
                prev_annotations = json.load(f)
        except FileNotFoundError:
            print(f"Previous annotations not found at '{prev_annotations_path}', "
                  "using only current annotations")
            prev_annotations = []
    else:
        prev_annotations = []

    merged = prev_annotations + curr_annotations

    with fsspec.open(result_path, "w") as f:
        json.dump(merged, f)

    return result_path


def main():
    parser = argparse.ArgumentParser(description="Annotation utilities for DEFT")
    parser.add_argument("--mode", required=True, choices=["create-llava", "merge"])

    # create-llava
    parser.add_argument("--cosmos-data-parquet")
    parser.add_argument("--output-json-path")

    # merge
    parser.add_argument("--prev-annotations-path", default="")
    parser.add_argument("--curr-annotations-path")
    parser.add_argument("--result-path")

    args = parser.parse_args()

    if args.mode == "create-llava":
        create_llava_training_json(args.cosmos_data_parquet, args.output_json_path)
    elif args.mode == "merge":
        merge_annotations(args.prev_annotations_path, args.curr_annotations_path,
                          args.result_path)


if __name__ == "__main__":
    main()
