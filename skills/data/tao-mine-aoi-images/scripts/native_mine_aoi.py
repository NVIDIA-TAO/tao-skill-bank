# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-Windows replacement for the data-services mining container.

Reproduces the tao-mine-aoi-images three-step contract without docker:
  1. SigLIP-embed target images        -> target_embeddings.parquet
  2. SigLIP-embed the source pool      -> source_embeddings.parquet
  3. top-K cosine nearest neighbours   -> mined_raw.parquet

Schemas match the container's so the DEFT skill's own
`filter_mined_by_cosine.py` (recompute + floor) and `emit_mined_sharegpt.py`
consume the output unchanged: embeddings parquet = `filepath` + `embedding`
(+ `label` when supplied), mined parquet = `filepath` + `label` +
`target_filepath` + `similarity`.

Encoder: google/siglip-base-patch16-224 via transformers, matching the mining
skill's default MODEL_PATH. Image features are L2-normalised so cosine is a
dot product, identically to the recompute in filter_mined_by_cosine.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

import parquet_io  # parquet via pyarrow, or duckdb where pyarrow has no wheel (Windows ARM64)


def load_encoder(model_id: str, device: str):
    from transformers import AutoImageProcessor, SiglipVisionModel

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = SiglipVisionModel.from_pretrained(model_id, dtype=torch.float32)
    model.eval().to(device)
    return processor, model


@torch.inference_mode()
def embed(paths: list[str], processor, model, device: str, batch_size: int = 16) -> torch.Tensor:
    from PIL import Image

    chunks = []
    for start in range(0, len(paths), batch_size):
        images = [Image.open(p).convert("RGB") for p in paths[start : start + batch_size]]
        inputs = processor(images=images, return_tensors="pt").to(device)
        features = model(**inputs).pooler_output
        chunks.append(torch.nn.functional.normalize(features, dim=-1).cpu())
    return torch.cat(chunks) if chunks else torch.empty(0, 0)


def write_embeddings(path: pathlib.Path, filepaths: list[str], vectors: torch.Tensor,
                     labels: list[str] | None) -> None:
    columns = {"filepath": list(filepaths), "embedding": vectors.tolist()}
    if labels is not None:
        columns["label"] = list(labels)
    parquet_io.write_columns(columns, path)  # "embedding" is written as list<float32>


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-parquet", required=True, help="parquet with `filepath` (query images)")
    parser.add_argument("--source-annotations", required=True, help="mining_pool.json (bare OK/NG ShareGPT)")
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--filter-label", choices=("OK", "NG"), default=None,
                        help="restrict the source pool to records with this label before mining "
                             "(per-class mining: run once per label with its own --top-k)")
    parser.add_argument("--reuse-source-embeddings", default=None,
                        help="parquet from a previous run over the SAME (filtered) pool; "
                             "skips re-embedding the sources (targets are always embedded fresh)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(args.media_root)

    targets = parquet_io.read_columns(args.targets_parquet)
    target_paths = targets["filepath"]
    for p in target_paths:
        if not pathlib.Path(p).is_file():
            raise FileNotFoundError(f"target image missing: {p}")

    records = json.loads(pathlib.Path(args.source_annotations).read_text(encoding="utf-8"))
    source_paths, source_labels = [], []
    for record in records:
        rel = record["images"][0]  # the AOI capture; images[1], when present, is the golden reference
        label = next(t["value"] for t in record["conversations"] if t["from"] == "gpt")
        if args.filter_label is not None and label != args.filter_label:
            continue
        absolute = str((root / rel).resolve())
        if not pathlib.Path(absolute).is_file():
            raise FileNotFoundError(f"mining-pool image missing: {absolute}")
        source_paths.append(absolute)
        source_labels.append(label)
    if not source_paths:
        raise ValueError(f"source pool is empty after --filter-label {args.filter_label}")

    processor, model = load_encoder(args.model, args.device)
    target_vecs = embed(target_paths, processor, model, args.device)
    source_vecs = None
    if args.reuse_source_embeddings:
        cached = parquet_io.read_columns(args.reuse_source_embeddings)
        if cached.get("filepath") == source_paths:  # exact same filtered pool, same order
            source_vecs = torch.tensor(cached["embedding"], dtype=torch.float32)
    reused = source_vecs is not None
    if source_vecs is None:
        source_vecs = embed(source_paths, processor, model, args.device)
    write_embeddings(out / "target_embeddings.parquet", target_paths, target_vecs, None)
    write_embeddings(out / "source_embeddings.parquet", source_paths, source_vecs, source_labels)

    # top-K per target; union over targets, keeping each source's best score
    similarity = target_vecs @ source_vecs.T          # [T, S], cosine (unit vectors)
    k = min(args.top_k, len(source_paths))
    best: dict[str, dict] = {}
    for t_index, t_path in enumerate(target_paths):
        scores, indices = torch.topk(similarity[t_index], k)
        for score, s_index in zip(scores.tolist(), indices.tolist()):
            s_path = source_paths[s_index]
            if s_path not in best or score > best[s_path]["similarity"]:
                best[s_path] = {
                    "filepath": s_path,
                    "label": source_labels[s_index],
                    "target_filepath": t_path,
                    "similarity": float(score),
                }
    rows = sorted(best.values(), key=lambda r: -r["similarity"])
    parquet_io.write_rows(rows, out / "mined_raw.parquet",
                          columns=("filepath", "label", "target_filepath", "similarity"))
    summary = {
        "encoder": args.model, "device": args.device, "top_k": args.top_k,
        "filter_label": args.filter_label, "source_embeddings_reused": reused,
        "targets": len(target_paths), "source_pool": len(source_paths),
        "mined_raw_rows": len(rows),
        "similarity_range": [rows[-1]["similarity"], rows[0]["similarity"]] if rows else None,
    }
    (out / "native_mining_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
