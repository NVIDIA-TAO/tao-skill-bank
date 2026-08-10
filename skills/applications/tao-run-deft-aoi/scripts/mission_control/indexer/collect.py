# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect every image in a TAO 7.0.1 PCB DEFT run into one embedding worklist.

The 7.0.1 loop only embeds the mining source pool and the KPI RCA-gap
targets. To populate the full Mission Control map we additionally embed:

- the full KPI test set        (``kpi/testing_set.csv``,   kind=kpi)
- the seed train / val splits  (``train/base/*_set.csv``,  kind=pool, split=train|val)
- the Cosmos AnomalyGen NG/OK pairs staged into training
  (``iterN/dataset/images/synthetic_iterN_{ng,ok}/``,      kind=pool, split=synth)

Returned columns: ``filepath, kind, split, label, object_name, input_path,
golden_path, board``. The container's ``embedding image_embeddings`` preserves
these extra columns alongside the added ``embedding`` column.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import images, layout


def _rows_from_component_csv(csv_path, base, kind, split, light, ext):
    """Turn one component CSV (KPI / train / val) into worklist rows.

    Each row names a component crop, whose real path is rebuilt from ``base``.
    ``kind`` draws it as a diamond (kpi) or circle (pool); ``split`` is the finer
    facet. Returns no rows when the CSV is missing.
    """
    if not Path(csv_path).is_file():
        return []
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    out = []
    for r in df.itertuples(index=False):
        fp = images.component_file(base, r.input_path, r.object_name, light, ext)
        out.append({
            "filepath": fp, "kind": kind, "split": split,
            "label": r.label, "object_name": r.object_name,
            "input_path": r.input_path, "golden_path": getattr(r, "golden_path", ""),
            "board": getattr(r, "project", ""),
        })
    return out


def _rows_from_synth(rd: Path, light, ext):
    """Collect the AnomalyGen synthetic NG/OK images staged into training.

    Reads ``iterN/dataset/images/synthetic_iterN_{ng,ok}/`` — what actually
    trained — not the raw generator dump under ``iterN/anomalygen/sdg*``. The NG
    filename ``<component>+<defect>_<idx>_<light><ext>`` supplies the label; its
    OK partner is PASS. Both are pool points tagged ``split=synth``.
    """
    out = []
    for ng_dir in sorted(rd.glob("iter*/dataset/images/synthetic_iter*_ng")):
        ok_dir = Path(str(ng_dir)[:-3] + "_ok")
        for f in sorted(ng_dir.glob(f"*{ext}")):
            stem = f.name[: -len(f"_{light}{ext}")] if f.name.endswith(f"_{light}{ext}") else f.stem
            defect = stem.split("+", 1)[1].rsplit("_", 1)[0] if "+" in stem else "ng"
            out.append({"filepath": os.path.realpath(f), "kind": "pool", "split": "synth",
                        "label": defect, "object_name": stem, "input_path": str(ng_dir.relative_to(rd.parent.parent)),
                        "golden_path": "", "board": "synthetic"})
        for f in sorted(ok_dir.glob(f"*{ext}")):
            stem = f.name[: -len(f"_{light}{ext}")] if f.name.endswith(f"_{light}{ext}") else f.stem
            out.append({"filepath": os.path.realpath(f), "kind": "pool", "split": "synth",
                        "label": "PASS", "object_name": stem, "input_path": str(ok_dir.relative_to(rd.parent.parent)),
                        "golden_path": "", "board": "synthetic"})
    return out


def collect(rd: Path, ws: Path, light: str, ext: str, paths: dict | None = None) -> pd.DataFrame:
    """Build the de-duplicated list of every image to SigLIP-embed for the map.

    Gathers the KPI test set, the seed train/val splits and the synthetic pairs,
    keeps only files present on disk, and drops duplicate paths so a KPI row wins
    over a pool row for the same file. One row per component, not per capture:
    ``light`` is the channel-0 name and a multi-lighting run stacks its captures
    into one sample. ``paths`` comes from ``layout.resolve``; ``ws`` is used only
    when it is omitted.
    """
    # Prefer the paths the run itself recorded; fall back to the documented
    # workspace layout (see indexer/layout.py).
    p = paths or layout.resolve(rd)
    images_dir = p["images_dir"]
    rows = []
    rows += _rows_from_component_csv(p["kpi_test_csv"], images_dir, "kpi", "kpi", light, ext)
    rows += _rows_from_component_csv(p["training_csv"], images_dir, "pool", "train", light, ext)
    rows += _rows_from_component_csv(p["validation_csv"], images_dir, "pool", "val", light, ext)
    rows += _rows_from_synth(rd, light, ext)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # keep only files that exist; de-dup by resolved path (KPI wins over pool)
    df = df[df["filepath"].map(images.exists)]
    df["_kpi"] = (df["kind"] == "kpi").astype(int)
    df = (df.sort_values("_kpi", ascending=False)
            .drop_duplicates("filepath", keep="first")
            .drop(columns="_kpi")
            .reset_index(drop=True))
    return df
