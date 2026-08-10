# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared synthetic DEFT run used by the Mission Control test suites."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

LIGHT, EXT = "SolderLight", ".jpg"

# The real KPI/train CSV header (see <workspace>/kpi/testing_set.csv).
CSV_COLS = ["input_path", "golden_path", "label", "object_name", "project"]

E0, E1, E2 = np.eye(4)[0], np.eye(4)[1], np.eye(4)[2]


def crop(ws, board, obj):
    """Create one component crop at the layout the loop expects."""
    d = Path(ws) / "kpi" / "images" / board
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{obj}_{LIGHT}{EXT}"
    f.write_bytes(b"x")
    return os.path.realpath(f)


def build_run(root, *, iterations=None, config=None, best_iteration="iter1",
              lights=("SolderLight",), kpi=None):
    """Create a minimal but complete DEFT run at ``<root>/ws/results/run_X``.

    Returns:
        tuple[Path, Path]: ``(workspace, results_dir)``.
    """
    ws = Path(root) / "ws"
    rd = ws / "results" / "run_X"
    (ws / "kpi" / "images").mkdir(parents=True)
    (ws / "train" / "base").mkdir(parents=True)
    (rd / "mission_control").mkdir(parents=True)

    kpi = kpi if kpi is not None else [("b1", "K0", "PASS"), ("b1", "K1", "Missing")]
    pool = [("b2", "T0", "PASS"), ("b2", "T1", "PASS"), ("b2", "T2", "PASS")]

    emb = []
    for board, obj, lab in kpi:
        emb.append({"filepath": crop(ws, board, obj), "embedding": E0.tolist(),
                    "kind": "kpi", "split": "kpi", "label": lab,
                    "object_name": obj, "board": board})
    for board, obj, lab in pool:
        emb.append({"filepath": crop(ws, board, obj), "embedding": E1.tolist(),
                    "kind": "pool", "split": "train", "label": lab,
                    "object_name": obj, "board": board})
    pd.DataFrame(emb).to_parquet(rd / "mission_control" / "embeddings.parquet", index=False)

    pd.DataFrame([[b, "", l, o, "proj"] for b, o, l in kpi],
                 columns=CSV_COLS).to_csv(ws / "kpi/testing_set.csv", index=False)
    pd.DataFrame([[b, "", l, o, "proj"] for b, o, l in pool],
                 columns=CSV_COLS).to_csv(ws / "train/base/training_set.csv", index=False)
    pd.DataFrame([], columns=CSV_COLS).to_csv(ws / "train/base/validation_set.csv", index=False)

    # Each extra lighting condition is another capture of the SAME component,
    # so the crop files multiply but the CSV rows (and map points) do not.
    for board, obj, _ in kpi:
        for lt in lights[1:]:
            (Path(ws) / "kpi" / "images" / board / f"{obj}_{lt}{EXT}").write_bytes(b"x")
    input_map = "\n".join(f"      {lt}: {i}" for i, lt in enumerate(lights))
    (rd / "baseline_spec.yaml").write_text(
        f"dataset:\n  classify:\n    image_ext: .jpg\n    num_input: {len(lights)}\n"
        f"    input_map:\n{input_map}\n")

    state = {
        "results_dir": str(rd),
        "best_iteration": best_iteration,
        "kpi_target": "FAR<1%@recall=100%",
        "config": config if config is not None else {
            "mining_filter": {"top_k_per_target": 5, "metric": "cosine",
                              "min_similarity": 0.9}},
        "iterations": iterations if iterations is not None else {
            "baseline": {"far_pct": 50.0, "threshold": 0.30, "val_loss": 0.50},
            "iter1": {"far_pct": 10.0, "threshold": 0.40, "val_loss": 0.40},
        },
    }
    # init_deft_state.py records the resolved input paths in config; mirror that
    # so the fixture exercises the same resolution a real run does.
    state["config"].update({
        "images_dir": str(ws / "kpi" / "images"),
        "kpi_test_csv": str(ws / "kpi" / "testing_set.csv"),
        "training_csv": str(ws / "train" / "base" / "training_set.csv"),
        "validation_csv": str(ws / "train" / "base" / "validation_set.csv"),
    })
    (rd / "deft_state.json").write_text(json.dumps(state))
    (rd / "loop_log.jsonl").write_text(json.dumps(
        {"seq": 1, "iter": "baseline", "stage": "train", "status": "ok",
         "summary": "FAR=50%", "duration_sec": 1}) + "\n")
    return ws, rd


def add_mining(ws, rd, iteration="iter1", ts="2026-01-01_120000", *,
               spec=True, mined=None, kept_count=2):
    """Add one ``mining_results/<ts>/`` attempt plus its mining_filter summary.

    Sources are E0/E1/E2 and targets E0/E1, so at cosine >= 0.9 exactly two
    source images clear the gate — the value written into knn_summary.csv.

    Returns:
        tuple[Path, list[str], list[str]]: ``(mining_dir, source_paths, target_paths)``.
    """
    srcs = [os.path.realpath(ws / "kpi/images/b2" / f"T{i}_{LIGHT}{EXT}") for i in range(3)]
    tgts = [os.path.realpath(ws / "kpi/images/b1" / f"K{i}_{LIGHT}{EXT}") for i in range(2)]

    md = rd / iteration / "mining_results" / ts
    md.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"filepath": srcs,
                  "embedding": [E0.tolist(), E1.tolist(), E2.tolist()],
                  "label": ["PASS"] * 3}).to_parquet(md / "source_embeddings.parquet", index=False)
    pd.DataFrame({"filepath": tgts,
                  "embedding": [E0.tolist(), E1.tolist()],
                  "label": ["PASS"] * 2}).to_parquet(md / "target_embeddings.parquet", index=False)
    if spec:
        (md / "mining_spec.yaml").write_text(
            'topn: 5\nknn_metric: cosine\nfilter_by_label: "false"\n')
    if mined is not None:
        pd.DataFrame({"filepath": mined}).to_parquet(md / "mined.parquet", index=False)

    mf = rd / iteration / "mining_filter"
    mf.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"candidate_count": 3, "kept_count": kept_count,
                   "rejected_count": 3 - kept_count,
                   "similarity_threshold": 0.9}]).to_csv(mf / "knn_summary.csv", index=False)
    return md, srcs, tgts


# --------------------------------------------------------------------------- #
# RCA fixture — a run shaped so every RcaChatTools branch is reachable
# --------------------------------------------------------------------------- #

# The real inference.csv header (see <run>/iterN/inference/best_val/inference.csv).
INF_COLS = ["input_path", "golden_path", "label", "object_name", "project", "boardname",
            "comp_type_1", "comp_type_2", "part_type", "number_of_pins",
            "description", "siamese_score"]

RCA_THRESHOLD = 0.40


def add_inference(rd, rows, iteration="iter1", kind="best_val"):
    """Write the iteration's inference.csv from (board, obj, label, score).

    ``kind=None`` writes it flat in ``inference/`` — the layout the loop
    actually produced on a 7.0.1 run.
    """
    inf = rd / iteration / "inference" / kind if kind else rd / iteration / "inference"
    inf.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[b, "", lab, o, "proj", "brd", "1", "ct", "pt", "4", "desc", s]
                  for b, o, lab, s in rows],
                 columns=INF_COLS).to_csv(inf / "inference.csv", index=False)
    return inf / "inference.csv"

RCA_KPI = (
    [(f"C{i}", "PASS", s, "C", 0) for i, s in enumerate([0.90, 0.80, 0.70, 0.60, 0.50, 0.10])]
    + [(f"U{i}", "PASS", s, "U", 2) for i, s in enumerate([0.45, 0.10, 0.10, 0.10, 0.10, 0.10])]
    + [("N0", "Missing", 0.95, "C", 0),      # defect caught (score above threshold)
       ("N1", "Missing", 0.30, "C", 1),      # defect missed
       ("N2", "Shift", 0.20, "U", 2)]        # defect missed
)


def build_rca_run(root, *, threshold=RCA_THRESHOLD, iterations=None):
    """A run tuned for the RCA tools: 15 KPI rows with inference scores, 6 seed
    pool images, and 3 unused mining candidates.

    Crops are real JPEGs (not stub bytes) because view_images opens them with
    PIL. Returns ``(workspace, results_dir)``.
    """
    from PIL import Image

    ws = Path(root) / "ws"
    rd = ws / "results" / "run_X"
    (ws / "kpi" / "images").mkdir(parents=True)
    (ws / "train" / "base").mkdir(parents=True)
    (rd / "mission_control").mkdir(parents=True)
    basis = np.eye(4)

    def jpeg(board, obj):
        d = ws / "kpi" / "images" / board
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{obj}_{LIGHT}{EXT}"
        Image.new("RGB", (8, 8), (120, 120, 120)).save(f, "JPEG")
        return os.path.realpath(f)

    pool = [(f"T{i}", 0) for i in range(6)]        # seed: enters training
    cands = [(f"P{i}", 1) for i in range(3)]       # mining pool: never kept

    emb = [{"filepath": jpeg("b1", o), "embedding": basis[e].tolist(), "kind": "kpi",
            "split": "kpi", "label": lab, "object_name": o, "board": "b1"}
           for o, lab, _, _, e in RCA_KPI]
    emb += [{"filepath": jpeg("b2", o), "embedding": basis[e].tolist(), "kind": "pool",
             "split": "train", "label": "PASS", "object_name": o, "board": "b2"}
            for o, e in pool]
    pd.DataFrame(emb).to_parquet(rd / "mission_control" / "embeddings.parquet", index=False)

    pd.DataFrame([["b1", "", lab, o, "proj"] for o, lab, _, _, _ in RCA_KPI],
                 columns=CSV_COLS).to_csv(ws / "kpi/testing_set.csv", index=False)
    pd.DataFrame([["b2", "", "PASS", o, "proj"] for o, _ in pool],
                 columns=CSV_COLS).to_csv(ws / "train/base/training_set.csv", index=False)
    pd.DataFrame([], columns=CSV_COLS).to_csv(ws / "train/base/validation_set.csv", index=False)

    (rd / "baseline_spec.yaml").write_text(
        "dataset:\n  classify:\n    image_ext: .jpg\n    input_map:\n      SolderLight: 0\n")
    (rd / "deft_state.json").write_text(json.dumps({
        "results_dir": str(rd), "best_iteration": "iter1", "kpi_target": "FAR<1%@recall=100%",
        "best_far_pct": 50.0, "kpi_met": False,
        "config": {"mining_filter": {"top_k_per_target": 5, "metric": "cosine",
                                     "min_similarity": 0.9}},
        "iterations": iterations if iterations is not None else {
            "baseline": {"far_pct": 90.0, "threshold": 0.20, "val_loss": 0.50},
            "iter1": {"far_pct": 50.0, "threshold": threshold, "val_loss": 0.40}},
    }))
    (rd / "loop_log.jsonl").write_text(json.dumps(
        {"seq": 1, "iter": "baseline", "stage": "train", "status": "ok",
         "summary": "FAR=90%", "duration_sec": 1}) + "\n")

    inf = rd / "iter1" / "inference" / "best_val"
    inf.mkdir(parents=True)
    pd.DataFrame([["b1", "", lab, o, "proj", "brd", "1", ct, "pt", "4", "desc", s]
                  for o, lab, s, ct, _ in RCA_KPI],
                 columns=INF_COLS).to_csv(inf / "inference.csv", index=False)

    # Candidates exist ONLY in source_embeddings and are never kept, which is
    # what makes _build tag them provenance="candidate" (unused mining pool).
    md = rd / "iter1" / "mining_results" / "2026-01-01_120000"
    md.mkdir(parents=True)
    pd.DataFrame({"filepath": [jpeg("b3", o) for o, _ in cands],
                  "embedding": [basis[e].tolist() for _, e in cands],
                  "label": ["PASS"] * len(cands)}).to_parquet(
        md / "source_embeddings.parquet", index=False)
    pd.DataFrame({"filepath": [jpeg("b1", "C0")], "embedding": [basis[0].tolist()],
                  "label": ["PASS"]}).to_parquet(md / "target_embeddings.parquet", index=False)
    (md / "mining_spec.yaml").write_text(
        'topn: 5\nknn_metric: cosine\nfilter_by_label: "false"\n')
    mf = rd / "iter1" / "mining_filter"
    mf.mkdir(parents=True)
    pd.DataFrame([{"candidate_count": 3, "kept_count": 0, "rejected_count": 3,
                   "similarity_threshold": 0.9}]).to_csv(mf / "knn_summary.csv", index=False)
    return ws, rd


def add_routing(rd, tgts, iteration="iter1", ts="2026-01-01_110000"):
    """Add the routing output that `weak_targets()` reads."""
    rr = rd / iteration / "routing_results" / ts
    rr.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"filepath": tgts, "label": ["PASS"] * len(tgts),
                  "siamese_score": [0.8, 0.7][:len(tgts)],
                  "weakness": [0.9, 0.5][:len(tgts)]}).to_parquet(
        rr / "mining_gaps.parquet", index=False)
    return rr
