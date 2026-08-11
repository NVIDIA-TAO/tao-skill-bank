# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""far_eval_sweep.py — per-epoch checkpoint sweep, runs INSIDE the TAO container.

For a completed training job:
  1. Enumerate all retained checkpoints (model_epoch_*.pth + model_best_*.pth).
  2. For each: run visual_changenet inference on the VALIDATION set, compute
     FAR@100%recall (exact analyze_kpi.py port).
  3. Select the checkpoint with the best (lowest) validation FAR
     (tie-break: earlier epoch — less overfit).
  4. Run KPI-set inference on the selected checkpoint only; its KPI FAR is the
     score reported to the AutoML brain.
  5. Retain the selected checkpoint as model_far_best.pth; delete the other
     periodic epoch checkpoints (storage control).
  6. Write metric_result.json with value=KPI FAR + best_epoch/val curve
     diagnostics.

Usage:
  python3 far_eval_sweep.py --template <infer_spec.yaml> --train-dir <dir>
      --val-csv <csv> --val-images <dir> --kpi-csv <csv> --kpi-images <dir>
      --workdir <dir> --out <metric_result.json>
"""
import argparse, copy, glob, json, math, os, re, shutil, subprocess, sys

import numpy as np
import pandas as pd
import yaml


def far_at_full_recall(csv_path):
    """Exact port of analyze_kpi.py semantics (validated bit-exact)."""
    df = pd.read_csv(csv_path)
    labels = df["label"].astype(str).str.strip()
    is_pass = (labels.str.upper() == "PASS").to_numpy()
    scores = df["siamese_score"].astype(float).to_numpy()
    n_nopass = int((~is_pass).sum()); n_pass = int(is_pass.sum())
    if n_nopass == 0:
        return None
    uniq = np.unique(scores)
    thresholds = np.concatenate(([np.nextafter(uniq[0], -np.inf)], uniq))
    np_sorted = np.sort(scores[~is_pass]); p_sorted = np.sort(scores[is_pass])
    tp = n_nopass - np.searchsorted(np_sorted, thresholds, side="right")
    fp = n_pass - np.searchsorted(p_sorted, thresholds, side="right")
    fn = n_nopass - tp; tn = n_pass - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), np.nan)
        far = np.where(fp + tn > 0, fp / (fp + tn), np.nan)
    eligible = np.isclose(recall, 1.0, rtol=0.0, atol=1e-12)
    if not eligible.any():
        return None
    f1k = np.where(np.isnan(f1), -np.inf, f1)
    prk = np.where(np.isnan(precision), -np.inf, precision)
    pool = np.flatnonzero(eligible)
    order = np.lexsort((thresholds[pool], prk[pool], f1k[pool]))
    b = pool[order[-1]]
    return {"far_pct": float(far[b] * 100.0), "threshold": float(thresholds[b]),
            "recall_pct": float(recall[b] * 100.0),
            "tp": int(tp[b]), "fp": int(fp[b])}


def run_inference(template, ckpt, csv_path, images_dir, out_dir):
    spec = copy.deepcopy(template)
    spec["task"] = "classify"
    spec["dataset"]["classify"]["infer_dataset"] = {
        "csv_path": csv_path, "images_dir": images_dir,
    }
    spec["inference"] = {
        "checkpoint": ckpt, "batch_size": 16, "results_dir": out_dir,
        "num_gpus": 1, "gpu_ids": [0], "num_nodes": 1,
    }
    spec["results_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    spec_path = os.path.join(out_dir, "infer_spec.yaml")
    with open(spec_path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False)
    r = subprocess.run(["visual_changenet", "inference", "-e", spec_path],
                       capture_output=True, text=True)
    csv_out = os.path.join(out_dir, "inference.csv")
    if r.returncode != 0 or not os.path.exists(csv_out):
        print(f"[sweep] inference FAILED for {os.path.basename(ckpt)} "
              f"(rc={r.returncode}); tail:\n{r.stdout[-600:]}\n{r.stderr[-600:]}",
              flush=True)
        return None
    return csv_out


def epoch_of(path):
    m = re.search(r"model_epoch_(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    m = re.search(r"model_best_(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--kpi-csv", required=True)
    ap.add_argument("--kpi-images", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.template) as f:
        template = yaml.safe_load(f)

    # Enumerate checkpoints; dedupe symlinks by realpath
    cands = sorted(glob.glob(os.path.join(args.train_dir, "model_epoch_*.pth"))) + \
            sorted(glob.glob(os.path.join(args.train_dir, "model_best_*.pth")))
    seen, ckpts = set(), []
    for c in cands:
        rp = os.path.realpath(c)
        if rp not in seen:
            seen.add(rp); ckpts.append(c)
    if not ckpts:
        latest = os.path.join(args.train_dir, "changenet_model_classify_latest.pth")
        if os.path.exists(latest):
            ckpts = [latest]
    if not ckpts:
        sys.exit("[sweep] no checkpoints found")

    print(f"[sweep] {len(ckpts)} checkpoint(s): "
          f"{[os.path.basename(c) for c in ckpts]}", flush=True)

    # 1) Validation-FAR per checkpoint
    val_curve = {}
    for i, ckpt in enumerate(ckpts):
        out_dir = os.path.join(args.workdir, f"val_{i:02d}")
        csv_out = run_inference(template, ckpt, args.val_csv, args.val_images, out_dir)
        if csv_out is None:
            continue
        m = far_at_full_recall(csv_out)
        if m is None:
            continue
        val_curve[ckpt] = m["far_pct"]
        print(f"[sweep] {os.path.basename(ckpt)} (ep {epoch_of(ckpt)}): "
              f"val FAR = {m['far_pct']:.2f}%", flush=True)

    if not val_curve:
        sys.exit("[sweep] no checkpoint produced a valid validation FAR")

    # 2) Select: lowest val FAR; tie → earlier epoch (less overfit)
    best_ckpt = min(val_curve, key=lambda c: (val_curve[c], epoch_of(c)))
    best_epoch = epoch_of(best_ckpt)
    print(f"[sweep] selected {os.path.basename(best_ckpt)} "
          f"(ep {best_epoch}, val FAR {val_curve[best_ckpt]:.2f}%)", flush=True)

    # 3) KPI FAR of the selected checkpoint
    kpi_dir = os.path.join(args.workdir, "kpi")
    kpi_csv = run_inference(template, best_ckpt, args.kpi_csv, args.kpi_images, kpi_dir)
    if kpi_csv is None:
        sys.exit("[sweep] KPI inference failed on selected checkpoint")
    kpi = far_at_full_recall(kpi_csv)
    if kpi is None:
        sys.exit("[sweep] KPI FAR computation failed")

    # 4) Retain the winner; prune other periodic checkpoints
    keep = os.path.join(args.train_dir, "model_far_best.pth")
    shutil.copyfile(os.path.realpath(best_ckpt), keep)
    for c in ckpts:
        rp = os.path.realpath(c)
        if rp != os.path.realpath(best_ckpt) and "model_epoch_" in os.path.basename(c):
            try:
                os.remove(rp)
            except OSError:
                pass

    result = {
        "name": "far_pct", "value": kpi["far_pct"], "unit": "%",
        "threshold": kpi["threshold"],
        "constraints": {"recall_pct": kpi["recall_pct"]},
        "diagnostics": {
            "recall_pct": kpi["recall_pct"], "tp": kpi["tp"], "fp": kpi["fp"],
            "best_epoch": best_epoch,
            "val_far_selected": val_curve[best_ckpt],
            "val_far_curve": {f"ep{epoch_of(c)}": round(v, 3)
                              for c, v in sorted(val_curve.items(),
                                                  key=lambda kv: epoch_of(kv[0]))},
            "selected_checkpoint": keep,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[sweep] DONE: KPI FAR@100%R = {kpi['far_pct']:.4f}% "
          f"@ best_epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
