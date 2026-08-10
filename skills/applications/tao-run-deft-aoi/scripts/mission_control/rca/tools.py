# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RCA chat tools (7.0.1 PCB) — the grounded evidence surface the VLM reasons
over. Every tool is a deterministic read over the loaded run (RunIndex) + the
best-iteration inference CSV (which carries PCB component metadata) + SigLIP
embeddings. No metadata store, no DT/GenAI render params, no quantities.

Each tool returns ``(result_dict, refs)`` where refs are map point ids the UI
highlights. The agent narrates; it never invents numbers.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from indexer import images

# component metadata columns worth slicing failures by (present in inference CSV)
META_COLS = ["comp_type_1", "comp_type_2", "part_type", "number_of_pins",
             "description", "boardname", "project"]


class RcaChatTools:
    def __init__(self, index):
        self.ix = index                       # RunIndex for the loaded run
        self.state = index.state
        self.best = index.state.get("best_iteration") or index.iter_order[-1]
        self.thr = index.state["iterations"].get(self.best, {}).get("threshold")
        if self.thr is None or self._inference_csv(self.best) is None:
            for lbl in reversed(index.iter_order):
                t = index.state["iterations"].get(lbl, {}).get("threshold")
                if t is not None and self._inference_csv(lbl) is not None:
                    self.thr, self.best = t, lbl
                    break
            else:
                raise RuntimeError(
                    f"No iteration in {index.rd.name} has both a decision threshold "
                    "and inference results; the RCA agent cannot label failures. "
                    "Run evaluation first."
                )
        self.inf = self._load_inference()     # best-iter KPI inference + meta + outcome

    def _inference_csv(self, iteration):
        """The iteration's inference.csv, or None when it has no inference output.

        Delegates so the RCA panel and the map always agree on which file the
        scores came from.
        """
        return self.ix.inference_csv(iteration)

    # ------------------------------------------------------------------ setup
    def _load_inference(self):
        f = self._inference_csv(self.best)
        df = pd.read_csv(f, dtype=str).fillna("")
        df["siamese_score"] = df["siamese_score"].astype(float)
        df["key"] = [images.component_file(self.ix.images_dir, ip, on, self.ix.light, self.ix.ext)
                     for ip, on in zip(df["input_path"], df["object_name"])]
        df["point_id"] = df["key"].map(self.ix.key2idx)
        df["is_pass"] = df["label"].str.upper() == "PASS"
        thr = self.thr
        # pred defect if score > thr; false alarm = PASS scored defect; missed = defect scored PASS
        df["pred_defect"] = df["siamese_score"] > thr
        df["outcome"] = np.where(df["is_pass"] & df["pred_defect"], "false_alarm",
                         np.where(~df["is_pass"] & ~df["pred_defect"], "missed_defect", "correct"))
        # margin: PASS wants low score (thr - s); defect wants high (s - thr)
        df["margin"] = np.where(df["is_pass"], thr - df["siamese_score"],
                                df["siamese_score"] - thr)
        return df

    def _canon(self, lbl):
        return self.ix._defect_canon.get(str(lbl).strip().lower(), str(lbl).strip().title())

    # ------------------------------------------------------------- run scope
    def run_overview(self):
        """Run headline + FAR/val_loss trajectory across iterations."""
        s = self.ix.summary()
        traj = [{"iter": i["label"], "far_pct": i["far_pct"], "threshold": i["threshold"],
                 "val_loss": i["val_loss"], "train_rows": i["train_rows"]} for i in s["iterations"]]
        n_fa = int((self.inf["outcome"] == "false_alarm").sum())
        n_miss = int((self.inf["outcome"] == "missed_defect").sum())
        return {"run_id": s["run_id"], "kpi_target": s["kpi_target"],
                "best_iteration": self.best, "threshold": self.thr,
                "best_far_pct": self.state.get("best_far_pct"),
                "kpi_met": self.state.get("kpi_met"),
                "kpi_images": s["counts"]["kpi"],
                "false_alarms": n_fa, "missed_defects": n_miss,
                "trajectory": traj}, []

    def data_breakdown(self):
        """THE tool for 'summary breakdown of data'. Full census of the run's
        images: pool vs KPI, PASS vs NP, per-defect-type counts, provenance
        (seed / synthetic / mined / candidate), and per-iteration training rows."""
        pts = self.ix.points
        prov = {}
        for p in pts:
            prov[p["provenance"]] = prov.get(p["provenance"], 0) + 1
        # KPI defect-type distribution
        kpi_def = {}
        for p in pts:
            if p["kind"] == "kpi" and p["defect_type"]:
                kpi_def[p["defect_type"]] = kpi_def.get(p["defect_type"], 0) + 1
        # synthetic (generated) defect distribution
        syn_def = {}
        for p in pts:
            if p["provenance"] == "synthetic" and p["defect_type"]:
                syn_def[p["defect_type"]] = syn_def.get(p["defect_type"], 0) + 1
        kpi_pass = sum(1 for p in pts if p["kind"] == "kpi" and p["label"] == "PASS")
        kpi_np = sum(1 for p in pts if p["kind"] == "kpi" and p["label"] != "PASS")
        per_iter = [{"iter": i["label"], "train_rows": i["train_rows"],
                     "added_this_iter": i.get("added_rows"),
                     "mined_by_class": i.get("mined_by_class")}
                    for i in self.ix.summary()["iterations"]]
        return {
            "totals": {"all_points": len(pts),
                       "kpi": sum(1 for p in pts if p["kind"] == "kpi"),
                       "pool": sum(1 for p in pts if p["kind"] == "pool")},
            "provenance": prov,
            "kpi": {"total": kpi_pass + kpi_np, "PASS": kpi_pass, "NP": kpi_np,
                    "by_defect_type": dict(sorted(kpi_def.items(), key=lambda kv: -kv[1]))},
            "synthetic_generated": {"total": sum(syn_def.values()),
                                    "by_defect_type": dict(sorted(syn_def.items(), key=lambda kv: -kv[1]))},
            "per_iteration_training": per_iter,
        }, []

    # --------------------------------------------------------- failure scope
    def list_failures(self, kind: str = "false_alarm", sort: str = "worst",
                      limit: int = 10):
        """Rank KPI failures at the best model's operating point.
        kind = false_alarm (PASS scored as defect) | missed_defect | all.
        sort = 'worst' (most confident error first) | 'best' (borderline first).
        Returns component metadata and map point ids. 'worst false positive' =
        highest-scoring PASS."""
        df = self.inf
        if kind in ("false_alarm", "missed_defect"):
            df = df[df["outcome"] == kind]
        else:
            df = df[df["outcome"] != "correct"]
        # worst = furthest on the wrong side. FA: highest score; miss: lowest score.
        # 'best' inverts it (closest to the threshold first). The param is
        # advertised in the tool schema, so it must actually be honoured.
        ascending = (kind == "missed_defect")
        if str(sort).lower() == "best":
            ascending = not ascending
        df = df.sort_values("siamese_score", ascending=ascending)
        rows, refs = [], []
        for r in df.head(int(limit)).itertuples(index=False):
            pid = int(r.point_id) if pd.notna(r.point_id) else None
            if pid is not None:
                refs.append(pid)
            rows.append({
                "point_id": pid, "object_name": r.object_name,
                "label": "PASS" if r.is_pass else self._canon(r.label),
                "score": round(r.siamese_score, 4), "threshold": round(self.thr, 4),
                "margin": round(r.margin, 4), "outcome": r.outcome,
                "board": r.boardname, "project": r.project,
                "comp_type": r.comp_type_2, "part_type": r.part_type,
                "pins": r.number_of_pins, "description": r.description,
            })
        return {"kind": kind, "shown": len(rows),
                "total_of_kind": int((self.inf["outcome"] == kind).sum()) if kind != "all"
                else int((self.inf["outcome"] != "correct").sum()),
                "failures": rows}, refs

    def defect_breakdown(self):
        """Per-defect-type performance at the best model: count in KPI, how many
        caught vs missed, and the tightest margin (closest to being missed).
        'most failing defect class' = highest missed count / smallest min margin."""
        ng = self.inf[~self.inf["is_pass"]].copy()
        ng["defect"] = ng["label"].map(self._canon)
        rows = []
        for d, g in ng.groupby("defect"):
            missed = int((g["outcome"] == "missed_defect").sum())
            rows.append({"defect_type": d, "kpi_count": len(g),
                         "caught": len(g) - missed, "missed": missed,
                         "min_margin": round(float(g["margin"].min()), 4),
                         "median_score": round(float(g["siamese_score"].median()), 4)})
        rows.sort(key=lambda r: (-r["missed"], r["min_margin"]))
        return {"threshold": round(self.thr, 4), "defects": rows}, []

    def failure_by(self, column: str = "comp_type_2", kind: str = "false_alarm",
                   limit: int = 12):
        """Slice failures by a component-metadata column to localize WHERE the
        model struggles. column in comp_type_1|comp_type_2|part_type|
        number_of_pins|boardname|project|description. Returns per-group
        failure rate, sorted worst-first (min 5 samples/group)."""
        if column not in META_COLS:
            return {"error": f"column must be one of {META_COLS}"}, []
        df = self.inf
        pop = df[df["is_pass"]] if kind == "false_alarm" else df[~df["is_pass"]]
        badcol = "false_alarm" if kind == "false_alarm" else "missed_defect"
        rows = []
        for val, g in pop.groupby(column):
            n = len(g)
            bad = int((g["outcome"] == badcol).sum())
            if n >= 5:
                rows.append({column: str(val), "n": n, "failures": bad,
                             "rate_pct": round(100 * bad / n, 1)})
        rows.sort(key=lambda r: (-r["rate_pct"], -r["n"]))
        return {"column": column, "kind": kind, "groups": rows[:int(limit)]}, []

    # --------------------------------------------------------- coverage scope
    def coverage_census(self, point_id: int = None, defect_type: str = None,
                        min_similarity: float = 0.9, topn: int = 8):
        """Is a failure a DATA GAP or a HARD CASE? For a pinned point (or a
        defect class), count how many similar images (SigLIP cosine >=
        min_similarity) are already in TRAINING vs sit UNUSED in the mining
        pool. High in-training + still failing -> tune/hard case; unused > 0 ->
        mine; ~0 anywhere -> generate (covered defect) or collect real."""
        E = self.ix.E
        mask = self.ix._siglip_mask
        if point_id is None:
            # class-level: use the centroid of that defect's KPI failures
            ids = [p_id for p_id, p in enumerate(self.ix.points)
                   if p["kind"] == "kpi" and p["defect_type"] == defect_type and mask[p_id]]
            if not ids:
                return {"error": f"no embedded KPI points for defect '{defect_type}'"}, []
            q = E[ids].mean(0)
            q = q / (np.linalg.norm(q) or 1.0)
            focus = {"defect_type": defect_type, "n_query_points": len(ids)}
        else:
            point_id = int(point_id)
            if not mask[point_id]:
                return {"error": "point has no embedding"}, []
            q = E[point_id]
            focus = {"point_id": point_id,
                     "object_name": self.ix.points[point_id].get("object_name")}
        sims = E @ q
        train_hits, pool_hits = [], []
        for j in np.argsort(-sims):
            j = int(j)
            if not mask[j] or (point_id is not None and j == point_id):
                continue
            if sims[j] < min_similarity:
                break
            p = self.ix.points[j]
            if p["kind"] == "kpi":
                continue
            entered = self.ix.membership.get(p["key"], "")
            if entered:  # this pool image is IN training (any iter)
                train_hits.append((j, float(sims[j]), p))
            elif p["provenance"] == "candidate":  # unused mining-pool image
                pool_hits.append((j, float(sims[j]), p))
        def fmt(hits):
            return [{"point_id": j, "cosine": round(s, 4),
                     "provenance": p["provenance"], "object_name": p.get("object_name")}
                    for j, s, p in hits[:topn]]
        route = ("hard_case_or_tune" if len(train_hits) >= 5
                 else "mine_unused_pool" if pool_hits
                 else "generate_or_collect")
        refs = [j for j, _, _ in (train_hits[:topn] + pool_hits[:topn])]
        return {**focus, "min_similarity": min_similarity,
                "in_training": len(train_hits), "unused_pool": len(pool_hits),
                "route_hint": route,
                "training_neighbors": fmt(train_hits),
                "unused_pool_neighbors": fmt(pool_hits)}, refs

    # ------------------------------------------------------------ view images
    def view_images(self, images: list, thumb: int = 384, light: str = ""):
        """LOOK at up to 6 images (map point ids), returned as pictures. Use
        before any visual claim. For a KPI failure, also pass its golden by id.

        On a multi-lighting run every capture of each component is returned
        (tagged with its lighting name), since a defect may only be visible
        under one illumination. Pass `light` to restrict to one."""
        out, refs = [], []
        for ref in list(images)[:6]:
            try:
                pid = int(ref)
            except (TypeError, ValueError):
                continue
            if not (0 <= pid < len(self.ix.points)):
                continue
            # A component is captured once per lighting condition, and a defect
            # is often visible under only one of them. Show every capture this
            # run has, so a visual claim is made on the whole evidence.
            wanted = [light] if light else self.ix.lights
            for lt in wanted:
                p = Path(self.ix.image_path(pid, lt))
                if not p.is_file():
                    continue
                im = Image.open(p).convert("RGB")
                im.thumbnail((thumb, thumb))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=80)
                entry = {"ref": pid, "b64": base64.b64encode(buf.getvalue()).decode()}
                if len(self.ix.lights) > 1:
                    entry["light"] = lt
                out.append(entry)
            refs.append(pid)
        return {"images": out}, refs


TOOL_SCHEMAS = [
    {"name": "run_overview", "description": "Run headline: best iteration, FAR @ recall=100%, threshold, KPI met?, false-alarm & missed-defect counts, and the FAR/val_loss/train-rows trajectory across iterations. Use for 'how did the run do', 'why did it regress', overall status.", "parameters": {"type": "object", "properties": {}}},
    {"name": "data_breakdown", "description": "THE tool for 'give me the summary breakdown of data'. Full census: all points, KPI vs pool, PASS vs NP, KPI per-defect-type counts, synthetic (AnomalyGen) generated defects by type, provenance split (seed/synthetic/mined/candidate), and per-iteration training composition. Use for any 'how much data / breakdown / what do we have' question.", "parameters": {"type": "object", "properties": {}}},
    {"name": "list_failures", "description": "Rank the model's KPI failures at the deployed operating point. kind='false_alarm' (a good board flagged as defect — use for 'worst false positive'), 'missed_defect', or 'all'. Returns each failure's score, margin, outcome, component metadata (board/comp_type/part_type/pins/description) and map point id. Worst = most confident error.", "parameters": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["false_alarm", "missed_defect", "all"]}, "sort": {"type": "string"}, "limit": {"type": "integer"}}}},
    {"name": "defect_breakdown", "description": "Per-defect-type performance at the best model: KPI count, caught vs missed, tightest margin (closest to being missed), median score. THE tool for 'most failing defect class' / 'which defect is at risk'.", "parameters": {"type": "object", "properties": {}}},
    {"name": "failure_by", "description": "Localize WHERE the model struggles by slicing failures by a component-metadata column. column in comp_type_1|comp_type_2|part_type|number_of_pins|boardname|project|description; kind='false_alarm'|'missed_defect'. Returns per-group failure rate, worst-first. THE tool for 'which component types drive false alarms'.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "kind": {"type": "string", "enum": ["false_alarm", "missed_defect"]}, "limit": {"type": "integer"}}, "required": ["column"]}},
    {"name": "coverage_census", "description": "For a failure (pinned point_id) or a defect_type, count how many SIMILAR images (SigLIP cosine>=min_similarity) are already IN TRAINING vs sit UNUSED in the mining pool. Grounds 'how to improve': high in_training + still failing -> tune/hard-case; unused_pool>0 -> mine; ~0 anywhere -> generate (if AnomalyGen covers it) or collect real. Pass point_id (preferred) or defect_type.", "parameters": {"type": "object", "properties": {"point_id": {"type": "integer"}, "defect_type": {"type": "string"}, "min_similarity": {"type": "number"}, "topn": {"type": "integer"}}}},
    {"name": "view_images", "description": "LOOK at up to 6 images (map point ids) — returned to you as pictures. ALWAYS call this before describing what an image looks like. For a KPI failure you can also pass nearby/golden point ids to compare. On a multi-lighting run each component is returned once per lighting condition, tagged with its name — a defect is often visible under only one illumination, so consider them all before judging. Pass 'light' to restrict to a single one.", "parameters": {"type": "object", "properties": {"images": {"type": "array", "items": {"type": "integer"}}, "thumb": {"type": "integer"}, "light": {"type": "string"}}, "required": ["images"]}},
]
