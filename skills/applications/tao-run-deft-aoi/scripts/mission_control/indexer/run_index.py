# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEFT run indexer — normalizes a PCB DEFT ${RESULTS_DIR} + workspace into the
JSON model the Mission Control frontend consumes.

Identity model: every image is keyed by its resolved absolute path, so whichever table references a given crop — the
KPI test CSV, an iteration's mining-pool provenance row, or an embedding row —
all unify to a single point on the map.

Provenance comes from each iteration's own
``iterN/dataset/train_combined_<iter>_provenance.csv`` ``source`` channel
(base_train / previous_iter_train / mining_pool / anomalygen), mapped to a
display facet via ``PROVENANCE_MAP`` (seed / mined / synthetic).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import images, layout

PROVENANCE_MAP = {
    "base_train": "seed",
    "previous_iter_train": "seed",
    "mining_pool": "mined",
    "anomalygen": "synthetic",
}


def _norm_label(label: str) -> str:
    """Binary display label: PASS stays PASS, everything else is NP."""
    return "PASS" if str(label).strip().upper() == "PASS" else "NP"


PATH_MAP: list[tuple[str, str]] = []  # (origin_prefix, local_prefix) — set for bundles
WS_ROOT: str | None = None  # workspace root; relative embedding paths resolve against it


def _set_ws_root(w: str) -> None:
    global WS_ROOT
    WS_ROOT = w


def _real(p: Path | str) -> str:
    s = str(p)
    # portable caches store workspace-relative paths -> resolve against the loaded
    # workspace so the map works wherever the data lives (no bundle_origin needed).
    # Absolute paths (legacy caches) fall through to the PATH_MAP bundle remap.
    if WS_ROOT and not os.path.isabs(s):
        s = os.path.join(WS_ROOT, s)
    for src, dst in PATH_MAP:
        if s.startswith(src):
            s = dst + s[len(src):]
            break
    return os.path.realpath(s)


class RunIndex:
    PROJECTIONS = ("tsne", "umap")

    def __init__(self, results_dir: str, artifacts_dir: str = None,
                 projection: str = None):
        """Load a finished DEFT run and build its in-memory point model.

        Reads the run's state and log, works out the crop-filename convention,
        then calls ``_build()`` to turn every image into a map point. Generated
        artifacts default to ``mission_control/`` inside the run dir, so a run's
        derived data lives with the run; pass ``artifacts_dir`` only to override.
        Raises RuntimeError when the dir has no ``deft_state.json``.
        """
        self.rd = Path(results_dir)
        self.cache = Path(artifacts_dir) if artifacts_dir else (self.rd / "mission_control")
        self.cache.mkdir(parents=True, exist_ok=True)
        # The build records which projection produced its coords, so serving
        # needs no flag — it reproduces whatever prepare.py laid down.
        self.projection = projection or self._recorded_projection() or "tsne"
        if self.projection not in self.PROJECTIONS:
            raise RuntimeError(
                f"unknown projection {self.projection!r}; expected one of "
                f"{', '.join(self.PROJECTIONS)}")
        state_f = self.rd / "deft_state.json"
        if not state_f.is_file():
            raise RuntimeError(
                f"{state_f} not found — not a standard DEFT results dir. "
                "(Non-standard/script-driven runs need an adapter.)"
            )
        self.state = json.loads(state_f.read_text())
        log_f = self.rd / "loop_log.jsonl"
        self.log = [json.loads(l) for l in open(log_f)] if log_f.is_file() else []

        # Workspace inputs come from the run's own record (deft_state.config)
        # where those paths still resolve, else from the documented layout.
        self.paths = layout.resolve(self.rd, self.state)
        self.ws = self.paths["workspace"]
        self.images_dir = self.paths["images_dir"]
        warn = layout.describe(self.paths)
        if warn:
            print(warn)
        _set_ws_root(str(self.ws))       # relative embedding filepaths resolve against here

        spec_f = self.rd / "baseline_spec.yaml"
        _spec = None
        if spec_f.is_file():
            _spec = yaml.safe_load(spec_f.read_text())
        self.lights, self.ext = images.read_lights_ext(_spec)
        self.light = self.lights[0]
        self._load_bundle_map()
        self.iter_order = self._iter_order()
        self._E = None
        self._siglip_mask = None
        self._spaces = None
        self._space_names = ["siglip"]
        self._lazy = False
        self._build()

    def _recorded_projection(self):
        """The projection the serve cache was built with, if there is one."""
        _, mf = self._serve_files()
        if not mf.is_file():
            return None
        try:
            return json.loads(mf.read_text()).get("projection")
        except Exception:  # noqa: BLE001
            return None

    def _load_bundle_map(self):
        """Portable-bundle support: if the workspace carries a bundle_origin.json
        (written when a run is exported into the git repo), remap the origin
        machine's absolute path prefixes to this machine's equivalents so all
        recorded paths (parquets, CSVs, sidecars) resolve locally."""
        f = self.ws / "metadata/bundle_origin.json"
        PATH_MAP.clear()
        if not f.is_file():
            return
        origin = json.loads(f.read_text())
        repo_root = self.ws
        while repo_root != repo_root.parent and not (repo_root / ".git").exists():
            repo_root = repo_root.parent
        mapping = []
        if origin.get("origin_workspace"):
            mapping.append((origin["origin_workspace"], str(self.ws)))
        if origin.get("origin_repo_root"):
            mapping.append((origin["origin_repo_root"], str(repo_root)))
        if origin.get("origin_run_dir"):
            mapping.append((origin["origin_run_dir"], str(self.rd)))
        PATH_MAP.extend(sorted(mapping, key=lambda t: -len(t[0])))

    # ---------------------------------------------------------------- helpers
    def _iter_order(self):
        labels = [k for k in self.state.get("iterations", {})]
        base = [l for l in labels if l == "baseline"]
        iters = sorted(
            (l for l in labels if l.startswith("iter")), key=lambda s: int(s[4:])
        )
        return base + iters

    def _build_defect_canon(self):
        """Canonical defect-type vocabulary = the KPI test-set's own names
        (Missing, Excess_Solder, Shift, …). The DEFT loop lowercases synthetic
        labels for training; this maps them back onto the KPI casing. Derived
        from kpi/testing_set.csv only (no embeddings), so it's built on every
        path — the RCA tools need it regardless of how points were loaded."""
        canon = {}
        ts_f = self.paths["kpi_test_csv"]
        if ts_f.is_file():
            for lbl in pd.read_csv(ts_f, dtype=str)["label"].dropna().unique():
                if _norm_label(lbl) != "PASS":
                    canon[str(lbl).strip().lower()] = str(lbl).strip()
        return canon

    # ------------------------------------------------------------------ build
    def _build(self):
        """Assemble ``self.points`` — the whole map model — from the run's files.

        Loads every image's SigLIP fingerprint,
        makes one point per image tagged with kind / label / defect_type /
        provenance / split, merges in the mining-pool images, then computes each
        point's (x, y) via t-SNE. Also fills the side tables (``self.meta``,
        ``self.membership``, ``self.kpi_scores``).

        Sets ``self.points``, ``self.key2idx``, ``self.E`` and the coords.
        Returns None (populates instance state).
        """
        self.meta = self._load_meta()
        self.membership = self._training_membership()
        self.kpi_scores = self._kpi_scores()
        self._defect_canon = self._build_defect_canon()

        if self._load_serve():
            return

        emb_f = self.cache / "embeddings.parquet"
        if not emb_f.is_file():
            raise RuntimeError(
                f"No embeddings at {emb_f}. Run the build step first:\n"
                f"    .venv/bin/python prepare.py --run {self.rd}\n"
                "to SigLIP-embed the run's images. This is required — a DEFT run "
                "does not produce these embeddings itself."
            )
        emb = pd.read_parquet(emb_f)
        emb["key"] = emb["filepath"].map(_real)
        emb = emb.drop_duplicates("key", keep="first")

        SPLIT_PROV = {"train": "seed", "val": "seed", "synth": "synthetic",
                      "mine": "mined"}

        canon = self._defect_canon

        def canon_defect(name):
            k = str(name).strip().lower()
            return canon.get(k) or str(name).strip().title()

        pts = []
        seen = set()
        for r in emb.itertuples(index=False):
            seen.add(r.key)
            is_kpi = (r.kind == "kpi")
            label = _norm_label(r.label)
            defect = "" if label == "PASS" else canon_defect(r.label)
            m = self.meta.get(r.key, {})
            vec = np.asarray(r.embedding, dtype=np.float32) if r.embedding is not None else None
            split = "kpi" if is_kpi else (m.get("split") or str(r.split) or "pool")
            prov = ("kpi" if is_kpi
                    else m.get("provenance") or SPLIT_PROV.get(str(r.split), "pool"))
            pts.append({
                "key": r.key, "kind": "kpi" if is_kpi else "pool",
                "label": label, "defect_type": defect,
                "kpi_defect_type": defect if is_kpi else "",
                "provenance": prov, "split": split, "role": split,
                "object_name": str(r.object_name), "folder": str(r.object_name),
                "board": str(r.board), "emb": vec,
            })

        for lbl in self.iter_order:
            src_glob = sorted((self.rd / lbl / "mining_results").glob("*/source_embeddings.parquet"))
            if not src_glob:
                continue
            sdf = pd.read_parquet(src_glob[-1])   # latest attempt wins (see _mining_dir)
            for r in sdf.itertuples(index=False):
                key = _real(r.filepath)
                if key in seen:
                    continue
                seen.add(key)
                pts.append({
                    "key": key, "kind": "pool",
                    "label": _norm_label(getattr(r, "label", "PASS")),
                    "defect_type": "", "kpi_defect_type": "",
                    "provenance": "mined" if key in self.membership else "candidate",
                    "split": "mine", "role": "mine",
                    "object_name": str(getattr(r, "object_name", Path(key).stem)),
                    "folder": str(getattr(r, "object_name", Path(key).stem)),
                    "board": "mining_pool",
                    "emb": np.asarray(r.embedding, dtype=np.float32) if r.embedding is not None else None,
                })
            break

        self.points = pts
        self.key2idx = {p["key"]: i for i, p in enumerate(pts)}
        self._path2idx = dict(self.key2idx)
        self._project()
        self._compute_spaces(project=True)
        self._dump_serve()

    def _mining_dir(self, iteration: str):
        """The timestamped mining_results/<ts>/ dir for an iteration (7.0.1
        writes embeddings + mined.parquet + mining_spec.yaml here), or None.

        Selection is by RECENCY, not by which files a dir happens to hold: dir
        names are <YYYY-MM-DD_HHMMSS> so they sort chronologically, and a re-run
        supersedes the earlier (possibly crashed) attempt. Picking by artifact
        presence instead would let a crashed attempt that got as far as writing
        mining_spec.yaml beat a later, complete attempt that legitimately has no
        spec (real runs vary — this run's iter3 has embeddings but no spec).
        """
        root = self.rd / iteration / "mining_results"
        if not root.is_dir():
            return None
        cands = sorted({
            p.parent
            for name in ("mining_spec.yaml", "source_embeddings.parquet", "mined.parquet")
            for p in root.glob(f"*/{name}")
        })
        return cands[-1] if cands else None

    def _kept_keys(self, iteration: str) -> set:
        """Resolved paths of the real neighbours mining actually retained this
        iteration. Prefers mined.parquet; if the run skipped it, recompute the
        kept set from the run's own {target,source}_embeddings at its own recipe
        (topn + cosine threshold) — this reproduces knn_summary.csv exactly,
        whereas the combined-CSV guess can under-resolve. Last resort: the
        mining_pool.csv source=='mining_pool' rows."""
        md = self._mining_dir(iteration)
        mined_f = md / "mined.parquet" if md else None
        if mined_f and mined_f.is_file():
            return {_real(f) for f in pd.read_parquet(mined_f)["filepath"]}
        # recompute from embeddings (exact; matches knn_summary)
        if md and (md / "target_embeddings.parquet").is_file() \
                and (md / "source_embeddings.parquet").is_file():
            spec = self.mining_spec(iteration)
            tgt = pd.read_parquet(md / "target_embeddings.parquet")
            src = pd.read_parquet(md / "source_embeddings.parquet")
            T = np.vstack(tgt["embedding"].to_numpy())
            T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
            S = np.vstack(src["embedding"].to_numpy())
            S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
            sims = T @ S.T
            s_keys = [_real(f) for f in src["filepath"]]
            s_labels = src["label"].astype(str).to_numpy() if "label" in src else None
            thr, topn, flab = spec["min_similarity"], spec["topn"], spec["filter_by_label"]
            kept = set()
            for ti in range(sims.shape[0]):
                t_label = str(tgt["label"].iloc[ti]) if "label" in tgt else ""
                rank = 0
                for c in np.argsort(-sims[ti]):
                    c = int(c)
                    if flab and s_labels is not None and s_labels[c] != t_label:
                        continue
                    rank += 1
                    if rank > topn:
                        break
                    if sims[ti, c] >= thr:
                        kept.add(s_keys[c])
            return kept
        mp = self.rd / iteration / "mining_filter/mining_pool.csv"
        if not mp.is_file():
            return set()
        df = pd.read_csv(mp, dtype=str).fillna("")
        df = df[df.get("source", "") == "mining_pool"] if "source" in df.columns else df.iloc[0:0]
        return {images.component_file(self.ws, r.input_path, r.object_name, self.light, self.ext)
                for r in df.itertuples(index=False)}

    def mining_spec(self, iteration: str) -> dict:
        """The knobs DEFT actually mined this iteration with.

        The two sources spell the same knobs differently — the YAML is the
        mining container's own vocabulary, the config is the loop's — so the
        returned dict is canonical and uses the YAML's names:

        =============== ============= ==========================================
        returned key    mining_spec.  config.mining_filter.
                        yaml
        =============== ============= ==========================================
        topn            topn          top_k_per_target
        knn_metric      knn_metric    metric
        min_similarity  (absent)      min_similarity
        filter_by_label filter_by_label  (absent)
        =============== ============= ==========================================

        Per key, the iteration's own ``mining_results/<ts>/mining_spec.yaml``
        wins where it says anything and the run-level config fills the rest, so
        a partial YAML blends rather than replacing. ``min_similarity`` is never
        in the YAML — the loop applies that filter in Python — and
        ``filter_by_label`` is never in the config, so a YAML that omits it
        reads as False rather than falling back.
        """
        cfg = self.state.get("config", {})
        mcfg = cfg.get("mining_filter", {})
        spec = {"topn": int(mcfg.get("top_k_per_target", cfg.get("top_k_per_target", 5))),
                "knn_metric": str(mcfg.get("metric", "cosine")),
                "filter_by_label": False,
                "min_similarity": float(mcfg.get("min_similarity", cfg.get("min_similarity", 0.9)))}
        md = self._mining_dir(iteration)
        f = md / "mining_spec.yaml" if md else None
        if f and f.is_file():
            raw = yaml.safe_load(f.read_text()) or {}
            spec["topn"] = int(raw.get("topn", spec["topn"]))
            spec["knn_metric"] = str(raw.get("knn_metric", spec["knn_metric"]))
            spec["filter_by_label"] = str(raw.get("filter_by_label", "false")).lower() == "true"
            if raw.get("min_similarity") is not None:
                spec["min_similarity"] = float(raw["min_similarity"])
        return spec

    def neighbors(self, i: int, iteration: str, k: int = 0, same_label: str = ""):
        """Neighbors of point i replayed with the iteration's own mining
        recipe: candidates are the mining source pool (split=mine), ranked by
        cosine in the SigLIP mining space, label-filtered when the run mined
        that way. k=0 means the run's own topn. same_label overrides the
        run's filter_by_label as a what-if ("1"/"0"; "" = follow the run)."""
        space_name = "siglip"   # the only space tao-run-deft-aoi mines in
        sp = self.spaces[space_name]
        E, mask = sp["E"], sp["mask"]
        if not mask[i]:
            return {"error": f"point has no embedding in {space_name}", "space": space_name}
        spec = self.mining_spec(iteration)
        label_filter = spec["filter_by_label"] if same_label == "" else same_label == "1"
        k = k or spec["topn"]
        kept_keys = self._kept_keys(iteration)
        me = self.points[i]
        sims = E @ E[i]
        order = np.argsort(-sims)
        rows, pool_rank = [], 0
        for j in order:
            j = int(j)
            p = self.points[j]
            if j == i or p.get("split") != "mine" or not mask[j]:
                continue
            if label_filter and p["label"] != me["label"]:
                continue
            pool_rank += 1
            rows.append({k2: _json_safe(v) for k2, v in {
                "id": j, "filename": Path(p["key"]).name,
                "cosine": round(float(sims[j]), 4),
                "provenance": p.get("provenance", ""), "split": p.get("split", ""),
                "label": p["label"], "defect_type": p.get("defect_type", ""),
                "retrieved": pool_rank <= spec["topn"],
                "above_thr": float(sims[j]) >= spec["min_similarity"],
                "kept": p["key"] in kept_keys,
                "entered_training": self.membership.get(p["key"], ""),
            }.items()})
            if len(rows) >= k:
                break
        siamese = (self.kpi_scores.get(me["key"], {}).get(iteration)
                   if me["kind"] == "kpi" else None)
        return {"space": space_name, "iteration": iteration,
                "siamese_score": _json_safe(siamese),
                "neighbors": rows, **spec,
                "label_filter_active": label_filter,
                "label_filter_is_runs": same_label == ""}

    def weak_targets(self, iteration: str):
        """The weak KPI samples that fed this iteration's mining — DEFT's
        routing output (routing_results/<ts>/mining_gaps.parquet: filepath,
        label, siamese_score, weakness), sorted by weakness."""
        cands = sorted((self.rd / iteration / "routing_results").glob("*/mining_gaps.parquet"))
        if not cands:
            return {"iteration": iteration, "targets": []}
        # latest routing attempt wins (see _mining_dir)
        df = pd.read_parquet(cands[-1]).sort_values("weakness", ascending=False)
        rows = []
        for r in df.itertuples():
            i = self._path2idx.get(_real(r.filepath))
            rows.append({
                "id": i, "folder": Path(r.filepath).stem.replace(f"_{self.light}", ""),
                "label": str(r.label),
                "siamese_score": _json_safe(r.siamese_score),
                "weakness": _json_safe(r.weakness),
            })
        return {"iteration": iteration, "targets": rows}

    def mining_edges(self, iteration: str):
        """Per-iteration k-NN retrieval edges, replayed with the iteration's
        own mining recipe (mining_spec.yaml: topn / label filter / cosine
        threshold) from the run's own target/source embedding parquets."""
        md = self._mining_dir(iteration)
        if md is None:
            return {"iteration": iteration, "edges": [], "note": "no mining artifacts"}
        tgt_f = md / "target_embeddings.parquet"
        src_f = md / "source_embeddings.parquet"
        if not (tgt_f.is_file() and src_f.is_file()):
            return {"iteration": iteration, "edges": [], "note": "no mining artifacts"}
        spec = self.mining_spec(iteration)
        tgt = pd.read_parquet(tgt_f)
        src = pd.read_parquet(src_f)
        kept = self._kept_keys(iteration)  # mined.parquet, or mining_pool.csv fallback
        T = np.stack(tgt["embedding"].values)
        T = T / np.linalg.norm(T, axis=1, keepdims=True)
        S = np.stack(src["embedding"].values)
        S = S / np.linalg.norm(S, axis=1, keepdims=True)
        sims = T @ S.T
        src_keys = [_real(f) for f in src["filepath"]]
        src_labels = src["label"].astype(str).values if "label" in src else None

        def to_idx(key):
            return self._path2idx.get(key)  # source-pool points are keyed by abs path

        edges = []
        for r, trow in enumerate(tgt.itertuples()):
            ti = self._path2idx.get(_real(trow.filepath))
            if ti is None:
                continue
            t_label = str(getattr(trow, "label", ""))
            cand = np.argsort(-sims[r])
            if spec["filter_by_label"] and src_labels is not None:
                cand = [c for c in cand if src_labels[int(c)] == t_label]
            rank = 0
            for c in cand:
                c = int(c)
                rank += 1  # a slot is consumed even if the point can't be drawn
                if rank > spec["topn"]:
                    break
                si = to_idx(src_keys[c])
                if si is None:
                    continue
                cos = float(sims[r, c])
                edges.append({
                    "target": ti, "neighbor": si, "rank": rank,
                    "cosine": round(cos, 4),
                    "above_thr": cos >= spec["min_similarity"],
                    "kept": src_keys[c] in kept,
                    "target_label": t_label,
                    "target_siamese": _json_safe(getattr(trow, "siamese_score", None)),
                    "target_weakness": _json_safe(getattr(trow, "weakness", None)),
                })
        return {"iteration": iteration, "edges": edges, **spec,
                "encoder": "siglip",   # tao-run-deft-aoi always mines with SigLIP
                "n_targets": len(tgt), "n_kept_edges": sum(1 for e in edges if e["kept"])}

    # ------------------------------------------------- extra embedding spaces
    def _compute_spaces(self, project=True):
        """Build the SigLIP {E, mask} matrix. When ``project`` is True (full
        build) it also writes per-point coords from the t-SNE already computed
        in ``_build``; when False (lazy path, coords already restored from the
        serve cache) it builds only the matrix needed for neighbor/coverage
        queries. Sets ``self._spaces`` and ``self._space_names``.

        SigLIP is the only space. ``tao-run-deft-aoi`` mines with SigLIP
        (``references/tao-mine-aoi-images.md``) and never writes per-iteration
        ChangeNet-encoder embeddings, so there is nothing else to build.
        """
        self._spaces = {"siglip": {"E": self.E, "mask": self._siglip_mask}}
        if project:
            for p in self.points:
                p["coords"] = {"siglip": [p["x"], p["y"]] if p["x"] is not None else None}
        self._space_names = list(self._spaces.keys())

    def get_space(self, space: str = "siglip"):
        return self.spaces.get(space) or self.spaces["siglip"]

    # ---------------------------------------------- lazy vectors / serve cache
    @property
    def E(self):
        """Full SigLIP matrix (N×D, normalized rows), built on first access.
        Only the RCA coverage tool needs it; the map and most sessions never do,
        so it stays out of the serving process until actually required."""
        if self._E is None:
            self._load_full_matrix()
        return self._E

    @property
    def spaces(self):
        """Per-space {E, mask} for neighbor replay, built on first access."""
        if self._spaces is None:
            self._compute_spaces(project=False)  # coords already loaded; no t-SNE
        return self._spaces

    def _serve_files(self):
        return (self.cache / "serve_points.parquet", self.cache / "serve_meta.json")

    def _dump_serve(self):
        """Persist the assembled points (minus vectors, plus coords) so the next
        load skips the embedding column, t-SNE, and the resident matrix.
        Best-effort — never fails the build."""
        try:
            pf, mf = self._serve_files()
            rows = []
            for p in self.points:
                q = {k: v for k, v in p.items() if k not in ("emb", "coords")}
                q["coords_json"] = json.dumps(p.get("coords"))
                rows.append(q)
            pd.DataFrame(rows).to_parquet(pf, index=False)
            mf.write_text(json.dumps({"space_names": list(self._space_names),
                                      "n": len(self.points),
                                      "projection": self.projection}))
        except Exception as e:  # noqa: BLE001
            print(f"[serve-cache] dump skipped ({type(e).__name__}: {e})")

    def _load_serve(self) -> bool:
        """Load points + coords from the serve cache. Returns False (→ full
        build) if it is absent, older than embeddings.parquet (stale after a
        re-embed), or older than the run's newest iteration (stale after a live
        run advanced — this is what makes /api/reload pick up new iterations)."""
        pf, mf = self._serve_files()
        if not (pf.is_file() and mf.is_file()):
            return False
        cache_mtime = pf.stat().st_mtime
        emb_f = self.cache / "embeddings.parquet"
        if emb_f.is_file() and emb_f.stat().st_mtime > cache_mtime:
            return False  # embeddings re-generated → serve cache is stale
        # A live run that gained an iteration (or re-ran one) invalidates the
        # cache too; without this the map silently pins to the old iteration set.
        for d in self.rd.glob("iter*"):
            if d.is_dir() and d.stat().st_mtime > cache_mtime:
                return False
        try:
            df = pd.read_parquet(pf)
            meta = json.loads(mf.read_text())
        except Exception:  # noqa: BLE001
            return False
        # Coords carry no method of their own; without this a switched
        # projection loads the previous layout and looks merely different.
        if meta.get("projection", "tsne") != self.projection:
            return False
        pts = []
        for r in df.to_dict("records"):
            cj = r.pop("coords_json", None)
            r["coords"] = json.loads(cj) if cj else None
            for c in ("x", "y"):
                v = r.get(c)
                r[c] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
            pts.append(r)
        self.points = pts
        self.key2idx = {p["key"]: i for i, p in enumerate(pts)}
        self._path2idx = dict(self.key2idx)
        self._space_names = meta.get("space_names", ["siglip"])
        self._lazy = True
        return True

    def _load_full_matrix(self):
        """Reconstruct the N×D SigLIP matrix (normalized rows, zeros for
        unembedded points), aligned to self.points, from the same sources the
        full build uses. Called lazily the first time a vector query needs it."""
        key2vec = {}
        emb_f = self.cache / "embeddings.parquet"
        if emb_f.is_file():
            e = pd.read_parquet(emb_f)
            e["key"] = e["filepath"].map(_real)
            for r in e.itertuples(index=False):
                if getattr(r, "embedding", None) is not None:
                    key2vec[r.key] = np.asarray(r.embedding, dtype=np.float32)
        for lbl in self.iter_order:
            srcs = sorted((self.rd / lbl / "mining_results").glob("*/source_embeddings.parquet"))
            if not srcs:
                continue
            for r in pd.read_parquet(srcs[-1]).itertuples(index=False):  # latest attempt wins
                k = _real(r.filepath)
                if k not in key2vec and getattr(r, "embedding", None) is not None:
                    key2vec[k] = np.asarray(r.embedding, dtype=np.float32)
            break
        n = len(self.points)
        dim = len(next(iter(key2vec.values()))) if key2vec else 1
        E = np.zeros((n, dim), dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        for k, v in key2vec.items():
            i = self.key2idx.get(k)
            if i is None:
                continue
            E[i] = v / (np.linalg.norm(v) or 1.0)
            mask[i] = True
        self._E, self._siglip_mask = E, mask

    def _load_meta(self):
        """resolved key -> {source, provenance, split, defect_type} for pool
        images, derived from the run's own train_combined_*_provenance.csv
        ``source`` channel (base_train / previous_iter_train / mining_pool /
        anomalygen). 7.0.1 has no pool_metadata.csv sidecar and no DT render
        params — provenance is the DEFT channel mapped to a display facet."""
        images_dir = self.images_dir
        meta: dict[str, dict] = {}
        base_f = self.paths["training_csv"]
        if base_f.is_file():
            for r in pd.read_csv(base_f, dtype=str).fillna("").itertuples(index=False):
                k = images.component_file(images_dir, r.input_path, r.object_name,
                                          self.light, self.ext)
                meta[k] = {"source": "base_train", "provenance": "seed", "split": "train",
                           "defect_type": "" if _norm_label(r.label) == "PASS" else r.label}
        for lbl in self.iter_order:
            if lbl == "baseline":
                continue
            prov = self.rd / lbl / "dataset" / f"train_combined_{lbl}_provenance.csv"
            if not prov.is_file():
                continue
            for r in pd.read_csv(prov, dtype=str).fillna("").itertuples(index=False):
                src = r.source
                if src in ("base_train", "previous_iter_train"):
                    continue
                k = images.component_file(self.ws, r.input_path, r.object_name,
                                          self.light, self.ext)
                meta.setdefault(k, {
                    "source": src,
                    "provenance": PROVENANCE_MAP.get(src, src),
                    "split": {"mining_pool": "mine", "anomalygen": "synth"}.get(src, src),
                    "defect_type": "" if _norm_label(r.label) == "PASS" else r.label,
                })
        return meta

    def _fit_projection(self, Em):
        """2-D coords for the normalized embedding matrix.

        Both use cosine geometry, matching every similarity the rest of the app
        reports — a Euclidean layout would disagree with the numbers beside it.
        t-SNE keeps local neighbourhoods; UMAP additionally keeps inter-cluster
        distance meaningful, which is what makes the map readable at a glance.
        """
        if self.projection == "umap":
            import umap  # lazy: pulls numba, and only this branch needs it

            return umap.UMAP(
                n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1,
                random_state=42,
            ).fit_transform(Em)
        from sklearn.manifold import TSNE

        return TSNE(
            n_components=2, init="pca", learning_rate="auto",
            perplexity=min(30, max(2, len(Em) - 1)), random_state=42,
        ).fit_transform(Em)

    def _project(self):
        """SigLIP projection — mask-based: points without a SigLIP embedding
        (e.g. crops present in a CSV but never embedded) get x/y = None."""
        have = [p["emb"] is not None for p in self.points]
        mask = np.asarray(have, dtype=bool)
        embs = [p["emb"] for p in self.points if p["emb"] is not None]
        dim = len(embs[0]) if embs else 1
        Em = np.stack(embs) if embs else np.zeros((0, dim), np.float32)
        Em = Em / np.linalg.norm(Em, axis=1, keepdims=True)
        # Keyed by method: one file per projection, so switching recomputes
        # instead of silently reusing the other layout.
        cache_f = self.cache / f"coords_{self.projection}.npy"
        xy = np.load(cache_f) if cache_f.is_file() else None
        if xy is None or len(xy) != len(Em):
            xy = self._fit_projection(Em)
            np.save(cache_f, xy)
        E = np.zeros((len(self.points), dim), dtype=np.float32)
        E[mask] = Em
        self._E = E  # row i = point id i; zero rows for unembedded points
        self._siglip_mask = mask
        k = 0
        for p, m in zip(self.points, mask):
            if m:
                p["x"], p["y"] = float(xy[k, 0]), float(xy[k, 1])
                k += 1
            else:
                p["x"] = p["y"] = None
            del p["emb"]

    def _training_membership(self):
        """resolved image key -> first iteration label it entered training.

        Base seed rows are ``kpi/images``-relative; per-iter *added* rows come
        from ``train_combined_<iter>_provenance.csv`` where ``source`` !=
        base/previous (i.e. mining_pool + anomalygen), with paths relative to
        the workspace root (synthetic rows carry ``results/.../synthetic_*``).
        """
        member = {}
        images_dir = self.images_dir
        base_f = self.paths["training_csv"]
        if base_f.is_file():
            for r in pd.read_csv(base_f, dtype=str).fillna("").itertuples(index=False):
                k = images.component_file(images_dir, r.input_path, r.object_name,
                                          self.light, self.ext)
                member[k] = "baseline"
        for lbl in self.iter_order:
            if lbl == "baseline":
                continue
            prov = self.rd / lbl / "dataset" / f"train_combined_{lbl}_provenance.csv"
            if not prov.is_file():
                continue
            df = pd.read_csv(prov, dtype=str).fillna("")
            added = df[~df["source"].isin(("base_train", "previous_iter_train"))]
            for r in added.itertuples(index=False):
                k = images.component_file(self.ws, r.input_path, r.object_name,
                                          self.light, self.ext)
                member.setdefault(k, lbl)
        return member

    def inference_csv(self, iteration: str):
        """The iteration's inference CSV, or None when it produced none.

        Only the documented layout is accepted — ``inference/<kind>/`` per
        ``references/data-layout.md``. A run that writes elsewhere is not
        conforming, and absorbing its layout here would hide that.
        """
        kind = (self.state.get("iterations", {}).get(iteration, {})
                .get("best_ckpt_kind") or "best_val")
        for rel in (f"inference/{kind}/inference.csv",
                    "inference/best_val/inference.csv",
                    "inference/latest/inference.csv"):
            f = self.rd / iteration / rel
            if f.is_file():
                return f
        return None

    def _kpi_scores(self):
        """resolved image key -> {iter: siamese_score}.

        7.0.1 inference CSVs carry ``input_path`` (a directory) + ``object_name``;
        the on-disk crop is ``{images_dir}/{input_path}/{object_name}_{light}{ext}``.
        We key by that resolved path so every KPI point matches its score row
        (the legacy ``input_path.split('/')[-1]`` = "PerComponent", non-unique).
        """
        scores = {}
        images_dir = self.images_dir
        for lbl in self.iter_order:
            # scores must come from the SAME checkpoint as the iteration's
            # recorded threshold — the KPI winner (best_val OR latest).
            f = self.inference_csv(lbl)
            if f is None:
                continue
            # fillna: a NaN input_path would otherwise build a literal "nan"
            # path segment and silently drop this KPI point's scores.
            df = pd.read_csv(f, dtype=str).fillna("")
            for r in df.itertuples(index=False):
                key = images.component_file(images_dir, r.input_path, r.object_name,
                                            self.light, self.ext)
                scores.setdefault(key, {})[lbl] = float(r.siamese_score)
        return scores

    # ------------------------------------------------------------------- API
    def summary(self):
        """Run headline for the UI (``/api/summary``): metrics and trajectory.

        Carries the run id, KPI target, embedding spaces, best iteration, the
        per-iteration list (FAR, threshold, val_loss, row and image counts), the
        stage log and pool/KPI point counts.
        """
        iters = []
        for lbl in self.iter_order:
            st = self.state["iterations"].get(lbl, {})
            # Images entering training for the FIRST time this iteration —
            # deliberately not the row delta of train_combined_<iter>.csv, which
            # grows faster because the loop re-appends images it already has.
            added = sum(1 for v in self.membership.values() if v == lbl)
            by_prov = {}
            for p in self.points:
                if self.membership.get(p["key"]) == lbl:
                    by_prov[p["provenance"]] = by_prov.get(p["provenance"], 0) + 1
            iters.append(
                {
                    "label": lbl,
                    "far_pct": st.get("far_pct"),
                    "threshold": st.get("threshold"),
                    "val_loss": st.get("val_loss"),
                    "best_ckpt": Path(st.get("best_ckpt_path", "")).name,
                    "added_rows": added if lbl != "baseline" else None,
                    # how those new images arrived: synthetic (AnomalyGen) vs mined
                    "added_by_provenance": by_prov if lbl != "baseline" else None,
                    # rows appended to the combined CSV this iteration. THIS is
                    # what reconciles train_rows across iterations; it is >=
                    # added_rows because some appended rows repeat an image the
                    # set already had.
                    "appended_rows": self._appended_rows(lbl),
                    "mined_by_class": self._mined_by_class(lbl),
                    "train_rows": self._train_rows(lbl),
                    "note": st.get("note", ""),
                }
            )
        stages = [
            {k: e.get(k) for k in ("seq", "iter", "stage", "status", "summary", "duration_sec")}
            for e in self.log
        ]
        scored = [i for i in iters if i["far_pct"] is not None]
        # Keep the degenerate case the SAME SHAPE as a real iteration entry —
        # the UI reads best.threshold/val_loss/etc. and a short dict would make
        # those silently undefined.
        best = (min(scored, key=lambda i: i["far_pct"]) if scored
                else {**{k: None for k in iters[0]}, "label": "—"} if iters
                else {"label": "—", "far_pct": None})
        mining_cfg = self.state.get("config", {}).get("mining_filter", {})
        return {
            "run_id": self.rd.name,
            "workspace": str(self.ws),
            "spaces": list(self._space_names),
            # Lighting conditions this run captured, channel order. One entry is
            # the single-light case; the UI only offers a selector past that.
            "lights": list(self.lights),
            "mining_encoder": mining_cfg.get("encoder", "siglip"),
            "kpi_target": self.state.get("config", {}).get("kpi_target")
            or self.state.get("kpi_target", "minimize FAR @ recall=100%"),
            "iterations": iters,
            "best": best,
            "stages": stages,
            "counts": {
                "pool": sum(1 for p in self.points if p["kind"] == "pool"),
                "kpi": sum(1 for p in self.points if p["kind"] == "kpi"),
            },
        }

    def _mined_by_class(self, lbl):
        """Class composition of the rows added this iteration (synthetic NG by
        defect type + mined real PASS), read from the iteration's
        mining_filter/mining_pool.csv (columns: label, source)."""
        if lbl == "baseline":
            return None
        f = self.rd / lbl / "mining_filter/mining_pool.csv"
        if not f.is_file():
            return None
        try:
            m = pd.read_csv(f, dtype=str).fillna("")
            if not len(m):
                return {}
            counts = m["label"].astype(str).value_counts().to_dict()
            return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
        except Exception:
            return None

    def _appended_rows(self, lbl):
        """Rows this iteration appended to the combined training CSV.

        The provenance sidecar tags every row with where it came from; anything
        not carried forward (``base_train`` / ``previous_iter_train``) is new
        this iteration. This is the delta that makes train_rows add up —
        ``added_rows`` counts distinct new IMAGES and is smaller whenever the
        loop re-appends one it already had.
        """
        if lbl == "baseline":
            return None
        f = self.rd / lbl / "dataset" / f"train_combined_{lbl}_provenance.csv"
        if not f.is_file():
            return None
        df = pd.read_csv(f, dtype=str).fillna("")
        if "source" not in df.columns:
            return None
        return int((~df["source"].isin(("base_train", "previous_iter_train"))).sum())

    def _train_rows(self, lbl):
        if lbl == "baseline":
            f = self.paths["training_csv"]
        else:
            f = self.rd / lbl / "dataset" / f"train_combined_{lbl}.csv"
        return sum(1 for _ in open(f)) - 1 if f.is_file() else None

    def _display_location(self, key):
        """The crop's directory, shortened against the run's own roots.

        Kept as the raw relative path rather than parsed into a panel id and
        timestamp: those conventions belong to one customer's AOI export, and
        every DEFT AOI run has to render here.
        """
        d = Path(key).parent
        for root in (self.images_dir, self.rd, self.ws):
            try:
                return str(d.relative_to(root))
            except ValueError:
                continue
        return str(d)

    def points_payload(self):
        """Serialize every map point to JSON for the frontend (``/api/points``).

        Adds per-point ``id``, ``filename``, ``location``, ``image_url`` and
        ``first_used_iter``; KPI points also carry their per-iteration ``scores``,
        which colour them correct / false-alarm / missed.
        """
        out = []
        for i, p in enumerate(self.points):
            q = {k: _json_safe(v) for k, v in p.items() if k != "key"}
            q["id"] = i
            q["filename"] = Path(p["key"]).name  # searchable source filename
            # object_name is a board reference designator, so it repeats once per
            # inspected panel; the directory is what tells two such points apart.
            q["location"] = self._display_location(p["key"])
            q["image_url"] = f"/api/image/{i}"
            q["first_used_iter"] = self.membership.get(p["key"], "")
            if p["kind"] == "kpi":
                q["scores"] = self.kpi_scores.get(p["key"], {})
            out.append(q)
        return out

    def image_path(self, idx: int, light: str | None = None) -> str:
        """On-disk path of point ``idx``'s capture, for serving its thumbnail.

        A point is a component, captured once per lighting condition; its key is
        the channel-0 capture and the others differ only by suffix. ``light``
        defaults to channel 0. A non-channel-0 path may not exist, so callers
        should check before serving.
        """
        key = self.points[idx]["key"]
        if not light or light == self.light:
            return key
        return images.swap_light(key, self.light, light, self.ext)

    def defect_margin_table(self):
        """Per-KPI-defect-type margin to threshold, per iteration."""
        rows = []
        for lbl in self.iter_order:
            thr = self.state["iterations"].get(lbl, {}).get("threshold")
            if thr is None:
                continue
            by_type = {}
            for p in self.points:
                if p["kind"] != "kpi" or p["label"] == "PASS":
                    continue
                s = self.kpi_scores.get(p["key"], {}).get(lbl)
                if s is None:
                    continue
                by_type.setdefault(p.get("kpi_defect_type") or "?", []).append(s - thr)
            for t, margins in sorted(by_type.items()):
                rows.append(
                    {
                        "iter": lbl,
                        "kpi_defect_type": t,
                        "n": len(margins),
                        "min_margin": round(min(margins), 4),
                        "median_margin": round(float(np.median(margins)), 4),
                        "at_risk": sum(1 for m in margins if m < 0.02),
                    }
                )
        return rows


def _json_safe(v):
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    # Recurse into sequences too — a NaN nested in a list (e.g. coords) would
    # otherwise serialize as bare `NaN`, which is not valid JSON.
    if isinstance(v, np.ndarray):
        return [_json_safe(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


def _f(m, key):
    if m is None:
        return None
    v = m.get(key)
    try:
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    except (TypeError, ValueError):
        return None
