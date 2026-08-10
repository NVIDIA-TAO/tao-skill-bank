# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate a run's workspace inputs — the KPI/train CSVs and the images dir.

``deft_state.json`` records the absolute paths the loop used
(``config.images_dir``, ``kpi_test_csv``, ``training_csv``, ``validation_csv``),
so those are preferred. They are absolute paths from the machine the run
executed on, so each is used only when it still resolves on disk; otherwise the
documented workspace layout under ``<results_dir>/../..`` applies — the same
relative paths ``scripts/init_deft_state.py`` builds those config values from.
"""

from __future__ import annotations

from pathlib import Path

# Workspace layout from references/data-layout.md, used when config is absent.
FALLBACK = {
    "images_dir": "kpi/images",
    "kpi_test_csv": "kpi/testing_set.csv",
    "training_csv": "train/base/training_set.csv",
    "validation_csv": "train/base/validation_set.csv",
}


def resolve(results_dir: Path, state: dict | None = None) -> dict:
    """Work out where a run's workspace inputs live.

    Each path comes from ``state["config"]`` when it still exists on disk and
    from the workspace layout otherwise, so a run that was moved still resolves.
    Returns the four input paths plus ``workspace``, and a ``sources`` map
    recording which of the two supplied each.
    """
    rd = Path(results_dir)
    ws = rd.parent.parent
    cfg = (state or {}).get("config") or {}

    out: dict = {}
    sources: dict[str, str] = {}
    for key, rel in FALLBACK.items():
        recorded = cfg.get(key)
        if recorded and Path(recorded).exists():
            out[key] = Path(recorded)
            sources[key] = "config"
        else:
            out[key] = ws / rel
            sources[key] = "layout"

    # Workspace-relative rows (synthetic images carry `results/run_<TS>/...`)
    # resolve against this root.
    out["workspace"] = (out["images_dir"].parent.parent
                        if sources["images_dir"] == "config" else ws)
    out["sources"] = sources
    return out


def describe(resolved: dict) -> str | None:
    """Return a warning line when a run's images could not be located, else None.

    A missing images dir degrades silently — KPI scores drop to zero and
    provenance empties without raising — so it is worth saying out loud.
    """
    if resolved["images_dir"].is_dir():
        return None
    return (f"WARNING: no KPI images at {resolved['images_dir']} — expected a run at "
            f"<workspace>/results/run_<TS>/. If this run was moved, KPI scores, "
            f"defect types and training provenance will be silently incomplete.")
