#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded appearance search over a scene whose geometry and camera are frozen.

The plan JSON is written before the search and fixes everything that could be
tuned afterwards: the seed, the evaluation budget, each stage's parameters
with their bounds, and the stop rule. A plan may only vary keys under
`materials/`, `lighting/` and `render/exposure`; any other path is refused, and
so is a base look with a `camera` key. Whatever else the base look holds
(render settings, solder assets) is written once and is the same in every
evaluation. The board and the camera layer are named in the plan's render
command and are not touched by this script.

Every evaluation is two recorded steps (render, score) in the run's receipt,
so a replay repeats the search exactly. The search directory keeps one look
file per evaluation, `history.json` with every result including the ones that
did not help, and `initial.json` / `best.json`.

Plan keys:
  seed, budget, stop {"min_stage_gain": x}
  base_look            look JSON for evaluation 0 (absolute, or relative to the plan file)
  render               argv for `kicad_light.py step --tool blender`; `{look}` and `{out}` are substituted
  score                argv for `kicad_light.py step --tool python`; `{out}` is substituted; must write {out}/rois.json
  stages               list of
    {"name", "method": "lhs", "evals": n, "params": [{"path": "a/b/0/c", "range": [lo, hi], "as": "scalar"|"gray"}],
     "then_ratio": [targets]?}   # optional: one colour correction after each sample, counted as an evaluation
    {"name", "method": "ratio", "evals": n, "targets": [{"roi": name, "path": "materials/.../color"}], "tolerance": x}
`lhs` draws a Latin hypercube inside the bounds (`evals` samples; twice as many
evaluations with `then_ratio`). `ratio` rescales a base colour
by reference/render mean colour of its region in linear light (assumes the
Standard view transform) and stops once every target is within `tolerance`.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def get(node, path):
    for key in path.split("/"):
        node = node[int(key)] if isinstance(node, list) else node[key]
    return node


def put(node, path, value):
    keys = path.split("/")
    for key, following in zip(keys, keys[1:]):
        if isinstance(node, list):
            node = node[int(key)]
        else:
            node = node.setdefault(key, [] if following.isdigit() else {})
    if isinstance(node, list):
        node[int(keys[-1])] = value
    else:
        node[keys[-1]] = value


ALLOWED = ("materials/", "lighting/", "render/exposure")


def check_plan(plan, base_look):
    """Refuse a plan that could change geometry or the camera."""
    if "camera" in base_look:
        raise SystemExit("The base look must not carry a camera; keep it in its own scene layer")
    paths = []
    for stage in plan["stages"]:
        paths += [p["path"] for p in stage.get("params", [])]
        paths += [t["path"] for t in stage.get("targets", []) + stage.get("then_ratio", [])]
    refused = sorted({p for p in paths if not p.startswith(ALLOWED)})
    if refused:
        raise SystemExit("Not appearance keys (allowed: " + ", ".join(ALLOWED) + "): " + ", ".join(refused))


def linear(srgb_255):
    c = np.asarray(srgb_255, dtype=np.float64) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run", required=True, help="run directory of kicad_light.py; steps are appended to it")
    parser.add_argument("--search-dir", required=True, help="new directory for look files and history")
    parser.add_argument("--prefix", default="fit", help="step-name and output-directory prefix inside the run")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    base_look = Path(args.plan).resolve().parent / plan["base_look"]  # absolute paths pass through unchanged
    out_dir = Path(args.search_dir)
    out_dir.mkdir(parents=True)  # refuses to reuse a directory: history is never overwritten
    light = [sys.executable, str(Path(__file__).resolve().parent / "kicad_light.py")]
    history, state = [], {"count": 0}

    def evaluate(look, stage, note):
        index = state["count"]
        if index >= plan["budget"]:
            return None
        state["count"] += 1
        look_path = out_dir / f"look_{index:02d}.json"
        look_path.write_text(json.dumps(look, indent=1) + "\n")
        out = f"{{run}}/{args.prefix}/{index:02d}"
        tick = time.time()
        for tool, name, argv in (("blender", "render", plan["render"]), ("python", "score", plan["score"])):
            argv = [a.replace("{look}", str(look_path)).replace("{out}", out) for a in argv]
            subprocess.run(light + ["step", "--run", args.run, "--name", f"{args.prefix}_{name}_{index:02d}",
                                    "--tool", tool, "--"] + argv, check=True, stdout=subprocess.DEVNULL)
        scored = json.loads((Path(args.run) / args.prefix / f"{index:02d}" / "rois.json").read_text())
        row = {"evaluation": index, "stage": stage, "note": note, "look": look_path.name, "loss": scored["loss"],
               "rois": {r["roi"]: {k: r[k] for k in ("loss", "colour_term", "tone_term", "contrast_term")}
                        for r in scored["rois"] if "loss" in r},
               "seconds": round(time.time() - tick, 1)}
        row["_stats"] = {r["roi"]: (r["reference"]["mean_rgb"], r["render"]["mean_rgb"]) for r in scored["rois"] if "loss" in r}
        history.append(row)
        (out_dir / "history.partial.json").write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "_stats"} for r in history], indent=1) + "\n")
        return row

    def corrected(look, row, targets):
        """Rescale each target colour by reference/render mean colour of its region, in linear light."""
        look, chosen, far = copy.deepcopy(look), {}, 0.0
        for target in targets:
            reference, rendered = row["_stats"][target["roi"]]
            ratio = linear(reference) / np.maximum(linear(rendered), 1e-4)
            far = max(far, float(np.abs(np.asarray(reference) - np.asarray(rendered)).max()) / 255)
            value = [round(float(v), 5) for v in np.clip(np.asarray(get(look, target["path"])) * ratio, 0.001, 1.0)]
            put(look, target["path"], value)
            chosen[target["path"]] = value
        return look, chosen, far

    best_look = json.loads(base_look.read_text())
    check_plan(plan, best_look)
    best = evaluate(best_look, "initial", "base look")
    stopped = "budget"
    for number, stage in enumerate(plan["stages"]):
        before = best["loss"]
        if stage["method"] == "lhs":
            rng = np.random.default_rng(plan["seed"] + number)
            n, params = stage["evals"], stage["params"]
            grid = np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in params], axis=1)
            start_look = best_look  # every sample of a stage varies the look the stage started from
            for sample in grid:
                look, chosen = copy.deepcopy(start_look), {}
                for spec, unit in zip(params, sample):
                    value = round(float(spec["range"][0] + unit * (spec["range"][1] - spec["range"][0])), 5)
                    put(look, spec["path"], [value] * 3 if spec.get("as") == "gray" else value)
                    chosen[spec["path"]] = value
                row = evaluate(look, stage["name"], chosen)
                if row is None:
                    break
                if row["loss"] < best["loss"]:
                    best, best_look = row, look
                if stage.get("then_ratio"):  # light changes shift every diffuse colour: correct once, then judge
                    look2, chosen2, _ = corrected(look, row, stage["then_ratio"])
                    row2 = evaluate(look2, stage["name"] + "+ratio", chosen2)
                    if row2 is None:
                        break
                    if row2["loss"] < best["loss"]:
                        best, best_look = row2, look2
        elif stage["method"] == "ratio":
            for _ in range(stage["evals"]):
                look, chosen, far = corrected(best_look, best, stage["targets"])
                if far <= stage["tolerance"]:
                    break
                row = evaluate(look, stage["name"], chosen)
                if row is None:
                    break
                if row["loss"] < best["loss"]:
                    best, best_look = row, look
                else:
                    break  # the correction stopped helping
        else:
            raise SystemExit("Unknown method: " + stage["method"])
        if state["count"] >= plan["budget"]:
            break
        if number and before - best["loss"] < plan["stop"]["min_stage_gain"]:
            stopped = f"stage '{stage['name']}' gained less than {plan['stop']['min_stage_gain']}"
            break
    else:
        stopped = "all stages done"
    for row in history:
        row.pop("_stats")
    (out_dir / "initial.json").write_text((out_dir / "look_00.json").read_text())
    (out_dir / "best.json").write_text(json.dumps(best_look, indent=1) + "\n")
    summary = {"plan": Path(args.plan).name, "seed": plan["seed"], "budget": plan["budget"],
               "evaluations": len(history), "stopped_by": stopped, "initial_loss": history[0]["loss"],
               "best_loss": best["loss"], "best_evaluation": best["evaluation"],
               "evaluations_that_improved_best": sum(1 for i, r in enumerate(history) if i and
                                                     r["loss"] < min(h["loss"] for h in history[:i])),
               "seconds": round(sum(r["seconds"] for r in history), 1), "history": history}
    (out_dir / "history.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out_dir / "history.partial.json").unlink()
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}))


if __name__ == "__main__":
    main()
