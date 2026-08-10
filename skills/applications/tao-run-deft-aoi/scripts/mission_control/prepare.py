#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-command onboarding of a DEFT run into Mission Control.

    scripts/mission_control/.venv/bin/python scripts/mission_control/prepare.py \
        --run <RESULTS_DIR>

REQUIRED before serving: a DEFT run does not embed its own train/val splits,
so server.py cannot start until this has produced embeddings.parquet.

1. Embeds the run's images with SigLIP (the DEFT loop itself never embeds the
   train/val splits) into <RESULTS_DIR>/mission_control/embeddings.parquet, via
   the tao_toolkit.data_services container. Skipped if already cached.
2. Warms the t-SNE projection cache so the first browser load is instant.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from indexer import collect, images, layout  # noqa: E402
from indexer.run_index import RunIndex  # noqa: E402


def _redact(tok: str) -> str:
    """Mask ``-e NAME=secret`` values (TOKEN/KEY/SECRET/PASSWORD) for logging."""
    if "=" in tok:
        name, _, _ = tok.partition("=")
        if any(s in name.upper() for s in ("TOKEN", "KEY", "SECRET", "PASSWORD")):
            return f"{name}=***"
    return tok


def sh(cmd, **kw):
    print("+", " ".join(_redact(str(c)) for c in cmd))
    return subprocess.run(list(map(str, cmd)), check=True, **kw)


def resolve_ds_image():
    """The data_services image, preferring the var the rest of the loop uses.

    Pre-Flight step 5 exports ``TAO_DS_IMAGE`` and hard-stops without it, so
    inside a loop run it is the pinned answer. ``TAO_SKILL_BANK_PATH`` is set by
    the plugin's session hook and is absent from a plain clone, which is why it
    is the fallback rather than the source of truth.
    """
    img = os.environ.get("TAO_DS_IMAGE")
    if img:
        return img
    sb = os.environ.get("TAO_SKILL_BANK_PATH")
    if not sb:
        return None
    return yaml.safe_load(open(f"{sb}/versions.yaml"))["images"]["tao_toolkit"]["data_services"]


def air_gapped() -> bool:
    """Global air-gap mode, per ``references/air-gap.md``.

    Declared, never inferred — the contract forbids probing the network to
    decide, since a failed pull is as likely to be a bad credential.
    """
    return os.environ.get("AIR_GAPPED", "") == "1"


def mount_roots(filepaths, cache: Path) -> list[str]:
    """Bind-mount arguments covering every image plus the cache.

    Mounting each top-level root once keeps the argument list flat however many
    directories the CSVs span, and mounting at the same path inside the
    container keeps the host paths in the parquet valid there.
    """
    roots = sorted({f"/{Path(p).parts[1]}" for p in filepaths}
                   | {f"/{Path(cache).resolve().parts[1]}"})
    return sum((["-v", f"{r}:{r}"] for r in roots), [])


def embed_all(rd: Path, ws: Path, cache: Path, force: bool = False) -> str | None:
    """Embed the run's images into ``cache/embeddings.parquet``.

    Returns None once the file is on disk, or a message explaining why it could
    not be produced. Every failure here is fatal to serving, so the caller must
    not treat a return value as advisory.
    """
    out = cache / "embeddings.parquet"
    if out.is_file() and not force:
        print(f"embeddings: already cached ({out}, {len(pd.read_parquet(out, columns=['filepath']))} images)")
        return None
    spec_path = rd / "baseline_spec.yaml"
    spec = yaml.safe_load(open(spec_path)) if spec_path.is_file() else None
    lights, ext = images.read_lights_ext(spec)
    light = lights[0]
    if len(lights) > 1:
        # One sample per component regardless — the captures are stacked
        # channels, so channel 0 is what the map is keyed and embedded on.
        print(f"lighting: {len(lights)} conditions {lights} — embedding channel 0 ({light})")

    state_f = rd / "deft_state.json"
    state = json.loads(state_f.read_text()) if state_f.is_file() else {}
    paths = layout.resolve(rd, state)
    used = [k for k, v in paths["sources"].items() if v == "config"]
    print(f"inputs: {len(used)}/4 from deft_state.config, rest from the workspace layout")
    work = collect.collect(rd, ws, light, ext, paths=paths)
    if work.empty:
        return (f"no images collected from {paths['images_dir']} — the run's CSVs "
                f"name no file that exists on disk")
    print(f"embeddings: {len(work)} unique images "
          f"({work['kind'].value_counts().to_dict()}) → embedding with SigLIP")
    ds = resolve_ds_image()
    if not ds:
        return ("no data_services image — export TAO_DS_IMAGE (Pre-Flight step 5) "
                "or TAO_SKILL_BANK_PATH, then re-run")
    # Pre-Flight resolves SIGLIP_MODEL_PATH to a local snapshot (air-gap) or
    # leaves the HuggingFace id (networked). Mining reads the same var; never
    # let this step pick an online default the rest of the loop rejected.
    siglip = os.environ.get("SIGLIP_MODEL_PATH") or "google/siglip-base-patch16-224"
    if air_gapped() and not Path(siglip).is_dir():
        return ("air-gap mode needs a staged SigLIP snapshot — export "
                "SIGLIP_MODEL_PATH to a local google/siglip-base-patch16-224 "
                "directory containing config.json (Pre-Flight step 8)")

    in_pq = cache / "embed_input.parquet"
    work.to_parquet(in_pq, index=False)
    (cache / "experiment_specs").mkdir(exist_ok=True)
    (cache / "embedding_spec.yaml").write_text(
        f"model: SigLIP\nmodel_path: {siglip}\nbatch_size: 64\n"
    )
    paths_to_mount = list(work["filepath"])
    if Path(siglip).is_dir():
        paths_to_mount.append(str(Path(siglip).resolve()))
    mounts = mount_roots(paths_to_mount, cache)
    offline = ["-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1"] if air_gapped() else []
    sh(["docker", "run", "--gpus", "all", "--rm", "--ipc=host",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"USER={os.environ.get('USER', 'app')}", "-e", "HOME=/tmp",
        # inherit the value rather than spelling it into argv, which is
        # world-readable in /proc for the life of the container
        "-e", "HF_TOKEN", "-e", "HF_HUB_DISABLE_XET=1", *offline,
        *mounts, "-w", str(cache.resolve()), ds,
        "embedding", "image_embeddings", "-e", str((cache / 'embedding_spec.yaml').resolve()),
        f"input_parquet={in_pq.resolve()}",
        f"output_parquet={out.resolve()}"])
    # the container can exit 0 without writing; only the file proves success
    if not out.is_file():
        return f"the embedding container exited 0 but wrote no {out}"
    print(f"embeddings: wrote {out} ({len(work)} images)")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--force", action="store_true", help="re-embed even if cached")
    ap.add_argument("--projection", choices=("tsne", "umap"), default="tsne",
                    help="2-D layout; recorded with the artifacts so server.py "
                         "reproduces it without a flag (default: tsne)")
    args = ap.parse_args()
    rd = Path(args.run).resolve()
    # Validate BEFORE mkdir — otherwise a typo'd --run silently fabricates a
    # whole tree under a nonexistent path and the script still exits 0.
    if not rd.is_dir():
        sys.exit(f"--run is not a directory: {rd}")
    if not (rd / "deft_state.json").is_file():
        sys.exit(
            f"{rd} is not a DEFT results dir (no deft_state.json).\n"
            "Point --run at a results/run_<TIMESTAMP>/ directory."
        )
    ws = rd.parent.parent
    if not (ws / "kpi").is_dir():
        print(f"WARNING: derived workspace {ws} has no kpi/ — expected the run at "
              f"<workspace>/results/run_<TS>/. Image paths and KPI scores will be "
              f"incomplete if the run was moved.")
    cache = rd / "mission_control"   # generate artifacts inside the run dir
    cache.mkdir(parents=True, exist_ok=True)

    problem = embed_all(rd, ws, cache, force=args.force)
    if problem:
        sys.exit(f"ERROR: {problem}.\n"
                 f"server.py cannot start without {cache / 'embeddings.parquet'}.")

    print("warming projection cache…")
    try:
        idx = RunIndex(str(rd), projection=args.projection)
        s = idx.summary()
    except Exception as e:  # noqa: BLE001 — surfaced here rather than at serve time
        sys.exit(f"ERROR: embeddings were written but the run would not index "
                 f"({type(e).__name__}: {e}).\nserver.py would fail the same way.")
    print(f"OK: {s['run_id']} — {s['counts']['pool']} pool + {s['counts']['kpi']} kpi points, "
          f"best FAR {s['best']['far_pct']} ({s['best']['label']})")
    # absolute: the user runs this from wherever the loop left them
    print(f"\nServe the map:  {HERE / '.venv/bin/python'} {HERE / 'server.py'} "
          f"--run {rd} --port 8090")


if __name__ == "__main__":
    main()
