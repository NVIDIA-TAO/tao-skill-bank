# Mission Control — run visualization

Read this after the DEFT loop completes to launch the interactive visualization
of a finished run. Not a pipeline stage — does not write to `deft_state.json` or
`loop_log.jsonl`.

Interactive visual analytics for a completed DEFT AOI run: every component crop
laid out on one embedding map, scrubbable across iterations, with FAR/KPI metrics,
defect-margin tables, and mining lineage. A browser app (FastAPI + canvas), served
locally; no GPU needed to view.

An optional RCA chat agent ships alongside the map (see **RCA chat agent** below).
It needs its own model endpoint and API key; the visualization does not use it and
works fully without it.

## Where things live

```
skills/applications/tao-run-deft-aoi/
├── assets/mission-control/          # frontend (index.html, app.js, style.css)
└── scripts/mission_control/
    ├── prepare.py                   # build step — generate artifacts for a run
    ├── server.py                    # serve the map for a run
    ├── indexer/                     # reads the run + builds the point model
    ├── rca/                         # optional RCA chat agent (see below)
    └── requirements.txt
```

Generated artifacts live **with the run**, in `${RESULTS_DIR}/mission_control/`
(embeddings, projection, and the serve-ready points). They are regenerated in
place per run — nothing is written into the skill dir.

## Setup

Mission Control has its own venv, separate from the loop's `~/.venvs/deft`. It is
`.gitignore`d, so a fresh clone or plugin install does not ship it. The loop-end
sequence creates it if absent; create it by hand only when running Mission
Control outside a loop:

```bash
scripts/mission_control/bootstrap_venv.sh
```

`bootstrap_venv.sh` handles the fallback chain (`python -m venv` → system
`virtualenv` → `venv --without-pip` with pip seeded from the base interpreter),
so `python3-venv` does not have to be installed via apt. It installs nothing
into the host environment, and re-running it is a no-op once the venv works.
Pass `--sync` to reinstall `requirements.txt` into an existing venv.

**Air-gap.** `bootstrap_venv.sh` never runs a package manager when `AIR_GAPPED=1`; pre-stage the venv on a networked host and copy the tree. The build then reads `SIGLIP_MODEL_PATH` (resolved by Pre-Flight step 8) instead of the HuggingFace id, mounts that snapshot, and passes `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`. Both are hard stops for the build only — Mission Control stays optional and never fatal.

Dependencies: `fastapi`, `uvicorn[standard]`, `starlette`, `pandas`, `pyarrow`,
`scikit-learn`, `numpy`, `pillow`, `pyyaml`, `openai>=2.44,<3`.

The build step (`prepare.py`) additionally needs:

| Requirement | Check | If missing |
|---|---|---|
| A data_services image | `[ -n "$TAO_DS_IMAGE" ] \|\| [ -n "$TAO_SKILL_BANK_PATH" ]` | `prepare.py` exits non-zero without embedding. It reads `$TAO_DS_IMAGE` first and falls back to `images.tao_toolkit.data_services` in `$TAO_SKILL_BANK_PATH/versions.yaml` |
| That image pulled | `docker image inspect "$TAO_DS_IMAGE"` | `docker pull "$TAO_DS_IMAGE"` first. Same data_services container the `data_mining` stage uses |
| `HF_TOKEN` | `[ -n "$HF_TOKEN" ]` | passed in via `-e HF_TOKEN` so SigLIP (`google/siglip-base-patch16-224`) can download; skipped if already cached in the image |
| A GPU | — | SigLIP embedding requires one |

`TAO_DS_IMAGE` and `HF_TOKEN` are exported by the loop's Pre-Flight (steps 5 and
3), so they are already in the environment if Mission Control runs in the same
session as the DEFT loop. `TAO_SKILL_BANK_PATH` comes from the plugin's session
hook instead, and is absent from a plain clone — which is why the image var is
read first. Serving needs neither; only the build step does.

## Two-step flow

Step 1 is **required, not optional**: a DEFT run never embeds its own train/val
splits, so `server.py` aborts with `RuntimeError: No embeddings at …` until
`prepare.py` has produced `${RESULTS_DIR}/mission_control/embeddings.parquet`.

1. **Build** — SigLIP-embed the run's crops + assemble the serve-ready model
   (one container pass; skipped if already present):
   
   ```bash
   scripts/mission_control/.venv/bin/python scripts/mission_control/prepare.py \
       --run ${RESULTS_DIR}
   ```
   `prepare.py` handles all container mounts automatically — it computes the
   filesystem roots of all image paths and mounts each one. Do not add mounts
   manually.

   Everything lands in `${RESULTS_DIR}/mission_control/`:

   | File | Written by | Notes |
   |---|---|---|
   | `embeddings.parquet` | the container | The build's only real output. `--force` regenerates it. |
   | `embed_input.parquet`, `embedding_spec.yaml`, `experiment_specs/` | `prepare.py` | Container inputs, written before launch. |
   | `tsne.npy`, `serve_points.parquet`, `serve_meta.json` | the indexer, on first load | Derived cache, not build output. Rebuilt automatically when the run gains an iteration or is re-embedded — safe to delete. |

   Only `embeddings.parquet` is required to serve; the rest is recomputed on
   demand.

   **Projection.** `--projection {tsne,umap}` picks the 2-D layout (default
   `tsne`). t-SNE keeps local neighbourhoods; UMAP additionally keeps
   inter-cluster distance meaningful. Both run on cosine geometry, matching the
   similarity the neighbour and mining panels report. The choice is recorded in
   `serve_meta.json`, so `server.py` reproduces it without a flag, and each
   method caches to its own `coords_<method>.npy` — switching recomputes rather
   than silently reusing the other layout. UMAP needs `umap-learn` (which pulls
   `numba`); it is imported only when selected. `--force` re-embeds even when artifacts are already cached (e.g.
   after adding new iterations to the run).

2. **Serve** — launch the map:
   ```bash
   scripts/mission_control/.venv/bin/python scripts/mission_control/server.py \
       --run ${RESULTS_DIR} --port 8090
   ```
   Open `http://localhost:8090/` (forward the port on a headless box).

The server loads from the serve cache when present, so startup skips t-SNE and
holds no embedding matrix — the map, timeline, and metrics serve from the
precomputed points.

## RCA chat agent (optional)

A chat panel in the right-hand sidebar that answers questions about the loaded
run — "worst false positive", "most failing defect class", "is this a data gap or
a hard case". It is a narrator over seven deterministic tools in
`scripts/mission_control/rca/tools.py` (run overview, data census, failure
ranking, per-defect breakdown, metadata slicing, coverage census, image viewing),
so every number it states comes from the run's own artifacts rather than the
model.

**It requires its own model endpoint and key.** Configure them in
`scripts/mission_control/rca/agent_config.yaml`:

```yaml
provider: openai_compatible
base_url: https://integrate.api.nvidia.com/v1
model: <model-id>
api_key_env: BUILD_KEY        # the env var holding the key
```

Any OpenAI-compatible chat-completions endpoint works; swapping models is a
config edit. With the key unset the panel reports that and the rest of Mission
Control is unaffected — the map, timeline, metrics and mining lineage never call
it.
