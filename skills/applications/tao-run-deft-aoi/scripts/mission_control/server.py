# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEFT Mission Control — FastAPI backend (visualization server).

Run:  scripts/mission_control/.venv/bin/python scripts/mission_control/server.py \
          --run <results_dir> [--port 8090]
      (or set $DEFT_MC_RUN instead of --run)
Then open http://localhost:8090/

Artifacts (embeddings, projection, serve points) are read from
<results_dir>/mission_control/ — produced by scripts/mission_control/prepare.py.
The frontend is served from <skill>/assets/mission-control/.
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).parent))
from indexer.run_index import RunIndex  # noqa: E402
from rca.agent import RcaChatAgent, load_config  # noqa: E402

HERE = Path(__file__).parent
# The frontend is shipped under the skill's assets/, two levels up from this
# server (scripts/mission_control/server.py -> <skill>/assets/mission-control/).
ASSETS = Path(__file__).resolve().parents[2] / "assets" / "mission-control"

app = FastAPI(title="DEFT Mission Control")
INDEX: RunIndex | None = None


@app.middleware("http")
async def _no_store(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


AGENTS: dict[str, RcaChatAgent] = {} 


def _load(run_dir: str):
    global INDEX
    INDEX = RunIndex(run_dir)  # artifacts default to <run_dir>/mission_control/
    AGENTS.clear()  # conversations are per-run


@app.get("/api/agent/config")
def agent_config():
    cfg = load_config()
    return {"provider": "openai_compatible", "model": cfg["model"],
            "base_url": cfg["base_url"],
            "key_set": bool(os.environ.get(cfg["api_key_env"]))}


@app.post("/api/agent/chat")
async def agent_chat(payload: dict):
    session = payload.get("session", "default")
    msg = (payload.get("message") or "").strip()
    if not msg:
        raise HTTPException(400, "empty message")
    agent = AGENTS.setdefault(session, RcaChatAgent(INDEX))

    def stream():
        try:
            for ev in agent.run(msg):
                # tools return map point ids directly as evidence refs
                if ev.get("type") == "tool_result":
                    ev["point_ids"] = [int(r) for r in (ev.get("refs") or [])
                                       if isinstance(r, (int, float))]
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/agent/reset")
def agent_reset(payload: dict):
    AGENTS.pop(payload.get("session", "default"), None)
    return {"ok": True}


@app.get("/api/runs")
def runs():
    """Sibling run_* dirs of the loaded run (same workspace)."""
    results_root = INDEX.rd.parent
    out = []
    for d in sorted(results_root.glob("run_*")):
        out.append(
            {"name": d.name, "loaded": d == INDEX.rd,
             "standard": (d / "deft_state.json").is_file()}
        )
    return out


@app.post("/api/load/{run_name}")
def load_run(run_name: str):
    cand = INDEX.rd.parent / run_name
    if not (cand / "deft_state.json").is_file():
        raise HTTPException(400, f"{run_name} is not a standard DEFT run (no deft_state.json)")
    _load(str(cand))
    return {"loaded": run_name}


@app.post("/api/reload")
def reload_run():
    """Re-read the run from disk (picks up new iterations of a live run)."""
    _load(str(INDEX.rd))
    return {"reloaded": INDEX.rd.name}


@app.get("/api/summary")
def summary():
    return INDEX.summary()


@app.get("/api/points")
def points():
    return INDEX.points_payload()


@app.get("/api/defect_margins")
def defect_margins():
    return INDEX.defect_margin_table()


@app.get("/api/mining_edges/{iteration}")
def mining_edges(iteration: str):
    return INDEX.mining_edges(iteration)


@app.get("/api/weak_targets/{iteration}")
def weak_targets(iteration: str):
    return INDEX.weak_targets(iteration)


@app.get("/api/neighbors/{idx}")
def neighbors(idx: int, iteration: str, k: int = 0, same_label: str = ""):
    try:
        return INDEX.neighbors(idx, iteration, k, same_label)
    except IndexError:
        raise HTTPException(404, f"no point {idx} in {INDEX.rd.name}")


@app.get("/api/image/{idx}")
def image(idx: int, thumb: int = 0, light: str = ""):
    try:
        path = INDEX.image_path(idx, light or None)
    except IndexError:
        raise HTTPException(404)
    if light and not Path(path).is_file():
        raise HTTPException(404, f"no {light} capture for point {idx}")
    if thumb:
        im = Image.open(path).convert("RGB")
        im.thumbnail((thumb, thumb))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return Response(buf.getvalue(), media_type="image/jpeg")
    return FileResponse(path)


@app.get("/")
@app.get("/index.html")
def _index():
    return FileResponse(ASSETS / "index.html", media_type="text/html",
                        headers={"Cache-Control": "no-store"})


@app.get("/app.js")
def _app_js():
    return FileResponse(ASSETS / "app.js", media_type="application/javascript",
                        headers={"Cache-Control": "no-store"})


@app.get("/style.css")
def _style_css():
    return FileResponse(ASSETS / "style.css", media_type="text/css",
                        headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=ASSETS, html=True), name="static")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.environ.get("DEFT_MC_RUN"),
                    help="path to a DEFT results dir (results/run_<TS>); "
                         "defaults to $DEFT_MC_RUN")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    if not args.run:
        ap.error("--run is required (path to a DEFT results dir), or set $DEFT_MC_RUN")
    _load(args.run)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)
