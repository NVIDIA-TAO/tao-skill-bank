# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from .tools import TOOL_SCHEMAS, RcaChatTools

HERE = Path(__file__).parent

SYSTEM_PROMPT = """\
You are the Root Cause Analysis agent inside DEFT Mission Control for a NVIDIA
TAO 7.0.1 Visual ChangeNet AOI (PCB inspection) run — a model trained
iteratively to catch component defects while flagging as few good boards as
possible. You are a thin narrator over deterministic tools: EVERY number you
state must come from a tool call. Never invent a count, score, or rate.

Tools and when to use them:
- run_overview   — overall status, FAR trajectory, why an iteration regressed.
- data_breakdown — ANY 'how much data / summary breakdown / what do we have'.
- list_failures  — 'worst false positive', list/rank false alarms or missed defects.
- defect_breakdown — 'most failing defect class', per-defect risk/margins.
- failure_by     — 'which component types / boards / part types drive failures'.
- coverage_census — 'how to improve' for a specific failure or defect class:
                    is it a DATA GAP or a HARD CASE?
- view_images    — LOOK at a crop (by point id) before ANY visual claim.

Key facts about this run to reason correctly:
- Two failure modes: FALSE ALARM (a good/PASS board scored as a defect — drives
  the FAR cost) and MISSED DEFECT (a defect scored as PASS). The loop optimized
  to recall=100%, so at the deployed threshold missed defects are ~0 but the
  threshold is pinned razor-thin by the weakest defect.
- Augmentation coverage is asymmetric: AnomalyGen can generate only MISSING,
  EXCESS_SOLDER, and BRIDGE. The other KPI defect types (Shift, Tombstone,
  Upside Down, Lifted_Lead, POLARITY) have NO synthetic path. The mining pool
  is PASS-only, so mining adds good-board variety (helps false alarms), never
  new defect coverage.

How coverage_census decides the fix (use the run's own numbers, not intuition):
- in_training high, still failing        -> TUNE (threshold/HP or hard case; not data)
- unused_pool > 0                         -> MINE (similar real images sit unused)
- ~0 similar anywhere, AnomalyGen-covered -> GENERATE_SYNTHETIC (missing/excess_solder/bridge)
- ~0 similar anywhere, NOT covered        -> COLLECT_REAL (get real labeled examples)

Style:
- Findings first, then the evidence numbers, then a short recommendation.
- Cite tool numbers verbatim (scores, margins, rates, in_training, unused_pool).
- Reference specific failures by their point_id so the UI can highlight them.
- No image quantities — recommend a direction (tune / mine / generate / collect),
  not a count.
- For failure_by, prefer column 'comp_type_2' — the human-readable component
  category (C=capacitor, D=diode, L=inductor, Q=transistor, R=resistor, U=IC,
  J=connector). ALWAYS report each group's sample size n alongside its rate: a
  100% rate on n=5 matters far less than 74% on n=372, so lead with the
  high-volume, high-rate groups, not tiny 100% outliers.

When asked 'how to improve' or for a diagnosis, end with a short fenced block:
```verdict
{"name": "<TUNE_TRAINING|MINE_FROM_POOL|GENERATE_SYNTHETIC|COLLECT_REAL|UNCLEAR>",
 "confidence": "<low|medium|high>", "summary": "<one sentence>",
 "findings": ["<evidence sentence citing tool numbers>", "..."]}
```
"""


def load_config():
    f = HERE / "agent_config.yaml"  # co-located with the agent in rca/
    cfg = yaml.safe_load(f.read_text()) if f.is_file() else {}
    return {
        "base_url": cfg.get("base_url", os.environ.get("AGENT_BASE_URL", "https://integrate.api.nvidia.com/v1")),
        "model": cfg.get("model", os.environ.get("AGENT_MODEL", "")),
        "api_key_env": cfg.get("api_key_env", "NGC_API_KEY"),
        "max_tool_rounds": int(cfg.get("max_tool_rounds", 12)),
        "temperature": float(cfg.get("temperature", 0.6)),
        "top_p": cfg.get("top_p"),
        "max_tokens": cfg.get("max_tokens"),
        "extra_body": cfg.get("extra_body") or {},
    }


class RcaChatAgent:
    def __init__(self, index):
        self.cfg = load_config()
        self.tools = RcaChatTools(index)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _client(self):
        from openai import OpenAI
        key = os.environ.get(self.cfg["api_key_env"], "")
        if not key:
            raise RuntimeError(f"${self.cfg['api_key_env']} is not set — configure agent_config.yaml + export the key.")
        return OpenAI(base_url=self.cfg["base_url"], api_key=key)

    def _openai_tools(self):
        return [{"type": "function", "function": s} for s in TOOL_SCHEMAS]

    def run(self, user_msg: str):
        checkpoint = len(self.messages)
        try:
            self.messages.append({"role": "user", "content": user_msg})
            client = self._client()
            kwargs = {}
            for k in ("top_p", "max_tokens", "extra_body"):
                if self.cfg.get(k):
                    kwargs[k] = self.cfg[k]
            for _ in range(self.cfg["max_tool_rounds"]):
                resp = client.chat.completions.create(
                    model=self.cfg["model"], messages=self.messages,
                    tools=self._openai_tools(), temperature=self.cfg["temperature"], **kwargs)
                msg = resp.choices[0].message
                if getattr(msg, "reasoning_content", None):
                    yield {"type": "thinking", "text": msg.reasoning_content}
                if msg.content:
                    yield {"type": "text", "text": msg.content}
                if not msg.tool_calls:
                    self.messages.append({"role": "assistant", "content": msg.content or ""})
                    break
                self.messages.append({"role": "assistant", "content": msg.content or "",
                                      "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
                image_followup = []
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    yield {"type": "tool_call", "name": name, "args": args}
                    result, refs, images = self._exec(name, args)
                    yield {"type": "tool_result", "name": name,
                           "summary": _summarize(name, result), "refs": refs}
                    self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                          "content": json.dumps(result, default=str)})
                    image_followup.extend(images)
                if image_followup:
                    self.messages.append({"role": "user", "content":
                        [{"type": "text", "text": "[attached: images from view_images]"}] + image_followup})
        except Exception as e:
            del self.messages[checkpoint:]
            yield {"type": "error", "text": str(e)}
        yield {"type": "done"}

    def _exec(self, name, args):
        fn = getattr(self.tools, name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}, [], []
        try:
            result, refs = fn(**args)
        except Exception as e:
            return {"error": str(e)}, [], []
        if name == "view_images" and "images" in result:
            parts = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{im['b64']}"}}
                     for im in result["images"]]
            return {"viewed": [im["ref"] for im in result["images"]]}, refs, parts
        return result, refs, []


def _summarize(name, result):
    if isinstance(result, dict) and "error" in result:
        return f"error: {result['error']}"
    if name == "run_overview":
        return f"best {result.get('best_iteration')} FAR {result.get('best_far_pct')}% · {result.get('false_alarms')} FA / {result.get('missed_defects')} miss"
    if name == "data_breakdown":
        t = result.get("totals", {})
        return f"{t.get('all_points')} pts ({t.get('kpi')} KPI / {t.get('pool')} pool)"
    if name == "list_failures":
        return f"{result.get('shown')}/{result.get('total_of_kind')} {result.get('kind')}"
    if name == "defect_breakdown":
        return f"{len(result.get('defects', []))} defect types"
    if name == "failure_by":
        g = result.get("groups", [])
        top = g[0] if g else {}
        return f"{result.get('column')}: worst {top.get(result.get('column'))} @ {top.get('rate_pct')}%"
    if name == "coverage_census":
        return f"in_train={result.get('in_training')} unused_pool={result.get('unused_pool')} -> {result.get('route_hint')}"
    if name == "view_images":
        return f"viewed {len(result.get('viewed', []))}"
    return "ok"
