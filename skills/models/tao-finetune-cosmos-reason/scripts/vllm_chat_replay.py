#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Replay a JSONL of archived multimodal chat requests against a local vLLM OpenAI endpoint.

Each input row carries an ``id`` and a ``messages`` list in the archived (Qwen-style) shape:

    {"role": "system", "content": "..."}
    {"role": "user", "content": [{"type": "image", "image": "<path relative to --media-root>", ...},
                                 {"type": "text", "text": "..."}]}

Images are sent as base64 data URLs in the given order (OpenAI ``image_url`` parts are passed
through unchanged), the text is sent verbatim, sampling is greedy (``temperature`` 0 -- sent
explicitly because Cosmos3's generation_config.json otherwise installs temperature 0.7 / top-k 20
/ top-p 0.8 as the server defaults), and nothing is retried with a different prompt. Per-image
``min_pixels`` / ``max_pixels`` hints in the archive are NOT honoured client-side: pin them on the
server with ``--mm-processor-kwargs`` and confirm via
``usage.prompt_tokens``.

Outputs (under --out):
  responses.jsonl   {"id": ..., "raw_prediction": "<model text, verbatim>"}   one row per input row
  telemetry.jsonl   id, group, n_images, latency_s, prompt_tokens, completion_tokens, finish_reason, error

Ground-truth files are never read here.

Usage:
  python vllm_chat_replay.py --requests inputs/model_requests.jsonl --media-root <package root> \
      --api http://127.0.0.1:8422 --out C:\work\run1 [--limit 8 --show] [--filter condition=A,D,H]
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request


def data_url(path: pathlib.Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def to_openai_messages(messages: list[dict], media_root: pathlib.Path) -> tuple[list[dict], int]:
    out, n_images = [], 0
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        parts = []
        for part in content:
            kind = part.get("type")
            if kind == "image":
                img = media_root / part["image"]
                if not img.is_file():
                    raise FileNotFoundError(img)
                parts.append({"type": "image_url", "image_url": {"url": data_url(img)}})
                n_images += 1
            elif kind == "image_url":
                parts.append(part)
                n_images += 1
            elif kind == "text":
                parts.append({"type": "text", "text": part["text"]})
            else:
                raise ValueError(f"unknown content part type {kind!r}")
        out.append({"role": msg["role"], "content": parts})
    return out, n_images


def call(api: str, model: str, messages: list[dict], max_tokens: int, timeout: int) -> tuple[dict, float]:
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(f"{api}/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    return body, time.perf_counter() - t0


def served_model(api: str) -> str:
    with urllib.request.urlopen(f"{api}/v1/models", timeout=10) as resp:
        return json.load(resp)["data"][0]["id"]


def parse_filter(spec: str) -> tuple[str, set[str]] | None:
    if not spec:
        return None
    field, _, values = spec.partition("=")
    if not field or not values:
        raise SystemExit("--filter expects FIELD=V1,V2,...")
    return field, set(values.split(","))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", type=pathlib.Path, required=True, help="JSONL with id + messages per row")
    ap.add_argument("--media-root", type=pathlib.Path, required=True, help="root that relative image paths resolve against")
    ap.add_argument("--api", default="http://127.0.0.1:8421")
    ap.add_argument("--model", default="", help="served model name; default: first entry of /v1/models")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N (filtered) rows")
    ap.add_argument("--filter", default="", help="FIELD=V1,V2 keep rows whose FIELD is one of the values")
    ap.add_argument("--group-field", default="condition", help="row field used for progress/latency grouping")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--show", action="store_true", help="print each raw model output")
    ap.add_argument("--progress-every", type=int, default=32)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.requests.read_text(encoding="utf-8").splitlines() if l.strip()]
    flt = parse_filter(args.filter)
    if flt:
        field, keep = flt
        rows = [r for r in rows if str(r.get(field)) in keep]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no rows selected")

    args.out.mkdir(parents=True, exist_ok=True)
    model = args.model or served_model(args.api)
    print(f"endpoint {args.api}  model {model}  rows {len(rows)}  max_tokens {args.max_tokens}", flush=True)

    latencies: list[float] = []
    with (args.out / "responses.jsonl").open("w", encoding="utf-8") as fr, \
         (args.out / "telemetry.jsonl").open("w", encoding="utf-8") as ft:
        for i, row in enumerate(rows, 1):
            messages, n_images = to_openai_messages(row["messages"], args.media_root)
            error = None
            try:
                body, latency = call(args.api, model, messages, args.max_tokens, args.timeout)
                text = body["choices"][0]["message"]["content"] or ""
                usage = body.get("usage") or {}
                finish = body["choices"][0].get("finish_reason")
            except urllib.error.HTTPError as exc:
                text, usage, finish, latency = "", {}, "http_error", float("nan")
                error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
            except Exception as exc:  # noqa: BLE001 - record and continue; never retry with a different prompt
                text, usage, finish, latency = "", {}, "error", float("nan")
                error = repr(exc)[:300]
            fr.write(json.dumps({"id": row["id"], "raw_prediction": text.strip()}, ensure_ascii=False) + "\n")
            ft.write(json.dumps({
                "id": row["id"], "group": row.get(args.group_field), "n_images": n_images, "latency_s": latency,
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": finish, "error": error,
            }) + "\n")
            fr.flush(); ft.flush()
            if latency == latency:
                latencies.append(latency)
            if args.show:
                print(f"[{row.get(args.group_field)} {row['id']}] images={n_images} "
                      f"prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')} "
                      f"latency={latency:.2f}s finish={finish}\n    RAW: {text!r}" + (f"\n    ERROR: {error}" if error else ""),
                      flush=True)
            elif i % args.progress_every == 0 or i == len(rows):
                med = statistics.median(latencies) if latencies else float("nan")
                print(f"progress {i}/{len(rows)}  median latency so far {med:.2f}s", flush=True)
    print(f"wrote {args.out / 'responses.jsonl'} and {args.out / 'telemetry.jsonl'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
