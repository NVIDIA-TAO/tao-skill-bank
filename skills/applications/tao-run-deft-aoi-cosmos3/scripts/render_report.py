#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Render a durable Cosmos Framework DEFT AOI run report from state."""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import sys
from typing import Any


def _escape(value: Any) -> str:
    return html.escape("—" if value is None else str(value))


def _rows(state: dict[str, Any]) -> str:
    entries: list[str] = []
    for event in state.get("events", []):
        if not isinstance(event, dict):
            continue
        entries.append(
            "<tr>"
            f"<td>{_escape(event.get('seq'))}</td>"
            f"<td>{_escape(event.get('iter'))}</td>"
            f"<td>{_escape(event.get('stage'))}</td>"
            f"<td>{_escape(event.get('status'))}</td>"
            f"<td>{_escape(event.get('duration_sec'))}</td>"
            f"<td>{_escape(event.get('summary'))}</td>"
            "</tr>"
        )
    return "".join(entries) or '<tr><td colspan="6">No committed stages yet.</td></tr>'


def _iteration_cards(state: dict[str, Any]) -> str:
    cards: list[str] = []
    for label, phase in state.get("iterations", {}).items():
        if not isinstance(phase, dict):
            continue
        metric = phase.get("metric_result")
        metric_text = "not evaluated"
        if isinstance(metric, dict):
            metric_text = (
                f"minimum F1={metric.get('minimum_f1', '—')} · "
                f"gate={'PASS' if metric.get('passed') else 'FAIL'}"
            )
        cards.append(
            '<section class="card">'
            f"<h3>{_escape(label)}</h3>"
            f"<p>Status: {_escape(phase.get('status', 'pending'))}</p>"
            f"<p>Last stage: {_escape(phase.get('stage_completed'))}</p>"
            f"<p>{_escape(metric_text)}</p>"
            f"<p>Mined records: {_escape(phase.get('mining_mined_count'))}</p>"
            f"<p>Framework DCP: {_escape(phase.get('best_ckpt_path'))}</p>"
            "</section>"
        )
    return "".join(cards) or '<section class="card"><p>No iterations committed.</p></section>'


def render(results_dir: pathlib.Path) -> pathlib.Path:
    root = results_dir.expanduser().resolve()
    state_path = root / "deft_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("deft_state.json root must be an object")
    config = state.get("config", {})
    kpi = config.get("kpi", {}) if isinstance(config, dict) else {}
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DEFT AOI Cosmos Framework Report</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f5f7f8;color:#1b1f23}}
main{{max-width:1180px;margin:auto;padding:32px}} h1{{margin-bottom:4px}} .lede{{color:#53606b}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;margin:22px 0}}
.card{{background:white;border:1px solid #dfe3e6;border-radius:10px;padding:16px}}
table{{width:100%;border-collapse:collapse;background:white}} th,td{{padding:10px;border-bottom:1px solid #e6eaed;text-align:left;vertical-align:top}} th{{background:#eef3f5}}
code{{overflow-wrap:anywhere}} .status{{color:#3b7d23;font-weight:700}}
</style></head><body><main>
<h1>DEFT AOI · Cosmos Framework</h1>
<p class="lede">Real-mining-only iterative adaptation with canonical NVPAW JSONL and Framework DCP checkpoints.</p>
<div class="grid">
<section class="card"><h3>Run</h3><p class="status">{_escape(state.get('status'))}</p><p>{_escape(root)}</p></section>
<section class="card"><h3>KPI authority</h3><p>{_escape(kpi.get('profile'))}</p><p>Component F1 threshold: {_escape(kpi.get('component_threshold'))}</p><code>{_escape(kpi.get('evaluator'))}</code></section>
<section class="card"><h3>Runtime</h3><p>Backend: {_escape(config.get('training', {}).get('backend') if isinstance(config, dict) else None)}</p><p>Checkpoint: Framework DCP</p><p>Training source: mined real samples</p></section>
</div>
<h2>Iterations</h2><div class="grid">{_iteration_cards(state)}</div>
<h2>Committed stages</h2><table><thead><tr><th>#</th><th>Iteration</th><th>Stage</th><th>Status</th><th>Seconds</th><th>Summary</th></tr></thead><tbody>{_rows(state)}</tbody></table>
</main></body></html>"""
    output = root / "DEFT_Loop_Report.html"
    output.write_text(body, encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        output = render(args.results_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"render_report: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
