#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find videos a finished video_reasoning_annotation run dropped, and plan a retry.

RESULTS_SUMMARY.md counts attrition per stage but never says which video was
lost or why. This helper compares the per-stage JSONL outputs in a results
directory, names each dropped video and the stage that lost it, and - when the
pipeline's own stdout log is supplied - attributes the cause (truncated model
output, API error, ...) from that stage's WARNING line.

Retrying is safe because the pipeline resumes: curation and steps 1a-2a skip
videos they already processed, step 3 skips (video, prompt_key) pairs already
present when workflow.qa_resume is true, and 4a/4b/5 always rebuild. Re-running
the same spec on the SAME results_dir therefore only reprocesses what is
missing. ``--plan-retry`` prints the extra ``prepare_vra_spec.py --set``
overrides for the next round (qa_resume plus a doubled output-token limit for
the model whose output was truncated) and records the round in
``<results_dir>/vra_retry_state.json``. At most three retry rounds are allowed
per results directory; after that the remaining videos are reported as
unrecoverable rather than retried again.

Reads only local files; never launches a container or calls a model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - only --spec needs it
    yaml = None

MAX_RETRY_ROUNDS = 3
TOKEN_CEILING = 65536
STATE_FILE = "vra_retry_state.json"

#: Stage order, with the JSONL each stage writes (relative to results_dir).
STAGES = [
    ("0a", "step_0a_filter/filter_results.jsonl"),
    ("0b", "step_0b_classify/classification.jsonl"),
    ("0c", "step_0c_routing"),  # directory of lane files
    ("1a", "step_1a_caption/captions.jsonl"),
    ("1b", "step_1b_chunks/chunk_captions.jsonl"),
    ("1c", "step_1c_highlight/highlight_captions.jsonl"),
    ("2a", "step_2_description/descriptions.jsonl"),
    ("3", "step_3_qa/qa_output.jsonl"),
]

#: Pipeline source file -> stage, for attributing WARNING lines in the log.
LOG_STAGE = {
    "step0a_filter.py": "0a", "step0b_classify.py": "0b", "step0c_routing.py": "0c",
    "step0d_dedup.py": "0d", "step0e_enhance.py": "0e", "step0f_snapshot.py": "0f",
    "step1a_caption.py": "1a", "step1a_merge.py": "1a", "step1b_chunks.py": "1b",
    "step1c_highlight.py": "1c", "step2a_description.py": "2a", "step3_qa.py": "3",
}

#: Stage -> the `pipeline:` routing key whose model produces that stage's output.
STAGE_ROUTE = {
    "0a": "video_filtering", "0b": "anomaly_normal_classification",
    "1a": "dense_caption", "1b": "chunk_caption", "1c": "highlight_caption",
    "2a": "description", "3": "mcq",
}

WARNING_RE = re.compile(r"WARNING - (.*?) \((\w+\.py):\d+\)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def stem(path):
    """Identity of a video across stages: file stem without the 0e `_enh` suffix."""
    name = Path(str(path)).name
    base = name.rsplit(".", 1)[0] if "." in name else name
    return base[:-4] if base.endswith("_enh") else base


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_video(row):
    return row.get("video_path") or row.get("video") or ""


def collect_stages(results_dir):
    """Return {stage: set(stems)} plus 0a rejections and QA rows per video."""
    present, rejected, qa_keys = {}, set(), {}
    for stage, rel in STAGES:
        path = results_dir / rel
        if stage == "0c":
            rows = [r for lane in sorted(path.glob("route_*.jsonl")) for r in read_jsonl(lane)]
        else:
            rows = read_jsonl(path)
        if stage == "0a":
            present[stage] = {stem(row_video(r)) for r in rows if r.get("is_valid")}
            rejected = {stem(row_video(r)) for r in rows if not r.get("is_valid")}
            present["0a_seen"] = {stem(row_video(r)) for r in rows}
        else:
            present[stage] = {stem(row_video(r)) for r in rows}
        if stage == "3":
            for r in rows:
                qa_keys.setdefault(stem(row_video(r)), set()).add(r.get("prompt_key", ""))
    return present, rejected, qa_keys


def parse_logs(log_paths, stems):
    """Map stem -> list of (stage, message) WARNINGs mentioning that video, in log order."""
    causes = {}
    for log in log_paths:
        for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = WARNING_RE.search(ANSI_RE.sub("", raw))
            if not m or m.group(2) not in LOG_STAGE:
                continue
            message, stage = m.group(1), LOG_STAGE[m.group(2)]
            names = {stem(tok) for tok in re.findall(r"[^\s\[\]:,()]+\.(?:mp4|webm|mkv|mov|avi)", message)}
            names |= {tok for tok in re.findall(r"[^\s\[\]:,()]+", message)}
            for s in stems:
                # Whole-name match: a plain substring test would attribute a
                # warning for "clip_scene10" to "clip_scene1" as well.
                if s and s in names:
                    entry = (stage, message)
                    if entry not in causes.setdefault(s, []):
                        causes[s].append(entry)
    return causes


def expected_qa_keys(qa_keys):
    """Expected prompt keys per QA lane: the union observed for videos in that lane."""
    by_mode = {}
    for keys in qa_keys.values():
        modes = {k.split("_", 1)[0] for k in keys if k.startswith(("anomaly_", "normal_"))}
        mode = modes.pop() if len(modes) == 1 else None
        if mode:
            by_mode.setdefault(mode, set()).update(keys)
    return by_mode


def find_dropouts(results_dir, inputs=None, log_paths=()):
    present, rejected, qa_keys = collect_stages(results_dir)
    universe = set(present["0a_seen"]) | set(inputs or ())
    for stage, _ in STAGES:
        universe |= present[stage]
    causes = parse_logs(log_paths, universe)

    report = []
    for s in sorted(universe):
        if s in rejected:
            report.append({"video": s, "dropped_at": "0a", "kind": "rejected",
                           "cause": "0a filter answered No (out-of-domain verdict, not a failure)",
                           "retryable": False, "truncated": False})
            continue
        dropped_at = None
        for stage, _ in STAGES:
            if stage == "0a" and s not in present["0a_seen"]:
                dropped_at = "0a"  # in --inputs but never filtered: a 0a error
                break
            if stage == "1c":
                continue  # only anomaly-lane videos get a highlight; 2a runs without it
            if s not in present[stage]:
                dropped_at = stage
                break
        partial = []
        if dropped_at is None:
            mode_keys = expected_qa_keys(qa_keys)
            mine = qa_keys.get(s, set())
            modes = {k.split("_", 1)[0] for k in mine if k.startswith(("anomaly_", "normal_"))}
            if len(modes) == 1:
                partial = sorted(mode_keys[modes.pop()] - mine)
            if not partial:
                continue
            dropped_at = "3"
        warnings = causes.get(s, [])
        stage_warnings = [m for st, m in warnings if st == dropped_at] or [m for _, m in warnings]
        message = stage_warnings[-1] if stage_warnings else (
            "no WARNING found for this video; pass the run's stdout with --log" if not log_paths
            else "no WARNING found in the supplied log(s)")
        report.append({
            "video": s, "dropped_at": dropped_at,
            "kind": "partial_qa" if partial else "dropped",
            "missing_prompt_keys": partial,
            "cause": message,
            "truncated": "truncated" in message.lower(),
            "retryable": True,
        })
    return report, present


def _model_for_stage(spec, stage):
    """(override_dotted_key, current_value) for the output-token limit of a stage's model."""
    vra = spec.get("video_reasoning_annotation", spec)
    route = (vra.get("pipeline") or {}).get(STAGE_ROUTE.get(stage, ""))
    model = (vra.get("models") or {}).get(route) if route else None
    text_role = "llm" if stage == "3" else "vlm"
    if model is None:
        model, owner = vra.get(text_role) or {}, text_role
    else:
        owner = f"models.{route}"
    backend = model.get("backend", "openai")
    block = model.get(backend) or {}
    field = "max_output_tokens" if backend == "gemini" else "max_tokens"
    return owner, backend, field, int(block.get(field) or 8192)


def override_key(spec_text, spec, stage):
    """Dotted --set key that controls `stage`'s output limit in this spec file.

    A `models.<name>` block either repeats an endpoint through a YAML alias
    (`openai: *vlm_endpoint`), in which case the limit lives on the anchored
    vlm/llm line, or merges it and sets its own `max_tokens` line, which then
    wins. prepare_vra_spec.py edits lines, so the key must name the line.
    """
    owner, backend, field, current = _model_for_stage(spec, stage)
    if owner.startswith("models."):
        name = owner.split(".", 1)[1]
        block = re.search(rf"^    {re.escape(name)}:\n((?:      .*\n|\s*\n)+)", spec_text, re.M)
        if block and re.search(rf"^\s+{field}:", block.group(1), re.M):
            return f"video_reasoning_annotation.models.{name}.{backend}.{field}", current
        alias = re.search(r"\*(\w+)_endpoint", block.group(1)) if block else None
        owner = alias.group(1) if alias else ("llm" if stage == "3" else "vlm")
    return f"video_reasoning_annotation.{owner}.{backend}.{field}", current


def load_state(results_dir):
    path = results_dir / STATE_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"max_rounds": MAX_RETRY_ROUNDS, "rounds": []}


def plan_retry(results_dir, report, spec_path):
    """Print overrides for the next retry round and record it; refuse past round 3."""
    state = load_state(results_dir)
    retryable = [r for r in report if r["retryable"]]
    done = len(state["rounds"])
    if not retryable:
        print("\nNothing to retry: every dropped video is a deliberate 0a rejection.")
        return 0
    if done >= MAX_RETRY_ROUNDS:
        print(f"\nRetry limit reached ({done}/{MAX_RETRY_ROUNDS} rounds). Not retrying again.")
        print("Unrecoverable videos - inspect them by hand or exclude them:")
        for r in retryable:
            print(f"  {r['video']}  (stage {r['dropped_at']}): {r['cause']}")
        return 2
    round_no = done + 1
    overrides = ['video_reasoning_annotation.workflow.qa_resume=true']
    if any(r["truncated"] for r in retryable):
        if spec_path is None or yaml is None:
            print("\n--spec (and PyYAML) are required to plan a token-limit increase.", file=sys.stderr)
            return 1
        spec_text = Path(spec_path).read_text(encoding="utf-8")
        spec = yaml.safe_load(spec_text)
        bumps = {}
        for r in retryable:
            if r["truncated"]:
                key, current = override_key(spec_text, spec, r["dropped_at"])
                bumps[key] = min(current * 2 ** round_no, TOKEN_CEILING)
        overrides += [f"{k}={v}" for k, v in sorted(bumps.items())]
    state["rounds"].append({
        "round": round_no,
        "planned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "videos": [r["video"] for r in retryable],
        "overrides": overrides,
    })
    (results_dir / STATE_FILE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"\nRetry round {round_no}/{MAX_RETRY_ROUNDS} recorded in {results_dir / STATE_FILE}.")
    print("Regenerate the spec with prepare_vra_spec.py using the SAME --set list as the")
    print("original run plus these, then relaunch on the SAME results_dir:")
    for o in overrides:
        print(f"  --set '{o}'")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results_dir", type=Path, help="Host path of the run's results_dir.")
    p.add_argument("--log", type=Path, action="append", default=[],
                   help="Pipeline stdout/stderr log(s) of this run and earlier retries; repeatable.")
    p.add_argument("--inputs", type=Path,
                   help="The run's input JSONL, to also catch videos 0a never recorded.")
    p.add_argument("--spec", type=Path, help="The spec the run used (needed by --plan-retry).")
    p.add_argument("--plan-retry", action="store_true",
                   help="Record the next retry round and print its prepare_vra_spec overrides.")
    p.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = p.parse_args(argv)

    if not (args.results_dir / "step_0a_filter").is_dir():
        p.error(f"{args.results_dir} does not look like a video_reasoning_annotation results_dir")
    inputs = None
    if args.inputs:
        inputs = {stem(row_video(r)) for r in read_jsonl(args.inputs)}
    report, present = find_dropouts(args.results_dir, inputs, args.log)

    if args.json:
        print(json.dumps({"dropouts": report,
                          "stage_counts": {k: len(v) for k, v in present.items() if k != "0a_seen"},
                          "retry_state": load_state(args.results_dir)}, indent=2))
    else:
        counts = "  ".join(f"{k}={len(v)}" for k, v in present.items() if k != "0a_seen")
        print(f"Videos per stage: {counts}")
        if not report:
            print("No dropped videos.")
        for r in report:
            tag = "RETRY" if r["retryable"] else "skip "
            extra = f" missing={','.join(r['missing_prompt_keys'])}" if r.get("missing_prompt_keys") else ""
            print(f"[{tag}] {r['video']}  stage {r['dropped_at']} ({r['kind']}){extra}\n         {r['cause']}")
        state = load_state(args.results_dir)
        print(f"Retry rounds used: {len(state['rounds'])}/{MAX_RETRY_ROUNDS}")
    if args.plan_retry:
        return plan_retry(args.results_dir, report, args.spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
