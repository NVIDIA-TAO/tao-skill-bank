#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a video_reasoning_annotation run: did it finish, and what went wrong.

PIPELINE_STATUS.md has one row per video, which stops being readable past a
few dozen videos, and step 5 writes it only when the run reaches step 5. This
helper gives a run-level answer that scales to any number of videos:

- a verdict: COMPLETED, COMPLETED_WITH_DROPS, INCOMPLETE (step 5 never wrote
  its report - the run crashed or is still running) or CRASHED (the supplied
  log ends in a traceback);
- one row per stage with status counts (ok / skipped / n/a / filtered_out /
  error / missing), instead of one row per video;
- only the problem videos, capped at ``--max-list``, each with the stage it
  dropped at and - when the run's stdout log is supplied - the WARNING that
  names it.

It reads ``live_report/pipeline_status.json`` when step 5 wrote one that is
current, and otherwise rebuilds the picture from
``live_report/pipeline_status_live.jsonl``, which every step appends to as it
goes, so a crashed run still gets a summary.

Reads only local files; never launches a container or calls a model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_vra_dropouts import parse_logs, stem  # noqa: E402  pylint: disable=wrong-import-position

#: Stage order of the live journal and the step 5 status report.
STAGE_ORDER = [
    "0a_filter", "0b_classify", "0c_routing", "0d_dedup", "0e_enhance", "0f_snapshot",
    "1a_caption", "1a_merge", "1b_chunks", "1c_highlight", "2_description", "3_qa",
    "4a_parse_qa", "4b_parse_contextual",
]

#: Status columns, in display order. A status value is classified by its first
#: word, so "ok (12/12 parsed)" counts as ok and "skipped (not low-quality)" as
#: skipped.
STATUS_COLUMNS = ["ok", "skipped", "n/a", "filtered_out", "error", "missing"]

#: Statuses that lose a video, as in the pipeline's own status report.
DROP_STATUSES = ("missing", "filtered_out", "error")

#: When one stage logs a video more than once, the best outcome wins - a
#: retry that succeeded is not a failure.
_RANK = {"ok": 0, "skipped": 1, "n/a": 1, "filtered_out": 2, "error": 3}

TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\):")
EXCEPTION_RE = re.compile(r"^(?:[\w.]+)(?:Error|Exception|Interrupt|Exit)\b.*")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def status_kind(value):
    """Return the status column a stage value belongs to."""
    word = str(value or "").split(" ", 1)[0]
    return word if word in STATUS_COLUMNS else "other"


def load_status_report(results_dir):
    """Return step 5's pipeline_status.json, or None when absent or stale.

    Stale means the live journal was appended to after the report was
    written - e.g. a later run into the same results_dir that crashed before
    step 5 - so the report no longer describes the latest run.
    """
    live_dir = results_dir / "live_report"
    report = live_dir / "pipeline_status.json"
    journal = live_dir / "pipeline_status_live.jsonl"
    if not report.exists():
        return None
    if journal.exists() and journal.stat().st_mtime > report.stat().st_mtime:
        return None
    with report.open(encoding="utf-8") as f:
        return json.load(f)


def status_from_journal(results_dir):
    """Rebuild per-video stage statuses from the live journal.

    Returns:
        list[dict]: ``{"video", "stages", "dropped_at"}`` per video, in the
            shape of pipeline_status.json's ``videos``. A stage the video has
            no event for is absent, not "missing": the run may simply not
            have reached it.
    """
    journal = results_dir / "live_report" / "pipeline_status_live.jsonl"
    by_video = {}
    if journal.exists():
        with journal.open(encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by a crash mid-write
                video, stage = event.get("video"), event.get("stage")
                status = event.get("status")
                if not (video and stage and status):
                    continue
                # 0e writes a "<stem>_enh" copy; later stages see that copy.
                stages = by_video.setdefault(stem(video), {"name": Path(video).name, "stages": {}})
                current = stages["stages"].get(stage)
                if current is None or _RANK.get(status, 9) < _RANK.get(current, 9):
                    stages["stages"][stage] = status
    videos = []
    for entry in by_video.values():
        stages = entry["stages"]
        dropped_at = next(
            (s for s in STAGE_ORDER if status_kind(stages.get(s)) in DROP_STATUSES), None,
        )
        videos.append({"video": entry["name"], "stages": stages, "dropped_at": dropped_at})
    return videos


def find_crash(log_paths):
    """Return the last exception line of the last traceback in the logs, or None."""
    crash = None
    for log in log_paths:
        in_traceback = False
        for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
            line = ANSI_RE.sub("", raw).rstrip()
            if TRACEBACK_RE.match(line):
                in_traceback = True
            elif in_traceback and EXCEPTION_RE.match(line):
                crash, in_traceback = line, False
    return crash


def summarize(results_dir, log_paths=(), max_list=50):
    """Build the run summary.

    Returns:
        dict: ``verdict``, ``source``, ``totals``, ``stage_counts``
            (stage -> {status: count}), ``last_stage`` (for an incomplete run:
            last stage reached -> video count), ``problems`` (capped list) and
            ``problems_total``.
    """
    report = load_status_report(results_dir)
    if report is not None:
        source = "live_report/pipeline_status.json (written by step 5)"
        videos = report.get("videos") or []
        stage_order = report.get("steps_checked") or STAGE_ORDER
    else:
        source = "live_report/pipeline_status_live.jsonl (step 5 did not write a current report)"
        videos = status_from_journal(results_dir)
        seen = {s for v in videos for s in v["stages"]}
        stage_order = [s for s in STAGE_ORDER if s in seen]

    stage_counts = {}
    for stage in stage_order:
        counts = Counter(status_kind(v["stages"][stage]) for v in videos if stage in v["stages"])
        stage_counts[stage] = {k: counts.get(k, 0) for k in STATUS_COLUMNS + ["other"]}

    dropped = [v for v in videos if v.get("dropped_at")]
    errors = sum(1 for v in videos for s in v["stages"].values() if status_kind(s) == "error")
    crash = find_crash(log_paths) if log_paths else None

    if crash:
        verdict = "CRASHED"
    elif report is None:
        verdict = "INCOMPLETE"
    elif dropped:
        verdict = "COMPLETED_WITH_DROPS"
    else:
        verdict = "COMPLETED"

    # Where an incomplete run left each video that it did not drop.
    last_stage = Counter()
    if report is None:
        for v in videos:
            if not v.get("dropped_at"):
                reached = [s for s in STAGE_ORDER if s in v["stages"]]
                last_stage[reached[-1] if reached else "none"] += 1

    causes = parse_logs(list(log_paths), {stem(v["video"]) for v in dropped}) if log_paths else {}
    problems = []
    # Errors first: a failure is actionable, a 0a "No" verdict is a domain decision.
    kind_order = {"error": 0, "missing": 1, "filtered_out": 2}
    for v in sorted(dropped, key=lambda v: (
            kind_order.get(status_kind(v["stages"].get(v["dropped_at"], "missing")), 3),
            STAGE_ORDER.index(v["dropped_at"]) if v["dropped_at"] in STAGE_ORDER else 99,
            v["video"])):
        notes = causes.get(stem(v["video"])) or []
        problems.append({
            "video": v["video"],
            "dropped_at": v["dropped_at"],
            "status": v["stages"].get(v["dropped_at"], "missing"),
            # The last WARNING naming the video is the one closest to the drop.
            "cause": notes[-1][1] if notes else None,
        })

    return {
        "results_dir": str(results_dir),
        "verdict": verdict,
        "crash": crash,
        "source": source,
        "totals": {"videos": len(videos), "completed": len(videos) - len(dropped),
                   "dropped": len(dropped), "stage_errors": errors},
        "stage_counts": stage_counts,
        "last_stage": dict(last_stage),
        "problems_total": len(problems),
        "problems": problems[:max_list] if max_list >= 0 else problems,
    }


def _cell(text, limit=160):
    text = str(text).replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def to_markdown(summary):
    """Render the summary as markdown."""
    t = summary["totals"]
    # Without a step 5 report, "not dropped" may still mean "not finished".
    finished = summary["verdict"] in ("COMPLETED", "COMPLETED_WITH_DROPS")
    lines = [
        "# VRA Run Summary", "",
        f"**Verdict: {summary['verdict']}** - {t['completed']}/{t['videos']} video(s) "
        f"{'completed' if finished else 'not dropped so far'}, "
        f"{t['dropped']} dropped, {t['stage_errors']} stage error(s).", "",
        f"- Results directory: `{summary['results_dir']}`",
        f"- Source: `{summary['source']}`",
    ]
    if summary["crash"]:
        lines.append(f"- Crash: `{_cell(summary['crash'], 300)}`")
    if summary["verdict"] in ("INCOMPLETE", "CRASHED"):
        lines.append("- Step 5 did not write a current report; stages the run never reached "
                     "are not counted as drops.")
    lines.append("")

    if summary["stage_counts"]:
        used = [c for c in STATUS_COLUMNS + ["other"]
                if any(row[c] for row in summary["stage_counts"].values())]
        lines += ["## Per-stage counts", "",
                  "| Stage | " + " | ".join(used) + " |",
                  "| :--- | " + " | ".join("---:" for _ in used) + " |"]
        for stage, row in summary["stage_counts"].items():
            lines.append(f"| {stage} | " + " | ".join(str(row[c]) for c in used) + " |")
        lines.append("")

    if summary["last_stage"]:
        lines += ["## Where the run stopped", "",
                  "Last stage recorded for each video that was not dropped:", "",
                  "| Last stage | Videos |", "| :--- | ---: |"]
        for stage in STAGE_ORDER + ["none"]:
            if stage in summary["last_stage"]:
                lines.append(f"| {stage} | {summary['last_stage'][stage]} |")
        lines.append("")

    if summary["problems_total"]:
        shown = len(summary["problems"])
        lines += [f"## Problem videos ({summary['problems_total']})", "",
                  "| Video | Dropped at | Status | Cause (from log) |",
                  "| :--- | :--- | :--- | :--- |"]
        for p in summary["problems"]:
            lines.append(f"| {_cell(p['video'])} | {p['dropped_at']} | {_cell(p['status'])} | "
                         f"{_cell(p['cause']) if p['cause'] else '-'} |")
        if shown < summary["problems_total"]:
            lines.append("")
            lines.append(f"+{summary['problems_total'] - shown} more; rerun with `--max-list -1` "
                         "or `--json` for the full list.")
        lines.append("")
    else:
        lines += ["No problem videos.", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    p.add_argument("results_dir", type=Path, help="Host path of the run's results_dir.")
    p.add_argument("--log", type=Path, action="append", default=[],
                   help="Pipeline stdout/stderr log of the run, for causes and crash detection; repeatable.")
    p.add_argument("--max-list", type=int, default=50,
                   help="Problem videos to list (default 50; -1 lists all).")
    p.add_argument("--out", type=Path, help="Also write the markdown summary to this file.")
    p.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    args = p.parse_args(argv)

    if not (args.results_dir / "live_report").is_dir():
        p.error(f"{args.results_dir}/live_report not found; the run has not recorded any status yet")
    summary = summarize(args.results_dir, args.log, args.max_list)
    markdown = to_markdown(summary)
    print(json.dumps(summary, indent=2) if args.json else markdown)
    if args.out:
        try:
            args.out.write_text(markdown + "\n", encoding="utf-8")
        except OSError as e:
            # The container usually runs as root, so live_report/ is often
            # not writable from the host; the summary above still printed.
            print(f"warning: could not write {args.out}: {e.strerror}; "
                  "pass an --out path you own", file=sys.stderr)
    # Non-zero when the run needs attention, so a wrapper can branch on it.
    return 0 if summary["verdict"] == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
