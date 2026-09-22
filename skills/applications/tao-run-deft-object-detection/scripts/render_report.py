#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render ``results/DEFT_Loop_Report.md`` from canonical disk state.

The report is a table-driven summary, and every table in it is a mechanical read
of a file the loop already wrote: the audit's verdict, ``deft_state.json``,
``loop_log.jsonl``, each phase's ``kpi_summary.json``, each iteration's mining
summary and staging report. Nothing in it needs judgement, so nothing in it needs
a model.

Rendering deterministically is what lets the report exist at all in a runtime with
no subagent tool, and it is also the stronger guarantee for the runtime that has
one: `init_deft_state.py` and every successful `commit_stage.py` call refresh the
report, so the end-of-loop render cannot be lost to a saturated parent context.

Two rules are worth stating because they are easy to get wrong when reading:

* The only mAP in the report is the KPI mAP, scored by `kpi_analyze` against the
  evaluation set. A `train` summary that improvised from the training container's
  stdout can carry `val_mAP`/`val_mAP50`, which score agreement with the Co-DETR
  pseudo-labels on the mined-data validation split rather than accuracy. Those are
  dropped from the timeline here; they remain in `loop_log.jsonl`.
* A number that is not on disk is left out rather than guessed. A phase with no
  readable per-class breakdown reports its mAP alone and says the breakdown was
  unavailable.

Inputs:  --results-dir, optionally --out, --require-terminal
Output:  DEFT_Loop_Report.md in the results dir; one status line on stdout

Exits 1 when the state cannot be read, --require-terminal is given for a run that
has not committed loop_stop, or the atomic rename fails.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_deft_run import audit  # noqa: E402
from deft_stages import iter_number, read_log, read_state  # noqa: E402

REPORT_NAME = "DEFT_Loop_Report.md"

# `val_mAP = 0.82`, `val_mAP50: 0.8266`, `val mAP 0.77`, `validation mAP50=0.8` -- and
# the bare name, so a summary that mentions the metric without a number does not
# leave the word behind.
_VAL_METRIC_RE = re.compile(
    r"\b(?:val_[A-Za-z0-9_]*|val(?:idation)?\s+mAP[A-Za-z0-9_]*)"
    r"(\s*[:=]?\s*[-+]?\d+(?:\.\d+)?)?",
    re.IGNORECASE)
# Separators orphaned by the removal above: a leading/trailing comma inside a
# parenthesis, doubled commas, a dangling trailing separator.
_EMPTY_GROUP_RE = re.compile(r"\(\s*[,;]?\s*\)")
_DOUBLE_SEP_RE = re.compile(r"([,;])(\s*[,;])+")
_TRAILING_SEP_RE = re.compile(r"[\s,;]+$")

# `| kpi_v3_with_labels | bicycle | 1688 | ... |` — the per-class table kpi_analyze
# prints. Used only to recover class names for a CSV written before
# tao-data-services#31 added the class_name column.
_LOG_ROW_RE = re.compile(r"^\s*\|(?P<cells>.+)\|\s*$")


def _load_json_object(path: Path | str | None) -> dict[str, Any]:
    """The JSON object at `path`, or {} when it is absent, unreadable or not an object.

    A report is a presentation of whatever the run managed to write; one
    unreadable input costs its own cells, not the whole document. The render runs
    after every commit, so a crash here would leave the report silently stale.
    """
    if not path:
        return {}
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _cell(value: Any) -> str:
    """Text safe inside a Markdown table cell: one line, pipes escaped."""
    return " ".join(str(value).split()).replace("|", "\\|")


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _num(value)
    return f"{number:.{digits}f}" if number is not None else "—"


def _fmt_int(value: Any) -> str:
    number = _num(value)
    return f"{int(number):,}" if number is not None else "—"


def _fmt_duration(seconds: Any) -> str:
    number = _num(seconds)
    # commit_stage.py records 0 when no start time was captured, so 0 means unknown.
    # Printing "0s" would state a measurement nobody took.
    if number is None or number <= 0:
        return "—"
    total = int(number)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {(total % 3600) // 60:02d}m"


def strip_val_metrics(summary: str) -> str:
    """Remove training validation metrics from a stage summary.

    They score the model against the validation split of the staged mined data,
    whose labels are Co-DETR pseudo-labels, so they measure agreement with the
    teacher rather than accuracy. Rendered beside the KPI mAP they read as a
    competing accuracy figure, and the agreement number can be the higher of the
    two. The rest of the summary is reproduced verbatim.
    """
    if not _VAL_METRIC_RE.search(summary):
        return summary
    text = _VAL_METRIC_RE.sub("", summary)
    text = _EMPTY_GROUP_RE.sub("", text)
    text = _DOUBLE_SEP_RE.sub(r"\1", text)
    text = re.sub(r"\s*[,;]\s*(?=\|)", " ", text)  # "2 epochs, | 3 sources"
    # A metric removed from the head of a list leaves its separator behind the
    # colon or bracket that introduced the list.
    text = re.sub(r"([:(])\s*[,;]\s*", r"\1 ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = _TRAILING_SEP_RE.sub("", text).strip()
    return text or "(summary carried only validation metrics)"


# ── phases ───────────────────────────────────────────────────────────────────

def _scored_phases(state: dict[str, Any]) -> list[str]:
    """`baseline`, then every `iterN` present, in run order.

    `prep` builds the pool and is never scored, so it has no row in the KPI or
    data-growth tables. It still appears in the timeline, which reads the log.
    """
    iterations = state.get("iterations")
    iterations = iterations if isinstance(iterations, dict) else {}
    phases = ["baseline"] if "baseline" in iterations else []
    numbered = sorted(
        (n, label)
        for label, n in ((label, iter_number(label)) for label in iterations)
        if n is not None
    )
    return phases + [label for _n, label in numbered]


def _entry(state: dict[str, Any], phase: str) -> dict[str, Any]:
    iterations = state.get("iterations")
    entry = iterations.get(phase) if isinstance(iterations, dict) else None
    return entry if isinstance(entry, dict) else {}


# ── section 1: KPI trend ─────────────────────────────────────────────────────

def _class_names_from_log(path: Path) -> list[str]:
    """Class names in CSV row order, read from the table kpi_analyze prints.

    Only needed for an image built before tao-data-services#31, whose kpi_calc.csv
    has no class_name column.
    """
    names: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        match = _LOG_ROW_RE.match(line)
        if not match:
            continue
        cells = [cell.strip() for cell in match.group("cells").split("|")]
        if len(cells) < 3 or not cells[1] or cells[1].lower() in {"class", "sequence"}:
            continue
        if set(cells[1]) <= {"-", "+"}:  # the table's rule lines
            continue
        if cells[1] not in names:
            names.append(cells[1])
    return names


def _kpi_from_csv(csv_path: Path, log_path: Path | None) -> tuple[float | None, dict[str, float] | None]:
    try:
        with csv_path.open(encoding="utf-8") as handle:
            # A row with more fields than the header carries the surplus as a list
            # under the key None, so only string values are tested for content.
            rows = [r for r in csv.DictReader(handle)
                    if any(isinstance(v, str) and v.strip() for v in r.values())]
    except (OSError, ValueError, csv.Error):  # ValueError covers UnicodeDecodeError
        return None, None
    if not rows or "AP" not in rows[0]:
        return None, None

    labelled = "class_name" in rows[0]
    if labelled:
        rows = [r for r in rows if (r.get("class_name") or "").strip().lower() != "summary"]
    aps = [_num(r.get("AP")) for r in rows]
    aps = [ap for ap in aps if ap is not None]
    if not aps:
        return None, None
    map_value = sum(aps) / len(aps)

    if labelled:
        names = [str(r.get("class_name", "")).strip() for r in rows]
    else:
        names = _class_names_from_log(log_path) if log_path else []
    # Row order is not a class order. Without names from the column or the log,
    # the mean is still exact but the breakdown would be a guess. A repeated name
    # -- one class scored in two KPI sequences -- would collapse to whichever row
    # came last, so that is refused too rather than rendered as one class's AP.
    if len(names) != len(aps) or len(set(names)) != len(names):
        return map_value, None
    return map_value, dict(zip(names, aps))


def _kpi_for_phase(state: dict[str, Any], phase: str) -> dict[str, Any]:
    """mAP and per-class AP for one phase.

    The mAP is the committed `--map-value` first: it is the number the audit checks
    and the loop compared phases on, so the report must not show a different one.
    Only a phase that committed none falls back to recomputing it. The per-class
    breakdown comes from `kpi_summary.json`, then from the CSV.
    """
    entry = _entry(state, phase)
    map_value = _num(entry.get("map_value"))
    per_class: dict[str, float] | None = None

    csv_path = entry.get("kpi_csv")
    if csv_path:
        summary = _load_json_object(Path(csv_path).parent / "kpi_summary.json")
        if map_value is None:
            map_value = _num(summary.get("map_value"))
        resolved = summary.get("per_class")
        if isinstance(resolved, dict) and resolved:
            cleaned = {str(k): _num(v) for k, v in resolved.items()}
            if all(v is not None for v in cleaned.values()):
                per_class = cleaned  # type: ignore[assignment]
        if map_value is None or per_class is None:
            log_path = entry.get("kpi_log")
            csv_map, csv_classes = _kpi_from_csv(
                Path(csv_path), Path(log_path) if log_path else None)
            map_value = map_value if map_value is not None else csv_map
            per_class = per_class or csv_classes
    return {"map": map_value, "per_class": per_class}


def _kpi_section(state: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    phases = _scored_phases(state)
    scored = {p: _kpi_for_phase(state, p) for p in phases}
    scored = {p: k for p, k in scored.items() if k["map"] is not None}
    if not scored:
        return [], {}

    classes: list[str] = []
    for kpi in scored.values():
        for name in (kpi["per_class"] or {}):
            if name not in classes:
                classes.append(name)
    classes.sort()

    lines = [
        "## 1. KPI Trend",
        "",
        "Every number below is scored by `kpi_analyze` against the KPI evaluation set and its",
        "human ground truth. These are the only accuracy figures in this report, and the only",
        "mAP anywhere in it.",
        "",
    ]
    header = ["Phase", "mAP"] + [f"AP50 {_cell(c)}" for c in classes]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    missing_breakdown: list[str] = []
    for phase, kpi in scored.items():
        row = [_cell(phase), _fmt(kpi["map"])]
        if kpi["per_class"] is None and classes:
            missing_breakdown.append(phase)
        for name in classes:
            row.append(_fmt((kpi["per_class"] or {}).get(name)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    ordered = list(scored)
    first, last = ordered[0], ordered[-1]
    if first != last:
        delta = scored[last]["map"] - scored[first]["map"]
        sentence = (f"{first} mAP {_fmt(scored[first]['map'])} → {last} "
                    f"{_fmt(scored[last]['map'])} ({delta:+.4f}).")
        against = []
        for name in classes:
            start = (scored[first]["per_class"] or {}).get(name)
            end = (scored[last]["per_class"] or {}).get(name)
            if start is None or end is None:
                continue
            moved = end - start
            if moved and delta and (moved > 0) != (delta > 0):
                against.append(f"{name} ({moved:+.4f})")
        if against:
            sentence += " Moved against the mean: " + ", ".join(against) + "."
        lines.extend([sentence, ""])
    if missing_breakdown:
        lines.extend([
            "Per-class breakdown unavailable for " + ", ".join(missing_breakdown)
            + ": the CSV carries no `class_name` column and no `kpi_analyze.log` was "
              "committed, and row order is not a class order.",
            "",
        ])
    return lines, scored


# ── section 2: data growth ───────────────────────────────────────────────────

def _train_sources(entry: dict[str, Any]) -> int | None:
    """Count of `dataset.train_data_sources` in the committed training spec.

    Optional: the renderer runs wherever `commit_stage.py` runs, and the column is
    worth less than a crash when no YAML parser is importable.
    """
    spec = entry.get("training_spec")
    if not spec:
        return None
    try:
        import yaml  # noqa: PLC0415 - optional, and only for this column
    except ImportError:
        return None
    try:
        with Path(spec).open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, ValueError, yaml.YAMLError):
        return None
    dataset = loaded.get("dataset") if isinstance(loaded, dict) else None
    sources = dataset.get("train_data_sources") if isinstance(dataset, dict) else None
    return len(sources) if isinstance(sources, list) else None


def _staging_report(results_dir: Path, state: dict[str, Any], phase: str) -> dict[str, Any]:
    entry = _entry(state, phase)
    # `--report-json` is written beside the staged annotations but is not itself a
    # committed artifact, so derive it from one that is before falling back to the
    # documented location.
    odvg = entry.get("odvg_jsonl")
    candidates = []
    if odvg:
        candidates.append(Path(odvg).parent.parent / "staging_report.json")
    candidates.append(results_dir / phase / "tmm" / "staging_report.json")
    for candidate in candidates:
        loaded = _load_json_object(candidate)
        if loaded:
            return loaded
    return {}


def _staging_gap(row: dict[str, Any]) -> int | None:
    """Mined images that did not reach training, or None when it cannot be known.

    Measured from `Mined`, not from the images copied: a mined image the pool no
    longer holds is never copied, so a gap measured from the copies would miss it.
    """
    mined, staged = _num(row["mined"]), _num(row["staged"])
    if mined is None or staged is None:
        return None
    return int(mined - staged)


def _growth_section(results_dir: Path, state: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for phase in _scored_phases(state):
        if iter_number(phase) is None:
            continue
        entry = _entry(state, phase)
        mining = _load_json_object(entry.get("mining_summary_json"))
        staging = _staging_report(results_dir, state, phase)
        missing_images = staging.get("missing_images")
        missing_annotations = staging.get("missing_annotations")
        rows.append({
            "phase": phase,
            "weak": entry.get("weak_image_count"),
            # The staging report's own count first: it is what staging worked from.
            "mined": (staging.get("mined_unique")
                      if staging.get("mined_unique") is not None
                      else mining.get("retrieved_unique_count")),
            "coverage": mining.get("coverage_pct"),
            "staged": staging.get("annotations_written"),
            "missing_images": len(missing_images) if isinstance(missing_images, list) else None,
            "missing_annotations": (len(missing_annotations)
                                    if isinstance(missing_annotations, list) else None),
            "sources": _train_sources(entry),
        })
    if not any(any(v is not None for k, v in row.items() if k != "phase") for row in rows):
        return [], rows

    lines = [
        "## 2. Data Growth",
        "",
        "| Iteration | Weak images | Mined | Coverage % | Staged | Train sources |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join([
            _cell(row["phase"]),
            _fmt_int(row["weak"]),
            _fmt_int(row["mined"]),
            _fmt(row["coverage"], 2),
            _fmt_int(row["staged"]),
            _fmt_int(row["sources"]),
        ]) + " |")
    lines.append("")
    # `Staged` is what actually reached training. It falls below `Mined` when a mined
    # image is missing from the pool or has no annotation there.
    gaps = []
    for row in rows:
        gap = _staging_gap(row)
        if not gap or gap < 0:
            continue
        causes = []
        if row["missing_images"]:
            causes.append(f"{row['missing_images']} missing from the pool")
        if row["missing_annotations"]:
            causes.append(f"{row['missing_annotations']} with no annotation in the pool")
        gaps.append(f"{row['phase']} lost {gap} of {_fmt_int(row['mined'])} mined images"
                    + (f" ({', '.join(causes)})" if causes else ""))
    if gaps:
        lines.extend(["`Staged` is below `Mined`: " + "; ".join(gaps) + ".", ""])
    return lines, rows


# ── section 3: stage timeline ────────────────────────────────────────────────

def _timeline_section(events: list[dict[str, Any]]) -> list[str]:
    if not events:
        return []
    lines = [
        "## 3. Stage Timeline",
        "",
        "| Seq | Phase | Stage | Status | Duration | Summary |",
        "|---|---|---|---|---|---|",
    ]
    for event in events:
        summary = strip_val_metrics(str(event.get("summary", "")).strip())
        lines.append("| " + " | ".join([
            _cell(event.get("seq", "—")),
            _cell(event.get("iter", "—")),
            _cell(event.get("stage", "—")),
            _cell(event.get("status", "—")),
            _fmt_duration(event.get("duration_sec")),
            _cell(summary),
        ]) + " |")
    lines.append("")
    return lines


# ── section 4: configuration ─────────────────────────────────────────────────

_CONFIG_ROWS = [
    ("embedding_model", "Encoder"),
    ("allocation_policy", "Allocation policy"),
    ("rare_class_list", "Rare classes"),
    ("multiplier", "Mining multiplier"),
    ("num_epochs", "Epochs"),
    ("learning_rate", "Learning rate"),
    ("num_gpus", "GPUs"),
    ("kpi_conf_threshold", "KPI confidence threshold"),
]


def _config_section(state: dict[str, Any]) -> list[str]:
    config = state.get("config")
    config = config if isinstance(config, dict) else {}
    if not config:
        return []
    lines = ["## 4. Configuration", "", "| Setting | Value |", "|---|---|"]
    for key, label in _CONFIG_ROWS:
        value = config.get(key)
        lines.append(f"| {label} | {'—' if value is None else _cell(value)} |")
    thresholds = config.get("ap50_thresholds")
    if isinstance(thresholds, dict) and thresholds:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(thresholds.items()))
        lines.append(f"| Per-class AP50 thresholds | {_cell(rendered)} |")
    lines.append("")
    conf = _num(config.get("kpi_conf_threshold"))
    if conf:
        # A run scored at a different threshold is not comparable to one scored at
        # 0.0, and a reader who cannot see the value cannot know that.
        lines.extend([
            f"Every KPI mAP above is scored at a confidence threshold of {conf}. A run "
            "scored at a different threshold is not comparable to this one.",
            "",
        ])
    return lines


# ── section 5: observations ──────────────────────────────────────────────────

def _observations_section(rows: list[dict[str, Any]], scored: dict[str, dict[str, Any]],
                          events: list[dict[str, Any]], report: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for row in rows:
        coverage = _num(row["coverage"])
        if coverage is not None and coverage < 50:
            notes.append(f"`{row['phase']}` mining coverage fell to {coverage:.2f}% — the "
                         "source pool is running dry and later iterations will add little.")
        gap = _staging_gap(row)
        if gap and gap > 0:
            notes.append(f"`{row['phase']}` staging gap: {gap} mined images did not reach "
                         f"training (see Data Growth).")
    for event in events:
        if event.get("status") == "error":
            notes.append(f"`{event.get('iter')}/{event.get('stage')}` committed "
                         f"`status=error`: {strip_val_metrics(str(event.get('summary', '')))}")
    if scored:
        best = max(scored.items(), key=lambda kv: kv[1]["map"])
        notes.append(f"Best KPI mAP: `{best[0]}` at {_fmt(best[1]['map'])}.")
    # The completion reason is not repeated here: the status line carries it wherever
    # the status alone would mislead (see _status).
    if report.get("status") == "INVALID":
        # The audit's errors are the repair list. Truncating them would hide the one
        # that matters, so every one is carried.
        for error in report.get("errors") or []:
            notes.append(f"Audit error: {error}")
    if not notes:
        return []
    return ["## 5. Observations", ""] + [f"- {n}" for n in notes] + [""]


# ── document ─────────────────────────────────────────────────────────────────

EARLY_STOP_PREFIX = "documented early stop"


def _status(report: dict[str, Any], state: dict[str, Any]) -> str:
    """The status line, with the reason beside it wherever the word alone misleads.

    A documented early stop -- an exhausted pool, or an iteration with no weak images
    -- is COMPLETE, and so is a run that finished every iteration; the reason is the
    only thing that tells them apart, so an early stop carries it. A STOPPED run
    always does, since "why did it stop short" is the first question it raises. The
    reason is read from state, where commit_stage.py records it at loop_stop, and
    falls back to the audit's.
    """
    # An inconsistent run has no trustworthy verdict to render. Say that at the top
    # rather than presenting the numbers as though the audit had accepted them.
    if report.get("status") == "INVALID":
        return "INVALID (audit found disk inconsistencies)"
    if report.get("run_failed"):
        return "FAILED"
    reason = state.get("completion_reason") or report.get("completion_reason") or ""
    if report.get("complete"):
        return f"COMPLETE ({reason})" if reason.startswith(EARLY_STOP_PREFIX) else "COMPLETE"
    if report.get("loop_stop_committed"):
        return "STOPPED (INCOMPLETE)" + (f" ({reason})" if reason else "")
    return "IN PROGRESS"


def compose(results_dir: Path, state: dict[str, Any], events: list[dict[str, Any]],
            report: dict[str, Any], generated: str) -> str:
    kpi_lines, scored = _kpi_section(state)
    growth_lines, rows = _growth_section(results_dir, state)
    config = state.get("config")
    config = config if isinstance(config, dict) else {}
    completed = report.get("iterations_completed")
    # The audit always carries the key, sometimes as None, so a .get default would
    # never reach the config's value.
    max_iterations = report.get("max_iterations")
    if max_iterations is None:
        max_iterations = config.get("max_iterations")

    lines = [
        "# DEFT OD Loop Report",
        "",
        f"**Status:** {_status(report, state)}",
        f"**Iterations completed:** {completed if completed is not None else '—'}"
        f" / {max_iterations if max_iterations is not None else '—'}",
        "**Model:** Grounding DINO (ODVG)",
        f"**Generated:** {generated}",
        "",
    ]
    lines += kpi_lines
    lines += growth_lines
    lines += _timeline_section(events)
    lines += _config_section(state)
    lines += _observations_section(rows, scored, events, report)
    lines.append("<!-- Rendered by scripts/render_report.py from deft_state.json, "
                 "loop_log.jsonl and the committed stage artifacts. -->")
    return "\n".join(lines).rstrip() + "\n"


def render(results_dir: str | Path, report: dict[str, Any] | None = None,
           out: str | Path | None = None) -> Path:
    """Write the report and return its path.

    `report` is the audit verdict. `commit_stage.py` already holds one at the point
    it renders, and re-auditing would be a third read of the same state. `out`
    replaces the default location; nothing is written to the results dir then.
    """
    results_dir = Path(results_dir)
    state = read_state(results_dir)
    events = read_log(results_dir)
    if report is None:
        report = audit(results_dir)
    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    target = Path(out) if out else results_dir / REPORT_NAME
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(compose(results_dir, state, events, report, generated), encoding="utf-8")
    os.replace(tmp, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", default=None,
                        help=f"Write here instead of {REPORT_NAME} in the results dir, "
                             "which is then left untouched.")
    parser.add_argument("--require-terminal", action="store_true",
                        help="Refuse to render unless loop_stop is committed. For an "
                             "end-of-loop render: a hard stop that has not been "
                             "finalized yet is not a finished run.")
    args = parser.parse_args()
    try:
        results_dir = Path(args.results_dir).expanduser().resolve()
        report = None
        if args.require_terminal:
            report = audit(results_dir)
            if not report.get("terminal"):
                raise ValueError(
                    "--require-terminal was given but the run is not terminal "
                    f"(next action: {report.get('next_action', 'unknown')}). The "
                    "existing report is left as it was.")
        out = Path(args.out).expanduser().resolve() if args.out else None
        written = render(results_dir, report, out)
        size = written.stat().st_size
        print(f"render_report: wrote {written.name} ({size}B)")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"render_report: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
