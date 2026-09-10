#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Prepare an offline pool evaluation plan or score its predictions on CPU.

Neither subcommand launches inference, submits jobs, or changes loop state.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
import pathlib
import re
import sys
from typing import Any, Iterable

from analyze_gaps import _load_evaluator
from build_target_profile import (
    TASKS, build_profile, dataset_family, read_rows, row_features,
    sha256_file, write_outputs, image_paths, prompt_and_response,
)
from cfw_action_plan import build_action_plan
from cfw_predictions import normalize_prediction
from render_cfw_evaluate import build_config
from render_cfw_sft import dump_toml

SAMPLE_SCHEMA = "nvpaw_pool_sample_v1"
RESIDUAL_SCHEMA = "nvpaw_pool_residual_v1"
COUNT_TASK = "Component Count"
SCORABLE_TASKS = (*TASKS, COUNT_TASK)
COUNT_CONTRACT = "analysis-only Component Count extension: exact nonnegative integer equality, wrong_class on mismatch, parse_failure on noninteger; not an official evaluator KPI"
ERROR_TYPES = ("correct", "wrong_class", "empty_on_positive", "boxes_on_empty",
               "missed_boxes", "extra_boxes", "misaligned", "parse_failure")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_id(row: dict, seen: set[str], context: str) -> str:
    key = row.get("id")
    if not isinstance(key, str) or not key or key in seen:
        raise ValueError(f"{context}: missing or duplicate string id {key!r}")
    seen.add(key)
    return key


def _parse_count(text: str) -> int | None:
    if not re.fullmatch(r"[0-9]+", text.strip()):
        return None
    digits = text.strip().lstrip("0") or "0"
    # The consumer column is int64. A hallucinated huge integer must become a
    # row parse failure, not abort publication after scoring the entire pool.
    if len(digits) > 19:
        return None
    value = int(digits)
    return value if value <= 2**63 - 1 else None


def residual_features(record: dict) -> dict:
    if record.get("task_type") != COUNT_TASK:
        return row_features(record)
    _, answer = prompt_and_response(record, context=str(record.get("id")))
    count = _parse_count(answer)
    if count is None:
        raise ValueError(f"{record.get('id')}: Component Count GT must be a nonnegative int64 integer")
    if len(image_paths(record, context=str(record.get("id")))) != 1:
        raise ValueError("Component Count requires one image")
    return {"task_type": COUNT_TASK, "dataset": dataset_family(record), "gt_count": count,
            "empty_status": "empty" if count == 0 else "non_empty", "count_bin": "NA",
            "size_bin": "NA", "pair_kind": "single", "gt_boxes": 0}


def build_residual_profile(records: Iterable[dict]) -> dict:
    """Reuse A for its six tasks, adding count-only cells without inventing boxes."""
    count_cells: Counter = Counter()
    labels: Counter = Counter()
    def benchmark_rows():
        for record in records:
            if record.get("task_type") == COUNT_TASK:
                features = residual_features(record)
                count_cells[(features["dataset"], features["empty_status"])] += 1
                labels[str(features["gt_count"])] += 1
            else:
                yield record
    profile = build_profile(benchmark_rows())
    count = sum(count_cells.values())
    if count:
        empty = sum(value for (_, status), value in count_cells.items() if status == "empty")
        datasets: Counter = Counter()
        for (dataset, status), rows in sorted(count_cells.items()):
            datasets[dataset] += rows
            profile["cells"].append({"task_type": COUNT_TASK, "dataset": dataset,
                "empty_status": status, "count_bin": "NA", "size_bin": "NA", "pair_kind": "single", "rows": rows})
        profile["tasks"][COUNT_TASK] = {"rows": count, "empty_rows": empty, "empty_rate": empty / count,
            "boxes": 0, "boxes_per_nonempty_row": None, "box_size_bins": {}, "count_bins": {"NA": count},
            "nonempty_count_share": {}, "relative_area_quantiles": {str(q): None for q in (10, 25, 50, 75, 90)},
            "aspect_ratio_median": None, "label_histogram": dict(sorted(labels.items())),
            "pair_kind": {"single": count}, "dataset_histogram": dict(sorted(datasets.items()))}
        profile["rows"] += count
    profile["definitions"]["counting_extension"] = COUNT_CONTRACT
    return profile


def sample_pool(records: Iterable[dict], *, per_group: int = 2000, seed: int = 17) -> tuple[list[dict], dict]:
    """Keep smallest seed/id hashes per task/family without rewriting records."""
    if not 1 <= per_group <= 2000:
        raise ValueError("per-group must be between 1 and 2000")
    heaps: dict[tuple, list] = defaultdict(list)
    available: Counter = Counter()
    ignored: Counter = Counter()
    seen: set[str] = set()
    for record in records:
        key = _row_id(record, seen, "mining pool")
        task = record.get("task_type")
        if task not in SCORABLE_TASKS:
            ignored[str(task)] += 1
            continue
        group = (task, dataset_family(record))
        available[group] += 1
        rank = int.from_bytes(hashlib.sha256(f"{seed}\0{key}".encode()).digest(), "big")
        entry = (-rank, key, record)
        heap = heaps[group]
        if len(heap) < per_group:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    selected = [entry[2] for group in sorted(heaps)
                for entry in sorted(heaps[group], key=lambda item: item[1])]
    for record in selected:
        residual_features(record)  # Validate native messages/GT before a plan can be emitted.
    return selected, {
        "schema_version": SAMPLE_SCHEMA, "analysis_only": True, "seed": seed,
        "per_group_cap": per_group, "rows_seen": len(seen), "rows": len(selected),
        "ignored_task_rows": dict(sorted(ignored.items())),
        "groups": [{"task_type": group[0], "dataset": group[1],
                    "available": available[group], "selected": len(heaps[group])}
                   for group in sorted(heaps)],
    }


def evaluation_plan(sample: pathlib.Path, output: pathlib.Path, *, checkpoint: str,
                    media_root: str, image: str, batch_size: int) -> tuple[dict, dict]:
    config_path = output / "evaluate_pool.toml"
    config = build_config(action="evaluate", annotation_path=str(sample),
                          media_root=media_root, model_path=checkpoint,
                          results_dir=str(output), num_gpus=8, batch_size=batch_size,
                          row_order="task_length_sorted")
    scripts = pathlib.Path(__file__).resolve().parent
    plan = build_action_plan(
        action="evaluate", image=image, config_path=str(config_path),
        results_dir=str(output), runtime_adapter=str(scripts / "cfw_jsonl_runtime.py"),
        output_jsonl=str(output / "shards/predictions_rank{rank}.jsonl"),
        source_jsonl=str(sample), media_root=media_root, checkpoint_path=checkpoint,
        base_model_path=checkpoint, num_gpus=8,
    )
    command = plan["command"]
    plan["command"] = [command[0], "-m", "torch.distributed.run", "--standalone",
                       "--nproc-per-node=8", *command[1:]]
    plan["resources"].update(partition="interactive", time="03:59:00", tasks=1)
    plan["merge_command"] = [command[0], str(scripts / "merge_cfw_prediction_shards.py"),
        "--source", str(sample), "--shard-dir", str(output / "shards"),
        "--expected-shards", "8", "--output", str(output / "predictions.jsonl"),
        "--summary-output", str(output / "predictions_manifest.json")]
    plan.update(analysis_only=True, operator_review_required=True)
    return config, plan


def prepare(args: argparse.Namespace) -> None:
    output = args.output_dir.expanduser().resolve()
    source = args.input.expanduser().resolve()
    checkpoint = str(args.model_path.expanduser().resolve())
    media_root = str(args.media_root.expanduser().resolve())
    selected, manifest = sample_pool(read_rows(source), per_group=args.per_group, seed=args.seed)
    if not selected:
        raise ValueError("mining pool supplies no supported rows")
    sample = output / "pool_sample.jsonl"
    sample_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    config_path = output / "evaluate_pool.toml"
    config, plan = evaluation_plan(sample, output, checkpoint=checkpoint, media_root=media_root,
                                   image=args.image, batch_size=args.batch_size)
    plan.update(rows=len(selected),
                exceeds_approx_60000_row_budget=len(selected) > 60000,
                execution_note="Descriptor only. Operator owns preflight, mounts, job record, submit and shard merge.")
    manifest.update(input={"path": str(source), "sha256": sha256_file(source)},
                    sample={"path": str(sample), "sha256": hashlib.sha256(sample_text.encode()).hexdigest()},
                    checkpoint=checkpoint, created_at=_utc_now(),
                    config_sha256=hashlib.sha256(dump_toml(config).encode()).hexdigest())
    plan["sample_manifest"] = str(output / "pool_sample_manifest.json")
    write_outputs({sample: sample_text, output / "pool_sample_manifest.json": _json(manifest),
                   config_path: dump_toml(config), output / "pool_scoring_plan.json": _json(plan)})


def validate_sample_manifest(sample: pathlib.Path, manifest: dict, checkpoint: str) -> None:
    if manifest.get("schema_version") != SAMPLE_SCHEMA:
        raise ValueError("unsupported pool sample manifest schema")
    if manifest.get("sample", {}).get("sha256") != sha256_file(sample):
        raise ValueError("pool sample SHA-256 does not match the prepared sample manifest")
    if manifest.get("checkpoint") != str(pathlib.Path(checkpoint).expanduser().resolve()):
        raise ValueError("checkpoint does not match the prepared sample manifest")


def score_row(evaluator: Any, source: dict, prediction: dict) -> dict:
    """Delegate canonical parsing and one-to-one matching, never approximate F1."""
    row_id = source["id"]
    raw = prediction.get("raw_prediction", "")
    raw = raw if isinstance(raw, str) else ""
    classification = evaluator.build_classification_examples({row_id: source}) if source.get("task_type") != COUNT_TASK else []
    gt_signature = predicted_signature = None
    gt_count_value = predicted_count = None
    tp = fp = fn = gt_count = 0
    if source.get("task_type") == COUNT_TASK:
        gt_count_value = residual_features(source)["gt_count"]
        predicted_count = _parse_count(raw)
        parse_ok = predicted_count is not None
        error = "parse_failure" if not parse_ok else "correct" if predicted_count == gt_count_value else "wrong_class"
        score = float(error == "correct")
    elif classification:
        example = classification[0]
        if example["answer_format"] == "direct_bcq":
            labels, parse_ok = evaluator.parse_direct_bcq(raw)
        else:
            labels, parse_ok = evaluator.parse_choice_labels(raw, example["option_letters"], example["option_text"])
        error = "parse_failure" if not parse_ok else "correct" if labels == example["gt_labels"] else "wrong_class"
        score = float(error == "correct")
        # Include option meanings: the same letter need not denote the same label.
        gt_signature = json.dumps([example["option_text"], sorted(example["gt_labels"])], sort_keys=True)
        predicted_signature = json.dumps(sorted(labels)) if parse_ok else None
    else:
        examples = evaluator.build_detection_examples({row_id: source}, 1.0)
        if not examples:
            raise ValueError(f"exact evaluator cannot classify source row {row_id!r}")
        gt = examples[0]["gt_boxes"]
        gt_count = len(gt)
        native, parse_ok = evaluator.parse_boxes(raw)
        boxes = evaluator.canonicalize_prediction_boxes(native, "xyxy") if parse_ok else []
        if parse_ok:
            tp, fp, fn = evaluator.one_to_one_detection_counts(gt, boxes, 0.5)
        else:
            fn = gt_count
        if not parse_ok:
            error = "parse_failure"
        elif fp == 0 and fn == 0:
            error = "correct"
        elif not boxes:
            error = "empty_on_positive"
        elif not gt:
            error = "boxes_on_empty"
        elif evaluator.one_to_one_detection_counts(gt, boxes, 0.3)[0] > tp:
            # More boxes become matchable only by admitting 0.3 < IoU <= 0.5.
            error = "misaligned"
        else:
            error = "missed_boxes" if fn else "extra_boxes"
        score = 1.0 if error == "correct" else tp / (tp + fp + fn) if parse_ok and tp else 0.0
    confidence = prediction.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"prediction {row_id!r}: confidence must be finite in [0,1]")
        confidence = float(confidence)
    review = bool(parse_ok and error != "correct" and confidence is not None and confidence >= 0.9)
    return {"row_score": score, "error_type": error, "parse_ok": bool(parse_ok),
            "gt_boxes": gt_count, "matched_boxes": tp, "false_positives": fp, "false_negatives": fn,
            "gt_count": gt_count_value, "predicted_count": predicted_count,
            "prediction_confidence": confidence, "review_label_noise": review,
            "review_reason": "human review: supplied confidence >=0.9 disagrees with GT; not proof of label noise" if review else "",
            "gt_label_pattern": gt_signature, "predicted_labels": predicted_signature}


def score_records(evaluator: Any, sources: Iterable[dict], predictions: Iterable[dict], *, checkpoint: str, scored_at: str,
                  flag_patterns: bool = True) -> list[dict]:
    source_by_id, prediction_by_id = {}, {}
    for rows, index, context in ((sources, source_by_id, "sample"), (predictions, prediction_by_id, "predictions")):
        seen: set[str] = set()
        for record in rows:
            index[_row_id(record, seen, context)] = record
    if not source_by_id:
        raise ValueError("sample is empty")
    missing = source_by_id.keys() - prediction_by_id.keys()
    unknown = prediction_by_id.keys() - source_by_id.keys()
    if missing or unknown:
        raise ValueError(f"prediction coverage mismatch: missing={sorted(missing)[:10]}, unknown={sorted(unknown)[:10]}")
    scored = []
    patterns: dict[tuple, list] = defaultdict(list)
    for key, source in source_by_id.items():
        prediction = prediction_by_id[key]
        if not isinstance(prediction.get("raw_prediction"), str):
            raise ValueError(f"prediction {key!r} requires string raw_prediction")
        expected = normalize_prediction(source, prediction)
        for field in ("task_type", "message", "GT"):
            if field in prediction and prediction[field] != expected[field]:
                raise ValueError(f"prediction {key!r}: {field} does not match the sampled source")
        features = residual_features(source)
        result = {field: features[field] for field in ("task_type", "dataset", "size_bin", "count_bin", "pair_kind")}
        result.update(score_row(evaluator, source, prediction), id=key, checkpoint=checkpoint, scored_at=scored_at)
        result["is_residual"] = result["error_type"] != "correct"
        scored.append(result)
        if result["gt_label_pattern"] is not None:
            patterns[(result["task_type"], result["dataset"], result["gt_label_pattern"])].append(result)
    for group in patterns.values() if flag_patterns else ():
        counts = Counter(item["predicted_labels"] for item in group if item["parse_ok"])
        if len(group) < 20 or not counts:
            continue
        pattern, count = counts.most_common(1)[0]
        if count / len(group) < 0.95:
            continue
        for item in group:
            if item["error_type"] == "wrong_class" and item["predicted_labels"] == pattern and not item["review_label_noise"]:
                item["review_label_noise"] = True
                item["review_reason"] = f"human review: repeated label-pattern disagreement ({count}/{len(group)}); not model confidence or proof of label noise"
    return scored


def parquet_schema(provenance: dict) -> Any:
    import pyarrow as pa
    strings = ("id", "task_type", "dataset", "error_type", "size_bin", "count_bin", "pair_kind",
               "checkpoint", "scored_at", "review_reason", "gt_label_pattern", "predicted_labels")
    integers = ("gt_boxes", "matched_boxes", "false_positives", "false_negatives", "gt_count", "predicted_count")
    fields = [(name, pa.string()) for name in strings] + [(name, pa.int64()) for name in integers]
    fields += [("row_score", pa.float64()), ("prediction_confidence", pa.float64())]
    fields += [(name, pa.bool_()) for name in ("parse_ok", "is_residual", "review_label_noise")]
    return pa.schema(fields, metadata={b"schema_version": RESIDUAL_SCHEMA.encode(), b"provenance": _json(provenance).encode()})


def _parquet_bytes(rows: list[dict], provenance: dict) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows, schema=parquet_schema(provenance)), sink)
    return sink.getvalue().to_pybytes()


def residual_report(rows: Iterable[dict], provenance: dict) -> str:
    lines = ["# Pool residual analysis", "", "Analysis only; this is row accuracy, not official F1.", "",
             "| Task | Dataset | Rows | Correct | Accuracy |", "| --- | --- | ---: | ---: | ---: |"]
    groups: dict[tuple, Counter] = defaultdict(Counter)
    counts: Counter = Counter()
    candidates = []
    total = with_confidence = candidate_count = 0
    for row in rows:
        groups[(row["task_type"], row["dataset"])][row["error_type"]] += 1
        counts[row["error_type"]] += 1
        total += 1
        with_confidence += row["prediction_confidence"] is not None
        if row["review_label_noise"]:
            candidate_count += 1
            if len(candidates) < 100:
                candidates.append(row)
    for (task, dataset), group in sorted(groups.items()):
        correct, size = group["correct"], sum(group.values())
        lines.append(f"| {task} | {dataset} | {size} | {correct} | {correct / size:.4f} |")
    lines += ["", "## Error mix", "", "| Error | Rows |", "| --- | ---: |"]
    lines += [f"| {error} | {counts[error]} |" for error in ERROR_TYPES]
    lines += ["", "## Suspicious-GT review candidates", "",
              f"{candidate_count} candidates; {with_confidence}/{total} rows have supplied confidence.", "",
              "The native runtime does not emit confidence. Disagreement cannot establish that the model is right "
              "or the GT is wrong. Flags mean human review only: supplied confidence >=0.9 on a parseable error, "
              "or >=95% repeated classification disagreement within a task/dataset/GT-option-label pattern of "
              "at least 20 rows. The latter is a consistency heuristic, not calibrated confidence. No relabeling is performed.", "",
              "First 100 candidates below; all flags and reasons are retained in parquet.", ""]
    lines += [f"- {row['id']}: {row['review_reason']}" for row in candidates[:100]] or ["No candidates meet the review criteria."]
    lines += ["", "## Provenance and scoring contract", "", "```json", _json(provenance).rstrip(), "```", ""]
    return "\n".join(lines)


def score(args: argparse.Namespace) -> None:
    sample = args.sample.expanduser().resolve()
    manifest_path = args.sample_manifest or sample.with_name("pool_sample_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = str(args.checkpoint.expanduser().resolve())
    validate_sample_manifest(sample, manifest, checkpoint)
    sources = list(read_rows(sample))
    if len(sources) != manifest.get("rows"):
        raise ValueError("sample row count does not match the prepared sample manifest")
    evaluator_path = args.evaluator.expanduser().resolve()
    evaluator_sha = sha256_file(evaluator_path)
    if args.evaluator_sha256 and args.evaluator_sha256 != evaluator_sha:
        raise ValueError("evaluator SHA-256 does not match --evaluator-sha256")
    evaluator = _load_evaluator(evaluator_path)
    timestamp = _utc_now()
    rows = score_records(evaluator, sources, read_rows(args.predictions), checkpoint=checkpoint, scored_at=timestamp)
    failed = {row["id"] for row in rows if row["is_residual"]}
    provenance = {"schema_version": RESIDUAL_SCHEMA, "analysis_only": True,
        "checkpoint": checkpoint, "scored_at": timestamp, "rows_scored": len(rows), "residual_rows": len(failed),
        "sample_sha256": manifest["sample"]["sha256"], "sample_manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": sha256_file(args.predictions), "evaluator": str(evaluator_path), "evaluator_sha256": evaluator_sha,
        "correctness": "canonical exact classification sets; threshold-aware one-to-one detection IoU > 0.5, labels ignored, explicit empty accepted",
        "row_score": "classification 1/0; detection correct=1, otherwise TP/(TP+FP+FN), parse failure=0; NOT official F1",
        "error_precedence": "parse_failure, correct, empty_on_positive, boxes_on_empty, misaligned, missed_boxes, extra_boxes",
        "misaligned": "additional one-to-one matches become possible at IoU >0.3 compared with >0.5",
        "coordinate_scale": 1.0, "prediction_coordinate_order": "xyxy", "counting_extension": COUNT_CONTRACT}
    profile = build_residual_profile(source for source in sources if source["id"] in failed)
    profile["scoring"] = provenance
    parquet = _parquet_bytes(rows, provenance)
    output = args.output_dir.expanduser().resolve()
    text_outputs = {output / "residual_profile.json": _json(profile),
                    output / "RESIDUAL_REPORT.md": residual_report(rows, provenance)}
    parquet_path = output / "pool_residual.parquet"
    for path in (*text_outputs, parquet_path):
        if path.exists():
            raise ValueError(f"refusing to overwrite existing output: {path}")
    write_outputs(text_outputs)
    with parquet_path.open("xb") as stream:
        stream.write(parquet)


def seal_chunk(output: pathlib.Path, index: int) -> None:
    from pool_residual_full import seal_chunk as seal
    seal(output, index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    planning = commands.add_parser("prepare", help="CPU sampling and evaluate descriptor only; never launch")
    for name in ("input", "output-dir", "model-path", "media-root"):
        planning.add_argument("--" + name, type=pathlib.Path, required=True)
    planning.add_argument("--image", required=True, help="approved CFW image pinned by @sha256 digest")
    planning.add_argument("--per-group", type=int, default=2000)
    planning.add_argument("--seed", type=int, default=17)
    planning.add_argument("--batch-size", type=int, default=1)
    planning.add_argument("--mode", choices=("sample", "full"), default="sample")
    planning.add_argument("--chunks", type=int, default=16)
    planning.add_argument("--concurrency", type=int, default=8)
    planning.add_argument("--partition", default="interactive")
    planning.add_argument("--sqsh", type=pathlib.Path)
    planning.add_argument("--container-mounts", help="explicit shared source:target Pyxis mounts, including all image roots")
    planning.add_argument("--cpus-per-task", type=int, default=64)
    planning.add_argument("--account", default="")
    planning.add_argument("--constraint", default="h100", help="operator-verified H100 feature name for the chosen partition")
    merging = commands.add_parser("merge", help="CPU merge and score all complete full-pool chunks")
    merging.add_argument("--output-dir", type=pathlib.Path, required=True)
    merging.add_argument("--evaluator", type=pathlib.Path, required=True)
    merging.add_argument("--evaluator-sha256")
    running = commands.add_parser("run-chunk", help="operator-only allocation entrypoint; runs the existing evaluate engine")
    running.add_argument("--output-dir", type=pathlib.Path, required=True)
    running.add_argument("--chunk", type=int, required=True)
    scoring = commands.add_parser("score", help="CPU scoring of operator-produced predictions")
    for name in ("sample", "predictions", "checkpoint", "evaluator", "output-dir"):
        scoring.add_argument("--" + name, type=pathlib.Path, required=True)
    scoring.add_argument("--sample-manifest", type=pathlib.Path)
    scoring.add_argument("--evaluator-sha256", help="optional expected evaluator code digest")
    args = parser.parse_args(argv)
    try:
        if args.action in {"merge", "run-chunk"} or args.action == "prepare" and args.mode == "full":
            from pool_residual_full import prepare_full, merge_full, run_chunk
            {"prepare": prepare_full, "merge": merge_full, "run-chunk": run_chunk}[args.action](args)
        else:
            prepare(args) if args.action == "prepare" else score(args)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        print(f"score_pool_residual: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
