# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Full-pool chunk orchestration for score_pool_residual; no submit operation.

Preparation/merge are CPU-only. The explicit run-chunk entrypoint delegates
inference to the existing CFW runtime inside an operator-owned allocation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
import fcntl
import json
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile

import score_pool_residual as scorer
from build_target_profile import read_rows, sha256_file, write_outputs
from cfw_jsonl_runtime import evaluation_row_sort_key
from cfw_predictions import normalize_prediction
from render_cfw_sft import _atomic_text, dump_toml

FULL_SCHEMA = "nvpaw_pool_full_v1"


def _load_manifest(output: pathlib.Path) -> dict:
    manifest = json.loads((output / "pool_full_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FULL_SCHEMA:
        raise ValueError("unsupported full-pool manifest schema")
    for path, digest in manifest["request"]["implementation"].items():
        if sha256_file(pathlib.Path(path)) != digest:
            raise ValueError(f"analysis implementation changed since preparation: {path}")
    return manifest


def _check_chunk(chunk: dict) -> None:
    for field in ("source", "config", "plan"):
        if sha256_file(pathlib.Path(chunk[field])) != chunk[field + "_sha256"]:
            raise ValueError(f"chunk {chunk['index']}: {field} SHA-256 changed")


def _seal_fields(manifest: dict, chunk: dict) -> dict:
    return {"state": "COMPLETE", "index": chunk["index"], "rows": chunk["rows"],
            "checkpoint": manifest["request"]["checkpoint"],
            **{key: chunk[key] for key in ("source_sha256", "config_sha256", "plan_sha256")},
            "predictions_sha256": sha256_file(pathlib.Path(chunk["predictions"]))}


def chunk_complete(manifest: dict, chunk: dict) -> bool:
    try:
        marker = json.loads(pathlib.Path(chunk["completion"]).read_text(encoding="utf-8"))
        return marker == _seal_fields(manifest, chunk)
    except (OSError, ValueError, KeyError):
        return False


def seal_chunk(output: pathlib.Path, index: int) -> None:
    manifest = _load_manifest(output)
    if not 0 <= index < len(manifest["chunks"]):
        raise ValueError("chunk index out of range")
    chunk = manifest["chunks"][index]
    _check_chunk(chunk)
    sources = list(read_rows(pathlib.Path(chunk["source"])))
    predictions = list(read_rows(pathlib.Path(chunk["predictions"])))
    by_id = {}
    seen: set[str] = set()
    for prediction in predictions:
        by_id[scorer._row_id(prediction, seen, "chunk predictions")] = prediction
    if len(sources) != chunk["rows"] or set(by_id) != {row["id"] for row in sources}:
        raise ValueError(f"chunk {index}: prediction coverage mismatch")
    for source in sources:
        prediction = by_id[source["id"]]
        expected = normalize_prediction(source, prediction)
        if prediction != expected:
            raise ValueError(f"chunk {index}: non-canonical prediction metadata for {source['id']}")
    _atomic_text(pathlib.Path(chunk["completion"]), scorer._json(_seal_fields(manifest, chunk)))


def _array_script(output: pathlib.Path, manifest: dict, args, pending: list[int]) -> str:
    if not pending:
        return "#!/usr/bin/env bash\n# All chunks have checksum-verified COMPLETE evidence; nothing to submit.\nexit 0\n"
    indices = f"0-{len(pending)-1}" if pending == list(range(len(pending))) and len(pending) > 1 else ",".join(map(str, pending))
    # This is the standard single-node Pyxis envelope; task semantics remain in
    # the existing action descriptor and evaluate runtime, not in shell code.
    q = shlex.quote
    lines = ["#!/usr/bin/env bash", "# ANALYSIS ONLY: operator must preflight and open a job record before submission.",
             "#SBATCH --job-name=pool-residual", "#SBATCH --nodes=1", "#SBATCH --ntasks-per-node=1",
             "#SBATCH --gres=gpu:8", "#SBATCH --exclusive", f"#SBATCH --cpus-per-task={args.cpus_per_task}",
             "#SBATCH --time=03:50:00", f"#SBATCH --partition={args.partition}",
             f"#SBATCH --constraint={args.constraint}",
             f"#SBATCH --array={indices}%{args.concurrency}", "#SBATCH --no-requeue",
             f"#SBATCH --output={q(str(output / 'array-%A_%a.out'))}",
             f"#SBATCH --error={q(str(output / 'array-%A_%a.err'))}"]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines += ["set -euo pipefail", ': "${TAO_JOB_ID:?Operator must open the TAO job record before submission}"',
              "ulimit -n 65536", "export HF_HUB_OFFLINE=1", "export TRANSFORMERS_OFFLINE=1",
              "unset WANDB_API_KEY", "srun --ntasks=1 --kill-on-bad-exit=1 --no-container-mount-home "
              f"--container-image={q(str(args.sqsh.expanduser().resolve()))} "
              f"--container-mounts={q(args.container_mounts)} --container-workdir={q(str(output))} "
              f"/workspace/.venv/bin/python {q(str(pathlib.Path(scorer.__file__).resolve()))} "
              f'run-chunk --output-dir {q(str(output))} --chunk "${{SLURM_ARRAY_TASK_ID}}"', ""]
    return "\n".join(lines)


def prepare_full(args) -> None:
    if args.chunks < 1 or args.chunks > 1024 or not 1 <= args.concurrency <= 8 or args.cpus_per_task < 1:
        raise ValueError("chunks must be 1..1024, concurrency 1..8, and CPUs positive")
    if (not re.fullmatch(r"[A-Za-z0-9_.,-]+", args.partition)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.constraint)
            or args.account and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.account)):
        raise ValueError("invalid partition/account/constraint directive")
    if not args.sqsh or not args.container_mounts:
        raise ValueError("full mode requires a pre-staged --sqsh and explicit --container-mounts")
    with args.sqsh.open("rb") as stream:
        if stream.read(4) != b"hsqs":
            raise ValueError("pre-staged SQSH is missing SquashFS magic; no registry fallback")
    output = args.output_dir.expanduser().resolve()
    source = args.input.expanduser().resolve()
    for value in (str(output), str(args.sqsh), args.container_mounts):
        if "\n" in value or "\r" in value:
            raise ValueError("paths and mounts cannot contain line breaks")
    scripts = pathlib.Path(__file__).resolve().parent
    request = {"input": str(source), "input_sha256": sha256_file(source),
               "checkpoint": str(args.model_path.expanduser().resolve()),
               "media_root": str(args.media_root.expanduser().resolve()), "image": args.image,
               "batch_size": args.batch_size, "chunks": args.chunks, "seed": args.seed,
               "implementation": {str(scripts / name): sha256_file(scripts / name) for name in
                   ("score_pool_residual.py", "pool_residual_full.py", "cfw_jsonl_runtime.py",
                    "cfw_predictions.py", "merge_cfw_prediction_shards.py", "render_cfw_evaluate.py",
                    "build_target_profile.py", "nvpaw_annotations.py", "validate_sharegpt.py",
                    "analyze_gaps.py", "cfw_action_plan.py", "render_cfw_sft.py")}}
    manifest_path = output / "pool_full_manifest.json"
    if manifest_path.exists():
        manifest = _load_manifest(output)
        if manifest["request"] != request:
            raise ValueError("full-pool preparation request changed; use a fresh output directory")
        for chunk in manifest["chunks"]:
            _check_chunk(chunk)
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("full-pool preparation requires a fresh directory or its sealed manifest")
        seen: set[str] = set()
        total = skipped = 0
        chunk_info = []
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
            staging = pathlib.Path(temporary)
            with ExitStack() as stack:
                streams = [stack.enter_context((staging / f"{i}.jsonl").open("x", encoding="utf-8")) for i in range(args.chunks)]
                for record in read_rows(source):
                    scorer._row_id(record, seen, "full mining pool")
                    task = str(record.get("task_type"))
                    if "segmentation" in task.casefold():
                        skipped += 1
                        continue
                    if task not in scorer.SCORABLE_TASKS:
                        raise ValueError(f"full pool has unsupported non-segmentation task {task!r}; cannot silently drop rows")
                    scorer.residual_features(record)
                    streams[total % args.chunks].write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
            if not total:
                raise ValueError("full pool supplies no scorable rows")
            for index in range(args.chunks):
                folder = output / "chunks" / f"{index:04d}"
                rows = sorted(read_rows(staging / f"{index}.jsonl"), key=evaluation_row_sort_key)
                chunk_source = folder / "source.jsonl"
                config, plan = scorer.evaluation_plan(chunk_source, folder, checkpoint=request["checkpoint"],
                    media_root=request["media_root"], image=args.image, batch_size=args.batch_size)
                plan["resources"].update(time="03:50:00", partition=args.partition)
                files = {chunk_source: "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                         folder / "evaluate_pool.toml": dump_toml(config), folder / "plan.json": scorer._json(plan)}
                write_outputs(files)
                chunk_info.append({"index": index, "rows": len(rows), "source": str(chunk_source),
                    "config": str(folder / "evaluate_pool.toml"), "plan": str(folder / "plan.json"),
                    "predictions": str(folder / "predictions.jsonl"), "completion": str(folder / "COMPLETE.json"),
                    "source_sha256": sha256_file(chunk_source), "config_sha256": sha256_file(folder / "evaluate_pool.toml"),
                    "plan_sha256": sha256_file(folder / "plan.json")})
        if sha256_file(source) != request["input_sha256"]:
            raise ValueError("mining input changed during preparation")
        manifest = {"schema_version": FULL_SCHEMA, "analysis_only": True, "request": request,
                    "rows": total, "rows_seen": len(seen), "skipped_segmentation_rows": skipped,
                    "chunks": chunk_info, "created_at": scorer._utc_now(),
                    "chunk_assignment": "source-order round robin; task/length sorted inside each chunk"}
        write_outputs({manifest_path: scorer._json(manifest)})
        for chunk in chunk_info:
            if chunk["rows"] == 0:
                write_outputs({pathlib.Path(chunk["predictions"]): ""})
                seal_chunk(output, chunk["index"])
    pending = [chunk["index"] for chunk in manifest["chunks"] if not chunk_complete(manifest, chunk)]
    # Re-render only this scheduling artifact; never overwrite chunk inputs or
    # finished predictions. Stale/partial results remain until a whole rerun.
    _atomic_text(output / "pool_array.sbatch", _array_script(output, manifest, args, pending))
    print(scorer._json({"rows": manifest["rows"], "pending_chunks": pending, "array": str(output / "pool_array.sbatch")}), flush=True)


def run_chunk(args) -> None:
    output = args.output_dir.expanduser().resolve()
    manifest = _load_manifest(output)
    if not 0 <= args.chunk < len(manifest["chunks"]):
        raise ValueError("chunk index out of range")
    chunk = manifest["chunks"][args.chunk]
    _check_chunk(chunk)
    with pathlib.Path(chunk["source"]).with_name(".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if chunk_complete(manifest, chunk):
            print(f"chunk {args.chunk}: COMPLETE, skipping", flush=True)
            return
        plan = json.loads(pathlib.Path(chunk["plan"]).read_text(encoding="utf-8"))
        for stage in ("command", "merge_command"):
            print(f"{scorer._utc_now()} chunk {args.chunk}: {stage} start", flush=True)
            result = subprocess.run(plan[stage], check=False)
            if result.returncode:
                raise SystemExit(result.returncode)
            print(f"{scorer._utc_now()} chunk {args.chunk}: {stage} end", flush=True)
        seal_chunk(output, args.chunk)


def merge_full(args) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = args.output_dir.expanduser().resolve()
    manifest = _load_manifest(output)
    for chunk in manifest["chunks"]:
        _check_chunk(chunk)
        if not chunk_complete(manifest, chunk):
            raise ValueError(f"chunk {chunk['index']} is incomplete; full-pool merge requires every chunk")
    names = ("predictions.jsonl", "pool_residual.parquet", "residual_profile.json", "RESIDUAL_REPORT.md")
    for name in names:
        if (output / name).exists():
            raise ValueError(f"refusing to overwrite existing output: {output / name}")
    evaluator_path = args.evaluator.expanduser().resolve()
    evaluator_sha = sha256_file(evaluator_path)
    if args.evaluator_sha256 and evaluator_sha != args.evaluator_sha256:
        raise ValueError("evaluator SHA-256 does not match --evaluator-sha256")
    evaluator = scorer._load_evaluator(evaluator_path)
    timestamp = scorer._utc_now()
    checkpoint = manifest["request"]["checkpoint"]
    provenance = {"schema_version": scorer.RESIDUAL_SCHEMA, "analysis_only": True, "mode": "full",
        "checkpoint": checkpoint, "scored_at": timestamp, "rows_scored": manifest["rows"],
        "full_manifest_sha256": sha256_file(output / "pool_full_manifest.json"),
        "evaluator": str(evaluator_path), "evaluator_sha256": evaluator_sha,
        "skipped_segmentation_rows": manifest["skipped_segmentation_rows"],
        "row_score": "canonical classification 1/0; detection TP/(TP+FP+FN), correct empty=1, parse failure=0; not F1",
        "matching": "authoritative threshold-aware Hungarian IoU >0.5; misaligned adds matches at >0.3; xyxy scale=1",
        "counting_extension": scorer.COUNT_CONTRACT}
    patterns: dict[tuple, Counter] = defaultdict(Counter)
    seen: set[str] = set()
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=output) as temporary:
        staging = pathlib.Path(temporary)
        with (staging / "scores.jsonl").open("x", encoding="utf-8") as scores, (staging / "predictions.jsonl").open("x", encoding="utf-8") as predictions:
            def failures():
                for chunk in manifest["chunks"]:
                    sources = list(read_rows(pathlib.Path(chunk["source"])))
                    native = list(read_rows(pathlib.Path(chunk["predictions"])))
                    if not sources:
                        continue
                    rows = scorer.score_records(evaluator, sources, native, checkpoint=checkpoint,
                                                scored_at=timestamp, flag_patterns=False)
                    for record in native:
                        predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
                    for source, row in zip(sources, rows, strict=True):
                        scorer._row_id(row, seen, "whole-pool residual")
                        scores.write(json.dumps(row, ensure_ascii=False) + "\n")
                        if row["gt_label_pattern"] is not None:
                            key = (row["task_type"], row["dataset"], row["gt_label_pattern"])
                            patterns[key][row["predicted_labels"]] += 1
                        if row["is_residual"]:
                            yield source
            profile = scorer.build_residual_profile(failures())
        if len(seen) != manifest["rows"]:
            raise ValueError("whole-pool residual coverage count mismatch")
        provenance["residual_rows"] = profile["rows"]
        provenance["predictions_sha256"] = sha256_file(staging / "predictions.jsonl")
        schema = scorer.parquet_schema(provenance)
        with pq.ParquetWriter(staging / "pool_residual.parquet", schema) as writer:
            def final_rows():
                batch = []
                for row in read_rows(staging / "scores.jsonl"):
                    if row["gt_label_pattern"] is not None and row["error_type"] == "wrong_class" and not row["review_label_noise"]:
                        counts = patterns[(row["task_type"], row["dataset"], row["gt_label_pattern"])]
                        total = sum(counts.values())
                        count = counts[row["predicted_labels"]]
                        if total >= 20 and count / total >= 0.95:
                            row["review_label_noise"] = True
                            row["review_reason"] = f"human review: repeated label-pattern disagreement ({count}/{total}); not model confidence or proof of label noise"
                    batch.append(row)
                    if len(batch) == 8192:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                    yield row
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            report = scorer.residual_report(final_rows(), provenance)
        profile["scoring"] = provenance
        # Exclusive destinations prevent concurrent merges from overwriting an
        # already published artifact. All computation completes before publish.
        for name in ("predictions.jsonl", "pool_residual.parquet"):
            with (staging / name).open("rb") as source, (output / name).open("xb") as target:
                shutil.copyfileobj(source, target)
        write_outputs({output / "residual_profile.json": scorer._json(profile), output / "RESIDUAL_REPORT.md": report})
