# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic host adapters for the local IAA generation stage."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml

from iaa_deft.sdg import (
    QUERY_LEVELS,
    accepted_augmentations,
    atomic_json,
    build_component_command,
    normalize_generated_pairs,
    residual_attribute_assignments,
    sha256,
    validate_config,
    verification_passed,
)


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    return validate_config(payload)


def _status_path(root: pathlib.Path, operation: str) -> pathlib.Path:
    return root / "status" / f"{operation}.host.status.json"


def _run_once(
    root: pathlib.Path, operation: str, inputs: list[pathlib.Path], outputs: list[pathlib.Path],
    handler,
) -> dict[str, Any]:
    status_path = _status_path(root, operation)
    if status_path.is_file():
        existing = json.loads(status_path.read_text())
        current_inputs = [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in inputs if path.is_file()
        ]
        if existing.get("status") == "success" and all(path.exists() for path in outputs):
            if existing.get("inputs") != current_inputs:
                raise ValueError(f"{operation} inputs changed after successful operation")
            if any(path.is_file() and path.stat().st_size == 0 for path in outputs):
                raise ValueError(f"{operation} has an empty committed output")
            return existing
        if int(existing.get("attempt", 0)) >= 2:
            raise ValueError(f"{operation} attempt budget exhausted; inspect {status_path}")
        attempt = int(existing.get("attempt", 0)) + 1
    else:
        attempt = 1
    started_ns = time.time_ns()
    try:
        result = handler()
        missing = [str(path) for path in outputs if not path.exists()]
        if missing:
            raise ValueError("operation did not produce: " + ", ".join(missing))
        status = {
            "schema_version": "1", "name": operation, "status": "success", "attempt": attempt,
            "started_ns": started_ns, "finished_ns": time.time_ns(),
            "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in inputs if path.is_file()],
            "fresh_outputs": [str(path.resolve()) for path in outputs], "result": result,
        }
    except Exception as exc:
        status = {
            "schema_version": "1", "name": operation, "status": "error", "attempt": attempt,
            "started_ns": started_ns, "finished_ns": time.time_ns(), "error": str(exc),
        }
        atomic_json(status_path, status)
        raise
    atomic_json(status_path, status)
    return status


def prepare_inputs(args: argparse.Namespace, config: dict[str, Any]) -> dict:
    output = args.output_root.resolve()
    plan_path = output / "sdg_plan.json"
    source_root = output / "source_ids"
    eval_names = {
        pathlib.Path(line.strip()).name
        for line in args.eval_list.read_text().splitlines() if line.strip()
    }

    def handler() -> dict:
        payload = json.loads(args.mined_pairs.read_text())
        if not isinstance(payload, list) or not payload:
            raise ValueError("mined pairs must be a non-empty list")
        unique: dict[str, dict] = {}
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("mined pair records must be objects")
            name = pathlib.Path(str(row.get("unique_name", ""))).name
            if not name or name in eval_names:
                if name in eval_names:
                    raise ValueError(f"evaluation image selected for generation: {name}")
                continue
            person = str(row.get("person_key") or row.get("person_id") or name)
            unique.setdefault(person, row)
        budget = min(
            len(unique),
            config["generation"]["max_samples_per_iteration"],
            max(1, int(len(unique) * float(config["generation"]["scale_factor"]))),
        )
        selected = list(unique.values())[:budget]
        if not selected:
            raise ValueError("generation plan selected zero images")
        import pandas as pd

        gaps = pd.read_parquet(args.gaps_parquet)
        if "image_attr_vector" not in gaps.columns or gaps.empty:
            raise ValueError("gaps parquet must contain non-empty image_attr_vector evidence")
        mined_vectors = [row.get("image_attr_values") for row in selected]
        if any(value is None for value in mined_vectors):
            raise ValueError("selected mined pairs lack image_attr_values")
        vocab_payload = json.loads(args.attribute_vocab.read_text())
        assignments, distribution = residual_attribute_assignments(
            gaps["image_attr_vector"], mined_vectors, vocab_payload,
            len(selected), float(config["generation"]["scale_factor"]),
        )
        records = []
        seen_keys: set[str] = set()
        for index, (row, target_attributes) in enumerate(zip(selected, assignments)):
            dataset_root = args.dataset_root.resolve()
            source = (dataset_root / str(row.get("image_path", ""))).resolve()
            try:
                source.relative_to(dataset_root)
            except ValueError as exc:
                raise ValueError(f"selected source escapes dataset root: {source}") from exc
            grouped_source = source.is_file() and source.parent != dataset_root / "images"
            if not source.is_file():
                fallback = dataset_root / "images" / pathlib.Path(str(row["unique_name"])).name
                source = fallback if fallback.is_file() else source
                grouped_source = False
            if not source.is_file() or source.stat().st_size == 0:
                raise ValueError(f"selected source image is missing: {source}")
            key = str(row.get("person_key") or row.get("person_id") or f"sample_{index:06d}")
            key = "".join(char if char.isalnum() or char in "_.-" else "-" for char in key).strip("-")
            if not key:
                key = f"sample_{index:06d}"
            if key in seen_keys:
                raise ValueError(f"two selected person keys normalize to the same path: {key}")
            seen_keys.add(key)
            destination_dir = source_root / key
            destination_dir.mkdir(parents=True, exist_ok=True)
            source_views = (
                sorted(
                    path for path in source.parent.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                if grouped_source
                else [source]
            )
            if not source_views:
                source_views = [source]
            staged_paths = []
            source_hashes = []
            for view in source_views:
                destination = destination_dir / view.name
                if destination.is_file():
                    if sha256(destination) != sha256(view):
                        raise ValueError(f"staged source changed during recovery: {destination}")
                else:
                    shutil.copyfile(view, destination)
                staged_paths.append(str(destination.resolve()))
                source_hashes.append({"path": str(view.resolve()), "sha256": sha256(view)})
            records.append({
                "source_key": key, "source_path": str(source.resolve()),
                "source_views": source_hashes, "staged_paths": staged_paths,
                "mined_unique_name": row["unique_name"],
                "target_attributes": target_attributes,
            })
        atomic_json(plan_path, {
            "schema_version": "1", "budget": budget, "selected": records,
            "eval_excluded": True, "source_pairs_sha256": sha256(args.mined_pairs),
            "gaps_sha256": sha256(args.gaps_parquet),
            "attribute_vocab_sha256": sha256(args.attribute_vocab),
            "residual_distribution": distribution,
        })
        return {"selected": len(records)}

    return _run_once(
        output, "sdg-prepare",
        [args.mined_pairs, args.gaps_parquet, args.attribute_vocab, args.eval_list],
        [plan_path], handler,
    )


def validate_augmentation(args: argparse.Namespace, config: dict[str, Any]) -> dict:
    output = args.output_root.resolve()
    accepted_path = output / "accepted_manifest.json"

    def handler() -> dict:
        accepted, rejected = accepted_augmentations(
            args.augmentation_root.resolve(), config["generation"]["verification_max_attempts"]
        )
        if not accepted:
            raise ValueError("no augmentation passed verification")
        atomic_json(accepted_path, {
            "schema_version": "1", "max_attempts": config["generation"]["verification_max_attempts"],
            "accepted": accepted, "rejected": rejected,
        })
        return {"accepted": len(accepted), "rejected": len(rejected)}

    return _run_once(output, "sdg-validate-augmentation", [], [accepted_path], handler)


def validate_labels(args: argparse.Namespace, config: dict[str, Any]) -> dict:
    output = args.output_root.resolve()
    report_path = output / "label_validation.json"

    def handler() -> dict:
        accepted = json.loads(args.accepted_manifest.read_text()).get("accepted", [])
        if not accepted:
            raise ValueError("accepted manifest is empty")
        checked = []
        for record in accepted:
            path = args.labels_root / record["source_key"] / "task" / "open_qa.json"
            payload = json.loads(path.read_text())
            if isinstance(payload, dict) and isinstance(payload.get("items"), list):
                items = payload["items"]
                if len(items) != 9 or any(
                    not isinstance(item, dict) or not str(item.get("answer", "")).strip()
                    for item in items
                ):
                    raise ValueError(f"{path} must contain nine non-empty open-QA items")
                checked.append({"source_key": record["source_key"], "path": str(path.resolve()), "sha256": sha256(path)})
                continue
            root = payload.get("queries", payload.get("open_qa", payload)) if isinstance(payload, dict) else None
            if not isinstance(root, dict):
                raise ValueError(f"invalid open-QA root: {path}")
            for level in QUERY_LEVELS:
                rows = root.get(level)
                if rows is None:
                    rows = [value for key, value in root.items() if str(key).startswith(level + "_")]
                if not isinstance(rows, list) or len(rows) != 3:
                    raise ValueError(f"{path} must contain exactly three {level} entries")
            checked.append({"source_key": record["source_key"], "path": str(path.resolve()), "sha256": sha256(path)})
        atomic_json(report_path, {"schema_version": "1", "validated": checked})
        return {"validated": len(checked)}

    return _run_once(output, "sdg-validate-labeling", [args.accepted_manifest], [report_path], handler)


def normalize(args: argparse.Namespace, config: dict[str, Any]) -> dict:
    output = args.output_root.resolve()
    dataset = output / "dataset"
    manifest = dataset / "sdg_manifest.json"
    eval_names = {pathlib.Path(line.strip()).name for line in args.eval_list.read_text().splitlines() if line.strip()}

    def handler() -> dict:
        path = normalize_generated_pairs(
            args.accepted_manifest.resolve(), args.labels_root.resolve(), dataset,
            args.attribute_vocab.resolve(), eval_names, config["generation"]["caption_policy"],
        )
        payload = json.loads(path.read_text())
        return {"num_source_images": payload["num_source_images"], "num_pairs": payload["num_pairs"]}

    return _run_once(
        output, "sdg-normalize", [args.accepted_manifest, args.attribute_vocab, args.eval_list],
        [manifest, dataset / "sdg_pairs.json", dataset / "sdg_image_list.txt"], handler,
    )


def component(args: argparse.Namespace, config: dict[str, Any]) -> dict:
    argv = build_component_command(
        config, args.action, input_root=args.input_root, output_root=args.output_root,
        source_key=args.source_key or "", attempt=args.attempt,
    )
    if args.dry_run:
        return {"command": argv}
    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        raise ValueError(f"component {args.action} exited {completed.returncode}")
    return {"command": argv, "returncode": completed.returncode}


def _component_call(
    config: dict[str, Any], action: str, input_root: pathlib.Path,
    output_root: pathlib.Path, log_path: pathlib.Path, source_key: str = "",
    attempt: int = 1,
    target_attributes: dict[str, str] | None = None,
) -> None:
    argv = build_component_command(
        config, action, input_root=input_root, output_root=output_root,
        source_key=source_key, attempt=attempt,
        target_attributes=target_attributes,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise ValueError(f"{action} exited {completed.returncode}; inspect {log_path}")


def execute(args: argparse.Namespace, config: dict[str, Any]) -> dict:
    """Run the bounded local component sequence with per-sample resume evidence."""
    output = args.output_root.resolve()
    plan = json.loads((output / "sdg_plan.json").read_text())
    selected = plan.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("sdg_plan.json has no selected inputs")
    progress_path = output / "sdg_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {
        "schema_version": "1", "preprocessed": False, "augmentation": {},
        "split": False, "labeling": {}, "command_attempts": {},
    }
    progress.setdefault("command_attempts", {})

    def call(
        action: str, input_root: pathlib.Path, log_path: pathlib.Path,
        source_key: str = "", attempt: int = 1,
        target_attributes: dict[str, str] | None = None,
    ) -> None:
        identity = f"{action}:{source_key or 'batch'}:{attempt}"
        used = int(progress["command_attempts"].get(identity, 0))
        if used >= 2:
            raise ValueError(f"component retry budget exhausted for {identity}; inspect {log_path}")
        progress["command_attempts"][identity] = used + 1
        atomic_json(progress_path, progress)
        _component_call(
            config, action, input_root, output, log_path,
            source_key, attempt, target_attributes,
        )
    logs = output / "logs"
    source_root = output / "source_ids"
    if not progress["preprocessed"]:
        call("preprocess", source_root, logs / "preprocess.log")
        panes = sorted((output / "panes").glob("*.jpg"))
        if len(panes) != len(selected):
            raise ValueError(f"preprocess produced {len(panes)} panes for {len(selected)} selected inputs")
        progress["preprocessed"] = True
        atomic_json(progress_path, progress)
    elif len(list((output / "panes").glob("*.jpg"))) != len(selected):
        raise ValueError("preprocessing is journaled complete but pane evidence is incomplete")

    accepted_root = output / "accepted"
    maximum = config["generation"]["verification_max_attempts"]
    for position, record in enumerate(selected):
        key = record["source_key"]
        outcome = progress["augmentation"].get(key)
        if isinstance(outcome, dict) and outcome.get("status") == "accepted":
            attempt_root = output / "augmentation" / key / f"attempt_{outcome['attempt']}"
            metadata_path = attempt_root / "output_metadata.json"
            image_path = attempt_root / "output.jpg"
            metadata_payload = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            if not image_path.is_file() or not verification_passed(metadata_payload):
                raise ValueError(f"accepted progress evidence is missing or invalid for {key}")
            destination = accepted_root / key / "aug_0"
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("output.jpg", "output.txt", "output_metadata.json"):
                source = attempt_root / name
                if source.is_file() and not (destination / name).is_file():
                    shutil.copyfile(source, destination / name)
            continue
        start_attempt = int(outcome.get("attempt", 0)) + 1 if isinstance(outcome, dict) else 1
        for attempt in range(start_attempt, maximum + 1):
            call(
                "augment", source_root, logs / f"augment-{key}-attempt-{attempt}.log",
                key, attempt, record.get("target_attributes") or {},
            )
            attempt_root = output / "augmentation" / key / f"attempt_{attempt}"
            metadata_path = attempt_root / "output_metadata.json"
            image_path = attempt_root / "output.jpg"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            accepted = image_path.is_file() and image_path.stat().st_size > 0 and verification_passed(metadata)
            progress["augmentation"][key] = {
                "attempt": attempt, "status": "accepted" if accepted else "rejected",
                "metadata": str(metadata_path.resolve()),
            }
            atomic_json(progress_path, progress)
            if accepted:
                destination = accepted_root / key / "aug_0"
                destination.mkdir(parents=True, exist_ok=True)
                for name in ("output.jpg", "output.txt", "output_metadata.json"):
                    source = attempt_root / name
                    if source.is_file():
                        shutil.copyfile(source, destination / name)
                break
        if progress["augmentation"][key]["status"] != "accepted":
            # Exhaustion is a finite sample rejection, not an unbounded workflow retry.
            continue
        if position == 0:
            # First accepted item is the mandatory smoke test. Validation above
            # is the exit condition before the remaining batch can proceed.
            atomic_json(output / "augmentation_smoke.json", progress["augmentation"][key])
    accepted_count = sum(item.get("status") == "accepted" for item in progress["augmentation"].values())
    if accepted_count == 0:
        raise ValueError("augmentation retry bound exhausted for every selected source")

    if not progress["split"]:
        call("split", source_root, logs / "split.log")
        crop_root = output / "augmented_dataset" / "augmented_imgs"
        crops = sorted(path for path in crop_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not crops:
            raise ValueError("accepted pane splitting produced no per-view images")
        progress["split"] = True
        atomic_json(progress_path, progress)
    elif not any(
        path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        for path in (output / "augmented_dataset" / "augmented_imgs").rglob("*")
    ):
        raise ValueError("split is journaled complete but per-view crops are missing")

    label_inputs = output / "label_inputs"
    label_inputs.mkdir(parents=True, exist_ok=True)
    crop_records = []
    seen_crop_keys: set[str] = set()
    for crop in sorted((output / "augmented_dataset" / "augmented_imgs").rglob("*")):
        if not crop.is_file() or crop.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parent_key = crop.parent.name.rsplit("_aug", 1)[0]
        safe_key = "".join(char if char.isalnum() or char in "_.-" else "-" for char in f"{parent_key}__{crop.stem}")
        if safe_key in seen_crop_keys:
            raise ValueError(f"two generated crops normalize to the same label key: {safe_key}")
        seen_crop_keys.add(safe_key)
        staged = label_inputs / f"{safe_key}.jpg"
        if staged.is_file() and sha256(staged) != sha256(crop):
            raise ValueError(f"staged label input changed during recovery: {staged}")
        if not staged.exists():
            shutil.copyfile(crop, staged)
        metadata = accepted_root / parent_key / "aug_0" / "output_metadata.json"
        if not metadata.is_file():
            matches = sorted(accepted_root.glob(f"{parent_key}*/aug_0/output_metadata.json"))
            if len(matches) != 1:
                raise ValueError(f"cannot bind crop {crop} to verification metadata")
            metadata = matches[0]
        crop_records.append({
            "source_key": safe_key, "attempt": 1, "image": str(crop.resolve()),
            "metadata": str(metadata.resolve()), "metadata_sha256": sha256(metadata),
            "source_unique_name": next(
                (item["mined_unique_name"] for item in selected if item["source_key"] == parent_key),
                parent_key,
            ),
        })
    crop_manifest = output / "accepted_crop_manifest.json"
    atomic_json(crop_manifest, {"schema_version": "1", "accepted": crop_records})

    for position, record in enumerate(crop_records):
        key = record["source_key"]
        if progress["labeling"].get(key) == "accepted":
            qa = output / "labels" / key / "task" / "open_qa.json"
            if not qa.is_file() or qa.stat().st_size == 0:
                raise ValueError(f"labeling is journaled complete but open QA is missing for {key}")
            continue
        call("label", label_inputs, logs / f"label-{key}.log", key)
        qa = output / "labels" / key / "task" / "open_qa.json"
        if not qa.is_file() or qa.stat().st_size == 0:
            raise ValueError(f"labeling produced no task/open_qa.json for {key}")
        progress["labeling"][key] = "accepted"
        atomic_json(progress_path, progress)
        if position == 0:
            shutil.copyfile(qa, output / "auto_label_smoke_open_qa.json")

    args.accepted_manifest = crop_manifest
    args.labels_root = output / "labels"
    validate_labels(args, config)
    result = normalize(args, config)
    atomic_json(output / "sdg_execution_manifest.json", {
        "schema_version": "1", "selected_sources": len(selected),
        "accepted_sources": accepted_count, "rejected_sources": len(selected) - accepted_count,
        "accepted_crops": len(crop_records), "progress": str(progress_path.resolve()),
        "normalized_manifest": str((output / "dataset" / "sdg_manifest.json").resolve()),
    })
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "execute", "validate-augmentation", "validate-labeling", "normalize", "component"))
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--mined-pairs", type=pathlib.Path)
    parser.add_argument("--dataset-root", type=pathlib.Path)
    parser.add_argument("--gaps-parquet", type=pathlib.Path)
    parser.add_argument("--eval-list", type=pathlib.Path)
    parser.add_argument("--augmentation-root", type=pathlib.Path)
    parser.add_argument("--accepted-manifest", type=pathlib.Path)
    parser.add_argument("--labels-root", type=pathlib.Path)
    parser.add_argument("--attribute-vocab", type=pathlib.Path)
    parser.add_argument("--action", choices=("preprocess", "augment", "split", "label"))
    parser.add_argument("--input-root", type=pathlib.Path)
    parser.add_argument("--source-key")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = _load_config(args.config)
        if args.operation == "prepare":
            report = prepare_inputs(args, config)
        elif args.operation == "execute":
            report = execute(args, config)
        elif args.operation == "validate-augmentation":
            report = validate_augmentation(args, config)
        elif args.operation == "validate-labeling":
            report = validate_labels(args, config)
        elif args.operation == "normalize":
            report = normalize(args, config)
        else:
            report = component(args, config)
        print(json.dumps(report, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"run_sdg_stage[{args.operation}]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
