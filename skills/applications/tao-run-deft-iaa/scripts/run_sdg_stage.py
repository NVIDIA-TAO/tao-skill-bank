# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic adapters for the platform-local IAA generation stage."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import yaml

from iaa_deft.sdg import (
    EDITABLE_ATTRIBUTES,
    QUERY_LEVELS,
    accepted_augmentations,
    atomic_json,
    bind_resumable_endpoint_pool,
    build_component_command,
    endpoint_url,
    normalize_generated_pairs,
    residual_attribute_assignments,
    sha256,
    validate_config,
    validate_image_edit_endpoint_pool,
    verification_passed,
)


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    return validate_config(payload)


EXECUTION_PLATFORMS = (
    "host", "docker", "slurm", "brev", "virtualenv", "kubernetes", "airflow",
)


def _status_path(
    root: pathlib.Path, operation: str, execution_platform: str = "host"
) -> pathlib.Path:
    if execution_platform not in EXECUTION_PLATFORMS:
        raise ValueError(f"unsupported SDG execution platform: {execution_platform}")
    return root / "status" / f"{operation}.{execution_platform}.status.json"


def _run_once(
    root: pathlib.Path, operation: str, inputs: list[pathlib.Path], outputs: list[pathlib.Path],
    handler, *, execution_platform: str = "host",
) -> dict[str, Any]:
    status_path = _status_path(root, operation, execution_platform)
    log_path = root / "logs" / f"{operation}.{execution_platform}.log"
    pre_action_path = root / "status" / f"{operation}.{execution_platform}.pre-action.json"

    def success_status(
        *, attempt: int, started_ns: int, finished_ns: int, result: Any,
        pre_action_sha256: str | None = None,
    ) -> dict[str, Any]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"{operation}: completed successfully\n", encoding="utf-8")
        payload = {
            "schema_version": "1", "workflow": "tao-run-deft-iaa",
            "name": operation, "status": "ok", "exit_code": 0,
            "execution_platform": execution_platform,
            "attempt": attempt, "started_ns": started_ns, "finished_ns": finished_ns,
            "finished_at": dt.datetime.fromtimestamp(
                finished_ns / 1_000_000_000, tz=dt.timezone.utc
            ).isoformat(),
            "log_path": str(log_path.resolve()),
            "inputs": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in inputs if path.is_file()
            ],
            "fresh_outputs": [str(path.resolve()) for path in outputs],
            "output_evidence": [
                {
                    "path": str(path.resolve()), "sha256": sha256(path),
                    "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in outputs if path.is_file()
            ],
            "result": result,
        }
        if pre_action_sha256 is not None:
            payload["pre_action"] = {
                "path": str(pre_action_path.resolve()),
                "sha256": pre_action_sha256,
            }
        return payload
    if status_path.is_file():
        existing = json.loads(status_path.read_text())
        current_inputs = [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in inputs if path.is_file()
        ]
        if existing.get("status") in {"success", "ok"} and all(path.exists() for path in outputs):
            if existing.get("inputs") != current_inputs:
                raise ValueError(f"{operation} inputs changed after successful operation")
            if any(path.is_file() and path.stat().st_size == 0 for path in outputs):
                raise ValueError(f"{operation} has an empty committed output")
            if existing.get("status") == "success":
                existing = success_status(
                    attempt=int(existing.get("attempt", 1)),
                    started_ns=int(existing.get("started_ns", time.time_ns())),
                    finished_ns=int(existing.get("finished_ns", time.time_ns())),
                    result=existing.get("result"),
                )
                atomic_json(status_path, existing)
            return existing
        if int(existing.get("attempt", 0)) >= 2:
            raise ValueError(f"{operation} attempt budget exhausted; inspect {status_path}")
        attempt = int(existing.get("attempt", 0)) + 1
    else:
        attempt = 1
    started_ns = time.time_ns()
    pre_action = {
        "schema_version": "1", "workflow": "tao-run-deft-iaa",
        "name": operation, "execution_platform": execution_platform,
        "attempt": attempt, "started_ns": started_ns,
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in inputs if path.is_file()
        ],
        "outputs": [
            {"path": str(path.resolve()), "absent": not path.exists()}
            for path in outputs
        ],
    }
    atomic_json(pre_action_path, pre_action)
    pre_action_sha256 = sha256(pre_action_path)
    try:
        result = handler()
        missing = [str(path) for path in outputs if not path.exists()]
        if missing:
            raise ValueError("operation did not produce: " + ", ".join(missing))
        status = success_status(
            attempt=attempt, started_ns=started_ns, finished_ns=time.time_ns(), result=result,
            pre_action_sha256=pre_action_sha256,
        )
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
        if gaps.empty:
            raise ValueError("gaps parquet must contain non-empty attribute evidence")
        if "image_attr_vector" in gaps.columns:
            weak_vectors = gaps["image_attr_vector"]
        else:
            if args.eval_pairs is None or not args.eval_pairs.is_file():
                raise ValueError(
                    "gaps parquet lacks image_attr_vector; provide the exact eval pairs "
                    "used by gap analysis"
                )
            if "unique_name" not in gaps.columns:
                raise ValueError(
                    "gaps parquet lacks both image_attr_vector and unique_name join evidence"
                )
            eval_payload = json.loads(args.eval_pairs.read_text())
            if not isinstance(eval_payload, list) or not eval_payload:
                raise ValueError("eval pairs must be a non-empty list")
            vectors: dict[str, Any] = {}
            for row in eval_payload:
                if not isinstance(row, dict):
                    raise ValueError("eval pair records must be objects")
                name = pathlib.Path(str(row.get("unique_name", ""))).name
                vector = row.get("image_attr_values")
                if not name or vector is None:
                    raise ValueError("eval pairs lack unique_name or image_attr_values")
                prior = vectors.setdefault(name, vector)
                if prior != vector:
                    raise ValueError(f"eval pairs disagree on image attributes for {name}")
            missing = sorted(
                {pathlib.Path(str(name)).name for name in gaps["unique_name"]}
                - set(vectors)
            )
            if missing:
                raise ValueError(
                    "gap rows lack matching eval-pair attribute evidence: "
                    + ", ".join(missing[:3])
                )
            weak_vectors = [
                vectors[pathlib.Path(str(name)).name] for name in gaps["unique_name"]
            ]
        mined_vectors = [row.get("image_attr_values") for row in selected]
        if any(value is None for value in mined_vectors):
            raise ValueError("selected mined pairs lack image_attr_values")
        vocab_payload = json.loads(args.attribute_vocab.read_text())
        assignments, distribution = residual_attribute_assignments(
            weak_vectors, mined_vectors, vocab_payload,
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
                "source_attribute_values": row["image_attr_values"],
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
        [
            args.mined_pairs,
            args.gaps_parquet,
            args.attribute_vocab,
            args.eval_list,
            *([args.eval_pairs] if args.eval_pairs is not None else []),
        ],
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
        plan_path = output / "sdg_plan.json"
        if not plan_path.is_file():
            raise ValueError("cannot bind accepted augmentations without sdg_plan.json")
        selected = json.loads(plan_path.read_text()).get("selected", [])
        vectors = {
            item.get("source_key"): item.get("source_attribute_values")
            for item in selected if isinstance(item, dict)
        }
        targets = {
            item.get("source_key"): item.get("target_attributes")
            for item in selected if isinstance(item, dict)
        }
        for record in accepted:
            record["source_attribute_values"] = vectors.get(record["source_key"])
            record["target_attributes"] = targets.get(record["source_key"])
            if not isinstance(record["source_attribute_values"], list):
                raise ValueError(
                    f"SDG plan lacks source attribute vector for {record['source_key']}"
                )
        atomic_json(accepted_path, {
            "schema_version": "1", "max_attempts": config["generation"]["verification_max_attempts"],
            "accepted": accepted, "rejected": rejected,
        })
        return {"accepted": len(accepted), "rejected": len(rejected)}

    return _run_once(
        output, "sdg-validate-augmentation", [], [accepted_path], handler,
        execution_platform=getattr(args, "execution_platform", "host"),
    )


def validate_labels(args: argparse.Namespace, config: dict[str, Any]) -> dict:
    output = args.output_root.resolve()
    report_path = output / "label_validation.json"
    accepted_records = json.loads(args.accepted_manifest.read_text()).get("accepted", [])
    if not isinstance(accepted_records, list) or not accepted_records:
        raise ValueError("accepted manifest is empty")
    binding_path = output / "label_validation_input.json"
    atomic_json(binding_path, {
        "schema_version": "1",
        "source_keys": sorted(str(record.get("source_key", "")) for record in accepted_records),
    })

    def handler() -> dict:
        checked = []
        for record in accepted_records:
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

    # Label validation depends on the accepted key set and label files, not on
    # unrelated crop-manifest fields. Bind that stable projection so adding
    # provenance fields cannot force duplicate model work during resume. Keep
    # the canonical name for a fresh run; use a versioned record only when a
    # legacy canonical checkpoint already exists.
    execution_platform = getattr(args, "execution_platform", "host")
    canonical_status = _status_path(output, "sdg-validate-labeling", execution_platform)
    operation = "sdg-validate-labeling"
    if canonical_status.is_file():
        prior = json.loads(canonical_status.read_text())
        current = [{"path": str(binding_path.resolve()), "sha256": sha256(binding_path)}]
        if prior.get("inputs") != current:
            operation = "sdg-validate-labeling-v3"
    return _run_once(
        output, operation, [binding_path], [report_path], handler,
        execution_platform=execution_platform,
    )


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

    execution_platform = getattr(args, "execution_platform", "host")
    canonical_status = _status_path(output, "sdg-normalize", execution_platform)
    operation = "sdg-normalize"
    if canonical_status.is_file():
        prior = json.loads(canonical_status.read_text())
        current = [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (args.accepted_manifest, args.attribute_vocab, args.eval_list)
        ]
        if prior.get("inputs") != current:
            operation = "sdg-normalize-v2"
    return _run_once(
        output, operation, [args.accepted_manifest, args.attribute_vocab, args.eval_list],
        [manifest, dataset / "sdg_pairs.json", dataset / "sdg_image_list.txt"], handler,
        execution_platform=execution_platform,
    )


def component(args: argparse.Namespace, config: dict[str, Any]) -> dict:
    argv = build_component_command(
        config, args.action, input_root=args.input_root, output_root=args.output_root,
        source_key=args.source_key or "", attempt=args.attempt,
        image_edit_url=args.image_edit_url,
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
    component_executor: pathlib.Path | None = None,
    component_executor_request: pathlib.Path | None = None,
    component_executor_job_id: str | None = None,
    image_edit_url: str | None = None,
    image_edit_endpoint_id: str | None = None,
) -> None:
    if component_executor is None:
        argv = build_component_command(
            config, action, input_root=input_root, output_root=output_root,
            source_key=source_key, attempt=attempt,
            target_attributes=target_attributes,
            image_edit_url=image_edit_url,
        )
    else:
        executor_lexical = component_executor.expanduser().absolute()
        request_lexical = (
            component_executor_request.expanduser().absolute()
            if component_executor_request is not None else None
        )
        executor = executor_lexical.resolve(strict=True)
        request = request_lexical.resolve(strict=True) if request_lexical is not None else None
        if executor != executor_lexical or not executor.is_file():
            raise ValueError("component executor must be a safe regular file")
        if request is None or request != request_lexical or not request.is_file():
            raise ValueError("component executor request must be a safe regular file")
        if (
            not isinstance(component_executor_job_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", component_executor_job_id)
        ):
            raise ValueError("component executor job ID is invalid")
        argv = (
            ([sys.executable, str(executor)] if executor.suffix == ".py" else [str(executor)])
            + [
                "component", "--request", str(request),
                "--job-id", component_executor_job_id,
                "--action", action,
                "--input-root", str(input_root.resolve()),
                "--output-root", str(output_root.resolve()),
                "--attempt", str(attempt),
            ]
        )
        if source_key:
            argv += ["--source-key", source_key]
        if target_attributes:
            argv += [
                "--target-attributes-json",
                json.dumps(target_attributes, sort_keys=True, separators=(",", ":")),
            ]
        if image_edit_url is not None:
            argv += ["--image-edit-url", image_edit_url]
        if image_edit_endpoint_id is not None:
            argv += ["--image-edit-endpoint-id", image_edit_endpoint_id]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise ValueError(f"{action} exited {completed.returncode}; inspect {log_path}")


class _ImageEditEndpointPool:
    """Bounded slot pool with run-local endpoint quarantine."""

    def __init__(self, config: dict[str, Any], entries: list[dict[str, Any]]):
        self.entries = entries
        self.active = [0 for _ in self.entries]
        self.healthy = [True for _ in self.entries]
        self.condition = threading.Condition()
        self.timeout_s = int(config["endpoints"]["request_timeout_s"])

    @property
    def capacity(self) -> int:
        return sum(int(item["capacity"]) for item in self.entries)

    def acquire(self, preferred: int) -> tuple[int, str]:
        deadline = time.monotonic() + self.timeout_s
        with self.condition:
            while True:
                if not any(self.healthy):
                    raise ValueError("all image-edit endpoints are quarantined")
                for offset in range(len(self.entries)):
                    index = (preferred + offset) % len(self.entries)
                    if (
                        self.healthy[index]
                        and self.active[index] < int(self.entries[index]["capacity"])
                    ):
                        self.active[index] += 1
                        return index, str(self.entries[index]["url"])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("timed out waiting for a free image-edit endpoint slot")
                self.condition.wait(remaining)

    def release(self, index: int, *, quarantine: bool = False) -> None:
        with self.condition:
            if self.active[index] < 1:
                raise RuntimeError("image-edit endpoint slot was released twice")
            self.active[index] -= 1
            if quarantine:
                self.healthy[index] = False
            self.condition.notify_all()


def _runtime_image_edit_endpoint_pool(
    config: dict[str, Any], manifest_path: pathlib.Path | None,
    execution_platform: str = "host",
) -> list[dict[str, Any]]:
    if manifest_path is not None:
        lexical = manifest_path.expanduser().absolute()
        resolved = lexical.resolve(strict=True)
        if resolved != lexical or not resolved.is_file():
            raise ValueError("image-edit endpoint pool must be a safe regular file")
        binding = validate_image_edit_endpoint_pool(json.loads(resolved.read_text()))
        if binding["platform"] != execution_platform:
            raise ValueError("image-edit endpoint pool platform does not match execution platform")
        expected_model = config["models"]["image_edit"]
        if binding["model"] != {
            "id": expected_model["id"], "revision": expected_model["revision"],
        }:
            raise ValueError("image-edit endpoint pool model does not match immutable config")
        return binding["endpoints"]
    url = endpoint_url(config, "image_edit")
    if config["endpoints"]["ownership"] == "managed":
        # The current local server exposes replicated workers behind one URL.
        # Model each explicit GPU as a bounded slot without claiming a distinct
        # routable per-GPU endpoint; runtime manifests use distinct URLs/IDs.
        return [
            {
                "id": f"managed-image-edit-slot-{gpu_id}", "url": url,
                "capacity": 1, "gpu_identity": str(gpu_id),
                "owner": {
                    "native_id": f"managed-local-gpu-{gpu_id}",
                    "name": "managed-image-edit",
                },
            }
            for gpu_id in config["endpoints"]["gpu_ids"]["image_edit"]
        ]
    return [{
        "id": "external-image-edit-0", "url": url, "capacity": 1,
        "gpu_identity": "user-managed",
        "owner": {"native_id": "user-managed", "name": "external-image-edit"},
    }]


def _augmentation_max_in_flight(
    config: dict[str, Any], remaining: int,
    entries: list[dict[str, Any]] | None = None,
) -> int:
    """Resolve bounded image-edit replica use without assuming external capacity."""
    if remaining < 1:
        return 0
    pool = entries if entries is not None else _runtime_image_edit_endpoint_pool(config, None)
    worker_count = len(pool)
    configured = config["generation"].get("max_in_flight", worker_count)
    if not isinstance(configured, int) or isinstance(configured, bool) or configured < 1:
        raise ValueError("generation.max_in_flight must be an integer >= 1")
    if configured > worker_count:
        raise ValueError("generation.max_in_flight exceeds the image-edit worker count")
    return min(configured, worker_count, remaining)


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
    progress.setdefault("endpoint_attempts", {})
    progress_lock = threading.Lock()
    endpoint_entries = _runtime_image_edit_endpoint_pool(
        config, getattr(args, "image_edit_endpoint_pool", None),
        getattr(args, "execution_platform", "host"),
    )
    endpoint_pool_evidence: dict[str, Any] | None = None
    endpoint_pool_binding: dict[str, Any] | None = None
    if getattr(args, "image_edit_endpoint_pool", None) is not None:
        endpoint_pool_path = args.image_edit_endpoint_pool.resolve(strict=True)
        endpoint_pool_binding = validate_image_edit_endpoint_pool(
            json.loads(endpoint_pool_path.read_text())
        )
        endpoint_pool_evidence = {
            "path": str(endpoint_pool_path),
            "sha256": sha256(endpoint_pool_path),
            "request_sha256": endpoint_pool_binding["request_sha256"],
            "required_capacity": endpoint_pool_binding["required_capacity"],
        }
    bind_resumable_endpoint_pool(
        progress, selected, endpoint_entries, endpoint_pool_binding,
        allow_unstarted_rebind=getattr(args, "explicit_unstarted_pool_rebind", False),
    )
    atomic_json(progress_path, progress)
    endpoint_pool = _ImageEditEndpointPool(config, endpoint_entries)

    # Runs prepared by an older compatible adapter can recover the source
    # vector from the immutable mined-pairs input. This avoids hand-editing
    # committed state while keeping every generated vector provenance-bound.
    source_vectors: dict[str, list[int]] = {}
    if any("source_attribute_values" not in record for record in selected):
        if args.mined_pairs is None or not args.mined_pairs.is_file():
            raise ValueError(
                "SDG plan lacks source attribute vectors; resume with the original --mined-pairs input"
            )
        mined_payload = json.loads(args.mined_pairs.read_text())
        if not isinstance(mined_payload, list):
            raise ValueError("mined pairs must be a list for source-vector recovery")
        for row in mined_payload:
            if isinstance(row, dict):
                name = pathlib.Path(str(row.get("unique_name", ""))).name
                value = row.get("image_attr_values")
                if name and isinstance(value, list):
                    prior = source_vectors.setdefault(name, value)
                    if prior != value:
                        raise ValueError(f"mined pairs disagree on source attributes for {name}")

    def call(
        action: str, input_root: pathlib.Path, log_path: pathlib.Path,
        source_key: str = "", attempt: int = 1,
        target_attributes: dict[str, str] | None = None,
        image_edit_url: str | None = None,
        image_edit_endpoint_id: str | None = None,
    ) -> None:
        identity = f"{action}:{source_key or 'batch'}:{attempt}"
        with progress_lock:
            used = int(progress["command_attempts"].get(identity, 0))
            if used >= 2:
                raise ValueError(f"component retry budget exhausted for {identity}; inspect {log_path}")
            progress["command_attempts"][identity] = used + 1
            atomic_json(progress_path, progress)
        executor = getattr(args, "component_executor", None)
        if executor is None:
            _component_call(
                config, action, input_root, output, log_path,
                source_key, attempt, target_attributes,
                image_edit_url=image_edit_url,
                image_edit_endpoint_id=image_edit_endpoint_id,
            )
        else:
            _component_call(
                config, action, input_root, output, log_path,
                source_key, attempt, target_attributes,
                executor,
                getattr(args, "component_executor_request", None),
                getattr(args, "component_executor_job_id", None),
                image_edit_url=image_edit_url,
                image_edit_endpoint_id=image_edit_endpoint_id,
            )
    logs = output / "logs"
    source_root = output / "source_ids"
    if not progress["preprocessed"]:
        panes = sorted((output / "panes").glob("*.jpg"))
        if len(panes) != len(selected):
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

    def accepted_evidence(record: dict[str, Any], outcome: dict[str, Any]) -> None:
        key = record["source_key"]
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

    def augment_source(record: dict[str, Any], source_index: int) -> str:
        key = record["source_key"]
        with progress_lock:
            outcome = progress["augmentation"].get(key)
        if isinstance(outcome, dict) and outcome.get("status") == "accepted":
            accepted_evidence(record, outcome)
            return "accepted"
        if not isinstance(outcome, dict):
            # A component process can exhaust its command retries before it
            # emits verification metadata (for example, on a bounded endpoint
            # timeout). Preserve that failed generation attempt in the journal
            # and advance; never reset or silently widen either retry budget.
            for exhausted_attempt in range(1, maximum + 1):
                identity = f"augment:{key}:{exhausted_attempt}"
                with progress_lock:
                    command_attempts = int(progress["command_attempts"].get(identity, 0))
                if command_attempts < 2:
                    break
                outcome = {
                    "attempt": exhausted_attempt,
                    "status": "command_failed",
                    "metadata": str((
                        output / "augmentation" / key /
                        f"attempt_{exhausted_attempt}" / "output_metadata.json"
                    ).resolve()),
                }
                with progress_lock:
                    progress["augmentation"][key] = outcome
                    atomic_json(progress_path, progress)
        start_attempt = int(outcome.get("attempt", 0)) + 1 if isinstance(outcome, dict) else 1
        preferred_endpoint = source_index % len(endpoint_pool.entries)
        for attempt in range(start_attempt, maximum + 1):
            identity = f"augment:{key}:{attempt}"
            while True:
                endpoint_index, endpoint_url_value = endpoint_pool.acquire(preferred_endpoint)
                endpoint = endpoint_pool.entries[endpoint_index]
                endpoint_identity = str(endpoint["id"])
                try:
                    call(
                        "augment", source_root, logs / f"augment-{key}-attempt-{attempt}.log",
                        key, attempt, record.get("target_attributes") or {},
                        endpoint_url_value, endpoint_identity,
                    )
                except Exception as exc:
                    endpoint_pool.release(endpoint_index, quarantine=True)
                    with progress_lock:
                        progress["endpoint_attempts"].setdefault(identity, []).append({
                            "endpoint_id": endpoint_identity, "url": endpoint_url_value,
                            "gpu_identity": endpoint["gpu_identity"], "owner": endpoint["owner"],
                            "status": "quarantined",
                        })
                        atomic_json(progress_path, progress)
                        used = int(progress["command_attempts"].get(identity, 0))
                    preferred_endpoint = (endpoint_index + 1) % len(endpoint_pool.entries)
                    if used >= 2:
                        raise
                    continue
                else:
                    endpoint_pool.release(endpoint_index)
                    with progress_lock:
                        progress["endpoint_attempts"].setdefault(identity, []).append({
                            "endpoint_id": endpoint_identity, "url": endpoint_url_value,
                            "gpu_identity": endpoint["gpu_identity"], "owner": endpoint["owner"],
                            "status": "completed",
                        })
                        atomic_json(progress_path, progress)
                    preferred_endpoint = (endpoint_index + 1) % len(endpoint_pool.entries)
                    break
            attempt_root = output / "augmentation" / key / f"attempt_{attempt}"
            metadata_path = attempt_root / "output_metadata.json"
            image_path = attempt_root / "output.jpg"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            accepted = image_path.is_file() and image_path.stat().st_size > 0 and verification_passed(metadata)
            outcome = {
                "attempt": attempt, "status": "accepted" if accepted else "rejected",
                "metadata": str(metadata_path.resolve()),
                "endpoint_id": endpoint_identity,
                "endpoint_url": endpoint_url_value,
            }
            with progress_lock:
                progress["augmentation"][key] = outcome
                atomic_json(progress_path, progress)
            if accepted:
                destination = accepted_root / key / "aug_0"
                destination.mkdir(parents=True, exist_ok=True)
                for name in ("output.jpg", "output.txt", "output_metadata.json"):
                    source = attempt_root / name
                    if source.is_file():
                        shutil.copyfile(source, destination / name)
                break
        with progress_lock:
            status = progress["augmentation"][key]["status"]
        return status

    # The first planned source is a single synchronous smoke gate. The
    # concurrent batch cannot start unless this exact source passes bounded
    # verification; a resumed accepted smoke is validated, never resubmitted.
    smoke_record = selected[0]
    smoke_status = augment_source(smoke_record, 0)
    if smoke_status != "accepted":
        raise ValueError(
            f"augmentation smoke source {smoke_record['source_key']} did not pass verification"
        )
    with progress_lock:
        smoke_result = dict(progress["augmentation"][smoke_record["source_key"]])
    smoke_result["source_key"] = smoke_record["source_key"]
    atomic_json(output / "augmentation_smoke.json", smoke_result)

    pending: list[tuple[int, dict[str, Any]]] = []
    for source_index, record in enumerate(selected[1:], start=1):
        outcome = progress["augmentation"].get(record["source_key"])
        if isinstance(outcome, dict) and outcome.get("status") == "accepted":
            accepted_evidence(record, outcome)
        else:
            pending.append((source_index, record))

    max_in_flight = _augmentation_max_in_flight(config, len(pending), endpoint_entries)
    failures: list[tuple[str, Exception]] = []
    if max_in_flight == 1:
        for source_index, record in pending:
            try:
                augment_source(record, source_index)
            except Exception as exc:  # keep later sources recoverable
                failures.append((record["source_key"], exc))
    elif max_in_flight > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_in_flight) as executor:
            futures = {
                executor.submit(augment_source, record, source_index): record["source_key"]
                for source_index, record in pending
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # finish other isolated source attempts
                    failures.append((futures[future], exc))
    if failures:
        details = "; ".join(f"{key}: {error}" for key, error in sorted(failures))
        raise ValueError(f"augmentation component failure(s); resume safely: {details}")

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
            "source_attribute_values": next(
                (
                    item.get("source_attribute_values")
                    or source_vectors.get(pathlib.Path(item["mined_unique_name"]).name)
                    for item in selected if item["source_key"] == parent_key
                ),
                None,
            ),
            "target_attributes": next(
                (
                    {
                        key: value for key, value in (item.get("target_attributes") or {}).items()
                        if key in EDITABLE_ATTRIBUTES
                    }
                    for item in selected if item["source_key"] == parent_key
                ),
                None,
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
    execution_manifest = {
        "schema_version": "1", "selected_sources": len(selected),
        "accepted_sources": accepted_count, "rejected_sources": len(selected) - accepted_count,
        "accepted_crops": len(crop_records), "progress": str(progress_path.resolve()),
        "normalized_manifest": str((output / "dataset" / "sdg_manifest.json").resolve()),
        "execution_platform": getattr(args, "execution_platform", "host"),
    }
    if endpoint_pool_evidence is not None:
        execution_manifest["endpoint_pool"] = endpoint_pool_evidence
        execution_manifest["image_edit_endpoint_history"] = progress.get(
            "image_edit_endpoint_history", []
        )
    atomic_json(output / "sdg_execution_manifest.json", execution_manifest)
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
    parser.add_argument("--eval-pairs", type=pathlib.Path)
    parser.add_argument("--augmentation-root", type=pathlib.Path)
    parser.add_argument("--accepted-manifest", type=pathlib.Path)
    parser.add_argument("--labels-root", type=pathlib.Path)
    parser.add_argument("--attribute-vocab", type=pathlib.Path)
    parser.add_argument("--action", choices=("preprocess", "augment", "split", "label"))
    parser.add_argument("--input-root", type=pathlib.Path)
    parser.add_argument("--source-key")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--image-edit-url")
    parser.add_argument("--image-edit-endpoint-id")
    parser.add_argument("--image-edit-endpoint-pool", type=pathlib.Path)
    parser.add_argument("--explicit-unstarted-pool-rebind", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execution-platform", choices=EXECUTION_PLATFORMS, default="host",
    )
    parser.add_argument("--component-executor", type=pathlib.Path)
    parser.add_argument("--component-executor-request", type=pathlib.Path)
    parser.add_argument("--component-executor-job-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        executor_fields = (
            args.component_executor,
            args.component_executor_request,
            args.component_executor_job_id,
        )
        if any(value is not None for value in executor_fields):
            if args.operation != "execute" or not all(
                value is not None for value in executor_fields
            ):
                raise ValueError(
                    "component executor fields must be supplied together for execute"
                )
        endpoint_override_fields = (args.image_edit_url, args.image_edit_endpoint_id)
        if any(value is not None for value in endpoint_override_fields):
            if args.operation != "component" or not all(
                value is not None for value in endpoint_override_fields
            ):
                raise ValueError(
                    "image-edit URL and endpoint ID must be supplied together for component"
                )
        if args.image_edit_endpoint_pool is not None and args.operation != "execute":
            raise ValueError("image-edit endpoint pool is valid only for execute")
        if args.explicit_unstarted_pool_rebind and (
            args.operation != "execute" or args.image_edit_endpoint_pool is None
        ):
            raise ValueError(
                "explicit unstarted pool rebind requires execute and a validated endpoint pool"
            )
        if args.execution_platform == "slurm" and args.component_executor is None:
            raise ValueError("SLURM SDG execution requires a component executor")
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
