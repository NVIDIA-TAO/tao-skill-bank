# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover coarse-filesystem SDG normalization evidence without rerunning inference."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import tempfile
from typing import Any

from commit_stage import _coarse_slurm_sdg_freshness_attested
from iaa_deft.sdg import normalize_generated_pairs
from run_sdg_stage import _load_config, _run_once


OUTPUTS = ("sdg_manifest.json", "sdg_pairs.json", "sdg_image_list.txt")


def _sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise


def _load_json(path: pathlib.Path, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ValueError(f"{name} is missing, empty, or unsafe: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return payload


def _paths(results_dir: pathlib.Path, iteration: int) -> dict[str, Any]:
    results = results_dir.resolve()
    datagen = results / f"iter_{iteration}" / "datagen"
    dataset = datagen / "dataset"
    repair = datagen / "freshness_repair" / "sdg-normalize-attempt-1"
    outputs = [dataset / name for name in OUTPUTS]
    backups = [repair / "backup" / name for name in OUTPUTS]
    return {
        "results": results, "datagen": datagen, "dataset": dataset,
        "repair": repair, "journal": repair / "repair.json",
        "outputs": outputs, "backups": backups,
        "status": datagen / "status" / "sdg-normalize.slurm.status.json",
    }


def _regular_record(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ValueError(f"normalization input is missing, empty, or unsafe: {path}")
    return {"path": str(path.resolve()), "sha256": _sha(path), "size": path.stat().st_size}


def _canonical_mount_path(
    path: pathlib.Path, *, runtime_root: pathlib.Path,
    canonical_root: pathlib.Path, allowed_root: pathlib.Path, name: str,
) -> pathlib.Path:
    """Map one request mount alias to its immutable host-path identity."""
    candidate = pathlib.Path(os.path.abspath(path.expanduser()))
    runtime = pathlib.Path(os.path.abspath(runtime_root))
    canonical = pathlib.Path(os.path.abspath(canonical_root))
    allowed = pathlib.Path(os.path.abspath(allowed_root))
    try:
        relative = candidate.relative_to(runtime)
        normalized = canonical / relative
    except ValueError:
        normalized = candidate
    try:
        normalized.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"{name} escapes its approved canonical root") from exc
    if normalized.is_symlink() or normalized.resolve(strict=False) != normalized:
        raise ValueError(f"{name} contains or traverses a symlink")
    return normalized


def _normalization_inputs(paths: dict[str, Any]) -> list[dict[str, Any]]:
    """Bind every file whose content can affect normalization output."""
    state = _load_json(paths["results"] / "deft_state.json", "DEFT state")
    canonical_results = pathlib.Path(str(state.get("results_dir", paths["results"]))).resolve()
    canonical_datagen = canonical_results / paths["datagen"].relative_to(paths["results"])
    dataset_root = pathlib.Path(str(state.get("config", {}).get("dataset_root", ""))).resolve()
    accepted = canonical_datagen / "accepted_crop_manifest.json"
    vocab = dataset_root / "attribute_vocab.json"
    eval_list = canonical_results / "iaa_splits" / "eval_list.txt"
    sdg_config = canonical_results / "config" / "sdg_config.yaml"
    manifest = _load_json(accepted, "accepted crop manifest")
    rows = manifest.get("accepted")
    if not isinstance(rows, list) or not rows:
        raise ValueError("accepted crop manifest must contain accepted records")
    inputs = [accepted, vocab, eval_list, sdg_config]
    labels_root = canonical_datagen / "labels"
    for row in sorted(rows, key=lambda item: (str(item.get("source_key", "")), int(item.get("attempt", 0)))):
        key = str(row.get("source_key", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise ValueError("accepted crop manifest contains an unsafe source key")
        image = _canonical_mount_path(
            pathlib.Path(str(row.get("image", ""))),
            runtime_root=paths["datagen"], canonical_root=canonical_datagen,
            allowed_root=canonical_datagen, name="accepted image",
        )
        metadata = _canonical_mount_path(
            pathlib.Path(str(row.get("metadata", ""))),
            runtime_root=paths["datagen"], canonical_root=canonical_datagen,
            allowed_root=canonical_datagen, name="accepted metadata",
        )
        inputs.extend((image, metadata, labels_root / key / "task" / "open_qa.json"))
    records = [_regular_record(path) for path in inputs]
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("normalization input set contains duplicate paths")
    return records


def validate_prepared(results_dir: pathlib.Path, iteration: int) -> dict[str, Any]:
    """Validate the exact typed repair gate before producing a signed action."""
    paths = _paths(results_dir, iteration)
    journal = _load_json(paths["journal"], "freshness repair journal")
    if (
        journal.get("schema_version") != "1"
        or journal.get("workflow") != "tao-run-deft-iaa"
        or journal.get("kind") != "sdg_normalize_freshness_repair"
        or journal.get("iteration") != iteration
        or journal.get("state") != "prepared"
    ):
        raise ValueError("SDG normalization repair journal is not prepared")
    records = journal.get("records")
    if not isinstance(records, list) or len(records) != len(paths["outputs"]):
        raise ValueError("prepared repair output records are malformed")
    for record, output, backup in zip(records, paths["outputs"], paths["backups"]):
        if (
            output.exists() or not backup.is_file() or backup.is_symlink()
            or _sha(backup) != record.get("sha256")
            or backup.stat().st_size != record.get("size")
        ):
            raise ValueError("prepared repair no longer has exact absent outputs and backups")
    current_inputs = _normalization_inputs(paths)
    if journal.get("normalization_inputs") != current_inputs:
        raise ValueError("normalization inputs changed after freshness repair preparation")
    return journal


def _tree_records(root: pathlib.Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"normalized dataset subtree is missing or unsafe: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"normalized dataset subtree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"normalized dataset subtree contains a non-file: {path}")
        records.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha(path),
            "size": path.stat().st_size,
        })
    return records


def _validate_normalized_shape(paths: dict[str, Any]) -> None:
    manifest = _load_json(paths["outputs"][0], "normalized manifest")
    pairs_path, image_list_path = paths["outputs"][1:]
    pairs = json.loads(pairs_path.read_text())
    lines = [line for line in image_list_path.read_text().splitlines() if line]
    if (
        not isinstance(pairs, list) or not pairs
        or not lines
        or manifest.get("num_pairs") != len(pairs)
        or len(lines) != len(pairs)
        or not isinstance(manifest.get("num_source_images"), int)
        or manifest["num_source_images"] < 1
        or manifest.get("rejected_samples_included") != 0
    ):
        raise ValueError("normalized SDG outputs fail pair/source/rejection shape checks")


def _terminal_binding(paths: dict[str, Any]) -> dict[str, Any]:
    datagen = paths["datagen"]
    endpoint_path = datagen / "endpoint_manifest.json"
    endpoint = _load_json(endpoint_path, "endpoint manifest")
    job_id = endpoint.get("job_id")
    if not isinstance(job_id, str) or re.fullmatch(r"[A-Za-z0-9.-]+", job_id) is None:
        raise ValueError("endpoint manifest lacks a safe SLURM job id")
    request_path = datagen / ".tao-runtime" / f"sdg.action.{job_id}.json"
    terminal_path = datagen / f"slurm_sdg_terminal.{job_id}.json"
    request = _load_json(request_path, "signed SDG request")
    terminal = _load_json(terminal_path, "SLURM SDG terminal")
    unsigned = dict(request)
    unsigned.pop("request_sha256", None)
    request_sha = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected = [str(path.resolve()) for path in paths["outputs"]]
    if (
        request.get("request_sha256") != request_sha
        or endpoint.get("request_sha256") != request_sha
        or endpoint.get("action_id") != request.get("action_id")
        or endpoint.get("attempt") != request.get("attempt")
        or request.get("expected_outputs", [])[:3] != expected
        or terminal.get("status") != "ok"
        or terminal.get("job_id") != job_id
        or terminal.get("request_sha256") != request_sha
        or terminal.get("action_id") != request.get("action_id")
        or terminal.get("attempt") != request.get("attempt")
        or terminal.get("expected_outputs", [])[:3] != expected
    ):
        raise ValueError("SLURM terminal/request/endpoint binding is inconsistent")
    return {
        "job_id": job_id, "action_id": request["action_id"],
        "request_sha256": request_sha, "attempt": request["attempt"],
        "endpoint_manifest": str(endpoint_path),
        "endpoint_manifest_sha256": _sha(endpoint_path),
        "request": str(request_path), "request_file_sha256": _sha(request_path),
        "terminal": str(terminal_path), "terminal_sha256": _sha(terminal_path),
    }


def prepare(results_dir: pathlib.Path, iteration: int) -> dict[str, Any]:
    paths = _paths(results_dir, iteration)
    state = _load_json(paths["results"] / "deft_state.json", "DEFT state")
    if (
        state.get("workflow") != "tao-run-deft-iaa"
        or state.get("config", {}).get("platform") != "slurm"
        or state.get("current_iteration") != iteration
    ):
        raise ValueError("freshness repair requires the current uncommitted SLURM iteration")
    journal_path = paths["journal"]
    if journal_path.exists():
        journal = _load_json(journal_path, "freshness repair journal")
        if (
            journal.get("schema_version") != "1"
            or journal.get("workflow") != "tao-run-deft-iaa"
            or journal.get("kind") != "sdg_normalize_freshness_repair"
            or journal.get("iteration") != iteration
            or journal.get("state") not in {"moving", "prepared"}
        ):
            raise ValueError("existing freshness repair journal has an invalid state or identity")
        records = journal.get("records")
        if not isinstance(records, list) or len(records) != len(paths["outputs"]):
            raise ValueError("existing freshness repair journal has invalid output records")
        if journal["state"] == "prepared":
            for record, output, backup in zip(records, paths["outputs"], paths["backups"]):
                if output.exists() or not backup.is_file() or _sha(backup) != record.get("sha256"):
                    raise ValueError("prepared freshness repair no longer has exact absent outputs/backups")
            current_inputs = _normalization_inputs(paths)
            previous_inputs = journal.get("normalization_inputs")
            if previous_inputs is None:
                journal["normalization_inputs"] = current_inputs
                _atomic_json(journal_path, journal)
            elif previous_inputs != current_inputs:
                raise ValueError("normalization inputs changed after freshness repair preparation")
    else:
        status = _load_json(paths["status"], "legacy normalize status")
        if (
            status.get("schema_version") != "1" or status.get("status") != "ok"
            or status.get("exit_code") != 0 or status.get("name") != "sdg-normalize"
            or status.get("execution_platform") != "slurm" or status.get("attempt") != 1
            or "pre_action" in status or "output_evidence" in status
        ):
            raise ValueError("normalize status is not the exact legacy coarse-filesystem signature")
        started_ns, finished_ns = status.get("started_ns"), status.get("finished_ns")
        if not isinstance(started_ns, int) or not isinstance(finished_ns, int):
            raise ValueError("normalize status lacks integer time bounds")
        for output in paths["outputs"]:
            if not output.is_file() or output.is_symlink() or output.stat().st_size == 0:
                raise ValueError(f"normalized output is missing, empty, or unsafe: {output}")
            mtime = output.stat().st_mtime_ns
            if not (
                mtime < started_ns <= finished_ns
                and mtime % 1_000_000_000 == 0
                and mtime // 1_000_000_000 == started_ns // 1_000_000_000
                and mtime <= finished_ns
            ):
                raise ValueError("normalized output does not have the exact same-second coarse mtime signature")
        _validate_normalized_shape(paths)
        binding = _terminal_binding(paths)
        records = [{
            "path": str(output), "backup": str(backup), "sha256": _sha(output),
            "size": output.stat().st_size, "mtime_ns": output.stat().st_mtime_ns,
        } for output, backup in zip(paths["outputs"], paths["backups"])]
        journal = {
            "schema_version": "1", "workflow": "tao-run-deft-iaa",
            "kind": "sdg_normalize_freshness_repair", "iteration": iteration,
            "state": "moving", "binding": binding, "records": records, "moved": [],
            "legacy_status": str(paths["status"]),
            "legacy_status_sha256": _sha(paths["status"]),
            "normalization_inputs": _normalization_inputs(paths),
        }
        _atomic_json(journal_path, journal)
    if journal.get("binding") != _terminal_binding(paths):
        raise ValueError("freshness repair terminal binding changed")
    for record, expected_output, expected_backup in zip(
        records, paths["outputs"], paths["backups"]
    ):
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "backup", "sha256", "size", "mtime_ns"}
            or record.get("path") != str(expected_output)
            or record.get("backup") != str(expected_backup)
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            or not isinstance(record.get("size"), int) or record["size"] < 1
            or not isinstance(record.get("mtime_ns"), int) or record["mtime_ns"] < 1
        ):
            raise ValueError("freshness repair output record is malformed or path-confused")
    if journal["state"] == "prepared":
        return {"status": "prepared", "journal": str(journal_path), "outputs_absent": True}
    for record in records:
        source, backup = pathlib.Path(record["path"]), pathlib.Path(record["backup"])
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            if source.exists() or _sha(backup) != record["sha256"]:
                raise ValueError("freshness repair move has ambiguous or changed outputs")
        else:
            if not source.is_file() or _sha(source) != record["sha256"]:
                raise ValueError("freshness repair source changed before atomic move")
            os.replace(source, backup)
        if record["path"] not in journal["moved"]:
            journal["moved"].append(record["path"])
            _atomic_json(journal_path, journal)
    journal["state"] = "prepared"
    journal["prepared_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _atomic_json(journal_path, journal)
    return {"status": "prepared", "journal": str(journal_path), "outputs_absent": True}


def recompute(results_dir: pathlib.Path, iteration: int) -> dict[str, Any]:
    """Recompute only normalized metadata inside a signed zero-GPU action."""
    paths = _paths(results_dir, iteration)
    journal = validate_prepared(results_dir, iteration)
    by_path = {record["path"]: pathlib.Path(record["path"]) for record in journal["normalization_inputs"]}
    state = _load_json(paths["results"] / "deft_state.json", "DEFT state")
    canonical_results = pathlib.Path(str(state.get("results_dir", paths["results"]))).resolve()
    canonical_datagen = canonical_results / paths["datagen"].relative_to(paths["results"])
    accepted = canonical_datagen / "accepted_crop_manifest.json"
    dataset_root = pathlib.Path(str(state["config"]["dataset_root"])).resolve()
    vocab = dataset_root / "attribute_vocab.json"
    eval_list = canonical_results / "iaa_splits" / "eval_list.txt"
    sdg_config = canonical_results / "config" / "sdg_config.yaml"
    required = (accepted, vocab, eval_list, sdg_config)
    if any(str(path.resolve()) not in by_path for path in required):
        raise ValueError("prepared normalization input set lacks a required canonical input")
    labels_root = canonical_datagen / "labels"
    work_root = paths["repair"] / "recomputed"
    temporary_dataset = work_root / "dataset"
    existing_dataset = paths["dataset"]
    canonical_dataset = canonical_datagen / "dataset"

    def handler() -> dict[str, Any]:
        if work_root.exists():
            shutil.rmtree(work_root)
        work_root.mkdir(parents=True)
        config = _load_config(sdg_config)
        eval_names = {
            pathlib.Path(line.strip()).name
            for line in eval_list.read_text().splitlines() if line.strip()
        }
        normalize_generated_pairs(
            accepted, labels_root, temporary_dataset, vocab, eval_names,
            config["generation"]["caption_policy"],
        )
        # The temporary compute-frame path is an implementation detail.  The
        # published manifest must retain the immutable host-path identity used
        # by the original platform-local normalization.
        manifest_path = temporary_dataset / "sdg_manifest.json"
        manifest = _load_json(manifest_path, "recomputed normalized manifest")
        manifest.update({
            "image_dir": str(canonical_dataset / "images"),
            "caption_dir": str(canonical_dataset / "captions"),
            "image_list_file": str(canonical_dataset / "sdg_image_list.txt"),
            "pairs_file": str(canonical_dataset / "sdg_pairs.json"),
            "attribute_vocab_file": str(canonical_dataset / "attribute_vocab.json"),
        })
        _atomic_json(manifest_path, manifest)
        for subtree in ("images", "captions"):
            if _tree_records(temporary_dataset / subtree) != _tree_records(existing_dataset / subtree):
                raise ValueError(f"recomputed {subtree} differ from preserved normalized data")
        if (temporary_dataset / "attribute_vocab.json").read_bytes() != (existing_dataset / "attribute_vocab.json").read_bytes():
            raise ValueError("recomputed attribute vocabulary differs from preserved normalized data")
        for output in paths["outputs"]:
            candidate = temporary_dataset / output.name
            if not candidate.is_file() or candidate.is_symlink():
                raise ValueError(f"recomputation did not produce {output.name}")
            os.replace(candidate, output)
        return {"repair": "normalization-only", "outputs": len(paths["outputs"])}

    input_paths = [pathlib.Path(record["path"]) for record in journal["normalization_inputs"]]
    status = _run_once(
        paths["datagen"], "sdg-normalize", input_paths, paths["outputs"], handler,
        execution_platform="slurm",
    )
    return {"status": "recomputed", "attempt": status["attempt"], "outputs": len(paths["outputs"])}


def verify(results_dir: pathlib.Path, iteration: int) -> dict[str, Any]:
    paths = _paths(results_dir, iteration)
    journal = _load_json(paths["journal"], "freshness repair journal")
    if journal.get("state") not in {"prepared", "verified"}:
        raise ValueError("freshness repair is not ready for verification")
    _validate_normalized_shape(paths)
    status = _load_json(paths["status"], "recomputed normalize status")
    evidence = status.get("output_evidence")
    normally_fresh = (
        isinstance(status.get("started_ns"), int)
        and isinstance(status.get("finished_ns"), int)
        and isinstance(evidence, list)
        and len(evidence) == len(paths["outputs"])
        and all(
            isinstance(record, dict)
            and record.get("path") == str(output.resolve())
            and output.is_file() and not output.is_symlink()
            and record.get("sha256") == _sha(output)
            and record.get("size") == output.stat().st_size > 0
            and record.get("mtime_ns") == output.stat().st_mtime_ns
            and status["started_ns"] <= output.stat().st_mtime_ns <= status["finished_ns"]
            for record, output in zip(evidence, paths["outputs"])
        )
    )
    if not normally_fresh and not _coarse_slurm_sdg_freshness_attested(
            status, paths["outputs"], scope=paths["datagen"].parent,
            status_path=paths["status"],
        ):
        raise ValueError("recomputed normalize output lacks strict coarse-filesystem evidence")
    for record, output, backup in zip(journal["records"], paths["outputs"], paths["backups"]):
        if (
            not backup.is_file() or _sha(backup) != record["sha256"]
            or not output.is_file() or _sha(output) != record["sha256"]
            or output.read_bytes() != backup.read_bytes()
        ):
            raise ValueError("recomputed normalized output is not byte-identical to its backup")
    journal["state"] = "verified"
    journal["verified_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    journal["recomputed_status_sha256"] = _sha(paths["status"])
    _atomic_json(paths["journal"], journal)
    return {"status": "verified", "journal": str(paths["journal"]), "byte_identical": True}


def restore(results_dir: pathlib.Path, iteration: int) -> dict[str, Any]:
    paths = _paths(results_dir, iteration)
    journal = _load_json(paths["journal"], "freshness repair journal")
    if journal.get("state") not in {"moving", "prepared"}:
        raise ValueError("only an incomplete freshness repair can be restored")
    failed_dir = paths["repair"] / "failed-recompute"
    for record, output, backup in zip(journal["records"], paths["outputs"], paths["backups"]):
        if not backup.is_file() or _sha(backup) != record["sha256"]:
            raise ValueError("backup is missing or changed; refusing partial restore")
        if output.exists() and (failed_dir / output.name).exists():
            raise ValueError(f"failed recompute evidence already exists: {failed_dir / output.name}")
    for record, output, backup in zip(journal["records"], paths["outputs"], paths["backups"]):
        if output.exists():
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed = failed_dir / output.name
            os.replace(output, failed)
        os.replace(backup, output)
        os.utime(output, ns=(record["mtime_ns"], record["mtime_ns"]))
    journal["state"] = "restored"
    journal["restored_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _atomic_json(paths["journal"], journal)
    return {"status": "restored", "journal": str(paths["journal"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "recompute", "verify", "restore"))
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--iteration", required=True, type=int)
    args = parser.parse_args()
    try:
        result = globals()[args.action](args.results_dir, args.iteration)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"repair_sdg_normalize_freshness: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
