#!/usr/bin/env python3
"""Exact compute-frame entrypoints for non-TAO IAA workflow operations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tarfile


ROOT = pathlib.Path(__file__).resolve().parent

ADAPTER_IMPORTS = {
    "dataset_rebuild": ("yaml",),
    "dataset_materialize": ("pandas", "pyarrow", "yaml"),
    "gap_analysis": ("numpy", "pandas", "pyarrow", "yaml"),
    "mining_postprocess": ("numpy", "pandas", "pyarrow", "yaml"),
    "history_select": ("pandas", "pyarrow", "yaml"),
    "visualize_prepare": ("matplotlib", "numpy", "pandas", "PIL", "pyarrow", "sklearn", "yaml"),
    "visualize_finish": ("matplotlib", "numpy", "pandas", "pyarrow", "sklearn", "yaml"),
    "eval_config": ("yaml",),
    "train_config": ("yaml",),
    "publish_checkpoint": ("torch", "yaml"),
    "iteration_summary": ("pandas", "pyarrow", "yaml"),
    "metric_parse": (),
    "report": ("matplotlib", "pandas", "pyarrow", "yaml"),
}


def _require_adapter_imports(operation: str) -> None:
    missing = [name for name in ADAPTER_IMPORTS[operation] if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError(
            f"{operation} execution profile lacks required imports: {', '.join(missing)}"
        )


def _state(results: pathlib.Path) -> tuple[dict, pathlib.Path]:
    payload = json.loads((results / "deft_state.json").read_text())
    if payload.get("workflow") != "tao-run-deft-iaa" or payload.get("schema_version") != "3":
        raise ValueError("invalid IAA state")
    canonical = pathlib.Path(payload["results_dir"])
    if results != pathlib.Path("/results") and canonical != results:
        raise ValueError("state results_dir does not match the compute-frame alias")
    if not (canonical / "deft_state.json").is_file():
        raise ValueError("canonical results alias is not mounted in the compute frame")
    return payload, canonical


def _run(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def _repair_history_candidates_after_proven_failure(
    results: pathlib.Path, state: dict, label: str
) -> None:
    """Regenerate omitted candidates only inside bounded attempt-2 recovery."""

    if not label.startswith("iter") or not label[4:].isdigit():
        raise ValueError("history candidate recovery requires an iterN label")
    if state.get("config", {}).get("history_aware") is not True:
        return
    number = int(label[4:])
    mining = results / f"iter_{number}" / "mining"
    candidates = mining / "history_candidates"
    required = (
        candidates / "mined_image_list.txt",
        candidates / "mined_pairs.json",
        candidates / "mined_dataset.json",
    )
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if not missing:
        return
    # Pairs are the platform-declared postprocess output. If they are absent,
    # this is not the legacy omitted-side-output defect and must hard-stop.
    if required[1] in missing:
        raise ValueError("history candidate pairs are missing; bounded repair is inapplicable")

    prior = mining / "history_select.attempt-1.status.json"
    if not prior.is_file() or prior.is_symlink():
        raise ValueError("missing archived history-select attempt-1 failure evidence")
    payload = json.loads(prior.read_text(encoding="utf-8"))
    log = pathlib.Path(str(payload.get("log_path", "")))
    try:
        log.relative_to(mining)
    except ValueError as exc:
        raise ValueError("history-select attempt-1 log escapes mining directory") from exc
    if (
        payload.get("workflow") != "tao-run-deft-iaa"
        or payload.get("name") != "history_select"
        or payload.get("attempt") != 1
        or payload.get("status") != "error"
        or payload.get("backend_state") != "ERROR"
        or payload.get("exit_code") in {None, 0}
        or not log.is_file()
        or log.is_symlink()
    ):
        raise ValueError("history-select attempt-1 evidence is not a bound failure")
    diagnostic = log.read_text(encoding="utf-8", errors="replace")
    if not any(str(path) in diagnostic for path in missing):
        raise ValueError("history-select failure does not identify an omitted candidate")

    _run([
        sys.executable, str(ROOT / "run_iaa_stage.py"), "mining-postprocess",
        "--results-dir", str(results),
        "--deft-config", str(results / "config" / "deft_config.yaml"),
        "--iter-num", str(number),
    ])
    remaining = [
        path for path in required
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0
    ]
    if remaining:
        raise ValueError(
            "bounded mining-candidate repair did not recreate: "
            + ", ".join(str(path) for path in remaining)
        )


def dataset_rebuild(results: pathlib.Path, state: dict) -> None:
    config = state["config"]
    dataset = pathlib.Path(config["dataset_root"])
    images = pathlib.Path(config["images_archive"])
    metadata = pathlib.Path(config["metadata_archive"])
    log = results / "dataset_setup" / "rebuild_verify.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def verify(root: pathlib.Path, *, rebuild: bool) -> bool:
        command = [
            sys.executable,
            str(ROOT / "iaa_deft" / "rebuild.py"),
            "--metadata-root", str(root), "--out", str(root), "--workers", "16",
        ]
        if not rebuild:
            command.append("--verify-only")
        with log.open("w", encoding="utf-8") as output:
            completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        return completed.returncode == 0 and "VERIFY: PASS" in log.read_text(errors="replace")

    # A complete destination is a committed-by-content result and is safe to
    # reuse after a launcher crash. Never merge archives into an existing or
    # partially built customer directory.
    if dataset.exists():
        if dataset.is_dir() and not dataset.is_symlink() and verify(dataset, rebuild=False):
            with log.open("a", encoding="utf-8") as output:
                output.write("REUSED: existing verified dataset\n")
            return
        raise ValueError(
            f"dataset destination already exists but is not a complete verified rebuild: {dataset}"
        )

    digest = state.get("active_runtime_sha256") or state["config"]["iaa_deft_bundle_sha256"]
    staging = dataset.parent / f".{dataset.name}.rebuild-{str(digest)[:12]}"
    if staging.exists():
        if staging.is_dir() and not staging.is_symlink() and verify(staging, rebuild=False):
            os.replace(staging, dataset)
            with log.open("a", encoding="utf-8") as output:
                output.write("RECOVERED: promoted verified staging dataset\n")
            return
        raise ValueError(
            f"incomplete rebuild staging directory retained for diagnosis: {staging}; "
            "move it aside after inspection before retrying"
        )

    staging.mkdir(parents=True)
    try:
        for archive, mode in ((metadata, "r:gz"), (images, "r:")):
            with tarfile.open(archive, mode) as handle:
                handle.extractall(staging, filter="data")
        if not verify(staging, rebuild=True):
            raise ValueError("dataset rebuild did not produce VERIFY: PASS")
        os.replace(staging, dataset)
    except Exception:
        # Preserve the run-scoped staging tree: it is never treated as the
        # dataset and contains the evidence needed to classify the failure.
        raise


def stage_adapter(results: pathlib.Path, state: dict, operation: str, label: str) -> None:
    config = results / "config" / "deft_config.yaml"
    stage = operation.replace("_", "-")
    if operation == "history_select":
        _repair_history_candidates_after_proven_failure(results, state, label)
    argv = [sys.executable, str(ROOT / "run_iaa_stage.py"), stage,
            "--results-dir", str(results), "--deft-config", str(config)]
    if label.startswith("iter"):
        number = label[4:]
        if operation == "eval_config":
            argv += ["--iter-label", label]
        else:
            argv += ["--iter-num", number]
    elif operation == "eval_config":
        argv += ["--iter-label", "baseline"]
    if operation == "publish_checkpoint":
        number = label[4:]
        argv += ["--train-command-status", str(results / f"iter_{number}" / "train" / "train.status.json")]
    if operation == "history_select":
        history_path = results / "mining_selection_history.json"
        if history_path.is_file():
            payload = json.loads(history_path.read_text())
            iterations = payload.get("iterations") if isinstance(payload, dict) else None
            if not isinstance(iterations, list):
                raise ValueError("mining selection history has an invalid iterations list")
            number = int(label[4:])
            matches = [
                row for row in iterations
                if isinstance(row, dict) and row.get("iteration") == number
            ]
            if len(matches) > 1:
                raise ValueError(f"mining selection history duplicates iteration {number}")
            if matches:
                argv.append("--resume")
    _run(argv)


def metric_argv(results: pathlib.Path, state: dict, label: str) -> list[str]:
    contract = state["metric_contract"]
    phase = results / ("zs" if label == "baseline" else f"iter_{label[4:]}") / "evaluate"
    argv = [sys.executable, str(ROOT / "parse_iaa_metrics.py"),
            "--metrics-csv", str(phase / "nvidia_pas_metrics_aggregate.csv"),
            "--metric-name", str(contract["metric_name"]),
            "--query-type", str(contract["query_type"]),
            "--op", str(contract["op"]), "--iter-label", label,
            "--output", str(phase / "metric_result.json")]
    target = contract.get("target")
    if target is not None:
        argv += ["--target", str(target)]
    return argv


def metric_parse(results: pathlib.Path, state: dict, label: str) -> None:
    _run(metric_argv(results, state, label))


def report(results: pathlib.Path) -> None:
    _run([sys.executable, str(ROOT / "render_deft_report.py"),
          "--results-dir", str(results), "--trigger", "loop-end"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=(
        "dataset_rebuild", "dataset_materialize", "gap_analysis",
        "mining_postprocess", "history_select", "visualize_prepare",
        "visualize_finish", "eval_config", "train_config",
        "publish_checkpoint", "iteration_summary", "metric_parse", "report",
    ))
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    results = args.results_dir.expanduser().absolute()
    state, results = _state(results)
    platform = state["config"]["platform"]
    if os.environ.get("IAA_COMPUTE_FRAME") != platform:
        raise ValueError(
            f"IAA mutators must run in the selected {platform} compute frame"
        )
    _require_adapter_imports(args.operation)
    if args.operation == "dataset_rebuild":
        dataset_rebuild(results, state)
    elif args.operation == "metric_parse":
        metric_parse(results, state, args.label)
    elif args.operation == "report":
        report(results)
    else:
        stage_adapter(results, state, args.operation, args.label)
    # Platform runners persist stdout as immutable native-job evidence. Keep
    # the line deterministic and free of paths, arguments, and environment.
    print(f"IAA_ADAPTER_COMPLETE operation={args.operation} label={args.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
