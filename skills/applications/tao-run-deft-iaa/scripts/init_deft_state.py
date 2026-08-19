# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialize ${RESULTS_DIR}/deft_state.json for an IAA CLIP DEFT run.

Why this exists: inline-dict writes drift from the canonical schema and can
produce duplicate top-level keys — Python 3.12+ emits a `SyntaxWarning` for
these and the loop's resume logic reads whichever copy parsing keeps, which is
not stable across edits.

This script builds the dict with literal-once keys and writes the JSON. Atomic
write (tmp + os.replace). It refuses to reinitialize an existing
deft_state.json — the resume path is supposed to read disk, not regenerate —
and it appends nothing to loop_log.jsonl.

CLI:

    python scripts/init_deft_state.py \
        --results-dir ~/workspace/results/run_20260805_090000 \
        --workspace ~/workspace \
        --dataset-root ~/workspace/data/iaa_v31_tao_ft \
        --max-iterations 3 \
        --metric-target 0.85 \
        --platform docker \
        # unpinned: illustrative CLI placeholder; executable default is release-managed below
        --pyt-image nvcr.io/nvidia/tao/tao-toolkit:pyt \
        # unpinned: illustrative CLI placeholder; executable default is release-managed below
        --ds-image nvcr.io/nvidia/tao/tao-toolkit:ds \
        --deft-config ~/workspace/specs/deft_config.yaml \
        --tao-spec ~/workspace/specs/tao_spec.yaml

On-disk layout recorded in `config.layout` follows the bundled IAA runtime
conventions: baseline (zero-shot) evaluate artifacts live under
${RESULTS_DIR}/zs/, iteration N artifacts under ${RESULTS_DIR}/iter_<N>/, and
run-level shared artifacts under ${RESULTS_DIR}/iaa_splits/,
${RESULTS_DIR}/embeddings/source/, and the two selection-history JSON files.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile

import yaml

from archive_contract import approved_sha256, verify_archive
from metric_contract import validate_contract


WORKFLOW = "tao-run-deft-iaa"
PINNED_PYT_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt"  # versions-key: images.tao_toolkit.pyt
PINNED_DS_IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services"  # versions-key: images.tao_toolkit.data_services
RUN_SPEC_NAMES = (
    "deft_config.yaml",
    "tao_spec.yaml",
    "text_embed_spec.yaml",
    "image_embed_spec.yaml",
    "mining_spec.yaml",
    "approval.json",
)

_COMPLETED_STEP_VALUES = [
    "dataset_setup",
    "pool_embed",
    "evaluate",
    "gap_analysis",
    "data_mining",
    "history_select",
    "visualize",
    "train",
    "loop_stop",
]
_STATUS_VALUES = ["pending", "in_progress", "complete", "failed"]


def _parse_gpu_ids(raw: str, num_gpus: int) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError('--gpu-ids must be a non-empty comma list, e.g. "0,1"')
    try:
        gpu_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"--gpu-ids must be a comma list of integers: {raw!r}") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"--gpu-ids must be non-negative device indices: {raw!r}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-ids contains duplicate device indices: {raw!r}")
    if len(gpu_ids) != num_gpus:
        raise ValueError(
            f"--gpu-ids lists {len(gpu_ids)} device(s) but --num-gpus is {num_gpus}"
        )
    return gpu_ids


def _resolve_dataset_root(root: pathlib.Path) -> pathlib.Path:
    """Resolve the intended dataset root before dataset_setup materializes it."""
    resolved = root.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"--dataset-root exists but is not a directory: {resolved}")
    return resolved


def _workspace_child(path: pathlib.Path, workspace: pathlib.Path, name: str) -> pathlib.Path:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{name} must be under --workspace {workspace}: {resolved}") from exc
    if relative == pathlib.Path("."):
        raise ValueError(f"{name} must be a child of --workspace, not the workspace itself")
    return resolved


def _validate_metric_gate(
    name: str, query_type: str, operator: str, target: float | None
) -> dict:
    """Reject malformed metric gates before they reach disk.

    A missing --metric-target is legal: the gate then never passes and the
    loop runs to --max-iterations. A present target must be a finite number,
    and Rank-style metrics are fractions in [0, 1].
    """
    contract = validate_contract(
        {
            "metric_name": name,
            "query_type": query_type,
            "op": operator,
            "target": target,
        }
    )
    if target is not None and name.strip().lower().startswith("rank") and not 0.0 <= target <= 1.0:
        raise ValueError(
            f"--metric-target for {name} must be a fraction in [0, 1], got {target}"
        )
    return contract


def _required_input_file(path: pathlib.Path, name: str) -> pathlib.Path:
    expanded = path.expanduser()
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError(f"{name} must be an existing file: {path}")
    if resolved.stat().st_size == 0:
        raise ValueError(f"{name} must not be empty: {path}")
    return resolved


def _required_input_dir(path: pathlib.Path, name: str) -> pathlib.Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an existing directory: {path}")
    return resolved


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_checksum_manifest(
    manifest: pathlib.Path, archives: tuple[pathlib.Path, pathlib.Path]
) -> None:
    referenced: set[pathlib.Path] = set()
    for line_number, raw in enumerate(manifest.read_text().splitlines(), 1):
        line = raw.strip("\n")
        if not line or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"[0-9A-Fa-f]{64} ([ *])(.+)", line)
        if not match:
            raise ValueError(
                f"--checksums-file line {line_number} is not GNU SHA256 format"
            )
        filename = match.group(2)
        if "\x00" in filename or filename.startswith("\\"):
            raise ValueError(
                "--checksums-file uses an unsupported escaped filename entry"
            )
        referenced.add((manifest.parent / filename).resolve())
    missing = [str(path) for path in archives if path.resolve() not in referenced]
    if missing:
        raise ValueError(
            "--checksums-file must bind both approved archives; missing "
            + ", ".join(missing)
        )


def _python_tree_sha256(root: pathlib.Path) -> str:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise ValueError(f"bundled IAA runtime contains no Python files: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _mapping(root: dict, key: str, source: str) -> dict:
    value = root.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source}.{key} must be an object")
    return value


def _load_run_config(args: argparse.Namespace) -> dict:
    config_dir = args.deft_config.parent.resolve()
    expected_config_dir = args.results_dir.resolve() / "config"
    if config_dir != expected_config_dir:
        raise ValueError(
            f"prepared config directory must be {expected_config_dir}, got {config_dir}"
        )
    if args.tao_spec.parent.resolve() != config_dir:
        raise ValueError("--deft-config and --tao-spec must share the run config directory")
    spec_sha256: dict[str, str] = {}
    for name in RUN_SPEC_NAMES:
        spec_path = _required_input_file(config_dir / name, f"run config {name}")
        spec_sha256[name] = _sha256(spec_path)
    approval_path = config_dir / "approval.json"
    approval = json.loads(approval_path.read_text())
    expected_approval = {
        "schema_version": "3",
        "workflow": WORKFLOW,
        "workspace": str(args.workspace.resolve()),
        "results_dir": str(args.results_dir.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "iaa_deft_bundle_sha256": args.iaa_deft_bundle_sha256,
        "images_archive": str(args.images_archive.resolve()),
        "images_archive_sha256": args.images_archive_sha256,
        "metadata_archive": str(args.metadata_archive.resolve()),
        "metadata_archive_sha256": args.metadata_archive_sha256,
        "checksums_file": (
            str(args.checksums_file.resolve())
            if args.checksums_file is not None
            else None
        ),
        "requires_hf_token": args.requires_hf_token,
        "max_iterations": args.max_iterations,
        "metric_contract": args.metric_contract,
        "pyt_image": args.pyt_image,
        "ds_image": args.ds_image,
    }
    if approval != expected_approval:
        raise ValueError(
            "approval.json does not match the approved initialization inputs; "
            "rerun config preparation in a fresh results directory"
        )
    deft = yaml.safe_load(args.deft_config.read_text())
    tao = yaml.safe_load(args.tao_spec.read_text())
    if not isinstance(deft, dict) or not isinstance(tao, dict):
        raise ValueError("--deft-config and --tao-spec must have object roots")
    iteration = _mapping(deft, "iteration", "deft_config")
    experiment = _mapping(deft, "experiment", "deft_config")
    mining = _mapping(deft, "mining", "deft_config")
    history = _mapping(mining, "history_aware", "deft_config.mining")
    training = _mapping(deft, "training", "deft_config")
    gap = _mapping(deft, "gap_analysis", "deft_config")
    train = _mapping(tao, "train", "tao_spec")
    evaluate = _mapping(tao, "evaluate", "tao_spec")
    if iteration.get("start") != 1:
        raise ValueError("prepared deft_config iteration.start must be 1")
    if iteration.get("end") != args.max_iterations:
        raise ValueError(
            "prepared deft_config iteration.end must equal --max-iterations"
        )
    if pathlib.Path(str(experiment.get("results_path", ""))).resolve() != args.results_dir.resolve():
        raise ValueError("prepared deft_config experiment.results_path must equal --results-dir")
    num_gpus = train.get("num_gpus")
    gpu_ids = train.get("gpu_ids")
    if not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError("prepared tao_spec train.num_gpus must be >= 1")
    if not isinstance(gpu_ids, list):
        raise ValueError("prepared tao_spec train.gpu_ids must be a list")
    parsed_gpu_ids = _parse_gpu_ids(",".join(str(item) for item in gpu_ids), num_gpus)
    if evaluate.get("num_gpus") != num_gpus or evaluate.get("gpu_ids") != gpu_ids:
        raise ValueError("prepared TAO train/evaluate GPU shapes must match")
    training_epochs = train.get("num_epochs")
    if not isinstance(training_epochs, int) or training_epochs < 1:
        raise ValueError("prepared tao_spec train.num_epochs must be >= 1")
    if gap.get("metric_name") != args.metric_contract["metric_name"]:
        raise ValueError(
            "prepared gap_analysis.metric_name must match the immutable "
            "metric contract"
        )
    replay = float(history.get("replay_fraction"))
    if not 0.0 <= replay <= 1.0:
        raise ValueError("prepared replay_fraction must be in [0, 1]")
    boolean_fields = {
        "history_aware.enabled": history.get("enabled"),
        "training.continual_dataset": training.get("continual_dataset"),
        "training.continual_model": training.get("continual_model"),
        "experiment.visualize": experiment.get("visualize"),
        "experiment.visualize_embeddings": experiment.get("visualize_embeddings"),
    }
    invalid_booleans = [name for name, value in boolean_fields.items() if not isinstance(value, bool)]
    if invalid_booleans:
        raise ValueError(
            "prepared boolean fields are invalid: " + ", ".join(invalid_booleans)
        )
    for name, value in (
        ("mining.topn", mining.get("topn")),
        ("gap_analysis.target_query_count", gap.get("target_query_count")),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"prepared {name} must be an integer >= 1")
    if mining.get("knn_metric") not in {"cosine", "euclidean"}:
        raise ValueError("prepared mining.knn_metric must be cosine or euclidean")
    for key in ("train_config", "eval_config"):
        if pathlib.Path(str(experiment.get(key, ""))).resolve() != args.tao_spec.resolve():
            raise ValueError(
                f"prepared deft_config experiment.{key} must equal --tao-spec"
            )
    return {
        "config_dir": str(config_dir),
        "approval_manifest": str(approval_path),
        "spec_sha256": spec_sha256,
        "deft_config_sha256": _sha256(args.deft_config),
        "tao_spec_sha256": _sha256(args.tao_spec),
        "training_epochs": training_epochs,
        "num_gpus": num_gpus,
        "gpu_ids": parsed_gpu_ids,
        "history_aware": history.get("enabled"),
        "replay_fraction": replay,
        "mining_topn": int(mining.get("topn")),
        "knn_metric": str(mining.get("knn_metric")),
        "target_query_count": int(gap.get("target_query_count")),
        "continual_dataset": training.get("continual_dataset"),
        "continual_model": training.get("continual_model"),
        "visualize": experiment.get("visualize"),
        "visualize_embeddings": experiment.get("visualize_embeddings"),
    }


def build_state(args: argparse.Namespace) -> dict:
    ws = args.workspace.expanduser().resolve()
    rd = args.results_dir.expanduser().resolve()

    state = {
        "schema_version": "3",
        "workflow": WORKFLOW,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "results_dir": str(rd),
        "max_iterations": args.max_iterations,
        "current_iteration": 0,
        "gate_met": False,
        # The canonical contract shape record_metric_result.py and
        # metric_contract.contract_from_state consume.
        "metric_contract": dict(args.metric_contract),
        "config": {
            "workspace": str(ws),
            "dataset_root": str(args.dataset_root),
            "iaa_deft_bundle_sha256": args.iaa_deft_bundle_sha256,
            "images_archive": str(args.images_archive),
            "images_archive_sha256": args.images_archive_sha256,
            "metadata_archive": str(args.metadata_archive),
            "metadata_archive_sha256": args.metadata_archive_sha256,
            "checksums_file": (
                str(args.checksums_file) if args.checksums_file is not None else None
            ),
            "checksums_file_sha256": (
                _sha256(args.checksums_file)
                if args.checksums_file is not None
                else None
            ),
            "requires_hf_token": args.requires_hf_token,
            "platform": args.platform,
            "pyt_image": args.pyt_image,
            "ds_image": args.ds_image,
            "deft_config": str(args.deft_config),
            "tao_spec": str(args.tao_spec),
            **args.run_config,
            # Bundled IAA runtime conventions; iteration labels map to
            # directories as baseline -> zs/ and iterN -> iter_<N>/.
            "layout": {
                "baseline_dir": str(rd / "zs"),
                "iteration_dir_template": str(rd / "iter_{N}"),
                "iaa_splits_dir": str(rd / "iaa_splits"),
                "source_embeddings_dir": str(rd / "embeddings" / "source"),
                "caption_selection_history": str(
                    rd / "caption_selection_history.json"
                ),
                "mining_selection_history": str(
                    rd / "mining_selection_history.json"
                ),
            },
        },
        "iterations": {"baseline": {"status": "pending"}},
        "_completed_step_values": list(_COMPLETED_STEP_VALUES),
        "_status_values": list(_STATUS_VALUES),
    }
    return state


def write_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize deft_state.json for an IAA CLIP DEFT run with a "
            "guaranteed-unique key set. Refuses to reinitialize an existing "
            "file."
        ),
    )
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=pathlib.Path,
        help=(
            "Intended iaa_v31_tao_ft directory. It may not exist until "
            "baseline/dataset_setup materializes it."
        ),
    )
    parser.add_argument("--images-archive", required=True, type=pathlib.Path)
    parser.add_argument("--images-archive-sha256", required=True)
    parser.add_argument("--metadata-archive", required=True, type=pathlib.Path)
    parser.add_argument("--metadata-archive-sha256", required=True)
    parser.add_argument("--checksums-file", type=pathlib.Path)
    parser.add_argument(
        "--requires-hf-token",
        action="store_true",
        help="Must match approval.json; token values are never persisted.",
    )
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument(
        "--metric-name",
        default="Rank-1",
        help="Primary metric key gating the loop (default Rank-1).",
    )
    parser.add_argument(
        "--metric-query-type",
        default="medium",
        help="Query split the gate reads the metric from (default medium).",
    )
    parser.add_argument(
        "--metric-op",
        default=">=",
        choices=(">=", ">", "<=", "<"),
        help="Success comparison operator.",
    )
    parser.add_argument(
        "--metric-target",
        type=float,
        help=(
            "Numeric success target. When absent the gate never passes and "
            "the loop runs to --max-iterations."
        ),
    )
    parser.add_argument("--platform", default="docker", choices=("docker",))
    parser.add_argument(
        "--pyt-image",
        required=True,
        help="TAO PyTorch container URI used for train/evaluate.",
    )
    parser.add_argument(
        "--ds-image",
        required=True,
        help="TAO data-services container URI used for embedding/mining.",
    )
    parser.add_argument(
        "--deft-config",
        required=True,
        type=pathlib.Path,
        help="The run's iaa_deft loop config file.",
    )
    parser.add_argument(
        "--tao-spec",
        required=True,
        type=pathlib.Path,
        help="The run's TAO experiment spec file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        args.metric_contract = _validate_metric_gate(
            args.metric_name, args.metric_query_type, args.metric_op, args.metric_target
        )
        args.dataset_root = _resolve_dataset_root(args.dataset_root)
        iaa_runtime = _required_input_dir(
            pathlib.Path(__file__).resolve().parent / "iaa_deft",
            "bundled IAA runtime",
        )
        args.iaa_deft_bundle_sha256 = _python_tree_sha256(iaa_runtime)
        args.workspace = _required_input_dir(args.workspace, "--workspace")
        if args.workspace == pathlib.Path(args.workspace.anchor):
            raise ValueError("--workspace must not be a filesystem root")
        args.results_dir = _workspace_child(
            args.results_dir, args.workspace, "--results-dir"
        )
        args.dataset_root = _workspace_child(
            args.dataset_root, args.workspace, "--dataset-root"
        )
        if (
            args.results_dir in args.dataset_root.parents
            or args.dataset_root in args.results_dir.parents
        ):
            raise ValueError(
                "--results-dir and --dataset-root must not contain one another"
            )
        if args.dataset_root.parent == args.workspace:
            raise ValueError(
                "--dataset-root must be nested below a workspace data directory"
            )
        args.images_archive = _required_input_file(args.images_archive, "--images-archive")
        args.metadata_archive = _required_input_file(args.metadata_archive, "--metadata-archive")
        args.images_archive_sha256 = approved_sha256(
            args.images_archive_sha256, "--images-archive-sha256"
        )
        args.metadata_archive_sha256 = approved_sha256(
            args.metadata_archive_sha256, "--metadata-archive-sha256"
        )
        verify_archive(
            args.images_archive,
            args.images_archive_sha256,
            "--images-archive",
            "--images-archive-sha256",
        )
        verify_archive(
            args.metadata_archive,
            args.metadata_archive_sha256,
            "--metadata-archive",
            "--metadata-archive-sha256",
        )
        if args.checksums_file is not None:
            args.checksums_file = _required_input_file(args.checksums_file, "--checksums-file")
            _validate_checksum_manifest(
                args.checksums_file,
                (args.images_archive, args.metadata_archive),
            )
        args.deft_config = _required_input_file(args.deft_config, "--deft-config")
        args.tao_spec = _required_input_file(args.tao_spec, "--tao-spec")
        args.run_config = _load_run_config(args)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        yaml.YAMLError,
    ) as exc:
        print(f"init_deft_state: {exc}", file=sys.stderr)
        return 2

    positive_ints = {
        "max_iterations": args.max_iterations,
    }
    invalid = {name: value for name, value in positive_ints.items() if value <= 0}
    if invalid:
        detail = ", ".join(f"{name}={value}" for name, value in invalid.items())
        print(
            f"init_deft_state: positive integers required ({detail})",
            file=sys.stderr,
        )
        return 2
    if args.pyt_image != PINNED_PYT_IMAGE or args.ds_image != PINNED_DS_IMAGE:
        print(
            "init_deft_state: --pyt-image and --ds-image must be the pinned "
            "IAA workflow images",
            file=sys.stderr,
        )
        return 2

    out = args.results_dir.expanduser().resolve() / "deft_state.json"
    if out.exists():
        print(
            f"init_deft_state: refusing to reinitialize {out}; deft_state.json "
            "is written exactly once per run. Resume reads disk state — to "
            "start over, choose a fresh --results-dir.",
            file=sys.stderr,
        )
        return 2
    try:
        state = build_state(args)
        write_atomic(out, state)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"init_deft_state: {exc}", file=sys.stderr)
        return 2
    print(f"init_deft_state: wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
