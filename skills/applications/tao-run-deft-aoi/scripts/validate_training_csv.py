# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate an assembled ChangeNet training CSV before launching training.

Why this exists: `augmentation/mining_pool/mining_pool.csv` is
append-only and accumulates production-line samples daily; rows can reference
images that were deleted, moved, or never staged. Launching ChangeNet training
on a CSV with broken `input_path` / `golden_path` wastes a GPU run because the
TAO container only fails per-batch and surfaces the root cause minutes in.

This script:

1. Reads the assembled training CSV, resolves every `input_path` and
   `golden_path` against a workspace root, with a CSV-directory fallback for
   flat source pools (or treats absolute paths as-is), and hard-stops on any
   missing file or schema error.
2. Enforces the PASS-preserving label rule: `label == "PASS"` must stay
   uppercase; every other label must be lowercase + stripped. Non-compliant
   rows hard-stop because TAO's ChangeNet classify dataloader does
   case-sensitive equality against the literal string "PASS" to detect
   class 0; any deviation produces silent class-collapse failures at
   training start.
3. Optionally diffs the training CSV against a validation CSV (when
   `--validation-csv` is supplied) on `(input_path, golden_path, label,
   object_name, boardname)` where present. Any validation row appearing
   in training is a hard-stop train/val leak — running this BEFORE CSV
   assembly is finalized lets the orchestrator avoid a wasted GPU run.

Exit code 2 on any validation failure; 0 on success.

CLI:

    python scripts/validate_training_csv.py \
        --csv ${RESULTS_DIR}/iter${N}/dataset/train_combined_iter${N}.csv \
        --workspace-root ~/workspace \
        [--validation-csv ~/workspace/train/base/validation_set.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

_shared_candidates = (pathlib.Path(__file__).resolve().parents[2] / "tao-generate-anomalies/scripts",
                      pathlib.Path(__file__).resolve().parents[3] / "data/tao-generate-anomalies/scripts")
sys.path.insert(0, str(next((p for p in _shared_candidates if p.is_dir()), _shared_candidates[-1])))
from anomalygen_consumer import validate_lineage, load_generation
from anomalygen_contract import sha256, load_exclusions, _excluded

_REQUIRED_COLUMNS = ("input_path", "golden_path", "label", "object_name")
_PATH_COLUMNS = ("input_path", "golden_path")
_LEAK_KEY_CANDIDATES = (
    "input_path",
    "golden_path",
    "label",
    "object_name",
    "boardname",
)


def _resolve(p: str, workspace_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(p)
    if path.is_absolute():
        return path
    return workspace_root / path


def normalize_label(label: str) -> str:
    """Preserve 'PASS' verbatim; lowercase + strip every other label."""
    if label == "PASS":
        return label
    return label.lower().strip()


def _check_label_case(rows: list[dict]) -> list[str]:
    """Return rows whose label is not in the canonical case.

    We compare the raw value (no caller-side strip) against normalize_label's
    output so trailing whitespace counts as non-canonical. The whole point of
    the normalization rule is that the on-disk row matches what the dataloader
    sees byte-for-byte — silently stripping here would mask the bug.
    """
    bad: list[tuple[int, str]] = []
    for i, row in enumerate(rows):
        raw = row.get("label") or ""
        if not raw.strip():
            bad.append((i, "<empty label>"))
            continue
        if raw != normalize_label(raw):
            bad.append((i, raw))
    if not bad:
        return []
    sample = ", ".join(f"row {i}: {p!r}" for i, p in bad[:5])
    return [
        f"{len(bad)} row(s) have non-canonical label case "
        f"(must be 'PASS' verbatim or lowercase+stripped); first: {sample}"
    ]


def _check_leakage(
    train_rows: list[dict],
    train_cols: list[str],
    validation_csv: pathlib.Path,
) -> list[str]:
    if not validation_csv.is_file():
        return [f"--validation-csv not found: {validation_csv}"]
    with validation_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        val_cols = reader.fieldnames or []
        val_rows = list(reader)

    join_keys = [k for k in _LEAK_KEY_CANDIDATES if k in train_cols and k in val_cols]
    if not join_keys:
        return [
            f"--validation-csv has no shared columns with training CSV "
            f"(tried {list(_LEAK_KEY_CANDIDATES)}); cannot leakage-check"
        ]

    def _key(row: dict) -> tuple:
        return tuple((row.get(k) or "").strip() for k in join_keys)

    val_keys = {_key(r) for r in val_rows}
    leaks: list[tuple[int, tuple]] = [
        (i, _key(r)) for i, r in enumerate(train_rows) if _key(r) in val_keys
    ]
    if not leaks:
        return []
    sample = ", ".join(f"row {i}: {k}" for i, k in leaks[:5])
    return [
        f"{len(leaks)} train/val leak(s) on keys {join_keys}; first: {sample}"
    ]


def validate(
    csv_path: pathlib.Path,
    workspace_root: pathlib.Path,
    validation_csv: pathlib.Path | None = None,
    light: str = "SolderLight",
    image_ext: str = ".jpg",
    exclude_stems: pathlib.Path | None = None,
    synthetic_csvs=(),
) -> list[str]:
    """Return a list of human-readable validation errors (empty == valid).

    Uses stdlib csv so the script runs on bare hosts without pandas.

    A path that already resolves to a file under the workspace root or beside
    the CSV is accepted as a flat-file source-pool row. Otherwise, path
    resolution follows TAO ChangeNet's siamese dataloader convention when
    `object_name` is present in the CSV:
        <workspace_root>/<input_path>/<object_name>_<light><image_ext>
    This lets the raw smoke-test mining pool use `images/<file>.jpg` while
    assembled training rows continue to use directory-style paths.
    """
    errors: list[str] = []

    if not csv_path.is_file():
        return [f"CSV not found: {csv_path}"]

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in columns]
    if missing_cols:
        errors.append(
            f"missing required column(s): {missing_cols}; got {list(columns)}"
        )
        # Continue so the user sees both schema and path errors in one shot.

    if not rows:
        errors.append("CSV is empty (0 data rows)")

    resolved_rows = [{} for _ in rows]
    siamese_mode = "object_name" in columns
    for col in _PATH_COLUMNS:
        if col not in columns:
            continue
        missing: list[tuple[int, str]] = []
        for i, row in enumerate(rows):
            raw = (row.get(col) or "").strip()
            if not raw:
                missing.append((i, f"<empty {col}>"))
                continue
            raw_path = pathlib.Path(raw)
            direct_candidates = [_resolve(raw, workspace_root)]
            if not raw_path.is_absolute():
                direct_candidates.append(csv_path.parent / raw_path)
            direct_file = next(
                (candidate for candidate in direct_candidates if candidate.is_file()),
                None,
            )

            if direct_file is not None:
                resolved = direct_file
            elif siamese_mode:
                obj = (row.get("object_name") or "").strip()
                if not obj:
                    missing.append((i, f"<empty object_name for siamese {col}>"))
                    continue
                # TAO siamese resolution: images_dir/input_path/object_name_light.ext
                filename = f"{obj}_{light}{image_ext}"
                reconstructed = [base / filename for base in direct_candidates]
                resolved = next(
                    (candidate for candidate in reconstructed if candidate.is_file()),
                    reconstructed[0],
                )
            else:
                resolved = direct_candidates[0]
            resolved_rows[i][col] = resolved.resolve()
            if not resolved.is_file():
                missing.append((i, f"{raw} -> {resolved}"))
        if missing:
            sample = ", ".join(f"row {i}: {p!r}" for i, p in missing[:5])
            errors.append(
                f"{len(missing)} row(s) reference a missing {col} on disk "
                f"(workspace_root={workspace_root}, siamese={siamese_mode}); first: {sample}"
            )
            image_prefix = next(
                (
                    prefix
                    for prefix in (pathlib.Path("images"), pathlib.Path("kpi/images"))
                    if any(
                        not pathlib.Path((row.get(col) or "").strip()).is_absolute()
                        and (workspace_root / prefix / (row.get(col) or "").strip()).exists()
                        for row in rows
                        if (row.get(col) or "").strip()
                    )
                ),
                None,
            )
            if image_prefix is not None:
                errors.append(
                    f"{col} appears to use base-CSV coordinates; prepend "
                    f"'{image_prefix.as_posix()}/' exactly once before assembling "
                    "an iteration CSV"
                )

    forbidden = set()
    if validation_csv is not None and validation_csv.is_file():
        with validation_csv.open(newline="") as stream:
            for row in csv.DictReader(stream):
                forbidden.update(str(_resolve(row[key], workspace_root)) for key in _PATH_COLUMNS if row.get(key))
                if row.get("object_name"):
                    forbidden.add(row["object_name"])
    try:
        sources = list(synthetic_csvs)
        if csv_path.with_name(csv_path.name + ".anomalygen.json").is_file() and csv_path not in sources:
            sources.append(csv_path)
        bindings = []
        for source in sources:
            source = pathlib.Path(source)
            receipt = json.loads(source.with_name(source.name + ".anomalygen.json").read_text())
            if receipt.get("output_sha256") != sha256(source):
                raise ValueError("synthetic source CSV changed after emission")
            report, native = load_generation(receipt["manifest"], exclude_stems)
            current = receipt.get("pair_bindings", [])
            expected = {json.dumps(row["lineage"], sort_keys=True) for row in native}
            if len(current) != report["generated"] or {json.dumps(b["lineage"], sort_keys=True) for b in current} != expected:
                raise ValueError("synthetic source lacks complete converted pair bindings")
            for binding in current:
                validate_lineage(binding["lineage"], forbidden, exclude_stems)
            bindings.extend(current)
        used = set()
        excluded = load_exclusions(exclude_stems)
        for row, resolved in zip(rows, resolved_rows):
            for path in resolved.values():
                if _excluded(path, excluded):
                    raise ValueError(f"training pair uses excluded image: {path}")
            candidates = [b for b in bindings if row.get("object_name") == b["object_name"] or
                          any(str(resolved.get(key)) == b[key]["path"] for key in _PATH_COLUMNS)]
            raw = row.get("anomalygen_lineage")
            lineage = json.loads(raw) if raw else None
            if lineage:
                validate_lineage(lineage, forbidden, exclude_stems)
                candidates.extend(b for b in bindings if b["lineage"] == lineage and b not in candidates)
                if not candidates:
                    raise ValueError("synthetic row requires authoritative --synthetic-csv source")
            if candidates:
                matches = [b for b in candidates if b["lineage"] == lineage and b["label"] == row.get("label") and
                           b["object_name"] == row.get("object_name") and
                           all(key in resolved and str(resolved[key]) == b[key]["path"] and
                               sha256(resolved[key]) == b[key]["sha256"] for key in _PATH_COLUMNS)]
                if len(matches) != 1:
                    raise ValueError("synthetic training pair/label/lineage differs from emission binding")
                used.add(json.dumps(matches[0], sort_keys=True))
        if used != {json.dumps(b, sort_keys=True) for b in bindings}:
            raise ValueError("assembled training CSV omits authoritative synthetic rows")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    if "label" in columns:
        errors.extend(_check_label_case(rows))

    if validation_csv is not None:
        errors.extend(_check_leakage(rows, list(columns), validation_csv))

    return errors


def state_synthetic_sources(state_path, iter_label):
    state_path = pathlib.Path(state_path)
    state = json.loads(state_path.read_text())
    limit = int(iter_label.removeprefix("iter"))
    sources = []
    for label, phase in state.get("iterations", {}).items():
        if not label.startswith("iter") or int(label[4:]) > limit:
            continue
        manifest = phase.get("anomalygen_manifest")
        if not manifest:
            if phase.get("anomalygen_training_ready"):
                raise ValueError("synthetic state lacks native manifest")
            continue
        report, _ = load_generation(manifest)
        if not report["generated"]:
            continue
        source = state_path.parent / label / "mining_filter/sdg_rows_raw.csv"
        receipt = json.loads(source.with_name(source.name + ".anomalygen.json").read_text())
        if pathlib.Path(receipt["manifest"]).resolve() != pathlib.Path(manifest).resolve():
            raise ValueError("synthetic source manifest differs from committed generation")
        sources.append(source)
    return sources


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an assembled ChangeNet training CSV: schema + existence "
            "of every input_path / golden_path, PASS-preserving label case, "
            "and (optionally) train/val leakage. Call this between CSV "
            "assembly and the training docker invocation."
        ),
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=pathlib.Path,
        help="Absolute path to the assembled training CSV.",
    )
    parser.add_argument(
        "--workspace-root",
        required=True,
        type=pathlib.Path,
        help=(
            "Absolute workspace root. Relative input_path / golden_path values "
            "are resolved against this directory, with a CSV-directory fallback "
            "for flat source-pool files; absolute values are used as-is."
        ),
    )
    parser.add_argument(
        "--validation-csv",
        required=False,
        default=None,
        type=pathlib.Path,
        help=(
            "Optional validation CSV. When supplied, the script diffs the "
            "training CSV against it on (input_path, golden_path, label, "
            "object_name, boardname) where present and hard-stops on any "
            "validation row that appears in training."
        ),
    )
    parser.add_argument(
        "--light",
        default="SolderLight",
        help=(
            "Lighting suffix for TAO siamese path resolution: "
            "<input_path>/<object_name>_<light><image_ext>. Default: SolderLight."
        ),
    )
    parser.add_argument(
        "--image-ext",
        default=".jpg",
        help="Image extension for siamese path resolution. Default: .jpg.",
    )
    parser.add_argument(
        "--report-json",
        type=pathlib.Path,
        help=(
            "Write a machine-verifiable success report containing rows_checked, "
            "missing_file_count, and train_val_leakage_overlap_count."
        ),
    )
    parser.add_argument("--exclude-stems", type=pathlib.Path)
    parser.add_argument("--synthetic-csv", action="append", default=[], type=pathlib.Path, help="Immutable emitted CSV; repeat for every contributing iteration")
    parser.add_argument("--state", type=pathlib.Path, help="Derive mandatory synthetic sources from durable DEFT state")
    parser.add_argument("--iter-label", help="Iteration paired with --state")
    return parser


def _write_success_report(args: argparse.Namespace) -> None:
    if args.report_json is None:
        return
    with args.csv.open(newline="") as f:
        rows_checked = sum(1 for _ in csv.DictReader(f))
    report = {
        "status": "ok",
        "training_csv_sha256": sha256(args.csv),
        "workspace_root": str(args.workspace_root.resolve()),
        "light": args.light, "image_ext": args.image_ext,
        "synthetic_csvs": [str(p.resolve()) for p in args.synthetic_csv],
        "training_csv": str(args.csv.resolve()),
        "validation_csv": (
            str(args.validation_csv.resolve())
            if args.validation_csv is not None
            else None
        ),
        "rows_checked": rows_checked,
        "missing_file_count": 0,
        "train_val_leakage_overlap_count": (
            0 if args.validation_csv is not None else None
        ),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.report_json.with_name(f".{args.report_json.name}.tmp")
    tmp_path.write_text(json.dumps(report, indent=2) + "\n")
    tmp_path.replace(args.report_json)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.state) != bool(args.iter_label):
        raise ValueError("--state requires --iter-label")
    if args.state:
        args.synthetic_csv.extend(state_synthetic_sources(args.state, args.iter_label))
    args.synthetic_csv = list(dict.fromkeys(args.synthetic_csv))
    errors = validate(
        args.csv,
        args.workspace_root,
        args.validation_csv,
        light=args.light,
        image_ext=args.image_ext,
        exclude_stems=args.exclude_stems,
        synthetic_csvs=args.synthetic_csv,
    )
    if errors:
        print(
            f"validate_training_csv: FATAL — {len(errors)} issue(s) in {args.csv}",
            file=sys.stderr,
        )
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2
    _write_success_report(args)
    print(f"validate_training_csv: ok ({args.csv})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
