#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single AnomalyGen 1.2 entry point; native commands own model execution."""

from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import yaml
from anomalygen_contract import (
    prepare_refinement,
    finalize_refinement,
    filter_testcases,
    load_exclusions,
    prepare_dataset_view,
    verify_dataset_inventory,
    read_jsonl,
    sha256,
    validate_testcases,
    verify_generation,
)

HERE = Path(__file__).resolve().parent
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def execute(command, args, *, allowed=(0,), environment=None):
    start = time.time()
    result = subprocess.run(
        [str(v) for v in command],
        cwd=args.repo,
        stdout=sys.stderr,
        env={**os.environ, **(environment or {})},
    )
    record = {
        "argv": [str(v) for v in command],
        "started_at_unix": start,
        "ended_at_unix": time.time(),
        "exit_code": result.returncode,
    }
    with (args.output_dir / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if result.returncode not in allowed:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def generation_command(args, testcase, output):
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.num_gpus}",
        str(args.repo / "anomalygen/scripts/texture/generate.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--recipe",
        str(args.recipe),
        "--input_data_path",
        str(testcase),
        "--output_dir",
        str(output),
        "--base_seed",
        str(args.base_seed),
        "--guardrail",
    ]
    if args.base_checkpoint:
        command += ["--base_checkpoint", str(args.base_checkpoint)]
    return command


def training_environment(args):
    return {
        "ANOMALYGEN_PARALLELISM": args.parallelism,
        "ANOMALYGEN_COMPILE": "1" if args.compile == "true" else "0",
    }


def training_commands(args, recipe):
    base = ["bash", str(args.repo / "scripts/skill_utility/train_control.sh")]
    return (
        base
        + [
            "start",
            "--name",
            args.name,
            "--recipe",
            str(recipe),
            "--num_gpus",
            str(args.num_gpus),
            "--scripts",
            str(args.repo / "anomalygen/scripts/texture"),
            "--task",
            "texture_ft",
        ],
        base + ["wait", "--name", args.name, "--timeout", "30", "--interval", "5"],
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode", choices=["prepare", "full", "finetune_only", "inference_only"]
    )
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path("/workspace/cdf-anomalygen"))
    p.add_argument("--source", choices=["texture", "od-gaps"], default="texture")
    p.add_argument(
        "--prepare-stage", choices=["inputs", "amp", "finalize"], default="inputs"
    )
    p.add_argument("--dataset-dir", type=Path)
    p.add_argument("--defect-spec", type=Path)
    p.add_argument("--input-data-path", type=Path)
    p.add_argument("--validation-testcase", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--recipe", type=Path)
    p.add_argument("--base-checkpoint", type=Path)
    p.add_argument("--checkpoint-root", type=Path)
    p.add_argument("--vae-path", type=Path)
    p.add_argument("--exclude-stems", type=Path)
    p.add_argument("--expected-dataset-inventory", type=Path, help="Optional exact relative-path/SHA256/bytes reference inventory")
    p.add_argument("--num-sdg", type=int)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--base-seed", type=int, default=1)
    p.add_argument("--amp-seed", type=int, default=43)
    p.add_argument("--mask-backend", choices=["amp", "amg"], default="amp")
    p.add_argument("--amg-gen-model")
    p.add_argument("--amg-judge-model")
    p.add_argument("--amg-model-registry", type=Path)
    p.add_argument("--per-defect-counts")
    p.add_argument("--num-search-run", type=int, default=3)
    p.add_argument(
        "--score",
        choices=[
            "nn",
            "mnn",
            "completeness",
            "precision",
            "boundary_iou",
            "aq_nn",
            "aq_rank",
        ],
        default="nn",
    )
    p.add_argument(
        "--skip-quality",
        action="store_true",
        help="Generate only; explicitly skip evaluation and refinement (for bounded smoke checks).",
    )
    p.add_argument(
        "--pseudo-label",
        action="store_true",
        help="Run native mask-based COCO export with captions disabled.",
    )
    p.add_argument("--min-nn-improvement", type=float, default=0.0)
    p.add_argument("--parallelism", choices=["ddp", "fsdp"], default="ddp")
    p.add_argument("--compile", choices=["true", "false"], default="true")
    p.add_argument("--config", type=Path)
    p.add_argument("--prepared-root", type=Path)
    p.add_argument("--pool-dataset-root", type=Path)
    p.add_argument("--published-root", type=Path)
    return p


def validate_args(a):
    if not SAFE_NAME.fullmatch(a.name):
        raise ValueError("name must use alphanumeric/dot/underscore/dash")
    if a.num_gpus < 1 or a.num_search_run < 0:
        raise ValueError("invalid GPU or search count")
    if a.mode in ("full", "finetune_only") and a.checkpoint:
        raise ValueError(
            "training modes start from the base; inference adapter checkpoint is not a resume contract"
        )
    if a.mode == "inference_only" and (not a.recipe or not a.recipe.is_file()):
        raise ValueError("inference_only requires the checkpoint training recipe file")
    if a.mode == "inference_only" and (not a.checkpoint or not a.checkpoint.exists()):
        raise ValueError(
            "inference_only requires --checkpoint run directory or iter_N.pt"
        )
    if a.source == "od-gaps":
        if a.mode != "prepare":
            raise ValueError(
                "od-gaps is a prepare source; generation consumes finalized native testcases"
            )
        if a.prepare_stage == "inputs" and not (a.config and a.pool_dataset_root):
            raise ValueError("od-gaps inputs requires --config and --pool-dataset-root")
        if a.prepare_stage != "inputs" and not a.prepared_root:
            raise ValueError("od-gaps continuation requires --prepared-root")
        if a.prepare_stage == "amp" and not (a.pool_dataset_root and a.checkpoint_root):
            raise ValueError("OD AMP requires pool and checkpoint-root bindings")
        return
    if not a.input_data_path and not a.dataset_dir:
        raise ValueError("dataset-dir or input-data-path is required")
    if a.mode in ("full", "finetune_only") and not a.dataset_dir:
        raise ValueError("training requires dataset-dir")
    if a.dataset_dir and not a.dataset_dir.is_dir():
        raise ValueError("dataset-dir does not exist")
    needs_placement = not a.input_data_path or (
        a.mode in ("full", "finetune_only") and not a.validation_testcase
    )
    if needs_placement and (a.num_sdg is None or a.num_sdg < 1):
        raise ValueError("mask placement requires positive num-sdg")
    if (
        a.mask_backend == "amg"
        and needs_placement
        and not all((a.amg_gen_model, a.amg_judge_model, a.amg_model_registry))
    ):
        raise ValueError(
            "AMG requires explicit generator, judge and model-registry; no implicit model service"
        )
    if a.mode == "inference_only" and not a.skip_quality and not a.dataset_dir:
        raise ValueError(
            "quality evaluation requires dataset-dir; bounded generation smoke may use --skip-quality"
        )


def prepare_od(a):
    if a.prepare_stage == "inputs":
        from prepare_od_inputs import prepare

        return prepare(a.config, a.output_dir, a.pool_dataset_root, a.exclude_stems)
    if a.exclude_stems:
        raise ValueError(
            "OD continuation uses its frozen exclusion list; set it at prepare inputs"
        )
    if a.prepare_stage == "amp":
        from run_od_amp import run

        return run(
            a.prepared_root,
            a.checkpoint_root,
            a.pool_dataset_root,
            a.published_root,
            a.repo,
        )
    from finalize_od_inputs import finalize

    return finalize(a.prepared_root)


def place(a, dataset, mode):
    target = a.output_dir / mode
    if a.mask_backend == "amp":
        command = [
            sys.executable,
            "-m",
            "anomalygen.scripts.auto_mask_placement.roi_place_pipeline",
            "--num_sdg",
            str(a.num_sdg),
            "--mode",
            mode,
            "--defect_desc",
            str(a.defect_spec or dataset / "defect_spec.jsonl"),
            "--dataset_dir",
            str(dataset),
            "--output_dir",
            str(target),
            "--seed",
            str(a.amp_seed if mode == "inference" else a.amp_seed - 1),
        ]
        if a.per_defect_counts and mode == "inference":
            command += ["--per_defect_counts", a.per_defect_counts]
    else:
        pairs = (
            yaml.safe_load(a.recipe.read_text())["anomaly_types"]
            if a.recipe
            else __import__("prepare_finetune_recipe")._pairs(dataset)
        )
        command = [
            sys.executable,
            "-m",
            "anomalygen.scripts.agentic_mask_generation.generate",
            "--dataset-path",
            str(dataset),
            "--anomaly-types",
            *[f"{t}+{d}" for t, d in pairs],
            "--gen-model",
            a.amg_gen_model,
            "--judge-model",
            a.amg_judge_model,
            "--model-registry",
            str(a.amg_model_registry),
            "--num-sdg",
            str(a.num_sdg),
            "--mode",
            mode,
            "--visual-prompt",
            "auto",
            "--defect-library",
            "auto",
            "--out-dir",
            str(target),
        ]
        if a.per_defect_counts and mode == "inference":
            command += ["--per-defect-counts", a.per_defect_counts]
    execute(command, a)
    testcase = a.output_dir / f"{mode}-testcase.jsonl"
    filter_testcases(
        target / "testcase.jsonl", testcase, a.exclude_stems, base_dir=a.repo
    )
    rows = read_jsonl(testcase)
    validate_testcases(rows, load_exclusions(a.exclude_stems))
    expected = (
        sum(json.loads(a.per_defect_counts).values())
        if a.per_defect_counts and mode == "inference"
        else a.num_sdg
    )
    if len(rows) != expected:
        raise ValueError(f"mask placement count mismatch: {len(rows)} != {expected}")
    return testcase


def train(a, dataset, validation):
    from prepare_finetune_recipe import _pairs

    template = a.recipe or HERE.parent / "assets/default_recipe.yaml"
    recipe = yaml.safe_load(template.read_text())
    discovered = _pairs(dataset)
    pairs = recipe.get("anomaly_types", discovered)
    if set(map(tuple, pairs)) - set(map(tuple, discovered)):
        raise ValueError("recipe anomaly types absent from filtered dataset")
    covered = {row["anomaly_type"] for row in read_jsonl(validation)}
    if {f"{t}+{d}" for t, d in pairs} - covered:
        raise ValueError("validation must cover every training anomaly type")
    recipe.update(
        dataset_name=a.name,
        dataset_path=str(dataset),
        testcase_jsonl=str(validation),
        anomaly_types=pairs,
        task_type="texture_ft",
        experiment="anomalygen_texture_ft",
        job_name=recipe.get("job_name", "anomalygen_texture_ft"),
    )
    if a.base_checkpoint:
        recipe["base_checkpoint_path"] = str(a.base_checkpoint)
    if a.vae_path:
        recipe["vae_path"] = str(a.vae_path)
    prepared = a.output_dir / "training_recipe.yaml"
    prepared.write_text(yaml.safe_dump(recipe, sort_keys=False))
    run_dir = a.repo / "results/anomalygen" / a.name / recipe["job_name"]
    if run_dir.exists():
        raise FileExistsError(
            "training run already exists; reconcile native lifecycle before resuming, never overwrite it"
        )
    start, wait = training_commands(a, prepared)
    execute(start, a, environment=training_environment(a))
    while execute(wait, a, allowed=(0, 2)) == 2:
        pass
    execute(
        [
            sys.executable,
            str(HERE / "validate_nn_improvement.py"),
            "--run-dir",
            str(run_dir),
            "--recipe",
            str(prepared),
            "--results-dir",
            str(a.output_dir / "training"),
            "--min-improvement",
            str(a.min_nn_improvement),
        ],
        a,
    )
    best = (run_dir / "checkpoints/best_checkpoint.txt").read_text().strip()
    if not re.fullmatch(r"iter_\d+\.pt", best):
        raise ValueError("native best checkpoint pointer is invalid")
    a.checkpoint = run_dir / "checkpoints/model" / best
    run_recipes = list(run_dir.glob("*.yaml"))
    copied = next(
        (p for p in run_recipes if yaml.safe_load(p.read_text()) == recipe), None
    )
    if copied is None:
        raise ValueError("training run lacks an exact saved recipe copy")
    a.recipe = copied
    return {
        "checkpoint": str(a.checkpoint),
        "recipe": str(copied),
        "recipe_sha256": sha256(copied),
        "run_dir": str(run_dir),
    }


def run(a):
    validate_args(a)
    if a.source == "od-gaps":
        return prepare_od(a)
    if a.output_dir.exists():
        raise FileExistsError("output-dir must be new; preserve failed/completed runs")
    if not a.repo.is_dir():
        raise FileNotFoundError(a.repo)
    a.output_dir.mkdir(parents=True)
    started = time.time()
    report = {
        "schema": "anomalygen-run/1",
        "mode": a.mode,
        "name": a.name,
        "guardrail_requested": True,
        "status": "RUNNING",
    }
    write(a.output_dir / "status.json", report)
    try:
        dataset = a.dataset_dir
        expected_inventory = getattr(a, "expected_dataset_inventory", None)
        if expected_inventory:
            expected_inventory = json.loads(expected_inventory.read_text())
            verify_dataset_inventory(dataset, expected_inventory)
        if dataset and a.exclude_stems:
            report["dataset_view"] = prepare_dataset_view(
                dataset, a.output_dir / "dataset-view", a.exclude_stems
            )
            dataset = a.output_dir / "dataset-view"
        if expected_inventory:
            verify_dataset_inventory(dataset, expected_inventory)
            report["expected_dataset_inventory"] = expected_inventory
        if a.mode in ("full", "finetune_only"):
            if a.validation_testcase:
                report["validation_filter"] = filter_testcases(
                    a.validation_testcase,
                    a.output_dir / "validation.jsonl",
                    a.exclude_stems,
                    base_dir=a.repo,
                )
                validation = a.output_dir / "validation.jsonl"
            else:
                validation = place(a, dataset, "validation")
            report["training"] = train(a, dataset, validation)
        if a.mode != "finetune_only":
            if a.input_data_path:
                report["input_filter"] = filter_testcases(
                    a.input_data_path,
                    a.output_dir / "testcase.jsonl",
                    a.exclude_stems,
                    base_dir=a.repo,
                )
                testcase = a.output_dir / "testcase.jsonl"
            else:
                testcase = place(a, dataset, "inference")
            report["testcase"] = str(testcase)
            report["testcase_sha256"] = sha256(testcase)
            if a.mode != "prepare":
                output = a.output_dir / "sdg"
                execute(generation_command(a, testcase, output), a)
                report["generation"] = verify_generation(
                    testcase, output, a.exclude_stems
                )
                timing = output / "timing_summary.json"
                if not timing.is_file():
                    raise ValueError(
                        "native generation timing/guardrail receipt is missing"
                    )
                native = json.loads(timing.read_text())
                if native.get("guardrail_enabled") is not True:
                    raise ValueError("native guardrail was disabled")
                report["native_timing"] = native
                report["checkpoint"] = str(a.checkpoint)
                report["recipe"] = str(a.recipe)
                report["recipe_sha256"] = sha256(a.recipe)
                # Model generation is complete independently of an all-blocked training-ineligible batch.
                if not a.skip_quality and report["generation"]["generated"]:
                    if expected_inventory:
                        verify_dataset_inventory(dataset, expected_inventory)
                    texture = a.repo / "anomalygen/scripts/texture"
                    kpi = output / "kpi.json"
                    execute(
                        [
                            sys.executable,
                            str(texture / "evaluate.py"),
                            "--gen_root",
                            str(output),
                            "--real_root",
                            str(dataset),
                            "--recipe",
                            str(a.recipe),
                            "--output_file",
                            str(kpi),
                        ],
                        a,
                    )
                    if expected_inventory:
                        verify_dataset_inventory(dataset, expected_inventory)
                    search_testcase = a.output_dir / "search-testcase.jsonl"
                    report["refinement_inputs"] = prepare_refinement(
                        testcase, output, search_testcase
                    )
                    command = [
                        sys.executable,
                        str(texture / "quality_refine.py"),
                        "run",
                        "--base_testcase",
                        str(search_testcase),
                        "--original",
                        str(output),
                        "--original_kpi",
                        str(kpi),
                        "--rounds_dir",
                        str(a.output_dir / "rounds"),
                        "--output",
                        str(a.output_dir / "searched"),
                        "--final_kpi",
                        str(a.output_dir / "searched/kpi.json"),
                        "--checkpoint",
                        str(a.checkpoint),
                        "--recipe",
                        str(a.recipe),
                        "--real_root",
                        str(dataset),
                        "--score",
                        a.score,
                        "--num_search_run",
                        str(a.num_search_run),
                        "--num_gpus",
                        str(a.num_gpus),
                        "--base_seed",
                        str(a.base_seed),
                        "--guardrail",
                    ]
                    if a.base_checkpoint:
                        command += ["--base_checkpoint", str(a.base_checkpoint)]
                    execute(command, a)
                    output = a.output_dir / "searched"
                    report["refined"] = finalize_refinement(
                        testcase, a.output_dir / "sdg", output, a.exclude_stems
                    )
                report["quality"] = {
                    "skipped": bool(
                        a.skip_quality or not report["generation"]["generated"]
                    ),
                    "score": a.score,
                    "search_rounds": 0 if a.skip_quality or not report["generation"]["generated"] else a.num_search_run,
                }
                report["selected_output"] = str(output)
                if a.pseudo_label and report["generation"]["generated"]:
                    labels = a.output_dir / "pseudo_labels"
                    execute(
                        [
                            sys.executable,
                            str(a.repo / "anomalygen/scripts/texture/pseudo_label.py"),
                            "--gen_root",
                            str(output),
                            "--output_dir",
                            str(labels),
                            "--no_caption",
                        ],
                        a,
                    )
                    report["pseudo_labels"] = str(labels / "coco_annotations.json")
        report.update(status="COMPLETE", host_wall_seconds=time.time() - started)
        write(a.output_dir / "manifest.json", report)
        write(a.output_dir / "status.json", {"status": "COMPLETE"})
        return report
    except Exception as error:
        write(
            a.output_dir / "status.json",
            {
                "status": "ERROR",
                "message": str(error),
                "host_wall_seconds": time.time() - started,
            },
        )
        raise


def main():
    a = parser().parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.expanduser().resolve())
    print(json.dumps(run(a), sort_keys=True))


if __name__ == "__main__":
    main()
