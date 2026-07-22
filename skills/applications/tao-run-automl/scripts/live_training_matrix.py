# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run every skill-owned AutoML policy against real TAO GPU training jobs.

This is an opt-in live regression harness, not a unit test.  It creates a tiny
two-class image dataset, drives ``automl_step.py`` through its public CLI, opens
canonical SDK-free TAO job records, launches ``classification_pyt`` in Docker,
reports the observed validation metric and checkpoint, and finalizes every
experiment.  Results and logs remain under ``--workspace``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
AUTOML_STEP = SCRIPT_DIR / "automl_step.py"
JOB_RECORD = REPO_ROOT / "scripts" / "tao_job_record.py"
MODEL_DIR = REPO_ROOT / "skills" / "models" / "tao-train-image-classification"
TRAIN_SCHEMA = MODEL_DIR / "schemas" / "train.schema.json"
TRAIN_TEMPLATE = MODEL_DIR / "references" / "spec_template_train.yaml"
EVALUATE_TEMPLATE = MODEL_DIR / "references" / "spec_template_evaluate.yaml"

ALGORITHMS = (
    "bayesian",
    "bfbo",
    "hyperband",
    "bohb",
    "asha",
    "pbt",
    "dehb",
    "hyperband_es",
    "llm",
    "hybrid",
    "autoresearch",
)

METRIC_RE = re.compile(
    r"(?:val_acc_1|val_accuracy|accuracy)[\"']?\s*[:=]\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=check,
    )


def _run_json(command: list[str], *, env: dict[str, str] | None = None) -> dict:
    result = _run(command, env=env)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Command did not return JSON: {' '.join(command)}\n{result.stdout}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Command returned a non-object: {' '.join(command)}")
    return value


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _container_path(workspace: Path, host_path: Path) -> str:
    return "/workspace/" + host_path.resolve().relative_to(workspace.resolve()).as_posix()


def _prepare_dataset(workspace: Path) -> None:
    """Generate a deterministic, learnable two-class classification fixture."""
    dataset = workspace / "data"
    classes = ("red", "blue")
    for split, count in (("train", 12), ("val", 6)):
        for class_index, class_name in enumerate(classes):
            target = dataset / split / class_name
            target.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                background = (210, 30, 30) if class_index == 0 else (30, 30, 210)
                image = Image.new("RGB", (224, 224), background)
                draw = ImageDraw.Draw(image)
                offset = 12 + (index * 7) % 80
                accent = (255, 220, 80) if class_index == 0 else (80, 220, 255)
                draw.rectangle((offset, offset, offset + 80, offset + 80), fill=accent)
                draw.text((16, 190), f"{class_name}-{split}-{index}", fill=(255, 255, 255))
                image.save(target / f"image-{index:02d}.png")
    (dataset / "classes.txt").write_text("red\nblue\n", encoding="utf-8")


def _disabled_augmentation() -> dict:
    return {
        "random_flip": {
            "vflip_probability": 0.0,
            "hflip_probability": 0.0,
            "enable": False,
        },
        "random_rotate": {
            "rotate_probability": 0.0,
            "angle_list": [90],
            "enable": False,
        },
        "random_color": {
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0,
            "enable": False,
            "color_probability": 0.0,
        },
        "random_erase": {"enable": False, "erase_probability": 0.0},
        "random_aug": {"enable": False},
        "with_scale_random_crop": {"scale_range": [1.0, 1.0], "enable": False},
        "with_random_blur": False,
        "with_random_crop": False,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "mixup_cutmix": False,
        "mixup_alpha": 0.4,
    }


def _base_spec() -> dict:
    spec = yaml.safe_load(TRAIN_TEMPLATE.read_text(encoding="utf-8"))
    spec["model_name"] = "automl-live-classifier"
    spec["results_dir"] = "/workspace/results/pending"
    spec["wandb"]["enable"] = False
    spec["model"]["backbone"].update(
        {"type": "fan_tiny_8_p4_hybrid", "pretrained_backbone_path": ""}
    )
    spec["model"]["head"]["in_channels"] = 192
    spec["dataset"].update(
        {
            "root_dir": "/workspace/data",
            "num_classes": 2,
            "img_size": 224,
            "batch_size": 8,
            "workers": 0,
            "classes_file": "/workspace/data/classes.txt",
            "augmentation": _disabled_augmentation(),
        }
    )
    spec["dataset"]["train_dataset"]["images_dir"] = "/workspace/data/train"
    spec["dataset"]["val_dataset"]["images_dir"] = "/workspace/data/val"
    spec["dataset"]["test_dataset"]["images_dir"] = "/workspace/data/val"
    spec["train"].update(
        {
            "num_epochs": 2,
            "checkpoint_interval": 1,
            "validation_interval": 1,
            "results_dir": "/workspace/results/pending",
            "resume_training_checkpoint_path": "",
            "precision": "fp32",
        }
    )
    spec["train"]["optim"].update(
        {
            "lr": 0.0003,
            "policy": "linear",
            "warmup_epochs": 0,
            "weight_decay": 0.01,
        }
    )
    return spec


def _find_values(value: Any, names: set[str]) -> list[float]:
    found: list[float] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in names and isinstance(child, (int, float)) and not isinstance(child, bool):
                found.append(float(child))
            found.extend(_find_values(child, names))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_values(child, names))
    return found


def _extract_metric(results_dir: Path, output: str) -> float:
    values: list[float] = []
    for path in results_dir.rglob("*.json"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            values.extend(_find_values(payload, {"val_acc_1", "val_accuracy", "accuracy"}))
    values.extend(float(match.group(1)) for match in METRIC_RE.finditer(output))
    finite = [value for value in values if value == value and abs(value) != float("inf")]
    if not finite:
        raise RuntimeError(f"No validation accuracy found under {results_dir}")
    return finite[-1]


def _checkpoint(results_dir: Path) -> Path:
    checkpoints = sorted(
        (path for path in results_dir.rglob("*.pth") if "epoch" in path.name),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not checkpoints:
        raise RuntimeError(f"No epoch checkpoint produced under {results_dir}")
    return checkpoints[-1]


class _LLMStub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                length = int(self.headers.get("Content-Length", "0"))
                envelope = json.loads(self.rfile.read(length))
                prompt = json.loads(envelope["messages"][1]["content"])
                with owner._lock:
                    owner._count += 1
                    count = owner._count
                if str(prompt["purpose"]).startswith("hybrid_"):
                    content = {
                        "algorithm": "bfbo" if count % 2 else "bayesian",
                        "reason": "live deterministic exploration/refinement plan",
                    }
                else:
                    dimensions = len(prompt["parameters"])
                    unit = ((count * 29) % 89 + 5) / 100.0
                    content = {
                        "normalized_vector": [unit] * dimensions,
                        "reason": "live deterministic proposal informed by reported jobs",
                    }
                body = json.dumps(
                    {
                        "choices": [{"message": {"content": json.dumps(content)}}],
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 10,
                            "total_tokens": 30,
                        },
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self) -> "_LLMStub":
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.thread.join()


class LiveMatrix:
    def __init__(self, args: argparse.Namespace, llm_endpoint: str) -> None:
        self.args = args
        self.workspace = args.workspace.resolve()
        self.llm_endpoint = llm_endpoint
        self.env = os.environ.copy()
        self.env["TAO_STATE_DIR"] = str(self.workspace / ".tao")
        self.checkpoints: dict[tuple[str, str], Path] = {}
        self.job_ids: dict[tuple[str, str], str] = {}
        self._gpu_lock = threading.Lock()
        self._available_gpus = list(args.gpus)

    def prepare(self) -> None:
        if self.workspace.exists() and self.args.clean:
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        _prepare_dataset(self.workspace)
        _write_yaml(self.workspace / "base_spec.yaml", _base_spec())
        probe = _run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                f"device={self.args.gpus[0]}",
                self.args.image,
                "python",
                "-c",
                "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
            ]
        )
        print(f"GPU probe: {probe.stdout.strip().splitlines()[-1]}", flush=True)

    def _engine(self, *arguments: str) -> dict:
        return _run_json([sys.executable, str(AUTOML_STEP), *arguments], env=self.env)

    def _init_algorithm(self, algorithm: str) -> Path:
        directory = self.workspace / "experiments" / algorithm
        directory.mkdir(parents=True, exist_ok=True)
        state = directory / "experiment.json"
        if state.exists():
            return state
        max_recommendations = 3 if algorithm in {"bayesian", "bfbo"} else 2
        max_epochs = 4 if algorithm in {"bohb", "dehb"} else 2
        max_experiments = 4 if algorithm == "hybrid" else 2
        command = [
            "init",
            "--state",
            str(state),
            "--schema",
            str(TRAIN_SCHEMA),
            "--base-spec",
            str(self.workspace / "base_spec.yaml"),
            "--parameter",
            "train.optim.lr",
            "--experiment-id",
            f"live-{algorithm}",
            "--network-arch",
            "classification_pyt",
            "--algorithm",
            algorithm,
            "--metric",
            "val_acc_1",
            "--direction",
            "maximize",
            "--max-recommendations",
            str(max_recommendations),
            "--max-concurrent",
            str(len(self.args.gpus)),
            "--candidate-count",
            "64",
            "--seed",
            str(700 + ALGORITHMS.index(algorithm)),
            "--max-epochs",
            str(max_epochs),
            "--reduction-factor",
            "2",
            "--population-size",
            "2",
            "--max-generations",
            "2",
            "--eval-interval",
            "1",
            "--max-trials",
            "2",
            "--min-points-in-model",
            "2",
            "--min-early-stop-epochs",
            "1",
            "--max-experiments",
            str(max_experiments),
        ]
        if algorithm in {"llm", "hybrid", "autoresearch"}:
            command.extend(
                [
                    "--llm-endpoint",
                    self.llm_endpoint,
                    "--llm-model",
                    "live-local-stub",
                    "--llm-max-retries",
                    "1",
                    "--llm-retry-delay",
                    "0",
                ]
            )
        if self.args.wandb and algorithm == "bayesian":
            command.extend(
                [
                    "--wandb",
                    "--wandb-mode",
                    "offline",
                    "--wandb-project",
                    "tao-automl-live-regression",
                ]
            )
        self._engine(*command)
        return state

    def _acquire_gpu(self) -> int:
        while True:
            with self._gpu_lock:
                if self._available_gpus:
                    return self._available_gpus.pop(0)
            time.sleep(0.05)

    def _release_gpu(self, gpu: int) -> None:
        with self._gpu_lock:
            self._available_gpus.append(gpu)

    def _job_record(self, algorithm: str, rec: dict) -> tuple[str, Path]:
        result_root = self.workspace / "results" / algorithm
        command = [
            sys.executable,
            str(JOB_RECORD),
            "open",
            "--platform",
            "docker",
            "--image",
            self.args.image,
            "--network-arch",
            "classification_pyt",
            "--action",
            "train",
            "--storage-tier",
            "A",
            "--results-root",
            str(result_root),
        ]
        parent = rec.get("parent_rec_id")
        if parent:
            parent_job_id = self.job_ids.get((algorithm, parent)) or rec.get(
                "resume_from_job_id"
            )
            if not parent_job_id:
                raise RuntimeError(f"{algorithm}/{rec['id']} has no parent job id")
            command.extend(["--parent-job", parent_job_id])
        job_id = _run(command, env=self.env).stdout.strip()
        return job_id, result_root / job_id

    def _mark(self, job_id: str, state: str, *extra: str) -> None:
        _run(
            [
                sys.executable,
                str(JOB_RECORD),
                "mark",
                job_id,
                "--state",
                state,
                *extra,
            ],
            env=self.env,
        )

    def _run_trial(self, algorithm: str, state: Path, rec: dict) -> dict:
        gpu = self._acquire_gpu()
        job_id = ""
        try:
            job_id, results_dir = self._job_record(algorithm, rec)
            results_dir.mkdir(parents=True, exist_ok=True)
            spec = copy.deepcopy(rec["spec"])
            container_results = _container_path(self.workspace, results_dir)
            spec["results_dir"] = container_results
            spec["train"]["results_dir"] = container_results
            parent = rec.get("parent_rec_id")
            if parent:
                parent_checkpoint = self.checkpoints.get((algorithm, parent))
                if parent_checkpoint is None:
                    state_payload = json.loads(state.read_text(encoding="utf-8"))
                    parent_record = next(
                        row
                        for row in state_payload["recommendations"]
                        if row["id"] == parent
                    )
                    parent_checkpoint = Path(parent_record["checkpoint_uri"])
                spec["train"]["resume_training_checkpoint_path"] = _container_path(
                    self.workspace, parent_checkpoint
                )
            spec_path = self.workspace / "specs" / algorithm / f"{rec['id']}-{job_id}.yaml"
            _write_yaml(spec_path, spec)
            self._engine(
                "bind-job",
                "--state",
                str(state),
                "--rec-id",
                rec["id"],
                "--job-id",
                job_id,
            )
            container_name = f"automl-{algorithm}-{job_id}".replace("_", "-")[:120]
            self._mark(
                job_id,
                "RUNNING",
                "--backend-ref",
                container_name,
                "--message",
                f"AutoML {algorithm} {rec['id']} on physical GPU {gpu}",
            )
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "--gpus",
                f"device={gpu}",
                "--ipc=host",
                "--ulimit",
                "memlock=-1",
                "--ulimit",
                "stack=67108864",
                "-v",
                f"{self.workspace}:/workspace",
                self.args.image,
                "classification_pyt",
                "train",
                "-e",
                _container_path(self.workspace, spec_path),
            ]
            started = time.monotonic()
            result = _run(command, check=False)
            duration = time.monotonic() - started
            log_path = self.workspace / "logs" / algorithm / f"{rec['id']}-{job_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(result.stdout, encoding="utf-8")
            if result.returncode != 0:
                self._mark(
                    job_id,
                    "ERROR",
                    "--err-class",
                    "ERR_PROGRAM",
                    "--message",
                    f"classification_pyt exited {result.returncode}; see {log_path}",
                )
                self._engine(
                    "report",
                    "--state",
                    str(state),
                    "--rec-id",
                    rec["id"],
                    "--outcome",
                    "ERR_PROGRAM",
                    "--job-id",
                    job_id,
                    "--message",
                    f"classification_pyt exited {result.returncode}",
                )
                raise RuntimeError(
                    f"{algorithm}/{rec['id']} failed; inspect {log_path}\n"
                    + "\n".join(result.stdout.splitlines()[-30:])
                )
            metric = _extract_metric(results_dir, result.stdout)
            checkpoint = _checkpoint(results_dir)
            feedback_path = self.workspace / "feedback" / algorithm / f"{rec['id']}.json"
            _write_json(
                feedback_path,
                {
                    "runtime_seconds": round(duration, 3),
                    "gpu": gpu,
                    "validation": {"val_acc_1": metric},
                    "checkpoint_bytes": checkpoint.stat().st_size,
                },
            )
            metric_record_path = (
                self.workspace / "metric-records" / algorithm / f"{rec['id']}.json"
            )
            feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
            _write_json(
                metric_record_path,
                {
                    "schema_version": 1,
                    "experiment_id": f"live-{algorithm}",
                    "rec_id": rec["id"],
                    "job_id": job_id,
                    "status": "COMPLETE",
                    "primary_metric": "val_acc_1",
                    "direction": "maximize",
                    "metrics": {"val_acc_1": metric},
                    "artifacts": {
                        "checkpoint_uri": str(checkpoint),
                        "metrics_uri": str(metric_record_path),
                        "output_uris": [str(results_dir)],
                    },
                    "failure": None,
                    "measured_at": datetime.now(timezone.utc).isoformat(),
                    "feedback": feedback,
                },
            )
            if algorithm == "hyperband_es":
                self._engine(
                    "observe",
                    "--state",
                    str(state),
                    "--rec-id",
                    rec["id"],
                    "--job-id",
                    job_id,
                    "--step",
                    str(rec.get("budget") or 1),
                    "--metric-value",
                    f"val_acc_1={metric}",
                    "--feedback",
                    str(feedback_path),
                )
            self._mark(
                job_id,
                "COMPLETE",
                "--message",
                f"val_acc_1={metric:.6f}; checkpoint={checkpoint.name}",
            )
            report = self._engine(
                "report",
                "--state",
                str(state),
                "--rec-id",
                rec["id"],
                "--metric-record",
                str(metric_record_path),
            )
            self.checkpoints[(algorithm, rec["id"])] = checkpoint
            self.job_ids[(algorithm, rec["id"])] = job_id
            print(
                f"{algorithm:14s} {rec['id']} budget={rec.get('budget')} "
                f"gpu={gpu} val_acc_1={metric:.4f} {duration:.1f}s",
                flush=True,
            )
            return report
        except Exception:
            if job_id:
                # The record may already be terminal; immutable-terminal failures
                # are intentionally ignored while preserving the original error.
                try:
                    self._mark(
                        job_id,
                        "ERROR",
                        "--err-class",
                        "ERR_INFRA",
                        "--message",
                        "live matrix harness failure",
                    )
                except subprocess.CalledProcessError:
                    pass
            raise
        finally:
            self._release_gpu(gpu)

    def _evaluate_best(self, algorithm: str, best_rec: dict) -> dict:
        checkpoint = Path(best_rec["best"]["checkpoint_uri"])
        parent_job_id = next(
            row["job_id"]
            for row in best_rec["all_recs"]
            if row["rec_id"] == best_rec["best"]["rec_id"]
        )
        result_root = self.workspace / "final-evaluation" / algorithm
        job_id = _run(
            [
                sys.executable,
                str(JOB_RECORD),
                "open",
                "--platform",
                "docker",
                "--image",
                self.args.image,
                "--network-arch",
                "classification_pyt",
                "--action",
                "evaluate",
                "--storage-tier",
                "A",
                "--results-root",
                str(result_root),
                "--parent-job",
                parent_job_id,
            ],
            env=self.env,
        ).stdout.strip()
        result_dir = result_root / job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        spec = yaml.safe_load(EVALUATE_TEMPLATE.read_text(encoding="utf-8"))
        base = _base_spec()
        for key in ("model", "dataset"):
            spec[key] = base[key]
        spec["wandb"]["enable"] = False
        spec["results_dir"] = _container_path(self.workspace, result_dir)
        spec["evaluate"]["results_dir"] = _container_path(self.workspace, result_dir)
        spec["evaluate"]["checkpoint"] = _container_path(self.workspace, checkpoint)
        spec_path = self.workspace / "specs" / algorithm / "final-evaluate.yaml"
        _write_yaml(spec_path, spec)
        gpu: int | None = None
        terminal = False
        try:
            gpu = self._acquire_gpu()
            container_name = f"automl-{algorithm}-final-{job_id}".replace("_", "-")[:120]
            self._mark(
                job_id,
                "RUNNING",
                "--backend-ref",
                container_name,
                "--message",
                f"Final evaluation for AutoML {algorithm}",
            )
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "--gpus",
                f"device={gpu}",
                "--ipc=host",
                "-v",
                f"{self.workspace}:/workspace",
                self.args.image,
                "classification_pyt",
                "evaluate",
                "-e",
                _container_path(self.workspace, spec_path),
            ]
            result = _run(command, check=False)
            log_path = self.workspace / "logs" / algorithm / "final-evaluate.log"
            log_path.write_text(result.stdout, encoding="utf-8")
            if result.returncode != 0:
                self._mark(
                    job_id,
                    "ERROR",
                    "--err-class",
                    "ERR_PROGRAM",
                    "--message",
                    f"classification_pyt evaluate exited {result.returncode}",
                )
                terminal = True
                raise RuntimeError(
                    f"Final evaluation failed for {algorithm}; inspect {log_path}\n"
                    + "\n".join(result.stdout.splitlines()[-30:])
                )
            metric = _extract_metric(result_dir, result.stdout)
            metric_record_path = result_dir / "metric_record.json"
            _write_json(
                metric_record_path,
                {
                    "schema_version": 1,
                    "experiment_id": best_rec["experiment_id"],
                    "rec_id": best_rec["best"]["rec_id"],
                    "job_id": job_id,
                    "status": "COMPLETE",
                    "primary_metric": "val_acc_1",
                    "direction": "maximize",
                    "metrics": {"val_acc_1": metric},
                    "artifacts": {
                        "checkpoint_uri": str(checkpoint),
                        "metrics_uri": str(metric_record_path),
                        "output_uris": [str(result_dir)],
                    },
                    "failure": None,
                    "measured_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            record = {
                "algorithm": algorithm,
                "job_id": job_id,
                "parent_job_id": parent_job_id,
                "checkpoint": str(checkpoint),
                "metric_name": "val_acc_1",
                "metric_value": metric,
                "log": str(log_path),
            }
            self._mark(
                job_id,
                "COMPLETE",
                "--message",
                f"final val_acc_1={metric:.6f}",
            )
            terminal = True
            return record
        except Exception:
            if not terminal:
                try:
                    self._mark(
                        job_id,
                        "ERROR",
                        "--err-class",
                        "ERR_INFRA",
                        "--message",
                        "live matrix final-evaluation harness failure",
                    )
                except subprocess.CalledProcessError:
                    pass
            raise
        finally:
            if gpu is not None:
                self._release_gpu(gpu)

    def run_algorithm(self, algorithm: str) -> dict:
        print(f"\n=== {algorithm} ===", flush=True)
        state = self._init_algorithm(algorithm)
        while True:
            response = self._engine("recommend", "--state", str(state))
            if response.get("status") == "COMPLETE" or response.get("reason") == "experiment_complete":
                break
            recommendations = response.get("recommendations") or []
            if not recommendations:
                raise RuntimeError(f"{algorithm} stalled: {json.dumps(response, sort_keys=True)}")
            with ThreadPoolExecutor(max_workers=len(recommendations)) as executor:
                futures = [
                    executor.submit(self._run_trial, algorithm, state, recommendation)
                    for recommendation in recommendations
                ]
                for future in as_completed(futures):
                    future.result()
        best_path = state.parent / "best_rec.json"
        finalized = self._engine(
            "finalize", "--state", str(state), "--out", str(best_path)
        )["payload"]
        evaluation = self._evaluate_best(algorithm, finalized)
        state_payload = json.loads(state.read_text(encoding="utf-8"))
        result = {
            "algorithm": algorithm,
            "training_jobs": len(state_payload["recommendations"]),
            "successful_training_jobs": sum(
                recommendation["state"] == "SUCCEEDED"
                for recommendation in state_payload["recommendations"]
            ),
            "best_rec": str(best_path),
            "best_training_metric": finalized["best"]["score"],
            "final_evaluation": evaluation,
        }
        _write_json(state.parent / "live_result.json", result)
        print(
            f"{algorithm}: PASS ({result['training_jobs']} real training jobs; "
            f"final val_acc_1={evaluation['metric_value']:.4f})",
            flush=True,
        )
        return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/tmp/tao-automl-live-matrix"),
    )
    parser.add_argument(
        "--image",
        default="nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.0.0-rc-226-multiarch",  # unpinned: qualification image is overridden per release
    )
    # Docker enumerates only GPUs exposed by the NVIDIA runtime. On DGX hosts
    # this can intentionally differ from host nvidia-smi numbering when the
    # display adapter is excluded (for example, host GPU 4 becomes device 3).
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--algorithm", action="append", choices=ALGORITHMS)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    algorithms = args.algorithm or list(ALGORITHMS)
    with _LLMStub() as llm:
        matrix = LiveMatrix(args, llm.endpoint)
        matrix.prepare()
        results = [matrix.run_algorithm(algorithm) for algorithm in algorithms]
    summary = {
        "image": args.image,
        "gpus": args.gpus,
        "algorithms": results,
        "total_training_jobs": sum(result["training_jobs"] for result in results),
        "passed": all(
            result["training_jobs"] == result["successful_training_jobs"]
            for result in results
        ),
    }
    _write_json(args.workspace.resolve() / "matrix_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
