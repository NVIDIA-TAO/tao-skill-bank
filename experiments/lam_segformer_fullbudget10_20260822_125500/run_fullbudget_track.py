#!/usr/bin/env python3
"""Run one reviewed 10 x 2,000-epoch LAM SegFormer AutoML brain."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path

import yaml


LOCAL_ROOT = Path(
    "/localhome/local-rarunachalam/workspace/"
    "lam_segformer_fullbudget10_20260822_125500"
)
REMOTE_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_fullbudget10_20260822_125500"
)
OLD_REMOTE_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_bayes_deft_20260820_231724"
)
SOURCE_DATA = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/lam_research"
)
SKILL_DIR = (
    LOCAL_ROOT
    / "skill_snapshot/skills/models/tao-train-segformer"
)
IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt"

BACKBONES = {
    "fan_base": {
        "type": "fan_base_16_p4_hybrid",
        "ptm": OLD_REMOTE_ROOT / "inputs/ptms/fan_base/fan_base_backbone_stripped.pth",
        "baseline": 0.9472963230,
    },
    "fan_large": {
        "type": "fan_large_16_p4_hybrid",
        "ptm": OLD_REMOTE_ROOT / "inputs/ptms/fan_large/fan_large_backbone_stripped.pth",
        "baseline": 0.9473685995,
    },
    "mit_b5": {
        "type": "mit_b5",
        "ptm": OLD_REMOTE_ROOT / "inputs/ptms/mit_b5/mit_b5_backbone_stripped.pth",
        "baseline": 0.9465497370,
    },
}

SEARCH_PARAMETERS = [
    "dataset.segment.augmentation.random_color.brightness",
    "dataset.segment.augmentation.random_color.color_probability",
    "dataset.segment.augmentation.random_color.contrast",
    "dataset.segment.augmentation.random_color.hue",
    "dataset.segment.augmentation.random_color.saturation",
    "dataset.segment.augmentation.random_flip.hflip_probability",
    "train.optim.lr",
    "train.optim.weight_decay",
]

SEARCH_RANGES = {
    "dataset.segment.augmentation.random_color.brightness": {
        "valid_min": 0.1,
        "valid_max": 0.5,
    },
    "dataset.segment.augmentation.random_color.color_probability": {
        "valid_min": 0.3,
        "valid_max": 0.8,
    },
    "dataset.segment.augmentation.random_color.contrast": {
        "valid_min": 0.1,
        "valid_max": 0.5,
    },
    "dataset.segment.augmentation.random_color.hue": {
        "valid_min": 0.05,
        "valid_max": 0.2,
    },
    "dataset.segment.augmentation.random_color.saturation": {
        "valid_min": 0.1,
        "valid_max": 0.5,
    },
    "dataset.segment.augmentation.random_flip.hflip_probability": {
        "valid_min": 0.3,
        "valid_max": 0.7,
    },
    "train.optim.lr": {"valid_min": 1.0e-5, "valid_max": 5.0e-4},
    "train.optim.weight_decay": {"valid_min": 0.001, "valid_max": 0.05},
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        handle.flush()


def build_spec(backbone: str) -> dict:
    spec = yaml.safe_load((SKILL_DIR / "references/spec_template_train.yaml").read_text())
    model = BACKBONES[backbone]
    spec["model_name"] = f"lam_fullbudget10_{backbone}"
    spec["encryption_key"] = "tlt_encode"
    spec["wandb"]["enable"] = False
    spec["model"]["backbone"].update(
        {
            "type": model["type"],
            "pretrained_backbone_path": str(model["ptm"]),
            "freeze_backbone": False,
        }
    )
    spec["dataset"]["segment"].update(
        {
            "root_dir": str(SOURCE_DATA),
            "num_classes": 4,
            "img_size": 1024,
            "batch_size": 1,
            "workers": 8,
            "label_transform": "None",
            "palette": [
                {"label_id": 0, "mapping_class": "background", "rgb": [0], "seg_class": "background"},
                {"label_id": 1, "mapping_class": "mask_height_1", "rgb": [85], "seg_class": "mask_height_1"},
                {"label_id": 2, "mapping_class": "mask_height_2", "rgb": [170], "seg_class": "mask_height_2"},
                {"label_id": 3, "mapping_class": "trench_depth", "rgb": [255], "seg_class": "trench_depth"},
            ],
        }
    )
    spec["train"].update(
        {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "num_epochs": 2000,
            "checkpoint_interval": 20,
            "validation_interval": 10,
            "use_distributed_sampler": True,
            "sync_batchnorm": True,
        }
    )
    spec["train"]["optim"].update(
        {"optim": "adamw", "policy": "linear", "lr": 6.0e-5, "weight_decay": 0.01}
    )
    spec["train"]["segment"].update(
        {"loss": "ce", "weights": [2.4690, 18.8720, 14.7437, 2.6614]}
    )
    spec["train"]["tensorboard"]["enabled"] = False
    return spec


def ssh_python(script: str, *args: str) -> str:
    host = os.environ["SLURM_HOSTNAME"].split(",", 1)[0]
    remote_command = " ".join(
        shlex.quote(value) for value in ("python3", "-", *map(str, args))
    )
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        f"{os.environ['SLURM_USER']}@{host}",
        remote_command,
    ]
    return subprocess.check_output(command, input=script, text=True).strip()


def lustre_path(value: str) -> str:
    """Convert the SDK's lustre:// URI to the absolute cluster path."""
    path = str(value)
    if path.startswith("lustre://"):
        path = path[len("lustre://"):]
    elif "://" in path:
        raise ValueError(f"unsupported results URI: {value}")
    if not path.startswith("/"):
        raise ValueError(f"results path is not absolute: {value}")
    return path


def read_best_val_miou(results_dir: str) -> dict:
    script = r'''import json,os,sys
root=os.path.realpath(sys.argv[1])
prefix=os.path.realpath(sys.argv[2])+os.sep
if not root.startswith(prefix):
    raise RuntimeError("refusing status read outside campaign results")
payload=os.path.join(root,"results_dir")
if not os.path.isdir(payload): payload=root
status=os.path.join(payload,"train","status.json")
best_epoch=None
best_value=float("-inf")
current_epoch=None
with open(status,errors="replace") as handle:
    for line in handle:
        try: row=json.loads(line)
        except Exception: continue
        epoch=row.get("epoch")
        if isinstance(epoch,(int,float)):
            current_epoch=int(epoch)
        kpi=row.get("kpi") or {}
        value=next((kpi.get(k) for k in ("val_miou","miou","mIoU") if isinstance(kpi.get(k),(int,float))),None)
        if value is not None and value>best_value:
            best_value=float(value); best_epoch=current_epoch
if best_value==float("-inf"):
    raise RuntimeError("status.json contains no validation mIoU")
print(json.dumps({"best_epoch":best_epoch,"best_val_miou":best_value}))'''
    prefix = str(REMOTE_ROOT / "results")
    return json.loads(ssh_python(script, lustre_path(results_dir), prefix))


def repair_missing_brain_design(workspace: Path) -> bool:
    """Recover the issued rec-0 design vector from durable controller specs.

    Older AutoML controllers persisted the pending recommendation before the
    Bayesian/BFBO brain state.  If the controller was restarted in that
    window, the recovered result had no corresponding normalized X vector.
    This campaign uses only linear continuous ranges, so rec-0 can be
    reconstructed exactly from its persisted hyperparameter values.
    """
    automl_dir = workspace / ".automl"
    brain_files = sorted((automl_dir / "brain").glob("*.json"))
    controller_files = sorted((automl_dir / "controller").glob("*.json"))
    if len(brain_files) != 1 or len(controller_files) != 1:
        raise RuntimeError(f"unexpected AutoML state layout in {workspace}")
    brain = json.loads(brain_files[0].read_text())
    if brain.get("Xs") or brain.get("ys"):
        return False
    recommendations = json.loads(controller_files[0].read_text())
    successful = [
        row for row in recommendations if row.get("status") in {"success", "done"}
    ]
    if not successful:
        return False
    if len(successful) != 1 or int(successful[0]["id"]) != 0:
        raise RuntimeError(
            "automatic design-vector repair is restricted to one completed rec-0"
        )
    specs = successful[0]["specs"]
    design = []
    for parameter in SEARCH_PARAMETERS:
        bounds = SEARCH_RANGES[parameter]
        value = float(specs[parameter])
        minimum = float(bounds["valid_min"])
        maximum = float(bounds["valid_max"])
        normalized = (value - minimum) / (maximum - minimum)
        if not -1.0e-9 <= normalized <= 1.0 + 1.0e-9:
            raise RuntimeError(
                f"persisted {parameter}={value} is outside [{minimum}, {maximum}]"
            )
        design.append(min(1.0, max(0.0, normalized)))
    brain["Xs"] = [design]
    brain["ys"] = []
    atomic_json(brain_files[0], brain)
    logging.warning(
        "Repaired missing rec-0 optimizer design vector in %s", brain_files[0]
    )
    return True


def prune_periodic_checkpoints(results_dir: str) -> dict:
    script = r'''import glob,json,os,re,sys
root=os.path.realpath(sys.argv[1])
prefix=os.path.realpath(sys.argv[2])+os.sep
if not root.startswith(prefix):
    raise RuntimeError("refusing checkpoint prune outside campaign results")
payload=os.path.join(root,"results_dir")
if not os.path.isdir(payload): payload=root
train=os.path.join(payload,"train")
rx=re.compile(r"model_epoch_(\d+)_step_(\d+)\.pth$")
checkpoints=[]
for path in glob.glob(os.path.join(train,"model_epoch_*_step_*.pth")):
    match=rx.search(path)
    if match and os.path.isfile(path) and os.path.getsize(path)>1000000:
        checkpoints.append((int(match.group(1)),int(match.group(2)),path))
if not checkpoints:
    raise RuntimeError("no durable checkpoint to retain")
best_epoch=None
best_value=float("-inf")
current_epoch=None
status=os.path.join(train,"status.json")
if os.path.isfile(status):
    with open(status,errors="replace") as handle:
        for line in handle:
            try: row=json.loads(line)
            except Exception: continue
            epoch=row.get("epoch")
            if isinstance(epoch,(int,float)):
                current_epoch=int(epoch)
            kpi=row.get("kpi") or {}
            value=next((kpi.get(k) for k in ("val_miou","miou","mIoU") if isinstance(kpi.get(k),(int,float))),None)
            if value is not None and value>best_value:
                best_value=float(value); best_epoch=current_epoch
latest=max(checkpoints)
keep={latest[2]}
if best_epoch is not None:
    keep.add(min(checkpoints,key=lambda row:(abs(row[0]-best_epoch),-row[0]))[2])
deleted=[]
for _,_,path in checkpoints:
    if path not in keep:
        os.remove(path); deleted.append(path)
print(json.dumps({"retained":sorted(keep),"deleted_count":len(deleted),"best_epoch":best_epoch,"best_val_miou":None if best_value==float("-inf") else best_value}))'''
    prefix = str(REMOTE_ROOT / "results")
    return json.loads(ssh_python(script, lustre_path(results_dir), prefix))


def run(backbone: str, variant: str, resume: bool) -> dict:
    from tao_automl.runner import AutoMLRunner
    from tao_sdk.platforms.slurm import SlurmSDK

    algorithm = "bfbo" if variant.startswith("bfbo") else "bayesian"
    llm_enabled = variant.endswith("_llm")
    track = f"{backbone}_{variant}"
    workspace = LOCAL_ROOT / "workspaces" / track
    if resume:
        prior_runs = sorted(workspace.glob("run_*"))
        if not prior_runs:
            raise RuntimeError(f"no timestamped AutoML workspace to resume: {workspace}")
        workspace = prior_runs[-1]
    state_dir = LOCAL_ROOT / "sdk_state" / track
    os.environ["TAO_SDK_STATE_DIR"] = str(state_dir)
    os.environ["SLURM_BASE_RESULTS_DIR"] = str(REMOTE_ROOT)
    os.environ["SLURM_TIME_HOURS"] = "4"
    os.environ["SLURM_TIMEOUT_HOURS"] = "3.8"
    os.environ["SLURM_USE_REQUEUE"] = "true"

    settings = {
        "algorithm": algorithm,
        "automl_max_recommendations": 10,
        "metric": "val_miou",
        "direction": "maximize",
        "train_sample_count": 316,
        "run_baseline": True,
        "baseline_metric": BACKBONES[backbone]["baseline"],
        "run_final_evaluation": True,
        "reuse_best_metric_for_final_evaluation": True,
        "automl_delete_intermediate_ckpt": False,
        "session_id": f"lam_full2000_v2_{backbone}_{algorithm}",
        "experiment_id": f"lam_full2000_v2_{backbone}_{variant}",
    }
    if llm_enabled:
        endpoint = os.environ.get("base_url") or os.environ.get("AUTOML_LLM_ENDPOINT")
        model = os.environ.get("model") or os.environ.get("AUTOML_LLM_MODEL")
        api_key = os.environ.get("AUTOML_LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not endpoint or not model or not api_key:
            raise RuntimeError("strict LLM analyzer endpoint, model, or API key is missing")
        settings.update(
            {
                "llm_endpoint": endpoint,
                "llm_model": model,
                "llm_api_key": api_key,
                "llm_analyzer_enabled": True,
                "llm_analyzer_interval": 1,
                "llm_analyzer_narrow_ranges": True,
                "llm_analyzer_strict": True,
            }
        )

    event_log = LOCAL_ROOT / "events" / f"{track}.jsonl"
    sdk = SlurmSDK(poll_interval=30, epoch_milestone_interval=10)

    def on_recommendation(rec) -> None:
        append_jsonl(
            event_log,
            {"event": "recommendation", "rec_id": rec.id, "specs": rec.specs},
        )

    def on_result(rec, metric, status) -> None:
        event = {
            "event": "result",
            "rec_id": rec.id,
            "job_id": rec.job_id,
            "metric": metric,
            "status": status,
        }
        if status == "success" and rec.job_id:
            results_dir = sdk.get_job_results_dir(rec.job_id)
            event["results_dir"] = results_dir
            try:
                event["checkpoint_retention"] = prune_periodic_checkpoints(results_dir)
            except Exception as exc:
                # Artifact cleanup is secondary and must never terminate a
                # healthy AutoML search after the result was reported.
                event["checkpoint_retention_error"] = repr(exc)
                logging.exception(
                    "checkpoint retention failed for rec %s job %s",
                    rec.id,
                    rec.job_id,
                )
        append_jsonl(event_log, event)

    def eval_result(rec, job_id: str) -> float:
        results_dir = sdk.get_job_results_dir(job_id)
        metric = read_best_val_miou(results_dir)
        logging.info(
            "Rec %s: exact status.json val_miou=%s at epoch=%s",
            rec.id,
            metric["best_val_miou"],
            metric["best_epoch"],
        )
        return float(metric["best_val_miou"])

    runner = AutoMLRunner(sdk=sdk, skill_dir=str(SKILL_DIR), action="train")
    if resume:
        # A prior controller may have persisted the three-decimal value found
        # in console logs before its callback failed.  Re-read status.json so
        # the resumed optimizer sees the full-precision objective.
        from tao_automl import AutoML, query_status

        persisted = query_status(str(workspace))
        reconciler = AutoML(
            workspace=str(workspace),
            network=runner.skill_ctx.network_arch,
            train_specs=build_spec(backbone),
            settings=settings,
            automl_hyperparameters=SEARCH_PARAMETERS,
            custom_param_ranges=SEARCH_RANGES,
            action="train",
            search_schema=runner.skill_ctx.schema,
            resume=True,
        )
        for prior in persisted.get("recommendations", []):
            if prior.get("status") not in {"success", "done"} or not prior.get("job_id"):
                continue
            try:
                precise = read_best_val_miou(
                    sdk.get_job_results_dir(prior["job_id"])
                )
                reconciler.report_result(
                    int(prior["rec_id"]),
                    float(precise["best_val_miou"]),
                    best_epoch=precise["best_epoch"],
                    status="success",
                )
                logging.info(
                    "Reconciled rec %s to exact val_miou=%s at epoch=%s",
                    prior["rec_id"],
                    precise["best_val_miou"],
                    precise["best_epoch"],
                )
            except Exception:
                logging.exception(
                    "Could not reconcile exact metric for rec %s; preserving persisted value",
                    prior.get("rec_id"),
                )
        repair_missing_brain_design(workspace)

    result = runner.run(
        image=IMAGE,
        spec_overrides=build_spec(backbone),
        automl_settings=settings,
        automl_hyperparameters=SEARCH_PARAMETERS,
        custom_param_ranges=SEARCH_RANGES,
        on_recommendation=on_recommendation,
        on_result=on_result,
        eval_fn=eval_result,
        workspace_path=str(workspace),
        resume=resume,
        gpu_count=8,
        num_nodes=1,
        account=os.environ.get("SLURM_ACCOUNT") or None,
        env_vars={"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"},
    )
    atomic_json(LOCAL_ROOT / "results" / f"{track}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=tuple(BACKBONES), required=True)
    parser.add_argument(
        "--variant",
        choices=("bayesian", "bfbo", "bayesian_llm", "bfbo_llm"),
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = run(args.backbone, args.variant, args.resume)
    print(json.dumps({"status": "complete", "best": result.get("best")}, default=str))


if __name__ == "__main__":
    main()
