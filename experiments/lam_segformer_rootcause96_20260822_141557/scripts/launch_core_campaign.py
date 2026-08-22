#!/usr/bin/env python3
"""Stage and submit the reviewed LAM root-cause and backbone campaign."""

from __future__ import annotations

import copy
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
import yaml


CAMPAIGN = "lam_segformer_rootcause96_20260822_141557"
CODE_ROOT = Path(__file__).resolve().parents[1]
TAO_PYTORCH_ROOT = Path("/localhome/local-rarunachalam/github/tao-pytorch")
EXTERNAL_PATCHES = (
    "nvidia_tao_pytorch/cv/segformer/scripts/train.py",
    "nvidia_tao_pytorch/cv/segformer/model/backbones/adapter_modules.py",
    "nvidia_tao_pytorch/cv/segformer/model/backbones/fan.py",
)
LOCAL_ROOT = Path("/localhome/local-rarunachalam/workspace") / CAMPAIGN
REMOTE_ROOT = Path("/lustre/fsw/portfolios/edgeai/users/rarunachalam") / CAMPAIGN
REMOTE_CONTROLLER = REMOTE_ROOT / "controller"
REMOTE_SPECS = REMOTE_ROOT / "specs"
REMOTE_RUNS = REMOTE_ROOT / "runs"
REMOTE_LOGS = REMOTE_ROOT / "slurm-logs"
REMOTE_SBATCH = REMOTE_ROOT / "sbatch"
DATA_ROOT = Path("/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/lam_research")
OLD_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
    "lam_segformer_bayes_deft_20260820_231724"
)
IMAGE_URI = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt"
IMAGE_SQSH = Path(
    "/lustre/fsw/portfolios/edgeai/projects/"
    "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
    "nvcr.io_nvidia_tao_tao-toolkit_7.1.0-pyt.sqsh"
)
PARTITIONS = "polar,polar3,polar4,grizzly"
PALETTE = [
    {"label_id": 0, "mapping_class": "background", "rgb": [0], "seg_class": "background"},
    {"label_id": 1, "mapping_class": "mask_height_1", "rgb": [85], "seg_class": "mask_height_1"},
    {"label_id": 2, "mapping_class": "mask_height_2", "rgb": [170], "seg_class": "mask_height_2"},
    {"label_id": 3, "mapping_class": "trench_depth", "rgb": [255], "seg_class": "trench_depth"},
]
WEIGHT_C03 = [0.815594, 3.042000, 2.551302, 0.864234]
PTMS = {
    "fan_base": OLD_ROOT / "inputs/ptms/fan_base/fan_base_backbone_stripped.pth",
    "fan_large": OLD_ROOT / "inputs/ptms/fan_large/fan_large_backbone_stripped.pth",
    "mit_b5": OLD_ROOT / "inputs/ptms/mit_b5/mit_b5_backbone_stripped.pth",
    "dinov3_large": REMOTE_ROOT / "inputs/ptms/vit_large_dinov3.safetensors",
    "dinov3_huge_plus": REMOTE_ROOT / "inputs/ptms/vit_huge_plus_dinov3.safetensors",
    "cradio_v3_large": REMOTE_ROOT / "inputs/ptms/c_radio_v3_l.safetensors",
    "cradio_v4_huge": REMOTE_ROOT / "inputs/ptms/c_radio_v4_h.safetensors",
    "cradio_v4_so400m": REMOTE_ROOT / "inputs/ptms/c_radio_v4_so400m.safetensors",
    "vit5_large": REMOTE_ROOT / "inputs/ptms/vit5_large_patch16_224.pth",
}
BACKBONES = {
    "dinov3_large": "vit_large_dinov3",
    "dinov3_huge_plus": "vit_huge_plus_dinov3",
    "cradio_v3_large": "c_radio_v3_vit_large_patch16_reg4_dinov2",
    "cradio_v4_huge": "c_radio_v4_vit_huge_patch16_224",
    "cradio_v4_so400m": "c_radio_v4_vit_so400m_patch16_224",
    "vit5_large": "vit5_large_patch16_224",
}
MODEL_SKILL = Path(
    "/localhome/local-rarunachalam/github/tao-skill-bank/"
    "skills/models/tao-train-segformer"
)
TEMPLATE = MODEL_SKILL / "references/spec_template_train.yaml"
RECORD_TOOL = Path(
    "/localhome/local-rarunachalam/github/tao-skill-bank/scripts/tao_job_record.py"
)
def resolve_plugin_bank() -> Path:
    """Resolve a usable installed bank without pinning a cache-buster version."""
    candidates: list[Path] = []
    configured = os.environ.get("TAO_SKILL_BANK_PATH")
    if configured:
        candidates.append(Path(configured).expanduser())
    cache_root = Path(
        "/localhome/local-rarunachalam/.codex/plugins/cache/tao-local-plugins/"
        "tao-skill-bank"
    )
    candidates.extend(sorted(cache_root.glob("*"), reverse=True))
    for candidate in candidates:
        submit_tool = (
            candidate
            / "skills/platform/tao-run-on-slurm/scripts/slurm_submit_action.py"
        )
        bundle_schema = (
            candidate
            / "skills/core/tao-artifacts/references/spec_bundle.schema.json"
        )
        if submit_tool.is_file() and bundle_schema.is_file():
            return candidate
    raise RuntimeError("no complete installed TAO skill bank was found")


PLUGIN_BANK = resolve_plugin_bank()
SUBMIT_TOOL = (
    PLUGIN_BANK
    / "skills/platform/tao-run-on-slurm/scripts/slurm_submit_action.py"
)
BUNDLE_SCHEMA = (
    PLUGIN_BANK
    / "skills/core/tao-artifacts/references/spec_bundle.schema.json"
)
NCCL_PROBE = PLUGIN_BANK / "scripts/nccl_allreduce_probe.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def login() -> str:
    hostname = os.environ["SLURM_HOSTNAME"].split(",", 1)[0]
    return f"{os.environ['SLURM_USER']}@{hostname}"


def remote(command: str, *, input_bytes: bytes | None = None) -> str:
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", login(), command],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    return completed.stdout.decode(errors="replace").strip()


def scp(source: Path, destination: Path) -> None:
    subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", str(source), f"{login()}:{destination}"],
        check=True,
    )


def record(*args: str) -> str:
    return subprocess.check_output(
        [sys.executable, str(RECORD_TOOL), *args], text=True
    ).strip()


def no_dotted_keys(value: object, location: str = "spec") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if "." in key:
                raise ValueError(f"dotted key at {location}: {key}")
            no_dotted_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            no_dotted_keys(child, f"{location}[{index}]")


def configure_augmentation(spec: dict, recipe: str) -> None:
    augmentation = spec["dataset"]["segment"]["augmentation"]
    if recipe == "historical":
        augmentation["random_flip"].update({"enable": True, "hflip_probability": 0.5, "vflip_probability": 0.5})
        augmentation["random_rotate"].update({"enable": True, "rotate_probability": 0.5})
        augmentation["random_color"].update({"enable": True, "color_probability": 1.0})
        augmentation["with_scale_random_crop"]["enable"] = True
        augmentation["with_random_blur"] = True
        augmentation["with_random_crop"] = True
        return
    augmentation["random_flip"].update({"enable": False, "hflip_probability": 0.0, "vflip_probability": 0.0})
    augmentation["random_rotate"].update({"enable": False, "rotate_probability": 0.0})
    augmentation["random_color"].update(
        {
            "enable": False,
            "color_probability": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0,
        }
    )
    augmentation["with_scale_random_crop"]["enable"] = False
    augmentation["with_random_blur"] = False
    augmentation["with_random_crop"] = False
    if recipe in {"d4", "d4_gray"}:
        augmentation["random_flip"].update(
            {"enable": True, "hflip_probability": 0.5, "vflip_probability": 0.5}
        )
        augmentation["random_rotate"].update(
            {"enable": True, "rotate_probability": 0.75, "angle_list": [90, 180, 270]}
        )
    elif recipe == "flips":
        augmentation["random_flip"].update(
            {"enable": True, "hflip_probability": 0.5, "vflip_probability": 0.5}
        )
    elif recipe != "none":
        raise ValueError(f"unknown augmentation recipe: {recipe}")
    if recipe == "d4_gray":
        augmentation["random_color"].update(
            {
                "enable": True,
                "color_probability": 0.5,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.0,
                "hue": 0.0,
            }
        )


def build_spec(
    *,
    label: str,
    backbone: str,
    ptm: Path,
    dataset: Path = DATA_ROOT,
    epochs: int = 2000,
    freeze: bool = False,
    augmentation: str = "d4",
    loss: str = "ce",
    weights: list[float] | None = WEIGHT_C03,
    iou_weight: float = 0.25,
    policy: str = "cosine",
    lr: float = 6.0e-5,
    precision: str = "32-true",
) -> dict:
    spec = yaml.safe_load(TEMPLATE.read_text())
    results_dir = REMOTE_RUNS / label
    spec.update(
        {
            "model_name": f"lam_root96_{label}",
            "encryption_key": "tlt_encode",
            "results_dir": str(results_dir),
        }
    )
    spec["wandb"]["enable"] = False
    spec["model"]["activation_checkpoint"] = True
    spec["model"]["backbone"].update(
        {
            "type": backbone,
            "pretrained_backbone_path": str(ptm),
            "freeze_backbone": freeze,
        }
    )
    spec["dataset"]["segment"].update(
        {
            "root_dir": str(dataset),
            "num_classes": 4,
            "img_size": 1024,
            "batch_size": 1,
            "workers": 8,
            "label_transform": "None",
            "palette": copy.deepcopy(PALETTE),
        }
    )
    configure_augmentation(spec, augmentation)
    spec["train"].update(
        {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "num_epochs": epochs,
            "checkpoint_interval": 20 if epochs > 1 else 1,
            "validation_interval": 10 if epochs > 1 else 1,
            "results_dir": str(results_dir),
            "resume_training_checkpoint_path": "",
            "precision": precision,
            "use_distributed_sampler": True,
            "sync_batchnorm": True,
        }
    )
    spec["train"]["optim"].update(
        {
            "optim": "adamw",
            "policy": policy,
            "lr": lr,
            "momentum": 0.9,
            "weight_decay": 0.01,
        }
    )
    spec["train"]["segment"].update(
        {
            "loss": loss,
            "weights": [] if weights is None else list(weights),
            "iou_weight": iou_weight,
        }
    )
    spec["train"]["tensorboard"]["enabled"] = False
    no_dotted_keys(spec)
    return spec


def full_run_rows() -> list[dict]:
    rows = []
    causal = [
        ("C01", "historical", "ce", None, 0.25, "linear"),
        ("C02", "d4", "ce", None, 0.25, "linear"),
        ("C03", "d4", "ce", WEIGHT_C03, 0.25, "linear"),
        ("C04", "d4", "ce", [0.661819, 5.058669, 3.952071, 0.713392], 0.25, "linear"),
        ("C05", "d4", "ce", [0.516171, 7.180647, 5.050901, 0.579573], 0.25, "linear"),
        ("C06", "d4", "ce_mmiou", WEIGHT_C03, 0.25, "linear"),
        ("C07", "d4", "ce_mmiou", WEIGHT_C03, 0.50, "linear"),
        ("C08", "d4", "mmiou", None, 1.00, "linear"),
        ("C09", "d4", "ce", WEIGHT_C03, 0.25, "cosine"),
        ("C10", "d4_gray", "ce", WEIGHT_C03, 0.25, "cosine"),
        ("C11", "none", "ce", WEIGHT_C03, 0.25, "cosine"),
        ("C12", "flips", "ce", WEIGHT_C03, 0.25, "cosine"),
        ("C13", "d4", "ce_lovasz", WEIGHT_C03, 0.25, "cosine"),
        ("C14", "d4", "ce_boundary", WEIGHT_C03, 0.25, "cosine"),
    ]
    for run_id, aug, loss, weights, iou_weight, policy in causal:
        label = f"{run_id.lower()}_fan_large"
        rows.append(
            {
                "run_id": run_id,
                "label": label,
                "group": "causal",
                "spec": build_spec(
                    label=label,
                    backbone="fan_large_16_p4_hybrid",
                    ptm=PTMS["fan_large"],
                    augmentation=aug,
                    loss=loss,
                    weights=weights,
                    iou_weight=iou_weight,
                    policy=policy,
                ),
            }
        )
    deft = [
        ("D01", "fan_base_16_p4_hybrid", PTMS["fan_base"], OLD_ROOT / "datasets/deft_mix100", "fan_base_mix100"),
        ("D02", "fan_large_16_p4_hybrid", PTMS["fan_large"], OLD_ROOT / "datasets/deft_mix50", "fan_large_mix50"),
    ]
    for run_id, backbone, ptm, dataset, suffix in deft:
        label = f"{run_id.lower()}_{suffix}"
        rows.append(
            {
                "run_id": run_id,
                "label": label,
                "group": "deft",
                "spec": build_spec(
                    label=label,
                    backbone=backbone,
                    ptm=ptm,
                    dataset=dataset,
                ),
            }
        )
    index = 1
    for key, backbone in BACKBONES.items():
        for freeze in (True, False):
            run_id = f"B{index:02d}"
            label = f"{run_id.lower()}_{key}_{'frozen' if freeze else 'full'}"
            rows.append(
                {
                    "run_id": run_id,
                    "label": label,
                    "group": "backbone",
                    "probe_key": key,
                    "spec": build_spec(
                        label=label,
                        backbone=backbone,
                        ptm=PTMS[key],
                        freeze=freeze,
                        lr=6.0e-5 if freeze else 1.0e-5,
                        precision=(
                            "bf16-mixed"
                            if key == "dinov3_huge_plus" and not freeze
                            else "32-true"
                        ),
                    ),
                }
            )
            index += 1
    if len(rows) != 28:
        raise RuntimeError(f"expected 28 full runs, found {len(rows)}")
    return rows


def bundle(spec: dict) -> dict:
    inputs = [
        {
            "spec_key": "dataset.segment.root_dir",
            "type": "folder",
            "uri": "lustre://" + spec["dataset"]["segment"]["root_dir"],
        },
        {
            "spec_key": "model.backbone.pretrained_backbone_path",
            "type": "file",
            "uri": "lustre://" + spec["model"]["backbone"]["pretrained_backbone_path"],
        },
    ]
    return {
        "network_arch": "segformer",
        "action": "train",
        "image": IMAGE_URI,
        "mode": "config",
        "command": "segformer train -e {config_path}",
        "config_format": "yaml",
        "spec": spec,
        "declared_inputs": inputs,
        "declared_outputs": [{"spec_key": "results_dir", "type": "folder"}],
        "upload_excludes": ["inputs/"],
        "compute_shape": {"gpus": 8, "nodes": 1},
        "gpu_spec_key": "train.num_gpus",
    }


def stage_controller() -> None:
    remote(
        "mkdir -p "
        + " ".join(
            shlex.quote(str(path))
            for path in (REMOTE_CONTROLLER, REMOTE_SPECS, REMOTE_RUNS, REMOTE_LOGS, REMOTE_SBATCH)
        )
    )
    with tempfile.NamedTemporaryFile(suffix=".tar") as archive_file:
        with tarfile.open(fileobj=archive_file, mode="w") as archive:
            archive.add(CODE_ROOT / "patches", arcname="patches")
            archive.add(CODE_ROOT / "scripts", arcname="scripts")
            for relative_path in EXTERNAL_PATCHES:
                archive.add(
                    TAO_PYTORCH_ROOT / relative_path,
                    arcname=f"patches/{relative_path}",
                )
        archive_file.flush()
        archive_file.seek(0)
        remote(
            f"tar -xf - -C {shlex.quote(str(REMOTE_CONTROLLER))}",
            input_bytes=archive_file.read(),
        )
    scp(NCCL_PROBE, REMOTE_CONTROLLER / "scripts/nccl_allreduce_probe.py")


def stage_specs(rows: list[dict]) -> None:
    schema = json.loads(BUNDLE_SCHEMA.read_text())
    local_specs = LOCAL_ROOT / "specs"
    local_bundles = LOCAL_ROOT / "bundles"
    local_specs.mkdir(parents=True, exist_ok=True)
    local_bundles.mkdir(parents=True, exist_ok=True)
    for row in rows:
        spec_path = local_specs / f"{row['label']}.yaml"
        spec_path.write_text(yaml.safe_dump(row["spec"], sort_keys=False))
        bundle_value = bundle(row["spec"])
        jsonschema.validate(bundle_value, schema)
        atomic_json(local_bundles / f"{row['label']}.json", bundle_value)
        scp(spec_path, REMOTE_SPECS / spec_path.name)
        row["spec_path"] = str(REMOTE_SPECS / spec_path.name)


def probe_rows() -> list[dict]:
    rows = []
    for key, backbone in BACKBONES.items():
        base = f"probe_{key}"
        frozen_label = f"{base}_frozen"
        unfrozen_label = f"{base}_unfrozen"
        frozen = build_spec(
            label=frozen_label,
            backbone=backbone,
            ptm=PTMS[key],
            epochs=1,
            freeze=True,
        )
        unfrozen = build_spec(
            label=unfrozen_label,
            backbone=backbone,
            ptm=PTMS[key],
            epochs=1,
            freeze=False,
            lr=1.0e-5,
            precision="bf16-mixed" if key == "dinov3_huge_plus" else "32-true",
        )
        rows.append(
            {
                "probe_key": key,
                "label": base,
                "frozen": frozen,
                "unfrozen": unfrozen,
                "results_dir": str(REMOTE_ROOT / "probes" / key),
            }
        )
    return rows


def stage_probe_specs(rows: list[dict]) -> None:
    schema = json.loads(BUNDLE_SCHEMA.read_text())
    local_specs = LOCAL_ROOT / "specs"
    local_bundles = LOCAL_ROOT / "bundles"
    for row in rows:
        for mode in ("frozen", "unfrozen"):
            label = f"{row['label']}_{mode}"
            spec_path = local_specs / f"{label}.yaml"
            spec_path.write_text(yaml.safe_dump(row[mode], sort_keys=False))
            bundle_value = bundle(row[mode])
            jsonschema.validate(bundle_value, schema)
            atomic_json(local_bundles / f"{label}.json", bundle_value)
            scp(spec_path, REMOTE_SPECS / spec_path.name)
            row[f"{mode}_spec_path"] = str(REMOTE_SPECS / spec_path.name)


def render_sbatch(
    *,
    job_id: str,
    command: str,
    dependency: list[str] | None = None,
    dependency_type: str = "afterok",
) -> Path:
    account = os.environ.get("SLURM_ACCOUNT", "")
    extra = []
    if account:
        extra.append(f"#SBATCH --account={account}")
    extra.append(f"#SBATCH --partition={PARTITIONS}")
    if dependency:
        if dependency_type not in {"afterok", "afterany"}:
            raise ValueError(f"unsupported dependency type: {dependency_type}")
        extra.append(f"#SBATCH --dependency={dependency_type}:{':'.join(dependency)}")
        extra.append("#SBATCH --kill-on-invalid-dep=yes")
    quoted_command = shlex.quote(command)
    script = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_id}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output={REMOTE_LOGS}/%x-%j.out
#SBATCH --error={REMOTE_LOGS}/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
{chr(10).join(extra)}
set -euo pipefail
export NCCL_DEBUG=INFO
export LOGLEVEL=INFO
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
set +e
timeout 228m srun --container-image={IMAGE_SQSH} \\
  --container-env=NCCL_DEBUG,LOGLEVEL,NCCL_P2P_DISABLE,NCCL_IB_DISABLE,NCCL_SOCKET_IFNAME,NCCL_IB_HCA,NCCL_NET \\
  --container-mounts=/lustre \\
  -- /bin/bash -lc {quoted_command}
status=$?
set -e
if [ "$status" -eq 124 ]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
"""
    path = LOCAL_ROOT / "sbatch" / f"job_{job_id}.sbatch"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    return path


def submit_job(
    *,
    label: str,
    action: str,
    results_dir: str,
    command: str,
    dependency: list[str] | None = None,
    dependency_type: str = "afterok",
    retry_of: str | None = None,
) -> dict:
    open_args = [
        "open",
        "--platform", "slurm",
        "--image", IMAGE_URI,
        "--network-arch", "segformer",
        "--action", action,
        "--storage-tier", "A",
        "--results-dir", results_dir,
    ]
    if retry_of:
        open_args.extend(("--retry-of", retry_of))
    job_id = record(*open_args)
    script = render_sbatch(
        job_id=job_id,
        command=command,
        dependency=dependency,
        dependency_type=dependency_type,
    )
    remote_script = REMOTE_SBATCH / f"job_{job_id}.sbatch"
    completed = subprocess.run(
        [
            sys.executable,
            str(SUBMIT_TOOL),
            "--login", login(),
            "--job-id", job_id,
            "--rendered-script", str(script),
            "--remote-script", str(remote_script),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode:
        record(
            "mark", job_id,
            "--state", "ERROR",
            "--source", "backend-hook",
            "--err-class", "ERR_PROGRAM",
            "--message", f"submission failed for {label}",
        )
        raise RuntimeError(completed.stderr.strip())
    submission = json.loads(completed.stdout)
    backend_ref = str(submission["backend_ref"])
    record(
        "mark", job_id,
        "--state", "RUNNING",
        "--source", "backend-hook",
        "--backend-ref", backend_ref,
        "--message", f"submitted {label} with 8 GPUs",
    )
    return {
        "label": label,
        "job_id": job_id,
        "backend_ref": backend_ref,
        "results_dir": results_dir,
        "dependency": dependency or [],
        "dependency_type": dependency_type if dependency else None,
        "state": "RUNNING",
        "submitted_at": now(),
        "retry_of": retry_of,
    }


def main() -> None:
    required = ("SLURM_USER", "SLURM_HOSTNAME", "SLURM_ACCOUNT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required environment names: {missing}")
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    stage_controller()
    full_rows = full_run_rows()
    probes = probe_rows()
    stage_specs(full_rows)
    stage_probe_specs(probes)
    manifest_path = LOCAL_ROOT / "core_launch_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing duplicate core launch: {manifest_path}")
    launched = []
    atomic_json(manifest_path, {"campaign": CAMPAIGN, "jobs": launched})

    topology_prefix = (
        "export SLURM_JOB_NAME=bash; "
        "unset SLURM_NTASKS SLURM_NNODES SLURM_PROCID SLURM_LOCALID "
        "SLURM_NTASKS_PER_NODE RANK LOCAL_RANK GROUP_RANK WORLD_SIZE NODE_RANK "
        "NUM_GPU_PER_NODE MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE; "
    )
    runtime_command = topology_prefix + (
        f"export TAO_NODE_COUNT=1 TAO_GPUS_PER_NODE=8 TAO_NODE_RANK=0 "
        f"MASTER_ADDR=127.0.0.1 MASTER_PORT=29500; "
        f"timeout 120s torchrun --standalone --nproc_per_node=8 "
        f"{REMOTE_CONTROLLER}/scripts/nccl_allreduce_probe.py && "
        f"python3 {REMOTE_CONTROLLER}/scripts/runtime_probe.py && "
        f"mkdir -p {REMOTE_ROOT}/probes/runtime && "
        f"touch {REMOTE_ROOT}/probes/runtime/probe_receipt.ok"
    )
    runtime = submit_job(
        label="runtime_nccl_probe",
        action="evaluate",
        results_dir=str(REMOTE_ROOT / "probes/runtime"),
        command=runtime_command,
    )
    launched.append(runtime)
    atomic_json(manifest_path, {"campaign": CAMPAIGN, "jobs": launched})

    probe_backend = {}
    for row in probes:
        command = topology_prefix + (
            f"python3 {REMOTE_CONTROLLER}/scripts/backbone_probe.py "
            f"--name {shlex.quote(row['probe_key'])} "
            f"--frozen-spec {shlex.quote(row['frozen_spec_path'])} "
            f"--unfrozen-spec {shlex.quote(row['unfrozen_spec_path'])} "
            f"--receipt {shlex.quote(row['results_dir'] + '/probe_receipt.json')}"
        )
        submitted = submit_job(
            label=row["label"],
            action="train",
            results_dir=row["results_dir"],
            command=command,
            dependency=[runtime["backend_ref"]],
        )
        probe_backend[row["probe_key"]] = submitted["backend_ref"]
        launched.append(submitted)
        atomic_json(manifest_path, {"campaign": CAMPAIGN, "jobs": launched})

    for row in full_rows:
        dependencies = [runtime["backend_ref"]]
        if row["group"] == "backbone":
            dependencies.append(probe_backend[row["probe_key"]])
        command = topology_prefix + (
            f"python3 {REMOTE_CONTROLLER}/scripts/resume_training_entrypoint.py "
            f"--spec {shlex.quote(row['spec_path'])}"
        )
        submitted = submit_job(
            label=row["label"],
            action="train",
            results_dir=row["spec"]["results_dir"],
            command=command,
            dependency=dependencies,
        )
        submitted.update({"run_id": row["run_id"], "group": row["group"], "spec": row["spec_path"]})
        launched.append(submitted)
        atomic_json(manifest_path, {"campaign": CAMPAIGN, "jobs": launched})
        print(
            f"SUBMITTED {row['run_id']} {row['label']} {submitted['backend_ref']}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "campaign": CAMPAIGN,
                "core_jobs": len(launched),
                "runtime": runtime["backend_ref"],
                "full_jobs": 28,
                "gpus_per_job": 8,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
