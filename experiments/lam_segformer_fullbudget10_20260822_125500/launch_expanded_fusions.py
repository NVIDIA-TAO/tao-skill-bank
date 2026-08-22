#!/usr/bin/env python3
"""Launch 9-source prediction fusion and checkpoint soups on validation."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


CAMPAIGN = "lam_segformer_fullbudget10_20260822_125500"
HERE = Path(__file__).resolve().parent
LOCAL_ROOT = Path("/localhome/local-rarunachalam/workspace") / CAMPAIGN
OLD_LOCAL = Path("/localhome/local-rarunachalam/workspace/lam_segformer_bayes_deft_20260820_231724")
REMOTE_ROOT = Path("/lustre/fsw/portfolios/edgeai/users/rarunachalam") / CAMPAIGN
OLD_REMOTE = Path("/lustre/fsw/portfolios/edgeai/users/rarunachalam/lam_segformer_bayes_deft_20260820_231724")
REMOTE_CONTROLLER = REMOTE_ROOT / "controller"
REMOTE_MANIFESTS = REMOTE_ROOT / "fusion9_manifests"
REMOTE_OUTPUTS = REMOTE_ROOT / "fusion9"
IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt"
RECORD = Path("/localhome/local-rarunachalam/github/tao-skill-bank/scripts/tao_job_record.py")
BACKBONES = ("fan_base", "fan_large", "mit_b5")
EXPECTED_PARTITIONS = {"polar", "polar3", "polar4", "grizzly"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def host() -> str:
    return os.environ["SLURM_HOSTNAME"].split(",", 1)[0]


def ssh(*args: str) -> str:
    command = " ".join(shlex.quote(arg) for arg in args)
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         f"{os.environ['SLURM_USER']}@{host()}", command],
        text=True,
    ).strip()


def scp(source: Path, destination: Path) -> None:
    subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         str(source), f"{os.environ['SLURM_USER']}@{host()}:{destination}"],
        check=True,
    )


def record(*args: str) -> str:
    return subprocess.check_output([sys.executable, str(RECORD), *args], text=True).strip()


def source_rows(backbone: str) -> list[dict]:
    original = json.loads((OLD_LOCAL / "full2000_all_brains_status.json").read_text())
    deft = json.loads((OLD_LOCAL / "deft_full2000_status.json").read_text())
    downstream = json.loads((OLD_LOCAL / "deft_automl_full2000_status.json").read_text())
    rows = []
    for row in original:
        if row["backbone"] == backbone and row["state"] == "COMPLETE":
            rows.append({
                "name": row["brain"], "source": "original_data_automl",
                "backbone": backbone,
                "variant": row["brain"].removeprefix(f"automl_{backbone}_"),
                "checkpoint": row["final_checkpoint"], "spec": row["spec"],
                "val_miou": row["selection_miou"],
            })
    for row in deft:
        if row["backbone"] == backbone and row["state"] == "COMPLETE":
            rows.append({
                "name": row["name"], "source": "standalone_deft",
                "backbone": backbone, "variant": "deft_mix25",
                "checkpoint": row["final_checkpoint"], "spec": row["spec"],
                "val_miou": -1.0,
            })
    for row in downstream:
        if row["backbone"] == backbone and row["state"] == "COMPLETE":
            rows.append({
                "name": row["name"], "source": "deft_snapshot_automl",
                "backbone": backbone, "variant": row["variant"],
                "checkpoint": row["final_checkpoint"], "spec": row["spec"],
                "val_miou": row["selection_metric"],
            })
    if len(rows) != 9:
        raise RuntimeError(f"{backbone}: expected 9 completed sources, got {len(rows)}")
    return rows


def resolve_nearest_best(rows: list[dict]) -> list[dict]:
    script = r'''import glob,json,os,re,sys
rows=json.loads(sys.argv[1]); out=[]
rx=re.compile(r"model_epoch_(\d+)_step_(\d+)\.pth$")
for row in rows:
 d=os.path.dirname(row["checkpoint"]); values=[]
 status=os.path.join(d,"status.json")
 if os.path.isfile(status):
  with open(status,errors="replace") as h:
   for line in h:
    try: event=json.loads(line)
    except Exception: continue
    epoch=event.get("epoch"); kpi=event.get("kpi") or {}
    value=next((kpi.get(k) for k in ("val_miou","miou","mIoU") if isinstance(kpi.get(k),(int,float))),None)
    if isinstance(epoch,int) and value is not None: values.append((float(value),epoch))
 checkpoints=[]
 for path in glob.glob(os.path.join(d,"model_epoch_*_step_*.pth")):
  match=rx.search(path)
  if match and os.path.getsize(path)>1024*1024: checkpoints.append((int(match.group(1)),path))
 if not checkpoints: raise RuntimeError("no durable checkpoints: "+d)
 if values:
  best_value,best_epoch=max(values); epoch,path=min(checkpoints,key=lambda x:(abs(x[0]-best_epoch),-x[0]))
  row["val_miou"]=best_value; row["best_metric_epoch"]=best_epoch
 else: epoch,path=max(checkpoints)
 row["checkpoint"]=path; row["checkpoint_epoch"]=epoch; out.append(row)
print(json.dumps(out))'''
    return json.loads(ssh("python3", "-c", script, json.dumps(rows)))


def candidates(rows: list[dict]) -> list[dict]:
    count = len(rows); output = []; seen = set()
    def add(name: str, weights: list[float], source: str) -> None:
        total=sum(weights); weights=[value/total for value in weights]
        key=tuple(round(value,12) for value in weights)
        if key not in seen:
            seen.add(key); output.append({"name":name,"weights":weights,"sources":[source]})
    for index in range(count):
        weights=[0.0]*count; weights[index]=1.0; add(f"single_{index}",weights,"single")
    for left in range(count):
        for right in range(left+1,count):
            for alpha in (0.25,0.5,0.75):
                weights=[0.0]*count; weights[left]=alpha; weights[right]=1-alpha
                add(f"pair_{left}_{right}_{int(alpha*100)}",weights,"pair_quarters")
    families={}
    for index,row in enumerate(rows): families.setdefault(row["source"],[]).append(index)
    for family,indices in sorted(families.items()):
        weights=[0.0]*count
        for index in indices: weights[index]=1/len(indices)
        add(f"uniform_{family}",weights,"family_uniform")
    add("uniform_all",[1/count]*count,"all_uniform")
    weights=[0.0]*count
    for indices in families.values():
        for index in indices: weights[index]=1/len(families)/len(indices)
    add("equal_source_families",weights,"family_balanced")
    ranked=sorted(range(count),key=lambda i:rows[i]["val_miou"],reverse=True)
    for size in range(2,count+1):
        weights=[0.0]*count
        for index in ranked[:size]: weights[index]=1/size
        add(f"cumulative_top_{size}",weights,"source_val_order")
    return output


def valid(row: dict) -> bool:
    script = r'''import json,os,sys
kind,root=sys.argv[1:3]; result=os.path.join(root,"results.json")
if not os.path.isfile(result) or os.path.getsize(result)<100: raise SystemExit(1)
p=json.load(open(result)); assert p["sample_count"]==1262 and p.get("best")
if kind=="checkpoint_soup": assert os.path.getsize(os.path.join(root,"best_soup.pth"))>1024*1024'''
    try: ssh("python3", "-c", script, row["kind"], row["results_dir"]); return True
    except subprocess.CalledProcessError: return False


def main() -> None:
    from tao_sdk.platforms.slurm import SlurmSDK
    required=("SLURM_USER","SLURM_HOSTNAME","SLURM_ACCOUNT","NGC_KEY")
    missing=[name for name in required if not os.environ.get(name)]
    if missing: raise RuntimeError(f"unset required variables: {missing}")
    partitions={part.strip() for part in os.environ.get("SLURM_PARTITION","").split(",") if part.strip()}
    if partitions != EXPECTED_PARTITIONS: raise RuntimeError("unexpected SLURM partition set")
    ssh("mkdir","-p",str(REMOTE_CONTROLLER),str(REMOTE_MANIFESTS),str(REMOTE_OUTPUTS))
    scripts={
        "prediction_fusion": HERE/"score_prediction_fusions_expanded.py",
        "checkpoint_soup": HERE/"score_checkpoint_soups_expanded.py",
    }
    for source in scripts.values(): scp(source,REMOTE_CONTROLLER/source.name)

    jobs=[]
    local_manifests=LOCAL_ROOT/"fusion9_manifests"
    for backbone in BACKBONES:
        rows=resolve_nearest_best(source_rows(backbone))
        common={"schema_version":1,"experiment":f"lam_{backbone}_nine_source",
                "backbone":backbone,"dataset_root":"/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/lam_research",
                "selection_split":"val","expected_samples":1262,"num_classes":4,
                "test_used_for_selection":False,"models":rows}
        fusion={**common,"method":"five_prediction_fusion_families","weight_candidates":candidates(rows)}
        soup={**common,"method":"same_backbone_checkpoint_interpolation"}
        fusion_local=local_manifests/f"prediction_fusion_{backbone}.json"
        soup_local=local_manifests/f"checkpoint_soup_{backbone}.json"
        write_json(fusion_local,fusion); write_json(soup_local,soup)
        fusion_remote=REMOTE_MANIFESTS/fusion_local.name
        soup_remote=REMOTE_MANIFESTS/soup_local.name
        scp(fusion_local,fusion_remote); scp(soup_local,soup_remote)
        schemes=("probability","geometric_probability","raw_logit",
                 "class_rank_probability_tiebreak","hard_vote_probability_tiebreak")
        for scheme in schemes:
            result=REMOTE_OUTPUTS/"prediction_fusion"/scheme/backbone
            command=(f"torchrun --standalone --nproc_per_node=8 "
                     f"{REMOTE_CONTROLLER}/{scripts['prediction_fusion'].name} "
                     f"--manifest {fusion_remote} --scheme {scheme} "
                     f"--output {result}/results.json")
            jobs.append({"name":f"fusion9_{scheme}_{backbone}","kind":"prediction_fusion",
                         "scheme":scheme,"backbone":backbone,"results_dir":str(result),
                         "command":command,"models":rows,"state":"PENDING"})
        result=REMOTE_OUTPUTS/"checkpoint_soup"/backbone
        command=(f"torchrun --standalone --nproc_per_node=8 "
                 f"{REMOTE_CONTROLLER}/{scripts['checkpoint_soup'].name} "
                 f"--manifest {soup_remote} --output {result}/results.json "
                 f"--best-checkpoint {result}/best_soup.pth")
        jobs.append({"name":f"fusion9_checkpoint_soup_{backbone}","kind":"checkpoint_soup",
                     "backbone":backbone,"results_dir":str(result),"command":command,
                     "models":rows,"state":"PENDING"})
    if len(jobs)!=18: raise RuntimeError(f"expected 18 parallel fusion jobs, got {len(jobs)}")
    ssh("python3","-c","import json,os,sys; [os.makedirs(x['results_dir'],exist_ok=True) for x in json.loads(sys.argv[1])]; print('FUSION9_PREFLIGHT_OK')",json.dumps(jobs))
    manifest=LOCAL_ROOT/"fusion9_launch_manifest.json"; status=LOCAL_ROOT/"fusion9_status.json"
    write_json(manifest,jobs)
    os.environ["TAO_SDK_STATE_DIR"]=str(LOCAL_ROOT/"sdk_state/fusion9")
    os.environ["SLURM_BASE_RESULTS_DIR"]=str(REMOTE_ROOT)
    os.environ["SLURM_SQSH_CACHE_DIR"]=(
        "/lustre/fsw/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam"
    )
    sdk=SlurmSDK(poll_interval=30,epoch_milestone_interval=5)
    for row in jobs:
        rid=record("open","--platform","slurm","--image",IMAGE,"--network-arch","segformer",
                   "--action","evaluate","--storage-tier","A","--results-dir",row["results_dir"])
        row["record_id"]=rid; write_json(manifest,jobs)
        try:
            job=sdk.create_job(image=IMAGE,command=row["command"],gpu_count=8,num_nodes=1,
                account=os.environ.get("SLURM_ACCOUNT") or None,
                env_vars={"PYTHONPATH":"/usr/local/lib/python3.12/dist-packages"})
            row.update({"job_id":job.id,"backend_ref":job.backend_job_id,"state":"RUNNING"})
            record("mark",rid,"--state","RUNNING","--source","backend-hook","--backend-ref",job.backend_job_id,
                   "--message",f"submitted {row['name']} with 8 GPUs")
            print(f"SUBMITTED {row['name']} {job.id} {job.backend_job_id}",flush=True)
        except Exception as exc:
            row.update({"state":"ERROR","error":f"{type(exc).__name__}: {exc}"})
            record("mark",rid,"--state","ERROR","--source","backend-hook","--err-class","ERR_PROGRAM",
                   "--message",f"submission failed for {row['name']}")
            write_json(manifest,jobs); raise
        write_json(manifest,jobs)
    terminal={"COMPLETE","ERROR","CANCELED"}
    while any(row["state"] not in terminal for row in jobs):
        for row in jobs:
            if row["state"] in terminal: continue
            state=sdk.get_job_status(row["job_id"]).status.upper()
            if state=="CANCELLED": state="CANCELED"
            if state=="COMPLETE" and not valid(row): row["error"]="missing result artifact"; state="ERROR"
            if state in terminal:
                row["state"]=state; args=["mark",row["record_id"],"--state",state,"--source","backend-hook","--message",f"terminal {row['name']}"]
                if state=="ERROR": args.extend(["--err-class","ERR_PROGRAM"])
                record(*args)
            else: row["state"]=state
        write_json(status,jobs)
        counts={state:sum(row["state"]==state for row in jobs) for state in terminal}
        print(f"FUSION9 running={len(jobs)-sum(counts.values())} complete={counts['COMPLETE']} error={counts['ERROR']} canceled={counts['CANCELED']}",flush=True)
        if any(row["state"] not in terminal for row in jobs): time.sleep(30)


if __name__ == "__main__": main()
