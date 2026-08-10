"""
far_eval.py — FAR@100%recall via SLURM inference job.

Everything runs on the cluster — no checkpoint download, no local Docker.
The inference container runs on a GPU node, writes inference.csv to Lustre,
then an inline Python step computes FAR and writes metric_result.json.
We read the result back with a single SSH cat.

Modes:
  bayesian    Score the retained v1 bayesian winner
  watch       Watch summary_*.json files from 5 running algorithms; score each on arrival
  score       Score a specific job (--job-id + --algo-name)

Usage:
  python far_eval.py --mode bayesian
  python far_eval.py --mode watch --log-dir ~/workspace/automl_logs_v2
  python far_eval.py --mode score --job-id <id> --algo-name bohb
"""

import argparse, json, logging, os, pathlib, subprocess, sys, time
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
SB        = pathlib.Path(os.environ.get("TAO_SKILL_BANK_PATH",
                          os.environ.get("TAO_SKILL_BANK_PATH", ".")))
WORKSPACE = pathlib.Path(os.environ.get("AOI_WORKSPACE", "./workspace"))
IMAGE     = "nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt"

LUSTRE_AOI = (
    os.environ.get("LUSTRE_AOI_ROOT", "/lustre/<project>/users/<user>/aoi_automl")
)
LUSTRE_RESULTS  = f"{LUSTRE_AOI}/results/results"
LUSTRE_KPI_CSV  = f"{LUSTRE_AOI}/train/testing_set.csv"
LUSTRE_IMGS     = f"{LUSTRE_AOI}/images"
LUSTRE_BACKBONE = f"{LUSTRE_AOI}/backbone/c_radio_v2_b.safetensors"

BAYESIAN_JOB = "33d13b05-af93-4638-8de1-7858d628e232"

# Inline Python that runs inside the TAO container after inference to compute FAR.
# Written as a single-line command so it can be embedded in the shell command string.
# Single source of truth for the FAR computation — exact port of analyze_kpi.py
# semantics (strict score > threshold; best F1 among recall==1.0 candidates).
# Validated bit-exact against DEFT iter9/baseline/iter1 stored results.
_FAR_INLINE_PY = (pathlib.Path(__file__).parent / "far_eval_inline.py").read_text()


def _creds():
    return {
        "key":  os.environ.get("SSH_KEY_PATH", ""),
        "user": os.environ.get("SLURM_USER", ""),
        "host": os.environ.get("SLURM_HOSTNAME", "").split(",")[0],
    }


def ssh_run(cmd: str) -> str:
    c = _creds()
    r = subprocess.run(
        ["ssh", "-i", c["key"], "-o", "StrictHostKeyChecking=no",
         f"{c['user']}@{c['host']}", cmd],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"SSH failed (rc={r.returncode}):\n{r.stderr[:500]}")
    return r.stdout.strip()


def scp_to_cluster(local: str, remote: str):
    c = _creds()
    log.info("scp → %s", remote)
    subprocess.run(
        ["scp", "-i", c["key"], "-o", "StrictHostKeyChecking=no",
         str(local), f"{c['user']}@{c['host']}:{remote}"],
        check=True,
    )


def make_sdk():
    from tao_sdk.platforms.slurm import SlurmSDK
    return SlurmSDK()


def build_infer_spec(experiment_yaml_str: str, ckpt_path: str,
                     infer_out_dir: str) -> dict:
    """
    Patch the training experiment.yaml for inference on the KPI set.
    All paths are Lustre paths — run unchanged on the cluster.
    """
    spec = yaml.safe_load(experiment_yaml_str)
    spec["task"] = "classify"
    spec["dataset"]["classify"]["infer_dataset"] = {
        "csv_path": LUSTRE_KPI_CSV,
        "images_dir": LUSTRE_IMGS,
    }
    spec["inference"] = {
        "checkpoint":  ckpt_path,
        "batch_size":  16,
        "results_dir": infer_out_dir,
        "num_gpus": 1, "gpu_ids": [0], "num_nodes": 1,
    }
    spec["results_dir"] = infer_out_dir

    # Strip 7.1.0-only fields unsupported by 7.0.1-pyt inference container
    spec.get("train", {}).pop("checkpointer", None)
    spec.get("train", {}).pop("precision", None)
    spec.get("train", {}).pop("sync_batchnorm", None)
    spec.get("train", {}).pop("use_distributed_sampler", None)

    return spec


def submit_far_job(algo_name: str, ckpt_path: str,
                   experiment_yaml_str: str, lustre_out: str,
                   sdk, partition: str, account: str) -> str:
    """
    Write spec to Lustre, submit a SLURM GPU job that runs:
      1. visual_changenet inference  (TAO container)
      2. inline Python FAR computation (same container, after inference exits)
    Returns the SLURM job ID string.
    """
    infer_out = f"{lustre_out}/inference"
    far_out   = f"{lustre_out}/far"
    spec_path = f"{lustre_out}/infer_spec.yaml"

    spec = build_infer_spec(experiment_yaml_str, ckpt_path, infer_out)

    # Write spec and FAR script locally then scp both to Lustre
    local_spec_tmp   = f"/tmp/far_infer_spec_{algo_name}.yaml"
    local_script_tmp = f"/tmp/far_compute_{algo_name}.py"
    pathlib.Path(local_spec_tmp).write_text(yaml.dump(spec, default_flow_style=False))
    pathlib.Path(local_script_tmp).write_text(_FAR_INLINE_PY)

    c = _creds()
    ssh_run(f"mkdir -p {lustre_out} {infer_out} {far_out}")

    far_script = f"{lustre_out}/far_compute.py"
    for local, remote in [(local_spec_tmp, spec_path), (local_script_tmp, far_script)]:
        subprocess.run(
            ["scp", "-i", c["key"], "-o", "StrictHostKeyChecking=no",
             local, f"{c['user']}@{c['host']}:{remote}"],
            check=True,
        )

    far_json  = f"{far_out}/metric_result.json"
    infer_csv = f"{infer_out}/inference.csv"

    command = (
        f"visual_changenet inference -e {spec_path} && "
        f"python3 {far_script} {infer_csv} {far_json}"
    )

    job = sdk.create_job(
        image=IMAGE,
        command=command,
        gpu_count=1,
        account=account or None,
        num_nodes=1,
    )
    job_id = job.id if hasattr(job, "id") else str(job)
    log.info("[%s] Submitted FAR job %s → out=%s", algo_name, job_id, lustre_out)
    return job_id, far_json


def poll_job(sdk, job_id: str, algo_name: str, timeout_min: int = 90) -> bool:
    """Poll until job reaches a terminal state. Returns True on success."""
    terminal_ok  = {"COMPLETE", "COMPLETED", "SUCCESS"}
    terminal_err = {"ERROR", "FAILED", "CANCELLED", "FAILURE", "TIMEOUT"}
    deadline = time.time() + timeout_min * 60

    while time.time() < deadline:
        time.sleep(30)
        try:
            raw = sdk.get_job_status(job_id)
            # JobStatus has .status attr ('Complete', 'Running', etc.) — not .value
            if hasattr(raw, "status"):
                status = str(raw.status).upper()
            elif hasattr(raw, "value"):
                status = str(raw.value).upper()
            else:
                status = str(raw).upper()
        except Exception as e:
            log.warning("[%s] status poll error: %s", algo_name, e)
            continue

        log.info("[%s] job %s → %s", algo_name, job_id, status)
        if status in terminal_ok:
            return True
        if status in terminal_err:
            try:
                logs = sdk.get_job_logs(job_id)
                log.error("[%s] job failed. Tail:\n%s", algo_name, logs[-1500:])
            except Exception:
                pass
            return False

    log.error("[%s] job %s timed out after %d min", algo_name, job_id, timeout_min)
    return False


def read_far_result(far_json_remote: str) -> dict:
    """Read metric_result.json from Lustre via SSH cat."""
    raw = ssh_run(f"cat {far_json_remote}")
    return json.loads(raw)


def _resolve_lustre_path(sdk, job_id: str) -> str:
    """Get the POSIX Lustre path for a job's results dir, stripping any URI scheme."""
    raw = sdk.get_job_results_dir(job_id)
    return raw.replace("lustre:///", "/").replace("lustre://", "/")


def score_job(algo_name: str, job_id: str,
              sdk, partition: str, account: str, local_log_dir: str) -> dict:
    """
    Full pipeline for one completed AutoML job:
      find best ckpt → get experiment.yaml → submit FAR SLURM job →
      poll → read result → return dict
    """
    job_results_dir = _resolve_lustre_path(sdk, job_id)
    log.info("[%s] results_dir=%s", algo_name, job_results_dir)

    # Find best checkpoint (prefer model_best_*.pth, fall back to latest symlink)
    ckpt = ssh_run(
        f"find {job_results_dir}/results_dir/train -name 'model_best_*.pth' 2>/dev/null | sort | tail -1 "
        f"|| readlink -f {job_results_dir}/results_dir/train/changenet_model_classify_latest.pth 2>/dev/null"
    )
    if not ckpt:
        raise FileNotFoundError(f"No checkpoint found in {job_results_dir}/results_dir/train")
    log.info("[%s] Using checkpoint: %s", algo_name, ckpt)

    experiment_yaml_str = ssh_run(f"cat {job_results_dir}/results_dir/train/experiment.yaml")

    lustre_out = f"{LUSTRE_AOI}/far_eval/{algo_name}"
    job_id, far_json = submit_far_job(
        algo_name, ckpt, experiment_yaml_str, lustre_out,
        sdk, partition, account,
    )

    ok = poll_job(sdk, job_id, algo_name)
    if not ok:
        raise RuntimeError(f"[{algo_name}] FAR SLURM job {job_id} failed")

    result = read_far_result(far_json)
    result["algo"]       = algo_name
    result["checkpoint"] = ckpt
    result["far_job_id"] = job_id

    # Save locally
    out = pathlib.Path(local_log_dir) / f"far_{algo_name}.json"
    out.write_text(json.dumps(result, indent=2))
    log.info("[%s] FAR@100%%R = %.2f%%  saved to %s", algo_name, result["value"], out)
    return result


def score_bayesian(local_log_dir: str, sdk, partition: str, account: str):
    result = score_job("bayesian_v1", BAYESIAN_JOB, sdk, partition, account, local_log_dir)

    print(f"\n{'='*60}")
    print(f"  Bayesian AutoML v1 — FAR@100%R = {result['value']:.2f}%")
    print(f"  Threshold  : {result['threshold']:.6f}")
    print(f"  DEFT best (iter9): 15.10%  |  HPO val_loss: 0.460")
    print(f"{'='*60}\n")
    return result


def watch_and_score(local_log_dir: str, sdk, partition: str, account: str):
    """
    Poll for summary_*.json files. When each appears, submit FAR job to SLURM,
    poll until done, read result, update comparison table.
    """
    log_dir = pathlib.Path(local_log_dir)
    algo_map = {
        "summary_bfbo.json":        "bfbo",
        "summary_bohb.json":        "bohb",
        "summary_bayesian_llm.json": "bayesian_llm",
        "summary_bfbo_llm.json":    "bfbo_llm",
        "summary_bohb_llm.json":    "bohb_llm",
    }
    remaining = set(algo_map.keys())
    scored    = {}

    log.info("Watching %s for %d algorithms...", log_dir, len(remaining))

    while remaining:
        for fname in list(remaining):
            p = log_dir / fname
            if not p.exists():
                continue

            algo = algo_map[fname]
            log.info("[%s] summary found — submitting FAR SLURM job", algo)
            remaining.discard(fname)

            try:
                summary   = json.loads(p.read_text())
                job_id    = summary.get("best_job_id", "")
                if not job_id:
                    log.warning("[%s] no best_job_id in summary", algo)
                    scored[algo] = {"algo": algo, "value": None,
                                    "error": "no best_job_id"}
                else:
                    result = score_job(algo, job_id, sdk,
                                       partition, account, local_log_dir)
                    result["val_loss"]           = summary.get("best_metric_value")
                    result["best_spec_overrides"] = summary.get("best_spec_overrides", {})
                    scored[algo] = result
            except Exception as e:
                log.exception("[%s] FAR scoring failed: %s", algo, e)
                scored[algo] = {"algo": algo, "value": None, "error": str(e)}

            _print_table(scored, log_dir)

        if remaining:
            log.info("Still waiting for: %s", [algo_map[f] for f in remaining])
            time.sleep(120)

    log.info("All algorithms scored.")
    _print_table(scored, log_dir, final=True)


def _print_table(scored: dict, log_dir: pathlib.Path, final: bool = False):
    # Merge in bayesian if already scored
    all_r = {}
    bay = log_dir / "far_bayesian_v1.json"
    if bay.exists():
        try:
            all_r["bayesian_v1"] = json.loads(bay.read_text())
        except Exception:
            pass
    all_r.update(scored)

    # Save comparison JSON
    (log_dir / "far_comparison.json").write_text(json.dumps(all_r, indent=2))

    print(f"\n{'─'*70}")
    print(f"  FAR Comparison {'(FINAL)' if final else '(running)'}")
    print(f"{'─'*70}")
    print(f"  {'Algorithm':<20}  {'FAR@100%R':>10}  {'val_loss(HPO)':>13}  {'Notes'}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*13}  {'─'*18}")
    print(f"  {'DEFT best (iter9)':<20}  {'15.10%':>10}  {'n/a':>13}  plain train baseline")
    for name, r in sorted(all_r.items()):
        far = f"{r['value']:.2f}%" if r.get("value") is not None else "FAIL/pending"
        vl  = f"{r['val_loss']:.4f}" if r.get("val_loss") else "n/a"
        print(f"  {name:<20}  {far:>10}  {vl:>13}")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bayesian", "watch", "score", "deft_check"], required=True)
    parser.add_argument("--log-dir",   default=str(WORKSPACE / "automl_logs_v2"))
    parser.add_argument("--job-id",    help="Lustre job ID for --mode score")
    parser.add_argument("--algo-name", help="Algorithm name for --mode score")
    args = parser.parse_args()

    # Load credentials
    cfg = pathlib.Path("~/.tao/config.env").expanduser()
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    partition = os.environ.get("SLURM_PARTITION", os.environ.get("SLURM_PARTITION", ""))
    account   = os.environ.get("SLURM_ACCOUNT",
                    os.environ.get("SLURM_ACCOUNT", ""))

    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    sdk = make_sdk()

    if args.mode == "bayesian":
        score_bayesian(str(log_dir), sdk, partition, account)

    elif args.mode == "watch":
        watch_and_score(str(log_dir), sdk, partition, account)

    elif args.mode == "score":
        if not args.job_id or not args.algo_name:
            sys.exit("--mode score requires --job-id and --algo-name")
        result = score_job(args.algo_name, args.job_id,
                           sdk, partition, account, str(log_dir))
        print(json.dumps(result, indent=2))

    elif args.mode == "deft_check":
        # End-to-end validation: run the DEFT iter9 checkpoint (known local
        # FAR=15.1018% @ 0.262223) through the exact SLURM eval pipeline.
        # A matching result proves inference spec + Lustre data + FAR math.
        DEFT_RESULTS = WORKSPACE / "results" / "run_20260804_205152"
        local_ckpt = DEFT_RESULTS / "iter9" / "train" / "model_epoch_009_step_00320.pth"
        local_yaml = DEFT_RESULTS / "iter9" / "train" / "experiment.yaml"
        lustre_out  = f"{LUSTRE_AOI}/far_eval/deft_iter9_check"
        lustre_ckpt = f"{lustre_out}/deft_iter9.pth"

        ssh_run(f"mkdir -p {lustre_out}")
        # Upload checkpoint only if not already there (1.8 GB)
        existing = ssh_run(f"stat -c %s {lustre_ckpt} 2>/dev/null || echo 0")
        local_size = local_ckpt.stat().st_size
        if existing.strip() != str(local_size):
            log.info("Uploading DEFT iter9 checkpoint (%.1f GB)...", local_size / 1e9)
            scp_to_cluster(str(local_ckpt), lustre_ckpt)
        else:
            log.info("Checkpoint already on Lustre (size match) — skipping upload")

        # Local experiment.yaml has container-local backbone path; patch to Lustre
        spec = yaml.safe_load(local_yaml.read_text())
        spec["model"]["backbone"]["pretrained_backbone_path"] = LUSTRE_BACKBONE
        exp_yaml_str = yaml.dump(spec, default_flow_style=False)

        job_id, far_json = submit_far_job(
            "deft_iter9_check", lustre_ckpt, exp_yaml_str, lustre_out,
            sdk, partition, account,
        )
        ok = poll_job(sdk, job_id, "deft_iter9_check")
        if not ok:
            sys.exit(f"deft_check SLURM job {job_id} failed")

        result = read_far_result(far_json)
        out = log_dir / "far_deft_iter9_check.json"
        out.write_text(json.dumps(result, indent=2))
        expected = 15.101847465656087
        got = result["value"]
        print(f"\n{'='*64}")
        print(f"  DEFT iter9 end-to-end SLURM validation")
        print(f"  Expected (local analyze_kpi): FAR = {expected:.4f}% @ 0.262223")
        print(f"  Got      (SLURM pipeline)   : FAR = {got:.4f}% @ {result['threshold']:.6f}")
        print(f"  Delta: {abs(got - expected):.4f} pp  →  "
              f"{'PIPELINE VALIDATED' if abs(got - expected) < 0.5 else 'MISMATCH — investigate'}")
        print(f"{'='*64}\n")
