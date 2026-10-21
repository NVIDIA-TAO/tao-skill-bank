# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the DEFT AOI loop on the poc-ongfq wafer workspace (native Windows).

Wraps the tao-run-deft-aoi-cosmos3 stage scripts into the canonical
disk-backed state machine: baseline (benchmark + proxy RCCA), then up to
MAX_ITERS iterations of routing -> per-class mining -> NG-floor balance ->
assemble -> validate -> train -> export -> benchmark gate, with a fresh proxy
RCCA after every non-final iteration. Resumable: stages already recorded ok
in deft_state.json are skipped.

Recipe (semicon mining-only run): accuracy >= 0.90 gate, <= 5 iterations,
10 epochs/iter, LoRA r16/a32/d0.05, lr 1e-4, per-class mining topn NG=15 /
OK=3, cosine floor 0.8, history exclusion, monotonic assemble. Balance policy
(stricter than the recipe's 40% floor, per user guidance): keep all NG, cap
new OK at combined parity, dropping lowest-similarity OK deterministically.
"""

import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time

# Host layout, all overridable. Defaults are the x64 reference workstation; on the RTX
# Spark N1X set DEFT_PYTHON (the TAO venv interpreter), DEFT_BANK, DEFT_FRAMEWORK_CWD (the
# patched cosmos-framework checkout) and DEFT_MODEL_BASE (the prepared VLM directory).
# Evaluate and export are invoked as modules so the venv needs no console-script wrappers.
PY = os.environ.get("DEFT_PYTHON", r"C:\cosmos-win\venv-tao\Scripts\python.exe")
EVAL = [PY, "-m", "cosmos_rl.evaluation.evaluate"]
EXPORT = [PY, "-m", "cosmos_rl.framework.export"]
BANK = pathlib.Path(os.environ.get("DEFT_BANK", str(pathlib.Path(__file__).resolve().parents[4])))


def _versions_key(key: str) -> str:
    """Resolve an image pin from the bank versions.yaml (never hard-code a URI here)."""
    out = subprocess.run([sys.executable, str(BANK / "scripts" / "resolve_versions_key.py"), key],
                         check=True, capture_output=True, text=True).stdout.strip()
    if not out:
        raise RuntimeError(f"versions.yaml has no value for {key}")
    return out
# Native TAO core otherwise imports its cloud status clients (ngcbase, not on PyPI) on every
# status write; required on ARM64, harmless everywhere.
os.environ.setdefault("TAO_DISABLE_CLOUD_STATUS_CALLBACK", "1")
DEFT = BANK / "skills/applications/tao-run-deft-aoi-cosmos3/scripts"
sys.path.insert(0, str(DEFT))
import parquet_io  # noqa: E402  (pyarrow or duckdb)
MINE = BANK / "skills/data/tao-mine-aoi-images/scripts"
WS = pathlib.Path(os.environ.get("DEFT_WAFER_WS", r"C:\cosmos-win\deft-wafer\workspace"))
RUN = WS / "results" / os.environ.get("DEFT_WAFER_RUN", "wafer-win-1")
FRAMEWORK_CWD = os.environ.get("DEFT_FRAMEWORK_CWD", r"C:\cosmos-win\cosmos-framework")
MODEL_BASE = os.environ.get("DEFT_MODEL_BASE", r"C:\cosmos-win\models\Cosmos3-Nano-VLM")
TRAIN_TOML = WS / "specs" / os.environ.get("DEFT_WAFER_TRAIN_TOML", "aoi_vlm_wafer.toml")
SYSTEM_PROMPT = "Return exactly OK or NG. Do not include reasoning or extra text."

KPI_METRIC, KPI_THRESHOLD = "accuracy", 0.90
MAX_ITERS, EPOCHS, BATCH = 5, 10, 8
NG_TOPN = int(os.environ.get("DEFT_NG_TOPN", "15"))
OK_TOPN = int(os.environ.get("DEFT_OK_TOPN", "3"))
COSINE_FLOOR = 0.8
BALANCE = os.environ.get("DEFT_BALANCE", "band")  # band | none

ANN = {r: WS / "annotations" / f"{r}.json" for r in ("proxy_kpi", "benchmark_kpi", "mining_pool")}
STATE = RUN / "deft_state.json"
LOGDIR = RUN / "logs"

EVAL_SPEC_TEMPLATE = """results_dir = '{results_dir}'
num_gpus = 1

[task]
type = ""

[dataset]
annotation_path = "{annotation}"
media_dir = "{media_dir}"
system_prompt = "{system_prompt}"

[model]
model_name = "{model}"
save_folder = "{save_folder}"
tokenizer_model_name = "qwen2.5-vl-7b"
dtype = "bfloat16"
max_length = 8192
tp_size = 1
enable_lora = false
base_model_path = "{model}"

[evaluation]
answer_type = "freeform"
num_processes = 1
skip_saved = false
seed = 1
limit = -1
shard_id = 0
batch_size = 1

[evaluation.soft_accuracy]
enabled = false
f1_threshold = 0.8

[vision]
nframes = 8

[generation]
max_retries = 3
max_tokens = 4
temperature = 0.0
repetition_penalty = 1.0
presence_penalty = 0.0
frequency_penalty = 0.0

[metrics]
names = []
bertscore_model = "microsoft/deberta-xlarge-mnli"
bertscore_lang = "en"
bertscore_device = "cpu"
bertscore_batch_size = 1

[results]
save_individual_results = true
save_confusion_matrix = false
save_metrics_summary = false
"""

TRAIN_OVERRIDES = [
    "optimizer=adamw", "trainer.distributed_parallelism=ddp",
    "model.config.parallelism.data_parallel_shard_degree=1",
    "model.config.parallelism.data_parallel_replicate_degree=1",
    # W14: attention stays at the recipe default ('cosmos' — torch cuDNN SDPA
    # adapter, native on Windows). sdpa/eager break learning on transformers 4.x
    # unless the mask-gate patch is applied.
    "model.config.policy.lora_enabled=true", "model.config.policy.lora_dropout=0.05",
    "dataloader_train.num_workers=0", "dataloader_train.persistent_workers=false",
    "dataloader_train.prefetch_factor=null",
    "dataloader_val.num_workers=0", "dataloader_val.persistent_workers=false",
    "dataloader_val.prefetch_factor=null",
    "checkpoint.load_path=",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(args, log_name, cwd=None, env=None):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGDIR / f"{log_name}.log"
    with open(log_path, "a", encoding="utf-8") as sink:
        sink.write(f"\n===== {time.strftime('%H:%M:%S')} {' '.join(map(str, args))}\n")
        sink.flush()
        rc = subprocess.run([str(a) for a in args], stdout=sink, stderr=subprocess.STDOUT,
                            cwd=cwd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"{log_name} failed rc={rc}; see {log_path}")


def done_stages():
    if not STATE.is_file():
        return set()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    return {(e["iter"], e["stage"])
            for e in state.get("events", []) if e.get("status") == "ok"}


def commit(iter_label, stage, summary, duration, **artifacts):
    args = [PY, DEFT / "commit_stage.py", "--results-dir", RUN, "--iter-label", iter_label,
            "--stage", stage, "--summary", summary, "--duration-sec", str(max(1, round(duration)))]
    for key, value in artifacts.items():
        args += [f"--{key.replace('_', '-')}", value]
    sh(args, f"{iter_label}_{stage}_commit")


def labels_of(path):
    records = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    out = []
    for rec in records:
        out.append((rec, next(t["value"] for t in rec["conversations"] if t["from"] == "gpt")))
    return out


def render_eval_spec(iter_label, role, model_dir):
    leg_dir = RUN / iter_label / f"evaluate_{role}"
    leg_dir.mkdir(parents=True, exist_ok=True)
    spec = RUN / "specs_generated" / f"eval_{role}_{iter_label}.toml"
    spec.parent.mkdir(parents=True, exist_ok=True)
    annotation = ANN["proxy_kpi" if role == "proxy" else "benchmark_kpi"]
    spec.write_text(EVAL_SPEC_TEMPLATE.format(
        results_dir=str(leg_dir).replace("\\", "/"), annotation=str(annotation).replace("\\", "/"),
        media_dir=str(WS).replace("\\", "/"), system_prompt=SYSTEM_PROMPT,
        model=str(model_dir).replace("\\", "/"), save_folder=f"cr3_wafer_{role}"), encoding="utf-8")
    return spec, leg_dir / f"cr3_wafer_{role}" / "general" / "results.json"


def run_eval_leg(iter_label, role, model_dir, skip):
    stage = f"evaluate_{role}"
    spec, results_json = render_eval_spec(iter_label, role, model_dir)
    if (iter_label, stage) not in skip:
        started = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        sh([*EVAL, "--config", spec], f"{iter_label}_{stage}", env=env)
        if not results_json.is_file():
            raise RuntimeError(f"{stage} produced no results.json at {results_json}")
        commit(iter_label, stage, f"{role} evaluate: {model_dir}", time.time() - started,
               **{("proxy_results" if role == "proxy" else "benchmark_results"): results_json})
        log(f"{iter_label} {stage} done")
    return results_json


def analyze_leg(iter_label, role, results_json, skip):
    stage = "proxy_rcca" if role == "proxy" else "benchmark_metrics"
    out_dir = RUN / iter_label / ("proxy_rcca" if role == "proxy" else "benchmark_metrics")
    if (iter_label, stage) in skip:
        return out_dir
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    sh([PY, DEFT / "analyze_gaps.py", "--results-json", results_json, "--output-dir", out_dir,
        "--kpi-metric", KPI_METRIC, "--kpi-threshold", str(KPI_THRESHOLD),
        "--evaluation-role", role], f"{iter_label}_{stage}")
    duration = time.time() - started
    if role == "proxy":
        write_rcca_report(iter_label, out_dir)
        commit(iter_label, stage, "proxy RCCA", duration,
               proxy_gaps_summary=out_dir / "gaps_summary.json",
               false_accepts=out_dir / "false_accepts.json",
               false_rejects=out_dir / "false_rejects.json",
               rcca_report=out_dir / "RCCA_Report.md")
    else:
        extra = {}
        if iter_label != "baseline":
            sh([PY, DEFT / "record_metric_result.py", "--state-path", STATE,
                "--iter-label", iter_label, "--result-json", out_dir / "metric_result.json",
                "--benchmark-results", results_json, "--training-spec", TRAIN_TOML,
                "--best-ckpt", RUN / iter_label / "export" / "merged"],
               f"{iter_label}_record_metric")
        commit(iter_label, stage, "benchmark metrics", duration,
               metric_result=out_dir / "metric_result.json",
               benchmark_metrics_summary=out_dir / "metrics_summary.json",
               benchmark_results=results_json, **extra)
    log(f"{iter_label} {stage} done")
    return out_dir


def defect_type(gap_id):
    # poc-ongfq filenames encode the source class as the third hyphen token,
    # e.g. 28734805-00-rdl_defect-81774-... -> rdl_defect
    parts = gap_id.split("-")
    return parts[2] if len(parts) > 2 else "unknown"


def breakdown_table(gaps):
    counts = {}
    for gap in gaps:
        counts[defect_type(gap["id"])] = counts.get(defect_type(gap["id"]), 0) + 1
    total = max(1, len(gaps))
    rows = [f"| `{t}` | {c} | {100 * c / total:.0f}% |"
            for t, c in sorted(counts.items(), key=lambda kv: -kv[1])]
    return ["| Defect type | Count | Share |", "|---|---:|---:|"] + (rows or ["| — | 0 | 0% |"])


def write_rcca_report(iter_label, out_dir):
    gaps = json.loads((out_dir / "gaps_summary.json").read_text(encoding="utf-8"))
    fa = json.loads((out_dir / "false_accepts.json").read_text(encoding="utf-8"))
    fr = json.loads((out_dir / "false_rejects.json").read_text(encoding="utf-8"))
    m = gaps["metrics"]
    fa_types = sorted({defect_type(g["id"]) for g in fa})
    per_defect = {}
    for kind, group in (("fa", fa), ("fr", fr)):
        for gap in group:
            per_defect.setdefault(defect_type(gap["id"]), {"fa": 0, "fr": 0})[kind] += 1
    worst = fa[:5] + fr[:5]
    lines = [f"# Cosmos3 Proxy RCCA Report: `{iter_label}`", "",
             "## 1. Verdict", "",
             f"Proxy accuracy {m['accuracy']:.4f} (recall_ng {m['recall_ng']:.3f}, "
             f"precision_ng {m['precision_ng']:.3f}); {len(fa)} false accepts, "
             f"{len(fr)} false rejects, {gaps['unknown_samples']} unknowns on "
             f"{gaps['samples']} records. The KPI gate ({KPI_METRIC} >= {KPI_THRESHOLD} "
             "on the frozen Benchmark) is judged only by the Benchmark leg; every proxy "
             "gap below is routed to per-class similarity mining. Headline: "
             + ("over-rejection dominates — OK records misread as NG."
                if len(fr) > len(fa) else "missed defects dominate — NG records passed."), "",
             "## 2. False-Accept Breakdown", ""] + breakdown_table(fa) + ["",
             "## 3. False-Reject Breakdown", ""] + breakdown_table(fr) + ["",
             "## 4. Top-K Worst Samples", "",
             "False accepts are shipped defects and rank above false alarms; first "
             "5 of each kind in evaluator order:", ""]
    lines += [f"- `{g['id']}` — {'false accept' if g in fa else 'false reject'} "
              f"(gt {g['gt']} -> pred {g['pred']}, type `{defect_type(g['id'])}`)"
              for g in worst] or ["- none"]
    lines += ["", "## 5. Per-Defect Analysis", "",
              "| Defect type | False accepts | False rejects | RCCA finding |",
              "|---|---:|---:|---|"]
    lines += [f"| `{t}` | {v['fa']} | {v['fr']} | "
              f"{'missed defect pattern; mine NG lookalikes' if v['fa'] else 'false-alarm pattern; mine OK lookalikes'} |"
              for t, v in sorted(per_defect.items(), key=lambda kv: -(kv[1]['fa'] + kv[1]['fr']))]
    lines += ["", "## 6. Recommended Actions", "",
              f"- Mine NG lookalikes for the {len(fa)} false-accept targets "
              f"(top-{NG_TOPN} each, pool filtered to NG"
              + (f"; defect types {', '.join(fa_types)}" if fa_types else "") + ").",
              f"- Mine OK lookalikes for the {len(fr)} false-reject targets "
              f"(top-{OK_TOPN} each, pool filtered to OK).",
              f"- Apply cosine floor {COSINE_FLOOR}, mining-history exclusion, and the "
              "combined-parity OK cap before assembly.",
              "- SDG: none — AnomalyGen is not ported to Windows; this is a "
              "mining-only run."]
    (out_dir / "RCCA_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def routing(iter_label, rcca_dir, skip):
    out = RUN / iter_label / "routing"
    targets_json = out / "mining_targets.json"
    if (iter_label, "routing") in skip:
        return targets_json
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    by_id = {rec.get("id"): rec for rec, _ in labels_of(ANN["proxy_kpi"])}
    targets = []
    for name, kind in (("false_accepts", "false_accept"), ("false_rejects", "false_reject")):
        for gap in json.loads((rcca_dir / f"{name}.json").read_text(encoding="utf-8")):
            rec = by_id.get(gap.get("id"))
            if rec is None:
                continue
            targets.append({"id": gap["id"], "kind": kind, "images": rec["images"],
                            "target_filepath": str(WS / rec["images"][0]),
                            "gt": gap["gt"], "response": gap.get("response", gap.get("pred"))})
    targets_json.write_text(json.dumps(targets, indent=1), encoding="utf-8")
    parquet_io.write_columns({"filepath": [t["target_filepath"] for t in targets],
                              "id": [t["id"] for t in targets],
                              "kind": [t["kind"] for t in targets]}, out / "mining_gaps.parquet")
    commit(iter_label, "routing",
           f"route {len(targets)} proxy gaps to mining", time.time() - started,
           mining_targets=targets_json)
    log(f"{iter_label} routing done ({len(targets)} targets)")
    return targets_json


def mine(iter_label, targets_json, skip):
    out = RUN / iter_label / "mining"
    balanced = out / "mined_balanced.json"
    if (iter_label, "data_mining") in skip:
        return balanced
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    targets = json.loads(targets_json.read_text(encoding="utf-8"))
    # false accepts are missed NG -> mine NG lookalikes; false rejects -> OK.
    plan = {"NG": ("false_accept", NG_TOPN), "OK": ("false_reject", OK_TOPN)}
    class_tables, class_dirs = [], []
    for label, (kind, topn) in plan.items():
        paths = sorted({t["target_filepath"] for t in targets if t["kind"] == kind})
        if not paths:
            continue
        class_dir = out / label.lower()
        class_dir.mkdir(exist_ok=True)
        parquet_io.write_columns({"filepath": paths}, class_dir / "targets.parquet")
        cache = []
        for j in range(int(iter_label.removeprefix("iter")) - 1, 0, -1):
            prior = RUN / f"iter{j}" / "mining" / label.lower() / "source_embeddings.parquet"
            if prior.is_file():
                cache = ["--reuse-source-embeddings", str(prior)]
                break
        sh([PY, MINE / "native_mine_aoi.py", "--targets-parquet", class_dir / "targets.parquet",
            "--source-annotations", ANN["mining_pool"], "--media-root", WS,
            "--output-dir", class_dir, "--top-k", str(topn), "--filter-label", label, *cache],
           f"{iter_label}_mine_{label}")
        sh([PY, DEFT / "filter_mined_by_cosine.py",
            "--mined-parquet", class_dir / "mined_raw.parquet",
            "--source-embeddings", class_dir / "source_embeddings.parquet",
            "--target-embeddings", class_dir / "target_embeddings.parquet",
            "--output", class_dir / "mined_cos.parquet",
            "--summary", class_dir / "cosine_filter_summary.json",
            "--min-similarity", str(COSINE_FLOOR)], f"{iter_label}_cosine_{label}")
        class_tables.append(parquet_io.read_rows(class_dir / "mined_cos.parquet"))
        class_dirs.append(class_dir)
    if not class_tables:
        return None
    for name in ("target_embeddings", "source_embeddings"):
        parts = [parquet_io.read_rows(d / f"{name}.parquet") for d in class_dirs]
        cols: list[str] = []  # union of columns in first-seen order (promote)
        for part in parts:
            for c in (part[0].keys() if part else ()):
                if c not in cols:
                    cols.append(c)
        parquet_io.write_rows([r for part in parts for r in part], out / f"{name}.parquet", columns=cols)
    # keep the best-similarity row per filepath across classes
    rows = sorted((r for part in class_tables for r in part), key=lambda r: -r["similarity"])
    best, seen = [], set()
    for row in rows:
        if row["filepath"] not in seen:
            seen.add(row["filepath"])
            best.append(row)
    parquet_io.write_rows(best, out / "mined_candidates.parquet",
                          columns=("filepath", "label", "target_filepath", "similarity"))
    history_file = RUN / "mining_history.json"
    iteration_no = int(iter_label.removeprefix("iter"))
    resume = []
    if history_file.is_file():
        ledger = json.loads(history_file.read_text(encoding="utf-8"))
        entries = ledger.get("iterations", [])
        committed = {int(e.get("iteration", 0)) for e in entries} if isinstance(entries, list) else set()
        if iteration_no in committed:
            resume = ["--resume"]
    sh([PY, MINE / "filter_mined_history.py", "--candidate-parquet", out / "mined_candidates.parquet",
        "--output-parquet", out / "mined_filtered.parquet",
        "--history-file", history_file,
        "--summary", out / "mining_history_summary.json",
        "--iteration", str(iteration_no), "--topn", str(NG_TOPN + OK_TOPN), *resume],
       f"{iter_label}_history")
    if parquet_io.num_rows(out / "mined_filtered.parquet") == 0:
        log(f"{iter_label}: history filter dropped every candidate — pool exhausted "
            "around the remaining gaps")
        return None
    sh([PY, DEFT / "emit_mined_sharegpt.py", "--mined-parquet", out / "mined_filtered.parquet",
        "--source-annotations", ANN["mining_pool"], "--media-root", WS,
        "--output", out / "mined_sharegpt.json", "--summary", out / "emit_mined_summary.json",
        "--emit-relative"], f"{iter_label}_emit")
    # Class-balance policy (user guidance, confirmed by wafer-win-1): keep every
    # NG (scarce class); cap new OK at COMBINED parity with NG so the assembled
    # set never goes majority-OK. Drop lowest-similarity OK first,
    # deterministically; previously assembled records are never modified.
    sim = {r["filepath"]: r["similarity"] for r in
           parquet_io.read_rows(out / "mined_filtered.parquet")}
    labelled = labels_of(out / "mined_sharegpt.json")
    def sim_of(rec):
        return sim.get(str(WS / rec["images"][0]), sim.get(rec["images"][0], 0.0))
    ng = [rec for rec, lab in labelled if lab == "NG"]
    ok = sorted((rec for rec, lab in labelled if lab == "OK"),
                key=lambda rec: (-sim_of(rec), rec["images"][0]))
    idx = int(iter_label.removeprefix("iter"))
    prev_ng = prev_ok = 0
    if idx > 1:
        prev_json = RUN / f"iter{idx - 1}" / "assemble" / f"train_iter_{idx - 1}.json"
        for _, lab in labels_of(prev_json):
            if lab == "NG":
                prev_ng += 1
            else:
                prev_ok += 1
    combined_ng = prev_ng + len(ng)
    if BALANCE == "none":
        # Error-profile-weighted assembly: keep everything mining selected.
        ok_budget, dropped = len(ok), 0
    else:
        # Balance band: the combined NG share never drops below 40% (the
        # recipe's floor, applied cumulatively). Strict parity dead-ends when
        # NG mining runs dry while the model still over-rejects; the band
        # keeps correction possible.
        ok_budget = max(0, int(combined_ng / 0.40) - combined_ng - prev_ok)
    dropped = max(0, len(ok) - ok_budget)
    ok = ok[:ok_budget]
    kept = ng + ok
    if not kept:
        log(f"{iter_label}: nothing addable under the balance band "
            f"(new NG {len(ng)}, prev {prev_ng} NG/{prev_ok} OK)")
        return None
    balanced.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    summary = {"kept": len(kept), "ng": len(ng), "ok": len(ok),
               "ok_dropped_for_parity": dropped,
               "prev_ng": prev_ng, "prev_ok": prev_ok,
               "combined_ng_share": round((prev_ng + len(ng)) /
                                          max(1, prev_ng + prev_ok + len(kept)), 4)}
    (out / "balance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    commit(iter_label, "data_mining",
           f"mined {len(kept)} (NG {len(ng)}/OK {len(ok)}, {dropped} OK dropped for floor)",
           time.time() - started,
           mining_parquet=out / "mined_filtered.parquet",
           mining_candidates=out / "mined_candidates.parquet",
           mining_summary=out / "emit_mined_summary.json",
           mining_history=RUN / "mining_history.json",
           mining_history_summary=out / "mining_history_summary.json",
           mining_target_embeddings=out / "target_embeddings.parquet",
           mining_source_embeddings=out / "source_embeddings.parquet",
           mined_sharegpt=out / "mined_sharegpt.json",
           mining_count=str(parquet_io.num_rows(out / "mined_filtered.parquet")))
    log(f"{iter_label} mining done: {summary}")
    return balanced


def assemble_validate(iter_label, iter_index, balanced, skip):
    out = RUN / iter_label / "assemble"
    train_json = out / f"train_iter_{iter_index}.json"
    if (iter_label, "assemble_data") not in skip:
        started = time.time()
        out.mkdir(parents=True, exist_ok=True)
        args = [PY, DEFT / "assemble_training_json.py", "--new-json", balanced,
                "--output", train_json, "--summary", out / "assemble_summary.json", "--dedupe"]
        if iter_index > 1:
            args += ["--previous-json",
                     RUN / f"iter{iter_index - 1}" / "assemble" / f"train_iter_{iter_index - 1}.json"]
        sh(args, f"{iter_label}_assemble")
        # Batch-size multiple (route doc rule). CosmosDataLoader's map-style resume
        # guard raises on the LAST batch of an epoch when len(train) % BATCH != 0:
        #   "Map-style resume cannot safely stamp a multi-sample batch whose
        #    _dp_stream_pos values are non-contiguous or span epochs (reordering
        #    batcher + batch_size>1)"
        # (wafer-win-arm-1, 3,444 records, died at step 430/431 after 27 min).
        # assemble_training_json keeps previous records first and appends the new
        # ones in mining order (NG, then OK by descending similarity), so the tail
        # is the lowest-similarity newly mined OK records: drop those.
        records = json.loads(train_json.read_text(encoding="utf-8"))
        excess = len(records) % BATCH
        if excess:
            dropped = records[-excess:]
            train_json.write_text(json.dumps(records[:-excess], indent=1), encoding="utf-8")
            summary_path = out / "assemble_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["batch_multiple_trim"] = {
                "batch": BATCH, "dropped": excess,
                "dropped_labels": [next(t["value"] for t in r["conversations"] if t["from"] == "gpt")
                                   for r in dropped],
                "dropped_images": [r["images"][0] for r in dropped],
                "reason": "CosmosDataLoader map-style resume guard rejects epoch-spanning batches",
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log(f"{iter_label} assemble: dropped {excess} tail record(s) "
                f"({','.join(summary['batch_multiple_trim']['dropped_labels'])}) "
                f"to reach a multiple of {BATCH}: {len(records) - excess}")
        commit(iter_label, "assemble_data", "monotonic train JSON", time.time() - started,
               combined_training=train_json, assemble_summary=out / "assemble_summary.json",
               mined_sharegpt=balanced)
        log(f"{iter_label} assemble done")
    if (iter_label, "validate_data") not in skip:
        started = time.time()
        vdir = RUN / iter_label / "validate"
        vdir.mkdir(parents=True, exist_ok=True)
        sh([PY, DEFT / "validate_sharegpt.py", "--annotations", train_json, "--media-root", WS,
            "--require-files", "--summary", vdir / "validation_report.json"],
           f"{iter_label}_validate")
        sh([PY, DEFT / "validate_split_contract.py", "--workspace", WS, "--train", train_json,
            "--proxy", ANN["proxy_kpi"], "--benchmark", ANN["benchmark_kpi"],
            "--mining", ANN["mining_pool"], "--summary", vdir / "split_contract.json"],
           f"{iter_label}_split")
        commit(iter_label, "validate_data", "sharegpt + split contract", time.time() - started,
               validation_report=vdir / "validation_report.json")
        log(f"{iter_label} validate done")
    return train_json


def train_export(iter_label, train_json, skip):
    merged = RUN / iter_label / "export" / "merged"
    if (merged / "model.safetensors.index.json").is_file():
        return merged
    out_root = RUN / iter_label / "train"
    existing = sorted(out_root.glob("**/checkpoints/epoch_*"),
                      key=lambda p: int(p.name.split("_")[1]))
    if (iter_label, "train") in skip and existing:
        ckpt = existing[-1]
        env = dict(os.environ, VLM_SAFETENSORS_PATH=MODEL_BASE, PYTHONIOENCODING="utf-8")
        return export_merged(iter_label, ckpt, merged, env)
    started = time.time()
    # W15: a leftover async-checkpoint writer on MASTER_PORT+2 makes this run
    # train fully, then hang forever at the final save with an empty ckpt dir.
    import socket
    for port in (29500, 29502):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(
                    f"port {port} is in use — kill the orphaned trainer/checkpoint "
                    f"writer first (netstat -ano | findstr :{port})")
    records = len(json.loads(train_json.read_text(encoding="utf-8")))
    steps = max(1, math.ceil(records / BATCH))
    if out_root.exists():
        shutil.rmtree(out_root)
    env = dict(os.environ, VLM_SAFETENSORS_PATH=MODEL_BASE,
               AOI_TRAIN_ANNOTATION=str(train_json), AOI_TRAIN_MEDIA=str(WS),
               AOI_VAL_ANNOTATION=str(train_json), AOI_VAL_MEDIA=str(WS), AOI_VAL_LIMIT="16",
               AOI_SYSTEM_PROMPT=SYSTEM_PROMPT, AOI_MAX_PIXELS="401408", AOI_MIN_PIXELS="50176",
               MASTER_ADDR="127.0.0.1", MASTER_PORT="29500", RANK="0", WORLD_SIZE="1",
               LOCAL_RANK="0", USE_LIBUV="0", PYTHONIOENCODING="utf-8",
               IMAGINAIRE_OUTPUT_ROOT=str(out_root))
    # Anneal the cosine over the ACTUAL run length (cosmos-rl parity): the
    # recipe's huge default cycle keeps lr at ~99% of peak for short runs.
    # +10 margin: stepping past the last cycle crashes the scheduler.
    total_steps = steps * EPOCHS
    extra = []
    lora_targets = os.environ.get("DEFT_LORA_TARGETS", "")
    if lora_targets:  # e.g. 7-module cosmos-rl parity: attention + MLP projections
        # embedded quotes: hydra treats bare commas as an ambiguous sweep
        extra.append(f'model.config.policy.lora_target_modules="{lora_targets}"')
    sh([PY, "-m", "cosmos_framework.scripts.train", f"--sft-toml={TRAIN_TOML}", "--",
        *TRAIN_OVERRIDES, f"trainer.steps_per_epoch={steps}",
        f"scheduler.cycle_lengths=[{total_steps + 10}]", *extra],
       f"{iter_label}_train", cwd=FRAMEWORK_CWD, env=env)
    ckpts = sorted(out_root.glob("**/checkpoints/epoch_*"),
                   key=lambda p: int(p.name.split("_")[1]))
    if not ckpts:
        raise RuntimeError("training saved no epoch checkpoint")
    ckpt = ckpts[-1]
    if (iter_label, "train") not in skip:
        commit(iter_label, "train",
               f"LoRA {EPOCHS} epochs x {steps} steps on {records} records",
               time.time() - started, best_ckpt=ckpt, training_spec=TRAIN_TOML)
    log(f"{iter_label} train done ({records} records, ckpt {ckpt.name})")
    return export_merged(iter_label, ckpt, merged, env)


def export_merged(iter_label, ckpt, merged, env):
    export_dir = RUN / iter_label / "export"
    # export takes the trainer's resolved config.yaml (TOML is rejected)
    resolved_config = ckpt.parent.parent / "config.yaml"
    sh([*EXPORT, "--checkpoint-path", ckpt, "--config-file", resolved_config,
        "--output-dir", merged, "--results-dir", export_dir,
        "--base-model-path-or-uri", MODEL_BASE], f"{iter_label}_export", env=env)
    index = json.loads((merged / "model.safetensors.index.json").read_text(encoding="utf-8"))
    lora_keys = [k for k in index.get("weight_map", {}) if "lora" in k.lower()]
    if lora_keys:
        raise RuntimeError(f"export left {len(lora_keys)} lora keys in the merged index")
    log(f"{iter_label} export done (0 lora keys)")
    # Disk hygiene: the ~33 GB DCP checkpoint is superseded by the verified
    # merged export; keep only config.yaml and the merged weights.
    shutil.rmtree(ckpt.parent, ignore_errors=True)
    return merged


def gate_met(iter_label):
    result = json.loads((RUN / iter_label / "benchmark_metrics" / "metric_result.json")
                        .read_text(encoding="utf-8"))
    met = result["value"] >= KPI_THRESHOLD and result.get("unknown_samples", 0) == 0
    log(f"{iter_label} benchmark {KPI_METRIC}={result['value']:.4f} "
        f"(recall_ng={result['metrics']['recall_ng']:.3f}) met={met}")
    return met


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    skip = done_stages()
    if not STATE.is_file():
        render_eval_spec("baseline", "benchmark", MODEL_BASE)
        render_eval_spec("baseline", "proxy", MODEL_BASE)
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True,
                             text=True).stdout.strip()
        sh([PY, DEFT / "init_deft_state.py", "--results-dir", RUN, "--workspace", WS,
            "--platform", "windows-venv", "--max-iterations", str(MAX_ITERS),
            "--kpi-metric", KPI_METRIC, "--kpi-threshold", str(KPI_THRESHOLD),
            "--base-model", "nvidia/Cosmos3-Nano", "--media-root", WS,
            "--train-spec", TRAIN_TOML,
            "--proxy-spec", RUN / "specs_generated" / "eval_proxy_baseline.toml",
            "--benchmark-spec", RUN / "specs_generated" / "eval_benchmark_baseline.toml",
            "--proxy-annotations", ANN["proxy_kpi"],
            "--benchmark-annotations", ANN["benchmark_kpi"],
            "--mining-annotations", ANN["mining_pool"],
            "--num-gpus", "1", "--gpu-model", gpu or "unknown",
            "--num-epochs", str(EPOCHS), "--batch-size", str(BATCH),
            "--learning-rate", "1e-4", "--lora-r", "16", "--lora-alpha", "32",
            "--top-k-per-target", str(NG_TOPN), "--min-similarity", str(COSINE_FLOOR),
            "--python-executable", PY, "--network-mode", "network-enabled",
            "--network-mode-source", "online Windows workstation",
            # lineage pins from versions.yaml (native route: recorded, not pulled)
            "--cosmos-container", _versions_key("images.tao_toolkit.cosmos_rl"),
            "--mining-container", _versions_key("images.tao_toolkit.data_services")], "init")
        log("state initialized")
        skip = done_stages()

    # baseline
    bench_results = run_eval_leg("baseline", "benchmark", MODEL_BASE, skip)
    analyze_leg("baseline", "benchmark", bench_results, skip)
    if gate_met("baseline"):
        log("baseline already meets the gate; nothing to improve")
        return 0
    proxy_results = run_eval_leg("baseline", "proxy", MODEL_BASE, skip)
    analyze_leg("baseline", "proxy", proxy_results, skip)

    rcca_dir = RUN / "baseline" / "proxy_rcca"
    total_started = time.time()
    for i in range(1, MAX_ITERS + 1):
        iter_label = f"iter{i}"
        skip = done_stages()
        targets = routing(iter_label, rcca_dir, skip)
        balanced = mine(iter_label, targets, skip)
        if balanced is None or not balanced.is_file():
            log(f"{iter_label}: mining pool exhausted or no candidates; stopping")
            sh([PY, DEFT / "finalize_run.py", "--results-dir", RUN, "--iter-label",
                f"iter{i - 1}" if i > 1 else "baseline", "--stop-reason", "pool_exhausted",
                "--mining-history-summary",
                RUN / iter_label / "mining" / "mining_history_summary.json",
                "--duration-sec", str(max(1, round(time.time() - total_started)))], "finalize")
            break
        train_json = assemble_validate(iter_label, i, balanced, skip)
        merged = train_export(iter_label, train_json, skip)
        bench_results = run_eval_leg(iter_label, "benchmark", merged, skip)
        analyze_leg(iter_label, "benchmark", bench_results, skip)
        if i > 1:  # drop the previous model only if the new one scores at least as well
            metric = lambda label: json.loads(
                (RUN / label / "benchmark_metrics" / "metric_result.json")
                .read_text(encoding="utf-8"))["value"]
            if metric(f"iter{i - 1}") <= metric(iter_label):
                shutil.rmtree(RUN / f"iter{i - 1}" / "export" / "merged", ignore_errors=True)
        if gate_met(iter_label):
            sh([PY, DEFT / "finalize_run.py", "--results-dir", RUN, "--iter-label", iter_label,
                "--stop-reason", "metric_met",
                "--duration-sec", str(max(1, round(time.time() - total_started)))], "finalize")
            log("gate met; run finalized")
            break
        if i == MAX_ITERS:
            sh([PY, DEFT / "finalize_run.py", "--results-dir", RUN, "--iter-label", iter_label,
                "--stop-reason", "max_iterations",
                "--duration-sec", str(max(1, round(time.time() - total_started)))], "finalize")
            log("max iterations reached; run finalized")
            break
        proxy_results = run_eval_leg(iter_label, "proxy", merged, skip)
        analyze_leg(iter_label, "proxy", proxy_results, skip)
        rcca_dir = RUN / iter_label / "proxy_rcca"

    sh([PY, DEFT / "render_report.py", "--results-dir", RUN, "--require-terminal"], "report")
    log("report rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
