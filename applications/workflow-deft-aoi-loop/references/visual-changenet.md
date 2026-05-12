# Visual ChangeNet — DEFT Loop Reference

Read this when the parent runs the `train`, `inference`, or `evaluate` stage. The
underlying skill `tao-skill-bank:visual-changenet` (`models/visual-changenet/SKILL.md`)
owns the docker invocation, spec format, CSV format, lighting conventions, and error
patterns — its `## Local Docker Invocation` section has the exact docker run command
(including `--shm-size=8g`, backbone file mount, and how to override
checkpoint/results_dir on the command line without editing the spec). This file only
covers the DEFT-loop-specific overlay: mounts, spec paths, two-checkpoint compare,
KPI sweep, and `deft_state.json` / `loop_log.jsonl` updates.

## DEFT-Loop Mount Layout

```
-v <workspace>/kpi/images:/data/datasets/NV_PCB_Siamese/images   # covers real + synthetic_iter*
-v <workspace>/train/base:/data/datasets/NV_PCB_Siamese/csv      # training_set.csv, validation_set.csv
-v <workspace>/kpi:/data/datasets/NV_PCB_Siamese/kpi             # testing_set.csv
```

## Spec Key Paths (container-side)

| What | Container path |
|---|---|
| Training CSV (iter N) | `/data/workspace/results/iter${N}/dataset/train_combined_iter${N}.csv` |
| Validation CSV | `/data/datasets/NV_PCB_Siamese/csv/validation_set.csv` |
| KPI test CSV | `/data/datasets/NV_PCB_Siamese/kpi/testing_set.csv` |
| images_dir | `/data/datasets/NV_PCB_Siamese/images` |
| Results dir (iter N) | `/results/iter${N}` |

## Two-Checkpoint Compare

Run inference on both the best-val checkpoint (lowest `val_loss`) and the latest checkpoint
(highest epoch). `val_loss` and FAR@100%-recall can diverge — pick the checkpoint with
**lower FAR@100%-recall**, not lower val_loss. See `scripts/analyze_kpi.py` for KPI sweep.

## analyze_kpi.py

```bash
python3 <skill_root>/scripts/analyze_kpi.py \
    <workspace>/results/iter${N}/inference/<label>/inference.csv \
    --output-dir <workspace>/results/iter${N}/inference/<label>
```

Key output line: `100% recall threshold: <T> (FAR=<FAR>%, ...)` — this is the KPI metric.

## Output to deft_state.json

```json
{
  "iterations": {
    "iter${N}": {
      "status": "complete",
      "best_ckpt_path": "<abs_host_path>",
      "best_ckpt_kind": "best_val|latest",
      "far_pct": <float>,
      "threshold": <float>,
      "val_loss": <float>,
      "inference_csv": "<abs_host_path>"
    }
  }
}
```

## Log Stage

```bash
python3 <skill_root>/scripts/log_stage.py \
    --log-path results/loop_log.jsonl \
    --iter-label <baseline|iter${N}> \
    --stage train --status ok \
    --summary "FAR=X% threshold=Y val_loss=Z best_ckpt=<kind>"
```
