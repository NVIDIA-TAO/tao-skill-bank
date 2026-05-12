# Bundled Script Usage

Detailed examples live here so `SKILL.md` stays focused on trigger behavior, workflow, and hard invariants.

## `run_script()` Invocation

Use `run_script()` when the harness provides it. Resolve every path argument to an absolute host path before calling.

```python
run_script(
    "scripts/log_stage.py",
    args=[
        "--log-path", f"{workspace_root}/results/loop_log.jsonl",
        "--iter-label", iter_label,
        "--stage", "anomalygen",
        "--status", "ok",
        "--summary", "generated 1024 triplets, 8 defect types",
        "--duration-sec", str(duration_sec),
        "--context-tokens", str(context_tokens),
    ],
)
```

## Direct Python Invocation

Use direct `python` invocation only when `run_script()` is unavailable.

```bash
python scripts/log_stage.py \
  --log-path /abs/path/results/loop_log.jsonl \
  --iter-label iter1 \
  --stage anomalygen \
  --status ok \
  --summary "generated 1024 triplets, 8 defect types" \
  --duration-sec 612 \
  --context-tokens 18432
```

## In-Process Library Use

When the parent runs a stage in-process, prefer the library API. Pass `log_path` as `pathlib.Path`; `append_stage()` intentionally rejects plain strings.

```python
from log_stage import append_stage
import pathlib

append_stage(
    pathlib.Path(f"{workspace_root}/results/loop_log.jsonl"),
    iter_label="iter1",
    stage="train",
    status="ok",
    summary="best_ckpt=ep049 FAR=0.42% threshold=0.31",
    duration_sec=duration_sec,
    context_tokens=context_tokens,
)
```

Never write `loop_log.jsonl` with `echo`, heredocs, or inline `jq`. The writer must compute `seq` from the live tail through `next_seq()`.
