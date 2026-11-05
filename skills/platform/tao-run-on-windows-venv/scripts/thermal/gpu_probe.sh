#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# gpu_probe.sh [seconds] [python] - bf16 8192^2 matmul loop under the 1600 MHz cap with the GPU otherwise idle, sampled by nvidia-smi
# once a second while it runs. Healthy N1X: ~1599 MHz, 95 % util, ~39 W, ~70 TFLOP/s. Latched low-power state (seen 2026-11-04
# twice, each time within a minute of a paced training run's end): 663 MHz, 95 %, 5-6 W, ~7 TFLOP/s, cleared only by a reboot.
# Run after any power event or reboot before launching a chain. Restores default clocks at the end and says so.
set -u
# Python: the second argument, or TAO_WINDOWS_PYTHON, or python on PATH. The clock cap script sits next to this one.
PY=${2:-${TAO_WINDOWS_PYTHON:-python}}
HERE=$(cd "$(dirname "$0")" && pwd)
CAP="$HERE/clock_cap.ps1"
SECS=${1:-8}
echo "[$(date -u +%FT%TZ)] probe: idle snapshot: $(nvidia-smi -i 0 --query-gpu=temperature.gpu,clocks.sm,pstate,utilization.gpu,power.draw,memory.used --format=csv,noheader)"
powershell -NoProfile -ExecutionPolicy Bypass -File "$CAP" -Apply 1600 >/dev/null 2>&1 && echo "[$(date -u +%FT%TZ)] cap 1600 MHz applied"
"$PY" - "$SECS" <<'PYEOF'
import subprocess, sys, time, torch
secs = float(sys.argv[1])
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
c = a @ b; torch.cuda.synchronize()
smi = subprocess.Popen(["nvidia-smi", "-i", "0", "--query-gpu=timestamp,temperature.gpu,clocks.sm,pstate,utilization.gpu,power.draw",
                        "--format=csv,noheader", "-lms", "1000"])
t0 = time.time(); n = 0
while time.time() - t0 < secs:
    c = a @ b; torch.cuda.synchronize(); n += 1
dt = time.time() - t0
smi.terminate()
print(f"probe: {n} bf16 8192^2 matmuls in {dt:.1f} s = {n * 2 * 8192**3 / dt / 1e12:.1f} TFLOP/s (healthy ~70, latched ~7)", flush=True)
PYEOF
powershell -NoProfile -ExecutionPolicy Bypass -File "$CAP" -Restore >/dev/null 2>&1 && echo "[$(date -u +%FT%TZ)] default clocks restored"
