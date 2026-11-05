# Thermal limiter and GPU health on Windows laptops (RTX Spark)

A fine-tune is a sustained load on a laptop. Run every GPU job through the scripts in
`../scripts/thermal/` so that the job has a start gate, a clock cap, an in-run pacer, a
hard guard and a receipt. All numbers below were measured on one RTX Spark N1X (Windows 11
on Arm, 128 GB, driver 6xx series) between 2026-10-28 and 2026-11-04 with one GPU job at a
time; treat them as the starting point for another unit, not as its specification.

## The toolkit

| Script | Role |
|---|---|
| `clock_cap.ps1 -Apply <MHz>` / `-Restore` / (no args) | Apply or restore the SM clock cap with `nvidia-smi -lgc` / `-rgc`; without arguments it probes which clock and power controls the driver accepts and leaves nothing applied. |
| `run_measured.ps1 -Tag <t> -CommandLine '<cmd>' [-Python <exe>] [-EvidenceDir <dir>] [-MaxStartTempC 62] [-TimeoutSec 3600] [-Spinner]` | Waits until the GPU is at or below the start temperature, runs the command with a 1 s sampler (`gpu_sampler.ps1`), writes `<tag>.receipt.json`, `<tag>.samples.csv` (+ `.summary.json`), stdout and stderr logs, and health snapshots before and after. `-Spinner` refuses to time a run on a degraded GPU. |
| `thermal_pacer.ps1 -Tag <t> [-PauseC 84] [-ResumeC 78] [-LimitC 87]` | Runs beside a measured job: suspends the whole job tree (NtSuspendProcess, descendants included) at the pause temperature, resumes it at the resume temperature, kills the tree at the limit, and writes `<tag>.pacer.json` with the pauses and paused seconds. The cap stays applied, so unpaused steps remain comparable. |
| `gpu_probe.sh [seconds] [python]` | Eight-second bf16 matmul loop under a 1600 MHz cap with the GPU otherwise idle, sampled once a second; prints TFLOP/s and restores the default clocks. |
| `gpu_spinner.py [seconds]` | Sustained matmul load for diagnostics and the `-Spinner` gate. |

Pattern for one job (the model skill's launcher does this for you):

```powershell
powershell -ExecutionPolicy Bypass -File clock_cap.ps1 -Apply 1600
Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','thermal_pacer.ps1','-Tag','train_job1','-EvidenceDir','.\evidence'
powershell -ExecutionPolicy Bypass -File run_measured.ps1 -Tag train_job1 -Python <venv>\Scripts\python.exe -CommandLine '"<venv>\Scripts\python.exe" -m cosmos_framework.scripts.train --sft-toml=<recipe>.toml -- <overrides>'
powershell -ExecutionPolicy Bypass -File clock_cap.ps1 -Restore
```

Restore the default clocks after every job and say so in the run log. Never run two GPU jobs at once.

## Measured operating point (RTX Spark N1X, 128 GB unit)

Cap 1600 MHz, start below 62 C, pacer pause 84 C / resume 78 C / kill 87 C.

| Cosmos3 Nano LoRA SFT input | s per step | W | GPU C | Pacer |
|---|---|---|---|---|
| 8 frames at 128x64 (clip budget 81920 px) | 0.5 to 0.6 | 17 to 18 | 73 to 77 | no pauses |
| 8 frames at 480x288 (clip budget 1179648 px) | 1.05 to 1.2 | 25 to 28 | 81 to 85 | about 250 short pauses (9 min) per 2880 steps |
| 8 frames at 832x480 | 2.9 | 27 to 28 | reaches the 87 C guard in about 10 minutes uncapped | not sustainable at 1600 |

Sim-Transfer inference (33-frame 640x480 clip): 25 to 27 s at 1600 MHz, 30 to 38 W, 73 to 80 C.

Healthy probe reading: about 70 TFLOP/s at 1599 MHz and 39 W. A reading near 7 TFLOP/s at
663 MHz and 5 to 6 W is the latched low-power state below.

## Failure modes seen, and what to do

- **Hard reset without an event-log cause** (Kernel-Power 41, bugcheck 0, no WHEA or display
  driver event). Seen under uncapped load at 90 to 94 C and, twice, at idle. Reboot, reapply
  the cap, resume; the receipts tell a chain which jobs are finished. Open items are tracked
  with the platform team.
- **Latched low-power state**: after the end of a run with hundreds of pacer cycles, the GPU
  stayed at 663 MHz and 5 to 6 W under full load with no throttle reason reported, running
  about seven times slower. `nvidia-smi -rgc` does not clear it; a reboot does. Run
  `gpu_probe.sh` after any power event and before a chain. Prefer a cap at which the pacer does
  not cycle over a higher cap with constant pausing (for the 480x288 input, 1400 MHz is the
  candidate; measure 100 steps at each before a night run).
- **Reliability clock limiter** after display-driver events (`clocks_event_reasons.active`
  non-zero at idle): every speed number taken under it is low by about 2.4x. Check the probe
  before any speed claim.

## Caveats

- The pacer identifies the job tree by a regular expression over process command lines
  (`-Pattern`); extend it when a new entry point is used.
- Under WSL the pacer cannot suspend processes inside the VM; it is a native-Windows tool.
- `nvidia-smi` memory on WDDM shows the dedicated segment, not the CUDA budget; use the
  preflight script in `../scripts/preflight_windows_arm64.py` for capacity.
