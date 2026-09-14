# Profile-matched calibration (`kpi_profile_count_bins`) — default off

## Why

Calibration rows are the loop's only lever on *what a detection answer looks
like* when the mined rows are all hard positives. The fixed buckets (512 empty
+ 512 rows with <= 2 boxes per iteration for single-image Defect Detection;
reference pairs split into no-change / <= 2-box changed at the KPI empty rate)
teach a box-count distribution the evaluation never asks for. Measured on the
frozen NVPAW benchmark vs. the training sets of finished runs (tool A profiles,
2026-09-10):

| task | benchmark empty | benchmark rows with >= 4 boxes | training set (v3b / v9b-redo) empty | training rows with >= 4 boxes |
|---|---:|---:|---:|---:|
| Defect Detection | 24.8% | 19% | 68.5% / 44.8% | 1% / 0.3% |
| Ref_based Defect Detection | 54.2% | 19% | 48.9% / 42.3% | 14% / 0.9% |

The two-image failure the image review found ("we find one defect and stop")
is the same mismatch: changed pairs in training carry one box, the benchmark's
carry 3.4 on average.

## What the policy does

`init_deft_state.py --calibration-task-total "Defect Detection=1024"
--calibration-task-total "Ref_based Defect Detection=500" [--calibration-min-fill 0.9]`
records `config.mining.calibration_quota_contract` with
`policy = kpi_profile_count_bins`:

- `task_profiles`: for every detection task, the KPI set's rows per
  ground-truth box-count bin `0 / 1 / 2-3 / 4-9 / 10+` and their shares
  (`select_detection_calibration.derive_task_count_profiles`);
- `task_bin_quotas`: each task total split across the bins by largest
  remainder (`profile_bin_quotas`);
- `min_fill_fraction`: fail-closed threshold per task (default 0.9).

Per iteration the mining runtime calls
`select_detection_calibration.select_calibration(records, media_root=...,
excluded_identities=..., pair_assets_dir=..., task_bin_quotas=contract["task_bin_quotas"],
min_fill_fraction=contract["min_fill_fraction"])` (or the CLI with
`--profile-task-total TASK=ROWS`); no `max_boxes`, cohort or legacy quota
arguments may be combined with it. Rows are read from
`config.annotations.calibration` (the Mining file unless a calibration pool was
recorded, see `preflight.md`). Selection walks the pool once, fills each
`(task, bin)` bucket in file order, keeps reference pairs content-unique and
skips previously mined / evaluation identities exactly as the cohort policy.

Rows keep the existing evidence labels (`calibration_empty_ground_truth`,
`calibration_few_box_ground_truth`, `calibration_reference_no_change_ground_truth`)
so downstream quota accounting is unchanged, and add `calibration_count_bin`
and `calibration_policy`. The summary (`detection_calibration_v4`) reports per
task and bin the requested / selected rows, the fill fraction and shortages; a
task below `min_fill_fraction` fails closed with the short bins listed.

Tasks not named in `--calibration-task-total` get no calibration rows
(Component Detection stays mining-only unless a total is given; its benchmark
profile is 0% empty and 5.3 boxes per row, so give it a total only together
with a corpus-size decision, because the row cap and the Defect Detection
floor act on the whole iteration).

## Selection order

Profile mode collects every eligible detection row first and fills each
`(task, bin)` bucket round-robin across datasets in seed/id-hash order
(`profile_seed`, default 17), so a pool sorted by dataset cannot fill a bucket
from its first dataset alone; the summary reports `datasets_per_bin`.
Reference pairs stay content-unique and any reference shortfall fails closed.

## Capability gate and acquisition mode (Phase 3)

`capability_gate.py --kpi-annotations <KPI set> --raw-report <raw_f1.json of
the gating checkpoint> --acquisition-rows N --refinement-rows TASK=ROWS
--output capability_gate.json` compares each detection task's F1 with the
KPI set's trivial always-empty F1 `2E / (2E + B)`. A task at or below it is in
**acquisition** mode: the skill is absent, so error-driven mining has nothing
to refine and the task gets a volume budget with broad dataset coverage.
Launch-record it with `init_deft_state.py --acquisition-task TASK=ROWS
--acquisition-gate capability_gate.json` (requires the profile policy): the
task's profile-binned calibration total becomes ROWS and
`calibration_quota_contract.acquisition` records tasks, rows and the gate
file. Pass the materializer `--acquisition-rows <rows>` so those rows are
excluded from the Defect Detection floor base; otherwise a large acquisition
slice forces an unsatisfiable DD quota. Re-run the gate every iteration; when
the task crosses the trivial baseline drop the override (refinement mode).

The quota manifest checks the reference empty rate over calibration **and**
mined reference rows together; the materializer therefore seeds the mined
slice's running total with the reserved calibration counts so both slices
complete one combined `floor(total × rate + 0.5)` target (separately rounded
slices missed it by one row at 3,000 + 80 rows, 2026-09-14).

## Supply check before launching

Pool box-count bins (full pool, 2026-09-10): Defect Detection `4-9` 1,358 rows,
`10+` 190; Ref_based Defect Detection `4-9` 13,558, `10+` 639. With the panel
shares above a 1,024-row DD quota asks for about 184 `4-9` and 23 `10+` rows
per iteration, so five iterations fit; report the shortage table anyway.
