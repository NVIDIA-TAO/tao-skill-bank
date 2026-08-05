# Failure analysis & retry — classify before resubmitting

When `status` reaches `ERROR`, **read the tail of the logs and classify the
failure before deciding to retry** — never blindly resubmit (a bad spec would
burn the whole retry budget on GPU-hours). This is a judgment you make from the
log; the signals below are guidance, not an exhaustive table.

- **Infrastructure → retriable** (a node / hardware / transport fault a resubmit
  may escape): SLURM `NODE_FAIL` / `BOOT_FAIL` / preemption; GPU `Xid` / `ECC` /
  "fell off the bus"; driver-too-old; genuine `NCCL` / InfiniBand / RDMA transport
  failures; `ImagePullBackOff`; pod `Evicted`. Resubmit under a NEW job-record
  with `--retry-of <id>` (mark the old one `ERROR --err-class ERR_INFRA`), up to
  **10**. SLURM's `#SBATCH --requeue` handles node-fault/preemption at the
  scheduler level for free.
- **Program → never retry** (a code / data / config fault a resubmit will just
  repeat): `OOMKilled` / CUDA out-of-memory (reduce batch); CUDA **device-side
  assert** / illegal-memory-access / CUBLAS/CUDNN status (a code or label-range
  bug); missing files; Python tracebacks; timeout / deadline (raise the limit).
  Surface the cause to the user.

Two context calls to apply with judgment: a **device-side assert** is a program
bug *even when it cascades into a CUDA/NCCL error* — the assert is the root cause;
and a bare traceback that is *downstream* of a real `Xid` / node fault is a
symptom — retry the node fault. (These are exactly what a regex table gets wrong,
which is why classification lives here as agent judgment, not in a script.)

**In-turn** monitoring is the agent polling `status` / `logs` at the interval.
**Post-turn** (the turn may end before a long job finishes): background a poller
with the harness (`run_in_background` wait-loop, `CronCreate`, or `/loop`). The
poller auto-resubmits only the **unambiguous state-based** infra cases
(`NODE_FAIL` / `BOOT_FAIL` / `PREEMPTED`) and, for auto-cleanup backends (K8s
`ttlSecondsAfterFinished`, docker `--rm`), writes the **daemon-independent
terminal record** (`tao_job_record mark --state <terminal> --source poller`)
*before* the backend object is deleted; anything nuanced re-wakes the agent to
classify. Pollers are idempotent — on re-attach, just re-establish one.
