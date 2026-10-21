# The Windows patch surface

Stock upstream does not run on Windows. The image named in
[`cosmos3-windows-contract.toml`](cosmos3-windows-contract.toml) (`image.sources`)
carries patched checkouts of **three** repositories. This file is the map of what
was changed and why, so that a rebase does not silently drop a fix.

The patch surface is a **union across three repositories**. Counting only
cosmos-framework, or only the TAO layer, understates it in both directions.

| Repository | Branch / tip | Shape of the change |
|---|---|---|
| cosmos-framework | `windows-port-pr164` (PR #164 plus the Windows port patch) | 13 files touched, including one new file |
| cosmos-rl | `windows-port`, tip `62da6f8` | exactly 4 commits, 2 files |
| tao-core | the pwd-guard commit | 1 commit, 1 file, 1 import |

---

## cosmos-framework

Patch file: `references/cosmos-framework-windows-port-pr164.patch` in this
skill. It applies cleanly on top of PR #164 (`git fetch origin pull/164/head`).

### The count is reported two different ways, and both are right

Upstream-facing notes say **nine patches**; the patch file touches **thirteen
files**. These are not in conflict and neither number should be quietly picked
over the other:

- **9** counts *distinct fixes* — one per POSIX assumption (affinity, SIGALRM,
  NCCL backend, DDP at world size 1, gloo CUDA staging, `fcntl`, `os.symlink`,
  `multistorageclient`, `import triton`).
- **13** counts *files in the diff* — several fixes span more than one call site
  (`fcntl` is imported in three places, `multistorageclient` in four), and the
  diff also adds a brand-new module.

When you report the surface, state both and say which you mean. "Nine patches
across thirteen files, one of them new" is the accurate sentence.

### Files in the diff

```
cosmos_framework/model/generator/reasoner/qwen3_vl_moe/moe_kernels.py
cosmos_framework/model/tokenizer/evaluation/lpips_cache.py
cosmos_framework/scripts/_train.py
cosmos_framework/scripts/dataset_utils.py
cosmos_framework/trainer/__init__.py
cosmos_framework/utils/checkpoint_db.py
cosmos_framework/utils/distributed.py
cosmos_framework/utils/easy_io/backends/__init__.py
cosmos_framework/utils/easy_io/backends/registry_utils.py
cosmos_framework/utils/easy_io/file_client.py
cosmos_framework/utils/easy_io/transient_retry.py
cosmos_framework/utils/reasoner/distributed.py
cosmos_framework/utils/win_compat.py          <- NEW
```

`cosmos_framework/utils/win_compat.py` is the only added file. It is where the
platform shims live rather than being scattered inline, which is what makes the
port reviewable — and what makes an accidental deletion during a rebase obvious.

### The nine fixes

| Area | Fix | What it prevents |
|---|---|---|
| `os.sched_setaffinity` / `sched_getaffinity` | `hasattr` guard | POSIX-only; the `AttributeError` escapes an `NVMLError`-only handler and aborts **every** entry point |
| `signal.SIGALRM` / `signal.alarm` | `hasattr` guard | no Windows equivalent; otherwise every run dies at trainer construction. The hang watchdog is disabled as a result |
| NCCL backend | fall back to gloo on win32 | `is_nccl_available()` is False on Windows, and a process group is still mandatory because trainer init runs an unconditional `dist.barrier()` |
| DDP wrapper | skip at `world_size == 1` | gloo cannot run a collective on a CUDA tensor; this is what makes single-GPU work at all |
| gloo CUDA collectives | stage tensors through host memory | same root cause, for the collectives that remain |
| `import fcntl` x3 | `msvcrt`-based `flock` shim | module-level imports. Two are `flock`; the third uses `SIOCGIFADDR` / `SIOCGIFCONF` ioctls with **no Windows analogue** — multi-node IB discovery only, which is unreachable here anyway |
| `os.symlink` for the rank-0 job dir | symlink, then directory junction, then pointer file | needs Developer Mode or `SeCreateSymbolicLinkPrivilege`, usually off |
| `multistorageclient` x4 | optional imports | no Windows wheel **and** no sdist (its dep `xattr` needs POSIX xattrs); backs only the optional `s3` / `msc` backends |
| unguarded `import triton` | guard, or install `triton-windows` | reached transitively for **dense, non-MoE** models via `qwen3_vl_moe/moe_kernels.py` |

Two adjacent findings are packaging, not portability, and are fixed the same way:
`checkpoint_db.py` shells out to the `uv` binary (a venv's `Scripts\` is not on
`PATH` unless activated, giving an opaque `FileNotFoundError [WinError 2]` that
surfaces as a Hydra `InstantiationException`), and its
`hf download ... --include '*'` parses `*` as a positional filename, fetches only
`README.md`, and **still exits 0** — failing much later as a `TypeError` in the
tokenizer.

---

## The PR #164 rebase rule

State this before any rebase. Verbatim:

> \#164 makes the multistorageclient ImportError guards unnecessary, BUT its
> affinity change widens the handler to OSError and STILL misses AttributeError,
> so the sched_setaffinity/sched_getaffinity hasattr guard MUST be retained.
> \#164's automatic sdpa selection is EDGE-recipe only. Nano must retain its
> recipe-default cosmos attention; do not resurrect the former Windows sdpa
> override unless the W14 mask-gate patch is also applied for debugging.

Upstream cosmos-framework fixed W14 itself in Release 2026-08-14 (#198, commit
`18d90ea`: `attention_mask` forwarded, `position_ids` no longer precomputed).
The mask-gate patch is therefore needed only while the Windows port stays on
the PR #164 base; a rebase onto that release or later drops it (W14 in
`cosmos3-windows-gaps.md`).

In other words, exactly one of the nine fixes becomes redundant on #164, and the
two that look redundant are not:

- **Drop**: the `multistorageclient` optional-import guards. #164 added its own
  `ImportError` guards.
- **Keep**: the affinity `hasattr` guard. #164 touches the same code and reads
  like it fixed this — it adds `sched_getaffinity` and widens the handler from
  `NVMLError` to `OSError` — but `AttributeError` is not an `OSError`, so on
  Windows the call still raises through the handler and still aborts every entry
  point. This is the single most likely fix to be dropped by someone reading the
  #164 diff.
- **Drop from the normal route**: both former `attn_implementation=sdpa`
  overrides. The Nano recipe's `cosmos` attention is a PyTorch/cuDNN-backed
  adapter and is the validated native Windows training path. Preserve the
  two-key rule only for controlled SDPA/eager debugging after applying
  `w14-attention-mask-gate.patch`; without that gate the run can exit zero while
  learning only the label prior.

---

## cosmos-rl

`windows-port` is **exactly four commits** on top of the TAO integration branch —
no more, and none of them is a merge:

| Commit | Change | File |
|---|---|---|
| `ed752be` | load NVDEC via `nvcuvid.dll` on win32 (was the hardcoded ELF SONAME `libnvcuvid.so.1`, loaded with `ctypes.CDLL`) | `cosmos_rl/utils/pynv_video_reader.py` |
| `5d191f6` | bump the tao-core submodule to the pwd-guard commit | `tao-core` (submodule pointer) |
| `c00a158` | allow `torchcodec` as an evaluation video decoder | `cosmos_rl/evaluation/base.py` |
| `62da6f8` | pin torchcodec explicitly instead of deferring to the environment | `cosmos_rl/evaluation/base.py` |

Two of the four are the whole of W1's fix: without `c00a158` and `62da6f8` there
is no config-only escape from the NVENC probe, because the stock decoder gate
raises on every value except its default.

`ed752be` is worth keeping in mind during review: its failure mode was three
layers from its cause (see the SONAME note in
[`cosmos3-windows-gaps.md`](cosmos3-windows-gaps.md)).

**Four of the five actions live in this repository**, so a cosmos-framework-only
rebuild does not carry these fixes at all.

---

## tao-core

One commit, one file, one import:

```
nvidia_tao_core/microservices/handlers/execution_handlers/slurm_handler.py:24
    import pwd            # POSIX-only, unconditional, module level
```

Guarded with `try` / `except ImportError`. The module is imported on the TAO
action path, so on Windows the unguarded import aborts `evaluate`, `inference`
and `export` before any work starts — for a SLURM handler that is never used.

**Availability caveat, stated because it affects reproduction:** the referenced
commit is *not reachable* in the public `tao-core` checkout. Anyone rebuilding
from public sources has to re-apply the guard rather than fetch it, and a
submodule init from the public remote will fail on that pointer. Do not describe
this change as "already upstream".

Separately, tao-core's `requirements.txt` hard-requires uWSGI, which cannot build
on Windows at all — that is an install problem, not a patch, and is handled with
`--no-deps` plus a curated list (see
[`cosmos3-windows-gaps.md`](cosmos3-windows-gaps.md)).

---

## What to check after any rebase

1. The affinity `hasattr` guard still exists. Grep for `sched_setaffinity`.
2. `cosmos_framework/utils/win_compat.py` still exists and is still imported.
3. The Nano recipe still uses its native `cosmos` attention default. If an
   SDPA/eager debug override is intentionally enabled, the W14 mask gate is
   present and both policy keys are changed together.
4. All four cosmos-rl commits are still present (`git log --oneline` on the branch
   should show them by name).
5. The tao-core submodule still points at a commit whose `slurm_handler.py`
   guards `import pwd`.
6. A training smoke run, not just evaluate and inference — W9 shows that training
   is the only action some regressions touch.

## Windows ARM64 only: cosmos-rl evaluator import guard

`cosmos_rl/evaluation/metrics/text_metrics.py` imports `evaluate` at module
scope; `evaluate -> datasets -> pyarrow`, and pyarrow has no Windows ARM64
wheel (W16), so the whole evaluator becomes unimportable. Apply
`cosmos-rl-evaluate-import-guard.patch`: the import is guarded, and when it
fails the BLEU/ROUGE/BERTScore metric names are dropped with a RuntimeWarning.
OK/NG accuracy is unaffected — it comes from `task_accuracy.py`, which does not
import `evaluate`, and `analyze_gaps.py` is the AOI scorer anyway. cosmos_rl is
installed **non-editable**, so apply the patch to the site-packages copy (and
to the source checkout, so a reinstall does not lose it).
