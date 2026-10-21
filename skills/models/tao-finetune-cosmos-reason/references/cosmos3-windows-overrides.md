# The eleven training overrides, and the failure each one prevents

Canonical form: `overrides.train` in
[`cosmos3-windows-contract.toml`](cosmos3-windows-contract.toml). Read the array from
there; this file explains it. Reasons are also machine-readable under
`[overrides.why]`, keyed by the left-hand side.

They are appended to the training command **after a bare `--` separator**, with
`launcher = "python"` — not `torchrun`:

```
python -m cosmos_framework.scripts.train --sft-toml=<recipe>.toml -- <11 overrides>
```

Nine of the eleven (`not_foldable_into_recipe_toml = 9`) cannot be moved into the
recipe TOML at all: `SFTExperimentConfig` rejects every `model.config.*` and
`dataloader_*` key with `Extra inputs are not permitted`, and TOML has no null
literal for `prefetch_factor`. They stay on the command line on every single run,
forever. Do not accept a "cleaned-up recipe" that claims to have absorbed them.

## The table

| # | Override | Default | What happens without it |
|---|---|---|---|
| 1 | `optimizer=adamw` | `fusedadamw` | `FusedAdam` imports `transformer_engine`, which has no Windows build. `ModuleNotFoundError` at optimizer construction — after the model is already loaded. |
| 2 | `trainer.distributed_parallelism=ddp` | `fsdp` | FSDP2 has no Windows path. DDP at world size 1 is the only working configuration, and a process group is still mandatory because trainer init runs an unconditional `dist.barrier()`. |
| 3 | `model.config.parallelism.data_parallel_shard_degree=1` | 8 (nano recipe) / -1 | The recipe assumes an 8-GPU allocation. Sharding needs collectives gloo cannot run on CUDA tensors: `0xC0000005`, no traceback. |
| 4 | `model.config.parallelism.data_parallel_replicate_degree=1` | -1 | Same as #3, by the replicate axis. |
| 5 | `model.config.policy.lora_enabled=true` | false | Full-parameter SFT of 8.8 B needs ~105 GB and does not fit 96 GB. Requires upstream PR #164; on `main` the reasoner/VLM path has no LoRA implementation at all. |
| 6 | `dataloader_train.num_workers=0` | 2 | Windows spawns rather than forks worker processes. Forked CUDA video decoding can also return zero frames. |
| 7 | `dataloader_train.persistent_workers=false` | true | Meaningless and rejected at `num_workers=0`. |
| 8 | `dataloader_train.prefetch_factor=null` | 2 | A positive prefetch factor is invalid at `num_workers=0`. CLI-only: TOML cannot express null. |
| 9 | `dataloader_val.num_workers=0` | 2 | As #6, on the validation loader. |
| 10 | `dataloader_val.persistent_workers=false` | true | As #7. |
| 11 | `dataloader_val.prefetch_factor=null` | 2 | As #8. |

## Attention: keep the native default

Do not append an attention override to the normal Windows command. The Nano
recipe's `cosmos` implementation is an `AttentionInterface` adapter over
PyTorch/cuDNN SDPA on this stack; it runs natively on Windows and was validated
to learn. This corrected the older guidance that treated `cosmos` as an
unavailable NATTEN-only path.

The former `sdpa`/`eager` override silently broke learning with Transformers
4.57.6: training removed `attention_mask`, and the HF-native attention path
derived false packed-sequence boundaries from the M-RoPE temporal positions.
The action still exited successfully. The full diagnosis and measured evidence
are retained under W14 in `cosmos3-windows-gaps.md` and
`cosmos3-windows-contract.toml`.

For a controlled SDPA/eager debug run, first apply
`w14-attention-mask-gate.patch`. Then set both
`policy.attn_implementation=sdpa` and
`model.config.policy.attn_implementation=sdpa`. The resolved config has two
policy objects and `VLMModel` reads the nested one, so setting only the first is
still ineffective. This optional pair is machine-readable under
`overrides.optional_attention_debug`; it is deliberately absent from
`overrides.train`.

The same doubling applies to `backbone.safetensors_path` and
`backbone.model_name` if you set them.

PR #164 selects an HF-native fallback automatically for the **Edge** recipe:

```python
_EDGE_ATTN_IMPLEMENTATION = (
    "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa"
)
```

That Edge-specific logic does not justify changing Nano. The **Nano** recipe's
`"cosmos"` default is the validated Windows training route.

## One optional override that is not in the eleven

- `checkpoint.load_path=` — `checkpoint.load_path` is silently ignored whenever
  `latest_checkpoint.txt` exists (contract `[checkpoint]`). Clear it explicitly
  when you want a genuinely fresh run; a Hydra mandatory-value error is the
  failure if the recipe leaves it as `???`.

## Accepted but ignored: `trainer.max_iter` does NOT shorten the run

Contract: `[overrides.inert]`. This is the training-side twin of the `num_frames`
trap (`[evaluate].num_frames_is_inert`): parsed, accepted, **not logged as
overridden, and then ignored**. It does not fail — it lies.

**Measured 2026-08-13**, real native LoRA SFT run on this bank's own runner:

| We passed | The run did |
|---|---|
| `trainer.max_iter=6` | executed all **20** iterations and saved `checkpoints/epoch_2` |

The shipped recipe's own length keys govern. In `wts_vlm_windows.toml`:

```toml
[scheduler]
cycle_lengths = [20]     # line 116 — this is what actually decides
[trainer]
max_iter = 20
num_epochs = 2           # x steps_per_epoch = 10 -> the same 20 optimizer steps
steps_per_epoch = 10
```

Consequences, in order of how much they cost:

1. **A "smoke run" that is not a smoke run.** Anyone budgeting a few minutes by
   passing `trainer.max_iter=6` gets the full 20-step run, the full VRAM peak
   (see below) and the full wall clock.
2. **A false record.** Do not report a run as "capped at N iterations" on the
   strength of the CLI override. Count the iterations the log actually printed.

To genuinely shorten a run, edit the recipe — `scheduler.cycle_lengths` together
with `trainer.num_epochs` / `trainer.steps_per_epoch` — then verify against the
logged iteration count. Any text you find claiming `trainer.max_iter=6` shortens
the run is wrong; this is the correction.

The same run is the source of the shipped-recipe VRAM figure: that recipe sets
`max_sequence_length = 81920`, `max_samples_per_batch = 1`, and LoRA SFT **peaked
at 72706 MiB (~71 GB)** — not the 46.4 GB that the seq-2672 estimate table
suggests. Contract: `[limits.vram_gb.shipped_recipe]`.

## The environment block that goes with them

Contract `[env.train]`, verbatim:

```
MASTER_ADDR=127.0.0.1   MASTER_PORT=29500
RANK=0   WORLD_SIZE=1   LOCAL_RANK=0
USE_LIBUV=0
PYTHONIOENCODING=utf-8
IMAGINAIRE_OUTPUT_ROOT=<output_root>
```

- `MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE` / `LOCAL_RANK`: the gloo
  `env://` bootstrap needs all five even at world size 1.
- `PYTHONIOENCODING=utf-8`: a cp1252 console cannot encode the block characters
  some libraries log, and the resulting `UnicodeEncodeError` kills the run.
- `IMAGINAIRE_OUTPUT_ROOT`: the default `/tmp/imaginaire4-output` is
  **drive-relative** on Windows and becomes `<cwd-drive>:\tmp\imaginaire4-output`.
  It degrades instead of crashing, which is why it goes unnoticed until someone
  looks for the checkpoints.

Plus `TAO_API_JOB_ID` **and** `TAO_API_RESULTS_DIR` together — contract `[env]`.
For training, `TAO_API_JOB_ID` alone is fatal at iteration 0 with
`TAO_API_RESULTS_DIR is required when TAO_API_JOB_ID is set`, raised in
`on_train_start` after the full model load and LoRA injection have already
happened. For the other four actions the requirement is the opposite way round.

## Verifying the overrides took effect

`lora.py` logs at injection time — verbatim from the passing run of 2026-08-13:

```
LoRA injection successful: 144 modules wrapped, 15,335,424 trainable LoRA params,
8,767,123,696 frozen (0.175%)
```

preceded by `inject_lora_pre_fsdp: rank=16 alpha=32.0 dropout=0.0 rslora=False
bias=none targets=['q_proj','k_proj','v_proj','o_proj']`.

Absent that line, the run is doing a full-parameter update and will OOM. The
shipped adapter defaults are in the contract's `[lora]` block (rank 16, alpha 32,
dropout 0.0, targets `q_proj,k_proj,v_proj,o_proj`, no `modules_to_save`) — read
them from there rather than assuming a Linux recipe's LoRA block applies.

The launch mechanics around all of this — how to get the env into a container,
how to stream and keep logs — belong to `skills/platform/tao-run-on-windows-docker`
and `skills/platform/tao-run-on-windows-venv`.
