# Pi adapter (guard + recorder + NVIDIA provider)

TypeScript extensions for the [Pi coding agent](https://pi.dev). Pack drivers
load them with `-e`; nothing to install beyond `pi` itself.

| File | Role |
|---|---|
| `guard.ts` | Blocks known dead ends BEFORE tokens burn: per-session tool-call budget (`PI_KIT_TURN_BUDGET`), near-duplicate-command loop breaker, no-fabricated-results check (a train "ok" commit requires `Execution status: PASS` in the real log), destructive-command safety net, and a secrets guard (blocks env dumps and any command naming a known credential var — the shipped packs never need credentials; a future pack that must pass one by reference, e.g. `-e HF_TOKEN`, should adjust that guard). Pi has no permission prompts, so this is also the only permission layer. |
| `recorder.ts` | Appends every substantive executed command to `$RD/commands.log` (guard-blocked calls excluded). Run dir comes from `PI_KIT_RD`, falling back to the newest `PI_KIT_RUN_PREFIX`* dir under `$PI_KIT_WS/results`. Extend the match list via `PI_KIT_RECORD_PATTERNS` — comma-separated regexes, so a pattern cannot itself contain a comma (e.g. `{1,3}` quantifiers are unrepresentable; use an unbounded form instead). Invalid fragments are logged and skipped, never fatal. |
| `nvidia-provider.ts` | Optional `nim/` provider for the NVIDIA Inference API: thinking disabled via chat-template kwargs with the `:off` model suffix, temperature pinned to 0 for tool-calling reliability. Export `NVIDIA_INFERENCE_API_KEY`. Not needed for `anthropic/` models. |

## Workflow-specific guards

`guard.ts` also carries three guards discovered on the DEFT AOI reference
hardware (a cuDNN/sm_75 embedding crash, a disk-headroom check before
training, a container CLI quirk). They are harmless elsewhere — each matches a
specific command signature — and they double as the pattern to copy: **one
guard per quirk your first run discovers**, with a reason that says why it is
blocked and what to do instead.

## Env contract

Set by pack drivers; set them yourself only when running sessions by hand:
`PI_KIT_TURN_BUDGET`, `PI_KIT_RD`, `PI_KIT_WS`, `PI_KIT_RUN_PREFIX`,
`PI_KIT_RECORD_PATTERNS`. For hand-run sessions the guard also honors bare
`RD`/`ITER` and the recorder honors bare `WS`; the recorder deliberately
trusts only `PI_KIT_RD` for the run dir (a bare-RD fallback could misfile
records into a previous run's commands.log).
