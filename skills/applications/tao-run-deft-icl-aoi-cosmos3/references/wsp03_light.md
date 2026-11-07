# Native Codex weight editing: light execution

The native editor has ordinary built-in shell, Python and file tools. It can
write its own scientific scripts in `/work`. Do **not** instantiate the old
`Service`, `Ledger`, `Supervisor`, `RoundBook`, dedicated-UID wrappers or relay.
The experiment owner supplies preregistration and the science plan in `/context`.

## Visible inputs

- `/development`: development images and development labels only. Its
  `gold.jsonl` is the allowed development label file, not withheld gold.
- `/development-embeddings.jsonl`: optional existing development SigLIP vectors.
- `/model`: read-only original base model.
- `/context`: fixed target/ICL prompts, class cues, explore IDs, twelve reference
  strategies, selection rules, strengths and the current task.
- `/bank`: public tool source; `/work`: this track's writable experiment state.
- `/frozen-r3`: optional read-only R3 export/patch and selected development
  artifacts for R4. Never mount an earlier editor's CODEX_HOME.
- `/codex`: this editor's fresh native home/auth/session logs. Do not print auth.
- `/results`: this native job's canary, stdout/stderr and exit marker.

No host workspace, source data root, holdout, Benchmark, historical predictions,
private gold, Docker socket or other runs are mounted. The single startup
`ls` canary checks the visible mounts and absence of forbidden paths.

## Directly reusable primitives

`PYTHONPATH` points at the application `scripts` directory; `python` is the
container's existing TAO/PyTorch environment. Read these modules directly and
use their functions or write a small script of your own:

| Module | Useful direct entrypoints |
| --- | --- |
| `wsp03_worker.py` | `run`, `apply_candidate`, `target_logprobs`; native generation and teacher-forced correct-class read/capture |
| `wsp03_capture.py` | `ActivationCapture`; hidden/MLP captures, diagnostic replacement and same-pass attention reconstruction |
| `wsp03_kernel.py` | `TensorStore.load_tensor`, `save_edit`, `commit_candidate`, `verify`, `reset`; CPU BF16 edits and replayable replacement patches |
| `wsp03_ops.py` | `Operations.export`; export an existing verified candidate as HF checkpoint plus replay patch |
| `wsp03_bindings.py` | `ReferenceSelector` and `PROFILES`; cached dev-only similarity, helpfulness and the twelve reference strategies |
| `tht_contract.py` | `external_evaluator` and `grade`; use only the sanitized frozen evaluator code and development labels supplied in context |

NumPy, PyTorch, safetensors and PIL are available in the image. Use
`model.eval()` and `torch.inference_mode()`. No gradients, optimizer or training
loops. Closed-form MEMIT-style edits are allowed and must be labeled separately
from strict weight-space edits.

A direct CPU patch example (replace the tensor/name and delta with the
preregistered development-derived edit):

```python
from wsp03_kernel import TensorStore
store = TensorStore('/model', '/work/patches')
weight = store.load_tensor('<existing tensor name>')
# edited = ... fixed preregistered strength and development evidence ...
store.save_edit('<existing tensor name>', edited)
manifest = store.commit_candidate('candidate01', 'record the mechanism',
                                  method='strict_wsp')
assert store.verify('candidate01')['passed']
```

Use `method='gradient_free_algebraic'` for closed-form fitting or `mixed` when
combining methods. Keep scripts and replay patch files; do not merely save a
metric. `apply_candidate(model, patch_directory)` replays absolute replacements
onto the original base and checks the expected BF16 before/after identities.
Apply once; never apply that base-relative patch again to an already edited HF
export.

For a direct worker invocation, write request JSON with `id`, `messages`, and
`references`. Images may use their actual `/development/...` paths. Read requests
also have the canonical `forced_text` (for example `{"labels":["Good"]}`) and
optional `capture` points/patches. The source provides the exact schemas.

```sh
python /bank/skills/applications/tao-run-deft-icl-aoi-cosmos3/scripts/wsp03_worker.py \
  --model /model --requests /work/read-requests.json --output /work/read01 \
  --mode read --split development
```

The command is a convenience, not a required service protocol. A persistent
Python process can reuse a loaded model to avoid repeated setup. Use only
preregistered strengths (at most five per edit type) and the fixed twelve R4
strategies; maintain decisions and counts in `/work/decisions.tsv`.

R4 keeps the exact frozen R3 weights; select references using development only.
Freeze R3, R4 and the fixed R4-C ablation before any withheld evaluation. Write
`/work/FREEZE.json` with the selected checkpoint/patch and strategy, input hashes,
selection metrics and method label. Then stop. A separate trusted script outside
the editor evaluates each frozen arm on holdout and Benchmark once; no feedback
returns to the editor.

## Owner launch commands

Create a small JSON configuration with paths `development`, `model`, `context`,
`work`, `codex_home`, `binary`, `bank`, `output`; immutable container `image`; and
`task` relative to the context directory. Optional `development_embeddings` and
`frozen_r3` become the mounts listed above. `forbidden_paths` can list the actual
host-only source/history paths for the one canary. Default model/effort is
`gpt-6-astra`/`xhigh`; native tools stay enabled. No heavy supervisor config or
custom provider/relay is used.

```sh
python scripts/wsp03_light.py prepare light-run.json
# Record the canary, mount list and preregistration hash in LIGHT_START.
python scripts/wsp03_light.py start <output>/run.json
python scripts/wsp03_light.py monitor <output>/run.json
```

`prepare` starts one container and waits at a start marker after the canary.
`start` releases that same container immediately; it is not another approval
or reviewer gate. `monitor` samples `nvidia-smi` every ten seconds and sums native
session `token_count` self-reports. Around five million tokens, or the configured
best-effort wall limit, it stops that owned container. Cached tokens are already
inside input tokens; in-flight or missing usage stays unknown. The native usage
logs are mutable/self-reported, and these observations are not strict quotas.

Native Codex deployments must include `codex-code-mode-host` and `codex-exec-server` beside the configured `binary`; the launcher mounts all three executables read-only. To recover an interrupted native session, keep its existing work and CODEX_HOME, set `resume_session` to its recorded UUID, and provide a continuation task file. Preserve and remove the old work/LIGHT_START marker before preparing the replacement container so it remains gated until `start`; do not reset session token logs or experiment elapsed time.
