# Integration with the DeepStream VLM skill

There are two skills covering Cosmos-3 on Windows, in two different repositories,
and they overlap. This file says which owns what, so neither re-implements the
other.

- **This skill** (`tao-skill-bank`, `skills/models/tao-finetune-cosmos-reason`)
  owns the Cosmos-3 **framework contract on Windows**: image, patched sources,
  the five actions, the training overrides, the evaluate/export correctness
  settings, and the platform limits.
- **`deepstream-vlm-eval-and-finetune`** (DeepStream, **MR 324**) owns the
  **deployed-accuracy loop**: measuring what the shipping DeepStream pipeline
  actually answers, before and after a fine-tune, and reporting the delta.

Neither is a superset of the other. The overlap is only the middle stage —
training and export — and that is exactly the stage this bank is upstream for.

## What the DeepStream skill owns, and this bank does not re-implement

None of the following exists in this bank, and none of it should be added here.
When a task needs one of them, route to `deepstream-vlm-eval-and-finetune`.

| Owned there | Why it stays there |
|---|---|
| The **video chunker** and the whole rtvi-vlm DeepStream pipeline | It is the deployed artifact being measured. Reproducing it here would produce a second, differently-behaving pipeline whose numbers mean something else. |
| **NVENC** and the five host driver DLLs bind-mounted read-only at `C:\host-driver` | The chunker genuinely needs the encoder path. This bank's rule is the opposite one (W1: never bake a driver DLL into the image, and select `torchcodec` for decode) and both are correct in their own context. |
| The **`rtvi-vlm` container image** and its `VLLM_MODEL` handling | DeepStream release artifact; the baked `cosmos3-nano-reasoner` name and the need to clear it are properties of that image. |
| `splitmuxsink` **`chunk_duration`** above the longest clip | A GStreamer property. Set too low it emits several fragments per clip and each question gets several answers with no defined way to choose one. |
| The **NAT topology**: vLLM in a sibling container, rtvi containers dialling its Docker NAT IP directly, no host-side TCP relay | Measured on that lane, and the reasoning (a process-isolated container cannot reach a host-served port) belongs with the pipeline. |
| **`eval_set_hash`** — the content hash over `(id, prompt, answer)` that both legs record | The identical-items invariant is the DeepStream skill's core correctness guarantee. |
| **`metrics.json`**, `predictions.jsonl`, and `compare_runs.py` | Its reporting contract, including fixed/regressed counts and unanswerable-by-reason. |
| The before/after web UI | Presentation, entirely theirs. |

## What this bank is upstream for

The DeepStream skill currently carries its own copies of the following. Every one
of them is a Cosmos-3-on-Windows platform fact, not a DeepStream fact, and every
one of them is already in
[`cosmos3-windows-contract.toml`](cosmos3-windows-contract.toml) in
machine-readable form.

| Duplicated there | Where it lives now | Contract key |
|---|---|---|
| GPU device class GUID `class/5B45201D-F2F2-4F3B-85BB-30FF1F953599`, plus `--isolation=process` | `scripts/dockerutil.py` (`GPU_DEVICE`), `references/windows.md` | `runtime.device_class`, `runtime.isolation`, `runtime.required_run_flags` |
| The cosmos-framework image tag | `scripts/dockerutil.py` (`COSMOS_IMAGE`) | `image.uri` (release-managed, stamped from `versions.yaml`) |
| The DCP bind-mount rule: never bind-mount the training output, no `--rm`, `docker cp` out, no `--memory` | `references/windows.md`, `scripts/train_lora.py` docstring | `runtime.output_dir_must_be_container_internal`, `runtime.results_extraction`, `runtime.memory_limit_min_multiple_of_model_size` |
| The training override set | `scripts/render_sft_toml.py` (`cli_overrides()` plus the rendered TOML) | `overrides.train`, `overrides.why`, `overrides.not_foldable_into_recipe_toml` |
| LoRA defaults: rank 16, alpha 32, dropout 0.0, targets `q_proj,k_proj,v_proj,o_proj` | `scripts/render_sft_toml.py` (`DEFAULT_LORA_TARGETS`, argparse defaults) | `[lora]` |
| The eval traps: `answer_type = "freeform"`, `fps` + `total_pixels` never `num_frames`, export with `--base-model-path-or-uri` and zero surviving `lora` keys | `SKILL.md` correctness table, `scripts/export_model.py` | `[evaluate]`, `[export]` |
| `TAO_API_JOB_ID` + `TAO_API_RESULTS_DIR` together | `scripts/train_lora.py` env block, `references/windows.md` | `[env]` |

**The replacement is the contract TOML**, read with `tomllib` from the standard
library — no third-party parser, no import of this bank's code:

```python
import tomllib, pathlib
c = tomllib.loads(pathlib.Path(CONTRACT).read_text(encoding="utf-8"))
GPU_DEVICE   = c["runtime"]["device_class"]
COSMOS_IMAGE = c["image"]["uri"]
OVERRIDES    = c["overrides"]["train"]
```

### One historical difference that now requires an update

The DeepStream copy predates the W14 resolution. Its `cli_overrides()` returned
twelve strings while the rendered recipe carried several related settings,
including the former attention override. That split was equivalent to the old
thirteen-override contract; it is **not** equivalent to the current normal
Windows route.

The current `overrides.train` contains eleven strings and deliberately contains
no attention override. DeepStream adoption must compare the **union** of its
TOML and CLI settings, remove both former SDPA settings from the normal route,
and use `overrides.not_foldable_into_recipe_toml = 9` as the boundary that
decides which side of the split a key may legally sit on. If DeepStream
explicitly chooses SDPA/eager for debugging, it must apply the W14 mask gate and
set both policy keys together; otherwise an exit-zero run may learn only the
label prior.

They also pass `lora_rank`, `lora_alpha` and `lora_dropout` explicitly with values
that match `[lora]`, so those overrides are redundant rather than wrong; they
become a divergence risk only when the shipped defaults change.

## Adoption status

**Adoption is a follow-up MR on the DeepStream side.** Nothing in this PR changes
`deepstream-vlm-eval-and-finetune`, and no file in MR 324 reads this contract
today.

Until that follow-up lands, describe this work as *available for adoption by the
DeepStream skill*, never as *integrated with* it. The distinction is not
cosmetic: an "integrated" claim implies that changing `versions.yaml` here moves
the image the DeepStream lane runs, and it does not — that tag is still a literal
in `dockerutil.py`.

### What the follow-up MR would do

1. Replace `GPU_DEVICE` and `COSMOS_IMAGE` in `dockerutil.py` with reads from the
   contract, with the contract path resolved from an environment variable so the
   DeepStream skill keeps working standalone.
2. Replace `cli_overrides()`'s hardcoded list with `overrides.train`, minus the
   keys its rendered TOML already carries, and assert the union.
3. Read `[lora]`, `[evaluate]` and `[export]` for defaults instead of restating
   them in argparse and prose.
4. Keep every DeepStream-owned item in the first table exactly where it is.
5. Add a check that fails loudly when the contract's `schema_version` is newer
   than the one it was written against, so a bank-side change surfaces as a test
   failure rather than a silently stale constant.

The direction of the dependency matters: the DeepStream skill reads this
contract, this bank never imports DeepStream code. That keeps the deployed-lane
measurement free to change without a bank release, and keeps the platform facts
in one place.
