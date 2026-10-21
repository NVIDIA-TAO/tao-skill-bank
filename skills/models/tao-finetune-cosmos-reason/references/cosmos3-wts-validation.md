# Optional Cosmos3 WTS validation fixture

Use this reference only when a reproducible native Windows-on-ARM64 smoke is
needed. WTS is a validation fixture, not a Cosmos3 dataset contract or a
default for customer training. The parent model skill remains dataset agnostic.

## Fixture layout

The staged archives use this layout:

```text
C:\data\wts\
  train\annotations.json
  train\videos\<clip>\overhead_view\<clip>.mp4
  val\annotations.json
  val\videos\<clip>\overhead_view\<clip>.mp4
```

Set each `WTS_*_MEDIA` variable to the `videos` directory. Annotation media
paths are relative to that directory; using `C:\data\wts\train` as the media
root produces a late `FileNotFoundError` after model loading.

## Native ARM64 commands

Use the public base weights converted to `C:\models\Cosmos3-Nano-VLM`, and the
Windows-safe recipe (`wts_vlm_windows.toml`: the Framework example
`examples/toml/sft_config/wts_vlm.toml` with the overrides from
`cosmos3-windows-overrides.md` applied). Complete the ARM64
environment and decoder setup in `cosmos3-windows-arm64.md` first.

```powershell
$PY = "<arm64-venv>\Scripts\python.exe"
$env:VLM_SAFETENSORS_PATH = "C:\models\Cosmos3-Nano-VLM"
$env:WTS_TRAIN_ANNOTATION = "C:\data\wts\train\annotations.json"
$env:WTS_TRAIN_MEDIA = "C:\data\wts\train\videos"
$env:WTS_VAL_ANNOTATION = "C:\data\wts\val\annotations.json"
$env:WTS_VAL_MEDIA = "C:\data\wts\val\videos"
$env:IMAGINAIRE_OUTPUT_ROOT = "C:\data\wts\runs"
$env:TAO_API_JOB_ID = "<unique-job-id>"
$env:TAO_API_RESULTS_DIR = "C:\data\wts\runs\cosmos3_reasoner\wts_sft\<unique-job-name>"
$env:TAO_DISABLE_CLOUD_STATUS_CALLBACK = "1"
$env:RANK = "0"; $env:LOCAL_RANK = "0"; $env:WORLD_SIZE = "1"
$env:MASTER_ADDR = "127.0.0.1"; $env:MASTER_PORT = "29601"
& $PY -m cosmos_framework.scripts.train `
  --sft-toml=<bank>\skills\models\tao-finetune-cosmos-reason\references\wts_vlm_windows.toml -- `
  optimizer=adamw model.config.policy.lora_enabled=true `
  dataloader_train.num_workers=0 dataloader_train.persistent_workers=false `
  dataloader_train.prefetch_factor=null `
  dataloader_val.num_workers=0 dataloader_val.persistent_workers=false `
  dataloader_val.prefetch_factor=null `
  trainer.max_iter=1 trainer.steps_per_epoch=1 trainer.num_epochs=1 `
  trainer.run_validation=false checkpoint.load_path= job.name=<unique-job-name>
```

That one-step run is only the pre-full-run smoke gate. For the recorded full
validation, use a fresh job name and run directory and replace its final
trainer overrides with the following. `scheduler.cycle_lengths` must remain
greater than or equal to `trainer.max_iter`; otherwise the scheduler, not the
requested trainer length, terminates the run early.

```powershell
& $PY -m cosmos_framework.scripts.train `
  --sft-toml=<bank>\skills\models\tao-finetune-cosmos-reason\references\wts_vlm_windows.toml -- `
  optimizer=adamw model.config.policy.lora_enabled=true `
  dataloader_train.num_workers=0 dataloader_train.persistent_workers=false `
  dataloader_train.prefetch_factor=null `
  dataloader_val.num_workers=0 dataloader_val.persistent_workers=false `
  dataloader_val.prefetch_factor=null `
  trainer.max_iter=500 trainer.steps_per_epoch=250 trainer.num_epochs=2 `
  'scheduler.cycle_lengths=[600]' trainer.run_validation=false `
  checkpoint.save_iter=999999 checkpoint.save_freq_in_epoch=2 `
  checkpoint.load_path= job.name=<unique-full-job-name>
```

Do not use `torchrun`, enable CPU video fallback, or reuse a run directory with
`latest_checkpoint.txt`. The first is not the validated launcher; the second
breaks the GPU decode assertion; the third silently overrides a requested
`checkpoint.load_path`. Do not add the old `attn_implementation=sdpa`
override: native `cosmos` attention is the current validated route. SDPA/eager
is debug-only and requires the W14 mask-gate patch described in the Windows
contract.

## What this fixture validated

On 2026-08-16, the fresh `wts_native_arm64_skill_r4` smoke completed one native
optimizer step using CUDA TorchCodec video decode: 144 wrapped LoRA modules,
15,335,424 trainable parameters, loss `0.744311`, and a 32.84 GiB DCP
checkpoint. It produced a 16.34 GiB five-shard merged HF export with 750 weight
tensors and zero residual LoRA keys. Direct Cosmos-RL inference and one-item
Framework evaluation both loaded that smoke export on CUDA and selected
TorchCodec. Inference generated six tokens in 8.5 seconds. Evaluation saved
metrics and a headless plot and reported NLL `0.595265`; its intentionally
synthetic label yielded zero accuracy, which is not a quality measurement.

On 2026-08-24, `wts_demo_lora_v1` completed the then-current command: 500
optimizer steps, 250 steps per epoch, two epochs, terminal `SUCCESS`, and a
token-weighted training loss of `0.173310205149136` over 1,000 valid labels.
The epoch-2 DCP checkpoint completed in 119.54 seconds. The exact-key exporter
merged all 144 adapter modules into a five-shard HF directory with 1,038 weight
tensors and zero residual LoRA keys. That merged export served through native
ARM64 vLLM in eager and compiled modes and passed text, image, video, Chat
Completions, and non-Harmony Responses requests.

That historical WTS run used the former SDPA override and therefore establishes
lifecycle execution, not learning quality. Later W14 qualification found that
this override could exit successfully while learning the label prior; the
commands above now retain the Nano recipe's native `cosmos` attention default.

The same checkpoint was also packaged as a standard rank-16 PEFT adapter with
288 A/B tensors across 144 modules and loaded dynamically by vLLM. The
adapter-only package is validation of dynamic serving, not the normal Framework
export path; the normal export remains the standalone merged HF model.

The staged 2,417-frame 1080p fixture clip decoded the same eight frame indices
in 0.879 s with the CUDA TorchCodec reader versus 11.187 s through PyAV CPU
fallback (about 12.7x), including the required CPU Pillow handoff.

These outcomes establish native lifecycle execution, a complete finite LoRA
schedule, decoder selection, export portability, and the serving paths listed
above. They do not establish convergence, accuracy, Linux performance parity,
or multi-request throughput on WTS or any other dataset.
