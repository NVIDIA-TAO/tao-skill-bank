# TAO 7.3 CLIP LoRA migration

The 7.3 CLIP schema replaces `peft.vision.enabled` and `peft.text.enabled` with `peft.vision.mode` and `peft.text.mode`. Migrate each old `enabled: true` to `mode: lora` and each `enabled: false` to `mode: frozen`; `mode: full` trains a whole tower. Keep the top-level `peft.enabled: true`. The new `peft.train_logit_calibration` flag defaults to `true` and independently controls training of logit scale and optional bias. A 7.2 tower `enabled` key fails 7.3 structured config merging with `ConfigKeyError` before training begins.

For example, replace the 7.2 tower keys:

```yaml
peft:
  enabled: true
  vision:
    enabled: true
  text:
    enabled: false
```

with this 7.3 configuration:

```yaml
peft:
  enabled: true
  train_logit_calibration: true
  vision:
    mode: lora
  text:
    mode: frozen
```

The packaged train templates and schema remain 7.2-shaped, and this skill pins a 7.2 PyTorch image. Do not pass their `peft.<tower>.enabled` fields to a 7.3 image. Use a 7.3-shaped spec and a confirmed 7.3 runtime for a 7.3 LoRA run.
