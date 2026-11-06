# AnomalyGen 1.2 — Cosmos3 paired AOI

Read `skills/applications/tao-run-deft-aoi/references/tao-generate-anomalies.md` for the
shared native runtime, public PCB-Nano checkpoint/recipe, bootstrap, BYO,
exclusion and quality-policy contract. This application also resolves
`images.metropolis_sdg.anomalygen` and calls the single shared skill in
`inference_only` mode. It retains its existing Proxy routing and KPI semantics.

Initialize `--anomalygen-checkpoint` with a native run directory or explicit
`iter_N.pt` and `--anomalygen-recipe` with its exact training recipe when using
BYO. Defaults bind PCB-Nano's `model/iter_000015000.pt` and
`model/exp_texture_ft_pcb.yaml`. `--anomalygen-dataset-dir` and
`--cosmos-models-dir` override dataset/cache paths. Mount resolved paths outside
the workspace explicitly. Pass the same `--exclude-stems` to all consumers.

```bash
"$PYTHON" "$SKILL_ROOT/scripts/emit_sdg_sharegpt.py" \
  --manifest "$RUN_DIR/manifest.json" --media-root "$MEDIA_ROOT" \
  --prompt-from "$MINING_ANNOTATIONS" --exclude-stems "$EXCLUDE_STEMS" \
  --emit-relative --output "$RUN_DIR/sdg_sharegpt.json"
"$PYTHON" "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label "iter${N}" --stage anomalygen \
  --anomalygen-manifest "$RUN_DIR/manifest.json" \
  --anomalygen-sharegpt "$RUN_DIR/sdg_sharegpt.json" \
  --duration-sec "$STAGE_DURATION_SEC" --summary "native generation accounted"
```

The emitter reads native CSV basenames in the selected output bucket. Each pair
is `[reconstructed_image, original_image]`, with the exact inherited prompt and
bare `NG` label. It retains testcase ID, original source, original mask donor and
manifest identity in `anomalygen_lineage`. No defect-name label conversion is
permitted. Source files and paired artifacts must be accessible in the same
compute namespace as generation; do not rewrite source CSVs to repair mounts.

Generated plus blocked identities must cover every testcase, including sparse
IDs. Mixed batches emit generated pairs only. All-blocked batches emit `[]` and
a bound `.anomalygen.json` receipt; assembly accepts that verified zero contribution.
The stage records COMPLETE generation and `training_ready=false`, never a skip
or fabricated replacement. Training still requires nonempty eligible real data.
`assemble_training_json.py` and `validate_split_contract.py` inspect source and
mask lineage against held-out targets and caller exclusions, in addition to
existing target, label, monotonic-history and Benchmark-hash checks.

A deliberate `--skip` remains permitted only with zero false accepts in the
driving Proxy RCCA. The native all-blocked outcome is a completed run and does
not use that skip mechanism. Missing/corrupt outputs, disabled guardrails,
stamp/recipe mismatch and forbidden lineage remain hard stops. Preserve native
guardrail capability fields; image enforcement false is not an all-guardrails-off
claim. The default quality policy runs three native search rounds with no automatic
discard or top-up; drop-ratio filtering requires an explicit request. Smoke
uses `--skip-quality` and does not establish production image quality.
Assembled Train records selected from known synthetic targets must retain the
exact verified pair, prompt, bare NG label and native lineage.

Shared Docker identity/mount commands belong in that reference. Do not duplicate them here.
