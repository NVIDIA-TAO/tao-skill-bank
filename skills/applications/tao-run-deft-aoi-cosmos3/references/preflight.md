# Preflight

Preflight is read-only and precedes the launch review.

1. Resolve `nvidia/Cosmos3-Nano` with the requested action and
   `workload=deft-aoi`; verify backend `cosmos-framework` and show rationale.
2. Ask the user to select among supported installed platforms, read that
   platform skill, and run its native-CLI/resource preflight.
3. Check `mining.jsonl`, `benchmark.jsonl`, and `proxy_kpi.jsonl` with
   `check_annotations.py --require-files`. Record each path, row count,
   image-item count, and SHA-256. The workspace root is the media root. The
   Mining report records both raw and eligible counts plus ignored
   count/segmentation task families; Proxy and Benchmark accept only the six
   supported classification/detection families.
4. Verify `models/Cosmos3-Nano-VLM` is a complete Qwen3-VL safetensors
   snapshot. Verify the exact evaluator is the absolute workspace
   `eval/calculate_f1_metrics.py`, record its SHA-256, and prove every frozen
   Benchmark row and all five components are evaluator-compatible:

   ```bash
   "$PYTHON" "$SKILL_ROOT/scripts/exact_f1_adapter.py" \
     --evaluator "$WORKSPACE/eval/calculate_f1_metrics.py" \
     --source "$WORKSPACE/annotations/benchmark.jsonl" \
     --preflight-source-contract
   ```

   This invokes the exact evaluator with perfect embedded predictions; it does
   not calculate F1 locally. Any rejected ground truth or missing component is
   a hard stop before state initialization.
5. Validate the full or explicitly named smoke TOML, writable results path,
   Python dependencies, and GPU topology. Full requires one 8-GPU node.
6. Resolve `images.tao_toolkit.cosmos_framework` and
   `images.tao_toolkit.data_services` from `versions.yaml`. Launch descriptors
   require immutable digests; resolving a local digest or pulling an image is
   post-approval if it changes state.

   ```bash
   $PYTHON "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" images.tao_toolkit.cosmos_framework
   $PYTHON "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" images.tao_toolkit.data_services
   ```
   After the approved image-resolution step, pass each immutable
   `tag@sha256:<digest>` value to `init_deft_state.py --framework-container`
   and `--mining-container`; state initialization rejects mutable tags.
7. Show nested configs, commands, mounts, resources, outputs, DCP format,
   exact metric paths, network policy, and credential variable names in one
   launch review. Wait for explicit approval before any pull, download, login,
   submit, train, evaluate, or inference.

Never ask for credential values and never place them on argv, in specs, logs,
records, or commits.
