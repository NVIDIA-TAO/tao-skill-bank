# DEFT CR ITS GenAI Branch

Read this reference only when `data_generation.mode` is `genai` or `both`. Resolve `DEFT_SKILL_ROOT` and the installed `paidf-cosmos-predict` directory as `PAIDF_SKILL_ROOT`; neither is a user input.

## Inputs

- `$ITER_DIR/gaps/kpi_gaps.jsonl`
- `genai.vlm_captioning_endpoint`
- `genai.paidf_num_gpus`
- Optional `genai.generation_settings`, otherwise the PAIDF skill default
- Optional `genai.caption_prompt_file`, otherwise `$DEFT_SKILL_ROOT/assets/qwen_its_caption_prompt.txt`
- `kpi_dataset.media_dir`, mounted 1:1 by PAIDF

The workflow requires an externally managed captioning endpoint and never starts vLLM.

## Verify Endpoint

Verification is an initialization preflight, before baseline evaluation:

```bash
python3 "$PAIDF_SKILL_ROOT/scripts/verify_vlm_captioning_base_url.py" \
  --vlm-captioning-endpoint "$VLM_CAPTIONING_ENDPOINT"
```

The configured value is the OpenAI-compatible base URL passed unchanged to PAIDF. The verifier derives and checks its `/models` endpoint. Stop if verification fails.

## Prepare PAIDF Input

```bash
python3 "$DEFT_SKILL_ROOT/scripts/prepare_paidf_input.py" \
  --gaps-jsonl "$ITER_DIR/gaps/kpi_gaps.jsonl" \
  --output-jsonl "$ITER_DIR/paidf/media.jsonl"
```

The helper writes one `id`/`media_path` row per unique weak media/question pair. Multiple failed questions for one video keep separate ids. The generic PAIDF skill deduplicates generation by media path and expands its handoff back to every input id.

## Run PAIDF

Invoke the registered `paidf-cosmos-predict` skill with:

- Input JSONL: `$ITER_DIR/paidf/media.jsonl`
- Output directory: `$ITER_DIR/paidf`
- Media directory: `kpi_dataset.media_dir`
- Captioning base URL: `genai.vlm_captioning_endpoint`
- PAIDF GPU count: `genai.paidf_num_gpus`
- Generation settings: user path or the PAIDF skill default
- Caption prompt: user path or `$DEFT_SKILL_ROOT/assets/qwen_its_caption_prompt.txt`

The PAIDF skill owns image resolution, Docker invocation, credentials, mounts, log streaming, config generation, and handoff generation. Monitor the container continuously. The stage is not complete merely because generated files appear; require Docker to exit successfully and retain `paidf_docker.log`.

Expected handoffs:

```text
$ITER_DIR/paidf/generated_videos.jsonl
$ITER_DIR/paidf/failed_videos.jsonl
```

Generated plus failed handoff row counts must equal `media.jsonl` row count, and every successful `generated_video_path` must exist.

## Build Converter Input

```bash
python3 "$DEFT_SKILL_ROOT/scripts/build_llava_input.py" \
  --gaps-jsonl "$ITER_DIR/gaps/kpi_gaps.jsonl" \
  --generated-videos-jsonl "$ITER_DIR/paidf/generated_videos.jsonl" \
  --output-jsonl "$ITER_DIR/genai/generated_samples.jsonl"
```

Each successful row contains exactly `id`, `video_path`, `question`, and `answer`. Failed PAIDF rows are skipped. If no successful rows remain, log the remaining GenAI stages as skipped. In GenAI-only mode, stop without training. In `both` mode, continue only when mining produced current annotations.

## Convert To LLaVA

Use the TAO Data Services image and keep the directory named `genai` even though the upstream command contains `qa`:

```bash
DS_IMAGE="nvcr.io/nvidia/tao/tao-toolkit:7.1.0-data-services" # versions-key: images.tao_toolkit.data_services

docker run --rm --ipc=host --network host \
  -v "$WORKSPACE:$WORKSPACE" \
  "$DS_IMAGE" \
  annotations qa_to_llava_annotation \
  input_jsonl="$ITER_DIR/genai/generated_samples.jsonl" \
  results_dir="$ITER_DIR/genai" \
  output_file="generated_llava_annotations.json"
```

Use the selected platform skill for non-local execution with the same image, command, 1:1 workspace visibility, and output contract.

The output is:

```text
$ITER_DIR/genai/generated_llava_annotations.json
```

Require valid LLaVA JSON and one row per successful PAIDF handoff. `prepare_cosmos_reason_train.py` reads this file as the GenAI branch's current annotation source and performs the mode-aware merge.

## Artifacts

```text
iter_<N>/paidf/media.jsonl
iter_<N>/paidf/config.yaml
iter_<N>/paidf/path_map.jsonl
iter_<N>/paidf/paidf_docker.log
iter_<N>/paidf/generated/videos/*.mp4
iter_<N>/paidf/generated_videos.jsonl
iter_<N>/paidf/failed_videos.jsonl
iter_<N>/genai/generated_samples.jsonl
iter_<N>/genai/generated_llava_annotations.json
```
