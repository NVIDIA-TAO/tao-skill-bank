---
name: tao-generate-video-reasoning-annotations
description: >-
  Multi-step video annotation pipeline that turns raw videos into
  Chain-of-Thought training data — curated and routed clips, multi-level
  captions, structured descriptions, QA pairs (MCQ, binary, open-ended,
  event verification, causal, temporal) with reasoning traces, and
  metropolis-v3.0 contextual annotations, via VLM/LLM distillation. Use when
  the user wants to "create video training data", "generate video QA
  datasets", "build CoT reasoning traces from videos", "auto-label videos",
  or run the video_reasoning_annotation (VRA) pipeline. Triggers include
  "video annotation", "video CoT", "video QA", "chain-of-thought", "video
  captioning pipeline", "video distillation", "VRA".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit + at least one VLM endpoint (Gemini API key or OpenAI-compatible).
metadata:
  author: NVIDIA Corporation
  version: "0.2.0"
allowed-tools: Read Bash Write
tags:
  - video
  - annotation
  - chain-of-thought
  - captioning
  - qa-generation
  - vlm
  - llm
  - auto-label
---

# Video Reasoning Annotation Pipeline

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Generate Chain-of-Thought training datasets from videos: curate and route clips, caption them, synthesize a description, generate QA with reasoning traces, and export `tao-vl-reason-v1.0` task files plus `metropolis-v3.0` contextual annotations.

The pipeline, its prompts, and its sample specs all ship in the data-services container (`nvidia_tao_ds.auto_label.video_reasoning_annotation`). This skill does not carry its own prompt copies — pick a bundled prompt module by name, or build a new one from the bundled template.

## Pipeline architecture

Three packages, in execution order:

```
CURATION (0a-0f) — picks the video bytes and the lane
  0a filter     VLM: is this footage the prompts expect?
  0b classify   VLM: anomaly/normal by content + ffprobe resolution
  0c routing    lanes: anomaly hi-q / anomaly lo-q / normal
  0d dedup      [opt] cut repeated/spliced segments (non-destructive copy)
  0e enhance    [opt] denoise -> CLAHE -> Lanczos upscale -> unsharp, lo-q lane
  0f snapshot   [opt] archive exact captioned bytes into final_videos/

LABELING (1a-4b) — turns the videos into text
  1a caption    global + dense timestamped captions (xN passes on hi-q lane, merged)
  1b chunks     fixed-duration segment captions
  1c highlight  anomaly lanes: clip around the incident timestamp, caption it
  2a description  synthesize captions -> narrative + final_verdict + incident_count
                (+ optional grounding-label injection)
  2b reroute    [opt] split 2a output by final_verdict (audit only; not consumed yet)
  3  qa         QA per qa_type, each with a reasoning trace
  4a parse      QA -> tao-vl-reason-v1.0 task JSONs
  4b contextual captions/description -> metropolis-v3.0 contextual JSONs

REVIEW (5)
  5  report     RESULTS_SUMMARY.md + annotation_report.json (+ optional tao-daft validation)
```

Steps 1a-1c look at pixels; from 2a on everything reasons over caption text, so **caption quality is the ceiling on everything downstream**.

This skill's default `steps` (in `skill_info.yaml`) is the full 14-step list the bundled specs use: `["0a","0b","0c","0d","0e","0f","1a","1b","1c","2a","3","4a","4b","5"]`. The pipeline's own dataclass default is shorter (`["0a","0b","1a","1b","1c","2a","3","4a","4b","5"]`, with 0c-0f opt-in). 2b is always opt-in. In `mode: auto`, 0a and 0b are added if absent.

**The `steps` argument overrides the spec's own list.** When you launch a spec whose steps differ from the default, pass that spec's list explicitly. The only bundled one is `smart_space_openai_grounding`: `["2a","2b","3","4a"]`.

Drop `0d` when the source has no repeated footage, and drop `0e` when there are no low-resolution clips. Keep `0f` whenever `0d` or `0e` runs.

## Initial consultation

Walk through these in order before any run.

### 1. Videos

- Directory (`data.video_root`, walked recursively) and/or JSONL (`data.input_jsonl_files`, `{"video_path": "..."}` per line; `video` key also accepted).
- Folder names starting with `anomal` / `normal` act as a routing floor (see `workflow.use_folder_floor`). For unlabelled pools, set `use_folder_floor: false`.
- If the user already has trusted per-video labels, set `data.routing_manifest` and skip the classifier.

### 2. Domain — drives `prompts_module`

`prompts_module` takes a short name resolved inside `nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts`, or a full dotted path for a user module.

| Domain | `prompts_module` | Bundled spec to start from |
|---|---|---|
| General / unknown | `""` (→ `generic`) | any spec, then set `prompts_module: ""` |
| Multi-category commercial CCTV | `smart_space` | `video_reasoning_annotation_smart_space_{openai,gemini,openai_gpt5}.yaml` |
| Warehouse / industrial CCTV | `warehouse` | `video_reasoning_annotation_warehouse_{openai,gemini,openai_gpt5}.yaml` |
| Public safety | `public_safety` | `video_reasoning_annotation_public_safety_openai.yaml` |
| Retail | `retail` | start from a smart_space spec, set `prompts_module: retail` |
| Traffic CCTV | `traffic` | start from a smart_space spec, set `prompts_module: traffic` |
| **Custom** | user module path | **Run the workshop in [references/domain_adaptation.md](references/domain_adaptation.md)** — copy `prompts/template.py`, fill its `[PLACEHOLDER]` markers, point `prompts_module` at the copy. Do this before any run. |

`public_safety` adds domain QA families (`event_type_mcq`, `weapon_detection`, `person_object`) usable in `workflow.qa_types`.

### 3. Anomaly / normal / mixed

- Mixed → `workflow.mode: "auto"` (0b classifies each video).
- Pre-split anomaly only → `mode: "anomaly"`. Pre-split normal only → `mode: "normal"` (1c is skipped).

### 4. VLM / LLM endpoints — confirm access **before** running

- **openai backend** is a protocol, not a vendor: self-hosted vLLM/NIM, any OpenAI-compatible gateway, or GPT. Needs `base_url`, `model_name`, and `OPENAI_API_KEY` (bare key, no `Bearer `). Check `video_content_type` (`video_url` for vLLM/NIM, `image_url` for some gateways) — a wrong value can return fluent "please provide the video" captions with HTTP 200.
- **gemini backend** uses the google-genai SDK: `GOOGLE_API_KEY`, `model` (not `model_name`), `max_output_tokens` (not `max_tokens`), no `base_url`.
- Image-only reasoning models (e.g. GPT-5.x): `frames_only: true` + `reasoning_model: true` (see the `_openai_gpt5.yaml` specs).
- Split by step with `models:` + `pipeline:` (captioner / describer / reasoner). Text steps can use a cheaper model.

No endpoint and want to self-host? Point the user at `skills/applications/tao-run-inference-service` (check its `references/service.yaml` `valid_network_arch_config_basenames` first). If no endpoint is ready, stop and help them get one first.

### 5. Optional inputs

- `data.gt_type_file` — `{stem: incident_label}` to steer caption/description prompts per category.
- `data.df_annotations_dir` — human contextual annotations, injected as authoritative context.
- **Grounding injection** — `workflow.sta_grounding_labels_dir` points at per-clip grounding labels (`<dir>/<clip_uuid>/**/events_reconciled_natural.json`, `clip_uuid` = video stem). Step 2a injects the frame-accurate, actor-attributed event timeline into the description prompt as the primary source for what happened. Start from `video_reasoning_annotation_smart_space_openai_grounding.yaml` (steps `["2a","2b","3","4a"]` over existing step 1 captions). See [references/configuration.md](references/configuration.md#grounding-injection).

### 6. Pilot vs full run

- **Recommend a 5–10 video pilot** for custom domains, edited prompts, a new endpoint/model, or a first run.
- Resume makes a pilot cheap: curation and 1a-2a skip already-processed `(video, prompt_key)` pairs.

## Quick start

Runs in the data-services container via the `auto_label` CLI. The bundled specs live *inside the image*, not in this repo — always edit a fresh copy pulled from the exact image you're about to run, never a cached copy, since the bundled spec's fields (workflow defaults, worker caps, even which steps run by default) move between image builds.

**Preferred: `scripts/prepare_vra_spec.py`.** It pulls the named spec out of the image with `docker create`/`docker cp` (no entrypoint execution, so no startup-banner noise), applies only the fields you name via `--set DOTTED.KEY=JSON_VALUE`, and prints a diff against the untouched original so every deviation from the bundled template is visible before launch. It edits lines in place rather than re-dumping the YAML, so every comment in the bundled spec survives in fields you didn't override — those comments carry the domain reasoning behind non-default values. Multi-line values (e.g. `qa_types`) are replaced whole; whole-section overrides and typos are refused; and the result is parsed and checked against the requested values before anything is written. It refuses to run against an image that isn't already pulled locally.

```bash
python3 skills/data/tao-generate-video-reasoning-annotations/scripts/prepare_vra_spec.py \
  --image <local-image-tag> \
  --spec-name video_reasoning_annotation_smart_space_openai.yaml \
  --output /workspace/vra_spec.yaml \
  --set 'results_dir="/results/vra_run1"' \
  --set 'video_reasoning_annotation.data.video_root="/path/to/clips"' \
  --set 'video_reasoning_annotation.vlm.openai.base_url="https://your-endpoint/v1"'

auto_label generate -e /workspace/vra_spec.yaml
```

A `--set` key must be the *dotted path down to the exact line* in the bundled file (e.g. `video_reasoning_annotation.vlm.openai.model_name`, not `vlm.model_name`) — the script fails loudly on any key that doesn't match a line, rather than silently no-op'ing a typo. This is also why endpoints must be set this way and not via a `video_reasoning_annotation.vlm.openai.*` CLI override at launch time: the bundled specs share `vlm`/`llm` endpoint blocks with `models:` via YAML anchors, so a launch-time override only changes `vlm`/`llm`, never the anchored copies every step actually reads through `pipeline:`. `--set` edits the one line in the file itself, so it reaches every anchored copy.

Manual fallback (equivalent, no diff safety net):

```bash
SPECS=$(python -c "import nvidia_tao_ds,os;print(os.path.join(os.path.dirname(nvidia_tao_ds.__file__),'auto_label/experiment_specs'))")
cp $SPECS/video_reasoning_annotation_smart_space_openai.yaml /workspace/vra_spec.yaml
# edit: results_dir, data.video_root, vlm/llm base_url + model_name, prompts_module
auto_label generate -e /workspace/vra_spec.yaml results_dir=/results/vra_run1
```

Keys go in the environment (`OPENAI_API_KEY` / `GOOGLE_API_KEY`), never in the spec or on argv. Hydra dot-overrides at launch time work for scalar fields not shared via an anchor (`video_reasoning_annotation.workflow.qa_resume=true`).

Full field reference, spec guide, and error patterns: [references/configuration.md](references/configuration.md).

## Re-running / resuming

- Point `results_dir` at a previous run: curation and 1a-2a resume, 0d never re-cuts, 0e never re-encodes.
- **Step 3 is wiped and regenerated by default.** Set `workflow.qa_resume=true` only when the staged video paths are identical to the previous run — not after changing the video source or toggling 0d/0e.
- `step_4_output/` and `step_4b_contextual/` are always rebuilt (local parsing, no API calls).

## Pilot workflow

1. Run the pilot subset with the chosen spec.
2. Check `step_0a_filter/filter_results.jsonl` row count against the "Filtering N videos" log line — filter errors silently drop videos.
3. Read a few `step_1a_caption/captions.jsonl` entries by eye — accurate, right detail, not refusals?
4. Read `step_2_description/descriptions.jsonl` (`final_verdict`, `incident_count`) and `step_3_qa/qa_output.jsonl`.
5. Read `step_5_report/RESULTS_SUMMARY.md` for stage-by-stage attrition.
6. Iterate on prompts (captions first), re-run, then scale to the full set.

Visual review (optional, from the pipeline repo root): `streamlit run nvidia_tao_ds/auto_label/video_reasoning_annotation/app_stage_review.py` (per-stage) and `app_qa_review.py` (step 4 tasks).

## Configuration summary

Key fields under `video_reasoning_annotation:` (full reference in [references/configuration.md](references/configuration.md)):

| Field | Default | Description |
|---|---|---|
| `workflow.steps` | `["0a","0b","0c","0d","0e","0f","1a","1b","1c","2a","3","4a","4b","5"]` | Steps to run (the skill default; 2b is opt-in) |
| `workflow.mode` | `"auto"` | `auto` / `anomaly` / `normal` |
| `workflow.qa_types` | 9 types incl. `event_verification` | QA types for step 3 |
| `workflow.caption_passes` | `1` | >1 = multi-pass 1a on the hi-q anomaly lane, merged |
| `workflow.low_quality_min_dim` | `480` | Smaller frame dim below this → lo-q lane |
| `workflow.qa_resume` | `false` | Keep step 3 output instead of regenerating |
| `workflow.qa_use_final_verdict` | `false` | Pick QA lane from 2a's `final_verdict` |
| `workflow.max_workers` / `vlm_workers` / `text_workers` / `qa_workers` | `4` / `0` / `0` / `0` | 0 = use `max_workers` |
| `workflow.daft_validator` | `""` | `tao-daft` binary; step 5 validates both exports |
| `vlm.backend` / `llm.backend` | `"gemini"` | `gemini` or `openai` |
| `models` / `pipeline` | `{}` | Named endpoints + per-step assignment |
| `prompts_module` | `""` | Short name or dotted path |
| `fp_guards` | `"auto"` | False-positive guard stance, inferred from the describer model |
| `license` / `description_extra` / `media_root` | `""` | Step 4 envelope metadata |
| `pipeline_version` / `model_suffix` | `""` | Provenance stamp / output-filename tag |

## Outputs

```
<results_dir>/
  step_0a_filter/       filter_results.jsonl
  step_0b_classify/     classification.jsonl       is_anomaly / is_normal / is_low_quality
  step_0c_routing/      route_*.jsonl              one file per lane
  step_0d_dedup/        videos/ + manifest.jsonl
  step_0e_enhanced/     videos/ + enhanced_map.json
  final_videos/         exactly what was captioned
  step_1a_caption/      captions.jsonl
  step_1b_chunks/       chunk_captions.jsonl
  step_1c_highlight/    highlight_captions.jsonl
  step_2_description/   descriptions.jsonl
  step_2b_reroute/      route_verdict_{anomaly,normal}.jsonl
  step_3_qa/            qa_output.jsonl            one row per (video, prompt_key)
  step_4_output/        <task>_<model_tag>.json    tao-vl-reason-v1.0
  step_4b_contextual/   <video>/contextual/{video,events,chunks,msted}.json   metropolis-v3.0
  step_5_report/        RESULTS_SUMMARY.md, annotation_report.json
```

Step 4a task files: `mcq`, `mcq_openended`, `bcq`, `bcq_openended`, `open_qa`, `event_verification`, `causal_linkage`, `temporal_localization`, `temporal_description`, `scene_description`, `video_summarization` (plus `public_safety` families when enabled). Envelope:

```json
{
  "format": "tao-vl-reason-v1.0",
  "metadata": {"type": "annotation", "task": "<task>", "date": "YYYY-MM-DD",
               "description": "<per-task + description_extra>", "license": "<license>"},
  "media_root": "<media_root or data.video_root>" | null,
  "items": [{"video_id": "...", "question": "...", "answer": "...", "reasoning": "..."}]
}
```

`video_id` resolves through `final_videos/` when it exists, so it shares a timeline with the timestamps. **If 0d or 0e runs, run 0f too.**

## Prerequisites

- **Container**: `nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services` <!-- versions-key: images.tao_toolkit.data_services -->. The pipeline must be the 0a-5 version described here (`workflow.steps` accepts `"0a"`); older images use the retired `"0"`/`"2"`/`"4"` steps. The image must also ship the bundled `auto_label/experiment_specs/video_reasoning_annotation_*.yaml` (packaged only in builds that include `experiment_specs/__init__.py`). There is no fallback: if the directory is missing, stop and ask for a newer image.
- **GPU with NVDEC**: images built on the current LGPL-only data-services base ship an ffmpeg that decodes H.264 **only** through `h264_cuvid`. Older images (e.g. `6.26.3-data-services`) use a GPL ffmpeg with software H.264 decode. Run with `--gpus all` (the image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`). Without a GPU, H.264 `.mp4` inputs fail in chunking, highlight clips and enhancement.
- **ffmpeg / ffprobe**: resolution probing, chunking, dedup, enhancement, highlight clips (VP9/WebM output). **Step 0e does not work on the current LGPL-only data-services base:** its ffmpeg lacks the `rawvideo` format 0e uses to re-encode frames, so 0e enhances 0 videos whatever `enhance_denoise` is set to (the base-image fix is pending). On such images, drop `0e` from `steps` (and keep `0f` if `0d` runs). The default denoiser `hqdn3d` also needs a GPL ffmpeg: it works in older images such as `6.26.3-data-services`.
- **opencv-python-headless + numpy**: only for 0d / 0e.
- **google-genai** and **openai** Python packages: both imported at load regardless of backend.
- **VLM endpoint**: at least one.
