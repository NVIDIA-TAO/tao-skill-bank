# Video Reasoning Annotation — Full Configuration Reference

Use this reference only when the parent `SKILL.md` points here for the current task. If this file conflicts with current `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current authoritative source wins. The source of truth for defaults is `VideoReasoningAnnotation*Config` in `nvidia_tao_ds/config/auto_label/default_config.py`.

## Contents

- Bundled Specs
- YAML Structure
- Endpoint Configuration (openai / gemini)
- Per-step Model Routing (`models` / `pipeline`)
- Grounding Injection
- Workflow Parameters
- Data Parameters
- Top-level Parameters
- Key Configuration Decisions
- Error Patterns


## Bundled Specs

Shipped in `nvidia_tao_ds/auto_label/experiment_specs/`. Copy one, fill the `???` placeholders (`results_dir`, `data.video_root`, and `base_url` for openai specs), and run `auto_label generate -e <copy>`.

| Spec | Backend | Prompts | Notes |
|---|---|---|---|
| `video_reasoning_annotation_smart_space_openai.yaml` | openai | `smart_space` | Any OpenAI-compatible endpoint. VLM captioner/describer + text reasoner |
| `video_reasoning_annotation_smart_space_gemini.yaml` | gemini | `smart_space` | google-genai SDK; no `base_url` |
| `video_reasoning_annotation_smart_space_openai_gpt5.yaml` | openai | `smart_space` | Image-only reasoning model: `frames_only: true`, `reasoning_model: true` |
| `video_reasoning_annotation_smart_space_openai_grounding.yaml` | openai | `smart_space` | Grounding injection: steps `["2a","2b","3","4a"]`, `mode: anomaly`; needs `workflow.sta_grounding_labels_dir` |
| `video_reasoning_annotation_warehouse_{openai,gemini,openai_gpt5}.yaml` | as named | `warehouse` | Same shapes as the smart_space trio |
| `video_reasoning_annotation_public_safety_openai.yaml` | openai | `public_safety` | All 14 steps; `qa_use_final_verdict: true`; domain QA families |

For `retail`, `traffic`, or `generic` prompts, start from the closest spec and change `prompts_module`.

**Preparing a copy:** use `scripts/prepare_vra_spec.py` (see `SKILL.md` Quick start) rather than hand-copying. It extracts the named spec from the target image, applies only the dotted-path fields you `--set` (including multi-line values such as `qa_types`), refuses typos and whole-section overrides, checks that the result parses to the requested values, and diffs it against the untouched original. Bundled specs change between image builds, so always prepare the spec from the image you are about to run; never reuse a copy prepared against another build.

## YAML Structure

Condensed from `video_reasoning_annotation_smart_space_openai.yaml`:

```yaml
results_dir: ???
autolabel_type: "video_reasoning_annotation"

video_reasoning_annotation:
  vlm:
    backend: "openai"
    openai: &vlm_endpoint
      api_key: ""                      # or export OPENAI_API_KEY (bare key)
      base_url: ???                    # https://<host>/v1
      model_name: "..."
      tag: "gemini3"                   # stamped into output filenames
      temperature: 0.2
      max_tokens: 8192                 # raise to 16384 for long clips
      timeout: 300
      video_content_type: "image_url"  # or "video_url" (vLLM/NIM) — verify one caption
      frames_only: false
      reasoning_model: false
  llm:
    backend: "openai"
    openai: &llm_endpoint
      api_key: ""
      base_url: ???
      model_name: "google/gemma-4-31B-it"
      tag: "gemma4"
      temperature: 1.0
      top_p: 0.95
      top_k: 64
      max_tokens: 8192
      timeout: 180

  models:
    captioner: {backend: "openai", openai: *vlm_endpoint}
    describer: {backend: "openai", openai: {<<: *vlm_endpoint, max_tokens: 16384}}
    reasoner:  {backend: "openai", openai: *llm_endpoint}
  pipeline:
    video_filtering: captioner
    anomaly_normal_classification: captioner
    global_caption: captioner
    dense_caption: captioner
    chunk_caption: captioner
    highlight_caption: captioner
    highlight_timestamp: captioner
    description: describer
    mcq: reasoner
    bcq: reasoner
    open_qa: reasoner
    mcq_open_qa: reasoner
    event_verification: reasoner
    causal_linkage: reasoner
    temporal_localization: reasoner
    temporal_event_desc: reasoner
    scene_description: reasoner
    event_summary: reasoner
    contextual_parsing: reasoner

  workflow:
    steps: ["0a","0b","0c","0d","0e","0f","1a","1b","1c","2a","3","4a","4b","5"]
    mode: "auto"
    caption_passes: 1
    promote_votes: 1
    use_folder_floor: true
    low_quality_min_dim: 480
    dedup_sensitivity: 3.0
    enhance_scale: 2.0
    enhance_denoise: "hqdn3d"
    max_workers: 4
    qa_types: ["mcq","bcq","open_qa","event_verification","causal_linkage",
               "temporal_localization","temporal_event_desc",
               "scene_description","event_summary"]
    qa_resume: false

  data:
    video_root: ???
    input_jsonl_files: []
    filter_field: null
    routing_manifest: ""
    gt_type_file: ""
    df_annotations_dir: ""

  prompts_module: "smart_space"
  fp_guards: "auto"
  license: ""
  description_extra: "smart_space multi-category commercial CCTV"
  media_root: ""
  strip_subpath_tail: ""
  model_suffix: ""
  pipeline_version: ""
  permute_mcq_options: true
  event_verification_instruction: false
```

**Anchors:** endpoint blocks are defined once under `vlm`/`llm` and reused in `models`. A CLI override like `video_reasoning_annotation.vlm.openai.model_name=...` changes only `vlm`, not the resolved copies under `models`, and every step routed through `pipeline` uses `models`. Edit endpoints in the YAML.

## Endpoint Configuration

Both `vlm` and `llm` (and each `models.<name>`) have the shape `{backend, gemini: {...}, openai: {...}}`. `backend` selects which block is used.

**The gemini block is not the openai block renamed.** `model` not `model_name`, `max_output_tokens` not `max_tokens`, no `base_url`; `frames_only` / `reasoning_model` / `video_content_type` do not exist there.

### openai (`VideoReasoningAnnotationOpenAIConfig`)

"openai" is a protocol: self-hosted vLLM/NIM, any OpenAI-compatible gateway, or GPT. To self-host, see `skills/applications/tao-run-inference-service`.

| Field | Default | Description |
|---|---|---|
| `api_key` | `""` | Or `OPENAI_API_KEY` env var (bare key, no `Bearer `) |
| `base_url` | `""` | `/v1` root or full `/v1/chat/completions` path |
| `base_urls` | `[]` | Interchangeable replicas; one picked per request; overrides `base_url` |
| `model_name` | `""` | Served model name |
| `tag` | `""` | Short label stamped into step 4/4b filenames |
| `temperature` | `0.7` | Ignored when `reasoning_model` is set |
| `max_tokens` | `4096` | Raise for long clips — a truncated caption is rejected on every retry and the video is dropped |
| `timeout` | `60` | Seconds |
| `video_content_type` | `"video_url"` | `video_url` (vLLM/NIM) or `image_url` (some gateways). Wrong value can silently drop video |
| `frames_only` | `false` | Image-only models: send timestamp-labelled JPEG frames instead of video |
| `frames_only_sample_fps` / `_min_frames` / `_max_frames` | `6.0` / `8` / `360` | Frame sampling when `frames_only` |
| `frames_max_dim` | `768` | Downscale-only longest side per frame |
| `reasoning_model` | `false` | Send only model/messages/max_tokens (for models that 400 on sampling params) |
| `top_p` / `top_k` | `-1` | Negative = omitted from the request |
| `enable_thinking` | `false` | Request visible reasoning via `chat_template_kwargs` |

### gemini (`VideoReasoningAnnotationGeminiConfig`)

| Field | Default | Description |
|---|---|---|
| `api_key` | `""` | Or `GOOGLE_API_KEY` env var |
| `model` | `"gemini-3.1-flash-lite-preview"` | Gemini model name |
| `tag` | `""` | Output-filename label |
| `media_resolution` | `"MEDIA_RESOLUTION_LOW"` | LOW / MEDIUM / HIGH |
| `temperature` | `0.3` | |
| `max_output_tokens` | `8192` | |
| `timeout` | `120` | Seconds |

## Per-step Model Routing

`models` defines named endpoints; `pipeline` maps a step name to a `models` key. Unmapped steps fall back to `vlm` (video steps) or `llm` (text steps); an empty map reproduces that default.

- **Video steps:** `video_filtering`, `anomaly_normal_classification`, `global_caption`, `dense_caption`, `chunk_caption`, `highlight_caption`.
- **Text steps:** `highlight_timestamp`, `description`, `mcq`, `bcq`, `open_qa`, `mcq_open_qa`, `event_verification`, `temporal_event_desc`, `causal_linkage`, `temporal_localization`, `scene_description`, `event_summary`, `contextual_parsing` (the full list is `VIDEO_STEPS` / `TEXT_STEPS` in `model_router.py`).

Interactions to know before changing a model:

- **`fp_guards: auto`** infers the false-positive guard stance from the **description** model: reasoning-family describer → guards off; otherwise on. Choosing a describer silently chooses a detection posture. Force with `on`/`off`.
- **`frames_only`** is a behavioural change: sampled frames can miss fast or occluded motion, which shows up as missed incidents, not errors.
- **Token budget** scales with clip length; the bundled specs give `describer` 16384.

## Grounding Injection

Optional. Step 2a can take per-clip grounding labels — a frame-accurate, actor-attributed event timeline from a person-tracking system — and inject them into the description prompt, where they outrank the VLM captions on *what happened*. Captions still supply scene context.

- **Enable:** set `workflow.sta_grounding_labels_dir`. Layout: `<dir>/<clip_uuid>/**/events_reconciled_natural.json` (change the name with `grounding_filename`). `clip_uuid` must match the video stem (including an `_enh` suffix when 0e produced the captioned copy). If there are several matches, the deepest one is used.
- **Domain instructions:** the pipeline imports `<resolved prompts_module>_grounding` (for `smart_space` that is the bundled `smart_space_grounding`) for `GROUNDING_INSTRUCTIONS_BLOCK` and `GROUNDING_SELF_AUDIT`. If that module is missing, the events are still injected but the domain judgment instructions are absent, and a warning is logged. Other domains need their own `<module>_grounding`.
- **Step 2b** splits `descriptions.jsonl` into `step_2b_reroute/route_verdict_{anomaly,normal}.jsonl` by `final_verdict`. It runs when listed, or implicitly when grounding is loaded with 2a and 3. **Nothing consumes its output yet.** To make QA follow the verdict, set `qa_use_final_verdict: true`.
- **Bundled spec:** `video_reasoning_annotation_smart_space_openai_grounding.yaml` runs `["2a","2b","3","4a"]` in `mode: anomaly` over captions from an earlier run. Step 2a reads captions only from `<results_dir>/step_1a_caption/`, `step_1b_chunks/` and `step_1c_highlight/`, so `results_dir` **must** be that earlier run's directory. The spec's `data.input_jsonl_files` (a `chunk_captions.jsonl` path) is not what 2a reads. The spec leaves out `event_verification` from `qa_types` and sets no `tag`.

## Workflow Parameters (`workflow.*`)

| Parameter | Default | Description |
|---|---|---|
| `steps` | `["0a","0b","1a","1b","1c","2a","3","4a","4b","5"]` | Steps to run. Curation `0a`–`0f`, labeling `1a`,`1b`,`1c`,`2a`,`2b`,`3`,`4a`,`4b`, review `5`. `auto` mode adds 0a/0b. This is the pipeline dataclass default; the skill's `skill_info.yaml` passes the full 14-step list (0a–0f included) unless you override it |
| `mode` | `"auto"` | `auto` (classifier decides), `anomaly`, `normal` |
| `max_workers` | `4` | Default concurrency |
| `vlm_workers` / `text_workers` / `qa_workers` | `0` | Per-group concurrency; 0 = `max_workers` |
| `caption_passes` | `1` | >1: N independent 1a passes on the hi-q anomaly lane, merged catch-if-any |
| `promote_votes` | `1` | Classifier votes to promote to anomaly; needs repeated passes to matter |
| `use_folder_floor` | `true` | `anomal*` folder = anomaly floor the classifier cannot demote. Off for unlabelled pools |
| `low_quality_min_dim` | `480` | Smaller frame dim below this → lo-q (single-pass) lane |
| `dedup_dir` | `""` | 0d output; default `<results_dir>/step_0d_dedup`. Put outside results_dir to reuse |
| `dedup_sensitivity` | `3.0` | Higher = stricter. Drives a destructive cut |
| `dedup_workers` | `8` | Decode-bound |
| `enhance_dir` | `""` | 0e output; default `<results_dir>/step_0e_enhanced` |
| `enhance_filter_field` | `"is_low_quality"` | Routing flag that selects videos for 0e |
| `enhance_workers` | `4` | CPU-bound |
| `enhance_scale` | `2.0` | Lanczos upscale factor |
| `enhance_denoise` | `"hqdn3d"` | ffmpeg `hqdn3d`/`nlmeans`/`atadenoise`/`vaguedenoiser` (need a GPL ffmpeg, e.g. `6.26.3-data-services`), opencv `nlm`/`bilateral`/`temporal_median`, or `none`. On the current LGPL-only base, 0e itself fails (no `rawvideo` format), so no setting helps there: drop `0e` |
| `enhance_clahe_clip` / `enhance_clahe_tile` | `2.0` / `8` | CLAHE on L channel |
| `enhance_unsharp_amount` / `_sigma` / `_thresh` | `0.6` / `1.5` / `3.0` | Unsharp mask; amount 0 disables |
| `max_video_length_sec` | `300` | Longer videos skipped |
| `chunk_duration_options` | `[5,10,15,20,30]` | 1b chunk-length candidates |
| `max_chunks` | `10` | Max chunks per video |
| `min_chunk_duration_sec` | `0` | Floor on auto-selected chunk length; 0 disables |
| `highlight_before_sec` / `highlight_after_sec` | `3.0` / `3.0` | 1c clip window |
| `long_video_threshold_sec` | `60` | Above this, videos are sent as sampled frames |
| `long_video_sample_fps` / `long_video_max_frames` | `0.5` / `60` | Long-video sampling |
| `qa_types` | `mcq, bcq, open_qa, event_verification, causal_linkage, temporal_localization, temporal_event_desc, scene_description, event_summary` | Step 3 types. `public_safety` adds `event_type_mcq`, `weapon_detection`, `person_object` |
| `joint_mcq_open_qa` | `false` | One call for MCQ + open QA with distinct topics; needs `*_mcq_open_qa` prompts |
| `qa_resume` | `false` | Keep step 3 output. Only when staged video paths are unchanged |
| `qa_use_final_verdict` | `false` | Choose QA lane from 2a `final_verdict` instead of 0c lane |
| `sta_grounding_labels_dir` | `""` | Root of `<clip_uuid>/**/<grounding_filename>`; enables grounding injection at 2a. Empty = off |
| `grounding_filename` | `"events_reconciled_natural.json"` | Grounding label file discovered under each clip dir |
| `daft_validator` | `""` | `tao-daft` path; step 5 runs `validate` on both exports. Empty = skipped (not a failure) |

## Data Parameters (`data.*`)

| Parameter | Default | Description |
|---|---|---|
| `video_root` | `""` | Walked recursively. At least one of `video_root` / `input_jsonl_files` |
| `input_jsonl_files` | `[]` | JSONL with `video_path` (or `video`) per line; merged with `video_root` |
| `filter_field` | `null` | Keep only JSONL rows where this boolean field is truthy |
| `routing_manifest` | `""` | Trusted per-video labels (JSON/JSONL); lanes built from it instead of the classifier |
| `gt_type_file` | `""` | `{stem: label}` or manifest list with `VideoID` + `IncidentSubCategory`; steers caption/description prompts |
| `df_annotations_dir` | `""` | `<stem>/contextual/{video_df,events_df}.json`; human annotations injected as authoritative context |

## Top-level Parameters (`video_reasoning_annotation.*`)

| Parameter | Default | Description |
|---|---|---|
| `prompts_module` | `""` | Short name (`smart_space`, `warehouse`, `retail`, `traffic`, `public_safety`, `generic`) or dotted path; `""` = generic |
| `fp_guards` | `"auto"` | `auto` / `on` / `off` |
| `license` | `""` | `metadata.license` in step 4 envelope |
| `description_extra` | `""` | Appended to per-task description |
| `media_root` | `""` | Envelope `media_root`; defaults to `data.video_root`. Use a portable label to avoid host paths |
| `strip_subpath_tail` | `""` | Drop a trailing path part when naming 4b output dirs (e.g. `raw/main`) |
| `model_suffix` | `""` | Extra tag in step 4 filenames |
| `pipeline_version` | `""` | Stamped onto every output record |
| `permute_mcq_options` | `true` | Seeded shuffle of MCQ options (counters letter bias) |
| `event_verification_instruction` | `false` | Append answer-format instruction to event_verification questions |

## Key Configuration Decisions

| Decision | Field | Guidance |
|---|---|---|
| Steps | `workflow.steps` | Bundled specs run 0a-0f. Drop `0d` if no repeated footage (most expensive non-API step). Drop `0e` if `route_anomaly_singlepass` is empty. **If 0d or 0e runs, run 0f** |
| Lanes | `mode`, `use_folder_floor`, `routing_manifest` | Mixed pool → `auto`. Trusted labels → `routing_manifest`. Unlabelled folders → `use_folder_floor: false` |
| Recall vs cost | `caption_passes` | >1 recovers secondary actors on hi-q anomalies, linear cost |
| Models | `models` / `pipeline` | Strong native-video model for captions; cheaper text model for QA. Describer choice sets `fp_guards` in auto |
| Resume | `qa_resume` | Leave off unless the staged video paths are identical |
| Verdict mismatch | `qa_use_final_verdict` | Turn on to stop asking about incidents the description says are absent |
| Metadata | `license`, `description_extra`, `media_root`, `pipeline_version` | Set before a release run |

## Error Patterns

| Symptom | Cause | Fix |
|---|---|---|
| Captions say "please provide the video" | Wrong `video_content_type` for the gateway (HTTP 200, video dropped) | Switch `image_url`↔`video_url`; check one caption by eye |
| Videos missing after a run, no error | 0a filter error leaves the video absent | Compare `filter_results.jsonl` rows with "Filtering N videos" log line |
| Long videos dropped | Caption truncated at `max_tokens` → completeness check rejects every retry | Raise captioner `max_tokens` / `max_output_tokens` |
| HTTP 400 on temperature/top_p | Reasoning-family model rejects sampling params | `reasoning_model: true` |
| No grounding events in descriptions | `clip_uuid` directories don't match video stems (e.g. missing `_enh` after 0e), or wrong `grounding_filename` | Rename the dirs or fix `grounding_filename`; check the "with grounding injection" log line |
| Warning "no grounding config found" | No `<prompts_module>_grounding` module | Expected for non-smart_space domains; add one for domain judgment instructions |
| CLI override of `vlm.openai.*` has no effect | Steps routed via `pipeline` use anchored copies in `models` | Edit the YAML, or `--set video_reasoning_annotation.vlm.openai.<field>=...` with `scripts/prepare_vra_spec.py` before launch |
| `prepare_vra_spec.py` exits with "override keys never matched a line" | `--set` dotted path doesn't match this image's bundled spec exactly — either a typo, or the field moved/was renamed in a newer image build | Re-run without `--no-diff` on an unmodified extract to see the file's real structure; fix the path |
| A spec field you remember from a prior run is gone / renamed / has a new default | The image was rebuilt and its bundled spec changed | Expected — bundled specs are versioned with the image, not with this repo. Always regenerate via `prepare_vra_spec.py` against the image you're about to run; never reuse a spec prepared against an older image build |
| Stale/mixed QA items in step 4 | `qa_resume: true` after paths changed (0d/0e toggled, new source) | Set `qa_resume: false` and re-run |
| `ffprobe` on an H.264 `.mp4` returns empty width/height, no visible error | Missing `--gpus all` — this image family's `ffmpeg` decodes H.264 only through NVDEC (`h264_cuvid`), which needs `libnvcuvid.so.1` | Add `--gpus all`; if still unclear, rerun with `-show_streams` (not `-v error -show_entries ...`) to surface the `Cannot load libnvcuvid.so.1` message |
| step 4 `video_id` / 4b metadata don't match timestamps | 0d or 0e ran without 0f | Add `0f` |
| `gemini` block ignored fields / errors | openai field names in the gemini block | Use `model`, `max_output_tokens`; no `base_url` |
| `GOOGLE_API_KEY` / `OPENAI_API_KEY` not set | Missing env var | Export before launch; never put keys in the spec |
| 429 / rate limits | Too much concurrency | Lower `max_workers` or split `vlm_workers` / `qa_workers` |
| Import error for `prompts_module` | Wrong short name / module not on `PYTHONPATH` | Use a bundled name or mount the module and set a full dotted path |
| `ImportError: cv2` | 0d/0e enabled without opencv | Install `opencv-python-headless` or drop 0d/0e |
| ffprobe not found | Missing ffmpeg | Install ffmpeg |
| 0e enhances 0 videos (`ffmpeg rc=234, frames=0`, `Requested output format 'rawvideo' is not known`) | Image on the current LGPL-only base: its ffmpeg has no `rawvideo` format, so 0e cannot re-encode | Drop `0e` from `steps` until the base image includes `rawvideo` |
| `Denoise pre-pass failed` warnings in 0e | The ffmpeg build has no `hqdn3d`/`nlmeans`/… filters (GPL filters, absent from the LGPL-only base) | Use an image with a GPL ffmpeg, or an opencv option (`nlm`, `bilateral`, `temporal_median`) once 0e works on that image |
| Step N reads empty input | Upstream step produced nothing | Check the previous step's JSONL and `step_5_report/RESULTS_SUMMARY.md` attrition |
