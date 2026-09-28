# Video Reasoning Annotation — Domain Adaptation Guide

Use this reference only when the parent `SKILL.md` points here for the current task. If this file conflicts with current `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current authoritative source wins.

## Contents

- Overview
- Bundled Prompt Modules
- Consultation Process
  - Phase 1 — Understand the annotation goals
  - Phase 2 — Infer caption requirements
  - Phase 3 — Write prompts
- Module Contract
- Placeholder Reference
- Wiring a Custom Module
- Iterative Prompt Tuning


## Overview

All prompts live in the pipeline package `nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts`. `prompts_module` selects one:

- a **short name** (`"retail"`) resolves to `nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts.retail`;
- a **dotted path** (`"my_pkg.prompts_mydomain"`) is imported as-is;
- `""` uses `generic`, the domain-agnostic default.

Check the bundled modules first. Write a custom module only when none of them fits.

`smart_space_grounding` is not a `prompts_module` value. It is the grounding companion of `smart_space`: when `workflow.sta_grounding_labels_dir` is set, the pipeline imports `<resolved prompts_module>_grounding` and fills the step 2a grounding slots from its `GROUNDING_INSTRUCTIONS_BLOCK` and `GROUNDING_SELF_AUDIT`. To use grounding with a custom domain, ship a sibling `<module>_grounding` with those two constants (see configuration.md, Grounding Injection).

## Bundled Prompt Modules

| Short name | Domain | Notes |
|---|---|---|
| `generic` (or `""`) | Any footage | Default. Full 30-key set |
| `smart_space` | Multi-category commercial CCTV | Used by the bundled smart_space specs. Defines false-positive guards (`fp_guards`) |
| `warehouse` | Warehouse / industrial site CCTV | Used by the bundled warehouse specs |
| `retail` | Retail store CCTV | |
| `traffic` | Traffic intersections / highways (no dashcam) | |
| `public_safety` | Public-safety CCTV | Adds QA families `event_type_mcq`, `weapon_detection`, `person_object`. Guards can also be toggled with the `VRA_FP_GUARDS` env var |
| `template` | Scaffold | Same 30 keys as `generic`, with `[PLACEHOLDER]` markers. **Do not run it directly** — copy it and fill it in |

To tune a bundled module (say, `traffic` for specific camera angles), copy its source out of the container (`python -c "import nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts.traffic as m; print(m.__file__)"`), edit the copy, and wire it as a custom module (below).

## Consultation Process

When no bundled module fits, run this consultation before writing any prompts. The goal is to learn **what the user wants their model to learn**, so the prompts capture the right information.

### Phase 1 — Understand the annotation goals

Ask: **"What types of questions do you want the trained model to be able to answer about these videos?"**

Walk through these categories. Not all will apply:

- *Identification / What*: What is happening? What type of event? What objects, people, entities?
- *Temporal / When*: When does the key event occur? What is the sequence?
- *Causal / Why*: What caused this? What led up to it?
- *Attribution / Who*: Who or what is responsible? What are the roles?
- *Consequence / Impact*: What changes after the event? How severe?
- *Spatial / Where*: Where in the scene? Spatial relationships?
- *Behavioral / How*: How do actors behave before, during, after?
- *Counterfactual / Prevention*: How could this have been prevented?
- *Classification / Category*: Normal or abnormal? Which category?
- *Verification*: Did a specific event happen? (maps to `event_verification`)

Then ask: **"What are the most important elements you want captured?"** Key actors/entities, identifying details (clothing, colour, position, labels), key actions and interactions, domain-specific details a general caption would miss, and bystander/environmental reactions.

### Phase 2 — Infer caption requirements

From the answers, infer what the captions MUST contain for the QA to be answerable, as a two-tier checklist:

> **Must capture (directly needed for the questions):**
> - [ ] [Items derived from the user's question types]
>
> **Should capture (context for reasoning):**
> - [ ] [Scene environment, timestamps, pre/post-event state, etc.]

For each question type, ask: "What would a captioner have to observe and write down for this question to be answerable from the caption alone?" From step 2a on the pipeline never sees the video again, so anything missing from the captions cannot be recovered later.

**Wait for user confirmation.** This checklist drives all prompt design.

### Phase 3 — Write prompts

Only after confirmation, copy `prompts/template.py` and fill its placeholders. The caption prompts (`*_global_caption`, `*_dense_caption`, `*_chunk_caption`, `highlight_chunk_caption`) must tell the VLM to observe and report each checklist item. The QA prompts must produce questions that match the user's question types.

**QA coverage:** each `qa_type` in `workflow.qa_types` needs its prompt keys and example placeholders filled. Default types: `mcq`, `bcq`, `open_qa`, `event_verification`, `causal_linkage`, `temporal_localization`, `temporal_event_desc`, `scene_description`, `event_summary`. Most have `anomaly_` and `normal_` variants; `scene_description` and `event_summary` are mode-agnostic. Placeholders for a dropped type can stay unfilled. With `joint_mcq_open_qa: true`, also fill `*_mcq_open_qa` and the topic pairings.

**Key principle:** design top-down, from questions to captions. Poor captions cannot be fixed by better QA prompts.

## Module Contract

A prompt module must expose:

- `PROMPT_TEMPLATES` — a dict with the 30 keys: `video_filtering`, `video_anomaly_classification`, `{anomaly,normal}_{global,dense,chunk}_caption`, `highlight_timestamp_extraction`, `highlight_chunk_caption`, `{anomaly,normal}_description`, `{anomaly,normal}_{mcq,bcq,open_qa,temporal_event_desc,causal_linkage,temporal_localization,event_verification,mcq_open_qa}`, `scene_description`, `event_summary`. Bundled domain modules may omit keys for types they never generate (`traffic` and `warehouse` ship 25).
- `get_prompt(key, **kwargs)`.
- `TOPIC_PAIRINGS` / `NORMAL_TOPIC_PAIRINGS` — `(mcq_topic, open_qa_topic)` pairs for the joint MCQ + open QA prompt. Pair different topics.

Optional hooks the pipeline uses when present:

- `SUBCATEGORY_GUIDANCE` (label → section overrides) and `INCIDENT_FOCUS_DEFAULTS` (fallback sections). Resolved against `data.gt_type_file` labels to steer caption and description prompts per incident category.
- `postprocess_description(text, prompt_key)` — normalizes step 2a output (heading numbering, contradictory section titles). If it raises, the raw text is kept.

Templates use `str.format` fields such as `{chunk_duration}`, `{duration}`, and `{dense_incident_type}`. A field the caller does not supply is filled with an empty string, so a missing field degrades the prompt without failing. Double any literal braces (`{{`, `}}`).

## Placeholder Reference

`template.py` has 89 distinct placeholders. The main groups:

| Placeholder(s) | Fill with | Example (traffic) |
|---|---|---|
| `[DOMAIN]`, `[DOMAIN_CONTEXT]` | Domain name / setting | "traffic surveillance" |
| `[POSITIVE_CRITERION_1..3]`, `[EXCLUSION_1..2]`, `[EDGE_CASE_GUIDANCE]` | What is / is not in-domain (step 0a) | "Fixed elevated view of a road"; "Dashcam footage" |
| `[ANOMALY_DEFINITION]`, `[ANOMALY_EXAMPLE_N]`, `[NORMAL_EXAMPLE_1..3]`, `[DOMAIN_ANOMALY_EVENTS]`, `[DOMAIN_NORMAL_EVENTS]`, `[DOMAIN_EVENT_TYPE]` | Anomaly vs normal (step 0b, captions) | "collision, near-miss, stalled vehicle, rule violation" |
| `[KEY_ASPECT_1..2]`, `[WHAT_DETAILS_TO_CAPTURE]`, `[WHAT_TO_TRACK_FOR_ACTORS]`, `[WHAT_SPATIAL_DETAILS_TO_NOTE]`, `[WHAT_ENVIRONMENTAL_FACTORS_TO_NOTE]`, `[WHAT_CONDITIONS_TO_MENTION]`, `[WHAT_ACTIONS_TO_DESCRIBE]`, `[DESCRIPTION_OF_WHAT_TO_OBSERVE]`, `[DESCRIPTION_FOR_NORMAL_SCENARIOS]` | Caption focus (from the Phase 2 checklist) | "Traffic signal state", "Vehicle colour, type, lane" |
| `[DOMAIN_ACTOR_DETAILS]`, `[DOMAIN_SPATIAL_CONTEXT]`, `[DOMAIN_ENVIRONMENTAL_FACTORS]`, `[DOMAIN_DYNAMICS]`, `[DOMAIN_BEHAVIORS]`, `[DOMAIN_INTERACTIONS]`, `[DOMAIN_INFRASTRUCTURE]`, `[DOMAIN_LAYOUT_DETAILS]` | Domain detail blocks in captions and descriptions | "Intersection layout — lanes, signals, crosswalks" |
| `[DOMAIN_SPECIFIC_PRE_EVENT_DETAILS]`, `[DOMAIN_SPECIFIC_CRITICAL_DETAILS]`, `[DOMAIN_SPECIFIC_AFTERMATH_DETAILS]`, `[DOMAIN_SPECIFIC_EXAMPLE_LINE_1..4]`, `[DOMAIN_SPECIFIC_NORMAL_EXAMPLE_LINE_1..4]` | Description structure (step 2a) | "Signal phase before impact" |
| `[WHAT_TO_DESCRIBE_*]`, `[WHAT_FIXED_ELEMENTS_TO_NOTE]`, `[WHAT_PATTERNS_TO_DESCRIBE]`, `[WHAT_NORMAL_ACTIVITIES_TO_DESCRIBE]`, `[WHAT_INTERACTIONS_TO_NOTE]` | Scene description / event summary guidance | |
| `[DOMAIN_{MCQ,BCQ,OPEN_QA}_EXAMPLE_{QUESTION,OPTIONS,ANSWER,REASONING}]` and `[DOMAIN_NORMAL_…]` variants | Worked QA examples per type | See `traffic.py` |
| `[DOMAIN_{CAUSAL_LINKAGE,TEMPORAL_EVENT_DESC,TEMPORAL_LOCALIZATION}_{ANOMALY,NORMAL}_EXAMPLE]`, `[DOMAIN_SCENE_DESCRIPTION_EXAMPLE]`, `[DOMAIN_EVENT_SUMMARY_EXAMPLE]` | Worked examples for the remaining types | |
| `[ANOMALY_TOPIC_A..C]`, `[NORMAL_TOPIC_A..C]` | Topics for `TOPIC_PAIRINGS` | "causal factor", "vehicle identification" |

For the full list, run `grep -o "\[[A-Z_0-9]*\]" template.py | sort -u`. Before running, check that no `[` placeholder remains in the filled module.

## Wiring a Custom Module

The module must be importable inside the container:

1. Write it (e.g. `prompts_mydomain.py`) into a directory that is mounted into the container.
2. Add that directory to `PYTHONPATH` for the run (for example `-e PYTHONPATH=/workspace/prompts` on `docker run`, or the platform skill's env mechanism).
3. Set `prompts_module: "prompts_mydomain"` — a **dotted** path is required for a top-level module outside the package, so use a package (`mypkg/prompts_mydomain.py` → `"mypkg.prompts_mydomain"`). A bare name without a dot is resolved inside the bundled package instead.
4. If the module defines guards, set `fp_guards` deliberately (see configuration.md).

## Iterative Prompt Tuning

1. Run 3–5 videos with the custom module (any bundled spec, with `prompts_module` changed).
2. `step_1a_caption/captions.jsonl` — does every caption cover the "Must capture" items?
3. `step_2_description/descriptions.jsonl` — accurate, complete, and is `final_verdict` right?
4. `step_3_qa/qa_output.jsonl` — are the QA pairs relevant to the user's question types?
5. `step_5_report/RESULTS_SUMMARY.md` — attrition per stage (prompts that return unparseable output show up as drops).
6. Revise and re-run. Curation and 1a-2a resume; step 3 regenerates unless `qa_resume: true`. Delete the step output directory for a step whose prompt changed, or it resumes with the old output.
7. Scale to the full dataset.

`app_stage_review.py` (Streamlit, from the pipeline repo root) shows every stage per video next to the source clip, which speeds up steps 2–4.

**Quality compounds downstream.** Tune the captions (steps 1a/1b) first.
