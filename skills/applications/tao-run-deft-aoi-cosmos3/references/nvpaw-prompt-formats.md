# NVPaw Multi-task Prompt Formats

This is the authoritative `nvpaw_multitask_v1` prompt and answer contract.
Prompt wording may vary only through an explicitly selected, versioned
`prompt_variant`; typed answers and metrics do not vary with wording.

| Task type | Image roles | Metric family | Empty answer | Mining | AnomalyGen |
| --- | --- | --- | --- | --- | --- |
| Component Classification | target | classification | `[]` | yes | no |
| Component Count | target | counting | `0` | yes | no |
| Component Detection | target | detection | `[]` | yes | no |
| Defect Classification | target | classification | `[]` or the documented no-defect response | yes | yes |
| Defect Detection | target | detection | `[]` | yes | no |
| Ref_based Defect Classification | golden, target | classification | `[]` or the documented no-defect response | yes | yes |
| Ref_based Defect Detection | golden, target | detection | `[]` | yes | no |

Reference-based prompts always present the golden image first and the target
image second. Coordinates always refer to the target image.

## Classification

Option prompts declare prompt-local choices as `A. ...`, `B. ...`, and so on.
The answer is one letter, a compact list such as `[B,D]`, or `[]`. The
materializer resolves letters to the semantic option text before evaluation;
letters are never treated as globally meaningful classes.

Direct defect-presence questions use these canonical responses:

```text
No, the target image does not contain any defects.
Yes, the target image contains a defect.
```

## Component Count `official_v1`

Use one target image and this wording:

```text
This is an industrial visual inspection task for PCBA component analysis. Use
only the provided image.
How many components are visible in this PCBA image?
Answer with only the integer count.
```

The answer is one non-negative base-10 integer with no prose, units, sign, or
decimal point. `0` is the empty count. Counting KPI uses count-only instance
F1: `TP=min(ground_truth,prediction)`, excess predictions are FP, and missing
predictions are FN. Reports also include exact-count accuracy and mean absolute
error.

## Detection `official_v1`

The task-specific subject can be substituted into this wording:

```text
Where is {object} located in the target image? Provide its bounding box
coordinates.

Return a JSON-formatted list of bounding boxes and labels for the associated
objects identified. Use [x1, y1, x2, y2] integer coordinates normalized to
[0, 1000] relative to the target image width and height.
```

The answer is a compact JSON list:

```json
[{"bbox_2d": [120, 80, 640, 720], "label": "missing component"}]
```

Every coordinate must be an integer in `[0, 1000]`, with `x2 > x1` and
`y2 > y1`. Each object requires a non-empty label. No objects is exactly `[]`.
AnomalyGen must not emit detection records unless an approved geometric
artifact supplies the boxes.

## Evaluation runtime

`bare_okng` retains its exact OK/NG system prompt and four-token generation
limit. `nvpaw_multitask_v1` preserves the task prompt, uses a neutral system
instruction, and defaults to `max_tokens=1024`. The evaluator generates raw
responses only; bundled task-aware metrics own parsing and scoring.
