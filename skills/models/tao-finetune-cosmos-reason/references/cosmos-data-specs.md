# Cosmos WTS and AETC runtime data contract

The workflow has no dataset default. Every request supplies annotation and
media paths for training and validation. Preserve the submitted string and add
an accessible resolved path; never replace a missing input with another file.

## WTS

Each split has one JSON annotation array and one media root. Every item must
have a media field (`video` for the native WTS contract) and at least two
conversation turns. Validation checks the complete manifest, all referenced
media, unique logical records, nonempty splits, train/validation overlap, and
record/media fingerprints. A directory name alone is not enough: annotation
and media mappings must be explicit in the generated backend spec.

## AETC / TAO VL Reason

Each split accepts one or more annotation files and one shared media root or
one media root per annotation. The canonical envelope is an object with
`format=tao-vl-reason-v1.0`, metadata containing the task, and an `items`
array. Task selection is optional but must produce at least one record.

Supported task names are `bcq`, `mcq`, `bcq_openended`, `mcq_openended`,
`open_qa`, `scene_description`, `video_summarization`,
`temporal_localization`, `temporal_description`, and `causal_linkage`.
Prompts, response targets, frame sampling, and media resolution come from the
versioned DAFT adapter in the repository-derived image.

BCQ and MCQ have deterministic accuracy. Other tasks retain their defined
text/task metrics. They are excluded from aggregate accuracy and listed with a
reason. The aggregate is example-weighted over accuracy-defined records.

## Smoke and full materialization

A smoke plan may materialize a new manifest under the runtime-supplied results
area and apply an explicit sample limit. It records the source manifest and
fingerprint. A full plan rereads the original runtime annotations and rejects
all sample-limit fields. A smoke manifest is never a full-run fallback.

Cosmos-RL may merge multiple AETC annotations into a generated manifest while
preserving every original path and logical record fingerprint. Cosmos
Framework consumes the explicit annotation list natively. Both representations
must fingerprint to the same logical records and media before a comparison.

## Compute-node validation

Submission-host validation is insufficient for SLURM. The allocated-node
preflight repeats readability checks for annotations, media, prepared model,
cache, results, checkpoints, and SQSH through the exact container mounts. It
also decodes representative media with the selected GPU decoder. A missing
mount or unreadable file blocks the full allocation.
