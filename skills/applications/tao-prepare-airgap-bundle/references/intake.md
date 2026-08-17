# Turning a goal into a selection

## Contents

- Ask for the outcome, not the module
- Mapping a goal to candidate skills
- Expanding an orchestrating skill
- Choosing the target platforms
- What to show before anything is downloaded

## Ask for the outcome, not the module

**The person running this is usually the one who cannot answer TAO questions.**
An operator can say "DINOv3 SSL"; a customer's engineer says "I need to find
scratches on painted metal parts". Opening with a list of skill names asks them
for the knowledge the skill exists to supply.

So the first question is what they are trying to achieve, and the selection is
derived from the answer. Someone who does know can name skills directly and skip
ahead — the intake is a front door, not a toll gate.

## Mapping a goal to candidate skills

Derive first, ask last. In order:

1. **When the request names a model, an architecture, or a HuggingFace
   repository**, resolve it rather than guessing:

   ```bash
   "${TAO_SKILL_BANK_PATH:?}/scripts/resolve_tao_model.py" --model <name or repo id> --format json
   ```

   It returns the owning skill, the network architecture, how it matched, and
   the container image. An unmatched model exits non-zero rather than guessing,
   which is the answer you want.

2. **When the request names a task rather than a model**, read the catalogue:

   ```bash
   "${TAO_SKILL_BANK_PATH:?}/scripts/list_tao_capabilities.py" --skill-bank "${TAO_SKILL_BANK_PATH:?}"
   "${TAO_SKILL_BANK_PATH:?}/scripts/list_tao_models.py" --skill-bank "${TAO_SKILL_BANK_PATH:?}" --format json
   ```

   Match on what the skills declare — their `description`, `data_format` and
   `network_arch` — rather than on a remembered mapping, which goes stale on the
   next release.

3. **Prefer an orchestrating skill over its parts** when the user described a
   pipeline rather than a single training run. Someone who wants defects found
   wants the loop, not one of its stages.

**Ask at most two clarifying questions, and only where the answer changes what
the user gets.** Labelled or unlabelled data on hand; a deployable engine or a
backbone. Anything determinable by reading the bank is determined, not asked.

## Expanding an orchestrating skill

An orchestrating skill declares no image of its own, so the eligibility screen
returns `undetermined` for every path and sets `requires_expansion`. Read its
stage table, resolve the underlying skill behind each stage, and screen those.

**Take the intersection of their eligible platforms**, not the union: a pipeline
does not run without one of its stages, so a stage that cannot run on a platform
removes that platform for the whole workload. Name the stage that removed it.

The assets are the union, deduplicated by identifier — which is the opposite
direction, and getting the two backwards produces either a bundle missing a
stage's weights or a platform claim the pipeline cannot honour.

## Choosing the target platforms

**Always asked, never derived.** Which platform the destination runs is a fact
about the customer's infrastructure that appears nowhere in the bank. It is also
asked *before* anything is downloaded, because it decides the on-disk form of
every staged asset — see `platform-payloads.md`.

Offer only the platforms that survived the screen and adjudication. Name the
excluded ones with the reason; an operator who sees a platform silently missing
assumes a bug. Allow several: an operator often does not yet know whether the
customer will land on one GPU box or their cluster.

Where a platform is selected that no available verification host can exercise —
a cluster artifact and no cluster to run it on — say so now rather than at
handover, because it changes what the delivery can claim. Preparing that payload
is still worth doing; claiming it was verified is not.

## What to show before anything is downloaded

Four things, together, while every one of them can still be changed:

| | Why it belongs here |
|---|---|
| the selected skills | the user confirms the goal was read correctly |
| the target platforms | and which were excluded, with the reason |
| the estimated size | a multi-platform selection carries the same image twice; say the number while a platform can still be dropped |
| **the capability boundary** | what the customer will *not* get |

The boundary is the most valuable line in the whole run. A workload that
produces a backbone rather than a deployable model, or that reaches a
deployment format only through another network's tooling, has to say so here —
not after a download measured in tens of gigabytes, and not in a handover note
the reader skims.

Write the selection to `.delivery/selection.yaml`, including the platforms and
the reason each was chosen or excluded, so the rest of the run has one source
for what was agreed.
