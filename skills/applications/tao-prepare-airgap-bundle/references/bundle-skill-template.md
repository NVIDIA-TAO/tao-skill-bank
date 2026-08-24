# The instructions the bundle carries

## Contents

- What the destination receives
- Bundle layout
- Writing the bundle's instructions
- What must not travel

The destination has its own agent. The bundle is therefore not a tarball with a
runbook beside it — it is a set of instructions with its assets attached, and
the agent on the far side reads it the way an agent here reads a skill.

## What the destination receives

Written as agent instructions, and runnable by a person without inference: every
command literal, no step that requires knowing something the file does not say.
One document, not two, because two drift.

## Bundle layout

```
<bundle root>/
  SKILL.md               generated: the destination agent's entry point
  skills/                the slice of the bank this bundle needs, verbatim
  payload/<platform>/    the execution artifact, per platform
  weights/               staged at the paths the entry point names
  wheels/                only when a wheel-consuming path is packaged
  specs/                 copied byte-for-byte from the skill
  MANIFEST.sha256        generated at packing; re-checked if verification runs
  .delivery/             the packager's record; excluded from the archive
    selection.yaml       what was chosen, including platforms and why
    eligibility.yaml     both sources' verdicts, and whether they agreed
    evidence.yaml        every asset with its provenance
    verify.log           what ran offline, what passed, what was not covered
```

**The bundle root is a skill, and its entry point is a real `SKILL.md` with
frontmatter.** That is what makes it discoverable: an agent harness at the
destination finds a skill by that filename and reads the frontmatter to know
when it applies. A differently named runbook has to be pointed at by hand, which
is the thing this bundle exists to avoid.

The bundle is generated into a working directory outside this repository, so the
no-nested-skill rule does not reach it. **The only file that must not be called
`SKILL.md` is the template in this repository** — hence the name of this file.

Give the generated frontmatter the same fields the bank requires, so it passes
the same validation if it is ever brought back in-tree:

```yaml
---
name: <workload>-airgapped
description: >-
  Run <workload> on a host with no network, from staged assets. Use when the
  user asks to "run <workload> offline", "train with no internet", or mentions
  this delivery by name.
license: Apache-2.0
compatibility: Requires <the destination platform runtime> and the assets staged in this bundle. No network access at any point.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- airgap
- offline
- <workload>
---
```

`name` must equal the bundle directory name, for the same reason it does here.
Keep the description free of angle brackets, and keep the body inside the same
size budget — a destination agent pays the same context cost this one does.

## Writing the bundle's instructions

Generate it from the packaged skill, rewritten for the offline path. It is a
rewrite, not a copy:

| In the source skill | In the bundle |
|---|---|
| a pull, fetch, login, or install step | a presence check against a staged path |
| several supported platforms | the selected ones only; the others named as unavailable, with the reason |
| a run command | the same command plus the platform's no-network and no-pull flags |
| a capability the bundle does not carry | stated as out of scope, up front |

Sections, in this order:

1. **What this is, and what it is not.** The deliverable in one line, then the
   boundary. If the packaged workload produces a backbone rather than a
   deployable model, that sentence is the most valuable one in the file.
2. **Integrity.** Verify the manifest before anything else.
3. **Load the payload.** Per selected platform, from `platform-payloads.md`.
4. **Lay out the working directory**, using the mounts the packaged skill's own
   documentation defines.
5. **Run.** The customer's command, with the offline flags, for **every action
   the bundle carries** — not only the ones that were exercised. An assembled
   bundle has verified none, and a delivery whose run section is empty because
   of that is useless. Mark each action verified or not; do not omit the ones
   that were not.
6. **Where output lands**, and which file is the deliverable.
7. **What was verified and what was not**, as a scope. If the bundle was never
   run offline — no verification host was available — say so in the first
   sentence of that section, in those words. A customer who believes a bundle
   was exercised will debug their own environment first when it fails, which is
   the most expensive place for them to start.

   **This section is the one part of the bundle that a later phase rewrites.**
   It is authored saying *assembled*, and a passing verification replaces it
   with what was actually exercised — which action, on which platform, at what
   configuration — after which the manifest is regenerated and the delivery
   re-packed. Write it as its own section with nothing else in it, so that
   rewrite is a replacement rather than surgery on a paragraph that also carries
   instructions.
8. **Troubleshooting**, symptom to cause, covering at least: a manifest
   mismatch, an image not found by name, and a run that stalls rather than
   failing.

**Point at the packaged skill's own documentation rather than restating it.**
The bundle carries that tree; duplicating its content into the entry point creates
two copies that disagree after the first correction.

**Name the telemetry attempt.** Some TAO commands attempt an outbound call when
they finish. It fails with no network and the run still passes, but the
customer's security team will see the attempt on every command — better they
hear it from the operator first than raise it as an incident.

## What must not travel

The bundle's entry point is read by the customer and cannot be recalled.

- No internal registry organisation names, and no internal host names. Resolve
  image pins through `versions.yaml` so only the resolved value travels.
- No statements about what was or was not verified internally, beyond the scope
  paragraph the bundle is entitled to.
- No credential values, and no instruction that would have the customer paste
  one into a file. Where an offline run needs no credential — which is the
  normal case — say that plainly, because a preflight asking for a key that
  cannot be used reads as a broken bundle.
- No other customer, and no earlier delivery.

Two checksums exist and they are not interchangeable. `MANIFEST.sha256` travels
inside the bundle and proves the files survived unpacking. A checksum of the
archive itself does **not** travel with the archive — it goes by a different
route, because a checksum sitting in the same drop as the file it certifies
proves nothing about a truncated transfer.
