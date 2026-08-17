---
name: tao-prepare-airgap-bundle
description: >-
  Prepare and pack an air-gapped delivery of TAO workloads for a site with no internet, on
  an ordinary machine with no GPU. Works out which skills serve a stated goal, screens which
  execution paths can run offline, downloads every asset those skills need in the form each
  target platform consumes, and packs it with instructions an agent at the destination can
  follow. Optionally verifies the result afterwards by running it with the network off. Use when the user asks to "package
  TAO for an air-gapped customer", "build an offline bundle", "prepare an air-gap
  delivery", "what do I need to download to run TAO offline", or mentions a customer site
  with no internet. Trigger phrases include "air-gapped", "air gap", "offline install",
  "no network", "isolated enclave", "disconnected environment".
license: Apache-2.0
compatibility: Requires Python 3.10+ and network access on the packaging host, plus the runtime of each target platform to convert images - docker to export one, enroot to build a squashfs image for SLURM. No GPU is needed to prepare a bundle. Verifying one by running it offline needs a GPU host, which is a separate step on a separate machine. The destination needs the bundle plus its platform runtime - a container runtime, or a matching Python interpreter for a virtualenv delivery.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- application
- workflow
- airgap
- offline
- packaging
- delivery
---

# TAO Air-Gap Bundle Preparation

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Takes a workload from "the customer has no internet" to a packed bundle, on an ordinary
machine with no GPU. The bundle carries its own instructions, so an agent at the destination
can drive it without this skill and without a network. Running it offline to prove it is a
separate, optional last step.

## Quick Start

```bash
# What could be packaged at all, and what needs a human judgement
scripts/probe_airgap_eligibility.py --all
scripts/probe_airgap_eligibility.py --all --needs-review
scripts/probe_airgap_eligibility.py --skill <skill name> --format json
```

Then state a goal rather than a module — see `references/intake.md`:

```
> I need to find scratches on painted metal parts, offline.
> Package TAO for a customer with no internet, running on their SLURM cluster.
```

## External dependencies

| Dependency | Purpose | Needed for |
|---|---|---|
| Python 3.10+ with PyYAML | the eligibility screen | phases 1–2 |
| network access | staging the assets | phase 3 |
| docker | export an image | phase 3, docker and kubernetes targets |
| enroot | build the squashfs image a SLURM site consumes | phase 3, slurm targets |
| a logging forward proxy | faster asset discovery; optional | see `references/asset-closure.md` |

**No GPU, and no accelerator of any kind.** The packaging host resolves, downloads and
converts files. Nothing it does executes the workload, and it does not need to.

## The deliverable is an assembled bundle

Phases 1–5 run start to finish on an ordinary non-GPU box and produce the thing you send.
That is the endpoint. **Verification is a separate, optional phase 6** on a machine that has a
GPU, offered at the end and skipped without ceremony when no such machine is available.

| Host | Runs | Needs |
|---|---|---|
| **packaging** | phases 1–5 — the whole delivery | network, docker or enroot. **No GPU** |
| **verification** *(optional)* | phase 6, against the packed bundle | a GPU, and the platform the bundle targets |
| **destination** | the delivery itself | the bundle, plus a container runtime — or a matching Python interpreter for a virtualenv delivery |

**Say which of the two you produced.** An assembled bundle's asset list rests on reading and
observation without a run, which `references/asset-closure.md` shows is incomplete by
construction. A verified one has been executed offline. Both are legitimate to hand over;
implying the stronger one is not.

**Verification does not transfer between platforms.** Running the docker payload offline on a
GPU box says nothing about the SLURM or Kubernetes payload — different image format, different
isolation mechanism, different scheduler. A bundle targeting a cluster is only verified for that
cluster by running it there, which is usually the customer's own site. Verify what you can,
record it per platform, and claim nothing about the rest.

## Where code stops and reading starts

`scripts/probe_airgap_eligibility.py` earns its place because its output is
reproducible, names the file behind every verdict — with the line wherever one exists —
covers every skill at once, and runs with no model available, so diffing two runs answers
what changed between two TAO releases. It **screens**; it does not decide. It reports
`eligible`, `ineligible` or `undetermined`, returns `ineligible` only on positive
evidence, and marks its own reasoning `presumed` where it applied a default. You
adjudicate the rest by reading the skill, and a disagreement between the two stops the run.

A verdict drawn from absence cites the file that was read and no line, because there is
no line that says a thing is missing. Treat a citation without a line as a place to look,
not as a fact already established.

Everything else here is judgement and has no script: which skills serve the goal, which
assets a run truly needs, what the bundle's instructions should say. A helper for those
would produce a confident fraction of the answer, and the missing fraction is what fails
at the customer.

## Phases

| Phase | What happens | Writes | Detail |
|---|---|---|---|
| 1 Intake | goal to candidates; screen; adjudicate; choose platforms; show the boundary | `.delivery/selection.yaml`, `.delivery/eligibility.yaml` | `references/intake.md`, `references/eligibility-rules.md` |
| 2 Resolve | candidate asset list, six classes, each with provenance | `.delivery/evidence.yaml` | `references/asset-closure.md` |
| 3 Prepare | download and stage every asset type in the form each platform consumes | `payload/ weights/ wheels/ specs/ skills/` | `references/platform-payloads.md` |
| 4 Author | write the bundle's own instructions | the bundle root | `references/bundle-skill-template.md` |
| 5 Pack | manifest, one archive, checksum by a separate route — **the delivery is now complete** | `MANIFEST.sha256` | `references/verification.md` |
| 6 Verify *(optional)* | on a GPU host: run the shipped instructions offline, every action; close the asset list | `.delivery/verify.log` | `references/verification.md` |

Load a reference when its phase starts; do not preload them. On any conflict the skill
being packaged wins — its `SKILL.md`, `references/skill_info.yaml` and spec templates are
the contract — and the platform skills own their own execution contracts.

## Instructions

1. **Intake.** Ask what they are trying to achieve. Derive candidates, screen them,
   adjudicate every `undetermined` and `presumed` verdict, expand any orchestrating
   skill, then ask which platforms the destination runs. Show the selection, its size and
   the capability boundary before downloading anything.
2. **Resolve.** Build the candidate asset list across all six classes with provenance.
   It is still **open** at the end of this phase, and saying so is part of the output.
3. **Prepare.** Stage every asset type the selection needs — images, wheels, model weights,
   code, specs. Confirm before downloading; these are side-effecting and often tens of
   gigabytes. Report each item as it lands, with its size.
4. **Author.** Rewrite the packaged skill for the offline path: every fetch, login and
   install becomes a presence check. The bundle root **is** a skill — its entry point is
   a real `SKILL.md` with frontmatter, so the destination agent discovers it rather than
   being pointed at it.
5. **Pack.** Generate the manifest, prove it round-trips, exclude `.delivery/`, make one
   archive. Send the archive, and send its checksum by a **different route** — a checksum in
   the same drop as the file it certifies proves nothing about a truncated transfer.
   **The delivery is complete here**, and it is an *assembled* bundle.
6. **Verify — optional, and only now.** Ask whether a GPU machine is available for it. If not,
   say plainly that the bundle is assembled rather than verified and stop; that is a normal
   outcome, not a failure. If one is available, run phase 6 there against the packed bundle:
   the shipped instructions, network off, every action. A fix goes into the bundle's
   instructions and the run restarts — adjusting a command without changing the file converts
   a caught defect into a shipped one. **Verification must leave the bundle byte-identical**,
   which `sha256sum -c MANIFEST.sha256` proves once it has cleaned up after itself.

**Phase 4 precedes phase 6, and that ordering is the guarantee.** Verification executes the
shipped instructions, so no separate procedure can drift friendlier than the delivery.

**Stop after Phase 1 when the user asked a question rather than for a delivery.** What a
bundle would contain is answered by the checklist, not by downloading sixty gigabytes.

## Credentials

Downloads happen on the packaging host while it still has network, so credentials belong
to that side only.

- Confirm a variable is present without reading it: `[ -n "$VAR_NAME" ] && echo SET || echo UNSET`.
- Never place a credential on a command line; prefer a standard-input form where a
  registry login is genuinely required.
- **A credential never enters the bundle.** An offline run authenticates to nothing, so
  the generated instructions record credentials as not applicable. A destination
  preflight asking for a key the run cannot use reads as a broken bundle.
- Where a pin resolves to a pre-release registry organisation, check whether a released
  equivalent exists before asking, and never ship it silently.

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| A workload the customer wanted is reported unpackageable | `undetermined` was read as "no" | adjudicate it; `ineligible` needs positive evidence |
| The bundle fails at the customer on a missing file | the asset list was never closed by an offline run | run phase 6 where a GPU exists; otherwise ship it labelled assembled |
| A SLURM site cannot use the delivered image | a docker export shipped where a squashfs image is consumed | convert on the packaging host |
| A wheel set installs on the packaging host and nowhere else | the closure resolved to a source distribution | wheels only, for the destination interpreter; stop on a package that has none |
| Verification passes but the customer's run does not | the test carried a flag the shipped instructions never had | run the shipped file unmodified |
| The manifest fails after verification | the run left files behind, or removed one it should not have | fix the cleanup and re-check; never regenerate the manifest to hide the difference |
| A pod stays pending on image pull | the cluster has no registry holding the image | settle registry or per-node preload during intake |
