# Eligibility: which paths of a skill can run with no network

## Contents

- The unit of the verdict
- Two sources, one verdict
- E-PATH, E-SVC, E-CLOSURE
- Adjudicating a provisional screen
- Recording the verdict

On any conflict, the current `SKILL.md` of the skill being screened, its
`references/skill_info.yaml`, and the platform skills win over this file.

## The unit of the verdict

**A skill is not air-gap-compatible or incompatible. Each pair of skill and
execution path is.** A skill reachable by docker, kubernetes, slurm and brev is
packageable via docker even though its brev path can never work offline. A skill
is unpackageable only when every one of its paths is.

This is why there is no `airgap:` field in `skill_info.yaml` and no allowlist.
Both would state a verdict about a skill, and the verdict does not exist at that
granularity. Both would also be wrong the release after they were written.

## Two sources, one verdict

The screen is mechanical. The verdict is not. Run both, and make them agree.

| | `probe_airgap_eligibility.py` | The agent |
|---|---|---|
| Reads | declared surfaces, by pattern | the skill's documentation, for meaning |
| Good at | breadth, repeatability, citations, running with no model | intent, novelty, what a command actually does |
| Fails by | matching vocabulary instead of meaning; going silent when the tree is silent | being unrepeatable and unauditable months later |

Neither is sufficient, and the failure modes are not symmetric. A false
*ineligible* hides a workload the customer could have had. A false *eligible*
ships a bundle that cannot run. Both were observed while building this:

- Matching the word `serve` anywhere disqualified two model skills, on the
  strength of "rebuild the engine for each (H, W) you serve" in a reference and
  one line in a reproducibility note.
- Requiring a declared `container_image` disqualified every skill that names its
  image some other way. Only 54 of 77 skills carry a `skill_info.yaml` at all.
  Of the 23 that do not, 3 name a single image URI in their body and 3 cite a
  `versions.yaml` key — 6 workloads a declaration-only rule reports as
  unpackageable, plus 6 more that name several images and need a person to say
  which one their actions run.

So the script reports three verdicts, never two:

| Verdict | Meaning | What the agent owes it |
|---|---|---|
| `eligible` | a path with an image and no live-service dependency | confirm, if the basis is `presumed` |
| `ineligible` | positive evidence removes the path | confirm the evidence says what the script thinks |
| `undetermined` | the tree is silent | **decide it** — this is the agent's to answer |

**`ineligible` is only ever returned from positive evidence.** Absence of a
declaration produces `undetermined`, never a confident no.

The `basis` field carries the same honesty one level down: `declared` means the
skill said so, `evidence` means a line was found, `presumed` means the screen
applied the container-image default that AGENTS.md documents, and
`absent-declaration` means nothing was found either way.

## E-PATH, E-SVC, E-CLOSURE

**E-PATH — which platforms can this skill reach?** Platforms are discovered by
globbing the installed platform skills; a platform present in the tree
self-registers, and there is no central registry to consult. A control-plane
platform is ineligible by construction: provisioning is a call to a service that
cannot be carried across the gap. Every other platform runs on hardware the
customer owns, so its compute side can sit inside an enclave — what is external
is the login node the skill reaches, not the cluster.

**E-SVC — does any action need a service the bundle cannot carry?** This reads
declared surfaces only: the frontmatter description and the `command` of each
declared action. It does not read reference prose, for the reason above. A skill
whose deliverable *is* a served endpoint has no offline form; a skill that
merely mentions serving does.

**E-CLOSURE — can everything the run reads be staged now?** Not decided by
either source. See `asset-closure.md`: it is settled by observation, because the
bank does not record what most skills download.

## Adjudicating a provisional screen

`probe_airgap_eligibility.py --all --needs-review` lists exactly what is
outstanding. For each item, read the skill's own `SKILL.md`, its
`references/skill_info.yaml`, and any reference the Quick Start points at, then
decide. Three rules:

1. **Read the skill, not the screen.** The screen's citation is a starting
   line, not the answer. Open it.
2. **A disagreement is a hard stop.** If the screen says `eligible` and reading
   says otherwise, or the reverse, stop and report both to the operator with the
   line each relied on. Do not resolve it silently in either direction — the
   two sources exist precisely so that this case becomes visible.
3. **Say which source decided.** Every path in the record carries whether the
   screen, the reading, or both produced it.

For an orchestrating skill the screen returns `undetermined` for every path and
sets `requires_expansion`. Expand it: read its stage table, resolve each
underlying skill, screen those, and take the intersection of their eligible
platforms. A stage that is ineligible makes the whole workload ineligible on
that path, because the pipeline does not run without it.

## Recording the verdict

Both sources go into `.delivery/eligibility.yaml`, and the bundle is not built
until every path in it is `eligible` or explicitly excluded by the operator:

```yaml
skill: tao-train-dinov3
paths:
  docker:
    screen:     { verdict: eligible, basis: presumed, source: "<skill_info.yaml line>" }
    adjudicated: { verdict: eligible, by: agent, reason: "Quick Start runs one docker command; no fetch in any action" }
    agreed: true
  brev:
    screen:     { verdict: ineligible, basis: presumed, source: "skills/platform/tao-run-on-brev/SKILL.md" }
    adjudicated: { verdict: ineligible, by: agent, reason: "provisioning calls the control plane" }
    agreed: true
selected_platforms: [docker]
```

`agreed: false` anywhere is a hard stop. It means the two sources read the same
skill and reached different conclusions, and a person has to say which is right
before anything is downloaded.
