# Knowledge a delivery proved that the bank does not declare

## Contents

- The gap this fills
- Where the record lives
- What an entry must carry
- Reading it during a run
- What does not belong here

## The gap this fills

The screen and the six asset classes both derive from what a skill declares.
That is deliberate: derivation stays correct across releases precisely because
it reads the tree rather than a memory. But it means a delivery can establish
something true, useful and undeclared — and the next delivery starts without it.

Two real cases from one packaging run:

- A workload exported ONNX but had no engine builder of its own, so the bundle
  shipped with no deployment path. A prior delivery had established that a
  related network's builder converts that ONNX correctly, and had the numerical
  comparison to prove it. Nothing in the bank says so, so the derivation could
  not know, and the customer got less than was available.
- A skill registered two backbone tiers in `huggingface_model_ids` while
  documenting six. The customer needed one of the four undeclared tiers. The
  weights exist and are reachable; only the declaration is missing.

Neither is a bug in the bank and neither is fixable by reading harder. They are
facts someone verified once, in a delivery, and had nowhere to put.

## Where the record lives

**Beside the skill it is about, not in this one.**

```
skills/<layer>/<skill name>/references/airgap-notes.md
```

The file is **optional and additive**. A skill without one is packaged exactly
as before — this changes no default and blocks nothing. A skill with one gets
the benefit of what a previous delivery learned.

It lives with the packaged skill rather than here for three reasons: it is
knowledge about *that* workload, so its owner is the person who maintains that
skill; it travels into the bundle already, because the skill's tree is staged
verbatim; and putting per-workload knowledge into this skill would make a
general packager grow a section per workload it has ever seen, which is the
shape this design exists to avoid.

## What an entry must carry

The same discipline the external-asset rule imposes, for the same reason: an
unverified note is a rumour, and a rumour that outlives its reason is worse than
no note at all.

```yaml
- id: trt-via-sibling
  claim: >
    A short statement of what is true and is not declared anywhere in the bank.
  verified_on: "2026-08-13"
  verified_by: >
    The command that established it and what it returned. For a numerical
    result, the measurement and its scale -- not "it worked".
  supersede_when: >
    The observable condition that makes this note obsolete. Check it before
    relying on the note; when it is true, delete the entry.
  affects: [the asset class or the phase this changes]
```

**`verified_by` carries the evidence, not the conclusion.** "We tested it" is
not evidence. A comparison against a reference implementation, with the metric
and the number, is.

**`supersede_when` is the part people skip, and it is the reason these notes
stay safe.** Every note here exists because the bank does not declare something.
The day it does, the note is not merely redundant — it is a second source that
can disagree with the first. Write the condition as something a command can
check ("this module exists in that repository", "this field lists more than two
ids"), and check it when the note is read.

## Reading it during a run

In phase 2, after the declared half of the asset list is built, read
`references/airgap-notes.md` under every selected skill if it exists.

For each entry: **evaluate `supersede_when` first.** If it now holds, the bank
has caught up — ignore the entry, say so, and remove it. If it does not, treat
the entry as a *candidate* the operator confirms, exactly like any other
undeclared asset. It is evidence from a previous delivery, not a licence to skip
asking.

Record what was used in `.delivery/evidence.yaml` with `discovered_by:
airgap-notes` and the entry's `id`, so a reader months later can tell which part
of the list came from the tree and which from a prior delivery.

**A note never silently widens the capability boundary.** Where an entry adds a
path the bank does not declare — a deployment format reached through another
network's tooling, say — the bundle says plainly that the route is not something
the vendor documents, alongside the measurement that justified carrying it. The
customer is entitled to make their own call, and can only do that if told.

## What does not belong here

- Anything the bank already declares. Duplicating a declaration creates two
  sources that drift.
- A preference, a default, or a convenience. This records facts that were
  established, not choices that were made.
- Customer-specific detail. A note is about the workload, and it travels into
  every bundle built from that skill.
- An untested belief. If it has no `verified_by`, it does not go in.
