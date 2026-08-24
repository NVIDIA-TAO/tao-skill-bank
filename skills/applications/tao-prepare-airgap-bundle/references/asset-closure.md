# Closing the asset list

## Contents

- Why reading is not enough
- The six classes
- Closing the list by observation
- Recording what was staged
- What to say about what is still open

On any conflict, the skill being packaged wins: its `SKILL.md`,
`references/skill_info.yaml`, and the spec templates it ships are the contract.

## Why reading is not enough

**The bank records what consumes an asset, not which asset.** Three of the
forty-three skills that declare a usable image set `huggingface_model_ids`, and no spec
template in the bank names a concrete asset URI — while the fields that take one
are everywhere. Counting declarations across every shipped spec template:
148 `pretrained_model_path`, 131 `checkpoint`, 100 `model_path`, 67
`pretrained_backbone_path`, 12 `pretrained_path`.

Where a skill does declare its weights, the declaration is usually a subset.
One model skill registers two backbone tiers and documents six; a delivery built
from the declared pair would omit the tier the customer selected.

**A previous delivery may already have answered part of this.** If the packaged
skill carries `references/airgap-notes.md` in its own tree, read it here — it records what an
earlier packaging run established and the bank still does not declare, with the
evidence and an expiry condition. See `undeclared-knowledge.md`. It supplements
the declared half; it never replaces asking.

**Do not resolve this by inference.** Reading a spec field and constructing a
plausible repository id produces a list that looks complete and is wrong, and
nothing downstream catches it — the bundle ships, and the run fails at the
customer where there is no fixing it. An asset is either observed or asked
about; it is never derived from the shape of a field name.

## The six classes

Every class is staged differently and only one of them is not packaged.

| Class | Where it comes from | Staged as |
|---|---|---|
| container image | the skill's declared image, resolved through `versions.yaml` | per platform — see `platform-payloads.md` |
| wheels | `versions.yaml` wheel pins plus their transitive closure | a wheel directory, resolved for the destination's exact interpreter |
| code | the skill's own tree | copied verbatim, so the destination can read it |
| model weights | observation, then confirmation with the operator | files at an explicit path the spec points at |
| specs | the skill's `references/spec_template_*.yaml` | copied byte-for-byte, never edited |
| data | the customer | **never packaged** |

Read the skill's `references/skill_info.yaml` for the declared half: the image,
the actions, and each action's `inputs` with their `type`. An input of
`type: file` or `type: folder` is a path the destination must be able to resolve
offline, so every one of them needs either a staged asset or an explicit note
that the customer supplies it.

**A weight is not one file.** Tokenizer files, `config.json`, preprocessor
configs and index files travel with it. Stage the directory, not the checkpoint.

**Point specs at files, not at a cache.** A staged snapshot is not a
HuggingFace hub cache; a cache expects a `models--<org>--<repo>/snapshots/<sha>/`
layout, and a spec field left empty sends resolution back to the hub, which
fails with the network off. The path in the spec must name the file.

## Closing the list by observation

Two mechanisms, and both belong to the optional verification phase. Use the
capture when the infrastructure is available; the offline loop is the one that
proves the result, and the capture only makes it converge faster. Where
verification is not run at all, neither happens and the list simply stays open —
see below.

**Both mechanisms run the workload, so neither runs on the packaging host.**
They belong to the verification host, which has the accelerator. That is why the
list this phase produces is explicitly *candidate* and explicitly *open*: on a
packaging host there is no way to close it, and pretending otherwise would put a
reading-derived list into a bundle labelled as checked.

Where no verification host is available at all, the list stays open for the life
of the delivery. That is a legitimate outcome and a much weaker one — the bundle
is assembled rather than verified, and both the record and the handover have to
say which.

**Observation pass, when a logging proxy is available.** Run the skill once with
the network up and every request logged. Each request is a candidate asset with
a URL and a size. This finds in one run what the loop finds in several, and it
finds fetches on code paths a short run would not otherwise reach.

**The offline loop, always.** Stage what is known, then run under no network,
with every action the selected path exposes — not just `train`. A reach-out
fails loudly and names itself. Stage what it named. Run again. The list is
closed when a full pass over every action completes with the network off.

```
   stage what is known
        |
        v
   run with no network, every action  ---- clean ----> closed
        |
        | named something missing
        v
   stage it, record where it came from
        |
        +--> run again
```

**Every action, not one.** A passing single-epoch training run says nothing
about whether `export` reaches for something. A skill with four actions needs
four passes before its list is closed.

**Never install anything to get past a failure.** A package-manager call during
an air-gapped run invalidates the run — see the air-gap contract that
`skills/applications/tao-run-deft-aoi/references/air-gap.md` defines, which this
skill satisfies rather than restates. Fix the bundle and restart from a clean
results directory. This is exactly why the loop runs here and not at the
customer.

## Recording what was staged

Every asset in `.delivery/evidence.yaml` carries where it came from and how that
was established. An asset with no provenance does not go in the bundle.

```yaml
assets:
  - id: A1
    class: image
    identifier: <image URI as resolved>
    source: <skill_info.yaml path and line>
  - id: A2
    class: weights
    identifier: <repository or file set>
    source_external: <URL>
    verified_on: <date>
    verified_by: <the command that confirmed it, and what it returned>
    discovered_by: observation      # declared | observation | operator | airgap-notes
    note_id: <the airgap-notes entry id, when discovered_by is airgap-notes>
```

An asset resolved outside the bank carries `source_external` **with**
`verified_on` and `verified_by`. That pair is the whole distinction between
checking and guessing, months later, for a reader who was not there.

## What to say about what is still open

A conditional fetch on a code path no action exercised will not appear in either
mechanism. The closure is proven for the actions run, at the shapes run.

Write that as a scope in the bundle, never as a capability claim: *"verified for
these actions at this configuration; other configurations were not exercised in
this delivery."* Do not write *"only these actions work"* — that asserts
something about the software nobody tested, and a customer who needs one of the
others will believe a door is shut that is merely unopened.
