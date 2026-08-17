# Packing a bundle, and optionally verifying it

## Contents

- Packing: the manifest, and the delivery
- Verification is optional, and it comes last
- The rule that makes it worth doing
- Smoke data
- The run
- Reading the result
- Cleaning up, and proving the bundle is unchanged
- What the log must say

## Packing: the manifest, and the delivery

Packing is the end of the delivery. It runs on the packaging host, needs no
accelerator, and produces the artifact you send.

`$BUNDLE` is the bundle root, set by the operator. The `:?` is load-bearing:
with the variable unset, a plain `cd "$BUNDLE"` succeeds and stays where it was,
so the manifest would describe the current directory instead and still verify
cleanly — the one failure mode this file's own proof step cannot catch.

```bash
cd "${BUNDLE:?set BUNDLE to the bundle root}"
find . -path ./.delivery -prune -o -name MANIFEST.sha256 -prune -o -type f -print0 \
  | sort -z | xargs -0 sha256sum > MANIFEST.sha256
sha256sum -c MANIFEST.sha256
```

**The manifest must be pruned from its own input.** The shell truncates the
output file when it sets the pipeline up, before `find` walks the tree — so
without that prune, `find` lists the now-empty manifest and records the digest
of the empty string against it. Every bundle then fails its own integrity check,
deterministically. That check is the customer's first action on a machine with
no network to ask about it, so this is the single worst line in the delivery to
get wrong.

`-print0`/`-0` are not decoration either: a weights filename containing a space
splits into two nonexistent paths without them.

The second line is not optional. Prove the manifest round-trips — exit 0, every
line `OK` — before calling the bundle done. Then exclude `.delivery/` from the
archive, and send the archive's own checksum by a different route.

**What you have at this point is an *assembled* bundle**, and that is a complete,
shippable outcome.

## Verification is optional, and it comes last

It needs a GPU, and the packaging host is deliberately an ordinary machine
without one. So verification is a separate phase against the already-packed
bundle, on whatever GPU machine is available — offered at the end, and skipped
without ceremony when there is none.

**Ask; do not assume.** If no GPU machine is available, say plainly that the
bundle is assembled rather than verified, record that, and stop.

**It does not transfer between platforms.** Running the docker payload offline
proves nothing about the SLURM or Kubernetes payload — different image format,
different isolation mechanism, different scheduler. A bundle aimed at a cluster
is verified for that cluster only by running it there, which is usually the
customer's own site. Verify what the available machine can, record it per
platform, and claim nothing about the rest.

**Verification must leave the bundle byte-identical.** It runs against the
packed delivery, so anything it writes and fails to remove changes what ships.
The manifest generated at packing time is the proof: re-run `sha256sum -c` after
cleanup, and a mismatch means the procedure left something behind.

## The rule that makes it worth doing

**Generate the bundle's instructions first, then verify by executing them.**

The artifact under test is the artifact that ships. Not a procedure written
beside it, not a command composed for the occasion — the file the customer will
receive, run as written.

This ordering is the whole guarantee. A verification procedure maintained
separately from the delivery drifts *friendlier* than the delivery: it acquires
an environment variable, a flag, a path that the shipped instructions never
carried, and it passes for a reason the customer cannot reproduce. When the
document under test is the document shipped, a step that only works because the
verifier knew something unwritten fails here, which is where it is cheap.

So: if a command has to be adjusted to make the run work, **the fix goes into
the bundle's instructions, and the run starts again.** Adjusting the command and
leaving the file alone converts a caught defect into a shipped one.

**Run every action the bundle claims**, not just the first. A passing training
run says nothing about whether an export step reaches for something. Four
declared actions means four passes before the bundle is verified.

**Verify each selected platform, or say which was not verified.** The record
says so in those words, and the handover repeats it, rather than letting the
reader infer that everything shipped was exercised.

## Smoke data

**Synthetic, always.** The goal is that the bundle comes up with no network,
which needs data of the right shape, not real content. Asking the operator for
the customer's data before the bundle can be checked turns a two-hour job into a
two-week one and proves nothing extra.

Derive the shape from the action's declared `inputs` in the skill's
`references/skill_info.yaml` — a `folder` of images, a file in a named format —
and generate the smallest set that exercises the path. Record in the log what
shape was chosen, so a later reader knows what the pass covered.

**Change only what makes it finish quickly.** Where a smoke configuration is
derived from a shipped spec, adjust epochs, batch size and interval — and leave
every path exactly as the shipped spec has it. Paths belong to the command line
the customer runs, and that command is the thing under test: a configuration
with the paths written in passes whether or not the customer's invocation is
correct.

## The run

**Load the payload from the artifact the customer receives**, not from a cache
already on the verification host. Remove any local copy first, so the load path
is the one being tested rather than skipped.

**Switch the network off at the container, not at the host.** A container with
no network interface fails immediately and loudly on anything reaching outward,
instead of hanging on a timeout — and it does not disconnect the person running
the test. The flags per platform are in `platform-payloads.md`.

**Do not add environment variables the shipped instructions do not carry.**
Offline-mode variables that the packaged software does not read make the test
easier than the delivery while appearing rigorous. If a variable is genuinely
required, it belongs in the bundle's instructions first.

**Never install anything to get past a failure**, even as a probe. A
package-manager call invalidates the run under the air-gap contract this skill
satisfies; fix the bundle and restart from a clean results directory.

## Reading the result

A completed run is not a passed run. Check, in order:

1. The command's own success signal, and a zero exit status.
2. The artifacts the action declares as outputs actually exist.
3. Nothing in the log reached the network and recovered silently.

A file-existence check is not a run. A completeness script that tests only
whether expected paths exist will report success on a tree missing the one file
a later stage needs, and fail only when that stage runs — at the customer.

**A produced artifact is not a correct artifact.** Where a bundle carries a
conversion step whose output cannot be eyeballed, compare it numerically against
the reference implementation on the same fixed input, and record the comparison
rather than the fact that a file appeared.

## Cleaning up, and proving the bundle is unchanged

The procedure removes what it created, while it is still obvious what belongs to
the smoke run and what belongs to the bundle. Name the smoke paths explicitly —
removing a parent directory takes the directory the customer's own run writes
into, which the bundle's instructions told them would be there.

**Then prove the bundle is exactly what was packed.** The manifest already
exists from packing, so this is a check, not a regeneration:

```bash
cd "${BUNDLE:?set BUNDLE to the bundle root}"
sha256sum -c MANIFEST.sha256
```

A mismatch means the procedure left something behind, or removed something it
should not have. Fix the cleanup and re-check; do not regenerate the manifest to
make the difference disappear, which would silently ship whatever the run left.

`.delivery/` is the packager's record and is excluded from the archive; the
verification log is appended to it rather than to the bundle proper.

## What the log must say

`.delivery/verify.log` records three things, and the third is the one a customer
benefits from most:

- **What ran** — every action, every platform, with the exact command.
- **What passed** — the success signal and the artifacts checked.
- **What was not covered** — actions not exercised, platforms staged but not
  run, configurations not tried. Write it as a scope, never as a limit of the
  software: it tells the customer where to validate on their own data.
