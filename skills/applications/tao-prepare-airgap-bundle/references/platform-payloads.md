# What to stage, per platform

## Contents

- Why the platform is chosen before anything is downloaded
- Docker
- SLURM
- Kubernetes
- Virtualenv
- The shared payload: identical on every platform
- One bundle, several payloads
- Keeping the payload small
- No-pull enforcement is not uniform, and the bundle must say so
- Cutting the network for a verification run

The platform skills own their execution contracts. Read
`skills/platform/tao-run-on-<platform>/SKILL.md` for mounts, GPU flags, user
mapping and the four-verb contract; this file covers only what changes when the
destination has no network, and it defers to those skills on everything else.

**Every command below belongs to one of two sides, and the sides never meet.**
The packaging host resolves, downloads and converts — it produces files and
nothing else. It has no route to the customer's daemon, cluster or scheduler, so
it never loads an image, pushes to a registry, touches a node, or installs
anything at the destination. Those commands travel *in* the bundle for the
customer to run on arrival. Where a section shows both, it says which is which;
if you find yourself about to run a destination command from the packaging host,
the air gap is the reason it will not work.

## Why the platform is chosen before anything is downloaded

**The same skill on two platforms needs different artifacts.** A container image
is not a portable thing — it is a thing in a local daemon, a squashfs file, or a
registry, depending on who consumes it. Choosing after the download means
downloading twice.

The choice is never derivable. It is a fact about the destination's
infrastructure that appears nowhere in the bank, so it is asked, and only the
platforms that survived the eligibility screen are offered. Multi-select is
normal: an operator often does not know whether the customer will land on one
GPU box or their cluster.

## Docker

The image is exported from a daemon that has pulled it, and loaded on the far
side. Resolve the URI rather than typing it, so the pin stays in `versions.yaml`:

```bash
IMAGE=$("${TAO_SKILL_BANK_PATH:?}/scripts/resolve_versions_key.py" <the skill's versions key>)
docker pull --platform linux/amd64 "$IMAGE"
docker save "$IMAGE" -o "$BUNDLE/payload/docker/<name>.tar"
gzip -f "$BUNDLE/payload/docker/<name>.tar"
gunzip -c "$BUNDLE/payload/docker/<name>.tar.gz" | tar -tf - >/dev/null
```

**Do not pipe `docker save` into `gzip`.** A pipeline reports only its last
command, so a failed export exits 0 and leaves a valid 20-byte gzip that
decompresses to nothing — which the manifest then records and `sha256sum -c`
certifies. The failure surfaces as `docker load` producing no image at a
customer with no network. `-o` makes the export's own exit status the one that
counts, and the third line is the docker equivalent of the squashfs magic-number
check the SLURM section requires.

At the destination the image is loaded and its name must match byte for byte —
the tooling resolves images by full URI, so a re-tag or an abbreviation makes a
run fail to find an image that is sitting on the disk:

```bash
gunzip -c "$BUNDLE/payload/docker/<name>.tar.gz" | docker load
docker images --format '{{.Repository}}:{{.Tag}}' | grep -Fx "$IMAGE"
```

Every run at the destination adds `--pull=never` so a missing image fails
immediately instead of reaching for a registry.

**Pre-create the runtime cache directories the platform skill documents.** The
docker platform redirects HuggingFace, torch, triton and matplotlib caches under
the results tree; a read-only or absent cache directory surfaces as an
unrelated-looking import error on a host with no network.

## SLURM

**A docker export is the wrong artifact here.** SLURM runs containers through
Pyxis and Enroot, which consume a squashfs file, and the platform skill converts
with `enroot import` and reuses an existing file when one is present. Shipping a
docker tarball to a SLURM site asks the customer to find a docker daemon — on a
login node they may not control — to produce a file that could have been
produced on the packaging host.

```bash
enroot import -o "$BUNDLE/payload/slurm/<name>.sqsh" "docker://<registry>#<image>:<tag>"
```

Convert off the GPU allocation, as the platform skill requires. A failed
conversion must not fall back to the registry image; at the destination there is
no registry to fall back to. A truncated file is detectable — the format carries
a magic number — and checking it is cheaper than a failed allocation.

**At the destination the run references the staged file by absolute path** on
the cluster's shared filesystem, and never a registry URI. The container-image
argument accepts both forms — a path, and two spellings of a registry reference
— which is the hazard: a registry reference left in a job script is accepted at
submission and fails inside the allocation, after the queue wait.

**The credential path has no offline form and must be neutralised.** Pyxis and
Enroot read a credentials file rather than the job environment, and with no
network there is nothing to authenticate to. Say so in the generated bundle
explicitly: a preflight that asks for a credential the run cannot use is how an
operator concludes a good bundle is broken.

## Kubernetes

The kubelet pulls, so the image must already be reachable from inside the
cluster.

**The packaging host stages exactly one thing: the image tar.** It never pushes
to a registry and never touches a node, because the cluster is on the far side
of the gap and unreachable from here by definition. Getting the image from the
tar into the cluster happens at the destination, by the customer, after the
delivery arrives — the bundle carries the command, not the outcome.

Two shapes for that destination-side step, and which applies is a fact about the
customer's infrastructure rather than a choice this skill makes:

| Shape | When | What the bundle carries | Who runs it |
|---|---|---|---|
| cluster-local registry | a registry already runs inside the enclave | the image tar, the push command, and the in-cluster reference the manifest should use | the customer, at the destination |
| per-node preload | no registry; the nodes are reachable | the image tar and the per-node import command | the customer, on each node |

Ask which during intake and record the answer, because it changes the image
reference written into the generated manifest — and that reference is baked into
the bundle here, where it cannot be corrected later without a network.

The platform skill declines to create pull secrets on the operator's behalf, and
neither job template sets an image pull policy — so a bundle that assumes a
registry where there is none produces a pod stuck pulling, with a message that
names the registry and not the mistake.

**The credential path has no offline form here either.** A pull secret exists to
authenticate to a registry outside the cluster, and an enclave has none: an
in-cluster registry is reached without one, and a preloaded node pulls nothing
at all. The generated manifest therefore carries no pull secret, and the
instructions say why — otherwise the first thing a cluster operator does is
create a credential the run will never use, and conclude the bundle is
incomplete when it goes unused.

On the packaging host, the image is exported exactly as for docker. When both
platforms are selected the same tar serves both — stage it once and point the
kubernetes instructions at `payload/docker/`; when kubernetes is the only target,
write it to its own directory:

```bash
docker save "$IMAGE" -o "$BUNDLE/payload/kubernetes/<name>.tar"
gzip -f "$BUNDLE/payload/kubernetes/<name>.tar"
gunzip -c "$BUNDLE/payload/kubernetes/<name>.tar.gz" | tar -tf - >/dev/null
```

The two destination-side commands the table promises, **both run by the customer
after the delivery arrives** — put whichever applies into the generated bundle:

```bash
# Shape A - a registry already runs inside the enclave.
gunzip -c <bundle>/payload/kubernetes/<name>.tar.gz | docker load
docker tag "$IMAGE" <in-cluster-registry>/<name>:<tag>
docker push <in-cluster-registry>/<name>:<tag>
# then the manifest's image: field must name <in-cluster-registry>/<name>:<tag>

# Shape B - no registry; import straight into each node's containerd namespace.
gunzip -c <bundle>/payload/kubernetes/<name>.tar.gz \
  | sudo ctr --namespace k8s.io images import -
# then the manifest keeps the original URI and the kubelet finds it locally
```

Shape B must run on **every** node the job can be scheduled to; a node that
missed it produces the same stuck-pulling pod as a missing registry. Confirm the
node runtime is containerd before using `ctr` — a cluster on a different runtime
needs its own import command, and guessing one is worse than asking.

A bound persistent volume is the air-gapped data answer the platform skill
already documents; the bundle stages into it rather than inventing a layout.

## Virtualenv

No container at all. The wheel closure replaces it, and it is the one payload
with a failure mode that does not appear until install time at the destination.

**Resolve to wheels only, for the destination's exact interpreter and platform
tag.** A dependency that publishes only a source distribution needs a compiler
and an index at install time, and will fail on a host that has neither. Pin the
target rather than inheriting the packaging host's:

```bash
python3 -m pip download --only-binary=:all: \
  --python-version <destination version> --platform <destination tag> \
  --dest "$BUNDLE/wheels" -r <the resolved pin list>
```

If any dependency has no wheel for the target tag, **stop and name the package**
rather than shipping a set that installs on the packaging host and nowhere else.
The pin list is generated into the bundle at packaging time; it is not a file
this skill carries in the repository.

**Wheels live at `$BUNDLE/wheels/`, not under `payload/`**, even though the
virtualenv path is the one that cannot run without them. They are tagged for the
destination interpreter rather than for a platform, and a second copy under a
platform directory is how a destination ends up pointed at the wrong one.

At the destination the install is index-free:

```bash
python3 -m pip install --no-index --find-links "$BUNDLE/wheels" <the pins>
```

## The shared payload: identical on every platform

Images differ per platform. These do not — stage them once, and they serve every
selected platform from the same directory.

### Model weights

Two staging forms, and choosing wrong fails at the destination rather than here.
Decide by what the consuming spec field takes: a **path** means stage to an
explicit directory; a **repository id** resolved by the code itself means stage a
cache and point the runtime at it.

```bash
# explicit-path: the spec field names a file or folder
hf download <repo id> --local-dir "$BUNDLE/weights/<name>"

# hf-cache: the code resolves a repo id itself at run time
HF_HOME="$BUNDLE/weights/hf" hf download <repo id>
```

**`hf` is the current command; `huggingface-cli` was renamed and is gone from
recent releases.** It is often not installed, and on a managed Python a
user-level install is refused. A throwaway virtual environment is enough, and it
is not part of the bundle:

```bash
python3 -m venv /tmp/hfdl && /tmp/hfdl/bin/pip install -q huggingface_hub
/tmp/hfdl/bin/hf download <repo id> --local-dir "$BUNDLE/weights/<name>"
```

For an ungated repository nothing more than HTTP is needed, which avoids the
tool question entirely:

```bash
curl -sSL --fail -o "$BUNDLE/weights/<name>/<file>" \
  "https://huggingface.co/<repo id>/resolve/main/<file>"
```

**Stage the directory, not the checkpoint.** Tokenizer files, `config.json`,
preprocessor configs and index files travel with the weights; a lone tensor file
loads on the packaging host and fails at the destination.

**A gated repository stops the run.** Report which one, and what approval the
operator has to obtain — do not silently ship a bundle missing it.

### The skill's own code

The destination agent reads the packaged skill's tree, so it travels verbatim:

```bash
mkdir -p "$BUNDLE/skills/<layer>"
cp -R "${TAO_SKILL_BANK_PATH:?}/skills/<layer>/<skill name>" "$BUNDLE/skills/<layer>/"
```

Copy every underlying skill an orchestrating workload runs, at the same paths
the bank uses, so relative references inside those files keep resolving.

### Specs

```bash
cp "${TAO_SKILL_BANK_PATH:?}/skills/<layer>/<skill>/references/spec_template_<action>.yaml" \
   "$BUNDLE/specs/"
```

Byte-for-byte. Everything that varies per delivery is a command-line override at
the destination, never an edit here.

### Wheels are not only a virtualenv concern

The wheel section above stages an interpreter's whole closure because a
virtualenv delivery has no container. **A container delivery can still need
wheels**: a workload whose skill installs a Python package at run time cannot do
that offline. AutoML is the case in this bank — it needs `nvidia-tao-automl`,
which pulls its own dependency chain.

Resolve those pins from `versions.yaml` like any other, and stage them into
`$BUNDLE/wheels/` with the same wheels-only rule. If the selection includes a
workload that installs anything at run time, its wheels are part of the bundle
whatever platform it targets.

## One bundle, several payloads

Packaging one selection for several platforms produces **one** bundle. Weights,
code and specs are shared and are usually the bulk; only the execution payload
differs, so only it is per-platform:

```
$BUNDLE/payload/<platform>/
```

The generated bundle instructions resolve which payload applies at the
destination. Duplicating the shared classes per platform would double a bundle
for no benefit.

## Keeping the payload small

- **Export every image in one command, not one command per image.** Images
  sharing a base layer share it within a single export and duplicate it across
  separate ones. On a multi-image selection this is the difference between one
  base layer and three.
- **Deduplicate by digest, not by tag.** Two skills in a selection commonly
  resolve to the same image.
- **Say the multi-platform cost at selection time**, while the operator can
  still drop a platform — the same image in two forms is two copies.
- **Do not compress an archive of compressed things.** The image exports are
  already gzipped and weights barely compress; a second pass costs minutes and
  saves almost nothing.
- **Ship a delta when the customer already has the bulk.** A site that took a
  delivery last quarter does not need the same image again. Compare against the
  manifest of the previous bundle, stage only what differs, and have the
  generated instructions state plainly what the delta assumes is already on
  disk — a delta that does not name its baseline is unusable the moment anyone
  forgets which delivery it followed.

## No-pull enforcement is not uniform, and the bundle must say so

Docker has a per-run flag that makes a missing image fail rather than reach for
a registry. **The other three have no equivalent that anything in this bank
enforces**, and their own air-gap notes defer it — a selected platform is
expected to apply the equivalent policy, and nothing checks that it did.

Treat that as the bundle's problem to state, not to silently assume:

| Platform | What prevents a reach-out | What the bundle must say |
|---|---|---|
| docker | the no-pull flag, per run | use it on every documented command |
| kubernetes | nothing; the kubelet pulls on its own | the image must already be in-cluster, and the manifest must not carry a pull policy that re-fetches |
| slurm | nothing; the run references a file | reference the staged squashfs file by path, never a registry URI |
| virtualenv | nothing; the installer reaches an index | install index-free, from the staged wheel directory only |

A destination that follows the generated instructions must not be able to reach
a registry by accident. Where the platform offers no flag to guarantee that, the
instruction is written so the reach-out has nowhere to go — a file path rather
than a registry reference, an index-free install rather than a pinned one.

## Cutting the network for a verification run

The optional verification phase runs the workload with no route out. **Only docker has a per-run flag
for this**, and nothing in this bank uses one anywhere, so there is no in-tree
example to copy — which is exactly why it is written down here.

| Platform | How the network is cut | Guaranteed? |
|---|---|---|
| docker | `--network=none` on the run | yes — the container gets no interface at all |
| kubernetes | a deny-all `NetworkPolicy` on the namespace, or a namespace already isolated by the cluster | only if a network plugin enforces policy; confirm, do not assume |
| slurm | whatever the site provides; Pyxis has no per-job equivalent | **no** |
| virtualenv | no container to isolate; the process inherits the host's network | **no** — `--no-index` covers the installer only |

**Isolate at the container, not by unplugging the host.** A container with no
interface fails immediately and loudly on anything reaching outward, rather than
hanging on a DNS timeout — and it does not disconnect the person running the
test.

**Where the column says no, say so in the record rather than implying the run
was sealed.** A slurm or virtualenv run on a networked host can silently succeed
by fetching something, which is the precise failure the phase exists to catch.
Either isolate the host by other means and note how, or record the path as
exercised-but-not-sealed — the same discipline the no-pull table above applies.
