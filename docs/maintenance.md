# Maintenance — bumping container images

All TAO container image tags **and** Python wheel pins live in **one file**: [`versions.yaml`](../versions.yaml) at the repo root. RC bumps and upgrades are a one-line edit there for both.

Container images live under `images:`, Python wheels under `wheels:`. Skills resolve both by dotted key via `scripts/resolve_versions_key.py` — there are no hardcoded image tags or `pip install` URLs in skill bodies. See "Bumping an SDK/AutoML wheel" below.

## Bumping a container image

Example: bumping the TAO Toolkit PyTorch image from `6.26.3` to `6.27.0`.

```diff
# versions.yaml
images:
  tao_toolkit:
-   pyt:        nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt
+   pyt:        nvcr.io/nvidia/tao/tao-toolkit:6.27.0-pyt
    cosmos_rl:  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-cosmos-rl
    vila:       nvcr.io/nvidia/tao/tao-toolkit:6.26.3-vila
```

That's it. Every skill referencing `tao_toolkit.pyt` (28 of them today) automatically picks up the new tag at runtime.

### Verify the bump

```bash
./scripts/validate-skills.sh                        # confirms all key references still resolve
python -c "
from tao_sdk.versions import resolve_image
print(resolve_image('tao_toolkit.pyt'))
"   # expect the new tag
```

### Commit + PR

```bash
git add versions.yaml
git commit -m "Bump tao_toolkit.pyt to 6.27.0-pyt"
git push -u origin <your-branch>
```

CI runs `validate-skills.sh` automatically. Merge once green.

## Bumping an SDK/AutoML wheel

`nvidia-tao-sdk` and `nvidia-tao-automl` are on public PyPI and pinned in the `wheels:` section of `versions.yaml`. Bumping is a one-line edit per entry — symmetric with images:

```diff
# versions.yaml
wheels:
-   tao_sdk_lepton:     nvidia-tao-sdk[lepton]==7.0.0
+   tao_sdk_lepton:     nvidia-tao-sdk[lepton]==7.1.0rc1
```

Every skill Preflight resolves its wheel key via `scripts/resolve_versions_key.py wheels.<key>`, so the new pin propagates automatically — no per-skill grep, no hardcoded URLs.

### Internal RC versions

To stage an RC internally before the public release:

1. Publish the RC wheel to the index pip is pointed at — an internal PyPI mirror, or `--extra-index-url` / `--index-url` supplied via pip config or `PIP_*` env. Index selection is an environment concern; the skill bank never bakes in a registry.
2. Pin the **exact** RC version in `versions.yaml` (e.g. `==7.1.0rc1`). pip installs an exact pre-release pin without `--pre`; a non-exact specifier like `>=7.1.0` would skip pre-releases unless `--pre` is passed.

That's the whole change: one line in `versions.yaml`, exactly like a container RC bump.

## Adding a new image

1. Add an entry to `versions.yaml` under the appropriate group:

   ```yaml
   images:
     tao_toolkit:
       my_new_image: nvcr.io/nvidia/tao/tao-toolkit:6.26.3-my-new-image
   ```

2. In the skill's `references/skill_info.yaml`, reference by key:

   ```yaml
   container_image: tao_toolkit.my_new_image
   ```

3. Run the validator — confirms the key resolves.

## When to use absolute paths instead of keys

Both `container_image: tao_toolkit.pyt` (key) and `container_image: nvcr.io/.../tao-toolkit:6.26.3-pyt` (absolute) are valid indefinitely. Use absolute paths when:

- The image is **experimental** and not worth promoting to the manifest.
- The image is **third-party** (non-NVIDIA registry).
- The image is used by **only one skill** and unlikely to need a coordinated bump.

Promote to a key (`versions.yaml` entry) when:

- The image is shared by **two or more skills**.
- The image will be **bumped on a release cadence**.
- You want to track it in changelogs / RC notes.

## Related: Python wheel install matrix

Users install the SDK by resolving the pin from `versions.yaml` (wheels are on public PyPI):

```bash
SB="${TAO_SKILL_BANK_PATH:-~/tao-skills-external}"
pip install "$($SB/scripts/resolve_versions_key.py wheels.tao_sdk)"             # core only
pip install "$($SB/scripts/resolve_versions_key.py wheels.tao_sdk_lepton)"      # + Lepton handler deps
pip install "$($SB/scripts/resolve_versions_key.py wheels.tao_sdk_brev)"        # + Brev handler
pip install "$($SB/scripts/resolve_versions_key.py wheels.tao_sdk_slurm)"       # + SLURM handler
pip install "$($SB/scripts/resolve_versions_key.py wheels.tao_sdk_kubernetes)"  # + Kubernetes handler
pip install "$($SB/scripts/resolve_versions_key.py wheels.tao_sdk_docker)"      # + local Docker handler
pip install "$($SB/scripts/resolve_versions_key.py wheels.tao_sdk_all)"         # all platforms
```

Legacy `tao-sdk` package: still installable as a thin alias that pulls in `nvidia-tao-sdk`. Prints a `DeprecationWarning` on import. Will be removed in a future major release.
