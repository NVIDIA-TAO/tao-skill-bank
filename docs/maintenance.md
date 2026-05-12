# Maintenance — bumping container images

All TAO container image tags live in **one file**: [`versions.yaml`](../versions.yaml) at the repo root. RC bumps and image upgrades are a one-line edit there.

Wheel install commands are NOT centrally pinned — each skill's Preflight section declares the explicit `pip install "...[<extra>] @ git+https://..."` direct-URL it needs. See "Bumping an SDK install command" below.

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

## Bumping an SDK install command

`nvidia-tao-sdk` and `nvidia-tao-automl` are not on public PyPI yet, so each skill's Preflight section uses a `pip install "...[<extra>] @ git+https://..."` direct-URL pointing at the GitLab repo. When packages eventually publish to PyPI, replace each skill's Preflight URL form with a versioned form (e.g., `pip install 'nvidia-tao-sdk[lepton]>=0.4.0'`). Use a repo-wide grep:

```bash
grep -rl "git+https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-sdk.git" .
grep -rl "git+https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-automl.git" .
```

Update each match. There is no central manifest to bump.

Why no central pin: a pinned `nvidia-tao-sdk==0.2.0` in `versions.yaml` would resolve to a wheel that doesn't exist on any pip-resolvable index. The direct-URL form in each Preflight is honest about where the package actually lives and works without `--extra-index-url` or auth.

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

Users install the SDK via the `pip` direct-URL form (the wheels aren't on public PyPI yet):

```bash
REPO='git+https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-sdk.git'
pip install "nvidia-tao-sdk @ $REPO"                  # core only
pip install "nvidia-tao-sdk[lepton] @ $REPO"          # + Lepton handler deps
pip install "nvidia-tao-sdk[brev] @ $REPO"            # + Brev handler
pip install "nvidia-tao-sdk[slurm] @ $REPO"           # + SLURM handler
pip install "nvidia-tao-sdk[kubernetes] @ $REPO"      # + Kubernetes handler
pip install "nvidia-tao-sdk[docker] @ $REPO"          # + local Docker handler
pip install "nvidia-tao-sdk[all] @ $REPO"             # all platforms
```

Legacy `tao-sdk` package: still installable as a thin alias that pulls in `nvidia-tao-sdk`. Prints a `DeprecationWarning` on import. Will be removed in a future major release.
