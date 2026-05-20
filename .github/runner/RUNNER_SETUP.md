# Self-Hosted GitHub Runner on a Horde Pod

Step-by-step to register a horde-style k8s pod (Ubuntu 22.04, L40 GPU, no GitLab access) as a self-hosted GitHub runner for [`NVIDIA-TAO/tao-skills-bank`](https://github.com/NVIDIA-TAO/tao-skills-bank), with the skill-eval container preinstalled.

## Architecture

```
GitHub PR → workflow .github/workflows/skill-eval.yml
              ↓ (matrix: skill × {claude,codex})
   GitHub job runner (on horde, label: self-hosted,horde,gpu-l40)
              ↓
   bash bootstrap-host.sh  (idempotent: docker + nvidia-toolkit + dockerd + image pull)
              ↓
   skill-eval container ← reads ~/skill-eval.env, ~/.claude.json, ~/.codex/auth.json
              ↓
   docker.sock → host dockerd → docker run --gpus all (TAO containers)
              ↓
   Artifacts uploaded to PR (actions/upload-artifact + actions/github-script comment)
```

The runner machine never needs GitLab access — everything ships through `nvcr.io/nvstaging/tao/skill-eval:latest`.

## One-time horde-pod provisioning

These are the only manual steps. Anything below is scriptable and idempotent.

### 1. Auth

```bash
ssh horde@horde-dgxc.nvidia.com -p <NN>   # current port

# Anthropic
claude                                     # browser OAuth → ~/.claude.json
# OpenAI Codex
codex login --device-auth                  # device flow → ~/.codex/auth.json

# Verify
ls -la ~/.claude.json ~/.codex/auth.json
```

### 2. Secrets

```bash
# Drop the same ~/skill-eval.env you use for local runs.
# Must contain at minimum: NGC_KEY (for nvcr.io pull),
# S3_PROFILE/S3_ACCESS_KEY/S3_SECRET_KEY/S3_ENDPOINT_URL/S3_REGION (for data),
# ARTIFACTORY_USER/ARTIFACTORY_TOKEN/ARTIFACTORY_PYPI_INDEX (for TAO SDK wheels),
# LEPTON_KEY (for Lepton case), HF_KEY, GITLAB_KEY (if needed), etc.

scp ~/skill-eval.env horde@horde-dgxc.nvidia.com:~/skill-eval.env
ssh horde@... 'chmod 600 ~/skill-eval.env'
```

### 3. bootstrap-host.sh

```bash
scp scripts/bootstrap-host.sh horde@...:~/bootstrap-host.sh
ssh horde@... '
  chmod +x ~/bootstrap-host.sh
  bash ~/bootstrap-host.sh        # installs docker, nvidia-toolkit, dockerd (fuse-overlayfs),
                                  # pulls image, installs ~/.local/bin/skill-eval launcher
'
```

The bootstrap is **idempotent**. The GitHub Actions workflow re-runs it at the top of every job — if the pod restarted (which it does periodically), the runner self-heals.

### 4. GitHub runner agent

Register the runner against the `NVIDIA-TAO/tao-skills-bank` repo:

```bash
# Get a registration token (UI: Settings → Actions → Runners → New self-hosted runner)
# Or via gh CLI from a maintainer's laptop:
gh api -X POST repos/NVIDIA-TAO/tao-skills-bank/actions/runners/registration-token \
  --jq .token

# On horde:
mkdir -p ~/actions-runner && cd ~/actions-runner
RUNNER_VERSION="2.319.1"   # check https://github.com/actions/runner/releases
curl -fsSL -o actions-runner.tar.gz \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
tar xzf actions-runner.tar.gz && rm actions-runner.tar.gz

./config.sh \
  --url https://github.com/NVIDIA-TAO/tao-skills-bank \
  --token <REGISTRATION_TOKEN_FROM_ABOVE> \
  --name horde-l40-$(hostname -s) \
  --labels self-hosted,horde,gpu-l40 \
  --work _work \
  --unattended \
  --replace
```

### 5. Run the runner under supervisord

Horde pods use supervisord as PID 1; there's no systemd. Add a runner-managed conf file:

```bash
sudo tee /etc/supervisor/conf.d/gha-runner.conf <<'CONF'
[program:gha-runner]
command=/home/horde/actions-runner/run.sh
user=horde
directory=/home/horde/actions-runner
autostart=true
autorestart=true
stdout_logfile=/var/log/gha-runner.out
stderr_logfile=/var/log/gha-runner.err
environment=HOME="/home/horde",USER="horde"
CONF
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status gha-runner       # should show RUNNING
```

After this, the runner picks up jobs from GitHub automatically.

> **Caveat**: when the horde pod is rescheduled (SSH port changes), `gha-runner` will restart automatically via supervisord; the GitHub side keeps the same runner registration (it just goes "offline" briefly). No re-registration needed unless the host changes identity.

## Workflow contract

`.github/workflows/skill-eval.yml` (see file) declares:

```yaml
runs-on: [self-hosted, horde, gpu-l40]
```

So **any horde-class runner with the GPU label can pick up the job**. To add a second runner, repeat steps 1–5 on another horde pod with the same labels — load-balancing happens automatically.

## Per-PR flow

1. PR opens against `tao-skills-bank` touching a skill dir.
2. Workflow's `detect` job runs `git diff` to find `eval.config` files affected — same logic as the GitLab template (`skill-eval` vs `plugin-workflow-eval` modes).
3. Matrix expands: `(skill, backend)` pairs, where backend ∈ `claude, codex` (or user-overridden via `workflow_dispatch`).
4. Each matrix shard:
   - Re-runs `bootstrap-host.sh` (idempotent, no-op on warm pod).
   - `docker pull` the latest image.
   - Invokes the launcher: `skill-eval eval /workspace/skill-source/<skill> --backend <backend>`.
   - Uploads `skill-test-outputs/` as an artifact.
   - Posts/updates a PR comment with the `eval_report.md` body, keyed by `<!-- skill-eval <skill> <backend> -->` so re-runs replace the prior comment.
5. PR shows pass/fail per shard.

## GitLab → GitHub mapping

| Concern | GitLab CI | GitHub Actions |
|---|---|---|
| Runner registration | runners with `omnistation` / `skill-test` tag, manually registered | self-hosted runner with `self-hosted,horde,gpu-l40` labels |
| Trigger | `if: $CI_PIPELINE_SOURCE == "merge_request_event"` | `on: pull_request` + `paths:` filters |
| Backend matrix | `parallel: matrix: SKILL_EVAL_BACKENDS: [claude,codex]` | `strategy.matrix.backend: [claude, codex]` |
| skill-eval | cloned per-job at `~/skill-eval-repo`, pre-installed venv | baked into `nvcr.io/nvstaging/tao/skill-eval:latest`, pulled per-job |
| Auth (claude/codex) | persistent on runner (`~/.claude/`, `~/.codex/`) | persistent on horde host (`~/.claude.json`, `~/.codex/auth.json`) |
| MR/PR comment | `gitlab-ci-token` + curl to `/api/v4/projects/.../notes` | `actions/github-script` with `secrets.GITHUB_TOKEN` |
| Artifacts | GitLab `artifacts: paths:` | `actions/upload-artifact@v4` |
| Cancellation | re-pipeline kills old | `concurrency.cancel-in-progress: true` |
| Slow-manual eval | second job with `when: manual` + `SWAP_SLOW_CONFIG=yes` | `workflow_dispatch` with `skill_path` input (or a second workflow file) |

## Smoke test (after registration)

Open a trivial PR that touches `samples/test-hello-world/SKILL.md` (or any path that matches the `paths:` filter). The workflow should fire, run, and post a comment with verdict PASS within ~3 min (smoke skills are fast).

## Tearing down a runner

```bash
sudo supervisorctl stop gha-runner
sudo rm /etc/supervisor/conf.d/gha-runner.conf
sudo supervisorctl reread && sudo supervisorctl update

# In the actions-runner dir:
./config.sh remove --token <REMOVAL_TOKEN_FROM_GITHUB>
```
