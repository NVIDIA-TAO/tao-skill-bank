#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Live four-verb smoke for the SDK-free platform skills.
#
# The static suite proves the skills SAY the right thing. This proves a real
# backend ACCEPTS it: submit -> status -> logs -> cancel, end to end, with the
# job record updated at each step. It is the layer that would have caught the
# `brev exec` regression, where every documented verb parsed fine as text and
# was rejected by the actual CLI.
#
# Deliberately does not need a GPU: the verb contract is about job lifecycle,
# not compute. A 5 MB busybox exercises it, so this can run on any runner and
# stay cheap enough for a nightly.
#
# Usage: platform_smoke.sh [targets]      # comma-separated: docker,kubernetes,slurm
#        PLATFORM_SMOKE_TARGETS=docker scripts/ci/platform_smoke.sh
set -uo pipefail

TARGETS="${1:-docker}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECORD="$REPO/scripts/tao_job_record.py"
REPORT="${PLATFORM_SMOKE_REPORT:-$REPO/platform-nightly-report.md}"
SMOKE_IMAGE="${PLATFORM_SMOKE_IMAGE:-busybox:1.36}"
rc=0

note() { printf '%s\n' "$*" | tee -a "$REPORT"; }
fail() { note "- **FAIL** $1 — $2"; rc=1; }
pass() { note "- **PASS** $1 — $2"; }

note "# Platform nightly — live four-verb smoke"
note ""

smoke_docker() {
  local p=docker jid cid state logs
  command -v docker >/dev/null 2>&1 || { note "- **SKIP** $p — docker not installed"; return 0; }
  docker info >/dev/null 2>&1 || { note "- **SKIP** $p — docker daemon unreachable"; return 0; }

  # submit: open the record FIRST so the id names the backend object, exactly
  # as the skill documents (an id minted after launch cannot be recovered).
  jid="$("$RECORD" open --platform docker --image "$SMOKE_IMAGE" \
          --network-arch smoke --action train --storage-tier A \
          --results-dir "${TMPDIR:-/tmp}/tao-smoke-$$" 2>/dev/null)" \
    || { fail "$p/submit" "tao_job_record.py open failed"; return 1; }
  cid="$(docker run -d --name "$jid" --label "tao-job=$jid" \
          "$SMOKE_IMAGE" sh -c 'echo tao-smoke-alive; sleep 30' 2>&1)" \
    || { fail "$p/submit" "docker run rejected: ${cid:0:200}"; return 1; }
  "$RECORD" mark "$jid" --state RUNNING --backend-ref "$cid" >/dev/null 2>&1 \
    || fail "$p/submit" "record mark RUNNING failed"
  pass "$p/submit" "container $jid started"

  # status
  state="$(docker inspect --format '{{.State.Status}}' "$jid" 2>/dev/null)"
  [ -n "$state" ] && pass "$p/status" "state=$state" \
                  || fail "$p/status" "docker inspect returned nothing"

  # logs — the container prints a known token; anything else means the log path
  # is wired to the wrong stream.
  logs="$(docker logs --tail 20 "$jid" 2>&1)"
  case "$logs" in
    *tao-smoke-alive*) pass "$p/logs" "log token observed" ;;
    *) fail "$p/logs" "expected token absent: ${logs:0:200}" ;;
  esac

  # cancel + teardown, then confirm the container is really gone.
  docker rm -f "$jid" >/dev/null 2>&1 \
    || fail "$p/cancel" "docker rm -f failed"
  "$RECORD" mark "$jid" --state CANCELED --source agent >/dev/null 2>&1 \
    || fail "$p/cancel" "record mark CANCELED failed"
  if docker inspect "$jid" >/dev/null 2>&1; then
    fail "$p/cancel" "container still present after rm -f"
  else
    pass "$p/cancel" "container removed and record closed"
  fi
}

smoke_kubernetes() {
  local p=kubernetes jid gpus rendered pod state logs ns
  command -v kubectl >/dev/null 2>&1 || { note "- **SKIP** $p — kubectl not installed"; return 0; }
  kubectl cluster-info >/dev/null 2>&1 || {
    note "- **SKIP** $p — no reachable cluster. On a GPU host: minikube start --driver=docker --gpus all"
    return 0; }
  ns="${PLATFORM_SMOKE_NAMESPACE:-default}"

  # GPU capacity decides what this run can actually prove. With GPUs we assert
  # the pod REACHES RUNNING (real scheduling + container start). Without them a
  # pod requesting nvidia.com/gpu would sit Pending forever, so we render with
  # zero and verify admission + lifecycle only — and say so, rather than
  # reporting a GPU-less pass as if it covered GPU.
  gpus=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null \
         | grep -v '^$' | awk '{s+=$1} END{print s+0}')
  gpus=${gpus:-0}

  jid="$("$RECORD" open --platform kubernetes --image "$SMOKE_IMAGE" \
          --network-arch smoke --action train --storage-tier A \
          --results-dir /tmp/tao-smoke-k8s 2>/dev/null)" \
    || { fail "$p/submit" "tao_job_record.py open failed"; return 1; }

  # A PVC-less Job: the packaged template mounts a claim, and a missing PVC
  # fails scheduling BEFORE any GPU complaint, masking the real signal.
  rendered=$(cat <<YAML
apiVersion: batch/v1
kind: Job
metadata: {name: "$jid", labels: {tao-job: "$jid"}}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 300
  template:
    metadata: {labels: {tao-job: "$jid"}}
    spec:
      restartPolicy: Never
      containers:
        - name: tao
          image: "$SMOKE_IMAGE"
          command: ["sh", "-c", "echo tao-smoke-alive; sleep 20"]
$( [ "$gpus" -gt 0 ] && printf '          resources:\n            limits:\n              "nvidia.com/gpu": "1"\n' )
YAML
)
  if ! printf '%s\n' "$rendered" | kubectl apply -n "$ns" -f - >/dev/null 2>&1; then
    "$RECORD" mark "$jid" --state ERROR --source agent >/dev/null 2>&1
    fail "$p/submit" "kubectl apply rejected the manifest"
    return 1
  fi
  "$RECORD" mark "$jid" --state RUNNING --backend-ref "$ns/$jid" >/dev/null 2>&1
  pass "$p/submit" "job $jid applied (gpu=$gpus)"

  # status — with GPUs, insist the pod actually reaches Running.
  for _ in $(seq 1 30); do
    state=$(kubectl get pods -n "$ns" -l "job-name=$jid" -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
    [ "$state" = Running ] || [ "$state" = Succeeded ] && break
    sleep 2
  done
  if [ "$gpus" -gt 0 ]; then
    case "$state" in
      Running|Succeeded) pass "$p/status" "pod reached $state on a GPU node" ;;
      *) fail "$p/status" "pod stuck in ${state:-unknown} with $gpus GPU(s) allocatable — \
$(kubectl get events -n "$ns" --field-selector reason=FailedScheduling -o jsonpath='{.items[-1].message}' 2>/dev/null | head -c 150)" ;;
    esac
  else
    [ -n "$state" ] && pass "$p/status" "phase=$state (no GPU on cluster — admission/lifecycle only, GPU scheduling UNVERIFIED)" \
                    || fail "$p/status" "no pod found for job $jid"
  fi

  # logs
  logs=$(kubectl logs -n "$ns" -l "job-name=$jid" --tail 20 2>&1)
  case "$logs" in
    *tao-smoke-alive*) pass "$p/logs" "log token observed" ;;
    *) [ "$state" = Running ] || [ "$state" = Succeeded ] \
         && fail "$p/logs" "expected token absent: ${logs:0:150}" \
         || note "  (logs skipped — pod never started)" ;;
  esac

  # cancel + teardown
  kubectl delete job "$jid" -n "$ns" --cascade=foreground >/dev/null 2>&1 \
    || fail "$p/cancel" "kubectl delete job failed"
  "$RECORD" mark "$jid" --state CANCELED --source agent >/dev/null 2>&1
  if kubectl get job "$jid" -n "$ns" >/dev/null 2>&1; then
    fail "$p/cancel" "job still present after delete"
  else
    pass "$p/cancel" "job deleted and record closed"
  fi
}

smoke_slurm() {
  [ -n "${SLURM_LOGIN_HOST:-}" ] || { note "- **SKIP** slurm — SLURM_LOGIN_HOST unset"; return 0; }
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$SLURM_LOGIN_HOST" true >/dev/null 2>&1 \
    || { note "- **SKIP** slurm — login host unreachable"; return 0; }
  note "- **TODO** slurm — four-verb smoke not implemented yet (submit to a short CPU partition)"
}

IFS=',' read -ra want <<<"$TARGETS"
for t in "${want[@]}"; do
  case "$t" in
    docker)     smoke_docker ;;
    kubernetes) smoke_kubernetes ;;
    slurm)      smoke_slurm ;;
    brev)       note "- **SKIP** brev — provisioning costs money; run manually" ;;
    *)          note "- **SKIP** $t — unknown target" ;;
  esac
done

note ""
note "Exit status: $rc"
exit "$rc"
