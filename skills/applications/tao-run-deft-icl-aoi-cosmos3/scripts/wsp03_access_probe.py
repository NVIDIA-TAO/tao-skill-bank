# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-denial probe with hashed path receipts; never reads protected contents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket


def probe(policy):
    checks=[]
    for path in policy['deny_paths']:
        # Use access/open-directory only: no protected file contents are read.
        accessible=os.path.exists(path) or os.access(path,os.R_OK)
        checks.append({'path_sha256':hashlib.sha256(path.encode()).hexdigest(),'read_denied':not accessible})
    canary=policy['host_canary']
    for path in (canary,'/proc/1/root'+canary):
        checks.append({'path_sha256':hashlib.sha256(path.encode()).hexdigest(),'read_denied':not(os.path.exists(path) or os.access(path,os.R_OK))})
    isolation={'socket_denied':not Path('/var/run/docker.sock').exists(),'gpu_devices_absent':not Path('/dev/nvidia0').exists(),
        'provider_credentials_absent':not any(k in os.environ for k in ('OPENAI_API_KEY','NVIDIA_API_KEY','AWS_SECRET_ACCESS_KEY'))}
    try:socket.create_connection(('1.1.1.1',443),timeout=.3).close();isolation['direct_network_denied']=False
    except OSError:isolation['direct_network_denied']=True
    return {'schema':'wsp03-path-denial/1','path_checks':checks,'isolation':isolation,
        'passed':all(r['read_denied'] for r in checks) and all(isolation.values()),'pid':os.getpid(),'uid':os.getuid(),
        'host_contents_read':False,'policy_sha256':hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',required=True);a=p.parse_args()
    result=probe(json.loads(Path(a.policy).read_text()));print(json.dumps(result,sort_keys=True));raise SystemExit(0 if result['passed'] else 1)
