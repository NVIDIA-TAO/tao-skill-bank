# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private file-mailbox server for a persistent CPU numeric subprocess."""
import argparse
import json
import os
from pathlib import Path
import socket
import time
from wsp03_io import json_write
from wsp03_kernel import Kernel,TensorStore
from wsp03_ops import Operations


def isolation_probe():
    import torch
    checks={'gpu_unavailable':not torch.cuda.is_available(),'docker_socket_absent':not Path('/var/run/docker.sock').exists(),
        'host_canary_denied':bool(os.environ.get('WSP_HOST_CANARY')) and not Path(os.environ['WSP_HOST_CANARY']).exists(),'credentials_absent':not any(k in os.environ for k in ('OPENAI_API_KEY','NVIDIA_API_KEY','AWS_SECRET_ACCESS_KEY'))}
    try:
        socket.create_connection(('1.1.1.1',443),timeout=.3).close();checks['network_denied']=False
    except OSError:checks['network_denied']=True
    try:
        Path('/wsp-isolation-test').write_text('forbidden');checks['root_readonly']=False
    except OSError:checks['root_readonly']=True
    return {'checks':checks,'passed':all(checks.values()),'pid':os.getpid(),'uid':os.getuid(),'network_namespace':os.readlink('/proc/self/ns/net')}


def main():
    p=argparse.ArgumentParser()
    for n in ('base','output','exchange'):p.add_argument('--'+n,required=True)
    a=p.parse_args();store=TensorStore(a.base,a.output);kernel=Kernel(store);operations=Operations(store);exchange=Path(a.exchange)
    proof=isolation_probe();json_write(exchange/'isolation.json',proof)
    if not proof['passed']:raise RuntimeError('CPU process isolation proof failed')
    while True:
        for path in sorted(exchange.glob('*.request.json')):
            response=path.with_name(path.name.replace('.request.json','.response.json'))
            started=path.with_name(path.name.replace('.request.json','.started.json'))
            if response.exists():continue
            if started.exists():raise RuntimeError('unresolved kernel request; no implicit replay after restart')
            json_write(started,{'time':time.time(),'pid':os.getpid()})
            try:
                request=json.loads(path.read_text())
                if request['op']=='python':value=kernel.execute(request['code'])
                elif request['op']=='verify':value=store.verify(request['candidate'])
                elif request['op']=='atomic':value=operations.call(request['operation'],request['parameters'])
                elif request['op']=='status':value={'executions':kernel.executions,'isolation':proof}
                else:raise ValueError('unknown kernel operation')
                result={'ok':True,'result':value}
            except BaseException as error:result={'ok':False,'error_type':type(error).__name__,'message':str(error)}
            json_write(response,result)
        time.sleep(.1)

if __name__=='__main__':main()
