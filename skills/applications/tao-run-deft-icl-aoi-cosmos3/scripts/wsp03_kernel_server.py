# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private file-mailbox server for a persistent CPU numeric subprocess."""
import argparse
import json
import os
from pathlib import Path
import socket
import time
from wsp03_io import json_write as private_json_write


def json_write(path,value):
    # This mailbox is an explicit UID boundary. Journals remain private0600.
    return private_json_write(path,value,mode=0o640)
from wsp03_kernel import Kernel,TensorStore
from wsp03_ops import Operations


def isolation_probe():
    import torch
    checks={'gpu_unavailable':not torch.cuda.is_available(),'docker_socket_absent':not Path('/var/run/docker.sock').exists(),
        'host_canary_denied':bool(os.environ.get('WSP_HOST_CANARY')) and not os.path.exists(os.environ['WSP_HOST_CANARY']),'credentials_absent':not any(k in os.environ for k in ('OPENAI_API_KEY','NVIDIA_API_KEY','AWS_SECRET_ACCESS_KEY'))}
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
    p.add_argument('--access-policy');p.add_argument('--editing-policy');p.add_argument('--captures')
    a=p.parse_args();store=TensorStore(a.base,a.output,capture_root=a.captures);operations=Operations(store,json.loads(Path(a.editing_policy).read_text()) if a.editing_policy else None);kernel=Kernel(store,operations=operations);exchange=Path(a.exchange)
    proof=isolation_probe()
    if a.access_policy:
        from wsp03_access_probe import probe
        access=probe(json.loads(Path(a.access_policy).read_text()));json_write(exchange/'path-denial.json',access)
        proof['checks']['path_denial']=access['passed'];proof['passed']=all(proof['checks'].values())
    json_write(exchange/'isolation.json',proof)
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
                elif request['op']=='atomic':
                    operations.namespace=kernel.namespace
                    value=operations.call(request['operation'],request['parameters'])
                elif request['op']=='evidence':
                    mapping=request['lineage']
                    if not isinstance(mapping,dict) or any(not isinstance(k,str) or not isinstance(v,list) or not v or any(not isinstance(h,str) or len(h)!=64 for h in v) for k,v in mapping.items()):raise ValueError('trusted evidence identity map required')
                    operations.trusted_lineage=mapping;value={'bound_reads':len(mapping)}
                elif request['op']=='limits':
                    amount=request['storage_bytes'];seconds=request['cell_seconds']
                    if type(amount)is not int or not 0<amount<=250*1024**3 or type(seconds)is not int or not 1<=seconds<=600:raise ValueError('invalid trusted kernel limits')
                    store.storage_limit=amount;kernel.cell_seconds=seconds;value={'storage_bytes':amount,'cell_seconds':seconds}
                elif request['op']=='status':value={'executions':kernel.executions,'isolation':proof}
                else:raise ValueError('unknown kernel operation')
                result={'ok':True,'result':value}
            except BaseException as error:result={'ok':False,'error_type':type(error).__name__,'message':str(error)}
            json_write(response,result)
        time.sleep(.1)

if __name__=='__main__':main()
