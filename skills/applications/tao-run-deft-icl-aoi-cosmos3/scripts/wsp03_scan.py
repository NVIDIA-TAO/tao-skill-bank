# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only finite-value inventory of actual immutable safetensor payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from wsp03_io import json_write
from wsp03_resources import cpu_memory


def scan(base,output):
    import torch
    from safetensors import safe_open
    if torch.cuda.is_available():raise ValueError('finiteness scan must run in no-GPU CPU container')
    base=Path(base).resolve(strict=True);output=Path(output).resolve(strict=False)
    if output.is_relative_to(base):raise ValueError('report cannot mutate immutable base')
    index=base/'model.safetensors.index.json';mapping=json.loads(index.read_text())['weight_map'] if index.exists() else None
    if mapping is not None and not mapping:raise ValueError('empty tensor index')
    if mapping is not None and set(mapping.values())!={p.name for p in base.glob('*.safetensors')}:raise ValueError('actual shard set differs from index')
    files=sorted(set(mapping.values())) if mapping else [p.name for p in base.glob('*.safetensors')]
    if not files:raise ValueError('no safetensors files')
    started=time.monotonic();rows=[];seen=set()
    for filename in files:
        path=(base/filename).resolve(strict=True)
        if path.parent!=base:raise ValueError('unsafe shard path')
        with safe_open(path,framework='pt',device='cpu') as file:
            for name in file.keys():
                if name in seen or mapping is not None and mapping.get(name)!=filename:raise ValueError('duplicate/index mismatch')
                seen.add(name);value=file.get_tensor(name);flat=value.reshape(-1);invalid=0
                for i in range(0,flat.numel(),1048576):invalid+=int((~torch.isfinite(flat[i:i+1048576])).sum())
                rows.append({'name':name,'shape':list(value.shape),'dtype':str(value.dtype),'elements':value.numel(),'nonfinite_elements':invalid})
    if not seen:raise ValueError('empty safetensors payload')
    if mapping is not None and seen!=set(mapping):raise ValueError('index has missing tensor payloads')
    result={'schema':'wsp03-finiteness/1','passed':all(r['nonfinite_elements']==0 for r in rows),'tensors':rows,
        'tensor_count':len(rows),'total_elements':sum(r['elements'] for r in rows),'nonfinite_elements':sum(r['nonfinite_elements'] for r in rows),
        'tensor_names_sha256':hashlib.sha256(json.dumps(sorted(seen),separators=(',',':')).encode()).hexdigest(),
        'wall_seconds':time.monotonic()-started,'cpu_memory':cpu_memory(),'gpu_used':False}
    json_write(output,result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=scan(args.base,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='tensors'}));raise SystemExit(0 if result['passed'] else 1)
