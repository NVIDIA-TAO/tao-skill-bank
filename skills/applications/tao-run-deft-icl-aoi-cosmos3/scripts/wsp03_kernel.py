# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent CPU-only numeric editing kernel. Run inside an isolated kernel container.

This interpreter is a restricted capability surface, NOT an OS sandbox. The
launcher must provide read-only allowlisted mounts, no GPU/network/host access.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

from wsp03_io import json_write
from wsp03_resources import cpu_memory, append_event


def digest_tensor(tensor):
    value = tensor.detach().contiguous().cpu()
    header = json.dumps({'shape': list(value.shape), 'dtype': str(value.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + value.view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def protected_name(name):
    parts = name.lower().split('.')
    if any('embed' in part or part in ('lm_head', 'output', 'output_layer') for part in parts):
        return True
    # Intermediate block norms are allowed; final norms outside blocks are not.
    return any(part in ('norm', 'final_layernorm', 'final_norm', 'ln_f') for part in parts) and not any(part in ('layers', 'blocks') for part in parts)


def whole_tensor_scaling(base, edited):
    """Reject any common scalar consistent with BF16 rounding intervals.

    Checking the final tensor against immutable base catches split slice writes.
    Interval intersection is conservative at exact half-way rounding ties.
    """
    import torch
    a, b = base.flatten(), edited.flatten()
    low, high = -float('inf'), float('inf')
    nonzero = False
    for offset in range(0, a.numel(), 1000000):
        x, y = a[offset:offset+1000000], b[offset:offset+1000000]
        zero = x == 0
        if bool((y[zero] != 0).any()):
            return False
        x, y = x[~zero].double(), y[~zero]
        if not x.numel():
            continue
        nonzero = True
        prev = torch.nextafter(y, torch.full_like(y, -float('inf'))).double()
        nxt = torch.nextafter(y, torch.full_like(y, float('inf'))).double()
        left, right = (prev+y.double())/2/x, (nxt+y.double())/2/x
        low = max(low, torch.minimum(left, right).max().item())
        high = min(high, torch.maximum(left, right).min().item())
        if low > high:
            return False
    return nonzero and low <= high


class TensorStore:
    def __init__(self, base, output, *, max_tensors=32, max_elements=2000000000, storage_limit=250*1024**3, max_candidate_attempts=98, capture_root=None):
        from safetensors import safe_open
        self.base, self.output = Path(base).resolve(strict=True), Path(output).resolve(strict=False)
        if self.output.is_relative_to(self.base):
            raise ValueError('edits must be outside immutable base')
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_tensors, self.max_elements, self.storage_limit = max_tensors, max_elements, storage_limit
        self.capture_root=Path(capture_root).resolve(strict=True) if capture_root else self.output/"reads"
        self.staged = {}
        self.max_candidate_attempts=max_candidate_attempts
        self.attempts_file=self.output/'candidate-attempts.jsonl'
        index = self.base / 'model.safetensors.index.json'
        if index.exists():
            self.index = json.loads(index.read_text())['weight_map']
        else:
            files = list(self.base.glob('*.safetensors'))
            if len(files) != 1:
                raise ValueError('indexed shards or single safetensors file required')
            with safe_open(files[0], framework='pt', device='cpu') as f:
                self.index = {name: files[0].name for name in f.keys()}
        for path in self.index.values():
            full = (self.base / path).resolve(strict=True)
            if full.parent != self.base or full.suffix != '.safetensors':
                raise ValueError('unsafe shard path')

    def load_tensor(self, name, candidate='base'):
        from safetensors import safe_open
        if name not in self.index:
            raise ValueError('unknown tensor; new parameters forbidden')
        if candidate != 'base':
            manifest = self.manifest(candidate)
            item = next((r for r in manifest['tensors'] if r['name'] == name), None)
            if item:
                path = (self.output / candidate / item['file']).resolve(strict=True)
                if path.parent != (self.output / candidate).resolve(strict=True):
                    raise ValueError('unsafe candidate tensor path')
                with safe_open(path, framework='pt', device='cpu') as f:
                    value = f.get_tensor('value').clone()
                if digest_tensor(value) != item['after_sha256']:
                    raise ValueError('candidate tensor changed')
                return value
        with safe_open(self.base / self.index[name], framework='pt', device='cpu') as f:
            value = f.get_tensor(name).clone()
        append_event(self.output / 'reads.jsonl', {'kind': 'tensor_value_read', 'name': name, 'candidate': candidate,
                     'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': digest_tensor(value), 'time': time.time()})
        return value

    def load_capture(self, request_id, record_id, key):
        from safetensors import safe_open
        if not re.fullmatch(r'[a-f0-9]{32}',request_id):raise ValueError('invalid trusted read identity')
        directory=(self.capture_root/request_id).resolve(strict=True)
        if not directory.is_relative_to(self.capture_root):raise ValueError('capture path escape')
        manifest=json.loads((directory/'manifest.json').read_text());item=manifest['records'][record_id]
        path=(directory/item['file']).resolve(strict=True)
        if path.parent!=directory:raise ValueError('capture artifact path escape')
        from wsp03_bindings import file_hash
        if file_hash(path)!=item['sha256']:raise ValueError('capture artifact changed')
        with safe_open(path,framework='pt',device='cpu') as file:value=file.get_tensor(key).clone()
        append_event(self.output/'reads.jsonl',{'kind':'activation_value_read','request_id':request_id,'record_id':record_id,'key':key,'shape':list(value.shape),'dtype':str(value.dtype),'time':time.time()})
        return value

    def save_edit(self, name, tensor):
        import torch
        if name not in self.index or protected_name(name):
            raise ValueError('protected or nonexistent parameter')
        base = self.load_tensor(name)
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or tensor.shape != base.shape:
            raise ValueError('CPU tensor with unchanged shape required')
        value = tensor.detach().to(dtype=base.dtype).contiguous().clone()
        if not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite patch')
        changed = int((value != base).sum())
        if not changed:
            raise ValueError('BF16 rounding produced no edit')
        if whole_tensor_scaling(base, value):
            raise ValueError('whole-tensor scaling forbidden')
        projected = dict(self.staged)
        projected[name] = value
        if len(projected) > self.max_tensors:
            raise ValueError('changed tensor limit')
        count = sum(int((v != self.load_tensor(n)).sum()) for n, v in projected.items())
        if count > self.max_elements:
            raise ValueError('changed element limit')
        self.staged = projected
        return {'name': name, 'changed_elements': changed, 'after_sha256': digest_tensor(value)}

    def manifest(self, candidate):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', candidate):
            raise ValueError('unsafe candidate ID')
        path = (self.output / candidate / 'manifest.json').resolve(strict=True)
        if not path.is_relative_to(self.output):
            raise ValueError('candidate path escape')
        return json.loads(path.read_text())

    def reset(self, parent='base'):
        self.edit_configuration=None;self.allowed_family_tensors=[]
        self.staged = {}
        if parent != 'base':
            for item in self.manifest(parent)['tensors']:
                self.staged[item['name']] = self.load_tensor(item['name'], parent)

    def commit_candidate(self, candidate, hypothesis, method='strict_wsp'):
        import torch
        from safetensors.torch import save_file
        attempts=len(self.attempts_file.read_text().splitlines()) if self.attempts_file.exists() else 0
        if attempts>=self.max_candidate_attempts:raise ValueError('candidate attempt budget exhausted')
        append_event(self.attempts_file,{'attempt':attempts+1,'candidate':candidate,'time':time.time(),'hypothesis':hypothesis,'method':method})
        if method=='algebraic_closed_form':method='gradient_free_algebraic'
        if getattr(self,'enforce_family',False):
            if not getattr(self,'edit_configuration',None) or not set(self.staged)<=set(self.allowed_family_tensors):raise ValueError('registered family/configuration required for every commit')
            if getattr(self,'requires_algebraic',False) and method=='strict_wsp':raise ValueError('observed algebraic fitting cannot be labelled strict_wsp')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', candidate) or candidate == 'base':
            raise ValueError('unsafe candidate ID')
        if not hypothesis or method not in ('strict_wsp', 'gradient_free_algebraic', 'mixed') or not self.staged:
            raise ValueError('hypothesis, method and nonzero patch required')
        dst = self.output / candidate
        if dst.exists():
            raise ValueError('candidate immutable; preserve previous attempt')
        needed = sum(v.numel()*v.element_size() for v in self.staged.values()) + 1024*1024
        used = sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())
        if used + needed > self.storage_limit:
            raise ValueError('retained storage budget')
        dst.mkdir(mode=0o750)
        records = []
        for i, (name, value) in enumerate(sorted(self.staged.items())):
            base = self.load_tensor(name)
            if protected_name(name) or whole_tensor_scaling(base, value) or not bool(torch.isfinite(value).all()):
                raise ValueError('candidate validation failed')
            delta = value.double() - base.double()
            changed = int((value != base).sum())
            filename = f'tensor-{i:03d}.safetensors'
            save_file({'value': value}, dst / filename)
            l2, bn = torch.linalg.vector_norm(delta).item(), torch.linalg.vector_norm(base.double()).item()
            records.append({'name': name, 'file': filename, 'shape': list(value.shape), 'dtype': str(value.dtype),
                'elements': value.numel(), 'changed_elements': changed, 'before_sha256': digest_tensor(base),
                'after_sha256': digest_tensor(value), 'delta_l1': delta.abs().sum().item(), 'delta_l2': l2,
                'delta_linf': delta.abs().max().item(), 'relative_delta_l2': l2/bn if bn else None})
        manifest = {'schema': 'wsp03-edit/1', 'id': candidate, 'hypothesis': hypothesis, 'method': method,
                    'valid': True, 'tensors': records, 'changed_elements': sum(x['changed_elements'] for x in records),
                    'changed_tensors': len(records), 'edit_configuration':getattr(self,'edit_configuration',None), 'replay': 'replace named tensors from immutable base; all unnamed tensors unchanged',
                    'source_cells_sha256': hashlib.sha256((self.output / 'cells-started.jsonl').read_bytes()).hexdigest() if (self.output / 'cells-started.jsonl').exists() else None}
        json_write(dst / 'manifest.json', manifest)
        for p in dst.iterdir(): p.chmod(0o440)
        self.staged = {}
        self.edit_configuration=None;self.allowed_family_tensors=[]
        return manifest

    def verify(self, candidate):
        manifest = self.manifest(candidate)
        for item in manifest['tensors']:
            base, edited = self.load_tensor(item['name']), self.load_tensor(item['name'], candidate)
            if digest_tensor(base) != item['before_sha256'] or int((base != edited).sum()) != item['changed_elements']:
                raise ValueError('patch replay lineage mismatch')
        return {'passed': True, 'candidate': candidate, 'changed_elements': manifest['changed_elements']}


class Facade:
    def __init__(self, module, allowed):
        object.__setattr__(self, '_module', module)
        object.__setattr__(self, '_allowed', allowed)
    def __getattr__(self, key):
        allowed = object.__getattribute__(self, '_allowed')
        if key not in allowed:
            raise ValueError('module capability not allowed: ' + key)
        value = getattr(object.__getattribute__(self, '_module'), key)
        return Facade(value, allowed[key]) if isinstance(allowed, dict) and isinstance(allowed[key], (set, dict)) else value


def bounded_range(*args):
    r = range(*args)
    if len(r) > 100000:
        raise ValueError('Python loop bound exceeded')
    return r


class LimitedBuffer(io.StringIO):
    def write(self,value):
        if self.tell()+len(value)>65536:raise ValueError('numeric output exceeds64KiB')
        return super().write(value)


class Kernel:
    def __init__(self, store, *, max_runs=500, cell_seconds=600, operations=None):
        import torch
        import numpy as np
        self.store, self.max_runs, self.cell_seconds = store, max_runs, cell_seconds
        from wsp03_ops import Operations
        self.operations=operations or Operations(store)
        self.executions = 0
        self.namespace = {}
        self.operations.namespace=self.namespace
        names = {'tensor', 'as_tensor', 'zeros', 'zeros_like', 'ones', 'ones_like', 'eye', 'arange', 'stack', 'cat', 'mean',
                 'sum', 'sqrt', 'abs', 'clamp', 'where', 'outer', 'matmul', 'einsum', 'isfinite', 'float32', 'float64', 'bfloat16',
                 'int64', 'sort', 'argsort', 'topk', 'diag', 'sign', 'maximum', 'minimum'}
        torch_allowed = {n: True for n in names}
        torch_allowed['linalg'] = {'svd', 'svdvals', 'qr', 'eigh', 'eigvalsh', 'norm', 'vector_norm', 'solve', 'lstsq', 'pinv'}
        numpy_allowed = {n: True for n in ['array', 'asarray', 'zeros', 'ones', 'eye', 'arange', 'stack', 'concatenate', 'mean', 'sum', 'sqrt', 'abs', 'outer', 'matmul', 'einsum', 'isfinite', 'float32', 'float64', 'diag']}
        numpy_allowed['linalg'] = {'svd', 'qr', 'eigh', 'norm', 'solve', 'lstsq', 'pinv'}
        self.capabilities = {'torch': Facade(torch, torch_allowed), 'np': Facade(np, numpy_allowed),
            'load_tensor': store.load_tensor, 'load_capture':store.load_capture, 'save_edit': store.save_edit, 'commit_candidate': store.commit_candidate,
            'reset_candidate': store.reset, 'register_edit_family':self.operations.family, 'select_edit_configuration':self.operations.config}

    def execute(self, code):
        import torch
        if self.executions >= self.max_runs:
            raise ValueError('Python execution budget')
        tree = ast.parse(code)
        denied = {'open', 'eval', 'exec', 'compile', 'getattr', 'setattr', 'globals', 'locals', 'vars', 'input',
                  'backward', 'autograd', 'optim', 'requires_grad_', 'grad', 'cuda', 'device', 'storage', 'untyped_storage',
                  'ctypes', 'data', 'load', 'save', 'tofile', 'dump', 'write', 'write_bytes', 'write_text', 'from_file', 'as_subclass', 'set_', 'resize_', 'manual_seed', 'random'}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                                 ast.While, ast.With, ast.AsyncWith, ast.Global, ast.Nonlocal)):
                raise ValueError('Python construct outside numeric capability surface')
            if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in denied):
                raise ValueError('forbidden attribute')
            if isinstance(node, ast.Name) and (node.id.startswith('_') or node.id in denied):
                raise ValueError('forbidden name')
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in self.capabilities:
                raise ValueError('cannot replace capability')
        if any(isinstance(n,ast.Attribute) and n.attr in ('lstsq','pinv','solve','inverse','pinverse','cholesky_inverse','triangular_solve','lu_solve','cholesky_solve','solve_triangular','lu_factor','lu','inv') for n in ast.walk(tree)):self.store.requires_algebraic=True
        self.executions += 1
        stdout, stderr = LimitedBuffer(), LimitedBuffer()
        namespace = dict(self.capabilities, **self.namespace)
        namespace['__builtins__'] = {'print': print, 'len': len, 'range': bounded_range, 'enumerate': enumerate, 'zip': zip,
                                    'min': min, 'max': max, 'sum': sum, 'float': float, 'int': int, 'abs': abs, 'list': list, 'tuple': tuple}
        started = time.monotonic()
        event = {'execution': self.executions, 'code': code, 'code_sha256': hashlib.sha256(code.encode()).hexdigest(), 'started_at': time.time()}
        append_event(self.store.output / 'cells-started.jsonl', event)
        old_handler = signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(TimeoutError('Python cell deadline')))
        signal.alarm(self.cell_seconds)
        try:
            with torch.inference_mode(), redirect_stdout(stdout), redirect_stderr(stderr):
                exec(compile(tree, '<numeric-cell>', 'exec'), namespace)
            self.namespace = {k: v for k, v in namespace.items() if k not in self.capabilities and not k.startswith('_')}
            event['status'] = 'complete'
        except BaseException as error:
            event.update(status='error', error_type=type(error).__name__)
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            event.update(stdout=stdout.getvalue()[:65536], stderr=stderr.getvalue()[:65536], wall_seconds=time.monotonic()-started, memory=cpu_memory())
            append_event(self.store.output / 'cells.jsonl', event)
        return event


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    import torch
    if torch.cuda.is_available() or os.environ.get('NVIDIA_VISIBLE_DEVICES', 'void') not in ('', 'void', 'none'):
        raise RuntimeError('numeric kernel must have no GPU')
    store, kernel = TensorStore(a.base, a.output), None
    kernel = Kernel(store)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request['op'] == 'python': result = kernel.execute(request['code'])
            elif request['op'] == 'verify': result = store.verify(request['candidate'])
            elif request['op'] == 'status': result = {'python_executions': kernel.executions, 'memory': cpu_memory()}
            else: raise ValueError('unknown kernel operation')
            print(json.dumps({'ok': True, 'result': result}, allow_nan=False), flush=True)
        except BaseException as error:
            print(json.dumps({'ok': False, 'error_type': type(error).__name__, 'message': str(error)}), flush=True)


if __name__ == '__main__': main()
