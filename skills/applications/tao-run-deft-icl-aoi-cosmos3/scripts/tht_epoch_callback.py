# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Framework callback: checkpoint handoff and bounded epoch decision wait.

The trusted supervisor grades Proxy in a separate process. This callback sees
only continue/stop, so no evaluator gold enters the training/inference runtime.
The model, optimizer, scheduler and native training loop remain unchanged.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from cosmos_framework.utils.callback import Callback

class ProxyEpochCallback(Callback):
    def _record(self,value):
        path=Path(self.config.job.path_local)/'training_telemetry.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(value,sort_keys=True,default=lambda x:x.item() if hasattr(x,'numel') and x.numel()==1 else str(x))+'\n');stream.flush();os.fsync(stream.fileno())

    def on_train_start(self,model,iteration=0):
        import torch
        torch.cuda.reset_peak_memory_stats()
        self._record({'event':'resume_boundary','iteration':iteration,'time':time.time()})

    def on_training_step_batch_start(self,model,data_batch,iteration=0):
        import torch
        labels=data_batch.get('labels')
        self.supervised_tokens=int((labels!=-100).sum().item()) if labels is not None else None
        torch.cuda.synchronize();self.batch_started=time.time()

    def on_training_step_batch_end(self,model,data_batch,output_batch,loss,iteration=0):
        import torch
        torch.cuda.synchronize()
        ids=data_batch.get('input_ids');labels=data_batch.get('labels')
        self._record({'event':'batch_complete','iteration':iteration,'started_at_unix':self.batch_started,
                      'ended_at_unix':time.time(),'batch_seconds':time.time()-self.batch_started,
                      'examples':int(ids.shape[0]) if ids is not None else None,
                      'input_tokens':int(ids.numel()) if ids is not None else None,
                      'supervised_tokens':self.supervised_tokens,
                      'loss':float(loss.item()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                      'peak_reserved_bytes':torch.cuda.max_memory_reserved()})

    def on_before_zero_grad(self,model,optimizer,scheduler,iteration=0):
        self._record({'event':'optimizer_state','iteration':iteration,
                      'learning_rates':scheduler.get_last_lr(),
                      'scheduler_state':scheduler.state_dict()})

    def on_save_checkpoint_success(self, iteration=0, elapsed_time=0):
        control=os.environ.get('TAO_EPOCH_CONTROL_DIR')
        if not control: return
        root=Path(control)
        root.mkdir(parents=True,exist_ok=True)
        offset=int(os.environ.get('TAO_EPOCH_ITERATION_OFFSET','0'))
        steps=int(os.environ['TAO_EPOCH_STEPS'])
        local=iteration-offset
        if local<=0 or local%steps: return
        epoch=local//steps
        checkpoints=Path(self.config.job.path_local)/'checkpoints'
        name=(checkpoints/'latest_checkpoint.txt').read_bytes().split(b'\0',1)[0].decode().strip()
        checkpoint=checkpoints/name
        if not any(p.is_file() for p in (checkpoint/'.metadata', checkpoint/'model/.metadata')):
            raise RuntimeError('native DCP completion metadata missing')
        ready={'epoch':epoch,'iteration':iteration,'checkpoint':str(checkpoint),'job_id':os.environ.get('TAO_JOB_ID'),
               'framework_config':str(Path(self.config.job.path_local)/'config.yaml'),
               'checkpoint_elapsed_seconds':elapsed_time,'ready_at_unix':time.time()}
        target=root/f'epoch-{epoch:03d}.ready.json'
        if target.exists() and json.loads(target.read_text())['checkpoint']!=str(checkpoint):
            raise RuntimeError('checkpoint identity changed on resume')
        temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(ready,sort_keys=True)+'\n')
        os.replace(temporary,target)
        decision=root/f'epoch-{epoch:03d}.decision.json'
        deadline=time.monotonic()+float(os.environ.get('TAO_EPOCH_DECISION_TIMEOUT','7200'))
        while not decision.exists():
            if time.monotonic()>=deadline:
                raise TimeoutError('Proxy epoch supervisor did not return within frozen timeout')
            time.sleep(1)
        result=json.loads(decision.read_text())
        if result.get('checkpoint')!=str(checkpoint) or result.get('epoch')!=epoch or result.get('action') not in ('continue','stop'):
            raise RuntimeError('invalid or stale epoch decision')
        if result['action']=='stop':
            # Native loop checks this public trainer limit before the next update.
            self.trainer.max_iterations=iteration
