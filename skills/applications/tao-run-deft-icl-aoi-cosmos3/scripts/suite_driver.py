# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autonomous six-arm suite, using native Framework and role-isolated workers.

No formal work starts without a checksum-bound preflight approval. Stage
attempts, costs, immutable responses and selections survive process restarts.
"""
from __future__ import annotations
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

from tht_contract import external_evaluator,grade,load_jsonl,json_hash,sha256
from suite_store import SuiteStore
from suite_runtime import BANK,Docker,json_write,infrastructure_failure
from suite_selection import epoch_decision,select_icl,select_round
from suite_data import RetrievalToolset,training_view,query_view
from astra_isolation import generic_mounts,isolation_probe,classify
from suite_costs import aggregate
from policy_agent import propose

APP=Path(__file__).resolve().parent
MODEL=BANK/'skills/models/tao-finetune-cosmos-reason/scripts'

class Suite:
    cost_arms = 'ABCDEF'
    def __init__(self,task,approval):
        self.task=task; self.root=Path(task['results_dir']).resolve()
        if self.root.is_relative_to(BANK): raise ValueError('experiment outputs must remain outside the code repository')
        self.store=SuiteStore(self.root)
        self.store.initialize(task,approval)
        self.store.seal('started_at',self.store.get('started_at',time.time()))
        self.deadline=self.store.get('started_at')+task['budgets']['suite_wall_seconds']
        self.public=Path(task['public_package']); self.private=Path(task['private_package'])
        self.prompt=json.loads(Path(task['direct_profile']).read_text())['prompt']
        self.evaluator=external_evaluator(self.public/'evaluator')
        self.mining=load_jsonl(self.public/'mining/samples.jsonl')
        self.proxy=load_jsonl(self.public/'proxy/queries.jsonl')
        self.runtime=Docker(self.root/'jobs',task['framework_image'])
        self.retrieval=RetrievalToolset(task,self.run_job)
        self.budget_scope=None

    def save_selection(self,arm,result):
        name=arm+'/select'
        if self.store.stage(name)['status']=='complete':
            if self.result(name)!=result: raise ValueError('selection changed on resume')
            return
        a=self.store.begin(name);self.store.finish(a,'complete',result)
        path=self.root/'selections'/f'{arm}.json';json_write(path,result);self.store.artifact(name,path)

    def result(self,stage):
        value=self.store.stage(stage)['result']
        return json.loads(value) if value else None

    def run_agent(self,stage,runtime,invoke,*,budget=None):
        """Persist the handle before launch and reconcile before any retry."""
        if self.store.stage(stage)['status']=='complete': return self.result(stage)
        for _ in range(2):
            previous=self.store.db.execute('SELECT * FROM attempts WHERE stage=? ORDER BY id DESC LIMIT 1',(stage,)).fetchone()
            job=None
            if previous and previous['status'] in ('pending','running'):
                attempt=previous['id']
                if previous['job_id']:
                    rec=json.loads(runtime.record('show',previous['job_id']))
                    job={'job_id':previous['job_id'],'results_dir':rec['results_dir'],'container_id':rec['backend_ref']}
                else:
                    self.store.finish(attempt,'infrastructure_error',{'reason':'interrupted before job record'})
                    attempt=self.store.begin(stage)
            else: attempt=self.store.begin(stage)
            value=invoke(job,lambda jid:self.store.attach_job(attempt,jid))
            usage=dict(value['usage'],invocation_id=value['job']['job_id'],category='formal',stage=stage,
                       started_at_unix=value['started_at_unix'],ended_at_unix=value['ended_at_unix'],
                       completed_final=value['completed'],backend_state=value['backend_state']['state'],
                       raw_answer=value.get('raw_answer'),invocation_path=str(Path(value['job']['results_dir'])/'invocation.json'))
            if budget and (usage.get('input_tokens') is None or usage.get('output_tokens') is None):
                self.store.record_usage(value['job']['job_id'],usage)
                self.store.finish(attempt,'program_error',{**value,'reason':'unknown usage prevents budget-safe continuation'})
                raise RuntimeError(stage+': agent usage unavailable; evidence retained, budgeted search incomplete')
            self.store.record_usage(value['job']['job_id'],usage,budget=budget)
            if value['completed']:
                self.store.finish(attempt,'complete',value);return value
            state=value['backend_state']
            reason='infrastructure_error' if infrastructure_failure(state.get('native',{}),runtime.logs(value['job']['job_id'])) else ('budget_exhausted' if state['state']=='CANCELED' or value.get('budget_exhausted') else 'program_error')
            self.store.finish(attempt,reason,value)
            if reason!='infrastructure_error':
                if reason=='budget_exhausted': raise TimeoutError(stage+': no final answer within budget; incomplete and not retryable')
                raise RuntimeError(stage+': no final answer; incomplete and not retryable')
        raise RuntimeError(stage+': infrastructure retry limit exhausted')

    def _native_cost(self,runtime,job,gpu,stage):
        native=runtime.status(job['job_id']).get('native',{})
        def dt(s):
            return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp() if s and not s.startswith('0001') else None
        start,end=dt(native.get('StartedAt')),dt(native.get('FinishedAt'))
        launch=json.loads((Path(job['results_dir'])/'launch.json').read_text())
        submitted=launch.get('submitted_at_unix')
        self.store.record_usage(job['job_id'],{'invocation_id':job['job_id'],'job_id':job['job_id'],
                'stage':stage,'arm':stage.split('/')[0] if stage.split('/')[0] in self.cost_arms else None,
                'phase':'benchmark' if '/benchmark/' in stage else ('proxy' if '/proxy/' in stage else 'preparation'),
                'category':self.task.get('cost_category','formal'),'started_at_unix':start,'ended_at_unix':end,
                'queue_seconds':max(0,start-submitted) if start and submitted else None,
                'gpu_seconds':end-start if gpu and start and end else None,
                'gpu_requested':gpu,'gpu_resource':self.task['environment']['host']+':GPU0' if gpu else None,
                'gpu_time_method':'allocated GPU count times native execution wall time' if gpu else 'not_applicable',
                'backend_state':native.get('Status'),'exit_code':native.get('ExitCode')})

    def run_job(self,stage,command,*,image=None,entrypoint='/workspace/.venv/bin/python',strip_entrypoint=False,
                mounts=(),env=(),required_output=None,gpu=True,heartbeat=None,resume_answers=False,recover_training=None,stage_wall_seconds=None,host_result_mount=False,extra_args=()):
        runtime=Docker(self.root/'jobs',image or self.task['framework_image'])
        stage_cap=stage_wall_seconds or self.task['budgets']['stage_wall_seconds']
        if strip_entrypoint: command=command[1:]
        if self.store.stage(stage)['status']=='complete':
            result=self.result(stage)
            for artifact in result.get('artifacts',[]):
                if sha256(artifact['path'])!=artifact['sha256']: raise ValueError('completed stage artifact changed')
            return result
        for _ in range(2):
            first=self.store.db.execute('SELECT MIN(started) FROM attempts WHERE stage=?',(stage,)).fetchone()[0]
            previous=self.store.db.execute('SELECT * FROM attempts WHERE stage=? ORDER BY id DESC LIMIT 1',(stage,)).fetchone()
            expired = ('suite wall budget exhausted' if time.time()>=self.deadline else
                       'original stage wall budget exhausted' if first is not None and time.time()>=first+stage_cap else
                       'ICL wall budget exhausted' if self.budget_scope and self.store.remaining(self.budget_scope)['wall_seconds']<=0 else None)
            if expired:
                if previous and previous['status'] in ('pending','running'):
                    detail={'reason':expired,'original_started':first,'stage_wall_seconds':stage_cap}
                    if previous['job_id']:
                        job_id=previous['job_id'];record=json.loads(runtime.record('show',job_id))
                        job={'job_id':job_id,'results_dir':record['results_dir'],'container_id':record['backend_ref']}
                        final=runtime.status(job_id)
                        if final['state'] in ('PENDING','RUNNING'):
                            runtime.cancel(job_id);final=runtime.status(job_id)
                        detail.update(job=job,backend=final)
                        try: self._native_cost(runtime,job,gpu,stage)
                        except (OSError,ValueError,KeyError) as error:
                            detail['cost_record_error']=str(error)
                            self.store.event('native_cost_unavailable',{'stage':stage,'job_id':job_id,'error':str(error)})
                    else:
                        detail['launch_interrupted_before_handle']=True
                    self.store.finish(previous['id'],'budget_exhausted',detail)
                raise TimeoutError(expired)
            retry_of=previous['job_id'] if previous else None
            if previous and previous['status'] in ('pending','running') and previous['job_id']:
                attempt=previous['id']; job_id=previous['job_id']
                record=json.loads(runtime.record('show',job_id))
                job={'job_id':job_id,'results_dir':record['results_dir'],'container_id':record['backend_ref']}
            else:
                if previous and previous['status']=='pending' and not previous['job_id']:
                    self.store.finish(previous['id'],'infrastructure_error',{'reason':'orchestrator interrupted before record/launch'})
                attempt=self.store.begin(stage,max_attempts=2)
                launch_env=list(env)
                launch_mounts=list(mounts)
                launch_command=list(command)
                if recover_training and retry_of:
                    launch_command,launch_mounts=recover_training(retry_of,launch_command,launch_mounts)
                if resume_answers and retry_of:
                    prior=Path(json.loads(runtime.record('show',retry_of))['results_dir'])/'raw_answers.jsonl'
                    if prior.is_file():
                        launch_mounts.append((prior,'/previous_raw_answers.jsonl',True))
                        launch_env.append(('TAO_RESUME_ANSWERS','/previous_raw_answers.jsonl'))
                job=runtime.submit(stage.replace('/','-'),launch_command,mounts=launch_mounts,env=launch_env,
                       entrypoint=entrypoint,gpu=gpu,retry_of=retry_of,host_result_mount=host_result_mount,extra_args=extra_args,on_open=lambda jid:self.store.attach_job(attempt,jid))
            first_started=self.store.db.execute('SELECT MIN(started) FROM attempts WHERE stage=?',(stage,)).fetchone()[0]
            stage_deadline=min(self.deadline,first_started+stage_cap)
            if self.budget_scope:
                stage_deadline=min(stage_deadline,time.time()+self.store.remaining(self.budget_scope)['wall_seconds'])
            final=runtime.wait(job['job_id'],stage_deadline,heartbeat=(lambda state:heartbeat(job)) if heartbeat else None)
            if heartbeat and final['state']=='COMPLETE': heartbeat(job)
            self._native_cost(runtime,job,gpu,stage)
            if final['state']=='COMPLETE':
                result={'job':job,'artifacts':[]}
                if required_output:
                    path=Path(required_output)
                    if not path.is_absolute(): path=Path(job['results_dir'])/path
                    if not path.is_file():
                        self.store.finish(attempt,'program_error',{'reason':'completion artifact missing','path':str(path),'job':job})
                        raise RuntimeError('native success without required output')
                    result['artifacts'].append(self.store.artifact(stage+'/output',path))
                self.store.finish(attempt,'complete',result)
                return result
            reason='budget_exhausted' if final['state']=='CANCELED' else ('infrastructure_error' if infrastructure_failure(final.get('native',{}),runtime.logs(job['job_id'])) else 'program_error')
            self.store.finish(attempt,reason,{'job':job,'backend':final})
            if reason=='budget_exhausted': raise TimeoutError(f'{stage}: wall budget exhausted')
            if reason!='infrastructure_error': raise RuntimeError(f'{stage} stopped: {reason}; see job {job["job_id"]}')
        raise RuntimeError(f'{stage}: bounded infrastructure recovery exhausted')

    def prepare_inference_view(self,view,queries,policy,phase):
        if not (view/'requests.json').exists():
            references=self.retrieval.references(phase,queries,policy,self.task['icl_reference_capacity']) if policy else None
            query_view(self.public,queries,self.prompt,view,policy=policy,references=references)

    def write_training_spec(self,path,text):
        if not path.exists(): path.write_text(text)

    def infer(self,arm,phase,candidate,model,*,policy=None,queries=None):
        if phase=='benchmark': self.store.verify_freeze()
        queries=queries or (self.proxy if phase=='proxy' else load_jsonl(self.public/'benchmark/queries.jsonl'))
        stage=f'{arm}/{phase}/{candidate}'
        view=self.root/'views'/arm/phase/candidate
        self.prepare_inference_view(view,queries,policy,phase)
        self.store.artifact(stage+'/requests',view/'requests.json')
        try:
            result=self.run_job(stage,['/worker.py','--model','/model','--requests','/input/requests.json','--output','/results',
                        '--seed',str(self.task['training']['seed'])],
                        mounts=[(APP/'inference_worker.py','/worker.py',True),(model,'/model',True),(view,'/input',True)],
                        required_output='completion.json',resume_answers=True)
        finally:
            # Retain completed answers and failed-call cost even if a native
            # attempt cannot finish the whole role. Raw completion is immutable.
            for attempt in self.store.db.execute('SELECT job_id FROM attempts WHERE stage=? ORDER BY id',(stage,)):
                if not attempt['job_id']: continue
                record=json.loads(self.runtime.record('show',attempt['job_id']))
                directory=Path(record['results_dir']);file=directory/'raw_answers.jsonl'
                for row in load_jsonl(file) if file.exists() else []:
                    self.store.record_answer(arm,phase,candidate,row)
                    usage={k:row.get(k) for k in ('invocation_id','inference_input_tokens','inference_output_tokens','started_at_unix','ended_at_unix')}
                    usage.update(category='formal',arm=arm,phase=phase,candidate=candidate,stage=stage)
                    self.store.record_usage(row['invocation_id'],usage)
                failures=directory/'attempts.jsonl'
                for row in load_jsonl(failures) if failures.exists() else []:
                    if row['status']=='error':
                        self.store.record_usage(row['invocation_id'],dict(row,category='formal',arm=arm,phase=phase,stage=stage))
        raw=self.store.answers(arm,phase,candidate)
        if {r['id'] for r in raw}!={r['id'] for r in queries}: raise ValueError('inference coverage incomplete')
        gold=load_jsonl(self.private/'evaluation'/f'{phase}-gold.jsonl')
        gold=[r for r in gold if r['id'] in {q['id'] for q in queries}]
        scored=grade(raw,gold,self.evaluator)
        path=self.root/'metrics'/arm/phase/(candidate+'.json')
        if not path.exists(): json_write(path,scored)
        self.store.artifact(stage+'/metrics',path)
        return scored

    def export(self,stage,train_job,ready):
        def relative(path): return str(Path(path).relative_to('/results'))
        checkpoint='/training/'+relative(ready['checkpoint'])
        config='/training/'+relative(ready['framework_config'])
        result=self.run_job(stage,['/helpers/framework_checkpoint_action.py','prepare','--action','inference',
                    '--checkpoint-path',checkpoint,'--config-file',config,'--base-model-path-or-uri','/model',
                    '--export-dir','/results/hf','--output','/results/action.json'],
                    mounts=[(MODEL,'/helpers',True),(self.task['model_path'],'/model',True),
                            (train_job['results_dir'],'/training',True)],required_output='action.json')
        action=json.loads((Path(result['job']['results_dir'])/'action.json').read_text())
        if action.get('pre_action_result') not in ('exported','reused'): raise ValueError('native export not verified')
        path=Path(result['job']['results_dir'])/'hf'
        self.store.artifact(stage+'/export',path/'export_manifest.json')
        return str(path)

    def training_population(self, selected_ids, view, round_index):
        if not (view/'train.json').exists(): training_view(self.public,selected_ids,self.prompt,view)
        return len(selected_ids)

    def train(self,arm,selected_ids,epochs,*,round_index=0,previous=None):
        import tomllib,tomli_w
        from render_cfw_sft import render_profile
        stage=f'{arm}/train/{round_index}'
        directory=self.root/'training'/arm/str(round_index);directory.mkdir(parents=True,exist_ok=True)
        view=directory/'input'
        population=self.training_population(selected_ids,view,round_index)
        self.store.artifact(stage+'/training-data',view/'train.json')
        offset=previous['iteration'] if previous else 0
        resume=previous['dcp_host'] if previous else '???'
        spec=tomllib.loads(render_profile('full',model_path='/model',annotation_path='/input/train.json',media_root='/input',
                           run_name=f'{arm}-{round_index}',num_epochs=epochs,train_records=population,annotation_mode='tht_single',resume_checkpoint=resume))
        training=self.task['training']; steps=population
        spec['model'].update(lora_rank=training['rank'],lora_alpha=training['alpha'],lora_dropout=training['dropout'],lora_target_modules=training['targets'])
        spec['model']['parallelism']['data_parallel_shard_degree']=1
        spec['optimizer'].update(lr=training['lr'],weight_decay=training['weight_decay'])
        spec['trainer']['max_iter']=offset+steps*epochs
        spec['trainer']['callbacks']['tao']['status_file_path']='/results/status.json'
        spec['scheduler']['cycle_lengths']=[offset+steps*epochs]
        spec['checkpoint']['save_iter']=steps if arm=='F' else offset+steps*epochs
        spec['custom']['seed']=training['seed']
        spec_path=directory/'train.toml'
        self.write_training_spec(spec_path,tomli_w.dumps(spec))
        self.store.artifact(stage+'/spec',spec_path)
        controls=directory/'control'; controls.mkdir(exist_ok=True)
        def hook(job):
            for path in sorted(controls.glob('epoch-*.ready.json')):
                ready=json.loads(path.read_text()); epoch=ready['epoch']
                key=f'{arm}/epoch/{epoch}'
                checkpoint_job=job
                if ready.get('job_id') and ready['job_id']!=job['job_id']:
                    prior=json.loads(self.runtime.record('show',ready['job_id']))
                    checkpoint_job={'job_id':ready['job_id'],'results_dir':prior['results_dir']}
                if self.store.get(key) is None:
                    exported=self.export(f'{arm}/export/{epoch}',checkpoint_job,ready)
                    scored=self.infer(arm,'proxy',f'epoch-{epoch:03d}',exported)
                    evidence=dict(phase='proxy',status='complete',n=scored['metrics']['n'],expected_n=len(self.proxy),
                                  mcc=scored['metrics']['mcc'],epoch=epoch,checkpoint=exported,
                                  dcp_host=str(Path(checkpoint_job['results_dir'])/Path(ready['checkpoint']).relative_to('/results')),
                                  framework_config=str(Path(checkpoint_job['results_dir'])/Path(ready['framework_config']).relative_to('/results')),
                                  iteration=ready['iteration'],unique_images=population,exposures=epoch*population,
                                  optimizer_steps=ready['iteration'],train_job=checkpoint_job['job_id'],checkpoint_ready_at=ready['ready_at_unix'])
                    self.store.seal(key,evidence)
                rows=[self.store.get(f'F/epoch/{i}') for i in range(1,epoch+1)]
                decision=epoch_decision(rows)
                answer=controls/f'epoch-{epoch:03d}.decision.json'
                if not answer.exists(): json_write(answer,{'epoch':epoch,'checkpoint':ready['checkpoint'],'action':'stop' if decision['stop'] else 'continue','selection':decision})
        mounts=[(APP,'/application',True),
                (APP/'cfw_cr3_aoi_adapter.py','/workspace/cosmos_framework/configs/base/reasoner/experiment/tao_cr3_aoi.py',True),
                (spec_path,'/train.toml',True),(self.task['model_path'],'/model',True),(view,'/input',True),(controls,'/control',False)]
        if previous: mounts.append((previous['dcp_host'],previous['dcp_host'],True))
        env=[('PYTHONPATH','/application'),('TAO_CR3_ANNOTATION_MODE','tht_single'),('TAO_CR3_TRAIN_ANNOTATION','/input/train.json'),
             ('TAO_CR3_MEDIA_ROOT','/input'),('TAO_CR3_SEED',training['seed']),('TAO_CR3_NEW_ROUND','1' if previous else '0'),
             ('IMAGINAIRE_OUTPUT_ROOT','/results'),('TAO_RESULTS_ROOT','/results'),
             ('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0'),('LOCAL_WORLD_SIZE','1'),('MASTER_ADDR','127.0.0.1'),('MASTER_PORT','29500')]
        if arm=='F': env += [('TAO_EPOCH_CONTROL_DIR','/control'),('TAO_EPOCH_STEPS',steps),('TAO_EPOCH_DECISION_TIMEOUT',self.task['budgets']['stage_wall_seconds'])]
        command=['-m','cosmos_framework.scripts.train','--sft-toml=/train.toml','--',f'trainer.seed={training["seed"]}',
                 'model.config.parallelism.fsdp_master_dtype=bfloat16','model.config.policy.model_max_length=40960',
                 'checkpoint.load_training_state=true']
        if previous:
            # Native resume restores model/optimizer/scheduler/trainer. A new
            # accumulated Mining dataset starts at its first sample.
            command.append('checkpoint.keys_not_to_resume=[dataloader]')
        def recover(retry_of,cmd,mnts):
            old=Path(json.loads(self.runtime.record('show',retry_of))['results_dir'])
            candidates=sorted(p.parent for p in old.rglob('model/.metadata'))
            candidates=[p.parent for p in candidates if all((p.parent/x).exists() for x in ('optim/.metadata','scheduler/.metadata','trainer/.metadata','dataloader'))]
            if not candidates:
                # No saved update: native initialization is the only valid restart.
                return cmd,mnts
            cp=candidates[-1]
            resumed=json.loads(json.dumps(spec));resumed['checkpoint']['load_path']=str(cp)
            retry_spec=directory/(retry_of+'-resume.toml');retry_spec.write_text(tomli_w.dumps(resumed))
            mnts=[m for m in mnts if m[1]!='/train.toml']+[(retry_spec,'/train.toml',True),(cp,str(cp),True)]
            cmd=[x for x in cmd if not x.startswith('checkpoint.keys_not_to_resume=')]
            return cmd,mnts
        result=self.run_job(stage,command,mounts=mounts,env=env,required_output='status.json',heartbeat=hook if arm=='F' else None,recover_training=recover)
        job=result['job']; native_root=Path(job['results_dir'])/'deft_cr3_aoi/train'/f'{arm}-{round_index}'
        import math
        events=load_jsonl(Path(job['results_dir'])/'status.json')
        if not events or events[-1].get('status')!='SUCCESS' or not math.isfinite(events[-1].get('kpi',{}).get('train/avg_loss',float('nan'))):
            raise ValueError('native terminal success with finite training loss is required')
        markers=list(Path(job['results_dir']).rglob('latest_checkpoint.txt'))
        if len(markers)!=1: raise ValueError('ambiguous native checkpoint root')
        cp=markers[0].parent/markers[0].read_bytes().split(b'\0',1)[0].decode().strip()
        config=cp.parent.parent/'config.yaml'
        telemetry=config.parent/'training_telemetry.jsonl'
        self.store.artifact(stage+'/telemetry',telemetry)
        if not all((cp/x).exists() for x in ('model/.metadata','optim/.metadata','scheduler/.metadata','trainer/.metadata','dataloader')):
            raise ValueError('native continuation checkpoint is incomplete')
        if arm=='F':
            rows=[self.store.get(f'F/epoch/{i}') for i in range(1,epochs+1) if self.store.get(f'F/epoch/{i}')]
            decision=epoch_decision(rows)
            if not decision['stop']: raise ValueError('F native run ended before the declared stopping rule')
            self.save_selection('F',dict(decision['best'],stop_reason=decision['reason']))
            return self.result('F/select')
        ready={'checkpoint':'/results/'+str(cp.relative_to(job['results_dir'])),'framework_config':'/results/'+str(config.relative_to(job['results_dir']))}
        exported=self.export(f'{arm}/export/{round_index}',job,ready)
        return {'checkpoint':exported,'dcp_host':str(cp),'framework_config':str(config),
                'iteration':offset+steps*epochs,'unique_images':population,'exposures':population*epochs,
                'optimizer_steps':steps*epochs,'scheduler_resume':'native optimizer and scheduler state; round schedule horizon is cumulative step cap',
                'dataloader_resume':'reset for changed accumulated Mining dataset; checkpoint resumes within a round retain cursor'}

    def search(self,arm,model):
        if self.store.stage(arm+'/select')['status']=='complete': return self.result(arm+'/select')
        budgets=self.task['budgets'];self.store.start_budget(arm,budgets['icl_wall_seconds'],budgets['icl_agent_tokens'])
        astra=self.task['astra']; runtime=Docker(self.root/'policy-jobs',astra['image'])
        mounts=generic_mounts(astra['codex_binary'],astra['python_binary'],astra['python_stdlib'],astra['system_libraries'],astra.get('site_packages',[]))
        trials=[]; stop_reason='trial_limit'; self.budget_scope=arm
        try:
          for index in range(1,13):
            trial=f'trial-{index:03d}'
            saved=self.store.get(arm+'/trial/'+trial)
            if saved: trials.append(saved); continue
            remaining=self.store.remaining(arm)
            if remaining['wall_seconds']<budgets['trial_admission_seconds'] or remaining['tokens']<=0:
                stop_reason='search_budget';break
            proposal_key=arm+'/proposal/'+trial
            context=self.store.get(proposal_key+'/context')
            if context is None:
                context={'arm':arm,'trials':trials,'remaining_budget':remaining,'reference_capacity':self.task['icl_reference_capacity'],
                         'max_trials':12,'selection':'highest MCC; within 0.01 fewer images/calls/prompt characters, then trial ID',
                         'baseline_metrics':self.result('B/proxy-summary')}
                self.store.seal(proposal_key+'/context',context)
            def invoke(job,on_open):
                first=self.store.db.execute('SELECT MIN(started) FROM attempts WHERE stage=?',(proposal_key,)).fetchone()[0]
                active_context=json.loads(json.dumps(context));remaining=self.store.remaining(arm)
                started=json.loads((Path(job['results_dir'])/'launch.json').read_text())['submitted_at_unix'] if job else time.time()
                wall=min(self.deadline,first+budgets['query_wall_seconds'],time.time()+remaining['wall_seconds'])-started
                if wall<=0 or (not job and remaining['tokens']<2): raise TimeoutError('proposal budget exhausted')
                if not job: active_context['remaining_budget']=remaining
                decision,details=propose(runtime,mounts,astra['auth_file'],astra['model'],astra['effort'],active_context,
                      self.root/'proposals'/arm/trial,wall,job=job,on_open=on_open)
                return {**details,'backend_state':details['backend'],'completed':decision is not None,'decision':decision}
            proposed=self.run_agent(proposal_key,runtime,invoke,budget=arm)
            decision=proposed['decision']
            if decision['action']=='stop': stop_reason='agent_stop';break
            if self.store.remaining(arm)['wall_seconds']<budgets['trial_admission_seconds'] or self.store.remaining(arm)['tokens']<=0:
                stop_reason='search_budget_after_proposal';break
            policy=decision['policy']; from retrieval import validate_policy,policy_hash
            validate_policy(policy,self.task['icl_reference_capacity'])
            policy_path=self.root/'policies'/arm/(trial+'.json')
            if not policy_path.exists(): json_write(policy_path,policy)
            self.store.artifact(arm+'/'+trial+'/policy',policy_path)
            score=self.infer(arm,'proxy',trial,model,policy=policy)
            requests=json.loads((self.root/'views'/arm/'proxy'/trial/'requests.json').read_text())
            chars=sum(sum(len(p['text']) for m in r['messages'] for p in m['content'] if p['type']=='text') for r in requests)/len(requests)
            value=dict(phase='proxy',status='complete',n=score['metrics']['n'],expected_n=len(self.proxy),mcc=score['metrics']['mcc'],
                       trial_id=trial,images_per_query=policy['parameters']['reference_count']+1,calls_per_query=1,
                       prompt_characters=chars,policy=policy,policy_sha256=policy_hash(policy),metrics=score['metrics'],
                       query_ids_sha256=json_hash(sorted(q['id'] for q in self.proxy)))
            self.store.seal(arm+'/trial/'+trial,value);trials.append(value)
        except TimeoutError:
            stop_reason='search_budget_partial_trial'
            if time.time()>=self.deadline: raise
        finally:
            self.budget_scope=None
        if not trials: raise RuntimeError(arm+': no completed Proxy trial; search incomplete')
        selected=select_icl(trials)
        self.save_selection(arm,dict(selected,checkpoint=model,search_stop_reason=stop_reason))
        return self.result(arm+'/select')

    def run(self):
        completed=self.store.get('completed')
        if completed:
            if sha256(completed['report'])!=completed['sha256']: raise ValueError('completed report snapshot changed')
            return json.loads(Path(completed['report']).read_text())
        if self.store.get('freeze'):
            return self.benchmark()
        # All selection below is Proxy-only. A configuration is inherited from
        # the verified M3 isolation smoke, not optimized on labels or examples.
        self.save_selection('A',self.task['astra'])
        base=self.task['model_path']
        baseline=self.infer('B','proxy','base',base)
        if self.store.stage('B/proxy-summary')['status']!='complete':
            a=self.store.begin('B/proxy-summary');self.store.finish(a,'complete',baseline['metrics'])
        f=self.train('F',[r['id'] for r in self.mining],20)
        c=self.search('C',base)
        previous=None; selected=[]; rounds=[]; gaps=baseline
        for number in range(1,6):
            key=f'D/round/{number}'
            saved=self.store.get(key)
            if saved:
                rounds.append(saved);previous=saved;selected=saved['selected_ids']
                gaps=json.loads(Path(saved['metrics_path']).read_text());continue
            from tht_rcca import route
            rcca=route(gaps,self.proxy,phase='proxy',neighbors=self.task['deft_neighbors'])
            rcca_path=self.root/'routing'/f'round-{number:03d}.json'
            if not rcca_path.exists(): json_write(rcca_path,rcca)
            self.store.artifact(f'D/routing/{number}',rcca_path)
            targets=[q for q in self.proxy if q['id'] in set(rcca['mining_targets'])]
            policy={'toolset':'siglip-retrieval-v1','version':1,'rationale':'DEFT Proxy error-neighbor mining; if no errors, all Proxy ties',
                    'max_calls':1,'parameters':{'reference_count':self.task.get('deft_neighbors',4),'ordering':'nearest_first','instruction':''}}
            # Embed full Proxy once; each subset uses the same immutable cache.
            refs=self.retrieval.references('proxy',self.proxy,policy,max(4,self.task['icl_reference_capacity']))
            newly={r['id'] for q in targets for r in refs[q['id']]}
            selected=sorted(set(selected)|newly)
            if not selected: raise RuntimeError('DEFT mining returned no training samples')
            trained=self.train('D',selected,f['epoch'],round_index=number,previous=previous)
            gaps=self.infer('D','proxy',f'round-{number:03d}',trained['checkpoint'])
            value=dict(trained,phase='proxy',status='complete',n=gaps['metrics']['n'],expected_n=len(self.proxy),mcc=gaps['metrics']['mcc'],
                       round=number,selected_ids=selected,new_images=len(newly-set(previous['selected_ids'])) if previous else len(newly),
                       metrics_path=str(self.root/'metrics/D/proxy'/f'round-{number:03d}.json'))
            value['routing_artifact']=str(rcca_path)
            self.store.seal(key,value);rounds.append(value);previous=value
        d=select_round(rounds);self.save_selection('D',d)
        e=self.search('E',d['checkpoint'])
        common={'base_fingerprint':self.task['base_fingerprint']}
        settings={'A':{**common,'configuration':self.task['astra']},'B':{**common,'checkpoint':base},
                  'C':{**common,'checkpoint':base,'policy':c['policy']},'D':{**common,'checkpoint':d['checkpoint']},
                  'E':{**common,'checkpoint':d['checkpoint'],'policy':e['policy']},'F':{**common,'checkpoint':f['checkpoint']}}
        manifests={}
        for path in {settings[a]['checkpoint'] for a in 'BCDEF'}:
            folder=Path(path)
            files=[{'path':str(p),'sha256':sha256(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                   for p in sorted(folder.iterdir()) if p.is_file()]
            manifest=self.root/'frozen-models'/(json_hash(path)+'.json');json_write(manifest,files)
            self.store.artifact('frozen-model/'+json_hash(path),manifest)
            manifests[path]={'path':str(manifest),'sha256':sha256(manifest)}
        for arm in 'BCDEF': settings[arm]['model_manifest']=manifests[settings[arm]['checkpoint']]
        for arm in settings:
            settings[arm].update(prompt_sha256=self.task['prompt_sha256'],preprocessing='RGB/ImageOps.pad/256x256/NEAREST/PNG',
                                 generation={'max_new_tokens':4096,'do_sample':False},seed=self.task['training']['seed'],
                                 statistics=self.task['statistics'])
        settings['A']['prompt_sha256']=self.task['astra']['prompt_template_sha256']
        settings['A']['prompt_template']=self.task['astra']['prompt_template']
        settings['A']['generation']={'provider':'Codex','native_rollout_tokens':self.task['budgets']['query_agent_tokens'],
                                     'wall_seconds':self.task['budgets']['query_wall_seconds'],'reasoning_effort':self.task['astra']['effort']}
        self.store.freeze(settings)
        return self.benchmark()

    def benchmark(self):
        settings=self.store.verify_freeze()['settings']
        queries=load_jsonl(self.public/'benchmark/queries.jsonl')
        for arm in 'BCDEF':
            self.infer(arm,'benchmark','final',settings[arm]['checkpoint'],policy=settings[arm].get('policy'),queries=queries)
        astra=self.task['astra'];runtime=Docker(self.root/'astra-jobs',astra['image'])
        mounts=generic_mounts(astra['codex_binary'],astra['python_binary'],astra['python_stdlib'],astra['system_libraries'],astra.get('site_packages',[]))
        if self.store.stage('A/access-probe')['status']!='complete':
            prior=self.store.db.execute("SELECT * FROM attempts WHERE stage='A/access-probe' ORDER BY id DESC LIMIT 1").fetchone()
            if prior and prior['job_id']:
                rec=json.loads(runtime.record('show',prior['job_id']));job={'job_id':prior['job_id'],'results_dir':rec['results_dir']}
                state=runtime.wait(prior['job_id'],prior['started']+60)
                if state['state']!='COMPLETE': raise RuntimeError('runtime access probe incomplete')
                a=prior['id']
            else:
                a=self.store.begin('A/access-probe');job=isolation_probe(runtime,mounts,on_open=lambda jid:self.store.attach_job(a,jid))
            self._native_cost(runtime,job,False,'A/access-probe');self.store.finish(a,'complete',job)
        completed={r['id'] for r in self.store.answers('A','benchmark','final')}
        from tht_contract import preprocess
        for index,query in enumerate(queries):
            if query['id'] in completed: continue
            if time.time()>=self.deadline: raise TimeoutError('suite wall budget exhausted')
            stage=f'A/benchmark/{index:04d}'
            path=self.root/'astra-views'/f'{index:04d}.png';preprocess(self.public/query['image'],path)
            def invoke(job,on_open):
                first=self.store.db.execute('SELECT MIN(started) FROM attempts WHERE stage=?',(stage,)).fetchone()[0]
                used=[json.loads(r['payload']) for r in self.store.db.execute('SELECT payload FROM usage') if json.loads(r['payload']).get('stage')==stage]
                if any(r.get('input_tokens') is None or r.get('output_tokens') is None for r in used):
                    raise RuntimeError('unknown failed-attempt usage blocks retry')
                tokens=self.task['budgets']['query_agent_tokens']-sum(r['input_tokens']+r['output_tokens'] for r in used)
                # The original query budget includes all failed attempts.
                wall=min(self.deadline,first+self.task['budgets']['query_wall_seconds'])-time.time()
                if tokens<=0 or wall<=0: raise TimeoutError('A query budget exhausted; incomplete')
                if job:
                    launch=json.loads((Path(job['results_dir'])/'launch.json').read_text())
                    wall=min(self.deadline,first+self.task['budgets']['query_wall_seconds'])-launch['submitted_at_unix']
                return classify(runtime,path,auth_file=astra['auth_file'],mounts=mounts,model=astra['model'],effort=astra['effort'],
                      wall_seconds=wall,token_budget=tokens,job=job,on_open=on_open)
            result=self.run_agent(stage,runtime,invoke)
            self.store.record_answer('A','benchmark','final',{'id':query['id'],'completed':True,'raw_answer':result['raw_answer'],
                                      'finish_reason':result['finish_reason'],'invocation_id':result['job']['job_id'],'usage':result['usage']})
        scored=grade(self.store.answers('A','benchmark','final'),load_jsonl(self.private/'evaluation/benchmark-gold.jsonl'),self.evaluator)
        json_write(self.root/'metrics/A/benchmark/final.json',scored)
        if not self.store.completed_benchmark([q['id'] for q in queries]): raise RuntimeError('six-arm Benchmark coverage incomplete')
        from suite_analysis import export
        report=export(self.store,self.task)
        digest=sha256(self.root/'report_data.json');snapshot=self.root/'exports'/('report-'+digest+'.json')
        snapshot.parent.mkdir(exist_ok=True);shutil.copyfile(self.root/'report_data.json',snapshot)
        self.store.seal('completed',{'report':str(snapshot),'sha256':digest})
        return report
