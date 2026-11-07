# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Fixed-checkpoint ICL only; the original E adaptive search remains unchanged.
import json
import time
from pathlib import Path
from icl_store import ICLStore
from suite_driver import Suite
from suite_data import RetrievalToolset
from suite_runtime import Docker
from tht_contract import external_evaluator, load_jsonl, json_hash, sha256


class ICLSuite(Suite):
    cost_arms = 'H'

    def __init__(self, task, approval, *, resume_origin='operator'):
        from icl_preflight import verify_resume
        verify_resume(task)
        self.task = task
        self.root = Path(task['results_dir']).resolve()
        self.store = ICLStore(self.root)
        self.store.initialize(task, approval)
        for artifact in self.store.db.execute('SELECT path,sha256 FROM artifacts'):
            if sha256(artifact['path']) != artifact['sha256']:
                raise ValueError('run artifact changed before resume')
        from generator_contract import verify_file
        for row in self.store.db.execute("SELECT value FROM meta WHERE key LIKE '%/view-files'"):
            for binding in json.loads(row['value']):
                verify_file(binding)
        resuming = self.store.get('started_at') is not None
        self.store.seal('started_at', self.store.get('started_at', time.time()))
        if resuming:
            self.store.event('resume', {'origin': resume_origin, 'policy_changed': False})
        self.deadline = self.store.get('started_at') + task['budgets']['suite_wall_seconds']
        self.public = Path(task['public_package'])
        self.private = Path(task['private_package'])
        self.prompt = json.loads(Path(task['direct_profile']).read_text())['prompt']
        self.evaluator = external_evaluator(self.public / 'evaluator')
        self.mining = load_jsonl(self.public / 'mining/samples.jsonl')
        self.proxy = load_jsonl(self.public / 'proxy/queries.jsonl')
        self.runtime = Docker(self.root / 'jobs', task['framework_image'])
        self.retrieval = RetrievalToolset(task, self.run_job)
        self.budget_scope = None
        self.store.seal('retrieval_cache_origin', {'mode': 'fresh', 'root': str(self.retrieval.root),
                                                'shared_historical_cache': False})

    def result(self, stage):
        # Exact permitted E first-context aggregate, without a new baseline run
        # or imported E policies, selected trials or Benchmark information.
        if stage == 'B/proxy-summary':
            return self.task['baseline_metrics']
        return super().result(stage)

    def train(self, *args, **kwargs):
        raise ValueError('training is forbidden in fixed-checkpoint ICL addendum')

    def export(self, *args, **kwargs):
        raise ValueError('model export/generation is forbidden in fixed-checkpoint ICL addendum')

    def run_job(self, stage, command, **kwargs):
        self.store.guard_stage(stage)
        if kwargs.get('recover_training'):
            raise ValueError('training recovery is forbidden')
        if stage.startswith('H/'):
            expected = ['/worker.py','--model','/model','--requests','/input/requests.json',
                        '--output','/results','--seed','42']
            if command != expected:
                raise ValueError('only the unchanged H inference worker command is permitted')
            views = [source for source,target,readonly in kwargs.get('mounts',[]) if target == '/input' and readonly]
            if len(views) != 1:
                raise ValueError('exact role-limited classifier view required')
            self.seal_view(stage,Path(views[0]))
        else:
            expected = ['embedding','image_embeddings','-e'] if stage.startswith('retrieval/embed-') else ['tmm','nearest_neighbors','-e']
            if len(command) != 4 or command[:3] != expected:
                raise ValueError('only native embedding/mining commands are permitted')
        result = super().run_job(stage, command, **kwargs)
        if stage.startswith('H/'):
            from icl_preflight import verify_gold
            verify_gold(self.task,stage.split('/')[1])
        return result

    def seal_view(self, stage, view):
        paths = sorted(view.rglob('*'))
        if any(p.is_symlink() for p in paths):
            raise ValueError('classifier views must not contain symlinks')
        files = [p for p in paths if p.is_file()]
        self.store.seal(stage+'/view-membership',[str(p.relative_to(view)) for p in files])
        from generator_contract import file_binding
        self.store.seal(stage+'/view-files',[file_binding(path) for path in files])


    def infer(self, arm, phase, candidate, model, *, policy=None, queries=None):
        if arm != 'H' or model != self.task['fixed_checkpoint'] or phase not in ('proxy', 'benchmark'):
            raise ValueError('H must use its exact sealed checkpoint and role')
        from retrieval import validate_policy
        validate_policy(policy, self.task['icl_reference_capacity'])
        self.store.guard_stage(f'{arm}/{phase}/{candidate}')
        if phase == 'benchmark':
            frozen = self.store.verify_freeze()['settings']['H']
            if policy != frozen['policy'] or model != frozen['checkpoint']:
                raise ValueError('Benchmark candidate differs from freeze')
            expected = load_jsonl(self.public / 'benchmark/queries.jsonl')
        else:
            expected = self.proxy
        if queries is not None and queries != expected:
            raise ValueError('partial or altered query population is forbidden')
        return super().infer(arm, phase, candidate, model, policy=policy, queries=expected)

    def reconcile_proposals(self):
        # E's admission rule runs before its next proposal. Reconcile already
        # issued work first, even when the admission reserve has since expired.
        # This path never submits another policy job or extends any deadline.
        pending = list(self.store.db.execute(
            "SELECT * FROM attempts WHERE stage LIKE 'H/proposal/%' AND status IN ('pending','running') ORDER BY id"))
        if not pending:
            return
        from astra_isolation import generic_mounts
        from policy_agent import propose
        from suite_runtime import infrastructure_failure
        astra = self.task['astra']
        runtime = Docker(self.root/'policy-jobs',astra['image'])
        mounts = generic_mounts(astra['codex_binary'],astra['python_binary'],astra['python_stdlib'],
                                astra['system_libraries'],astra.get('site_packages',[]))
        budget = self.store.db.execute("SELECT * FROM budgets WHERE name='H'").fetchone()
        if budget is None:
            raise ValueError('pending policy attempt lacks its original budget')
        for attempt in pending:
            stage = attempt['stage']
            first = self.store.db.execute('SELECT MIN(started) FROM attempts WHERE stage=?',(stage,)).fetchone()[0]
            deadline = min(self.deadline,first+self.task['budgets']['query_wall_seconds'],
                           budget['started']+budget['wall_cap'])
            if not attempt['job_id']:
                reason = 'budget_exhausted' if time.time() >= deadline else 'infrastructure_error'
                self.store.finish(attempt['id'],reason,{'reason':'interrupted before native job record'})
                if reason == 'budget_exhausted':
                    raise TimeoutError('unlaunched proposal original deadline expired')
                continue
            record = json.loads(runtime.record('show',attempt['job_id']))
            job = {'job_id':attempt['job_id'],'results_dir':record['results_dir'],'container_id':record['backend_ref']}
            started = json.loads((Path(job['results_dir'])/'launch.json').read_text())['submitted_at_unix']
            output = self.root/'proposals/H'/stage.split('/')[-1]
            context = json.loads((output/'context.json').read_text())
            sealed = self.store.get(stage+'/context')
            if ({k:v for k,v in context.items() if k != 'remaining_budget'} !=
                    {k:v for k,v in sealed.items() if k != 'remaining_budget'}):
                raise ValueError('pending policy context drift')
            decision,details = propose(runtime,mounts,astra['auth_file'],astra['model'],astra['effort'],
                                       context,output,deadline-started,job=job)
            value = {**details,'backend_state':details['backend'],'completed':decision is not None,'decision':decision}
            usage = dict(value['usage'],invocation_id=job['job_id'],category='formal',stage=stage,
                         started_at_unix=value['started_at_unix'],ended_at_unix=value['ended_at_unix'],
                         completed_final=value['completed'],backend_state=value['backend_state']['state'],
                         raw_answer=value.get('raw_answer'),invocation_path=str(Path(job['results_dir'])/'invocation.json'))
            if usage.get('input_tokens') is None or usage.get('output_tokens') is None:
                self.store.record_usage(job['job_id'],usage)
                self.store.finish(attempt['id'],'program_error',value)
                raise RuntimeError('reconciled proposal usage unknown; no continuation permitted')
            self.store.record_usage(job['job_id'],usage,budget='H')
            if value['completed']:
                self.store.finish(attempt['id'],'complete',value)
                continue
            state = value['backend_state']
            reason = ('infrastructure_error' if infrastructure_failure(state.get('native',{}),runtime.logs(job['job_id']))
                      else 'budget_exhausted' if state['state']=='CANCELED' or value.get('budget_exhausted') else 'program_error')
            self.store.finish(attempt['id'],reason,value)
            if reason == 'budget_exhausted':
                raise TimeoutError('reconciled proposal has no final within original budget')
            raise RuntimeError('reconciled proposal incomplete: '+reason)

    def collect_native_answers(self, stage):
        # Same immutable answer/usage contract as Suite.infer's finally block,
        # also used when reconciliation cannot re-enter inference successfully.
        arm,phase,candidate = stage.split('/')
        for attempt in self.store.db.execute('SELECT job_id FROM attempts WHERE stage=? ORDER BY id',(stage,)):
            if not attempt['job_id']:
                continue
            record = json.loads(self.runtime.record('show',attempt['job_id']))
            directory = Path(record['results_dir'])
            raw = directory/'raw_answers.jsonl'
            for row in load_jsonl(raw) if raw.exists() else []:
                self.store.record_answer(arm,phase,candidate,row)
                usage = {k:row.get(k) for k in ('invocation_id','inference_input_tokens','inference_output_tokens','started_at_unix','ended_at_unix')}
                usage.update(category='formal',arm=arm,phase=phase,candidate=candidate,stage=stage)
                self.store.record_usage(row['invocation_id'],usage)
            failures = directory/'attempts.jsonl'
            for row in load_jsonl(failures) if failures.exists() else []:
                if row['status'] == 'error':
                    self.store.record_usage(row['invocation_id'],dict(row,category='formal',arm=arm,phase=phase,stage=stage))

    def reconcile_native(self):
        # Reconcile only previously recorded native work. No submit, retry, new
        # trial, fresh deadline or admission decision occurs in this method.
        import datetime
        from suite_runtime import infrastructure_failure
        pending = list(self.store.db.execute(
            "SELECT * FROM attempts WHERE stage NOT LIKE 'H/proposal/%' AND status IN ('pending','running') ORDER BY id"))
        for attempt in pending:
            stage = attempt['stage']
            self.store.guard_stage(stage)
            first = self.store.db.execute('SELECT MIN(started) FROM attempts WHERE stage=?',(stage,)).fetchone()[0]
            deadline = min(self.deadline,first+self.task['budgets']['stage_wall_seconds'])
            if not self.store.get('freeze'):
                budget = self.store.db.execute("SELECT * FROM budgets WHERE name='H'").fetchone()
                if budget is None:
                    raise ValueError('pending native search job lacks original H budget')
                deadline = min(deadline,budget['started']+budget['wall_cap'])
            if not attempt['job_id']:
                reason = 'budget_exhausted' if time.time() >= deadline else 'infrastructure_error'
                self.store.finish(attempt['id'],reason,{'reason':'interrupted before native record'})
                raise RuntimeError('native reconciliation has no recorded handle: '+reason)
            image = self.task['data_services_image'] if stage.startswith('retrieval/') else self.task['framework_image']
            runtime = Docker(self.root/'jobs',image)
            record = json.loads(runtime.record('show',attempt['job_id']))
            job = {'job_id':attempt['job_id'],'results_dir':record['results_dir'],'container_id':record['backend_ref']}
            try:
                final = runtime.wait(job['job_id'],deadline)
                self._native_cost(runtime,job,True,stage)
                if final['state'] == 'COMPLETE':
                    finished = final.get('native',{}).get('FinishedAt')
                    if not finished or finished.startswith('0001'):
                        self.store.finish(attempt['id'],'program_error',{'job':job,'reason':'native completion timestamp unavailable'})
                        raise RuntimeError('cannot verify completion within original native deadline')
                    ended = datetime.datetime.fromisoformat(finished.replace('Z','+00:00')).timestamp()
                    if ended > deadline:
                        self.store.finish(attempt['id'],'budget_exhausted',{'job':job,'backend':final,'deadline':deadline})
                        raise TimeoutError('native job completed beyond its original deadline')
                    if stage.startswith('H/'):
                        output = Path(job['results_dir'])/'completion.json'
                    else:
                        name = stage.split('/')[1]
                        output = self.root/'retrieval'/name/('embeddings.parquet' if name.startswith('embed-') else 'mined.parquet')
                    if not output.is_file():
                        self.store.finish(attempt['id'],'program_error',{'job':job,'reason':'completion artifact missing','path':str(output)})
                        raise RuntimeError('reconciled native success lacks required output')
                    result = {'job':job,'artifacts':[self.store.artifact(stage+'/output',output)]}
                    self.store.finish(attempt['id'],'complete',result)
                    continue
                reason = ('budget_exhausted' if final['state']=='CANCELED' else
                          'infrastructure_error' if infrastructure_failure(final.get('native',{}),runtime.logs(job['job_id'])) else 'program_error')
                self.store.finish(attempt['id'],reason,{'job':job,'backend':final,'deadline':deadline})
                if reason == 'budget_exhausted':
                    raise TimeoutError('native job exhausted its original deadline')
                raise RuntimeError('native reconciliation incomplete: '+reason)
            finally:
                if stage.startswith('H/'):
                    self.collect_native_answers(stage)

    def recover_trials(self):
        # A supervisor can stop between native completion and sealing a trial.
        # Recover only complete stages, using E's exact trial-summary fields.
        from retrieval import policy_hash
        for row in self.store.db.execute("SELECT name FROM stages WHERE name LIKE 'H/proxy/%' AND status='complete' ORDER BY name"):
            trial = row['name'].split('/')[-1]
            if self.store.get('H/trial/'+trial):
                continue
            if not (self.root/'views/H/proxy'/trial/'requests.json').is_file():
                raise ValueError('completed trial classifier view is missing; no reconstruction launch permitted')
            policy = json.loads((self.root/'policies/H'/(trial+'.json')).read_text())
            score = self.infer('H','proxy',trial,self.task['fixed_checkpoint'],policy=policy)
            requests = json.loads((self.root/'views/H/proxy'/trial/'requests.json').read_text())
            chars = sum(sum(len(p['text']) for message in r['messages'] for p in message['content'] if p['type']=='text') for r in requests)/len(requests)
            value = dict(phase='proxy',status='complete',n=score['metrics']['n'],expected_n=len(self.proxy),mcc=score['metrics']['mcc'],
                         trial_id=trial,images_per_query=policy['parameters']['reference_count']+1,calls_per_query=1,
                         prompt_characters=chars,policy=policy,policy_sha256=policy_hash(policy),metrics=score['metrics'],
                         query_ids_sha256=json_hash(sorted(q['id'] for q in self.proxy)))
            self.store.seal('H/trial/'+trial,value)

    def run(self):
        if self.store.db.execute("SELECT 1 FROM attempts WHERE status='program_error' LIMIT 1").fetchone():
            raise RuntimeError('terminal deterministic program failure forbids continuation')
        self.reconcile_native()
        if self.store.get('completed'):
            self.store.verify_freeze()
            from icl_analysis import export
            return export(self.store, self.task)
        if not self.store.get('freeze'):
            self.reconcile_proposals()
            self.recover_trials()
            selected = self.search('H', self.task['fixed_checkpoint'])
            from icl_preflight import verify_checkpoint
            verify_checkpoint(self.task['source_model'], self.task['model_path'])
            manifest = self.task['source_model']['model_manifest']
            self.store.artifact('fixed-model-manifest', manifest['path'])
            settings = {'H': {'checkpoint': self.task['fixed_checkpoint'],
                        'policy': selected['policy'], 'policy_sha256': selected['policy_sha256'],
                        'model_manifest': manifest, 'base_fingerprint': self.task['base_fingerprint'],
                        'preflight_sha256': json_hash(self.task), 'prompt_sha256': self.task['prompt_sha256'],
                        'preprocessing': 'RGB/ImageOps.pad/256x256/NEAREST/PNG',
                        'generation': {'max_new_tokens':4096, 'do_sample':False},
                        'seed':42, 'statistics':self.task['statistics']}}
            self.store.freeze(settings)
        return self.benchmark()

    def benchmark(self):
        settings = self.store.verify_freeze()['settings']['H']
        queries = load_jsonl(self.public / 'benchmark/queries.jsonl')
        if len(queries) != 271:
            raise ValueError('Benchmark requires exact 271 coverage')
        self.infer('H', 'benchmark', 'final', settings['checkpoint'], policy=settings['policy'], queries=queries)
        if not self.store.completed_benchmark([q['id'] for q in queries]):
            raise ValueError('final Benchmark incomplete')
        path = self.root / 'metrics/H/benchmark/final.json'
        self.store.seal('completed', {'status':'complete', 'metrics':str(path), 'sha256':sha256(path)})
        from icl_analysis import export
        return export(self.store, self.task)
