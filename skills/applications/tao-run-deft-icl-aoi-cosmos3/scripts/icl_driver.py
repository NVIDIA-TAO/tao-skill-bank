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

    def run(self):
        if self.store.get('completed'):
            self.store.verify_freeze()
            from icl_analysis import export
            return export(self.store, self.task)
        if not self.store.get('freeze'):
            self.reconcile_proposals()
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
