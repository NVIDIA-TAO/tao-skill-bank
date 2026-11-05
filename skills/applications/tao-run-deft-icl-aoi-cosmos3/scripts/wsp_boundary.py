#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted WSP capability boundary, durable admission ledger and final holdout.

Only development capabilities are exposed to the editor. The evaluator and
model runner are injected by trusted code, never selected by an editor request.
Reservations precede work and are never refunded, including on process death.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

from suite_runtime import json_write
from tht_contract import grade, json_hash, load_jsonl, sha256
from wsp_weights import apply_patch, catalog, inventory, read_values, verify_inventory
from wsp_usage import response_completion


class LimitError(ValueError):
    pass


class Journal:
    def __init__(self, root, task, *, now=time.time):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.now = now
        self.task = task
        self.lock = (self.root / 'owner.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise ValueError('another WSP boundary owns this run') from None
        self.mutex = threading.RLock()
        self.db = sqlite3.connect(self.root / 'journal.sqlite', check_same_thread=False)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, value TEXT NOT NULL, digest TEXT NOT NULL)')
        row = self.db.execute('SELECT value FROM state WHERE id=1').fetchone()
        if row is None:
            initialization = self.root / 'initialization.json'
            started_at = json.loads(initialization.read_text())['started_at_unix'] if initialization.exists() else now()
            self.db.execute('INSERT INTO state VALUES (1,?)', (json.dumps({
                'task_sha256': json_hash(task), 'started_at_unix': started_at, 'phase': 'DEVELOPMENT',
                'used': {}, 'actual': {'input_tokens': 0, 'output_tokens': 0, 'gpu_wall_seconds': 0},
                'candidates': {}, 'forwards': {}, 'freeze': None, 'agent_usage': None,
                'runtime_profile_sha256': None}),))
            self.db.commit()
        elif json.loads(row[0])['task_sha256'] != json_hash(task):
            self.close()
            raise ValueError('cannot reset/reconfigure an existing run')

    def close(self):
        self.db.close()
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()

    def state(self):
        with self.mutex:
            return json.loads(self.db.execute('SELECT value FROM state WHERE id=1').fetchone()[0])

    def change(self, kind, payload, change=lambda state: None):
        with self.mutex:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                state = self.state()
                result = change(state)
                previous = self.db.execute('SELECT seq,digest FROM events ORDER BY seq DESC LIMIT 1').fetchone()
                event = {'seq': previous[0] + 1 if previous else 1, 'previous_sha256': previous[1] if previous else None,
                         'at_unix': self.now(), 'kind': kind, 'payload': payload}
                self.db.execute('INSERT INTO events VALUES (?,?,?)', (event['seq'], json.dumps(event, sort_keys=True), json_hash(event)))
                self.db.execute('UPDATE state SET value=? WHERE id=1', (json.dumps(state, sort_keys=True),))
                self.db.commit()
                return result
            except BaseException:
                self.db.rollback()
                raise

    def reserve(self, kind, charges, payload, *, phase='DEVELOPMENT'):
        def mutate(state):
            if state['phase'] != phase:
                raise LimitError('operation is closed in this phase')
            if self.now() - state['started_at_unix'] >= self.task['budgets']['experiment_wall_seconds']:
                raise LimitError('experiment wall exhausted')
            if phase == 'DEVELOPMENT' and state.get('editor_admissions_closed'):
                raise LimitError('editor admissions closed by observed stop')
            usage = state.get('agent_usage')
            if phase == 'DEVELOPMENT' and usage and usage.get('input_tokens') is not None and usage.get('output_tokens') is not None:
                if usage['input_tokens'] + usage['output_tokens'] >= self.task['budgets']['editor_agent_tokens']:
                    raise LimitError('actual input plus output token threshold reached')
            for key, amount in charges.items():
                if type(amount) is not int or amount < 0:
                    raise ValueError('invalid reservation')
                if state['used'].get(key, 0) + amount > self.task['budgets'][key]:
                    raise LimitError('budget exhausted: ' + key)
            for key, amount in charges.items():
                state['used'][key] = state['used'].get(key, 0) + amount
            return dict(state['used'])
        return self.change(kind, {'charges': charges, **payload}, mutate)

    def export(self):
        with self.mutex:
            rows = self.db.execute('SELECT value,digest FROM events ORDER BY seq').fetchall()
        previous = None
        events = []
        for value, digest in rows:
            event = json.loads(value)
            if event['previous_sha256'] != previous or json_hash(event) != digest:
                raise ValueError('journal hash chain failed')
            events.append({**event, 'sha256': digest})
            previous = digest
        return {'schema': 'wsp-ledger/1', 'state': self.state(), 'events': events, 'head_sha256': previous}


class Boundary:
    def __init__(self, root, task, base, base_manifest, views, forward, evaluator, *, now=time.time):
        self.root, self.task, self.base = Path(root), task, Path(base)
        self.base_manifest, self.views = base_manifest, Path(views)
        self.forward, self.evaluator, self.now = forward, evaluator, now
        self.journal = Journal(root, task, now=now)
        self.tensors = catalog(base)['tensors']
        self.mutex = threading.RLock()
        self.b = task['budgets']

    def close(self):
        self.journal.close()

    def check_storage(self):
        total = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink())
        if total >= self.b['storage_bytes']:
            raise LimitError('storage budget exhausted')
        return total

    def handle(self, request):
        """Serial dispatch, exact schemas, no paths/commands/configs from callers."""
        with self.mutex:
            started = self.now()
            self.journal.change('tool_request', {'request': request})
            try:
                self.check_storage()
                self.journal.reserve('tool_admission', {'tool_requests': 1}, {})
                if not isinstance(request, dict) or not isinstance(request.get('op'), str):
                    raise ValueError('JSON operation required')
                op = request['op']
                schemas = {'metadata': {'op', 'offset', 'limit'}, 'values': {'op', 'tensor', 'indices'},
                           'candidate': {'op', 'patch', 'code'}, 'forward': {'op', 'candidate'},
                           'freeze': {'op', 'candidate', 'rationale'}, 'status': {'op'}}
                if op not in schemas or set(request) != schemas[op]:
                    raise ValueError('unknown operation or fields')
                if op == 'metadata':
                    offset, limit = request['offset'], request['limit']
                    if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 256:
                        raise ValueError('metadata page bounds')
                    names = sorted(self.tensors)[offset:offset + limit]
                    result = {'base_fingerprint': self.base_manifest['fingerprint'], 'total_tensors': len(self.tensors),
                              'tensors': {n: {k: self.tensors[n][k] for k in ('dtype', 'shape', 'elements')} for n in names}}
                elif op == 'values':
                    indices = request['indices']
                    count = len(indices) if isinstance(indices, list) else 0
                    self.journal.reserve('inspection_admission', {'inspection_requests': 1, 'inspection_elements': count}, {})
                    result = read_values(self.base, self.tensors, request['tensor'], indices, self.b['elements_per_inspection'])
                elif op == 'candidate':
                    result = self.candidate(request['patch'], request['code'])
                elif op == 'forward':
                    result = self.development(request['candidate'])
                elif op == 'freeze':
                    result = self.freeze(request['candidate'], request['rationale'])
                else:
                    state = self.journal.state()
                    result = {'phase': state['phase'], 'used': state['used'], 'budgets': self.b,
                              'candidates': {k: {'valid': v['valid'], 'development_complete': v.get('development_complete', False)} for k, v in state['candidates'].items()}}
                self.journal.change('tool_result', {'op': op, 'result': result, 'wall_seconds': self.now() - started})
                return {'ok': True, 'result': result}
            except Exception as error:
                # Host trace retains details; the editor receives no host path,
                # private row, holdout state/metrics or native exception text.
                self.journal.change('tool_failure', {'error_type': type(error).__name__, 'error': str(error),
                                                      'wall_seconds': self.now() - started})
                result = {'ok': False, 'error': 'closed_or_budget_exhausted' if isinstance(error, LimitError) else 'invalid_or_failed_request'}
                safe_reasons = ('wrong patch base lineage', 'zero/rounded-away element change', 'nonfinite tensor',
                                'nonfinite delta', 'replacement overflows dtype', 'tensor budget exceeded',
                                'element budget exceeded', 'duplicate/invalid tensor edit', 'invalid/duplicate flat indices',
                                'base and candidate must both complete development before freeze',
                                'forward already attempted; no repeated feedback')
                if str(error) in safe_reasons:
                    result['reason'] = str(error)
                return result

    def candidate(self, patch, code):
        used = self.journal.reserve('candidate_admission', {'candidate_attempts': 1}, {})
        identity = 'c%04d' % used['candidate_attempts']
        directory = self.root / 'candidates' / identity
        directory.mkdir(parents=True, exist_ok=False)
        json_write(directory / 'patch.json', patch)
        if not isinstance(code, str) or not 1 <= len(code.encode()) <= 128 * 1024:
            raise ValueError('bounded actual patch-construction source required')
        (directory / 'source.py').write_text(code)
        status = {'valid': False, 'attempt_directory': str(directory)}
        self.journal.change('candidate_open', {'candidate': identity}, lambda state: state['candidates'].update({identity: status}))
        try:
            if not isinstance(patch, dict):
                raise ValueError('patch object required')
            count = sum(len(e.get('indices', [])) for e in patch.get('edits', []) if isinstance(e, dict) and isinstance(e.get('indices'), list))
            self.journal.reserve('edit_elements_admission', {'edited_elements_total': count}, {'candidate': identity})
            base_bytes = sum(f['size'] for f in self.base_manifest['files'])
            if self.check_storage() + base_bytes >= self.b['storage_bytes']:
                raise LimitError('insufficient storage for retained candidate copy')
            info = apply_patch(self.base, directory / 'model', self.base_manifest, patch,
                               max_tensors=self.b['tensors_per_candidate'], max_elements=self.b['elements_per_candidate'])
            info['source_sha256'] = sha256(directory / 'source.py')
            info['patch_file_sha256'] = sha256(directory / 'patch.json')
            json_write(directory / 'validation.json', info)
            status.update(valid=True, validation=info, model=str(directory / 'model'))
            self.journal.change('candidate_valid', {'candidate': identity, 'validation': info},
                                lambda state: state['candidates'].update({identity: status}))
            return {'candidate': identity, **{k: info[k] for k in ('changed_tensors', 'changed_elements', 'delta_norms', 'loadability')}}
        except BaseException as error:
            json_write(directory / 'failure.json', {'error_type': type(error).__name__, 'error': str(error)})
            self.journal.change('candidate_failed', {'candidate': identity, 'error_type': type(error).__name__})
            raise

    def model_for(self, candidate):
        if candidate == 'base':
            return self.base, self.base_manifest
        if not isinstance(candidate, str) or not re.fullmatch(r'c[0-9]{4}', candidate):
            raise ValueError('invalid candidate identity')
        item = self.journal.state()['candidates'].get(candidate)
        if not item or not item['valid']:
            raise ValueError('candidate not validated')
        return Path(item['model']), item['validation']['manifest']

    def _forward(self, candidate, role, *, phase):
        model, manifest = self.model_for(candidate)
        # IDs/labels are always trusted role views, never supplied by the editor.
        queries = load_jsonl(self.views / role / 'queries.jsonl')
        key = role + ':' + candidate
        if key in self.journal.state()['forwards']:
            raise LimitError('forward already attempted; no repeated feedback')
        if len(queries) > self.b['images_per_request']:
            raise LimitError('image count exceeds admitted batch')
        charges = {role + '_requests': 1, 'forward_images': len(queries),
                   'forward_output_tokens': len(queries) * self.b['max_new_tokens'],
                   'gpu_wall_seconds': self.b['forward_wall_seconds']}
        self.journal.reserve('forward_admission', charges, {'key': key, 'model_fingerprint': manifest['fingerprint']}, phase=phase)
        request_id = 'f%04d' % sum(self.journal.state()['used'].get(k, 0) for k in ('development_requests', 'holdout_requests'))
        directory = self.root / 'forwards' / request_id
        directory.mkdir(parents=True, exist_ok=False)
        self.journal.change('forward_open', {'key': key, 'request_id': request_id},
                            lambda state: state['forwards'].update({key: {'status': 'ATTEMPTED', 'request_id': request_id}}))
        started = self.now()
        result = None
        try:
            verify_inventory(model, manifest)
            deadline = min(started + self.b['forward_wall_seconds'],
                           self.journal.state()['started_at_unix'] + self.b['experiment_wall_seconds'])
            result = self.forward(model, role, directory, deadline)
            raw = result['raw']
            json_write(directory / 'result.json', result)
            if result.get('complete') is not True or len(raw) != len(queries) or {r['id'] for r in raw} != {r['id'] for r in queries}:
                raise ValueError('incomplete native output coverage')
            if any(r.get('completed') is not True or type(r.get('inference_output_tokens')) is not int or not 0 <= r['inference_output_tokens'] <= 4096 for r in raw):
                raise ValueError('native output/token contract failure')
            if result['profile_sha256'] != json_hash(self.task['inference']):
                raise ValueError('inference profile changed')
            profile_hash = result['runtime_profile_sha256']
            previous_profile = self.journal.state()['runtime_profile_sha256']
            if not isinstance(profile_hash, str) or len(profile_hash) != 64 or (previous_profile and profile_hash != previous_profile):
                raise ValueError('actual processor/template/runtime changed between calls')
            actual = {'input_tokens': sum(r['inference_input_tokens'] for r in raw),
                      'output_tokens': sum(r['inference_output_tokens'] for r in raw),
                      'gpu_wall_seconds': result['gpu_wall_seconds']}
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in actual.values()):
                raise ValueError('invalid actual usage')
            if actual['gpu_wall_seconds'] > self.b['forward_wall_seconds']:
                raise LimitError('native allocated GPU interval exceeded reservation')
            gold = load_jsonl(self.views / role / 'gold.jsonl')
            scored = grade(raw, gold, self.evaluator)
            json_write(directory / 'scored.json', scored)
            def complete(state):
                state['forwards'][key].update(status='COMPLETE', directory=str(directory), metrics=scored['metrics'],
                                               model_fingerprint=manifest['fingerprint'])
                state['runtime_profile_sha256'] = profile_hash
                for name, value in actual.items():
                    state['actual'][name] += value
                if candidate != 'base' and role == 'development':
                    state['candidates'][candidate]['development_complete'] = True
            self.journal.change('forward_complete', {'key': key, 'actual': actual}, complete)
            return {'candidate': candidate, 'raw': raw, 'metrics': scored['metrics'], 'actual': actual}
        except BaseException as error:
            # Reservations remain spent. Partial native outputs and usage remain
            # in the runner directory even if no complete comparison is possible.
            failure = {'key': key, 'error_type': type(error).__name__, 'error': str(error),
                       'wall_seconds': self.now() - started,
                       'observed_completed_images': len(result['raw']) if result is not None else None,
                       'observed_completed_output_tokens': sum(r['inference_output_tokens'] for r in result['raw']) if result is not None and all(type(r.get('inference_output_tokens')) is int for r in result['raw']) else None,
                       'allocated_gpu_wall_seconds': result.get('gpu_wall_seconds') if result is not None else None,
                       'incomplete_request_output_tokens': None}
            json_write(directory / 'failure.json', failure)
            self.journal.change('forward_failed', failure,
                                lambda state: state['forwards'][key].update(status='FAILED', directory=str(directory), failure_cost=failure))
            raise

    def development(self, candidate):
        return self._forward(candidate, 'development', phase='DEVELOPMENT')

    def freeze(self, candidate, rationale):
        if candidate == 'base' or not isinstance(rationale, str) or not 1 <= len(rationale.strip()) <= 8192:
            raise ValueError('edited candidate and development-only rationale required')
        model, manifest = self.model_for(candidate)
        state = self.journal.state()
        if state['phase'] != 'DEVELOPMENT' or state['freeze'] is not None:
            raise LimitError('only one freeze')
        if not state['candidates'][candidate].get('development_complete') or state['forwards'].get('development:base', {}).get('status') != 'COMPLETE':
            raise ValueError('base and candidate must both complete development before freeze')
        verify_inventory(model, manifest)
        validation = state['candidates'][candidate]['validation']
        record = {'schema': 'wsp-freeze/1', 'candidate': candidate, 'model_fingerprint': manifest['fingerprint'],
                  'base_fingerprint': self.base_manifest['fingerprint'], 'patch_sha256': validation['patch_sha256'],
                  'task_sha256': json_hash(self.task), 'split_sha256': self.task['split_sha256'],
                  'inference_sha256': json_hash(self.task['inference']), 'rationale': rationale,
                  'at_unix': self.now()}
        def commit(state):
            state['freeze'] = record
            state['phase'] = 'FROZEN'
        # Freeze in SQLite first. The exported file may be recovered from it;
        # a crash can never reopen editing or create a second selection.
        self.journal.change('freeze', record, commit)
        json_write(self.root / 'freeze.json', record)
        return {'frozen': True, 'candidate': candidate, 'freeze_sha256': json_hash(record), 'editor_closed': True}

    def holdout(self):
        """Trusted-only entrypoint, called after the editor process is stopped.

        Never an RPC operation. An interruption/failure is terminal; there is no
        second holdout attempt or way to return its feedback to the editor.
        """
        with self.mutex:
            def start(state):
                if state['phase'] != 'FROZEN' or not state['freeze']:
                    raise LimitError('holdout requires exactly one frozen candidate')
                if not self.task.get('synthetic_only'):
                    usage = state.get('agent_usage')
                    if not usage or not state.get('agent_usage_final') or any(type(usage.get(k)) is not int for k in ('input_tokens', 'output_tokens')):
                        raise LimitError('final actual editor usage is required before holdout')
                    completion = state.get('editor_completion')
                    if not completion or completion['status'] != 'HOLDOUT_ELIGIBLE' or completion['usage'] != usage or completion['threshold_tokens'] != self.b['editor_agent_tokens']:
                        raise LimitError('accepted response-boundary completion evidence is required')
                state['phase'] = 'HOLDOUT'
            self.journal.change('holdout_open', {}, start)
            candidate = self.journal.state()['freeze']['candidate']
            try:
                before = self._forward('base', 'holdout', phase='HOLDOUT')
                after = self._forward(candidate, 'holdout', phase='HOLDOUT')
                comparison = {'schema': 'wsp-holdout/1', 'candidate': candidate,
                              'freeze': self.journal.state()['freeze'], 'before': before, 'after': after,
                              'inference_sha256': json_hash(self.task['inference']),
                              'selection': 'development-only; one fixed holdout comparison'}
                json_write(self.root / 'holdout_comparison.private.json', comparison)
                self.journal.change('complete', {'comparison_sha256': sha256(self.root / 'holdout_comparison.private.json')},
                                    lambda state: state.update(phase='COMPLETE'))
                return comparison
            except BaseException as error:
                self.journal.change('holdout_failed', {'error_type': type(error).__name__, 'error': str(error)},
                                    lambda state: state.update(phase='ERROR'))
                raise

    def close_editor_admissions(self, notice):
        # Only the journal lock, never the operation lock: an admitted forward
        # may still run under its existing reservation while new work is closed.
        self.journal.change('editor_admissions_closed', notice,
                            lambda state: state.update(editor_admissions_closed=notice))

    def record_agent_usage(self, usage, source, *, final=False):
        """Trusted cumulative snapshots only; missing counts remain unknown."""
        def record(state):
            previous = state['agent_usage']
            for key in ('input_tokens', 'output_tokens'):
                value = usage.get(key)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError('invalid agent usage')
                if previous and previous.get(key) is not None and (value is None or value < previous[key]):
                    raise ValueError('agent counters cannot reset')
            cached = usage.get('cached_input_tokens')
            if cached is not None and (type(cached) is not int or cached < 0 or usage.get('input_tokens') is None or cached > usage['input_tokens']):
                raise ValueError('cached input must be a subset of total input')
            state['agent_usage'] = usage
            state['agent_usage_final'] = final
        self.journal.change('agent_usage', {'usage': usage, 'source': source}, record)

    def accept_editor_completion(self, events, evidence, *, exit_code, forced_stop=None):
        """Trusted final accounting/policy gate, separate from editor admission."""
        self.check_storage()
        state = self.journal.state()
        completion = response_completion(events, evidence, self.b['editor_agent_tokens'],
                                          frozen=state['phase'] == 'FROZEN' and state['freeze'] is not None,
                                          exit_code=exit_code, forced_stop=forced_stop)
        self.record_agent_usage(completion['usage'], 'captured_raw_response_evidence', final=completion['provider_usage_complete'])
        self.journal.change('editor_completion', completion, lambda state: state.update(editor_completion=completion))
        return completion

    def export(self):
        ledger = self.journal.export()
        json_write(self.root / 'ledger.private.json', ledger)
        state = ledger['state']
        report = {'schema': 'wsp-export/1', 'status': state['phase'], 'separate_from_six_arm_suite': True,
                  'task_sha256': json_hash(self.task), 'split_sha256': self.task['split_sha256'],
                  'base_fingerprint': self.base_manifest['fingerprint'], 'freeze': state['freeze'],
                  'budget_reservations': state['used'], 'known_complete_forward_usage': state['actual'],
                  'failed_forward_costs': {key: row.get('failure_cost') for key, row in state['forwards'].items() if row['status'] != 'COMPLETE'},
                  'agent_usage': state['agent_usage'], 'editor_completion': state.get('editor_completion'),
                  'failed_forward_usage': 'see retained native outputs; missing values remain unknown',
                  'prices_usd': {'agent': None, 'gpu': None}, 'ledger_head_sha256': ledger['head_sha256'],
                  'limitations': ['small Mining-only feasibility subset', 'no claim of paper reproduction',
                                  'observed raw stopping requires reviewed live evidence; interrupted provider usage stays unknown; no zero-overshoot spend ceiling',
                                  'failed requests consume full reservations; actual usage reported separately']}
        if state['phase'] == 'COMPLETE':
            comparison = json.loads((self.root / 'holdout_comparison.private.json').read_text())
            before, after = comparison['before']['metrics'], comparison['after']['metrics']
            report['holdout_metrics'] = {'base': before, 'edited': after,
                                         'mcc_delta': after['mcc'] - before['mcc'] if 'mcc' in before and 'mcc' in after else None}
            base_scored = json.loads((Path(state['forwards']['holdout:base']['directory']) / 'scored.json').read_text())['predictions']
            edited_scored = json.loads((Path(state['forwards']['holdout:' + state['freeze']['candidate']]['directory']) / 'scored.json').read_text())['predictions']
            edited_by_id = {r['id']: r for r in edited_scored}
            changes = []
            for row in base_scored:
                edited = edited_by_id[row['id']]
                changes.append({'id': row['id'], 'truth': row['truth'], 'before': row['prediction'], 'after': edited['prediction'],
                                'correctness_delta': int(edited['prediction'] == row['truth']) - int(row['prediction'] == row['truth'])})
            report['paired_behavior'] = {'gains': sum(r['correctness_delta'] == 1 for r in changes),
                                        'regressions': sum(r['correctness_delta'] == -1 for r in changes), 'rows': changes}
        else:
            report['holdout_metrics'] = None
            report['paired_behavior'] = None
        json_write(self.root / 'report_data.private.json', report)
        return report
