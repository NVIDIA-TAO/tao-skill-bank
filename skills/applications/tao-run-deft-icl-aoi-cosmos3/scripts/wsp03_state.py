# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable budget admission and explore→shortlist→full-development→freeze boundary."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


class AdmissionError(RuntimeError):
    pass




class Ledger:
    def __init__(self, path, *, track, clock=time.time, probe_debit=None, caps=None,
                 full_size=2917, explore_size=512, total_wall=86400, development_wall=79200,
                 soft_tokens=4500000, target_tokens=5000000):
        self.clock = clock
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=20, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,kind TEXT NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS usage(id TEXT PRIMARY KEY,input INTEGER NOT NULL,output INTEGER NOT NULL,cached INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS results(id TEXT NOT NULL,phase TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(id,phase));
          CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,prev TEXT NOT NULL,payload TEXT NOT NULL,sha TEXT NOT NULL);''')
        debit = probe_debit or {'identity': 'none', 'generation_passes': 0, 'wall_seconds': 0, 'gpu_allocated_seconds': 0, 'input_tokens': 0, 'output_tokens': 0}
        for name in ('generation_passes', 'wall_seconds', 'gpu_allocated_seconds', 'input_tokens', 'output_tokens'):
            if not isinstance(debit.get(name), (int, float)) or not math.isfinite(debit[name]) or debit[name] < 0:
                raise ValueError('invalid probe debit')
        if caps is None:raise ValueError('explicit frozen versioned budget allocation required')
        cap = dict(caps)
        cap.setdefault('baseline', 0)
        if sum(cap[k] for k in ('probe', 'explore', 'full', 'trusted', 'baseline')) > 100 or cap['full'] > 3:
            raise ValueError('authorized generation allocation must be≤100 and finalists≤3; unused capacity is unavailable')
        if debit['generation_passes'] > cap['probe'] or debit['wall_seconds'] >= development_wall:
            raise ValueError('probe exhausted allocation')
        contract = {'track': track, 'caps': cap, 'probe_debit': debit, 'full_size': full_size, 'explore_size': explore_size,
                    'total_wall': total_wall, 'development_wall': development_wall, 'soft_tokens': soft_tokens, 'target_tokens': target_tokens}
        with self.tx():
            old = self.get('contract')
            if old is not None and old != contract:
                raise ValueError('immutable run contract/debit changed')
            if old is None:
                self.set('contract', contract)
                self.set('phase', 'explore')
                self.set('started_at', None)
                self.set('full_reserve_seconds', None)
                self.set('explore_reserve_seconds', 0)
                self.event('initialize', contract)

    @contextmanager
    def tx(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def close(self):
        self.db.close()

    def get(self, key):
        row = self.db.execute('SELECT v FROM meta WHERE k=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', (key, canonical(value)))

    def event(self, kind, data):
        last = self.db.execute('SELECT seq,sha FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        seq, prev = (last[0] + 1, last[1]) if last else (1, '0' * 64)
        payload = canonical({'kind': kind, 'data': data, 'time': self.clock()})
        digest = hashlib.sha256((prev + payload).encode()).hexdigest()
        self.db.execute('INSERT INTO events VALUES(?,?,?,?)', (seq, prev, payload, digest))

    def start_clock(self, started_at=None):
        with self.tx():
            value = self.clock() if started_at is None else started_at
            old = self.get('started_at')
            if old is not None and old != value:
                raise ValueError('cannot reset clock')
            self.set('started_at', value)
            self.event('start', value)

    def set_finalist_reserve(self, per_full_pass_seconds, explore_seconds=0):
        if not all(math.isfinite(v) and v >= 0 for v in (per_full_pass_seconds, explore_seconds)):
            raise ValueError('invalid reserve')
        with self.tx():
            if self.get('full_reserve_seconds') is not None:
                raise ValueError('throughput reserve already frozen')
            self.set('full_reserve_seconds', per_full_pass_seconds)
            self.set('explore_reserve_seconds', explore_seconds)
            self.event('throughput_reserve', {'full': per_full_pass_seconds, 'explore': explore_seconds})

    def status(self):
        c = self.get('contract')
        usage = self.db.execute('SELECT coalesce(sum(input),0),coalesce(sum(output),0),coalesce(sum(cached),0) FROM usage').fetchone()
        used = {kind: n for kind, n in self.db.execute('SELECT kind,count(*) FROM requests GROUP BY kind')}
        used['probe'] = used.get('probe', 0) + c['probe_debit']['generation_passes']
        elapsed = max(0, self.clock() - self.get('started_at')) if self.get('started_at') is not None else 0
        elapsed += c['probe_debit']['wall_seconds']
        total = usage[0] + usage[1]
        return {'phase': self.get('phase'), 'input_tokens': usage[0], 'output_tokens': usage[1], 'cached_tokens': usage[2],
                'observed_tokens': total, 'budget_overrun': total > c['target_tokens'], 'soft_stopped': total >= c['soft_tokens'],
                'used': used, 'remaining_wall_seconds': c['total_wall'] - elapsed,
                'remaining_development_seconds': c['development_wall'] - elapsed,
                'unknown_inflight_usage': [r[0] for r in self.db.execute("SELECT id FROM requests WHERE kind='editor' AND state='running'")],
                'provider_usage_complete': not bool(self.db.execute("SELECT 1 FROM requests WHERE kind='editor' AND state!='complete'").fetchone())}

    def admit(self, kind, request_id, payload=None):
        with self.tx():
            c, status = self.get('contract'), self.status()
            if kind not in set(c['caps']) | {'editor'}:
                raise ValueError('unknown request kind')
            if not request_id or self.db.execute('SELECT 1 FROM requests WHERE id=?', (request_id,)).fetchone():
                raise AdmissionError('request identity already admitted')
            if self.db.execute("SELECT 1 FROM requests WHERE state='running'").fetchone():
                raise AdmissionError('one request in flight; reconcile first')
            if status['remaining_wall_seconds'] <= 0:
                raise AdmissionError('wall exhausted')
            if kind in ('editor', 'python', 'candidate', 'explore', 'read', 'baseline'):
                if status['phase'] != 'explore' or status['soft_stopped']:
                    raise AdmissionError('editing admission closed; trusted finalization remains available')
                if status['remaining_development_seconds'] <= 0:
                    raise AdmissionError('development deadline')
            if kind == 'full':
                if status['phase'] != 'finalists' or status['remaining_development_seconds'] <= 0:
                    raise AdmissionError('full development only for frozen shortlist within development wall')
            if kind == 'trusted' and status['phase'] not in ('frozen', 'no_valid_patch'):
                raise AdmissionError('trusted evaluation requires terminal scientific editing')
            if kind in c['caps'] and status['used'].get(kind, 0) >= c['caps'][kind]:
                raise AdmissionError('request budget exhausted')
            if kind in ('explore', 'baseline', 'read', 'python', 'editor'):
                reserve = self.get('full_reserve_seconds')
                if reserve is not None and status['remaining_development_seconds'] <= 3 * reserve + self.get('explore_reserve_seconds'):
                    raise AdmissionError('preserve full-development finalist reserve')
            self.db.execute('INSERT INTO requests VALUES(?,?,?,?)', (request_id, kind, 'running', canonical(payload or {})))
            self.event('admit', {'kind': kind, 'id': request_id, 'payload': payload or {}})

    def finish(self, request_id, result):
        with self.tx():
            row = self.db.execute('SELECT state,payload FROM requests WHERE id=?', (request_id,)).fetchone()
            if not row or row[0] != 'running':
                raise AdmissionError('unknown or terminal request')
            merged=dict(json.loads(row[1]),**result)
            self.db.execute('UPDATE requests SET state=?,payload=? WHERE id=?', ('complete' if result.get('complete') else 'failed', canonical(merged), request_id))
            self.event('finish', {'id': request_id, 'result': result})

    def usage(self, response_id, input_tokens, output_tokens, cached_tokens):
        if any(type(v) is not int or v < 0 for v in (input_tokens, output_tokens, cached_tokens)) or cached_tokens > input_tokens:
            raise ValueError('invalid native usage')
        values = (input_tokens, output_tokens, cached_tokens)
        with self.tx():
            old = self.db.execute('SELECT input,output,cached FROM usage WHERE id=?', (response_id,)).fetchone()
            if old:
                if old != values:
                    raise ValueError('conflicting repeated native response identity')
                return
            row = self.db.execute('SELECT kind FROM requests WHERE id=?', (response_id,)).fetchone()
            if not row or row[0] != 'editor':
                raise ValueError('usage lacks admitted editor response')
            self.db.execute('INSERT INTO usage VALUES(?,?,?,?)', (response_id, *values))
            self.db.execute("UPDATE requests SET state='complete' WHERE id=?", (response_id,))
            self.event('usage', {'id': response_id, 'input': input_tokens, 'output': output_tokens, 'cached': cached_tokens})

    def candidate(self, candidate_id, manifest):
        with self.tx():
            if self.get('phase') != 'explore' or self.status()['soft_stopped']:
                raise AdmissionError('candidate creation closed')
            if self.db.execute('SELECT count(*) FROM candidates').fetchone()[0] >= self.get('contract')['caps']['candidate']:
                raise AdmissionError('candidate creation budget exhausted')
            if not manifest.get('valid') or manifest.get('changed_elements', 0) <= 0:
                raise ValueError('nonzero validated patch required')
            self.db.execute('INSERT INTO candidates VALUES(?,?)', (candidate_id, canonical(manifest)))
            self.event('candidate', {'id': candidate_id, 'manifest': manifest})

    def result(self, candidate_id, phase, metrics):
        with self.tx():
            c = self.get('contract')
            if phase not in ('explore', 'full'):
                raise ValueError('unknown development phase')
            expected = c['explore_size'] if phase == 'explore' else c['full_size']
            if metrics.get('complete') is not True or metrics.get('n') != expected:
                raise ValueError('partial result cannot enter ranking')
            if not self.db.execute('SELECT 1 FROM candidates WHERE id=?', (candidate_id,)).fetchone():
                raise ValueError('unknown candidate')
            if phase == 'full' and (self.get('phase') != 'finalists' or candidate_id not in self.get('shortlist')):
                raise AdmissionError('not in frozen shortlist')
            if phase == 'explore' and self.get('phase') != 'explore':
                raise AdmissionError('exploration closed')
            for key in ('mcc', 'macro_f1', 'invalid'):
                if not isinstance(metrics.get(key), (int, float)) or not math.isfinite(metrics[key]):
                    raise ValueError('finite metrics required')
            self.db.execute('INSERT INTO results VALUES(?,?,?)', (candidate_id, phase, canonical(metrics)))
            self.event('result', {'id': candidate_id, 'phase': phase, 'metrics': metrics})

    def ranked(self, phase):
        records = []
        for cid, payload, manifest in self.db.execute('SELECT r.id,r.payload,c.payload FROM results r JOIN candidates c ON r.id=c.id WHERE r.phase=?', (phase,)):
            m, c = json.loads(payload), json.loads(manifest)
            records.append(((-m['mcc'], -m['macro_f1'], m['invalid'], c.get('mean_reference_count',c['changed_elements']), cid), cid))
        return [cid for _, cid in sorted(records)]

    def nominate(self):
        with self.tx():
            if self.get('phase') != 'explore' or self.db.execute("SELECT 1 FROM requests WHERE state='running'").fetchone():
                raise AdmissionError('cannot nominate with closed phase or unresolved work')
            selected = self.ranked('explore')[:3]
            self.set('shortlist', selected)
            self.set('phase', 'finalists')
            self.event('shortlist', selected)
            return selected

    def freeze(self):
        with self.tx():
            if self.get('phase') != 'finalists' or self.db.execute("SELECT 1 FROM requests WHERE state='running'").fetchone():
                raise AdmissionError('freeze requires resolved finalist phase')
            ranked = self.ranked('full')
            for cid in self.get('shortlist'):
                attempted=self.db.execute("SELECT 1 FROM requests WHERE kind='full' AND state='failed' AND json_extract(payload,'$.candidate')=?",(cid,)).fetchone()
                if cid not in ranked and not attempted and self.status()['remaining_development_seconds']>0:
                    raise AdmissionError('unresolved finalist must be evaluated before freeze')
            selected = ranked[0] if ranked else None
            self.set('selected', selected)
            self.set('phase', 'frozen' if selected else 'no_valid_patch')
            self.event('freeze', selected)
            return selected
