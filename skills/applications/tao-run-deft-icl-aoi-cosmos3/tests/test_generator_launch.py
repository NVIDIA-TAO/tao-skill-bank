# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real driver-to-Docker argv boundary without starting a container."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from generator_driver import GeneratorSuite
from generator_store import GeneratorStore
from suite_driver import APP
from suite_runtime import Docker


@pytest.mark.parametrize('route', ['generator', 'historical'])
def test_actual_driver_launch_contract_has_scoped_localhost_hostname(tmp_path, route):
    suite = GeneratorSuite.__new__(GeneratorSuite)
    suite.root = tmp_path / 'run'
    suite.store = GeneratorStore(suite.root)
    suite.deadline = 10**12
    suite.budget_scope = None
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    generator = {'image': 'fixture-anomalygen@sha256:pinned'}
    for key in ('prepared_package', 'checkpoint', 'recipe', 'base_checkpoint', 'checkpoint_root', 'base_source'):
        path = inputs / key
        if key in ('checkpoint', 'recipe'):
            path.write_text('fixture')
        else:
            path.mkdir()
        generator[key] = str(path)
    suite.task = {'generator': generator, 'framework_image': 'fixture-framework@sha256:pinned',
                  'budgets': {'stage_wall_seconds': 172800, 'generator_wall_seconds': 7200}}
    request = inputs / 'request.json'
    request.write_text('{}')
    job_id = 'fixture-generator-job'
    result = suite.root / 'jobs' / job_id
    dispatched = []

    def record(self, *args):
        if args[0] == 'open':
            return job_id
        if args == ('show', job_id):
            return json.dumps({'results_dir': str(result), 'backend_ref': None})
        pytest.fail('No backend operation should occur in the CPU launch-contract test')

    class BeforeContainerLaunch(Exception):
        pass

    def dispatch(argv, **kwargs):
        assert argv[:2] == ['docker', 'run']
        dispatched.append(argv)
        raise BeforeContainerLaunch

    with patch.object(Docker, 'record', record), \
         patch('suite_runtime.subprocess.run', return_value=SimpleNamespace(returncode=0)), \
         patch('suite_runtime.subprocess.check_output', side_effect=dispatch):
        with pytest.raises(BeforeContainerLaunch):
            if route == 'generator':
                suite.generate(1, request)
            else:
                suite.run_job('D/proxy/base', ['/worker.py'])

    assert len(dispatched) == 1
    argv = dispatched[0]
    assert json.loads((result / 'launch.json').read_text())['argv'] == argv
    assert argv[argv.index('--network') + 1] == 'none'
    if route == 'generator':
        assert argv.count('--hostname') == 1
        assert argv[argv.index('--hostname'):] == [
            '--hostname', 'localhost', '--entrypoint', '/opt/venv/bin/python',
            generator['image'], str(APP / 'generator_worker.py'), '--request', str(request)]
        assert 'TAO_HOST_RESULTS_DIR=' + str(result) in argv
        assert 'type=bind,src=' + str(result) + ',dst=' + str(result) in argv
        assert 'type=bind,src=' + generator['prepared_package'] + ',dst=' + generator['prepared_package'] + ',readonly' in argv
    else:
        assert '--hostname' not in argv
        assert argv[argv.index('--entrypoint') + 1] == '/workspace/.venv/bin/python'
