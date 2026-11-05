#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate WSP preparation/runner CLI. No six-arm suite state is consumed."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    views = commands.add_parser('prepare-views', help='CPU-only, Mining-only grouped views in a new directory')
    views.add_argument('--public', required=True)
    views.add_argument('--exclusions', required=True, help='two-joint identity-only JSON projection')
    views.add_argument('--output', required=True)
    views.add_argument('--seed', type=int, default=42)
    views.add_argument('--development-groups', type=int, default=16)
    views.add_argument('--holdout-groups', type=int, default=16)
    preflight = commands.add_parser('preflight', help='CPU-only proposed preflight; no tensor payload read')
    for name in ('model', 'views', 'direct-profile', 'evaluator', 'output', 'run-dir'):
        preflight.add_argument('--' + name, required=True)
    verify = commands.add_parser('verify-preparation', help='read-only code/view fingerprints; no real weight values')
    verify.add_argument('--preflight', required=True)
    run = commands.add_parser('run', help='FUTURE real experiment; requires review and primary-GPU release')
    for name in ('preflight', 'approval', 'gpu-release', 'usage-proof', 'codex-binary', 'auth-file'):
        run.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.command == 'prepare-views':
        from wsp_prepare import prepare_views
        split = prepare_views(args.public, args.exclusions, args.output, seed=args.seed,
                              development_groups=args.development_groups, holdout_groups=args.holdout_groups)
        # Never print holdout rows, gold, class counts or future private scores.
        print(json.dumps({'split_sha256': split['split_sha256'], 'counts': {k: v['images'] for k, v in split['roles'].items()},
                          'excluded_images': split['excluded_images'], 'unused_eligible_images': split['unused_eligible_images']}))
    elif args.command == 'preflight':
        from wsp_prepare import proposed_preflight
        result = proposed_preflight(model=args.model, views=args.views, direct_profile=args.direct_profile,
                                    evaluator=args.evaluator, output=args.output, run_dir=args.run_dir)
        print(json.dumps({'status': result['task']['status'], 'preflight_sha256': result['preflight_sha256'],
                          'tensors': len(result['task']['header_catalog']['tensors']), 'real_weight_values_read': False}))
    elif args.command == 'verify-preparation':
        from wsp_runtime import load_preflight, verify_views
        result = load_preflight(args.preflight)
        verify_views(result['task'])
        print(json.dumps({'verified': True, 'preflight_sha256': result['preflight_sha256'], 'execution_authorized': False}))
    else:
        from wsp_runtime import run_approved
        result = run_approved(args.preflight, args.approval, args.gpu_release,
                              usage_proof=args.usage_proof, codex_binary=args.codex_binary, auth_file=args.auth_file)
        print(json.dumps({'status': result['status'], 'task_sha256': result['task_sha256'], 'private_export_written': True}))


if __name__ == '__main__':
    main()
