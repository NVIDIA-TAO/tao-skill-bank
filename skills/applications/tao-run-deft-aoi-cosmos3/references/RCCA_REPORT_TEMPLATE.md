# RCCA report template

## Executive Summary

State the Proxy coverage, incorrect/parse-failure counts, and selected real
Mining targets. Proxy does not gate the loop.

## Failure Mode Analysis

Summarize weaknesses by task type, evaluator family, reference cohort, and
dataset. Link claims to `gap_candidates.parquet` rows.

## Root Cause Analysis

Separate observed evidence from hypotheses. Identify prompt, visual,
localization, reference-comparison, or data-coverage causes.

## Corrective Actions

List the routed Mining targets, task-aware quotas, similarity policy, history
deduplication, and expected training-data contribution.

## Validation Plan

Require split isolation, canonical JSONL validation, complete Framework DCP,
exact prediction coverage, and the frozen Benchmark F1 gate.
