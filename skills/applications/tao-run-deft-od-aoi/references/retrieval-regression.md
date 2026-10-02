# Retrieval regression plan

Use this plan when changing `retrieval.preprocessing.profile` or
`retrieval.selection.strategy`. The two axes are independent and every test or
acceptance run must name both values.

## Preserved behavioral references

- `test_prepare_deft_od_aoi_retrieval.py` covers all four configuration pairs:
  `tight_context`/`square_context` crossed with
  `max_similarity`/`round_robin_similarity`.
- `fixtures/round_robin_similarity_reference.json` freezes the semantic parent
  order from historical commit `7ebdfbb`; the corresponding unit test ignores
  scheduler and path-layout details and compares query, candidate, and parent
  identities exactly.
- `dev/deft-commercial-round-robin-paired-artifact` remains the full four-arm,
  four-iteration behavioral reference. Its orchestration code is evidence, not
  product code.
- `dev/deft-embedding-performance-ablation` preserves the current-versus-square
  embedding experiments, including the square-context plus max-similarity
  follow-up.

## Acceptance layers

1. Run the application unit suite. Require exact crop dimensions, profile and
   strategy manifests, pocket quotas, stable tie handling, exclusions, parent
   deduplication, and the frozen historical semantic selection.
2. Run a deterministic CPU replay over frozen candidate/query embeddings.
   Compare selected `(query_id, candidate_id, source_filepath)` rows exactly;
   compare cosine similarity with absolute tolerance `1e-6`. Record row counts,
   duplicate-parent counts, shortlist depth, quota requested/admitted, and
   per-pocket shortfall.
3. Run a GPU embedding replay over a frozen image subset for all four arms.
   Record crop count and size distribution, embedding count/dimension,
   non-finite or zero-norm rows, wall time, images/second, and peak memory.
   Candidate and query banks must use the same preprocessing profile.
4. Run the four-arm DEFT trajectory on frozen roles, checkpoint, seed, factor,
   and training budget. Hold synthesis, hardware shape, dataloader workers,
   and all non-retrieval policy values constant. Persist one typed artifact set
   per arm and iteration.

## Full-run comparison metrics

Primary model metrics:

- KPI `val_mAP50` used for checkpoint selection at every iteration;
- sealed-test `mAP50`, report-only, for every selected checkpoint;
- iteration-to-best KPI, final-minus-baseline delta, and area under the
  per-iteration KPI trajectory.

Routing and dose metrics:

- strict FN, near-miss FP, and background FP query counts;
- requested and admitted unique real/clean parent images;
- per-pocket quota fulfillment and shortfall;
- prior-source exclusions, duplicate-parent rate, and cumulative pool size.

Retrieval diagnostics:

- selected similarity minimum, median, p10, and p90;
- number and share of queries represented by at least one admitted parent;
- selected rank-depth distribution and candidates consumed per admitted parent;
- per-pocket admitted-parent distribution and clean-to-real cumulative ratio.

Operational metrics:

- candidate/query preprocessing seconds, embedding seconds and throughput;
- selection wall time and peak resident memory;
- training and evaluation wall time, while confirming identical GPU, epoch,
  batch-size, worker, seed, and checkpoint inputs across arms.

Do not judge a strategy only by iteration-one parent count or by test mAP50.
Require exact deterministic replay first, then review KPI trajectory, sealed-test
trajectory, pocket coverage, dose parity, and runtime together. Scheduler IDs,
timestamps, and result-root prefixes are excluded from semantic comparisons.
