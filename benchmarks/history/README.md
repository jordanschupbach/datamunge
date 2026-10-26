# Benchmark history

Every successful `just bench-backend` run automatically stores its original JSON
in `runs/<sha256>.json` and a compact summary in `summaries/<sha256>.json`.
Snapshots are content-addressed and written atomically; recording the same file
again is idempotent. Failed fixtures are rejected and optional LAPACK skips are
explicit. Raw archives are gitignored; **commit the compact summaries** to keep
long-term history in the repository. Existing result files remain intact.
CI uploads raw results and summaries for 90 days; committed summaries outlive
artifact retention. No timing regression gate is enabled on shared CI runners.

```sh
# Capture a repeated observation (automatic persistent recording).
just bench-backend --label daily-baseline -- \
  --benchmark_filter='BM_TensorMatmulPartialBatch|BM_Network/MLP' \
  --benchmark_min_time=0.1s --benchmark_repetitions=5

# Import a historical metadata-backed result without rerunning it.
just benchmark-record benchmarks/results/performance/cache_pass_before.json --label cache-pass-before

# TSV output for plotting or spreadsheets, restricted to one compatible profile.
just benchmark-history --case 'BM_TensorMatmulPartialBatch' --profile PROFILE_PREFIX > trend.tsv
```

History includes case names, medians, range, repetition CV, tensor-memory peaks,
optimizer quality/call counters, source revision/dirty hash, actual binary hashes,
and compiler/backend/thread configuration. Each summary has a `profile` hash for
compatible machine/build/thread settings. Compare within a profile and within the
same fixture/source family; different shapes are separate time series.

By default, history shows only runs with at least three raw repetitions per case
and an explicit minimum time of at least 0.1 seconds. `1x` runs are classified as
smoke checks; other short/single runs are exploratory. Use `--include-smoke` to
inspect those records. Smoke timing is for execution checks, not trend claims.
The full raw file remains available for detailed before/after comparisons using
`just benchmark-compare`. Local timings can still vary with host load and power.
