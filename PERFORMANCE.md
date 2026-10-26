# Performance plan, results, and TensorFlow comparisons

2026-10-24. This is the single performance planning and comparison document.
Current conclusions are in the first sections. All TensorFlow/native per-case
comparisons and the complete earlier planning/results record are preserved below.
Raw JSON/TSV, immutable comparison snapshots and content-addressed history remain separate evidence, not parallel
plans. New TensorFlow runs in the standard results directory automatically refresh
the full comparison-table section without overwriting the human-written plan.

## Completed pass and next priorities

1. **Completed:** expanded cross-framework coverage from 12 operation families /
   72 fixtures to **35 families / 164 Float32/Float64 fixtures**. C++ and Python
   read the same `benchmarks/tensorflow_cases.def` catalog.
2. **Completed:** checked every native/eager/graph output before collecting a
   five-repetition, 0.1-second-minimum Release baseline; froze the native library.
3. **Completed:** optimized measured max, activation, transpose/binary and
   normalization bottlenecks; fused ordinary softmax VJP while preserving
   differentiable composition fallbacks and double accumulation.
4. **Completed:** repeated the full comparison, confirmed native improvements
   against the frozen core in alternating adjacent order, and validated accelerated
   and portable correctness/fixture matrices.
5. **Completed:** consolidated the eight earlier plans, five TensorFlow result
   writeups, container audit and historical benchmark notes into this file.
6. **Next, measured priorities:** investigate the remaining gaps listed below,
   including accurate Float64 GELU/scalar erf and workload-specific reduction or
   convolution costs. Require repeated matched controls before changing dispatch.
7. **Next, coverage priorities:** rectangular/ill-conditioned linear algebra,
   additional convolution shapes and filter gradients, non-unit affine norm
   parameters/training gradients, real model/data distributions, and matched
   multi-thread scaling. Native optimizer updates already have broad coverage;
   cross-TensorFlow optimizer comparison needs an agreed state/reset/resource-read
   scope first. GPU/binding overhead and RL remain separate, unmeasured work.

## What the expanded comparison measures

| Family | Size parameter | Inputs and result scope |
|---|---|---|
| Original 12 families | 32/512/8192 × 128; GEMM orders 64/256/512 | Add, broadcast add, multiply, row/global sum, mean, ReLU, exp, softmax/log-softmax, square-loss VJP and square GEMM |
| Reductions | 32/8192 × 128 | Column sum, row maximum, row population variance and log-sum-exp |
| Activations/layout/indexing | 32/8192 × 128 | Stable sigmoid, tanh, exact-mode GELU, gather every third row from a fixed index tensor, and transpose-plus-add |
| GEMM and gradients | Orders 64/512; batch 4 at 16/128; dense rows 32/256 | AᵀA; batched square GEMM; gradient of sum((XW)²) for W, X [N,64], W [64,32] |
| Convolution and pooling | Batch 1/16 | NHWC [N,16,16,4], valid 3×3×4×8 convolution, input gradient of summed outputs, valid 2×2 stride-2 max/average pooling |
| Normalization | 32/8192 × 128 | Last-axis layer norm and column batch norm, epsilon 1e-5, unit scale/zero bias |
| Attention | 16/128 tokens | Two batches, depth 32; softmax(QQᵀ/√32)Q, no masks/dropout |
| FFT features | Length 64/2048, batch 8 | Real FFT power, real²+imag²; produces the same real feature output despite different complex storage layouts |
| Decompositions | Orders 8/64 | Cholesky and four-RHS solve on independently constructed, well-conditioned SPD matrices |
| Additional backward | 32/8192 × 128 | Gradient of summed squared softmax probabilities |

Each new family has small and large scales in both dtypes; original case names and
sizes remain available. `rows` is a common case-name field: it means batch for
convolution/pooling, order for decompositions, length for FFT and tokens for
attention. It is not universally the first matrix dimension.

Inputs use the deterministic nonconstant sequence `(i % 251 - 125) / 128`, with
explicit SPD construction for decompositions and fixed default affine parameters
for norms. C++ references use scalar loops/DFT/Cholesky substitutions; TensorFlow
outputs are checked against independent NumPy/math references. No benchmark accepts
one framework's output as the other's oracle. Shape and full-output checks are
untimed. `--validate-only` checks the complete triple and records smoke evidence,
not performance claims.

Float64 tolerance is 1e-10 except GELU at 1e-8: the pinned TensorFlow built-in
exact-mode GELU showed about 4e-9 deviation from the scalar erf reference. Float32
uses 2e-5, with 2e-4 for FFT power and matrix-loss gradients. Raw new TensorFlow
records include tolerance and maximum absolute reference error. Strict native unit
and higher-derivative tests provide additional numerical checks. Native Float32
sums retain double accumulation; FFT computes internally in double. No global
fast-math or lower-precision summation policy was introduced.

## Measurement and fairness contract

- Native core and driver are verified optimized Release (`-O3 -DNDEBUG` locally),
  with runtime build tags and loaded-library SHA-256 checked. The earlier
  benchmark was already Release; reported improvements are code changes.
- Pinned TensorFlow 2.21.0, synchronous CPU execution, one compute thread and
  BLAS/inter-op/intra-op limits. No GPU or XLA. Local host: ARM64 macOS.
- Five repeated samples with an explicit 0.1-second minimum calibration. Paired
  framework calls alternate native-first/native-last per fixture. Inputs, imports,
  tracing, warmup, validation and host copies are outside timing; operation and
  output allocations are included.
- TensorFlow eager and warmed `tf.function` are separate targets. Graphs may fuse
  compositions or prune forward loss work when only a gradient is returned.
  Transpose happens inside transpose-plus-add on both sides. Norms use default
  affine values: TF compositions can elide identity arithmetic while native reads
  its stored scale/bias. Native batch norm also creates its public statistics.
- These are warm user-call comparisons, not isolated TensorFlow C++ kernel
  timings. Python/graph dispatch dominates tiny cases. Counts of wins are not a
  suite-wide speed score or a portable guarantee. Native tensor-storage peaks
  exclude STL/node metadata and are not a TensorFlow memory comparison.
- Host load/CV and source, driver, case-catalog, fixture-factory and loaded-library
  hashes are retained. Older disturbed/superseded runs below remain historical
  evidence, not current conclusions; absolute times from different phases must
  not be treated as causal speedups without matched frozen-core confirmation.

## Expanded baseline and final comparison

The baseline had 147/164 eager and 148/164 graph wins. The final full run had **144/164 eager and 150/164 graph wins**. These counts include tiny-call overhead and host variation.

The full-run Float32 mean cell had **55.4% native CV** and is not treated as a stable regression. Its focused repeat measured **61.47 µs native / 50.00 µs eager / 109.41 µs graph** (native CV 0.063%), returning the established ~1.23× eager gap. The original noisy row remains in its full raw table. The ranked view below substitutes the explicitly labeled Mean/SumAll repeat and excludes cells with native/eager CV above 10%; it is not a synthesized new full run.

Largest remaining stable native/eager ratios (above 1 means slower):

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph |
|---|---:|---:|---:|---:|---:|
| `Gelu/rows:8192/f64:1/real_time` | 3932.148 | 1352.937 | 1284.270 | 2.906 | 3.062 |
| `Gelu/rows:8192/f64:0/real_time` | 1682.207 | 585.393 | 586.035 | 2.874 | 2.870 |
| `Tanh/rows:8192/f64:0/real_time` | 546.957 | 212.407 | 284.354 | 2.575 | 1.924 |
| `Conv2d/rows:16/f64:0/real_time` | 86.319 | 53.026 | 120.846 | 1.628 | 0.714 |
| `Conv2d/rows:16/f64:1/real_time` | 117.606 | 72.463 | 139.667 | 1.623 | 0.842 |
| `MaxLast/rows:8192/f64:1/real_time` | 130.882 | 83.588 | 154.865 | 1.566 | 0.845 |
| `Tanh/rows:8192/f64:1/real_time` | 1146.321 | 785.666 | 860.502 | 1.459 | 1.332 |
| `Variance/rows:8192/f64:0/real_time` | 489.812 | 344.576 | 393.832 | 1.421 | 1.244 |
| `MaxLast/rows:8192/f64:0/real_time` | 69.244 | 50.291 | 120.819 | 1.377 | 0.573 |
| `Sigmoid/rows:8192/f64:1/real_time` | 1345.036 | 1056.905 | 1131.965 | 1.273 | 1.188 |
| `Sigmoid/rows:8192/f64:0/real_time` | 484.412 | 385.507 | 456.377 | 1.257 | 1.061 |
| `Mean/rows:8192/f64:0/real_time` | 61.470 | 50.001 | 109.414 | 1.229 | 0.562 |

Every case, including cases without a native win, appears in the full tables below.
Do not infer that a gap is closed just because an eager ratio improved: graph
ratios and uncertainty are separate columns.

## Retained implementation changes

- Floating row maxima reuse NaN-propagating vector max instead of a serial
  comparator dependency. Floating max/min identities now correctly handle all
  -infinity/+infinity groups; tie gradients remain shared correctly.
- Larger contiguous sigmoid/tanh use the existing Accelerate vector math backend;
  sigmoid reuses its output as bounded-magnitude exponential scratch. Tiny/strided
  inputs retain scalar paths; non-Accelerate sigmoid retains a single scalar loop.
- Exact GELU without recording fuses the pointwise expression into one output.
  Recorded first/higher derivatives keep the original expression. Approximate
  GELU remains a distinct option; no approximation mode was silently changed.
- No-gradient, contiguous, dtype-matched layer norm fuses row statistics and
  affine writes into one output buffer. Intermediate products/means still round
  in the tensor dtype, with double reduction accumulation. Recorded, mixed-dtype
  and strided inputs retain the original composition.
- Ordinary contiguous last-axis softmax VJP computes its dot and final gradient
  without full-size multiply/subtract intermediates. Higher-order and strided
  seed paths retain differentiable operations; saved-output mutation checks remain.
- Large compatible transposed binary matrices use 16×32 cache tiles. Work spans
  both tile dimensions to retain parallelism for thin matrices; offsets, padded
  strides, noncommutative operands, bool outputs and odd edges are preserved.

Frozen-core confirmation (same Release driver, alternating order per fixture):

| Case | Frozen before µs | Final µs | Time reduction | Speedup |
|---|---:|---:|---:|---:|
| `Add/rows:8192/f64:0/real_time` | 68.244 | 68.292 | -0.1% | 1.00× |
| `Add/rows:8192/f64:1/real_time` | 136.887 | 137.030 | -0.1% | 1.00× |
| `Gelu/rows:8192/f64:0/real_time` | 1907.857 | 1490.501 | 21.9% | 1.28× |
| `Gelu/rows:8192/f64:1/real_time` | 4373.815 | 4087.742 | 6.5% | 1.07× |
| `LayerNorm/rows:8192/f64:0/real_time` | 1243.764 | 289.635 | 76.7% | 4.29× |
| `LayerNorm/rows:8192/f64:1/real_time` | 1694.757 | 452.929 | 73.3% | 3.74× |
| `LogSumExp/rows:8192/f64:0/real_time` | 1921.078 | 784.246 | 59.2% | 2.45× |
| `LogSumExp/rows:8192/f64:1/real_time` | 2700.027 | 1588.884 | 41.2% | 1.70× |
| `MaxLast/rows:8192/f64:0/real_time` | 1181.321 | 69.047 | 94.2% | 17.11× |
| `MaxLast/rows:8192/f64:1/real_time` | 1181.873 | 130.837 | 88.9% | 9.03× |
| `Multiply/rows:8192/f64:0/real_time` | 68.179 | 68.112 | 0.1% | 1.00× |
| `Multiply/rows:8192/f64:1/real_time` | 136.695 | 136.875 | -0.1% | 1.00× |
| `Sigmoid/rows:8192/f64:0/real_time` | 1304.868 | 483.769 | 62.9% | 2.70× |
| `Sigmoid/rows:8192/f64:1/real_time` | 1860.381 | 1345.061 | 27.7% | 1.38× |
| `SoftmaxGradient/rows:8192/f64:0/real_time` | 1350.174 | 976.921 | 27.6% | 1.38× |
| `SoftmaxGradient/rows:8192/f64:1/real_time` | 2821.185 | 2272.365 | 19.5% | 1.24× |
| `Tanh/rows:8192/f64:0/real_time` | 1468.533 | 547.030 | 62.7% | 2.68× |
| `Tanh/rows:8192/f64:1/real_time` | 2432.907 | 1147.527 | 52.8% | 2.12× |
| `TransposeAdd/rows:8192/f64:0/real_time` | 1200.504 | 450.375 | 62.5% | 2.67× |
| `TransposeAdd/rows:8192/f64:1/real_time` | 1248.387 | 492.726 | 60.5% | 2.53× |

Large native peak tensor payload also fell: exact GELU **80%**, inference layer
norm **about 75%**, and softmax-loss gradient **about 33%**, in both dtypes. Raw
counters retain byte values. These are native payload counters, not process RSS,
cache-miss rates, or cross-framework memory measurements.

## Validation and evidence for this pass

- **541** Accelerate correctness tests; **539** built-in passes with two expected
  LAPACK-only skips. New cases cover vector thresholds/tails, offsets/strides,
  nonfinite/signed-zero activations, precise no-grad normalization/GELU parity,
  VJP seed layout/higher derivatives, extrema identities/ties, and tiled thin/odd
  matrices in one/four threads.
- **164** native/eager/graph validation triples passed before and after.
- **1,568** accelerated fixtures; built-in **1,440 executed / 128 expected
  optional LAPACK skips**. Smoke timings are excluded from speedup claims.
- **29** Python runner/report contract tests; whitespace checks; normal Release
  `just build` completed. The prior optional sanitizer startup limitation is
  preserved in the container audit archive below; no sanitizer pass is claimed.
- `expanded_before_{native,tf}.json` and `expanded_after_{native,tf}.json` under
  `benchmarks/results/tensorflow/`: full calibrated observations, with comparison
  JSON/TSV. The original working tree was dirty; hashes identify the snapshot,
  rather than presenting it as a clean committed baseline.
- `expanded_reduction_repeat_{native,tf}.json`: same-image follow-up of the noisy Float32 mean/global-sum cells.
- `expanded_optimization_{before,after}_native.json`: adjacent frozen confirmation;
  runtime proof links to the baseline's Release provenance.
- `expanded_validation.json`, `expanded_after_validation.json` and matching native
  files: untimed validation records. `expanded_final_matrix.json` and
  `expanded_final_portable_matrix.json`: full native fixture checks.

## Run and maintain

```sh
just build
just bench-backend-check
just bench-tensorflow --validate-only --output benchmarks/results/tensorflow/local-validation.json
just bench-tensorflow --paired-native --label local-expanded \
  --native benchmarks/results/tensorflow/local-native.json \
  --output benchmarks/results/tensorflow/local-tf.json
# Direct Python invocation preserves shell regex quoting for targeted runs:
build/tensorflow-venv/bin/python scripts/benchmark_tensorflow.py --paired-native \
  --case '^(MaxLast|Sigmoid|Tanh|LayerNorm|SoftmaxGradient)/' \
  --native benchmarks/results/tensorflow/local-target-native.json \
  --output benchmarks/results/tensorflow/local-target-tf.json
python3 scripts/performance_report.py
```

TensorFlow is optional and isolated in `build/tensorflow-venv`; install the pinned
`benchmarks/tensorflow-requirements.txt`. The shared `.def` catalog drives sizes,
C++ registration and Python selection; the coverage manifest and native smoke
selection also cover these fixtures. Add references and validation before adding
a timing case. `scripts/performance_report.py` deterministically refreshes all
archived per-case tables; successful standard-directory TF runs invoke it
automatically. It preserves manually written planning/history sections.

<!-- tensorflow-comparisons:start -->

<a id="tensorflow-comparisons"></a>

## All TensorFlow / in-house comparison tables

Every archived comparison is included below, including superseded and exploratory runs. Use the dated results above for conclusions; do not combine different builds, fixture sets or noisy historical runs into one score. Ratios above 1 mean native took longer. Times are medians in microseconds; CV columns are percentages. Immutable snapshots retain older comparisons even when default output filenames are reused.
**17 runs; 1049 per-case comparisons.**

<details><summary>expanded_after_tf — 164 cases — e5a686c61138</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/e5a686c61138949b2f431c039a19615c99edcab65eca378a7cf65d8124e2df24.json). Run date: `2026-10-24T21:02:02.850448+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:32/f64:0/real_time` | 2.24176 | 53.901 | 328.279 | 0.0415904 | 0.00682883 | 12.6701 | 1.53431 | 1.16101 |
| `Add/rows:32/f64:1/real_time` | 3.3058 | 53.0515 | 242.766 | 0.0623129 | 0.0136172 | 14.9366 | 11.8498 | 10.9216 |
| `Add/rows:512/f64:0/real_time` | 26.1299 | 75.3475 | 359.924 | 0.346792 | 0.0725983 | 6.36868 | 13.4559 | 6.09291 |
| `Add/rows:512/f64:1/real_time` | 38.5853 | 103.803 | 328.898 | 0.371716 | 0.117317 | 17.9419 | 8.5248 | 4.7534 |
| `Add/rows:8192/f64:0/real_time` | 492.825 | 592.284 | 1025.34 | 0.832076 | 0.480647 | 3.61718 | 3.30441 | 1.91082 |
| `Add/rows:8192/f64:1/real_time` | 1066.55 | 1225.88 | 1695.09 | 0.870033 | 0.629202 | 1.84395 | 1.81878 | 2.04585 |
| `Attention/rows:128/f64:0/real_time` | 86.8973 | 137.244 | 154.111 | 0.633157 | 0.56386 | 0.138271 | 0.151692 | 0.345554 |
| `Attention/rows:128/f64:1/real_time` | 151.045 | 211.365 | 229.36 | 0.714618 | 0.658551 | 0.586256 | 0.0659245 | 0.80618 |
| `Attention/rows:16/f64:0/real_time` | 3.94023 | 60.6356 | 73.7858 | 0.0649821 | 0.0534009 | 0.559583 | 2.53695 | 0.180114 |
| `Attention/rows:16/f64:1/real_time` | 5.29607 | 62.2451 | 75.3633 | 0.0850841 | 0.0702738 | 0.219775 | 0.0855602 | 0.438189 |
| `AvgPool/rows:1/f64:0/real_time` | 0.664639 | 9.32414 | 72.3131 | 0.0712816 | 0.00919113 | 2.45301 | 0.2103 | 0.348384 |
| `AvgPool/rows:1/f64:1/real_time` | 0.945923 | 9.30821 | 71.8506 | 0.101622 | 0.0131652 | 2.94564 | 0.174577 | 1.49298 |
| `AvgPool/rows:16/f64:0/real_time` | 7.11379 | 21.96 | 89.0908 | 0.323943 | 0.0798488 | 0.789243 | 0.058209 | 0.232672 |
| `AvgPool/rows:16/f64:1/real_time` | 11.3688 | 22.478 | 90.0161 | 0.505777 | 0.126298 | 1.18857 | 0.141987 | 4.78708 |
| `BatchMatmul/rows:128/f64:0/real_time` | 16.0625 | 196.436 | 264.923 | 0.0817695 | 0.0606307 | 0.320078 | 0.134717 | 0.155966 |
| `BatchMatmul/rows:128/f64:1/real_time` | 42.7952 | 380.483 | 447.254 | 0.112476 | 0.0956843 | 0.180392 | 1.69358 | 0.906945 |
| `BatchMatmul/rows:16/f64:0/real_time` | 1.68809 | 9.51459 | 71.9256 | 0.177422 | 0.02347 | 4.65913 | 0.236142 | 0.159173 |
| `BatchMatmul/rows:16/f64:1/real_time` | 1.73581 | 9.93888 | 72.2908 | 0.174648 | 0.0240115 | 0.28279 | 0.0724565 | 1.36589 |
| `BatchNorm/rows:32/f64:0/real_time` | 3.90611 | 161.868 | 75.2271 | 0.0241314 | 0.0519243 | 0.303315 | 0.243776 | 5.89382 |
| `BatchNorm/rows:32/f64:1/real_time` | 5.19656 | 171.958 | 79.0815 | 0.0302198 | 0.0657114 | 0.253404 | 0.66006 | 0.366858 |
| `BatchNorm/rows:8192/f64:0/real_time` | 762.466 | 1340.11 | 1227.1 | 0.568956 | 0.621356 | 2.41894 | 0.126089 | 1.21139 |
| `BatchNorm/rows:8192/f64:1/real_time` | 1114.28 | 2731.19 | 2531.53 | 0.407984 | 0.44016 | 0.19491 | 0.527918 | 0.610115 |
| `BroadcastAdd/rows:32/f64:0/real_time` | 2.75954 | 56.0018 | 320.032 | 0.0492759 | 0.00862268 | 2.83956 | 2.05594 | 2.5963 |
| `BroadcastAdd/rows:32/f64:1/real_time` | 3.88701 | 61.6139 | 299.883 | 0.0630867 | 0.0129618 | 9.87674 | 3.51499 | 12.0593 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 35.2221 | 107.959 | 398.776 | 0.326254 | 0.0883256 | 12.457 | 1.51458 | 6.36686 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 56.2467 | 139.005 | 422.312 | 0.404639 | 0.133187 | 10.0118 | 12.0108 | 14.5517 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 566.132 | 867.461 | 1573.13 | 0.652631 | 0.359876 | 12.7742 | 14.5942 | 1.82829 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 1021.02 | 1825.05 | 2500.25 | 0.559451 | 0.408368 | 5.43035 | 5.97828 | 10.3812 |
| `Cholesky/rows:64/f64:0/real_time` | 6.36232 | 13.0882 | 85.9804 | 0.486112 | 0.0739974 | 0.0796083 | 0.143871 | 0.135373 |
| `Cholesky/rows:64/f64:1/real_time` | 5.35353 | 14.8646 | 85.7924 | 0.360154 | 0.0624009 | 1.06833 | 0.0877566 | 0.35579 |
| `Cholesky/rows:8/f64:0/real_time` | 0.450552 | 5.64946 | 69.8075 | 0.0797513 | 0.00645421 | 0.85111 | 2.4724 | 0.279686 |
| `Cholesky/rows:8/f64:1/real_time` | 0.280212 | 5.62311 | 70.2241 | 0.0498322 | 0.00399026 | 0.835519 | 0.199174 | 1.33591 |
| `Conv2d/rows:1/f64:0/real_time` | 7.79809 | 13.2933 | 74.6872 | 0.586619 | 0.10441 | 1.56107 | 0.470922 | 1.36749 |
| `Conv2d/rows:1/f64:1/real_time` | 10.6102 | 14.4938 | 75.8905 | 0.732052 | 0.139809 | 1.94408 | 0.147692 | 6.50024 |
| `Conv2d/rows:16/f64:0/real_time` | 86.3186 | 53.0263 | 120.846 | 1.62784 | 0.714286 | 0.93656 | 0.101241 | 4.30311 |
| `Conv2d/rows:16/f64:1/real_time` | 117.606 | 72.4628 | 139.667 | 1.62299 | 0.84205 | 1.87417 | 0.0820476 | 0.181881 |
| `ConvInputGradient/rows:1/f64:0/real_time` | 15.7318 | 151.348 | 77.1709 | 0.103944 | 0.203857 | 0.948818 | 0.137219 | 0.487914 |
| `ConvInputGradient/rows:1/f64:1/real_time` | 26.6331 | 159.669 | 81.5141 | 0.166801 | 0.32673 | 0.3669 | 0.164894 | 0.411964 |
| `ConvInputGradient/rows:16/f64:0/real_time` | 164.165 | 373.174 | 152.719 | 0.439915 | 1.07495 | 0.8002 | 0.523012 | 0.513428 |
| `ConvInputGradient/rows:16/f64:1/real_time` | 217.615 | 461.994 | 174.972 | 0.471034 | 1.24371 | 0.581173 | 0.328956 | 0.370754 |
| `Exp/rows:32/f64:0/real_time` | 1.42298 | 7.21334 | 72.3642 | 0.197271 | 0.0196642 | 0.648919 | 0.17112 | 1.50215 |
| `Exp/rows:32/f64:1/real_time` | 4.17094 | 9.12043 | 74.151 | 0.457319 | 0.0562493 | 1.95126 | 0.121789 | 0.240694 |
| `Exp/rows:512/f64:0/real_time` | 21.0284 | 30.1394 | 99.2198 | 0.697705 | 0.211938 | 0.0483286 | 0.154243 | 0.204101 |
| `Exp/rows:512/f64:1/real_time` | 65.0421 | 60.5463 | 129.657 | 1.07425 | 0.501647 | 0.102871 | 0.104116 | 0.101983 |
| `Exp/rows:8192/f64:0/real_time` | 334.097 | 392.963 | 465.738 | 0.8502 | 0.717351 | 0.0924186 | 0.352036 | 0.197807 |
| `Exp/rows:8192/f64:1/real_time` | 1038.7 | 878.777 | 952.169 | 1.18199 | 1.09088 | 0.0769888 | 0.12728 | 0.21501 |
| `FftPower/rows:2048/f64:0/real_time` | 36.2968 | 75.9681 | 92.0047 | 0.47779 | 0.39451 | 0.215076 | 3.87872 | 1.20812 |
| `FftPower/rows:2048/f64:1/real_time` | 37.9989 | 90.334 | 105.339 | 0.420649 | 0.360729 | 0.413756 | 0.165178 | 0.180853 |
| `FftPower/rows:64/f64:0/real_time` | 1.64728 | 59.3371 | 73.3559 | 0.0277614 | 0.022456 | 0.45319 | 0.200468 | 0.183631 |
| `FftPower/rows:64/f64:1/real_time` | 1.6663 | 60.7764 | 73.099 | 0.0274169 | 0.0227951 | 0.333879 | 0.256731 | 5.49019 |
| `GatherRows/rows:32/f64:0/real_time` | 0.340562 | 12.0975 | 69.7635 | 0.0281514 | 0.00488167 | 2.58203 | 2.40828 | 0.295209 |
| `GatherRows/rows:32/f64:1/real_time` | 0.411245 | 12.1811 | 69.8314 | 0.0337609 | 0.00588911 | 0.979802 | 0.162025 | 0.345748 |
| `GatherRows/rows:8192/f64:0/real_time` | 27.4908 | 34.3528 | 98.6751 | 0.800249 | 0.278599 | 0.935815 | 0.337556 | 0.336442 |
| `GatherRows/rows:8192/f64:1/real_time` | 51.1012 | 56.0651 | 120.904 | 0.911463 | 0.422661 | 3.71458 | 0.638207 | 0.378243 |
| `Gelu/rows:32/f64:0/real_time` | 6.61059 | 53.1027 | 73.164 | 0.124487 | 0.090353 | 1.68731 | 0.586008 | 0.248248 |
| `Gelu/rows:32/f64:1/real_time` | 15.0608 | 62.8514 | 75.7792 | 0.239625 | 0.198745 | 0.72999 | 0.203532 | 0.394831 |
| `Gelu/rows:8192/f64:0/real_time` | 1682.21 | 585.393 | 586.035 | 2.87364 | 2.87049 | 2.68 | 0.119091 | 0.0834407 |
| `Gelu/rows:8192/f64:1/real_time` | 3932.15 | 1352.94 | 1284.27 | 2.90638 | 3.06178 | 2.08619 | 3.79826 | 0.81349 |
| `LayerNorm/rows:32/f64:0/real_time` | 1.2776 | 161.326 | 74.9968 | 0.00791937 | 0.0170354 | 0.360666 | 0.248978 | 1.12167 |
| `LayerNorm/rows:32/f64:1/real_time` | 1.88598 | 171.402 | 78.0901 | 0.0110033 | 0.0241513 | 0.39546 | 0.0953826 | 0.413772 |
| `LayerNorm/rows:8192/f64:0/real_time` | 290.556 | 809.227 | 694.844 | 0.359054 | 0.41816 | 0.0830899 | 0.880769 | 0.151126 |
| `LayerNorm/rows:8192/f64:1/real_time` | 451.44 | 1510.24 | 1357.49 | 0.298919 | 0.332554 | 0.215406 | 3.63837 | 0.184523 |
| `LogSoftmax/rows:32/f64:0/real_time` | 2.56039 | 9.97935 | 74.1854 | 0.256569 | 0.0345134 | 0.204512 | 0.184432 | 0.184394 |
| `LogSoftmax/rows:32/f64:1/real_time` | 6.6728 | 13.7467 | 78.0141 | 0.48541 | 0.0855333 | 0.265974 | 0.430291 | 0.525328 |
| `LogSoftmax/rows:512/f64:0/real_time` | 38.91 | 53.9951 | 123.104 | 0.720621 | 0.316075 | 0.0835614 | 0.0668493 | 1.25971 |
| `LogSoftmax/rows:512/f64:1/real_time` | 104.713 | 106.609 | 175.938 | 0.982214 | 0.595169 | 0.206733 | 0.0782909 | 0.10499 |
| `LogSoftmax/rows:8192/f64:0/real_time` | 620.1 | 755.915 | 829.807 | 0.82033 | 0.747281 | 1.20643 | 0.166209 | 0.971075 |
| `LogSoftmax/rows:8192/f64:1/real_time` | 1675.4 | 1620.78 | 1714.62 | 1.0337 | 0.97713 | 1.76439 | 0.0767578 | 0.130774 |
| `LogSumExp/rows:32/f64:0/real_time` | 5.03442 | 78.5147 | 75.7863 | 0.0641208 | 0.0664292 | 0.390317 | 0.251597 | 0.159051 |
| `LogSumExp/rows:32/f64:1/real_time` | 8.04042 | 82.1931 | 84.3044 | 0.0978235 | 0.0953736 | 0.345839 | 1.01004 | 0.183153 |
| `LogSumExp/rows:8192/f64:0/real_time` | 784.558 | 724.599 | 717.434 | 1.08275 | 1.09356 | 0.415846 | 0.268553 | 0.381334 |
| `LogSumExp/rows:8192/f64:1/real_time` | 1576.37 | 1496.35 | 1469.5 | 1.05348 | 1.07273 | 0.0930472 | 0.132656 | 0.833144 |
| `Matmul/rows:256/f64:0/real_time` | 25.9486 | 288.177 | 354.867 | 0.090044 | 0.0731221 | 0.883601 | 0.125 | 0.228758 |
| `Matmul/rows:256/f64:1/real_time` | 81.5088 | 704.186 | 772.694 | 0.115749 | 0.105487 | 0.272968 | 0.0755817 | 0.502203 |
| `Matmul/rows:512/f64:0/real_time` | 170.134 | 2133.56 | 2202.39 | 0.0797419 | 0.0772495 | 0.615034 | 0.063892 | 0.739055 |
| `Matmul/rows:512/f64:1/real_time` | 597.48 | 5516.41 | 5585.77 | 0.108309 | 0.106965 | 4.06501 | 0.110425 | 0.405592 |
| `Matmul/rows:64/f64:0/real_time` | 1.48272 | 15.3607 | 77.035 | 0.0965274 | 0.0192474 | 5.1663 | 0.148315 | 0.474705 |
| `Matmul/rows:64/f64:1/real_time` | 2.58367 | 22.2169 | 88.3323 | 0.116293 | 0.0292494 | 0.244886 | 0.13672 | 0.983717 |
| `MatmulGradient/rows:256/f64:0/real_time` | 7.24802 | 149.231 | 99.3239 | 0.0485692 | 0.0729736 | 0.419396 | 0.221519 | 0.189761 |
| `MatmulGradient/rows:256/f64:1/real_time` | 12.5501 | 184.11 | 129.972 | 0.0681663 | 0.0965595 | 2.50146 | 0.616669 | 0.168977 |
| `MatmulGradient/rows:32/f64:0/real_time` | 4.29396 | 127.41 | 75.6289 | 0.033702 | 0.0567767 | 0.480984 | 2.93762 | 0.221359 |
| `MatmulGradient/rows:32/f64:1/real_time` | 5.01604 | 131.716 | 84.8057 | 0.0380822 | 0.0591474 | 0.31691 | 0.831996 | 0.202259 |
| `MatmulTranspose/rows:512/f64:0/real_time` | 353.789 | 2171.61 | 2235.52 | 0.162915 | 0.158258 | 0.166536 | 0.130408 | 0.14955 |
| `MatmulTranspose/rows:512/f64:1/real_time` | 670.873 | 5566.73 | 5618.53 | 0.120515 | 0.119404 | 0.33779 | 2.74266 | 0.0438148 |
| `MatmulTranspose/rows:64/f64:0/real_time` | 2.1212 | 16.2298 | 78.0707 | 0.130698 | 0.0271703 | 5.14625 | 0.136627 | 0.449449 |
| `MatmulTranspose/rows:64/f64:1/real_time` | 2.63494 | 22.8967 | 88.7778 | 0.11508 | 0.0296802 | 1.18679 | 0.0660296 | 0.313905 |
| `MaxLast/rows:32/f64:0/real_time` | 0.551977 | 7.0757 | 69.6861 | 0.0780102 | 0.00792091 | 2.72797 | 0.29098 | 0.243154 |
| `MaxLast/rows:32/f64:1/real_time` | 0.802099 | 7.26164 | 70.0368 | 0.110457 | 0.0114525 | 4.04535 | 0.569849 | 0.344601 |
| `MaxLast/rows:8192/f64:0/real_time` | 69.2436 | 50.2908 | 120.819 | 1.37686 | 0.573121 | 0.170598 | 0.0688498 | 1.28128 |
| `MaxLast/rows:8192/f64:1/real_time` | 130.882 | 83.5883 | 154.865 | 1.5658 | 0.845136 | 0.387618 | 0.208754 | 5.26537 |
| `MaxPool/rows:1/f64:0/real_time` | 1.11923 | 9.68869 | 71.836 | 0.115519 | 0.0155803 | 0.468167 | 0.899058 | 0.230355 |
| `MaxPool/rows:1/f64:1/real_time` | 1.12161 | 9.79747 | 72.0094 | 0.114479 | 0.0155758 | 0.594025 | 0.809654 | 0.161669 |
| `MaxPool/rows:16/f64:0/real_time` | 13.3893 | 19.1765 | 86.2919 | 0.698215 | 0.155163 | 1.30732 | 0.99692 | 0.21809 |
| `MaxPool/rows:16/f64:1/real_time` | 13.9087 | 21.41 | 88.0286 | 0.649637 | 0.158002 | 0.911843 | 0.221446 | 1.17028 |
| `Mean/rows:32/f64:0/real_time` | 3.1727 | 59.7201 | 304.255 | 0.0531262 | 0.0104278 | 11.8422 | 1.02685 | 2.00309 |
| `Mean/rows:32/f64:1/real_time` | 2.99726 | 59.1166 | 308.194 | 0.0507007 | 0.00972522 | 5.69375 | 2.43077 | 1.92462 |
| `Mean/rows:512/f64:0/real_time` | 19.5147 | 66.4764 | 298.673 | 0.293558 | 0.065338 | 3.01553 | 15.02 | 12.7422 |
| `Mean/rows:512/f64:1/real_time` | 23.6212 | 83.7435 | 354.018 | 0.282066 | 0.0667231 | 6.66138 | 10.9522 | 2.00436 |
| `Mean/rows:8192/f64:0/real_time` | 234.393 | 50.0746 | 111.371 | 4.68087 | 2.10461 | 55.4122 | 1.67406 | 0.644875 |
| `Mean/rows:8192/f64:1/real_time` | 72.2445 | 98.7147 | 163.047 | 0.731852 | 0.443091 | 0.128889 | 8.79345 | 5.22295 |
| `Multiply/rows:32/f64:0/real_time` | 1.95216 | 40.1715 | 308.695 | 0.0485957 | 0.00632392 | 3.24773 | 0.945319 | 1.92805 |
| `Multiply/rows:32/f64:1/real_time` | 3.14303 | 40.9184 | 309.776 | 0.076812 | 0.0101461 | 10.1767 | 11.2475 | 9.09742 |
| `Multiply/rows:512/f64:0/real_time` | 26.4869 | 57.8821 | 298.629 | 0.4576 | 0.0886948 | 4.33795 | 10.1864 | 13.3926 |
| `Multiply/rows:512/f64:1/real_time` | 53.6233 | 89.328 | 395.913 | 0.600297 | 0.135442 | 6.02343 | 13.3905 | 13.1726 |
| `Multiply/rows:8192/f64:0/real_time` | 543.956 | 635.559 | 756.947 | 0.85587 | 0.718618 | 2.20843 | 8.70671 | 11.1214 |
| `Multiply/rows:8192/f64:1/real_time` | 989.624 | 992.431 | 1417.23 | 0.997172 | 0.69828 | 11.1112 | 10.8408 | 9.91335 |
| `Relu/rows:32/f64:0/real_time` | 0.27604 | 4.82521 | 69.1937 | 0.0572079 | 0.00398938 | 2.68482 | 0.24013 | 4.63417 |
| `Relu/rows:32/f64:1/real_time` | 0.386273 | 5.0098 | 69.1649 | 0.0771035 | 0.00558481 | 3.74547 | 0.0905011 | 1.82651 |
| `Relu/rows:512/f64:0/real_time` | 4.23788 | 8.93018 | 74.3953 | 0.474557 | 0.0569644 | 4.10901 | 0.316788 | 0.132317 |
| `Relu/rows:512/f64:1/real_time` | 8.25877 | 13.0933 | 83.2316 | 0.630762 | 0.0992264 | 7.22533 | 0.581782 | 0.865772 |
| `Relu/rows:8192/f64:0/real_time` | 67.7269 | 73.0244 | 143.492 | 0.927457 | 0.471991 | 0.138614 | 0.0771867 | 0.151418 |
| `Relu/rows:8192/f64:1/real_time` | 135.342 | 140.758 | 212.283 | 0.961522 | 0.637553 | 0.548471 | 0.147691 | 0.282015 |
| `Sigmoid/rows:32/f64:0/real_time` | 1.83527 | 8.35948 | 72.5485 | 0.219544 | 0.0252972 | 0.514776 | 0.106665 | 0.140913 |
| `Sigmoid/rows:32/f64:1/real_time` | 4.97184 | 11.0339 | 74.9138 | 0.450598 | 0.0663674 | 0.210018 | 0.837447 | 0.38559 |
| `Sigmoid/rows:8192/f64:0/real_time` | 484.412 | 385.507 | 456.377 | 1.25656 | 1.06143 | 0.0936176 | 0.128686 | 0.0789552 |
| `Sigmoid/rows:8192/f64:1/real_time` | 1345.04 | 1056.9 | 1131.96 | 1.27262 | 1.18823 | 0.073142 | 0.433926 | 0.209269 |
| `Softmax/rows:32/f64:0/real_time` | 2.41874 | 9.56753 | 73.9533 | 0.252807 | 0.0327063 | 0.377635 | 0.221945 | 0.352563 |
| `Softmax/rows:32/f64:1/real_time` | 6.36871 | 12.8258 | 76.6448 | 0.496555 | 0.0830938 | 0.157472 | 2.30703 | 0.286971 |
| `Softmax/rows:512/f64:0/real_time` | 36.6623 | 49.1602 | 118.381 | 0.745772 | 0.309698 | 0.304009 | 2.2613 | 0.338486 |
| `Softmax/rows:512/f64:1/real_time` | 99.8753 | 96.5376 | 165.563 | 1.03457 | 0.603246 | 0.0487546 | 0.024055 | 1.31348 |
| `Softmax/rows:8192/f64:0/real_time` | 585.549 | 673.636 | 748.267 | 0.869236 | 0.782539 | 0.109894 | 0.229609 | 0.28648 |
| `Softmax/rows:8192/f64:1/real_time` | 1597.81 | 1451.31 | 1533.96 | 1.10095 | 1.04162 | 0.0921344 | 0.71374 | 0.0844127 |
| `SoftmaxGradient/rows:32/f64:0/real_time` | 5.60413 | 155.379 | 76.7394 | 0.0360674 | 0.0730281 | 0.251826 | 0.345819 | 0.427879 |
| `SoftmaxGradient/rows:32/f64:1/real_time` | 10.7302 | 163.397 | 85.9613 | 0.0656692 | 0.124825 | 0.709401 | 0.850914 | 5.54461 |
| `SoftmaxGradient/rows:8192/f64:0/real_time` | 976.971 | 1625.06 | 1343.99 | 0.60119 | 0.726918 | 0.0252453 | 0.234951 | 4.61952 |
| `SoftmaxGradient/rows:8192/f64:1/real_time` | 2274.62 | 3375.4 | 2790.78 | 0.673882 | 0.815049 | 0.66098 | 0.234969 | 0.269839 |
| `Solve/rows:64/f64:0/real_time` | 11.871 | 16.9698 | 87.5813 | 0.699539 | 0.135543 | 0.588508 | 0.185536 | 0.225725 |
| `Solve/rows:64/f64:1/real_time` | 10.9505 | 19.6062 | 90.3937 | 0.558522 | 0.121142 | 0.314148 | 2.73548 | 0.207592 |
| `Solve/rows:8/f64:0/real_time` | 0.801504 | 6.32756 | 71.8201 | 0.126669 | 0.0111599 | 0.243656 | 0.669802 | 0.234953 |
| `Solve/rows:8/f64:1/real_time` | 0.547767 | 6.28833 | 71.5795 | 0.0871085 | 0.00765256 | 1.97124 | 0.134197 | 0.20242 |
| `SquareGradient/rows:32/f64:0/real_time` | 2.11648 | 104.603 | 70.0508 | 0.0202335 | 0.0302135 | 0.66751 | 2.02138 | 0.259213 |
| `SquareGradient/rows:32/f64:1/real_time` | 2.47449 | 174.353 | 70.7734 | 0.0141924 | 0.0349635 | 1.23756 | 0.114649 | 0.320003 |
| `SquareGradient/rows:512/f64:0/real_time` | 13.747 | 135.292 | 86.8789 | 0.10161 | 0.158232 | 1.1014 | 0.386497 | 0.227091 |
| `SquareGradient/rows:512/f64:1/real_time` | 22.2665 | 226.962 | 96.0243 | 0.098107 | 0.231885 | 0.474461 | 5.27942 | 0.205929 |
| `SquareGradient/rows:8192/f64:0/real_time` | 198.857 | 579.999 | 251.503 | 0.342857 | 0.790674 | 0.13173 | 4.89616 | 0.0689497 |
| `SquareGradient/rows:8192/f64:1/real_time` | 364.417 | 1194.11 | 429.64 | 0.305179 | 0.848193 | 0.274256 | 0.4394 | 0.597592 |
| `SumAll/rows:32/f64:0/real_time` | 2.26189 | 58.5107 | 301.837 | 0.0386578 | 0.00749377 | 9.7884 | 10.0259 | 3.14751 |
| `SumAll/rows:32/f64:1/real_time` | 2.04983 | 60.708 | 308.702 | 0.0337655 | 0.00664018 | 1.58485 | 1.90036 | 1.38487 |
| `SumAll/rows:512/f64:0/real_time` | 17.3424 | 71.988 | 322.869 | 0.240907 | 0.0537135 | 9.81194 | 3.08524 | 2.00938 |
| `SumAll/rows:512/f64:1/real_time` | 24.6151 | 87.1306 | 302.078 | 0.282508 | 0.0814859 | 10.0136 | 4.56613 | 16.2859 |
| `SumAll/rows:8192/f64:0/real_time` | 281.943 | 240.876 | 599.17 | 1.17049 | 0.470557 | 3.5494 | 9.98618 | 5.56452 |
| `SumAll/rows:8192/f64:1/real_time` | 524.877 | 602.075 | 1041.44 | 0.871779 | 0.503992 | 11.9264 | 13.3461 | 7.46166 |
| `SumFirst/rows:32/f64:0/real_time` | 0.706143 | 7.08538 | 70.5313 | 0.0996619 | 0.0100118 | 1.13086 | 0.142927 | 1.06263 |
| `SumFirst/rows:32/f64:1/real_time` | 0.70829 | 7.29681 | 70.9631 | 0.0970684 | 0.0099811 | 0.45669 | 0.209907 | 1.308 |
| `SumFirst/rows:8192/f64:0/real_time` | 109.788 | 304.331 | 377.055 | 0.360753 | 0.291173 | 12.514 | 0.120561 | 3.24282 |
| `SumFirst/rows:8192/f64:1/real_time` | 119.493 | 617.681 | 687.265 | 0.193454 | 0.173867 | 0.140245 | 0.0476434 | 1.07301 |
| `SumLast/rows:32/f64:0/real_time` | 2.61556 | 44.7389 | 296.509 | 0.0584627 | 0.00882119 | 8.07622 | 0.961179 | 9.58494 |
| `SumLast/rows:32/f64:1/real_time` | 2.38823 | 45.657 | 315.099 | 0.0523081 | 0.00757931 | 9.35246 | 10.5123 | 1.30817 |
| `SumLast/rows:512/f64:0/real_time` | 17.175 | 60.875 | 320.977 | 0.282135 | 0.0535085 | 17.1142 | 1.00113 | 9.06453 |
| `SumLast/rows:512/f64:1/real_time` | 20.3833 | 72.9035 | 344.954 | 0.279593 | 0.0590901 | 20.5029 | 6.89796 | 5.97059 |
| `SumLast/rows:8192/f64:0/real_time` | 317.396 | 345.096 | 829.788 | 0.919732 | 0.382503 | 8.5547 | 8.88966 | 2.40979 |
| `SumLast/rows:8192/f64:1/real_time` | 558.847 | 768.896 | 1044.29 | 0.726818 | 0.535146 | 3.65422 | 1.3356 | 11.3248 |
| `Tanh/rows:32/f64:0/real_time` | 2.25747 | 5.7213 | 72.2854 | 0.394574 | 0.03123 | 0.205958 | 0.481483 | 1.3325 |
| `Tanh/rows:32/f64:1/real_time` | 4.60224 | 7.98444 | 74.0645 | 0.576401 | 0.0621383 | 0.35113 | 2.11088 | 0.251037 |
| `Tanh/rows:8192/f64:0/real_time` | 546.957 | 212.407 | 284.354 | 2.57504 | 1.92351 | 0.0361064 | 2.644 | 0.128512 |
| `Tanh/rows:8192/f64:1/real_time` | 1146.32 | 785.666 | 860.502 | 1.45904 | 1.33215 | 0.0434401 | 0.0542102 | 0.147141 |
| `TransposeAdd/rows:32/f64:0/real_time` | 2.08226 | 20.0953 | 72.7074 | 0.103619 | 0.0286389 | 0.440212 | 0.108352 | 0.0457805 |
| `TransposeAdd/rows:32/f64:1/real_time` | 2.11041 | 20.7067 | 73.2341 | 0.101919 | 0.0288173 | 1.03559 | 0.087103 | 5.77085 |
| `TransposeAdd/rows:8192/f64:0/real_time` | 450.993 | 691.459 | 746.28 | 0.652233 | 0.604321 | 0.0770562 | 0.109739 | 4.29965 |
| `TransposeAdd/rows:8192/f64:1/real_time` | 492.236 | 1003.57 | 1044.46 | 0.490487 | 0.471284 | 0.534363 | 0.187543 | 0.203003 |
| `Variance/rows:32/f64:0/real_time` | 3.22464 | 33.0943 | 73.1517 | 0.0974378 | 0.0440815 | 0.208037 | 0.294896 | 6.17032 |
| `Variance/rows:32/f64:1/real_time` | 3.45056 | 34.2928 | 74.1432 | 0.100621 | 0.0465391 | 0.677111 | 0.0351931 | 0.92901 |
| `Variance/rows:8192/f64:0/real_time` | 489.812 | 344.576 | 393.832 | 1.42149 | 1.24371 | 1.36863 | 0.751484 | 0.19338 |
| `Variance/rows:8192/f64:1/real_time` | 605.424 | 685.108 | 724.578 | 0.883691 | 0.835554 | 0.128395 | 0.253159 | 0.256099 |

</details>

<details><summary>expanded_before_tf — 164 cases — 2e4bc6678786</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/2e4bc66787867a59fe741f3ca5a3e3961a65f1e9d3f2b27e4b2b2f44c18244ab.json). Run date: `2026-10-24T20:19:30.509855+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:32/f64:0/real_time` | 2.18531 | 43.5808 | 311.447 | 0.0501438 | 0.00701662 | 11.3173 | 16.4144 | 14.8331 |
| `Add/rows:32/f64:1/real_time` | 4.23206 | 43.2796 | 274.902 | 0.0977841 | 0.0153948 | 3.38048 | 11.9273 | 11.2295 |
| `Add/rows:512/f64:0/real_time` | 28.1294 | 79.9509 | 356.109 | 0.351833 | 0.0789908 | 0.703276 | 2.11464 | 4.43073 |
| `Add/rows:512/f64:1/real_time` | 54.4376 | 108.963 | 389.425 | 0.4996 | 0.13979 | 1.15766 | 10.8465 | 0.330657 |
| `Add/rows:8192/f64:0/real_time` | 509.045 | 608.075 | 1082.65 | 0.837141 | 0.470184 | 1.99832 | 0.977355 | 10.6658 |
| `Add/rows:8192/f64:1/real_time` | 1150.76 | 1062.87 | 1795.24 | 1.08269 | 0.641004 | 1.42902 | 6.65248 | 6.91361 |
| `Attention/rows:128/f64:0/real_time` | 261.837 | 679.728 | 705.104 | 0.385208 | 0.371345 | 5.32717 | 9.61541 | 2.12972 |
| `Attention/rows:128/f64:1/real_time` | 510.686 | 1089.3 | 1098.64 | 0.46882 | 0.464832 | 5.58012 | 3.69488 | 9.29673 |
| `Attention/rows:16/f64:0/real_time` | 12.7271 | 321.975 | 340.722 | 0.0395281 | 0.0373533 | 1.91489 | 8.05157 | 11.8278 |
| `Attention/rows:16/f64:1/real_time` | 18.8912 | 280.296 | 298.479 | 0.0673973 | 0.0632916 | 1.70271 | 10.629 | 6.97571 |
| `AvgPool/rows:1/f64:0/real_time` | 3.4364 | 64.1195 | 276.166 | 0.0535937 | 0.0124432 | 2.32651 | 4.21835 | 12.1959 |
| `AvgPool/rows:1/f64:1/real_time` | 4.17942 | 53.1648 | 300.531 | 0.0786126 | 0.0139068 | 13.0421 | 4.30278 | 12.4272 |
| `AvgPool/rows:16/f64:0/real_time` | 27.672 | 158.667 | 474.922 | 0.174403 | 0.0582664 | 8.63148 | 15.5297 | 1.50733 |
| `AvgPool/rows:16/f64:1/real_time` | 66.2866 | 196.981 | 479.151 | 0.336512 | 0.138342 | 12.2892 | 2.48728 | 3.30693 |
| `BatchMatmul/rows:128/f64:0/real_time` | 116.124 | 656.063 | 1077.84 | 0.177002 | 0.107738 | 7.09231 | 9.25303 | 5.31652 |
| `BatchMatmul/rows:128/f64:1/real_time` | 362.906 | 1537.57 | 2125.47 | 0.236026 | 0.170741 | 10.3849 | 5.18527 | 13.7282 |
| `BatchMatmul/rows:16/f64:0/real_time` | 3.44421 | 55.0944 | 271.805 | 0.0625148 | 0.0126716 | 14.0238 | 14.3486 | 13.6714 |
| `BatchMatmul/rows:16/f64:1/real_time` | 3.78024 | 56.01 | 279.142 | 0.0674922 | 0.0135423 | 13.9665 | 6.51104 | 5.43589 |
| `BatchNorm/rows:32/f64:0/real_time` | 20.2772 | 879.535 | 342.689 | 0.0230545 | 0.0591709 | 8.54301 | 3.06429 | 1.061 |
| `BatchNorm/rows:32/f64:1/real_time` | 28.3751 | 941.295 | 361.303 | 0.0301448 | 0.0785357 | 1.92646 | 2.66981 | 7.13172 |
| `BatchNorm/rows:8192/f64:0/real_time` | 4017.18 | 7132.53 | 6560.55 | 0.56322 | 0.612324 | 11.7741 | 9.82995 | 1.60136 |
| `BatchNorm/rows:8192/f64:1/real_time` | 5061.72 | 20557.6 | 19020.2 | 0.246222 | 0.266123 | 16.1842 | 1.55995 | 3.3978 |
| `BroadcastAdd/rows:32/f64:0/real_time` | 2.32788 | 53.4022 | 308.274 | 0.0435915 | 0.00755134 | 13.5171 | 8.25995 | 5.41036 |
| `BroadcastAdd/rows:32/f64:1/real_time` | 4.01118 | 48.5605 | 328.078 | 0.0826017 | 0.0122263 | 13.7301 | 16.7452 | 3.50688 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 32.2088 | 107.236 | 380.138 | 0.300355 | 0.0847292 | 6.86155 | 9.62161 | 3.4712 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 57.527 | 140.946 | 354.601 | 0.408148 | 0.16223 | 3.45751 | 12.5195 | 8.82069 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 529.104 | 1202.66 | 1594.33 | 0.439946 | 0.331866 | 9.34443 | 5.61908 | 5.28064 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 1197.64 | 1933.97 | 2847 | 0.619264 | 0.420666 | 4.90055 | 9.47982 | 6.29619 |
| `Cholesky/rows:64/f64:0/real_time` | 20.5634 | 66.5717 | 342.842 | 0.30889 | 0.0599791 | 2.61354 | 2.46933 | 2.46407 |
| `Cholesky/rows:64/f64:1/real_time` | 16.1582 | 70.0193 | 356.607 | 0.230768 | 0.0453109 | 1.63339 | 11.447 | 2.2426 |
| `Cholesky/rows:8/f64:0/real_time` | 1.61983 | 26.6299 | 310.342 | 0.0608275 | 0.0052195 | 14.8849 | 19.5189 | 4.82096 |
| `Cholesky/rows:8/f64:1/real_time` | 1.02229 | 29.1357 | 293.958 | 0.035087 | 0.00347766 | 1.72579 | 8.42888 | 7.57916 |
| `Conv2d/rows:1/f64:0/real_time` | 22.3589 | 60.7481 | 283.873 | 0.36806 | 0.0787637 | 11.2453 | 7.28824 | 8.68966 |
| `Conv2d/rows:1/f64:1/real_time` | 25.9452 | 60.6379 | 269.777 | 0.42787 | 0.0961725 | 5.06395 | 5.46863 | 14.0847 |
| `Conv2d/rows:16/f64:0/real_time` | 326.609 | 272.692 | 566.885 | 1.19772 | 0.576146 | 9.1455 | 2.59049 | 10.2274 |
| `Conv2d/rows:16/f64:1/real_time` | 434.678 | 348.493 | 494.396 | 1.24731 | 0.879211 | 7.41142 | 11.8121 | 17.4869 |
| `ConvInputGradient/rows:1/f64:0/real_time` | 51.8079 | 791.38 | 270.264 | 0.0654653 | 0.191694 | 7.40503 | 13.9348 | 12.0339 |
| `ConvInputGradient/rows:1/f64:1/real_time` | 57.6895 | 908.058 | 348.764 | 0.0635306 | 0.165411 | 11.8505 | 5.85399 | 12.268 |
| `ConvInputGradient/rows:16/f64:0/real_time` | 712.286 | 2007.57 | 670.577 | 0.354801 | 1.0622 | 6.42337 | 13.1684 | 4.81247 |
| `ConvInputGradient/rows:16/f64:1/real_time` | 849.935 | 2181.79 | 806.017 | 0.389558 | 1.05449 | 10.4296 | 12.5879 | 9.50573 |
| `Exp/rows:32/f64:0/real_time` | 5.09566 | 38.127 | 267.435 | 0.13365 | 0.0190538 | 12.0642 | 10.7168 | 11.4339 |
| `Exp/rows:32/f64:1/real_time` | 17.7567 | 39.9108 | 322.699 | 0.444909 | 0.0550255 | 11.9744 | 12.4017 | 6.08396 |
| `Exp/rows:512/f64:0/real_time` | 77.4069 | 127.62 | 433.121 | 0.606542 | 0.178719 | 10.7165 | 11.9115 | 9.42898 |
| `Exp/rows:512/f64:1/real_time` | 272.551 | 377.623 | 678.575 | 0.721754 | 0.401651 | 9.09863 | 13.1239 | 5.27282 |
| `Exp/rows:8192/f64:0/real_time` | 1678.67 | 2242.16 | 2811.39 | 0.748684 | 0.597096 | 8.66407 | 4.73907 | 3.18368 |
| `Exp/rows:8192/f64:1/real_time` | 4887.39 | 5275.32 | 5353.72 | 0.926464 | 0.912896 | 3.84448 | 13.1492 | 8.57827 |
| `FftPower/rows:2048/f64:0/real_time` | 189.036 | 426.243 | 384.333 | 0.443493 | 0.491854 | 5.93442 | 5.25451 | 12.0759 |
| `FftPower/rows:2048/f64:1/real_time` | 250.236 | 457.832 | 470.965 | 0.546568 | 0.531326 | 1.65931 | 10.9804 | 13.1742 |
| `FftPower/rows:64/f64:0/real_time` | 10.0738 | 338.177 | 330.769 | 0.0297885 | 0.0304557 | 1.90706 | 1.10819 | 3.12315 |
| `FftPower/rows:64/f64:1/real_time` | 9.81027 | 345.233 | 340.663 | 0.0284164 | 0.0287976 | 17.6489 | 0.688679 | 1.37908 |
| `GatherRows/rows:32/f64:0/real_time` | 1.11464 | 74.328 | 256.071 | 0.0149962 | 0.00435286 | 16.4406 | 6.85364 | 9.13111 |
| `GatherRows/rows:32/f64:1/real_time` | 1.34968 | 68.4155 | 295.952 | 0.0197277 | 0.00456046 | 5.29179 | 5.98111 | 9.90233 |
| `GatherRows/rows:8192/f64:0/real_time` | 160.366 | 234.818 | 549.994 | 0.682939 | 0.291578 | 6.53 | 1.63781 | 6.91212 |
| `GatherRows/rows:8192/f64:1/real_time` | 313.315 | 499.657 | 1019.33 | 0.627061 | 0.307372 | 12.8417 | 2.65741 | 4.96304 |
| `Gelu/rows:32/f64:0/real_time` | 35.9584 | 279.193 | 329.771 | 0.128794 | 0.10904 | 6.48834 | 8.90073 | 4.0978 |
| `Gelu/rows:32/f64:1/real_time` | 81.7048 | 346.617 | 320.015 | 0.235721 | 0.255316 | 8.01348 | 6.24015 | 14.294 |
| `Gelu/rows:8192/f64:0/real_time` | 7931.21 | 3703.62 | 4162.95 | 2.14148 | 1.90519 | 6.60554 | 10.3646 | 8.58082 |
| `Gelu/rows:8192/f64:1/real_time` | 20919.2 | 7974.38 | 7125.03 | 2.6233 | 2.93601 | 6.37837 | 9.51893 | 7.90279 |
| `LayerNorm/rows:32/f64:0/real_time` | 31.4627 | 828.348 | 345.309 | 0.0379825 | 0.0911147 | 11.0666 | 14.919 | 7.01317 |
| `LayerNorm/rows:32/f64:1/real_time` | 45.1576 | 832.525 | 351.827 | 0.0542418 | 0.128352 | 2.5591 | 17.9032 | 10.0762 |
| `LayerNorm/rows:8192/f64:0/real_time` | 6786.57 | 5563.41 | 4508 | 1.21986 | 1.50545 | 5.46725 | 1.01483 | 9.77581 |
| `LayerNorm/rows:8192/f64:1/real_time` | 9493.98 | 9925.28 | 8163.71 | 0.956544 | 1.16295 | 8.04617 | 1.88973 | 1.80034 |
| `LogSoftmax/rows:32/f64:0/real_time` | 10.9417 | 48.0157 | 283.007 | 0.227878 | 0.0386624 | 7.85931 | 9.52377 | 10.007 |
| `LogSoftmax/rows:32/f64:1/real_time` | 25.4283 | 70.0279 | 293.428 | 0.363117 | 0.0866596 | 10.6318 | 12.9815 | 5.89812 |
| `LogSoftmax/rows:512/f64:0/real_time` | 171.093 | 236.164 | 487.664 | 0.724468 | 0.350843 | 7.50606 | 12.3432 | 12.4933 |
| `LogSoftmax/rows:512/f64:1/real_time` | 473.101 | 624.297 | 1014.37 | 0.757815 | 0.466397 | 12.9095 | 7.40544 | 11.1821 |
| `LogSoftmax/rows:8192/f64:0/real_time` | 3197.64 | 4648.74 | 5455.68 | 0.687851 | 0.586112 | 4.59833 | 4.15013 | 3.37701 |
| `LogSoftmax/rows:8192/f64:1/real_time` | 6441 | 9890.63 | 10503.3 | 0.651222 | 0.613238 | 13.6287 | 8.11794 | 3.19346 |
| `LogSumExp/rows:32/f64:0/real_time` | 33.1135 | 404.647 | 283.698 | 0.0818331 | 0.116721 | 9.22593 | 8.25465 | 5.63194 |
| `LogSumExp/rows:32/f64:1/real_time` | 54.862 | 364.655 | 325.511 | 0.150449 | 0.168541 | 7.56453 | 7.27425 | 14.0375 |
| `LogSumExp/rows:8192/f64:0/real_time` | 6667.13 | 4519.09 | 4686.41 | 1.47533 | 1.42265 | 9.42101 | 6.16711 | 3.59641 |
| `LogSumExp/rows:8192/f64:1/real_time` | 10502.9 | 9676.46 | 9102.02 | 1.08541 | 1.15391 | 8.34572 | 11.2533 | 3.27677 |
| `Matmul/rows:256/f64:0/real_time` | 213.115 | 1103.48 | 1628.76 | 0.19313 | 0.130845 | 12.1335 | 9.84551 | 6.12853 |
| `Matmul/rows:256/f64:1/real_time` | 824.215 | 2721.84 | 2957.92 | 0.302815 | 0.278647 | 7.59479 | 5.77487 | 9.33507 |
| `Matmul/rows:512/f64:0/real_time` | 1590.94 | 10272.4 | 8858.61 | 0.154875 | 0.179592 | 12.3991 | 1.45185 | 10.0894 |
| `Matmul/rows:512/f64:1/real_time` | 6400.64 | 21440.7 | 21611.7 | 0.298528 | 0.296165 | 13.9813 | 6.21354 | 2.6624 |
| `Matmul/rows:64/f64:0/real_time` | 4.63454 | 68.6592 | 283.803 | 0.0675006 | 0.0163302 | 10.6253 | 13.664 | 15.1998 |
| `Matmul/rows:64/f64:1/real_time` | 13.263 | 94.7717 | 318.532 | 0.139947 | 0.0416378 | 9.25819 | 5.5415 | 10.3597 |
| `MatmulGradient/rows:256/f64:0/real_time` | 29.0997 | 694.179 | 311.371 | 0.0419196 | 0.0934565 | 6.89557 | 14.1256 | 14.4293 |
| `MatmulGradient/rows:256/f64:1/real_time` | 61.2839 | 751.452 | 508.677 | 0.0815539 | 0.120477 | 16.4498 | 5.98878 | 12.8419 |
| `MatmulGradient/rows:32/f64:0/real_time` | 13.5521 | 594.578 | 274.007 | 0.0227928 | 0.049459 | 12.6035 | 11.0315 | 9.8049 |
| `MatmulGradient/rows:32/f64:1/real_time` | 15.7376 | 642.754 | 317.074 | 0.0244846 | 0.0496338 | 5.26818 | 10.3709 | 7.35934 |
| `MatmulTranspose/rows:512/f64:0/real_time` | 1439.78 | 9503.51 | 10488.9 | 0.1515 | 0.137268 | 5.876 | 12.2812 | 6.87318 |
| `MatmulTranspose/rows:512/f64:1/real_time` | 4176.14 | 22052 | 22705.9 | 0.189377 | 0.183923 | 6.55975 | 2.22701 | 1.44173 |
| `MatmulTranspose/rows:64/f64:0/real_time` | 5.79286 | 69.7469 | 315.327 | 0.0830555 | 0.018371 | 8.45737 | 9.05362 | 8.75248 |
| `MatmulTranspose/rows:64/f64:1/real_time` | 10.9783 | 103.104 | 346.41 | 0.106478 | 0.0316917 | 15.2099 | 12.3958 | 10.7149 |
| `MaxLast/rows:32/f64:0/real_time` | 14.7003 | 31.6187 | 277.955 | 0.464924 | 0.0528872 | 9.38251 | 12.4031 | 9.35972 |
| `MaxLast/rows:32/f64:1/real_time` | 12.7459 | 41.1716 | 244.51 | 0.30958 | 0.0521282 | 8.5006 | 9.01475 | 10.1817 |
| `MaxLast/rows:8192/f64:0/real_time` | 3139.28 | 308.834 | 641.767 | 10.1649 | 4.89162 | 6.07146 | 12.9187 | 5.3521 |
| `MaxLast/rows:8192/f64:1/real_time` | 3901.76 | 611.234 | 1028 | 6.3834 | 3.79547 | 6.42225 | 4.24097 | 5.59236 |
| `MaxPool/rows:1/f64:0/real_time` | 4.35327 | 48.4365 | 257.189 | 0.0898756 | 0.0169263 | 16.5703 | 11.617 | 2.6886 |
| `MaxPool/rows:1/f64:1/real_time` | 5.0415 | 56.9365 | 325.293 | 0.0885461 | 0.0154983 | 9.76257 | 12.3001 | 9.62156 |
| `MaxPool/rows:16/f64:0/real_time` | 59.4829 | 147.22 | 428.079 | 0.404041 | 0.138953 | 12.8579 | 3.83679 | 2.46063 |
| `MaxPool/rows:16/f64:1/real_time` | 69.1117 | 170.574 | 452.073 | 0.405171 | 0.152877 | 15.3051 | 11.4863 | 1.3605 |
| `Mean/rows:32/f64:0/real_time` | 2.92822 | 52.0495 | 274.73 | 0.0562584 | 0.0106586 | 4.87983 | 9.47742 | 16.908 |
| `Mean/rows:32/f64:1/real_time` | 2.39696 | 54.6228 | 250.398 | 0.0438821 | 0.00957259 | 9.09507 | 16.0379 | 4.15223 |
| `Mean/rows:512/f64:0/real_time` | 16.0248 | 64.2713 | 276.526 | 0.24933 | 0.0579503 | 16.3552 | 8.95907 | 7.8904 |
| `Mean/rows:512/f64:1/real_time` | 23.5258 | 67.1948 | 280.492 | 0.350113 | 0.0838733 | 12.3922 | 7.30887 | 16.4215 |
| `Mean/rows:8192/f64:0/real_time` | 239.545 | 285.765 | 603.899 | 0.83826 | 0.396665 | 11.0197 | 9.97849 | 12.6845 |
| `Mean/rows:8192/f64:1/real_time` | 590.664 | 671.814 | 1058.88 | 0.879207 | 0.557817 | 6.52135 | 3.36419 | 5.01389 |
| `Multiply/rows:32/f64:0/real_time` | 1.63856 | 33.1567 | 302.979 | 0.0494188 | 0.00540817 | 11.2387 | 12.5622 | 14.172 |
| `Multiply/rows:32/f64:1/real_time` | 3.20497 | 33.6255 | 313.393 | 0.0953139 | 0.0102267 | 16.8263 | 15.0092 | 3.90714 |
| `Multiply/rows:512/f64:0/real_time` | 22.4749 | 62.3995 | 330.957 | 0.360177 | 0.0679087 | 11.6343 | 6.11228 | 12.5163 |
| `Multiply/rows:512/f64:1/real_time` | 53.5584 | 85.7056 | 378.035 | 0.624911 | 0.141676 | 2.07359 | 7.09145 | 7.78682 |
| `Multiply/rows:8192/f64:0/real_time` | 563.92 | 639.352 | 1017.68 | 0.882017 | 0.554124 | 10.6111 | 3.12696 | 6.19294 |
| `Multiply/rows:8192/f64:1/real_time` | 1121.99 | 1249.4 | 1627.67 | 0.898027 | 0.689327 | 1.64744 | 2.12172 | 1.96351 |
| `Relu/rows:32/f64:0/real_time` | 1.01799 | 22.8871 | 258.873 | 0.0444788 | 0.00393239 | 1.92543 | 13.1561 | 7.77402 |
| `Relu/rows:32/f64:1/real_time` | 2.0732 | 24.0372 | 267.396 | 0.0862495 | 0.0077533 | 18.0117 | 13.9547 | 8.91264 |
| `Relu/rows:512/f64:0/real_time` | 19.5834 | 39.4637 | 299.289 | 0.496238 | 0.065433 | 3.19804 | 16.9555 | 8.54601 |
| `Relu/rows:512/f64:1/real_time` | 32.031 | 67.7346 | 294.776 | 0.47289 | 0.108662 | 7.53387 | 7.03239 | 13.3738 |
| `Relu/rows:8192/f64:0/real_time` | 424.602 | 538.933 | 974.259 | 0.787857 | 0.435821 | 10.1352 | 8.52297 | 2.82317 |
| `Relu/rows:8192/f64:1/real_time` | 840.462 | 946.336 | 1651.32 | 0.888122 | 0.508964 | 8.285 | 6.08504 | 2.87622 |
| `Sigmoid/rows:32/f64:0/real_time` | 22.8911 | 49.8554 | 254.764 | 0.459151 | 0.0898524 | 5.0177 | 5.88064 | 11.4607 |
| `Sigmoid/rows:32/f64:1/real_time` | 30.6585 | 54.083 | 333.812 | 0.566878 | 0.0918434 | 4.32176 | 9.59375 | 6.90167 |
| `Sigmoid/rows:8192/f64:0/real_time` | 6297.41 | 2504.32 | 3104.99 | 2.51462 | 2.02816 | 7.25498 | 3.39253 | 2.51632 |
| `Sigmoid/rows:8192/f64:1/real_time` | 9138.71 | 6681.06 | 7312.88 | 1.36785 | 1.24967 | 1.07125 | 2.28194 | 12.1517 |
| `Softmax/rows:32/f64:0/real_time` | 10.5334 | 49.2269 | 326.271 | 0.213975 | 0.0322841 | 20.6531 | 12.8705 | 9.4869 |
| `Softmax/rows:32/f64:1/real_time` | 23.9085 | 56.937 | 342.424 | 0.419911 | 0.0698213 | 11.3392 | 11.2833 | 11.3076 |
| `Softmax/rows:512/f64:0/real_time` | 170.846 | 200.187 | 542.324 | 0.853431 | 0.315025 | 11.5517 | 9.88943 | 9.28186 |
| `Softmax/rows:512/f64:1/real_time` | 427.824 | 470.158 | 928.842 | 0.909958 | 0.460599 | 5.9115 | 9.13722 | 8.55114 |
| `Softmax/rows:8192/f64:0/real_time` | 2944.12 | 4038.29 | 4956.41 | 0.729052 | 0.594003 | 12.4517 | 15.6289 | 4.45165 |
| `Softmax/rows:8192/f64:1/real_time` | 6852.97 | 7757.55 | 9576.21 | 0.883394 | 0.715624 | 18.4127 | 11.9781 | 9.73297 |
| `SoftmaxGradient/rows:32/f64:0/real_time` | 42.9738 | 733.945 | 345.699 | 0.0585518 | 0.12431 | 1.82377 | 11.9825 | 15.2115 |
| `SoftmaxGradient/rows:32/f64:1/real_time` | 57.7855 | 902.295 | 402.065 | 0.0640428 | 0.143722 | 5.9675 | 11.0836 | 4.91703 |
| `SoftmaxGradient/rows:8192/f64:0/real_time` | 6214.53 | 10010.2 | 7868.91 | 0.620818 | 0.789757 | 11.3334 | 9.76851 | 10.0671 |
| `SoftmaxGradient/rows:8192/f64:1/real_time` | 13864.2 | 19922.8 | 14960.9 | 0.695895 | 0.926695 | 8.14597 | 8.76085 | 8.43905 |
| `Solve/rows:64/f64:0/real_time` | 42.3493 | 82.7744 | 375.027 | 0.511623 | 0.112923 | 5.57634 | 8.24769 | 8.69243 |
| `Solve/rows:64/f64:1/real_time` | 36.9252 | 109.85 | 351.596 | 0.33614 | 0.105021 | 12.5866 | 5.39284 | 11.1503 |
| `Solve/rows:8/f64:0/real_time` | 3.98024 | 37.8691 | 315.815 | 0.105105 | 0.0126031 | 9.64796 | 1.98802 | 7.29343 |
| `Solve/rows:8/f64:1/real_time` | 2.57149 | 36.9435 | 308.678 | 0.069606 | 0.00833065 | 13.5117 | 2.49624 | 4.46701 |
| `SquareGradient/rows:32/f64:0/real_time` | 10.7703 | 565.787 | 292.344 | 0.019036 | 0.0368413 | 14.6506 | 11.2597 | 16.0211 |
| `SquareGradient/rows:32/f64:1/real_time` | 12.4073 | 825.394 | 307.815 | 0.015032 | 0.0403078 | 7.19096 | 12.407 | 10.4779 |
| `SquareGradient/rows:512/f64:0/real_time` | 65.9082 | 648.023 | 364.479 | 0.101707 | 0.180828 | 8.49823 | 16.0957 | 8.83056 |
| `SquareGradient/rows:512/f64:1/real_time` | 130.297 | 1316.02 | 467.397 | 0.0990083 | 0.278771 | 6.65668 | 16.0778 | 12.759 |
| `SquareGradient/rows:8192/f64:0/real_time` | 1451.85 | 5026.9 | 1849.36 | 0.288815 | 0.785053 | 1.39525 | 4.9675 | 3.56034 |
| `SquareGradient/rows:8192/f64:1/real_time` | 2492.24 | 9324.43 | 3077.5 | 0.267281 | 0.809826 | 2.32507 | 7.94007 | 5.1075 |
| `SumAll/rows:32/f64:0/real_time` | 1.78763 | 60.2181 | 233.859 | 0.0296859 | 0.00764403 | 15.0687 | 8.06647 | 10.2376 |
| `SumAll/rows:32/f64:1/real_time` | 1.92526 | 47.9011 | 283.926 | 0.0401923 | 0.00678084 | 17.6972 | 14.1266 | 8.4291 |
| `SumAll/rows:512/f64:0/real_time` | 13.3505 | 70.1857 | 318.217 | 0.190217 | 0.0419541 | 17.7715 | 12.0918 | 4.39769 |
| `SumAll/rows:512/f64:1/real_time` | 24.5765 | 81.7769 | 310.055 | 0.300531 | 0.079265 | 7.7867 | 3.3925 | 7.23473 |
| `SumAll/rows:8192/f64:0/real_time` | 241.667 | 313.437 | 727.786 | 0.771022 | 0.332058 | 16.9798 | 3.16752 | 2.59574 |
| `SumAll/rows:8192/f64:1/real_time` | 607.131 | 730.86 | 1145.18 | 0.830708 | 0.530163 | 8.28675 | 4.30837 | 3.75855 |
| `SumFirst/rows:32/f64:0/real_time` | 3.90277 | 44.9779 | 327.241 | 0.0867707 | 0.0119263 | 1.62077 | 16.8344 | 1.79735 |
| `SumFirst/rows:32/f64:1/real_time` | 3.75469 | 49.4214 | 321.74 | 0.0759729 | 0.0116699 | 7.96778 | 1.35568 | 3.16076 |
| `SumFirst/rows:8192/f64:0/real_time` | 695.854 | 795.008 | 1463.2 | 0.875279 | 0.475571 | 10.9863 | 5.53855 | 12.2948 |
| `SumFirst/rows:8192/f64:1/real_time` | 629.278 | 5216.72 | 5688.65 | 0.120627 | 0.11062 | 16.2791 | 14.7306 | 6.74676 |
| `SumLast/rows:32/f64:0/real_time` | 2.4967 | 41.8878 | 274.454 | 0.0596045 | 0.00909699 | 2.53728 | 13.2013 | 8.97822 |
| `SumLast/rows:32/f64:1/real_time` | 2.16076 | 36.23 | 268.572 | 0.0596401 | 0.00804537 | 7.88917 | 7.04845 | 8.70352 |
| `SumLast/rows:512/f64:0/real_time` | 18.4962 | 52.8941 | 266.599 | 0.349683 | 0.0693784 | 4.41813 | 2.12105 | 16.0927 |
| `SumLast/rows:512/f64:1/real_time` | 20.4017 | 65.4872 | 374.226 | 0.311537 | 0.0545171 | 8.31802 | 8.75825 | 11.2682 |
| `SumLast/rows:8192/f64:0/real_time` | 221.605 | 233.16 | 818.302 | 0.95044 | 0.27081 | 4.97499 | 17.1972 | 11.3415 |
| `SumLast/rows:8192/f64:1/real_time` | 577.783 | 680.848 | 1099.19 | 0.848622 | 0.525647 | 4.44491 | 3.63561 | 6.23226 |
| `Tanh/rows:32/f64:0/real_time` | 25.2377 | 30.7901 | 291.851 | 0.819668 | 0.0864745 | 13.0844 | 11.6289 | 6.46657 |
| `Tanh/rows:32/f64:1/real_time` | 39.7022 | 46.2041 | 330.308 | 0.859279 | 0.120198 | 18.929 | 6.1299 | 9.38786 |
| `Tanh/rows:8192/f64:0/real_time` | 6660.79 | 1138.2 | 1600.94 | 5.85203 | 4.16054 | 1.98614 | 0.798985 | 4.47164 |
| `Tanh/rows:8192/f64:1/real_time` | 14111.8 | 4438.69 | 4773.29 | 3.17926 | 2.9564 | 2.89609 | 2.66578 | 9.27827 |
| `TransposeAdd/rows:32/f64:0/real_time` | 9.04968 | 117.559 | 335.003 | 0.0769796 | 0.0270137 | 11.5906 | 7.50513 | 9.40074 |
| `TransposeAdd/rows:32/f64:1/real_time` | 10.8455 | 125.156 | 330.292 | 0.0866551 | 0.0328359 | 3.52935 | 14.7969 | 4.48038 |
| `TransposeAdd/rows:8192/f64:0/real_time` | 3795.13 | 2764.24 | 3091 | 1.37294 | 1.2278 | 1.98845 | 3.48601 | 2.46901 |
| `TransposeAdd/rows:8192/f64:1/real_time` | 10337.3 | 4199.88 | 4011.91 | 2.46133 | 2.57664 | 10.2147 | 2.75829 | 3.83709 |
| `Variance/rows:32/f64:0/real_time` | 14.8925 | 155.67 | 310.094 | 0.0956674 | 0.0480257 | 9.13307 | 18.3477 | 6.82517 |
| `Variance/rows:32/f64:1/real_time` | 17.1469 | 195.928 | 352.662 | 0.0875161 | 0.0486213 | 6.53421 | 7.9145 | 8.20425 |
| `Variance/rows:8192/f64:0/real_time` | 2494.2 | 2194.22 | 2859.88 | 1.13672 | 0.872134 | 5.55094 | 11.3633 | 2.57119 |
| `Variance/rows:8192/f64:1/real_time` | 3619.83 | 4189 | 4413.78 | 0.864128 | 0.820119 | 9.77269 | 3.97159 | 8.47973 |

</details>

<details><summary>expanded_reduction_repeat_tf — 2 cases — 865445a7b659</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/865445a7b6592ee49cc33dea5b325a4061533e45648440c484f8e86bec8542a7.json). Run date: `2026-10-24T21:18:01.312504+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Mean/rows:8192/f64:0/real_time` | 61.47 | 50.0006 | 109.414 | 1.22939 | 0.561812 | 0.0627987 | 0.125669 | 0.170841 |
| `SumAll/rows:8192/f64:0/real_time` | 60.9145 | 50.0598 | 108.431 | 1.21683 | 0.561781 | 0.122232 | 0.537001 | 9.46007 |

</details>

<details><summary>gap_batch_check_tf — 1 cases — b24c24ab0012</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/b24c24ab0012e71d05ced6a5120df1f537263d002b893c000b090f74424e8111.json). Run date: `2026-10-24T17:44:10.855936+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Mean/rows:32/f64:0/real_time` | 3.11367 | 62.905 | 310.786 | 0.049498 | 0.0100187 | 4.83844 | 2.50295 | 2.1298 |

</details>

<details><summary>gap_tf_after — 28 cases — 83c0c3990eed</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/83c0c3990eed1f116c835a7182b3bf93d7b49674260de6d6e6c0f228a3a6b6eb.json). Run date: `2026-10-24T17:34:06.076769+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:512/f64:0/real_time` | 4.35142 | 12.5765 | 69.6032 | 0.345996 | 0.0625175 | 0.286458 | 0.496902 | 1.09942 |
| `Add/rows:512/f64:1/real_time` | 8.39666 | 16.7031 | 74.0464 | 0.502702 | 0.113397 | 0.374072 | 0.150872 | 4.02122 |
| `Add/rows:8192/f64:0/real_time` | 68.1714 | 77.2076 | 131.312 | 0.882962 | 0.519157 | 0.16734 | 0.816986 | 0.188304 |
| `Add/rows:8192/f64:1/real_time` | 137.112 | 145.328 | 206.689 | 0.943471 | 0.663376 | 0.765692 | 0.0995121 | 0.187977 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 4.94473 | 18.6068 | 80.1792 | 0.265748 | 0.0616711 | 0.705227 | 0.313591 | 0.250869 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 8.52289 | 28.1887 | 88.3909 | 0.302351 | 0.0964227 | 3.31185 | 0.439503 | 0.36413 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 68.1783 | 167.135 | 228.308 | 0.407924 | 0.298624 | 2.35683 | 0.539786 | 0.179493 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 136.239 | 318.176 | 379.927 | 0.428186 | 0.358592 | 9.89499 | 0.102126 | 3.03206 |
| `Mean/rows:512/f64:0/real_time` | 4.27247 | 12.0202 | 68.0689 | 0.355441 | 0.0627669 | 0.28439 | 0.0832242 | 1.28817 |
| `Mean/rows:512/f64:1/real_time` | 5.87818 | 14.5074 | 70.8555 | 0.405184 | 0.0829601 | 0.402268 | 0.193975 | 21.2697 |
| `Mean/rows:8192/f64:0/real_time` | 61.5178 | 49.8576 | 115.271 | 1.23387 | 0.533678 | 0.113112 | 3.72394 | 16.5883 |
| `Mean/rows:8192/f64:1/real_time` | 88.1 | 90.8727 | 152.647 | 0.969488 | 0.577147 | 0.160834 | 0.137025 | 17.1394 |
| `Multiply/rows:512/f64:0/real_time` | 4.30991 | 10.1666 | 69.4854 | 0.423929 | 0.062026 | 0.451736 | 0.755261 | 5.51364 |
| `Multiply/rows:512/f64:1/real_time` | 8.35515 | 14.1776 | 78.3316 | 0.589319 | 0.106664 | 0.125833 | 0.168064 | 0.197621 |
| `Multiply/rows:8192/f64:0/real_time` | 68.1782 | 74.4655 | 138.182 | 0.915568 | 0.493394 | 4.65583 | 0.0632907 | 0.234028 |
| `Multiply/rows:8192/f64:1/real_time` | 136.989 | 142.72 | 206.703 | 0.959845 | 0.662733 | 0.115308 | 0.410107 | 0.0493008 |
| `SquareGradient/rows:512/f64:0/real_time` | 13.9091 | 128.9 | 106.445 | 0.107906 | 0.130669 | 1.64308 | 6.98252 | 16.5958 |
| `SquareGradient/rows:512/f64:1/real_time` | 23.7416 | 220.864 | 92.7778 | 0.107494 | 0.255897 | 1.91466 | 0.0989204 | 0.0451116 |
| `SquareGradient/rows:8192/f64:0/real_time` | 199.546 | 572.096 | 245.754 | 0.348799 | 0.811977 | 0.603434 | 0.213864 | 0.163207 |
| `SquareGradient/rows:8192/f64:1/real_time` | 375.861 | 1178.18 | 424.078 | 0.319017 | 0.886301 | 0.251656 | 0.440332 | 0.510695 |
| `SumAll/rows:512/f64:0/real_time` | 4.10015 | 12.0958 | 67.5787 | 0.338974 | 0.0606722 | 0.264967 | 3.46141 | 0.193083 |
| `SumAll/rows:512/f64:1/real_time` | 5.69877 | 14.548 | 70.488 | 0.391721 | 0.0808473 | 0.213879 | 0.790995 | 0.371705 |
| `SumAll/rows:8192/f64:0/real_time` | 61.4726 | 50.0098 | 110.932 | 1.22921 | 0.554149 | 0.0793458 | 0.084368 | 0.304306 |
| `SumAll/rows:8192/f64:1/real_time` | 88.3334 | 90.8954 | 152.049 | 0.971814 | 0.580955 | 2.16961 | 0.0529129 | 0.230148 |
| `SumLast/rows:512/f64:0/real_time` | 4.38981 | 9.50491 | 68.1125 | 0.461846 | 0.0644493 | 0.373619 | 0.0972851 | 0.299403 |
| `SumLast/rows:512/f64:1/real_time` | 4.51294 | 11.6424 | 71.0466 | 0.387631 | 0.0635208 | 0.270419 | 0.753159 | 1.40278 |
| `SumLast/rows:8192/f64:0/real_time` | 65.8704 | 48.5391 | 112.614 | 1.35706 | 0.584922 | 0.119977 | 0.152015 | 0.245819 |
| `SumLast/rows:8192/f64:1/real_time` | 74.4523 | 86.7411 | 150.884 | 0.858327 | 0.49344 | 0.151596 | 3.58194 | 0.15693 |

</details>

<details><summary>gap_tf_before — 28 cases — 3a51e143295d</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/3a51e143295dcf60cda48aded3ab8eb776c4e5e36a558dfd2365830dc6069892.json). Run date: `2026-10-24T17:25:40.610177+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:512/f64:0/real_time` | 4.30401 | 12.5052 | 66.8015 | 0.344178 | 0.0644297 | 1.31778 | 0.372425 | 2.20851 |
| `Add/rows:512/f64:1/real_time` | 8.3636 | 16.7013 | 70.5108 | 0.500776 | 0.118614 | 0.497547 | 0.229735 | 7.86602 |
| `Add/rows:8192/f64:0/real_time` | 68.1089 | 77.0708 | 124.06 | 0.883718 | 0.548998 | 0.0660307 | 0.524289 | 1.49476 |
| `Add/rows:8192/f64:1/real_time` | 136.83 | 145.312 | 205.311 | 0.941625 | 0.66645 | 0.114658 | 0.113712 | 0.217864 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 4.41689 | 18.7263 | 75.4196 | 0.235865 | 0.0585642 | 3.11767 | 0.575477 | 0.310754 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 8.36053 | 27.8213 | 85.7413 | 0.300508 | 0.0975088 | 5.41065 | 0.427396 | 0.545847 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 68.2413 | 167.034 | 225.395 | 0.408548 | 0.302764 | 3.22252 | 0.120611 | 0.197463 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 136.011 | 318.119 | 377.177 | 0.427549 | 0.360604 | 0.074942 | 0.236247 | 0.0977043 |
| `Mean/rows:512/f64:0/real_time` | 6.19719 | 11.9805 | 67.8265 | 0.517273 | 0.0913683 | 0.0751213 | 0.895443 | 0.269515 |
| `Mean/rows:512/f64:1/real_time` | 5.50157 | 14.4118 | 70.9124 | 0.381742 | 0.0775827 | 0.497136 | 0.08136 | 5.59687 |
| `Mean/rows:8192/f64:0/real_time` | 92.2498 | 49.9243 | 110.837 | 1.84779 | 0.832302 | 0.270406 | 0.210236 | 4.30678 |
| `Mean/rows:8192/f64:1/real_time` | 85.1794 | 90.8442 | 152.682 | 0.937643 | 0.557889 | 0.975603 | 0.102986 | 0.515662 |
| `Multiply/rows:512/f64:0/real_time` | 4.2914 | 10.0455 | 68.6595 | 0.427197 | 0.0625026 | 1.23643 | 0.140598 | 0.119059 |
| `Multiply/rows:512/f64:1/real_time` | 8.34492 | 12.421 | 76.4179 | 0.671838 | 0.109201 | 0.334818 | 3.51565 | 1.25138 |
| `Multiply/rows:8192/f64:0/real_time` | 68.193 | 60.2454 | 138.682 | 1.13192 | 0.491722 | 0.113506 | 0.295926 | 0.132603 |
| `Multiply/rows:8192/f64:1/real_time` | 136.824 | 142.622 | 206.694 | 0.959351 | 0.661966 | 0.130076 | 0.0866093 | 0.211367 |
| `SquareGradient/rows:512/f64:0/real_time` | 30.0624 | 128.356 | 81.4474 | 0.234211 | 0.369102 | 1.04043 | 0.960812 | 0.432142 |
| `SquareGradient/rows:512/f64:1/real_time` | 33.2785 | 220.031 | 92.8068 | 0.151244 | 0.358579 | 0.922947 | 0.251248 | 0.998895 |
| `SquareGradient/rows:8192/f64:0/real_time` | 449.948 | 573.785 | 245.422 | 0.784175 | 1.83337 | 0.0539333 | 0.147161 | 0.105207 |
| `SquareGradient/rows:8192/f64:1/real_time` | 525.496 | 1179.32 | 423.472 | 0.445591 | 1.24092 | 0.317174 | 0.293792 | 0.317086 |
| `SumAll/rows:512/f64:0/real_time` | 5.99506 | 11.978 | 67.4941 | 0.500508 | 0.0888235 | 0.128363 | 3.71779 | 1.54658 |
| `SumAll/rows:512/f64:1/real_time` | 5.59188 | 14.4466 | 70.2931 | 0.387073 | 0.0795508 | 1.18666 | 0.14619 | 0.488322 |
| `SumAll/rows:8192/f64:0/real_time` | 92.096 | 50.0159 | 111.502 | 1.84133 | 0.825962 | 0.1165 | 0.438747 | 0.589688 |
| `SumAll/rows:8192/f64:1/real_time` | 83.5564 | 91.0096 | 152.846 | 0.918105 | 0.54667 | 1.99862 | 0.192505 | 0.303537 |
| `SumLast/rows:512/f64:0/real_time` | 6.2207 | 9.4594 | 68.1249 | 0.657621 | 0.0913131 | 0.256743 | 0.239944 | 0.40591 |
| `SumLast/rows:512/f64:1/real_time` | 4.68535 | 11.6908 | 70.4925 | 0.400772 | 0.066466 | 1.48287 | 0.604647 | 0.483851 |
| `SumLast/rows:8192/f64:0/real_time` | 94.7875 | 48.4669 | 113.047 | 1.95572 | 0.838479 | 0.104623 | 0.0869584 | 0.314648 |
| `SumLast/rows:8192/f64:1/real_time` | 74.8593 | 85.7715 | 150.981 | 0.872777 | 0.495819 | 0.915847 | 3.46686 | 0.251453 |

</details>

<details><summary>release_latest_repeat_tf — 2 cases — daf6388e89bc</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/daf6388e89bc45aafd5e29151b37061c9736c6578384d9034babc494e7f108cd.json). Run date: `2026-10-24T19:04:40.778198+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SquareGradient/rows:8192/f64:0/real_time` | 1411.76 | 4131.53 | 1559.93 | 0.341704 | 0.905018 | 3.23328 | 4.91214 | 11.0532 |
| `SquareGradient/rows:8192/f64:1/real_time` | 2473.44 | 7429.41 | 2636.96 | 0.332926 | 0.93799 | 5.34291 | 5.77379 | 4.05091 |

</details>

<details><summary>release_latest_tf — 72 cases — e316ff812b7d</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/e316ff812b7dd38b1d8348a9446cc818f5797f2e1846bcb17f9f7de34e3c00fc.json). Run date: `2026-10-24T18:59:34.706176+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:32/f64:0/real_time` | 1.90924 | 53.2375 | 284.725 | 0.0358627 | 0.00670556 | 6.69733 | 3.11117 | 10.1724 |
| `Add/rows:32/f64:1/real_time` | 4.11819 | 55.813 | 288.958 | 0.0737854 | 0.0142518 | 1.38407 | 6.00367 | 6.15878 |
| `Add/rows:512/f64:0/real_time` | 23.3511 | 65.5774 | 296.25 | 0.356084 | 0.0788222 | 11.3589 | 10.2244 | 8.134 |
| `Add/rows:512/f64:1/real_time` | 43.4647 | 93.4984 | 344.089 | 0.464871 | 0.126318 | 6.92491 | 7.6945 | 9.41577 |
| `Add/rows:8192/f64:0/real_time` | 415.494 | 661.921 | 985.372 | 0.627709 | 0.421662 | 10.2171 | 3.66533 | 7.42567 |
| `Add/rows:8192/f64:1/real_time` | 1091.75 | 1111.87 | 1470.6 | 0.981897 | 0.742381 | 2.98379 | 8.34498 | 5.30686 |
| `BroadcastAdd/rows:32/f64:0/real_time` | 2.7771 | 55.8362 | 233.598 | 0.0497365 | 0.0118883 | 4.44249 | 5.07678 | 30.5273 |
| `BroadcastAdd/rows:32/f64:1/real_time` | 4.36161 | 58.469 | 297.957 | 0.0745969 | 0.0146384 | 10.4886 | 4.38702 | 2.02164 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 32.6083 | 104.925 | 354.007 | 0.310777 | 0.092112 | 2.29538 | 14.5331 | 5.19809 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 46.0938 | 165.986 | 365.984 | 0.277696 | 0.125945 | 9.31833 | 2.87738 | 14.5115 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 564.899 | 1169.59 | 1383.94 | 0.482989 | 0.408181 | 6.94514 | 4.63061 | 9.46079 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 1212.42 | 1819.96 | 2206.36 | 0.666178 | 0.549511 | 6.68443 | 7.06697 | 3.44079 |
| `Exp/rows:32/f64:0/real_time` | 6.59561 | 45.2717 | 309.046 | 0.14569 | 0.0213418 | 4.16641 | 14.5277 | 1.9771 |
| `Exp/rows:32/f64:1/real_time` | 15.3245 | 57.9685 | 292.891 | 0.264359 | 0.0523216 | 10.9384 | 1.92784 | 11.9199 |
| `Exp/rows:512/f64:0/real_time` | 84.7055 | 140.112 | 374.353 | 0.604556 | 0.226272 | 7.40168 | 9.05944 | 13.3826 |
| `Exp/rows:512/f64:1/real_time` | 289.516 | 310.617 | 566.143 | 0.932067 | 0.511382 | 3.26501 | 5.75662 | 12.5376 |
| `Exp/rows:8192/f64:0/real_time` | 1674.7 | 2240.63 | 2799.91 | 0.747424 | 0.598126 | 10.113 | 2.13498 | 1.79887 |
| `Exp/rows:8192/f64:1/real_time` | 4500.14 | 5153 | 5989.89 | 0.873306 | 0.75129 | 14.4271 | 16.5153 | 3.01679 |
| `LogSoftmax/rows:32/f64:0/real_time` | 12.5807 | 56.4977 | 288.921 | 0.222677 | 0.0435439 | 1.45044 | 1.05527 | 11.6569 |
| `LogSoftmax/rows:32/f64:1/real_time` | 23.3112 | 76.092 | 318.764 | 0.306355 | 0.0731298 | 23.1804 | 1.23899 | 3.45395 |
| `LogSoftmax/rows:512/f64:0/real_time` | 156.236 | 274.822 | 453.769 | 0.568498 | 0.344307 | 5.89115 | 6.83939 | 10.5138 |
| `LogSoftmax/rows:512/f64:1/real_time` | 409.408 | 526.006 | 856.166 | 0.778333 | 0.478187 | 7.92097 | 7.61168 | 13.2253 |
| `LogSoftmax/rows:8192/f64:0/real_time` | 2955.09 | 4590.17 | 5317.88 | 0.643786 | 0.555689 | 5.57644 | 5.82938 | 12.1418 |
| `LogSoftmax/rows:8192/f64:1/real_time` | 7434.35 | 9458.43 | 10337.7 | 0.786003 | 0.719153 | 1.75371 | 10.0426 | 8.1894 |
| `Matmul/rows:256/f64:0/real_time` | 212.386 | 1222.75 | 1575.78 | 0.173696 | 0.134782 | 9.46846 | 3.2852 | 4.88577 |
| `Matmul/rows:256/f64:1/real_time` | 655.741 | 2753.3 | 2910.46 | 0.238166 | 0.225305 | 15.9611 | 8.28698 | 10.6918 |
| `Matmul/rows:512/f64:0/real_time` | 1619.55 | 8725.38 | 10253.4 | 0.185614 | 0.157954 | 2.88621 | 9.49883 | 2.14422 |
| `Matmul/rows:512/f64:1/real_time` | 5358.53 | 21170.2 | 21478.6 | 0.253117 | 0.249482 | 5.41158 | 8.61896 | 4.37486 |
| `Matmul/rows:64/f64:0/real_time` | 5.98358 | 80.0674 | 316.949 | 0.0747318 | 0.0188787 | 1.44715 | 1.32733 | 1.82132 |
| `Matmul/rows:64/f64:1/real_time` | 15.1627 | 101.152 | 338.448 | 0.1499 | 0.0448007 | 1.34633 | 11.8184 | 4.04184 |
| `Mean/rows:32/f64:0/real_time` | 2.65641 | 53.2627 | 241.084 | 0.0498738 | 0.0110186 | 11.72 | 8.18596 | 12.0587 |
| `Mean/rows:32/f64:1/real_time` | 2.68772 | 49.3111 | 262.658 | 0.0545054 | 0.0102328 | 10.903 | 8.17866 | 12.819 |
| `Mean/rows:512/f64:0/real_time` | 16.2214 | 60.5135 | 295.823 | 0.268063 | 0.054835 | 5.94274 | 11.4931 | 3.0271 |
| `Mean/rows:512/f64:1/real_time` | 24.3911 | 81.2857 | 324.538 | 0.300067 | 0.0751565 | 7.93177 | 10.9044 | 3.24212 |
| `Mean/rows:8192/f64:0/real_time` | 290.349 | 294.366 | 648.883 | 0.986354 | 0.447459 | 11.3865 | 1.73897 | 7.24865 |
| `Mean/rows:8192/f64:1/real_time` | 467.277 | 749.321 | 1229.87 | 0.623601 | 0.37994 | 8.74885 | 6.77441 | 17.2964 |
| `Multiply/rows:32/f64:0/real_time` | 1.70721 | 35.7084 | 278.178 | 0.0478096 | 0.00613709 | 8.59883 | 5.88513 | 5.37453 |
| `Multiply/rows:32/f64:1/real_time` | 3.8875 | 39.2418 | 302.951 | 0.0990652 | 0.0128321 | 1.47524 | 13.2849 | 2.89346 |
| `Multiply/rows:512/f64:0/real_time` | 26.9099 | 63.1984 | 313.073 | 0.425801 | 0.0859541 | 2.06891 | 4.0484 | 2.06481 |
| `Multiply/rows:512/f64:1/real_time` | 50.7608 | 93.3452 | 346.79 | 0.543797 | 0.146373 | 10.7796 | 2.22403 | 2.08919 |
| `Multiply/rows:8192/f64:0/real_time` | 573.648 | 593.305 | 817.069 | 0.966869 | 0.70208 | 6.56784 | 14.3591 | 11.7702 |
| `Multiply/rows:8192/f64:1/real_time` | 1071.38 | 1230.31 | 1503.57 | 0.870827 | 0.71256 | 8.3167 | 7.71245 | 9.45925 |
| `Relu/rows:32/f64:0/real_time` | 1.09923 | 23.2009 | 272.202 | 0.0473789 | 0.0040383 | 6.4353 | 6.88857 | 7.78935 |
| `Relu/rows:32/f64:1/real_time` | 2.69861 | 29.7326 | 278.449 | 0.0907625 | 0.00969155 | 6.54834 | 5.52229 | 6.71108 |
| `Relu/rows:512/f64:0/real_time` | 17.0074 | 48.0259 | 296.061 | 0.35413 | 0.0574455 | 4.85796 | 5.25683 | 6.25732 |
| `Relu/rows:512/f64:1/real_time` | 42.4731 | 68.1166 | 341.827 | 0.623535 | 0.124253 | 5.47414 | 6.48109 | 5.02066 |
| `Relu/rows:8192/f64:0/real_time` | 453.888 | 519.246 | 829.82 | 0.87413 | 0.546972 | 2.65835 | 8.14496 | 14.0166 |
| `Relu/rows:8192/f64:1/real_time` | 988.965 | 903.188 | 1612.77 | 1.09497 | 0.613208 | 4.21858 | 10.0023 | 2.07055 |
| `Softmax/rows:32/f64:0/real_time` | 12.1325 | 54.2585 | 307.742 | 0.223606 | 0.0394245 | 0.720535 | 2.8851 | 10.1073 |
| `Softmax/rows:32/f64:1/real_time` | 23.2831 | 73.5751 | 325.512 | 0.316454 | 0.0715278 | 17.326 | 1.44212 | 10.6107 |
| `Softmax/rows:512/f64:0/real_time` | 152.174 | 201.024 | 466.813 | 0.756998 | 0.325986 | 6.0474 | 9.55946 | 11.1583 |
| `Softmax/rows:512/f64:1/real_time` | 424.572 | 445.257 | 731.595 | 0.953544 | 0.580337 | 7.05508 | 4.21898 | 6.80768 |
| `Softmax/rows:8192/f64:0/real_time` | 2870.81 | 4105.68 | 4564.41 | 0.699229 | 0.628956 | 10.3743 | 2.57246 | 3.17686 |
| `Softmax/rows:8192/f64:1/real_time` | 6880.05 | 8364.8 | 9960.08 | 0.822501 | 0.690763 | 6.09573 | 4.11792 | 2.68915 |
| `SquareGradient/rows:32/f64:0/real_time` | 12.7043 | 561.462 | 298.857 | 0.0226273 | 0.0425097 | 12.4066 | 6.08192 | 4.4586 |
| `SquareGradient/rows:32/f64:1/real_time` | 16.4922 | 932.186 | 292.612 | 0.017692 | 0.0563621 | 2.40254 | 2.90435 | 20.7053 |
| `SquareGradient/rows:512/f64:0/real_time` | 74.3492 | 739.112 | 369.715 | 0.100593 | 0.201099 | 1.42887 | 6.13224 | 2.93195 |
| `SquareGradient/rows:512/f64:1/real_time` | 111.669 | 1208.99 | 384.987 | 0.092366 | 0.29006 | 6.19651 | 6.96103 | 6.94919 |
| `SquareGradient/rows:8192/f64:0/real_time` | 1259.16 | 4514.15 | 1680.55 | 0.278937 | 0.749256 | 8.02015 | 9.11185 | 9.09742 |
| `SquareGradient/rows:8192/f64:1/real_time` | 2662.37 | 9291.25 | 3030.51 | 0.286546 | 0.878523 | 3.76565 | 1.41409 | 2.94752 |
| `SumAll/rows:32/f64:0/real_time` | 2.17846 | 54.2569 | 268.61 | 0.0401508 | 0.00811011 | 12.1732 | 6.90975 | 8.79979 |
| `SumAll/rows:32/f64:1/real_time` | 2.12826 | 56.2743 | 226.696 | 0.0378194 | 0.00938815 | 11.2813 | 4.3847 | 6.97508 |
| `SumAll/rows:512/f64:0/real_time` | 16.8861 | 62.6129 | 243.128 | 0.269691 | 0.0694537 | 6.45189 | 12.599 | 4.38767 |
| `SumAll/rows:512/f64:1/real_time` | 22.5791 | 75.5385 | 316.547 | 0.298908 | 0.0713292 | 8.51632 | 8.75937 | 10.5452 |
| `SumAll/rows:8192/f64:0/real_time` | 235.785 | 269.915 | 591.467 | 0.873551 | 0.398643 | 8.65369 | 7.43429 | 12.0166 |
| `SumAll/rows:8192/f64:1/real_time` | 555.797 | 715.272 | 1181.04 | 0.777043 | 0.470598 | 7.64158 | 12.0439 | 10.6256 |
| `SumLast/rows:32/f64:0/real_time` | 1.96954 | 43.8849 | 291.623 | 0.0448797 | 0.00675371 | 10.617 | 6.85019 | 3.78484 |
| `SumLast/rows:32/f64:1/real_time` | 1.78476 | 43.3941 | 265.445 | 0.041129 | 0.00672364 | 8.91933 | 10.7298 | 6.61031 |
| `SumLast/rows:512/f64:0/real_time` | 19.6741 | 54.9968 | 264.397 | 0.357732 | 0.0744112 | 6.95171 | 5.89259 | 8.3737 |
| `SumLast/rows:512/f64:1/real_time` | 23.5336 | 61.3211 | 296.503 | 0.383776 | 0.0793704 | 5.22537 | 11.0445 | 6.72313 |
| `SumLast/rows:8192/f64:0/real_time` | 281.793 | 291.925 | 666.269 | 0.965293 | 0.422942 | 5.58741 | 5.30976 | 8.2316 |
| `SumLast/rows:8192/f64:1/real_time` | 608.525 | 779.239 | 1222.62 | 0.780922 | 0.497723 | 19.4913 | 3.93739 | 2.67431 |

</details>

<details><summary>release_pass_after_tf — 72 cases — 90861f341f0f</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/90861f341f0f775b2afa4e0b3bf3777690273af6cbd2e4fb6b9931a7cefb1f6c.json). Run date: `2026-10-24T18:39:08.328303+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:32/f64:0/real_time` | 0.451929 | 9.1854 | 65.6516 | 0.0492008 | 0.00688375 | 3.59169 | 0.909289 | 11.534 |
| `Add/rows:32/f64:1/real_time` | 0.652579 | 9.33677 | 66.4562 | 0.0698935 | 0.00981969 | 1.6951 | 0.686055 | 2.25344 |
| `Add/rows:512/f64:0/real_time` | 4.51067 | 13.8484 | 71.6114 | 0.325718 | 0.0629883 | 4.57807 | 1.32334 | 1.16053 |
| `Add/rows:512/f64:1/real_time` | 8.85543 | 18.0437 | 76.8089 | 0.490777 | 0.115292 | 5.1399 | 0.390494 | 1.05945 |
| `Add/rows:8192/f64:0/real_time` | 75.7686 | 83.3192 | 146.453 | 0.909376 | 0.517359 | 2.52819 | 5.38306 | 2.2877 |
| `Add/rows:8192/f64:1/real_time` | 153.1 | 165.524 | 223.306 | 0.924944 | 0.685608 | 1.69742 | 2.32891 | 0.821373 |
| `BroadcastAdd/rows:32/f64:0/real_time` | 0.566057 | 10.0128 | 68.1785 | 0.0565336 | 0.00830258 | 3.41637 | 2.36386 | 0.39203 |
| `BroadcastAdd/rows:32/f64:1/real_time` | 0.762932 | 10.8726 | 68.2253 | 0.0701703 | 0.0111825 | 1.36965 | 6.0937 | 1.20159 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 5.51732 | 19.5709 | 78.7437 | 0.281915 | 0.0700668 | 1.27093 | 2.11278 | 1.12673 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 10.6121 | 29.199 | 89.6014 | 0.363441 | 0.118437 | 1.74459 | 1.26431 | 0.889584 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 89.4176 | 178.51 | 236.81 | 0.50091 | 0.377593 | 0.668506 | 1.94315 | 1.61074 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 159.334 | 358.737 | 386.9 | 0.444153 | 0.411822 | 3.63123 | 2.1401 | 0.0875589 |
| `Exp/rows:32/f64:0/real_time` | 1.42028 | 7.34598 | 67.3475 | 0.193341 | 0.0210888 | 0.381187 | 0.420052 | 0.240744 |
| `Exp/rows:32/f64:1/real_time` | 4.16012 | 9.24555 | 69.5296 | 0.44996 | 0.0598324 | 0.831291 | 0.094215 | 1.64048 |
| `Exp/rows:512/f64:0/real_time` | 21.0493 | 30.3475 | 95.7074 | 0.693609 | 0.219934 | 0.355339 | 0.182431 | 0.122773 |
| `Exp/rows:512/f64:1/real_time` | 65.0373 | 60.6791 | 130.083 | 1.07182 | 0.49997 | 0.0833477 | 0.0368522 | 2.87313 |
| `Exp/rows:8192/f64:0/real_time` | 334.679 | 394.173 | 469.456 | 0.849066 | 0.712908 | 0.0943598 | 0.128486 | 2.45348 |
| `Exp/rows:8192/f64:1/real_time` | 1039.24 | 880.735 | 947.711 | 1.17997 | 1.09658 | 0.0563511 | 0.349739 | 0.114516 |
| `LogSoftmax/rows:32/f64:0/real_time` | 2.57009 | 9.99325 | 69.0843 | 0.257182 | 0.0372022 | 1.47828 | 0.301627 | 0.211315 |
| `LogSoftmax/rows:32/f64:1/real_time` | 6.57598 | 13.8856 | 72.936 | 0.473582 | 0.090161 | 0.207188 | 0.169163 | 0.413191 |
| `LogSoftmax/rows:512/f64:0/real_time` | 39.0235 | 53.9114 | 118.729 | 0.723845 | 0.328678 | 0.279914 | 0.0721003 | 0.130052 |
| `LogSoftmax/rows:512/f64:1/real_time` | 103.702 | 106.924 | 172.383 | 0.969868 | 0.601582 | 0.157892 | 0.10825 | 0.703939 |
| `LogSoftmax/rows:8192/f64:0/real_time` | 622.076 | 756.717 | 824.965 | 0.822072 | 0.754063 | 0.295416 | 0.0994175 | 0.0887558 |
| `LogSoftmax/rows:8192/f64:1/real_time` | 1656.64 | 1625.95 | 1713.54 | 1.01888 | 0.966796 | 0.073157 | 2.42209 | 0.271735 |
| `Matmul/rows:256/f64:0/real_time` | 27.3264 | 322.609 | 352.788 | 0.0847045 | 0.0774584 | 4.42322 | 0.717229 | 4.35997 |
| `Matmul/rows:256/f64:1/real_time` | 93.1651 | 721.166 | 842.203 | 0.129187 | 0.110621 | 3.90317 | 0.647479 | 0.423167 |
| `Matmul/rows:512/f64:0/real_time` | 197.546 | 2282.43 | 2311.29 | 0.0865507 | 0.0854699 | 2.65274 | 1.27106 | 4.31907 |
| `Matmul/rows:512/f64:1/real_time` | 623.901 | 6084 | 5643.03 | 0.102548 | 0.110561 | 1.36714 | 3.06699 | 1.71002 |
| `Matmul/rows:64/f64:0/real_time` | 0.9935 | 16.6771 | 76.8778 | 0.0595725 | 0.0129231 | 0.556066 | 1.66094 | 2.3992 |
| `Matmul/rows:64/f64:1/real_time` | 2.25388 | 23.372 | 82.6714 | 0.096435 | 0.0272631 | 0.939128 | 2.12874 | 0.947282 |
| `Mean/rows:32/f64:0/real_time` | 0.696233 | 9.63444 | 65.057 | 0.0722651 | 0.0107019 | 0.282192 | 0.280221 | 0.219967 |
| `Mean/rows:32/f64:1/real_time` | 0.626398 | 9.91567 | 66.6607 | 0.0631725 | 0.00939681 | 0.36303 | 0.590302 | 5.54745 |
| `Mean/rows:512/f64:0/real_time` | 4.28934 | 12.3347 | 69.0875 | 0.347745 | 0.0620857 | 0.120551 | 0.160805 | 4.73719 |
| `Mean/rows:512/f64:1/real_time` | 4.30799 | 14.8216 | 71.2378 | 0.290656 | 0.0604733 | 0.329499 | 0.745942 | 0.410975 |
| `Mean/rows:8192/f64:0/real_time` | 61.8176 | 50.3715 | 112.193 | 1.22723 | 0.550993 | 1.44386 | 0.162589 | 1.24867 |
| `Mean/rows:8192/f64:1/real_time` | 72.2669 | 91.1789 | 152.811 | 0.792583 | 0.472916 | 0.645526 | 0.0374297 | 0.226185 |
| `Multiply/rows:32/f64:0/real_time` | 0.376824 | 6.5879 | 69.6551 | 0.0571994 | 0.00540985 | 1.9588 | 1.10752 | 2.51193 |
| `Multiply/rows:32/f64:1/real_time` | 0.628018 | 6.9413 | 66.5525 | 0.0904755 | 0.00943644 | 4.02099 | 2.12865 | 10.1662 |
| `Multiply/rows:512/f64:0/real_time` | 4.62158 | 11.3763 | 75.2371 | 0.406246 | 0.0614269 | 0.912001 | 1.88218 | 12.289 |
| `Multiply/rows:512/f64:1/real_time` | 9.19025 | 15.7488 | 80.2707 | 0.583553 | 0.114491 | 8.84076 | 1.40059 | 1.10482 |
| `Multiply/rows:8192/f64:0/real_time` | 75.5876 | 106.475 | 139.438 | 0.709908 | 0.542088 | 2.2262 | 0.760744 | 0.81001 |
| `Multiply/rows:8192/f64:1/real_time` | 153.119 | 192.475 | 221.059 | 0.795528 | 0.692662 | 0.860055 | 3.38381 | 1.70953 |
| `Relu/rows:32/f64:0/real_time` | 0.291605 | 4.91543 | 64.5375 | 0.0593244 | 0.00451838 | 2.81073 | 0.534723 | 0.328506 |
| `Relu/rows:32/f64:1/real_time` | 0.444433 | 5.08958 | 65.574 | 0.0873222 | 0.00677758 | 2.2607 | 0.389291 | 0.247258 |
| `Relu/rows:512/f64:0/real_time` | 4.22338 | 9.03282 | 70.0037 | 0.467559 | 0.0603307 | 0.316742 | 0.629048 | 0.310312 |
| `Relu/rows:512/f64:1/real_time` | 8.26737 | 12.8663 | 75.5422 | 0.642559 | 0.10944 | 0.11357 | 6.06151 | 0.919212 |
| `Relu/rows:8192/f64:0/real_time` | 67.8292 | 60.2611 | 139.109 | 1.12559 | 0.487597 | 0.365947 | 11.1209 | 0.110879 |
| `Relu/rows:8192/f64:1/real_time` | 135.531 | 141.071 | 207.488 | 0.960733 | 0.653202 | 0.13057 | 0.214448 | 0.135396 |
| `Softmax/rows:32/f64:0/real_time` | 2.42198 | 9.63259 | 69.0143 | 0.251436 | 0.0350939 | 2.12888 | 0.318517 | 1.39037 |
| `Softmax/rows:32/f64:1/real_time` | 6.274 | 12.8261 | 71.6947 | 0.489158 | 0.08751 | 1.7184 | 0.140832 | 0.261139 |
| `Softmax/rows:512/f64:0/real_time` | 36.6715 | 48.9699 | 114.407 | 0.748857 | 0.320534 | 0.0868905 | 0.0896626 | 0.935271 |
| `Softmax/rows:512/f64:1/real_time` | 98.2445 | 96.0008 | 160.819 | 1.02337 | 0.610901 | 0.0497058 | 0.0586624 | 0.316776 |
| `Softmax/rows:8192/f64:0/real_time` | 585.356 | 672.186 | 742.546 | 0.870825 | 0.78831 | 0.0772324 | 0.567 | 2.57802 |
| `Softmax/rows:8192/f64:1/real_time` | 1575.61 | 1449.39 | 1528.19 | 1.08708 | 1.03103 | 0.126969 | 0.0463261 | 0.173135 |
| `SquareGradient/rows:32/f64:0/real_time` | 2.33468 | 99.8575 | 65.3545 | 0.0233801 | 0.0357233 | 0.764219 | 3.7799 | 0.381876 |
| `SquareGradient/rows:32/f64:1/real_time` | 2.68131 | 167.365 | 65.5381 | 0.0160207 | 0.0409123 | 0.572039 | 0.888597 | 0.541249 |
| `SquareGradient/rows:512/f64:0/real_time` | 13.9203 | 130.776 | 81.7808 | 0.106444 | 0.170215 | 0.78882 | 0.192852 | 0.385861 |
| `SquareGradient/rows:512/f64:1/real_time` | 22.8535 | 226.724 | 93.2934 | 0.100799 | 0.244964 | 1.50438 | 0.556162 | 0.252838 |
| `SquareGradient/rows:8192/f64:0/real_time` | 199.524 | 611.617 | 253.665 | 0.326224 | 0.786565 | 0.101471 | 0.811587 | 5.94098 |
| `SquareGradient/rows:8192/f64:1/real_time` | 405.39 | 1321.39 | 430.883 | 0.306791 | 0.940835 | 24.3798 | 2.35691 | 5.64122 |
| `SumAll/rows:32/f64:0/real_time` | 0.551908 | 10.8812 | 65.1684 | 0.050721 | 0.00846896 | 1.95592 | 1.06658 | 1.45452 |
| `SumAll/rows:32/f64:1/real_time` | 0.439726 | 10.3378 | 65.3075 | 0.0425358 | 0.00673317 | 1.82401 | 2.07281 | 0.402418 |
| `SumAll/rows:512/f64:0/real_time` | 4.10752 | 12.2925 | 68.1703 | 0.334147 | 0.0602537 | 0.381172 | 0.206628 | 1.58107 |
| `SumAll/rows:512/f64:1/real_time` | 4.09713 | 14.7156 | 71.6312 | 0.278421 | 0.0571976 | 0.220086 | 0.0816485 | 1.22926 |
| `SumAll/rows:8192/f64:0/real_time` | 61.5499 | 50.2606 | 111.97 | 1.22462 | 0.549699 | 2.25911 | 0.085293 | 0.362569 |
| `SumAll/rows:8192/f64:1/real_time` | 72.0606 | 91.227 | 153.273 | 0.789904 | 0.470146 | 3.11802 | 0.104091 | 0.255937 |
| `SumLast/rows:32/f64:0/real_time` | 0.623657 | 7.81597 | 64.8113 | 0.0797927 | 0.00962266 | 11.6975 | 8.3742 | 2.90256 |
| `SumLast/rows:32/f64:1/real_time` | 0.543024 | 9.75613 | 68.4807 | 0.0556597 | 0.00792959 | 1.84025 | 6.35033 | 1.68023 |
| `SumLast/rows:512/f64:0/real_time` | 4.76564 | 11.9209 | 69.6669 | 0.399771 | 0.0684061 | 5.11033 | 5.70566 | 3.00481 |
| `SumLast/rows:512/f64:1/real_time` | 4.54282 | 12.8734 | 75.7228 | 0.352885 | 0.0599928 | 11.683 | 1.63872 | 0.597705 |
| `SumLast/rows:8192/f64:0/real_time` | 71.7126 | 54.2413 | 116.686 | 1.3221 | 0.614576 | 1.2285 | 0.771899 | 1.20021 |
| `SumLast/rows:8192/f64:1/real_time` | 77.8969 | 100.294 | 159.561 | 0.776686 | 0.488195 | 2.65587 | 0.931168 | 8.10146 |

</details>

<details><summary>release_pass_before_tf — 72 cases — 7a7d16edba78</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/7a7d16edba781f00c2c95f136ceb6fe4b1eca4a459c65cb280b6c2326305b50f.json). Run date: `2026-10-24T18:31:42.878887+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:32/f64:0/real_time` | 0.456222 | 8.33942 | 64.249 | 0.0547067 | 0.00710085 | 9.56033 | 0.231851 | 0.482413 |
| `Add/rows:32/f64:1/real_time` | 0.603664 | 8.66986 | 65.0538 | 0.0696279 | 0.00927947 | 1.19389 | 0.691797 | 0.202844 |
| `Add/rows:512/f64:0/real_time` | 4.3449 | 11.8439 | 69.1794 | 0.366846 | 0.0628064 | 0.810157 | 0.853637 | 0.204238 |
| `Add/rows:512/f64:1/real_time` | 8.40699 | 16.7123 | 78.5122 | 0.503041 | 0.107079 | 0.249775 | 0.0909219 | 0.325541 |
| `Add/rows:8192/f64:0/real_time` | 68.3407 | 77.0564 | 135.013 | 0.886892 | 0.506178 | 0.257034 | 0.0785267 | 2.36005 |
| `Add/rows:8192/f64:1/real_time` | 137.275 | 145.781 | 207.294 | 0.941652 | 0.662224 | 0.228205 | 0.186402 | 0.0374655 |
| `BroadcastAdd/rows:32/f64:0/real_time` | 0.487136 | 8.98552 | 66.0254 | 0.0542134 | 0.007378 | 1.34432 | 0.30405 | 0.279183 |
| `BroadcastAdd/rows:32/f64:1/real_time` | 0.667807 | 9.82733 | 66.7762 | 0.0679541 | 0.0100007 | 0.606259 | 3.50288 | 1.25353 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 4.9453 | 18.7057 | 80.039 | 0.264373 | 0.0617861 | 0.895712 | 0.500781 | 0.307564 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 9.51931 | 28.0181 | 90.4241 | 0.339755 | 0.105274 | 0.203199 | 0.319441 | 0.326531 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 78.9703 | 166.567 | 227.785 | 0.474106 | 0.346687 | 0.41805 | 0.129807 | 0.340666 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 136.326 | 319.784 | 380.914 | 0.426306 | 0.357891 | 10.8204 | 0.864457 | 0.14299 |
| `Exp/rows:32/f64:0/real_time` | 1.50662 | 7.74077 | 68.6259 | 0.194635 | 0.0219542 | 1.25622 | 4.2384 | 2.62603 |
| `Exp/rows:32/f64:1/real_time` | 4.45274 | 9.58191 | 70.5391 | 0.464703 | 0.0631244 | 0.744681 | 0.638215 | 0.48774 |
| `Exp/rows:512/f64:0/real_time` | 22.7991 | 31.7878 | 94.6315 | 0.717227 | 0.240925 | 1.73842 | 0.75606 | 0.799848 |
| `Exp/rows:512/f64:1/real_time` | 69.5883 | 65.4401 | 128.922 | 1.06339 | 0.53977 | 0.649237 | 2.20788 | 9.98401 |
| `Exp/rows:8192/f64:0/real_time` | 359.495 | 419.992 | 564.434 | 0.855958 | 0.636913 | 1.23121 | 2.33184 | 6.67741 |
| `Exp/rows:8192/f64:1/real_time` | 1152.11 | 1048.06 | 1029.91 | 1.09927 | 1.11865 | 1.46649 | 6.99041 | 0.961687 |
| `LogSoftmax/rows:32/f64:0/real_time` | 2.77864 | 10.494 | 72.506 | 0.264783 | 0.0383229 | 0.871648 | 0.75567 | 10.4884 |
| `LogSoftmax/rows:32/f64:1/real_time` | 7.29788 | 15.0345 | 76.2593 | 0.48541 | 0.0956982 | 2.78837 | 0.40358 | 0.619659 |
| `LogSoftmax/rows:512/f64:0/real_time` | 43.2948 | 57.5257 | 118.592 | 0.752615 | 0.365074 | 3.535 | 0.895122 | 0.415118 |
| `LogSoftmax/rows:512/f64:1/real_time` | 115.579 | 117.858 | 181.505 | 0.980663 | 0.63678 | 1.14679 | 0.293278 | 0.576592 |
| `LogSoftmax/rows:8192/f64:0/real_time` | 682.509 | 824.853 | 896.148 | 0.827431 | 0.761603 | 1.02369 | 0.515813 | 0.617616 |
| `LogSoftmax/rows:8192/f64:1/real_time` | 1813.74 | 1822.91 | 1831.75 | 0.994969 | 0.990167 | 1.28727 | 0.186338 | 0.990723 |
| `Matmul/rows:256/f64:0/real_time` | 25.084 | 319.823 | 373.554 | 0.0784309 | 0.0671497 | 4.55433 | 1.64333 | 1.17285 |
| `Matmul/rows:256/f64:1/real_time` | 92.0436 | 809.515 | 876.913 | 0.113702 | 0.104963 | 4.00348 | 0.689486 | 0.34729 |
| `Matmul/rows:512/f64:0/real_time` | 193.959 | 2481.56 | 2445.68 | 0.0781604 | 0.0793069 | 1.78767 | 1.67929 | 0.346791 |
| `Matmul/rows:512/f64:1/real_time` | 647.07 | 6128.83 | 6511.33 | 0.105578 | 0.099376 | 3.42976 | 0.179313 | 2.26232 |
| `Matmul/rows:64/f64:0/real_time` | 1.14725 | 16.908 | 70.456 | 0.0678524 | 0.0162832 | 3.09744 | 0.602672 | 0.803798 |
| `Matmul/rows:64/f64:1/real_time` | 2.21243 | 24.2451 | 87.1487 | 0.091253 | 0.0253869 | 1.04897 | 5.35543 | 7.51631 |
| `Mean/rows:32/f64:0/real_time` | 0.697012 | 9.35896 | 63.8051 | 0.0744753 | 0.0109241 | 1.20365 | 0.144981 | 0.844576 |
| `Mean/rows:32/f64:1/real_time` | 0.793195 | 9.49507 | 64.3018 | 0.0835376 | 0.0123355 | 2.64153 | 0.0846581 | 0.235175 |
| `Mean/rows:512/f64:0/real_time` | 4.28612 | 11.96 | 68.284 | 0.358372 | 0.0627691 | 0.141716 | 0.236975 | 0.420548 |
| `Mean/rows:512/f64:1/real_time` | 5.90523 | 14.5418 | 70.7765 | 0.406087 | 0.0834348 | 0.546875 | 0.221654 | 5.51274 |
| `Mean/rows:8192/f64:0/real_time` | 61.6966 | 50.0749 | 111.53 | 1.23209 | 0.553186 | 0.103157 | 0.428449 | 4.63096 |
| `Mean/rows:8192/f64:1/real_time` | 88.2574 | 91.0791 | 152.896 | 0.96902 | 0.577237 | 0.0633825 | 0.0692191 | 0.392609 |
| `Multiply/rows:32/f64:0/real_time` | 0.3321 | 5.9944 | 64.8731 | 0.0554018 | 0.00511923 | 0.628592 | 0.120982 | 0.324999 |
| `Multiply/rows:32/f64:1/real_time` | 0.520944 | 6.22782 | 64.5353 | 0.0836479 | 0.00807223 | 2.01812 | 0.182183 | 6.19882 |
| `Multiply/rows:512/f64:0/real_time` | 4.30911 | 10.0854 | 69.5729 | 0.427262 | 0.0619367 | 0.476736 | 0.357001 | 6.4778 |
| `Multiply/rows:512/f64:1/real_time` | 8.36898 | 14.1319 | 77.9152 | 0.592206 | 0.107411 | 0.267765 | 0.18234 | 0.288157 |
| `Multiply/rows:8192/f64:0/real_time` | 68.1458 | 74.7899 | 138.598 | 0.911164 | 0.49168 | 0.216249 | 0.636385 | 0.217721 |
| `Multiply/rows:8192/f64:1/real_time` | 137.309 | 142.44 | 206.797 | 0.963978 | 0.66398 | 0.183086 | 0.201989 | 0.267347 |
| `Relu/rows:32/f64:0/real_time` | 0.288835 | 4.62918 | 64.2188 | 0.0623944 | 0.00449767 | 4.47191 | 0.737958 | 0.548112 |
| `Relu/rows:32/f64:1/real_time` | 0.499295 | 4.81762 | 64.348 | 0.103639 | 0.0077593 | 6.36739 | 0.0965738 | 0.19683 |
| `Relu/rows:512/f64:0/real_time` | 4.29604 | 8.53738 | 72.7288 | 0.503203 | 0.0590693 | 1.58672 | 0.76236 | 1.13315 |
| `Relu/rows:512/f64:1/real_time` | 8.64442 | 12.6895 | 80.2626 | 0.681226 | 0.107702 | 4.68585 | 1.07421 | 1.4393 |
| `Relu/rows:8192/f64:0/real_time` | 70.5825 | 77.1035 | 135.539 | 0.915424 | 0.520753 | 1.35173 | 0.331765 | 0.32332 |
| `Relu/rows:8192/f64:1/real_time` | 141.996 | 150.859 | 221.873 | 0.941249 | 0.639987 | 1.29727 | 4.84756 | 0.353908 |
| `Softmax/rows:32/f64:0/real_time` | 2.58372 | 10.5759 | 73.5694 | 0.244301 | 0.0351195 | 0.998769 | 7.0697 | 0.395656 |
| `Softmax/rows:32/f64:1/real_time` | 6.99105 | 14.1807 | 76.9652 | 0.492996 | 0.090834 | 0.925609 | 0.380141 | 0.679083 |
| `Softmax/rows:512/f64:0/real_time` | 42.5202 | 53.9996 | 117.197 | 0.787416 | 0.362809 | 4.58282 | 1.59818 | 1.70172 |
| `Softmax/rows:512/f64:1/real_time` | 109 | 103.582 | 168.608 | 1.05231 | 0.646472 | 3.84685 | 1.25099 | 0.576914 |
| `Softmax/rows:8192/f64:0/real_time` | 644.637 | 758.091 | 811.186 | 0.850343 | 0.794685 | 0.585069 | 1.36182 | 1.4522 |
| `Softmax/rows:8192/f64:1/real_time` | 1734.78 | 1609.71 | 1721.3 | 1.0777 | 1.00783 | 1.36486 | 0.214754 | 5.06867 |
| `SquareGradient/rows:32/f64:0/real_time` | 2.65529 | 105.053 | 65.3907 | 0.0252757 | 0.0406066 | 2.05123 | 1.5345 | 0.579712 |
| `SquareGradient/rows:32/f64:1/real_time` | 3.26424 | 180.26 | 66.9396 | 0.0181086 | 0.048764 | 1.95132 | 5.55279 | 0.366975 |
| `SquareGradient/rows:512/f64:0/real_time` | 14.9049 | 146.578 | 97.3536 | 0.101686 | 0.153101 | 0.64305 | 7.30697 | 2.47088 |
| `SquareGradient/rows:512/f64:1/real_time` | 31.8789 | 285.503 | 94.472 | 0.111659 | 0.337443 | 2.68211 | 6.75273 | 1.3148 |
| `SquareGradient/rows:8192/f64:0/real_time` | 226.926 | 674.454 | 257.178 | 0.336459 | 0.882368 | 2.44656 | 3.33247 | 2.00828 |
| `SquareGradient/rows:8192/f64:1/real_time` | 449.716 | 1665.39 | 543.844 | 0.270037 | 0.82692 | 18.4565 | 1.72845 | 1.45449 |
| `SumAll/rows:32/f64:0/real_time` | 0.500191 | 9.3505 | 64.3571 | 0.0534935 | 0.00777212 | 0.922129 | 0.301535 | 0.356528 |
| `SumAll/rows:32/f64:1/real_time` | 0.580716 | 9.49996 | 64.5183 | 0.0611282 | 0.00900079 | 0.400458 | 0.271979 | 0.135949 |
| `SumAll/rows:512/f64:0/real_time` | 4.08674 | 12.033 | 67.7547 | 0.339628 | 0.0603166 | 0.101304 | 0.531596 | 0.17038 |
| `SumAll/rows:512/f64:1/real_time` | 5.70269 | 14.5471 | 70.1688 | 0.392017 | 0.0812711 | 0.25939 | 0.752778 | 0.886452 |
| `SumAll/rows:8192/f64:0/real_time` | 61.5099 | 50.0772 | 110.424 | 1.2283 | 0.557031 | 0.0588653 | 0.0770574 | 0.336627 |
| `SumAll/rows:8192/f64:1/real_time` | 88.0413 | 90.9857 | 152.75 | 0.967639 | 0.576376 | 0.281545 | 0.10141 | 0.170526 |
| `SumLast/rows:32/f64:0/real_time` | 0.542645 | 6.97764 | 64.2508 | 0.0777692 | 0.00844573 | 0.49966 | 0.847521 | 0.127452 |
| `SumLast/rows:32/f64:1/real_time` | 0.555561 | 7.17793 | 64.3149 | 0.0773985 | 0.00863814 | 0.412085 | 1.09014 | 0.222916 |
| `SumLast/rows:512/f64:0/real_time` | 4.38913 | 9.53074 | 67.9549 | 0.460524 | 0.0645889 | 0.155431 | 0.0966988 | 0.326449 |
| `SumLast/rows:512/f64:1/real_time` | 4.52451 | 11.6912 | 70.5474 | 0.387001 | 0.0641343 | 0.147993 | 3.83189 | 0.33348 |
| `SumLast/rows:8192/f64:0/real_time` | 65.885 | 49.5678 | 112.256 | 1.32919 | 0.58692 | 0.369255 | 4.51445 | 0.247463 |
| `SumLast/rows:8192/f64:1/real_time` | 74.512 | 86.3423 | 152.238 | 0.862983 | 0.489443 | 0.946564 | 0.272241 | 1.50654 |

</details>

<details><summary>release_pass_final_tf — 72 cases — a332244c2534</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/a332244c253416192084fbdad99cf762c0b8f7c8a69104f4b9830f7203a92b8b.json). Run date: `2026-10-24T19:27:54.764975+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:32/f64:0/real_time` | 0.399053 | 8.43307 | 64.0637 | 0.04732 | 0.006229 | 0.87449 | 0.259488 | 0.438725 |
| `Add/rows:32/f64:1/real_time` | 0.589175 | 8.74839 | 64.4139 | 0.0673467 | 0.00914671 | 1.59847 | 2.67043 | 0.440474 |
| `Add/rows:512/f64:0/real_time` | 4.35637 | 12.4411 | 69.1193 | 0.350159 | 0.0630268 | 0.670207 | 3.65613 | 0.0750381 |
| `Add/rows:512/f64:1/real_time` | 8.4025 | 16.793 | 78.1316 | 0.500356 | 0.107543 | 0.133528 | 0.270221 | 0.270351 |
| `Add/rows:8192/f64:0/real_time` | 68.2018 | 77.3034 | 139.778 | 0.882261 | 0.487928 | 0.384636 | 0.0988672 | 8.80879 |
| `Add/rows:8192/f64:1/real_time` | 137.078 | 175.775 | 207.431 | 0.779848 | 0.660838 | 0.0999364 | 20.5797 | 0.112158 |
| `BroadcastAdd/rows:32/f64:0/real_time` | 0.489091 | 9.08989 | 66.2112 | 0.053806 | 0.00738684 | 1.93947 | 0.390179 | 0.225737 |
| `BroadcastAdd/rows:32/f64:1/real_time` | 0.663748 | 9.90529 | 66.78 | 0.0670095 | 0.00993933 | 0.903046 | 0.164914 | 0.340565 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 4.47402 | 18.844 | 79.485 | 0.237424 | 0.0562876 | 0.377171 | 0.433969 | 0.374847 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 9.55725 | 28.0774 | 91.0876 | 0.340389 | 0.104924 | 0.324362 | 0.521371 | 0.275179 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 68.1168 | 167.47 | 228.408 | 0.406741 | 0.298224 | 0.709566 | 0.138409 | 0.430032 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 172.317 | 318.073 | 380.595 | 0.541753 | 0.452757 | 2.24975 | 0.114902 | 0.0836793 |
| `Exp/rows:32/f64:0/real_time` | 1.41654 | 7.13875 | 67.8324 | 0.198429 | 0.0208829 | 2.43269 | 0.510111 | 0.348134 |
| `Exp/rows:32/f64:1/real_time` | 4.16217 | 9.02681 | 68.9467 | 0.46109 | 0.0603679 | 0.292092 | 0.120908 | 0.308836 |
| `Exp/rows:512/f64:0/real_time` | 21.0058 | 30.1627 | 95.8494 | 0.696415 | 0.219154 | 0.060992 | 0.456069 | 0.270628 |
| `Exp/rows:512/f64:1/real_time` | 65.0094 | 60.7506 | 126.668 | 1.0701 | 0.513228 | 0.0717334 | 0.397419 | 0.213784 |
| `Exp/rows:8192/f64:0/real_time` | 334.151 | 393.555 | 459.48 | 0.849058 | 0.727237 | 0.319779 | 0.0809018 | 0.11124 |
| `Exp/rows:8192/f64:1/real_time` | 1038.28 | 879.391 | 947.12 | 1.18068 | 1.09625 | 0.0786015 | 2.35027 | 1.26524 |
| `LogSoftmax/rows:32/f64:0/real_time` | 2.55708 | 9.74459 | 68.5504 | 0.26241 | 0.0373021 | 0.264019 | 2.91708 | 1.48874 |
| `LogSoftmax/rows:32/f64:1/real_time` | 6.65232 | 13.5166 | 73.1964 | 0.492158 | 0.0908832 | 0.285746 | 0.0837373 | 0.438032 |
| `LogSoftmax/rows:512/f64:0/real_time` | 39.0394 | 53.7717 | 118.791 | 0.726021 | 0.32864 | 0.065128 | 0.470717 | 0.280775 |
| `LogSoftmax/rows:512/f64:1/real_time` | 104.55 | 106.651 | 171.732 | 0.980307 | 0.608801 | 1.02279 | 0.0696457 | 0.123851 |
| `LogSoftmax/rows:8192/f64:0/real_time` | 620.59 | 755.883 | 824.623 | 0.821014 | 0.752574 | 0.0859642 | 0.415007 | 0.141669 |
| `LogSoftmax/rows:8192/f64:1/real_time` | 1672.98 | 1620.34 | 1707.56 | 1.03249 | 0.979746 | 0.0702449 | 0.1068 | 3.2101 |
| `Matmul/rows:256/f64:0/real_time` | 26.0236 | 287.098 | 348.325 | 0.0906435 | 0.0747107 | 0.0396263 | 2.65459 | 0.138622 |
| `Matmul/rows:256/f64:1/real_time` | 82.0175 | 703.742 | 765.043 | 0.116545 | 0.107206 | 0.593514 | 0.143032 | 0.112177 |
| `Matmul/rows:512/f64:0/real_time` | 169.74 | 2133.64 | 2198.34 | 0.0795544 | 0.0772129 | 0.144522 | 0.0262092 | 0.0852277 |
| `Matmul/rows:512/f64:1/real_time` | 594.942 | 5513.38 | 5583.11 | 0.107909 | 0.106561 | 0.0968983 | 0.123798 | 0.375656 |
| `Matmul/rows:64/f64:0/real_time` | 1.48153 | 15.1358 | 71.8227 | 0.0978823 | 0.0206276 | 0.197304 | 0.182894 | 0.261169 |
| `Matmul/rows:64/f64:1/real_time` | 2.58087 | 21.8933 | 81.9993 | 0.117884 | 0.0314743 | 0.200899 | 2.49315 | 0.146896 |
| `Mean/rows:32/f64:0/real_time` | 0.691155 | 9.397 | 64.9495 | 0.0735507 | 0.0106414 | 0.440486 | 0.314792 | 0.349762 |
| `Mean/rows:32/f64:1/real_time` | 0.62834 | 9.56268 | 64.7235 | 0.0657076 | 0.00970807 | 0.325477 | 0.220057 | 0.212856 |
| `Mean/rows:512/f64:0/real_time` | 4.29713 | 12.0443 | 68.8324 | 0.356776 | 0.0624288 | 0.829571 | 0.145949 | 0.3437 |
| `Mean/rows:512/f64:1/real_time` | 4.28076 | 14.5197 | 70.6768 | 0.294825 | 0.0605681 | 0.185537 | 0.232815 | 1.01851 |
| `Mean/rows:8192/f64:0/real_time` | 61.659 | 50.1153 | 110.951 | 1.23034 | 0.555733 | 0.0566244 | 0.138925 | 0.550233 |
| `Mean/rows:8192/f64:1/real_time` | 72.0334 | 90.8461 | 152.676 | 0.792917 | 0.471807 | 0.133197 | 0.0728255 | 0.286461 |
| `Multiply/rows:32/f64:0/real_time` | 0.337254 | 6.08798 | 64.6653 | 0.0553968 | 0.00521539 | 1.24412 | 15.0506 | 7.76454 |
| `Multiply/rows:32/f64:1/real_time` | 0.515415 | 6.27457 | 64.9337 | 0.0821434 | 0.00793755 | 0.782168 | 0.192535 | 18.1375 |
| `Multiply/rows:512/f64:0/real_time` | 4.30987 | 9.45578 | 69.7405 | 0.455792 | 0.0617986 | 0.498963 | 16.391 | 6.38326 |
| `Multiply/rows:512/f64:1/real_time` | 8.535 | 14.2578 | 90.3295 | 0.59862 | 0.0944874 | 12.7292 | 0.444715 | 11.7328 |
| `Multiply/rows:8192/f64:0/real_time` | 68.3745 | 74.5027 | 138.8 | 0.917746 | 0.492613 | 4.95018 | 0.119488 | 0.284223 |
| `Multiply/rows:8192/f64:1/real_time` | 137.111 | 142.687 | 207.326 | 0.960924 | 0.66133 | 0.288266 | 0.0921265 | 0.142879 |
| `Relu/rows:32/f64:0/real_time` | 0.272193 | 4.7451 | 64.5773 | 0.0573629 | 0.004215 | 6.02977 | 0.354884 | 0.195277 |
| `Relu/rows:32/f64:1/real_time` | 0.451075 | 4.93182 | 64.5913 | 0.0914621 | 0.00698352 | 4.18078 | 1.31837 | 0.151096 |
| `Relu/rows:512/f64:0/real_time` | 4.23586 | 8.87378 | 70.2596 | 0.477345 | 0.0602886 | 0.210449 | 0.182121 | 0.230936 |
| `Relu/rows:512/f64:1/real_time` | 8.25736 | 13.0081 | 79.006 | 0.634787 | 0.104516 | 0.426551 | 0.13625 | 1.94621 |
| `Relu/rows:8192/f64:0/real_time` | 67.6898 | 72.9847 | 139.857 | 0.927451 | 0.483993 | 0.0340609 | 0.0748852 | 5.78559 |
| `Relu/rows:8192/f64:1/real_time` | 135.381 | 140.828 | 207.215 | 0.961325 | 0.653335 | 0.495179 | 0.116136 | 0.114149 |
| `Softmax/rows:32/f64:0/real_time` | 2.412 | 9.37214 | 67.8889 | 0.257358 | 0.0355286 | 0.634443 | 3.5425 | 0.411664 |
| `Softmax/rows:32/f64:1/real_time` | 6.35893 | 12.5574 | 71.3527 | 0.506388 | 0.0891197 | 0.221957 | 0.0425278 | 1.28974 |
| `Softmax/rows:512/f64:0/real_time` | 36.6764 | 48.7273 | 113.509 | 0.752687 | 0.323115 | 0.112572 | 0.187122 | 0.751306 |
| `Softmax/rows:512/f64:1/real_time` | 99.9528 | 96.2971 | 160.604 | 1.03796 | 0.622355 | 0.135059 | 0.440189 | 0.119783 |
| `Softmax/rows:8192/f64:0/real_time` | 584.957 | 672.008 | 739.254 | 0.870461 | 0.79128 | 0.067951 | 0.116375 | 0.694272 |
| `Softmax/rows:8192/f64:1/real_time` | 1598.32 | 1450.57 | 1527.71 | 1.10185 | 1.04622 | 0.266598 | 3.59493 | 0.220998 |
| `SquareGradient/rows:32/f64:0/real_time` | 2.27733 | 98.2251 | 64.9625 | 0.0231848 | 0.0350561 | 1.12661 | 0.224318 | 6.91515 |
| `SquareGradient/rows:32/f64:1/real_time` | 2.76159 | 165.347 | 65.3511 | 0.0167018 | 0.0422577 | 0.663869 | 0.204767 | 1.13274 |
| `SquareGradient/rows:512/f64:0/real_time` | 13.8842 | 128.848 | 81.5072 | 0.107756 | 0.170343 | 0.740519 | 0.282838 | 0.253586 |
| `SquareGradient/rows:512/f64:1/real_time` | 22.5143 | 222.463 | 93.1051 | 0.101205 | 0.241816 | 1.03057 | 0.194199 | 1.10489 |
| `SquareGradient/rows:8192/f64:0/real_time` | 199.359 | 575.796 | 245.863 | 0.346233 | 0.810856 | 0.101613 | 0.244042 | 0.383024 |
| `SquareGradient/rows:8192/f64:1/real_time` | 365.656 | 1181.96 | 426.773 | 0.309365 | 0.856792 | 0.23018 | 0.514243 | 1.40658 |
| `SumAll/rows:32/f64:0/real_time` | 0.499048 | 9.3712 | 64.8482 | 0.0532533 | 0.00769562 | 0.255071 | 0.0813624 | 1.28615 |
| `SumAll/rows:32/f64:1/real_time` | 0.42937 | 9.5464 | 64.9221 | 0.0449772 | 0.00661362 | 1.34285 | 0.0972887 | 0.248551 |
| `SumAll/rows:512/f64:0/real_time` | 4.09409 | 12.0567 | 68.6161 | 0.33957 | 0.0596666 | 0.345052 | 0.0946003 | 0.256567 |
| `SumAll/rows:512/f64:1/real_time` | 4.10026 | 14.614 | 70.5768 | 0.28057 | 0.0580965 | 0.315106 | 2.562 | 0.344178 |
| `SumAll/rows:8192/f64:0/real_time` | 61.4398 | 50.3793 | 111.017 | 1.21954 | 0.553426 | 0.0454454 | 2.99885 | 0.172221 |
| `SumAll/rows:8192/f64:1/real_time` | 71.9174 | 91.3919 | 153.018 | 0.786912 | 0.469993 | 0.148425 | 0.389099 | 0.144854 |
| `SumLast/rows:32/f64:0/real_time` | 0.548951 | 7.04287 | 64.6562 | 0.0779441 | 0.0084903 | 0.263199 | 0.424509 | 0.0832925 |
| `SumLast/rows:32/f64:1/real_time` | 0.554772 | 7.17689 | 65.4646 | 0.0772997 | 0.00847438 | 0.273731 | 0.243504 | 6.10775 |
| `SumLast/rows:512/f64:0/real_time` | 4.38627 | 9.64361 | 68.6822 | 0.454837 | 0.0638632 | 0.244897 | 0.149762 | 6.33443 |
| `SumLast/rows:512/f64:1/real_time` | 4.51593 | 11.7376 | 70.8646 | 0.38474 | 0.0637261 | 0.194522 | 0.379312 | 0.110333 |
| `SumLast/rows:8192/f64:0/real_time` | 65.9069 | 48.7304 | 113.32 | 1.35248 | 0.5816 | 2.0041 | 0.108195 | 1.45195 |
| `SumLast/rows:8192/f64:1/real_time` | 74.3079 | 85.5464 | 150.594 | 0.868627 | 0.493433 | 0.348125 | 0.81519 | 0.20962 |

</details>

<details><summary>release_repeat_tf — 12 cases — 9f8cc75d4bbf</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/9f8cc75d4bbff5e09ae0df407032c995a3a34770b2909949503a5c6aeeeacc74.json). Run date: `2026-10-24T18:42:41.099596+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:8192/f64:0/real_time` | 72.4297 | 82.8838 | 139.72 | 0.873871 | 0.518393 | 2.33439 | 13.5879 | 0.160287 |
| `Add/rows:8192/f64:1/real_time` | 148.874 | 195.523 | 220.833 | 0.761414 | 0.674148 | 1.64372 | 7.94127 | 2.45536 |
| `Exp/rows:8192/f64:0/real_time` | 374.387 | 426.309 | 502.373 | 0.878206 | 0.745238 | 2.00603 | 1.56032 | 1.97035 |
| `Exp/rows:8192/f64:1/real_time` | 1040.15 | 980.357 | 948.053 | 1.06099 | 1.09715 | 0.262655 | 1.53436 | 2.89051 |
| `Multiply/rows:8192/f64:0/real_time` | 91.2596 | 81.37 | 154.985 | 1.12154 | 0.588829 | 9.1276 | 1.69499 | 7.01203 |
| `Multiply/rows:8192/f64:1/real_time` | 150.48 | 159.704 | 234.817 | 0.942244 | 0.640841 | 1.60897 | 11.9511 | 2.26529 |
| `Relu/rows:8192/f64:0/real_time` | 75.9515 | 61.669 | 143.404 | 1.2316 | 0.529633 | 1.85412 | 2.56266 | 2.16684 |
| `Relu/rows:8192/f64:1/real_time` | 150.655 | 162.336 | 224.073 | 0.928049 | 0.672349 | 3.63345 | 2.52759 | 1.17497 |
| `Softmax/rows:8192/f64:0/real_time` | 583.571 | 674.424 | 744.616 | 0.865287 | 0.783721 | 0.0194075 | 0.19849 | 0.792433 |
| `Softmax/rows:8192/f64:1/real_time` | 1574.22 | 1460.86 | 1533.08 | 1.0776 | 1.02683 | 0.220169 | 6.98042 | 2.41271 |
| `SquareGradient/rows:8192/f64:0/real_time` | 199.976 | 593.785 | 248.803 | 0.336783 | 0.803755 | 0.102768 | 6.4326 | 1.91237 |
| `SquareGradient/rows:8192/f64:1/real_time` | 367.339 | 1180.36 | 426.378 | 0.311209 | 0.861532 | 0.406081 | 0.3421 | 0.393045 |

</details>

<details><summary>release_tf — 72 cases — a68ac8d3c004</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/a68ac8d3c004b9120725bda4592ab06379a8dfbb3cc50e63f438647cf0b626a3.json). Run date: `2026-10-24T18:36:29.246084+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:32/f64:0/real_time` | 0.388371 | 8.54302 | 63.7635 | 0.0454606 | 0.00609081 | 2.03142 | 3.42524 | 1.36655 |
| `Add/rows:32/f64:1/real_time` | 0.584334 | 8.68865 | 64.275 | 0.0672525 | 0.00909115 | 0.604535 | 0.249005 | 0.231255 |
| `Add/rows:512/f64:0/real_time` | 4.35204 | 12.7382 | 69.2299 | 0.341652 | 0.0628636 | 1.08427 | 0.302994 | 0.198059 |
| `Add/rows:512/f64:1/real_time` | 8.36852 | 16.7611 | 78.4953 | 0.499282 | 0.106612 | 0.25441 | 0.148373 | 0.170497 |
| `Add/rows:8192/f64:0/real_time` | 68.1655 | 77.1496 | 137.924 | 0.883549 | 0.494223 | 0.134588 | 0.18757 | 0.147115 |
| `Add/rows:8192/f64:1/real_time` | 136.693 | 145.221 | 195.16 | 0.941269 | 0.700412 | 0.151522 | 0.208925 | 0.573602 |
| `BroadcastAdd/rows:32/f64:0/real_time` | 0.476808 | 8.97355 | 65.8007 | 0.0531348 | 0.00724625 | 1.72659 | 0.375384 | 0.498584 |
| `BroadcastAdd/rows:32/f64:1/real_time` | 0.6611 | 9.80743 | 65.7609 | 0.0674081 | 0.0100531 | 0.819251 | 0.234166 | 1.10854 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 4.45495 | 18.8244 | 80.0345 | 0.236658 | 0.0556629 | 2.00698 | 0.244068 | 0.326687 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 8.36867 | 27.9328 | 90.3666 | 0.2996 | 0.092608 | 1.75422 | 0.520569 | 0.968282 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 68.0552 | 166.835 | 227.411 | 0.407919 | 0.299261 | 6.09297 | 0.0815478 | 0.291739 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 135.672 | 317.365 | 377.568 | 0.427495 | 0.359332 | 0.145492 | 0.503886 | 2.29997 |
| `Exp/rows:32/f64:0/real_time` | 1.44868 | 7.19871 | 66.7401 | 0.201242 | 0.0217063 | 2.57919 | 3.38076 | 1.28237 |
| `Exp/rows:32/f64:1/real_time` | 4.17178 | 9.01618 | 67.8764 | 0.462699 | 0.0614614 | 0.582589 | 0.0756573 | 0.163802 |
| `Exp/rows:512/f64:0/real_time` | 21.0291 | 30.2718 | 94.8411 | 0.694675 | 0.221729 | 0.223817 | 0.173107 | 0.326594 |
| `Exp/rows:512/f64:1/real_time` | 68.1824 | 60.571 | 130.532 | 1.12566 | 0.522344 | 0.188267 | 0.115121 | 5.31609 |
| `Exp/rows:8192/f64:0/real_time` | 362.076 | 394.6 | 460.486 | 0.917576 | 0.786291 | 1.03262 | 0.885032 | 1.98556 |
| `Exp/rows:8192/f64:1/real_time` | 1119.23 | 881.881 | 958.446 | 1.26914 | 1.16775 | 0.480226 | 0.768593 | 3.91779 |
| `LogSoftmax/rows:32/f64:0/real_time` | 2.77164 | 10.1382 | 71.7564 | 0.273387 | 0.0386257 | 2.60037 | 0.838767 | 1.21643 |
| `LogSoftmax/rows:32/f64:1/real_time` | 7.03028 | 14.2342 | 77.4879 | 0.493901 | 0.0907275 | 0.450176 | 1.73584 | 5.30358 |
| `LogSoftmax/rows:512/f64:0/real_time` | 41.6632 | 55.914 | 123.148 | 0.74513 | 0.338317 | 1.01514 | 1.47164 | 1.6912 |
| `LogSoftmax/rows:512/f64:1/real_time` | 111.604 | 110.931 | 169.952 | 1.00607 | 0.656677 | 0.974021 | 1.11663 | 3.5808 |
| `LogSoftmax/rows:8192/f64:0/real_time` | 671.775 | 785.823 | 912.545 | 0.854869 | 0.736155 | 1.17326 | 0.394565 | 1.6249 |
| `LogSoftmax/rows:8192/f64:1/real_time` | 1806.33 | 1727.14 | 1970.89 | 1.04585 | 0.916507 | 1.32535 | 4.60059 | 1.01719 |
| `Matmul/rows:256/f64:0/real_time` | 26.7645 | 316.329 | 350.344 | 0.0846095 | 0.0763949 | 6.15336 | 0.939486 | 0.517649 |
| `Matmul/rows:256/f64:1/real_time` | 90.3197 | 787.046 | 841.718 | 0.114758 | 0.107304 | 6.39813 | 5.21774 | 2.16112 |
| `Matmul/rows:512/f64:0/real_time` | 187.958 | 2303.21 | 2400.64 | 0.0816068 | 0.0782948 | 3.06939 | 1.59242 | 12.2988 |
| `Matmul/rows:512/f64:1/real_time` | 659.026 | 5860.84 | 6104.16 | 0.112446 | 0.107963 | 1.27991 | 3.5685 | 1.41101 |
| `Matmul/rows:64/f64:0/real_time` | 0.998451 | 16.7939 | 72.5323 | 0.0594532 | 0.0137656 | 2.65488 | 2.50853 | 5.69566 |
| `Matmul/rows:64/f64:1/real_time` | 2.17822 | 24.5814 | 82.1208 | 0.0886127 | 0.0265246 | 0.480534 | 2.41101 | 1.56898 |
| `Mean/rows:32/f64:0/real_time` | 0.689354 | 9.36589 | 65.0771 | 0.0736026 | 0.0105929 | 0.377912 | 0.213664 | 1.88014 |
| `Mean/rows:32/f64:1/real_time` | 0.62001 | 9.46124 | 63.7646 | 0.0655315 | 0.00972342 | 0.215986 | 0.200485 | 6.60957 |
| `Mean/rows:512/f64:0/real_time` | 4.26732 | 11.9815 | 67.8804 | 0.35616 | 0.0628653 | 0.149862 | 0.8592 | 4.19993 |
| `Mean/rows:512/f64:1/real_time` | 4.28196 | 14.5942 | 70.4715 | 0.293402 | 0.0607616 | 0.287263 | 1.26521 | 0.185259 |
| `Mean/rows:8192/f64:0/real_time` | 61.8456 | 49.9447 | 111.101 | 1.23828 | 0.556661 | 1.31317 | 0.224284 | 0.323466 |
| `Mean/rows:8192/f64:1/real_time` | 71.9356 | 90.7489 | 151.597 | 0.792688 | 0.47452 | 0.256955 | 0.227409 | 0.109618 |
| `Multiply/rows:32/f64:0/real_time` | 0.326931 | 6.06043 | 63.3318 | 0.0539452 | 0.0051622 | 0.824868 | 0.149435 | 3.43884 |
| `Multiply/rows:32/f64:1/real_time` | 0.501219 | 6.21782 | 64.2539 | 0.0806101 | 0.00780059 | 1.4253 | 0.711694 | 0.537007 |
| `Multiply/rows:512/f64:0/real_time` | 4.29739 | 9.14634 | 68.7029 | 0.469848 | 0.0625503 | 1.24125 | 1.56942 | 1.33237 |
| `Multiply/rows:512/f64:1/real_time` | 8.41435 | 14.2114 | 78.0424 | 0.592085 | 0.107818 | 2.30563 | 0.0775802 | 0.532957 |
| `Multiply/rows:8192/f64:0/real_time` | 72.0262 | 76.7012 | 135.967 | 0.939049 | 0.529732 | 2.45309 | 2.15581 | 0.406191 |
| `Multiply/rows:8192/f64:1/real_time` | 148.372 | 162.372 | 211.96 | 0.913777 | 0.699999 | 1.16167 | 2.58214 | 0.483481 |
| `Relu/rows:32/f64:0/real_time` | 0.275053 | 4.94503 | 65.8664 | 0.0556221 | 0.00417592 | 2.04507 | 3.96041 | 4.81475 |
| `Relu/rows:32/f64:1/real_time` | 0.513347 | 5.29212 | 98.8143 | 0.0970023 | 0.00519507 | 4.24424 | 0.719377 | 24.8292 |
| `Relu/rows:512/f64:0/real_time` | 4.26001 | 8.91584 | 69.8537 | 0.477803 | 0.0609848 | 1.93012 | 0.0737563 | 0.325418 |
| `Relu/rows:512/f64:1/real_time` | 8.2761 | 12.3946 | 78.8052 | 0.667717 | 0.10502 | 0.164442 | 4.39894 | 0.236475 |
| `Relu/rows:8192/f64:0/real_time` | 67.8873 | 56.2272 | 138.716 | 1.20737 | 0.489397 | 0.215072 | 6.22309 | 0.499135 |
| `Relu/rows:8192/f64:1/real_time` | 135.521 | 141.398 | 206.659 | 0.958436 | 0.655769 | 0.0569105 | 0.837756 | 0.098723 |
| `Softmax/rows:32/f64:0/real_time` | 2.54561 | 10.0917 | 68.2695 | 0.252247 | 0.0372877 | 2.72285 | 0.425103 | 2.73813 |
| `Softmax/rows:32/f64:1/real_time` | 6.76024 | 12.8598 | 76.5235 | 0.525686 | 0.088342 | 2.30612 | 3.92798 | 3.1329 |
| `Softmax/rows:512/f64:0/real_time` | 39.4361 | 50.154 | 117.617 | 0.786301 | 0.335292 | 1.55393 | 0.577589 | 2.18877 |
| `Softmax/rows:512/f64:1/real_time` | 105.096 | 106.468 | 162.081 | 0.987113 | 0.648413 | 0.279724 | 2.99872 | 0.69837 |
| `Softmax/rows:8192/f64:0/real_time` | 631.491 | 705.361 | 795.602 | 0.895273 | 0.793727 | 1.12382 | 0.545318 | 7.42138 |
| `Softmax/rows:8192/f64:1/real_time` | 1764.26 | 1536.81 | 1564.54 | 1.14801 | 1.12766 | 3.35024 | 1.65666 | 7.58441 |
| `SquareGradient/rows:32/f64:0/real_time` | 2.56766 | 117.074 | 70.7685 | 0.0219319 | 0.0362825 | 1.32068 | 8.11583 | 3.02815 |
| `SquareGradient/rows:32/f64:1/real_time` | 3.14879 | 174.616 | 68.4557 | 0.0180326 | 0.0459975 | 1.64089 | 1.53139 | 0.385097 |
| `SquareGradient/rows:512/f64:0/real_time` | 14.8957 | 136.216 | 81.8601 | 0.109354 | 0.181966 | 2.7193 | 0.825204 | 0.366981 |
| `SquareGradient/rows:512/f64:1/real_time` | 25.0766 | 235.209 | 92.2816 | 0.106614 | 0.27174 | 1.82803 | 1.52323 | 1.17671 |
| `SquareGradient/rows:8192/f64:0/real_time` | 219.769 | 621.04 | 266.124 | 0.353873 | 0.825815 | 2.93035 | 0.634675 | 1.3885 |
| `SquareGradient/rows:8192/f64:1/real_time` | 652.186 | 1291.12 | 431.821 | 0.505131 | 1.51032 | 5.97479 | 4.73365 | 1.32543 |
| `SumAll/rows:32/f64:0/real_time` | 0.490966 | 9.38864 | 63.9515 | 0.0522937 | 0.00767717 | 0.242222 | 0.237016 | 0.304887 |
| `SumAll/rows:32/f64:1/real_time` | 0.430983 | 9.51259 | 64.9147 | 0.0453066 | 0.00663922 | 0.306618 | 1.0306 | 5.07165 |
| `SumAll/rows:512/f64:0/real_time` | 4.08422 | 12.0265 | 68.6037 | 0.339602 | 0.0595336 | 0.388604 | 0.432147 | 5.91811 |
| `SumAll/rows:512/f64:1/real_time` | 4.09713 | 14.5525 | 70.8662 | 0.281542 | 0.0578151 | 0.281291 | 0.772316 | 0.330244 |
| `SumAll/rows:8192/f64:0/real_time` | 61.2955 | 49.9511 | 111.311 | 1.22711 | 0.550669 | 1.44382 | 0.12502 | 0.932129 |
| `SumAll/rows:8192/f64:1/real_time` | 71.9232 | 90.6062 | 151.822 | 0.7938 | 0.473735 | 2.13479 | 0.215733 | 0.196789 |
| `SumLast/rows:32/f64:0/real_time` | 0.576043 | 7.70425 | 65.8601 | 0.0747695 | 0.00874646 | 0.930149 | 1.27348 | 0.318915 |
| `SumLast/rows:32/f64:1/real_time` | 0.536171 | 7.92002 | 66.3718 | 0.0676982 | 0.0080783 | 2.22916 | 6.60466 | 0.272282 |
| `SumLast/rows:512/f64:0/real_time` | 4.63396 | 10.6932 | 70.1394 | 0.433355 | 0.0660679 | 2.11509 | 4.10337 | 1.12375 |
| `SumLast/rows:512/f64:1/real_time` | 4.50244 | 12.8722 | 73.809 | 0.349781 | 0.0610012 | 1.1234 | 0.507546 | 0.520577 |
| `SumLast/rows:8192/f64:0/real_time` | 74.7871 | 55.6629 | 116.184 | 1.34357 | 0.643698 | 3.1663 | 1.33542 | 2.09358 |
| `SumLast/rows:8192/f64:1/real_time` | 73.2247 | 96.4552 | 151.698 | 0.759157 | 0.482701 | 3.07317 | 1.26493 | 1.93605 |

</details>

<details><summary>tensorflow — 72 cases — b2f5f61d8958</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/b2f5f61d8958b06428d2c5f94a5c50c6056fb8484982273e6937c5fae3f46117.json). Run date: `2026-10-24T17:15:34.659765+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:32/f64:0/real_time` | 0.401137 | 8.19678 | 63.0028 | 0.0489384 | 0.00636697 | 1.65488 | 2.22647 | 0.438894 |
| `Add/rows:32/f64:1/real_time` | 0.589783 | 8.49589 | 63.7493 | 0.0694198 | 0.0092516 | 1.42237 | 0.178277 | 0.396244 |
| `Add/rows:512/f64:0/real_time` | 4.35486 | 12.4949 | 67.3639 | 0.34853 | 0.0646469 | 0.362924 | 0.785005 | 0.408049 |
| `Add/rows:512/f64:1/real_time` | 8.35004 | 16.5533 | 76.2888 | 0.504433 | 0.109453 | 0.306699 | 0.25189 | 0.400782 |
| `Add/rows:8192/f64:0/real_time` | 67.9349 | 76.6858 | 137.427 | 0.885887 | 0.494334 | 0.449005 | 4.61471 | 0.310746 |
| `Add/rows:8192/f64:1/real_time` | 136.091 | 144.369 | 204.311 | 0.942665 | 0.666098 | 0.0476379 | 0.122664 | 0.227614 |
| `BroadcastAdd/rows:32/f64:0/real_time` | 0.473889 | 8.82679 | 64.0463 | 0.0536875 | 0.00739917 | 1.36686 | 0.348371 | 0.229918 |
| `BroadcastAdd/rows:32/f64:1/real_time` | 0.668059 | 9.67809 | 64.9974 | 0.0690281 | 0.0102783 | 2.97423 | 0.137744 | 1.5951 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 4.43422 | 18.5557 | 76.5111 | 0.238968 | 0.0579552 | 0.574274 | 0.169297 | 0.219116 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 8.35886 | 27.7543 | 86.9762 | 0.301173 | 0.0961052 | 0.54415 | 0.414201 | 4.03085 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 67.8521 | 166.306 | 225.889 | 0.407997 | 0.300379 | 4.04054 | 0.0587699 | 3.53037 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 135.102 | 316.011 | 374.021 | 0.427522 | 0.361215 | 0.0823546 | 0.404878 | 0.131764 |
| `Exp/rows:32/f64:0/real_time` | 1.41602 | 7.00484 | 65.1905 | 0.202148 | 0.0217212 | 0.323786 | 0.32288 | 0.218605 |
| `Exp/rows:32/f64:1/real_time` | 4.15772 | 8.86268 | 67.9898 | 0.469127 | 0.0611521 | 0.113204 | 0.0640088 | 6.41494 |
| `Exp/rows:512/f64:0/real_time` | 21.0047 | 29.8846 | 94.8328 | 0.702863 | 0.221492 | 0.197058 | 0.193026 | 3.8964 |
| `Exp/rows:512/f64:1/real_time` | 64.8513 | 60.3112 | 124.871 | 1.07528 | 0.519346 | 0.136994 | 0.365464 | 0.191522 |
| `Exp/rows:8192/f64:0/real_time` | 332.823 | 391.843 | 457.645 | 0.849376 | 0.72725 | 1.46003 | 0.0779185 | 0.270057 |
| `Exp/rows:8192/f64:1/real_time` | 1041.53 | 874.998 | 940.119 | 1.19032 | 1.10787 | 1.44323 | 0.0524692 | 0.0576537 |
| `LogSoftmax/rows:32/f64:0/real_time` | 2.66139 | 9.4746 | 66.4111 | 0.280897 | 0.0400744 | 0.418414 | 0.393349 | 0.26892 |
| `LogSoftmax/rows:32/f64:1/real_time` | 6.61858 | 13.3282 | 71.3356 | 0.496585 | 0.0927808 | 0.200922 | 0.178221 | 1.40347 |
| `LogSoftmax/rows:512/f64:0/real_time` | 40.411 | 53.473 | 117.158 | 0.755728 | 0.344928 | 0.130089 | 0.0545401 | 0.82414 |
| `LogSoftmax/rows:512/f64:1/real_time` | 104.346 | 106.567 | 170.226 | 0.979163 | 0.612986 | 0.373851 | 2.17557 | 1.15017 |
| `LogSoftmax/rows:8192/f64:0/real_time` | 643.062 | 755.721 | 817.144 | 0.850925 | 0.786963 | 0.158966 | 1.99214 | 0.033349 |
| `LogSoftmax/rows:8192/f64:1/real_time` | 1664.43 | 1599.49 | 1703.52 | 1.0406 | 0.977052 | 0.0432248 | 0.632506 | 0.28248 |
| `Matmul/rows:256/f64:0/real_time` | 25.9665 | 286.925 | 347.605 | 0.0904992 | 0.0747011 | 0.0718503 | 0.380244 | 0.126661 |
| `Matmul/rows:256/f64:1/real_time` | 81.2043 | 702.909 | 758.68 | 0.115526 | 0.107034 | 0.245543 | 0.203691 | 0.172593 |
| `Matmul/rows:512/f64:0/real_time` | 169.406 | 2129.24 | 2196.51 | 0.0795618 | 0.0771251 | 1.18537 | 0.0943665 | 3.29991 |
| `Matmul/rows:512/f64:1/real_time` | 595.289 | 5507.14 | 5545.92 | 0.108094 | 0.107338 | 0.125909 | 0.560524 | 0.280685 |
| `Matmul/rows:64/f64:0/real_time` | 1.49408 | 14.9616 | 70.9094 | 0.099861 | 0.0210702 | 17.9134 | 0.538214 | 0.758438 |
| `Matmul/rows:64/f64:1/real_time` | 2.64622 | 21.7948 | 81.5205 | 0.121416 | 0.0324608 | 1.40799 | 0.171589 | 0.262361 |
| `Mean/rows:32/f64:0/real_time` | 0.813812 | 9.2839 | 65.0431 | 0.0876585 | 0.0125119 | 1.23694 | 0.19656 | 6.15103 |
| `Mean/rows:32/f64:1/real_time` | 0.752016 | 9.48662 | 63.1962 | 0.0792712 | 0.0118997 | 3.5446 | 0.130741 | 0.310449 |
| `Mean/rows:512/f64:0/real_time` | 6.18519 | 11.9024 | 66.0878 | 0.51966 | 0.0935906 | 1.70473 | 0.761068 | 0.405508 |
| `Mean/rows:512/f64:1/real_time` | 5.63665 | 14.3529 | 69.2084 | 0.392719 | 0.0814446 | 0.81056 | 0.228111 | 0.389902 |
| `Mean/rows:8192/f64:0/real_time` | 91.7382 | 49.7892 | 108.539 | 1.84253 | 0.845211 | 0.0869454 | 0.668457 | 0.478428 |
| `Mean/rows:8192/f64:1/real_time` | 83.8611 | 90.4036 | 149.646 | 0.92763 | 0.560398 | 0.414382 | 0.0506131 | 0.327241 |
| `Multiply/rows:32/f64:0/real_time` | 0.34815 | 5.87611 | 62.3351 | 0.0592485 | 0.00558514 | 2.61656 | 0.239573 | 0.0856627 |
| `Multiply/rows:32/f64:1/real_time` | 0.527145 | 6.10388 | 63.1062 | 0.0863622 | 0.0083533 | 1.8883 | 0.31348 | 0.475616 |
| `Multiply/rows:512/f64:0/real_time` | 4.32182 | 9.98316 | 67.646 | 0.432911 | 0.0638888 | 0.411928 | 0.222245 | 1.93203 |
| `Multiply/rows:512/f64:1/real_time` | 8.3325 | 14.0398 | 75.7218 | 0.59349 | 0.110041 | 0.570086 | 0.265762 | 0.193641 |
| `Multiply/rows:8192/f64:0/real_time` | 67.8829 | 74.0143 | 136.303 | 0.917159 | 0.49803 | 0.108402 | 0.600705 | 0.47831 |
| `Multiply/rows:8192/f64:1/real_time` | 135.973 | 129.244 | 203.58 | 1.05206 | 0.66791 | 0.0738543 | 5.92366 | 0.324069 |
| `Relu/rows:32/f64:0/real_time` | 0.273752 | 4.52795 | 62.0326 | 0.0604583 | 0.00441304 | 4.98221 | 0.95788 | 0.393552 |
| `Relu/rows:32/f64:1/real_time` | 0.427481 | 4.85189 | 63.2837 | 0.088106 | 0.00675499 | 0.917796 | 2.43251 | 0.216712 |
| `Relu/rows:512/f64:0/real_time` | 4.22801 | 8.68027 | 67.1999 | 0.487083 | 0.062917 | 0.65152 | 5.29746 | 0.143413 |
| `Relu/rows:512/f64:1/real_time` | 8.24388 | 12.5694 | 76.5747 | 0.655868 | 0.107658 | 0.349042 | 1.05536 | 1.05529 |
| `Relu/rows:8192/f64:0/real_time` | 67.6924 | 72.5202 | 138.288 | 0.933428 | 0.489502 | 0.432745 | 0.129397 | 0.158856 |
| `Relu/rows:8192/f64:1/real_time` | 135.274 | 140.733 | 206.609 | 0.961213 | 0.654734 | 3.41817 | 0.100164 | 0.95786 |
| `Softmax/rows:32/f64:0/real_time` | 2.52138 | 9.24074 | 67.591 | 0.272854 | 0.0373034 | 0.270069 | 0.159348 | 0.415348 |
| `Softmax/rows:32/f64:1/real_time` | 6.35183 | 12.5201 | 70.8634 | 0.507329 | 0.0896349 | 0.173 | 0.0916501 | 0.446815 |
| `Softmax/rows:512/f64:0/real_time` | 38.384 | 48.6214 | 112.76 | 0.789447 | 0.340404 | 0.134683 | 0.439269 | 0.278263 |
| `Softmax/rows:512/f64:1/real_time` | 99.6639 | 95.7142 | 157.955 | 1.04127 | 0.630965 | 0.0727956 | 0.77024 | 0.351739 |
| `Softmax/rows:8192/f64:0/real_time` | 610.351 | 669.261 | 734.319 | 0.911977 | 0.83118 | 1.48153 | 0.11749 | 0.19865 |
| `Softmax/rows:8192/f64:1/real_time` | 1598.62 | 1427.32 | 1492.87 | 1.12002 | 1.07084 | 0.378542 | 0.387704 | 0.410867 |
| `SquareGradient/rows:32/f64:0/real_time` | 3.45634 | 96.5342 | 63.965 | 0.0358043 | 0.0540348 | 0.265865 | 0.261399 | 0.253089 |
| `SquareGradient/rows:32/f64:1/real_time` | 3.63921 | 161.895 | 64.2783 | 0.0224789 | 0.0566165 | 3.35097 | 0.1335 | 0.775365 |
| `SquareGradient/rows:512/f64:0/real_time` | 29.607 | 126.712 | 79.5224 | 0.233655 | 0.37231 | 0.43225 | 0.352003 | 1.49657 |
| `SquareGradient/rows:512/f64:1/real_time` | 33.3855 | 219.145 | 92.2853 | 0.152344 | 0.361764 | 0.558635 | 0.480248 | 4.22144 |
| `SquareGradient/rows:8192/f64:0/real_time` | 449.502 | 573.018 | 245.959 | 0.784447 | 1.82755 | 0.055751 | 0.215645 | 3.42361 |
| `SquareGradient/rows:8192/f64:1/real_time` | 513.551 | 1185.72 | 422.381 | 0.433112 | 1.21585 | 0.336261 | 0.240352 | 0.0811569 |
| `SumAll/rows:32/f64:0/real_time` | 0.614235 | 9.21248 | 62.6267 | 0.0666743 | 0.00980788 | 2.53789 | 0.176517 | 0.339721 |
| `SumAll/rows:32/f64:1/real_time` | 0.578225 | 9.38714 | 62.7132 | 0.0615975 | 0.00922015 | 0.555449 | 0.0844101 | 0.117874 |
| `SumAll/rows:512/f64:0/real_time` | 5.97278 | 11.8561 | 65.9439 | 0.503772 | 0.0905737 | 0.738465 | 0.286394 | 0.21599 |
| `SumAll/rows:512/f64:1/real_time` | 5.54717 | 14.4378 | 69.2727 | 0.384212 | 0.0800773 | 1.56352 | 0.524385 | 0.266659 |
| `SumAll/rows:8192/f64:0/real_time` | 91.4677 | 49.7365 | 108.866 | 1.83905 | 0.840185 | 0.040348 | 0.0969388 | 0.271415 |
| `SumAll/rows:8192/f64:1/real_time` | 83.8444 | 90.2865 | 150.625 | 0.928648 | 0.556643 | 0.280806 | 0.108668 | 0.654239 |
| `SumLast/rows:32/f64:0/real_time` | 0.648687 | 6.81527 | 62.7332 | 0.0951814 | 0.0103404 | 0.406379 | 2.92153 | 0.211298 |
| `SumLast/rows:32/f64:1/real_time` | 0.554177 | 7.02308 | 63.0858 | 0.078908 | 0.0087845 | 0.245453 | 0.245075 | 0.437174 |
| `SumLast/rows:512/f64:0/real_time` | 6.178 | 9.46775 | 66.3904 | 0.652532 | 0.0930557 | 0.536954 | 0.218502 | 0.138233 |
| `SumLast/rows:512/f64:1/real_time` | 4.50498 | 11.5674 | 69.2225 | 0.389455 | 0.0650798 | 1.44333 | 0.290526 | 0.186521 |
| `SumLast/rows:8192/f64:0/real_time` | 94.315 | 48.322 | 110.937 | 1.9518 | 0.850169 | 0.325379 | 0.270839 | 0.527495 |
| `SumLast/rows:8192/f64:1/real_time` | 74.857 | 84.8063 | 149.775 | 0.882681 | 0.499796 | 0.45169 | 0.21166 | 5.56764 |

</details>

<details><summary>tensorflow_before — 72 cases — bbbe5abecabd</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/bbbe5abecabdda8c64589266e20dafb2adbd4fd6cb7afa73fb76f8aad938fa12.json). Run date: `2026-10-24T16:31:11.607651+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:32/f64:0/real_time` | 0.399425 | 8.10415 | 64.5938 | 0.0492865 | 0.00618365 | — | — | — |
| `Add/rows:32/f64:1/real_time` | 0.585908 | 8.36283 | 64.5196 | 0.0700609 | 0.00908108 | — | — | — |
| `Add/rows:512/f64:0/real_time` | 4.34043 | 12.3073 | 69.5436 | 0.352671 | 0.062413 | — | — | — |
| `Add/rows:512/f64:1/real_time` | 8.38412 | 16.4994 | 78.6453 | 0.508148 | 0.106607 | — | — | — |
| `Add/rows:8192/f64:0/real_time` | 68.0576 | 76.9424 | 138.859 | 0.884526 | 0.490119 | — | — | — |
| `Add/rows:8192/f64:1/real_time` | 137.206 | 129.875 | 207.776 | 1.05645 | 0.660356 | — | — | — |
| `BroadcastAdd/rows:32/f64:0/real_time` | 0.477403 | 8.704 | 66.217 | 0.0548487 | 0.00720967 | — | — | — |
| `BroadcastAdd/rows:32/f64:1/real_time` | 0.667897 | 9.64698 | 66.8183 | 0.0692338 | 0.00999572 | — | — | — |
| `BroadcastAdd/rows:512/f64:0/real_time` | 4.45187 | 18.4194 | 80.3286 | 0.241695 | 0.0554208 | — | — | — |
| `BroadcastAdd/rows:512/f64:1/real_time` | 8.41691 | 27.7132 | 90.101 | 0.303715 | 0.0934164 | — | — | — |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 68.03 | 166.993 | 228.5 | 0.407382 | 0.297725 | — | — | — |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 135.96 | 318.829 | 381.685 | 0.426434 | 0.356208 | — | — | — |
| `Exp/rows:32/f64:0/real_time` | 4.30135 | 7.13697 | 151.522 | 0.602686 | 0.0283875 | — | — | — |
| `Exp/rows:32/f64:1/real_time` | 6.84712 | 34.3882 | 198.821 | 0.199113 | 0.0344386 | — | — | — |
| `Exp/rows:512/f64:0/real_time` | 67.0838 | 97.8584 | 271.716 | 0.685519 | 0.24689 | — | — | — |
| `Exp/rows:512/f64:1/real_time` | 107.754 | 223.916 | 427.966 | 0.481227 | 0.251782 | — | — | — |
| `Exp/rows:8192/f64:0/real_time` | 1094.51 | 1343.79 | 2468.24 | 0.814493 | 0.443437 | — | — | — |
| `Exp/rows:8192/f64:1/real_time` | 1725.74 | 4155.94 | 5740.53 | 0.415248 | 0.300625 | — | — | — |
| `LogSoftmax/rows:32/f64:0/real_time` | 13.5258 | 64.125 | 314.275 | 0.210929 | 0.0430381 | — | — | — |
| `LogSoftmax/rows:32/f64:1/real_time` | 16.5058 | 84.5749 | 340.96 | 0.195162 | 0.0484099 | — | — | — |
| `LogSoftmax/rows:512/f64:0/real_time` | 195.072 | 328.8 | 622.678 | 0.593283 | 0.313278 | — | — | — |
| `LogSoftmax/rows:512/f64:1/real_time` | 242.901 | 669.512 | 1005.86 | 0.362803 | 0.241487 | — | — | — |
| `LogSoftmax/rows:8192/f64:0/real_time` | 3100.86 | 4448.99 | 4845.12 | 0.696982 | 0.639997 | — | — | — |
| `LogSoftmax/rows:8192/f64:1/real_time` | 3862.79 | 9689.88 | 10455.6 | 0.398642 | 0.369446 | — | — | — |
| `Matmul/rows:256/f64:0/real_time` | 25.8763 | 1418.24 | 1611.67 | 0.0182453 | 0.0160556 | — | — | — |
| `Matmul/rows:256/f64:1/real_time` | 82.7557 | 2952.52 | 3478.2 | 0.0280289 | 0.0237927 | — | — | — |
| `Matmul/rows:512/f64:0/real_time` | 169.882 | 10083.8 | 11037.1 | 0.016847 | 0.0153919 | — | — | — |
| `Matmul/rows:512/f64:1/real_time` | 594.382 | 22182 | 22877.4 | 0.0267957 | 0.0259811 | — | — | — |
| `Matmul/rows:64/f64:0/real_time` | 1.47232 | 87.5102 | 353.994 | 0.0168245 | 0.00415916 | — | — | — |
| `Matmul/rows:64/f64:1/real_time` | 2.57973 | 109.103 | 352.29 | 0.0236449 | 0.00732275 | — | — | — |
| `Mean/rows:32/f64:0/real_time` | 2.42253 | 9.18 | 64.3331 | 0.263892 | 0.037656 | — | — | — |
| `Mean/rows:32/f64:1/real_time` | 2.40715 | 9.38315 | 64.4047 | 0.25654 | 0.0373754 | — | — | — |
| `Mean/rows:512/f64:0/real_time` | 32.2309 | 11.861 | 67.9376 | 2.71738 | 0.47442 | — | — | — |
| `Mean/rows:512/f64:1/real_time` | 31.9872 | 14.3238 | 70.8034 | 2.23315 | 0.451775 | — | — | — |
| `Mean/rows:8192/f64:0/real_time` | 509.148 | 49.8359 | 111.447 | 10.2165 | 4.56852 | — | — | — |
| `Mean/rows:8192/f64:1/real_time` | 505.469 | 90.7184 | 153.192 | 5.57185 | 3.29958 | — | — | — |
| `Multiply/rows:32/f64:0/real_time` | 0.343089 | 5.93231 | 64.305 | 0.057834 | 0.00533534 | — | — | — |
| `Multiply/rows:32/f64:1/real_time` | 0.543957 | 5.9454 | 65.0001 | 0.0914921 | 0.00836855 | — | — | — |
| `Multiply/rows:512/f64:0/real_time` | 4.29702 | 9.92917 | 69.5845 | 0.432768 | 0.0617525 | — | — | — |
| `Multiply/rows:512/f64:1/real_time` | 8.33515 | 13.9417 | 72.1126 | 0.597856 | 0.115585 | — | — | — |
| `Multiply/rows:8192/f64:0/real_time` | 68.1206 | 74.2876 | 138.513 | 0.916985 | 0.491801 | — | — | — |
| `Multiply/rows:8192/f64:1/real_time` | 136.951 | 142.691 | 206.77 | 0.959777 | 0.662337 | — | — | — |
| `Relu/rows:32/f64:0/real_time` | 0.277581 | 4.40611 | 63.5302 | 0.0629991 | 0.00436928 | — | — | — |
| `Relu/rows:32/f64:1/real_time` | 0.427759 | 4.59454 | 64.4286 | 0.0931015 | 0.00663927 | — | — | — |
| `Relu/rows:512/f64:0/real_time` | 4.21549 | 8.52267 | 69.5183 | 0.494621 | 0.0606386 | — | — | — |
| `Relu/rows:512/f64:1/real_time` | 8.24096 | 12.6057 | 73.8085 | 0.65375 | 0.111653 | — | — | — |
| `Relu/rows:8192/f64:0/real_time` | 67.7846 | 72.9458 | 127.226 | 0.929247 | 0.532791 | — | — | — |
| `Relu/rows:8192/f64:1/real_time` | 135.333 | 115.261 | 206.555 | 1.17414 | 0.655191 | — | — | — |
| `Softmax/rows:32/f64:0/real_time` | 13.1649 | 63.3801 | 317.924 | 0.207714 | 0.041409 | — | — | — |
| `Softmax/rows:32/f64:1/real_time` | 15.9955 | 72.524 | 332.315 | 0.220555 | 0.0481334 | — | — | — |
| `Softmax/rows:512/f64:0/real_time` | 192.324 | 277.354 | 425.653 | 0.693427 | 0.451834 | — | — | — |
| `Softmax/rows:512/f64:1/real_time` | 241.765 | 571.176 | 924.531 | 0.423277 | 0.261501 | — | — | — |
| `Softmax/rows:8192/f64:0/real_time` | 3071.06 | 3734.3 | 4034.9 | 0.822393 | 0.761124 | — | — | — |
| `Softmax/rows:8192/f64:1/real_time` | 3849.01 | 7855.87 | 7905.48 | 0.489953 | 0.486878 | — | — | — |
| `SquareGradient/rows:32/f64:0/real_time` | 7.01574 | 656.556 | 322.208 | 0.0106857 | 0.0217739 | — | — | — |
| `SquareGradient/rows:32/f64:1/real_time` | 7.62412 | 1073.51 | 333.138 | 0.00710203 | 0.0228858 | — | — | — |
| `SquareGradient/rows:512/f64:0/real_time` | 81.411 | 784.272 | 380.75 | 0.103805 | 0.213817 | — | — | — |
| `SquareGradient/rows:512/f64:1/real_time` | 92.1762 | 1446.7 | 465.914 | 0.0637146 | 0.197839 | — | — | — |
| `SquareGradient/rows:8192/f64:0/real_time` | 1277.33 | 3934.37 | 1356.12 | 0.32466 | 0.941902 | — | — | — |
| `SquareGradient/rows:8192/f64:1/real_time` | 1480.59 | 6911.83 | 2556.77 | 0.214211 | 0.579084 | — | — | — |
| `SumAll/rows:32/f64:0/real_time` | 2.23934 | 9.06331 | 64.1422 | 0.247078 | 0.0349121 | — | — | — |
| `SumAll/rows:32/f64:1/real_time` | 2.20768 | 9.19306 | 64.6502 | 0.240147 | 0.0341481 | — | — | — |
| `SumAll/rows:512/f64:0/real_time` | 32.0642 | 11.8213 | 67.7162 | 2.71241 | 0.473508 | — | — | — |
| `SumAll/rows:512/f64:1/real_time` | 31.8025 | 14.3353 | 71.0222 | 2.21847 | 0.447782 | — | — | — |
| `SumAll/rows:8192/f64:0/real_time` | 508.745 | 49.7652 | 111.306 | 10.2229 | 4.57069 | — | — | — |
| `SumAll/rows:8192/f64:1/real_time` | 505.738 | 90.658 | 152.35 | 5.57853 | 3.31959 | — | — | — |
| `SumLast/rows:32/f64:0/real_time` | 1.05547 | 6.70422 | 63.8428 | 0.157434 | 0.0165324 | — | — | — |
| `SumLast/rows:32/f64:1/real_time` | 0.988721 | 6.84798 | 64.2171 | 0.144381 | 0.0153965 | — | — | — |
| `SumLast/rows:512/f64:0/real_time` | 12.2251 | 9.30985 | 68.0104 | 1.31314 | 0.179754 | — | — | — |
| `SumLast/rows:512/f64:1/real_time` | 11.2202 | 11.4259 | 70.5017 | 0.982003 | 0.159148 | — | — | — |
| `SumLast/rows:8192/f64:0/real_time` | 190.909 | 48.337 | 112.314 | 3.94955 | 1.69978 | — | — | — |
| `SumLast/rows:8192/f64:1/real_time` | 176.184 | 85.2284 | 150.083 | 2.06719 | 1.17391 | — | — | — |

</details>

<details><summary>tensorflow_final — 72 cases — 535f52c7b8cc</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/535f52c7b8cceee4e02f7d6093d161aae45ad2bc7a1e148fb9075b6ced527243.json). Run date: `2026-10-24T16:50:00.067413+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:32/f64:0/real_time` | 2.19737 | 52.5151 | 286 | 0.0418427 | 0.00768314 | 11.3108 | 4.28957 | 6.74133 |
| `Add/rows:32/f64:1/real_time` | 3.16222 | 54.475 | 298.728 | 0.0580489 | 0.0105856 | 5.0249 | 2.0964 | 12.7298 |
| `Add/rows:512/f64:0/real_time` | 26.4057 | 69.8292 | 280.387 | 0.378147 | 0.0941761 | 5.09403 | 14.9767 | 10.6717 |
| `Add/rows:512/f64:1/real_time` | 40.2405 | 98.8718 | 341.305 | 0.406997 | 0.117902 | 9.10228 | 9.36736 | 12.9427 |
| `Add/rows:8192/f64:0/real_time` | 563.385 | 587.415 | 1016.89 | 0.959092 | 0.55403 | 7.27383 | 5.57719 | 1.61406 |
| `Add/rows:8192/f64:1/real_time` | 1045.51 | 1229.76 | 1685.19 | 0.850169 | 0.620409 | 1.63916 | 2.70579 | 4.60391 |
| `BroadcastAdd/rows:32/f64:0/real_time` | 2.39324 | 47.7594 | 253.47 | 0.0501104 | 0.00944191 | 8.97722 | 8.01546 | 15.2936 |
| `BroadcastAdd/rows:32/f64:1/real_time` | 4.61329 | 55.776 | 298.012 | 0.0827111 | 0.0154802 | 3.16878 | 15.3376 | 1.36918 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 31.7558 | 102.34 | 299.342 | 0.310298 | 0.106086 | 11.4292 | 16.1831 | 6.75957 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 50.8626 | 118.138 | 400.512 | 0.430536 | 0.126994 | 13.1633 | 13.9293 | 5.70664 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 567.82 | 1045.27 | 1601.82 | 0.54323 | 0.354485 | 17.5323 | 16.0205 | 8.71315 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 969.277 | 2189.78 | 2581.73 | 0.442636 | 0.375437 | 9.72486 | 5.19246 | 9.77224 |
| `Exp/rows:32/f64:0/real_time` | 6.61936 | 41.2572 | 285.734 | 0.160441 | 0.0231662 | 1.57755 | 12.1558 | 3.22781 |
| `Exp/rows:32/f64:1/real_time` | 19.2965 | 52.2468 | 292.867 | 0.369334 | 0.0658885 | 0.744518 | 2.27708 | 2.30125 |
| `Exp/rows:512/f64:0/real_time` | 85.9957 | 142.056 | 427.59 | 0.605364 | 0.201117 | 16.4293 | 12.6768 | 15.8621 |
| `Exp/rows:512/f64:1/real_time` | 287.798 | 309.362 | 612.238 | 0.930296 | 0.470076 | 9.86314 | 5.29317 | 10.5846 |
| `Exp/rows:8192/f64:0/real_time` | 1377.38 | 2166.07 | 2646.09 | 0.635889 | 0.520534 | 9.54784 | 2.3651 | 4.45252 |
| `Exp/rows:8192/f64:1/real_time` | 4023.78 | 4737.9 | 5773.64 | 0.849274 | 0.696922 | 5.98813 | 7.13923 | 2.76419 |
| `LogSoftmax/rows:32/f64:0/real_time` | 9.27924 | 46.1054 | 247.327 | 0.201261 | 0.0375181 | 9.09507 | 7.05644 | 11.7938 |
| `LogSoftmax/rows:32/f64:1/real_time` | 29.0682 | 70.5067 | 295.128 | 0.412276 | 0.0984936 | 9.39166 | 10.4711 | 11.8008 |
| `LogSoftmax/rows:512/f64:0/real_time` | 160.488 | 277.485 | 516.581 | 0.578368 | 0.310674 | 10.5019 | 2.71034 | 7.12682 |
| `LogSoftmax/rows:512/f64:1/real_time` | 420.908 | 560.915 | 875.068 | 0.750396 | 0.481001 | 7.56446 | 1.80313 | 2.08269 |
| `LogSoftmax/rows:8192/f64:0/real_time` | 2746.53 | 3614.35 | 5161.73 | 0.759894 | 0.532094 | 11.2475 | 8.15632 | 3.06201 |
| `LogSoftmax/rows:8192/f64:1/real_time` | 7286.44 | 9678.11 | 8873.72 | 0.752878 | 0.821126 | 4.24703 | 8.82816 | 7.63345 |
| `Matmul/rows:256/f64:0/real_time` | 198.16 | 1008.7 | 1371.09 | 0.196451 | 0.144527 | 3.62003 | 5.30706 | 14.2811 |
| `Matmul/rows:256/f64:1/real_time` | 679.076 | 2253.53 | 2576.15 | 0.301339 | 0.263601 | 11.3111 | 17.2177 | 20.3447 |
| `Matmul/rows:512/f64:0/real_time` | 1364.91 | 7653.07 | 8726.41 | 0.178348 | 0.156411 | 10.2125 | 10.7382 | 8.5854 |
| `Matmul/rows:512/f64:1/real_time` | 5272.36 | 19018.6 | 18210.9 | 0.277221 | 0.289516 | 13.3826 | 3.69801 | 6.56199 |
| `Matmul/rows:64/f64:0/real_time` | 4.85694 | 75.3123 | 305.819 | 0.0644907 | 0.0158817 | 13.1891 | 8.28785 | 7.45994 |
| `Matmul/rows:64/f64:1/real_time` | 16.0925 | 108.306 | 331.875 | 0.148584 | 0.0484897 | 9.69691 | 5.2903 | 5.24357 |
| `Mean/rows:32/f64:0/real_time` | 3.55405 | 58.5523 | 273.775 | 0.0606987 | 0.0129816 | 2.95299 | 1.18267 | 3.81905 |
| `Mean/rows:32/f64:1/real_time` | 2.69967 | 59.3497 | 217.666 | 0.0454874 | 0.0124028 | 16.4165 | 4.08026 | 5.31916 |
| `Mean/rows:512/f64:0/real_time` | 27.0678 | 59.828 | 227.505 | 0.452427 | 0.118977 | 9.76245 | 11.755 | 18.3442 |
| `Mean/rows:512/f64:1/real_time` | 28.047 | 78.9811 | 320.543 | 0.35511 | 0.0874984 | 2.7251 | 10.0131 | 2.23323 |
| `Mean/rows:8192/f64:0/real_time` | 419.97 | 287.063 | 640.412 | 1.46299 | 0.65578 | 3.06304 | 11.3037 | 4.97285 |
| `Mean/rows:8192/f64:1/real_time` | 570.81 | 727.301 | 1112.22 | 0.784833 | 0.513217 | 1.8547 | 3.58963 | 2.79465 |
| `Multiply/rows:32/f64:0/real_time` | 1.96863 | 38.7414 | 297.826 | 0.0508145 | 0.00660999 | 10.3116 | 2.4164 | 1.4972 |
| `Multiply/rows:32/f64:1/real_time` | 3.79446 | 40.3207 | 274.125 | 0.094107 | 0.0138421 | 5.4417 | 4.6974 | 14.9804 |
| `Multiply/rows:512/f64:0/real_time` | 27.3621 | 63.18 | 316.824 | 0.433082 | 0.0863637 | 3.33786 | 3.90054 | 12.1151 |
| `Multiply/rows:512/f64:1/real_time` | 44.9449 | 86.8173 | 361.117 | 0.517695 | 0.124461 | 7.06906 | 14.4693 | 4.52764 |
| `Multiply/rows:8192/f64:0/real_time` | 567.453 | 649.834 | 861.881 | 0.873227 | 0.658389 | 0.884272 | 2.77446 | 5.9109 |
| `Multiply/rows:8192/f64:1/real_time` | 1169.49 | 1100.3 | 1570.62 | 1.06289 | 0.744606 | 3.87578 | 5.44994 | 3.24463 |
| `Relu/rows:32/f64:0/real_time` | 1.20714 | 29.8982 | 278.111 | 0.040375 | 0.00434049 | 13.6711 | 2.19537 | 1.72919 |
| `Relu/rows:32/f64:1/real_time` | 2.33321 | 31.178 | 246.91 | 0.0748351 | 0.00944965 | 18.5 | 1.20388 | 15.4881 |
| `Relu/rows:512/f64:0/real_time` | 20.2125 | 49.2111 | 309.625 | 0.41073 | 0.0652806 | 8.94206 | 4.038 | 3.10247 |
| `Relu/rows:512/f64:1/real_time` | 39.2856 | 64.7613 | 320.649 | 0.606621 | 0.122519 | 12.6734 | 12.9716 | 8.94749 |
| `Relu/rows:8192/f64:0/real_time` | 487.506 | 551.919 | 919.077 | 0.883293 | 0.53043 | 0.64725 | 5.29701 | 4.58874 |
| `Relu/rows:8192/f64:1/real_time` | 928.562 | 1027.99 | 1577.67 | 0.903278 | 0.588565 | 3.90448 | 2.5191 | 1.2742 |
| `Softmax/rows:32/f64:0/real_time` | 12.4439 | 55.1469 | 305.67 | 0.22565 | 0.0407103 | 9.76379 | 2.28406 | 3.53636 |
| `Softmax/rows:32/f64:1/real_time` | 29.3744 | 71.8513 | 323.792 | 0.408823 | 0.0907201 | 1.31648 | 4.88629 | 4.35653 |
| `Softmax/rows:512/f64:0/real_time` | 181.061 | 267.371 | 531.617 | 0.677188 | 0.340585 | 7.87601 | 3.7107 | 2.57549 |
| `Softmax/rows:512/f64:1/real_time` | 393.914 | 521.316 | 857.261 | 0.755614 | 0.459503 | 7.44164 | 7.43054 | 2.36385 |
| `Softmax/rows:8192/f64:0/real_time` | 2890.59 | 4275.6 | 4177 | 0.676067 | 0.692026 | 9.50016 | 7.68028 | 10.928 |
| `Softmax/rows:8192/f64:1/real_time` | 6900.64 | 7023.2 | 9487.16 | 0.982549 | 0.727366 | 12.8899 | 11.0238 | 3.02082 |
| `SquareGradient/rows:32/f64:0/real_time` | 17.4904 | 549.131 | 274.402 | 0.031851 | 0.06374 | 4.40405 | 10.9254 | 9.21461 |
| `SquareGradient/rows:32/f64:1/real_time` | 19.0539 | 864.322 | 287.593 | 0.022045 | 0.0662532 | 12.2023 | 8.30433 | 0.648122 |
| `SquareGradient/rows:512/f64:0/real_time` | 148.782 | 709.328 | 350.004 | 0.20975 | 0.425086 | 1.45365 | 6.49042 | 1.58428 |
| `SquareGradient/rows:512/f64:1/real_time` | 175.666 | 1439.61 | 414.984 | 0.122024 | 0.423308 | 9.47089 | 2.09433 | 2.98554 |
| `SquareGradient/rows:8192/f64:0/real_time` | 2245.41 | 4814.41 | 1860.77 | 0.466393 | 1.20671 | 12.0793 | 3.10183 | 2.45622 |
| `SquareGradient/rows:8192/f64:1/real_time` | 3531.54 | 9434.18 | 3147.07 | 0.374335 | 1.12217 | 3.20644 | 1.48239 | 6.22715 |
| `SumAll/rows:32/f64:0/real_time` | 2.30632 | 57.303 | 246.974 | 0.0402479 | 0.00933833 | 14.6375 | 2.41328 | 7.42329 |
| `SumAll/rows:32/f64:1/real_time` | 2.57387 | 56.1761 | 280.866 | 0.045818 | 0.00916406 | 14.0386 | 9.42454 | 1.91048 |
| `SumAll/rows:512/f64:0/real_time` | 20.7676 | 60.555 | 265.55 | 0.342954 | 0.078206 | 7.36357 | 8.63808 | 5.52605 |
| `SumAll/rows:512/f64:1/real_time` | 22.8173 | 81.9686 | 303.875 | 0.278367 | 0.0750878 | 11.9564 | 1.76042 | 2.02102 |
| `SumAll/rows:8192/f64:0/real_time` | 377.338 | 287.35 | 603.703 | 1.31317 | 0.62504 | 6.47531 | 3.5292 | 0.989941 |
| `SumAll/rows:8192/f64:1/real_time` | 538.586 | 721.94 | 1002.91 | 0.746027 | 0.537023 | 3.14282 | 7.10872 | 7.4549 |
| `SumLast/rows:32/f64:0/real_time` | 2.79367 | 34.8931 | 239.466 | 0.0800638 | 0.0116663 | 11.5662 | 16.6274 | 14.2162 |
| `SumLast/rows:32/f64:1/real_time` | 2.46892 | 39.7287 | 250.01 | 0.0621445 | 0.00987529 | 4.38831 | 15.2175 | 14.6859 |
| `SumLast/rows:512/f64:0/real_time` | 21.8597 | 57.0253 | 288.981 | 0.383333 | 0.075644 | 11.5884 | 2.29204 | 7.48334 |
| `SumLast/rows:512/f64:1/real_time` | 24.8179 | 62.1194 | 278.152 | 0.399519 | 0.0892241 | 7.24988 | 6.27311 | 6.48469 |
| `SumLast/rows:8192/f64:0/real_time` | 340.502 | 242.008 | 649.987 | 1.40699 | 0.523859 | 8.94984 | 12.3033 | 8.79836 |
| `SumLast/rows:8192/f64:1/real_time` | 524.135 | 691.183 | 1093.78 | 0.758316 | 0.479196 | 8.83479 | 4.77984 | 7.75833 |

</details>

<details><summary>tensorflow_paired_after — 72 cases — 778322254170</summary>

Immutable data: [JSON](benchmarks/results/tensorflow/comparisons/778322254170f6680b9b8e90a34391ed9e0aae9b4bf59dc08eefb36be2b48fad.json). Run date: `2026-10-24T16:36:53.831080+00:00`.

| Case | Native µs | TF eager µs | TF graph µs | Native/eager | Native/graph | Native CV% | Eager CV% | Graph CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Add/rows:32/f64:0/real_time` | 1.99566 | 50.4647 | 251.311 | 0.0395457 | 0.00794101 | 6.71131 | 9.61488 | 7.60494 |
| `Add/rows:32/f64:1/real_time` | 4.02637 | 46.8467 | 269.443 | 0.0859477 | 0.0149433 | 8.01839 | 8.63747 | 2.23573 |
| `Add/rows:512/f64:0/real_time` | 21.4232 | 71.1137 | 324.063 | 0.301253 | 0.0661082 | 14.4176 | 9.25675 | 6.91787 |
| `Add/rows:512/f64:1/real_time` | 44.8405 | 96.7898 | 345.985 | 0.463277 | 0.129602 | 15.6026 | 5.2762 | 3.73507 |
| `Add/rows:8192/f64:0/real_time` | 539.831 | 634.094 | 1073.94 | 0.851342 | 0.502665 | 3.39117 | 7.37159 | 7.34688 |
| `Add/rows:8192/f64:1/real_time` | 966.906 | 1128.92 | 1198.42 | 0.856488 | 0.806819 | 8.22925 | 12.5148 | 5.57693 |
| `BroadcastAdd/rows:32/f64:0/real_time` | 2.85376 | 48.6643 | 278.68 | 0.0586416 | 0.0102402 | 4.76361 | 7.70843 | 14.6598 |
| `BroadcastAdd/rows:32/f64:1/real_time` | 3.80448 | 45.2361 | 287.275 | 0.0841028 | 0.0132433 | 5.44383 | 7.23667 | 14.3782 |
| `BroadcastAdd/rows:512/f64:0/real_time` | 31.2392 | 101.57 | 283.021 | 0.307562 | 0.110378 | 11.7962 | 7.74318 | 12.5068 |
| `BroadcastAdd/rows:512/f64:1/real_time` | 48.4747 | 154.486 | 388.245 | 0.313781 | 0.124856 | 14.1768 | 8.11722 | 4.73225 |
| `BroadcastAdd/rows:8192/f64:0/real_time` | 498.067 | 1016.02 | 1459.77 | 0.490212 | 0.341195 | 8.12691 | 10.526 | 7.38873 |
| `BroadcastAdd/rows:8192/f64:1/real_time` | 1196.35 | 2119.2 | 2733.12 | 0.564528 | 0.437722 | 5.02364 | 3.02318 | 4.23273 |
| `Exp/rows:32/f64:0/real_time` | 16.8368 | 34.4165 | 235.343 | 0.489207 | 0.0715416 | 6.79195 | 8.6322 | 10.9367 |
| `Exp/rows:32/f64:1/real_time` | 29.2557 | 49.4949 | 323.263 | 0.591085 | 0.090501 | 3.21861 | 6.56411 | 11.64 |
| `Exp/rows:512/f64:0/real_time` | 316.645 | 163.847 | 424.129 | 1.93256 | 0.746578 | 0.46627 | 2.58242 | 1.58057 |
| `Exp/rows:512/f64:1/real_time` | 475.655 | 312.212 | 640.2 | 1.5235 | 0.742979 | 1.74442 | 10.7982 | 2.08928 |
| `Exp/rows:8192/f64:0/real_time` | 4831.19 | 2198.58 | 2164.51 | 2.19742 | 2.23201 | 12.7642 | 13.8423 | 8.03312 |
| `Exp/rows:8192/f64:1/real_time` | 6701.84 | 5192.5 | 6053.7 | 1.29068 | 1.10706 | 4.38613 | 10.3907 | 14.2647 |
| `LogSoftmax/rows:32/f64:0/real_time` | 41.7614 | 50.7674 | 287.772 | 0.822604 | 0.14512 | 18.2048 | 11.8577 | 15.61 |
| `LogSoftmax/rows:32/f64:1/real_time` | 54.4124 | 68.8042 | 311.612 | 0.79083 | 0.174616 | 11.0176 | 8.10316 | 5.63278 |
| `LogSoftmax/rows:512/f64:0/real_time` | 594.793 | 290.443 | 555.195 | 2.04789 | 1.07132 | 15.0398 | 13.9204 | 1.99862 |
| `LogSoftmax/rows:512/f64:1/real_time` | 961.013 | 574.03 | 913.73 | 1.67415 | 1.05175 | 0.599608 | 10.3224 | 1.54714 |
| `LogSoftmax/rows:8192/f64:0/real_time` | 10485.6 | 3393.46 | 5527.69 | 3.08995 | 1.89693 | 18.4014 | 3.21022 | 2.87834 |
| `LogSoftmax/rows:8192/f64:1/real_time` | 14729.1 | 9806.61 | 9335.79 | 1.50195 | 1.5777 | 7.73992 | 1.1993 | 11.1987 |
| `Matmul/rows:256/f64:0/real_time` | 178.927 | 1028.89 | 1433.96 | 0.173903 | 0.124779 | 10.3855 | 11.2028 | 12.0556 |
| `Matmul/rows:256/f64:1/real_time` | 754.203 | 2880.46 | 3423.74 | 0.261834 | 0.220287 | 12.7579 | 1.84134 | 9.48944 |
| `Matmul/rows:512/f64:0/real_time` | 1379.89 | 9281.05 | 8012.15 | 0.148678 | 0.172225 | 11.9981 | 3.37331 | 8.93625 |
| `Matmul/rows:512/f64:1/real_time` | 5364.74 | 21171.1 | 19870.6 | 0.253399 | 0.269983 | 13.1685 | 6.53497 | 7.46038 |
| `Matmul/rows:64/f64:0/real_time` | 5.55633 | 67.9931 | 277.891 | 0.081719 | 0.0199947 | 4.20888 | 7.28743 | 5.06853 |
| `Matmul/rows:64/f64:1/real_time` | 13.2576 | 88.3989 | 313.929 | 0.149974 | 0.0422312 | 9.2836 | 8.42425 | 8.31011 |
| `Mean/rows:32/f64:0/real_time` | 3.73522 | 60.2617 | 291.418 | 0.0619834 | 0.0128174 | 2.99812 | 8.39349 | 3.08308 |
| `Mean/rows:32/f64:1/real_time` | 3.53176 | 57.3957 | 294.415 | 0.0615335 | 0.0119958 | 2.86364 | 11.1427 | 1.45369 |
| `Mean/rows:512/f64:0/real_time` | 26.8165 | 71.3322 | 323.538 | 0.375938 | 0.0828853 | 2.27581 | 5.56475 | 13.3842 |
| `Mean/rows:512/f64:1/real_time` | 24.2258 | 86.1518 | 257.786 | 0.281198 | 0.0939762 | 12.6036 | 4.56151 | 10.838 |
| `Mean/rows:8192/f64:0/real_time` | 410.365 | 268.244 | 650.044 | 1.52982 | 0.631289 | 6.50607 | 13.1182 | 18.4232 |
| `Mean/rows:8192/f64:1/real_time` | 608.848 | 643.901 | 1038.37 | 0.945562 | 0.586347 | 9.49289 | 3.31524 | 9.75916 |
| `Multiply/rows:32/f64:0/real_time` | 1.85705 | 36.4148 | 277.317 | 0.050997 | 0.00669648 | 5.79564 | 9.06297 | 6.12268 |
| `Multiply/rows:32/f64:1/real_time` | 3.62633 | 38.6622 | 280.75 | 0.0937953 | 0.0129166 | 5.26968 | 7.16093 | 2.75096 |
| `Multiply/rows:512/f64:0/real_time` | 24.084 | 61.389 | 291.811 | 0.392318 | 0.0825329 | 8.20008 | 15.1642 | 8.10188 |
| `Multiply/rows:512/f64:1/real_time` | 42.1125 | 78.3644 | 326.524 | 0.537393 | 0.128972 | 6.31461 | 6.3654 | 7.2699 |
| `Multiply/rows:8192/f64:0/real_time` | 479.55 | 632.597 | 916.077 | 0.758066 | 0.523482 | 7.22806 | 1.51598 | 8.37285 |
| `Multiply/rows:8192/f64:1/real_time` | 994.241 | 1023.31 | 1631.34 | 0.971595 | 0.609463 | 8.66156 | 3.3844 | 5.92604 |
| `Relu/rows:32/f64:0/real_time` | 1.40578 | 29.0362 | 294.104 | 0.0484147 | 0.00477987 | 12.6174 | 4.03762 | 4.88596 |
| `Relu/rows:32/f64:1/real_time` | 2.81159 | 31.5789 | 270.755 | 0.0890337 | 0.0103843 | 2.44086 | 1.48226 | 5.19002 |
| `Relu/rows:512/f64:0/real_time` | 22.1515 | 52.6307 | 326.463 | 0.420885 | 0.067853 | 2.79313 | 1.79328 | 2.76487 |
| `Relu/rows:512/f64:1/real_time` | 35.215 | 74.0115 | 356.242 | 0.475804 | 0.0988512 | 15.3546 | 10.8279 | 1.3693 |
| `Relu/rows:8192/f64:0/real_time` | 467.14 | 541.761 | 889.053 | 0.862262 | 0.525436 | 2.39579 | 2.6579 | 3.8248 |
| `Relu/rows:8192/f64:1/real_time` | 852.68 | 1076.61 | 1566.01 | 0.792008 | 0.544493 | 5.03312 | 1.86089 | 2.65349 |
| `Softmax/rows:32/f64:0/real_time` | 52.0576 | 46.1366 | 307.958 | 1.12834 | 0.169041 | 5.50798 | 9.33304 | 3.25618 |
| `Softmax/rows:32/f64:1/real_time` | 49.7187 | 55.3254 | 289.344 | 0.89866 | 0.171832 | 14.3693 | 12.6641 | 10.5897 |
| `Softmax/rows:512/f64:0/real_time` | 788.946 | 217.138 | 554.766 | 3.63339 | 1.42212 | 8.33907 | 6.8212 | 4.30158 |
| `Softmax/rows:512/f64:1/real_time` | 801.185 | 488.186 | 688.371 | 1.64115 | 1.16389 | 10.7097 | 7.28241 | 7.49854 |
| `Softmax/rows:8192/f64:0/real_time` | 11398.5 | 3312.7 | 4787.91 | 3.44085 | 2.38068 | 5.95796 | 9.55684 | 7.39578 |
| `Softmax/rows:8192/f64:1/real_time` | 15063.5 | 7640.22 | 8773.97 | 1.97161 | 1.71684 | 4.29282 | 5.56268 | 9.32228 |
| `SquareGradient/rows:32/f64:0/real_time` | 28.3943 | 548.063 | 262.919 | 0.0518084 | 0.107996 | 15.0106 | 3.52154 | 10.904 |
| `SquareGradient/rows:32/f64:1/real_time` | 32.9787 | 861.035 | 280.968 | 0.0383013 | 0.117376 | 9.96094 | 8.98692 | 8.23686 |
| `SquareGradient/rows:512/f64:0/real_time` | 254.053 | 721.868 | 332.63 | 0.351939 | 0.763771 | 9.84854 | 8.65839 | 6.15805 |
| `SquareGradient/rows:512/f64:1/real_time` | 303.511 | 1362.96 | 421.227 | 0.222685 | 0.720542 | 8.57484 | 12.2665 | 4.83294 |
| `SquareGradient/rows:8192/f64:0/real_time` | 4313.61 | 4676.77 | 1781.4 | 0.922347 | 2.42148 | 4.30686 | 4.26405 | 4.07657 |
| `SquareGradient/rows:8192/f64:1/real_time` | 6649.19 | 9091.29 | 2877.12 | 0.73138 | 2.31105 | 3.81723 | 1.77919 | 5.29989 |
| `SumAll/rows:32/f64:0/real_time` | 2.86009 | 58.5605 | 288.888 | 0.0488398 | 0.00990032 | 1.93387 | 2.43754 | 2.85632 |
| `SumAll/rows:32/f64:1/real_time` | 2.19874 | 54.5079 | 299.243 | 0.040338 | 0.00734768 | 11.7168 | 8.75551 | 1.91228 |
| `SumAll/rows:512/f64:0/real_time` | 18.9924 | 69.5408 | 281.565 | 0.273112 | 0.0674529 | 13.0512 | 11.7186 | 5.06409 |
| `SumAll/rows:512/f64:1/real_time` | 23.7228 | 73.6284 | 320.528 | 0.322197 | 0.0740119 | 15.5008 | 6.82141 | 7.05042 |
| `SumAll/rows:8192/f64:0/real_time` | 370.07 | 294.995 | 546.702 | 1.25449 | 0.676914 | 6.84138 | 6.47757 | 8.22627 |
| `SumAll/rows:8192/f64:1/real_time` | 545.163 | 651.857 | 1017 | 0.836322 | 0.536048 | 8.32548 | 7.09306 | 5.76078 |
| `SumLast/rows:32/f64:0/real_time` | 3.04816 | 46.5862 | 283.821 | 0.0654305 | 0.0107397 | 3.88281 | 1.86513 | 10.771 |
| `SumLast/rows:32/f64:1/real_time` | 2.66167 | 46.1933 | 296.205 | 0.0576203 | 0.00898592 | 14.196 | 0.937355 | 1.59277 |
| `SumLast/rows:512/f64:0/real_time` | 27.1696 | 52.0793 | 227.296 | 0.521697 | 0.119534 | 7.47555 | 11.0707 | 16.5509 |
| `SumLast/rows:512/f64:1/real_time` | 24.0436 | 65.2708 | 281.826 | 0.368368 | 0.0853137 | 9.98195 | 9.55927 | 8.19114 |
| `SumLast/rows:8192/f64:0/real_time` | 372.727 | 341.183 | 698.758 | 1.09245 | 0.533413 | 9.73788 | 4.28662 | 4.84299 |
| `SumLast/rows:8192/f64:1/real_time` | 580.454 | 696.732 | 1134.5 | 0.83311 | 0.511641 | 2.54191 | 2.53468 | 3.80923 |

</details>

<!-- tensorflow-comparisons:end -->

## Consolidated historical plans and results

The following archives preserve the complete earlier constraints, plans, accepted
and rejected experiments, counters and conclusions. They are **superseded historical
snapshots**; their original “current/latest/next” wording and counts refer to their
own date/image, not the final pass above. The original separate planning/result
files have been retired so future maintenance has one destination. Raw measurements
remain unchanged. Earlier comparison tables are also fully reproduced above.

<a id="history-plans-backend-benchmark-plan-md"></a>
<details><summary>Historical source: plans/backend-benchmark-plan.md</summary>

Date: 2026-10-23. Status: agreed benchmark matrix audited, implemented and validated (1,452 cases after performance-work extensions).

#### Scope and assessment limits

Prioritize the canonical C++ backend, then measure binding overhead separately if needed. The initial assessment inspected source, benchmark registrations, CMake, task recipes, and existing test locations without running executables. That original inventory is historical; the implementation progress and follow-up audit below record successful runtime validation. The starting working directory is a generated CMake dependency build directory; this plan belongs in the source repository's plans directory.

#### Agreed focus and implementation order

Focus confirmed on 2026-10-23: tensor linear algebra, tensor operations, autograd, general optimizers, and neural networks. Implementation followed the slices below; the progress record documents the resulting coverage.

| Order | Slice | Initial deliverable |
| --- | --- | --- |
| 1 | Shared harness | Release core verification, bounded smoke runs, JSON metadata, reliable comparisons |
| 2 | Tensor linear algebra | Rectangular/batched matmul; solve, triangular solve, Cholesky, QR, SVD and symmetric eigendecomposition |
| 3 | Tensor primitives | Allocation/copy, structural operations, elementwise/broadcast operations, reductions and indexing in both tensor APIs |
| 4 | Autograd | Recording overhead, complete gradients, graph depth/fan-out, broadcast/reduction gradients and differentiable linear algebra |
| 5 | General optimizers | Deterministic bounded solves for gradient descent, Adam, L-BFGS and Newton; then derivative-free/population and constrained/multi-objective families |
| 6 | Neural networks | Dense inference/training steps, compact CNN, sequence and transformer workloads, with component timings |

Tensor linear algebra deserves a dedicated slice before the broader tensor sweep. `include/datamunge/nn/linalg.hpp` already exposes batched float32/float64 solves and decompositions, plus selectable built-in/LAPACK backends. Benchmark Auto, Builtin and available Lapack modes explicitly; skip unavailable LAPACK instead of reporting fallback results as LAPACK. Restore process-wide backend settings after each case. Inspect `linalg::Tensor` separately and benchmark only its supported operations.

For linear algebra, start with square sizes 16/64/256, selected tall/wide QR/SVD cases, batch sizes 1/16, and one/multiple right-hand sides. Curate combinations to bound runtime. Generate well-conditioned invertible/SPD matrices outside timing; use matrices with distinct singular/eigenvalues for gradient cases. Validate solve residuals, decomposition reconstruction/orthogonality and eigen residuals outside timing, with dtype-appropriate tolerances. Add least squares, inverse, determinant/logdet, pseudoinverse and general eigendecomposition in a second slice. Compare residual/gradient quality as well as runtime. Avoid assuming automatic backend crossover sizes; inspect the implementation and include cases around those thresholds.

Autograd should initially prioritize `nn::grad`, including solve/Cholesky and selected QR/SVD/eigh gradients supported by the API. Keep standalone dual/hyperdual/tape benchmarks as a follow-on within this area rather than blocking NN autograd coverage.

General optimizers refer to `include/datamunge/optim/`; prioritize that family independently of `nn::Optimizer`. Tensor optimizer microbenchmarks remain useful supporting work for NN training attribution, starting with SGD/Adam/AdamW, but the full 15-kind sweep is secondary.

Deferred from the active scope: reinforcement learning, binding overhead, unrelated data structures, and broad image/audio/statistical kernel coverage. Preserve existing sparse benchmarks and run relevant checks, but defer their expansion. The detailed sections below retain the longer-term backlog; execute them in the order in this table, not their original numbering.

#### Original coverage assessment (before implementation)

| Area | Existing benchmark coverage | Missing coverage |
| --- | --- | --- |
| NN tensors | `benchmarks/nn/nn_bench.cpp`: square GEMM at 128/512/1024 with serial/default threading; one batched GEMM shape | Allocation, copies/views, dtype conversion, broadcasting, reductions, indexing, layout effects, rectangular GEMM |
| NN autograd | Pooling, batch norm, bias/ReLU, and attention include forward plus backward | Isolated graph construction/traversal, gradient accumulation, depth/fan-out, backward-only and memory scaling |
| Standalone autodiff | No dedicated benchmark found for `include/datamunge/autodiff/{dual,hyperdual,tape}.hpp` | Forward mode, second derivatives, tape recording and reverse sweeps |
| General linear algebra tensors | Sparse matrix benchmarks in `benchmarks/cpp/linalg_bench.cpp`; no dedicated `linalg::Tensor` benchmarks found | Dense tensor operations supported by `include/datamunge/linalg/tensor.hpp` |
| Tensor optimizers | Adam appears inside a full CNN epoch | Isolated updates for all 15 `nn::OptimizerKind` values, state initialization, clipping and accumulation |
| General optimizers | No dedicated timing benchmarks found under `benchmarks/` | First/second-order, constrained, derivative-free, population and multi-objective families |
| Neural networks | Conv2d forward/input/filter gradients, max pool, batch norm, bias/ReLU; plain/fused attention with causal and backward variants; one synthetic MNIST CNN epoch | Small repeatable training steps, inference, dense networks, recurrent/sequence workloads, losses, broad shape/dtype/layout sweeps |
| Reinforcement learning | No dedicated benchmark found | Environment stepping, replay, returns/advantages, learner updates and rollout throughput |
| Supporting backend | ThreadSafeVector push/snapshot/apply; sparse SpMV, compression, conversion and expressions | Broader dense/sparse kernels, threading scaling, memory and other backend families |

Existing integration: Google Benchmark; `benchmarks/CMakeLists.txt` builds `linalg_bench` (including data structures) and `nn_bench`; `just benchmark`, `just bench-nn *ARGS`, `just benchmark-compare BASE NEW`; C++ Nix shell includes Google Benchmark. An archived linalg JSON result exists in `benchmarks/results/`.

RL currently lives primarily in `examples/nn/common/rl/` and executable examples under `examples/nn/6_deep_reinforcement_learning/`, rather than an installed `datamunge::rl` API. Benchmark reusable helpers directly where possible; label example-level workloads accordingly. Do not turn this effort into a public RL API redesign.

Correctness tests already exist for both tensor systems, standalone autodiff, NN autograd, layers, convolution, sequence, attention, training, memory, and general optimizers. Reuse their small reference problems outside timed regions; correctness tests are not timing benchmarks.

#### Measurement contract

- Use Google Benchmark consistently, with deterministic inputs and stable names identifying operation, shape, dtype, layout, thread configuration and backend.
- Build both the benchmark drivers and linked core in Release. Verify actual compiler flags; driver-only `-O3` cannot establish that the library is optimized. Replace compiler-specific flags with appropriate CMake configuration when touching infrastructure.
- Keep fixtures, random generation, reference checks and resets outside measured regions unless allocation, initialization or setup is the stated workload. Clearly distinguish kernel-only, forward-only, forward-plus-backward, backward-only where supported, and complete training steps.
- Prevent dead-code elimination by consuming outputs and using suitable optimization barriers. Check finite values and representative results before timing. Do not copy huge results merely to consume them.
- For tensor mutation and optimizer/training state, explicitly define reset policy. Separate first-step state allocation from warmed updates; use bounded deterministic update blocks and reset outside timing so calibration/repetitions do not change the workload or drive values into invalid ranges.
- Fix seeds; explicitly set and restore library thread counts. Record BLAS/backend and its thread controls separately: library thread count does not necessarily constrain BLAS threads. Compare one thread with a fixed parallel count; label machine-default runs separately.
- Use wall time for threaded throughput and report CPU time where useful. Record compiler/version, flags, build revision and dirty state, OS, CPU, backend, seeds and thread environment with JSON results.
- Report latency and useful work: elements/s, updates/s, transitions/s or samples/s. Report FLOP rates only with an explicit operation count. Label estimated byte traffic as an estimate.
- Use `nn::PeakMemoryScope`, live bytes and storage counts for separate memory measurements. These count tensor storage, not total allocations or RSS; do not call peak resets concurrently. Separate persistent optimizer state from temporary peak storage.
- Keep smoke runs small and bounded; keep large attention/CNN epochs opt-in. Use repeated measurements and medians/dispersion for baselines. Avoid universal timing assertions on shared CI.

#### Implementation sequence

Each numbered item is an independently reviewable change. Implement and validate one at a time.

##### 1. Reliable harness and baseline (first change)

- Preserve current benchmark names/results. Add consistent argument forwarding, output-directory creation, suite selection and short smoke presets to `just` workflows.
- Fix comparison tooling discovery: `benchmark-compare` currently assumes a CPM source checkout although CMake prefers an installed package.
- Verify Release flags for the linked core and record build/backend/thread metadata. Keep optional backend dependencies optional.
- Audit existing cases: consume SpMV outputs; correct/document sparse input generation (LCG modulo correlations and claimed duplicate distribution); distinguish actual shared-container contention from per-thread independent containers in data-structure benchmarks; check throughput accounting.
- Audit GEMM rate units: the current counter is named GFLOPS but uses raw FLOPs with automatic scaling; make reported units unambiguous.
- Separate the single-iteration 55,000-example CNN epoch and large attention sweep from normal smoke/default runs; add a bounded training-step case later.
- Deliverable: existing suites list and smoke-run, valid JSON and comparison output, documented commands, and a same-machine baseline. No performance thresholds yet.

##### 2. Tensor primitives (highest coverage payoff)

- Add separate cases for `nn::Tensor` and `linalg::Tensor`, using only operations each actually supports. Inspect APIs and tests before choosing exact cases.
- Measure allocation/initialization, clone/copy, view creation versus materialization, supported dtype casts, elementwise arithmetic/activations, broadcasting, sum/mean/max reductions, transpose/reshape, slices/gather/scatter and concatenation.
- Cover float32/float64 where supported; representative sizes around 1K, 64K and 1M elements; contiguous and supported strided/transposed paths; scalar, row and channel broadcasts. Use a curated matrix rather than every Cartesian combination.
- Extend GEMM to small overhead-sensitive, rectangular and batched shapes; label built-in versus BLAS builds. Include supported einsum contraction paths.
- Deliverable: latency/throughput baselines and selected storage peaks with references checked outside timing.

##### 3. Autodiff and autograd

- Standalone dual/hyperdual: fixed scalar expressions and vector objectives at increasing dimensions; compare against equivalent analytic value/derivative work where useful.
- Tape: recording-only, reverse-only using a valid prebuilt tape, and complete gradients; sweep chain depth and branching/shared subexpressions. Consume derivative values and report nodes/s.
- NN autograd: no-grad versus recording forwards, graph construction, complete gradients, fan-out accumulation, broadcasting/reduction gradients and dense-layer loss gradients. Add backward-only only after confirming graph reuse/lifetime semantics.
- Track storage after repeated gradients to detect retained tensor buffers; avoid interpreting tensor storage as total graph memory.
- Deliverable: separate fixed graph overhead from numerical kernel cost; validate gradients using analytic references or existing gradcheck facilities outside timing.

##### 4. Tensor optimizer updates

- Register all current kinds: SGD, Momentum, Nesterov, Adagrad, RMSProp, Adam, AdamW, Nadam, AMSGrad, Adamax, Lamb, Lars, Lion, Adafactor and Adadelta.
- Start with SGD/Adam/AdamW, then complete remaining kinds. Use identical finite deterministic gradients and representative vector/matrix parameters.
- Hold total elements fixed while comparing one large tensor with many small tensors; sweep total parameter size. Include matrix cases for Adafactor's factored state.
- Separate construction/first step, warmed updates, clipping and gradient accumulation. Validate one update against a small reference; report updated elements/s, persistent state and temporary storage.

##### 5. General optimization algorithms

- Start with gradient descent, Adam, L-BFGS, Newton and representative derivative-free/population methods; add constrained and multi-objective methods in later batches.
- Use well-conditioned and ill-conditioned quadratics, Rosenbrock, and bounded multimodal objectives; add small constrained and multi-objective problems suited to the relevant algorithms.
- Separate fixed-work throughput from time-to-target quality. Count objective/gradient/Hessian evaluations; do not compare methods solely by iteration count. Record stopping tolerances, achieved objective/error and success.
- For stochastic methods use a fixed seed set and aggregate quality, evaluations and time; compare like evaluation budgets. Confirm supported step APIs before promising isolated iterations; otherwise benchmark bounded solves.

##### 6. Neural network workloads

- Add dense/MLP inference and training steps (forward, loss, gradient, optimizer separately and combined); then a compact CNN, recurrent/sequence step and transformer block using existing APIs/helpers.
- Expand losses, softmax, normalization, embedding/indexing, supported convolutions/pooling and attention with selected batch/sequence/channel/dtype shapes. Keep forward, backward and combined labels precise.
- Compare plain/fused attention outputs and gradients before timing. Include memory scaling with sequence length and causal masks.
- Reuse deterministic synthetic data; keep downloads, file I/O and parameter initialization outside training-step measurements. Keep epochs as optional integration workloads.
- Deliverable: samples/tokens per second and peak tensor storage, attributable to component timings.

##### 7. Reinforcement learning (deferred)

- Begin with existing deterministic classic-control/grid environments: step/reset separately, then vectorized stepping at selected environment counts; exclude rendering and external simulators.
- Benchmark uniform and prioritized replay insertion, sampling and priority updates with capacity, observation shape and batch size sweeps. Specify whether sampled batches are copied/materialized and keep buffers prefilled.
- Benchmark discounted returns and advantage computation using existing policy-gradient helpers where available; tabular Q/SARSA updates using existing helpers.
- Add bounded DQN learner updates with prefabricated transitions, followed by existing policy-gradient/PPO and SAC/TD3 update paths where their code can be reused cleanly. Separate replay/environment costs, network evaluation, loss/backprop and optimizer updates.
- Add short seeded rollout-plus-update workloads only after components are measured. Throughput and learning quality are separate outputs; learning curves require multiple seeds and longer opt-in runs.
- External ALE/vsim integrations and rendering remain optional suites with declared dependencies, not prerequisites for backend benchmarking.

##### 8. Broader backend (deferred) and regression tracking

- Extend dense/sparse linear algebra to supported solves/decompositions and structured sparsity; add FFT, image/audio transforms, RNG/data batching and representative statistical/algorithm kernels based on API inventory and observed hotspots.
- Benchmark thread scaling on sufficiently large kernels; measure small-task scheduling overhead separately. Keep each backend variant in a separate configured build.
- Add CI build/list/smoke correctness checks first. Archive complete baseline JSON and metadata from a controlled runner; later flag statistically credible regressions using both relative and absolute tolerances and rerun confirmation.
- Separate algorithm quality dashboards from runtime regression comparisons. Add optional binding call/serialization overhead only after backend baselines stabilize.

#### Completion criteria for every implementation slice

1. Exact benchmark scope and timing boundaries documented; input sizes, seeds and thread settings reproducible.
2. Native C++ build works through the C++ dev shell and an idiomatic `just` entrypoint.
3. Benchmark list and bounded smoke run succeed; representative outputs/gradients are checked outside timing and failures surface clearly.
4. JSON contains stable names, appropriate counters and configuration metadata; results are comparable on the same machine/configuration.
5. Relevant existing correctness tests pass; new correctness tests are added only when the benchmark exposes an uncovered behavior.
6. Record findings and remaining gaps without claiming measured speedups from source inspection.

The agreed focus table determined implementation order. The original detailed sequence above remains useful as an expansion backlog; the delivered coverage and validation are recorded below.


#### Implementation progress (2026-10-23)

The five requested backend areas now have a dedicated Google Benchmark executable,
`backend_bench`, with 932 registered shape/dtype/backend/mode cases. See
`benchmarks/README.md` for the exact workloads, timing boundaries and commands.

- [x] Shared harness: Release configuration for core/drivers; actual compiler-command checks; argument forwarding; output creation; source/build/thread metadata; dependency-independent median comparison; bounded smoke selection.
- [x] Tensor linear algebra: matmul, solves, Cholesky, QR, SVD, eigh, inverse, logdet, least squares, pseudoinverse, determinant/slogdet, nonsymmetric eigenvalues and einsum; batched/rectangular cases, backend/crossover sweep and numerical residual checks.
- [x] Tensor primitives: both tensor APIs, allocation/copy/structure/arithmetic/reductions/indexing, supported dtype/layout cases and explicit threading sweep.
- [x] Autograd: no-grad/record/full-gradient/retained-backward, depth/fan-out and broadcast gradients, differentiable solve/Cholesky/QR/SVD/eigh; standalone tape, dual gradients and hyperdual Hessians. Selected tensor-storage peaks and post-gradient storage checks.
- [x] General optimizers: gradient descent, Adam, L-BFGS, Newton, Nelder-Mead, differential evolution and PSO over quadratic/ill-conditioned/Rosenbrock objectives; evaluation counts, achieved objective and target status; SQP constrained quadratic and NSGA-II tradeoff cases with multiple stochastic seeds.
- [x] Neural networks: compact MLP/CNN/RNN/transformer inference, gradients and Adam steps; small plain/fused/causal attention parity and gradient checks; activation/softmax/cross-entropy; all 15 tensor optimizer kinds, first/warmed updates, parameter grouping, clipping and gradient accumulation.
- [x] Integration: `just bench-backend`, `just bench-backend-smoke`, `just bench-backend-check`; existing suites preserved; focused existing GTest target; Linux CI smoke/artifact job added (remote execution awaits CI).
- [x] Existing-suite audit: sparse output barriers and improved random/duplicate fixture generation, correct logical thread accounting/labels, explicit FLOP/s units and previous-thread restoration. Historical sparse baselines must be regenerated after the fixture change.

Validation completed so far:

- All 932 focused cases executed with one timed iteration and passed their fixture checks; zero optional skips on this Accelerate/LAPACK machine.
- All 224 curated smoke cases passed.
- All 350 reused C++ correctness tests passed against the benchmark core.
- Five benchmark runner contract tests passed.
- Legacy sparse/data-structure and NN GEMM smoke checks passed.
- Saved `benchmarks/results/backend_baseline_20261023.json`: 224 cases, three calibrated repetitions per case (`0.01s` minimum each), 672 iteration records plus median/mean/dispersion aggregates; zero errors/skips.
- A second repeated run passed; the comparator matched all 224 cases, with no additions/removals. Self-comparison returned zero changes for every case.
- `just` recipe parsing, Python runner tests, new C++ formatting checks and focused diff whitespace checks passed.
- Local validation used macOS arm64, Clang 21.1.8, Release, Accelerate/LAPACK, one default library/BLAS thread (explicit thread-scaling cases use 1/2/4). The working tree was already dirty; metadata records that fact. Short local baselines are not statistically established regression thresholds.

This completes initial representative coverage across all agreed families, not an
exhaustive benchmark for every optimizer or model architecture. Larger production
shape sweeps, additional algorithms, controlled learning-quality studies,
time-to-target experiments, and statistical performance gates remain extensions.
RL, bindings and unrelated backend families remain deferred as agreed.


Entry points:

```sh
just bench-backend-smoke
just bench-backend-check
just bench-backend -- --benchmark_filter=BM_TensorLinalg --benchmark_repetitions=5
just benchmark-compare baseline.json candidate.json
```

The implementation changes no core APIs or generated bindings. Existing unrelated
working-tree changes were preserved. Detailed measurement semantics and expansion
points are in `benchmarks/README.md`.


#### Follow-up coverage audit

The follow-up audit filled the remaining explicit coverage gaps and introduced an
executable inventory in `benchmarks/coverage.json`. The suite now registers 1,089
cases; all passed one-iteration execution, with 382 existing C++ correctness tests
and six runner tests passing. See `backend-benchmark-coverage-audit.md` for the
precise checklist and `backend-performance-improvement-plan.md` for proposed
performance work. The older 932-case/350-test counts above describe the first
implementation and are retained as history.

</details>

<a id="history-plans-tensorflow-remaining-gaps-md"></a>
<details><summary>Historical source: plans/tensorflow-remaining-gaps.md</summary>

2026-10-24. Status: complete.

1. Freeze the current library/driver and remeasure Float32 row/global sum, mean,
   and square-gradient workloads, with Float64 and unchanged operation controls.
2. Inspect widening/accumulator throughput and gradient materialization. Preserve
   double accumulation, higher derivatives, independent outputs and mutation checks.
3. Retain measured improvements; add targeted coverage for affected paths and
   compare adjacent TensorFlow eager/graph samples, including uncertainty.
4. Persist raw/summary evidence and update results. Run the full native benchmark
   matrix, correctness tests and built-in fallback checks.


#### Implemented candidates and early evidence

- ARM Float32 reductions widen vectors to eight independent Float64 vector
  accumulators (sixteen elements per iteration) and combine them in a tree.
  Scalar/non-ARM and Float64 reduction paths remain available. No Float32 partial
  sums or global fast-math flags are used. Cancellation and odd-tail tests pass.
- Binary kernels recognize fully zero-stride broadcast views as uniform scalar
  storage, including nonzero offsets. A contiguous peer can then use the regular
  vector loop without expanded seeds or generic strided indexing. The shape and
  original storage remain intact. Partial broadcasts and empty cases are checked.
- Public `square()` ordinary backward now fuses its scale-and-multiply into one
  output pass; higher-order derivatives retain recorded tensor operations. Its
  original scale-before-seed order is preserved, including overflow behavior.
  Eight sum/mean VJP benchmarks cover both sizes and dtypes.

Early candidate API measurements: large Float32 sums/mean improved ~1.4–1.5×;
`x*x` square-gradient improved ~2.3× (Float32) and ~1.4× (Float64). Unchanged
large add/multiply controls were near parity. The quiet-host TensorFlow baseline
had low repetition dispersion and confirmed larger gaps than the previous noisy
pass; final ranking will use a new adjacent-order comparison.

Exploratory standalone reduction kernels found eight widened vector accumulators
better than four for large scans. Sixteen accumulators and 2×/4× loop unrolling
showed no material benefit and were not retained. Final speedup claims use the
full API benchmark runs, not the exploratory kernels.


#### Final measurements and disposition

- Retained eight widened ARM vector accumulators and uniform-storage binary fast
  paths. Five-repetition native comparisons, including unchanged add/multiply and
  MLP inference controls, confirm the gains with controls within about 1%.
- In adjacent TensorFlow comparisons, large Float32 row sum improved 94.79 →
  65.87 μs; global sum 92.10 → 61.47 μs; mean 92.25 → 61.52 μs. Native / TF eager
  ratios narrowed from ~1.84–1.96 to ~1.23–1.36, preserving double accumulation.
- The x*x gradient gap closed on both dtypes: Float32 native / TF graph 1.83 →
  0.81, Float64 1.24 → 0.89. This remains a warm API-call comparison, including
  TensorFlow Python/graph dispatch; it is not a TensorFlow C++ kernel benchmark.
- Public-square sum/mean VJPs improved 1.75–2.64× in eight cases; large native
  gradient peak tensor storage fell 33.3%. Existing scale order and higher-order
  differentiation were preserved.
- Extra 2×/4× unrolling and sixteen accumulators did not improve the retained
  kernel materially. Adjacent-row batching had no target-width benefit and
  regressed wider rows; it was rejected. Exploratory logs are retained separately
  and do not support cross-framework timing claims. An initially contended
  adjacent-row run was excluded and repeated after suite execution.

#### Validation and persistent evidence

- **501** correctness tests passed on Accelerate; built-in **499 passed, two
  expected LAPACK-only skips**. New checks cover SIMD boundaries/tails,
  cancellation, offset views, uniform versus partial broadcasts, operand order,
  empty tensors, strided VJPs, differentiable seeds and Square overflow ordering.
- Full native matrix: **1,452** accelerated cases passed; built-in **1,324** ran
  with **128** expected optional LAPACK skips. Eight new public-square VJP cases
  are covered by the audit and the ordinary native CI smoke selection.
- **18** Python runner/history/comparison contract tests passed. The TensorFlow
  driver can target exact subsets with `--case`, rejects incomplete pairings,
  and defaults to the existing full 72-case comparison.
- Results, frozen-reference comparisons, counters and source/library hashes are
  archived under `benchmarks/results/tensorflow/` and content-addressed history.
  Summary and exact next priorities: `benchmarks/results/tensorflow/GAP_RESULTS.md`.

The remaining Float32 reduction gap is reduced, not eliminated. Further work
should profile widening throughput and other CPU implementations while retaining
precision. Lower-precision partial sums would be a separate explicit numerical
policy decision. No public APIs or generated bindings changed.

</details>

<a id="history-plans-release-benchmark-verification-md"></a>
<details><summary>Historical source: plans/release-benchmark-verification.md</summary>

2026-10-24. Status: complete.

1. Inspect actual core/driver compiler commands, CMake configuration, runtime
   build tags and loaded-library hashes. Existing benchmark builds report
   Release with -O3 -DNDEBUG; the surrounding examples/debug working directory
   does not determine the benchmark configuration.
2. Build the current backend and benchmark driver in a fresh, explicit Release
   directory; reject Debug/non-Release configurations before timing.
3. Verify the frozen pre-gap baseline library against its archived Release-backed
   hash, then repeat before/after native comparisons and all 72 adjacent-order
   TensorFlow CPU comparisons with fixed single-thread settings.
4. Archive raw samples, compiler evidence, summaries and a new results report.
   Run appropriate correctness/fixture and runner-contract checks.


Additional consistency check: the shared workspace gained a softmax reciprocal
normalization change during the first rerun. The first-image observations remain
archived under `release_*`. The affected tensor translation units were forced to
recompile, with a core-source hash captured before/after rebuild. Final report
comparisons use `release_latest_*` so they represent one consistent binary of the
latest captured code. Pre-existing workspace edits were preserved.


#### Completion evidence

- Existing accelerated and built-in builds were already Release (`-O3 -DNDEBUG`).
  A fresh explicit Release build and a stable-source forced rebuild both passed
  audits covering 271 core and nine focused-driver translation units.
- Final comparisons use one captured core-source version and matching driver/core
  hashes, recorded in `release_latest_build_evidence.json`. All final current-core
  runs and the two-case gradient repeat loaded those identical binaries.
- Reran 82 native before/after cases and all 72 adjacent-order TensorFlow CPU
  comparisons, with five repetitions, explicit 0.1-second calibration and one
  compute thread. Also saved a same-image two-case gradient repeat.
- Final accelerated correctness: 503 passed. Built-in correctness: 501 passed,
  two expected LAPACK-only skips. Full fixture validation: 1,452 accelerated;
  1,324 built-in with 128 expected optional LAPACK skips.
- All 23 Python contracts passed. Explicit Debug CMake configuration was rejected;
  Ninja Multi-Config was verified to expose/compile-command-generate only Release.
- Runners now check effective optimization and NDEBUG, configuration, runtime tags
  and focused-core identity/provenance. Paired native preflight happens before
  warmup/timing. Legacy recipes use the verified runner and --config Release.
- Saved compiler evidence, raw observations, comparison tables, repeat samples
  and content-addressed history. First-image runs remain archived separately.

Final report: `benchmarks/results/tensorflow/RELEASE_RESULTS.md`. Current timings
show substantial host variability, so the report includes complete tables and
repeat observations instead of treating small margins as portable wins. These
are code-version comparisons under Release, not Debug-versus-Release speedups.

</details>

<a id="history-plans-backend-benchmark-coverage-audit-md"></a>
<details><summary>Historical source: plans/backend-benchmark-coverage-audit.md</summary>

Initial audit: 2026-10-23; performance extensions: 2026-10-24. Result: the agreed representative benchmark matrix is implemented and passes execution checks.

#### Scope and executable checks

The active scope is tensor linear algebra, tensor operations, autograd/autodiff,
general optimizers and neural networks. Coverage is checked using
`benchmarks/coverage.json` and `just bench-backend-audit`; every focused benchmark
run performs the same check before timing. The check also reads the current public
`nn::OptimizerKind` enum and verifies every kind in both phases and parameter shapes.
A runner contract test confirms missing registrations fail the audit.

This is an explicit operation/measurement matrix. It does not claim every public
method or all general optimizer algorithms have individual benchmarks. General
optimizer coverage is the agreed representative set of first/second-order,
derivative-free, population, constrained and multi-objective methods. RL, bindings,
and unrelated backend expansion remain deferred. Legacy sparse/data-structure and
large NN suites remain available separately.

#### Current inventory

| Area | Registered cases | Coverage |
| --- | ---: | --- |
| Tensor linear algebra | 494 | Batched/rectangular GEMM, solve/triangular solve, QR/SVD/eigh/Cholesky, inverse, least squares/pseudoinverse, determinant/logdet, nonsymmetric eigenvalues/vectors and einsum; single/multiple RHS; dtype/backend sweeps |
| NN tensor primitives, scaling and matched CPU operations | 297 | Allocation/initialization, copy/materialization, views/shape operations, broadcast/arithmetic, sum/mean/max, indexing/slicing/concat/cast; contiguous/transposed and scalar/channel/spatial paths |
| General tensor API | 53 | Float64 structural/arithmetic/broadcast/reduction operations and rectangular matrix multiplication |
| NN autograd | 45 | No-grad/record/full gradient/retained backward; chain depth, shared fan-out, broadcast accumulation and linear algebra gradients |
| Standalone autodiff | 21 | Tape record/reverse/full gradients and branching, dual gradients, Jacobians and hyperdual Hessians |
| General optimizers | 114 | Gradient descent, Adam, L-BFGS, Newton, Nelder-Mead, DE/global and ring PSO; well/ill-conditioned quadratics, Rosenbrock, population Rastrigin; SQP, NSGA-II and separate instrumented time-to-target cases |
| Tensor optimizer updates/utilities | 244 | All 15 kinds in float32/float64 first/warmed modes and matrix/vector shapes; grouping/size sweeps, slot/peak storage, clipping and gradient accumulation |
| Neural network models | 84 | MLP/CNN/single-step RNN/RNN-sequence/LSTM/GRU/transformer inference, forward/backward, full Adam step, fixed-logit loss, retained backward and optimizer-only modes |
| Neural components | 100 | Plain/fused/causal attention, ReLU/softmax/cross-entropy, layer/batch norm, embedding, convolution/pooling, simple RNN/LSTM/GRU; forward and gradient modes |
| **Total** | **1,452** | All registered focused cases executed successfully |

Additional performance-work coverage includes shared-operand GEMM, shared-matrix solve, larger legacy GEMM, float64 optimizer phases, ring PSO, Jacobians, sequence-model components and multi-size/thread submission sweeps. The legacy sparse/support suite separately contains 108 cases, including sorted recompression and CSR-to-COO. Final performance evidence and experiment dispositions are in `backend-performance-results.md`.

#### Gaps filled during the initial audit

- Single-RHS solves and full nonsymmetric eigenvectors.
- Explicit scalar/channel/spatial broadcasts, NN max reduction, and general tensor mean/max/scalar arithmetic.
- Vector parameter cases for every tensor optimizer, including Adafactor's unfactored path.
- Model loss-only, retained-backward and optimizer-only timings, with cached inputs/graphs/gradients outside timing and state reset outside updates.
- Standalone tape branching/shared variables.
- Small isolated layer norm, batch norm, embedding, convolution, pooling and gated recurrent workloads with float32/float64 coverage.
- Rastrigin for population methods and separately labeled time-to-target cases for deterministic gradient-based solvers.
- Quadratic coefficients moved into untimed fixture construction; reverse-tape throughput counts actual tape nodes.
- Focused correctness target extended to include existing sequence and additional optimizer tests.

Time-to-target uses an explicit adapter that checks objective values before gradient
evaluations and on objective evaluations. It counts the extra work and includes
exception unwinding. It stops at the first successful evaluated point, which can
be a line-search trial; it does not assert that the public solver returned that
point through its ordinary termination path. Bounded-solve cases retain their
separate timing and quality counters.

#### Validation

- Release build of `backend_bench` succeeded in the C++ tools Nix shell.
- Registration audit passed all requirements and all 15 tensor optimizer kinds.
- All **1,089** cases passed one-timed-iteration execution with their fixture checks; zero errors or optional LAPACK skips on this Accelerate/LAPACK machine.
- All **382** existing C++ correctness tests in the focused target passed.
- All **six** runner contract tests passed; new C++ formatting and focused whitespace checks passed.
- `benchmarks/results/backend_audit_20261023.json` contains **317** smoke-selected cases with three calibrated repetitions each (0.01-second minimum), plus mean/median/dispersion aggregates: **2,219** records total. No errors/skips. Stored benchmark source hashes match final code.

The full matrix checks successful execution; single-iteration times are not used
for performance conclusions. The repeated baseline was collected serially after
builds/tests completed. It is a local dirty-tree measurement, not a portable
performance gate. The earlier baseline and user-produced smoke files were preserved.
Linux CI is configured to build/audit/smoke/check; remote execution is outside this
local audit.

#### Next work

The optimization plan is `backend-performance-improvement-plan.md`. It starts with
measurement/profiling, general tensor numeric traversal and fused optimizer updates,
then addresses reductions, autograd, linear algebra dispatch, model kernels,
attention, general optimizers and supporting kernels. No core performance code was
changed by this audit.

The TensorFlow pass added 72 matched native fixtures (including GEMM and square gradients), grouped with tensor primitives above. See `tensorflow-performance-plan.md` and `benchmarks/results/tensorflow/RESULTS.md`.

The remaining-gap pass added eight public-square sum/mean VJP cases; final validation and retained/rejected experiments are in `tensorflow-remaining-gaps.md`.

</details>

<a id="history-plans-backend-performance-results-md"></a>
<details><summary>Historical source: plans/backend-performance-results.md</summary>

Goal: execute `backend-performance-improvement-plan.md` in priority order, retaining measured gains and preserving APIs/correctness. Status: completed 2026-10-24. All P0–P5 slices evaluated; measured changes retained and final validation passed.

#### Accepted results at a glance

These are local matched-case results, not a suite-wide average or portable speedup promise.

| Family | Representative confirmed result |
| --- | --- |
| General tensor traversal | Roughly 15–200x on targeted structural/arithmetic/broadcast cases |
| General tensor GEMM bridge | 144–861x on the accelerated target shapes |
| Tensor optimizers | About 1.2–14x across optimized cases; warmed 65K Adam temporary tensor peak reduced 80% |
| Shared coefficient solves | 1.6–4.7x; order-64 float64 peak tensor storage 557,056→32,768 bytes |
| Recurrent models | Larger sequence-model training about 1.04–1.13x; isolated components up to 1.36x |
| General optimizers | L-BFGS median ~1.15x, Nelder–Mead ~1.49x; DE/ring PSO/NSGA-II retain seeded quality and call counts |
| Standalone seed drivers | Gradients 1.91–2.15x, Hessians 1.32–2.59x, Jacobians 1.27–1.60x |
| Sparse kernels | Large CSR SpMV 1.7–2.9x, CSR-to-COO 1.6–2.4x, sorted recompression 1.7–4.0x |
| Pool submission | Multi-thread submission about 1.3–1.6x in repeated sweeps |

Small metadata/autograd/reduction/attention gains and all negative experiments are detailed below. Public signatures and agreed numerical/aliasing/checkpoint behavior are preserved; imported differentiable optimizer state explicitly takes the compatible general path.

#### P0: reproducible baseline and profiles

- Collected `benchmarks/results/performance/p0_baseline.json`: all 317 curated cases, five repetitions, 0.1-second minimum per repetition, no concurrent builds or tests, 1,089-case coverage audit passed. This replaces the earlier short audit baseline for optimization decisions.
- Retained original executable and shared core at `build/performance/reference/`, and source snapshots/hashes at `build/performance/source-original/`. These are local build artifacts; no staged or unrelated working-tree edits were reverted.
- Collected macOS sample traces for legacy tensor broadcast arithmetic, Adam, and autograd at `build/performance/profiles/`. Tensor traces contain repeated index access and allocation/free calls; timed Adam stacks contain chained tensor arithmetic and copies. Optimizer traces also contain untimed reset work, so total sample share is not treated as update latency. Initial traces contain process startup and support qualitative attribution only; obtain steady-state samples for final attribution.
- The benchmark binary and core library were unchanged during baseline collection. Source preparation for P1 happened without compilation. Before/candidate artifacts record the retained reference library's SHA-256 so the reference remains identifiable even though `--no-build` captures the current dirty source state.

#### P1: general tensor traversal (complete)

Prepared changes dispatch once over the stored numeric type, compute broadcast strides once, advance offsets without per-element allocation, copy contiguous spans for structural operations, and traverse reductions using flat strides. Public headers/API and binding artifacts are unchanged.

Boundary checks exposed existing empty-dimension issues: broadcasting a zero-length dimension against one must remain empty; structural operations and reductions over nonempty axes of empty tensors must return empty results. Added coordinate-based, mixed bool/float, string-copy and empty-tensor tests alongside existing tests.

Validation: 386 focused C++ correctness tests passed, all 1,089 benchmark cases passed their fixture checks, and 33 legacy tensor cases were measured with five 0.1-second repetitions. Collected A/B then B/A runs using the retained original library, recorded library hashes, and compared matching cases. Artifacts: `p1_tensor_before.json`, `p1_tensor_after.json`, `p1_tensor_after_repeat.json`, `p1_tensor_before_repeat.json` under `benchmarks/results/performance/`.

Repeated-run gains: BroadcastAdd 46.2–58.9x, Multiply 146.8–206.0x, Transpose 15.7–35.6x, Concat 35.9–41.2x, Slice 29.2–30.7x, Reshape 1.5–1.5x. Unchanged allocation/matmul controls had no regressions above both 5% and 1 microsecond in the repeated experiment. The first run's 65K zero-allocation fluctuation was below 1 microsecond absolute; repeated controls were checked explicitly.

The optimized variant is retained in `build/performance/p1-traversal/`. Matrix multiplication remains unchanged for the independent next slice.

#### P1: general tensor matrix multiplication (complete on accelerated build)

Reused the shared internal GEMM backend with packed double buffers; bool operands convert once. Dot/outer now traverse packed numeric buffers. Kept zero-size behavior and a non-narrowing fallback for dimensions beyond the shared CBLAS interface's int range. Added nontrivial floating-point reference, bool, zero-inner-dimension and independence checks. Added 128/256 square benchmark shapes before capturing the baseline.

Validation: 387 focused C++ tests passed and all 1,091 benchmark cases passed. A/B then B/A comparisons used five 0.1-second repetitions over five shapes, with source-independent retained reference libraries. Repeated speedups range from 144.4 to 860.9x. Artifacts: `p1_gemm_before/after` and `p1_gemm_before/after_repeat` JSON files under `benchmarks/results/performance/`. This reuses the existing blocked fallback when BLAS is unavailable; explicit portable-kernel validation and a backend-disabled build are scheduled with the cross-backend work rather than claiming an untested platform result.

The stage library/executable are retained at `build/performance/p1-gemm/`.

#### P1: fused SGD/Adam/AdamW (complete)

Capturing baseline for the three update kinds and every model component. Prepared fused update kernels with delayed parameter writes, dtype-aware staged arithmetic, storage-version updates, and a general fallback for strided/unsupported restored slot layouts. Added tests for fallback equivalence, checkpoint continuation, aliased gradients/overlapping parameters and saved-graph mutation detection. The candidate has been applied and built. 392 focused correctness tests and all 1,091 benchmark cases passed, including portable GEMM block-boundary checks, packed-versus-strided optimizer equivalence, checkpoint continuation, parameter/gradient aliasing, saved-graph mutation detection, and malformed restored-slot validation.

First A/B pass: SGD 2.0–4.2x, Adam 2.5–9.9x, AdamW 2.8–11.7x across tested shapes/phases; direct model optimizer-only cases 7.5–8.9x, with complete training steps 1.1–2.2x. Warmed 65K-element Adam temporary tensor peak fell from 1,310,720 to 262,144 bytes (80%). The reverse-order B/A repeat confirmed SGD 2.2–5.0x, Adam 2.8–12.0x and AdamW 3.2–12.7x improvements, with no neighboring regression above both 5% and 1 microsecond. The initial untouched inference/backward fluctuations did not repeat. Accepted after this control check. Artifacts: `p1_optimizer_before.json`, `p1_optimizer_after.json` and repeat variants.

#### Execution map

| Priority | Final disposition |
| --- | --- |
| P0 | Matched Release baselines, provenance and profiles collected |
| P1 | Typed legacy traversal/GEMM and basic fused optimizers retained |
| P2 | Remaining optimizers/utilities, metadata/reductions and autograd retained |
| P3 | Linear algebra setup/factor reuse, GEMM, recurrence and attention retained; convolution experiment rejected |
| P4 | L-BFGS/Nelder–Mead workspaces, population kernels and seed reuse retained; tape rewrite rejected |
| P5 | Sparse traversal/compression and pool completion retained; vector worker policy unchanged after profiling |

Historical entries below describe experiments as they occurred; their acceptance/rejection paragraphs give the final disposition.

#### P2: remaining optimizers and gradient utilities (complete)

Collecting a full optimizer-kind baseline plus gradient utilities and MLP consumers.
The common candidate fuses Momentum/Nesterov, Adagrad, RMSProp, Adadelta, Lion, Adamax, AMSGrad and Nadam; avoids float64 norm/square tensor buffers; and skips unnecessary gradient clones during repeated unweighted accumulation. It passed 394 focused tests and all 1,091 cases, including packed/strided and checkpoint checks for all 15 kinds. Full 166-case calibrated measurements are running. LARS/LAMB/Adafactor specialized candidates are prepared but not yet applied; their normalization, clipping and factored-state precision remain explicit requirements.


P2 common-stage measurement (five 0.1-second repetitions, 166 cases): Momentum/Nesterov/Adagrad/RMSProp/Adadelta/Lion/Adamax/AMSGrad/Nadam improved 1.1–12.4x depending on shape, grouping and phase. Existing SGD/Adam/AdamW and the not-yet-modified norm-based kinds remained near parity, with no regression above both 5% and 1 microsecond. Global norm clipping improved 1.4x with a 75% lower temporary tensor peak; four-way accumulation improved 1.3–1.5x with about one-third lower peak storage. Artifact: `p2_optimizer_after_common.json` versus `p2_optimizer_before.json`.

The specialized LARS/LAMB/Adafactor candidate has now been applied and passed the same 394 tests and all 1,091 cases. Its full calibrated comparison is running. It retains the existing dtype-specific norm/reduction calculations, mean/clipping order, factored matrix state, vector fallback and delayed parameter writes.


P2 factored Adafactor experiment: the first specialized kernel was 7% slower on the large single-matrix case because it repeated row normalization and integer division per element. Replaced that part with a row loop normalizing once per row. The tuned 10-case comparison improved every tested Adafactor case by 1.3–2.0x; vector and higher-rank packed/strided state checks now pass as well (395 total tests). The initial slower variant is not retained in the source; its artifacts remain as a negative experiment.

Added explicit float64 first/warmed vector/matrix optimizer registrations to validate both template instantiations. Registration audit now requires all 15 kinds in both dtypes; the full 1,181-case matrix passed. Final B/A runs covered 256 optimizer/utility/MLP cases and passed, including explicit float64 registrations. The runner now fingerprints the actually loaded core library (verified against current and retained-reference paths), avoiding ambiguity from --no-build dirty source snapshots.

P2 NN metadata/reduction candidate is prepared but not yet applied: remove detached handles from pure map kernels that already run under NoGradGuard and allocate independent outputs; bypass view construction for contiguous suffix reductions. Existing public detach semantics stay unchanged. Measurement and correctness checks remain pending.


Measurement provenance follow-up: benchmark JSON now records `context.core_library_path` (via the loaded symbol's image on POSIX) and `backend_benchmark_metadata.loaded_core_sha256`. Functional checks proved that retained-library runs resolve to the retained path and matching hash, while normal runs resolve to the rebuilt core. Source fingerprints alone were insufficient to disambiguate prepared source with --no-build, so future comparisons retain binary provenance directly.


P2 tensor/autograd preparation: source inspection confirms elementwise `map_*` kernels only read storage and allocate new outputs under an existing NoGradGuard, making their detached TensorImpl copies redundant. Reduction packing similarly constructs extra metadata for packed suffix axes. A separate bookkeeping experiment will replace the backward discovery ownership hash map plus raw-pointer postorder vector with one owning postorder vector; traversal order, release rules and capture routing stay unchanged. Both experiments are prepared as separate steps, not yet applied or accepted.


P2 final acceptance: retained the tuned norm-based kernels and all common kernels after reverse-order confirmation across 256 optimizer/utility/MLP cases (including both dtypes), five 0.1-second repetitions each. No neighboring regression exceeded both 5% and 1 microsecond. Conservative repeated improvements for the newly optimized kinds were about 1.2–14x float32 and 1.4–4x float64 depending on size/phase; initial control timing drift was checked against earlier A/B data rather than attributed to unchanged code. All 395 correctness tests and the 1,181-case matrix passed. Artifacts: `p2_optimizer_before/after_repeat.json`, with verified loaded-core paths/hashes; retained stage at `build/performance/p2-final-optimizers/`.

#### P2: elementwise metadata and reduction setup (complete)

Capturing a matched baseline before removing redundant detached metadata inside guarded pure map kernels and unnecessary views for contiguous suffix reductions. No public detach semantics change. The following autograd ownership-vector experiment remains separate and pending.

P2 metadata accepted after A/B then B/A over 192 cases, five 0.1-second repetitions. The initial run had broad machine drift (unchanged controls improved too); use the reverse-order repeat for conservative estimates. Controls returned near parity, with no regression above both 5% and 1 microsecond. Depth-16 tiny chains improved 1.13–1.25x, layernorm 1.13–1.22x, recurrent layer components 1.09–1.18x, model forward/backward 1.05–1.12x and selected small multiply up to 1.47x. Changes remove detached TensorImpl copies from pure guarded map kernels and avoid views for packed trailing-axis reductions, preserving numerical iteration order. All 395 correctness tests and 1,181 benchmark cases passed. Artifacts: `p2_tensor_metadata_{before,after}{,_repeat}.json`; accepted stage `build/performance/p2-metadata/`.

#### P2: autograd ownership bookkeeping (complete)

Replacing duplicate DFS ownership hash entries with shared ownership directly in the postorder vector. Capturing the matched baseline before applying this independent change.

P2 autograd accepted: owning postorder vector replaces duplicate node-ownership hash storage while preserving DFS and execution order. A/B covered 61 chain/fan-out/linalg/model cases; reverse B/A repeated 51 affected/adjacent cases, five 0.1-second repetitions. Repeated chains improved about 1–7%, MLP retained backward 3–7%, RNN training about 2%; transformer was near parity. This small simplification removes one hash node allocation per discovered autograd node and a duplicate pointer-key table. No reverse-repeat regression exceeded both 5% and 1 microsecond; the first no-grad control fluctuation disappeared. All 395 checks and 1,181 smoke cases passed, including higher derivatives, saved mutation checks and graph-storage accounting. Artifacts: `p2_autograd_{before,after}{,_repeat}.json`; retained stage `build/performance/p2-autograd/`.

#### P3: linear algebra setup/workspaces (complete)

P2 complete. Profiling the accepted stage before choosing scoped linear algebra changes; built-in and LAPACK controls remain in the measurement set.

P3 read-only setup accepted after 220-case A/B and 68-case reverse B/A, five 0.1-second repetitions. No repeated regression exceeded both 5% and 1 microsecond; the first inverse/eigh fluctuations disappeared. Repeated small solve RHS cases improved 1.17–1.19x, triangular solves 1.07–1.59x, while decomposition controls were near parity. Raw float64 packing no longer detaches read-only inputs; matching batch shapes bypass broadcast views. All 395 checks and 1,181 benchmark fixtures passed. Steady-state solve sample: `build/performance/profiles/solve-steady.txt`; artifacts `p3_linalg_{before,after}{,_repeat}.json`. Retained stage `build/performance/p3-linalg-setup/`.

Adding shared-operand matmul and shared-coefficient solve cases before evaluating copy/factor reuse. These extend coverage rather than substituting for the existing independent-batch cases. Shared solve correctness tests compare both backends/dtypes/adjoints against separately factored batches and check first/second derivatives.

P3 shared factorization accepted after A/B then B/A over 44 cases. Repeated shared solves improved 1.64–3.43x at order 16 and 2.01–4.66x at order 64 across both dtypes, adjoints and backend selections; independent solve controls remained near parity with no material regressions. The singleton coefficient matrix is packed/factored once per call; workers reuse private RHS conversion scratch, with immutable factors shared during solves. Float64 order-64 auto peak tensor storage fell from 557,056 to 32,768 bytes; counters exclude STL scratch. All 397 checks, including matching independent factors and first/second derivatives, and all 1,237 benchmark fixtures passed. Artifacts: `p3_shared_solve_{before,after}{,_repeat}.json`. Existing backend crossover defaults remain unchanged: these measurements support removing setup/repeated work, not portable vendor threshold retuning.

Portable validation: separate Release build with `DATAMUNGE_USE_BLAS=OFF`, `DATAMUNGE_USE_LAPACK=OFF`, `DATAMUNGE_ENABLE_ARROW=OFF` passed 397 checks (two LAPACK-only checks skipped) and the full 1,237-case fixture run (128 expected unavailable-backend skips). This also validates the P1 general-tensor GEMM bridge and all optimizer/autograd changes against the built-in backend. Artifact: `p3_portable_smoke.json`.

P3 shared GEMM accepted: packed singleton operands are reused directly, with zero batch stride, instead of materializing repeated copies; redundant detached metadata is removed from the guarded floating-point input path. Both-dtype/transposition/shared-side reference tests pass. A/B and reverse B/A over 69 cases confirmed approximately 1.46–1.48x median gains among batch-16 shared cases, up to 2.40x, with ordinary products/einsum near parity. Reverse-run machine drift also slowed an untouched fan-out control; a separate eight-case, five-repetition 0.2-second A/B cleared those regressions (fan-out within 1%, model controls equal/faster). No vendor crossover changes. All 399 checks and 1,237 benchmark cases passed. Artifacts: `p3_matmul_{before,after}{,_repeat}.json`, `p3_matmul_controls_{before,after}.json`; retained stage `build/performance/p3-matmul/`.

#### P3: model kernels (complete)

Evaluating one input projection per recurrent sequence and one parameter lookup per call. Existing split/unstack backward joins timestep gradients once, avoiding per-timestep full scatter buffers. Compare against single-step cells with identical parameter creation order and seeded state, including all parameter and input gradients in both dtypes.

P3 recurrent accepted after tuning and reverse-order confirmation over 72 layer/model/control cases. Parameters are fetched once per sequence; larger batches and float64 inputs share a time-major input projection, while batch-1 float32 retains per-step GEMM. The first all-batched candidate regressed small GRU inference and selected retained backward cases, so it was rejected (`p3_recurrent_extended_after.json`, source retained in `build/performance/p3-recurrent-first/`). Time-major projected timestep views and the small-cell path cleared those regressions. Conservative matched gains across repeats: simple RNN layer up to 1.36x, LSTM up to 1.22x, GRU up to 1.28x; larger sequence-model training about 1.04–1.13x, depending on model/batch. Some tiny model components drifted between runs, so these are not blanket speedup promises. No repeated >5% and >1-microsecond regressions. All 400 checks and 1,281 benchmark cases passed. Added explicit RNNSequence/LSTM/GRU network component modes and SimpleRNN layer cases because the original RNN benchmark intentionally calls single-step cells and cannot validate sequence projection. Artifacts: `p3_recurrent_extended_before.json`, `p3_recurrent_tuned.json`, `p3_recurrent_{before,after}_repeat.json`; retained stage `build/performance/p3-recurrent/`.

Convolution/pooling/batch-normalization: existing im2col buffers already span all chunks within a call, so avoid adding an unbounded scratch cache. Evaluate eliminating read-only detached metadata in the guarded raw kernels independently of recurrent changes.

P3 convolution metadata experiment rejected: 36-case A/B suggested small 1–4% gains, but reverse B/A did not reproduce them, with unchanged loss/optimizer controls also drifting. The extra removal of detached input metadata was restored from the exact pre-experiment source snapshot; do not attribute its tentative gains to the final implementation. Existing per-call im2col reuse and specialized pointwise/depthwise kernels remain the baseline. Artifacts: `p3_conv_{before,after}{,_repeat}.json`. P2 elementwise/reduction gains and accepted GEMM/model changes still benefit these consumers; no unmeasured conv fusion/cache added.

#### P3: attention scratch (complete)

Short sequences allocate full 64x128 score scratch even when the active tile is 16x16. Evaluate allocating only the maximum tile dimensions this call can use, in both forward and backward; retain tile sizes, masks, arithmetic order and Auto dispatch. For 16-token float32 inputs, score scratch per participant falls from 32 KiB to 1 KiB in forward, and from 64 KiB to 2 KiB in backward (STL storage is not included in tensor memory counters).

P3 attention accepted after 36-case A/B then B/A: fused forward gains repeated at about 1.06–1.26x and full forward/backward at about 1.02–1.16x depending on sequence length/mask; plain attention and transformer model controls were near parity with no material regression. Score/accumulator scratch is bounded by actual call tile dimensions instead of always allocating the maximum tile. The 16-token float32 forward score scratch falls 32 KiB→1 KiB and backward score scratch 64 KiB→2 KiB per participant; tensor-only counters intentionally exclude this STL storage. All 400 correctness checks and 1,281 benchmark fixtures passed, including masks, unequal lengths, higher derivatives and plain/fused parity. Artifacts: `p3_attention_{before,after}{,_repeat}.json`. P3 complete: retained measured setup/factor reuse, shared GEMM, recurrent and attention changes; rejected unconfirmed convolution metadata experiment; no unsupported vendor-threshold or tile retuning.

#### P4: general optimization workspaces (complete)

Added ring-topology PSO cases before evaluating its attractor-copy overhead. Untimed coordinate and Pareto signatures supplement existing objective/call/success counters, making seeded result changes detectable. Profile L-BFGS and NSGA-II, then reuse L-BFGS iteration scratch and evicted history buffers without changing evaluation or line-search order. SQP remains a correctness/performance control: its current allocations come from virtual user callbacks, so avoiding them would require an API change or fewer callback calls.

P4 L-BFGS accepted after a compact workspace revision. Iteration vectors are reused in one call-local array, and evicted history pair buffers are recycled. Initial separate workspace objects produced a consistent ~5% neighboring Nelder–Mead slowdown despite an identical normalized Nelder–Mead instruction stream; that variant was rejected/retained as negative evidence at `build/performance/p4-lbfgs-first/`. Compact array storage cleared the neighbor regression in A/B then B/A. Repeated bounded L-BFGS medians improved about 1.15–1.18x across its six cases (up to 1.31–1.33x); time-to-target improved 1.04–1.13x. Tiny one-step solves can be slightly slower below 1 microsecond absolute. All objective/gradient/Hessian counts and final result signatures match in the retained variant; the first variant's differences were quantified at at most 1.8e-15 before rejection. All 400 checks and 1,305 fixtures passed. Artifacts: `p4_lbfgs_tuned_{before,after}{,_repeat}.json`, original/controls/coordinate evidence preserved. Profiles: `lbfgs-steady.txt`, `nsga-steady.txt`; NSGA-II samples show duplicate dominance comparisons dominate its sort. Retained stage `build/performance/p4-lbfgs/`.

Evaluate Nelder–Mead centroid/trial buffer reuse as a separate workspace change. Newton already solves directly in callback-returned Hessian/gradient storage; SQP similarly has no redundant core copies to remove without changing callback APIs/call counts.

P4 Nelder–Mead workspace accepted after 13-case A/B then B/A: centroid/trial vectors persist within the solve, and accepted trial buffers swap with replaced simplex storage. Exact coordinate signatures, objective quality and evaluation counts match all six cases. Repeated speedups 1.05–2.20x (median about 1.49x); L-BFGS/SQP controls had no material regression. All 400 checks and 1,305 fixtures passed. Artifacts: `p4_nelder_{before,after}{,_repeat}.json`; retained combined workspace stage `build/performance/p4-workspaces/`.

#### P4: population and multi-objective kernels (complete)

Profile confirms dominance checks dominate NSGA-II sorting. Evaluate each unordered pair once while preserving ascending dominated-list order, move selected generation storage instead of copying, and reuse objective-order scratch. DE reuses mutant/trial buffers and uses a fixed five-index selection buffer without changing RNG draws; PSO ring reads the selected personal-best buffer directly, whose contents remain unchanged until coordinate updates finish. Every seed/objective/budget stays fixed. Results now include executable SHA-256 as well as loaded-core provenance where supported, needed for later header/template optimizations.

P4 population accepted after 88-case A/B then B/A, all seeds/objectives: DE about 1.34–3.0x conservatively across runs, ring PSO 1.22–2.10x, NSGA-II 1.85–2.03x. Exact coordinate/Pareto signatures, final quality, success and objective/gradient/Hessian counts match. Initial neighboring control drift cleared in reverse order; no >5% and >1-microsecond repeated regression. All 400 checks and 1,305 benchmark fixtures passed. Artifacts: `p4_population_{before,after}{,_repeat}.json`; retained stage `build/performance/p4-population/`. General Newton/SQP and non-ring PSO are unchanged controls, so their incidental timing movement is not counted as a gain.

#### P4: standalone autodiff (complete)

Added Jacobian cases before changing generic seed drivers; seed storage will be allocated once per call and fully reset before each evaluation, including callable mutations. Tape backward retains every node/parent update, including zero-adjoint nonfinite propagation; only its known zero/one/two-parent loop form changes. Added edge checks for mutable seed inputs and duplicate/nonfinite parents. Corrected legacy general-tensor slice/concat throughput counters to use actual output size; latency fixtures are unchanged. Freeze the baseline executable as well as the library because template code is compiled into the driver.

P4 standalone seed reuse accepted after 21-case A/B and reverse B/A with the frozen reference executable/library. Baseline executable hashes match exactly across reference runs, so template changes are not accidentally measured against a newly compiled reference. Repeated gradients improved 1.91–2.15x, hyperdual Hessians 1.32–2.59x, Jacobians 1.27–1.60x. Each call allocates one seed buffer and still resets all value/derivative components for every evaluation. All 402 checks and 1,308 fixtures passed. The separate tape two-parent-loop experiment had no clear throughput gain and was restored to the original implementation; its edge-case tests remain. Tiny unchanged tape controls have sub-microsecond code-layout/allocation drift, which is not counted as a gain. Artifacts: `p4_autodiff_{before,after}{,_repeat}.json`; frozen reference at `build/performance/p4-autodiff-reference/`, accepted stage `build/performance/p4-final/`. P4 complete.

#### P5: sparse kernels and supporting structures (complete)

Add sorted recompression and CSR→COO cases before evaluating sorted in-place compression and local CSR row accumulation. Preserve stable duplicate summation, zero removal after scaling, and aliased vector/matrix input behavior. Include existing sparse-format/COO and ThreadSafeVector correctness tests in backend checks. Sparse timing uses the corrected deterministic fixture generator, not historical pre-correction matrices. Supporting-vector lock/worker policy remains unchanged pending evidence; benchmark labels continue to distinguish independent containers from shared contention.

P5 sparse accepted after tuning: CSR assigns every ordinary output row once, retaining the old zero-and-accumulate path for aliased or differently sized buffers. Sorted COO input merges/compacts in place with stable duplicate order; a cheap monotonic sample rejects common concatenated runs before a full order scan. The sorted helper was moved out of `compress()` to preserve the original unsorted comparator specialization: earlier inline variants showed random-sort regressions and are retained as negative evidence (`p5-sparse-first/second` sources). Final 20-case helper B/A cleared random-sort/expression regressions; independent CSR repeat confirmed large random SpMV gains ~1.7–2.9x, CSR→COO ~1.6–2.4x, sorted recompression ~1.7–4.0x. Tiny CSR timings drift below 1 microsecond; no blanket small-case promise. All 490 checks, 108 legacy fixtures and 1,308 backend fixtures passed during this slice. Artifacts: `p5_sparse_{before,after}{,_repeat}.json`, `p5_sparse_tuned.json`, `p5_sparse_helper_{before,after}.json`; retained stage `build/performance/p5-sparse-final/`.

Supporting-vector profile `vector-steady.txt` shows thread create/join dominates the tiny apply fixture. Leave its requested worker/locking policy unchanged rather than infer a universal grain rule from a trivial functor. Push/snapshot operations already perform the required protected append/copy.

#### P5: thread-pool completion (complete)

Add 1/2/4-thread ReLU/reduction sweeps at 1K/32K/64K/1M elements and direct submission cases before changing completion synchronization. Add concurrent submitter coverage alongside existing nested/exception and 20,000-job stress checks. A final optimizer compatibility edge check also verifies imported differentiable checkpoint slots follow the original out-of-place history behavior.


P1/P2 compatibility follow-up: a new edge test showed that imported moment buffers carrying gradient history retained that history under in-place fused updates, unlike the original formulas. The test failed against the fused stage and passed against the original retained core. Such slots now use the existing out-of-place fallback; ordinary non-differentiable checkpoint slots keep the fused path. All 14 stateful optimizer kinds are covered. A 34-case comparison of representative float32/float64 first/warmed/vector/matrix updates and model controls found no material normal-path regression. All 492 checks passed. Artifacts: `p5_slot_guard_{before,after}.json`; baseline-failure/original-pass logs retained locally.

P5 pool completion accepted: each participant publishes its completed chunk count once, with a relaxed atomic claim counter and the existing mutex/condition completion barrier. Shared job ownership and body lifetime rules remain intact. All 492 checks (including nested exceptions, 20,000 small jobs and concurrent submitters) and all 1,338 fixtures passed. A/B then B/A over 33 thread/size cases confirmed roughly 1.29–1.61x faster multi-thread direct submission; a longer seven-case control A/B cleared an initial serial ReLU timing fluctuation and confirmed about 1.36–1.73x submission improvements. Large kernel sweeps were mostly near parity or modestly faster; no universal grain/worker-count retuning was applied. Artifacts: `p5_thread_{before,after}{,_repeat}.json`, `p5_thread_controls_{before,after}.json`; profile `parallel-steady.txt`.

#### Final validation

Completed on macOS arm64 with Release builds:

- Accelerate: all 492 backend checks passed; the complete 1,338-case focused matrix passed, and the final 451-case smoke selection passed.
- Built-in-only (`DATAMUNGE_USE_BLAS=OFF`, `DATAMUNGE_USE_LAPACK=OFF`, Arrow disabled): 490 checks passed, with the two LAPACK-only checks skipped. All 1,338 focused registrations executed, with 128 expected unavailable-LAPACK cases skipped.
- Legacy sparse/support matrix: all 108 cases passed in both builds.
- Six runner contract tests passed; official comparisons accept the matched Release/thread contexts; `git diff --check` passed.
- Manifest inventory accounts for every focused registration across nine areas and discovers all 15 optimizer kinds in both dtypes/phases and vector/matrix shapes.

Final functional artifacts: `final_backend_matrix.json`, `final_backend_smoke.json`, `final_legacy_matrix.json`, `final_portable_backend_matrix.json`, `final_portable_legacy_matrix.json` under `benchmarks/results/performance/`. These one-iteration files validate fixtures; calibrated before/after files above support timing claims. Final executable/core/cache/source hashes are retained locally in `build/performance/final/`.

Dedicated-runner alerts and platform-specific crossover tuning remain conditional operational follow-up, rather than unmeasured code changes. This completes the prioritized backend performance evaluation/implementation plan; it does not assert that every possible backend optimization or production shape has been exhausted.

</details>

<a id="history-plans-backend-performance-improvement-plan-md"></a>
<details><summary>Historical source: plans/backend-performance-improvement-plan.md</summary>

Date: 2026-10-23; execution completed 2026-10-24. Status: P0–P5 evaluated and completed; measured changes retained, unsupported experiments rejected. See `backend-performance-results.md` for measurements and remaining priorities.

#### Scope and benchmark readiness

Optimize the canonical C++ implementation behind tensor linear algebra, both tensor APIs, autograd/autodiff, general optimizers and neural networks. Preserve public API behavior and deterministic regeneration of bindings. Existing sparse/data-structure benchmarks are a secondary workstream. RL and binding overhead remain deferred.

The coverage checklist is now executable: `benchmarks/coverage.json` describes the agreed representative operations; `python3 scripts/benchmark_backend.py audit` verifies registered cases and discovers current tensor optimizer kinds from the public enum. Every kind must have first-step and warmed cases for both vectors and matrices. Every focused run performs this audit automatically. This is complete coverage of the agreed benchmark matrix, not a claim that every public method, general optimization algorithm, architecture or possible shape has a benchmark.

The coverage audit added missing single-RHS solves, general eigenvectors, scalar/channel/spatial broadcasting, max/mean reductions, vector optimizer parameters, standalone tape branching, explicit model loss/backward/optimizer components, normalization/embedding/conv/pooling/gated recurrence, multimodal population objectives and instrumented time-to-target cases. The coverage audit report and benchmark README document the exact current inventory.

#### Evidence and limits

The following are local median wall times from the repeated audit baseline, in microseconds. Each case has three calibrated repetitions with a 0.01-second minimum. They identify candidates for profiling; they do not establish portable speedup promises. The original baseline was collected under different machine load and must not be treated as a before/after optimization comparison: this audit changed benchmark coverage and fixtures, not backend performance.

| Workload | Median wall time (microseconds) | Interpretation |
| --- | --- | --- |
| Float64 broadcast add, 1,024 elements | general tensor 32.232 / NN tensor 0.450 | Profile numeric indexing and broadcast bookkeeping in the general tensor implementation; both results contain the same logical values. |
| Float64 multiply, 1,024 elements | general tensor 33.458 / NN tensor 0.313 | A same-shape numeric fast path is a high-priority candidate; these are distinct APIs, not interchangeable implementations. |
| Batch-1 MLP | inference 1.634 / optimizer only 20.306 / training step 30.916 | Fused optimizer updates may benefit even tiny models. Independently timed components are not additive accounting. |
| Batch-1 transformer | optimizer only 73.745 / training step 172.103 | Validate optimizer gains in a second model and profile projection/norm/backward work separately. |
| 32-element chain, depth 16 | no-grad 8.667 / record 12.471 / retained backward 16.581 | Inspect node/metadata allocation and routing overhead with a sampling/allocation profile. |
| Causal attention, 16 tokens | plain forward 12.782 / fused forward 8.767 | Investigate short-sequence dispatch only after repeating a broader shape/mask sweep. |
| ReLU, 1,048,576 float32 elements | 1 thread 66.560 / 2 threads 41.638 / 4 threads 34.221 | Use a measured grain/thread policy; extra workers do not imply proportional gains. |

The audit baseline contains 317 repeated cases; median within-case coefficient of variation is 0.76%. Three short repetitions do not quantify run-to-run thermal/load variation. Baseline: `../benchmarks/results/backend_audit_20261023.json`.

Source inspection supports these hypotheses:

- `src/datamunge/linalg/tensor.cpp`: generic elementwise operations build coordinate vectors and broadcast operand indices per element; matrix multiplication calls checked, multidimensional `at()` inside its innermost loop. These are concrete candidates for numeric fast paths.
- `src/datamunge/nn/optim.cpp`: update rules compose many out-of-place tensor operations even though updates run under `NoGradGuard`. Many small parameters also repeat slot lookup and dispatch. Fusion may reduce storage traffic and dispatch overhead.
- `src/datamunge/nn/detail.hpp`: same-shape contiguous and scalar-broadcast fast paths already exist. Extend remaining channel/spatial/layout paths only when a profile demonstrates a gap; do not spend a change recreating existing paths.
- `src/datamunge/nn/reduce.cpp`: reductions already have packed-row and selected leading-axis column paths. Other axis/layout combinations can materialize contiguous storage. Measure layout-copy cost before changing the reduction arithmetic.
- `src/datamunge/nn/autograd.cpp`: traversal and gradient routing use dynamic containers, tensor metadata and reference-counted nodes. Tiny chain cases can expose bookkeeping; profile before choosing a replacement representation.
- `src/datamunge/nn/gemm.cpp` already has a blocked fallback kernel and BLAS dispatch. `linalg.cpp` already has per-operation LAPACK crossovers. LSTM/GRU already use packed gate parameters. Improve these implementations based on measurements instead of proposing those existing features as new optimizations.

Exact profile attribution remains to be collected. Tensor memory counters exclude node metadata and allocator overhead; use an allocation profiler alongside sampling when those costs matter.

#### Measurement and acceptance contract

1. Establish a fresh baseline from the exact source revision/configuration being optimized. Do not mix the older 932-case baseline with newly named optimizer shapes or changed objective fixtures. Record dirty state/source hashes, compiler flags, BLAS/LAPACK and thread settings; keep input fixtures identical between baseline and candidate.
2. Run without concurrent builds/tests, with the same power/thermal conditions. Warm up, use at least five repetitions and a 0.1-second minimum for the target families, and repeat the entire baseline/candidate experiment in alternating order. Increase duration when variance remains high. Keep first-step allocation and warmed updates separate.
3. Profile representative small/large, contiguous/strided, float32/float64 cases using local sampling/allocation tools (macOS Time Profiler/sample; Linux perf when available). Identify inclusive/exclusive time and allocation/copy volume. Do not infer a hotspot solely from a large absolute duration on a larger workload.
4. Land one focused change at a time. Run `just bench-backend-check`, the complete one-iteration matrix, then calibrated comparisons for targeted and adjacent families. Check the built-in and available BLAS/LAPACK paths separately. Local macOS checks do not replace Linux CI.
5. Report per-family medians and dispersion, absolute time saved, worst affected case and memory changes. Avoid a single unweighted mean across 1,000+ shape variants. If reporting a summary, group by operation first so large registration grids do not dominate it.
6. An initial optimization target is a reproducible 10% improvement in its intended workload or a substantial demonstrated storage reduction. This is a planning target, not an automatic universal gate. Accept smaller improvements only when they exceed measured noise and have clear practical value. Investigate adjacent-case regressions above both 5% and 1 microsecond; use tighter absolute limits for tiny kernels once stable runner noise is known.
7. Preserve numerical tolerances, dtype promotion, broadcasting, aliasing/copy semantics, empty/invalid input behavior, gradient accumulation, graph release/retention, higher-order differentiation, optimizer checkpoint state and seeded algorithm behavior. Do not obtain a faster result by reducing iteration budgets or weakening accuracy.
8. For general optimizers, compare speed within the same algorithm and objective/budget first. Across algorithms also report objective/gradient/Hessian calls, success, quality and seed dispersion. Time-to-target includes the benchmark adapter's target checks and unwinding; count that work and keep it unchanged. Do not equate a faster unsuccessful bounded solve with an improvement.

#### Proposed implementation slices

Execute in this order unless profiling changes the evidence. Each slice should contain the relevant correctness checks, before/after artifacts and an explanation of the observed cause.

| Priority | Slice | Implementation candidates | Benchmarks and success evidence | Main constraints |
| --- | --- | --- | --- | --- |
| P0 | Stable measurement set and profiles | Freeze configurations and collect sampling/allocation profiles | Calibrated baseline across every family; profiles for general tensor arithmetic, Adam, small autograd and model components | No concurrent workload; distinguish fixture/reset overhead and backend work |
| P1 | General tensor numeric traversal | Replace repeated coordinate construction with typed contiguous loops; hoist dtype/shape validation; increment broadcast coordinates without per-element allocation; copy contiguous spans for transpose/slice/concat where valid | `BM_LinalgTensor`, `BM_LinalgTensorExtra`, `BM_LinalgTensorSum`; seek improvement on 1K and 64K without degrading 1M | Retain copy semantics and numeric/bool/string behavior |
| P1 | General tensor matrix multiplication | Use checked-once raw numeric buffers, cache-friendly loop order/blocking, then evaluate reuse of existing GEMM backend where compatible | `BM_LinalgTensorMatmul`; check rectangular products and float64 references; add larger target shapes before changing dispatch | Avoid a second unrelated GEMM implementation; preserve bool numeric conversion and standalone packaging |
| P1 | Fused SGD/Adam/AdamW updates | Replace tensor-expression chains with dtype-dispatched update loops; mutate slot storage safely; hoist per-step scalar math/validation; optimize many-small-parameter dispatch | `BM_TensorOptimizer` first/warmed, vector/matrix, one/16 tensors; `BM_Network/*_OptimizerOnly` and training steps; demonstrate lower temporary storage as well as time | Keep epsilon placement, bias correction, weight decay, storage version updates, aliasing and checkpoint names/values identical |
| P2 | Remaining tensor optimizers and gradient utilities | Extend fusion by shared update structure; specialize factored Adafactor separately; combine gradient norm/reduction passes only when equivalent | All 15 optimizer kinds plus clipping and accumulation; report factored-state memory and grouped-parameter overhead | Preserve LAMB/LARS trust ratios, Lion old-moment ordering, Adafactor vector fallback and accumulation weighting |
| P2 | NN broadcast, layout and reductions | Profile channel/spatial cases; collapse compatible axes, avoid needless metadata/packing and redundant contiguous copies; vectorize inner loops and tune reduction accumulation | `BM_TensorBroadcast`, `BM_TensorOps`, `BM_TensorMax`, `BM_LayerComponent/*Norm`, `BM_GradientUtilities`; compare layout/dtype sweeps and memory peaks | Existing contiguous/scalar fast paths are baselines; preserve offsets, promotion and summation/NaN semantics |
| P2 | Autograd scheduling and allocation | Reserve per-invocation containers, reduce repeated metadata construction and gradient routing allocations; investigate cheaper topological ordering only after profiles | Chain depth/fan-out, retained/full backward, solve/QR/SVD gradients, direct model retained backward; allocation profiles and post-gradient live-storage checks | No global mutable scratch cache; preserve saved-tensor mutation checks, release timing, unused inputs, checkpointing and second derivatives |
| P3 | Linear algebra dispatch and workspaces | Profile layout conversion/workspace queries; reuse scratch within safe call/worker lifetimes; tune per-operation crossovers for shape/dtype/batch and compare built-in vs LAPACK | `BM_TensorLinalg`, solve RHS variants, rectangular QR/SVD, general eig, differentiation cases; residuals/orthogonality/gradients remain unchanged | Do not copy macOS crossover values blindly to OpenBLAS; preserve internal float64 decomposition accuracy and eigenvector/sign conventions |
| P3 | Tiny/batched GEMM and einsum | Measure vendor call overhead, transposition/packing and per-batch dispatch; evaluate a small-kernel crossover and avoid avoidable packing in repeated compatible shapes | `BM_TensorMatmul`, `BM_TensorEinsum`, dense layers and attention; built-in and accelerated builds, rectangular and batched inputs | Existing blocked fallback is the starting point; avoid nested BLAS/pool oversubscription and architecture-specific defaults without evidence |
| P3 | Dense/conv/norm and recurrent models | Fuse measured bias/activation sequences, reuse eligible conv packing buffers, and reduce layer-name/parameter lookup overhead; batch input projections across timesteps where valid | Isolated layer components, CNN/MLP/RNN/LSTM/GRU and model inference/backward/training modes; require an end-to-end gain alongside microbenchmarks | Packed gates already exist; preserve recurrent order, parameter sharing, padding, gradients and training/inference distinctions |
| P3 | Attention | Tune tile sizes and Auto dispatch for short/long sequences; optimize causal boundary tiles and scratch reuse after profiling | Plain/fused, causal/unmasked, forward/backward and memory scaling; transformer modes confirm application benefit | Maintain masks, output/gradient parity and bounded memory; small-case wins alone do not justify globally changing Auto thresholds |
| P4 | General optimization workspaces | Reuse L-BFGS vectors/history storage, line-search temporaries and dense solve buffers; avoid redundant objective/gradient work when semantics permit | Bounded solves and time-to-target; all objective families and seeds; count calls and compare achieved objective/error | Precomputed benchmark coefficients are now untimed; do not optimize fixture math instead of backend code or change accepted-step/termination behavior |
| P4 | Population/constrained/multi-objective kernels | Profile candidate cloning, selection/sorting, distance checks and small system solves; reuse generation-local buffers and cache already computed evaluations | DE/PSO including multimodal cases, SQP and NSGA-II; evaluations, quality/success and per-seed results | Preserve bounds, RNG draw/order for deterministic comparisons, Pareto selection/ties and constraint tolerances |
| P4 | Standalone autodiff | Reuse per-call derivative seed buffers where possible; reduce tape node allocation/copy overhead without changing user-facing API; inspect reverse workspace costs | Dual gradients, hyperdual Hessians, tape record/reverse/full and branching; actual tape node throughput and derivative references | Separate algorithmic O(n)/O(n²) forward-pass counts from avoidable overhead; no differentiation-mode substitution hidden in a timing comparison |
| P5 | Sparse and supporting data structures | After primary backend work: profile SpMV memory traffic, compression/conversion sorting and allocation, then independent-container vector operations | Existing `linalg_bench`; regenerate sparse baselines after the fixture correction; same nnz/distribution and truthful thread labels | No claims of shared-container contention from independent containers; compare structured and random formats on equivalent appropriate workloads |
| P5 | Threading and scaling | Tune grain sizes and submission overhead using multiple element counts; reduce per-chunk synchronization only if profiles show it matters | Thread scaling, large reductions/conv/GEMM and small serial cases; 1/2/4 library threads plus explicit BLAS controls | More threads can be slower; preserve nested-call behavior, exception propagation and race-free completion |

#### Rollout and reporting

The first optimization change should be either typed traversal in `linalg::Tensor` or fused SGD/Adam updates, depending on the profiles. Both have narrow implementations, direct checks and visible consumers. Do not start by retuning all linear algebra thresholds or rewriting the autograd engine.

For every change, retain a small report with source revision, exact command, targeted case list, numerical checks, before/after timing distribution, allocation/storage changes and neighboring regressions. Keep negative experiments documented but out of the implementation if they add complexity without a measured win.

Continue ordinary CI build/list/audit/smoke checks. Introduce performance alerts only on a controlled runner after multiple baseline runs establish noise. Add absolute and relative tolerances plus rerun confirmation; shared hosted CI remains a functional smoke check. If a proposed optimization needs a new case, add it to the coverage manifest and smoke selection before evaluating the candidate.

Suggested commands:

```sh
just bench-backend-audit
just bench-backend-check
just bench-backend -- --benchmark_min_time=1x
just bench-backend -- --benchmark_filter='BM_LinalgTensor' --benchmark_min_time=0.1s --benchmark_repetitions=5
just bench-backend -- --benchmark_filter='BM_TensorOptimizer|BM_Network/MLP_' --benchmark_min_time=0.1s --benchmark_repetitions=5
just benchmark-compare baseline.json candidate.json
```

Save separate baseline/candidate files with `--output` before the `--` separator.

</details>

<a id="history-plans-backend-copy-cache-pass-md"></a>
<details><summary>Historical source: plans/backend-copy-cache-pass.md</summary>

Date: 2026-10-24. Status: complete.

1. Persist benchmark observations as immutable raw files plus versionable summaries, with compatible profile IDs and smoke/calibrated separation. Preserve existing results and export trends through `just`.
2. Avoid partial-batch matmul expansion: pack each original operand once and map output batches to original matrices. Validate strided/transposed inputs, broadcasting, first/second derivatives and tensor peaks against explicit expansion.
3. Evaluate cache-blocked double matrix transpose and direct identity-permutation copies. Keep generic dtype/rank paths and deep-copy semantics.
4. Keep only repeated measured wins; compare reverse order, check neighbors and both accelerated/built-in builds. Save raw results and compact summaries and report gains as verified.

Baseline: `benchmarks/results/performance/cache_pass_before.json`; raw/compact persistent history: `benchmarks/history/`.


#### Retained improvements and measurements

- Partial-batch Float32/Float64 matmul packs each original input once instead of
  expanding broadcast batches. All dtype/transpose combinations are covered;
  strided inputs and first/second derivatives are checked. Final 16-case speedups:
  **1.57–3.60x**. Larger Float64 peak tracked tensor storage:
  **1,441,792 → 262,144 bytes (81.8% lower)**.
- Larger Double matrix transpose uses 8-row × 32-column tiles with contiguous
  output writes, gated to dimensions >=64 and >=32,768 elements. Final eligible
  cases improved **1.53–2.75x**. Direct identity copies improved **3.37–4.74x**;
  returned tensors remain independent copies. Odd tile edges are tested.
- Rejected the initial input-contiguous 32×32 transpose variant because tall
  matrices regressed. Also rejected applying blocked traversal to thin/small
  matrices and placing it inside the generic visitor. Original generic paths
  remain in use outside the eligible matrix shapes.

Final matched comparison: 79 cases, five repetitions, explicit 0.1-second minimum,
Accelerate, same local profile `72f1cd94dc71`. Neighboring shared matmul, einsum,
reshape/concat and MLP controls had no regression exceeding both 5% and 1 μs.
Repeated/reverse-order runs confirmed the retained changes. Speedups are local
measurements; smoke runs provide execution coverage only.

#### Persistent evidence

- Raw final comparison: `benchmarks/results/performance/cache_pass_final_before.json`
  and `cache_pass_final_after.json`; intermediate accepted/rejected experiments
  remain alongside them.
- Compact summaries: `benchmarks/history/summaries/`; immutable raw snapshots:
  `benchmarks/history/runs/` (gitignored). Commit summaries for lasting Git history.
- Comparison: `benchmarks/history/cache-pass-comparison.tsv`; sample compatible
  time series: `benchmarks/history/cache-pass-trend.tsv`.
- `just benchmark-record FILE --label LABEL` imports historical observations;
  `just benchmark-history --case REGEX --profile PREFIX` exports TSV. Normal
  benchmark runs record automatically. See `benchmarks/history/README.md`.
- CI uploads raw/summarized observations for 90 days. Smoke/exploratory results are
  excluded from trend output by default. Raw hashes, loaded-library hashes,
  source/build metadata, counters and repetition dispersion are preserved.

#### Validation

- **1,372 focused cases** on Accelerate: all executed successfully.
- Built-in-only matrix: **1,244 executed, 128 expected optional LAPACK skips**.
- Correctness: **494 passed** on Accelerate; **492 passed, two expected skips**
  without LAPACK. Existing 108-case legacy suite was unchanged by this pass.
- **12 Python runner/history tests passed**; coverage audit and whitespace checks
  passed. Added 16 matmul and 18 transpose benchmark cases.

</details>

<a id="history-plans-tensorflow-performance-plan-md"></a>
<details><summary>Historical source: plans/tensorflow-performance-plan.md</summary>

2026-10-24. Status: complete.

1. Add matched deterministic Float32/Float64 CPU fixtures: add, broadcast add,
   multiply, row/global sum, mean, ReLU, exp, softmax, log-softmax, matmul and
   square-loss forward/backward. Validate full outputs against independent
   references before timing. Keep allocation/output materialization semantics
   explicit and fix both frameworks to one compute thread.
2. Measure the native Release backend and TensorFlow eager/traced execution
   separately, excluding import, tracing, input creation, reference validation
   and host output copies. Warm up and take at least five calibrated repetitions.
3. Rank the observed gaps, optimize the C++ core for the largest reproducible
   bottlenecks, and retain only measured gains with gradients/edge cases covered.
4. Save raw data, history summaries, an ordered remaining-gap report and repeatable
   commands. Validate the wider backend matrix and built-in fallback.


#### First measurements and retained candidate

The initial 72-case comparison identified global sum/mean as the largest measured
losses (5.6–10.2× at 1,048,576 elements) and row sums (2.1–4.0×). Other measured
operations were generally competitive in this local packaged CPU configuration;
small eager calls include TensorFlow Python overhead, and many late-suite samples
were noisy. These are CPU package results, not conclusions about TensorFlow GPUs,
XLA, all hardware or isolated native TensorFlow kernels.

The C++ reduction now uses eight independent double accumulators for packed
floating rows. Float32 retains double accumulation; integer and column-reduction
paths stay as before. This removes the long serial-add dependency without global
fast-math flags, expanded inputs or additional tensor copies. A matched A/B rerun
with large add controls near parity measured sums 3.3–5.7× and row sums 1.9–3.5×
faster; mean also improved. Raw observations are
`benchmarks/results/tensorflow/native_sum_{reference_repeat,repeat}.json`.

Because host performance shifted during the first longer run, the harness now
supports `--paired-native`: time each native fixture next to TensorFlow and
alternate native-first/native-last order. Record host load and per-case CV.
Final comparisons use this mode rather than combining measurements from different
host-performance periods.


#### Paired follow-up and second pass

The adjacent-order comparison after sum optimization exposed larger residual
softmax/log-softmax, exponential and traced square-gradient gaps that the noisy
batch-order run had obscured. Saved as `tensorflow_paired_after.comparison.tsv`.

Retained improvements (verified by final paired and frozen-library comparisons):

- Contiguous exp uses Accelerate vForce when the existing Apple BLAS backend is
  enabled, with bounded integer-length calls; strided and non-Apple builds retain
  standard math. No new public API or dependency is required.
- Contiguous last-axis softmax/log-softmax reuse their output as exponential
  scratch, replacing full-sized intermediate tensors. Other axis/layout cases
  retain the composition. ARM SIMD maximum propagates NaNs, and sums retain
  double accumulation. Backward formulas retain higher-order differentiation.
- Identical-implementation `x*x` records one shared edge instead of duplicate
  contributions. Ordinary backward writes `2*(g*x)` in one pass; graph-building
  backward retains differentiable operations. The order avoids overflowing `2*x`
  before multiplication by a small gradient. Views sharing storage but having
  different tensor implementations use the general multiplication path.

The first fused-candidate rerun measured exp 1.5–3.4×, softmax 1.2–2.4×,
log-softmax 1.2–1.8×, and the square-gradient workload 1.3–1.6× faster than the
sum-only candidate. Neighbor controls drifted, so final claims require the final
paired and matched frozen-library reruns. Keep these intermediate observations as
experiment history, not universal performance claims.


#### Final results and validation

- 72 final adjacent comparisons: `benchmarks/results/tensorflow/tensorflow_final.comparison.tsv`.
- Original-versus-final confirmation: `native_confirmation_before.json` and
  `native_confirmation_after.json` in that directory. Approximate local speedups:
  row sum 1.84–4.11×; global sum 3.63–5.77×; mean 2.67–5.25×; exp 1.36–3.30×;
  softmax 2.13–4.60×; log-softmax 2.25–4.70×; square gradient 1.97–2.90×.
- Float64 8192×128 peak tracked tensor storage fell 50.2% for softmax, 66.8% for
  log-softmax, and 50.0% for square gradients. These exclude node/STL metadata
  and are not TensorFlow memory measurements.
- Unchanged controls drifted up to about 17%; retain CV and treat timing ranges as
  approximate. Final paired measurements put softmax/log-softmax/exp at or ahead
  of eager on these fixtures. Large Float32 reductions remain roughly 1.3–1.5×
  slower than eager, and the largest Float32 square-gradient case about 1.2×
  slower than the traced graph. Some differences are within noisy samples.
- **1,444** accelerated fixtures passed; built-in-only **1,316** executed with
  **128** expected optional LAPACK skips.
- Correctness: **498 passed** on Accelerate; **496 passed, two skips** without
  LAPACK. **17 Python contract tests passed**. Subnormal vector-exp outputs were
  also checked against standard Float32/Float64 exp near underflow.
- Native matched small cases are in the normal CI smoke selection; comparison
  tests run in CI without importing TensorFlow. The optional environment is pinned
  separately, and core/source/library hashes and raw samples enter persistent
  history. No generated bindings or public core API changed.

Detailed results, caveats and ordered next work are in
`benchmarks/results/tensorflow/RESULTS.md`. Priority: large Float32 reductions
while preserving double accumulation; remaining traced-gradient allocation/setup
cost; quieter-host reruns of marginal cases; then broader model/platform/thread
comparisons. GPU/XLA and Python-binding comparisons remain outside this CPU pass.

</details>

<a id="history-benchmarks-results-tensorflow-release-pass-results-md"></a>
<details><summary>Historical source: benchmarks/results/tensorflow/RELEASE_PASS_RESULTS.md</summary>

2026-10-24. This report describes the final retained code after the earlier
Release-verification and candidate runs. The benchmark was already Release;
these are code improvements, not Debug-to-Release speedups.

#### Build behavior

- `just build` now runs the actual Release core build through `build-cpp`, rather
  than the Debug configure recipe. The tools-only Nix shell avoids rebuilding an
  installed copy of the library before building the working tree.
- Standalone single-config CMake defaults to Release when no type is selected.
  Explicit Debug configurations remain supported. `just build-cpp` explicitly
  selects Release, including when its existing cache had an unspecified type.
- Benchmark configuration and compiler/runtime checks verify optimized Release
  for the core and driver. The final run used Clang 21.1.8, `-O3 -DNDEBUG`,
  Accelerate, and TensorFlow 2.21.0 on the local ARM64 macOS host.

#### Retained improvement

Long ARM64 Float64 reductions now use eight independent double vector
accumulators, a tree combination and a scalar tail. The threshold is 256
values; shorter rows retain their previous path. Float32 still accumulates in
double. Other architectures retain the portable implementation. No public API,
SWIG regeneration, fast-math flag or lower-precision accumulation was introduced.
The summation order can change ordinary floating-point rounding.

The final frozen-baseline/candidate confirmation uses the same benchmark driver,
with the two libraries measured adjacent for each fixture and their order
alternated. Both library hashes are checked against Release provenance. Five
samples and an explicit 0.1-second minimum calibration are used per measurement.
At 8192 × 128:

| Workload | Before µs | Final µs | Time reduction | Speedup |
|---|---:|---:|---:|---:|
| SumAll Float64 | 88.00 | 72.02 | 18.16% | 1.22× |
| Mean Float64 | 88.28 | 72.21 | 18.21% | 1.22× |

All twelve other confirmation cases (Float32 reductions, short-row Float64
reduction, softmax/log-softmax and add/multiply controls) stayed within 0.6%.
This is local CPU evidence, not a portable speed guarantee.

#### Final TensorFlow comparison

All 72 matched fixtures passed independent output checks. Native was faster in
64/72 TensorFlow eager comparisons and 70/72 warmed graph comparisons. Small
cases include TensorFlow Python/graph dispatch overhead; these counts do not
establish isolated kernel superiority. CPU execution is synchronous and uses one
compute thread. Inputs, imports, tracing, validation and host copies are excluded;
operation/output allocation is included. No GPU or XLA is used.

Ratios are native / TensorFlow; above 1 means native took longer. Times are
medians in microseconds for 8192 × 128 inputs.

| Case | Native µs | TF eager µs | TF graph µs | Native / eager | Native / graph |
|---|---:|---:|---:|---:|---:|
| SumLast Float32 | 65.91 | 48.73 | 113.32 | 1.35 | 0.58 |
| SumAll Float32 | 61.44 | 50.38 | 111.02 | 1.22 | 0.55 |
| Mean Float32 | 61.66 | 50.12 | 110.95 | 1.23 | 0.56 |
| SumLast Float64 | 74.31 | 85.55 | 150.59 | 0.87 | 0.49 |
| SumAll Float64 | 71.92 | 91.39 | 153.02 | 0.79 | 0.47 |
| Mean Float64 | 72.03 | 90.85 | 152.68 | 0.79 | 0.47 |
| Exp Float64 | 1038.28 | 879.39 | 947.12 | 1.18 | 1.10 |
| Softmax Float64 | 1598.32 | 1450.57 | 1527.71 | 1.10 | 1.05 |
| LogSoftmax Float64 | 1672.98 | 1620.34 | 1707.56 | 1.03 | 0.98 |
| SquareGradient Float32 | 199.36 | 575.80 | 245.86 | 0.35 | 0.81 |
| SquareGradient Float64 | 365.66 | 1181.96 | 426.77 | 0.31 | 0.86 |

Float32 reductions remain 1.22–1.35× slower than TensorFlow eager. Large
Float64 exp remains 1.18× slower, softmax 1.10×, and log-softmax 1.03×. This pass
does not close those gaps. Large square-gradient workloads remain faster than
the warmed graph measurements on both dtypes.

#### Rejected candidates and timing disturbance

The first candidate enabled the double vector loop on short rows and replaced
softmax elementwise division with multiplication by a row reciprocal. Neither
short-row nor softmax gains were repeatable after accounting for controls, so
short rows were restored and the reciprocal change was removed. The final
comparison measures the narrower retained optimization.

A later sequential confirmation showed unchanged add/multiply operations taking
5–7× as long. An initial adjacent retry was also unstable. Those runs are retained
for audit and excluded from speedup claims. The final adjacent confirmation above
returned stable controls and independently confirmed the long Float64 gain.
Earlier reports with “latest” in their title refer to earlier compiled images.

#### Validation

- Final Accelerate core: **502 tests passed**.
- Final built-in core: **500 passed, two expected LAPACK skips**.
- New coverage exercises 255/256/257-element boundaries, odd tails, cancellation,
  offset views, infinities and NaNs, alongside existing gradient/softmax tests.
- Accelerate fixture matrix: **1,452 passed**.
- Built-in matrix: **1,324 executed, 128 expected optional LAPACK skips**.
- Python runner contracts: **23 passed**. `git diff --check` passed.
- Fresh standalone CMake selected Release by default.
- `just JOBS=10 build` completed successfully; all **271 core translation units**
  in its compiler database use `-O3 -DNDEBUG`, and its cache records Release.

#### Reproduce and evidence

```sh
just build
just bench-backend-check
just bench-tensorflow --paired-native --label local-release-pass \
  --native benchmarks/results/tensorflow/local-release-native.json \
  --output benchmarks/results/tensorflow/local-release-tf.json
```

The frozen-library confirmation uses the same native driver, selecting the
fourteen large SumLast/SumAll/Mean/Softmax/LogSoftmax/Add/Multiply dtype fixtures,
with baseline-first and candidate-first order alternating. The baseline is loaded
through `DYLD_LIBRARY_PATH`; its original `release_pass_before_native.json` is
supplied as `--core-provenance`. Each raw result records its loaded library hash,
command, five timings, output checks and compiler manifest. Source and binary
hashes are archived with each run.

Final evidence:

- `release_pass_final_native.json`, `release_pass_final_tf.json` and
  `release_pass_final_tf.comparison.{json,tsv}`: all 72 final comparisons.
- `release_pass_final_adjacent_{before,after}.json` and
  `release_pass_final_native_comparison.tsv`: stable same-driver confirmation.
- `release_pass_final_matrix.json` and `release_pass_final_portable_matrix.json`:
  complete fixture correctness matrices, not timing comparisons.
- `release_pass_before_{native,tf}.json`: original Release baseline.
- `release_pass_after_{native,tf}.json`: superseded first candidate.
- `release_pass_confirm_{before,after}.json` and
  `release_pass_adjacent_{before,after}.json`: disturbed exploratory confirmations;
  excluded from final performance claims.
- Content-addressed raw archives and summaries are in `benchmarks/history/`.

Final core source SHA-256: `15f54aa33a45b105a5c51f00c9dc7d4e1cf377ff9801bc1ec34f4006ff0a1cca`.
Final loaded library SHA-256: `959f3542c38b7dd3299fb8f5d632789b53e7e0c6172add9326254fe7f724a45f`.

</details>

<a id="history-benchmarks-results-tensorflow-gap-results-md"></a>
<details><summary>Historical source: benchmarks/results/tensorflow/GAP_RESULTS.md</summary>

2026-10-24. Matched CPU measurements against TensorFlow 2.21.0; native Accelerate Release build, one compute thread, five repetitions and a 0.1-second minimum calibration. Float32 reductions retain Float64 accumulation.

#### Measured changes

| Workload (8192 × 128) | Native before µs | Native after µs | Native speedup | TF target | Native / TF before → after |
|---|---:|---:|---:|---|---:|
| SumLast Float32 | 94.79 | 65.87 | 1.44× | eager | 1.96 → 1.36 |
| SumAll Float32 | 92.10 | 61.47 | 1.50× | eager | 1.84 → 1.23 |
| Mean Float32 | 92.25 | 61.52 | 1.50× | eager | 1.85 → 1.23 |
| SquareGradient Float32 | 449.95 | 199.55 | 2.25× | graph | 1.83 → 0.81 |
| SquareGradient Float64 | 525.50 | 375.86 | 1.40× | graph | 1.24 → 0.89 |

Ratios above 1 mean the native call is slower. These are paired framework calls with alternating native-first/native-last order, not isolated TensorFlow C++ kernel timings. Setup/tracing/host copies are excluded; output allocation is included. Raw per-case CV is available in the comparison TSVs. Most reduction/gradient samples in this pass have substantially lower dispersion than the earlier noisy run.

The 38-case frozen-reference confirmation (`gap_final_reference.json` versus `gap_final_native.json`) also checks unchanged large add/multiply and MLP inference controls. Those controls stayed within about 1%; no material neighboring regression appeared. Small differences should not be treated as portable guarantees.

#### Implementation

- On ARM64, Float32 sums widen each vector before addition and use eight independent Float64 vector accumulators. The scalar tail preserves odd lengths and offsets. Float64 and non-ARM reductions retain their existing paths. No Float32 partial sums, full-size conversion buffers, or fast-math flags were introduced.
- A fully zero-stride broadcast view now uses its scalar storage when paired with a contiguous operand. This lets backward seeds use a contiguous vector loop rather than generic strided indexing. Shape, offset, broadcast semantics, operand order and independent output storage are preserved. Partial broadcasts and empty tensors keep their appropriate paths.
- Ordinary public `square()` backward computes its scale and multiply in one pass. Higher-order backward retains differentiable operations. The existing scale-before-seed order, saved-value mutation checks and overflow behavior remain covered by tests.

Eight new `BM_SquarePrimitiveVjp` cases cover unary-square sum/mean losses at two sizes and both dtypes. The frozen-reference comparison measured **1.75–2.64×** improvements across those cases. Large unary-square gradient peak storage fell **33.3%**, e.g. Float64 sum-loss VJP from **25,165,840 to 16,777,232 bytes**. Counters measure native tensor storage only, excluding node/STL metadata; TensorFlow memory was not measured.

#### Remaining differences

The previous square-gradient gap is closed on both dtypes in this run. Large Float32 row sums are still about **1.36×** behind TensorFlow eager, and global sum/mean about **1.23×**. Those differences are not closed, although they are substantially smaller. The implementation deliberately preserves double accumulation rather than replacing it with lower-precision partial sums.

Exploratory kernels found no material benefit from sixteen accumulators or extra 2×/4× unrolling. Adjacent-row batching did not improve the target width and regressed wider rows, so it was rejected; see the plan/experiment log. Further reduction work needs evidence of a faster widening/reduction schedule or a separately specified numerical mode, rather than silently relaxing accuracy.

#### Reproduce and inspect

```sh
build/tensorflow-venv/bin/python scripts/benchmark_tensorflow.py --paired-native \
  --case '^(SumLast|SumAll|Mean|SquareGradient)/rows:8192/f64:0/' \
  --native benchmarks/results/tensorflow/local-gap-native.json \
  --output benchmarks/results/tensorflow/local-gap-tensorflow.json --label local-gap
```

`--case` filters the canonical 72 matched fixtures and rejects empty or incomplete pairings. Omitting it still runs the complete comparison. Runner contract tests and native smoke coverage do not require TensorFlow.

Persistent evidence:

- `gap_baseline.json` / `gap_candidate.json`: first native candidate check.
- `gap_tf_before.json` / `gap_tf_after.json`, matching native JSON and `.comparison.{json,tsv}`: 28 adjacent-order comparisons including controls and smaller fixtures.
- `gap_final_reference.json` / `gap_final_native.json` / `gap_native_comparison.txt`: 38 native cases, including public-square VJPs and MLP controls, compared using the same driver and frozen pre-pass library.
- `gap_sum_experiments.txt`: exploratory accumulator/unrolling measurements; not cross-framework evidence.
- `gap_accelerated_matrix.json` / `gap_portable_matrix.json`: full one-iteration fixture validation; not timing comparisons.
- Content-addressed raw archives and versionable summaries remain under `benchmarks/history/`.

The local execution plan is `plans/tensorflow-remaining-gaps.md`.


Validation: **501** correctness tests passed on Accelerate; **499 passed with two
expected skips** on the built-in backend. All **1,452** accelerated fixtures
passed; built-in execution ran **1,324**, with **128** expected optional LAPACK
skips. All **18** Python contract tests passed. Whitespace/coverage checks passed.

</details>

<a id="history-benchmarks-results-tensorflow-results-md"></a>
<details><summary>Historical source: benchmarks/results/tensorflow/RESULTS.md</summary>

2026-10-24; TensorFlow 2.21.0, Python 3.13, Apple ARM64 macOS, native Accelerate Release build.
Both frameworks use one compute thread, synchronous CPU calls, five repetitions and 0.1-second minimum calibration.

72 matched cases cover Float32/Float64 and three sizes for 12 operations. Full outputs are checked against independent references. [Method and commands](benchmarks/TENSORFLOW.md). Native inputs use the same deterministic sequence as TensorFlow.

#### Changes and original-versus-final results

| Operation | Local speedup range |
|---|---:|
| SumLast | 1.84–4.11× |
| SumAll | 3.63–5.77× |
| Mean | 2.67–5.25× |
| Exp | 1.36–3.30× |
| Softmax | 2.13–4.60× |
| LogSoftmax | 2.25–4.70× |
| SquareGradient | 1.97–2.90× |

These are native before/after speedups, **not speedups over TensorFlow**. Final confirmation measured the candidate first, then the frozen original library with the same driver. The earlier sum experiment used the reverse order. Unchanged large add/multiply/ReLU controls drifted by up to about 17% in the final confirmation; treat these as approximate local ranges, not precise portable guarantees. No timing regression gate was added.

- Independent reduction lanes remove serial additions while retaining double accumulation for Float32.
- Last-axis contiguous softmax/log-softmax reuse one output buffer as exponential scratch. ARM maximum scans use NaN-propagating SIMD; other layouts/axes keep the general path.
- Contiguous exponentials use already-linked Accelerate vForce on Apple BLAS builds. Other builds use standard math.
- `x*x` records one gradient edge; ordinary backward fuses the contribution into one output pass. Higher-order backward retains differentiable operations and saved-version checks.

#### Tensor storage savings

For the 8192×128 Float64 fixtures (tracked tensor storage above the input baseline; excludes node/STL memory):

| Workload | Original bytes | Final bytes | Reduction |
|---|---:|---:|---:|
| Softmax | 16,842,752 | 8,388,608 | 50.2% |
| LogSoftmax | 25,296,896 | 8,388,608 | 66.8% |
| SquareGradient | 33,554,448 | 16,777,232 | 50.0% |

These are native before/after memory savings; TensorFlow memory was not measured.

#### Remaining gaps, in priority order

Ratios below are native / TensorFlow; above 1 means slower. Values come from adjacent paired measurements with alternating framework order. Small calls include Python/graph dispatch overhead; compare large fixtures for kernel conclusions.

| Case | Native / eager | Native / graph | Native CV | Eager CV | Graph CV |
|---|---:|---:|---:|---:|---:|
| Mean/rows:8192/f64:0 | 1.46 | 0.66 | 3.1% | 11.3% | 5.0% |
| SumLast/rows:8192/f64:0 | 1.41 | 0.52 | 8.9% | 12.3% | 8.8% |
| SumAll/rows:8192/f64:0 | 1.31 | 0.63 | 6.5% | 3.5% | 1.0% |
| SquareGradient/rows:8192/f64:0 | 0.47 | 1.21 | 12.1% | 3.1% | 2.5% |

1. Investigate larger Float32 row/global sums and mean (roughly 1.3–1.5× behind eager here). Preserve double accumulation; evaluate wider double-accumulator kernels and conversion/reduction throughput. An optional lower-precision accumulation mode would require a separate explicit API decision.
2. Reduce remaining overhead in the large Float32 square-gradient workload (about 1.2× behind the traced graph). Inspect gradient seeding/capture and allocation lifetimes; extend fused square VJPs to `square()` if worthwhile. Preserve higher-order behavior and IEEE edge cases.
3. Repeat marginal/noisy cases on an otherwise quiet host before tuning small differences. Some samples have >10% CV, so sub-20% differences are uncertain.
4. Extend matched coverage to dense-layer/training steps and convolution/normalization, additional thread counts, and another CPU platform. GPU/XLA and Python-binding overhead are outside this comparison.

Final softmax/log-softmax/exp fixtures are at or ahead of eager in this run, while Float64 multiply is near parity. This does not imply all tensor operations outperform TensorFlow.

#### Persistent evidence

- `native_before.json` / `tensorflow_before.json`: initial batch-order comparison (late-suite host drift; not the final gap ranking).
- `native_sum_reference_repeat.json` / `native_sum_repeat.json`: matched reduction experiment.
- `native_paired_after.json` / `tensorflow_paired_after.json`: sum-only candidate, adjacent framework measurements.
- `native_fused.json`: intermediate fusion candidate.
- `native_final.json` / `tensorflow_final.json` and `tensorflow_final.comparison.{json,tsv}`: final adjacent comparison.
- `native_confirmation_before.json` / `native_confirmation_after.json`: frozen original versus final native core, including tensor-memory counters.
- `accelerated_matrix.json` / `portable_matrix.json`: one-iteration full fixture validation.
- Immutable raw archives and versionable summaries: `benchmarks/history/`; each observation records source/build/binary provenance.

Validation: 498 correctness tests passed on Accelerate; 496 passed with two expected skips on the built-in backend. All 1,444 accelerated fixtures passed; built-in execution ran 1,316 with 128 expected optional LAPACK skips. All 17 Python contract tests passed. Timings above are separate from one-iteration coverage runs. The full local plan is `plans/tensorflow-performance-plan.md`.

</details>

<a id="history-benchmarks-results-tensorflow-release-first-image-results-md"></a>
<details><summary>Historical source: benchmarks/results/tensorflow/RELEASE_FIRST_IMAGE_RESULTS.md</summary>

Superseded by [the consistent latest-image rerun](#history-benchmarks-results-tensorflow-release-results-md) after a concurrent workspace softmax change. Retained for provenance.

### Fresh Release benchmark comparisons

2026-10-24. **The previous benchmark builds were already Release.** Actual compiler databases for the existing accelerated and built-in builds contain `-O3 -DNDEBUG`; the earlier frozen baseline hash matches its archived Release-backed result. The surrounding `build/debug/examples/...` directory does not determine the benchmark build configuration.

This verification rebuilt the current workspace in a fresh `build/benchmarks-release` directory with explicit `CMAKE_BUILD_TYPE=Release` and `CMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG`. All **271 core** and **9 focused-driver** translation units passed the effective-flag check. Runtime tags report `Clang-21.1.8/Release` and Google Benchmark `release`; the loaded core path and SHA-256 are verified and saved.

The current workspace includes its pre-existing Float64 vector-reduction optimization. These are current-code versus frozen pre-gap Release-baseline comparisons, **not Debug-versus-Release speedups**.

#### Reruns

- **82 native before/after cases**: all 72 matched operations, eight public-square VJPs, and two MLP controls.
- **72 adjacent-order TensorFlow comparisons**: Float32/Float64, three sizes and 12 operations, with eager and traced-graph timings separate.
- **12 focused repeats** of large arithmetic, exp/softmax and gradient cells to check timing variability.
- One compute thread, warm synchronous CPU execution, five repetitions, explicit 0.1-second minimum calibration. Input setup, graph tracing, reference checks and host copies are excluded. Output allocation is included.

#### Current native versus TensorFlow

Ratios are native / TensorFlow: above 1 is slower. A dash means the cell was not included in the focused repeat. Eager/Python and graph-call overhead are included; these are not isolated TensorFlow C++ kernel measurements.

| Case (8192 × 128) | Native µs, full run | TF eager µs | TF graph µs | Native / eager | Native / graph | Repeat native / eager | Repeat native / graph |
|---|---:|---:|---:|---:|---:|---:|---:|
| SumLast Float32 | 74.79 | 55.66 | 116.18 | 1.34 | 0.64 | — | — |
| SumAll Float32 | 61.30 | 49.95 | 111.31 | 1.23 | 0.55 | — | — |
| Mean Float32 | 61.85 | 49.94 | 111.10 | 1.24 | 0.56 | — | — |
| SumLast Float64 | 73.22 | 96.46 | 151.70 | 0.76 | 0.48 | — | — |
| SumAll Float64 | 71.92 | 90.61 | 151.82 | 0.79 | 0.47 | — | — |
| Mean Float64 | 71.94 | 90.75 | 151.60 | 0.79 | 0.47 | — | — |
| Exp Float64 | 1119.23 | 881.88 | 958.45 | 1.27 | 1.17 | 1.06 | 1.10 |
| Softmax Float64 | 1764.26 | 1536.81 | 1564.54 | 1.15 | 1.13 | 1.08 | 1.03 |
| SquareGradient Float32 | 219.77 | 621.04 | 266.12 | 0.35 | 0.83 | 0.34 | 0.80 |
| SquareGradient Float64 | 652.19 | 1291.12 | 431.82 | 0.51 | 1.51 | 0.31 | 0.86 |

Float32 row/global reductions remain about **23–34% slower than TensorFlow eager** while retaining double accumulation. The large Float32 square-gradient workload uses about **17–20% less time than the traced graph** in both runs.

The large Float64 gradient result changes direction between the full run and repeat (**1.51× versus 0.86×** relative to the traced graph). Float64 exp/softmax margins also vary. Both runs load the **same verified Release core and driver binaries**, so no stable win/loss should be claimed for those cells from a single run. Raw CV, load averages and both observations are retained. This supersedes any earlier unconditional claim that the Float64 gradient gap is closed.

#### Native before/after confirmation

Representative latest-code versus frozen pre-gap baseline measurements, using the same fresh Release driver:

| Case | Baseline µs | Current µs | Speedup |
|---|---:|---:|---:|
| SumLast Float32 | 104.89 | 68.58 | 1.53× |
| SumAll Float32 | 92.08 | 65.66 | 1.40× |
| Mean Float32 | 99.75 | 68.29 | 1.46× |
| SumAll Float64 | 91.63 | 74.42 | 1.23× |
| SquareGradient Float32 | 449.03 | 221.27 | 2.03× |
| SquareGradient Float64 | 519.67 | 415.36 | 1.25× |

The complete 82-case comparison is `release_native_comparison.txt`. Some unchanged operations drift across the two sequential processes, so these remain local observations rather than portable performance guarantees. No optimization was attributed to changing build mode.

#### Release enforcement

- CMake rejects single-config Debug benchmark builds; multi-config benchmark projects expose only Release. Actual negative Debug and Ninja Multi-Config configuration checks passed.
- The runner checks every core and selected-suite driver compilation command, requires optimized mode plus an effective `NDEBUG`, and rejects a non-Release cache configuration. Debug symbols alone do not cause rejection.
- Native runtime checks reject Debug Google Benchmark/driver tags. Focused POSIX runs also verify loaded-core identity. A frozen/overridden core requires `--core-provenance RESULT.json` with matching SHA-256 and Release compiler evidence.
- TensorFlow paired runs perform Release preflight before import/warmup/timing. Reused native observations must carry Release evidence.
- Legacy `benchmark`, `run-benchmark` and `bench-nn` workflows now use the verified runner; builds explicitly select `--config Release`.

#### Evidence and validation

- `release_build_evidence.json` and `release_compile_flags.tsv`: configuration, effective flags, verified unit counts, compiler-database hashes and frozen-baseline provenance.
- `release_debug_rejection.txt` / `release_multiconfig_manifest.json`: actual configuration guard checks; the multi-config check was configuration-only, not a timed build.
- `release_native_{before,after}.json` and `release_native_comparison.txt`: native comparisons.
- `release_tf_native.json`, `release_tf.json`, `release_tf.comparison.{json,tsv}`: full current TensorFlow comparison.
- `release_repeat_native.json`, `release_repeat_tf.json` and comparison tables: the focused repeat.
- `release_matrix.json`: all **1,452** fresh Release fixtures passed; this one-iteration run is coverage evidence, not timing evidence.
- **502** current correctness tests and **23** Python runner/history/comparison contract tests passed.
- Raw data and compact summaries are archived in the existing content-addressed benchmark history.

The execution plan is `plans/release-benchmark-verification.md`.

</details>

<a id="history-benchmarks-results-tensorflow-release-results-md"></a>
<details><summary>Historical source: benchmarks/results/tensorflow/RELEASE_RESULTS.md</summary>

2026-10-24. **The prior benchmark builds were already Release**, with `-O3 -DNDEBUG` in their compiler databases. This rerun uses a freshly configured `build/benchmarks-release` build, explicitly selecting Release for both core and driver. The surrounding examples/debug working directory does not select the benchmark configuration.

All **271 core** and **9 focused-driver** translation units passed effective optimization/`NDEBUG` checks. Runtime tags identify `Clang-21.1.8/Release` and Google Benchmark `release`. The frozen pre-gap baseline library matches its archived Release-backed SHA-256.

#### Consistent latest compiled image

The shared workspace gained a softmax normalization change during the initial verification. Those first-image runs are retained as historical observations. The affected translation units were forced to rebuild; the core-source hash stayed unchanged across that rebuild. Final results below use only the resulting latest image. The current code also includes its pre-existing Float64 vector reduction optimization. These are code-version comparisons, **not Debug-versus-Release speedups**.

- Compiled core-source SHA-256: `21c6ae92c53bf89c87b80c572d5ca041283519c34fa5276badaa10068a456481`
- Loaded core SHA-256: `2947bfb686598f83306b7271489369b9b6e5bfd8ee4703a9db23a21285a44f97`
- Driver SHA-256: `798996b8067e0b6c1c18525440347d46e839d081ebcfd79b57854dd1aa1059e5`

#### Repeated comparisons

- **82 native before/after cases**: 72 matched operations, eight public-square VJPs and two MLP controls, using the same driver.
- **72 paired TensorFlow comparisons**, with eager and traced graph timings separate.
- One compute thread, warm synchronous CPU execution, five repetitions and explicit 0.1-second minimum calibration. Setup, tracing, validation and host copies are excluded; output allocation is included.

Ratios below are native / TensorFlow: above 1 is slower. These include Python/graph-call overhead, rather than isolating TensorFlow C++ kernels.

| Case (8192 × 128) | Native µs | TF eager µs | TF graph µs | Native / eager | Native / graph |
|---|---:|---:|---:|---:|---:|
| SumLast Float32 | 281.79 | 291.92 | 666.27 | 0.97 | 0.42 |
| SumAll Float32 | 235.78 | 269.92 | 591.47 | 0.87 | 0.40 |
| Mean Float32 | 290.35 | 294.37 | 648.88 | 0.99 | 0.45 |
| SumLast Float64 | 608.52 | 779.24 | 1222.62 | 0.78 | 0.50 |
| SumAll Float64 | 555.80 | 715.27 | 1181.04 | 0.78 | 0.47 |
| Mean Float64 | 467.28 | 749.32 | 1229.87 | 0.62 | 0.38 |
| Exp Float64 | 4500.14 | 5153.00 | 5989.89 | 0.87 | 0.75 |
| Softmax Float64 | 6880.05 | 8364.80 | 9960.08 | 0.82 | 0.69 |
| SquareGradient Float32 | 1259.16 | 4514.15 | 1680.55 | 0.28 | 0.75 |
| SquareGradient Float64 | 2662.37 | 9291.25 | 3030.51 | 0.29 | 0.88 |

A focused same-image gradient repeat checks whether the gradient comparison is stable:

| Case | Full native / graph | Repeat native / graph |
|---|---:|---:|
| SquareGradient/rows:8192/f64:0 | 0.75 | 0.91 |
| SquareGradient/rows:8192/f64:1 | 0.88 | 0.94 |

Raw per-case CV and host-load observations are retained in the JSON and TSV files. Earlier first-image runs varied substantially on some Float64 cells even with identical Release binaries. Interpret small margins and direction changes cautiously; build-mode verification does not remove host-load or allocation variability. Float32 reductions continue to accumulate in double.

#### Native code-version comparison

| Case | Frozen Release baseline µs | Latest Release µs | Speedup |
|---|---:|---:|---:|
| SumLast Float32 | 359.73 | 283.47 | 1.27× |
| SumAll Float32 | 382.64 | 222.55 | 1.72× |
| Mean Float32 | 400.38 | 268.21 | 1.49× |
| SumAll Float64 | 642.96 | 613.90 | 1.05× |
| SquareGradient Float32 | 2510.09 | 1364.84 | 1.84× |
| SquareGradient Float64 | 3434.30 | 2374.54 | 1.45× |

Unchanged add/multiply/MLP controls span 0.81–1.24× across the two sequential processes. Treat before/after ratios as local measurements; the complete table includes every case and the paired TensorFlow observations limit long-run drift.

#### Release enforcement

- Single-config Debug benchmark configuration is rejected; multi-config benchmark projects expose only Release. Actual negative Debug and Ninja Multi-Config configuration checks passed.
- Every core and selected-suite driver compilation command must enable optimization and effective `NDEBUG`; a non-Release configuration is rejected. Debug symbols alone are permitted.
- Debug Google Benchmark and driver runtime tags are rejected. Focused POSIX runs also validate the loaded core path/hash. Frozen external cores require `--core-provenance RESULT.json` attesting the same library hash and Release compiler flags.
- Paired TensorFlow runs perform Release preflight before import/warmup/timing. Cached native observations require Release evidence.
- Legacy benchmark entrypoints use the verified runner, and build commands explicitly select `--config Release`.

#### Evidence

- `release_latest_build_evidence.json`: stable-source rebuild proof, actual compiler manifest and compiled binary hashes.
- `release_build_evidence.json` / `release_compile_flags.tsv`: compiler audits of existing/fresh build directories and frozen baseline provenance.
- `release_debug_rejection.txt` / `release_multiconfig_manifest.json`: configuration guard checks; multi-config was configured/audited, not timed.
- `release_latest_native_{before,after}.json` / `release_latest_native_comparison.txt`: 82 native comparisons.
- `release_latest_tf_native.json` / `release_latest_tf.json` / `release_latest_tf.comparison.{json,tsv}`: all 72 paired comparisons.
- `release_latest_repeat_native.json` / `release_latest_repeat_tf.json`: focused gradient repeat, if listed above.
- `release_latest_matrix.json`: all **1,452** latest-image accelerated fixtures passed in a one-iteration coverage run.
- `release_portable_matrix.json`: **1,324** built-in fixtures ran with **128** expected optional LAPACK skips.
- Correctness: **503 passed** for the latest accelerated image; built-in **501 passed with two expected skips**. **23** Python contract tests passed.
- Raw observations and compact summaries enter content-addressed benchmark history. The first-image observations/report remain available in `RELEASE_FIRST_IMAGE_RESULTS.md` and `release_*` files.

The completed execution plan is `plans/release-benchmark-verification.md`.

</details>

<a id="history-benchmarks-containers-md"></a>
<details><summary>Historical source: benchmarks/CONTAINERS.md</summary>

2026-10-24. The numeric tensor buffers were already contiguous. This pass removes
node-based lookup/allocation from important metadata paths without changing
numeric kernels, public source APIs, parameter creation order or checkpoint names.
Release builds use Clang 21.1.8 with `-O3 -DNDEBUG` on the local ARM64 macOS host.

#### Audited structures and final changes

| Area | Before | Final / reason |
|---|---|---|
| NN numeric storage | Shared `unique_ptr<unsigned char[]>` buffer | Already contiguous; retained uninitialized allocation and shared-view semantics. Replacing it with a default-initialized vector would add work without improving layout. |
| Tensor shapes, strides, kernel arrays | `std::vector` | Already contiguous; retained. |
| Legacy dense/sparse tensors and scalar autodiff tape | Vector-backed values, shapes, coordinates and tape nodes | Already contiguous; retained. |
| Backward engine | Separate visited/needed hash sets, capture maps and per-node gradient-buffer map | One dense vector of node records, a vector-backed open-addressed identity index, sorted capture entries and one contiguous pending-gradient-handle buffer. The per-call gradient vector is reused. |
| ParameterStore | Tree map with duplicated string keys plus creation-order vectors | Vector-backed open-addressed integer index at <=50% occupancy, referencing the existing name/value vectors. Expected constant-time lookup and amortized insertion; no duplicate key strings or per-key tree allocations. |
| Optimizer slots | Deque of named slots and tree map searched with constructed strings | Vector of named slots plus a lazy contiguous `[parameter][slot kind]` integer index. Warm steps use direct offsets without string construction. |
| Batch-normalization partials | Ordered map of separately allocated vectors, with a merge mutex | One aligned/padded double buffer, disjoint worker regions and ordered final merge. Serial work writes directly to the total. Scratch remains bounded by four chunks per configured thread. |
| Einsum label bookkeeping | Trees for label membership and counts | Sorted compact label vectors and a fixed character-count array, preserving implicit output order and ellipsis semantics. |
| FFT plans | Node-based hash table for a bounded cache | Sorted vector of length/plan pairs; existing mutex, shared plan ownership and 65-entry clearing policy retained. |

The batch-normalization worker regions are aligned and spaced for 128-byte cache
lines, also covering 64-byte hosts. Ordered merging remains deterministic for a
fixed thread configuration. Float32 reductions continue accumulating in double.
No fast-math or precision policy was changed.

Vectors are used as storage, not as an excuse for linear scans of large graphs.
The graph and parameter indexes retain expected constant-time hashed access;
optimizer kinds have fixed direct indexes. A preliminary sorted optimizer index
regressed the 4096-parameter first step by 8.7%; direct indexing removed that
tradeoff and produced the final gains below.

Graph nodes themselves retain shared ownership: saved values, aliased tensors,
retained graphs and recursive checkpoints must continue to outlive their users.
Optimizer storage reserves room for a parameter's maximum three new slots before
acquiring any slot reference; vector growth cannot invalidate references in use.
Restore still accepts unknown names, preserves order, and uses the last exact
canonical duplicate. `m/00` does not become an alias of `m/0`.

Some containers remain outside the main tensor engine: checkpoint/reversible
boundary membership uses hash sets for unbounded sparse identities; NMS groups
use a map; facade gradient names use a map; filesystem class validation uses a
set. Those are explicitly not claimed to be fully optimized by this pass. Blind
linear scans would risk quadratic graph membership work; profiling those separate
workloads would be needed to justify further replacements.

#### Measurements

Five repetitions per fixture, explicit 0.1-second minimum calibration, CPU
execution and one compute thread except the named four-thread normalization
fixtures. Timings include operation/output allocations and exclude setup and
validation. Parameter names use a deterministic shuffled sequence with common
prefixes; scalar-parameter optimizer fixtures expose bookkeeping cost.

The main suite compared 55 common fixtures plus three FFT cases. A second
confirmation measured 16 baseline/candidate pairs adjacent, alternating which
variant ran first. Frozen library and executable hashes were verified; parameter
and optimizer fixtures use their matching compiled clients, because their private
class layouts changed. The same core binary is used by the final suite, adjacent
confirmation and complete fixture matrix.

Adjacent confirmation medians (microseconds per fixture, not per parameter):

| Workload | Before µs | Final µs | Time reduction | Speedup |
|---|---:|---:|---:|---:|
| Parameter lookup, 4096 names | 566.707 | 59.485 | 89.5% | 9.53× |
| Parameter creation, 4096 names | 2083.203 | 1160.969 | 44.3% | 1.79× |
| AMSGrad first step, 4096 scalar parameters | 4608.212 | 2074.496 | 55.0% | 2.22× |
| AMSGrad warm step, 4096 scalar parameters | 2031.536 | 829.804 | 59.2% | 2.45× |
| Retained backward, 32 elements / depth 128 | 104.889 | 86.531 | 17.5% | 1.21× |
| Retained backward, 4096 elements / depth 128 | 171.162 | 151.564 | 11.4% | 1.13× |
| MLP training, batch 1 | 10.992 | 9.621 | 12.5% | 1.14× |
| Transformer training, batch 32 | 339.930 | 326.731 | 3.9% | 1.04× |

A supplementary retained-backward workload captures gradients for many independent
leaves, testing wide graphs rather than only deep chains:

| Capture targets | Before µs | Final µs | Time reduction |
|---|---:|---:|---:|
| 16 | 8.320 | 4.605 | 44.7% |
| 256 | 143.881 | 75.980 | 47.2% |
| 4096 | 2770.654 | 1437.282 | 48.1% |

The adjacent inference controls changed by -1.24% (MLP) and +0.17% (CNN). The
small Float32 normalization fixture changed by +2.23% (about 0.13 µs); the
Float64 counterpart was effectively unchanged. The largest four-thread
normalization fixture was also similar (-1.5%). No serious degradation appeared
in these cases. FFT repeats were -5.8%, +0.1% and -0.3% at 1/16/64 active lengths;
initial tiny-case differences changed direction, so no robust FFT speedup is
claimed. Large numeric kernels and unmeasured workloads are not guaranteed to
improve. These are local CPU timings, not hardware cache-miss measurements.

#### Correctness and limitations

- **535** Release correctness tests passed on Accelerate.
- **533** passed on the built-in backend, with **two expected LAPACK skips**.
- Coverage includes 100,000-node graph lifetime tests, higher derivatives,
  checkpoints/reversible recomputation, large/permuted/duplicate capture lists,
  parameter rehash/move/clone behavior, optimizer growth and restore semantics,
  deterministic parallel normalization, implicit einsum order and FFT eviction.
- Complete Accelerate matrix: **1,476 fixtures passed**.
- Complete built-in matrix: **1,348 executed**, **128 expected LAPACK skips**.
- **23** Python runner-contract tests passed; `git diff --check` passed.
- The normal `just JOBS=6 build` Release workflow completed successfully.
- Optional ASan/UBSan instrumentation compiled, but its Clang runtime stalled
  during macOS dyld/malloc initialization before `main`. A second startup attempt
  also stalled and was stopped. This is **not** a passed sanitizer test; see
  `containers_sanitizer_startup.txt` for the sampled stack.

Native C++ consumers must rebuild after the private ParameterStore/Optimizer
layout changes. Public source APIs, tensor/view ownership and checkpoint format
remain compatible; generated binding source does not require regeneration.

#### Reproduce and inspect

```sh
just build
just bench-backend-check
nix develop .#cpp-tools --command python3 scripts/benchmark_backend.py run \
  --output benchmarks/results/performance/local-containers.json \
  --label local-containers -- \
  '--benchmark_filter=^(BM_Container|BM_AutogradChain/(ForwardBackward|RetainedBackward)|BM_AutogradFanout|BM_TensorEinsum|BM_LayerComponent/BatchNorm|BM_Network/(MLP|CNN|Transformer)_(Inference|Train))' \
  --benchmark_min_time=0.1s --benchmark_repetitions=5
```

New probes are ordinary Google Benchmark registrations in
`benchmarks/backend/containers.cpp`, included in the coverage audit and smoke
selection. The frozen baseline is a pre-change working-tree binary, not a clean
Git revision. Its source/library/driver hashes are recorded, so unrelated staged
work is not represented as a committed baseline.

Evidence in `benchmarks/results/performance/`:

- `containers_before.json`: frozen original Release baseline, 55 cases.
- `containers_after.json`: preliminary sorted-index candidate (superseded).
- `containers_final.json` / `containers_final_comparison.tsv`: final core's 58-case
  targeted measurement set before adding the three wide-graph registrations.
- `containers_adjacent_{before,after}.json` /
  `containers_adjacent_comparison.tsv`: final sixteen-pair confirmation, including
  exact subprocess commands, contexts, client hashes and loaded-library proof.
- `containers_supplement_{before,after}.json` /
  `containers_supplement_comparison.tsv`: wide captures and FFT repeats. These
  fixtures use only the unchanged Tensor/FFT client ABI, so the current driver can
  safely load the verified frozen core for those specific filters.
- `containers_final_matrix.json` / `containers_final_portable_matrix.json`:
  complete correctness matrices after adding all 24 new fixtures. Smoke timings
  are not used for speedup claims.
- `containers_fft_before.json`: initial FFT reference observations.
- Raw runs and compact summaries are also in content-addressed benchmark history.

Final compiled core source SHA-256: `47f91030b684dc9fc3785a038a44e7904c1c6d7ad22442f9db416ceea9f23918`.
Final loaded core SHA-256: `bd46e7d33290ea56019822bcf5da214f720b2584db39c61745abf0a8ed0c0984`.

</details>

<a id="history-benchmarks-readme-md-historical-results"></a>
<details><summary>Historical source: benchmarks/README.md (historical results)</summary>

#### Initial local baseline

`results/backend_baseline_20261023.json` records the 224-case smoke selection with
three calibrated repetitions per case and a 0.01-second minimum per repetition.
The complete 932-case matrix also passed one-iteration validation, and all 350
existing correctness tests in `backend_checks` passed. A separate repeated run
produced 224 comparable cases with no missing cases. These local macOS arm64
Accelerate results came from a dirty working tree and are a starting point for
measurement, not established regression limits or a performance improvement claim.


#### Coverage audit follow-up

The initial audited suite registered **1,089 cases** across the agreed areas. Every case
passed one-iteration execution and its fixture checks on the local Accelerate build;
382 existing C++ tests and six runner tests also passed. The new repeated audit
baseline is `results/backend_audit_20261023.json`; the earlier baseline remains a
historical artifact. Optimizer shape names and some fixtures expanded, so use a
fresh matching baseline for optimization work.

The detailed coverage report is `../plans/backend-benchmark-coverage-audit.md` and
the proposed optimization sequence is `../plans/backend-performance-improvement-plan.md`.
These local plans may be gitignored by repository policy.


#### Performance work and current coverage

The performance execution report is `../plans/backend-performance-results.md`.
The focused suite now registers **1,452 cases**, with all 15 tensor optimizers in
both dtypes, phases and vector/matrix shapes. Extensions cover shared GEMM/solve,
larger legacy GEMM, sequence-model components, Jacobians, ring PSO and multiple
thread/element-count sweeps. The legacy sparse/support suite has **108 cases**.

Retained changes include typed legacy tensor traversal, shared GEMM/factor reuse,
fused tensor optimizers, reduction/autograd setup, recurrent projections,
attention scratch sizing, general optimizer workspaces, population selection,
autodiff seed reuse, sparse kernels and pool completion. The report records
calibrated comparisons, correctness checks and rejected experiments; timings are
local macOS results, not portable speedup promises.

Results record executable SHA-256 and, on POSIX focused runs, the actually loaded
core library path/SHA-256. Header/template experiments freeze the executable as
well as the core. Use a matching benchmark/coverage revision when rerunning a
historical executable. Artifacts are under `results/performance/`; older short
baselines remain historical evidence. Tensor memory counters exclude STL scratch
and node metadata.

Latest explicit Release validation: 503 accelerated correctness checks passed;
the built-in build passed 501 with two LAPACK-only skips. All 1,452 accelerated
fixtures passed; built-in execution ran 1,324 with 128 expected LAPACK skips.
Twenty-three runner/history/comparison tests passed. The 108-case legacy suite
was validated in the earlier performance pass. One-iteration runs validate
fixtures; calibrated comparisons and compiler/hash evidence are in
[the fresh Release report](#history-benchmarks-results-tensorflow-release-results-md).



#### Copy and cache pass

Partial-batch floating matmul now packs the original operands once and maps batch
indices without materializing expanded inputs. On this machine the 16 new cases
improved **1.57–3.60x**; one Float64 case reduced peak tracked tensor storage from
1,441,792 to 262,144 bytes (**81.8%**). Double matrix transpose uses cache tiles for
larger matrices (**1.53–2.75x** in the final comparison); identity permutations use
a direct independent copy (**3.37–4.74x**). Generic paths retain their semantics.
These are five-repetition, 0.1-second minimum local measurements, not portable
speedup promises. Neighboring shared GEMM, einsum, reshape/concat and MLP cases
showed no regression above both 5% and 1 microsecond in the final 79-case comparison.

Successful runs now archive raw JSON and compact, versionable summaries
with compatible-machine/build profile IDs. See [benchmark history](benchmarks/history/README.md)
for recording, importing and exporting trends. The final comparison and an example
time series are in `history/cache-pass-comparison.tsv` and
`history/cache-pass-trend.tsv`. The execution report is
`../plans/backend-copy-cache-pass.md`; full raw comparisons are
`results/performance/cache_pass_final_{before,after}.json`.


#### TensorFlow comparison

The optional [matched TensorFlow CPU harness](benchmarks/TENSORFLOW.md) adds 72 fixtures
covering both floating dtypes, three sizes and 12 operations, including a
forward/backward workload. It validates complete outputs before timing, records
TensorFlow eager and traced calls separately, and can alternate adjacent native
and TensorFlow measurements to reduce host drift. Install the isolated optional
environment described there, then run `just bench-tensorflow --paired-native`.
Results and sorted gap tables persist under `results/tensorflow/` and benchmark
history. The measured gaps motivated independent-lane floating sums, fused last-axis
softmax/log-softmax, vector exponentials on Accelerate, and fused self-multiply
gradients. Float32 sums retain double accumulation without fast-math flags.
See [results and remaining priorities](#history-benchmarks-results-tensorflow-results-md).


The [remaining-gap pass](#history-benchmarks-results-tensorflow-gap-results-md) adds widened ARM
Float32 reductions, vector loops for uniform broadcast seeds, and fused unary
square VJPs. It closes the measured square-gradient gap and narrows Float32
reduction differences while retaining double accumulation. Eight additional
native sum/mean VJP fixtures cover public `square()` independently of `x*x`.
Use the TensorFlow driver's `--case` option for targeted paired reruns.


The verified runner requires Release configuration, effective optimization and
`NDEBUG`, and checks runtime build tags. Frozen core overrides in focused runs
require a matching Release-backed `--core-provenance` result. Benchmark CMake
projects expose only Release for multi-config generators; recipes explicitly
build that configuration. Legacy native entrypoints also use the verified runner.

Tensor metadata/container locality, measured gains and retained exceptions:
[container cache audit](#history-benchmarks-containers-md).

</details>

