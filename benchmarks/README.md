# Backend benchmarks

The focused `backend_bench` executable measures tensor linear algebra, both tensor
APIs, autograd/autodiff, general optimization, tensor optimizers and compact neural
networks. Google Benchmark drives all timing. The older `linalg_bench` and
`nn_bench` executables remain available; `nn_bench` includes large attention cases
and a full synthetic CNN epoch, so it is intentionally separate from the smoke suite.

## Run

From the repository root:

```sh
just bench-backend-smoke
just bench-backend-check
just bench-backend-audit
just bench-backend -- --benchmark_filter=BM_TensorMatmul --benchmark_repetitions=5
just bench-backend -- --benchmark_filter=BM_Autograd --benchmark_min_time=0.1s
just bench-backend -- --benchmark_repetitions=3
just benchmark-compare baseline.json candidate.json
```

`just` enters the C++ tools Nix shell. Without Nix, use an environment with CMake,
a C++23 compiler, Python 3 and Google Benchmark (CMake can fetch Benchmark):

```sh
python3 scripts/benchmark_backend.py run --smoke
python3 scripts/benchmark_backend.py run -- --benchmark_filter=BM_GeneralOptimizer
./build/benchmarks/backend_bench --benchmark_list_tests=true
```

Every focused run first checks registrations against `coverage.json`, including all
current tensor optimizer kinds and both parameter shapes. Run `just bench-backend-audit`
to check an existing build without timing it. The manifest describes agreed operation
coverage; it does not claim every public API or general optimizer algorithm is benchmarked.

The runner accepts `--jobs`, `--build-dir`, `--output` and repeated `--cmake-arg`
options before `--`. Google Benchmark options go after `--`. `--no-build` is for
an already up-to-date build. The default invocation runs the entire focused matrix;
use `--smoke` for one timed iteration on a curated small matrix across every family.
Smoke times establish that cases execute; use repeated calibrated runs for analysis.
For example, `--smoke -- --benchmark_min_time=0.01s --benchmark_repetitions=3`
creates a short repeated baseline across the same small matrix.

Examples of separate backend builds:

```sh
python3 scripts/benchmark_backend.py run --build-dir build/benchmarks-builtin \
  --cmake-arg=-DDATAMUNGE_USE_BLAS=OFF --cmake-arg=-DDATAMUNGE_USE_LAPACK=OFF --smoke
python3 scripts/benchmark_backend.py run --suite linalg_bench -- \
  --benchmark_filter=BM_CSR_spmv --benchmark_min_time=0.1s
```

## What each family measures

| Family | Timed work |
| --- | --- |
| `BM_TensorLinalg` | Batched solve, triangular solve, Cholesky, QR, SVD, eigh, inverse, logdet, least squares and pseudoinverse; f32/f64, backend mode and crossover sizes |
| `BM_TensorMatmul`, `BM_TensorRectangular`, `BM_TensorDetEig`, `BM_TensorEinsum` | Rectangular/batched GEMM, tall/wide QR/SVD, determinant/log-determinant, nonsymmetric eigenvalues and an einsum contraction |
| `BM_TensorOps`, `BM_LinalgTensor*` | Allocation, initialization, copies/materialization, supported reshape/transpose, broadcasting/arithmetic, reduction, cast, indexing, slicing and concatenation |
| `BM_TensorThreadScaling` | Large elementwise ReLU with 1/2/4 library threads |
| `BM_AutogradChain`, `BM_AutogradFanout` | No-grad forwards, recording forwards (including graph destruction), full gradients and retained-graph backward; depth, fan-out and broadcast-gradient accumulation |
| `BM_AutogradLinalg` | Forward plus backward of solve, Cholesky, QR, SVD and eigh, with directional derivative checks |
| `BM_AutodiffTape`, `BM_AutodiffForward` | Standalone reverse-tape recording/reverse/full gradients; dual gradients and hyperdual Hessians |
| `BM_GeneralOptimizer` | Bounded complete solves using gradient descent, Adam, L-BFGS, Newton, Nelder-Mead, differential evolution and PSO |
| `BM_ConstrainedSQP`, `BM_MultiObjectiveNSGA2` | An equality-constrained quadratic and a bounded two-objective tradeoff |
| `BM_TensorOptimizer`, `BM_GradientUtilities` | First-step construction/update versus warmed updates for every current optimizer kind; fixed total parameter counts split across tensors; clipping and four-way gradient accumulation |
| `BM_Network` | MLP, compact CNN, 16-step vanilla RNN and compact transformer: inference, forward/backward and Adam training steps |
| `BM_AttentionSmall`, `BM_NeuralPrimitive` | Plain/fused attention, causal and unmasked, forward and forward/backward; ReLU, softmax and cross-entropy |

`nn::Tensor` cases use float32 and float64 where registered. General
`linalg::Tensor` arithmetic is float64; its structural operations can copy,
whereas NN operations may create views. Compare their documented semantics before
interpreting their timings. Tensor throughput is logical input elements/s (gather
and slice use output elements, concatenation uses output elements); view cases
measure metadata overhead, not memory bandwidth. Allocation-only `empty` never
reads uninitialized storage. FLOP/s counters use raw operation counts, without a
misleading giga prefix.

Linear algebra fixture generation and reference checks are outside timing.
Fixtures check solve residuals, decomposition reconstruction, QR orthogonality,
eigen residuals, and determinant/eigenvalue references as appropriate. General
linear algebra cases select Auto/Builtin/LAPACK and restore the previous backend.
Unavailable LAPACK cases produce explicit skips; they never masquerade as LAPACK
results. Auto crossover sizes are represented in the sweep (16, 32, 64, 96, 128,
256), but depend on the operation and build.

General optimizers reset coordinates outside timing. The timed solve includes
objective/gradient/Hessian evaluation and algorithm allocations. Problems 0/1/2
are an isotropic quadratic, a quadratic with condition number 100, and Rosenbrock.
Population methods also cover problem 3, Rastrigin. Quadratic coefficients are
precomputed outside timing so repeated power calculations do not dominate the fixture.
The deterministic methods have a 100-iteration cap; population methods use 24
members and 30 generations, with seeds 42/43/44. Default library stopping tolerances
still apply. Counters report evaluations per solve, achieved objective and whether
the common objective target `1e-6` was reached. Iteration budgets differ in work;
compare evaluation counts and objective quality with runtime. These are bounded
solver benchmarks, not a learning-quality study or guaranteed time-to-target runs.
`BM_OptimizerTimeToTarget` separately measures reaching objective `1e-6` on the
16-dimensional isotropic and condition-100 quadratics, with a 10,000-iteration cap
and strict library tolerance. Its adapter checks the objective before gradients
and at objective evaluations, and stops at the first successful evaluated point
using an exception. Timings include these extra checks and unwinding; evaluation
counters include the extra work. This is distinct from the bounded-solve cases.
NSGA-II uses a population of 32 and 20 generations; SQP uses library defaults.

Neural network training and tensor optimizer cases reset parameters and restore a
snapshot of warmed optimizer state outside timing. Model `LossOnly` uses fixed
logits; `RetainedBackward` uses a retained prebuilt loss graph; `OptimizerOnly` uses
fixed gradients and the same reset policy. These component times are measured
separately and need not sum exactly to a complete step. Each timed update therefore
starts at the same point rather than continuing to train during calibration.
`FirstStep` constructs a fresh optimizer and includes state initialization;
`Warmed` excludes that construction. Gradient utilities include their result
allocation. No model downloads, dataset I/O, rendering or RL dependencies are needed.

## Memory, threads and result interpretation

Selected cases report tensor storage peaks measured in a separate untimed probe.
They use the same execution mode as the timed workload. Optimizer cases also report
persistent slot tensor bytes (excluding serialized scalar metadata). First-step
update peaks include new optimizer slots; warmed update peaks exclude existing
slots. These are live tensor-buffer counts, not total
allocator use, graph metadata or process RSS. Autograd chain cases verify that a
completed gradient call leaves no extra live tensor storage. Peak resets are never
performed concurrently.

The focused suite uses one library thread except explicitly named thread-scaling
cases. The runner defaults `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and
`OMP_NUM_THREADS` to 1 only when unset; explicit user values are preserved and
recorded. Library and BLAS thread controls are distinct. Wall time is the comparison
metric; CPU time is also recorded by Google Benchmark. Restore thread and backend
settings after each case.

Results are saved under `benchmarks/results/` with timestamped filenames. JSON
includes Google Benchmark context plus revision/dirty state, working-diff digest,
benchmark source hashes, platform, thread environment, invocation, CMake settings
and example compiler commands. The runner verifies optimized Release commands for
both core and drivers, rather than assuming that driver-only optimization suffices.

Comparison uses the median of repetitions per case, normalizes timing units, and
rejects known incompatible configurations or benchmark errors. Negative change is
faster. Legacy JSON may be compared with other legacy JSON with a warning about
missing metadata. It cannot be compared silently with a metadata-backed run.
Timing deltas alone do not establish a regression: use the same machine/load,
repeat runs, inspect dispersion and verify changed numerical quality. CI only
builds, checks fixtures and archives smoke results; it has no timing thresholds.

## Correctness and runner validation

Reuse the repository's existing correctness tests against the benchmark core:

```sh
nix develop .#cpp-tools --command cmake -S benchmarks -B build/benchmarks \
  -DCMAKE_BUILD_TYPE=Release -DDATAMUNGE_BENCH_CHECKS=ON
nix develop .#cpp-tools --command cmake --build build/benchmarks --target backend_checks -j 4
./build/benchmarks/backend_checks
python3 -m unittest discover -s benchmarks -p test_runner.py
```

GTest must be installed for the optional `backend_checks` target (provided by the
C++ tools shell). This target reuses tensor, linear algebra, autodiff/autograd,
optimizer, layer, attention, convolution, training and memory tests. It does not
replace the complete repository test suite.

The sparse benchmark generator now mixes low bits before power-of-two indexing
and creates explicit duplicate insertions. Old sparse baselines used a different
input distribution and must be regenerated. The vector concurrency cases label
independent-container behavior; they do not claim shared-container contention.

Additional coverage includes `BM_TensorBroadcast` (scalar/channel/spatial),
`BM_TensorMax`, `BM_LinalgTensorExtra` (mean/max/scalar add), `BM_TensorSolveRhs`,
`BM_GeneralEigenvectors`, `BM_AutodiffBranch`, and `BM_LayerComponent`
(layer norm, embedding, convolution, pooling, batch norm, LSTM and GRU). Layer
components include float32/float64 and forward/forward-plus-backward modes.

RL, bindings, external simulators and unrelated backend families remain outside the
agreed scope. Additional algorithms and production-sized model sweeps can extend
this harness without changing the measurement contract.

## Performance plan and comparisons

All plans, current results, container/cache audit, rejected experiments and every
TensorFlow/native comparison table are consolidated in [PERFORMANCE.md](../PERFORMANCE.md).
Use this README for benchmark workflow and scope; update the single report for
performance conclusions. [TensorFlow quick start](TENSORFLOW.md).
