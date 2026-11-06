# SqueezeNet CIFAR-10 training throughput

Goal: train as many epochs as possible in a fixed wall-clock budget with the
exact loop of the SqueezeNet report
(`docs/mepml/examples/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/squeezenet.mepml`):
SqueezeNet v1.0 with the CIFAR stem, batch 32, per-image pad/crop/flip
augmentation, L2 penalty on every weight, Adam with warm-up + cosine schedule,
CPU training. The metric is **epochs per minute** (250 steps of 32 images).

Everything here is reproducible with the two benchmark binaries:

| Binary | Recipe | Measures |
| :--- | :--- | :--- |
| `benchmarks/nn/squeezenet_train.cpp` | `just bench-squeezenet --threads 16 --seconds 20` | The real training step: ms/step, img/s, epochs/min, per-phase split (augment / forward / backward / update), optional `nn::Profiler` trace (`--profile trace.json`) and JSON results (`--json out.json`). |
| `benchmarks/nn/squeezenet_ops_bench.cpp` | `just bench-squeezenet-ops` | Google Benchmark micro-benchmarks of every op class at the exact layer shapes (11 convolution geometries forward/backward-input/backward-filter, bias+ReLU, ReLU backward with contiguous and strided seeds, dropout, max pool, concat, the L2 objective, the Adam step, augmentation, pool dispatch latency, whole-model forward and forward+backward), each at 1 and 16 threads. |
| `benchmarks/frameworks/` | `benchmarks/frameworks/run_all.sh` | The same network, data pipeline and optimizer in PyTorch and TensorFlow, CPU and GPU, for comparison (see `benchmarks/frameworks/README.md`). |

Raw results live in `benchmarks/results/squeezenet/` (JSON per run) and
`benchmarks/results/frameworks/`.

Reference machine: AMD Ryzen 9 9955HX (Zen 5, 16 cores / 32 threads, AVX-512),
60 GB RAM, RTX 5080 Laptop GPU, NixOS, GCC 15.2, OpenBLAS 0.3.30 (pthread build,
`DYNAMIC_ARCH`, picks the Haswell AVX2 kernels on this CPU).

## 1. Baseline (before any change)

`squeezenet_train`, augmented, 20 s per run:

| threads | ms/step | img/s | s/epoch | epochs/min | forward | backward | update |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 132.2 | 242 | 33.1 | 1.81 | 54.1 ms | 75.6 ms | 2.0 ms |
| 4 | 100.4 | 319 | 25.1 | 2.39 | 38.9 | 59.6 | 1.5 |
| 8 | 95.3 | 336 | 23.8 | 2.52 | 36.2 | 57.2 | 1.3 |
| 16 | 99.2 | 323 | 24.8 | 2.42 | 38.1 | 59.2 | 1.4 |
| 32 | 106.0 | 302 | 26.5 | 2.26 | 41.0 | 62.9 | 1.5 |

Augmentation is 0.4 ms/step (0.4 %) and is not a target. Sixteen cores buy
only 1.33x over one core; `perf stat` shows **2.3 CPUs busy on average at 16
threads**: the step is essentially serial.

### Where the time goes (16 threads)

`nn::Profiler` trace of 5 steps (backward nodes are traced individually,
forward ops only as a region):

| region / backward node | share of step |
| :--- | ---: |
| forward (all ops) | 39 % |
| Conv2D backward (input + filter GEMMs) | 49 % |
| Relu backward | 5.3 % |
| MaxPool backward | 1.9 % |
| Add backward (bias reductions) | 2.0 % |
| Adam update | 1.3 % |

`perf record -g` at 16 threads (share of CPU samples; parallel kernels are
over-represented relative to wall time, serial ones under-represented):

| symbol | CPU share | what it is |
| :--- | ---: | :--- |
| `sgemm_kernel_HASWELL` (+ pack routines) | 33 % | every convolution GEMM, **single-threaded**: `OPENBLAS_NUM_THREADS=1` in the dev shell and no split across the `nn` pool |
| `strided_rows<3, map_binary<leaky_relu_backward>>` | 17 % | ReLU backward on the strided split views that `concat`'s backward produces: generic scalar strided loop |
| `max_pool_with_global_argmax` | 11 % | branchy scalar argmax loop with an `int64` index store per channel per tap |
| `strided_rows<3, map_binary<add>>` | 6.6 % | `bias_add` of `[N,H,W,C] + [C]`: same generic scalar strided loop |
| `map_unary<relu>` | 5.7 % | ReLU forward (fine, memory-bound) |
| `memmove` / `memset` | 10 % | im2col copies, `std::vector` value-initialisation of every im2col buffer, `dx` zero-fill, output allocations |
| `max_pool_scatter` | 4.0 % | max pool backward |
| `im2col` / `col2im_add` | 3.7 % | patch extraction |

Flame graph: `benchmarks/results/squeezenet/flame_baseline.svg`.

### Micro-benchmark baseline

`squeezenet_ops_bench` (`benchmarks/results/squeezenet/ops_baseline*.json`),
wall time per call at 1 and 16 threads:

| op (shape) | 1 thread | 16 threads | note |
| :--- | ---: | ---: | :--- |
| conv2d forward conv1 `[32,32,32,3]*[3,3,3,96]` | 3.46 ms | 1.78 ms | 49 GFLOP/s: im2col-bound (K = 27) |
| conv2d forward fire4 expand3x3 `[32,16,16,32]*[3,3,32,128]` | 4.73 ms | 4.59 ms | 128 GFLOP/s = one AVX2 core; **no scaling** |
| conv2d forward fire8 expand3x3 `[32,8,8,64]*[3,3,64,256]` | 4.62 ms | 4.66 ms | same |
| conv2d backprop input, fire4 expand3x3 | 4.95 ms | 4.93 ms | same |
| conv2d backprop filter, fire4 expand3x3 | 4.98 ms | 4.66 ms | same |
| conv2d backprop filter, conv1 | 2.57 ms | 2.33 ms | M = 27 rows: only a K split can parallelise it |
| bias_add + relu `[32,16,16,128]` | 235 us | 229 us | |
| relu backward, contiguous seed, 1 M elements | 6.60 ms | 1.32 ms | **4 ns/element on one core: scalar compare-and-branch** |
| relu backward, strided seed (split view) | 7.32 ms | 2.24 ms | generic strided loop |
| bias_add + relu backward (incl. bias reduction) | 6.73 ms | 1.86 ms | |
| max_pool2d forward `[32,33,33,96]` 3x3/2 | 15.6 ms | 2.23 ms | 2.2 ns per tap-channel compare |
| max_pool2d forward + backward | 18.2 ms | 2.97 ms | |
| dropout `[32,4,4,512]` | 1.86 ms | 2.05 ms | serial RNG, three passes |
| concat of two `[32,16,16,64]` | 113 us | 109 us | |
| L2 objective over 26 weights, forward + backward | 308 us | 492 us | 208 tiny ops; slower with more threads |
| Adam step, 729 k parameters | 2.26 ms | 1.46 ms | memory-bound, poor scaling |
| augmentation, one batch | 280 us | 392 us | |
| one `parallel_for` of 64 trivial chunks | 0.67 us | 8.6 us | pool dispatch + futex wake |
| SqueezeNet forward (training) | 85.5 ms | 72.7 ms | |
| SqueezeNet forward + backward | 190 ms | 117 ms | |

Summing the convolution rows over the 26 layers gives about 29 ms of forward
and 58 ms of backward convolution per step on one core, i.e. **convolution is
~85 % of the 99 ms step and runs on one core**. The elementwise, pooling and
reduction kernels are the other 15 % and scale, but are 5-20x slower than
memory bandwidth allows.

`perf annotate` on the contiguous ReLU backward shows why: the loop is
`comiss / jbe / movss` per element. The op lambda's captured slope `a` is
reached through a reference inside the type-erased `parallel_for` chunk
callback, so the compiler cannot prove the output stores do not alias it and
gives up vectorising. A standalone copy of the same loop runs at 0.52 ms per
million elements (12x faster) once the functor is a local copy.

## 2. Plan, in priority order

Expected gains are wall-clock ms per 99 ms step at 16 threads, estimated from
the profile above and verified by re-measuring after each item.

| # | change | why | expected |
| :---: | :--- | :--- | ---: |
| P1 | **Parallel GEMM across the `nn` pool** (`src/datamunge/nn/gemm.cpp`): split M across threads (forward, backward-input), split K with per-chunk partial products reduced in a fixed order (backward-filter, whose M is only `kh*kw*cin`), split N as a fallback; pin OpenBLAS to one thread so the two pools never oversubscribe. | 33 % of CPU time on one core while 15 idle | 25-35 ms |
| P2 | **Vectorisable elementwise kernels** (`detail::map_binary`, `map_unary`, `parallel_elements`, `Tensor::copy_`): kernels capture their pointers and scalars by value and the chunk callback runs a local copy of the functor, so stores can no longer alias the captures and the loops vectorise; the strided path gets plain inner loops for unit (or broadcast) inner strides instead of the generic `off + j * step` loop. Fixes ReLU backward (contiguous and on split views), `bias_add`, `.contiguous()` of split gradients and `concat`. | 24 % of CPU time in scalar loops; 12x available per the standalone test | 10-15 ms |
| P3 | **Branchless max pool** forward (first tap copies, later taps update value and index with selects over the channel vector) and a simpler backward scatter. | 15 % of CPU time for 10 M tap-channel compares | 5-8 ms |
| P4 | **No per-call allocation in convolution**: thread-local reusable im2col scratch (no value-initialisation), parallel `dx` zero-fill. | memset/memmove 10 % | 3-5 ms |
| P5 | **Pool wake-up latency**: workers spin briefly on the job generation before sleeping on the condition variable; the submitter likewise. A step issues several hundred `parallel_for`s, each paying a futex wake of up to 15 threads. | measured by `BM_ParallelForDispatch` | 2-10 ms |
| P6 | **Fuse `bias_add` + ReLU** into one forward kernel and one backward kernel that also produces the bias gradient (the conv layer's `finish`), and dropout into a single mask-multiply pass. | 3 passes become 1 in forward and in backward | 2-4 ms |
| P7 | **AVX-512 GEMM kernels**: OpenBLAS in nixpkgs is built without AVX-512 kernels (`OPENBLAS_CORETYPE=SKYLAKEX` is refused, "Core: Haswell"); try an `avx512Support` override, else a packed in-house kernel compiled with `target_clones`. | up to 2x on the GEMM share | 2-5 ms |
| P8 | **Smaller items**: L2 penalty as one fused sum-of-squares op (104 tiny ops forward + backward today), autograd per-node overhead, Adam update bandwidth, `-march` option for the in-house kernels. | measured | 1-3 ms |

Items P1-P4 are algorithmic and keep results bit-identical except P1's
K-split, whose partial-product order is fixed (independent of the thread
count) and therefore deterministic but not identical to one long dot product.

## 3. Results per item

Each row is `squeezenet_train --seconds 10`, augmented, same data and seeds,
measured right after the item landed (so rows are comparable to their
neighbours; the machine's state drifts over hours, see the sweep below).
Threads: 8 (the best count on this machine, see section 6) and 16.

| after | 8 thr ms/step | 16 thr ms/step | epochs/min (8 thr) | vs baseline (8 thr) |
| :--- | ---: | ---: | ---: | ---: |
| baseline | 95.3 | 99.2 | 2.52 | 1.00x |
| P1 parallel GEMM + P2 vectorisable elementwise + P3 blocked max pool + P4 no per-call allocation | - | 46.6 | - | - |
| P5 spin-wait pool, branch-free ReLU backward, Adam capture fix | 37.7 | 43.7 | 6.37 | 2.5x |
| P6a fused `bias_add_relu` op (no gain yet: the loop still compiled to a branch, see section 6) | 38.7 | 42.8 | 6.20 | 2.5x |
| P6b `-fno-trapping-math -fno-math-errno` for the nn kernels (`DATAMUNGE_NN_FAST_FP`) | 33.6 | - | 7.14 | 2.8x |
| P6c one-pass parallel dropout (SplitMix64 blocks, one seed per call) | 32.3 | 38.6 | 7.44 | 3.0x |
| P7a tiled convolution: im2col + GEMM per L2-sized tile on one thread | 29.6 | 35.2 | 8.10 | 3.2x |
| P7b fused `conv2d_bias_relu` with a single-pass backward (dz formed once per tile) | 22.6 | 25.7 | 10.60 | 4.2x |
| P8 OpenBLAS with AVX-512 kernels (`enableAVX512`, Cooperlake kernels picked at run time) | 18.9 | 22.9 | 12.72 | 5.0x |

Phase split of the final step at 8 threads: augment 0.3 ms, forward 8.2 ms,
backward 9.7 ms, update 0.6 ms.

The 18.9 ms row is the best observed; later runs of the same binary gave
24-26 ms at 8 threads while an editor process kept a quarter of a core busy
(the 1-thread time was unchanged at 70-71 ms, so an 8-thread region waits
for whichever worker shares a core with it). Expect 19-25 ms/step (9.3-12.7
epochs/min) on this machine depending on what else runs.

Thread sweep of the final build (`benchmarks/results/squeezenet/final_t*.json`,
one run, measured later with an editor process taking a quarter of a core,
which is why its 8-thread row is slower than the 18.9 ms above):

| threads | 1 | 2 | 4 | 8 | 12 | 16 | 32 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms/step | 70.0 | 40.7 | 27.9 | 24.6 | 27.7 | 29.1 | 32.8 |
| epochs/min | 3.43 | 5.90 | 8.59 | 9.75 | 8.67 | 8.26 | 7.31 |

One thread went from 132 ms to 70 ms; the report's original 4-thread setting
from 100 ms to 28 ms. Flame graph of the final step at 8 threads:
`benchmarks/results/squeezenet/flame_final.svg` (GEMM kernel 25 %, pool
spin while the main thread runs serial sections 14 %, OpenBLAS's beta-zero
memset and packing 20 %, the fused conv epilogue/mask kernels 10 %, concat
copies 7 %).

Correctness: the kernel tests (`tests/cpp/nn_{conv,conv3d,autograd,ops,pooling_variants,tensor,views,layers,training}_tests.cpp`,
209 tests, three new ones for the fused ops and dropout), the model tests that
go through the conv layers (16 files, 134 of 136 pass; the two
`NnVision*` failures are 3e-6 differences against TensorFlow reference values
that fail identically on a pristine checkout of the previous commit built the
same way) and `just test-cuda` (230 tests) all pass.

## 4. Micro-benchmarks

`squeezenet_ops_bench`, wall time per call, baseline
(`ops_baseline*.json`) against final (`ops_final.json`, AVX-512 BLAS); 1 and
16 threads. The baseline numbers were taken with the default glibc
allocator, whose heap trimming made every fresh 4 MB tensor page-fault; the
page faults, not the kernels, were most of the "memory-bound" rows (see
section 6), so the final rows are the honest kernel cost.

| op (shape) | baseline 1 thr | baseline 16 thr | final 1 thr | final 16 thr |
| :--- | ---: | ---: | ---: | ---: |
| conv2d forward conv1 `[32,32,32,3]*[3,3,3,96]` | 3.46 ms | 1.78 ms | 1.21 ms | 0.31 ms |
| conv2d forward fire4 expand3x3 `[32,16,16,32]*[3,3,32,128]` | 4.73 ms | 4.59 ms | 2.28 ms | 0.56 ms |
| conv2d forward fire8 expand3x3 `[32,8,8,64]*[3,3,64,256]` | 4.62 ms | 4.66 ms | 2.56 ms | 0.56 ms |
| conv2d backprop input, fire4 expand3x3 | 4.95 ms | 4.93 ms | 2.42 ms | 0.54 ms |
| conv2d backprop filter, fire4 expand3x3 | 4.98 ms | 4.66 ms | 2.33 ms | 0.60 ms |
| conv2d backprop filter, conv1 | 2.57 ms | 2.33 ms | 1.55 ms | 0.35 ms |
| fused conv2d + bias + ReLU forward, fire4 expand3x3 | - | - | 2.42 ms | 0.52 ms |
| fused conv2d + bias + ReLU forward + backward (dx, dw, db) | 7.70 ms (composed) | 1.99 ms (composed) | 7.48 ms | 1.81 ms |
| bias_add + relu `[32,16,16,128]` | 235 us | 229 us | 71 us (fused) | 60 us (fused) |
| bias_add + relu backward (incl. bias reduction) | 6.73 ms | 1.86 ms | 325 us (fused) | 157 us (fused) |
| relu backward, contiguous seed, 1 M elements | 6.60 ms | 1.32 ms | 167 us | 199 us |
| relu backward, strided seed (split view) | 7.32 ms | 2.24 ms | 0.33 ms | 0.37 ms |
| max_pool2d forward `[32,33,33,96]` 3x3/2 | 15.6 ms | 2.23 ms | 2.01 ms | 0.36 ms |
| max_pool2d forward + backward | 18.2 ms | 2.97 ms | 3.20 ms | 0.68 ms |
| dropout `[32,4,4,512]` | 1.86 ms | 2.05 ms | 147 us | 39 us |
| concat of two `[32,16,16,64]` | 113 us | 109 us | 80 us | 81 us |
| L2 objective over 26 weights, forward + backward | 308 us | 492 us | 270 us | 335 us |
| Adam step, 729 k parameters | 2.26 ms | 1.46 ms | 0.50 ms | 0.40 ms |
| augmentation, one batch | 280 us | 392 us | 228 us | 266 us |
| one `parallel_for` of 64 trivial chunks | 0.67 us | 8.6 us | 0.63 us | 5.4 us |
| SqueezeNet forward (training) | 85.5 ms | 72.7 ms | 41.0 ms | 23.2 ms |
| SqueezeNet forward + backward | 190 ms | 117 ms | 84.6 ms | 38.8 ms |

The 8-thread columns are in `ops_final.json` (its strided ReLU backward row
predates the last fix: the generic strided map ran 23x slower than the
contiguous one until its row lambda took a local copy of the op, the same
closure-aliasing issue as in section 6).

## 5. Framework comparison

`benchmarks/frameworks/` trains the same network (SqueezeNet v1.0, CIFAR
stem, batch 32, the same augmentation, L2 and Adam) in PyTorch 2.11 and
TensorFlow 2.21; full tables in `benchmarks/results/frameworks/summary.md`.
The CPU rows there were measured while this session was building and
benchmarking on the same machine, so they are lower bounds; the GPU rows are
clean.

| implementation | device | threads | steps/s | img/s | s/epoch |
| :--- | :--- | ---: | ---: | ---: | ---: |
| datamunge, baseline | CPU | 16 | 10.1 | 323 | 24.8 |
| datamunge, baseline | CPU | 8 | 10.5 | 336 | 23.8 |
| **datamunge, final** | CPU | 8 | **52.9** | **1693** | **4.7** |
| datamunge, final | CPU | 4 | 35.8 | 1145 | 7.0 |
| datamunge, final | CPU | 16 | 43.7 | 1395 | 5.7 |
| PyTorch 2.11 (eager) | CPU | 8 | 31.6 | 1011 | 7.9 |
| PyTorch 2.11 (eager) | CPU | 16 | 28.0-32.1 | 896-1027 | 7.8-8.9 |
| TensorFlow 2.21 (graph, inter-op 2) | CPU | 8 | 47.6 | 1522 | 5.3 |
| TensorFlow 2.21 (graph, inter-op 2) | CPU | 16 | 43.8 | 1401 | 5.7 |
| PyTorch 2.11 | RTX 5080 Laptop | - | 224-307 | 7164-9823 | 0.8-1.1 |
| TensorFlow 2.21 (graph) | RTX 5080 Laptop | - | 178-189 | 5695-6058 | 1.3-1.4 |
| TensorFlow 2.21 (XLA) | RTX 5080 Laptop | - | 366 | 11719 | 0.7 |

On the CPU the library went from a third of PyTorch's throughput to above
TensorFlow's graph mode at 8 threads. The GPU numbers are 4-7x higher
still: at batch 32 the step is launch-bound on the GPU (and inference
already runs there in the report), so moving training to the GPU is the next
order of magnitude, not another CPU item. Both frameworks also collapse at
32 threads on this 16-core machine, as the library does.

Training parity: with a constant learning rate the ReLU head dies in all
three implementations (NLL stuck at ln 10); the `--schedule warmup_cosine`
runs mirror the report's schedule and reproduce its loss curve.

## 6. What was learned

- **Trapping math blocks if-conversion.** With GCC's default `-ftrapping-math`
  a float compare may trap, so `v > 0 ? g : 0` compiles to a branch and the
  loop never vectorises; that single flag was worth 10-15x on every mask,
  argmax and Adam loop. `DATAMUNGE_NN_FAST_FP` (default ON) adds
  `-fno-trapping-math -fno-math-errno` to the nn sources only; results are
  bit-identical. Clang already defaults to this.
- **Closures reached through a pointer alias everything.** A `[=]` lambda
  wrapped in `std::function` lives behind a pointer, so a captured loop bound
  can alias the stores inside the loop and gets reloaded every iteration.
  Kernels copy every captured scalar and pointer into locals first
  (`detail::parallel_elements`, `bias_relu_backward`, `dropout_kernel`).
- **Micro-benchmarks of allocating ops measured page faults.** glibc trims
  the heap after each free of a multi-MB block, so every iteration of a
  "fill a fresh 4 MB tensor" benchmark took 1000 page faults (2.8 M faults
  per second): 6.6 ms for a 1 M-element ReLU backward of which 6.4 ms were
  faults. The training step itself keeps enough tensors alive that it faults
  only ~700 pages per step, so the real cost of these kernels was always
  the memory traffic, not the compute.
- **The step is memory-bound at 8 threads.** Measured DRAM bandwidth is 61
  GB/s read, 57 GB/s copy (4+ threads); 16 threads get less (41 GB/s copy)
  because the two CCDs contend. The 46 MB of activations plus their
  gradients leave the 2 x 32 MB L3 every step (batch 64 is 1.5x slower per
  image, batch 16 only 3 % faster), so passes over activations cost ~1 ms
  per 46 MB and the wins came from removing passes: the fused conv op turns
  the five passes of conv + bias + ReLU backward (mask, bias sum, two GEMM
  inputs) into one, and the tiled convolution keeps the im2col matrix in L2
  instead of streaming 76 MB of it per forward through DRAM.
- **8 threads beat 16 and 32 on this 16-core/2-CCD machine** for a
  bandwidth-bound step; the pool now defaults to one thread per physical
  core (sysfs topology) instead of hardware concurrency, and the SqueezeNet
  report's training block uses 8.
- **nixpkgs' OpenBLAS has no AVX-512 kernels** (`Core: Haswell` on a Zen 5).
  `openblas.override { enableAVX512 = true; }` (now in `flake.nix` and
  `nix/datamunge.nix`) picks the Cooperlake kernels at run time: 1.8x per
  core on the convolution GEMMs, 16 % on the step.
- **Determinism is kept.** Every split (GEMM K-split, tile partials for the
  filter gradient and bias sum, dropout blocks) is fixed by the geometry, not
  the thread count, and the tests check the fused ops against their
  compositions with contiguous and strided (concat/split) gradients and
  second-order gradients.

Remaining items after this pass, in order of expected gain: write the two
expand convolutions of a fire module straight into the concatenated buffer
(concat copies are 7 % of CPU samples); int32 argmax for max pool; folding
the L2 penalty into the Adam update (the 104 tiny ops are serial time the
pool idles through); training on the GPU. The first two are done in
section 7.

## 7. Second pass: in-house BLAS and the fire module (2026-10-30)

The GEMM, GEMV and vector routines under the training step are now the
library's own kernels and OpenBLAS is no longer linked
([BLAS.md](BLAS.md) has the kernels and the library comparison). This pass
first checked that the swap costs nothing in the step, then profiled again.

### The GEMM backend in the step

Same source and build options, only the GEMM backend differs
(`benchmarks/results/squeezenet/blas_*.json`, 10 s runs measured before the
changes below, alternating backends):

| GEMM backend | 1 thr ms/step | 1 thr epochs/min | 8 thr ms/step | 8 thr epochs/min |
| :--- | ---: | ---: | ---: | ---: |
| nixpkgs OpenBLAS 0.3.30 (Haswell kernels) | 98.3 | 2.44 | - | - |
| OpenBLAS 0.3.30 with AVX-512 kernels (`enableAVX512`) | 71.9 | 3.34 | 25.9, 26.4 | 9.28, 9.11 |
| in-house, AVX-512 (`inhouse-avx512`) | **69.5** | **3.45** | **25.5, 25.4** | **9.41, 9.46** |

The in-house kernels are 3 % faster than the best OpenBLAS build on one
thread and 2-3 % at 8, and 1.41x the OpenBLAS that nixpkgs ships. The
convolution GEMMs are small and skinny (k = 16-576, n = 16-256), where a
fixed 12 x 32 / 12 x 16 AVX-512 tile and no library call overhead matter
more than peak square-matrix throughput.

### Profile-driven changes

| change | what | measured |
| :--- | :--- | :--- |
| Fused fire expand (`conv2d_bias_relu_concat`, layer `conv2d_concat`, used by `Fire`) | Both expand convolutions write their channel bands of one output, so the forward concat copy and the split of its gradient into two strided views (which the backward then copied to contiguous) disappear; the backward accumulates both input gradients into one buffer. | 1 thread -1.3 ms/step |
| Ceil-mode max pool without a padded copy (`CeilMaxPool`) | It appended a row and column of negative infinity with `pad` and pooled VALID; it now pools SAME, whose windows are clipped to the input and whose single pad row sits at the bottom/right, which is exactly the infinite padding. No pad copy, no strided gradient slice. Bit-identical values and gradients (new test against the padded form). | with the two neighbours |
| No zero-fill of the filter-gradient partials | The filter gradient's 16 fixed partial products were zeroed (about 35 MB per step) and then accumulated into; the first GEMM into each partial now overwrites it (beta = 0). | the three together: 1 thread 69.4 -> 63.6 ms/step, 8 threads 25.6 -> 18.6 |
| Max pool keeps a one-byte winning tap | The forward stores the window position (`uint8_t`) of each maximum instead of an int64 index into the input, and the backward scatters from it in a loop that now vectorises (unconditional gradient load, `__restrict`); higher-order gradients convert the taps to the index form. Same values and switches (ties, NaN, SAME/VALID, all strides; tested against the old path). | op benchmark, 1 thread: forward 5.2 -> 3.5 ms, backward 8.8 -> 6.7 ms |
| Threaded GEMM sharing one packed copy of op(B) | See BLAS.md. Neutral on these shapes (each B block is small), so it only runs when op(B) has more than 16 K elements. | step unchanged |

### Result

Alternating before/after runs (ABBA order, 20 s cool-down between runs, the
laptop at 79-94 C; before = this section's starting point with the in-house
BLAS; `benchmarks/results/squeezenet/pass2_*`):

| threads | before ms/step | after ms/step | before epochs/min | after epochs/min | gain |
| ---: | :--- | :--- | ---: | ---: | ---: |
| 8 | 27.7, 27.5, 27.7, 27.2 | 19.6, 19.4, 18.6, 19.0 | 8.7 | **12.5** | +44 % |
| 1 | 70.0, 70.8 | 62.4, 62.4 | 3.41 | **3.85** | +13 % |

Phase split at 8 threads: forward 12.5 -> 6.1 ms, backward 13.8 -> 11.1 ms,
update 0.9 -> 0.8 ms. At 1 thread: forward 27.4 -> 21.5 ms, backward 41.3 ->
39.6 ms. The removed copies were a larger share of the multi-threaded step
because they ran on bandwidth the GEMMs also need.

Against the original baseline (section 1): 95.3 -> 19.2 ms/step at 8 threads
(2.5 -> 12.5 epochs/min, 5.0x), 132.2 -> 62.4 ms/step on one thread (2.1x),
with no external BLAS. Expect 12-13 epochs/min at 8 threads on a quiet
machine; anything else holding a core costs more at 8 threads than at 1 (see
section 3).

Correctness: the BLAS, convolution, pooling, autograd and training kernel
tests (245), the linear algebra tests (35), the SqueezeNet model tests with
the fused `Fire`, and the CUDA tests (231 + 5) pass; the two `NnVision*`
failures noted in section 3 are unchanged. The training loss of the fused
model differs from the unfused one in the last digits (different summation
order), not in its trajectory.

Where the time goes now (`perf`, 8 threads, after the first three changes):
the in-house GEMM kernel 52 % of CPU samples, pool workers spinning through
serial sections 14 %, the fused conv backward epilogue (mask, bias sum) 12 %,
memmove 6 % (many small copies; no single caller above 64 KB), max pool 6 %
(since reduced by the tap form), memset 2 %. The GEMM kernel already runs
at 250-300 GFLOP/s per core on most of the step's shapes, as fast as on
large squares (`benchmarks/results/blas/cpu.md`); the exception is the
squeeze convolutions (n = 16: 186 GFLOP/s, one 16-wide vector per tile row).
Remaining items, in order of expected gain: shortening the serial sections
the pool idles through (autograd bookkeeping, the 104 tiny L2 penalty ops,
which could fold into the Adam update); a narrow-n micro-kernel for the
squeeze products (more rows per tile when n = 16); the small copies;
training on the GPU, whose in-house CUDA GEMM is now 0.7-1.45x cuBLAS
(BLAS.md).

## 8. Third pass: the linear algebra improvement plan (2026-10-31)

The BLAS work of this pass ([BLAS.md](BLAS.md), improvement plan) touched
three things under the step: 512-deep AVX-512 GEMM blocks with C-tile
prefetch, the narrow-output GEMM path that reads A in place (the squeeze
convolutions, n = 16), and the `nn` thread pool. Alternating runs on a
quiet machine (`benchmarks/results/squeezenet/pass3_abba_summary.txt`):

| change | threads | before ms/step | after ms/step | epochs/min |
| :--- | ---: | :--- | :--- | :--- |
| 512-deep blocks + C prefetch (on top of the narrow path) | 8 | 13.5, 13.4, 13.7 | 13.6, 13.4, 13.5 | 17.7 -> 17.7 |
| | 1 | 57.6, 57.4 | 57.6, 57.3 | 4.17 -> 4.18 |
| claim-word pool with home blocks | 8 | 13.8, 14.1, 13.9 | 13.5, 13.6, 13.4 | 17.2 -> **17.8** |
| | 16 | 15.8, 15.6 | 14.7, 14.4 | 15.3 -> 16.5 |

The GEMM changes are neutral here: the step's products are already split
one level up, and the large-square and narrow-shape gains (BLAS.md) are on
shapes the step does not have or that were not its bottleneck. The pool
change is 3 % at 8 threads and 7 % at 16: every parallel region picks up
its job without a lock (dispatch 2.2-2.7 -> 1.2 us) and a range lands on
the same core in consecutive calls, so the elementwise passes over an
activation find it in that core's cache.

The machine matters more than any of these. Section 7 measured 19.2 ms at 8
threads (12.5 epochs/min) while two other processes (editor sessions) used
about 60 % of a core each; on a quiet machine the same build runs at
13.4-13.7 ms and 57.3-57.6 ms on one thread. With this pass: **13.4-13.6
ms/step, 17.6-18.0 epochs/min at 8 threads** (7.1x the 2.5 of section 1;
part of that factor is the quieter machine), still the best thread count
(16 threads: 14.4-14.7 ms).
