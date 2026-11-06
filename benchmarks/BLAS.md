# In-house linear algebra: CPU and CUDA kernels against OpenBLAS, BLIS, LAPACK and cuBLAS

2026-10-30; second pass 2026-10-31 (the improvement plan at the end). The
library no longer needs an external BLAS, LAPACK or cuBLAS.
Matrix products, vector routines and the dense decompositions of
`datamunge::nn` run on in-house kernels by default; the external libraries
remain only as opt-in comparison paths. This page describes the kernels, how
to benchmark them and where they stand.

| What | Default (in-house) | Opt-in comparison |
| :--- | :--- | :--- |
| CPU GEMM, GEMV, DOT, AXPY, SCAL (`float`, `double`) | `src/datamunge/nn/blas_kernels.hpp`, public API `include/datamunge/nn/blas.hpp` | `-DDATAMUNGE_USE_BLAS=ON` (Accelerate, OpenBLAS or another CBLAS for the nn GEMM) |
| Cholesky, LU (solve, det, inverse), triangular solve, QR, SVD, eigh, eig | built-in kernels in `src/datamunge/nn/linalg.cpp`; Cholesky, LU and the triangular solve blocked around the GEMM; QR, SVD and eigh as blocked Householder reductions and divide and conquer (`linalg_householder.cpp`) | `-DDATAMUNGE_USE_LAPACK=ON`, then `set_linalg_backend` (Auto / Builtin / Lapack); the triangular solve uses the library's `dtrsm`; CMake detects an OpenBLAS with 64-bit integers (nixpkgs' default) |
| CUDA matrix product (`compute::Context::matmul`, float and double) | `shaders/compute/blas.cu`, compiled to PTX and embedded (`src/datamunge/compute/cuda_blas_ptx.hpp`); the library needs no nvcc and no CUDA toolkit at run time | `-DDATAMUNGE_ENABLE_CUBLAS=ON` and `DATAMUNGE_CUDA_GEMM=cublas` |

`nn::gemm_backend()` reports what a build uses (`inhouse-avx512`, `inhouse-avx2`,
`inhouse-generic`, or `openblas` / `cblas` / `accelerate` in a comparison build).

## Design

### CPU kernels

- **Packed GEMM** (GotoBLAS/BLIS algorithm). For each block of columns (NC) and
  depth (KC) the block of op(B) is packed into panels of NR columns; for each
  block of rows (MC) op(A) is packed into panels of MR rows. A register-tiled
  micro-kernel keeps an MR x NR tile of C in vector registers for a whole KC
  block (one broadcast of A and NR/lanes vector loads of B per FMA row). Edge
  tiles compute into a scratch tile. Tiles: AVX-512 12 x 32 (float) and
  12 x 16 (double); AVX2 6 x 16 / 6 x 8; baseline 4 x 8 / 4 x 4.
- **Blocking.** AVX-512 depth blocks are up to 512 deep (256 for the other
  instruction sets), and k is cut into equal blocks (576 = 288 + 288, not
  512 + 64). C is read and written once per depth block, so deeper blocks
  halve that traffic, and the micro-kernel prefetches its C tile before the
  depth loop instead of stalling on it afterwards (rows ldc apart, often on
  as many pages). This was the large-square gap of the first version
  (2048 and 4096: 0.84-0.91x of OpenBLAS on one core). A sweep of KC, MC, NC,
  C prefetch and huge pages put it at +5 % on one core at 4096 on an idle
  machine and +9 % (float) to +15-20 % (double) when other processes shared
  the caches, within 2 % on every other benchmark shape. Huge pages
  (`MADV_HUGEPAGE` on the packing buffers) were slower.
- **Narrow outputs** (at most one B panel: n <= 32 float, 16 double on
  AVX-512). Each element of A is then used once per depth block, so packing
  it costs about as much as the products it feeds (a third of the time at
  8192 x 16 x 128). The micro-kernel reads A in place instead (strided
  broadcasts, either transpose), and only the last, partial row tile is
  packed. Every element of C sees the same instructions, so results are
  unchanged. A taller tile for these shapes (24 or 28 x 16, the first idea)
  was slower: 150 GFLOP/s against 270.
- **One source, three instruction sets.** The kernels are written with GCC/Clang
  vector extensions and compiled three times (`blas_avx512.cpp`, `blas_avx2.cpp`,
  `blas_generic.cpp`) under `#pragma GCC target` / `#pragma clang attribute`
  with fused multiply-add contraction, so the library itself stays buildable
  for the baseline ISA and still ships AVX-512 code. `blas.cpp` picks the best
  set the CPU supports at first use; `DATAMUNGE_BLAS_ISA=avx2` or
  `blas::set_isa("generic")` forces another. Everything in the units has
  internal linkage so an AVX-512 inline can never be merged into the baseline path.
- **Threading** (`src/datamunge/nn/gemm.cpp`, on the nn pool):
  - short and wide outputs with a long depth (a convolution's filter gradient)
    split K into a fixed number of partial products, summed in order;
  - when op(B) is larger than 16 K elements, the kernels' threaded driver
    (`gemm_packed_mt`) packs each block of op(B) once, in parallel, and all
    threads share it; rows of C are split into MR-aligned blocks sized for L2
    (and at least 2 MFLOP each), and into panel groups when there are few rows;
  - otherwise rows (or columns) of C are split into one block per thread, each
    packing its own (small) op(B).
  Every split is aligned to the register tile and fixed by the shape, so
  results are bit-identical for any thread count (tested at 1, 2, 3, 7 and 16
  threads with alpha and beta not 0/1).
- **Thread pool** (`src/datamunge/nn/parallel.cpp`, shared by every nn
  kernel). Workers spin for 50 us after a job before sleeping. A job is
  published as one word per participant (the job's tag, the next chunk and
  the end of that participant's block of chunks), and one compare-exchange
  both checks that the job is current and takes a chunk: no allocation and
  no lock per job. Each participant runs its own contiguous block first and
  then takes what is left of the others', so with every thread present a
  range lands on the same core in every call, and data one call left in
  that core's L2 is still there for the next. Before, workers queued on a
  mutex to pick up each job and chunks went to whichever thread asked
  first: an empty 8-chunk job took 2.2-2.7 us to dispatch at 8 threads (now
  1.2 us), and level-1 routines on vectors that fit in the caches were 2-3.5x
  slower than now (AXPY of 2^16 floats 2.3 -> 0.74 us, SCAL of 2^20 floats
  13 -> 3.7 us). Which thread runs a chunk never changes what it computes.
- **GEMV** processes four rows of A per pass in the transposed case. From
  2^17 matrix entries a product is split over y in 64-element blocks, so
  every element is computed as in one serial call. DOT/AXPY/SCAL use eight
  independent vector accumulators. They split into 8 K-element pieces from
  32 K elements and 256 KB of vectors (below that, one call on the calling
  thread is faster at 8 threads even with the vectors in cache). DOT always
  (also on one thread) adds the partial sums of fixed 8 K-element chunks in
  order. Before the first improvement pass DOT summed a single chunk on one thread
  and 256 K-element chunks on several, so its result depended on the thread
  count above 256 K elements.
- **Decompositions.** Cholesky (from n = 65; right-looking, in place, the
  trailing update restricted to the lower triangle by row blocks of 256),
  LU with partial pivoting (from n = 96; each 64-column panel factored in a
  transposed copy, recursively as LAPACK's `dgetrf2` so most panel flops are
  GEMMs, with rank-1 updates only in 8-column leaves; row swaps applied
  afterwards; U12 by substitution; trailing update by one GEMM) and the
  triangular solve with several right-hand sides (from n = 16; one GEMM per
  64-row block, then the diagonal block row by row) put almost all flops
  into the GEMM. On several threads LU looks one panel ahead: the next
  panel's columns are updated first, then one pool task factors that panel
  while the others update the remaining columns. The panel factorisation
  and its transposes do not scale with threads (1.7 of 3.3 ms at n = 1024 on
  8 threads), and the look-ahead hides them (1.3x at 512 and 1024). The same
  look-ahead did not pay for Cholesky, whose next column's L21 solve would
  then run on one thread and outlast the update. Cholesky's diagonal blocks
  and the unblocked kernel work right-looking on L^T, so every update is a
  vectorised axpy over a contiguous row instead of a dot product's serial
  sum (2x at n = 64, the same subtractions in the same order). The row work that remains goes through the BLAS units'
  GEMV/AXPY kernels (`linalg.cpp` itself is compiled for the baseline
  instruction set) and is split over the pool in 64-column blocks, which
  leaves every element's arithmetic unchanged: results are identical at any
  thread count (tested). A single right-hand side uses dot-product or axpy
  forms over contiguous rows.
- **QR, eigh and SVD** (`linalg_householder.cpp`; from min(m, n) = 48 for
  QR, n = 24 for eigh (16 for values only) and min(m, n) = 16 for the SVD,
  the measured break-even points against the unblocked kernels) follow
  LAPACK on column-major copies. QR is `dgeqrf`/`dorgqr`: 32
  reflectors per block, applied in the compact WY form (three GEMMs per
  block). eigh is `dsytrd` (each 32-column panel's rank-2 updates folded into
  two GEMMs) followed by Cuppen's divide and conquer (`dstedc`: deflation,
  the secular equation, Gu-Eisenstat vectors and one pair of GEMMs per merge)
  with implicit QL on leaves of up to 32 rows. The SVD is a QR when m > n,
  then `dgebrd` and a bidiagonal divide and conquer (`dbdsdc`/`dlasd1-3`)
  with one-sided Jacobi on leaves of up to 12 rows; the reflectors are then
  applied to the singular vectors. For values only, implicit QL runs on the
  tridiagonal (eigh) or on the Golub-Kahan tridiagonal of the bidiagonal
  (SVD). Three things make these fast on eight cores. The merges skip the
  zero quarters of the block-diagonal factors (LAPACK's column types). The
  subtrees below about n / (2 x threads) rows run as independent pool tasks.
  The thousands of matrix-vector products in each reduction copy strided
  vectors to contiguous ones and are split over the pool in cache-line
  pieces. Before that copy, the strided products of the bidiagonal reduction
  ran scalar loops and took 34 of the 46 ms of a 512 x 512 SVD. Results do
  not depend on the thread count (tested).

### CUDA kernels

- **Tiles.** Each entry runs one block per BM x BN tile of C on a
  one-dimensional grid, numbered down groups of eight tile rows so blocks that
  run together share A and B panels in L2. Warps form a 2-D grid over the
  block tile and lanes a 2-D grid over the warp tile. Each thread accumulates
  TM x TN outputs made of 4 x 4 (float) or 2 x 2 (double) sub-tiles, strided
  so every shared-memory fragment read and every store of C is a
  conflict-free 128-bit access.
- **Main loop.** A and B are staged k-major in two shared buffers. Each k
  step issues the next tile's global loads (128-bit when aligned) into
  registers, runs the BK outer products while loading the next slice's
  fragments into a second register set, then stores the staged tile and
  synchronises once. Operands stored k-contiguous are scattered into padded
  shared rows (no bank conflicts); edge rows are clamped and the k tail is
  zero-filled, so any shape, transpose and leading dimension works. The four
  transposes are template parameters.
- **Entries.** Float 128 x 256 (256 threads, 8 x 16 outputs each), 128 x 128
  (128 threads, 8 x 16) and 128 x 64 (8 x 8), plus 64 x 64 and 128 x 32
  (4 x 4) for small or narrow outputs; double 64 x 64 (4 x 4) and 64 x 32
  (4 x 2). With 8 x 8 outputs per thread the kernel was bound by
  shared-memory reads (four 128-bit reads per 64 FMAs, about 0.66x of
  cuBLAS); 8 x 16 needs six per 128.
- **Planner.** The host picks the entry and the k split with a cost model.
  One wave of an entry (multiprocessors x resident blocks, from the
  driver's occupancy query) costs a + b x k-chunk microseconds, with
  constants measured on the RTX 5080 Laptop GPU (only their ratios matter
  elsewhere). The candidates are every tile whole; every tile split into
  2-64 k chunks when there are fewer tiles than resident slots; and with
  more tiles, full waves of whole tiles plus only the leftover tiles split.
  A 2048 square, for example, computes 120 of its 128 tiles of 128 x 256
  whole and splits the last 8 seven ways, instead of running a third wave
  of 8 blocks on 60 multiprocessors. Wave quantisation, not the per-tile
  rate (0.88-0.93x of cuBLAS), was the largest single loss before.
- **Deterministic split K.** Split chunks write raw sums to their own
  workspace slab, and a second kernel adds each tile's slabs in chunk order
  and applies alpha and beta. There are no atomics and the order is fixed, so
  results are bit-identical for a given device and shape (tested). Plans are
  cached per shape, and the workspace is capped at 64 MiB.

## Running the benchmarks

| Recipe | Driver | Compares |
| :--- | :--- | :--- |
| `just bench-blas` | `benchmarks/blas/blas_compare.cpp` | in-house (each ISA) against OpenBLAS (nixpkgs, Haswell kernels), OpenBLAS with AVX-512 kernels, BLIS and the netlib reference BLAS, loaded with `dlopen` from the `.#blas-bench` shell; GEMM at squares and SqueezeNet shapes, GEMV, DOT, AXPY, SCAL; float and double; 1, 8 and 16 threads |
| `just bench-blas-cuda` | `benchmarks/blas/cuda_gemm_compare.cpp` | the in-house CUDA kernels against cuBLAS through the same `Context::matmul` call (`CUBLAS_PEDANTIC_MATH`, no TF32) |
| `just bench-linalg` | `benchmarks/blas/linalg_compare.cpp` | built-in decompositions against LAPACK in one `-DDATAMUNGE_USE_LAPACK=ON` build; the run swaps in the AVX-512 OpenBLAS of the `.#blas-bench` shell |
| `just bench-squeezenet` | `benchmarks/nn/squeezenet_train.cpp` | the end-to-end training step (see [SQUEEZENET.md](SQUEEZENET.md)) |

Extra arguments go to the drivers, e.g. `just bench-blas --threads 1,8 --ops gemm --quick`.
Another CBLAS joins the comparison through the library list (an explicit
`--libs` replaces the shell's list), e.g. ATLAS's threaded library:
`DATAMUNGE_BLAS_LIBS="$DATAMUNGE_BLAS_LIBS;atlas=/path/to/libtatlas.so:fixed-threads"`
inside `nix develop .#blas-bench`, then `./build/benchmarks/blas_compare`.
Results: `benchmarks/results/blas/{cpu,cuda,linalg}.json` and the generated
tables `{cpu,cuda,linalg}.md` (`benchmarks/blas/summarize.py`).

`blas_compare` runs each operation, thread count and library in its own
child process (libraries loaded side by side keep competing thread pools),
and `linalg_compare` each thread count. The children start with
`OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS`, `OMP_NUM_THREADS` (and
`BLIS_NUM_THREADS`) set, because OpenBLAS reads them when it is loaded:
a shell that exports `OPENBLAS_NUM_THREADS=1` (as this machine's profile
does) otherwise silently runs every "8-thread" OpenBLAS row on one thread,
and with the variables unset nixpkgs' OpenMP build of OpenBLAS ignores a
later `openblas_set_num_threads(1)` and runs its "1-thread" LAPACK rows on
several cores. `blas_compare` warns when a library reports a different
thread count than requested.

## Results

Reference machine: AMD Ryzen 9 9955HX (Zen 5, 16 cores / 32 threads, AVX-512
with a full 512-bit datapath), 60 GB RAM, RTX 5080 Laptop GPU, NixOS, GCC
15.2; 2026-10-30 and 2026-10-31. A laptop: it throttles at 90-96 C, so long runs drift by
10-20 % and the comparisons below alternate backends within each case. The
competitors are the libraries nixpkgs ships: OpenBLAS 0.3.30 (its
`DYNAMIC_ARCH` build picks the Haswell AVX2 kernels on this CPU), the same
OpenBLAS rebuilt with `enableAVX512` (Cooperlake kernels), BLIS 2.0
(generic `x86_64` configuration) and the netlib reference BLAS 3.12.

### CPU kernels against OpenBLAS, BLIS and the reference BLAS

`just bench-blas`; GFLOP/s, median of at least five calls, each operation,
thread count and library in its own process; 2026-10-31, with every change
of the second pass (the first version's run is kept in `cpu_baseline.md`;
its 16-thread rows are not valid). Every in-house instruction set, every
library and the vector routines are in `benchmarks/results/blas/cpu.md`.
"ratio" is the in-house AVX-512 kernel over the best library (the AVX-512
OpenBLAS, except three 16-thread rows where the Haswell build is ahead).
The SqueezeNet shapes are the step's convolution GEMMs (forward im2col
products, the input gradient `dx` and the filter gradient `dw`; `tile` is
one per-thread tile of the tiled convolution).

| float GEMM (m x n x k) | in-house | OpenBLAS (Haswell) | OpenBLAS AVX-512 | BLIS | ratio | in-house 8 thr | OpenBLAS AVX-512 8 thr | ratio 8 thr | in-house 16 thr | OpenBLAS AVX-512 16 thr | ratio 16 thr |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| square 256 | 294 | 160 | 292 | 53 | 1.01 | 941 | 895 | 1.05 | 1017 | 633 | 1.39 |
| square 512 | 310 | 163 | 298 | 53 | 1.04 | 1624 | 1689 | 0.96 | 1761 | 1535 | 1.15 |
| square 1024 | 313 | 167 | 302 | 53 | 1.03 | 1878 | 1917 | 0.98 | 2142 | 2140 | 1.00 |
| square 2048 | 304 | 165 | 301 | 52 | 1.01 | 1764 | 1920 | 0.92 | 2488 | 2524 | 0.99 |
| square 4096 | 295 | 162 | 302 | 53 | 0.98 | 1771 | 1928 | 0.92 | 2543 | 2785 | 0.91 |
| conv1 fwd 32768x64x27 | 260 | 141 | 229 | 34 | 1.14 | 1675 | 1486 | 1.13 | 2456 | 2165 | 1.13 |
| squeeze fwd 8192x16x128 | 208 | 130 | 197 | 41 | 1.06 | 1400 | 1275 | 1.10 | 2087 | 1670 | 1.25 |
| expand1x1 fwd 8192x64x16 | 254 | 131 | 199 | 26 | 1.27 | 1275 | 1401 | 0.91 | 1550 | 1195 | 1.30 |
| expand3x3 fwd 8192x64x144 | 279 | 155 | 270 | 48 | 1.03 | 1886 | 1770 | 1.07 | 2944 | 2543 | 1.16 |
| fire8 fwd 2048x256x576 | 307 | 166 | 305 | 51 | 1.01 | 1675 | 1808 | 0.93 | 2500 | 2476 | 1.01 |
| expand3x3 dx 8192x144x64 NT | 283 | 157 | 273 | 45 | 1.04 | 1919 | 1804 | 1.06 | 2734 | 2397 | 1.14 |
| expand3x3 dw 144x64x8192 TN | 255 | 151 | 260 | 50 | 0.98 | 1417 | 726 | 1.95 | 1376 | 581 | 1.74 |
| tile 128x64x144 | 255 | 153 | 260 | 46 | 0.98 | 645 | 583 | 1.11 | 553 | 276 | 1.90 |

| double GEMM (m x n x k) | in-house | OpenBLAS (Haswell) | OpenBLAS AVX-512 | BLIS | ratio | in-house 8 thr | OpenBLAS AVX-512 8 thr | ratio 8 thr | in-house 16 thr | OpenBLAS AVX-512 16 thr | ratio 16 thr |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| square 256 | 142 | 75 | 122 | 26 | 1.16 | 806 | 718 | 1.12 | 575 | 519 | 1.11 |
| square 512 | 152 | 78 | 128 | 26 | 1.19 | 782 | 803 | 0.97 | 821 | 728 | 1.13 |
| square 1024 | 151 | 80 | 132 | 26 | 1.15 | 912 | 860 | 1.06 | 943 | 1069 | 0.88 |
| square 2048 | 148 | 80 | 132 | 26 | 1.12 | 781 | 843 | 0.93 | 1064 | 1225 | 0.87 |
| square 4096 | 145 | 81 | 135 | 26 | 1.08 | 818 | 848 | 0.96 | 1188 | 1089 | 1.09 |
| conv1 fwd 32768x64x27 | 127 | 64 | 94 | 17 | 1.36 | 849 | 600 | 1.42 | 1322 | 904 | 1.46 |
| squeeze fwd 8192x16x128 | 152 | 58 | 78 | 22 | 1.95 | 974 | 510 | 1.91 | 1494 | 739 | 2.02 |
| expand1x1 fwd 8192x64x16 | 130 | 63 | 82 | 14 | 1.58 | 790 | 569 | 1.39 | 946 | 717 | 1.32 |
| expand3x3 fwd 8192x64x144 | 140 | 74 | 108 | 25 | 1.30 | 914 | 692 | 1.32 | 1408 | 1054 | 1.34 |
| fire8 fwd 2048x256x576 | 153 | 81 | 127 | 26 | 1.21 | 873 | 815 | 1.07 | 1210 | 1203 | 1.01 |
| expand3x3 dx 8192x144x64 NT | 150 | 77 | 111 | 23 | 1.35 | 971 | 725 | 1.34 | 1478 | 1067 | 1.38 |
| expand3x3 dw 144x64x8192 TN | 125 | 72 | 104 | 25 | 1.20 | 639 | 461 | 1.38 | 617 | 353 | 1.75 |
| tile 128x64x144 | 132 | 73 | 106 | 24 | 1.24 | 360 | 300 | 1.20 | 320 | 209 | 1.53 |

- **One core:** 0.98-1.27x of the AVX-512 OpenBLAS in float and 1.08-1.95x
  in double, and 1.6-2.6x the OpenBLAS nixpkgs ships. Large squares, the
  first version's gap (2048 and 4096: 0.84-0.91x), are now 0.98-1.01x
  (float) and 1.08-1.12x (double) after the deeper blocks and the C
  prefetch; the narrow squeeze product is 1.06x (was 0.97x).
- **Eight cores:** large squares 0.92-1.05x (float) and 0.93-1.12x
  (double), the SqueezeNet shapes 0.91-1.95x, ahead on the filter gradient
  (1.4-2.0x: the k split) and now on `tile` (1.11-1.20x, was 0.45x: the
  new pool's dispatch). Repeated 8-thread runs of one small product are
  bimodal, with both pools: square 256 takes 22 or 40 us depending on the
  run, likely where the scheduler puts the threads (across the two
  8-core dies or not); single rows can move by 0.6-1.4x between runs.
- **Sixteen cores** (each library in its own process; the first version ran
  them in one process, whose competing thread pools made those rows
  meaningless): 0.87-1.39x on squares and 1.0-2.0x on the SqueezeNet
  shapes. Large products scale past 8 threads (float 4096: 1770 -> 2540
  GFLOP/s in-house, 1930 -> 2790 OpenBLAS), but the SqueezeNet step does
  not (8 threads stays its best count).
- **Vector routines** (GEMV, DOT, AXPY, SCAL), one core: 0.85-1.09x for
  the level-1 routines and 0.72-1.20x for GEMV; the gap is the
  non-transposed float GEMV on matrices larger than the caches (4096 and
  8192: 0.72-0.78x; OpenBLAS streams 65 GB/s from one core, the in-house
  kernel 47). At 8 threads, with the new pool and split threshold, the
  level-1 routines on 2^16-2^20 elements went from 0.2-0.5x to 0.93-6.9x
  (OpenBLAS runs the float DOT on one thread at that size), and GEMV is
  0.73-1.97x (the transposed 1024 product is the low end). At 16 threads, 2^16-element
  AXPY and SCAL are 0.33-0.66x: the split threshold is tuned for 8 threads,
  and OpenBLAS keeps them on one. Above the caches all libraries run at
  memory bandwidth, except that OpenBLAS's static split keeps a 2^24-float
  SCAL in the two dies' L3 at 16 threads (0.3x).
- **BLIS** is nixpkgs' generic `x86_64` build without the Zen kernels (25-50
  GFLOP/s on one core), so it is not representative of BLIS on this CPU.
  The **reference BLAS** runs at 7-9 GFLOP/s (35x slower) and only on one
  thread.

### Dense decompositions against LAPACK

`just bench-linalg`: float64, one matrix per call, ms per call (median of
five interleaved rounds, each the median of at least three calls); LAPACK is
the AVX-512 OpenBLAS 0.3.30's. Speed ratio = LAPACK time / in-house time
(above 1: in-house faster). All sizes (16-1024) in
`benchmarks/results/blas/linalg.md`; 2026-10-31, with every change of the
second pass (the first version: `linalg_baseline.md`).

| n = 1024 | in-house 1 thr | LAPACK 1 thr | speed ratio | in-house 8 thr | LAPACK 8 thr | speed ratio |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cholesky | 5.26 | 4.83 | 0.92 | 1.84 | 1.32 | 0.72 |
| LU solve, 1 right-hand side | 8.25 | 8.42 | 1.02 | 2.64 | 2.59 | 0.98 |
| LU solve, n right-hand sides | 30.10 | 35.90 | 1.19 | 8.03 | 7.60 | 0.95 |
| slogdet (LU) | 8.08 | 8.31 | 1.03 | 2.54 | 2.51 | 0.99 |
| triangular solve, 1 right-hand side | 0.05 | 0.20 | 4.12 | 0.04 | 0.20 | 5.55 |
| triangular solve, n right-hand sides | 11.19 | 11.06 | 0.99 | 2.59 | 2.37 | 0.92 |
| QR | 30.55 | 51.49 | 1.69 | 9.13 | 29.65 | 3.25 |
| eigh | 93.57 | 96.06 | 1.03 | 39.92 | 56.25 | 1.41 |
| SVD | 180.29 | 233.41 | 1.29 | 86.97 | 157.41 | 1.81 |
| eigenvalues only (`eigvalsh`) | 46.55 | 36.48 | 0.78 | 24.39 | 18.04 | 0.74 |
| singular values only | 113.71 | 106.12 | 0.93 | 72.43 | 73.71 | 1.02 |

- **QR, eigh and the SVD** were the unblocked algorithms in the first
  version: 0.26-0.48x, 0.12-0.27x and 0.05-0.08x of LAPACK from n = 256,
  falling with size. As blocked Householder reductions with divide and
  conquer they now run at 1.7-2.4x, 1.0-1.3x and 1.1-1.4x on one core and
  2.8-3.7x, 1.4-2.0x and 1.4-2.0x at 8 threads from n = 96-128 (at n = 512
  on one core: QR 26.2 -> 4.2 ms, eigh 111.7 -> 14.8 ms, SVD 715.9 ->
  29.2 ms). The threaded GEMM carries their flops, and the divide-and-conquer
  subtrees run as independent pool tasks. Below n = 96 they are at
  0.93-1.85x.
- **Values only** still trail LAPACK at some sizes: `eigvalsh` 0.71-0.78x
  at n = 512-1024 (the reduction to tridiagonal form reads the whole
  matrix where `dsytrd`'s `dsymv` reads half), the singular values 0.41-0.67x
  up to n = 256 (implicit QL on the Golub-Kahan tridiagonal; LAPACK uses
  dqds) and 0.90-1.02x above.
- **Cholesky and LU, one core:** at parity from n = 48 (0.86-1.12x; LU
  0.86-0.91x at 64-128) and faster below (1.0-1.6x at 16-32: no call
  overhead). Solves with many right-hand sides are 1.2-2.4x faster: the
  in-house LU and triangular solve are each a little faster, and the LAPACK
  path also pays for `dgetrs`'s row interchanges and for two transposes of
  the right-hand sides into and out of the column-major layout LAPACK needs.
  (The first version of the comparison wrapper made those transposes with
  strided loops and overstated the lead as 1.4-2.2x; they are now
  cache-tiled.) A single right-hand side triangular solve is 1.1-5.6x faster
  (no call overhead and a dot-product form over contiguous rows).
- **Cholesky and LU, eight cores:** LU (one right-hand side, slogdet)
  0.66-0.99x of LAPACK from n = 64, 0.98-0.99x at 1024 with the look-ahead
  (the first version: 0.54-0.85x, 0.69-0.70x at 1024); Cholesky at parity up
  to n = 256 and 0.71-0.72x at 512-1024 (was 0.6-0.8x); the triangular solve
  with many right-hand sides 0.6-0.9x from n = 256. The new thread pool
  added most of the Cholesky gain at 256-1024 (0.61-0.73x the run before).
- **Auto routing.** In builds with LAPACK, `set_linalg_backend(Auto)` (the
  default) sends a matrix to LAPACK where these measurements put LAPACK
  ahead: Cholesky from n = 384, LU from 64, the many right-hand-side
  triangular solve from 96, eigenvalues only from 384 and singular values
  only from 8; QR, eigh and the SVD always stay built-in, and so does a
  single right-hand side triangular solve. The thresholds for Apple's
  Accelerate were measured against the first version's kernels and need
  re-measuring.
- **Run-to-run spread at 8 threads.** Runs in one session agree within
  about 10 % on most rows (up to 26 %; the small 8-thread solves move more,
  e.g. the many right-hand-side triangular solve at n = 32: 1.49x in one
  run, 2.47x in the next), but absolute 8-thread times moved by up to 1.7x
  between sessions for both libraries alike. On this laptop a pool worker
  that shares a core with another process (two editor processes at 20-60 %
  of a core each, in some runs) stalls every dispatch, and the LU issues
  many. Compare ratios, not absolute 8-thread times, across runs.

### CUDA matrix product against cuBLAS

RTX 5080 Laptop GPU (driver 580.142), `just bench-blas-cuda`, GFLOP/s, median
of at least five calls, in-house and cuBLAS alternating within each case,
cuBLAS with `CUBLAS_PEDANTIC_MATH` (no TF32, no reduced-precision tricks).
Full table: `benchmarks/results/blas/cuda.md`; the first in-house version is
kept in `cuda_baseline.md` and the first tuning pass in `cuda_pass1.md`.

| case (m x n x k) | float: first version | float: pass 1 | float: now | float: cuBLAS | now / cuBLAS | pass 1 / cuBLAS | double: now / cuBLAS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| square 256 | 700 | 2063 | 2185 | 2347 | 0.93 | 1.00 | 1.60 |
| square 512 | 3467 | 6552 | 10085 | 7819 | 1.29 | 0.83 | 1.28 |
| square 1024 | 4221 | 10596 | 12621 | 13421 | 0.94 | 0.79 | 1.14 |
| square 2048 | 10803 | 12610 | 17678 | 16887 | 1.05 | 0.71 | 1.12 |
| square 4096 | 9434 | 12803 | 16855 | 17029 | 0.99 | 0.76 | - |
| conv1 forward 32768 x 64 x 27 | 3192 | 3629 | 4221 | 3381 | 1.25 | 1.01 | 1.12 |
| squeeze forward 8192 x 16 x 128 | 817 | 1556 | 1889 | 1717 | 1.10 | 0.90 | 1.21 |
| expand 1x1 forward 8192 x 64 x 16 | 1276 | 2001 | 1932 | 1688 | 1.14 | 1.21 | 1.18 |
| expand 3x3 forward 8192 x 64 x 144 | 3399 | 5980 | 7454 | 6433 | 1.16 | 0.93 | 1.28 |
| fire8 forward 2048 x 256 x 576 | 6864 | 7956 | 11824 | 10311 | 1.15 | 0.73 | 1.18 |
| fire squeeze 2048 x 32 x 512 | 920 | 2734 | 3657 | 2985 | 1.22 | 0.92 | 1.85 |
| expand 3x3 input gradient (NT) | 3803 | 4620 | 6160 | 5305 | 1.16 | 0.85 | 1.11 |
| expand 3x3 filter gradient (TN, k = 8192) | 161 | 4759 | 5281 | 6690 | 0.79 | 0.71 | 0.93 |

- **First pass** (k splits, the 128 x 32 tile for narrow outputs, the big
  tile with a k split for wide ones): from 0.02-1.0x of cuBLAS to
  0.71-1.45x.
- **Second pass** (128-bit loads, warp tiling with 8 x 16 outputs per
  thread and register double buffering, the 128 x 256 tile, the
  occupancy-aware planner that splits only the leftover tiles): large
  single-precision products from 0.71-0.83x to 0.94-1.29x (2048: 1.05,
  4096: 0.99), the SqueezeNet shapes to 1.10-1.25x, double precision to
  1.11-1.85x. The exception is the filter gradient (three output tiles,
  k = 8192) at 0.79-0.91x (0.93x in double), bound by launch and reduction
  latency. Square 256 (0.93-1.0x) is bound by launch overhead.
- **Run-to-run spread.** This power-capped laptop GPU (80-105 W) moves by
  15-30 % between runs: a second full run, back to back, put square 1024 at
  1.19x instead of 0.94x and the filter gradient at 0.91x. Compare ratios.
  A 105 W sustained run at 4096 measured 17.4 against 18.2 TFLOP/s (0.96x).
- **Accuracy.** Worst float error 1.0e-5 (square 2048; cuBLAS 6.6e-6), worst
  double error 8.4e-15 (cuBLAS 7.4e-15), identical across runs.
- **Module size.** The four transpose variants of every entry make the
  embedded PTX 2.3 MB (was 243 KB): the first, uncached load JIT-compiles in
  2.6 s (was 0.22 s; cached loads take 4 ms). A variant with the transposes
  as runtime flags is 0.66 MB and 0.65 s, and was within run-to-run noise
  of this one in a single comparison (see the next steps).

### SqueezeNet training step

The end-to-end check (details in [SQUEEZENET.md](SQUEEZENET.md), section 7):
the same training step with only the GEMM backend swapped.

| GEMM backend | 1 thread ms/step | 8 threads ms/step |
| :--- | ---: | ---: |
| nixpkgs OpenBLAS (Haswell kernels) | 98.3 | - |
| OpenBLAS with AVX-512 kernels | 71.9 | 25.9, 26.4 |
| in-house AVX-512 | **69.5** | **25.5, 25.4** |

With the step-level changes of the same pass on top, the step measured 62.4
ms on one thread and 19.2 ms at 8 threads (12.5 epochs per minute, against
2.5 at the start of the SqueezeNet work), with two other processes using
about 60 % of a core each. On a quiet machine that build runs at 13.4-13.7
ms (8 threads) and 57.3-57.6 ms (1 thread). The second pass's GEMM changes
are neutral in the step; the new thread pool takes it to 13.4-13.6 ms at
8 threads (**17.6-18.0 epochs per minute**, +3 %) and 14.4-14.7 ms at 16
(+7 %); SQUEEZENET.md, section 8.

### ATLAS

ATLAS is not in the tables. Current nixpkgs has no ATLAS package; the
NixOS 18.09 build (3.10.3, still on the binary cache) ships static archives
only, and a shared library linked from them needs an executable stack,
which glibc 2.41+ refuses to `dlopen` ("cannot enable executable stack as
shared object requires"). Clearing the flag (`patchelf --clear-execstack`)
or building ATLAS from source should work, but the automated session that
produced these numbers was not permitted to run that downloaded code. To
add it locally:

```sh
nix develop .#blas-bench
# a libtatlas.so / libsatlas.so built for this machine (ATLAS tunes itself at build time)
export DATAMUNGE_BLAS_LIBS="$DATAMUNGE_BLAS_LIBS;atlas=/path/to/libtatlas.so:fixed-threads"
./build/benchmarks/blas_compare --threads 1,8 --json atlas.json
python3 benchmarks/blas/summarize.py atlas.json
```

`:fixed-threads` because ATLAS fixes its thread count at build time. ATLAS
tunes its kernels to the build machine, so a build made elsewhere (like the
cached one) would understate it.

## Improvement plan

The first version's plan had seven items, ordered by expected gain; the
second pass (2026-10-31) worked through all of them. Status, then what is
left.

1. **QR, symmetric eigensolver and SVD: done.** Blocked Householder
   reductions with the compact WY form, `dsytrd`/`dgebrd`-style panels and
   divide and conquer (see Design). From 0.26-0.48x (QR), 0.12-0.27x (eigh)
   and 0.05-0.08x (SVD) of LAPACK to 1.7-2.4x, 1.0-1.3x and 1.1-1.4x on one
   core and 2.8-3.7x, 1.4-2.0x and 1.4-2.0x at 8 threads (n >= 96-128).
   `set_linalg_backend(Auto)` no longer sends them to LAPACK.
2. **CUDA single-precision GEMM on large outputs: done.** 128-bit loads,
   warp tiling with register double buffering, a 128 x 256 tile and an
   occupancy-aware planner: large squares from 0.71-0.83x of cuBLAS to
   0.94-1.29x, the SqueezeNet shapes 1.10-1.25x, double 1.11-1.85x.
3. **Decompositions at 8 threads: partly done.** LU looks one panel ahead
   (1.3x at n = 512 and 1024) and the new pool shortened every dispatch: LU
   0.66-0.99x of LAPACK from n = 64 (was 0.54-0.85x), Cholesky at parity up
   to 256 and 0.71-0.72x at 512-1024 (was 0.6-0.8x), with vectorised
   diagonal blocks (small sizes from 0.4-0.6x to 0.9-1.2x) and new cutoffs
   (blocked Cholesky from 65, LU from 96, the many right-hand-side
   triangular solve from 16). Still serial at n = 1024 on 8 threads: the LU
   panel factorisation (1.2 ms) and its transposed copies (0.5 ms), which
   the look-ahead hides, and Cholesky's transposed copies around the L21
   solve (0.5 ms of 1.8), which it does not.
4. **CPU GEMM on large squares, one core: done.** 512-deep blocks, equal
   depth blocks and C-tile prefetch: square 2048 and 4096 from 0.84-0.91x to
   0.98-1.01x (float) and 1.08-1.12x (double).
5. **Small products and vector routines at 8 threads: done.** The pool
   rewrite (claim words, home blocks) and a split threshold for the level-1
   routines: `tile` from 0.45x to 1.11-1.20x, level-1 on 2^16-2^20 elements
   from 0.2-0.5x to 0.93-6.9x, GEMV 1024 from 0.6-0.8x to 0.73-1.21x; the
   SqueezeNet step +3 % at 8 threads and +7 % at 16 (SQUEEZENET.md,
   section 8).
6. **Narrow outputs: done.** A read in place when B is one panel: 1.3-2x
   on narrow shapes, squeeze 8192 x 16 x 128 from 0.97x to 1.06x on one
   core (1.95x in double). The taller-tile idea was slower.
7. **The 16-thread comparison: done.** Every operation, thread count and
   library runs in its own process; the 16-thread columns are now in the
   tables above.

### Next steps

In order of expected gain:

1. **Values-only eigensolver and SVD** (`linalg.md`: `eigvalsh`
   0.71-0.78x at n = 512-1024, singular values 0.41-0.67x up to n = 256).
   A symmetric matrix-vector product that reads one triangle (`dsymv`) in
   the tridiagonal reduction, and dqds (`dlasq1-6`) for the singular values
   of the bidiagonal. Until then, Auto sends both to LAPACK where LAPACK is
   built in.
2. **Cholesky and the many right-hand-side triangular solve at 8 threads**
   (0.71-0.72x and 0.6-0.9x from n = 256-512). Candidates: a panel kernel
   on the row-major layout (no transposed copies), a parallel L21 solve,
   and recursive Cholesky (`dpotrf2`) so that most of the panel's flops
   become GEMMs.
3. **CUDA module size: a decision to make.** The four transpose variants
   of every entry made the embedded PTX 2.3 MB and the first, uncached JIT
   2.6 s (was 0.22 s; cached loads take 4 ms). The variant with the
   transposes as runtime flags (0.66 MB, 0.65 s) was within run-to-run
   noise in a single comparison. Recommended: confirm with a longer
   alternating run and switch to it. Then the filter gradient (three output
   tiles, k = 8192: 0.79-0.91x; a fused split-k reduction), `cp.async`
   staging (a second PTX module for sm_80 and newer) and TF32 as an opt-in.
4. **One-core GEMV on large matrices** (`cpu.md`: non-transposed float 4096
   and 8192, 0.72-0.78x: 47 against 65 GB/s). Software prefetch further
   ahead, or more rows per pass.
5. **Thread placement and thread-count-aware thresholds.** Repeated
   8-thread runs of one small product are bimodal (square 256: 22 or
   40 us), likely depending on whether the scheduler spreads the threads
   over both dies; pinning the pool to one die is worth a measurement. The
   level-1 split threshold is tuned for 8 threads (2^16-element AXPY and
   SCAL at 16 threads: 0.33-0.66x).
6. **ATLAS** is still unmeasured (see above).

The Auto thresholds for Apple's Accelerate (`lapack_min_size`, `__APPLE__`
branch of `linalg.cpp`) were measured against the first version's kernels
and need re-measuring on a Mac.
