// In-house CUDA GEMM kernels behind compute::Context::matmul.
//
// C[m,n] = alpha * op(A)[m,k] op(B)[k,n] + beta * C[m,n] on row-major matrices with the
// compute::MatrixProduct conventions: stored A is [m,k] (lda >= k) or, when transpose_a is set,
// [k,m] (lda >= m); stored B is [k,n] (ldb >= n) or, when transpose_b is set, [n,k] (ldb >= k);
// C is [m,n] (ldc >= n). The Context API passes packed matrices with alpha = 1, beta = 0;
// beta = 0 never reads C.
//
// Compiled to PTX by scripts/prebuild_nn_cuda_kernels.py (`just prebuild-nn-cuda-kernels`) and
// embedded in src/datamunge/compute/cuda_blas_ptx.hpp; the library never needs nvcc. Unlike
// shaders/nn/inference.cu this file is compiled with fused multiply-add enabled.
//
// Every entry takes its buffers first, then one by-value parameter block, matching
// compute::KernelSource. A GEMM block of NW warps computes one BM x BN output tile (or one k chunk
// of it, see split-K below) over a one-dimensional grid. Tiles are numbered down groups of eight
// tile rows before moving right, so the blocks resident at one time share A and B panels in L2.
// The host (src/datamunge/compute/cuda.cpp) picks the entry, the number of whole tiles and the
// split from a cost model of the shape.
//
// Register tiling: the warps form a WGM x (NW / WGM) grid over the block tile and the 32 lanes of
// a warp an LM x (32 / LM) grid over the warp tile. Each thread accumulates TM x TN outputs made
// of V x V sub-tiles (V = 4 floats or 2 doubles, one 128-bit vector) strided LM * V rows and
// (32 / LM) * V columns apart, so a warp's fragment reads are whole 128-bit vectors of contiguous
// shared memory (no bank conflicts) and its stores of C are 128-bit too. The large float tiles
// give each thread 8 x 16 outputs (6 fragment vectors per 128 multiply-adds), which keeps the
// shared-memory pipe below the FMA rate; the 4 x 4 tiles of the narrow entries are for small or
// thin outputs, where more blocks matter more than that ratio.
//
// Main loop: A and B tiles are staged in shared memory k-major (As[kk][row], Bs[kk][col]) with
// two buffers. Each k step issues the next tile's global loads into registers first, then runs
// the BK outer products of the current tile while it also loads the next k slice's fragments
// from shared memory into a second register set (register double buffering); after the last
// slice the staged registers go to the other buffer and one barrier per k step publishes them.
// Global loads are 128-bit when the operand's
// base address and leading dimension allow it; operands stored k-contiguous (A untransposed, B
// transposed) are scattered k-major into shared memory, and their rows are padded by PAD
// elements so those scattered stores hit distinct banks. Rows beyond the matrix edge are clamped
// onto the last row (their results are never stored) and k beyond the range is zero-filled, so
// any m, n, k and leading dimension work; only the last, partial k step and misaligned operands
// take per-element checked loads.
//
// Transposes: only the staging (global loads and their shared-memory stores) depends on the
// operands' layouts. An entry either compiles each layout separately and picks one per block (its
// main loop then carries no layout branches) or reads the layouts from uniform runtime flags in
// one shared copy of the loop (Stage / StageR below). Compiling the four layouts of every entry
// made this module 2.3 MB of PTX, JIT-compiled in 2.6 s on first, uncached use. Measured with
// alternating runs on an RTX 5080 Laptop GPU against that build: the 4 x 4 entries are as fast or
// faster with flags (6 % on the NT input gradient; one copy fits the 80-register cap without
// spills), but the 8 x 16 and 8 x 8 entries lose 4-7 % on untransposed products. They run at
// their register caps, and the driver's code for their main loop shifts with anything else in
// the function: even a build of only the untransposed layout, or the same four layouts with a
// shorter store, lost 4-9 %. So gemm_f32_128x256 keeps its four compiled layouts unchanged,
// gemm_f32_128x64 compiles B's layout only (two copies), gemm_f32_128x128 and the 4 x 4 entries
// use flags (128x128 gives up about 4 % on untransposed products), and gemm_f64_64x32 keeps four
// (with flags it fits three blocks per multiprocessor instead of two and runs slower per block).
// The module is 1.28 MB and JIT-compiles in 1.4 s. A 64 x 64 double entry was dropped: the
// planner never chose it.
//
// Split-K: blocks [0, whole) compute tiles [0, whole) whole. Every later tile is cut into
// `splits` k chunks of `kchunk`, one block each, which write raw sums to their own BM x BN slab of
// a workspace; reduce_<type> then adds each tile's slabs (in groups of eight chunks, the groups
// in order) and applies alpha/beta.
// whole = 0 splits every tile (a small output with a long k: the filter gradient of a
// convolution); 0 < whole < tiles runs full waves of whole tiles and splits only the tiles that
// would otherwise form a last, mostly idle wave (a 2048 square has 128 tiles of 128 x 256 for 60
// multiprocessors: two waves, then 8 tiles split 7 ways instead of a third wave of 8 blocks).
// Every output element is one register accumulation in increasing k order per chunk and the
// chunks are added in a fixed order, so for a given device and shape the result is deterministic.
// Fusing that reduction into the GEMM (the last chunk block of a group, found with a counter, adds
// the group's slabs) was measured too and was slower: 38-45 against 29-34 us on the filter
// gradient, whose 57 slabs then pass through one block per group and per tile.

typedef unsigned long long u64;
typedef unsigned int       u32;

/// By-value GEMM parameter block; src/datamunge/compute/cuda.cpp mirrors and static_asserts it.
struct Gemm {
  u64    m, n, k;
  u64    lda, ldb, ldc;
  u32    transpose_a, transpose_b;
  double alpha, beta;
  u64    whole;  ///< Tiles [0, whole) are computed whole, one block each.
  u64    splits; ///< Each later tile is split into this many k chunks (blocks).
  u64    kchunk; ///< k per chunk, a multiple of 16.
};

/// By-value reduction parameter block (cuda.cpp mirrors it): the split tiles of one product.
struct Reduce {
  u64    m, n, ldc;
  u64    bm, bn;   ///< Tile shape of the GEMM entry.
  u64    whole;    ///< First split tile (as Gemm::whole).
  u64    splits;   ///< Chunks per split tile.
  u64    count;    ///< Vectors to reduce: (tiles - whole) * bm * bn / V.
  double alpha, beta;
};

/// Origin of tile `t` of a BM x BN tiling. Consecutive tiles walk down a group of G tile rows,
/// then right, then on to the next group, so the blocks resident at one time share A and B
/// panels in L2.
__device__ __forceinline__ void tile_origin(u64 t, u64 m, u64 n, u64 bm, u64 bn, u64& m0, u64& n0) {
  constexpr u32 G       = 8;
  const u32     tiles_n = u32((n + bn - 1) / bn), tiles_m = u32((m + bm - 1) / bm);
  const u32     tile = u32(t), group = tile / (G * tiles_n), first = group * G;
  const u32     height = tiles_m - first < G ? tiles_m - first : G;
  const u32     within = tile - group * G * tiles_n;
  m0                   = u64(first + within % height) * bm;
  n0                   = u64(within / height) * bn;
}

/// 128-bit vector of T, with element access by (after unrolling) constant index.
template <class T>
struct Vec;
template <>
struct Vec<float> {
  typedef float4       type;
  static constexpr int width = 4;
  static __device__ __forceinline__ float get(const float4& v, int e) {
    return e == 0 ? v.x : e == 1 ? v.y : e == 2 ? v.z : v.w;
  }
  static __device__ __forceinline__ void set(float4& v, int e, float x) {
    if (e == 0)
      v.x = x;
    else if (e == 1)
      v.y = x;
    else if (e == 2)
      v.z = x;
    else
      v.w = x;
  }
  static __device__ __forceinline__ float4 zero() { return make_float4(0.f, 0.f, 0.f, 0.f); }
};
template <>
struct Vec<double> {
  typedef double2      type;
  static constexpr int width = 2;
  static __device__ __forceinline__ double get(const double2& v, int e) { return e == 0 ? v.x : v.y; }
  static __device__ __forceinline__ void   set(double2& v, int e, double x) {
    if (e == 0)
      v.x = x;
    else
      v.y = x;
  }
  static __device__ __forceinline__ double2 zero() { return make_double2(0.0, 0.0); }
};

/// Global-to-shared staging of one operand's R x BK tile per k step (A: R = BM rows of m;
/// B: R = BN columns of n). Each of the NT threads moves L vectors of V elements. Stored
/// k-contiguous (KMAJOR), a vector runs along k (vector idx -> row idx / (BK / V), k offset
/// (idx % (BK / V)) * V) and is scattered into tile[k][row]; stored row-contiguous, it runs along
/// the rows (k offset idx / (R / V), row (idx % (R / V)) * V) and lands in shared memory whole.
/// The layout is a template parameter here; StageR reads it from a flag.
template <class T, int R, int BK, int NT, bool KMAJOR>
struct Stage {
  typedef Vec<T>            VV;
  typedef typename VV::type VT;
  static constexpr int      V = VV::width;
  static constexpr int      N = R * BK / V;      ///< Vectors per tile.
  static constexpr int      L = (N + NT - 1) / NT; ///< Vectors per thread.
  static_assert(BK % V == 0 && R % V == 0, "tile edges are whole vectors");

  const T* ptr[L]; ///< First element of each vector at the next k step.
  VT       reg[L]; ///< Staged values.
  u64      step;   ///< Pointer advance per k step.
  int      limit;  ///< Rows from the tile origin to the matrix edge (row-contiguous checks).
  bool     vec;    ///< 128-bit loads are aligned.

  static __device__ __forceinline__ bool active(int j) {
    return N % NT == 0 || int(threadIdx.x) + NT * j < N;
  }
  static __device__ __forceinline__ int row(int j) {
    const int idx = int(threadIdx.x) + NT * j;
    return KMAJOR ? idx / (BK / V) : idx % (R / V) * V;
  }
  static __device__ __forceinline__ int kk(int j) {
    const int idx = int(threadIdx.x) + NT * j;
    return KMAJOR ? idx % (BK / V) * V : idx / (R / V);
  }

  template <int S>
  __device__ __forceinline__ void init(const T* base, u64 ld, u64 rows, u64 r0, u64 kbeg, bool) {
    vec   = reinterpret_cast<u64>(base) % 16 == 0 && ld % V == 0 && (KMAJOR || rows % V == 0);
    step  = KMAJOR ? u64(BK) : u64(BK) * ld;
    limit = int(rows - r0 < u64(R) ? rows - r0 : u64(R));
#pragma unroll
    for (int j = 0; j < L; ++j) {
      if (KMAJOR) {
        // Rows past the edge read the last row instead; their results are never stored.
        const u64 r = r0 + row(j) < rows ? r0 + row(j) : rows - 1;
        ptr[j]      = base + r * ld + kbeg + kk(j);
      } else {
        u64 r = r0 + row(j);
        if (vec && r + V > rows)
          r = rows - V; // same, and still aligned (rows % V == 0)
        ptr[j] = base + (kbeg + kk(j)) * ld + r;
      }
    }
  }
  /// Load a step with all BK k values in range.
  __device__ __forceinline__ void load() {
#pragma unroll
    for (int j = 0; j < L; ++j) {
      if (!active(j))
        continue;
      if (vec) {
        reg[j] = *reinterpret_cast<const VT*>(ptr[j]);
      } else if (KMAJOR) {
#pragma unroll
        for (int e = 0; e < V; ++e)
          VV::set(reg[j], e, ptr[j][e]);
      } else {
#pragma unroll
        for (int e = 0; e < V; ++e)
          VV::set(reg[j], e, row(j) + e < limit ? ptr[j][e] : T(0));
      }
      ptr[j] += step;
    }
  }
  /// Load the last step, of `rem` < BK k values; the rest is zero-filled.
  __device__ __forceinline__ void load(int rem) {
#pragma unroll
    for (int j = 0; j < L; ++j) {
      if (!active(j))
        continue;
      if (KMAJOR) {
#pragma unroll
        for (int e = 0; e < V; ++e)
          VV::set(reg[j], e, kk(j) + e < rem ? ptr[j][e] : T(0));
      } else if (vec) {
        reg[j] = kk(j) < rem ? *reinterpret_cast<const VT*>(ptr[j]) : VV::zero();
      } else {
#pragma unroll
        for (int e = 0; e < V; ++e)
          VV::set(reg[j], e, kk(j) < rem && row(j) + e < limit ? ptr[j][e] : T(0));
      }
    }
  }
  template <int S>
  __device__ __forceinline__ void store(T (*tile)[S]) const {
#pragma unroll
    for (int j = 0; j < L; ++j) {
      if (!active(j))
        continue;
      if (KMAJOR) {
#pragma unroll
        for (int e = 0; e < V; ++e)
          tile[kk(j) + e][row(j)] = VV::get(reg[j], e);
      } else {
        *reinterpret_cast<VT*>(&tile[kk(j)][row(j)]) = reg[j];
      }
    }
  }
};

/// Stage with the layout as a uniform runtime flag (kmajor), so both layouts share one copy of the
/// main loop; the thread's shared-memory offsets are computed once in init.
template <class T, int R, int BK, int NT>
struct StageR {
  typedef Vec<T>            VV;
  typedef typename VV::type VT;
  static constexpr int      V = VV::width;
  static constexpr int      N = R * BK / V;      ///< Vectors per tile.
  static constexpr int      L = (N + NT - 1) / NT; ///< Vectors per thread.
  static_assert(BK % V == 0 && R % V == 0, "tile edges are whole vectors");

  const T* ptr[L]; ///< First element of each vector at the next k step.
  VT       reg[L]; ///< Staged values.
  int      at[L];  ///< Shared-memory offset of each vector's first element.
  /// Row (row-contiguous) or k offset (k-contiguous) of the thread's vectors: they are NT apart,
  /// a multiple of R / V and BK / V, so it is the same for all of them.
  int      first;
  u64      step;   ///< Pointer advance per k step.
  int      limit;  ///< Rows from the tile origin to the matrix edge (row-contiguous checks).
  bool     kmajor; ///< Stored k-contiguous.
  bool     vec;    ///< 128-bit loads are aligned.

  static __device__ __forceinline__ bool active(int j) {
    return N % NT == 0 || int(threadIdx.x) + NT * j < N;
  }

  template <int S>
  __device__ __forceinline__ void init(const T* base, u64 ld, u64 rows, u64 r0, u64 kbeg, bool km) {
    static_assert(NT % (R / V) == 0 && NT % (BK / V) == 0, "vector coordinates repeat every NT");
    kmajor = km;
    vec    = reinterpret_cast<u64>(base) % 16 == 0 && ld % V == 0 && (km || rows % V == 0);
    step   = km ? u64(BK) : u64(BK) * ld;
    limit  = int(rows - r0 < u64(R) ? rows - r0 : u64(R));
    const int t = int(threadIdx.x);
    first       = km ? t % (BK / V) * V : t % (R / V) * V;
#pragma unroll
    for (int j = 0; j < L; ++j) {
      const int idx = t + NT * j;
      const int row = km ? idx / (BK / V) : idx % (R / V) * V;
      const int kk  = km ? idx % (BK / V) * V : idx / (R / V);
      at[j]         = kk * S + row;
      if (km) {
        // Rows past the edge read the last row instead; their results are never stored.
        const u64 r = r0 + row < rows ? r0 + row : rows - 1;
        ptr[j]      = base + r * ld + kbeg + kk;
      } else {
        u64 r = r0 + row;
        if (vec && r + V > rows)
          r = rows - V; // same, and still aligned (rows % V == 0)
        ptr[j] = base + (kbeg + kk) * ld + r;
      }
    }
  }
  /// Load a step with all BK k values in range.
  __device__ __forceinline__ void load() {
#pragma unroll
    for (int j = 0; j < L; ++j) {
      if (!active(j))
        continue;
      if (vec) {
        reg[j] = *reinterpret_cast<const VT*>(ptr[j]);
      } else {
#pragma unroll
        for (int e = 0; e < V; ++e)
          VV::set(reg[j], e, kmajor || first + e < limit ? ptr[j][e] : T(0));
      }
      ptr[j] += step;
    }
  }
  /// Load the last step, of `rem` < BK k values; the rest is zero-filled.
  __device__ __forceinline__ void load(int rem) {
#pragma unroll
    for (int j = 0; j < L; ++j) {
      if (!active(j))
        continue;
      const int kk = (int(threadIdx.x) + NT * j) / (R / V); // k offset when row-contiguous
      if (kmajor) {
#pragma unroll
        for (int e = 0; e < V; ++e)
          VV::set(reg[j], e, first + e < rem ? ptr[j][e] : T(0));
      } else if (vec) {
        reg[j] = kk < rem ? *reinterpret_cast<const VT*>(ptr[j]) : VV::zero();
      } else {
#pragma unroll
        for (int e = 0; e < V; ++e)
          VV::set(reg[j], e, kk < rem && first + e < limit ? ptr[j][e] : T(0));
      }
    }
  }
  template <int S>
  __device__ __forceinline__ void store(T (*tile)[S]) const {
    T* base = &tile[0][0];
#pragma unroll
    for (int j = 0; j < L; ++j) {
      if (!active(j))
        continue;
      if (kmajor) {
#pragma unroll
        for (int e = 0; e < V; ++e)
          base[at[j] + e * S] = VV::get(reg[j], e);
      } else {
        *reinterpret_cast<VT*>(base + at[j]) = reg[j];
      }
    }
  }
};

/// Picks the staging of one operand: KM = 1 / 0 fixes the layout at compile time (k-contiguous /
/// row-contiguous), KM = -1 reads it from a runtime flag.
template <class T, int R, int BK, int NT, int KM>
struct PickStage {
  typedef Stage<T, R, BK, NT, KM == 1> type;
};
template <class T, int R, int BK, int NT>
struct PickStage<T, R, BK, NT, -1> {
  typedef StageR<T, R, BK, NT> type;
};

/// Shared-memory tiles of one entry, declared once in the entry and shared by its layout
/// instantiations.
template <class T, int BM, int BN, int BK, int PAD>
struct Tiles {
  __align__(16) T a[2][BK][BM + PAD];
  __align__(16) T b[2][BK][BN + PAD];
};

/// One block's tile. KA / KB: A / B stored k-contiguous (1: A not transposed, B transposed),
/// row-contiguous (0), or read from p.transpose_a / p.transpose_b (-1).
template <class T, int NW, int BM, int BN, int BK, int WGM, int LM, int PAD, int KA, int KB>
__device__ __forceinline__ void gemm(
    const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, T* __restrict__ W,
    const Gemm& p, Tiles<T, BM, BN, BK, PAD>& s) {
  typedef Vec<T>            VV;
  typedef typename VV::type VT;
  constexpr int             V   = VV::width;
  constexpr int             NT = 32 * NW, WGN = NW / WGM, WM = BM / WGM, WN = BN / WGN, LN = 32 / LM;
  constexpr int             TM = WM / LM, TN = WN / LN, FM = TM / V, FN = TN / V;
  static_assert(WGM * WGN == NW && LM * LN == 32, "NW warps of 32 lanes");
  static_assert(WM * WGM == BM && WN * WGN == BN && TM * LM == WM && TN * LN == WN, "tiling");
  static_assert(FM * V == TM && FN * V == TN && FM > 0 && FN > 0, "whole vectors per thread");
  static_assert(PAD % V == 0, "padded rows keep 128-bit alignment");

  const int tid = threadIdx.x;
  // Block b < whole computes tile b; the others compute k chunk (b - whole) % splits of tile
  // whole + (b - whole) / splits into its own BM x BN workspace slab.
  const u64 tiles = ((p.m + BM - 1) / BM) * ((p.n + BN - 1) / BN);
  const u64 b     = u64(blockIdx.y) * gridDim.x + blockIdx.x;
  const u64 tile  = b < p.whole ? b : p.whole + (b - p.whole) / p.splits;
  if (tile >= tiles)
    return;
  const bool raw  = b >= p.whole;
  const u64  kbeg = raw ? (b - p.whole) % p.splits * p.kchunk : 0;
  const u64  kend = raw && kbeg + p.kchunk < p.k ? kbeg + p.kchunk : p.k;
  u64        m0, n0;
  tile_origin(tile, p.m, p.n, BM, BN, m0, n0);
  // k <= INT_MAX (Context checks), so step counts fit an int.
  const int     len   = kbeg < kend ? int(kend - kbeg) : 0;
  const int     steps = (len + BK - 1) / BK, full = len / BK;

  typename PickStage<T, BM, BK, NT, KA>::type sa;
  typename PickStage<T, BN, BK, NT, KB>::type sb;
  sa.template init<BM + PAD>(A, p.lda, p.m, m0, kbeg, p.transpose_a == 0);
  sb.template init<BN + PAD>(B, p.ldb, p.n, n0, kbeg, p.transpose_b != 0);
  const auto load = [&](int step) {
    if (step < full) {
      sa.load();
      sb.load();
    } else {
      sa.load(len - step * BK);
      sb.load(len - step * BK);
    }
  };

  const int lane = tid & 31, warp = tid >> 5;
  const int fa   = (warp / WGN) * WM + (lane / LN) * V; // + f * LM * V
  const int fb   = (warp % WGN) * WN + (lane % LN) * V; // + f * LN * V

  T acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = T(0);
  T    af[2][TM], bf[2][TN]; // fragments of slice kk (set kk & 1) and kk + 1
  auto fragments = [&](int set, int buf, int kk) {
#pragma unroll
    for (int f = 0; f < FM; ++f) {
      const VT v = *reinterpret_cast<const VT*>(&s.a[buf][kk][fa + f * LM * V]);
#pragma unroll
      for (int e = 0; e < V; ++e)
        af[set][f * V + e] = VV::get(v, e);
    }
#pragma unroll
    for (int f = 0; f < FN; ++f) {
      const VT v = *reinterpret_cast<const VT*>(&s.b[buf][kk][fb + f * LN * V]);
#pragma unroll
      for (int e = 0; e < V; ++e)
        bf[set][f * V + e] = VV::get(v, e);
    }
  };

  if (steps) {
    load(0);
    sa.store(s.a[0]);
    sb.store(s.b[0]);
  }
  __syncthreads();
  fragments(0, 0, 0);
  int buf = 0;
  for (int step = 0; step < steps; ++step) {
    const bool more = step + 1 < steps;
    if (more)
      load(step + 1); // in flight while this tile is multiplied
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      const int set = kk & 1;
      if (kk == BK - 1 && more) {
        // The other buffer was last read before the previous barrier.
        sa.store(s.a[buf ^ 1]);
        sb.store(s.b[buf ^ 1]);
        __syncthreads();
        buf ^= 1;
      }
      if (kk < BK - 1 || more)
        fragments(set ^ 1, buf, (kk + 1) % BK);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] += af[set][i] * bf[set][j];
    }
  }

  if (raw) {
    // Raw sums to the chunk's slab, all BM x BN (rows and columns past the edge included).
    T* slab = W + (b - p.whole) * u64(BM * BN);
#pragma unroll
    for (int f = 0; f < FM; ++f)
#pragma unroll
      for (int e = 0; e < V; ++e)
#pragma unroll
        for (int g = 0; g < FN; ++g) {
          VT v;
#pragma unroll
          for (int x = 0; x < V; ++x)
            VV::set(v, x, acc[f * V + e][g * V + x]);
          *reinterpret_cast<VT*>(slab + (fa + f * LM * V + e) * BN + fb + g * LN * V) = v;
        }
    return;
  }
  const T    alpha = T(p.alpha), beta = T(p.beta);
  const bool whole = reinterpret_cast<u64>(C) % 16 == 0 && p.ldc % V == 0 && beta == T(0);
#pragma unroll
  for (int f = 0; f < FM; ++f)
#pragma unroll
    for (int e = 0; e < V; ++e) {
      const int i = f * V + e;
      const u64 r = m0 + fa + f * LM * V + e;
      if (r >= p.m)
        continue;
      T* line = C + r * p.ldc;
#pragma unroll
      for (int g = 0; g < FN; ++g) {
        const u64 c = n0 + fb + g * LN * V;
        if (whole && c + V <= p.n) {
          VT v;
#pragma unroll
          for (int x = 0; x < V; ++x)
            VV::set(v, x, alpha * acc[i][g * V + x]);
          *reinterpret_cast<VT*>(line + c) = v;
        } else {
#pragma unroll
          for (int x = 0; x < V; ++x) {
            if (c + x >= p.n)
              continue;
            const T s   = acc[i][g * V + x];
            line[c + x] = beta == T(0) ? alpha * s : alpha * s + beta * line[c + x];
          }
        }
      }
    }
}

/// Split tiles: C = alpha * (sum of the tile's chunk slabs) + beta * C. A block is X x G threads
/// (G = ceil(splits / 8) <= 8, X a multiple of 32 with X * G <= 256, chosen by the host): column x
/// is one vector of V consecutive outputs and row y adds chunks 8y .. 8y + 7 of it in chunk order
/// with all eight loads in flight; row 0 then adds the rows' sums in row order. The order is
/// fixed, so for a given plan the result is deterministic.
template <class T>
__device__ __forceinline__ void reduce(const T* __restrict__ W, T* __restrict__ C, const Reduce& p) {
  typedef Vec<T>            VV;
  typedef typename VV::type VT;
  constexpr int             V = VV::width;
  __shared__ VT             sums[256]; // rows 1 .. G - 1 of the block: [y * X + x]
  const int                 x = threadIdx.x, y = threadIdx.y, X = int(blockDim.x);
  const u64                 i = u64(blockIdx.x) * u64(X) + u64(x);
  const u64                 per = p.bm * p.bn / V, local = i / per, at = (i % per) * V; // tile, element
  const u64                 step = p.bm * p.bn;
  const bool                live = i < p.count;
  const T*                  slab = W + local * p.splits * step + at;
  const u64                 z0 = u64(y) * 8, z1 = z0 + 8 < p.splits ? z0 + 8 : p.splits;
  VT                        s  = VV::zero();
  if (live && z0 < z1) {
    VT t[8];
#pragma unroll
    for (int u = 0; u < 8; ++u)
      if (z0 + u < z1)
        t[u] = *reinterpret_cast<const VT*>(slab + (z0 + u) * step);
    s = t[0];
#pragma unroll
    for (int u = 1; u < 8; ++u)
      if (z0 + u < z1)
#pragma unroll
        for (int e = 0; e < V; ++e)
          VV::set(s, e, VV::get(s, e) + VV::get(t[u], e));
  }
  if (blockDim.y > 1) { // uniform over the block
    if (y > 0)
      sums[y * X + x] = s;
    __syncthreads();
  }
  if (y > 0 || !live)
    return;
  for (int g = 1; g < int(blockDim.y); ++g)
#pragma unroll
    for (int e = 0; e < V; ++e)
      VV::set(s, e, VV::get(s, e) + VV::get(sums[g * X + x], e));
  u64 m0, n0;
  tile_origin(p.whole + local, p.m, p.n, p.bm, p.bn, m0, n0);
  const u64 r = m0 + at / p.bn, c = n0 + at % p.bn;
  if (r >= p.m || c >= p.n)
    return;
  const T alpha = T(p.alpha), beta = T(p.beta);
  T*      out   = C + r * p.ldc + c;
#pragma unroll
  for (int e = 0; e < V; ++e)
    if (c + e < p.n)
      out[e] = beta == T(0) ? alpha * VV::get(s, e) : alpha * VV::get(s, e) + beta * out[e];
}

// How an entry handles the transposes (see the file comment).
#define DM_LAYOUTS_COMPILED 0 // the four layouts compiled separately, one picked per block
#define DM_LAYOUTS_B 1        // B's layout compiled (two copies), A's read from its flag
#define DM_LAYOUTS_FLAGS 2    // both read from runtime flags (one copy)

// NAME, type, warps, BM, BN, BK, warp rows (WGM), lane rows (LM), row padding, the minimum
// resident blocks per multiprocessor (caps registers per thread), and the transpose handling.
#define DM_GEMM(NAME, T, NW, BM, BN, BK, WGM, LM, PAD, MINB, LAYOUTS)                             \
  extern "C" __global__ void __launch_bounds__(32 * NW, MINB)                                      \
      NAME(const T* A, const T* B, T* C, T* W, Gemm p) {                                           \
    __shared__ Tiles<T, BM, BN, BK, PAD> s;                                                        \
    if (LAYOUTS == DM_LAYOUTS_COMPILED) {                                                          \
      if (!p.transpose_a) {                                                                        \
        if (!p.transpose_b)                                                                        \
          gemm<T, NW, BM, BN, BK, WGM, LM, PAD, 1, 0>(A, B, C, W, p, s);                           \
        else                                                                                       \
          gemm<T, NW, BM, BN, BK, WGM, LM, PAD, 1, 1>(A, B, C, W, p, s);                           \
      } else {                                                                                     \
        if (!p.transpose_b)                                                                        \
          gemm<T, NW, BM, BN, BK, WGM, LM, PAD, 0, 0>(A, B, C, W, p, s);                           \
        else                                                                                       \
          gemm<T, NW, BM, BN, BK, WGM, LM, PAD, 0, 1>(A, B, C, W, p, s);                           \
      }                                                                                            \
    } else if (LAYOUTS == DM_LAYOUTS_B) {                                                          \
      if (!p.transpose_b)                                                                          \
        gemm<T, NW, BM, BN, BK, WGM, LM, PAD, -1, 0>(A, B, C, W, p, s);                            \
      else                                                                                         \
        gemm<T, NW, BM, BN, BK, WGM, LM, PAD, -1, 1>(A, B, C, W, p, s);                            \
    } else {                                                                                       \
      gemm<T, NW, BM, BN, BK, WGM, LM, PAD, -1, -1>(A, B, C, W, p, s);                             \
    }                                                                                              \
  }

// Threads x outputs per thread: 128x256 256 x (8x16); 128x128 128 x (8x16); 128x64 128 x (8x8);
// 64x64 256 x (4x4); 128x32 256 x (4x4); f64 64x32 256 x (4x2).
DM_GEMM(gemm_f32_128x256, float, 8, 128, 256, 8, 2, 8, 4, 1, DM_LAYOUTS_COMPILED)
DM_GEMM(gemm_f32_128x128, float, 4, 128, 128, 8, 2, 8, 4, 2, DM_LAYOUTS_FLAGS)
DM_GEMM(gemm_f32_128x64, float, 4, 128, 64, 8, 2, 8, 4, 3, DM_LAYOUTS_B)
DM_GEMM(gemm_f32_64x64, float, 8, 64, 64, 16, 2, 8, 4, 3, DM_LAYOUTS_FLAGS)
DM_GEMM(gemm_f32_128x32, float, 8, 128, 32, 8, 8, 4, 4, 3, DM_LAYOUTS_FLAGS)
DM_GEMM(gemm_f64_64x32, double, 8, 64, 32, 8, 4, 4, 2, 2, DM_LAYOUTS_COMPILED)

extern "C" __global__ void reduce_f32(const float* W, float* C, Reduce p) { reduce(W, C, p); }
extern "C" __global__ void reduce_f64(const double* W, double* C, Reduce p) { reduce(W, C, p); }
