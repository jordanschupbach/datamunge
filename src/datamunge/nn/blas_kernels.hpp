#pragma once

// In-house BLAS kernels: packed GEMM (GotoBLAS/BLIS algorithm), GEMV, DOT,
// AXPY and SCAL for float and double, written with GCC/Clang vector
// extensions so one source serves every instruction set. The header is
// compiled into several translation units (blas_avx512.cpp, blas_avx2.cpp,
// blas_generic.cpp), each with its own -m flags and its own `Isa` tag; blas.cpp
// picks one at run time from the CPU features (see blas_kernels()).
//
// Everything here has internal linkage on purpose: an inline function compiled
// with -mavx512f in one unit must never be merged with the same function from
// the baseline unit, or the baseline path would execute AVX-512 instructions.
//
// GEMM: C = alpha * op(A) @ op(B) + beta * C, row-major, with op(A) [M, K] and
// op(B) [K, N]. For each (N block, K block) the K x N block of op(B) is packed
// into micro-panels of NR columns (one panel stays in L1 while it is used),
// and for each M block the M x K block of op(A) is packed into micro-panels of
// MR rows (the block stays in L2; when B is a single panel, A is used once and
// read in place instead). The micro-kernel keeps an MR x NR tile of C
// in vector registers for the whole K block, so every FMA reads one broadcast
// of A and one vector of B. Edge tiles compute into a scratch tile and copy
// out their valid part. Sums run in k order within a block and blocks in order,
// so results are deterministic and independent of the thread count (threading
// over M, N or K chunks happens in gemm.cpp).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

namespace datamunge::nn::detail {
namespace {

using I64 = std::int64_t;

inline I64 min_i(I64 a, I64 b) { return a < b ? a : b; }
inline I64 max_i(I64 a, I64 b) { return a < b ? b : a; }

#if defined(__GNUC__) || defined(__clang__)
#define DATAMUNGE_BLAS_INLINE __attribute__((always_inline)) inline
/// Vector type of `Bytes` bytes holding `T` lanes.
template <typename T, int Bytes>
struct Vec {
  typedef T type __attribute__((vector_size(Bytes)));
  static constexpr int lanes = Bytes / static_cast<int>(sizeof(T));
};
#else
#define DATAMUNGE_BLAS_INLINE inline
/// Portable stand-in for the vector extension: the same lane-wise arithmetic
/// on a plain array (compilers without vector_size auto-vectorise what they can).
template <typename T, int Bytes>
struct Vec {
  static constexpr int lanes = Bytes / static_cast<int>(sizeof(T));
  struct type {
    T lane[lanes];
    T&       operator[](int i) { return lane[i]; }
    const T& operator[](int i) const { return lane[i]; }
    friend type operator+(type a, const type& b) {
      for (int i = 0; i < lanes; ++i) a.lane[i] += b.lane[i];
      return a;
    }
    friend type operator*(type a, const type& b) {
      for (int i = 0; i < lanes; ++i) a.lane[i] *= b.lane[i];
      return a;
    }
    friend type operator/(type a, const type& b) {
      for (int i = 0; i < lanes; ++i) a.lane[i] /= b.lane[i];
      return a;
    }
    type& operator+=(const type& b) { return *this = *this + b; }
  };
};
#endif

/// All lanes set to `s`, built as one vector constructor so the compiler emits a
/// single broadcast (V{} + s would add a real zero first; lane stores become
/// masked inserts and keep the accumulators out of registers).
template <typename V, typename T, std::size_t... I>
DATAMUNGE_BLAS_INLINE V broadcast_impl(T s, std::index_sequence<I...>) {
#if defined(__GNUC__) || defined(__clang__)
  return V{(static_cast<void>(I), s)...};
#else
  V v;
  ((v[static_cast<int>(I)] = s), ...);
  return v;
#endif
}
template <typename V, typename T>
DATAMUNGE_BLAS_INLINE V broadcast(T s) {
  return broadcast_impl<V>(s, std::make_index_sequence<sizeof(V) / sizeof(T)>{});
}

/// 64-byte aligned scratch that grows on demand and belongs to one thread.
struct Scratch {
  void*       p{nullptr};
  std::size_t bytes{0};
  ~Scratch() { std::free(p); }
  template <typename T>
  T* get(std::size_t count) {
    const std::size_t need = (count * sizeof(T) + 63) & ~std::size_t{63};
    if (need > bytes) {
      std::free(p);
      bytes = need < 4096 ? 4096 : need;
      p     = std::aligned_alloc(64, bytes);
      if (!p) throw std::bad_alloc();
    }
    return static_cast<T*>(p);
  }
};

// ---- blocking parameters ------------------------------------------------------------------
//
// MR x NR is the register tile (NR = NV vectors), KC the largest depth of a
// packed block (k is cut into equal blocks of at most KC; a B micro-panel of
// KC x NR is reused from L1/L2 by every row panel of A), MC the rows of packed
// A (MC x KC stays in L2) and NC the columns of packed B (KC x NC in L3).

template <typename T, int VecBytes>
struct Blocking;

// AVX-512 blocks are 512 deep: half the passes over C of KC = 256 (C is read
// and written once per depth block), worth 5 % on one core at 4096 x 4096 and
// up to 15-20 % (double) when other processes share the caches; within 2 % of
// KC = 256 on the other benchmark shapes. MC halves so the packed A block
// stays at 192 KB.
template <>
struct Blocking<float, 64> {  // AVX-512: 32 zmm registers, 24 accumulators
  static constexpr int MR = 12, NV = 2, KC = 512, MC = 96, NC = 2048;
};
template <>
struct Blocking<double, 64> {
  static constexpr int MR = 12, NV = 2, KC = 512, MC = 48, NC = 1024;
};
template <>
struct Blocking<float, 32> {  // AVX2: 16 ymm registers, 12 accumulators
  static constexpr int MR = 6, NV = 2, KC = 256, MC = 192, NC = 2048;
};
template <>
struct Blocking<double, 32> {
  static constexpr int MR = 6, NV = 2, KC = 256, MC = 96, NC = 1024;
};
template <>
struct Blocking<float, 16> {  // SSE2 / NEON baseline
  static constexpr int MR = 4, NV = 2, KC = 256, MC = 128, NC = 1024;
};
template <>
struct Blocking<double, 16> {
  static constexpr int MR = 4, NV = 2, KC = 256, MC = 64, NC = 512;
};

// ---- micro-kernel ---------------------------------------------------------------------------

/// C[MR x NV*VL] (row stride ldc) = alpha * A~ @ B~ + beta * C, where A~ is an
/// MR x kc block of A with element (i, k) at a[i * ars + k * acs] and B~ a
/// packed kc x NRP micro-panel (b[k * NRP + j]) of which the first NV vectors
/// are used. beta == 0 does not read C.
template <typename T, int VecBytes, int MR, int NV, int NRP>
DATAMUNGE_BLAS_INLINE void micro_kernel_strided(I64 kc, const T* __restrict a, I64 ars, I64 acs,
                                                const T* __restrict b, T alpha, T beta, T* __restrict c, I64 ldc) {
  using V          = typename Vec<T, VecBytes>::type;
  constexpr int VL = Vec<T, VecBytes>::lanes;
  constexpr int NR = NRP;
  V             acc[MR][NV];
  // The C tile is read (or written) only after the whole depth loop: fetch it
  // now so its rows, ldc apart and often on as many pages, arrive in time.
  constexpr int kLines = (NR * static_cast<int>(sizeof(T)) + 63) / 64;
#pragma GCC unroll 16
  for (int i = 0; i < MR; ++i)
#pragma GCC unroll 4
    for (int l = 0; l < kLines; ++l) __builtin_prefetch(c + i * ldc + l * (64 / static_cast<int>(sizeof(T))), 1, 3);
#pragma GCC unroll 16
  for (int i = 0; i < MR; ++i)
#pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) acc[i][v] = V{};
  for (I64 k = 0; k < kc; ++k) {
    V bv[NV];
#pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) std::memcpy(&bv[v], b + k * NR + v * VL, sizeof(V));
#pragma GCC unroll 16
    for (int i = 0; i < MR; ++i) {
      const V av = broadcast<V>(a[i * ars + k * acs]);
#pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) acc[i][v] += av * bv[v];
    }
  }
  const V valpha = broadcast<V>(alpha);
  if (beta == T(0)) {
#pragma GCC unroll 16
    for (int i = 0; i < MR; ++i)
#pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        const V r = valpha * acc[i][v];
        std::memcpy(c + i * ldc + v * VL, &r, sizeof(V));
      }
  } else {
    const V vbeta = broadcast<V>(beta);
#pragma GCC unroll 16
    for (int i = 0; i < MR; ++i)
#pragma GCC unroll 4
      for (int v = 0; v < NV; ++v) {
        V old;
        std::memcpy(&old, c + i * ldc + v * VL, sizeof(V));
        const V r = valpha * acc[i][v] + vbeta * old;
        std::memcpy(c + i * ldc + v * VL, &r, sizeof(V));
      }
  }
}

/// micro_kernel_strided on a packed A micro-panel (k-major: a[k * MR + i]).
template <typename T, int VecBytes, int MR, int NV, int NRP>
DATAMUNGE_BLAS_INLINE void micro_kernel(I64 kc, const T* __restrict a, const T* __restrict b, T alpha, T beta,
                                        T* __restrict c, I64 ldc) {
  micro_kernel_strided<T, VecBytes, MR, NV, NRP>(kc, a, 1, MR, b, alpha, beta, c, ldc);
}

// ---- packing --------------------------------------------------------------------------------

/// Packs rows [0, mc) x cols [0, kc) of op(A) into micro-panels of MR rows,
/// k-major inside a panel; rows past mc in the last panel are zero.
template <typename T, int MR>
void pack_a(bool trans, I64 mc, I64 kc, const T* a, I64 lda, T* __restrict out) {
  for (I64 i0 = 0; i0 < mc; i0 += MR) {
    const I64 rows = min_i(MR, mc - i0);
    T* __restrict dst = out + i0 * kc;
    if (!trans) {
      // op(A) stored [M, K]: panel rows are MR strided streams.
      if (rows == MR) {
        for (I64 k = 0; k < kc; ++k) {
          const T* src = a + i0 * lda + k;
#pragma GCC unroll 16
          for (int i = 0; i < MR; ++i) dst[k * MR + i] = src[i * lda];
        }
      } else {
        for (I64 k = 0; k < kc; ++k) {
          const T* src = a + i0 * lda + k;
          for (I64 i = 0; i < rows; ++i) dst[k * MR + i] = src[i * lda];
          for (I64 i = rows; i < MR; ++i) dst[k * MR + i] = T(0);
        }
      }
    } else {
      // op(A) stored transposed [K, M]: each k gives MR contiguous values.
      if (rows == MR) {
        for (I64 k = 0; k < kc; ++k) {
          const T* src = a + k * lda + i0;
#pragma GCC unroll 16
          for (int i = 0; i < MR; ++i) dst[k * MR + i] = src[i];
        }
      } else {
        for (I64 k = 0; k < kc; ++k) {
          const T* src = a + k * lda + i0;
          for (I64 i = 0; i < rows; ++i) dst[k * MR + i] = src[i];
          for (I64 i = rows; i < MR; ++i) dst[k * MR + i] = T(0);
        }
      }
    }
  }
}

/// Packs rows [0, kc) x cols [0, nc) of op(B) into micro-panels of NR columns,
/// k-major inside a panel; columns past nc in the last panel are zero.
template <typename T, int NR>
void pack_b(bool trans, I64 kc, I64 nc, const T* b, I64 ldb, T* __restrict out) {
  for (I64 j0 = 0; j0 < nc; j0 += NR) {
    const I64 cols = min_i(NR, nc - j0);
    T* __restrict dst = out + j0 * kc;
    if (!trans) {
      // op(B) stored [K, N]: each k gives NR contiguous values.
      if (cols == NR) {
        for (I64 k = 0; k < kc; ++k) {
          const T* src = b + k * ldb + j0;
#pragma GCC unroll 32
          for (int j = 0; j < NR; ++j) dst[k * NR + j] = src[j];
        }
      } else {
        for (I64 k = 0; k < kc; ++k) {
          const T* src = b + k * ldb + j0;
          for (I64 j = 0; j < cols; ++j) dst[k * NR + j] = src[j];
          for (I64 j = cols; j < NR; ++j) dst[k * NR + j] = T(0);
        }
      }
    } else {
      // op(B) stored transposed [N, K]: panel columns are NR strided streams.
      if (cols == NR) {
        for (I64 k = 0; k < kc; ++k) {
          const T* src = b + j0 * ldb + k;
#pragma GCC unroll 32
          for (int j = 0; j < NR; ++j) dst[k * NR + j] = src[j * ldb];
        }
      } else {
        for (I64 k = 0; k < kc; ++k) {
          const T* src = b + j0 * ldb + k;
          for (I64 j = 0; j < cols; ++j) dst[k * NR + j] = src[j * ldb];
          for (I64 j = cols; j < NR; ++j) dst[k * NR + j] = T(0);
        }
      }
    }
  }
}

// ---- GEMM driver ----------------------------------------------------------------------------

/// Depth of each block when k is cut into the fewest blocks of at most KC:
/// equal blocks (576 = 288 + 288, not 512 + 64), so no block is too shallow
/// to amortise its packing.
inline I64 depth_step(I64 k, I64 KC) {
  const I64 blocks = (k + KC - 1) / KC;
  return (k + blocks - 1) / blocks;
}

/// Scales the m x n block of C by beta (beta == 0 writes zeros without reading).
template <typename T>
void scale_c(I64 m, I64 n, T beta, T* c, I64 ldc) {
  if (beta == T(1)) return;
  for (I64 i = 0; i < m; ++i) {
    T* row = c + i * ldc;
    if (beta == T(0))
      for (I64 j = 0; j < n; ++j) row[j] = T(0);
    else
      for (I64 j = 0; j < n; ++j) row[j] *= beta;
  }
}

/// C[0:mc, panels p0..p1) of one block: packed A (mc rows, MR-row panels) times
/// the packed B panels p0..p1 (NR columns each, the last one possibly
/// partial; nc columns in all). Full tiles go straight into C, edge tiles
/// through a scratch tile. Serial and threaded drivers share it, so every C
/// element is computed by the same instructions in the same order.
template <typename T, int VecBytes>
DATAMUNGE_BLAS_INLINE void macro_kernel(I64 mc, I64 nc, I64 kc, const T* ap, const T* bp, I64 p0, I64 p1, T alpha,
                                        T beta_eff, T* c, I64 ldc) {
  using B          = Blocking<T, VecBytes>;
  constexpr int MR = B::MR, NV = B::NV, VL = Vec<T, VecBytes>::lanes, NR = NV * VL;
  alignas(64) T tile[MR * NR];
  for (I64 p = p0; p < p1; ++p) {
    const I64 jr = p * NR, nr = min_i(NR, nc - jr);
    const T*  bpanel = bp + jr * kc;
    for (I64 ir = 0; ir < mc; ir += MR) {
      const I64 mr     = min_i(MR, mc - ir);
      const T*  apanel = ap + ir * kc;
      T*        cblk   = c + ir * ldc + jr;
      if (mr == MR && nr == NR) {
        micro_kernel<T, VecBytes, MR, NV, NR>(kc, apanel, bpanel, alpha, beta_eff, cblk, ldc);
      } else if (mr == MR && nr == VL && NV > 1) {
        micro_kernel<T, VecBytes, MR, 1, NR>(kc, apanel, bpanel, alpha, beta_eff, cblk, ldc);
      } else {
        // Edge tile: full kernel into scratch, then the valid part into C.
        if (nr <= VL && NV > 1)
          micro_kernel<T, VecBytes, MR, 1, NR>(kc, apanel, bpanel, T(1), T(0), tile, NR);
        else
          micro_kernel<T, VecBytes, MR, NV, NR>(kc, apanel, bpanel, T(1), T(0), tile, NR);
        for (I64 i = 0; i < mr; ++i) {
          T*       crow = cblk + i * ldc;
          const T* trow = tile + i * NR;
          if (beta_eff == T(0))
            for (I64 j = 0; j < nr; ++j) crow[j] = alpha * trow[j];
          else
            for (I64 j = 0; j < nr; ++j) crow[j] = alpha * trow[j] + beta_eff * crow[j];
        }
      }
    }
  }
}

/// C[0:mc, 0:nc) for a B block of a single panel (nc <= NR). Each element of A
/// is then used once per depth block, so packing A would cost about as much as
/// the products it feeds (a third of the time at 8192 x 16 x 128): full row
/// tiles read A in place (row i, column k at a[i * ars + k * acs]) and only
/// the last, partial tile is packed (zero rows). Every C element sees the
/// same instructions as through macro_kernel.
template <typename T, int VecBytes>
DATAMUNGE_BLAS_INLINE void macro_kernel_direct(I64 mc, I64 nc, I64 kc, const T* a, I64 ars, I64 acs, const T* bp,
                                               T alpha, T beta_eff, T* c, I64 ldc) {
  using B          = Blocking<T, VecBytes>;
  constexpr int MR = B::MR, NV = B::NV, VL = Vec<T, VecBytes>::lanes, NR = NV * VL;
  alignas(64) T tile[MR * NR];
  for (I64 ir = 0; ir < mc; ir += MR) {
    const I64 mr   = min_i(MR, mc - ir);
    const T*  ablk = a + ir * ars;
    T*        cblk = c + ir * ldc;
    if (mr == MR && nc == NR) {
      micro_kernel_strided<T, VecBytes, MR, NV, NR>(kc, ablk, ars, acs, bp, alpha, beta_eff, cblk, ldc);
      continue;
    }
    if (mr == MR && nc == VL && NV > 1) {
      micro_kernel_strided<T, VecBytes, MR, 1, NR>(kc, ablk, ars, acs, bp, alpha, beta_eff, cblk, ldc);
      continue;
    }
    // Edge tile into scratch, then the valid part into C.
    if (mr == MR) {
      if (nc <= VL && NV > 1)
        micro_kernel_strided<T, VecBytes, MR, 1, NR>(kc, ablk, ars, acs, bp, T(1), T(0), tile, NR);
      else
        micro_kernel_strided<T, VecBytes, MR, NV, NR>(kc, ablk, ars, acs, bp, T(1), T(0), tile, NR);
    } else {
      alignas(64) T apanel[MR * B::KC];
      for (I64 k = 0; k < kc; ++k) {
        for (I64 i = 0; i < mr; ++i) apanel[k * MR + i] = ablk[i * ars + k * acs];
        for (I64 i = mr; i < MR; ++i) apanel[k * MR + i] = T(0);
      }
      if (nc <= VL && NV > 1)
        micro_kernel<T, VecBytes, MR, 1, NR>(kc, apanel, bp, T(1), T(0), tile, NR);
      else
        micro_kernel<T, VecBytes, MR, NV, NR>(kc, apanel, bp, T(1), T(0), tile, NR);
    }
    for (I64 i = 0; i < mr; ++i) {
      T*       crow = cblk + i * ldc;
      const T* trow = tile + i * NR;
      if (beta_eff == T(0))
        for (I64 j = 0; j < nc; ++j) crow[j] = alpha * trow[j];
      else
        for (I64 j = 0; j < nc; ++j) crow[j] = alpha * trow[j] + beta_eff * crow[j];
    }
  }
}

template <typename T, int VecBytes>
void gemm_packed(bool ta, bool tb, I64 m, I64 n, I64 k, T alpha, const T* a, I64 lda, const T* b, I64 ldb, T beta,
                 T* c, I64 ldc) {
  using B           = Blocking<T, VecBytes>;
  constexpr int MR  = B::MR, NV = B::NV, VL = Vec<T, VecBytes>::lanes, NR = NV * VL;
  constexpr I64 KC  = B::KC, MC = B::MC, NC = B::NC;
  if (m <= 0 || n <= 0) return;
  if (k <= 0 || alpha == T(0)) {
    scale_c<T>(m, n, beta, c, ldc);
    return;
  }
  static thread_local Scratch a_scratch, b_scratch;
  if (n <= NR) {
    // One B panel: A is read in place (macro_kernel_direct).
    const I64        ks = depth_step(k, KC);
    T* __restrict bp = b_scratch.get<T>(static_cast<std::size_t>(NR * ks));
    for (I64 pc = 0; pc < k; pc += ks) {
      const I64 kc = min_i(ks, k - pc);
      pack_b<T, NR>(tb, kc, n, tb ? b + pc : b + pc * ldb, ldb, bp);
      macro_kernel_direct<T, VecBytes>(m, n, kc, ta ? a + pc * lda : a + pc, ta ? 1 : lda, ta ? lda : 1, bp, alpha,
                                       pc == 0 ? beta : T(1), c, ldc);
    }
    return;
  }
  const I64 mc_max = min_i(MC, (m + MR - 1) / MR * MR);
  const I64 nc_max = min_i(NC, (n + NR - 1) / NR * NR);
  const I64 kc_max = depth_step(k, KC);
  T* __restrict ap = a_scratch.get<T>(static_cast<std::size_t>(mc_max * kc_max));
  T* __restrict bp = b_scratch.get<T>(static_cast<std::size_t>(nc_max * kc_max));
  for (I64 jc = 0; jc < n; jc += NC) {
    const I64 nc = min_i(NC, n - jc);
    for (I64 pc = 0; pc < k; pc += kc_max) {
      const I64 kc       = min_i(kc_max, k - pc);
      const T   beta_eff = pc == 0 ? beta : T(1);
      pack_b<T, NR>(tb, kc, nc, tb ? b + jc * ldb + pc : b + pc * ldb + jc, ldb, bp);
      for (I64 ic = 0; ic < m; ic += MC) {
        const I64 mc = min_i(MC, m - ic);
        pack_a<T, MR>(ta, mc, kc, ta ? a + pc * lda + ic : a + ic * lda + pc, lda, ap);
        macro_kernel<T, VecBytes>(mc, nc, kc, ap, bp, 0, (nc + NR - 1) / NR, alpha, beta_eff, c + ic * ldc + jc, ldc);
      }
    }
  }
}

/// Runs body(ctx, i) for every i in [0, count) on a thread pool and returns
/// once all have finished (gemm.cpp passes the nn pool). A plain function
/// pointer, so the instruction-set units instantiate no library templates.
using ParallelRun = void (*)(I64 count, void (*body)(void*, I64), void* ctx);

/// Packed-B blocks up to this many elements are packed by the calling thread
/// (cheaper than a pool dispatch: the narrow outputs of most convolutions).
constexpr I64 kSerialPackElems = I64{1} << 14;

/// Smallest task of the threaded GEMM, in flops (~10 us on one core).
constexpr I64 kMinTaskFlop = I64{1} << 21;

/// gemm_packed on `threads` threads sharing each packed block of B: per
/// (jc, pc) block the threads first pack B's panels together (the caller alone
/// when the block is small), then split C
/// into row blocks (multiples of MR, at most MC) and, when that gives too few
/// tasks, panel groups, each packing its own rows of A. The tiles and the
/// order of the depth blocks are those of gemm_packed, so the result is
/// identical for every thread count.
template <typename T, int VecBytes>
void gemm_packed_mt(bool ta, bool tb, I64 m, I64 n, I64 k, T alpha, const T* a, I64 lda, const T* b, I64 ldb,
                    T beta, T* c, I64 ldc, I64 threads, ParallelRun run) {
  using Bk          = Blocking<T, VecBytes>;
  constexpr int MR  = Bk::MR, NR = Bk::NV * Vec<T, VecBytes>::lanes;
  constexpr I64 KC  = Bk::KC, MC = Bk::MC, NC = Bk::NC;
  if (threads <= 1 || !run || m <= 0 || n <= 0 || k <= 0 || alpha == T(0)) {
    gemm_packed<T, VecBytes>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return;
  }
  static thread_local Scratch b_shared;
  const I64 nc_max = min_i(NC, (n + NR - 1) / NR * NR), kc_max = depth_step(k, KC);
  T*        bp     = b_shared.get<T>(static_cast<std::size_t>(nc_max * kc_max));
  // Row blocks (MR-aligned): at most what keeps a packed A block in L2 (MC x KC
  // elements, so a shallow product takes more rows), split evenly over rounds
  // of `threads` tasks, and at least kMinTaskFlop of work each so a shallow,
  // narrow product is not cut into tasks that cost less than claiming them.
  const auto align     = [](I64 r) { return (r + MR - 1) / MR * MR; };
  const I64  rows_cap  = max_i(MR, MC * KC / kc_max / MR * MR);
  const I64  rounds    = (m + threads * rows_cap - 1) / (threads * rows_cap);
  const I64  even      = align((m + threads * rounds - 1) / (threads * rounds));
  const I64  min_work  = align(kMinTaskFlop / (2 * min_i(n, NC) * kc_max));
  const I64  rows      = min_i(rows_cap, max_i(even, min_work));
  const I64  row_blocks = (m + rows - 1) / rows;

  struct Job {
    bool     ta, tb;
    I64      m, k, lda, ldb, ldc, jc, pc, nc, kc, panels, rows, groups, per_group, pack_tasks;
    T        alpha, beta_eff;
    const T *a, *b;
    T *      c, *bp;
  } job{ta, tb, m, k, lda, ldb, ldc, 0, 0, 0, 0, 0, rows, 0, 0, 0, alpha, beta, a, b, c, bp};

  // Task t packs a contiguous run of B's panels into the shared block.
  const auto pack_task = [](void* p, I64 t) {
    const Job& j     = *static_cast<const Job*>(p);
    const I64  per   = (j.panels + j.pack_tasks - 1) / j.pack_tasks;
    const I64  first = t * per, last = min_i(j.panels, first + per);
    if (first >= last) return;
    const I64 c0 = first * NR, cols = min_i(j.nc, last * NR) - c0;
    const T*  src = j.tb ? j.b + (j.jc + c0) * j.ldb + j.pc : j.b + j.pc * j.ldb + j.jc + c0;
    pack_b<T, NR>(j.tb, j.kc, cols, src, j.ldb, j.bp + c0 * j.kc);
  };
  // Task t = (row block, panel group): packs its rows of A, multiplies them by its panels.
  const auto compute_task = [](void* p, I64 t) {
    const Job& j  = *static_cast<const Job*>(p);
    const I64  rb = t / j.groups, g = t % j.groups;
    const I64  r0 = rb * j.rows, mc = min_i(j.rows, j.m - r0);
    const I64  p0 = g * j.per_group, p1 = min_i(j.panels, p0 + j.per_group);
    if (mc <= 0 || p0 >= p1) return;
    const T* asrc = j.ta ? j.a + j.pc * j.lda + r0 : j.a + r0 * j.lda + j.pc;
    if (j.panels == 1) {
      macro_kernel_direct<T, VecBytes>(mc, j.nc, j.kc, asrc, j.ta ? 1 : j.lda, j.ta ? j.lda : 1, j.bp, j.alpha,
                                       j.beta_eff, j.c + r0 * j.ldc + j.jc, j.ldc);
      return;
    }
    static thread_local Scratch a_scratch;
    T* ap = a_scratch.get<T>(static_cast<std::size_t>(((mc + MR - 1) / MR * MR) * j.kc));
    pack_a<T, MR>(j.ta, mc, j.kc, asrc, j.lda, ap);
    macro_kernel<T, VecBytes>(mc, j.nc, j.kc, ap, j.bp, p0, p1, j.alpha, j.beta_eff, j.c + r0 * j.ldc + j.jc, j.ldc);
  };

  for (I64 jc = 0; jc < n; jc += NC) {
    const I64 nc     = min_i(NC, n - jc);
    const I64 panels = (nc + NR - 1) / NR;
    // Panel groups: enough tasks to fill the threads when C has few row blocks.
    const I64 groups = min_i(panels, max_i(1, (threads + row_blocks - 1) / row_blocks));
    for (I64 pc = 0; pc < k; pc += kc_max) {
      job.jc = jc, job.pc = pc, job.nc = nc, job.kc = min_i(kc_max, k - pc), job.panels = panels;
      job.beta_eff   = pc == 0 ? beta : T(1);
      job.groups     = groups;
      job.per_group  = (panels + groups - 1) / groups;
      job.pack_tasks = min_i(panels, threads);
      if (nc * job.kc <= kSerialPackElems)
        pack_b<T, NR>(tb, job.kc, nc, tb ? b + jc * ldb + pc : b + pc * ldb + jc, ldb, bp);
      else
        run(job.pack_tasks, pack_task, &job);
      run(row_blocks * groups, compute_task, &job);
    }
  }
}

// ---- level-1 / level-2 ----------------------------------------------------------------------

/// Sum of x[i] * y[i] with eight independent vector accumulators (unit stride).
template <typename T, int VecBytes>
T dot_unit(I64 n, const T* __restrict x, const T* __restrict y) {
  using V          = typename Vec<T, VecBytes>::type;
  constexpr int VL = Vec<T, VecBytes>::lanes;
  V             acc[8];
  for (auto& v : acc) v = V{};
  I64 i = 0;
  for (; i + 8 * VL <= n; i += 8 * VL) {
#pragma GCC unroll 8
    for (int u = 0; u < 8; ++u) {
      V xv, yv;
      std::memcpy(&xv, x + i + u * VL, sizeof(V));
      std::memcpy(&yv, y + i + u * VL, sizeof(V));
      acc[u] += xv * yv;
    }
  }
  for (; i + VL <= n; i += VL) {
    V xv, yv;
    std::memcpy(&xv, x + i, sizeof(V));
    std::memcpy(&yv, y + i, sizeof(V));
    acc[0] += xv * yv;
  }
  V total = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
  T s     = T(0);
  for (int l = 0; l < VL; ++l) s += total[l];
  for (; i < n; ++i) s += x[i] * y[i];
  return s;
}

template <typename T, int VecBytes>
T dot_kernel(I64 n, const T* x, I64 incx, const T* y, I64 incy) {
  if (n <= 0) return T(0);
  if (incx == 1 && incy == 1) return dot_unit<T, VecBytes>(n, x, y);
  T s = T(0);
  for (I64 i = 0; i < n; ++i) s += x[i * incx] * y[i * incy];
  return s;
}

template <typename T, int VecBytes>
void axpy_kernel(I64 n, T alpha, const T* x, I64 incx, T* y, I64 incy) {
  if (n <= 0 || alpha == T(0)) return;
  if (incx == 1 && incy == 1) {
    for (I64 i = 0; i < n; ++i) y[i] += alpha * x[i];
    return;
  }
  for (I64 i = 0; i < n; ++i) y[i * incy] += alpha * x[i * incx];
}

template <typename T, int VecBytes>
void scal_kernel(I64 n, T alpha, T* x, I64 incx) {
  if (n <= 0) return;
  if (incx == 1) {
    for (I64 i = 0; i < n; ++i) x[i] *= alpha;
    return;
  }
  for (I64 i = 0; i < n; ++i) x[i * incx] *= alpha;
}

/// y_i += alpha * (A_i . x) over rows [0, rows) of the row-major A (unit-stride x), four rows at
/// a time sharing each load of x (four streams through memory keep more lines in flight than
/// one; 1.3-1.4x on matrices past the caches). Every row sums with the same two accumulators
/// whether it runs in a group of four or alone, so the results do not depend on where a caller
/// splits the rows.
template <typename T, int VecBytes>
void gemv_n_unit(I64 rows, I64 cols, T alpha, const T* a, I64 lda, const T* x, T* y, I64 incy) {
  using V          = typename Vec<T, VecBytes>::type;
  constexpr int VL = Vec<T, VecBytes>::lanes;
  const I64     body = cols / (2 * VL) * (2 * VL);
  const auto    finish = [&](const V& acc0, const V& acc1, const T* row) {
    const V total = acc0 + acc1;
    T       s     = T(0);
    for (int l = 0; l < VL; ++l) s += total[l];
    for (I64 j = body; j < cols; ++j) s += row[j] * x[j];
    return s;
  };
  I64 i = 0;
  for (; i + 4 <= rows; i += 4) {
    const T* __restrict r0 = a + i * lda;
    const T* __restrict r1 = r0 + lda;
    const T* __restrict r2 = r1 + lda;
    const T* __restrict r3 = r2 + lda;
    V a00 = V{}, a01 = V{}, a10 = V{}, a11 = V{}, a20 = V{}, a21 = V{}, a30 = V{}, a31 = V{};
    for (I64 j = 0; j < body; j += 2 * VL) {
      V x0, x1, v;
      std::memcpy(&x0, x + j, sizeof(V));
      std::memcpy(&x1, x + j + VL, sizeof(V));
      std::memcpy(&v, r0 + j, sizeof(V));
      a00 += v * x0;
      std::memcpy(&v, r0 + j + VL, sizeof(V));
      a01 += v * x1;
      std::memcpy(&v, r1 + j, sizeof(V));
      a10 += v * x0;
      std::memcpy(&v, r1 + j + VL, sizeof(V));
      a11 += v * x1;
      std::memcpy(&v, r2 + j, sizeof(V));
      a20 += v * x0;
      std::memcpy(&v, r2 + j + VL, sizeof(V));
      a21 += v * x1;
      std::memcpy(&v, r3 + j, sizeof(V));
      a30 += v * x0;
      std::memcpy(&v, r3 + j + VL, sizeof(V));
      a31 += v * x1;
    }
    y[i * incy] += alpha * finish(a00, a01, r0);
    y[(i + 1) * incy] += alpha * finish(a10, a11, r1);
    y[(i + 2) * incy] += alpha * finish(a20, a21, r2);
    y[(i + 3) * incy] += alpha * finish(a30, a31, r3);
  }
  for (; i < rows; ++i) {
    const T* __restrict r0 = a + i * lda;
    V a00 = V{}, a01 = V{};
    for (I64 j = 0; j < body; j += 2 * VL) {
      V x0, x1, v;
      std::memcpy(&x0, x + j, sizeof(V));
      std::memcpy(&x1, x + j + VL, sizeof(V));
      std::memcpy(&v, r0 + j, sizeof(V));
      a00 += v * x0;
      std::memcpy(&v, r0 + j + VL, sizeof(V));
      a01 += v * x1;
    }
    y[i * incy] += alpha * finish(a00, a01, r0);
  }
}

/// y = alpha * op(A) x + beta * y with op(A) [m, n] (A stored [m, n] row-major,
/// or [n, m] when trans). Rows [r0, r1) of the stored matrix only, so callers
/// can split the work; for trans the split is over the stored rows too (each
/// contributes a scaled row to all of y), so callers must then use one range.
template <typename T, int VecBytes>
void gemv_kernel(bool trans, I64 m, I64 n, T alpha, const T* a, I64 lda, const T* x, I64 incx, T beta, T* y,
                 I64 incy) {
  const I64 rows = trans ? n : m;  // length of y
  const I64 cols = trans ? m : n;  // length of x
  if (rows <= 0) return;
  if (beta == T(0)) {
    // BLAS: beta = 0 does not read y (a NaN there must not survive).
    for (I64 i = 0; i < rows; ++i) y[i * incy] = T(0);
  } else if (beta != T(1)) {
    scal_kernel<T, VecBytes>(rows, beta, y, incy);
  }
  if (cols <= 0 || alpha == T(0)) return;
  if (!trans) {
    if (incx == 1) {
      gemv_n_unit<T, VecBytes>(rows, cols, alpha, a, lda, x, y, incy);
    } else {
      for (I64 i = 0; i < rows; ++i) {
        const T* row = a + i * lda;
        T        s   = T(0);
        for (I64 j = 0; j < cols; ++j) s += row[j] * x[j * incx];
        y[i * incy] += alpha * s;
      }
    }
    return;
  }
  // y += alpha * A^T x: accumulate rows of the stored [cols, rows] matrix,
  // four at a time so y is loaded and stored once per four rows (the adds
  // still run in row order, so the result equals one row at a time).
  if (incy == 1) {
    T* __restrict yv = y;
    I64           j  = 0;
    for (; j + 4 <= cols; j += 4) {
      const T x0 = alpha * x[j * incx], x1 = alpha * x[(j + 1) * incx], x2 = alpha * x[(j + 2) * incx],
              x3 = alpha * x[(j + 3) * incx];
      const T* __restrict r0 = a + j * lda;
      const T* __restrict r1 = r0 + lda;
      const T* __restrict r2 = r1 + lda;
      const T* __restrict r3 = r2 + lda;
      for (I64 i = 0; i < rows; ++i) {
        T v = yv[i];
        v += x0 * r0[i];
        v += x1 * r1[i];
        v += x2 * r2[i];
        v += x3 * r3[i];
        yv[i] = v;
      }
    }
    for (; j < cols; ++j) {
      const T             xj  = alpha * x[j * incx];
      const T* __restrict row = a + j * lda;
      for (I64 i = 0; i < rows; ++i) yv[i] += xj * row[i];
    }
  } else {
    for (I64 j = 0; j < cols; ++j) {
      const T  xj  = alpha * x[j * incx];
      const T* row = a + j * lda;
      for (I64 i = 0; i < rows; ++i) y[i * incy] += xj * row[i];
    }
  }
}

/// Two products from one pass over the stored [m, n] row-major A (unit strides): y += alpha A^T x
/// (y has n entries, x has m) and t += alpha A z (t has m entries, z has n). Rows go four at a
/// time, so y and z are read once per four rows of A. These are the off-diagonal blocks of a
/// symmetric matrix-vector product that reads one triangle (BLAS dsymv).
template <typename T, int VecBytes>
void gemv_pair_kernel(I64 m, I64 n, T alpha, const T* a, I64 lda, const T* x, const T* z, T* y, T* t) {
  using V          = typename Vec<T, VecBytes>::type;
  constexpr int VL = Vec<T, VecBytes>::lanes;
  if (m <= 0 || n <= 0 || alpha == T(0)) return;
  T* __restrict yv = y;
  I64           r  = 0;
  for (; r + 4 <= m; r += 4) {
    const T* __restrict r0 = a + r * lda;
    const T* __restrict r1 = r0 + lda;
    const T* __restrict r2 = r1 + lda;
    const T* __restrict r3 = r2 + lda;
    const T x0 = alpha * x[r], x1 = alpha * x[r + 1], x2 = alpha * x[r + 2], x3 = alpha * x[r + 3];
    const V bx0 = broadcast<V>(x0), bx1 = broadcast<V>(x1), bx2 = broadcast<V>(x2), bx3 = broadcast<V>(x3);
    V       acc0 = V{}, acc1 = V{}, acc2 = V{}, acc3 = V{};
    I64     i    = 0;
    for (; i + VL <= n; i += VL) {
      V a0, a1, a2, a3, zi, yi;
      std::memcpy(&a0, r0 + i, sizeof(V));
      std::memcpy(&a1, r1 + i, sizeof(V));
      std::memcpy(&a2, r2 + i, sizeof(V));
      std::memcpy(&a3, r3 + i, sizeof(V));
      std::memcpy(&zi, z + i, sizeof(V));
      std::memcpy(&yi, yv + i, sizeof(V));
      yi += bx0 * a0;
      yi += bx1 * a1;
      yi += bx2 * a2;
      yi += bx3 * a3;
      std::memcpy(yv + i, &yi, sizeof(V));
      acc0 += a0 * zi;
      acc1 += a1 * zi;
      acc2 += a2 * zi;
      acc3 += a3 * zi;
    }
    T s0 = T(0), s1 = T(0), s2 = T(0), s3 = T(0);
    for (int l = 0; l < VL; ++l) {
      s0 += acc0[l];
      s1 += acc1[l];
      s2 += acc2[l];
      s3 += acc3[l];
    }
    for (; i < n; ++i) {
      T v = yv[i];
      v += x0 * r0[i];
      v += x1 * r1[i];
      v += x2 * r2[i];
      v += x3 * r3[i];
      yv[i] = v;
      s0 += r0[i] * z[i];
      s1 += r1[i] * z[i];
      s2 += r2[i] * z[i];
      s3 += r3[i] * z[i];
    }
    t[r] += alpha * s0;
    t[r + 1] += alpha * s1;
    t[r + 2] += alpha * s2;
    t[r + 3] += alpha * s3;
  }
  for (; r < m; ++r) {
    const T* __restrict row = a + r * lda;
    const T xr              = alpha * x[r];
    for (I64 i = 0; i < n; ++i) yv[i] += xr * row[i];
    t[r] += alpha * dot_unit<T, VecBytes>(n, row, z);
  }
}


// ---- small triangular solves and Cholesky (blocks of the factorizations in linalg.cpp) ----------

/// Columns [0, S * VL) of the rows of Y: see trsm_rows_kernel. Two rows at a time share the
/// loads of every solved row; S = 0 is one scalar column.
template <typename T, int VecBytes, int S>
DATAMUNGE_BLAS_INLINE void trsm_rows_strip(I64 m, const T* L, I64 rs, I64 cs, bool unit, T* Y, I64 ldy) {
  using V          = typename Vec<T, VecBytes>::type;
  constexpr int VL = Vec<T, VecBytes>::lanes;
  if constexpr (S == 0) {
    for (I64 j = 0; j < m; ++j) {
      const T* lj = L + j * rs;
      T        v  = Y[j * ldy];
      for (I64 p = 0; p < j; ++p) v -= lj[p * cs] * Y[p * ldy];
      Y[j * ldy] = unit ? v : v / lj[j * cs];
    }
  } else {
    I64 j = 0;
    for (; j + 2 <= m; j += 2) {
      T* __restrict ya = Y + j * ldy;
      T* __restrict yb = ya + ldy;
      const T*      la = L + j * rs;
      const T*      lb = la + rs;
      V             a[S], b[S];
      for (int s = 0; s < S; ++s) {
        std::memcpy(&a[s], ya + s * VL, sizeof(V));
        std::memcpy(&b[s], yb + s * VL, sizeof(V));
      }
      for (I64 p = 0; p < j; ++p) {
        const T* yp = Y + p * ldy;
        const V  fa = broadcast<V>(-la[p * cs]), fb = broadcast<V>(-lb[p * cs]);
        for (int s = 0; s < S; ++s) {
          V v;
          std::memcpy(&v, yp + s * VL, sizeof(V));
          a[s] += fa * v;
          b[s] += fb * v;
        }
      }
      if (!unit) {
        const V d = broadcast<V>(la[j * cs]);
        for (int s = 0; s < S; ++s) a[s] = a[s] / d;
      }
      const V f = broadcast<V>(-lb[j * cs]);
      for (int s = 0; s < S; ++s) {
        std::memcpy(ya + s * VL, &a[s], sizeof(V));
        b[s] += f * a[s];
      }
      if (!unit) {
        const V d = broadcast<V>(lb[(j + 1) * cs]);
        for (int s = 0; s < S; ++s) b[s] = b[s] / d;
      }
      for (int s = 0; s < S; ++s) std::memcpy(yb + s * VL, &b[s], sizeof(V));
    }
    if (j < m) {
      T* __restrict ya = Y + j * ldy;
      const T*      la = L + j * rs;
      V             a[S];
      for (int s = 0; s < S; ++s) std::memcpy(&a[s], ya + s * VL, sizeof(V));
      for (I64 p = 0; p < j; ++p) {
        const T* yp = Y + p * ldy;
        const V  fa = broadcast<V>(-la[p * cs]);
        for (int s = 0; s < S; ++s) {
          V v;
          std::memcpy(&v, yp + s * VL, sizeof(V));
          a[s] += fa * v;
        }
      }
      if (!unit) {
        const V d = broadcast<V>(la[j * cs]);
        for (int s = 0; s < S; ++s) a[s] = a[s] / d;
      }
      for (int s = 0; s < S; ++s) std::memcpy(ya + s * VL, &a[s], sizeof(V));
    }
  }
}

/// Forward substitution on the rows of Y [m, w] (row stride ldy): for j in order,
/// Y_j = (Y_j - sum_{p<j} L(j, p) Y_p) / L(j, j) with L(j, p) = L[j * rs + p * cs], or without
/// the division when unit. Strides may be negative (an upper triangle runs as a reversed lower
/// one). Each column's arithmetic is the same wherever it sits in Y.
template <typename T, int VecBytes>
void trsm_rows_kernel(I64 m, I64 w, const T* L, I64 rs, I64 cs, bool unit, T* Y, I64 ldy) {
  constexpr int VL = Vec<T, VecBytes>::lanes;
  I64           c  = 0;
  for (; c + 4 * VL <= w; c += 4 * VL) trsm_rows_strip<T, VecBytes, 4>(m, L, rs, cs, unit, Y + c, ldy);
  for (; c + VL <= w; c += VL) trsm_rows_strip<T, VecBytes, 1>(m, L, rs, cs, unit, Y + c, ldy);
  for (; c < w; ++c) trsm_rows_strip<T, VecBytes, 0>(m, L, rs, cs, unit, Y + c, ldy);
}

/// Lane masks for potrf_small_kernel: row r has zeros in lanes [0, r) and ones after.
template <typename T, int VL>
struct HeadMasks {
  T m[VL][VL];
  constexpr HeadMasks() : m{} {
    for (int r = 0; r < VL; ++r)
      for (int l = 0; l < VL; ++l) m[r][l] = l < r ? T(0) : T(1);
  }
};

/// Lower Cholesky factor of the lower triangle of a [n, n] (row stride lda) into the lower
/// triangle of l (row stride ldl; may alias a); nothing above the diagonal is written. False if
/// not positive definite. Right-looking on U = L^T in work (n * (n + 8) + 8 values), so every
/// update runs along a contiguous row of U. Rows of U are vector aligned and every vector loop
/// starts at an aligned column: the lanes before the diagonal hold zeros (masked out of each
/// update), the columns past n are zero padding, and zeros stay zero.
template <typename T, int VecBytes>
bool potrf_small_kernel(I64 n, const T* a, I64 lda, T* l, I64 ldl, T* work) {
  using V          = typename Vec<T, VecBytes>::type;
  constexpr int VL = Vec<T, VecBytes>::lanes;
  alignas(64) static constexpr HeadMasks<T, VL> kMask{};
  const I64     ld = (n + VL - 1) / VL * VL;
  T* __restrict u  = reinterpret_cast<T*>((reinterpret_cast<std::uintptr_t>(work) + 63) & ~std::uintptr_t{63});
  for (I64 i = 0; i < n; ++i) {
    for (I64 j = i / VL * VL; j < i; ++j) u[i * ld + j] = T(0);
    for (I64 j = n; j < ld; ++j) u[i * ld + j] = T(0);
  }
  for (I64 j = 0; j < n; ++j)  // rows of a's lower triangle into columns of U
    for (I64 i = 0; i <= j; ++i) u[i * ld + j] = a[j * lda + i];
  for (I64 k = 0; k < n; ++k) {
    T* __restrict uk = u + k * ld;
    if (!(uk[k] > T(0))) return false;
    const T   d  = std::sqrt(uk[k]);
    const V   dv = broadcast<V>(d);
    const I64 k0 = k / VL * VL;
    V         v;
    std::memcpy(&v, uk + k0, sizeof(V));
    v         = v / dv;
    v[k - k0] = d;
    std::memcpy(uk + k0, &v, sizeof(V));
    for (I64 j = k0 + VL; j < n; j += VL) {
      std::memcpy(&v, uk + j, sizeof(V));
      v = v / dv;
      std::memcpy(uk + j, &v, sizeof(V));
    }
    for (I64 i = k + 1; i < n; ++i) {
      T* __restrict ui = u + i * ld;
      const I64     i0 = i / VL * VL;
      const V       f  = broadcast<V>(-uk[i]);
      V             x, y, fm;
      std::memcpy(&fm, kMask.m[i - i0], sizeof(V));
      fm = fm * f;
      std::memcpy(&x, uk + i0, sizeof(V));
      std::memcpy(&y, ui + i0, sizeof(V));
      y += fm * x;
      std::memcpy(ui + i0, &y, sizeof(V));
      for (I64 j = i0 + VL; j < n; j += VL) {
        std::memcpy(&x, uk + j, sizeof(V));
        std::memcpy(&y, ui + j, sizeof(V));
        y += f * x;
        std::memcpy(ui + j, &y, sizeof(V));
      }
    }
  }
  for (I64 j = 0; j < n; ++j)
    for (I64 i = 0; i <= j; ++i) l[j * ldl + i] = u[i * ld + j];
  return true;
}

}  // namespace
}  // namespace datamunge::nn::detail

/// Defines the exported entry points of one instruction-set unit.
#define DATAMUNGE_BLAS_DEFINE_ISA(ns, VEC_BYTES)                                                                   \
  namespace datamunge::nn::detail::ns {                                                                            \
  void sgemm(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,       \
             std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c, std::int64_t ldc) {        \
    gemm_packed<float, VEC_BYTES>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                           \
  }                                                                                                                \
  void dgemm(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double* a,     \
             std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c, std::int64_t ldc) {      \
    gemm_packed<double, VEC_BYTES>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                          \
  }                                                                                                                \
  void sgemm_mt(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,    \
                std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c, std::int64_t ldc,          \
                std::int64_t threads, ParallelRun run) {                                                           \
    gemm_packed_mt<float, VEC_BYTES>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threads, run);          \
  }                                                                                                                \
  void dgemm_mt(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double* a,  \
                std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c, std::int64_t ldc,       \
                std::int64_t threads, ParallelRun run) {                                                           \
    gemm_packed_mt<double, VEC_BYTES>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threads, run);         \
  }                                                                                                                \
  void sgemv(bool trans, std::int64_t m, std::int64_t n, float alpha, const float* a, std::int64_t lda,           \
             const float* x, std::int64_t incx, float beta, float* y, std::int64_t incy) {                         \
    gemv_kernel<float, VEC_BYTES>(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);                             \
  }                                                                                                                \
  void dgemv(bool trans, std::int64_t m, std::int64_t n, double alpha, const double* a, std::int64_t lda,         \
             const double* x, std::int64_t incx, double beta, double* y, std::int64_t incy) {                      \
    gemv_kernel<double, VEC_BYTES>(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);                            \
  }                                                                                                                \
  float sdot(std::int64_t n, const float* x, std::int64_t incx, const float* y, std::int64_t incy) {              \
    return dot_kernel<float, VEC_BYTES>(n, x, incx, y, incy);                                                      \
  }                                                                                                                \
  double ddot(std::int64_t n, const double* x, std::int64_t incx, const double* y, std::int64_t incy) {           \
    return dot_kernel<double, VEC_BYTES>(n, x, incx, y, incy);                                                     \
  }                                                                                                                \
  void saxpy(std::int64_t n, float alpha, const float* x, std::int64_t incx, float* y, std::int64_t incy) {       \
    axpy_kernel<float, VEC_BYTES>(n, alpha, x, incx, y, incy);                                                     \
  }                                                                                                                \
  void daxpy(std::int64_t n, double alpha, const double* x, std::int64_t incx, double* y, std::int64_t incy) {    \
    axpy_kernel<double, VEC_BYTES>(n, alpha, x, incx, y, incy);                                                    \
  }                                                                                                                \
  void sscal(std::int64_t n, float alpha, float* x, std::int64_t incx) {                                          \
    scal_kernel<float, VEC_BYTES>(n, alpha, x, incx);                                                              \
  }                                                                                                                \
  void dscal(std::int64_t n, double alpha, double* x, std::int64_t incx) {                                        \
    scal_kernel<double, VEC_BYTES>(n, alpha, x, incx);                                                             \
  }                                                                                                                \
  void dgemv_pair(std::int64_t m, std::int64_t n, double alpha, const double* a, std::int64_t lda,                \
                  const double* x, const double* z, double* y, double* t) {                                       \
    gemv_pair_kernel<double, VEC_BYTES>(m, n, alpha, a, lda, x, z, y, t);                                         \
  }                                                                                                               \
  void dtrsm_rows(std::int64_t m, std::int64_t w, const double* l, std::int64_t rs, std::int64_t cs, bool unit,    \
                  double* y, std::int64_t ldy) {                                                                  \
    trsm_rows_kernel<double, VEC_BYTES>(m, w, l, rs, cs, unit, y, ldy);                                           \
  }                                                                                                               \
  bool dpotrf_small(std::int64_t n, const double* a, std::int64_t lda, double* l, std::int64_t ldl,               \
                    double* work) {                                                                               \
    return potrf_small_kernel<double, VEC_BYTES>(n, a, lda, l, ldl, work);                                        \
  }                                                                                                               \
  }
