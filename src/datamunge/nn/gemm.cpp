// GEMM backends: the in-house packed kernels (blas_kernels.hpp, selected per
// instruction set by blas.cpp) or, when the library was built with
// DATAMUNGE_USE_BLAS for comparison, a CBLAS. Either way one product is split
// across the nn thread pool (see gemm_parallel), so a convolution's single
// large GEMM uses every core instead of one.

#include "gemm.hpp"

#include <algorithm>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include "blas_dispatch.hpp"
#include "datamunge/nn/parallel.hpp"

#if defined(DATAMUNGE_HAVE_CBLAS)
#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

namespace datamunge::nn::detail {

namespace {

using I64 = std::int64_t;

/// The in-house kernel set for this CPU, on the calling thread.
template <typename T>
void gemm_inhouse(bool ta, bool tb, I64 m, I64 n, I64 k, T alpha, const T* a, I64 lda, const T* b, I64 ldb, T beta,
                  T* c, I64 ldc) {
  const BlasKernels& kernels = blas_kernels();
  if constexpr (std::is_same<T, float>::value)
    kernels.sgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  else
    kernels.dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/// One serial product on the calling thread: CBLAS when built with it, else the in-house kernel.
template <typename T>
void gemm_serial(bool ta, bool tb, I64 m, I64 n, I64 k, T alpha, const T* a, I64 lda, const T* b, I64 ldb, T beta,
                 T* c, I64 ldc) {
#if defined(DATAMUNGE_HAVE_CBLAS)
  if (k > 0) {
    if constexpr (std::is_same<T, float>::value)
      cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans, static_cast<int>(m),
                  static_cast<int>(n), static_cast<int>(k), alpha, a, static_cast<int>(lda), b, static_cast<int>(ldb),
                  beta, c, static_cast<int>(ldc));
    else
      cblas_dgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans, static_cast<int>(m),
                  static_cast<int>(n), static_cast<int>(k), alpha, a, static_cast<int>(lda), b, static_cast<int>(ldb),
                  beta, c, static_cast<int>(ldc));
    return;
  }
#endif
  gemm_inhouse<T>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// ---- splitting one product across the pool -------------------------------------------------
//
// Products below kSerialFlop are cheaper on one thread than the pool wake-up.
// With the in-house kernels, a product whose op(B) has more than
// kSharedPackElems elements runs on the kernels' threaded driver, which packs
// each block of op(B) once for all threads (gemm_packed_mt); a smaller op(B)
// is cheaper to re-pack per thread than to share. Otherwise the rows of C (M)
// are split when there are enough of them, the
// depth (K) is split into partial products when M and N are both short (the
// filter gradient of a convolution: M = kh kw cin, N = cout, K = every output
// pixel of the batch), and the columns (N) as a last resort. Whether K is split
// and into how many chunks depends on the shape alone (a single thread computes
// the same chunks one after another), and the partials are summed in chunk
// order, so the result is deterministic and independent of the thread count;
// M and N splits do not change any element's summation order.

constexpr I64 kSerialFlop      = I64{1} << 20;  // 1 MFLOP, ~10 us on one core (a few pool dispatches)
constexpr I64 kMinRows         = 32;            // rows of C per M chunk
constexpr I64 kMinCols         = 32;            // columns of C per N chunk
constexpr I64 kMinDepth        = 256;           // K per partial product
constexpr I64 kMaxDepthChunks  = 32;
constexpr I64 kMaxPartialElems = I64{1} << 18;  // m * n above this keeps the K split off (1 MB float partials)
constexpr I64 kPartialBudget   = I64{1} << 22;  // elements of all partials together (16 MB float)
constexpr I64 kMinDepthChunks  = 4;
constexpr I64 kDepthMaxRows    = 8 * kMinRows;  // with more rows than this, splitting M keeps 8 threads busy
constexpr I64 kSharedPackElems = I64{1} << 14;  // op(B) above this is packed once and shared by the threads
constexpr I64 kRowAlign        = 12;  // multiple of every kernel set's register tile height (MR)
constexpr I64 kColAlign        = 32;  // and width (NR)

#if defined(DATAMUNGE_HAVE_OPENBLAS)
/// OpenBLAS would otherwise start its own pool inside every chunk; the nn pool
/// already supplies the parallelism, so BLAS runs single-threaded.
void pin_blas_threads() {
  static std::once_flag once;
  std::call_once(once, [] { openblas_set_num_threads(1); });
}
#else
void pin_blas_threads() {}
#endif

#if !defined(DATAMUNGE_HAVE_CBLAS)
/// The nn pool as the kernels' task runner (BlasParallelRun).
void pool_run(I64 count, void (*body)(void*, I64), void* ctx) {
  parallel_for(0, count, 1, [&](I64 lo, I64 hi) {
    for (I64 i = lo; i < hi; ++i) body(ctx, i);
  });
}
#endif

template <typename T>
void gemm_parallel(bool ta, bool tb, I64 m, I64 n, I64 k, T alpha, const T* a, I64 lda, const T* b, I64 ldb, T beta,
                   T* c, I64 ldc) {
  if (m <= 0 || n <= 0) return;
  pin_blas_threads();
  const I64 threads = get_num_threads();
  const I64 flop    = 2 * m * n * k;
  if (k <= 0 || flop < kSerialFlop) {
    gemm_serial<T>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return;
  }
  const I64  row_chunks   = m / kMinRows;
  const I64  col_chunks   = n / kMinCols;
  const I64  depth_chunks = std::min<I64>({k / kMinDepth, kMaxDepthChunks, kPartialBudget / (m * n)});
  const bool split_k      = m < kDepthMaxRows && m * n <= kMaxPartialElems && depth_chunks >= kMinDepthChunks;

  // Blocks start at multiples of the register tile, so every element of C lands
  // in the same kind of tile (full or edge) as in a serial product and is
  // rounded the same way whatever the thread count.
  const auto blocks_of = [&](I64 extent, I64 minimum, I64 align) {
    const I64 per = (std::max<I64>(minimum, (extent + threads - 1) / threads) + align - 1) / align * align;
    return std::pair<I64, I64>{per, (extent + per - 1) / per};
  };
  // Rows of C: op(A)'s row block starts at a + lo (stored transposed as [K, M]) or a + lo * lda.
  // One block per thread: every block re-packs op(B), so more blocks cost more than they balance.
  const auto split_rows = [&] {
    const auto [per, count] = blocks_of(m, kMinRows, kRowAlign);
    parallel_for(0, count, 1, [&](I64 b0, I64 b1) {
      for (I64 blk = b0; blk < b1; ++blk) {
        const I64 lo = blk * per, hi = std::min(m, lo + per);
        gemm_serial<T>(ta, tb, hi - lo, n, k, alpha, ta ? a + lo : a + lo * lda, lda, b, ldb, beta, c + lo * ldc, ldc);
      }
    });
  };
  // Columns of C: op(B)'s column block starts at b + lo (stored [K, N]) or b + lo * ldb (stored transposed as [N, K]).
  const auto split_cols = [&] {
    const auto [per, count] = blocks_of(n, kMinCols, kColAlign);
    parallel_for(0, count, 1, [&](I64 b0, I64 b1) {
      for (I64 blk = b0; blk < b1; ++blk) {
        const I64 lo = blk * per, hi = std::min(n, lo + per);
        gemm_serial<T>(ta, tb, m, hi - lo, k, alpha, a, lda, tb ? b + lo * ldb : b + lo, ldb, beta, c + lo, ldc);
      }
    });
  };
  // Depth: chunk c multiplies op(A)[:, k0:k1] by op(B)[k0:k1, :] into its own partial,
  // then C = beta C + alpha * sum of partials in chunk order.
  const auto split_depth = [&] {
    const I64 chunks = depth_chunks;
    const I64 depth  = (k + chunks - 1) / chunks;
    const I64 plane  = m * n;
    static thread_local std::vector<T> partials;
    if (partials.size() < static_cast<std::size_t>(chunks * plane)) partials.resize(static_cast<std::size_t>(chunks * plane));
    T* parts = partials.data();
    parallel_for(0, chunks, 1, [&](I64 lo, I64 hi) {
      for (I64 ch = lo; ch < hi; ++ch) {
        const I64 k0 = ch * depth, k1 = std::min(k, k0 + depth);
        if (k0 >= k1) {
          std::fill(parts + ch * plane, parts + (ch + 1) * plane, T(0));
          continue;
        }
        gemm_serial<T>(ta, tb, m, n, k1 - k0, T(1), ta ? a + k0 * lda : a + k0, lda, tb ? b + k0 : b + k0 * ldb, ldb,
                       T(0), parts + ch * plane, n);
      }
    });
    parallel_for(0, m, std::max<I64>(1, 16384 / n), [&](I64 lo, I64 hi) {
      for (I64 i = lo; i < hi; ++i) {
        T*       crow = c + i * ldc;
        const T* p0   = parts + i * n;
        if (beta == T(0))
          for (I64 j = 0; j < n; ++j) crow[j] = alpha * p0[j];
        else if (beta == T(1))
          for (I64 j = 0; j < n; ++j) crow[j] += alpha * p0[j];
        else
          for (I64 j = 0; j < n; ++j) crow[j] = beta * crow[j] + alpha * p0[j];
        for (I64 ch = 1; ch < chunks; ++ch) {
          const T* p = parts + ch * plane + i * n;
          for (I64 j = 0; j < n; ++j) crow[j] += alpha * p[j];
        }
      }
    });
  };

  if (split_k)
    split_depth();  // also with one thread, so the sums never depend on the thread count
  else if (threads <= 1)
    gemm_serial<T>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#if !defined(DATAMUNGE_HAVE_CBLAS)
  else if ((row_chunks >= 2 || col_chunks >= 2) && n * k > kSharedPackElems) {
    const BlasKernels& kernels = blas_kernels();
    if constexpr (std::is_same<T, float>::value)
      kernels.sgemm_mt(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threads, pool_run);
    else
      kernels.dgemm_mt(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threads, pool_run);
  }
#endif
  else if (row_chunks >= 2)
    split_rows();
  else if (col_chunks >= 2)
    split_cols();
  else
    gemm_serial<T>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

}  // namespace

void gemm_local(bool ta, bool tb, I64 m, I64 n, I64 k, float alpha, const float* a, I64 lda, const float* b, I64 ldb,
                float beta, float* c, I64 ldc) {
  if (m <= 0 || n <= 0) return;
  pin_blas_threads();
  gemm_serial<float>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm_local(bool ta, bool tb, I64 m, I64 n, I64 k, double alpha, const double* a, I64 lda, const double* b,
                I64 ldb, double beta, double* c, I64 ldc) {
  if (m <= 0 || n <= 0) return;
  pin_blas_threads();
  gemm_serial<double>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm_builtin(bool ta, bool tb, I64 m, I64 n, I64 k, float alpha, const float* a, I64 lda, const float* b, I64 ldb,
                  float beta, float* c, I64 ldc) {
  gemm_inhouse<float>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm_builtin(bool ta, bool tb, I64 m, I64 n, I64 k, double alpha, const double* a, I64 lda, const double* b,
                  I64 ldb, double beta, double* c, I64 ldc) {
  gemm_inhouse<double>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(bool ta, bool tb, I64 m, I64 n, I64 k, float alpha, const float* a, I64 lda, const float* b, I64 ldb,
          float beta, float* c, I64 ldc) {
  gemm_parallel<float>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(bool ta, bool tb, I64 m, I64 n, I64 k, double alpha, const double* a, I64 lda, const double* b, I64 ldb,
          double beta, double* c, I64 ldc) {
  gemm_parallel<double>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

}  // namespace datamunge::nn::detail
