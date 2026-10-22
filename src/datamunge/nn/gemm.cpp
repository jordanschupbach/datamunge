// GEMM backends: BLAS when available, otherwise a blocked, threaded kernel.

#include "gemm.hpp"

#include <algorithm>
#include <vector>

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

constexpr std::int64_t kBlockM = 64;
constexpr std::int64_t kBlockK = 256;

/// Copies op(X) ([rows, cols]) into a packed row-major buffer.
template <typename T>
std::vector<T> pack(bool trans, std::int64_t rows, std::int64_t cols, const T* x, std::int64_t ld) {
  std::vector<T> out(static_cast<std::size_t>(rows * cols));
  for (std::int64_t i = 0; i < rows; ++i)
    for (std::int64_t j = 0; j < cols; ++j)
      out[static_cast<std::size_t>(i * cols + j)] = trans ? x[j * ld + i] : x[i * ld + j];
  return out;
}

template <typename T>
void gemm_blocked(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, T alpha, const T* a,
                  std::int64_t lda, const T* b, std::int64_t ldb, T beta, T* c, std::int64_t ldc) {
  if (m <= 0 || n <= 0) return;
  // Scale C by beta first, so the kernel only accumulates.
  parallel_for(0, m, std::max<std::int64_t>(1, 16384 / std::max<std::int64_t>(n, 1)), [&](std::int64_t lo, std::int64_t hi) {
    for (std::int64_t i = lo; i < hi; ++i) {
      T* row = c + i * ldc;
      if (beta == T(0))
        std::fill(row, row + n, T(0));
      else if (beta != T(1))
        for (std::int64_t j = 0; j < n; ++j) row[j] *= beta;
    }
  });
  if (k <= 0 || alpha == T(0)) return;

  std::vector<T> a_packed, b_packed;
  const T*       ap = a;
  const T*       bp = b;
  std::int64_t   a_ld = lda, b_ld = ldb;
  if (trans_a) {
    a_packed = pack(true, m, k, a, lda);
    ap       = a_packed.data();
    a_ld     = k;
  }
  if (trans_b) {
    b_packed = pack(true, k, n, b, ldb);
    bp       = b_packed.data();
    b_ld     = n;
  }

  const std::int64_t blocks = (m + kBlockM - 1) / kBlockM;
  const std::int64_t work   = kBlockM * n * k;
  parallel_for(0, blocks, std::max<std::int64_t>(1, (1 << 16) / std::max<std::int64_t>(work, 1)),
               [&](std::int64_t lo, std::int64_t hi) {
                 for (std::int64_t blk = lo; blk < hi; ++blk) {
                   const std::int64_t i0 = blk * kBlockM, i1 = std::min(m, i0 + kBlockM);
                   for (std::int64_t l0 = 0; l0 < k; l0 += kBlockK) {
                     const std::int64_t l1 = std::min(k, l0 + kBlockK);
                     for (std::int64_t i = i0; i < i1; ++i) {
                       T*       crow = c + i * ldc;
                       const T* arow = ap + i * a_ld;
                       for (std::int64_t l = l0; l < l1; ++l) {
                         const T  av   = alpha * arow[l];
                         const T* brow = bp + l * b_ld;
                         for (std::int64_t j = 0; j < n; ++j) crow[j] += av * brow[j];
                       }
                     }
                   }
                 }
               });
}

}  // namespace

void gemm_builtin(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,
                  std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c, std::int64_t ldc) {
  gemm_blocked<float>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm_builtin(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double* a,
                  std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c, std::int64_t ldc) {
  gemm_blocked<double>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,
          std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c, std::int64_t ldc) {
  if (m <= 0 || n <= 0) return;
#if defined(DATAMUNGE_HAVE_CBLAS)
  if (k > 0) {
    cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans, static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha, a, static_cast<int>(lda), b, static_cast<int>(ldb),
                beta, c, static_cast<int>(ldc));
    return;
  }
#endif
  gemm_blocked<float>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double* a,
          std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c, std::int64_t ldc) {
  if (m <= 0 || n <= 0) return;
#if defined(DATAMUNGE_HAVE_CBLAS)
  if (k > 0) {
    cblas_dgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans, static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha, a, static_cast<int>(lda), b, static_cast<int>(ldb),
                beta, c, static_cast<int>(ldc));
    return;
  }
#endif
  gemm_blocked<double>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

}  // namespace datamunge::nn::detail
