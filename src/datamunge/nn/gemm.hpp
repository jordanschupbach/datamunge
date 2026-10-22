#pragma once

// Row-major GEMM used by matmul and convolution. Not installed.
//
//   C = alpha * op(A) @ op(B) + beta * C
//
// op(A) is [M, K] and op(B) is [K, N]; lda/ldb/ldc are row strides of the
// matrices as stored (so a transposed A is stored [K, M] with lda >= M).
// float and double go to BLAS when the library was built with one
// (DATAMUNGE_HAVE_CBLAS), else to a blocked, threaded kernel.

#include <cstdint>

namespace datamunge::nn::detail {

void gemm(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,
          std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c, std::int64_t ldc);
void gemm(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double* a,
          std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c, std::int64_t ldc);

/// The portable kernel, exposed so tests and benchmarks can compare it with BLAS.
void gemm_builtin(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                  const float* a, std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c,
                  std::int64_t ldc);
void gemm_builtin(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                  const double* a, std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c,
                  std::int64_t ldc);

}  // namespace datamunge::nn::detail
