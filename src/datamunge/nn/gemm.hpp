#pragma once

// Row-major GEMM used by matmul and convolution. Not installed.
//
//   C = alpha * op(A) @ op(B) + beta * C
//
// op(A) is [M, K] and op(B) is [K, N]; lda/ldb/ldc are row strides of the
// matrices as stored (so a transposed A is stored [K, M] with lda >= M).
// float and double go to the in-house packed kernels (blas_kernels.hpp,
// one instruction-set unit chosen at run time) split across the nn thread
// pool, or to a CBLAS when the library was built with DATAMUNGE_USE_BLAS.

#include <cstdint>

namespace datamunge::nn::detail {

void gemm(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,
          std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c, std::int64_t ldc);
void gemm(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double* a,
          std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c, std::int64_t ldc);

/// One product on the calling thread (CBLAS when built with it, else the
/// in-house kernel): for callers that are already inside a parallel region
/// and own a cache-sized tile.
void gemm_local(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,
                std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c, std::int64_t ldc);
void gemm_local(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                const double* a, std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c,
                std::int64_t ldc);

/// The in-house kernel on the calling thread whatever the build, so tests and
/// benchmarks can compare it with a CBLAS.
void gemm_builtin(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                  const float* a, std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c,
                  std::int64_t ldc);
void gemm_builtin(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                  const double* a, std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c,
                  std::int64_t ldc);

}  // namespace datamunge::nn::detail
