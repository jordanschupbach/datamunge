#pragma once

// Run-time selection between the instruction-set units of the in-house BLAS
// (blas_avx512.cpp, blas_avx2.cpp, blas_generic.cpp). Not installed.

#include <cstdint>

namespace datamunge::nn::detail {

/// Runs body(ctx, i) for every i in [0, count) on a thread pool and returns
/// when all have finished; the threaded GEMM of a kernel set calls it.
using BlasParallelRun = void (*)(std::int64_t count, void (*body)(void*, std::int64_t), void* ctx);

/// Entry points one instruction-set unit exports (see DATAMUNGE_BLAS_DEFINE_ISA).
#define DATAMUNGE_BLAS_DECLARE_ISA(ns)                                                                             \
  namespace ns {                                                                                                   \
  void   sgemm(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,     \
               std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c, std::int64_t ldc);       \
  void   dgemm(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double* a,   \
               std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c, std::int64_t ldc);    \
  void   sgemm_mt(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,  \
                  std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c, std::int64_t ldc,        \
                  std::int64_t threads, ::datamunge::nn::detail::BlasParallelRun run);                             \
  void   dgemm_mt(bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double* a,\
                  std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c, std::int64_t ldc,     \
                  std::int64_t threads, ::datamunge::nn::detail::BlasParallelRun run);                             \
  void   sgemv(bool trans, std::int64_t m, std::int64_t n, float alpha, const float* a, std::int64_t lda,         \
               const float* x, std::int64_t incx, float beta, float* y, std::int64_t incy);                       \
  void   dgemv(bool trans, std::int64_t m, std::int64_t n, double alpha, const double* a, std::int64_t lda,       \
               const double* x, std::int64_t incx, double beta, double* y, std::int64_t incy);                    \
  float  sdot(std::int64_t n, const float* x, std::int64_t incx, const float* y, std::int64_t incy);              \
  double ddot(std::int64_t n, const double* x, std::int64_t incx, const double* y, std::int64_t incy);            \
  void   saxpy(std::int64_t n, float alpha, const float* x, std::int64_t incx, float* y, std::int64_t incy);      \
  void   daxpy(std::int64_t n, double alpha, const double* x, std::int64_t incx, double* y, std::int64_t incy);   \
  void   sscal(std::int64_t n, float alpha, float* x, std::int64_t incx);                                         \
  void   dscal(std::int64_t n, double alpha, double* x, std::int64_t incx);                                       \
  void   dgemv_pair(std::int64_t m, std::int64_t n, double alpha, const double* a, std::int64_t lda,              \
                    const double* x, const double* z, double* y, double* t);                                      \
  void   dtrsm_rows(std::int64_t m, std::int64_t w, const double* l, std::int64_t rs, std::int64_t cs, bool unit,  \
                    double* y, std::int64_t ldy);                                                                 \
  bool   dpotrf_small(std::int64_t n, const double* a, std::int64_t lda, double* l, std::int64_t ldl,             \
                      double* work);                                                                              \
  }

/// One kernel set: entry points that run on the calling thread, plus the threaded GEMMs.
struct BlasKernels {
  const char* name;
  void (*sgemm)(bool, bool, std::int64_t, std::int64_t, std::int64_t, float, const float*, std::int64_t, const float*,
                std::int64_t, float, float*, std::int64_t);
  void (*dgemm)(bool, bool, std::int64_t, std::int64_t, std::int64_t, double, const double*, std::int64_t,
                const double*, std::int64_t, double, double*, std::int64_t);
  /// GEMM split over `threads` pool tasks sharing each packed block of op(B);
  /// identical results for every thread count.
  void (*sgemm_mt)(bool, bool, std::int64_t, std::int64_t, std::int64_t, float, const float*, std::int64_t,
                   const float*, std::int64_t, float, float*, std::int64_t, std::int64_t, BlasParallelRun);
  void (*dgemm_mt)(bool, bool, std::int64_t, std::int64_t, std::int64_t, double, const double*, std::int64_t,
                   const double*, std::int64_t, double, double*, std::int64_t, std::int64_t, BlasParallelRun);
  void (*sgemv)(bool, std::int64_t, std::int64_t, float, const float*, std::int64_t, const float*, std::int64_t, float,
                float*, std::int64_t);
  void (*dgemv)(bool, std::int64_t, std::int64_t, double, const double*, std::int64_t, const double*, std::int64_t,
                double, double*, std::int64_t);
  float (*sdot)(std::int64_t, const float*, std::int64_t, const float*, std::int64_t);
  double (*ddot)(std::int64_t, const double*, std::int64_t, const double*, std::int64_t);
  void (*saxpy)(std::int64_t, float, const float*, std::int64_t, float*, std::int64_t);
  void (*daxpy)(std::int64_t, double, const double*, std::int64_t, double*, std::int64_t);
  void (*sscal)(std::int64_t, float, float*, std::int64_t);
  void (*dscal)(std::int64_t, double, double*, std::int64_t);
  /// y += alpha A^T x and t += alpha A z in one pass over the row-major [m, n] A (unit strides).
  void (*dgemv_pair)(std::int64_t, std::int64_t, double, const double*, std::int64_t, const double*, const double*,
                     double*, double*);
  /// Forward substitution on the rows of Y [m, w] (row stride ldy) with the triangle
  /// L(j, p) = l[j * rs + p * cs] (strides may be negative; unit: L(j, j) = 1):
  /// Y_j = (Y_j - sum_{p<j} L(j, p) Y_p) / L(j, j) in order of j.
  void (*dtrsm_rows)(std::int64_t, std::int64_t, const double*, std::int64_t, std::int64_t, bool, double*,
                     std::int64_t);
  /// Lower Cholesky factor of the lower triangle of a [n, n] (row stride lda) into the lower
  /// triangle of l (row stride ldl; may alias a), with n * (n + 8) + 8 values of work; false if not
  /// positive definite. For the small blocks of the factorizations.
  bool (*dpotrf_small)(std::int64_t, const double*, std::int64_t, double*, std::int64_t, double*);
};

/// The kernel set in use (CPU detection, DATAMUNGE_BLAS_ISA, or blas::set_isa).
const BlasKernels& blas_kernels();

}  // namespace datamunge::nn::detail
