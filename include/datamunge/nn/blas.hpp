#pragma once

/// @file blas.hpp
/// @brief In-house BLAS for the CPU: GEMM, GEMV, DOT, AXPY and SCAL on float and
///        double, with kernels selected at run time for the instruction set.
///
/// The library ships its own linear-algebra kernels (no OpenBLAS, ATLAS or MKL):
/// a packed GEMM in the GotoBLAS/BLIS style with register-tiled micro-kernels,
/// compiled once per instruction set (AVX-512, AVX2+FMA, and a baseline that
/// works everywhere) and chosen from the CPU features at first use. GEMM, GEMV
/// and the level-1 routines run on the nn thread pool (see parallel.hpp:
/// `set_num_threads`) and give the same result for any thread count.
///
/// All matrices are row-major. `gemm` computes
/// `C = alpha * op(A) @ op(B) + beta * C` with op(A) [m, k] and op(B) [k, n];
/// `lda`, `ldb`, `ldc` are the row strides of the matrices as stored, so a
/// transposed A is stored [k, m] with `lda >= m`. `gemv` computes
/// `y = alpha * op(A) x + beta * y` with A stored [m, n] (op(A) is [n, m] when
/// `trans`). Increments follow BLAS (`incx`, `incy` are element strides).
///
/// @par Example
/// @code
/// std::vector<float> a(64 * 32, 1.f), b(32 * 16, 2.f), c(64 * 16, 0.f);
/// dm::nn::blas::gemm(false, false, 64, 16, 32, 1.f, a.data(), 32, b.data(), 16, 0.f, c.data(), 16);
/// std::print("{} {}\n", c[0], dm::nn::blas::isa()); // 64 avx512
/// @endcode

#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::nn::blas {

/// @brief Name of the kernel set in use: "avx512", "avx2" or "generic".
/// @return The instruction set selected by `set_isa` or, by default, the best
///         one this CPU supports (the environment variable DATAMUNGE_BLAS_ISA
///         overrides the detection).
std::string isa();

/// @brief Kernel sets this CPU can run, best first.
std::vector<std::string> available_isas();

/// @brief Selects a kernel set by name ("avx512", "avx2", "generic"; "auto" restores detection).
/// @return False, with no change, when the name is unknown or the CPU lacks the instructions.
bool set_isa(const std::string& name);

/// @brief C = alpha * op(A) @ op(B) + beta * C, threaded over the nn pool.
/// @param trans_a Whether A is stored transposed, as [k, m].
/// @param trans_b Whether B is stored transposed, as [n, k].
void gemm(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,
          std::int64_t lda, const float* b, std::int64_t ldb, float beta, float* c, std::int64_t ldc);
/// @copydoc gemm(bool,bool,std::int64_t,std::int64_t,std::int64_t,float,const float*,std::int64_t,const float*,std::int64_t,float,float*,std::int64_t)
void gemm(bool trans_a, bool trans_b, std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double* a,
          std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c, std::int64_t ldc);

/// @brief y = alpha * op(A) x + beta * y with A stored [m, n] row-major (op(A) is A^T when `trans`).
void gemv(bool trans, std::int64_t m, std::int64_t n, float alpha, const float* a, std::int64_t lda, const float* x,
          std::int64_t incx, float beta, float* y, std::int64_t incy);
/// @copydoc gemv(bool,std::int64_t,std::int64_t,float,const float*,std::int64_t,const float*,std::int64_t,float,float*,std::int64_t)
void gemv(bool trans, std::int64_t m, std::int64_t n, double alpha, const double* a, std::int64_t lda,
          const double* x, std::int64_t incx, double beta, double* y, std::int64_t incy);

/// @brief Sum of x[i] * y[i] over n elements with the given strides.
float  dot(std::int64_t n, const float* x, std::int64_t incx, const float* y, std::int64_t incy);
/// @copydoc dot(std::int64_t,const float*,std::int64_t,const float*,std::int64_t)
double dot(std::int64_t n, const double* x, std::int64_t incx, const double* y, std::int64_t incy);

/// @brief y += alpha * x over n elements with the given strides.
void axpy(std::int64_t n, float alpha, const float* x, std::int64_t incx, float* y, std::int64_t incy);
/// @copydoc axpy(std::int64_t,float,const float*,std::int64_t,float*,std::int64_t)
void axpy(std::int64_t n, double alpha, const double* x, std::int64_t incx, double* y, std::int64_t incy);

/// @brief x *= alpha over n elements with the given stride.
void scal(std::int64_t n, float alpha, float* x, std::int64_t incx);
/// @copydoc scal(std::int64_t,float,float*,std::int64_t)
void scal(std::int64_t n, double alpha, double* x, std::int64_t incx);

}  // namespace datamunge::nn::blas
