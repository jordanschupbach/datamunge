/// @file blas.cpp
/// @brief Run-time instruction-set selection for the in-house BLAS and the
///        threaded public entry points (datamunge/nn/blas.hpp).
///
/// The kernels live in blas_kernels.hpp and are compiled three times with
/// different -m flags (blas_avx512.cpp, blas_avx2.cpp, blas_generic.cpp). The
/// best set the CPU supports is chosen on first use; DATAMUNGE_BLAS_ISA or
/// blas::set_isa() can force another one (benchmarks compare them). GEMM goes
/// through gemm.cpp, which splits one product across the nn thread pool; the
/// level-1/2 routines are split here so that results do not depend on the
/// thread count: GEMV, AXPY and SCAL over their outputs, DOT in fixed chunks
/// whose partial sums are added in order (also on one thread).

#include "datamunge/nn/blas.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "blas_dispatch.hpp"
#include "datamunge/nn/parallel.hpp"
#include "gemm.hpp"

namespace datamunge::nn::detail {

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#define DATAMUNGE_BLAS_X86 1
DATAMUNGE_BLAS_DECLARE_ISA(blas_avx512)
DATAMUNGE_BLAS_DECLARE_ISA(blas_avx2)
#endif
DATAMUNGE_BLAS_DECLARE_ISA(blas_generic)

namespace {

#define DATAMUNGE_BLAS_TABLE(ns, label)                                                                            \
  BlasKernels {                                                                                                    \
    label, ns::sgemm, ns::dgemm, ns::sgemm_mt, ns::dgemm_mt, ns::sgemv, ns::dgemv, ns::sdot, ns::ddot, ns::saxpy,  \
        ns::daxpy, ns::sscal, ns::dscal, ns::dgemv_pair, ns::dtrsm_rows, ns::dpotrf_small                          \
  }

#if defined(DATAMUNGE_BLAS_X86)
const BlasKernels kAvx512 = DATAMUNGE_BLAS_TABLE(blas_avx512, "avx512");
const BlasKernels kAvx2   = DATAMUNGE_BLAS_TABLE(blas_avx2, "avx2");
#endif
const BlasKernels kGeneric = DATAMUNGE_BLAS_TABLE(blas_generic, "generic");

/// Whether this CPU (and OS) can run the given kernel set.
bool cpu_supports(const BlasKernels& k) {
  if (&k == &kGeneric) return true;
#if defined(DATAMUNGE_BLAS_X86) && (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  if (&k == &kAvx512)
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("fma");
  if (&k == &kAvx2) return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
  return false;
}

/// Every kernel set, best first.
std::vector<const BlasKernels*> all_kernels() {
  return {
#if defined(DATAMUNGE_BLAS_X86)
      &kAvx512, &kAvx2,
#endif
      &kGeneric};
}

const BlasKernels* find_kernels(const std::string& name) {
  for (const BlasKernels* k : all_kernels())
    if (name == k->name) return k;
  return nullptr;
}

/// The best supported set, or the one named by DATAMUNGE_BLAS_ISA when usable.
const BlasKernels* detect() {
  if (const char* env = std::getenv("DATAMUNGE_BLAS_ISA"))
    if (const BlasKernels* k = find_kernels(env); k && cpu_supports(*k)) return k;
  for (const BlasKernels* k : all_kernels())
    if (cpu_supports(*k)) return k;
  return &kGeneric;
}

std::atomic<const BlasKernels*> g_kernels{nullptr};

/// Elements per partial sum of DOT: fixed (whatever the thread count, also one
/// thread), so the result is reproducible.
constexpr std::int64_t kDotChunk = std::int64_t{1} << 13;
/// Smallest piece of a level-1 routine worth a pool task.
constexpr std::int64_t kVectorGrain = std::int64_t{1} << 13;
/// Level-1 routines run on the calling thread below 4 pieces or 256 KB of
/// vectors: measured at 8 threads with the vectors in cache, two pieces take
/// longer than one call (a pool dispatch is ~1 us), and so does a float SCAL
/// of 2^15 elements; AXPY and DOT of 2^15 floats are 1.4-1.7x faster split.
constexpr std::int64_t kVectorParallelMinElems = 4 * kVectorGrain;
constexpr std::int64_t kVectorParallelMinBytes = std::int64_t{1} << 18;

}  // namespace

const BlasKernels& blas_kernels() {
  const BlasKernels* k = g_kernels.load(std::memory_order_acquire);
  if (!k) {
    k = detect();
    const BlasKernels* expected = nullptr;
    if (!g_kernels.compare_exchange_strong(expected, k, std::memory_order_acq_rel)) k = expected;
  }
  return *k;
}

}  // namespace datamunge::nn::detail

namespace datamunge::nn::blas {

using detail::blas_kernels;
using I64 = std::int64_t;

std::string isa() { return blas_kernels().name; }

std::vector<std::string> available_isas() {
  std::vector<std::string> out;
  for (const detail::BlasKernels* k : detail::all_kernels())
    if (detail::cpu_supports(*k)) out.emplace_back(k->name);
  return out;
}

bool set_isa(const std::string& name) {
  const detail::BlasKernels* k = name == "auto" ? detail::detect() : detail::find_kernels(name);
  if (!k || !detail::cpu_supports(*k)) return false;
  detail::g_kernels.store(k, std::memory_order_release);
  return true;
}

void gemm(bool trans_a, bool trans_b, I64 m, I64 n, I64 k, float alpha, const float* a, I64 lda, const float* b,
          I64 ldb, float beta, float* c, I64 ldc) {
  detail::gemm(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(bool trans_a, bool trans_b, I64 m, I64 n, I64 k, double alpha, const double* a, I64 lda, const double* b,
          I64 ldb, double beta, double* c, I64 ldc) {
  detail::gemm(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

namespace {

/// Matrices below this many elements run on the calling thread (a pool
/// dispatch costs more than they take; at 2^17 a product takes ~30 us).
constexpr I64 kGemvParallelElems = I64{1} << 17;

/// y = alpha * op(A) x + beta * y over the pool, split over y: every element is
/// computed by one thread in the kernel's own order (A x: one row's dot
/// product; A^T x: the stored rows accumulated in order), and the pieces are
/// whole 64-element blocks, so the kernel's vector loop covers the same
/// elements as in one call. The result is that of one call on one thread.
template <typename T, typename Fn>
void gemv_threaded(bool trans, I64 m, I64 n, T alpha, const T* a, I64 lda, const T* x, I64 incx, T beta, T* y,
                   I64 incy, Fn kernel) {
  const I64 rows = trans ? n : m, cols = trans ? m : n;  // lengths of y and x
  if (rows <= 0) return;
  constexpr I64 kAlign = 64;
  if (rows * cols < kGemvParallelElems || cols <= 0 || alpha == T(0) || rows < 2 * kAlign) {
    kernel(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    return;
  }
  const I64 blocks = (rows + kAlign - 1) / kAlign;
  parallel_for(0, blocks, std::max<I64>(1, (I64{1} << 16) / (kAlign * cols)), [&](I64 b0, I64 b1) {
    const I64 lo = b0 * kAlign, hi = std::min(rows, b1 * kAlign);
    if (trans)  // columns [lo, hi) of the stored [m, n] matrix
      kernel(true, m, hi - lo, alpha, a + lo, lda, x, incx, beta, y + lo * incy, incy);
    else
      kernel(false, hi - lo, n, alpha, a + lo * lda, lda, x, incx, beta, y + lo * incy, incy);
  });
}

/// Whether a level-1 routine over n elements of `vectors` vectors is worth
/// splitting across the pool (see kVectorParallelMinElems).
template <typename T>
bool split_worthwhile(I64 n, I64 vectors) {
  return n >= detail::kVectorParallelMinElems &&
         n * vectors * static_cast<I64>(sizeof(T)) >= detail::kVectorParallelMinBytes;
}

/// DOT as partial sums over fixed kDotChunk-element chunks, added in order: the
/// same arithmetic on one thread or many.
template <typename T, typename Fn>
T dot_threaded(I64 n, const T* x, I64 incx, const T* y, I64 incy, Fn kernel) {
  if (n <= 0) return T(0);
  if (n <= detail::kDotChunk) return kernel(n, x, incx, y, incy);
  const I64      chunks = (n + detail::kDotChunk - 1) / detail::kDotChunk;
  std::vector<T> partial(static_cast<std::size_t>(chunks));
  const auto     run = [&](I64 lo, I64 hi) {
    for (I64 c = lo; c < hi; ++c) {
      const I64 first = c * detail::kDotChunk, count = std::min(detail::kDotChunk, n - first);
      partial[static_cast<std::size_t>(c)] = kernel(count, x + first * incx, incx, y + first * incy, incy);
    }
  };
  if (split_worthwhile<T>(n, 2))
    parallel_for(0, chunks, std::max<I64>(1, detail::kVectorGrain / detail::kDotChunk), run);
  else
    run(0, chunks);
  T s = T(0);
  for (T p : partial) s += p;
  return s;
}

/// AXPY and SCAL have no reduction: any split gives the same result. Pieces are
/// whole multiples of 64 elements (cache lines of both types).
template <typename T, typename F>
void split_vector(I64 n, I64 vectors, F&& piece) {
  if (!split_worthwhile<T>(n, vectors)) {
    piece(I64{0}, n);
    return;
  }
  constexpr I64 kAlign = 64;
  const I64     blocks = (n + kAlign - 1) / kAlign;
  parallel_for(0, blocks, detail::kVectorGrain / kAlign,
               [&](I64 lo, I64 hi) { piece(lo * kAlign, std::min(n, hi * kAlign)); });
}

template <typename T, typename Fn>
void axpy_threaded(I64 n, T alpha, const T* x, I64 incx, T* y, I64 incy, Fn kernel) {
  if (n <= 0) return;
  split_vector<T>(n, 2, [&](I64 lo, I64 hi) { kernel(hi - lo, alpha, x + lo * incx, incx, y + lo * incy, incy); });
}

template <typename T, typename Fn>
void scal_threaded(I64 n, T alpha, T* x, I64 incx, Fn kernel) {
  if (n <= 0) return;
  split_vector<T>(n, 1, [&](I64 lo, I64 hi) { kernel(hi - lo, alpha, x + lo * incx, incx); });
}

}  // namespace

void gemv(bool trans, I64 m, I64 n, float alpha, const float* a, I64 lda, const float* x, I64 incx, float beta,
          float* y, I64 incy) {
  gemv_threaded<float>(trans, m, n, alpha, a, lda, x, incx, beta, y, incy, blas_kernels().sgemv);
}

void gemv(bool trans, I64 m, I64 n, double alpha, const double* a, I64 lda, const double* x, I64 incx, double beta,
          double* y, I64 incy) {
  gemv_threaded<double>(trans, m, n, alpha, a, lda, x, incx, beta, y, incy, blas_kernels().dgemv);
}

float dot(I64 n, const float* x, I64 incx, const float* y, I64 incy) {
  return dot_threaded<float>(n, x, incx, y, incy, blas_kernels().sdot);
}

double dot(I64 n, const double* x, I64 incx, const double* y, I64 incy) {
  return dot_threaded<double>(n, x, incx, y, incy, blas_kernels().ddot);
}

void axpy(I64 n, float alpha, const float* x, I64 incx, float* y, I64 incy) {
  axpy_threaded<float>(n, alpha, x, incx, y, incy, blas_kernels().saxpy);
}

void axpy(I64 n, double alpha, const double* x, I64 incx, double* y, I64 incy) {
  axpy_threaded<double>(n, alpha, x, incx, y, incy, blas_kernels().daxpy);
}

void scal(I64 n, float alpha, float* x, I64 incx) { scal_threaded<float>(n, alpha, x, incx, blas_kernels().sscal); }

void scal(I64 n, double alpha, double* x, I64 incx) {
  scal_threaded<double>(n, alpha, x, incx, blas_kernels().dscal);
}

}  // namespace datamunge::nn::blas
