// Batched dense linear algebra with gradients.
//
// Kernels work on one packed float64 matrix at a time (inputs are converted
// on the way in and results back to the input dtype on the way out) and run
// over the batch with parallel_for. Gradients mirror TensorFlow's
// linalg_grad.py and are built from differentiable ops, so they can be
// differentiated again.
//
// With DATAMUNGE_HAVE_LAPACK, large matrices go to LAPACK (and triangular
// solves to BLAS trsm with DATAMUNGE_HAVE_CBLAS); the built-in kernels stay
// for small matrices and as the fallback. LAPACK is column-major: a packed
// row-major matrix is the column-major transpose, which several drivers use
// directly instead of copying.

#include "datamunge/nn/linalg.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <string>

#include "datamunge/nn/parallel.hpp"
#include "detail.hpp"

#if defined(__APPLE__) && (defined(DATAMUNGE_HAVE_LAPACK) || defined(DATAMUNGE_HAVE_CBLAS))
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#elif defined(DATAMUNGE_HAVE_CBLAS)
#include <cblas.h>
#endif

#if defined(DATAMUNGE_HAVE_LAPACK)
#if defined(__APPLE__)
using datamunge_lapack_int = __LAPACK_int;
#else
// OpenBLAS built with INTERFACE64 (as in nixpkgs) takes 64-bit integers in its LAPACK entry
// points too; its cblas.h, included above, defines OPENBLAS_USE64BITINT in that case.
#if defined(DATAMUNGE_LAPACK_ILP64) || defined(OPENBLAS_USE64BITINT)
using datamunge_lapack_int = std::int64_t;
#else
using datamunge_lapack_int = int;
#endif
// Fortran LAPACK entry points (reference / OpenBLAS / MKL naming). The hidden
// string-length arguments are omitted: every character argument has length 1.
// Matrices of a batch are factorised concurrently, so the LAPACK must be
// thread-safe (OpenBLAS's and reference LAPACK built with -frecursive are; the
// legacy f2c CLAPACK ABI in Accelerate is not, hence ACCELERATE_NEW_LAPACK).
extern "C" {
using dm_lpint = datamunge_lapack_int;
void dgesdd_(const char* jobz, const dm_lpint* m, const dm_lpint* n, double* a, const dm_lpint* lda, double* s,
             double* u, const dm_lpint* ldu, double* vt, const dm_lpint* ldvt, double* work, const dm_lpint* lwork,
             dm_lpint* iwork, dm_lpint* info);
void dgesvd_(const char* jobu, const char* jobvt, const dm_lpint* m, const dm_lpint* n, double* a, const dm_lpint* lda,
             double* s, double* u, const dm_lpint* ldu, double* vt, const dm_lpint* ldvt, double* work,
             const dm_lpint* lwork, dm_lpint* info);
void dsyevd_(const char* jobz, const char* uplo, const dm_lpint* n, double* a, const dm_lpint* lda, double* w,
             double* work, const dm_lpint* lwork, dm_lpint* iwork, const dm_lpint* liwork, dm_lpint* info);
void dgeqrf_(const dm_lpint* m, const dm_lpint* n, double* a, const dm_lpint* lda, double* tau, double* work,
             const dm_lpint* lwork, dm_lpint* info);
void dorgqr_(const dm_lpint* m, const dm_lpint* n, const dm_lpint* k, double* a, const dm_lpint* lda, const double* tau,
             double* work, const dm_lpint* lwork, dm_lpint* info);
void dpotrf_(const char* uplo, const dm_lpint* n, double* a, const dm_lpint* lda, dm_lpint* info);
void dgetrf_(const dm_lpint* m, const dm_lpint* n, double* a, const dm_lpint* lda, dm_lpint* ipiv, dm_lpint* info);
void dgetrs_(const char* trans, const dm_lpint* n, const dm_lpint* nrhs, const double* a, const dm_lpint* lda,
             const dm_lpint* ipiv, double* b, const dm_lpint* ldb, dm_lpint* info);
void dgeev_(const char* jobvl, const char* jobvr, const dm_lpint* n, double* a, const dm_lpint* lda, double* wr,
            double* wi, double* vl, const dm_lpint* ldvl, double* vr, const dm_lpint* ldvr, double* work,
            const dm_lpint* lwork, dm_lpint* info);
}
#endif
#endif

namespace datamunge::nn {

using detail::nograd;

namespace {

using I64   = std::int64_t;
using Grads = std::vector<Tensor>;

// ---- shape helpers -------------------------------------------------------------------------

DType float_dtype(const Tensor& t) { return is_floating(t.dtype()) ? t.dtype() : DType::Float32; }

Tensor as_float(const Tensor& t) { return detail::as_dtype(t, float_dtype(t)); }

void require_matrix(const Tensor& a, const char* what) {
  if (a.dim() < 2)
    throw std::invalid_argument(std::string(what) + ": expected a tensor of rank >= 2, got " + shape_str(a.shape()));
}

void require_square(const Tensor& a, const char* what) {
  require_matrix(a, what);
  if (a.size(-1) != a.size(-2))
    throw std::invalid_argument(std::string(what) + ": expected square matrices, got " + shape_str(a.shape()));
}

Shape batch_shape(const Shape& s) { return Shape(s.begin(), s.end() - 2); }

Shape with_matrix(Shape batch, I64 m, I64 n) {
  batch.push_back(m);
  batch.push_back(n);
  return batch;
}

Tensor mT(const Tensor& x) { return swapaxes(x, -1, -2); }

/// Packed float64 copy (or alias) for read-only kernel input.
Tensor packed_f64(const Tensor& x) {
  NoGradGuard guard;
  return detail::cast_raw(x, DType::Float64).contiguous();
}

/// Converts a kernel result back to dtype.
Tensor from_f64(const Tensor& x, DType dtype) { return detail::cast_raw(x, dtype); }

/// Runs f(p) for every matrix p of the batch; `cost` is the work per matrix.
template <typename F>
void for_batch(I64 nb, I64 cost, F&& f) {
  const I64 grain = std::max<I64>(1, I64{1 << 16} / std::max<I64>(cost, 1));
  parallel_for(0, nb, grain, [&](I64 lo, I64 hi) {
    for (I64 p = lo; p < hi; ++p) f(p);
  });
}

// ---- backend selection ---------------------------------------------------------------------

std::atomic<int> g_backend{static_cast<int>(LinalgBackend::Auto)};

#if defined(DATAMUNGE_HAVE_LAPACK)
constexpr bool kHaveLapack = true;
#else
constexpr bool kHaveLapack = false;
#endif

/// Kernels with a LAPACK (or BLAS) path.
enum class Op { Svd, SvdValues, Eigh, EighValues, Qr, Cholesky, Lu, TriSolve, Eig };

/// Smallest size (min(m, n), or n) from which Auto uses LAPACK, the measured
/// break-even of LAPACK (Accelerate, M-series Mac) against the built-in
/// kernels for single matrices and batches with 1 and 10 threads. Below it the
/// per-call overhead (workspace queries, layout copies) dominates. LU and trsm
/// stay built-in up to 64 also because Accelerate's own threading
/// oversubscribes the pool on batches of 32 x 32 matrices (0.4x). geev is
/// always used when available: its balancing makes it more robust on badly
/// scaled input, while the built-in eig (faster for small batches) is only
/// the fallback without LAPACK.
I64 lapack_min_size(Op op) {
  switch (op) {
    case Op::Svd: return 16;        // 1.2x at 16, 4-5x at 64, 40x at 512
    case Op::SvdValues: return 8;   // 1.0-1.2x at 8, 9x at 64
    case Op::Eigh: return 96;       // ~1x at 64, 1.9x at 96, 26x at 512
    case Op::EighValues: return 8;  // 1.2-1.3x at 8, 2x at 64
    case Op::Qr: return 128;        // 0.7-1.5x up to 128 (not monotone), 2.5x at 256
    case Op::Cholesky: return 32;   // 1.3x at 32, 25x at 512
    case Op::Lu: return 64;         // 1.0-1.6x at 64, 5x at 512
    case Op::TriSolve: return 64;   // 1.5-3x at 64, 13x at 512
    case Op::Eig: return 0;
  }
  return 0;
}

[[maybe_unused]] bool use_lapack(Op op, I64 size) {
  const auto b = static_cast<LinalgBackend>(g_backend.load(std::memory_order_relaxed));
  if (b == LinalgBackend::Builtin || size <= 0) return false;  // LAPACK rejects empty matrices (lda >= 1)
  const bool avail = op == Op::TriSolve ?
#if defined(DATAMUNGE_HAVE_CBLAS)
                                        true
#else
                                        false
#endif
                                        : kHaveLapack;
  if (!avail) return false;
  return b == LinalgBackend::Lapack || size >= lapack_min_size(op);
}

#if defined(DATAMUNGE_HAVE_LAPACK)
// ---- LAPACK wrappers (column-major; workspace queries; return info) -----------------------------

namespace lapack {

using LI = datamunge_lapack_int;

LI li(I64 x) { return static_cast<LI>(x); }

/// Workspace size from a query's work[0].
LI lwork_of(double w) { return std::max<LI>(1, static_cast<LI>(w)); }

LI gesdd(char jobz, I64 m, I64 n, double* a, I64 lda, double* s, double* u, I64 ldu, double* vt, I64 ldvt) {
  const LI M = li(m), N = li(n), LDA = li(lda), LDU = li(ldu), LDVT = li(ldvt);
  LI       info = 0, lwork = -1;
  double   q    = 0;
  std::vector<LI> iwork(static_cast<std::size_t>(8 * std::max<I64>(1, std::min(m, n))));
  dgesdd_(&jobz, &M, &N, a, &LDA, s, u, &LDU, vt, &LDVT, &q, &lwork, iwork.data(), &info);
  if (info != 0) return info;
  lwork = lwork_of(q);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dgesdd_(&jobz, &M, &N, a, &LDA, s, u, &LDU, vt, &LDVT, work.data(), &lwork, iwork.data(), &info);
  return info;
}

LI gesvd(char job, I64 m, I64 n, double* a, I64 lda, double* s, double* u, I64 ldu, double* vt, I64 ldvt) {
  const LI M = li(m), N = li(n), LDA = li(lda), LDU = li(ldu), LDVT = li(ldvt);
  LI       info = 0, lwork = -1;
  double   q    = 0;
  dgesvd_(&job, &job, &M, &N, a, &LDA, s, u, &LDU, vt, &LDVT, &q, &lwork, &info);
  if (info != 0) return info;
  lwork = lwork_of(q);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dgesvd_(&job, &job, &M, &N, a, &LDA, s, u, &LDU, vt, &LDVT, work.data(), &lwork, &info);
  return info;
}

LI syevd(char jobz, char uplo, I64 n, double* a, double* w) {
  const LI N = li(n);
  LI       info = 0, lwork = -1, liwork = -1, iq = 0;
  double   q    = 0;
  dsyevd_(&jobz, &uplo, &N, a, &N, w, &q, &lwork, &iq, &liwork, &info);
  if (info != 0) return info;
  lwork  = lwork_of(q);
  liwork = std::max<LI>(1, iq);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  std::vector<LI>     iwork(static_cast<std::size_t>(liwork));
  dsyevd_(&jobz, &uplo, &N, a, &N, w, work.data(), &lwork, iwork.data(), &liwork, &info);
  return info;
}

LI geqrf(I64 m, I64 n, double* a, I64 lda, double* tau) {
  const LI M = li(m), N = li(n), LDA = li(lda);
  LI       info = 0, lwork = -1;
  double   q    = 0;
  dgeqrf_(&M, &N, a, &LDA, tau, &q, &lwork, &info);
  if (info != 0) return info;
  lwork = lwork_of(q);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dgeqrf_(&M, &N, a, &LDA, tau, work.data(), &lwork, &info);
  return info;
}

LI orgqr(I64 m, I64 n, I64 k, double* a, I64 lda, const double* tau) {
  const LI M = li(m), N = li(n), K = li(k), LDA = li(lda);
  LI       info = 0, lwork = -1;
  double   q    = 0;
  dorgqr_(&M, &N, &K, a, &LDA, tau, &q, &lwork, &info);
  if (info != 0) return info;
  lwork = lwork_of(q);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dorgqr_(&M, &N, &K, a, &LDA, tau, work.data(), &lwork, &info);
  return info;
}

LI potrf(char uplo, I64 n, double* a) {
  const LI N = li(n);
  LI       info = 0;
  dpotrf_(&uplo, &N, a, &N, &info);
  return info;
}

LI getrf(I64 n, double* a, LI* ipiv) {
  const LI N = li(n);
  LI       info = 0;
  dgetrf_(&N, &N, a, &N, ipiv, &info);
  return info;
}

LI getrs(char trans, I64 n, I64 nrhs, const double* a, const LI* ipiv, double* b) {
  const LI N = li(n), NRHS = li(nrhs);
  LI       info = 0;
  dgetrs_(&trans, &N, &NRHS, a, &N, ipiv, b, &N, &info);
  return info;
}

LI geev(bool vectors, I64 n, double* a, double* wr, double* wi, double* vr) {
  const char jobvl = 'N', jobvr = vectors ? 'V' : 'N';
  const LI   N = li(n), one = 1, LDVR = vectors ? N : 1;
  LI         info = 0, lwork = -1;
  double     q = 0, dummy = 0;
  dgeev_(&jobvl, &jobvr, &N, a, &N, wr, wi, &dummy, &one, vectors ? vr : &dummy, &LDVR, &q, &lwork, &info);
  if (info != 0) return info;
  lwork = lwork_of(q);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dgeev_(&jobvl, &jobvr, &N, a, &N, wr, wi, &dummy, &one, vectors ? vr : &dummy, &LDVR, work.data(), &lwork, &info);
  return info;
}

}  // namespace lapack
#endif

// ---- sign conventions ------------------------------------------------------------------------

/// -1 if the largest-magnitude entry (first on ties) of column c of x [rows, cols] is negative, else 1.
double column_sign(const double* x, I64 rows, I64 cols, I64 c) {
  double best = -1, val = 0;
  for (I64 i = 0; i < rows; ++i) {
    const double v = x[i * cols + c];
    if (std::abs(v) > best) best = std::abs(v), val = v;
  }
  return val < 0 ? -1.0 : 1.0;
}

void scale_column(double* x, I64 rows, I64 cols, I64 c, double f) {
  for (I64 i = 0; i < rows; ++i) x[i * cols + c] *= f;
}

/// SVD convention: U's columns have their largest entry positive, V's paired columns follow;
/// unpaired completion columns (full_matrices) get the same rule on their own.
void normalize_svd_signs(double* u, double* v, I64 m, I64 n, I64 uc, I64 vc) {
  const I64 k = std::min(m, n);
  for (I64 j = 0; j < uc; ++j) {
    const double f = column_sign(u, m, uc, j);
    if (f > 0) continue;
    scale_column(u, m, uc, j, f);
    if (j < k) scale_column(v, n, vc, j, f);
  }
  for (I64 j = k; j < vc; ++j) {
    const double f = column_sign(v, n, vc, j);
    if (f < 0) scale_column(v, n, vc, j, f);
  }
}

/// eigh convention: every eigenvector has its largest entry positive.
void normalize_eigh_signs(double* v, I64 n) {
  for (I64 j = 0; j < n; ++j) {
    const double f = column_sign(v, n, n, j);
    if (f < 0) scale_column(v, n, n, j, f);
  }
}

/// A constant [m, n] mask of `dtype` with value(i, j) as entries.
template <typename F>
Tensor matrix_mask(I64 m, I64 n, DType dtype, F&& value) {
  Tensor  out = empty({m, n}, DType::Float64);
  double* p   = out.data_ptr<double>();
  for (I64 i = 0; i < m; ++i)
    for (I64 j = 0; j < n; ++j) p[i * n + j] = value(i, j);
  return from_f64(out, dtype);
}

Tensor eye_mask(I64 m, I64 n, DType dtype) {
  return matrix_mask(m, n, dtype, [](I64 i, I64 j) { return i == j ? 1.0 : 0.0; });
}

/// x * reciprocal(x^2 + eps), TF's _SafeReciprocal.
Tensor safe_reciprocal(const Tensor& x) { return mul(x, reciprocal(add(square(x), 1e-20))); }

/// f[..., i, j] = 1 / (v_j - v_i) off the diagonal, 0 on it.
Tensor pairwise_reciprocal(const Tensor& v) {
  const I64 n = v.size(-1);
  Tensor    f = safe_reciprocal(sub(expand_dims(v, -2), expand_dims(v, -1)));
  return mul(f, matrix_mask(n, n, v.dtype(), [](I64 i, I64 j) { return i == j ? 0.0 : 1.0; }));
}

/// Identity broadcast to [..., n, n] for the batch of `like`.
Tensor batched_eye(const Shape& batch, I64 n, DType dtype) { return broadcast_to(eye(n, dtype), with_matrix(batch, n, n)); }

// ---- diagonal primitives ---------------------------------------------------------------------

Tensor diag_part_impl(const Tensor& a);

/// [..., k] -> [..., m, n] with the values on the main diagonal (k = min(m, n)).
Tensor diag_embed(const Tensor& d, I64 m, I64 n) {
  const Tensor dc = as_float(d);
  const I64    k  = dc.size(-1);
  if (k != std::min(m, n)) throw std::invalid_argument("matrix_diag: diagonal length does not fit the matrix");
  Shape  batch(dc.shape().begin(), dc.shape().end() - 1);
  Tensor out;
  {
    NoGradGuard guard;
    out = zeros(with_matrix(batch, m, n), dc.dtype());
    detail::dispatch_float(dc.dtype(), "matrix_diag", [&](auto tag) {
      using T            = typename decltype(tag)::type;
      const Tensor src   = nograd(dc).contiguous();
      const T*     s     = src.data_ptr<T>();
      T*           o     = out.data_ptr<T>();
      const I64    nb    = shape_numel(batch);
      for (I64 p = 0; p < nb; ++p)
        for (I64 i = 0; i < k; ++i) o[p * m * n + i * n + i] = s[p * k + i];
    });
  }
  record("MatrixDiag", {dc}, {out}, [](const Grads& g, const Grads&) { return Grads{diag_part_impl(g[0])}; });
  return out;
}

Tensor diag_part_impl(const Tensor& a) {
  require_matrix(a, "matrix_diag_part");
  const Tensor ac = as_float(a);
  const I64    m = ac.size(-2), n = ac.size(-1), k = std::min(m, n);
  const Shape  batch = batch_shape(ac.shape());
  Shape        so    = batch;
  so.push_back(k);
  Tensor out;
  {
    NoGradGuard guard;
    out = empty(so, ac.dtype());
    detail::dispatch_float(ac.dtype(), "matrix_diag_part", [&](auto tag) {
      using T          = typename decltype(tag)::type;
      const Tensor src = nograd(ac).contiguous();
      const T*     s   = src.data_ptr<T>();
      T*           o   = out.data_ptr<T>();
      const I64    nb  = shape_numel(batch);
      for (I64 p = 0; p < nb; ++p)
        for (I64 i = 0; i < k; ++i) o[p * k + i] = s[p * m * n + i * n + i];
    });
  }
  record("MatrixDiagPart", {ac}, {out}, [m, n](const Grads& g, const Grads&) { return Grads{diag_embed(g[0], m, n)}; });
  return out;
}

// ---- kernels (float64, one packed matrix) ---------------------------------------------------

/// Lower Cholesky factor of the lower triangle of a; false if not positive definite.
bool cholesky_kernel(const double* a, double* l, I64 n) {
  std::fill(l, l + n * n, 0.0);
  for (I64 i = 0; i < n; ++i) {
    for (I64 j = 0; j < i; ++j) {
      double s = a[i * n + j];
      for (I64 k = 0; k < j; ++k) s -= l[i * n + k] * l[j * n + k];
      l[i * n + j] = s / l[j * n + j];
    }
    double s = a[i * n + i];
    for (I64 k = 0; k < i; ++k) s -= l[i * n + k] * l[i * n + k];
    if (!(s > 0.0)) return false;
    l[i * n + i] = std::sqrt(s);
  }
  return true;
}

/// x = op(A)^-1 b for triangular A [n, n], b and x [n, k] (may alias).
void triangular_solve_kernel(const double* a, const double* b, double* x, I64 n, I64 k, bool lower, bool adjoint) {
  if (x != b) std::copy(b, b + n * k, x);
  const auto A   = [&](I64 i, I64 j) { return adjoint ? a[j * n + i] : a[i * n + j]; };
  const bool low = lower != adjoint;
  if (low) {
    for (I64 i = 0; i < n; ++i) {
      double* xi = x + i * k;
      for (I64 j = 0; j < i; ++j) {
        const double  c  = A(i, j);
        const double* xj = x + j * k;
        for (I64 c2 = 0; c2 < k; ++c2) xi[c2] -= c * xj[c2];
      }
      const double d = A(i, i);
      for (I64 c2 = 0; c2 < k; ++c2) xi[c2] /= d;
    }
  } else {
    for (I64 i = n - 1; i >= 0; --i) {
      double* xi = x + i * k;
      for (I64 j = i + 1; j < n; ++j) {
        const double  c  = A(i, j);
        const double* xj = x + j * k;
        for (I64 c2 = 0; c2 < k; ++c2) xi[c2] -= c * xj[c2];
      }
      const double d = A(i, i);
      for (I64 c2 = 0; c2 < k; ++c2) xi[c2] /= d;
    }
  }
}

/// In-place LU with partial pivoting: lu = P A with unit-lower L below the
/// diagonal. Returns the permutation sign, or 0 if a pivot is exactly zero.
int lu_kernel(double* lu, I64* perm, I64 n) {
  int sign = 1;
  for (I64 i = 0; i < n; ++i) perm[i] = i;
  for (I64 c = 0; c < n; ++c) {
    I64    piv  = c;
    double best = std::abs(lu[c * n + c]);
    for (I64 r = c + 1; r < n; ++r)
      if (std::abs(lu[r * n + c]) > best) best = std::abs(lu[r * n + c]), piv = r;
    if (best == 0.0) return 0;
    if (piv != c) {
      std::swap_ranges(lu + c * n, lu + c * n + n, lu + piv * n);
      std::swap(perm[c], perm[piv]);
      sign = -sign;
    }
    const double d = lu[c * n + c];
    for (I64 r = c + 1; r < n; ++r) {
      const double f = lu[r * n + c] / d;
      lu[r * n + c]  = f;
      if (f == 0.0) continue;
      for (I64 j = c + 1; j < n; ++j) lu[r * n + j] -= f * lu[c * n + j];
    }
  }
  return sign;
}

/// Householder QR of a [m, n]: q [m, qc] (qc = m when full, else min(m, n)),
/// r [rr, n] (rr = m when full, else min(m, n)).
void qr_kernel(const double* a, double* q, double* r, I64 m, I64 n, bool full) {
  const I64           k = std::min(m, n), qc = full ? m : k, rr = full ? m : k;
  std::vector<double> w(a, a + m * n), vs(static_cast<std::size_t>(k * m), 0.0);
  std::vector<char>   used(static_cast<std::size_t>(k), 0);
  std::vector<double> dots(static_cast<std::size_t>(std::max(n, m)));
  for (I64 j = 0; j < k; ++j) {
    double tail = 0;
    for (I64 i = j + 1; i < m; ++i) tail += w[i * n + j] * w[i * n + j];
    if (tail == 0.0) continue;  // already reduced: no reflection (as LAPACK dlarfg and Eigen)
    const double xnorm = std::sqrt(tail + w[j * n + j] * w[j * n + j]);
    double* v = vs.data() + j * m;
    for (I64 i = j; i < m; ++i) v[i] = w[i * n + j];
    v[j] += w[j * n + j] >= 0 ? xnorm : -xnorm;
    double vnorm = 0;
    for (I64 i = j; i < m; ++i) vnorm += v[i] * v[i];
    vnorm = std::sqrt(vnorm);
    if (vnorm == 0.0) continue;
    for (I64 i = j; i < m; ++i) v[i] /= vnorm;
    used[static_cast<std::size_t>(j)] = 1;
    // w[j:, j:] -= 2 v (v^T w[j:, j:]), row by row for contiguous access
    std::fill(dots.begin(), dots.end(), 0.0);
    for (I64 i = j; i < m; ++i)
      for (I64 p = j; p < n; ++p) dots[static_cast<std::size_t>(p)] += v[i] * w[i * n + p];
    for (I64 i = j; i < m; ++i)
      for (I64 p = j; p < n; ++p) w[i * n + p] -= 2 * v[i] * dots[static_cast<std::size_t>(p)];
  }
  for (I64 i = 0; i < rr; ++i)
    for (I64 p = 0; p < n; ++p) r[i * n + p] = (i < m && p >= i) ? w[i * n + p] : 0.0;
  // Q = H_0 ... H_{k-1} I[:, :qc]
  std::fill(q, q + m * qc, 0.0);
  for (I64 i = 0; i < qc; ++i) q[i * qc + i] = 1.0;
  for (I64 j = k - 1; j >= 0; --j) {
    if (!used[static_cast<std::size_t>(j)]) continue;
    const double* v = vs.data() + j * m;
    std::fill(dots.begin(), dots.end(), 0.0);
    for (I64 i = j; i < m; ++i)
      for (I64 p = 0; p < qc; ++p) dots[static_cast<std::size_t>(p)] += v[i] * q[i * qc + p];
    for (I64 i = j; i < m; ++i)
      for (I64 p = 0; p < qc; ++p) q[i * qc + p] -= 2 * v[i] * dots[static_cast<std::size_t>(p)];
  }
}

/// Fills the columns of q [m, cols] flagged in `missing` with unit vectors
/// orthogonal to all other columns (Gram-Schmidt on the best standard basis vector).
void complete_basis(double* q, I64 m, I64 cols, std::vector<char> missing) {
  std::vector<double> cand(static_cast<std::size_t>(m)), best(static_cast<std::size_t>(m));
  for (I64 c = 0; c < cols; ++c) {
    if (!missing[static_cast<std::size_t>(c)]) continue;
    double best_norm = -1;
    for (I64 e = 0; e < m; ++e) {
      std::fill(cand.begin(), cand.end(), 0.0);
      cand[static_cast<std::size_t>(e)] = 1.0;
      for (int pass = 0; pass < 2; ++pass)
        for (I64 o = 0; o < cols; ++o) {
          if (missing[static_cast<std::size_t>(o)]) continue;
          double dot = 0;
          for (I64 i = 0; i < m; ++i) dot += q[i * cols + o] * cand[static_cast<std::size_t>(i)];
          for (I64 i = 0; i < m; ++i) cand[static_cast<std::size_t>(i)] -= dot * q[i * cols + o];
        }
      double nrm = 0;
      for (double v : cand) nrm += v * v;
      if (nrm > best_norm) best_norm = nrm, best = cand;
      if (nrm > 0.5) break;
    }
    const double inv = 1.0 / std::sqrt(best_norm);
    for (I64 i = 0; i < m; ++i) q[i * cols + c] = best[static_cast<std::size_t>(i)] * inv;
    missing[static_cast<std::size_t>(c)] = 0;
  }
}

/// Symmetric eigen decomposition of the lower triangle of a [n, n]
/// (Householder tridiagonalisation + implicit QL, after JAMA's tred2/tql2).
/// e ascending; v [n, n] with eigenvectors as columns.
void eigh_kernel(const double* a, double* e_out, double* v_out, I64 n) {
  if (n == 0) return;
  std::vector<std::vector<double>> V(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n)));
  for (I64 i = 0; i < n; ++i)
    for (I64 j = 0; j <= i; ++j) V[i][j] = V[j][i] = a[i * n + j];
  std::vector<double> d(static_cast<std::size_t>(n)), e(static_cast<std::size_t>(n));
  // tred2
  for (I64 j = 0; j < n; ++j) d[j] = V[n - 1][j];
  for (I64 i = n - 1; i > 0; --i) {
    double scale = 0.0, h = 0.0;
    for (I64 k = 0; k < i; ++k) scale += std::abs(d[k]);
    if (scale == 0.0) {
      e[i] = d[i - 1];
      for (I64 j = 0; j < i; ++j) {
        d[j]    = V[i - 1][j];
        V[i][j] = 0.0;
        V[j][i] = 0.0;
      }
    } else {
      for (I64 k = 0; k < i; ++k) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      double f = d[i - 1];
      double g = std::sqrt(h);
      if (f > 0) g = -g;
      e[i]     = scale * g;
      h        = h - f * g;
      d[i - 1] = f - g;
      for (I64 j = 0; j < i; ++j) e[j] = 0.0;
      for (I64 j = 0; j < i; ++j) {
        f       = d[j];
        V[j][i] = f;
        g       = e[j] + V[j][j] * f;
        for (I64 k = j + 1; k <= i - 1; ++k) {
          g += V[k][j] * d[k];
          e[k] += V[k][j] * f;
        }
        e[j] = g;
      }
      f = 0.0;
      for (I64 j = 0; j < i; ++j) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      const double hh = f / (h + h);
      for (I64 j = 0; j < i; ++j) e[j] -= hh * d[j];
      for (I64 j = 0; j < i; ++j) {
        f = d[j];
        g = e[j];
        for (I64 k = j; k <= i - 1; ++k) V[k][j] -= (f * e[k] + g * d[k]);
        d[j]    = V[i - 1][j];
        V[i][j] = 0.0;
      }
    }
    d[i] = h;
  }
  for (I64 i = 0; i < n - 1; ++i) {
    V[n - 1][i]    = V[i][i];
    V[i][i]        = 1.0;
    const double h = d[i + 1];
    if (h != 0.0) {
      for (I64 k = 0; k <= i; ++k) d[k] = V[k][i + 1] / h;
      for (I64 j = 0; j <= i; ++j) {
        double g = 0.0;
        for (I64 k = 0; k <= i; ++k) g += V[k][i + 1] * V[k][j];
        for (I64 k = 0; k <= i; ++k) V[k][j] -= g * d[k];
      }
    }
    for (I64 k = 0; k <= i; ++k) V[k][i + 1] = 0.0;
  }
  for (I64 j = 0; j < n; ++j) {
    d[j]        = V[n - 1][j];
    V[n - 1][j] = 0.0;
  }
  V[n - 1][n - 1] = 1.0;
  e[0]            = 0.0;
  // tql2, on the transposed eigenvector matrix (rotations then touch two contiguous rows)
  for (I64 i = 0; i < n; ++i)
    for (I64 j = i + 1; j < n; ++j) std::swap(V[i][j], V[j][i]);
  for (I64 i = 1; i < n; ++i) e[i - 1] = e[i];
  e[n - 1]          = 0.0;
  double       f    = 0.0, tst1 = 0.0;
  const double eps  = std::ldexp(1.0, -52);
  for (I64 l = 0; l < n; ++l) {
    tst1  = std::max(tst1, std::abs(d[l]) + std::abs(e[l]));
    I64 m = l;
    while (m < n) {
      if (std::abs(e[m]) <= eps * tst1) break;
      ++m;
    }
    if (m == n) m = n - 1;  // e[n-1] == 0, never reached for finite input
    if (m > l) {
      int iter = 0;
      do {
        if (++iter > 300) throw std::runtime_error("eigh: QL iteration did not converge (non-finite input?)");
        double g = d[l];
        double p = (d[l + 1] - g) / (2.0 * e[l]);
        double r = std::hypot(p, 1.0);
        if (p < 0) r = -r;
        d[l]             = e[l] / (p + r);
        d[l + 1]         = e[l] * (p + r);
        const double dl1 = d[l + 1];
        double       h   = g - d[l];
        for (I64 i = l + 2; i < n; ++i) d[i] -= h;
        f                = f + h;
        p                = d[m];
        double       c   = 1.0, c2 = c, c3 = c;
        const double el1 = e[l + 1];
        double       s = 0.0, s2 = 0.0;
        for (I64 i = m - 1; i >= l; --i) {
          c3       = c2;
          c2       = c;
          s2       = s;
          g        = c * e[i];
          h        = c * p;
          r        = std::hypot(p, e[i]);
          e[i + 1] = s * r;
          s        = e[i] / r;
          c        = p / r;
          p        = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          double* vi  = V[i].data();
          double* vi1 = V[i + 1].data();
          for (I64 k = 0; k < n; ++k) {
            h      = vi1[k];
            vi1[k] = s * vi[k] + c * h;
            vi[k]  = c * vi[k] - s * h;
          }
        }
        p    = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (std::abs(e[l]) > eps * tst1);
    }
    d[l] = d[l] + f;
    e[l] = 0.0;
  }
  // sort ascending (selection sort keeps the pairing)
  for (I64 i = 0; i < n - 1; ++i) {
    I64    k = i;
    double p = d[i];
    for (I64 j = i + 1; j < n; ++j)
      if (d[j] < p) k = j, p = d[j];
    if (k != i) {
      d[k] = d[i];
      d[i] = p;
      std::swap(V[i], V[k]);
    }
  }
  for (I64 i = 0; i < n; ++i) {
    e_out[i] = d[i];
    if (v_out)
      for (I64 j = 0; j < n; ++j) v_out[i * n + j] = V[j][i];
  }
}

/// SVD of a [m, n]: s [k] descending; u [m, uc], v [n, vc] where uc/vc are
/// k, or m/n with full. u, v may be null (values only).
void svd_kernel(const double* a, double* s_out, double* u_out, double* v_out, I64 m, I64 n, bool full) {
  const I64 k = std::min(m, n);
  if (m < n) {
    // svd(A^T) = V S U^T
    std::vector<double> at(static_cast<std::size_t>(m * n));
    for (I64 i = 0; i < m; ++i)
      for (I64 j = 0; j < n; ++j) at[j * m + i] = a[i * n + j];
    svd_kernel(at.data(), s_out, v_out, u_out, n, m, full);
    return;
  }
  // m >= n: B = R from a thin QR (tall) or A itself, n x n.
  std::vector<double> qmat, b(static_cast<std::size_t>(n * n));
  if (m > n) {
    qmat.resize(static_cast<std::size_t>(m * n));
    qr_kernel(a, qmat.data(), b.data(), m, n, false);
  } else {
    std::copy(a, a + n * n, b.begin());
  }
  // One-sided Jacobi on the columns of B: bt rows are B's columns, vt rows V's columns.
  std::vector<double> bt(static_cast<std::size_t>(n * n)), vt(static_cast<std::size_t>(n * n), 0.0);
  for (I64 i = 0; i < n; ++i)
    for (I64 j = 0; j < n; ++j) bt[j * n + i] = b[i * n + j];
  for (I64 i = 0; i < n; ++i) vt[i * n + i] = 1.0;
  const double tol = 1e-15;
  for (int sweep = 0; sweep < 80; ++sweep) {
    bool rotated = false;
    for (I64 p = 0; p < n - 1; ++p)
      for (I64 q = p + 1; q < n; ++q) {
        double* bp = bt.data() + p * n;
        double* bq = bt.data() + q * n;
        double  alpha = 0, beta = 0, gamma = 0;
        for (I64 i = 0; i < n; ++i) alpha += bp[i] * bp[i], beta += bq[i] * bq[i], gamma += bp[i] * bq[i];
        if (gamma == 0.0 || std::abs(gamma) <= tol * std::sqrt(alpha * beta)) continue;
        rotated            = true;
        const double zeta  = (beta - alpha) / (2 * gamma);
        const double t     = (zeta >= 0 ? 1.0 : -1.0) / (std::abs(zeta) + std::sqrt(1 + zeta * zeta));
        const double c     = 1 / std::sqrt(1 + t * t), sn = c * t;
        for (I64 i = 0; i < n; ++i) {
          const double x = bp[i], y = bq[i];
          bp[i] = c * x - sn * y;
          bq[i] = sn * x + c * y;
        }
        double* vp = vt.data() + p * n;
        double* vq = vt.data() + q * n;
        for (I64 i = 0; i < n; ++i) {
          const double x = vp[i], y = vq[i];
          vp[i] = c * x - sn * y;
          vq[i] = sn * x + c * y;
        }
      }
    if (!rotated) break;
  }
  std::vector<double> sv(static_cast<std::size_t>(n));
  for (I64 j = 0; j < n; ++j) {
    double s = 0;
    for (I64 i = 0; i < n; ++i) s += bt[j * n + i] * bt[j * n + i];
    sv[static_cast<std::size_t>(j)] = std::sqrt(s);
  }
  std::vector<I64> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), I64{0});
  std::stable_sort(order.begin(), order.end(), [&](I64 x, I64 y) { return sv[x] > sv[y]; });
  for (I64 j = 0; j < k; ++j) s_out[j] = sv[static_cast<std::size_t>(order[j])];
  if (!u_out) return;
  const I64 uc = full ? m : k, vc = full ? n : k;  // vc == n here
  // V columns
  for (I64 j = 0; j < vc; ++j)
    for (I64 i = 0; i < n; ++i) v_out[i * vc + j] = vt[order[j] * n + i];
  // U = Q (B columns / s)
  const double        smax = s_out[0];
  std::vector<double> ub(static_cast<std::size_t>(n * n), 0.0);  // [n, n], columns
  std::vector<char>   missing(static_cast<std::size_t>(uc), 0);
  for (I64 j = 0; j < n; ++j) {
    const double s = sv[static_cast<std::size_t>(order[j])];
    if (!(s > smax * 1e-14 * static_cast<double>(n))) {  // rank deficient: complete below
      missing[static_cast<std::size_t>(j)] = 1;
      continue;
    }
    for (I64 i = 0; i < n; ++i) ub[i * n + j] = bt[order[j] * n + i] / s;
  }
  std::fill(u_out, u_out + m * uc, 0.0);
  for (I64 i = 0; i < m; ++i)
    for (I64 j = 0; j < n; ++j) {
      if (missing[static_cast<std::size_t>(j)]) continue;
      double acc = 0;
      if (m > n)
        for (I64 l = 0; l < n; ++l) acc += qmat[i * n + l] * ub[l * n + j];
      else
        acc = ub[i * n + j];
      u_out[i * uc + j] = acc;
    }
  for (I64 j = n; j < uc; ++j) missing[static_cast<std::size_t>(j)] = 1;
  complete_basis(u_out, m, uc, missing);
}

void solve_lu_builtin_factors(const double* lu, const I64* perm, const double* bp, double* x, I64 n, I64 k) {
  for (I64 i = 0; i < n; ++i) std::copy(bp + perm[i] * k, bp + perm[i] * k + k, x + i * k);
  // unit lower, then upper
  for (I64 i = 0; i < n; ++i)
    for (I64 j = 0; j < i; ++j) {
      const double c = lu[i * n + j];
      for (I64 c2 = 0; c2 < k; ++c2) x[i * k + c2] -= c * x[j * k + c2];
    }
  for (I64 i = n - 1; i >= 0; --i) {
    for (I64 j = i + 1; j < n; ++j) {
      const double c = lu[i * n + j];
      for (I64 c2 = 0; c2 < k; ++c2) x[i * k + c2] -= c * x[j * k + c2];
    }
    const double d = lu[i * n + i];
    for (I64 c2 = 0; c2 < k; ++c2) x[i * k + c2] /= d;
  }
}

/// Built-in LU solve of op(A) X = B for one matrix; false if A is singular.
bool lu_solve_builtin(const double* ap, const double* bp, double* x, I64 n, I64 k, bool adjoint) {
  std::vector<double> lu(static_cast<std::size_t>(n * n));
  std::vector<I64>    perm(static_cast<std::size_t>(n));
  for (I64 i = 0; i < n; ++i)
    for (I64 j = 0; j < n; ++j) lu[i * n + j] = adjoint ? ap[j * n + i] : ap[i * n + j];
  if (lu_kernel(lu.data(), perm.data(), n) == 0) return false;
  solve_lu_builtin_factors(lu.data(), perm.data(), bp, x, n, k);
  return true;
}

/// Built-in sign, log|det| and det of one matrix (sign 0 when singular).
void lu_det_builtin(const double* a, I64 n, double& sign, double& logabs, double& det) {
  std::vector<double> lu(a, a + n * n);
  std::vector<I64>    perm(static_cast<std::size_t>(n));
  const int           sg = lu_kernel(lu.data(), perm.data(), n);
  if (sg == 0) {
    sign = 0, logabs = -std::numeric_limits<double>::infinity(), det = 0;
    return;
  }
  double s = sg, l = 0, d = sg;
  for (I64 i = 0; i < n; ++i) {
    const double v = lu[i * n + i];
    if (v < 0) s = -s;
    l += std::log(std::abs(v));
    d *= v;
  }
  sign = s, logabs = l, det = d;
}

// ---- general eigen decomposition, built-in (JAMA's orthes + hqr2, after EISPACK) ----------------

/// Real Schur based eigen decomposition of a [n, n] (row-major). wr/wi get the
/// eigenvalues; with v, v [n, n] gets the eigenvectors in LAPACK geev layout:
/// a real eigenvalue's vector in its column, a complex pair (wi[j] > 0,
/// wi[j+1] < 0) as v[:, j] +- i v[:, j+1].
void eig_builtin(const double* a, double* wr, double* wi, double* v, I64 n) {
  if (n == 0) return;
  // Scale by a power of two (exact) so the entries are O(1): the iteration squares them.
  double amax = 0;
  for (I64 i = 0; i < n * n; ++i) amax = std::max(amax, std::abs(a[i]));
  const int scale_exp = amax > 0 && std::isfinite(amax) ? -std::ilogb(amax) : 0;
  std::vector<double> hbuf(a, a + n * n), vbuf(static_cast<std::size_t>(n * n), 0.0), ort(static_cast<std::size_t>(n));
  if (scale_exp != 0)
    for (double& x : hbuf) x = std::ldexp(x, scale_exp);
  const auto H = [&](I64 i, I64 j) -> double& { return hbuf[static_cast<std::size_t>(i * n + j)]; };
  const auto V = [&](I64 i, I64 j) -> double& { return vbuf[static_cast<std::size_t>(i * n + j)]; };
  double*    d = wr;
  double*    e = wi;
  const I64  low = 0, high = n - 1;
  // orthes: Householder reduction to upper Hessenberg form, accumulated in V.
  for (I64 m = low + 1; m <= high - 1; ++m) {
    double scale = 0.0;
    for (I64 i = m; i <= high; ++i) scale += std::abs(H(i, m - 1));
    if (scale == 0.0) continue;
    double h = 0.0;
    for (I64 i = high; i >= m; --i) {
      ort[i] = H(i, m - 1) / scale;
      h += ort[i] * ort[i];
    }
    double g = std::sqrt(h);
    if (ort[m] > 0) g = -g;
    h      = h - ort[m] * g;
    ort[m] = ort[m] - g;
    for (I64 j = m; j < n; ++j) {
      double f = 0.0;
      for (I64 i = high; i >= m; --i) f += ort[i] * H(i, j);
      f /= h;
      for (I64 i = m; i <= high; ++i) H(i, j) -= f * ort[i];
    }
    for (I64 i = 0; i <= high; ++i) {
      double f = 0.0;
      for (I64 j = high; j >= m; --j) f += ort[j] * H(i, j);
      f /= h;
      for (I64 j = m; j <= high; ++j) H(i, j) -= f * ort[j];
    }
    ort[m]       = scale * ort[m];
    H(m, m - 1)  = scale * g;
  }
  for (I64 i = 0; i < n; ++i) V(i, i) = 1.0;
  for (I64 m = high - 1; m >= low + 1; --m) {
    if (H(m, m - 1) == 0.0) continue;
    for (I64 i = m + 1; i <= high; ++i) ort[i] = H(i, m - 1);
    for (I64 j = m; j <= high; ++j) {
      double g = 0.0;
      for (I64 i = m; i <= high; ++i) g += ort[i] * V(i, j);
      g = (g / ort[m]) / H(m, m - 1);  // double division avoids possible underflow
      for (I64 i = m; i <= high; ++i) V(i, j) += g * ort[i];
    }
  }
  // hqr2: shifted double QR iteration to real Schur form, then back substitution.
  const I64    nn  = n;
  I64          top = nn - 1;
  const double eps = std::ldexp(1.0, -52);
  double       exshift = 0.0, p = 0, q = 0, r = 0, s = 0, z = 0, t, w, x, y;
  double       norm = 0.0;
  for (I64 i = 0; i < nn; ++i)
    for (I64 j = std::max<I64>(i - 1, 0); j < nn; ++j) norm += std::abs(H(i, j));
  if (norm == 0.0) {  // zero matrix: zero eigenvalues, identity eigenvectors
    std::fill(d, d + n, 0.0);
    std::fill(e, e + n, 0.0);
    if (v)
      for (I64 i = 0; i < n * n; ++i) v[i] = i % (n + 1) == 0 ? 1.0 : 0.0;
    return;
  }
  int iter = 0, total = 0;
  while (top >= low) {
    I64 l = top;
    while (l > low) {
      s = std::abs(H(l - 1, l - 1)) + std::abs(H(l, l));
      if (s == 0.0) s = norm;
      if (std::abs(H(l, l - 1)) <= eps * s) break;  // "<=": exact zeros deflate (JAMA's "<" loops on them)
      --l;
    }
    if (l == top) {  // one root
      H(top, top) = H(top, top) + exshift;
      d[top]      = H(top, top);
      e[top]      = 0.0;
      --top;
      iter = 0;
    } else if (l == top - 1) {  // two roots
      const I64 m1 = top - 1;
      w            = H(top, m1) * H(m1, top);
      p            = (H(m1, m1) - H(top, top)) / 2.0;
      q            = p * p + w;
      z            = std::sqrt(std::abs(q));
      H(top, top)  = H(top, top) + exshift;
      H(m1, m1)    = H(m1, m1) + exshift;
      x            = H(top, top);
      if (q >= 0) {  // real pair
        z     = p >= 0 ? p + z : p - z;
        d[m1] = x + z;
        d[top] = d[m1];
        if (z != 0.0) d[top] = x - w / z;
        e[m1] = 0.0;
        e[top] = 0.0;
        x     = H(top, m1);
        s     = std::abs(x) + std::abs(z);
        p     = x / s;
        q     = z / s;
        r     = std::sqrt(p * p + q * q);
        p /= r;
        q /= r;
        for (I64 j = m1; j < nn; ++j) {
          z          = H(m1, j);
          H(m1, j)   = q * z + p * H(top, j);
          H(top, j)  = q * H(top, j) - p * z;
        }
        for (I64 i = 0; i <= top; ++i) {
          z          = H(i, m1);
          H(i, m1)   = q * z + p * H(i, top);
          H(i, top)  = q * H(i, top) - p * z;
        }
        for (I64 i = low; i <= high; ++i) {
          z          = V(i, m1);
          V(i, m1)   = q * z + p * V(i, top);
          V(i, top)  = q * V(i, top) - p * z;
        }
      } else {  // complex pair
        d[m1]  = x + p;
        d[top] = x + p;
        e[m1]  = z;
        e[top] = -z;
      }
      top -= 2;
      iter = 0;
    } else {  // no convergence yet
      x = H(top, top);
      y = 0.0;
      w = 0.0;
      if (l < top) {
        y = H(top - 1, top - 1);
        w = H(top, top - 1) * H(top - 1, top);
      }
      if (iter == 10) {  // Wilkinson's original ad hoc shift
        exshift += x;
        for (I64 i = low; i <= top; ++i) H(i, i) -= x;
        s = std::abs(H(top, top - 1)) + std::abs(H(top - 1, top - 2));
        x = y = 0.75 * s;
        w     = -0.4375 * s * s;
      }
      if (iter == 30) {  // MATLAB's ad hoc shift
        s = (y - x) / 2.0;
        s = s * s + w;
        if (s > 0) {
          s = std::sqrt(s);
          if (y < x) s = -s;
          s = x - w / ((y - x) / 2.0 + s);
          for (I64 i = low; i <= top; ++i) H(i, i) -= s;
          exshift += s;
          x = y = w = 0.964;
        }
      }
      ++iter;
      if (++total > 100 * std::max<I64>(nn, 10))
        throw std::runtime_error("eig: QR iteration did not converge (non-finite input?)");
      // two consecutive small subdiagonal elements
      I64 m = top - 2;
      while (m >= l) {
        z = H(m, m);
        r = x - z;
        s = y - z;
        p = (r * s - w) / H(m + 1, m) + H(m, m + 1);
        q = H(m + 1, m + 1) - z - r - s;
        r = H(m + 2, m + 1);
        s = std::abs(p) + std::abs(q) + std::abs(r);
        p /= s;
        q /= s;
        r /= s;
        if (m == l) break;
        if (std::abs(H(m, m - 1)) * (std::abs(q) + std::abs(r)) <
            eps * (std::abs(p) * (std::abs(H(m - 1, m - 1)) + std::abs(z) + std::abs(H(m + 1, m + 1)))))
          break;
        --m;
      }
      for (I64 i = m + 2; i <= top; ++i) {
        H(i, i - 2) = 0.0;
        if (i > m + 2) H(i, i - 3) = 0.0;
      }
      // double QR step on rows l..top and columns m..top
      for (I64 k = m; k <= top - 1; ++k) {
        const bool notlast = k != top - 1;
        if (k != m) {
          p = H(k, k - 1);
          q = H(k + 1, k - 1);
          r = notlast ? H(k + 2, k - 1) : 0.0;
          x = std::abs(p) + std::abs(q) + std::abs(r);
          if (x == 0.0) continue;
          p /= x;
          q /= x;
          r /= x;
        }
        s = std::sqrt(p * p + q * q + r * r);
        if (p < 0) s = -s;
        if (s == 0.0) continue;
        if (k != m)
          H(k, k - 1) = -s * x;
        else if (l != m)
          H(k, k - 1) = -H(k, k - 1);
        p = p + s;
        x = p / s;
        y = q / s;
        z = r / s;
        q = q / p;
        r = r / p;
        for (I64 j = k; j < nn; ++j) {
          p = H(k, j) + q * H(k + 1, j);
          if (notlast) {
            p = p + r * H(k + 2, j);
            H(k + 2, j) -= p * z;
          }
          H(k, j) -= p * x;
          H(k + 1, j) -= p * y;
        }
        for (I64 i = 0; i <= std::min(top, k + 3); ++i) {
          p = x * H(i, k) + y * H(i, k + 1);
          if (notlast) {
            p = p + z * H(i, k + 2);
            H(i, k + 2) -= p * r;
          }
          H(i, k) -= p;
          H(i, k + 1) -= p * q;
        }
        for (I64 i = low; i <= high; ++i) {
          p = x * V(i, k) + y * V(i, k + 1);
          if (notlast) {
            p = p + z * V(i, k + 2);
            V(i, k + 2) -= p * r;
          }
          V(i, k) -= p;
          V(i, k + 1) -= p * q;
        }
      }
    }
  }
  const auto unscale = [&] {
    if (scale_exp != 0)
      for (I64 i = 0; i < n; ++i) wr[i] = std::ldexp(wr[i], -scale_exp), wi[i] = std::ldexp(wi[i], -scale_exp);
  };
  if (!v) return unscale();
  // back substitution: eigenvectors of the quasi-triangular form
  const auto cdiv = [](double xr, double xi, double yr, double yi) {
    if (std::abs(yr) > std::abs(yi)) {
      const double rr = yi / yr, dd = yr + rr * yi;
      return std::complex<double>((xr + rr * xi) / dd, (xi - rr * xr) / dd);
    }
    const double rr = yr / yi, dd = yi + rr * yr;
    return std::complex<double>((rr * xr + xi) / dd, (rr * xi - xr) / dd);
  };
  for (I64 c = nn - 1; c >= 0; --c) {
    p = d[c];
    q = e[c];
    if (q == 0) {  // real vector
      I64 l   = c;
      H(c, c) = 1.0;
      for (I64 i = c - 1; i >= 0; --i) {
        w = H(i, i) - p;
        r = 0.0;
        for (I64 j = l; j <= c; ++j) r += H(i, j) * H(j, c);
        if (e[i] < 0.0) {
          z = w;
          s = r;
        } else {
          l = i;
          if (e[i] == 0.0) {
            H(i, c) = w != 0.0 ? -r / w : -r / (eps * norm);
          } else {  // real 2x2 equations
            x       = H(i, i + 1);
            y       = H(i + 1, i);
            q       = (d[i] - p) * (d[i] - p) + e[i] * e[i];
            t       = (x * s - z * r) / q;
            H(i, c) = t;
            H(i + 1, c) = std::abs(x) > std::abs(z) ? (-r - w * t) / x : (-s - y * t) / z;
          }
          t = std::abs(H(i, c));
          if ((eps * t) * t > 1)
            for (I64 j = i; j <= c; ++j) H(j, c) /= t;
        }
      }
    } else if (q < 0) {  // complex vector (columns c-1, c)
      I64 l = c - 1;
      if (std::abs(H(c, c - 1)) > std::abs(H(c - 1, c))) {
        H(c - 1, c - 1) = q / H(c, c - 1);
        H(c - 1, c)     = -(H(c, c) - p) / H(c, c - 1);
      } else {
        const auto cd   = cdiv(0.0, -H(c - 1, c), H(c - 1, c - 1) - p, q);
        H(c - 1, c - 1) = cd.real();
        H(c - 1, c)     = cd.imag();
      }
      H(c, c - 1) = 0.0;
      H(c, c)     = 1.0;
      for (I64 i = c - 2; i >= 0; --i) {
        double ra = 0.0, sa = 0.0;
        for (I64 j = l; j <= c; ++j) {
          ra += H(i, j) * H(j, c - 1);
          sa += H(i, j) * H(j, c);
        }
        w = H(i, i) - p;
        if (e[i] < 0.0) {
          z = w;
          r = ra;
          s = sa;
        } else {
          l = i;
          if (e[i] == 0) {
            const auto cd = cdiv(-ra, -sa, w, q);
            H(i, c - 1)   = cd.real();
            H(i, c)       = cd.imag();
          } else {  // complex 2x2 equations
            x         = H(i, i + 1);
            y         = H(i + 1, i);
            double vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
            double vi = (d[i] - p) * 2.0 * q;
            if (vr == 0.0 && vi == 0.0)
              vr = eps * norm * (std::abs(w) + std::abs(q) + std::abs(x) + std::abs(y) + std::abs(z));
            const auto cd = cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi);
            H(i, c - 1)   = cd.real();
            H(i, c)       = cd.imag();
            if (std::abs(x) > (std::abs(z) + std::abs(q))) {
              H(i + 1, c - 1) = (-ra - w * H(i, c - 1) + q * H(i, c)) / x;
              H(i + 1, c)     = (-sa - w * H(i, c) - q * H(i, c - 1)) / x;
            } else {
              const auto cd2  = cdiv(-r - y * H(i, c - 1), -s - y * H(i, c), z, q);
              H(i + 1, c - 1) = cd2.real();
              H(i + 1, c)     = cd2.imag();
            }
          }
          t = std::max(std::abs(H(i, c - 1)), std::abs(H(i, c)));
          if ((eps * t) * t > 1)
            for (I64 j = i; j <= c; ++j) {
              H(j, c - 1) /= t;
              H(j, c) /= t;
            }
        }
      }
    }
  }
  // back transformation to the eigenvectors of the original matrix
  for (I64 j = nn - 1; j >= low; --j)
    for (I64 i = low; i <= high; ++i) {
      z = 0.0;
      for (I64 k = low; k <= std::min(j, high); ++k) z += V(i, k) * H(k, j);
      V(i, j) = z;
    }
  std::copy(vbuf.begin(), vbuf.end(), v);
  unscale();
}

/// Turns geev-layout results (wr, wi, real v [n, n]) into sorted complex
/// eigenpairs with the documented normalisation; v may be null.
void eig_finish(I64 n, const double* wr, const double* wi, const double* v, double* er, double* ei, double* vre,
                double* vim) {
  using C = std::complex<double>;
  std::vector<C> vals(static_cast<std::size_t>(n)), vecs;
  for (I64 j = 0; j < n; ++j) vals[j] = C(wr[j], wi[j]);
  if (v) {
    vecs.assign(static_cast<std::size_t>(n * n), C());  // vecs[j * n + i]: column j
    for (I64 j = 0; j < n; ++j) {
      if (wi[j] != 0.0 && j + 1 < n && wi[j] > 0) {
        for (I64 i = 0; i < n; ++i) {
          vecs[j * n + i]       = C(v[i * n + j], v[i * n + j + 1]);
          vecs[(j + 1) * n + i] = C(v[i * n + j], -v[i * n + j + 1]);
        }
        ++j;
      } else {
        for (I64 i = 0; i < n; ++i) vecs[j * n + i] = C(v[i * n + j], 0.0);
      }
    }
    for (I64 j = 0; j < n; ++j) {
      C*     col  = vecs.data() + j * n;
      double nrm2 = 0, best = -1;
      C      piv  = 1;
      for (I64 i = 0; i < n; ++i) {
        const double m2 = std::norm(col[i]);
        nrm2 += m2;
        if (m2 > best) best = m2, piv = col[i];
      }
      if (nrm2 == 0) continue;
      // unit norm, largest-modulus entry real and positive
      const C f = std::conj(piv) / (std::abs(piv) * std::sqrt(nrm2));
      for (I64 i = 0; i < n; ++i) col[i] *= f;
    }
  }
  std::vector<I64> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), I64{0});
  std::stable_sort(order.begin(), order.end(), [&](I64 x, I64 y) {
    const double ax = std::abs(vals[x]), ay = std::abs(vals[y]);
    if (ax != ay) return ax < ay;
    if (vals[x].real() != vals[y].real()) return vals[x].real() < vals[y].real();
    return vals[x].imag() < vals[y].imag();
  });
  for (I64 j = 0; j < n; ++j) {
    const I64 o = order[j];
    er[j]       = vals[o].real();
    ei[j]       = vals[o].imag();
    if (v)
      for (I64 i = 0; i < n; ++i) {
        vre[i * n + j] = vecs[o * n + i].real();
        vim[i * n + j] = vecs[o * n + i].imag();
      }
  }
}

// ---- LAPACK / BLAS paths ---------------------------------------------------------------------

#if defined(DATAMUNGE_HAVE_LAPACK)
/// SVD via gesdd (gesvd if that fails). A row-major [m, n] is At column-major
/// [n, m] = U' S VT', so A's U is VT'^T (VT' as stored is U row-major) and V = U'.
bool svd_lapack(const double* a, double* s, double* u, double* v, I64 m, I64 n, bool full) {
  const I64           k = std::min(m, n), uc = full ? m : k, vc = full ? n : k;
  std::vector<double> at(a, a + m * n), up(u ? static_cast<std::size_t>(n * vc) : 1), dummy(1);
  const char          job = !u ? 'N' : full ? 'A' : 'S';
  double*             pu  = u ? up.data() : dummy.data();
  double*             pvt = u ? u : dummy.data();
  const I64           ldvt = u ? uc : 1;
  if (lapack::gesdd(job, n, m, at.data(), n, s, pu, n, pvt, ldvt) != 0) {
    std::copy(a, a + m * n, at.begin());
    if (lapack::gesvd(job, n, m, at.data(), n, s, pu, n, pvt, ldvt) != 0) return false;
  }
  if (v)
    for (I64 i = 0; i < n; ++i)
      for (I64 j = 0; j < vc; ++j) v[i * vc + j] = up[static_cast<std::size_t>(i + j * n)];
  return true;
}

/// syevd on the lower triangle (row-major lower == column-major upper).
bool eigh_lapack(const double* a, double* e, double* v, I64 n) {
  std::vector<double> w(a, a + n * n);
  if (lapack::syevd(v ? 'V' : 'N', 'U', n, w.data(), e) != 0) return false;
  if (v)
    for (I64 i = 0; i < n; ++i)
      for (I64 j = 0; j < n; ++j) v[i * n + j] = w[static_cast<std::size_t>(i + j * n)];
  return true;
}

/// geqrf + orgqr on a column-major copy.
bool qr_lapack(const double* a, double* q, double* r, I64 m, I64 n, bool full) {
  const I64           k = std::min(m, n), qc = full ? m : k, rr = full ? m : k;
  std::vector<double> c(static_cast<std::size_t>(m * std::max(n, qc)), 0.0), tau(static_cast<std::size_t>(std::max<I64>(k, 1)));
  for (I64 i = 0; i < m; ++i)
    for (I64 j = 0; j < n; ++j) c[static_cast<std::size_t>(i + j * m)] = a[i * n + j];
  if (lapack::geqrf(m, n, c.data(), m, tau.data()) != 0) return false;
  for (I64 i = 0; i < rr; ++i)
    for (I64 p = 0; p < n; ++p) r[i * n + p] = (i < m && p >= i) ? c[static_cast<std::size_t>(i + p * m)] : 0.0;
  if (lapack::orgqr(m, qc, k, c.data(), m, tau.data()) != 0) return false;
  for (I64 i = 0; i < m; ++i)
    for (I64 j = 0; j < qc; ++j) q[i * qc + j] = c[static_cast<std::size_t>(i + j * m)];
  return true;
}

/// potrf: column-major upper U (A = U^T U) is row-major lower L.
bool cholesky_lapack(const double* a, double* l, I64 n) {
  std::copy(a, a + n * n, l);
  if (lapack::potrf('U', n, l) != 0) return false;
  for (I64 i = 0; i < n; ++i) std::fill(l + i * n + i + 1, l + (i + 1) * n, 0.0);
  return true;
}

/// getrf of the row-major buffer (= A^T column-major), then getrs with the
/// transposed operation; false if A is singular.
bool lu_solve_lapack(const double* a, const double* b, double* x, I64 n, I64 k, bool adjoint) {
  std::vector<double>       lu(a, a + n * n), bc(static_cast<std::size_t>(n * k));
  std::vector<lapack::LI>   ipiv(static_cast<std::size_t>(n));
  if (lapack::getrf(n, lu.data(), ipiv.data()) != 0) return false;
  for (I64 i = 0; i < n; ++i)
    for (I64 c = 0; c < k; ++c) bc[static_cast<std::size_t>(i + c * n)] = b[i * k + c];
  if (lapack::getrs(adjoint ? 'N' : 'T', n, k, lu.data(), ipiv.data(), bc.data()) != 0) return false;
  for (I64 i = 0; i < n; ++i)
    for (I64 c = 0; c < k; ++c) x[i * k + c] = bc[static_cast<std::size_t>(i + c * n)];
  return true;
}

void lu_det_lapack(const double* a, I64 n, double& sign, double& logabs, double& det) {
  std::vector<double>     lu(a, a + n * n);
  std::vector<lapack::LI> ipiv(static_cast<std::size_t>(n));
  if (lapack::getrf(n, lu.data(), ipiv.data()) != 0) {
    sign = 0, logabs = -std::numeric_limits<double>::infinity(), det = 0;
    return;
  }
  double s = 1, l = 0, d = 1;
  for (I64 i = 0; i < n; ++i) {
    const double v = lu[static_cast<std::size_t>(i * n + i)];
    if (ipiv[i] != i + 1) s = -s, d = -d;
    if (v < 0) s = -s;
    l += std::log(std::abs(v));
    d *= v;
  }
  sign = s, logabs = l, det = d;
}

/// geev on a column-major copy; v (row-major) in geev's real layout.
bool eig_lapack(const double* a, double* wr, double* wi, double* v, I64 n) {
  std::vector<double> c(static_cast<std::size_t>(n * n)), vr(v ? static_cast<std::size_t>(n * n) : 1);
  for (I64 i = 0; i < n; ++i)
    for (I64 j = 0; j < n; ++j) c[static_cast<std::size_t>(i + j * n)] = a[i * n + j];
  if (lapack::geev(v != nullptr, n, c.data(), wr, wi, vr.data()) != 0) return false;
  if (v)
    for (I64 i = 0; i < n; ++i)
      for (I64 j = 0; j < n; ++j) v[i * n + j] = vr[static_cast<std::size_t>(i + j * n)];
  return true;
}
#endif

// ---- per-matrix dispatch ------------------------------------------------------------------------

void svd_any(const double* a, double* s, double* u, double* v, I64 m, I64 n, bool full) {
  bool done = false;
#if defined(DATAMUNGE_HAVE_LAPACK)
  if (use_lapack(u ? Op::Svd : Op::SvdValues, std::min(m, n))) done = svd_lapack(a, s, u, v, m, n, full);
#endif
  if (!done) svd_kernel(a, s, u, v, m, n, full);
  if (u) normalize_svd_signs(u, v, m, n, full ? m : std::min(m, n), full ? n : std::min(m, n));
}

void eigh_any(const double* a, double* e, double* v, I64 n) {
  bool done = false;
#if defined(DATAMUNGE_HAVE_LAPACK)
  if (use_lapack(v ? Op::Eigh : Op::EighValues, n)) done = eigh_lapack(a, e, v, n);
#endif
  if (!done) eigh_kernel(a, e, v, n);
  if (v) normalize_eigh_signs(v, n);
}

void qr_any(const double* a, double* q, double* r, I64 m, I64 n, bool full) {
#if defined(DATAMUNGE_HAVE_LAPACK)
  if (use_lapack(Op::Qr, std::min(m, n)) && qr_lapack(a, q, r, m, n, full)) return;
#endif
  qr_kernel(a, q, r, m, n, full);
}

bool cholesky_any(const double* a, double* l, I64 n) {
#if defined(DATAMUNGE_HAVE_LAPACK)
  if (use_lapack(Op::Cholesky, n)) return cholesky_lapack(a, l, n);
#endif
  return cholesky_kernel(a, l, n);
}

bool lu_solve_any(const double* a, const double* b, double* x, I64 n, I64 k, bool adjoint) {
#if defined(DATAMUNGE_HAVE_LAPACK)
  if (use_lapack(Op::Lu, n)) return lu_solve_lapack(a, b, x, n, k, adjoint);
#endif
  return lu_solve_builtin(a, b, x, n, k, adjoint);
}

void lu_det_any(const double* a, I64 n, double& sign, double& logabs, double& det) {
#if defined(DATAMUNGE_HAVE_LAPACK)
  if (use_lapack(Op::Lu, n)) return lu_det_lapack(a, n, sign, logabs, det);
#endif
  lu_det_builtin(a, n, sign, logabs, det);
}

void triangular_solve_any(const double* a, const double* b, double* x, I64 n, I64 k, bool lower, bool adjoint) {
#if defined(DATAMUNGE_HAVE_CBLAS)
  if (use_lapack(Op::TriSolve, n) && n > 0 && k > 0) {
    std::copy(b, b + n * k, x);
    cblas_dtrsm(CblasRowMajor, CblasLeft, lower ? CblasLower : CblasUpper, adjoint ? CblasTrans : CblasNoTrans,
                CblasNonUnit, static_cast<int>(n), static_cast<int>(k), 1.0, a, static_cast<int>(n), x,
                static_cast<int>(k));
    return;
  }
#endif
  triangular_solve_kernel(a, b, x, n, k, lower, adjoint);
}

void eig_any(const double* a, double* er, double* ei, double* vre, double* vim, I64 n) {
  std::vector<double> wr(static_cast<std::size_t>(n)), wi(static_cast<std::size_t>(n)),
      v(vre ? static_cast<std::size_t>(n * n) : 0);
  bool done = false;
#if defined(DATAMUNGE_HAVE_LAPACK)
  if (use_lapack(Op::Eig, n)) done = eig_lapack(a, wr.data(), wi.data(), vre ? v.data() : nullptr, n);
#endif
  if (!done) eig_builtin(a, wr.data(), wi.data(), vre ? v.data() : nullptr, n);
  eig_finish(n, wr.data(), wi.data(), vre ? v.data() : nullptr, er, ei, vre, vim);
}

/// Factors a broadcast singleton matrix once; scratch belongs to this call.
class SharedLu {
 public:
  SharedLu(const double* a, I64 n, bool adjoint) : n_(n), adjoint_(adjoint), lu_(static_cast<std::size_t>(n * n)) {
#if defined(DATAMUNGE_HAVE_LAPACK)
    lapack_ = use_lapack(Op::Lu, n);
    if (lapack_) {
      std::copy(a, a + n * n, lu_.begin());
      ipiv_.resize(static_cast<std::size_t>(n));
      valid_ = lapack::getrf(n, lu_.data(), ipiv_.data()) == 0;
      return;
    }
#endif
    perm_.resize(static_cast<std::size_t>(n));
    for (I64 i = 0; i < n; ++i)
      for (I64 j = 0; j < n; ++j) lu_[i * n + j] = adjoint ? a[j * n + i] : a[i * n + j];
    valid_ = lu_kernel(lu_.data(), perm_.data(), n) != 0;
  }
  bool valid() const { return valid_; }
  bool solve(const double* b, double* x, I64 k, std::vector<double>& scratch) const {
#if defined(DATAMUNGE_HAVE_LAPACK)
    if (lapack_) {
      scratch.resize(static_cast<std::size_t>(n_ * k));
      for (I64 i = 0; i < n_; ++i)
        for (I64 c = 0; c < k; ++c) scratch[i + c * n_] = b[i * k + c];
      if (lapack::getrs(adjoint_ ? 'N' : 'T', n_, k, lu_.data(), ipiv_.data(), scratch.data()) != 0) return false;
      for (I64 i = 0; i < n_; ++i)
        for (I64 c = 0; c < k; ++c) x[i * k + c] = scratch[i + c * n_];
      return true;
    }
#else
    (void)scratch;
#endif
    solve_lu_builtin_factors(lu_.data(), perm_.data(), b, x, n_, k);
    return true;
  }
 private:
  I64 n_;
  bool adjoint_, valid_{false};
  std::vector<double> lu_;
  std::vector<I64> perm_;
#if defined(DATAMUNGE_HAVE_LAPACK)
  bool lapack_{false};
  std::vector<lapack::LI> ipiv_;
#endif
};

// ---- forward drivers -------------------------------------------------------------------------

struct Broadcast2 {
  Tensor a, b;  // packed float64, broadcast to a common batch
  Shape  batch;
};

Broadcast2 broadcast_pair(const Tensor& a, const Tensor& b) {
  Broadcast2 r;
  r.batch = broadcast_shapes(batch_shape(a.shape()), batch_shape(b.shape()));
  NoGradGuard guard;
  // Equal batch shapes already have the right layout. The guarded packing
  // path only reads its inputs, so no detached TensorImpl copies are needed.
  r.a = packed_f64(batch_shape(a.shape()) == r.batch ? a : broadcast_to(a, with_matrix(r.batch, a.size(-2), a.size(-1))));
  r.b = packed_f64(batch_shape(b.shape()) == r.batch ? b : broadcast_to(b, with_matrix(r.batch, b.size(-2), b.size(-1))));
  return r;
}

Tensor triangular_solve_raw(const Tensor& a, const Tensor& b, bool lower, bool adjoint, DType dt) {
  Broadcast2 bc = broadcast_pair(a, b);
  const I64  n = a.size(-1), k = b.size(-1), nb = shape_numel(bc.batch);
  Tensor     out = empty(with_matrix(bc.batch, n, k), DType::Float64);
  const double* pa = bc.a.data_ptr<double>();
  const double* pb = bc.b.data_ptr<double>();
  double*       po = out.data_ptr<double>();
  for_batch(nb, n * n * k, [&](I64 p) { triangular_solve_any(pa + p * n * n, pb + p * n * k, po + p * n * k, n, k, lower, adjoint); });
  return from_f64(out, dt);
}

Tensor solve_raw(const Tensor& a, const Tensor& b, bool adjoint, DType dt) {
  const Shape batch = broadcast_shapes(batch_shape(a.shape()), batch_shape(b.shape()));
  const I64 count = shape_numel(batch);
  if (count > 1 && shape_numel(batch_shape(a.shape())) == 1) {
    // All RHS batches use the same coefficient matrix. Keep its packed storage
    // once and reuse its factors; each worker owns its RHS conversion scratch.
    NoGradGuard guard;
    const I64 n = a.size(-1), k = b.size(-1);
    const Tensor pa = packed_f64(a);
    const Tensor pb = packed_f64(batch_shape(b.shape()) == batch ? b : broadcast_to(b, with_matrix(batch, n, k)));
    Tensor out = empty(with_matrix(batch, n, k), DType::Float64);
    const SharedLu factors(pa.data_ptr<double>(), n, adjoint);
    if (!factors.valid()) throw std::invalid_argument("solve: input matrix is not invertible");
    const double* rhs = pb.data_ptr<double>();
    double* dst = out.data_ptr<double>();
    const I64 grain = std::max<I64>(1, I64{1 << 16} / std::max<I64>(n * n * (n + k), 1));
    parallel_for(0, count, grain, [&](I64 lo, I64 hi) {
      std::vector<double> scratch;
      for (I64 p = lo; p < hi; ++p)
        if (!factors.solve(rhs + p * n * k, dst + p * n * k, k, scratch))
          throw std::invalid_argument("solve: input matrix is not invertible");
    });
    return from_f64(out, dt);
  }
  Broadcast2 bc = broadcast_pair(a, b);
  const I64  n = a.size(-1), k = b.size(-1), nb = shape_numel(bc.batch);
  Tensor     out = empty(with_matrix(bc.batch, n, k), DType::Float64);
  const double* pa = bc.a.data_ptr<double>();
  const double* pb = bc.b.data_ptr<double>();
  double*       po = out.data_ptr<double>();
  for_batch(nb, n * n * (n + k), [&](I64 p) {
    if (!lu_solve_any(pa + p * n * n, pb + p * n * k, po + p * n * k, n, k, adjoint))
      throw std::invalid_argument("solve: input matrix is not invertible");
  });
  return from_f64(out, dt);
}

/// sign and log|det| (and det) of each matrix, as float64 [batch].
void lu_det_raw(const Tensor& a, Tensor* sign, Tensor* logabs, Tensor* det_out) {
  const Tensor  pa = packed_f64(a);
  const I64     n = a.size(-1);
  const Shape   batch = batch_shape(a.shape());
  const I64     nb    = shape_numel(batch);
  Tensor        ts = empty(batch, DType::Float64), tl = empty(batch, DType::Float64), td = empty(batch, DType::Float64);
  const double* src = pa.data_ptr<double>();
  double *      ps = ts.data_ptr<double>(), *pl = tl.data_ptr<double>(), *pd = td.data_ptr<double>();
  for_batch(nb, n * n * n, [&](I64 p) { lu_det_any(src + p * n * n, n, ps[p], pl[p], pd[p]); });
  if (sign) *sign = ts;
  if (logabs) *logabs = tl;
  if (det_out) *det_out = td;
}

// ---- complex helpers for the eig gradient (differentiable nn ops) ------------------------------

ComplexTensor cmatmul(const ComplexTensor& a, const ComplexTensor& b) {
  return {sub(matmul(a.real, b.real), matmul(a.imag, b.imag)), add(matmul(a.real, b.imag), matmul(a.imag, b.real))};
}

ComplexTensor cadjoint(const ComplexTensor& a) { return {mT(a.real), neg(mT(a.imag))}; }

ComplexTensor cdiag(const ComplexTensor& d) { return {matrix_diag(d.real), matrix_diag(d.imag)}; }

/// Solves A X = B for complex A via the real embedding [[Re, -Im], [Im, Re]].
ComplexTensor csolve(const ComplexTensor& a, const ComplexTensor& b) {
  const I64    n   = a.size(-1);
  const Tensor m   = concat({concat({a.real, neg(a.imag)}, -1), concat({a.imag, a.real}, -1)}, -2);
  const Tensor x   = solve(m, concat({b.real, b.imag}, -2));
  return {slice_axis(x, -2, 0, n), slice_axis(x, -2, n, 2 * n)};
}

}  // namespace

// ---- structure -------------------------------------------------------------------------------

Tensor matrix_diag(const Tensor& diagonal) {
  if (diagonal.dim() < 1) throw std::invalid_argument("matrix_diag: expected rank >= 1");
  const I64 n = diagonal.size(-1);
  return diag_embed(diagonal, n, n);
}

Tensor matrix_diag_part(const Tensor& a) { return diag_part_impl(a); }

Tensor set_diag(const Tensor& a, const Tensor& diagonal) {
  require_matrix(a, "set_diag");
  const Tensor ac = as_float(a);
  const I64    m = ac.size(-2), n = ac.size(-1);
  const Tensor d = detail::as_dtype(diagonal, ac.dtype());
  const Tensor emb  = diag_embed(d, m, n);
  const Tensor mask = detail::cast_raw(eye_mask(m, n, DType::Float64), DType::Bool);
  return where(mask, emb, ac);
}

Tensor band_part(const Tensor& a, std::int64_t num_lower, std::int64_t num_upper) {
  require_matrix(a, "band_part");
  const I64 m = a.size(-2), n = a.size(-1);
  const Tensor mask = matrix_mask(m, n, DType::Float64, [&](I64 i, I64 j) {
    return ((num_lower < 0 || i - j <= num_lower) && (num_upper < 0 || j - i <= num_upper)) ? 1.0 : 0.0;
  });
  return where(detail::cast_raw(mask, DType::Bool), a, scalar_like(0.0, a));
}

Tensor trace(const Tensor& a) { return sum(matrix_diag_part(a), {-1}); }

Tensor matrix_transpose(const Tensor& a) {
  require_matrix(a, "matrix_transpose");
  return mT(a);
}

// ---- decompositions ----------------------------------------------------------------------------

Tensor cholesky(const Tensor& a) {
  require_square(a, "cholesky");
  const Tensor ac = as_float(a);
  const DType  dt = ac.dtype();
  const I64    n  = ac.size(-1);
  Tensor       out;
  {
    NoGradGuard   guard;
    const Tensor  pa  = packed_f64(ac);
    Tensor        l   = empty(ac.shape(), DType::Float64);
    const double* src = pa.data_ptr<double>();
    double*       dst = l.data_ptr<double>();
    for_batch(shape_numel(batch_shape(ac.shape())), n * n * n, [&](I64 p) {
      if (!cholesky_any(src + p * n * n, dst + p * n * n, n))
        throw std::invalid_argument("cholesky: matrix is not positive definite (Cholesky decomposition was not successful)");
    });
    out = from_f64(l, dt);
  }
  record(
      "Cholesky", {ac}, {out},
      [n, dt](const Grads& g, const Grads& o) {
        const Tensor& L     = o[0];
        const Tensor  linv  = triangular_solve(L, batched_eye(batch_shape(L.shape()), n, dt), true, false);
        // phi(L^T G): lower triangle with a halved diagonal.
        const Tensor  phi   = matrix_mask(n, n, dt, [](I64 i, I64 j) { return i > j ? 1.0 : i == j ? 0.5 : 0.0; });
        const Tensor  mid   = mul(matmul(L, g[0], true, false), phi);
        const Tensor  grad  = matmul(matmul(linv, mid, true, false), linv);
        return Grads{mul(add(grad, mT(grad)), 0.5)};
      },
      true);
  return out;
}

QrResult qr(const Tensor& a, bool full_matrices) {
  require_matrix(a, "qr");
  const Tensor ac = as_float(a);
  const DType  dt = ac.dtype();
  const I64    m = ac.size(-2), n = ac.size(-1), k = std::min(m, n);
  const I64    qc = full_matrices ? m : k, rr = full_matrices ? m : k;
  const Shape  batch = batch_shape(ac.shape());
  QrResult     res;
  {
    NoGradGuard   guard;
    const Tensor  pa = packed_f64(ac);
    Tensor        q = empty(with_matrix(batch, m, qc), DType::Float64), r = empty(with_matrix(batch, rr, n), DType::Float64);
    const double* src = pa.data_ptr<double>();
    double *      pq = q.data_ptr<double>(), *pr = r.data_ptr<double>();
    for_batch(shape_numel(batch), m * n * k, [&](I64 p) {
      qr_any(src + p * m * n, pq + p * m * qc, pr + p * rr * n, m, n, full_matrices);
    });
    res.q = from_f64(q, dt);
    res.r = from_f64(r, dt);
  }
  SavedTensor sa(ac);
  record(
      "Qr", {ac}, {res.q, res.r},
      [sa, m, n, full_matrices](const Grads& g, const Grads& o) {
        if (full_matrices && m != n)
          throw std::invalid_argument("qr: gradient not implemented for full_matrices with non-square input (as TF)");
        // TF _QrGradSquareAndDeepMatrices
        const auto deep = [](const Tensor& q, const Tensor& r, const Tensor& dq, const Tensor& dr) {
          const auto tri_solve = [](const Tensor& x, const Tensor& rr) {  // x R^{-T}
            return mT(triangular_solve(rr, mT(x), false, false));
          };
          const Tensor qdq  = matmul(q, dq, true, false);
          const Tensor qdq_ = sub(qdq, mT(qdq));
          const Tensor rdr  = matmul(r, dr, false, true);
          const Tensor rdr_ = sub(rdr, mT(rdr));
          const Tensor tril = band_part(add(qdq_, rdr_), -1, 0);
          const Tensor ga   = matmul(q, add(dr, tri_solve(tril, r)));
          const Tensor gb   = tri_solve(sub(dq, matmul(q, qdq)), r);
          return add(ga, gb);
        };
        const Tensor &q = o[0], &r = o[1];
        if (m >= n) return Grads{deep(q, r, g[0], g[1])};
        // Wide: A = [X | Y], R = [U | V]; reduce to the square case.
        const Tensor& A  = sa.get();
        const Tensor  y  = slice_axis(A, -1, m, n);
        const Tensor  u  = slice_axis(r, -1, 0, m);
        const Tensor  dv = slice_axis(g[1], -1, m, n);
        const Tensor  du = slice_axis(g[1], -1, 0, m);
        const Tensor  dq = add(g[0], matmul(y, dv, false, true));
        const Tensor  dx = deep(q, u, dq, du);
        const Tensor  dy = matmul(q, dv);
        return Grads{concat({dx, dy}, -1)};
      },
      true);
  return res;
}

SvdResult svd(const Tensor& a, bool compute_uv, bool full_matrices) {
  require_matrix(a, "svd");
  const Tensor ac = as_float(a);
  const DType  dt = ac.dtype();
  const I64    m = ac.size(-2), n = ac.size(-1), k = std::min(m, n);
  const I64    uc = full_matrices ? m : k, vc = full_matrices ? n : k;
  const Shape  batch = batch_shape(ac.shape());
  SvdResult    res;
  {
    NoGradGuard  guard;
    const Tensor pa = packed_f64(ac);
    Shape        ss = batch;
    ss.push_back(k);
    Tensor s = empty(ss, DType::Float64), u, v;
    if (compute_uv) u = empty(with_matrix(batch, m, uc), DType::Float64), v = empty(with_matrix(batch, n, vc), DType::Float64);
    const double* src = pa.data_ptr<double>();
    double*       ps  = s.data_ptr<double>();
    double*       pu  = compute_uv ? u.data_ptr<double>() : nullptr;
    double*       pv  = compute_uv ? v.data_ptr<double>() : nullptr;
    for_batch(shape_numel(batch), 10 * m * n * k, [&](I64 p) {
      svd_any(src + p * m * n, ps + p * k, pu ? pu + p * m * uc : nullptr, pv ? pv + p * n * vc : nullptr, m, n,
              full_matrices);
    });
    res.s = from_f64(s, dt);
    if (compute_uv) res.u = from_f64(u, dt), res.v = from_f64(v, dt);
  }
  SavedTensor sa(ac);
  if (!compute_uv) {
    record("Svd", {ac}, {res.s}, [sa](const Grads& g, const Grads&) {
      const SvdResult full = svd(sa.get(), true, false);
      return Grads{matmul(full.u, matmul(matrix_diag(g[0]), full.v, false, true))};
    });
    return res;
  }
  record(
      "Svd", {ac}, {res.s, res.u, res.v},
      [m0 = m, n0 = n, full_matrices](const Grads& g, const Grads& o) {
        I64    m = m0, n = n0;
        Tensor s = o[0], u = o[1], v = o[2], gs = g[0], gu = g[1], gv = g[2];
        const bool use_adjoint = m > n;
        if (use_adjoint) {
          std::swap(m, n);
          std::swap(u, v);
          std::swap(gu, gv);
        }
        if (full_matrices && std::abs(m - n) > 1)
          throw std::invalid_argument("svd: gradient not implemented for full_matrices with |m - n| > 1 (as TF)");
        const Tensor s_mat     = matrix_diag(s);
        const Tensor f         = pairwise_reciprocal(square(s));
        const Tensor s_inv_mat = matrix_diag(safe_reciprocal(s));
        const Tensor v1        = slice_axis(v, -1, 0, m);
        const Tensor gv1       = slice_axis(gv, -1, 0, m);
        const Tensor u_gu      = matmul(u, gu, true, false);
        const Tensor v_gv      = matmul(v1, gv1, true, false);
        const Tensor f_u       = mul(f, u_gu);
        const Tensor f_v       = mul(f, v_gv);
        const Tensor term1_nouv =
            add(add(matrix_diag(gs), matmul(add(f_u, mT(f_u)), s_mat)), matmul(s_mat, add(f_v, mT(f_v))));
        Tensor grad = matmul(u, matmul(term1_nouv, v1, false, true));
        if (m != n) {
          const Tensor gv1t       = mT(gv1);
          const Tensor gv1t_v1    = matmul(gv1t, v1);
          Tensor       term2_nous = sub(gv1t, matmul(gv1t_v1, v1, false, true));
          if (full_matrices) {
            const Tensor v2     = slice_axis(v, -1, m, n);
            const Tensor gv2    = slice_axis(gv, -1, m, n);
            const Tensor v1t_gv2 = matmul(v1, gv2, true, false);
            term2_nous          = sub(term2_nous, matmul(v1t_gv2, v2, false, true));
          }
          grad = add(grad, matmul(matmul(u, s_inv_mat), term2_nous));
        }
        return Grads{use_adjoint ? mT(grad) : grad};
      },
      true);
  return res;
}

EighResult eigh(const Tensor& a, bool compute_v) {
  require_square(a, "eigh");
  const Tensor ac = as_float(a);
  const DType  dt = ac.dtype();
  const I64    n  = ac.size(-1);
  const Shape  batch = batch_shape(ac.shape());
  EighResult   res;
  {
    NoGradGuard  guard;
    const Tensor pa = packed_f64(ac);
    Shape        es = batch;
    es.push_back(n);
    Tensor        e = empty(es, DType::Float64), v;
    if (compute_v) v = empty(ac.shape(), DType::Float64);
    const double* src = pa.data_ptr<double>();
    double *      pe = e.data_ptr<double>(), *pv = compute_v ? v.data_ptr<double>() : nullptr;
    for_batch(shape_numel(batch), 10 * n * n * n,
              [&](I64 p) { eigh_any(src + p * n * n, pe + p * n, pv ? pv + p * n * n : nullptr, n); });
    res.e = from_f64(e, dt);
    if (compute_v) res.v = from_f64(v, dt);
  }
  // The forward reads only the lower triangle: symmetrise the gradient and keep its lower part.
  const auto finish = [](const Tensor& grad) {
    const Tensor sym = band_part(add(grad, mT(grad)), -1, 0);
    return set_diag(sym, mul(matrix_diag_part(sym), 0.5));
  };
  if (!compute_v) {
    SavedTensor sa(ac);
    record("SelfAdjointEigV2", {ac}, {res.e}, [sa, finish](const Grads& g, const Grads&) {
      const Tensor v = eigh(sa.get(), true).v;
      return Grads{finish(matmul(v, matmul(matrix_diag(g[0]), v, false, true)))};
    });
    return res;
  }
  record(
      "SelfAdjointEigV2", {ac}, {res.e, res.v},
      [finish](const Grads& g, const Grads& o) {
        const Tensor &e = o[0], &v = o[1];
        const Tensor  f = pairwise_reciprocal(e);
        const Tensor  inner = add(matrix_diag(g[0]), mul(f, matmul(v, g[1], true, false)));
        return Grads{finish(matmul(v, matmul(inner, v, false, true)))};
      },
      true);
  return res;
}

EighResult self_adjoint_eig(const Tensor& a, bool compute_v) { return eigh(a, compute_v); }

Tensor eigvalsh(const Tensor& a) { return eigh(a, false).e; }

// ---- solves and determinants ---------------------------------------------------------------------

Tensor triangular_solve(const Tensor& a, const Tensor& b, bool lower, bool adjoint) {
  require_square(a, "triangular_solve");
  require_matrix(b, "triangular_solve");
  const DType dt = promote_types(float_dtype(a), float_dtype(b));
  const Tensor ac = detail::as_dtype(a, dt), bc = detail::as_dtype(b, dt);
  if (bc.size(-2) != ac.size(-1))
    throw std::invalid_argument("triangular_solve: shapes " + shape_str(ac.shape()) + " and " + shape_str(bc.shape()) +
                                " do not fit");
  Tensor out;
  {
    NoGradGuard guard;
    out = triangular_solve_raw(ac, bc, lower, adjoint, dt);
  }
  SavedTensor sa(ac);
  const Shape as = ac.shape(), bs = bc.shape();
  record(
      "MatrixTriangularSolve", {ac, bc}, {out},
      [sa, as, bs, lower, adjoint](const Grads& g, const Grads& o) {
        const Tensor& A  = sa.get();
        const Tensor& c  = o[0];
        const Tensor  gb = triangular_solve(A, g[0], lower, !adjoint);
        Tensor        ga = adjoint ? neg(matmul(c, gb, false, true)) : neg(matmul(gb, c, false, true));
        ga               = lower ? band_part(ga, -1, 0) : band_part(ga, 0, -1);
        return Grads{sum_to(ga, as), sum_to(gb, bs)};
      },
      true);
  return out;
}

Tensor solve(const Tensor& a, const Tensor& b, bool adjoint) {
  require_square(a, "solve");
  require_matrix(b, "solve");
  const DType  dt = promote_types(float_dtype(a), float_dtype(b));
  const Tensor ac = detail::as_dtype(a, dt), bc = detail::as_dtype(b, dt);
  if (bc.size(-2) != ac.size(-1))
    throw std::invalid_argument("solve: shapes " + shape_str(ac.shape()) + " and " + shape_str(bc.shape()) + " do not fit");
  Tensor out;
  {
    NoGradGuard guard;
    out = solve_raw(ac, bc, adjoint, dt);
  }
  SavedTensor sa(ac);
  const Shape as = ac.shape(), bs = bc.shape();
  record(
      "MatrixSolve", {ac, bc}, {out},
      [sa, as, bs, adjoint](const Grads& g, const Grads& o) {
        const Tensor& A  = sa.get();
        const Tensor& c  = o[0];
        const Tensor  gb = solve(A, g[0], !adjoint);
        const Tensor  ga = adjoint ? neg(matmul(c, gb, false, true)) : neg(matmul(gb, c, false, true));
        return Grads{sum_to(ga, as), sum_to(gb, bs)};
      },
      true);
  return out;
}

Tensor cholesky_solve(const Tensor& chol, const Tensor& b) {
  return triangular_solve(chol, triangular_solve(chol, b, true, false), true, true);
}

Tensor inverse(const Tensor& a, bool adjoint) {
  require_square(a, "inverse");
  const Tensor ac = as_float(a);
  return solve(ac, batched_eye(batch_shape(ac.shape()), ac.size(-1), ac.dtype()), adjoint);
}

Tensor det(const Tensor& a) {
  require_square(a, "det");
  const Tensor ac = as_float(a);
  Tensor       out;
  {
    NoGradGuard guard;
    Tensor      d;
    lu_det_raw(ac, nullptr, nullptr, &d);
    out = from_f64(d, ac.dtype());
  }
  SavedTensor sa(ac);
  record(
      "MatrixDeterminant", {ac}, {out},
      [sa](const Grads& g, const Grads& o) {
        const Tensor scale = expand_dims(expand_dims(mul(g[0], o[0]), -1), -1);
        return Grads{mul(scale, inverse(sa.get(), true))};
      },
      true);
  return out;
}

SlogdetResult slogdet(const Tensor& a) {
  require_square(a, "slogdet");
  const Tensor  ac = as_float(a);
  SlogdetResult res;
  {
    NoGradGuard guard;
    Tensor      s, l;
    lu_det_raw(ac, &s, &l, nullptr);
    res.sign        = from_f64(s, ac.dtype());
    res.log_abs_det = from_f64(l, ac.dtype());
  }
  SavedTensor sa(ac);
  record("LogMatrixDeterminant", {ac}, {res.sign, res.log_abs_det}, [sa](const Grads& g, const Grads&) {
    return Grads{mul(expand_dims(expand_dims(g[1], -1), -1), inverse(sa.get(), true))};
  });
  return res;
}

Tensor logdet(const Tensor& a) { return mul(sum(log(matrix_diag_part(cholesky(a))), {-1}), 2.0); }

// ---- least squares, pseudo-inverse, general eigen -------------------------------------------------

Tensor lstsq(const Tensor& a, const Tensor& b, double l2_regularizer, bool fast) {
  require_matrix(a, "lstsq");
  require_matrix(b, "lstsq");
  const DType  dt = promote_types(float_dtype(a), float_dtype(b));
  const Tensor ac = detail::as_dtype(a, dt), bc = detail::as_dtype(b, dt);
  if (bc.size(-2) != ac.size(-2))
    throw std::invalid_argument("lstsq: shapes " + shape_str(ac.shape()) + " and " + shape_str(bc.shape()) + " do not fit");
  if (!fast) return matmul(pinv(ac), bc);
  const I64 m = ac.size(-2), n = ac.size(-1);
  // TF's composite: Cholesky of the regularised Gramian of the smaller side.
  const auto gram_chol = [&](bool first_kind) {
    Tensor gram = first_kind ? matmul(ac, ac, true, false) : matmul(ac, ac, false, true);
    if (l2_regularizer != 0.0) gram = add(gram, mul(eye(first_kind ? n : m, dt), l2_regularizer));
    return cholesky(gram);
  };
  if (m >= n) return cholesky_solve(gram_chol(true), matmul(ac, bc, true, false));
  return matmul(ac, cholesky_solve(gram_chol(false), bc), true, false);
}

Tensor matrix_solve_ls(const Tensor& a, const Tensor& b, double l2_regularizer, bool fast) {
  return lstsq(a, b, l2_regularizer, fast);
}

Tensor pinv(const Tensor& a, double rcond) {
  require_matrix(a, "pinv");
  const Tensor ac = as_float(a);
  const I64    m = ac.size(-2), n = ac.size(-1);
  if (std::min(m, n) == 0) return zeros(with_matrix(batch_shape(ac.shape()), n, m), ac.dtype());
  if (rcond < 0) {
    const double eps = ac.dtype() == DType::Float64 ? std::numeric_limits<double>::epsilon()
                                                    : static_cast<double>(std::numeric_limits<float>::epsilon());
    rcond = 10.0 * static_cast<double>(std::max(m, n)) * eps;
  }
  const SvdResult r      = svd(ac, true, false);
  const Tensor    cutoff = mul(max(r.s, {-1}, true), rcond);
  const Tensor    s      = where(greater(r.s, cutoff), r.s, std::numeric_limits<double>::infinity());
  return matmul(div(r.v, expand_dims(s, -2)), r.u, false, true);
}

EigResult eig(const Tensor& a, bool compute_v) {
  require_square(a, "eig");
  const Tensor ac = as_float(a);
  const DType  dt = ac.dtype();
  const I64    n  = ac.size(-1);
  const Shape  batch = batch_shape(ac.shape());
  Tensor       er, ei, vr, vi;
  {
    NoGradGuard  guard;
    const Tensor pa = packed_f64(ac);
    Shape        es = batch;
    es.push_back(n);
    Tensor ter = empty(es, DType::Float64), tei = empty(es, DType::Float64), tvr, tvi;
    if (compute_v) tvr = empty(ac.shape(), DType::Float64), tvi = empty(ac.shape(), DType::Float64);
    const double* src = pa.data_ptr<double>();
    double *      per = ter.data_ptr<double>(), *pei = tei.data_ptr<double>();
    double *      pvr = compute_v ? tvr.data_ptr<double>() : nullptr, *pvi = compute_v ? tvi.data_ptr<double>() : nullptr;
    for_batch(shape_numel(batch), 30 * n * n * n, [&](I64 p) {
      eig_any(src + p * n * n, per + p * n, pei + p * n, pvr ? pvr + p * n * n : nullptr, pvi ? pvi + p * n * n : nullptr,
              n);
    });
    er = from_f64(ter, dt);
    ei = from_f64(tei, dt);
    if (compute_v) vr = from_f64(tvr, dt), vi = from_f64(tvi, dt);
  }
  if (compute_v) {
    // TF's _EigGrad.
    record(
        "Eig", {ac}, {er, ei, vr, vi},
        [n, dt](const Grads& g, const Grads& o) {
          const ComplexTensor e{o[0], o[1]}, v{o[2], o[3]}, ge{g[0], g[1]}, gv{g[2], g[3]};
          const ComplexTensor vt = cadjoint(v);
          // f[i, j] = conj(1 / (e_j - e_i)) off the diagonal, 0 on it.
          const Tensor dr   = sub(expand_dims(e.real, -2), expand_dims(e.real, -1));
          const Tensor di   = sub(expand_dims(e.imag, -2), expand_dims(e.imag, -1));
          const Tensor off  = matrix_mask(n, n, dt, [](I64 i, I64 j) { return i == j ? 0.0 : 1.0; });
          const Tensor inv  = mul(reciprocal(add(add(square(dr), square(di)), 1e-20)), off);
          const ComplexTensor f{mul(dr, inv), mul(di, inv)};
          const ComplexTensor vgv = cmatmul(vt, gv);
          const Tensor        dgp = matrix_diag(matrix_diag_part(vgv.real));
          const ComplexTensor vtv = cmatmul(vt, v);
          const ComplexTensor t{matmul(vtv.real, dgp), matmul(vtv.imag, dgp)};
          const ComplexTensor mid = complex_add(cdiag(ge), complex_mul(f, complex_sub(vgv, t)));
          return Grads{csolve(vt, cmatmul(mid, vt)).real};
        },
        true);
  } else {
    SavedTensor sa(ac);
    record("Eig", {ac}, {er, ei}, [sa](const Grads& g, const Grads&) {
      const ComplexTensor vt = cadjoint(eig(sa.get(), true).v);
      return Grads{csolve(vt, cmatmul(cdiag(ComplexTensor{g[0], g[1]}), vt)).real};
    });
  }
  EigResult res;
  res.e = ComplexTensor{er, ei};
  if (compute_v) res.v = ComplexTensor{vr, vi};
  return res;
}

ComplexTensor eigvals(const Tensor& a) { return eig(a, false).e; }

// ---- backend selection ---------------------------------------------------------------------------

void set_linalg_backend(LinalgBackend backend) { g_backend.store(static_cast<int>(backend)); }

LinalgBackend linalg_backend() { return static_cast<LinalgBackend>(g_backend.load()); }

bool linalg_has_lapack() { return kHaveLapack; }

}  // namespace datamunge::nn
