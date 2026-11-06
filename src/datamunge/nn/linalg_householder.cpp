// Blocked Householder QR, symmetric eigensolver and SVD (see linalg_householder.hpp).
//
// The kernels follow LAPACK and work on column-major copies, where Householder
// vectors are contiguous columns. A column-major matrix is the row-major
// transpose, so cm_gemm / cm_gemv call the row-major in-house BLAS with the
// operands swapped. Blocks of kBlock reflectors are applied in the compact WY
// form I - V T V^T, which turns most of the work into three GEMMs per block.
//
//   QR    dgeqrf (unblocked panels, block reflector on the trailing columns),
//         dorgqr (block reflectors applied backwards to the identity).
//   eigh  dsytrd/dlatrd (panels of rank-2 updates folded into two GEMMs),
//         dstedc (Cuppen's divide and conquer: rank-one tearing, deflation,
//         secular equation, Gu-Eisenstat vectors, one GEMM per merge),
//         dormtr (block reflectors).
//   SVD   QR first when m > n, dgebrd/dlabrd, a bidiagonal divide and conquer
//         (Gu-Eisenstat; dbdsdc/dlasd1-3 arrangement), dormbr.
//
// Leaves of the divide and conquer use implicit QL (eigen) and one-sided
// Jacobi (SVD) on blocks of at most a few dozen rows.

#include "linalg_householder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "blas_dispatch.hpp"
#include "datamunge/nn/blas.hpp"
#include "datamunge/nn/parallel.hpp"
#include "gemm.hpp"

namespace datamunge::nn::detail {
namespace {

using I64 = std::int64_t;
using Vec = std::vector<double>;

/// LAPACK's relative machine precision, dlamch('E') = 2^-53.
constexpr double kEps = std::numeric_limits<double>::epsilon() / 2;
/// Reflectors per block (LAPACK's default block size for these drivers).
constexpr I64 kBlock = 32;
/// Largest tridiagonal / bidiagonal block solved directly in the divide and conquer.
constexpr I64 kEigLeaf = 32;
constexpr I64 kSvdLeaf = 12;

/// Matrices and products with fewer entries than this stay on the calling thread (a pool dispatch
/// costs about as much as copying or multiplying them).
constexpr I64 kSplitElems = I64{1} << 16;

std::size_t sz(I64 x) { return static_cast<std::size_t>(x); }

/// fn(lo, hi) over [0, count), on the pool when the items (each touching about `cost` entries)
/// add up to enough work.
template <typename F>
void split_for(I64 count, I64 cost, F&& fn) {
  cost = std::max<I64>(cost, 1);
  if (count * cost < kSplitElems)
    fn(I64{0}, count);
  else
    parallel_for(0, count, std::max<I64>(1, kSplitElems / (8 * cost)), fn);
}


/// Reverses the order of the first cols columns of the column-major Q [rows, cols].
void reverse_columns(double* Q, I64 ldq, I64 rows, I64 cols) {
  for (I64 j = 0; j < cols / 2; ++j) std::swap_ranges(Q + j * ldq, Q + j * ldq + rows, Q + (cols - 1 - j) * ldq);
}

// ---- column-major BLAS on the row-major kernels ---------------------------------------------------

/// C [m, n] = alpha op(A) op(B) + beta C, column-major: C^T = op(B)^T op(A)^T in row-major terms.
void cm_gemm(bool ta, bool tb, I64 m, I64 n, I64 k, double alpha, const double* A, I64 lda, const double* B, I64 ldb,
             double beta, double* C, I64 ldc) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0) {
    for (I64 j = 0; j < n; ++j)
      for (I64 i = 0; i < m; ++i) C[i + j * ldc] = beta == 0.0 ? 0.0 : beta * C[i + j * ldc];
    return;
  }
  gemm(tb, ta, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);
}

/// y = alpha op(A) x + beta y with A [m, n] column-major (the row-major [n, m] matrix).
///
/// Strided vectors are copied to contiguous scratch first (the kernel's strided paths are scalar),
/// and the product is split over y in cache-line multiples, so each entry is still computed by one
/// thread in one order and the result does not depend on the thread count. blas::gemv splits
/// only from 2^20 entries; the reductions' panels make thousands of smaller products that still
/// gain from the pool.
void cm_gemv(bool trans, I64 m, I64 n, double alpha, const double* A, I64 lda, const double* x, I64 incx, double beta,
             double* y, I64 incy) {
  const I64 ylen = trans ? n : m, xlen = trans ? m : n;
  if (ylen <= 0) return;
  if (xlen <= 0) {
    for (I64 i = 0; i < ylen; ++i) y[i * incy] = beta == 0.0 ? 0.0 : beta * y[i * incy];
    return;
  }
  thread_local Vec xs, ys;
  if (incx != 1) {
    xs.resize(sz(xlen));
    for (I64 i = 0; i < xlen; ++i) xs[sz(i)] = x[i * incx];
    x = xs.data();
  }
  double* yc = y;
  if (incy != 1) {
    ys.resize(sz(ylen));
    if (beta != 0.0)
      for (I64 i = 0; i < ylen; ++i) ys[sz(i)] = y[i * incy];
    yc = ys.data();
  }
  const auto& kern = blas_kernels();
  const auto  part = [&](I64 lo, I64 hi) {
    if (trans)  // y = A^T x: rows [lo, hi) of the row-major [n, m] matrix
      kern.dgemv(false, hi - lo, m, alpha, A + lo * lda, lda, x, 1, beta, yc + lo, 1);
    else  // y = A x: columns [lo, hi) of the row-major [n, m] matrix
      kern.dgemv(true, n, hi - lo, alpha, A + lo, lda, x, 1, beta, yc + lo, 1);
  };
  constexpr I64 kChunk = 64;  // entries of y per piece: whole cache lines
  if (ylen * xlen < 2 * kSplitElems || ylen < 2 * kChunk) {
    part(0, ylen);
  } else {
    const I64 pieces = (ylen + kChunk - 1) / kChunk;
    parallel_for(0, pieces, std::max<I64>(1, kSplitElems / (kChunk * xlen)),
                 [&](I64 lo, I64 hi) { part(lo * kChunk, std::min(ylen, hi * kChunk)); });
  }
  if (incy != 1)
    for (I64 i = 0; i < ylen; ++i) y[i * incy] = ys[sz(i)];
}

/// Euclidean norm without overflow or underflow (scaled sum when the plain one is out of range).
double nrm2(I64 n, const double* x, I64 incx) {
  if (n <= 0) return 0.0;
  const double s = blas::dot(n, x, incx, x, incx);
  if (s > 1e-280 && s < 1e280) return std::sqrt(s);
  double scale = 0.0, ssq = 1.0;
  for (I64 i = 0; i < n; ++i) {
    const double v = std::abs(x[i * incx]);
    if (v == 0.0) continue;
    if (scale < v) {
      ssq   = 1.0 + ssq * (scale / v) * (scale / v);
      scale = v;
    } else {
      ssq += (v / scale) * (v / scale);
    }
  }
  return scale * std::sqrt(ssq);
}

/// Rotates the columns x, y (n entries): x = c x + s y, y = c y - s x (BLAS drot).
void rot(I64 n, double* x, double* y, double c, double s) {
  for (I64 i = 0; i < n; ++i) {
    const double a = x[i], b = y[i];
    x[i] = c * a + s * b;
    y[i] = c * b - s * a;
  }
}

// ---- Householder reflectors ----------------------------------------------------------------------

/// LAPACK dlarfg: H = I - tau v v^T, v = [1; x'], maps [alpha; x] (n entries) to [beta; 0]; alpha
/// becomes beta (with the opposite sign of alpha) and x becomes x'. tau = 0 (H = I) when x is zero.
void larfg(I64 n, double& alpha, double* x, I64 incx, double& tau) {
  tau = 0.0;
  if (n <= 1) return;
  double xnorm = nrm2(n - 1, x, incx);
  if (xnorm == 0.0) return;
  const auto beta_of = [&] { return alpha >= 0.0 ? -std::hypot(alpha, xnorm) : std::hypot(alpha, xnorm); };
  double       beta   = beta_of();
  const double safmin = std::numeric_limits<double>::min() / kEps;
  int          knt    = 0;
  while (std::abs(beta) < safmin && knt < 20) {  // rescale tiny vectors, as LAPACK
    ++knt;
    blas::scal(n - 1, 1.0 / safmin, x, incx);
    beta /= safmin;
    alpha /= safmin;
  }
  if (knt > 0) {
    xnorm = nrm2(n - 1, x, incx);
    beta  = beta_of();
  }
  tau = (beta - alpha) / beta;
  blas::scal(n - 1, 1.0 / (alpha - beta), x, incx);
  for (int j = 0; j < knt; ++j) beta *= safmin;
  alpha = beta;
}

/// LAPACK dlarft (forward, columnwise): T [kb, kb] upper triangular (ld kb) with
/// H_0 ... H_{kb-1} = I - V T V^T for the explicit reflectors V [rows, kb]. S is scratch.
void larft(I64 rows, I64 kb, const double* V, const double* tau, double* T, Vec& S) {
  S.resize(sz(kb * kb));
  cm_gemm(true, false, kb, kb, rows, 1.0, V, rows, V, rows, 0.0, S.data(), kb);  // S = V^T V
  for (I64 i = 0; i < kb; ++i) {
    double* ti = T + i * kb;
    std::fill(ti, ti + kb, 0.0);
    if (tau[i] == 0.0) continue;
    for (I64 r = 0; r < i; ++r) {  // T[:i, i] = -tau_i T[:i, :i] S[:i, i]
      double acc = 0.0;
      for (I64 c = r; c < i; ++c) acc += T[r + c * kb] * S[sz(c + i * kb)];
      ti[r] = -tau[i] * acc;
    }
    ti[i] = tau[i];
  }
}

/// LAPACK dlarfb (left, forward, columnwise): C [rows, nc] = (I - V op(T) V^T) C with op(T) = T^T
/// when trans. Three GEMMs; W1, W2 are scratch.
void larfb(bool trans, I64 rows, I64 nc, I64 kb, const double* V, const double* T, double* C, I64 ldc, Vec& W1,
           Vec& W2) {
  if (rows <= 0 || nc <= 0 || kb <= 0) return;
  W1.resize(sz(kb * nc));
  W2.resize(sz(kb * nc));
  cm_gemm(true, false, kb, nc, rows, 1.0, V, rows, C, ldc, 0.0, W1.data(), kb);
  cm_gemm(trans, false, kb, nc, kb, 1.0, T, kb, W1.data(), kb, 0.0, W2.data(), kb);
  cm_gemm(false, false, rows, nc, kb, -1.0, V, rows, W2.data(), kb, 1.0, C, ldc);
}

/// Applies k reflectors H_j = I - tau_j v_j v_j^T to C [rows, nc]: C = H_0 ... H_{k-1} C, or with
/// trans C = H_{k-1} ... H_0 C. v_j is zero above entry j, one at j, and get(j, i) below (i > j).
/// With identity, C starts as the first nc columns of the identity and the columns left of each
/// block are skipped (still unit vectors there), as LAPACK dorgqr.
template <typename Get>
void apply_reflectors(bool trans, I64 rows, I64 nc, I64 k, Get get, const double* tau, double* C, I64 ldc,
                      bool identity) {
  if (k <= 0 || nc <= 0) return;
  Vec       V, T(sz(kBlock * kBlock)), S, W1, W2;
  const I64 blocks = (k + kBlock - 1) / kBlock;
  for (I64 bi = 0; bi < blocks; ++bi) {
    const I64 b = trans ? bi : blocks - 1 - bi;
    const I64 j = b * kBlock, jb = std::min(kBlock, k - j), vr = rows - j;
    V.assign(sz(vr * jb), 0.0);
    for (I64 c = 0; c < jb; ++c) {
      double* vc = V.data() + c * vr;
      vc[c]      = 1.0;
      for (I64 i = c + 1; i < vr; ++i) vc[i] = get(j + c, j + i);
    }
    larft(vr, jb, V.data(), tau + j, T.data(), S);
    const I64 c0 = identity ? std::min(j, nc) : 0;
    larfb(trans, vr, nc - c0, jb, V.data(), T.data(), C + j + c0 * ldc, ldc, W1, W2);
  }
}

// ---- QR -------------------------------------------------------------------------------------------

/// LAPACK dgeqr2: unblocked QR of the panel A [m, n] in place (reflectors below the diagonal).
void geqr2(I64 m, I64 n, double* A, I64 lda, double* tau, Vec& w) {
  const I64   k    = std::min(m, n);
  const auto& kern = blas_kernels();
  w.resize(sz(std::max<I64>(n, 1)));
  for (I64 i = 0; i < k; ++i) {
    double* aii = A + i + i * lda;
    larfg(m - i, *aii, A + std::min(i + 1, m - 1) + i * lda, 1, tau[i]);
    if (i + 1 >= n || tau[i] == 0.0) continue;
    const double keep = *aii;
    *aii              = 1.0;  // v = A[i:, i] with its implicit leading one
    cm_gemv(true, m - i, n - i - 1, 1.0, aii + lda, lda, aii, 1, 0.0, w.data(), 1);
    for (I64 c = 0; c < n - i - 1; ++c) kern.daxpy(m - i, -tau[i] * w[sz(c)], aii, 1, aii + (c + 1) * lda, 1);
    *aii = keep;
  }
}

/// LAPACK dgeqrf: blocked QR of A [m, n] in place.
void geqrf(I64 m, I64 n, double* A, I64 lda, double* tau) {
  const I64 k = std::min(m, n);
  Vec       w, V, T(sz(kBlock * kBlock)), S, W1, W2;
  for (I64 j = 0; j < k; j += kBlock) {
    const I64 jb = std::min(kBlock, k - j), vr = m - j;
    double*   panel = A + j + j * lda;
    geqr2(vr, jb, panel, lda, tau + j, w);
    if (j + jb >= n) continue;
    V.assign(sz(vr * jb), 0.0);
    for (I64 c = 0; c < jb; ++c) {
      double* vc = V.data() + c * vr;
      vc[c]      = 1.0;
      std::copy(panel + c + 1 + c * lda, panel + vr + c * lda, vc + c + 1);
    }
    larft(vr, jb, V.data(), tau + j, T.data(), S);
    larfb(true, vr, n - j - jb, jb, V.data(), T.data(), panel + jb * lda, lda, W1, W2);
  }
}

// ---- symmetric tridiagonal reduction ------------------------------------------------------------
//
// As LAPACK dsytrd, the reduction reads and updates one triangle (the lower): the matrix-vector
// product of each column reads every entry once and the trailing update of each panel takes half
// the flops. The column blocks of both lie on a grid of kSymBlock fixed in the full matrix, and
// each update writes its blocks' whole diagonal squares, so those stay valid on both sides of the
// diagonal and are multiplied as ordinary blocks.

/// Width of the column blocks of the one-triangle products and updates.
constexpr I64 kSymBlock = 64;

/// Starts of the column blocks of [0, len) on the kSymBlock grid, with row 0 at row g0 of the grid;
/// the last entry is len.
void sym_blocks(I64 len, I64 g0, std::vector<I64>& starts) {
  starts.clear();
  for (I64 c = 0; c < len;) {
    starts.push_back(c);
    c = std::min(len, c == 0 ? kSymBlock - g0 % kSymBlock : c + kSymBlock);
  }
  starts.push_back(len);
}

struct SymScratch {
  std::vector<I64> starts, offs;
  Vec              part, xy;
};

/// y = A x for the symmetric A [len, len] (column-major) of which the lower triangle and the
/// diagonal blocks of the grid (row 0 at grid row g0) are stored. Each column block multiplies
/// its diagonal block and, in one pass over the panel below it, both that panel and its
/// transpose, into a partial vector of its own; the partials are summed in block order, so the
/// result does not depend on the thread count.
void symv_lower(I64 len, const double* A, I64 lda, const double* x, double* y, I64 g0, SymScratch& s) {
  if (len <= 0) return;
  const auto& kern = blas_kernels();
  sym_blocks(len, g0, s.starts);
  const I64 nb = static_cast<I64>(s.starts.size()) - 1;
  s.offs.resize(sz(nb + 1));
  s.offs[0] = 0;
  for (I64 k = 0; k < nb; ++k) s.offs[sz(k + 1)] = s.offs[sz(k)] + (len - s.starts[sz(k)]);
  s.part.resize(sz(s.offs[sz(nb)]));
  const auto block = [&](I64 k0, I64 k1) {
    for (I64 k = k0; k < k1; ++k) {
      const I64 c0 = s.starts[sz(k)], c1 = s.starts[sz(k + 1)], w = c1 - c0, below = len - c1;
      double*   pk = s.part.data() + s.offs[sz(k)];
      // Diagonal block: rows of the row-major view are its columns (it is symmetric).
      kern.dgemv(false, w, w, 1.0, A + c0 + c0 * lda, lda, x + c0, 1, 0.0, pk, 1);
      if (below <= 0) continue;
      std::fill(pk + w, pk + w + below, 0.0);
      kern.dgemv_pair(w, below, 1.0, A + c1 + c0 * lda, lda, x + c0, x + c1, pk + w, pk);
    }
  };
  if (nb > 1 && len * len >= kSplitElems)
    parallel_for(0, nb, 1, block);
  else
    block(0, nb);
  const auto reduce = [&](I64 lo, I64 hi) {
    std::copy(s.part.begin() + lo, s.part.begin() + hi, y + lo);
    for (I64 k = 1; k < nb && s.starts[sz(k)] < hi; ++k) {
      const I64 a = std::max(lo, s.starts[sz(k)]);
      kern.daxpy(hi - a, 1.0, s.part.data() + s.offs[sz(k)] + (a - s.starts[sz(k)]), 1, y + a, 1);
    }
  };
  constexpr I64 kRows = 256;
  split_for((len + kRows - 1) / kRows, kRows * nb / 2,
            [&](I64 lo, I64 hi) { reduce(lo * kRows, std::min(len, hi * kRows)); });
}

/// C -= V W^T + W V^T on the lower triangle and the diagonal blocks (grid row g0 at row 0) of
/// C [n, n] (column-major); V and W are [n, k]. One GEMM per column block, [V W] [W V]^T, so
/// each entry is written by one serial product whatever the thread count.
void syr2k_lower(I64 n, I64 k, const double* V, I64 ldv, const double* W, I64 ldw, double* C, I64 ldc, I64 g0,
                 SymScratch& s) {
  if (n <= 0 || k <= 0) return;
  const I64 k2 = 2 * k;
  s.xy.resize(sz(3 * n * k));  // [V W V]: [V W] and [W V] are its first and last 2k columns
  double* X = s.xy.data();
  double* Y = X + n * k;
  for (I64 c = 0; c < k; ++c) {
    std::copy(V + c * ldv, V + c * ldv + n, X + c * n);
    std::copy(W + c * ldw, W + c * ldw + n, X + (k + c) * n);
    std::copy(V + c * ldv, V + c * ldv + n, X + (k2 + c) * n);
  }
  sym_blocks(n, g0, s.starts);
  const I64  nb    = static_cast<I64>(s.starts.size()) - 1;
  const auto block = [&](I64 b0, I64 b1) {
    for (I64 b = b0; b < b1; ++b) {
      const I64 c0 = s.starts[sz(b)], c1 = s.starts[sz(b + 1)];
      cm_gemm(false, true, n - c0, c1 - c0, k2, -1.0, X + c0, n, Y + c0, n, 1.0, C + c0 + c0 * ldc, ldc);
    }
  };
  if (nb > 1 && n * n * k >= 16 * kSplitElems)
    parallel_for(0, nb, 1, block);
  else
    block(0, nb);
}

/// LAPACK dlatrd (lower): reduces the first nb columns of the symmetric A [nn, nn] (column-major,
/// lower triangle and grid diagonal blocks stored; row 0 is row g0 of the full matrix) and
/// returns W [nn, nb] (ld ldw) for the trailing update A -= V W^T + W V^T. e[i], tau[i] for the
/// nb reflectors; A[i + 1, i] is left at one.
void latrd(I64 nn, I64 nb, double* A, I64 lda, double* e, double* tau, double* W, I64 ldw, I64 g0,
           SymScratch& s) {
  for (I64 i = 0; i < nb; ++i) {
    if (i > 0) {  // A[i:, i] -= A[i:, :i] W[i, :i]^T + W[i:, :i] A[i, :i]^T
      cm_gemv(false, nn - i, i, -1.0, A + i, lda, W + i, ldw, 1.0, A + i + i * lda, 1);
      cm_gemv(false, nn - i, i, -1.0, W + i, ldw, A + i, lda, 1.0, A + i + i * lda, 1);
    }
    if (i + 1 >= nn) continue;
    larfg(nn - i - 1, A[i + 1 + i * lda], A + std::min(i + 2, nn - 1) + i * lda, 1, tau[i]);
    e[i]                = A[i + 1 + i * lda];
    A[i + 1 + i * lda]  = 1.0;
    const I64     len   = nn - i - 1;
    const double* v     = A + (i + 1) + i * lda;
    double*       w     = W + (i + 1) + i * ldw;
    // w = A[i+1:, i+1:] v, then the corrections for the panel's earlier columns (not yet applied).
    symv_lower(len, A + (i + 1) + (i + 1) * lda, lda, v, w, g0 + i + 1, s);
    if (i > 0) {
      double* t = W + i * ldw;  // W[:i, i] as scratch
      cm_gemv(true, len, i, 1.0, W + (i + 1), ldw, v, 1, 0.0, t, 1);
      cm_gemv(false, len, i, -1.0, A + (i + 1), lda, t, 1, 1.0, w, 1);
      cm_gemv(true, len, i, 1.0, A + (i + 1), lda, v, 1, 0.0, t, 1);
      cm_gemv(false, len, i, -1.0, W + (i + 1), ldw, t, 1, 1.0, w, 1);
    }
    blas::scal(len, tau[i], w, 1);
    const double alpha = -0.5 * tau[i] * blas::dot(len, w, 1, v, 1);
    blas::axpy(len, alpha, v, 1, w, 1);
  }
}

/// LAPACK dsytrd (lower): A [n, n] symmetric (column-major; the lower triangle is read, the
/// upper one is left stale outside the grid's diagonal blocks) = Q T Q^T with T tridiagonal
/// (d [n], e [n - 1]) and Q = H_0 ... H_{n-2}; reflector j is stored below A[j + 1, j] (implicit
/// one there).
void sytrd(I64 n, double* A, I64 lda, double* d, double* e, double* tau) {
  Vec        W(sz(n * kBlock));
  SymScratch s;
  for (I64 i0 = 0; i0 + 1 < n; i0 += kBlock) {
    const I64 kb = std::min(kBlock, n - 1 - i0), nn = n - i0;
    latrd(nn, kb, A + i0 + i0 * lda, lda, e + i0, tau + i0, W.data(), nn, i0, s);
    const I64 r0 = i0 + kb, rest = n - r0;
    if (rest > 0) syr2k_lower(rest, kb, A + r0 + i0 * lda, lda, W.data() + kb, nn, A + r0 + r0 * lda, lda, r0, s);
    for (I64 j = i0; j < i0 + kb; ++j) A[j + 1 + j * lda] = e[j];
  }
  for (I64 j = 0; j < n; ++j) d[j] = A[j + j * lda];
}

// ---- symmetric tridiagonal eigensolver ----------------------------------------------------------

/// Implicit QL on the tridiagonal (d [n], e [n] with e[i] coupling i and i + 1; destroyed), after
/// JAMA's tql2. With Z, the rotations accumulate on its first n columns (nz rows, column-major).
/// Eigenvalues are left unsorted. Expects entries of order one (callers scale).
void tql(I64 n, double* d, double* e, double* Z, I64 ldz, I64 nz) {
  if (n <= 1) return;
  e[n - 1]          = 0.0;
  const double eps  = std::numeric_limits<double>::epsilon();
  double       f    = 0.0, tst1 = 0.0;
  for (I64 l = 0; l < n; ++l) {
    tst1  = std::max(tst1, std::abs(d[l]) + std::abs(e[l]));
    I64 m = l;
    while (m + 1 < n && std::abs(e[m]) > eps * tst1) ++m;
    if (m > l) {
      int iter = 0;
      do {
        if (++iter > 300) throw std::runtime_error("eigh: QL iteration did not converge (non-finite input?)");
        double g = d[l];
        double p = (d[l + 1] - g) / (2.0 * e[l]);
        double r = std::sqrt(p * p + 1.0);
        if (p < 0) r = -r;
        d[l]             = e[l] / (p + r);
        d[l + 1]         = e[l] * (p + r);
        const double dl1 = d[l + 1];
        double       h   = g - d[l];
        for (I64 i = l + 2; i < n; ++i) d[i] -= h;
        f += h;
        p                = d[m];
        double       c   = 1.0, c2 = 1.0, c3 = 1.0, s = 0.0, s2 = 0.0;
        const double el1 = e[l + 1];
        for (I64 i = m - 1; i >= l; --i) {
          c3       = c2;
          c2       = c;
          s2       = s;
          g        = c * e[i];
          h        = c * p;
          r        = std::sqrt(p * p + e[i] * e[i]);
          e[i + 1] = s * r;
          s        = e[i] / r;
          c        = p / r;
          p        = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          if (Z) {
            double* zi  = Z + i * ldz;
            double* zi1 = Z + (i + 1) * ldz;
            for (I64 k = 0; k < nz; ++k) {
              const double t = zi1[k];
              zi1[k]         = s * zi[k] + c * t;
              zi[k]          = c * zi[k] - s * t;
            }
          }
        }
        p    = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (std::abs(e[l]) > eps * tst1);
    }
    d[l] += f;
    e[l] = 0.0;
  }
}

/// Root in (lo, hi) of g(t) = 1 + rho sum_i z_i^2 / (P_i - t), with poles P ascending, P_j <= lo and
/// P_{j+1} >= hi (no right pole when j + 1 == K); g increases on the interval. Each step fits
/// psi (poles up to j) and phi (the rest) with c + b / (pole - t) by value and slope and solves the
/// model (Bunch-Nielsen-Sorensen / Gragg); bisection keeps the iterate inside the bracket.
double secular_root(I64 K, I64 j, const double* P, const double* z, double rho, double lo, double hi) {
  double t = 0.5 * (lo + hi);
  for (int it = 0; it < 200; ++it) {
    double psi = 0.0, dpsi = 0.0, phi = 0.0, dphi = 0.0;
    for (I64 i = 0; i <= j; ++i) {
      const double r = 1.0 / (P[i] - t), w = rho * z[i] * z[i] * r;
      psi += w;
      dpsi += w * r;
    }
    for (I64 i = j + 1; i < K; ++i) {
      const double r = 1.0 / (P[i] - t), w = rho * z[i] * z[i] * r;
      phi += w;
      dphi += w * r;
    }
    const double f = 1.0 + psi + phi;
    if (std::abs(f) <= kEps * (8.0 * (phi - psi) + 2.0 + std::abs(t) * (dpsi + dphi))) return t;
    if (f < 0.0)
      lo = t;
    else
      hi = t;
    const double dp = P[j] - t;
    double       tn;
    if (j + 1 < K) {
      const double dq = P[j + 1] - t;
      const double b1 = dpsi * dp * dp, b2 = dphi * dq * dq;
      const double C = f - b1 / dp - b2 / dq;  // the model is C + b1 / (dp - eta) + b2 / (dq - eta)
      const double B = C * (dp + dq) + b1 + b2, C0 = dp * dq * f;
      double       eta;
      if (C == 0.0) {
        eta = C0 / B;
      } else {  // C eta^2 - B eta + C0 = 0: take the root between the poles
        const double qq = 0.5 * (B + std::copysign(std::sqrt(std::max(0.0, B * B - 4.0 * C * C0)), B));
        const double r1 = qq / C, r2 = qq != 0.0 ? C0 / qq : r1;
        eta             = (r1 > dp && r1 < dq) ? r1 : r2;
      }
      tn = t + eta;
    } else {
      const double b1 = dpsi * dp * dp, C = f - b1 / dp;
      tn              = C > 0.0 ? t + dp + b1 / C : 0.5 * (lo + hi);
    }
    if (!(tn > lo && tn < hi)) {
      tn = 0.5 * (lo + hi);
      if (!(tn > lo && tn < hi)) return t;
    }
    if (tn == t) return t;
    t = tn;
  }
  return t;
}

/// Root j of 1 + rho sum_i z_i^2 / (d_i - x) (d ascending and distinct, z nonzero, rho > 0), found
/// relative to the nearer pole d_o: returns the root and delta[i] = d_i - root = (d_i - d_o) - tau.
double eig_secular(I64 K, I64 j, const double* d, const double* z, double rho, double* delta) {
  if (K == 1) {
    delta[0] = -rho * z[0] * z[0];
    return d[0] + rho * z[0] * z[0];
  }
  I64    o;
  double lo, hi;
  if (j + 1 < K) {
    const double half = 0.5 * (d[j + 1] - d[j]);
    double       f    = 1.0;
    for (I64 i = 0; i < K; ++i) f += rho * z[i] * z[i] / ((d[i] - d[j]) - half);
    if (f >= 0.0)
      o = j, lo = 0.0, hi = half;
    else
      o = j + 1, lo = -half, hi = 0.0;
  } else {
    double ss = 0.0;
    for (I64 i = 0; i < K; ++i) ss += z[i] * z[i];
    o = K - 1, lo = 0.0, hi = rho * ss;
  }
  for (I64 i = 0; i < K; ++i) delta[i] = d[i] - d[o];
  const double tau = secular_root(K, j, delta, z, rho, lo, hi);
  for (I64 i = 0; i < K; ++i) delta[i] -= tau;
  return d[o] + tau;
}

/// Sorts eigenvalues ascending with their columns of Z (n rows), for small n.
void sort_pairs(I64 n, double* d, double* Z, I64 ldz, I64 rows) {
  for (I64 i = 0; i + 1 < n; ++i) {
    I64 k = i;
    for (I64 j = i + 1; j < n; ++j)
      if (d[j] < d[k]) k = j;
    if (k == i) continue;
    std::swap(d[i], d[k]);
    if (Z) std::swap_ranges(Z + i * ldz, Z + i * ldz + rows, Z + k * ldz);
  }
}

/// Merge step of the eigen divide and conquer (LAPACK dlaed1-3). On entry d[:m] and d[m:] hold the
/// ascending eigenvalues of the two halves and Z [n, n] = diag(Q1, Q2); the tridiagonal is
/// diag(T1, T2) + |beta| u u^T with u = e_{m-1} + sign(beta) e_m, i.e. D + rho z z^T in the
/// eigenbasis with rho = 2 |beta| and z = Z^T u / sqrt(2). On exit d ascending, Z its eigenvectors.
void eig_merge(I64 n, I64 m, double* d, double* Z, I64 ldz, double beta) {
  const double     rho = 2.0 * std::abs(beta), sgn = beta < 0.0 ? -1.0 : 1.0, r2 = std::sqrt(0.5);
  Vec              z(sz(n));
  std::vector<int> type(sz(n));  // 1: rows < m only, 3: rows >= m only, 2: both
  for (I64 i = 0; i < n; ++i) {
    z[sz(i)]    = i < m ? Z[(m - 1) + i * ldz] * r2 : sgn * Z[m + i * ldz] * r2;
    type[sz(i)] = i < m ? 1 : 3;
  }
  std::vector<I64> order(sz(n));
  std::iota(order.begin(), order.end(), I64{0});
  std::inplace_merge(order.begin(), order.begin() + m, order.end(), [&](I64 x, I64 y) { return d[x] < d[y]; });
  double dmax = 0.0, zmax = 0.0;
  for (I64 i = 0; i < n; ++i) dmax = std::max(dmax, std::abs(d[i])), zmax = std::max(zmax, std::abs(z[sz(i)]));
  const double tol = 8.0 * kEps * std::max(dmax, rho * zmax);

  // Deflation (dlaed2): tiny z components, and pairs of close eigenvalues rotated so one z vanishes.
  std::vector<I64> keep, defl;
  I64              prev = -1;
  for (const I64 j : order) {
    if (rho * std::abs(z[sz(j)]) <= tol) {
      defl.push_back(j);
      continue;
    }
    if (prev < 0) {
      prev = j;
      continue;
    }
    double       s = z[sz(prev)], c = z[sz(j)];
    const double tau = std::hypot(c, s), t = d[j] - d[prev];
    c /= tau;
    s = -s / tau;
    if (std::abs(t * c * s) <= tol) {
      z[sz(j)]    = tau;
      z[sz(prev)] = 0.0;
      rot(n, Z + prev * ldz, Z + j * ldz, c, s);
      if (type[sz(prev)] != type[sz(j)]) type[sz(j)] = 2;
      const double dp = d[prev], dj = d[j];
      d[prev]         = dp * c * c + dj * s * s;
      d[j]            = dp * s * s + dj * c * c;
      defl.push_back(prev);
    } else {
      keep.push_back(prev);
    }
    prev = j;
  }
  if (prev >= 0) keep.push_back(prev);

  const I64 K = static_cast<I64>(keep.size());
  Vec       lam(sz(K)), znew;
  Vec       Znew;
  if (K > 0) {
    Vec dl(sz(K)), zl(sz(K)), delta(sz(K * K));
    for (I64 i = 0; i < K; ++i) dl[sz(i)] = d[keep[sz(i)]], zl[sz(i)] = z[sz(keep[sz(i)])];
    parallel_for(0, K, std::max<I64>(1, 4096 / K), [&](I64 lo, I64 hi) {
      for (I64 j = lo; j < hi; ++j) lam[sz(j)] = eig_secular(K, j, dl.data(), zl.data(), rho, delta.data() + j * K);
    });
    // Gu-Eisenstat: the z for which the computed roots are exact, then the vectors from it.
    Vec zh(sz(K));
    parallel_for(0, K, std::max<I64>(1, 4096 / K), [&](I64 lo, I64 hi) {
      for (I64 i = lo; i < hi; ++i) {
        double p = -delta[sz(i + i * K)];
        for (I64 j = 0; j < K; ++j)
          if (j != i) p *= delta[sz(i + j * K)] / (dl[sz(i)] - dl[sz(j)]);
        zh[sz(i)] = std::copysign(std::sqrt(p / rho), zl[sz(i)]);
      }
    });
    // Columns grouped by type, so the GEMMs skip the zero quarters of diag(Q1, Q2).
    std::vector<I64> kord;
    I64              cnt[4] = {0, 0, 0, 0};
    for (int t = 1; t <= 3; ++t)
      for (I64 i = 0; i < K; ++i)
        if (type[sz(keep[sz(i)])] == t) kord.push_back(i), ++cnt[t];
    Vec Qc(sz(n * K)), Uc(sz(K * K));
    split_for(K, n, [&](I64 lo, I64 hi) {
      for (I64 q = lo; q < hi; ++q) {
        const double* src = Z + keep[sz(kord[sz(q)])] * ldz;
        std::copy(src, src + n, Qc.data() + q * n);
      }
    });
    parallel_for(0, K, std::max<I64>(1, 4096 / K), [&](I64 lo, I64 hi) {
      for (I64 j = lo; j < hi; ++j) {
        double  nrm = 0.0;
        double* uj  = Uc.data() + j * K;
        for (I64 q = 0; q < K; ++q) {
          const I64 i = kord[sz(q)];
          uj[q]       = zh[sz(i)] / delta[sz(i + j * K)];
          nrm += uj[q] * uj[q];
        }
        const double inv = 1.0 / std::sqrt(nrm);
        for (I64 q = 0; q < K; ++q) uj[q] *= inv;
      }
    });
    Znew.assign(sz(n * K), 0.0);
    const I64 n1 = cnt[1], n12 = cnt[1] + cnt[2], n23 = cnt[2] + cnt[3];
    cm_gemm(false, false, m, K, n12, 1.0, Qc.data(), n, Uc.data(), K, 0.0, Znew.data(), n);
    cm_gemm(false, false, n - m, K, n23, 1.0, Qc.data() + m + n1 * n, n, Uc.data() + n1, K, 0.0, Znew.data() + m, n);
  }
  // Ascending order of the new eigenvalues and the deflated ones.
  std::vector<std::pair<double, I64>> items;  // index < K: new column; else deflated column (index - K)
  items.reserve(sz(n));
  for (I64 j = 0; j < K; ++j) items.emplace_back(lam[sz(j)], j);
  for (const I64 c : defl) items.emplace_back(d[c], K + c);
  std::stable_sort(items.begin(), items.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
  Vec out(sz(n * n));
  split_for(n, n, [&](I64 lo, I64 hi) {
    for (I64 c = lo; c < hi; ++c) {
      const I64     src = items[sz(c)].second;
      const double* col = src < K ? Znew.data() + src * n : Z + (src - K) * ldz;
      std::copy(col, col + n, out.data() + c * n);
    }
  });
  for (I64 c = 0; c < n; ++c) d[c] = items[sz(c)].first;
  split_for(n, n, [&](I64 lo, I64 hi) {
    for (I64 c = lo; c < hi; ++c) std::copy(out.data() + c * n, out.data() + (c + 1) * n, Z + c * ldz);
  });
}

/// Merges the solved halves [0, n / 2) and [n / 2, n) of Z [n, n] (see eig_merge).
void eig_join(I64 n, double* d, double* Z, I64 ldz, double beta) {
  const I64 m = n / 2;
  split_for(n, n, [&](I64 lo, I64 hi) {
    for (I64 c = lo; c < hi; ++c) {
      if (c < m)
        std::fill(Z + m + c * ldz, Z + n + c * ldz, 0.0);
      else
        std::fill(Z + c * ldz, Z + m + c * ldz, 0.0);
    }
  });
  eig_merge(n, m, d, Z, ldz, beta);
}

/// Divide and conquer on the tridiagonal (d [n], e [n] with e[i] coupling i and i + 1): d becomes
/// the ascending eigenvalues, Z [n, n] the eigenvectors. Halves are torn apart by subtracting
/// |e[m-1]| from the two diagonal entries next to the cut and merged by eig_merge.
void stedc_rec(I64 n, double* d, double* e, double* Z, I64 ldz) {
  if (n <= kEigLeaf) {
    for (I64 c = 0; c < n; ++c)
      for (I64 r = 0; r < n; ++r) Z[r + c * ldz] = r == c ? 1.0 : 0.0;
    tql(n, d, e, Z, ldz, n);
    sort_pairs(n, d, Z, ldz, n);
    return;
  }
  const I64    m    = n / 2;
  const double beta = e[m - 1];
  d[m - 1] -= std::abs(beta);
  d[m] -= std::abs(beta);
  stedc_rec(m, d, e, Z, ldz);
  stedc_rec(n - m, d + m, e + m, Z + m + m * ldz, ldz);
  eig_join(n, d, Z, ldz, beta);
}

/// One node of a divide and conquer: rows [off, off + n); beta is the coupling torn out (eigen)
/// and sqre the extra column (SVD).
struct DcNode {
  I64    off, n;
  int    sqre;
  double beta;
};

/// Smallest subtree size that runs whole on one thread: the subtrees then number about two per
/// thread, and the merges above them run one at a time on the whole pool.
I64 subtree_limit(I64 n, I64 leaf) {
  const I64 threads = get_num_threads();
  return threads <= 1 ? n : std::max(leaf, (n + 2 * threads - 1) / (2 * threads));
}

/// Tears the tridiagonal down to subtrees of at most limit rows (as stedc_rec), listing them and the
/// merges above them in post-order (children before parents).
void eig_plan(I64 off, I64 n, double* d, double* e, I64 limit, std::vector<DcNode>& subtrees,
              std::vector<DcNode>& merges) {
  if (n <= limit) {
    subtrees.push_back({off, n, 0, 0.0});
    return;
  }
  const I64    m    = n / 2;
  const double beta = e[off + m - 1];
  d[off + m - 1] -= std::abs(beta);
  d[off + m] -= std::abs(beta);
  eig_plan(off, m, d, e, limit, subtrees, merges);
  eig_plan(off + m, n - m, d, e, limit, subtrees, merges);
  merges.push_back({off, n, 0, beta});
}

/// Eigen decomposition of the symmetric tridiagonal (d [n], e [n - 1]): d ascending, Z [n, n] (ld n).
void stedc(I64 n, double* d, const double* e_in, double* Z) {
  Vec e(sz(n), 0.0);
  std::copy(e_in, e_in + std::max<I64>(n - 1, 0), e.begin());
  double scale = 0.0;
  for (I64 i = 0; i < n; ++i) scale = std::max({scale, std::abs(d[i]), std::abs(e[sz(i)])});
  if (scale == 0.0) {
    for (I64 c = 0; c < n; ++c)
      for (I64 r = 0; r < n; ++r) Z[r + c * n] = r == c ? 1.0 : 0.0;
    return;
  }
  for (I64 i = 0; i < n; ++i) d[i] /= scale, e[sz(i)] /= scale;
  // The subtrees on the pool (each on one thread), then the merges above them in order.
  std::vector<DcNode> subtrees, merges;
  eig_plan(0, n, d, e.data(), subtree_limit(n, kEigLeaf), subtrees, merges);
  parallel_for(0, static_cast<I64>(subtrees.size()), 1, [&](I64 lo, I64 hi) {
    for (I64 t = lo; t < hi; ++t) {
      const DcNode& s = subtrees[sz(t)];
      stedc_rec(s.n, d + s.off, e.data() + s.off, Z + s.off * (1 + n), n);
    }
  });
  for (const DcNode& mg : merges) eig_join(mg.n, d + mg.off, Z + mg.off * (1 + n), n, mg.beta);
  for (I64 i = 0; i < n; ++i) d[i] *= scale;
}

/// LAPACK dlae2: eigenvalues of [[a, b], [b, c]], rt1 the larger in absolute value.
void lae2(double a, double b, double c, double& rt1, double& rt2) {
  const double sm = a + c, adf = std::abs(a - c), ab = std::abs(b + b);
  const double acmx = std::abs(a) > std::abs(c) ? a : c, acmn = std::abs(a) > std::abs(c) ? c : a;
  double       rt;
  if (adf > ab)
    rt = adf * std::sqrt(1.0 + (ab / adf) * (ab / adf));
  else if (adf < ab)
    rt = ab * std::sqrt(1.0 + (adf / ab) * (adf / ab));
  else
    rt = ab * std::sqrt(2.0);
  if (sm == 0.0) {
    rt1 = 0.5 * rt;
    rt2 = -0.5 * rt;
    return;
  }
  rt1 = 0.5 * (sm < 0.0 ? sm - rt : sm + rt);
  rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
}

/// LAPACK dsterf: eigenvalues of the symmetric tridiagonal (d [n], e [n - 1], destroyed),
/// ascending in d, by the Pal-Walker-Kahan root-free QL or QR iteration (whichever chases toward
/// the smaller end of each block), on the squared off-diagonal. False if it does not converge.
bool sterf(I64 n, double* __restrict d, double* __restrict e) {
  if (n <= 1) return true;
  const double eps = kEps, eps2 = eps * eps, safmin = std::numeric_limits<double>::min();
  const double ssfmax = std::sqrt(1.0 / safmin) / 3.0, ssfmin = std::sqrt(safmin) / eps2;
  const I64    nmaxit = 30 * n;
  I64          jtot   = 0;
  // A shift from the leading 2 x 2 block at index l, toward neighbour l + dir.
  const auto shift = [&](I64 l, I64 dir) {
    const double rte   = std::sqrt(e[dir > 0 ? l : l - 1]);
    const double sigma = (d[l + dir] - d[l]) / (2.0 * rte);
    const double r     = std::hypot(sigma, 1.0);
    return d[l] - rte / (sigma + std::copysign(r, sigma));
  };
  for (I64 l1 = 0; l1 < n;) {
    if (l1 > 0) e[l1 - 1] = 0.0;
    I64 m = l1;
    for (; m < n - 1; ++m)
      if (std::abs(e[m]) <= std::sqrt(std::abs(d[m])) * std::sqrt(std::abs(d[m + 1])) * eps) {
        e[m] = 0.0;
        break;
      }
    I64       l = l1, lend = m;
    const I64 lsv = l, lendsv = lend;
    l1            = m + 1;
    if (lend == l) continue;
    // Scale the block into range, then square its off-diagonal.
    double anorm = 0.0;
    for (I64 i = l; i <= lend; ++i) anorm = std::max(anorm, std::abs(d[i]));
    for (I64 i = l; i < lend; ++i) anorm = std::max(anorm, std::abs(e[i]));
    if (anorm == 0.0) continue;
    const double to = anorm > ssfmax ? ssfmax : anorm < ssfmin ? ssfmin : anorm;
    if (to != anorm) {
      const double f = to / anorm;
      for (I64 i = l; i <= lend; ++i) d[i] *= f;
      for (I64 i = l; i < lend; ++i) e[i] *= f;
    }
    for (I64 i = l; i < lend; ++i) e[i] *= e[i];
    if (std::abs(d[lend]) < std::abs(d[l])) std::swap(l, lend);
    if (lend >= l) {  // QL: deflate at the top
      while (l <= lend) {
        for (m = l; m < lend; ++m)
          if (std::abs(e[m]) <= eps2 * std::abs(d[m] * d[m + 1])) break;
        if (m < lend) e[m] = 0.0;
        if (m == l) {
          ++l;
          continue;
        }
        if (m == l + 1) {
          double rt1, rt2;
          lae2(d[l], std::sqrt(e[l]), d[l + 1], rt1, rt2);
          d[l]     = rt1;
          d[l + 1] = rt2;
          e[l]     = 0.0;
          l += 2;
          continue;
        }
        if (jtot == nmaxit) break;
        ++jtot;
        const double sigma = shift(l, 1);
        double       c = 1.0, s = 0.0, gamma = d[m] - sigma, p = gamma * gamma;
        for (I64 i = m - 1; i >= l; --i) {
          const double bb = e[i], r = p + bb;
          if (i != m - 1) e[i + 1] = s * r;
          const double oldc = c, oldgam = gamma, alpha = d[i];
          c        = p / r;
          s        = bb / r;
          gamma    = c * (alpha - sigma) - s * oldgam;
          d[i + 1] = oldgam + (alpha - gamma);
          p        = c != 0.0 ? (gamma * gamma) / c : oldc * bb;
        }
        e[l] = s * p;
        d[l] = sigma + gamma;
      }
    } else {  // QR: deflate at the bottom
      while (l >= lend) {
        for (m = l; m > lend; --m)
          if (std::abs(e[m - 1]) <= eps2 * std::abs(d[m] * d[m - 1])) break;
        if (m > lend) e[m - 1] = 0.0;
        if (m == l) {
          --l;
          continue;
        }
        if (m == l - 1) {
          double rt1, rt2;
          lae2(d[l], std::sqrt(e[l - 1]), d[l - 1], rt1, rt2);
          d[l]     = rt1;
          d[l - 1] = rt2;
          e[l - 1] = 0.0;
          l -= 2;
          continue;
        }
        if (jtot == nmaxit) break;
        ++jtot;
        const double sigma = shift(l, -1);
        double       c = 1.0, s = 0.0, gamma = d[m] - sigma, p = gamma * gamma;
        for (I64 i = m; i < l; ++i) {
          const double bb = e[i], r = p + bb;
          if (i != m) e[i - 1] = s * r;
          const double oldc = c, oldgam = gamma, alpha = d[i + 1];
          c     = p / r;
          s     = bb / r;
          gamma = c * (alpha - sigma) - s * oldgam;
          d[i]  = oldgam + (alpha - gamma);
          p     = c != 0.0 ? (gamma * gamma) / c : oldc * bb;
        }
        e[l - 1] = s * p;
        d[l]     = sigma + gamma;
      }
    }
    if (to != anorm) {
      const double f = anorm / to;
      for (I64 i = lsv; i <= lendsv; ++i) d[i] *= f;
    }
    if (jtot >= nmaxit) return false;
  }
  std::sort(d, d + n);
  return true;
}

/// Eigenvalues only of the tridiagonal (d [n], e [n - 1]), ascending: sterf, or implicit QL if
/// that does not converge.
void tridiagonal_values(I64 n, double* d, const double* e_in) {
  Vec e(sz(n), 0.0);
  std::copy(e_in, e_in + std::max<I64>(n - 1, 0), e.begin());
  Vec d0(d, d + n);
  if (sterf(n, d, e.data())) return;
  std::copy(d0.begin(), d0.end(), d);
  std::copy(e_in, e_in + std::max<I64>(n - 1, 0), e.begin());
  double scale = 0.0;
  for (I64 i = 0; i < n; ++i) scale = std::max({scale, std::abs(d[i]), std::abs(e[sz(i)])});
  if (scale == 0.0) return;
  for (I64 i = 0; i < n; ++i) d[i] /= scale, e[sz(i)] /= scale;
  tql(n, d, e.data(), nullptr, 0, 0);
  std::sort(d, d + n);
  for (I64 i = 0; i < n; ++i) d[i] *= scale;
}

// ---- bidiagonal reduction -----------------------------------------------------------------------

/// LAPACK dlabrd (m >= n): reduces the first nb rows and columns of A [m, n] to upper bidiagonal
/// form and returns X [m, nb], Y [n, nb] for the trailing update A -= V Y^T + X U^T. A[i, i] and
/// A[i, i + 1] are left at one for that update. r [n] is scratch: row i is worked on there, since
/// A's rows are strided.
void labrd(I64 m, I64 n, I64 nb, double* A, I64 lda, double* d, double* e, double* tauq, double* taup, double* X,
           I64 ldx, double* Y, I64 ldy, double* r) {
  for (I64 i = 0; i < nb; ++i) {
    double* aii = A + i + i * lda;
    cm_gemv(false, m - i, i, -1.0, A + i, lda, Y + i, ldy, 1.0, aii, 1);
    cm_gemv(false, m - i, i, -1.0, X + i, ldx, A + i * lda, 1, 1.0, aii, 1);
    larfg(m - i, *aii, A + std::min(i + 1, m - 1) + i * lda, 1, tauq[i]);
    d[i] = *aii;
    if (i + 1 >= n) continue;
    *aii       = 1.0;
    double* yi = Y + i * ldy;  // Y[:, i]
    cm_gemv(true, m - i, n - i - 1, 1.0, aii + lda, lda, aii, 1, 0.0, yi + i + 1, 1);
    cm_gemv(true, m - i, i, 1.0, A + i, lda, aii, 1, 0.0, yi, 1);
    cm_gemv(false, n - i - 1, i, -1.0, Y + i + 1, ldy, yi, 1, 1.0, yi + i + 1, 1);
    cm_gemv(true, m - i, i, 1.0, X + i, ldx, aii, 1, 0.0, yi, 1);
    cm_gemv(true, i, n - i - 1, -1.0, A + (i + 1) * lda, lda, yi, 1, 1.0, yi + i + 1, 1);
    blas::scal(n - i - 1, tauq[i], yi + i + 1, 1);
    // Row i right of the diagonal, in r.
    const I64 nr  = n - i - 1;
    double*   row = A + i + (i + 1) * lda;  // stride lda
    for (I64 c = 0; c < nr; ++c) r[c] = row[c * lda];
    cm_gemv(false, nr, i + 1, -1.0, Y + i + 1, ldy, A + i, lda, 1.0, r, 1);
    cm_gemv(true, i, nr, -1.0, A + (i + 1) * lda, lda, X + i, ldx, 1.0, r, 1);
    larfg(nr, r[0], r + 1, 1, taup[i]);
    e[i] = r[0];
    r[0] = 1.0;
    for (I64 c = 0; c < nr; ++c) row[c * lda] = r[c];
    double* xi = X + i * ldx;  // X[:, i]
    cm_gemv(false, m - i - 1, nr, 1.0, A + (i + 1) + (i + 1) * lda, lda, r, 1, 0.0, xi + i + 1, 1);
    cm_gemv(true, nr, i + 1, 1.0, Y + i + 1, ldy, r, 1, 0.0, xi, 1);
    cm_gemv(false, m - i - 1, i + 1, -1.0, A + i + 1, lda, xi, 1, 1.0, xi + i + 1, 1);
    cm_gemv(false, i, nr, 1.0, A + (i + 1) * lda, lda, r, 1, 0.0, xi, 1);
    cm_gemv(false, m - i - 1, i, -1.0, X + i + 1, ldx, xi, 1, 1.0, xi + i + 1, 1);
    blas::scal(m - i - 1, taup[i], xi + i + 1, 1);
  }
}

/// LAPACK dgebrd (m >= n): A [m, n] = Q B P^T with B upper bidiagonal (d [n], e [n - 1]),
/// Q = H_0 ... H_{n-1} (reflector i below A[i, i]) and P = G_0 ... G_{n-2} (reflector i right of
/// A[i, i + 1]).
void gebrd(I64 m, I64 n, double* A, I64 lda, double* d, double* e, double* tauq, double* taup) {
  Vec X(sz(m * kBlock)), Y(sz(n * kBlock)), r(sz(n));
  for (I64 i = 0; i < n; i += kBlock) {
    const I64 kb = std::min(kBlock, n - i), mr = m - i, nr = n - i;
    labrd(mr, nr, kb, A + i + i * lda, lda, d + i, e + i, tauq + i, taup + i, X.data(), mr, Y.data(), nr, r.data());
    if (kb < nr) {
      double* C = A + (i + kb) + (i + kb) * lda;
      cm_gemm(false, true, mr - kb, nr - kb, kb, -1.0, A + (i + kb) + i * lda, lda, Y.data() + kb, nr, 1.0, C, lda);
      cm_gemm(false, false, mr - kb, nr - kb, kb, -1.0, X.data() + kb, mr, A + i + (i + kb) * lda, lda, 1.0, C, lda);
    }
    for (I64 j = i; j < i + kb; ++j) {
      A[j + j * lda] = d[j];
      if (j + 1 < n) A[j + (j + 1) * lda] = e[j];
    }
  }
}

// ---- bidiagonal SVD -----------------------------------------------------------------------------

/// Fills the columns of Q [rows, cols] (column-major) flagged in missing with unit vectors
/// orthogonal to all other columns (Gram-Schmidt, twice, on the best standard basis vector).
void complete_columns(double* Q, I64 ldq, I64 rows, I64 cols, std::vector<char> missing) {
  Vec cand(sz(rows)), best(sz(rows));
  for (I64 c = 0; c < cols; ++c) {
    if (!missing[sz(c)]) continue;
    double best_norm = -1.0;
    for (I64 e = 0; e < rows; ++e) {
      std::fill(cand.begin(), cand.end(), 0.0);
      cand[sz(e)] = 1.0;
      for (int pass = 0; pass < 2; ++pass)
        for (I64 o = 0; o < cols; ++o) {
          if (missing[sz(o)]) continue;
          const double* qo  = Q + o * ldq;
          double        dot = 0.0;
          for (I64 i = 0; i < rows; ++i) dot += qo[i] * cand[sz(i)];
          for (I64 i = 0; i < rows; ++i) cand[sz(i)] -= dot * qo[i];
        }
      double nrm = 0.0;
      for (const double v : cand) nrm += v * v;
      if (nrm > best_norm) best_norm = nrm, best = cand;
      if (nrm > 0.5) break;
    }
    const double inv = 1.0 / std::sqrt(best_norm);
    for (I64 i = 0; i < rows; ++i) Q[i + c * ldq] = best[sz(i)] * inv;
    missing[sz(c)] = 0;
  }
}

/// SVD of a small upper bidiagonal block B [nr, nc = nr + sqre] (d [nr], e [nc - 1]) by one-sided
/// Jacobi on B^T: d becomes the ascending singular values, U [nr, nr] and V [nc, nc] the vectors
/// (B = U [diag(d) 0] V^T); with sqre, V's last column spans B's null space.
void bd_leaf(I64 nr, int sqre, double* d, const double* e, double* U, I64 ldu, double* V, I64 ldv) {
  const I64 nc = nr + sqre;
  Vec       G(sz(nc * nr), 0.0), W(sz(nr * nr), 0.0);  // G = B^T: column r is row r of B
  for (I64 r = 0; r < nr; ++r) {
    G[sz(r + r * nc)] = d[r];
    if (r + 1 < nc) G[sz(r + 1 + r * nc)] = e[r];
    W[sz(r + r * nr)] = 1.0;
  }
  for (int sweep = 0; sweep < 80; ++sweep) {
    bool rotated = false;
    for (I64 p = 0; p + 1 < nr; ++p)
      for (I64 q = p + 1; q < nr; ++q) {
        double* gp    = G.data() + p * nc;
        double* gq    = G.data() + q * nc;
        double  alpha = 0.0, beta = 0.0, gamma = 0.0;
        for (I64 i = 0; i < nc; ++i) alpha += gp[i] * gp[i], beta += gq[i] * gq[i], gamma += gp[i] * gq[i];
        if (gamma == 0.0 || std::abs(gamma) <= 1e-15 * std::sqrt(alpha * beta)) continue;
        rotated           = true;
        const double zeta = (beta - alpha) / (2.0 * gamma);
        const double t    = (zeta >= 0 ? 1.0 : -1.0) / (std::abs(zeta) + std::hypot(1.0, zeta));
        const double c = 1.0 / std::sqrt(1.0 + t * t), s = c * t;
        rot(nc, gp, gq, c, -s);
        rot(nr, W.data() + p * nr, W.data() + q * nr, c, -s);
      }
    if (!rotated) break;
  }
  Vec sv(sz(nr));
  for (I64 r = 0; r < nr; ++r) sv[sz(r)] = nrm2(nc, G.data() + r * nc, 1);
  std::vector<I64> order(sz(nr));
  std::iota(order.begin(), order.end(), I64{0});
  std::stable_sort(order.begin(), order.end(), [&](I64 x, I64 y) { return sv[sz(x)] < sv[sz(y)]; });
  std::vector<char> missing(sz(nc), 0);
  for (I64 c = 0; c < nr; ++c) {
    const I64 r = order[sz(c)];
    d[c]        = sv[sz(r)];
    std::copy(W.data() + r * nr, W.data() + (r + 1) * nr, U + c * ldu);
    if (sv[sz(r)] > std::numeric_limits<double>::min()) {
      for (I64 i = 0; i < nc; ++i) V[i + c * ldv] = G[sz(i + r * nc)] / sv[sz(r)];
    } else {
      missing[sz(c)] = 1;
    }
  }
  if (sqre) missing[sz(nr)] = 1;
  complete_columns(V, ldv, nc, nc, missing);
}

/// Root j of the bidiagonal secular equation 1 + sum_i z_i^2 / (D_i^2 - s^2) (D ascending from
/// D_0 = 0, distinct; z nonzero), solved for nu = s^2 - D_o^2 with the poles shifted to the nearer
/// D_o (computed as products of sums and differences). Returns s with dm[i] = D_i - s and
/// dp[i] = D_i + s; P is scratch [K].
double svd_secular(I64 K, I64 j, const double* D, const double* z, double* dm, double* dp, double* P) {
  if (K == 1) {
    const double s = std::hypot(D[0], z[0]);
    dm[0] = D[0] - s, dp[0] = D[0] + s;
    return s;
  }
  I64    o;
  double lo, hi;
  if (j + 1 < K) {
    const double half = 0.5 * (D[j + 1] - D[j]) * (D[j + 1] + D[j]);
    double       f    = 1.0;
    for (I64 i = 0; i < K; ++i) f += z[i] * z[i] / ((D[i] - D[j]) * (D[i] + D[j]) - half);
    if (f >= 0.0)
      o = j, lo = 0.0, hi = half;
    else
      o = j + 1, lo = -half, hi = 0.0;
  } else {
    double ss = 0.0;
    for (I64 i = 0; i < K; ++i) ss += z[i] * z[i];
    o = K - 1, lo = 0.0, hi = ss;
  }
  for (I64 i = 0; i < K; ++i) P[i] = (D[i] - D[o]) * (D[i] + D[o]);
  const double nu  = secular_root(K, j, P, z, 1.0, lo, hi);
  const double tau = nu / (D[o] + std::sqrt(D[o] * D[o] + nu));  // s = D_o + tau
  for (I64 i = 0; i < K; ++i) {
    dm[i] = (D[i] - D[o]) - tau;
    dp[i] = (D[i] + D[o]) + tau;
  }
  return D[o] + tau;
}

/// Out [rows, K] (ld rows) = Q[:, cols] W with W [K, K] (ld K), where each selected column of Q
/// (ld ldq) is nonzero in rows [0, split), in rows [split, rows) or in both. The columns are
/// grouped that way (LAPACK dlasd3's column types) and the two row blocks multiplied separately,
/// which skips the zero quarters of the block-diagonal Q: about half the flops of one GEMM.
void block_product(I64 rows, I64 split, const double* Q, I64 ldq, const std::vector<I64>& cols, const double* W,
                   I64 K, double* Out) {
  std::vector<I64> group[3];  // top only (or zero), both, bottom only
  for (I64 i = 0; i < K; ++i) {
    const double* q   = Q + cols[sz(i)] * ldq;
    const auto    nz  = [](double x) { return x != 0.0; };
    const bool    top = std::any_of(q, q + split, nz), bot = std::any_of(q + split, q + rows, nz);
    group[bot ? (top ? 1 : 2) : 0].push_back(i);
  }
  std::vector<I64> perm;
  perm.reserve(sz(K));
  for (const auto& g : group) perm.insert(perm.end(), g.begin(), g.end());
  Vec Qp(sz(rows * K)), Wp(sz(K * K));
  split_for(K, rows + K, [&](I64 lo, I64 hi) {
    for (I64 j = lo; j < hi; ++j) {
      const double* q = Q + cols[sz(perm[sz(j)])] * ldq;
      std::copy(q, q + rows, Qp.data() + j * rows);
      for (I64 p = 0; p < K; ++p) Wp[sz(p + j * K)] = W[sz(perm[sz(p)] + j * K)];
    }
  });
  const I64 n0 = static_cast<I64>(group[0].size()), n1 = static_cast<I64>(group[1].size());
  cm_gemm(false, false, split, K, n0 + n1, 1.0, Qp.data(), rows, Wp.data(), K, 0.0, Out, rows);
  cm_gemm(false, false, rows - split, K, K - n0, 1.0, Qp.data() + n0 * rows + split, rows, Wp.data() + n0, K, 0.0,
          Out + split, rows);
}

/// Merge step of the bidiagonal divide and conquer (LAPACK dlasd1-3). B [nr, nc = nr + sqre] is
/// [B1 0; alpha e_k^T beta e_{k+1}^T; 0 B2] with B1 [k, k + 1] and B2 [nr - k - 1, nc - k - 1],
/// whose SVDs sit in U = diag(U1, 1, U2) and V = diag(V1, V2) (null vectors last in each block).
/// In those bases B is M = e_0 z^T + diag(D) with D_0 = 0 (coordinate 0 is row k and B1's null
/// vector, then B1's and B2's singular pairs) plus, with sqre, a column holding only B2's null
/// vector's z, rotated into coordinate 0. On exit d holds the ascending singular values of B and
/// U, V its vectors.
void bd_merge(I64 nr, int sqre, I64 k, double* d, double* U, I64 ldu, double* V, I64 ldv, double alpha,
              double beta) {
  const I64  nc = nr + sqre, n2 = nr - k - 1;
  const auto col = [k](I64 c) { return c == 0 ? k : (c <= k ? c - 1 : c); };  // coordinate -> stored column
  Vec        D(sz(nr)), z(sz(nr));
  D[0] = 0.0;
  z[0] = alpha * V[k + k * ldv];
  for (I64 a = 0; a < k; ++a) D[sz(1 + a)] = d[a], z[sz(1 + a)] = alpha * V[k + a * ldv];
  for (I64 b = 0; b < n2; ++b)
    D[sz(k + 1 + b)] = d[k + 1 + b], z[sz(k + 1 + b)] = beta * V[(k + 1) + (k + 1 + b) * ldv];
  if (sqre) {
    const double znull = beta * V[(k + 1) + (nc - 1) * ldv], r = std::hypot(z[0], znull);
    if (r > 0.0) {
      rot(nc, V + k * ldv, V + (nc - 1) * ldv, z[0] / r, znull / r);
      z[0] = r;
    }
  }
  double dmax = 0.0;
  for (I64 c = 1; c < nr; ++c) dmax = std::max(dmax, D[sz(c)]);
  double tol = 64.0 * kEps * std::max({std::abs(alpha), std::abs(beta), dmax});
  if (tol == 0.0) tol = 64.0 * kEps;
  if (std::abs(z[0]) <= tol) z[0] = tol;

  // Deflation (dlasd2): tiny z components; singular values next to D_0 = 0 rotated into
  // coordinate 0; pairs of close singular values rotated so one z vanishes.
  std::vector<I64> order(sz(nr - 1));
  std::iota(order.begin(), order.end(), I64{1});
  std::inplace_merge(order.begin(), order.begin() + k, order.end(), [&](I64 x, I64 y) { return D[sz(x)] < D[sz(y)]; });
  std::vector<I64> keep{0}, defl;
  Vec              dval(sz(nr), 0.0);
  I64              pending = -1;
  for (const I64 c : order) {
    if (std::abs(z[sz(c)]) <= tol) {
      defl.push_back(c);
      dval[sz(c)] = D[sz(c)];
      continue;
    }
    if (D[sz(c)] <= tol) {
      const double r = std::hypot(z[0], z[sz(c)]), cs = z[0] / r, sn = z[sz(c)] / r;
      rot(nc, V + k * ldv, V + col(c) * ldv, cs, sn);
      z[0]        = r;
      z[sz(c)]    = 0.0;
      dval[sz(c)] = std::abs(cs) * D[sz(c)];
      if (cs < 0.0)
        for (I64 i = 0; i < nc; ++i) V[i + col(c) * ldv] = -V[i + col(c) * ldv];
      defl.push_back(c);
      continue;
    }
    if (pending >= 0 && D[sz(c)] - D[sz(pending)] <= tol) {
      const double r = std::hypot(z[sz(pending)], z[sz(c)]), cs = z[sz(c)] / r, sn = z[sz(pending)] / r;
      // new pending = cs pending - sn c, new c = sn pending + cs c (on both sides)
      rot(nr, U + col(pending) * ldu, U + col(c) * ldu, cs, -sn);
      rot(nc, V + col(pending) * ldv, V + col(c) * ldv, cs, -sn);
      z[sz(pending)]    = 0.0;
      z[sz(c)]          = r;
      const double dpv = D[sz(pending)], dcv = D[sz(c)];
      dval[sz(pending)] = cs * cs * dpv + sn * sn * dcv;
      D[sz(c)]          = sn * sn * dpv + cs * cs * dcv;
      defl.push_back(pending);
      pending = c;
      continue;
    }
    if (pending >= 0) keep.push_back(pending);
    pending = c;
  }
  if (pending >= 0) keep.push_back(pending);

  const I64 K = static_cast<I64>(keep.size());
  Vec       Dk(sz(K)), zk(sz(K)), sig(sz(K)), dm(sz(K * K)), dpl(sz(K * K));
  for (I64 i = 0; i < K; ++i) Dk[sz(i)] = D[sz(keep[sz(i)])], zk[sz(i)] = z[sz(keep[sz(i)])];
  parallel_for(0, K, std::max<I64>(1, 4096 / K), [&](I64 lo, I64 hi) {
    Vec P(sz(K));
    for (I64 j = lo; j < hi; ++j)
      sig[sz(j)] = svd_secular(K, j, Dk.data(), zk.data(), dm.data() + j * K, dpl.data() + j * K, P.data());
  });
  const auto delta = [&](I64 i, I64 j) { return dm[sz(i + j * K)] * dpl[sz(i + j * K)]; };  // D_i^2 - s_j^2
  Vec        zh(sz(K));
  parallel_for(0, K, std::max<I64>(1, 4096 / K), [&](I64 lo, I64 hi) {
    for (I64 i = lo; i < hi; ++i) {
      double p = -delta(i, i);
      for (I64 j = 0; j < K; ++j)
        if (j != i) p *= delta(i, j) / ((Dk[sz(i)] - Dk[sz(j)]) * (Dk[sz(i)] + Dk[sz(j)]));
      zh[sz(i)] = std::copysign(std::sqrt(p), zk[sz(i)]);
    }
  });
  Vec Uu(sz(K * K)), Vv(sz(K * K));
  parallel_for(0, K, std::max<I64>(1, 4096 / K), [&](I64 lo, I64 hi) {
    for (I64 j = lo; j < hi; ++j) {
      double* uj = Uu.data() + j * K;
      double* vj = Vv.data() + j * K;
      double  nu = 0.0, nv = 0.0;
      for (I64 i = 0; i < K; ++i) {
        const double v = zh[sz(i)] / delta(i, j);
        vj[i]          = v;
        uj[i]          = i == 0 ? -1.0 : Dk[sz(i)] * v;
        nu += uj[i] * uj[i];
        nv += v * v;
      }
      const double iu = 1.0 / std::sqrt(nu), iv = 1.0 / std::sqrt(nv);
      for (I64 i = 0; i < K; ++i) uj[i] *= iu, vj[i] *= iv;
    }
  });
  std::vector<I64> kept_cols(sz(K));
  for (I64 i = 0; i < K; ++i) kept_cols[sz(i)] = col(keep[sz(i)]);
  Vec Unew(sz(nr * K)), Vnew(sz(nc * K));
  block_product(nr, k + 1, U, ldu, kept_cols, Uu.data(), K, Unew.data());
  block_product(nc, k + 1, V, ldv, kept_cols, Vv.data(), K, Vnew.data());

  std::vector<std::pair<double, I64>> items;  // index < K: new pair; else deflated coordinate (index - K)
  items.reserve(sz(nr));
  for (I64 j = 0; j < K; ++j) items.emplace_back(sig[sz(j)], j);
  for (const I64 c : defl) items.emplace_back(dval[sz(c)], K + c);
  std::stable_sort(items.begin(), items.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
  Vec Uo(sz(nr * nr)), Vo(sz(nc * nc));
  split_for(nr, nr + nc, [&](I64 lo, I64 hi) {
    for (I64 t = lo; t < hi; ++t) {
      const I64     src = items[sz(t)].second;
      const double* uc  = src < K ? Unew.data() + src * nr : U + col(src - K) * ldu;
      const double* vc  = src < K ? Vnew.data() + src * nc : V + col(src - K) * ldv;
      std::copy(uc, uc + nr, Uo.data() + t * nr);
      std::copy(vc, vc + nc, Vo.data() + t * nc);
    }
  });
  if (sqre) std::copy(V + (nc - 1) * ldv, V + (nc - 1) * ldv + nc, Vo.data() + (nc - 1) * nc);
  for (I64 t = 0; t < nr; ++t) d[t] = items[sz(t)].first;
  split_for(nc, nr + nc, [&](I64 lo, I64 hi) {
    for (I64 t = lo; t < hi; ++t) {
      if (t < nr) std::copy(Uo.data() + t * nr, Uo.data() + (t + 1) * nr, U + t * ldu);
      std::copy(Vo.data() + t * nc, Vo.data() + (t + 1) * nc, V + t * ldv);
    }
  });
}

/// Splits B [nr, nr + sqre] down to subtrees of at most limit rows (as bd_rec), listing them and
/// the merges above them in post-order.
void bd_plan(I64 off, I64 nr, int sqre, I64 limit, std::vector<DcNode>& subtrees, std::vector<DcNode>& merges) {
  if (nr <= limit) {
    subtrees.push_back({off, nr, sqre, 0.0});
    return;
  }
  const I64 k = nr / 2;
  bd_plan(off, k, 1, limit, subtrees, merges);
  bd_plan(off + k + 1, nr - k - 1, sqre, limit, subtrees, merges);
  merges.push_back({off, nr, sqre, 0.0});
}

/// Merges the solved blocks B1 = rows [0, k) and B2 = rows (k, nr), k = nr / 2, of bd_rec.
void bd_join(I64 nr, int sqre, double* d, const double* e, double* U, I64 ldu, double* V, I64 ldv) {
  const I64    nc = nr + sqre, k = nr / 2;
  const double alpha = d[k], beta = e[k];
  const auto ublock = [k](I64 i) { return i < k ? 0 : (i == k ? 1 : 2); };
  split_for(nc, nr + nc, [&](I64 lo, I64 hi) {
    for (I64 c = lo; c < hi; ++c) {
      if (c < nr)
        for (I64 r = 0; r < nr; ++r)
          if (ublock(r) != ublock(c)) U[r + c * ldu] = 0.0;
      for (I64 r = 0; r < nc; ++r)
        if ((r <= k) != (c <= k)) V[r + c * ldv] = 0.0;
    }
  });
  U[k + k * ldu] = 1.0;
  bd_merge(nr, sqre, k, d, U, ldu, V, ldv, alpha, beta);
}

/// Bidiagonal divide and conquer on B [nr, nr + sqre] (d [nr], e [nr - 1 + sqre]): d becomes the
/// ascending singular values, U [nr, nr] and V [nc, nc] the vectors. Row k = nr / 2 joins
/// B1 = rows [0, k) (square plus one column) and B2 = rows (k, nr).
void bd_rec(I64 nr, int sqre, double* d, double* e, double* U, I64 ldu, double* V, I64 ldv) {
  if (nr <= kSvdLeaf) {
    bd_leaf(nr, sqre, d, e, U, ldu, V, ldv);
    return;
  }
  const I64 k = nr / 2;
  bd_rec(k, 1, d, e, U, ldu, V, ldv);
  bd_rec(nr - k - 1, sqre, d + k + 1, e + k + 1, U + (k + 1) + (k + 1) * ldu, ldu, V + (k + 1) + (k + 1) * ldv, ldv);
  bd_join(nr, sqre, d, e, U, ldu, V, ldv);
}

/// SVD of the upper bidiagonal B [n, n] (d [n], e [n - 1]): d ascending; U, V [n, n] (ld n).
void bdsdc(I64 n, double* d, const double* e_in, double* U, double* V) {
  Vec e(sz(n), 0.0);
  std::copy(e_in, e_in + std::max<I64>(n - 1, 0), e.begin());
  double scale = 0.0;
  for (I64 i = 0; i < n; ++i) scale = std::max({scale, std::abs(d[i]), std::abs(e[sz(i)])});
  if (scale == 0.0) {
    for (I64 c = 0; c < n; ++c)
      for (I64 r = 0; r < n; ++r) U[r + c * n] = V[r + c * n] = r == c ? 1.0 : 0.0;
    return;
  }
  for (I64 i = 0; i < n; ++i) d[i] /= scale, e[sz(i)] /= scale;
  // As stedc: the subtrees on the pool, then the merges above them.
  std::vector<DcNode> subtrees, merges;
  bd_plan(0, n, 0, subtree_limit(n, kSvdLeaf), subtrees, merges);
  parallel_for(0, static_cast<I64>(subtrees.size()), 1, [&](I64 lo, I64 hi) {
    for (I64 t = lo; t < hi; ++t) {
      const DcNode& s = subtrees[sz(t)];
      bd_rec(s.n, s.sqre, d + s.off, e.data() + s.off, U + s.off * (1 + n), n, V + s.off * (1 + n), n);
    }
  });
  for (const DcNode& mg : merges)
    bd_join(mg.n, mg.sqre, d + mg.off, e.data() + mg.off, U + mg.off * (1 + n), n, V + mg.off * (1 + n), n);
  for (I64 i = 0; i < n; ++i) d[i] *= scale;
}

/// Singular values only of the upper bidiagonal (d [n], e [n - 1]), descending into s: the
/// non-negative eigenvalues of the Golub-Kahan tridiagonal (zero diagonal, off-diagonal
/// d_0, e_0, d_1, ..., d_{n-1}), by implicit QL. The fallback of bidiagonal_values.
void bidiagonal_values_ql(I64 n, const double* d, const double* e, double* s) {
  double scale = 0.0;
  for (I64 i = 0; i < n; ++i) scale = std::max(scale, std::abs(d[i]));
  for (I64 i = 0; i + 1 < n; ++i) scale = std::max(scale, std::abs(e[i]));
  if (scale == 0.0) {
    std::fill(s, s + n, 0.0);
    return;
  }
  Vec td(sz(2 * n), 0.0), te(sz(2 * n), 0.0);
  for (I64 i = 0; i < n; ++i) {
    te[sz(2 * i)] = d[i] / scale;
    if (i + 1 < n) te[sz(2 * i + 1)] = e[i] / scale;
  }
  tql(2 * n, td.data(), te.data(), nullptr, 0, 0);
  std::sort(td.begin(), td.end(), std::greater<>());
  for (I64 i = 0; i < n; ++i) s[i] = std::abs(td[sz(i)]) * scale;
}

// ---- bidiagonal singular values: dqds -----------------------------------------------------------
//
// LAPACK's dlasq1-6 (Fernando, Parlett, Marques): the squared singular values are the eigenvalues
// of B^T B, which the differential qd algorithm with shifts finds from q_k = d_k^2, e_k = e_k^2
// without forming the product, to high relative accuracy. One division per entry and step on n
// entries, where QL on the Golub-Kahan matrix takes a square root and a division on 2n.
//
// The qd array is indexed from one as in LAPACK: z(4k-3) = q_k, z(4k-1) = e_k ("ping") and
// z(4k-2), z(4k) the transformed values ("pong"); pp = 0 reads ping and writes pong, pp = 1 the
// reverse. A non-positive e marks a split; it holds minus the shift accumulated when it was set.

/// LAPACK dlamch('P'): eps * base = 2^-52.
constexpr double kQdEps = std::numeric_limits<double>::epsilon();

struct Qd {
  double* p;
  double& operator()(I64 i) const { return p[i - 1]; }
};

/// min that keeps a NaN in b: a failed transform is detected from dmin.
double qd_min(double a, double b) { return (b < a || std::isnan(b)) ? b : a; }

/// The shift bookkeeping dlasq3-5 pass around: the smallest d of the last transform (dmin),
/// without its last one or two entries (dmin1, dmin2), its last three d (dn, dn1, dn2), the shift
/// tau, its kind ttype and dlasq4's adaptive fraction g.
struct QdState {
  double dmin = 0, dmin1 = 0, dmin2 = 0, dn = 0, dn1 = 0, dn2 = 0, g = 0, tau = 0;
  int    ttype = 0;
};

/// LAPACK dlasq4: the shift for the next transform of rows i0..n0. n0in is n0 before this
/// step's deflations. (LAPACK's early exits keep the previous shift; these take the cautious
/// one set just before them.)
void lasq4(I64 i0, I64 n0, Qd z, int pp, I64 n0in, QdState& st) {
  constexpr double cnst1 = 0.563, cnst2 = 1.01, cnst3 = 1.05, qurtr = 0.25, third = 0.333, half = 0.5;
  const double     dmin = st.dmin, dmin1 = st.dmin1, dmin2 = st.dmin2, dn = st.dn, dn1 = st.dn1, dn2 = st.dn2;
  if (dmin <= 0.0) {
    st.tau   = -dmin;
    st.ttype = -1;
    return;
  }
  const I64 nn = 4 * n0 + pp;
  double    s = 0.0, a2 = 0.0, b1 = 0.0, b2 = 0.0, gam = 0.0;
  // Sum of the ratios z(i4) / z(i4 - 2) products down to row i0 (a norm estimate); false when a
  // ratio exceeds one (no estimate).
  const auto tail = [&](I64 from) {
    for (I64 i4 = from; i4 >= 4 * i0 - 1 + pp; i4 -= 4) {
      if (b2 == 0.0) break;
      b1 = b2;
      if (z(i4) > z(i4 - 2)) return false;
      b2 *= z(i4) / z(i4 - 2);
      a2 += b2;
      if (100.0 * std::max(b2, b1) < a2 || cnst1 < a2) break;
    }
    return true;
  };
  if (n0in == n0) {  // no values deflated
    if (dmin == dn || dmin == dn1) {
      b1 = std::sqrt(z(nn - 3)) * std::sqrt(z(nn - 5));
      b2 = std::sqrt(z(nn - 7)) * std::sqrt(z(nn - 9));
      a2 = z(nn - 7) + z(nn - 5);
      if (dmin == dn && dmin1 == dn1) {  // cases 2 and 3
        const double gap2 = dmin2 - a2 - dmin2 * qurtr;
        const double gap1 = (gap2 > 0.0 && gap2 > b2) ? a2 - dn - (b2 / gap2) * b2 : a2 - dn - (b1 + b2);
        if (gap1 > 0.0 && gap1 > b1) {
          s        = std::max(dn - (b1 / gap1) * b1, half * dmin);
          st.ttype = -2;
        } else {
          s = 0.0;
          if (dn > b1) s = dn - b1;
          if (a2 > b1 + b2) s = std::min(s, a2 - (b1 + b2));
          s        = std::max(s, third * dmin);
          st.ttype = -3;
        }
      } else {  // case 4
        st.ttype = -4;
        s        = qurtr * dmin;
        I64 np;
        bool ok = true;
        if (dmin == dn) {
          gam = dn;
          a2  = 0.0;
          if (z(nn - 5) > z(nn - 7)) ok = false;
          else b2 = z(nn - 5) / z(nn - 7);
          np = nn - 9;
        } else {
          np  = nn - 2 * pp;
          gam = dn1;
          if (z(np - 4) > z(np - 2) || z(nn - 9) > z(nn - 11)) {
            ok = false;
          } else {
            a2 = z(np - 4) / z(np - 2);
            b2 = z(nn - 9) / z(nn - 11);
          }
          np = nn - 13;
        }
        if (ok) {
          a2 += b2;
          if (tail(np)) {
            a2 *= cnst3;
            if (a2 < cnst1) s = gam * (1.0 - std::sqrt(a2)) / (1.0 + a2);  // Rayleigh quotient residual bound
          }
        }
      }
    } else if (dmin == dn2) {  // case 5
      st.ttype     = -5;
      s            = qurtr * dmin;
      const I64 np = nn - 2 * pp;
      b1           = z(np - 2);
      b2           = z(np - 6);
      gam          = dn2;
      if (!(z(np - 8) > b2 || z(np - 4) > b1)) {
        a2      = (z(np - 8) / b2) * (1.0 + z(np - 4) / b1);
        bool ok = true;
        if (n0 - i0 > 2) {
          b2 = z(nn - 13) / z(nn - 15);
          a2 += b2;
          ok = tail(nn - 17);
          if (ok) a2 *= cnst3;
        }
        if (ok && a2 < cnst1) s = gam * (1.0 - std::sqrt(a2)) / (1.0 + a2);
      }
    } else {  // case 6: no information to guide the shift
      if (st.ttype == -6)
        st.g += third * (1.0 - st.g);
      else if (st.ttype == -18)
        st.g = qurtr * third;
      else
        st.g = qurtr;
      s        = st.g * dmin;
      st.ttype = -6;
    }
  } else if (n0in == n0 + 1) {  // one value just deflated: use dmin1, dn1
    if (dmin1 == dn1 && dmin2 == dn2) {  // cases 7 and 8
      st.ttype = -7;
      s        = third * dmin1;
      if (!(z(nn - 5) > z(nn - 7))) {
        b1      = z(nn - 5) / z(nn - 7);
        b2      = b1;
        bool ok = true;
        if (b2 != 0.0) {
          for (I64 i4 = 4 * n0 - 9 + pp; i4 >= 4 * i0 - 1 + pp; i4 -= 4) {
            a2 = b1;
            if (z(i4) > z(i4 - 2)) {
              ok = false;
              break;
            }
            b1 *= z(i4) / z(i4 - 2);
            b2 += b1;
            if (100.0 * std::max(b1, a2) < b2) break;
          }
        }
        if (ok) {
          b2                = std::sqrt(cnst3 * b2);
          a2                = dmin1 / (1.0 + b2 * b2);
          const double gap2 = half * dmin2 - a2;
          if (gap2 > 0.0 && gap2 > b2 * a2) {
            s = std::max(s, a2 * (1.0 - cnst2 * a2 * (b2 / gap2) * b2));
          } else {
            s        = std::max(s, a2 * (1.0 - cnst2 * b2));
            st.ttype = -8;
          }
        }
      }
    } else {  // case 9
      s = dmin1 == dn1 ? half * dmin1 : qurtr * dmin1;
      st.ttype = -9;
    }
  } else if (n0in == n0 + 2) {  // two values deflated: use dmin2, dn2
    if (dmin2 == dn2 && 2.0 * z(nn - 5) < z(nn - 7)) {  // case 10
      st.ttype = -10;
      s        = third * dmin2;
      if (!(z(nn - 5) > z(nn - 7))) {
        b1      = z(nn - 5) / z(nn - 7);
        b2      = b1;
        bool ok = true;
        if (b2 != 0.0) {
          for (I64 i4 = 4 * n0 - 9 + pp; i4 >= 4 * i0 - 1 + pp; i4 -= 4) {
            if (z(i4) > z(i4 - 2)) {
              ok = false;
              break;
            }
            b1 *= z(i4) / z(i4 - 2);
            b2 += b1;
            if (100.0 * b1 < b2) break;
          }
        }
        if (ok) {
          b2                = std::sqrt(cnst3 * b2);
          a2                = dmin2 / (1.0 + b2 * b2);
          const double gap2 = z(nn - 7) + z(nn - 9) - std::sqrt(z(nn - 11)) * std::sqrt(z(nn - 9)) - a2;
          if (gap2 > 0.0 && gap2 > b2 * a2)
            s = std::max(s, a2 * (1.0 - cnst2 * a2 * (b2 / gap2) * b2));
          else
            s = std::max(s, a2 * (1.0 - cnst2 * b2));
        }
      }
    } else {  // case 11
      s        = qurtr * dmin2;
      st.ttype = -11;
    }
  } else {  // case 12: more than two deflated, no information
    s        = 0.0;
    st.ttype = -12;
  }
  st.tau = s;
}

/// Positions of row j's entries: read (in) and written (out) by a transform from side pp.
struct QdSide {
  int pp;
  I64 qin(I64 j) const { return 4 * j - 3 + pp; }
  I64 ein(I64 j) const { return 4 * j - 1 + pp; }
  I64 qout(I64 j) const { return 4 * j - 2 - pp; }
  I64 eout(I64 j) const { return 4 * j - pp; }
};

/// dlasq5's main loop over rows j0..j1 from side PP; returns the last d. Specialised so that the
/// chain from one d to the next is just an add, a division and a multiply-add; the minima are
/// kept in registers and branch-free, with a NaN in d carried by a flag.
template <int PP, bool Clamp>
double qd_sweep(Qd z, I64 j0, I64 j1, double d, double tau, double dthresh, double& dmin_io, double& emin_io) {
  double  dmin = dmin_io, emin = emin_io;
  bool    nan  = false;
  double* p    = &z(4 * j0);  // p[k] = z(4j + k) for the current row j
  for (I64 j = j0; j <= j1; ++j, p += 4) {
    const double e    = p[PP - 1];
    const double qq   = d + e;
    p[-2 - PP]        = qq;
    const double temp = p[1 + PP] / qq;
    d                 = d * temp - tau;
    if constexpr (Clamp) d = d < dthresh ? 0.0 : d;
    nan |= d != d;
    dmin            = d < dmin ? d : dmin;
    const double ee = e * temp;
    p[-PP]          = ee;
    emin            = ee < emin ? ee : emin;
  }
  dmin_io = nan ? std::numeric_limits<double>::quiet_NaN() : dmin;
  emin_io = emin;
  return d;
}

/// LAPACK dlasq5 (IEEE version): one dqds transform of rows i0..n0 with shift tau (zeroed when
/// below eps * (sigma + tau); then d values below that threshold are set to zero).
void lasq5(I64 i0, I64 n0, Qd z, int pp, double sigma, QdState& st) {
  if (n0 - i0 - 1 <= 0) return;
  const QdSide s{pp};
  const double dthresh = kQdEps * (sigma + st.tau);
  if (st.tau < dthresh * 0.5) st.tau = 0.0;
  const double tau  = st.tau;
  double       emin = z(s.qin(i0) + 4);
  double       d    = z(s.qin(i0)) - tau;
  double       dmin = d;
  if (n0 - 3 >= i0) {
    if (tau == 0.0)
      d = pp == 0 ? qd_sweep<0, true>(z, i0, n0 - 3, d, tau, dthresh, dmin, emin)
                  : qd_sweep<1, true>(z, i0, n0 - 3, d, tau, dthresh, dmin, emin);
    else
      d = pp == 0 ? qd_sweep<0, false>(z, i0, n0 - 3, d, tau, dthresh, dmin, emin)
                  : qd_sweep<1, false>(z, i0, n0 - 3, d, tau, dthresh, dmin, emin);
  }
  // The last two steps unrolled, keeping the last three d.
  const double dnm2 = d;
  st.dmin2          = dmin;
  I64 j             = n0 - 2;
  z(s.qout(j))      = dnm2 + z(s.ein(j));
  z(s.eout(j))      = z(s.qin(j + 1)) * (z(s.ein(j)) / z(s.qout(j)));
  const double dnm1 = z(s.qin(j + 1)) * (dnm2 / z(s.qout(j))) - tau;
  dmin              = qd_min(dmin, dnm1);
  st.dmin1          = dmin;
  j                 = n0 - 1;
  z(s.qout(j))      = dnm1 + z(s.ein(j));
  z(s.eout(j))      = z(s.qin(j + 1)) * (z(s.ein(j)) / z(s.qout(j)));
  const double dn   = z(s.qin(j + 1)) * (dnm1 / z(s.qout(j))) - tau;
  dmin              = qd_min(dmin, dn);
  z(s.qout(n0))     = dn;
  z(s.eout(n0))     = emin;
  st.dn = dn, st.dn1 = dnm1, st.dn2 = dnm2, st.dmin = dmin;
}

/// LAPACK dlasq6: one unshifted transform (dqd) of rows i0..n0 guarding against underflow.
void lasq6(I64 i0, I64 n0, Qd z, int pp, QdState& st) {
  if (n0 - i0 - 1 <= 0) return;
  const QdSide s{pp};
  const double safmin = std::numeric_limits<double>::min();
  double       emin   = z(s.qin(i0) + 4);
  double       d      = z(s.qin(i0));
  double       dmin   = d;
  // Row j's step from d; plain (temp) form while the quotient is safe, else divide first.
  const auto step = [&](I64 j, double dcur, bool loop_form) {
    const I64 qo = s.qout(j), eo = s.eout(j), ei = s.ein(j), qn = s.qin(j + 1);
    z(qo)        = dcur + z(ei);
    double dnew;
    if (z(qo) == 0.0) {
      z(eo) = 0.0;
      dnew  = z(qn);
      dmin  = dnew;
      emin  = 0.0;
    } else if (safmin * z(qn) < z(qo) && safmin * z(qo) < z(qn)) {
      const double temp = z(qn) / z(qo);
      z(eo)             = z(ei) * temp;
      dnew              = dcur * temp;
    } else {
      z(eo) = z(qn) * (z(ei) / z(qo));
      dnew  = z(qn) * (dcur / z(qo));
    }
    dmin = std::min(dmin, dnew);
    if (loop_form) emin = std::min(emin, z(eo));
    return dnew;
  };
  for (I64 j = i0; j <= n0 - 3; ++j) d = step(j, d, true);
  const double dnm2 = d;
  st.dmin2          = dmin;
  const double dnm1 = step(n0 - 2, dnm2, false);
  st.dmin1          = dmin;
  const double dn   = step(n0 - 1, dnm1, false);
  z(s.qout(n0))     = dn;
  z(s.eout(n0))     = emin;
  st.dn = dn, st.dn1 = dnm1, st.dn2 = dnm2, st.dmin = dmin;
}

/// LAPACK dlasq3: deflates converged values at the bottom of rows i0..n0 (lowering n0), then
/// takes one transform with a shift from lasq4, smaller ones again while the shift is too big.
/// pp = 2 on entry: the array was just reversed, skip the deflation checks.
void lasq3(I64 i0, I64& n0, Qd z, int& pp, double& sigma, double& desig, double& qmax, QdState& st) {
  constexpr double cbias = 1.5;
  const double     tol = kQdEps * 100.0, tol2 = tol * tol;
  const I64        n0in = n0;
  while (pp != 2) {
    if (n0 < i0) return;
    if (n0 > i0) {
      const I64 nn = 4 * n0 + pp;
      if (n0 > i0 + 1) {
        if (z(nn - 5) > tol2 * (sigma + z(nn - 3)) && z(nn - 2 * pp - 4) > tol2 * z(nn - 7)) {
          if (z(nn - 9) > tol2 * sigma && z(nn - 2 * pp - 8) > tol2 * z(nn - 11)) break;  // nothing to deflate
        } else {
          z(4 * n0 - 3) = z(4 * n0 + pp - 3) + sigma;  // e(n0 - 1) negligible: one value
          --n0;
          continue;
        }
      }
      // e(n0 - 2) negligible, or two rows left: the 2 x 2 block's two values.
      if (z(nn - 3) > z(nn - 7)) std::swap(z(nn - 3), z(nn - 7));
      double t = 0.5 * ((z(nn - 7) - z(nn - 3)) + z(nn - 5));
      if (z(nn - 5) > z(nn - 3) * tol2 && t != 0.0) {
        double s = z(nn - 3) * (z(nn - 5) / t);
        if (s <= t)
          s = z(nn - 3) * (z(nn - 5) / (t * (1.0 + std::sqrt(1.0 + s / t))));
        else
          s = z(nn - 3) * (z(nn - 5) / (t + std::sqrt(t) * std::sqrt(t + s)));
        t = z(nn - 7) + (s + z(nn - 5));
        z(nn - 3) *= z(nn - 7) / t;
        z(nn - 7) = t;
      }
      z(4 * n0 - 7) = z(nn - 7) + sigma;
      z(4 * n0 - 3) = z(nn - 3) + sigma;
      n0 -= 2;
      continue;
    }
    z(4 * n0 - 3) = z(4 * n0 + pp - 3) + sigma;  // a single row
    --n0;
  }
  if (pp == 2) pp = 0;
  // Reverse the rows if the bottom is much larger than the top (dqds converges at the bottom).
  if ((st.dmin <= 0.0 || n0 < n0in) && cbias * z(4 * i0 + pp - 3) < z(4 * n0 + pp - 3)) {
    const I64 ipn4 = 4 * (i0 + n0);
    for (I64 j4 = 4 * i0; j4 <= 2 * (i0 + n0 - 1); j4 += 4) {
      std::swap(z(j4 - 3), z(ipn4 - j4 - 3));
      std::swap(z(j4 - 2), z(ipn4 - j4 - 2));
      std::swap(z(j4 - 1), z(ipn4 - j4 - 5));
      std::swap(z(j4), z(ipn4 - j4 - 4));
    }
    if (n0 - i0 <= 4) {
      z(4 * n0 + pp - 1) = z(4 * i0 + pp - 1);
      z(4 * n0 - pp)     = z(4 * i0 - pp);
    }
    st.dmin2           = std::min(st.dmin2, z(4 * n0 + pp - 1));
    z(4 * n0 + pp - 1) = std::min({z(4 * n0 + pp - 1), z(4 * i0 + pp - 1), z(4 * i0 + pp + 3)});
    z(4 * n0 - pp)     = std::min({z(4 * n0 - pp), z(4 * i0 - pp), z(4 * i0 - pp + 4)});
    qmax               = std::max({qmax, z(4 * i0 + pp - 3), z(4 * i0 + pp + 1)});
    st.dmin            = -0.0;
  }
  lasq4(i0, n0, z, pp, n0in, st);
  for (;;) {
    lasq5(i0, n0, z, pp, sigma, st);
    if (st.dmin >= 0.0 && st.dmin1 >= 0.0) break;
    if (st.dmin < 0.0 && st.dmin1 > 0.0 && z(4 * (n0 - 1) - pp) < tol * (sigma + st.dn1) &&
        std::abs(st.dn) < tol * sigma) {  // convergence hidden by a negative dn
      z(4 * (n0 - 1) - pp + 2) = 0.0;
      st.dmin                  = 0.0;
      break;
    }
    if (st.dmin < 0.0) {  // shift too big
      if (st.ttype < -22) {
        st.tau = 0.0;  // failed twice: play safe
      } else if (st.dmin1 > 0.0) {
        st.tau = (st.tau + st.dmin) * (1.0 - 2.0 * kQdEps);  // late failure: a good shift
        st.ttype -= 11;
      } else {
        st.tau *= 0.25;  // early failure
        st.ttype -= 12;
      }
      continue;
    }
    if (std::isnan(st.dmin) && st.tau != 0.0) {
      st.tau = 0.0;
      continue;
    }
    lasq6(i0, n0, z, pp, st);  // NaN without a shift, or possible underflow
    st.tau = 0.0;
    break;
  }
  // sigma += tau, carrying the rounding error in desig.
  double t;
  if (st.tau < sigma) {
    desig += st.tau;
    t = sigma + desig;
    desig -= t - sigma;
  } else {
    t     = sigma + st.tau;
    desig = sigma - (t - st.tau) + desig;
  }
  sigma = t;
}

/// LAPACK dlasq2: the eigenvalues of the positive definite tridiagonal given by the qd array
/// z(1..2n) = q_1, e_1, ..., q_n (z has 4n entries), descending into z(1..n). n >= 3. Returns
/// non-zero when it fails (negative data or no convergence).
int lasq2(I64 n, Qd z) {
  constexpr double cbias = 1.5;
  const double     tol = kQdEps * 100.0, tol2 = tol * tol, safmin = std::numeric_limits<double>::min();
  z(2 * n)               = 0.0;
  double emin = z(2), qmax = 0.0, dsum = 0.0, esum = 0.0;
  for (I64 k = 1; k <= 2 * (n - 1); k += 2) {
    if (z(k) < 0.0 || z(k + 1) < 0.0) return -2;
    dsum += z(k);
    esum += z(k + 1);
    qmax = std::max(qmax, z(k));
    emin = std::min(emin, z(k + 1));
  }
  if (z(2 * n - 1) < 0.0) return -2;
  dsum += z(2 * n - 1);
  if (esum == 0.0) {  // diagonal
    for (I64 k = 2; k <= n; ++k) z(k) = z(2 * k - 1);
    std::sort(z.p, z.p + n, std::greater<>());
    return 0;
  }
  // Interleave: z = (q1, qq1, e1, ee1, q2, ...).
  for (I64 k = 2 * n; k >= 2; k -= 2) {
    z(2 * k)     = 0.0;
    z(2 * k - 1) = z(k);
    z(2 * k - 2) = 0.0;
    z(2 * k - 3) = z(k - 1);
  }
  I64 i0 = 1, n0 = n;
  if (cbias * z(4 * i0 - 3) < z(4 * n0 - 3)) {  // reverse if warranted
    const I64 ipn4 = 4 * (i0 + n0);
    for (I64 i4 = 4 * i0; i4 <= 2 * (i0 + n0 - 1); i4 += 4) {
      std::swap(z(i4 - 3), z(ipn4 - i4 - 3));
      std::swap(z(i4 - 1), z(ipn4 - i4 - 5));
    }
  }
  // Initial split checks (Li's test) and two unshifted transforms.
  int pp = 0;
  for (int k = 0; k < 2; ++k) {
    double d = z(4 * n0 + pp - 3);
    for (I64 i4 = 4 * (n0 - 1) + pp; i4 >= 4 * i0 + pp; i4 -= 4) {
      if (z(i4 - 1) <= tol2 * d) {
        z(i4 - 1) = -0.0;
        d         = z(i4 - 3);
      } else {
        d = z(i4 - 3) * (d / (d + z(i4 - 1)));
      }
    }
    emin = z(4 * i0 + pp + 1);
    d    = z(4 * i0 + pp - 3);
    for (I64 i4 = 4 * i0 + pp; i4 <= 4 * (n0 - 1) + pp; i4 += 4) {
      z(i4 - 2 * pp - 2) = d + z(i4 - 1);
      if (z(i4 - 1) <= tol2 * d) {
        z(i4 - 1)          = -0.0;
        z(i4 - 2 * pp - 2) = d;
        z(i4 - 2 * pp)     = 0.0;
        d                  = z(i4 + 1);
      } else if (safmin * z(i4 + 1) < z(i4 - 2 * pp - 2) && safmin * z(i4 - 2 * pp - 2) < z(i4 + 1)) {
        const double temp = z(i4 + 1) / z(i4 - 2 * pp - 2);
        z(i4 - 2 * pp)    = z(i4 - 1) * temp;
        d *= temp;
      } else {
        z(i4 - 2 * pp) = z(i4 + 1) * (z(i4 - 1) / z(i4 - 2 * pp - 2));
        d              = z(i4 + 1) * (d / z(i4 - 2 * pp - 2));
      }
      emin = std::min(emin, z(i4 - 2 * pp));
    }
    z(4 * n0 - pp - 2) = d;
    qmax               = z(4 * i0 - pp - 2);
    for (I64 i4 = 4 * i0 - pp + 2; i4 <= 4 * n0 - pp - 2; i4 += 4) qmax = std::max(qmax, z(i4));
    pp = 1 - pp;
  }
  QdState st;
  for (I64 outer = 0; outer <= n; ++outer) {
    if (n0 < 1) break;
    // Rows i0..n0: the last unreduced block. e(n0) holds minus the shift at its split.
    double desig = 0.0;
    double sigma = n0 == n ? 0.0 : -z(4 * n0 - 1);
    if (sigma < 0.0) return 1;
    double emax = 0.0, qmin = z(4 * n0 - 3);
    qmax = qmin;
    I64 i4 = 4 * n0;
    for (; i4 >= 8; i4 -= 4) {
      if (z(i4 - 5) <= 0.0) break;
      if (qmin >= 4.0 * emax) {
        qmin = std::min(qmin, z(i4 - 3));
        emax = std::max(emax, z(i4 - 5));
      }
      qmax = std::max(qmax, z(i4 - 7) + z(i4 - 5));
    }
    i0 = i4 / 4;
    pp = 0;
    if (n0 - i0 > 1) {  // reverse when the smallest d of a dqd pass is near the top
      double dee = z(4 * i0 - 3), deemin = dee;
      I64    kmin = i0;
      for (I64 j4 = 4 * i0 + 1; j4 <= 4 * n0 - 3; j4 += 4) {
        dee = z(j4) * (dee / (dee + z(j4 - 2)));
        if (dee <= deemin) {
          deemin = dee;
          kmin   = (j4 + 3) / 4;
        }
      }
      if ((kmin - i0) * 2 < n0 - kmin && deemin <= 0.5 * z(4 * n0 - 3)) {
        const I64 ipn4 = 4 * (i0 + n0);
        pp             = 2;
        for (I64 j4 = 4 * i0; j4 <= 2 * (i0 + n0 - 1); j4 += 4) {
          std::swap(z(j4 - 3), z(ipn4 - j4 - 3));
          std::swap(z(j4 - 2), z(ipn4 - j4 - 2));
          std::swap(z(j4 - 1), z(ipn4 - j4 - 5));
          std::swap(z(j4), z(ipn4 - j4 - 4));
        }
      }
    }
    st.dmin        = -std::max(0.0, qmin - 2.0 * std::sqrt(qmin) * std::sqrt(emax));  // minus the first shift
    const I64 nbig = 100 * (n0 - i0 + 1);
    bool      done = false;
    for (I64 it = 0; it < nbig; ++it) {
      if (i0 > n0) {
        done = true;
        break;
      }
      lasq3(i0, n0, z, pp, sigma, desig, qmax, st);
      pp = 1 - pp;
      // When an e is tiny, split.
      if (pp == 0 && n0 - i0 >= 3 && (z(4 * n0) <= tol2 * qmax || z(4 * n0 - 1) <= tol2 * sigma)) {
        I64    splt   = i0 - 1;
        double oldemn = z(4 * i0);
        qmax          = z(4 * i0 - 3);
        emin          = z(4 * i0 - 1);
        for (I64 j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
          if (z(j4) <= tol2 * z(j4 - 3) || z(j4 - 1) <= tol2 * sigma) {
            z(j4 - 1) = -sigma;
            splt      = j4 / 4;
            qmax      = 0.0;
            emin      = z(j4 + 3);
            oldemn    = z(j4 + 4);
          } else {
            qmax   = std::max(qmax, z(j4 + 1));
            emin   = std::min(emin, z(j4 - 1));
            oldemn = std::min(oldemn, z(j4));
          }
        }
        z(4 * n0 - 1) = emin;
        z(4 * n0)     = oldemn;
        i0            = splt + 1;
      }
    }
    if (!done) return 2;
  }
  if (n0 >= 1) return 3;
  for (I64 k = 2; k <= n; ++k) z(k) = z(4 * k - 3);
  std::sort(z.p, z.p + n, std::greater<>());
  return 0;
}

/// LAPACK dlas2: singular values of [[f, g], [0, h]].
void las2(double f, double g, double h, double& ssmin, double& ssmax) {
  const double fa = std::abs(f), ga = std::abs(g), ha = std::abs(h);
  const double fhmn = std::min(fa, ha), fhmx = std::max(fa, ha);
  if (fhmn == 0.0) {
    ssmin = 0.0;
    ssmax = fhmx == 0.0 ? ga : std::max(fhmx, ga) * std::sqrt(1.0 + std::pow(std::min(fhmx, ga) / std::max(fhmx, ga), 2));
    return;
  }
  if (ga < fhmx) {
    const double as = 1.0 + fhmn / fhmx, at = (fhmx - fhmn) / fhmx, au = (ga / fhmx) * (ga / fhmx);
    const double c  = 2.0 / (std::sqrt(as * as + au) + std::sqrt(at * at + au));
    ssmin           = fhmn * c;
    ssmax           = fhmx / c;
    return;
  }
  const double au = fhmx / ga;
  if (au == 0.0) {  // the off-diagonal dominates beyond double range
    ssmin = (fhmn * fhmx) / ga;
    ssmax = ga;
    return;
  }
  const double as = 1.0 + fhmn / fhmx, at = (fhmx - fhmn) / fhmx;
  const double c  = 1.0 / (std::sqrt(1.0 + (as * au) * (as * au)) + std::sqrt(1.0 + (at * au) * (at * au)));
  ssmin           = 2.0 * ((fhmn * c) * au);
  ssmax           = ga / (c + c);
}

/// Singular values only of the upper bidiagonal (d [n], e [n - 1]), descending into s (LAPACK
/// dlasq1: dqds on the squares, scaled into range first). Falls back to QL on the Golub-Kahan
/// matrix if dqds reports a failure.
void bidiagonal_values(I64 n, const double* d, const double* e, double* s) {
  if (n <= 0) return;
  if (n == 1) {
    s[0] = std::abs(d[0]);
    return;
  }
  if (n == 2) {
    las2(d[0], e[0], d[1], s[1], s[0]);
    return;
  }
  double sigmx = 0.0;
  for (I64 i = 0; i + 1 < n; ++i) sigmx = std::max(sigmx, std::abs(e[i]));
  if (sigmx == 0.0) {  // diagonal
    for (I64 i = 0; i < n; ++i) s[i] = std::abs(d[i]);
    std::sort(s, s + n, std::greater<>());
    return;
  }
  for (I64 i = 0; i < n; ++i) sigmx = std::max(sigmx, std::abs(d[i]));
  // Scale the largest entry to sqrt(eps / safmin) so that the squares neither overflow nor lose
  // the small entries.
  const double scale = std::sqrt(kQdEps / std::numeric_limits<double>::min());
  thread_local Vec work;
  work.assign(sz(4 * n), 0.0);
  for (I64 i = 0; i < n; ++i) {
    const double q = (std::abs(d[i]) / sigmx) * scale;
    work[sz(2 * i)] = q * q;
    if (i + 1 < n) {
      const double f      = (e[i] / sigmx) * scale;
      work[sz(2 * i + 1)] = f * f;
    }
  }
  if (lasq2(n, Qd{work.data()}) != 0) {
    bidiagonal_values_ql(n, d, e, s);
    return;
  }
  for (I64 i = 0; i < n; ++i) s[i] = (std::sqrt(work[sz(i)]) / scale) * sigmx;
}

}  // namespace

void transpose_into(const double* src, std::int64_t lds, double* dst, std::int64_t ldd, std::int64_t rows,
                    std::int64_t cols) {
  constexpr I64 kTile = 8;  // with power-of-two strides more rows would share an L1 set
  split_for((cols + kTile - 1) / kTile, rows * kTile, [&](I64 t0, I64 t1) {
    const I64 jend = std::min(cols, t1 * kTile);
    for (I64 i0 = 0; i0 < rows; i0 += kTile)
      for (I64 j0 = t0 * kTile; j0 < jend; j0 += kTile) {
        const I64 i1 = std::min(rows, i0 + kTile), j1 = std::min(jend, j0 + kTile);
        for (I64 j = j0; j < j1; ++j)
          for (I64 i = i0; i < i1; ++i) dst[j * ldd + i] = src[i * lds + j];
      }
  });
}

// ---- drivers --------------------------------------------------------------------------------------

void qr_householder(const double* a, double* q, double* r, I64 m, I64 n, bool full) {
  const I64 k = std::min(m, n), qc = full ? m : k, rr = full ? m : k;
  Vec       A(sz(m * n)), tau(sz(std::max<I64>(k, 1)));
  transpose_into(a, n, A.data(), m, m, n);
  geqrf(m, n, A.data(), m, tau.data());
  for (I64 i = 0; i < rr; ++i)
    for (I64 p = 0; p < n; ++p) r[i * n + p] = p >= i ? A[sz(i + p * m)] : 0.0;
  Vec Q(sz(m * qc), 0.0);
  for (I64 i = 0; i < std::min(m, qc); ++i) Q[sz(i + i * m)] = 1.0;
  apply_reflectors(false, m, qc, k, [&](I64 c, I64 i) { return A[sz(i + c * m)]; }, tau.data(), Q.data(), m, true);
  transpose_into(Q.data(), m, q, qc, qc, m);
}

void eigh_householder(const double* a, double* e_out, double* v_out, I64 n) {
  if (n <= 0) return;
  // Full symmetric column-major copy: the transpose is right on and below the diagonal, and
  // column j above it is row j of a's lower triangle.
  Vec A(sz(n * n));
  transpose_into(a, n, A.data(), n, n, n);
  split_for(n, n / 2, [&](I64 j0, I64 j1) {
    for (I64 j = j0; j < j1; ++j) std::copy(a + j * n, a + j * n + j, A.data() + j * n);
  });
  Vec d(sz(n)), e(sz(n), 0.0), tau(sz(n), 0.0);
  sytrd(n, A.data(), n, d.data(), e.data(), tau.data());
  if (!v_out) {
    tridiagonal_values(n, d.data(), e.data());
    std::copy(d.begin(), d.end(), e_out);
    return;
  }
  Vec Z(sz(n * n));
  stedc(n, d.data(), e.data(), Z.data());
  apply_reflectors(false, n - 1, n, n - 1, [&](I64 c, I64 i) { return A[sz((i + 1) + c * n)]; }, tau.data(),
                   Z.data() + 1, n, false);
  std::copy(d.begin(), d.end(), e_out);
  transpose_into(Z.data(), n, v_out, n, n, n);
}

void svd_householder(const double* a, double* s, double* u, double* v, I64 m, I64 n, bool full) {
  if (m < n) {  // svd(A^T) = V S U^T
    Vec at(sz(m * n));
    transpose_into(a, n, at.data(), m, m, n);
    svd_householder(at.data(), s, v, u, n, m, full);
    return;
  }
  if (n <= 0) return;
  // m >= n: R [n, n] from a QR when m > n, else A itself; then R = Qb B Pb^T.
  Vec A(sz(m * n)), tauqr(sz(n), 0.0), R(sz(n * n), 0.0);
  transpose_into(a, n, A.data(), m, m, n);
  if (m > n) {
    geqrf(m, n, A.data(), m, tauqr.data());
    for (I64 j = 0; j < n; ++j) std::copy(A.data() + j * m, A.data() + j * m + j + 1, R.data() + j * n);
  } else {
    R = A;
  }
  Vec d(sz(n)), e(sz(n), 0.0), tauq(sz(n), 0.0), taup(sz(n), 0.0);
  gebrd(n, n, R.data(), n, d.data(), e.data(), tauq.data(), taup.data());
  if (!u) {
    bidiagonal_values(n, d.data(), e.data(), s);
    return;
  }
  Vec UB(sz(n * n)), VB(sz(n * n));
  bdsdc(n, d.data(), e.data(), UB.data(), VB.data());
  apply_reflectors(false, n, n, n, [&](I64 c, I64 i) { return R[sz(i + c * n)]; }, tauq.data(), UB.data(), n, false);
  apply_reflectors(false, n - 1, n, n - 1, [&](I64 c, I64 i) { return R[sz(c + (i + 1) * n)]; }, taup.data(),
                   VB.data() + 1, n, false);
  // U = Qqr [UB 0; 0 I]: the QR's reflectors applied to the padded factor (no explicit Q).
  const I64 uc = full ? m : n;
  Vec       Uf(sz(m * uc), 0.0);
  for (I64 j = 0; j < n; ++j) std::copy(UB.data() + j * n, UB.data() + (j + 1) * n, Uf.data() + j * m);
  for (I64 j = n; j < uc; ++j) Uf[sz(j + j * m)] = 1.0;
  if (m > n)
    apply_reflectors(false, m, uc, n, [&](I64 c, I64 i) { return A[sz(i + c * m)]; }, tauqr.data(), Uf.data(), m,
                     false);
  // Descending order; row-major outputs.
  for (I64 t = 0; t < n; ++t) s[t] = d[sz(n - 1 - t)];
  reverse_columns(Uf.data(), m, m, n);
  reverse_columns(VB.data(), n, n, n);
  transpose_into(Uf.data(), m, u, uc, uc, m);
  transpose_into(VB.data(), n, v, n, n, n);
}

}  // namespace datamunge::nn::detail
