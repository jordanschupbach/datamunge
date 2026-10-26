#pragma once

/// @file linalg.hpp
/// @brief Batched dense linear algebra with gradients (TF linalg ops).
///
/// Every op works on the last two axes ([..., m, n]) and batches over the
/// leading ones, in float32 or float64 (other dtypes convert to float32).
/// Decompositions run in float64 internally, one matrix per task, spread over
/// the thread pool for large batches. Gradients follow TensorFlow's
/// registered gradients and are written with differentiable nn ops, so second
/// derivatives work wherever TF supports them.
///
/// Like TF, `cholesky` and `eigh` read only the lower triangle of their input,
/// and `triangular_solve` only the triangle it is told to use.
///
/// Backends: when the library is built with LAPACK (DATAMUNGE_HAVE_LAPACK:
/// Accelerate on macOS, OpenBLAS's LAPACK or another LAPACK elsewhere), the
/// decompositions of larger matrices call it (gesdd, syevd, geqrf/orgqr,
/// potrf, getrf/getrs, trsm, geev); small matrices, where the built-in
/// kernels are faster, keep using them (see `set_linalg_backend`). Both give
/// the same results to rounding, with the same sign conventions:
///  - QR: R's diagonal has the opposite sign of the pivot column's leading
///    entry (Householder, as LAPACK and Eigen/TF), unless the column below is
///    already zero (then the reflection is skipped).
///  - SVD: each column of U has its largest-magnitude entry positive (V's
///    matching column follows); extra columns of U or V with full_matrices,
///    and columns for zero singular values, are any orthonormal completion.
///  - eigh: each eigenvector has its largest-magnitude entry positive.
///  - eig: eigenvalues sorted by modulus (ascending, as TF/Eigen), then by
///    real and imaginary part; each eigenvector has unit norm and its
///    largest-magnitude entry real and positive.
/// TF leaves these signs and phases to the solver, so individual vectors can
/// differ from TF's by a sign (or phase); gauge-invariant results agree.

#include <cstdint>

#include "datamunge/nn/fft.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

// ---- structure -----------------------------------------------------------------------

/// [..., n] -> [..., n, n] with the input on the diagonal (TF MatrixDiag).
Tensor matrix_diag(const Tensor& diagonal);

/// [..., m, n] -> [..., min(m, n)], the main diagonal (TF MatrixDiagPart).
Tensor matrix_diag_part(const Tensor& a);

/// `a` with its main diagonal replaced by `diagonal` [..., min(m, n)] (TF MatrixSetDiag).
Tensor set_diag(const Tensor& a, const Tensor& diagonal);

/// Keeps the band of `a` with `num_lower` subdiagonals and `num_upper`
/// superdiagonals and zeroes the rest; a negative count keeps that whole
/// triangle (TF MatrixBandPart). band_part(a, -1, 0) is the lower triangle,
/// the usual causal mask.
Tensor band_part(const Tensor& a, std::int64_t num_lower, std::int64_t num_upper);

/// Sum of the main diagonal: [..., m, n] -> [...].
Tensor trace(const Tensor& a);

/// Swaps the last two axes.
Tensor matrix_transpose(const Tensor& a);

// ---- decompositions --------------------------------------------------------------------

/// Lower Cholesky factor L of a symmetric positive definite matrix, A = L L^T
/// (TF Cholesky). Reads the lower triangle only; throws if A is not positive
/// definite. Gradient as TF (Murray 2016), symmetric in A.
Tensor cholesky(const Tensor& a);

/// Thin (or full) QR decomposition, A = Q R.
struct QrResult {
  Tensor q;  ///< [..., m, k] with k = min(m, n) (or [..., m, m] with full_matrices)
  Tensor r;  ///< [..., k, n] upper triangular (or [..., m, n])
};

/// Householder QR (TF Qr; LAPACK geqrf/orgqr for large inputs). Differentiable for full_matrices = false and any
/// shape, and for square inputs with full_matrices = true, like TF.
QrResult qr(const Tensor& a, bool full_matrices = false);

/// Singular value decomposition A = U diag(s) V^T.
struct SvdResult {
  Tensor s;  ///< [..., k] singular values, descending, k = min(m, n)
  Tensor u;  ///< [..., m, k] ([..., m, m] with full_matrices); undefined without compute_uv
  Tensor v;  ///< [..., n, k] ([..., n, n] with full_matrices); undefined without compute_uv
};

/// SVD (TF Svd): LAPACK gesdd (gesvd if it fails) or the built-in one-sided
/// Jacobi after a QR of tall inputs. Note that V is
/// returned, not V^T, as in TF. The gradient follows TF's SvdGrad (needs
/// distinct nonzero singular values when U or V receive gradient; with
/// full_matrices only for |m - n| <= 1).
SvdResult svd(const Tensor& a, bool compute_uv = true, bool full_matrices = false);

/// Eigen decomposition of a symmetric matrix.
struct EighResult {
  Tensor e;  ///< [..., n] eigenvalues, ascending
  Tensor v;  ///< [..., n, n] eigenvectors as columns; undefined without compute_v
};

/// Symmetric eigen decomposition (TF SelfAdjointEigV2; LAPACK syevd or the
/// built-in tridiagonal QL): reads the lower triangle, A v_i = e_i v_i. Gradient as TF (distinct eigenvalues).
EighResult eigh(const Tensor& a, bool compute_v = true);

/// Same as `eigh` (TF's name).
EighResult self_adjoint_eig(const Tensor& a, bool compute_v = true);

/// Eigenvalues only, ascending (TF self_adjoint_eigvals).
Tensor eigvalsh(const Tensor& a);

// ---- solves and determinants ------------------------------------------------------------

/// Solves A X = B (or A^T X = B with `adjoint`) for triangular A [..., n, n]
/// and B [..., n, k], reading only A's lower (or upper) triangle (TF
/// MatrixTriangularSolve). Batch axes broadcast.
Tensor triangular_solve(const Tensor& a, const Tensor& b, bool lower = true, bool adjoint = false);

/// Solves A X = B (or A^T X = B) by LU with partial pivoting (TF MatrixSolve).
/// Batch axes broadcast. Throws if A is singular.
Tensor solve(const Tensor& a, const Tensor& b, bool adjoint = false);

/// Solves A X = B given the Cholesky factor L of A (TF cholesky_solve).
Tensor cholesky_solve(const Tensor& chol, const Tensor& b);

/// Matrix inverse (TF MatrixInverse); with `adjoint`, the inverse of A^T.
Tensor inverse(const Tensor& a, bool adjoint = false);

/// Determinant via LU (TF MatrixDeterminant).
Tensor det(const Tensor& a);

/// Sign and log of the absolute determinant (TF LogMatrixDeterminant). A
/// singular matrix gives sign 0 and -inf. Only log_abs_det is differentiable.
struct SlogdetResult {
  Tensor sign;
  Tensor log_abs_det;
};
SlogdetResult slogdet(const Tensor& a);

/// log det A for symmetric positive definite A, via Cholesky (TF linalg.logdet).
Tensor logdet(const Tensor& a);

// ---- least squares, pseudo-inverse, general eigen ----------------------------------------

/// Least squares solution X [..., n, k] of A X ~= B for A [..., m, n] and
/// B [..., m, k] (TF matrix_solve_ls / tf.linalg.lstsq).
/// fast = true: Cholesky of the regularised normal equations, as TF:
///   m >= n: X = (A^T A + l2 I)^-1 A^T B;  m < n: X = A^T (A A^T + l2 I)^-1 B.
///   A must have full rank unless l2_regularizer > 0. Differentiable (built
///   from cholesky / cholesky_solve / matmul, so TF's gradient, any order).
/// fast = false: the minimum-norm least squares solution pinv(A) B via the
///   SVD, also for rank-deficient A; l2_regularizer is ignored, as TF. TF has
///   no gradient here; ours flows through `pinv` (needs distinct singular values).
Tensor lstsq(const Tensor& a, const Tensor& b, double l2_regularizer = 0.0, bool fast = true);

/// Same as `lstsq` (TF's op name).
Tensor matrix_solve_ls(const Tensor& a, const Tensor& b, double l2_regularizer = 0.0, bool fast = true);

/// Moore-Penrose pseudo-inverse [..., n, m] of A [..., m, n] (tf.linalg.pinv):
/// V diag(1/s) U^T keeping singular values above rcond * max(s). A negative
/// rcond means TF's default 10 * max(m, n) * eps(dtype). Differentiable
/// through `svd` (distinct singular values), as TF.
Tensor pinv(const Tensor& a, double rcond = -1.0);

/// Eigen decomposition of a general real square matrix (TF Eig).
struct EigResult {
  ComplexTensor e;  ///< [..., n] eigenvalues (sorted, see the file comment)
  ComplexTensor v;  ///< [..., n, n] eigenvectors as columns (unit norm); undefined without compute_v
};

/// General (non-symmetric) eigen decomposition A v_i = e_i v_i with complex
/// results in the input's float dtype. Uses LAPACK geev when available, else
/// a built-in Hessenberg reduction + shifted QR (EISPACK hqr2). Gradient as
/// TF's EigGrad (diagonalisable A, distinct eigenvalues; for eigenvectors only
/// functions invariant to each vector's phase are meaningful).
EigResult eig(const Tensor& a, bool compute_v = true);

/// Eigenvalues of a general real square matrix (TF eigvals), as `eig(a, false).e`.
ComplexTensor eigvals(const Tensor& a);

// ---- backend selection ---------------------------------------------------------------------

/// Which kernels the decompositions and solves use.
enum class LinalgBackend {
  Auto,     ///< LAPACK for matrices large enough to gain from it (the default)
  Builtin,  ///< always the built-in kernels
  Lapack,   ///< LAPACK for every size (falls back to Builtin without LAPACK)
};

/// Sets the backend for subsequent calls (process-wide; mainly for tests and benchmarks).
void set_linalg_backend(LinalgBackend backend);

/// The current backend setting.
LinalgBackend linalg_backend();

/// True when the library was built with LAPACK.
bool linalg_has_lapack();

}  // namespace datamunge::nn
