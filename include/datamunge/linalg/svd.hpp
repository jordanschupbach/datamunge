#pragma once

/// @file svd.hpp
/// @brief Thin singular value decomposition of a dense matrix by one-sided Jacobi rotations.
///
/// `svd(A)` factors an \f$m\times n\f$ matrix as \f$A = U\,\mathrm{diag}(s)\,V^\top\f$ with
/// orthonormal columns in \f$U\f$ (\f$m\times k\f$, \f$k=\min(m,n)\f$) and \f$V\f$
/// (\f$n\times k\f$) and singular values \f$s_1\ge\dots\ge s_k\ge 0\f$. The algorithm is
/// Hestenes' one-sided Jacobi method: plane rotations applied to the columns of \f$A\f$ until
/// every pair of columns is orthogonal. It needs no bidiagonalization, is backward stable, and
/// computes small singular values to high relative accuracy, which makes it a good reference
/// solver for the moderate sizes used in statistics (up to a few hundred columns).

#include <datamunge/linalg/dense_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace datamunge::linalg {

/// @brief The thin SVD \f$A = U\,\mathrm{diag}(s)\,V^\top\f$ returned by svd().
///
/// @tparam T  Floating-point type (`float`, `double` or `long double`).
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> A(3, 2, {3, 0, 0, 4, 0, 0});
/// const auto f = dm::linalg::svd(A);
/// std::print("{} {} {}\n", f.s[0], f.s[1], f.condition_number());  // 4 3 1.3333333333333333
/// @endcode
///
/// @see svd(), least_squares()
template <typename T = double>
struct SVDDecomposition {
    DenseMatrix<T> U;            ///< m x k matrix whose columns are the left singular vectors (a column is zero when its singular value is 0).
    std::vector<T> s;            ///< The k = min(m, n) singular values, in decreasing order.
    DenseMatrix<T> V;            ///< n x k matrix whose columns are the right singular vectors.
    std::size_t    sweeps{0};    ///< Number of complete Jacobi sweeps over all column pairs.
    std::size_t    rotations{0}; ///< Number of plane rotations applied in total.
    bool           converged{false}; ///< True when a sweep found every column pair orthogonal to the tolerance.

    /// @brief The default relative cut-off for "numerically zero" singular values.
    /// @return \f$\max(m,n)\,\varepsilon\f$, the LAPACK/NumPy convention.
    [[nodiscard]] T default_rcond() const {
        return static_cast<T>(std::max(U.rows(), V.rows())) * std::numeric_limits<T>::epsilon();
    }

    /// @brief Numerical rank: the number of singular values above `rcond * s[0]`.
    /// @param rcond  Relative cut-off; a negative value selects default_rcond().
    /// @return The rank (0 for a zero matrix).
    [[nodiscard]] std::size_t rank(T rcond = T{-1}) const {
        if (s.empty() || s[0] == T{}) return 0;
        const T cut = (rcond < T{} ? default_rcond() : rcond) * s[0];
        return static_cast<std::size_t>(std::count_if(s.begin(), s.end(), [&](T v) { return v > cut; }));
    }

    /// @brief The spectral norm \f$\lVert A\rVert_2 = s_1\f$.
    /// @return The largest singular value (0 for an empty matrix).
    [[nodiscard]] T norm_2() const { return s.empty() ? T{} : s[0]; }

    /// @brief The 2-norm condition number \f$\kappa_2(A) = s_1/s_k\f$.
    /// @return The ratio of the extreme singular values; infinity when the smallest is zero.
    [[nodiscard]] T condition_number() const {
        if (s.empty()) return T{};
        if (s.back() == T{}) return std::numeric_limits<T>::infinity();
        return s.front() / s.back();
    }

    /// @brief Minimum-norm least-squares solution \f$x = V\,\Sigma_r^{+}\,U^\top b\f$.
    ///
    /// Singular values at or below `rcond * s[0]` are treated as zero, so for a rank-deficient
    /// matrix the result is the solution of smallest Euclidean norm among all minimizers of
    /// \f$\lVert Ax-b\rVert_2\f$.
    ///
    /// @param b      Right-hand side of length m.
    /// @param rcond  Relative cut-off; a negative value selects default_rcond().
    /// @return The solution of length n.
    /// @throws std::invalid_argument  If `b.size() != m`.
    [[nodiscard]] std::vector<T> solve(const std::vector<T>& b, T rcond = T{-1}) const {
        if (b.size() != U.rows()) throw std::invalid_argument("SVDDecomposition::solve: b.size() != rows");
        const std::size_t r = rank(rcond);
        std::vector<T>    x(V.rows(), T{});
        for (std::size_t j = 0; j < r; ++j) {
            T c = T{};
            for (std::size_t i = 0; i < U.rows(); ++i) c += U(i, j) * b[i];
            c /= s[j];
            for (std::size_t i = 0; i < V.rows(); ++i) x[i] += V(i, j) * c;
        }
        return x;
    }

    /// @brief The Moore-Penrose pseudo-inverse \f$A^{+} = V\,\Sigma_r^{+}\,U^\top\f$ (n x m).
    /// @param rcond  Relative cut-off; a negative value selects default_rcond().
    /// @return The n x m pseudo-inverse.
    [[nodiscard]] DenseMatrix<T> pseudo_inverse(T rcond = T{-1}) const {
        const std::size_t r = rank(rcond);
        DenseMatrix<T>    P(V.rows(), U.rows(), T{});
        for (std::size_t j = 0; j < r; ++j)
            for (std::size_t a = 0; a < V.rows(); ++a) {
                const T va = V(a, j) / s[j];
                for (std::size_t b = 0; b < U.rows(); ++b) P(a, b) += va * U(b, j);
            }
        return P;
    }

    /// @brief The rank-`k` truncation \f$A_k=\sum_{j\le k} s_j u_j v_j^\top\f$.
    ///
    /// By the Eckart-Young theorem this is the best rank-`k` approximation of \f$A\f$ in both the
    /// spectral and Frobenius norms, with errors \f$s_{k+1}\f$ and
    /// \f$(\sum_{j>k}s_j^2)^{1/2}\f$.
    ///
    /// @param k  Number of singular triplets to keep (clamped to the number available).
    /// @return The m x n matrix \f$A_k\f$.
    [[nodiscard]] DenseMatrix<T> reconstruct(std::size_t k) const {
        k = std::min(k, s.size());
        DenseMatrix<T> A(U.rows(), V.rows(), T{});
        for (std::size_t j = 0; j < k; ++j)
            for (std::size_t a = 0; a < U.rows(); ++a) {
                const T ua = U(a, j) * s[j];
                for (std::size_t b = 0; b < V.rows(); ++b) A(a, b) += ua * V(b, j);
            }
        return A;
    }
};

/// @brief Thin SVD of `A` by one-sided (Hestenes) Jacobi rotations.
///
/// For \f$m\ge n\f$ the method works on the columns \f$a_p\f$ of a copy of \f$A\f$. For each pair
/// \f$(p,q)\f$ with \f$\alpha=\lVert a_p\rVert^2\f$, \f$\beta=\lVert a_q\rVert^2\f$,
/// \f$\gamma=a_p^\top a_q\f$ it applies the rotation with \f$\zeta=(\beta-\alpha)/(2\gamma)\f$,
/// \f$t=\mathrm{sign}(\zeta)/(|\zeta|+\sqrt{1+\zeta^2})\f$, \f$c=1/\sqrt{1+t^2}\f$,
/// \f$s=ct\f$, which makes the two columns orthogonal, and accumulates the same rotation into
/// \f$V\f$. A pair is skipped when \f$|\gamma|\le\mathrm{tol}\sqrt{\alpha\beta}\f$; the
/// iteration stops after a sweep that skips every pair. Then \f$s_j=\lVert a_j\rVert\f$ and
/// \f$u_j=a_j/s_j\f$. For \f$m<n\f$ the routine factors \f$A^\top\f$ and swaps \f$U\f$ and
/// \f$V\f$. Each sweep costs about \f$3mn^2\f$ multiply-adds (plus \f$1.5n^3\f$ for \f$V\f$);
/// convergence is quadratic in the end and typically takes 6-15 sweeps.
///
/// @tparam T  Floating-point type.
/// @param A           Matrix to factor (any shape).
/// @param tol         Relative orthogonality tolerance; a non-positive value selects
///                    \f$m\,\varepsilon\f$.
/// @param max_sweeps  Upper bound on the number of sweeps.
/// @return The decomposition, with singular values sorted in decreasing order; check
///         `converged` when `max_sweeps` is small.
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> A(2, 2, {1, 1, 0, 1});
/// const auto f = dm::linalg::svd(A);
/// std::print("{:.6f} {:.6f}\n", f.s[0], f.s[1]);  // 1.618034 0.618034 (golden ratio and its inverse)
/// @endcode
///
/// @par References
/// - M. R. Hestenes, "Inversion of matrices by biorthogonalization and related results",
///   J. SIAM 6 (1958) 51-90.
/// - J. Demmel and K. Veselic, "Jacobi's method is more accurate than QR",
///   SIAM J. Matrix Anal. Appl. 13 (1992) 1204-1245.
///
/// @see SVDDecomposition, jacobi_eigen(), qr()
template <typename T>
SVDDecomposition<T> svd(const DenseMatrix<T>& A, T tol = T{}, std::size_t max_sweeps = 60) {
    if (A.rows() < A.cols()) {
        auto f = svd(A.transpose(), tol, max_sweeps);
        std::swap(f.U, f.V);
        return f;
    }
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    if (tol <= T{}) tol = static_cast<T>(std::max<std::size_t>(m, 1)) * std::numeric_limits<T>::epsilon();

    // Rows of W are the columns of A and rows of Vt the columns of V, so every
    // rotation touches two contiguous rows.
    DenseMatrix<T> W  = A.transpose();
    DenseMatrix<T> Vt = DenseMatrix<T>::identity(n);
    SVDDecomposition<T> f;

    for (; f.sweeps < max_sweeps && !f.converged; ++f.sweeps) {
        bool rotated = false;
        for (std::size_t p = 0; p + 1 < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                T alpha = T{}, beta = T{}, gamma = T{};
                for (std::size_t i = 0; i < m; ++i) {
                    alpha += W(p, i) * W(p, i);
                    beta += W(q, i) * W(q, i);
                    gamma += W(p, i) * W(q, i);
                }
                if (gamma == T{} || std::abs(gamma) <= tol * std::sqrt(alpha * beta)) continue;
                rotated         = true;
                const T zeta    = (beta - alpha) / (T{2} * gamma);
                const T t       = (zeta >= T{} ? T{1} : T{-1}) / (std::abs(zeta) + std::sqrt(T{1} + zeta * zeta));
                const T c       = T{1} / std::sqrt(T{1} + t * t);
                const T sn      = c * t;
                for (std::size_t i = 0; i < m; ++i) {
                    const T wp = W(p, i), wq = W(q, i);
                    W(p, i)    = c * wp - sn * wq;
                    W(q, i)    = sn * wp + c * wq;
                }
                for (std::size_t i = 0; i < n; ++i) {
                    const T vp = Vt(p, i), vq = Vt(q, i);
                    Vt(p, i)   = c * vp - sn * vq;
                    Vt(q, i)   = sn * vp + c * vq;
                }
                ++f.rotations;
            }
        }
        if (!rotated) f.converged = true;
    }
    if (n <= 1) f.converged = true;

    std::vector<T> norms(n);
    for (std::size_t j = 0; j < n; ++j) {
        T ss = T{};
        for (std::size_t i = 0; i < m; ++i) ss += W(j, i) * W(j, i);
        norms[j] = std::sqrt(ss);
    }
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return norms[a] > norms[b]; });

    f.s.resize(n);
    f.U = DenseMatrix<T>(m, n, T{});
    f.V = DenseMatrix<T>(n, n, T{});
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t j = order[k];
        f.s[k]              = norms[j];
        for (std::size_t i = 0; i < m; ++i) f.U(i, k) = norms[j] > T{} ? W(j, i) / norms[j] : T{};
        for (std::size_t i = 0; i < n; ++i) f.V(i, k) = Vt(j, i);
    }
    return f;
}

/// @brief The 2-norm condition number \f$\kappa_2(A)=s_{\max}/s_{\min}\f$, computed from svd().
/// @tparam T  Floating-point type.
/// @param A  Any matrix.
/// @return \f$\kappa_2(A)\f$; infinity if \f$A\f$ is rank-deficient to working precision.
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> A(2, 2, {1, 0, 0, 1e-6});
/// std::print("{:.0e}\n", dm::linalg::condition_number(A));  // 1e+06
/// @endcode
template <typename T>
T condition_number(const DenseMatrix<T>& A) {
    return svd(A).condition_number();
}

} // namespace datamunge::linalg
