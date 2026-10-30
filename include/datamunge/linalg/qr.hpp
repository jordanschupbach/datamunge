#pragma once

/// @file qr.hpp
/// @brief Householder QR factorization \f$A=QR\f$ of a tall dense matrix and the
///        least-squares solve built on it.

#include <datamunge/linalg/dense_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::linalg {

/// @brief The factorization \f$A=QR\f$ returned by qr(): \f$Q\f$ is n x n orthogonal and
///        \f$R\f$ is n x m upper trapezoidal (zero below the diagonal).
///
/// @tparam T  Floating-point type.
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> A(3, 2, {1, 0, 1, 1, 1, 2});
/// const auto f = dm::linalg::qr(A);
/// const auto x = f.solve(std::vector<double>{1, 3, 5});  // least squares: y = 1 + 2t
/// std::print("{:.3f} {:.3f}\n", x[0], x[1]);             // 1.000 2.000
/// @endcode
///
/// @see qr(), least_squares()
template <typename T = double>
struct QRDecomposition {
    DenseMatrix<T> Q;  ///< n x n orthogonal factor.
    DenseMatrix<T> R;  ///< n x m upper-trapezoidal factor.

    /// @brief The rank cut-off used by solve() and is_full_rank() by default:
    ///        \f$\max(n,m)\,\varepsilon\,\max_i|R_{ii}|\f$.
    /// @return The absolute threshold below which a diagonal entry of R counts as zero.
    [[nodiscard]] T rank_threshold() const {
        const std::size_t k = std::min(Q.rows(), R.cols());
        T big = T{};
        for (std::size_t i = 0; i < k; ++i) big = std::max(big, std::abs(R(i, i)));
        return static_cast<T>(std::max(Q.rows(), R.cols())) * std::numeric_limits<T>::epsilon() * big;
    }

    /// @brief Least-squares solution of \f$\min_x\lVert Ax-b\rVert_2\f$ (the exact solution
    ///        when A is square and nonsingular): \f$Rx=(Q^\top b)_{1:m}\f$ by back substitution.
    /// @param b  Right-hand side of length n.
    /// @return The solution of length m.
    /// @throws std::invalid_argument  If `b.size() != n`.
    /// @throws std::runtime_error     If a diagonal entry of R is at or below rank_threshold().
    std::vector<T> solve(const std::vector<T>& b) const {
        const std::size_t n = Q.rows();
        const std::size_t m = R.cols();
        if (b.size() != n)
            throw std::invalid_argument("QRDecomposition::solve: b.size() != n");

        const std::size_t k = std::min(n, m);

        // c = Q^T b: c[i] = sum_j Q(j,i)*b[j]
        std::vector<T> c(k);
        for (std::size_t i = 0; i < k; ++i) {
            T s = T{};
            for (std::size_t j = 0; j < n; ++j)
                s += Q(j, i) * b[j];
            c[i] = s;
        }

        // Back-substitute R[0:k,0:k] x = c[0:k]
        std::vector<T> x(m, T{});
        const T        cut = rank_threshold();
        for (std::size_t i = k; i-- > 0;) {
            if (!(std::abs(R(i, i)) > cut))
                throw std::runtime_error(
                    "QRDecomposition::solve: R is rank-deficient");
            T s = c[i];
            for (std::size_t j = i + 1; j < k; ++j)
                s -= R(i, j) * x[j];
            x[i] = s / R(i, i);
        }

        return x;
    }

    /// @brief Solves column by column for a matrix right-hand side.
    /// @param B  n x r matrix of right-hand sides.
    /// @return The m x r matrix of solutions.
    /// @throws std::invalid_argument  If `B.rows() != n`.
    DenseMatrix<T> solve(const DenseMatrix<T>& B) const {
        const std::size_t n   = Q.rows();
        const std::size_t m   = R.cols();
        const std::size_t rhs = B.cols();
        if (B.rows() != n)
            throw std::invalid_argument(
                "QRDecomposition::solve(B): row mismatch");
        DenseMatrix<T> X(m, rhs);
        for (std::size_t j = 0; j < rhs; ++j) {
            auto col = B.col(j);
            auto x   = solve(col);
            for (std::size_t i = 0; i < m; ++i)
                X(i, j) = x[i];
        }
        return X;
    }

    /// @brief Whether every diagonal entry of R exceeds the threshold.
    /// @param tol  Absolute threshold; a negative value (the default) selects rank_threshold(),
    ///             which scales with the size of A.
    /// @return True if A has full column rank to working precision.
    [[nodiscard]] bool is_full_rank(T tol = T{-1}) const {
        const std::size_t k = std::min(Q.rows(), R.cols());
        if (tol < T{}) tol = rank_threshold();
        for (std::size_t i = 0; i < k; ++i)
            if (!(std::abs(R(i, i)) > tol)) return false;
        return true;
    }

    /// @brief The inverse of the leading m x m block of R, by back substitution.
    ///
    /// For a full-rank tall A, \f$(A^\top A)^{-1}=R^{-1}R^{-\top}\f$, so this gives the
    /// unscaled covariance of least-squares coefficients without forming \f$A^\top A\f$
    /// (which would square the condition number).
    /// @return The m x m upper-triangular matrix \f$R_{1:m,1:m}^{-1}\f$.
    /// @throws std::runtime_error  If R is rank-deficient (see is_full_rank()).
    [[nodiscard]] DenseMatrix<T> r_inverse() const {
        const std::size_t m = R.cols();
        if (Q.rows() < m || !is_full_rank())
            throw std::runtime_error("QRDecomposition::r_inverse: R is rank-deficient");
        DenseMatrix<T> Ri(m, m, T{});
        for (std::size_t j = 0; j < m; ++j) {
            Ri(j, j) = T{1} / R(j, j);
            for (std::size_t i = j; i-- > 0;) {
                T s = T{};
                for (std::size_t k = i + 1; k <= j; ++k) s += R(i, k) * Ri(k, j);
                Ri(i, j) = -s / R(i, i);
            }
        }
        return Ri;
    }
};

/// @brief Householder QR factorization of an n x m matrix with \f$n\ge m\f$.
///
/// Step j reflects column j below the diagonal onto a multiple of \f$e_j\f$ with
/// \f$H_j=I-2vv^\top\f$, \f$v\propto x+\mathrm{sign}(x_1)\lVert x\rVert e_1\f$ (the sign
/// avoids cancellation), so \f$R=H_{m}\cdots H_1A\f$ and \f$Q=H_1\cdots H_m\f$. Costs about
/// \f$2nm^2-2m^3/3\f$ flops for R plus about \f$4n^2m-2nm^2\f$ for the explicit n x n Q, and is backward
/// stable: the computed factors are exact for \f$A+\Delta A\f$ with
/// \f$\lVert\Delta A\rVert\le c\,nm\,u\lVert A\rVert\f$.
///
/// @tparam T  Floating-point type.
/// @param A_orig  Matrix to factor (n x m, \f$n\ge m\f$).
/// @return The factors Q and R.
/// @throws std::invalid_argument  If `A.rows() < A.cols()`.
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> A(2, 2, {3, 1, 4, 2});
/// const auto f = dm::linalg::qr(A);
/// std::print("{:.1f}\n", std::abs(f.R(0, 0)));  // 5.0 (the norm of the first column)
/// @endcode
///
/// @see QRDecomposition, least_squares(), svd()
template <typename T>
QRDecomposition<T> qr(const DenseMatrix<T>& A_orig) {
    const std::size_t n = A_orig.rows();
    const std::size_t m = A_orig.cols();
    if (n < m)
        throw std::invalid_argument("qr: A.rows() < A.cols()");

    DenseMatrix<T> A    = A_orig;          // working copy
    DenseMatrix<T> Q_tmp = DenseMatrix<T>::identity(n);  // accumulates H_k...H_0

    std::vector<T> v(n, T{});

    const std::size_t iters = std::min(n, m);

    for (std::size_t j = 0; j < iters; ++j) {
        // Compute ||A[j:n, j]||
        T xnorm = T{};
        for (std::size_t i = j; i < n; ++i)
            xnorm += A(i, j) * A(i, j);
        xnorm = std::sqrt(xnorm);

        if (xnorm == T{}) continue;

        // sigma = sign(A(j,j)) * xnorm  (same sign to avoid cancellation)
        T sigma = (A(j, j) >= T{}) ? xnorm : -xnorm;

        // Build v: v[i] = A(i,j) for i>=j, v[j] += sigma
        std::fill(v.begin(), v.end(), T{});
        for (std::size_t i = j; i < n; ++i)
            v[i] = A(i, j);
        v[j] += sigma;

        // Normalize v[j:n]
        T vnorm = T{};
        for (std::size_t i = j; i < n; ++i)
            vnorm += v[i] * v[i];
        vnorm = std::sqrt(vnorm);
        if (vnorm == T{}) continue;
        for (std::size_t i = j; i < n; ++i)
            v[i] /= vnorm;

        // Apply H = I - 2vv^T to A[j:n, j:m] (left multiply)
        for (std::size_t p = j; p < m; ++p) {
            T dot = T{};
            for (std::size_t i = j; i < n; ++i)
                dot += v[i] * A(i, p);
            for (std::size_t i = j; i < n; ++i)
                A(i, p) -= T{2} * v[i] * dot;
        }

        // Apply H to Q_tmp[j:n, :] (left multiply — accumulate Q^T)
        for (std::size_t p = 0; p < n; ++p) {
            T dot = T{};
            for (std::size_t i = j; i < n; ++i)
                dot += v[i] * Q_tmp(i, p);
            for (std::size_t i = j; i < n; ++i)
                Q_tmp(i, p) -= T{2} * v[i] * dot;
        }
    }

    QRDecomposition<T> dec;
    dec.Q = Q_tmp.transpose();
    dec.R = std::move(A);
    return dec;
}

} // namespace datamunge::linalg
