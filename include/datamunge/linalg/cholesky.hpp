#pragma once

/// @file cholesky.hpp
/// @brief Cholesky factorization \f$A=LL^\top\f$ of a symmetric positive definite matrix.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

/// @brief Dense and sparse linear algebra: matrices, factorizations, iterative solvers and regression.
namespace datamunge::linalg {

/// @brief The factor of \f$A=LL^\top\f$ returned by cholesky().
///
/// @tparam T  Floating-point type.
///
/// @par Example
/// @code
/// const auto f = dm::linalg::cholesky(dm::linalg::DenseMatrix<double>(2, 2, {4, 2, 2, 5}));
/// std::print("{} {} {}\n", f.L(0, 0), f.L(1, 0), f.L(1, 1));  // 2 1 2
/// @endcode
///
/// @see cholesky()
template <typename T = double>
struct CholeskyDecomposition {
    DenseMatrix<T> L;       ///< Lower-triangular factor with positive diagonal (meaningful only when ok).
    bool           ok{true}; ///< False if a non-positive pivot showed A is not (numerically) positive definite.

    /// @brief Solves \f$Ax=b\f$ by \f$Ly=b\f$ then \f$L^\top x=y\f$ (\f$2n^2\f$ flops).
    /// @param b  Right-hand side of length n.
    /// @return The solution x.
    /// @throws std::runtime_error     If the factorization failed (`ok == false`).
    /// @throws std::invalid_argument  If `b.size() != n`.
    std::vector<T> solve(const std::vector<T>& b) const {
        if (!ok)
            throw std::runtime_error(
                "CholeskyDecomposition::solve: matrix was not SPD");
        const std::size_t n = L.rows();
        if (b.size() != n)
            throw std::invalid_argument(
                "CholeskyDecomposition::solve: b size mismatch");

        // Forward substitution: L y = b
        std::vector<T> y(n);
        for (std::size_t i = 0; i < n; ++i) {
            T s = b[i];
            for (std::size_t k = 0; k < i; ++k)
                s -= L(i, k) * y[k];
            y[i] = s / L(i, i);
        }

        // Backward substitution: L^T x = y  (L^T(i,j) = L(j,i))
        std::vector<T> x(n);
        for (std::size_t i = n; i-- > 0;) {
            T s = y[i];
            for (std::size_t k = i + 1; k < n; ++k)
                s -= L(k, i) * x[k];
            x[i] = s / L(i, i);
        }

        return x;
    }

    /// @brief Solves \f$AX=B\f$ column by column.
    /// @param B  n x r matrix of right-hand sides.
    /// @return The n x r solution.
    /// @throws std::invalid_argument  If `B.rows() != n`.
    DenseMatrix<T> solve(const DenseMatrix<T>& B) const {
        const std::size_t n   = L.rows();
        const std::size_t rhs = B.cols();
        if (B.rows() != n)
            throw std::invalid_argument(
                "CholeskyDecomposition::solve(B): row mismatch");
        DenseMatrix<T> X(n, rhs);
        for (std::size_t j = 0; j < rhs; ++j) {
            auto col = B.col(j);
            auto x   = solve(col);
            for (std::size_t i = 0; i < n; ++i)
                X(i, j) = x[i];
        }
        return X;
    }

    /// @brief The determinant \f$\det A=\prod_i L_{ii}^2\f$.
    /// @return det(A), or 0 if the factorization failed.
    T det() const {
        if (!ok) return T{};
        const std::size_t n = L.rows();
        T d = T{1};
        for (std::size_t i = 0; i < n; ++i)
            d *= L(i, i);
        return d * d;
    }
};

/// @brief Cholesky-Banachiewicz factorization \f$A=LL^\top\f$ (\f$n^3/3\f$ flops).
///
/// Row by row, \f$\ell_{ij}=(a_{ij}-\sum_{k<j}\ell_{ik}\ell_{jk})/\ell_{jj}\f$ and
/// \f$\ell_{ii}=(a_{ii}-\sum_{k<i}\ell_{ik}^2)^{1/2}\f$. The square root's argument is the
/// Schur complement pivot, positive exactly when A is positive definite, so the routine doubles
/// as a definiteness test. Only the lower triangle of A is read. No pivoting is needed: every
/// \f$|\ell_{ij}|\le\sqrt{a_{ii}}\f$, so there is no element growth.
///
/// @tparam T  Floating-point type.
/// @param A  Symmetric matrix.
/// @return The factor; `ok` is false (and L partial) if a pivot was \f$\le 0\f$.
/// @throws std::invalid_argument  If A is not square.
///
/// @par Example
/// @code
/// const auto f = dm::linalg::cholesky(dm::linalg::DenseMatrix<double>(2, 2, {1, 2, 2, 1}));
/// std::print("{}\n", f.ok);  // false (eigenvalues 3 and -1)
/// @endcode
///
/// @see CholeskyDecomposition, lu()
template <typename T>
CholeskyDecomposition<T> cholesky(const DenseMatrix<T>& A) {
    const std::size_t n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("cholesky: matrix must be square");

    CholeskyDecomposition<T> dec;
    dec.L = DenseMatrix<T>(n, n, T{});

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            T s = A(i, j);
            for (std::size_t k = 0; k < j; ++k)
                s -= dec.L(i, k) * dec.L(j, k);
            dec.L(i, j) = s / dec.L(j, j);
        }
        T s = A(i, i);
        for (std::size_t k = 0; k < i; ++k)
            s -= dec.L(i, k) * dec.L(i, k);
        if (s <= T{}) {
            dec.ok = false;
            return dec;
        }
        dec.L(i, i) = std::sqrt(s);
    }

    return dec;
}

} // namespace datamunge::linalg
