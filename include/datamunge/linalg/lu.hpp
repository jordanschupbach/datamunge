#pragma once

/// @file lu.hpp
/// @brief LU factorization with partial pivoting, \f$PA=LU\f$, for square dense systems.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::linalg {

/// @brief The factorization \f$PA=LU\f$ returned by lu(), stored compactly.
///
/// @tparam T  Floating-point type.
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> A(2, 2, {1, 2, 3, 4});
/// const auto f = dm::linalg::lu(A);
/// const auto x = f.solve(std::vector<double>{5, 6});
/// std::print("{:.1f} {:.1f} det={:.1f}\n", x[0], x[1], f.det());  // -4.0 4.5 det=-2.0
/// @endcode
///
/// @see lu(), cholesky(), qr()
template <typename T = double>
struct LUDecomposition {
    DenseMatrix<T>           LU;  ///< Strict lower triangle holds L (unit diagonal implied); upper triangle holds U.
    std::vector<std::size_t> piv; ///< Row i of LU came from row piv[i] of A, i.e. (PA)(i, :) = A(piv[i], :).
    bool                     singular{false}; ///< True if an exactly zero pivot was met.

    /// @brief Solves \f$Ax=b\f$: permute, then forward (L) and back (U) substitution, \f$2n^2\f$ flops.
    /// @param b  Right-hand side of length n.
    /// @return The solution x.
    /// @throws std::runtime_error     If the matrix was found singular.
    /// @throws std::invalid_argument  If `b.size() != n`.
    std::vector<T> solve(const std::vector<T>& b) const {
        if (singular)
            throw std::runtime_error("LUDecomposition::solve: matrix is singular");
        const std::size_t n = LU.rows();
        if (b.size() != n)
            throw std::invalid_argument("LUDecomposition::solve: b size mismatch");

        // Apply permutation
        std::vector<T> x(n);
        for (std::size_t i = 0; i < n; ++i)
            x[i] = b[piv[i]];

        // Forward substitution: L y = x (L has unit diagonal)
        for (std::size_t i = 1; i < n; ++i)
            for (std::size_t j = 0; j < i; ++j)
                x[i] -= LU(i, j) * x[j];

        // Backward substitution: U x = y
        for (std::size_t i = n; i-- > 0;) {
            for (std::size_t j = i + 1; j < n; ++j)
                x[i] -= LU(i, j) * x[j];
            x[i] /= LU(i, i);
        }

        return x;
    }

    /// @brief Solves \f$AX=B\f$ column by column.
    /// @param B  n x r matrix of right-hand sides.
    /// @return The n x r solution.
    /// @throws std::invalid_argument  If `B.rows() != n`.
    DenseMatrix<T> solve(const DenseMatrix<T>& B) const {
        const std::size_t n   = LU.rows();
        const std::size_t rhs = B.cols();
        if (B.rows() != n)
            throw std::invalid_argument("LUDecomposition::solve(B): row mismatch");
        DenseMatrix<T> X(n, rhs);
        for (std::size_t j = 0; j < rhs; ++j) {
            auto col = B.col(j);
            auto x   = solve(col);
            for (std::size_t i = 0; i < n; ++i)
                X(i, j) = x[i];
        }
        return X;
    }

    /// @brief The determinant \f$\det A=\mathrm{sign}(P)\prod_i U_{ii}\f$.
    /// @return det(A) (0 if singular). The product can overflow for large n; sum
    ///         \f$\log|U_{ii}|\f$ yourself for a log-determinant.
    T det() const {
        const std::size_t n = LU.rows();
        T d = T{1};
        for (std::size_t i = 0; i < n; ++i)
            d *= LU(i, i);

        // Count sign from permutation using cycle decomposition
        // A cycle of length l requires l-1 transpositions.
        // Sign flip if total transpositions is odd.
        std::vector<bool> visited(n, false);
        int sign = 1;
        for (std::size_t i = 0; i < n; ++i) {
            if (!visited[i]) {
                std::size_t len = 0;
                std::size_t cur = i;
                while (!visited[cur]) {
                    visited[cur] = true;
                    cur = piv[cur];
                    ++len;
                }
                // cycle of length len needs len-1 transpositions
                if ((len - 1) % 2 == 1)
                    sign = -sign;
            }
        }

        return d * static_cast<T>(sign);
    }

    /// @brief The inverse \f$A^{-1}\f$, by solving against the identity (\f$\approx 2n^3\f$ flops after lu()).
    /// @return The n x n inverse.
    /// @throws std::runtime_error  If the matrix was found singular.
    DenseMatrix<T> inverse() const {
        const std::size_t n = LU.rows();
        return solve(DenseMatrix<T>::identity(n));
    }
};

/// @brief Gaussian elimination with partial (row) pivoting: \f$PA=LU\f$.
///
/// At step k the row with the largest \f$|a_{ik}|\f$, \f$i\ge k\f$, is swapped into the pivot
/// position, so every multiplier satisfies \f$|\ell_{ik}|\le 1\f$. Costs \f$2n^3/3\f$ flops.
/// The computed factors satisfy \f$\hat L\hat U=P(A+\Delta A)\f$ with
/// \f$|\Delta A|\le\gamma_n|\hat L||\hat U|\f$, which is small in practice (the growth factor
/// is modest except on contrived matrices).
///
/// @tparam T  Floating-point type.
/// @param A  Square matrix (taken by value and overwritten by the factors).
/// @return The packed factors and pivot vector; `singular` is set if a pivot was exactly 0.
/// @throws std::invalid_argument  If A is not square.
///
/// @par Example
/// @code
/// const auto f = dm::linalg::lu(dm::linalg::DenseMatrix<double>(2, 2, {0, 1, 1, 0}));
/// std::print("{} {}\n", f.piv[0], f.det());  // 1 -1
/// @endcode
///
/// @see LUDecomposition
template <typename T>
LUDecomposition<T> lu(DenseMatrix<T> A) {
    const std::size_t n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("lu: matrix must be square");

    LUDecomposition<T> dec;
    dec.LU  = std::move(A);
    dec.piv.resize(n);
    for (std::size_t i = 0; i < n; ++i) dec.piv[i] = i;
    if (n == 0) return dec;

    auto& M = dec.LU;

    for (std::size_t k = 0; k < n - 1; ++k) {
        // Find pivot
        std::size_t pivot_row = k;
        T max_val = std::abs(M(k, k));
        for (std::size_t i = k + 1; i < n; ++i) {
            T av = std::abs(M(i, k));
            if (av > max_val) {
                max_val   = av;
                pivot_row = i;
            }
        }

        if (max_val == T{}) {
            dec.singular = true;
            continue;
        }

        if (pivot_row != k) {
            for (std::size_t j = 0; j < n; ++j)
                std::swap(M(k, j), M(pivot_row, j));
            std::swap(dec.piv[k], dec.piv[pivot_row]);
        }

        for (std::size_t i = k + 1; i < n; ++i) {
            M(i, k) /= M(k, k);
            for (std::size_t j = k + 1; j < n; ++j)
                M(i, j) -= M(i, k) * M(k, j);
        }
    }

    if (M(n - 1, n - 1) == T{})
        dec.singular = true;

    return dec;
}

} // namespace datamunge::linalg
