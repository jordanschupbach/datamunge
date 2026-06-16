#pragma once

#include <datamunge/linalg/dense_matrix.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::linalg {

template <typename T = double>
struct LUDecomposition {
    DenseMatrix<T>           LU;
    std::vector<std::size_t> piv;
    bool                     singular{false};

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

    DenseMatrix<T> inverse() const {
        const std::size_t n = LU.rows();
        return solve(DenseMatrix<T>::identity(n));
    }
};

template <typename T>
LUDecomposition<T> lu(DenseMatrix<T> A) {
    const std::size_t n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("lu: matrix must be square");

    LUDecomposition<T> dec;
    dec.LU  = std::move(A);
    dec.piv.resize(n);
    for (std::size_t i = 0; i < n; ++i) dec.piv[i] = i;

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
