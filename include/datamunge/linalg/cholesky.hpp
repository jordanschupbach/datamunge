#pragma once

#include <datamunge/linalg/dense_matrix.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::linalg {

template <typename T = double>
struct CholeskyDecomposition {
    DenseMatrix<T> L;
    bool           ok{true};

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

    T det() const {
        if (!ok) return T{};
        const std::size_t n = L.rows();
        T d = T{1};
        for (std::size_t i = 0; i < n; ++i)
            d *= L(i, i);
        return d * d;
    }
};

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
