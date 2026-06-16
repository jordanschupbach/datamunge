#pragma once

#include <datamunge/linalg/dense_matrix.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::linalg {

template <typename T = double>
struct QRDecomposition {
    DenseMatrix<T> Q;  // n×n orthogonal
    DenseMatrix<T> R;  // n×m upper trapezoidal

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
        for (std::size_t i = k; i-- > 0;) {
            if (std::abs(R(i, i)) < T{1e-14})
                throw std::runtime_error(
                    "QRDecomposition::solve: R is rank-deficient");
            T s = c[i];
            for (std::size_t j = i + 1; j < k; ++j)
                s -= R(i, j) * x[j];
            x[i] = s / R(i, i);
        }

        return x;
    }

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

    bool is_full_rank(T tol = T{1e-12}) const {
        const std::size_t k = std::min(Q.rows(), R.cols());
        for (std::size_t i = 0; i < k; ++i)
            if (std::abs(R(i, i)) <= tol) return false;
        return true;
    }
};

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
