#pragma once

#include <datamunge/linalg/dense_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace datamunge::linalg {

template <typename T = double>
struct SymmetricEigenResult {
    std::vector<T> eigenvalues;   // descending order
    DenseMatrix<T> eigenvectors;  // orthonormal columns, matching eigenvalues order
};

// Classical cyclic Jacobi eigenvalue algorithm for a symmetric matrix.
// Only the upper triangle of `A` is read (it is symmetrized defensively).
// Robust and simple for the small matrices (a handful of predictors) that
// show up in statistics use cases like LDA; not intended for large n.
template <typename T = double>
SymmetricEigenResult<T> jacobi_eigen(DenseMatrix<T> A, T tol = static_cast<T>(1e-12),
                                     std::size_t max_sweeps = 100) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("jacobi_eigen: matrix must be square");
    const std::size_t n = A.rows();

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            A(j, i) = A(i, j);

    DenseMatrix<T> V = DenseMatrix<T>::identity(n);

    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        T off = T{};
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                off += A(i, j) * A(i, j);
        if (off <= tol) break;

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                if (std::abs(A(p, q)) < tol) continue;

                const T theta = (A(q, q) - A(p, p)) / (T{2} * A(p, q));
                const T sign  = (theta >= T{}) ? T{1} : T{-1};
                const T t     = sign / (std::abs(theta) + std::sqrt(theta * theta + T{1}));
                const T c     = T{1} / std::sqrt(t * t + T{1});
                const T s     = t * c;

                const T app = A(p, p);
                const T aqq = A(q, q);

                A(p, p) = app - t * A(p, q);
                A(q, q) = aqq + t * A(p, q);
                A(p, q) = T{};
                A(q, p) = T{};

                for (std::size_t i = 0; i < n; ++i) {
                    if (i == p || i == q) continue;
                    const T aip = A(i, p);
                    const T aiq = A(i, q);
                    A(i, p) = c * aip - s * aiq;
                    A(p, i) = A(i, p);
                    A(i, q) = s * aip + c * aiq;
                    A(q, i) = A(i, q);
                }

                for (std::size_t i = 0; i < n; ++i) {
                    const T vip = V(i, p);
                    const T viq = V(i, q);
                    V(i, p) = c * vip - s * viq;
                    V(i, q) = s * vip + c * viq;
                }
            }
        }
    }

    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return A(a, a) > A(b, b); });

    SymmetricEigenResult<T> result;
    result.eigenvalues.resize(n);
    result.eigenvectors = DenseMatrix<T>(n, n, T{});
    for (std::size_t new_index = 0; new_index < n; ++new_index) {
        const std::size_t old_index   = order[new_index];
        result.eigenvalues[new_index] = A(old_index, old_index);
        for (std::size_t i = 0; i < n; ++i)
            result.eigenvectors(i, new_index) = V(i, old_index);
    }
    return result;
}

} // namespace datamunge::linalg
