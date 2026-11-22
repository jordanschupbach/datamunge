#pragma once

/// @file
/// @brief Shared symmetric eigensolver used by covariance and exponential updates.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::optim::detail {

/// @brief Owned eigenpairs of a real symmetric matrix.
struct EigenSystem {
    std::vector<double> values; ///< Eigenvalues floored at the requested lower limit.
    std::vector<double> vectors; ///< Row-major matrix whose columns are eigenvectors.
};

/// @brief Apply largest-off-diagonal Jacobi rotations to a symmetric covariance copy.
/// @param matrix Owned row-major n by n symmetric matrix.
/// @param n Matrix dimension.
/// @param floor Smallest permitted returned eigenvalue.
/// @return Unsorted eigenvectors and associated eigenvalues.
/// Uses at most 20*n*n+1 rotations and absolute 1e-14 off-diagonal stopping.
inline EigenSystem symmetric_eigensystem(std::vector<double> matrix, const std::size_t n, double floor) {
    EigenSystem result{std::vector<double>(n), std::vector<double>(n * n, 0.0)};
    for (std::size_t i = 0; i < n; ++i) result.vectors[i * n + i] = 1.0;
    const std::size_t max_sweeps = 20 * n * n + 1;
    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        std::size_t p = 0, q = 0;
        double largest = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const double candidate = std::abs(matrix[i * n + j]);
                if (candidate > largest) {
                    largest = candidate;
                    p = i;
                    q = j;
                }
            }
        }
        if (largest <= 1e-14) break;
        const double app = matrix[p * n + p];
        const double aqq = matrix[q * n + q];
        const double apq = matrix[p * n + q];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (std::size_t k = 0; k < n; ++k) {
            const double mkp = matrix[k * n + p];
            const double mkq = matrix[k * n + q];
            matrix[k * n + p] = cosine * mkp - sine * mkq;
            matrix[k * n + q] = sine * mkp + cosine * mkq;
        }
        for (std::size_t k = 0; k < n; ++k) {
            const double mpk = matrix[p * n + k];
            const double mqk = matrix[q * n + k];
            matrix[p * n + k] = cosine * mpk - sine * mqk;
            matrix[q * n + k] = sine * mpk + cosine * mqk;
        }
        matrix[p * n + q] = matrix[q * n + p] = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double vkp = result.vectors[k * n + p];
            const double vkq = result.vectors[k * n + q];
            result.vectors[k * n + p] = cosine * vkp - sine * vkq;
            result.vectors[k * n + q] = sine * vkp + cosine * vkq;
        }
    }
    for (std::size_t i = 0; i < n; ++i) result.values[i] = std::max(matrix[i * n + i], floor);
    return result;
}

} // namespace datamunge::optim::detail
