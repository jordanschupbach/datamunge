#pragma once

#include <datamunge/algebra/polynomial.hpp>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algebra {

namespace detail {

[[nodiscard]] inline std::vector<std::vector<double>> sylvester_matrix(const Polynomial& a, const Polynomial& b) {
    const int m = a.degree();
    const int n = b.degree();
    const std::size_t size = static_cast<std::size_t>(m + n);
    std::vector<std::vector<double>> mat(size, std::vector<double>(size, 0.0));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= m; ++j) mat[static_cast<std::size_t>(i)][static_cast<std::size_t>(i + j)] = a.coefficient(m - j);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j <= n; ++j) mat[static_cast<std::size_t>(n + i)][static_cast<std::size_t>(i + j)] = b.coefficient(n - j);

    return mat;
}

[[nodiscard]] inline double determinant(std::vector<std::vector<double>> mat) {
    const std::size_t n = mat.size();
    double det = 1.0;
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t row = col + 1; row < n; ++row)
            if (std::fabs(mat[row][col]) > std::fabs(mat[pivot][col])) pivot = row;
        if (std::fabs(mat[pivot][col]) < 1e-14) return 0.0;
        if (pivot != col) {
            std::swap(mat[pivot], mat[col]);
            det = -det;
        }
        det *= mat[col][col];
        for (std::size_t row = col + 1; row < n; ++row) {
            const double factor = mat[row][col] / mat[col][col];
            for (std::size_t k = col; k < n; ++k) mat[row][k] -= factor * mat[col][k];
        }
    }
    return det;
}

} // namespace detail

/// @brief The resultant of a and b: the determinant of their Sylvester matrix. Zero if and
///        only if a and b share a common root (equivalently, a nonconstant common factor).
[[nodiscard]] inline double resultant(const Polynomial& a, const Polynomial& b) {
    if (a.is_zero() || b.is_zero()) return 0.0;
    if (a.degree() == 0 || b.degree() == 0) return std::pow(a.evaluate(0.0), b.degree()) * std::pow(b.coefficient(b.degree()), a.degree());
    return detail::determinant(detail::sylvester_matrix(a, b));
}

/// @brief The discriminant of p: `(-1)^(n(n-1)/2) * resultant(p, p') / lc(p)`, where n =
///        degree(p). Zero if and only if p has a repeated root.
[[nodiscard]] inline double discriminant(const Polynomial& p) {
    const int n = p.degree();
    if (n <= 0) return 0.0;
    const double lead = p.coefficient(n);
    const double sign = ((n * (n - 1) / 2) % 2 == 0) ? 1.0 : -1.0;
    return sign * resultant(p, p.derivative()) / lead;
}

} // namespace datamunge::algebra
