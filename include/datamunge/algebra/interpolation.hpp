#pragma once

#include <datamunge/algebra/polynomial.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::algebra {

/// @brief The unique polynomial of degree < n through the n points (xs[i], ys[i]), built via
///        Lagrange's basis-polynomial formula (O(n^2)).
/// @throws std::invalid_argument if xs/ys sizes disagree, are empty, or xs has a duplicate.
[[nodiscard]] inline Polynomial lagrange_interpolate(const std::vector<double>& xs, const std::vector<double>& ys) {
    if (xs.size() != ys.size() || xs.empty()) throw std::invalid_argument("xs/ys must be the same nonzero size");

    Polynomial result(0.0);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        Polynomial basis(1.0);
        double denom = 1.0;
        for (std::size_t j = 0; j < xs.size(); ++j) {
            if (i == j) continue;
            if (xs[j] == xs[i]) throw std::invalid_argument("xs must not contain duplicate values");
            basis = basis.multiply(Polynomial(std::vector<double>{-xs[j], 1.0}));
            denom *= (xs[i] - xs[j]);
        }
        result = result.add(basis.scale(ys[i] / denom));
    }
    return result;
}

/// @brief The same unique interpolating polynomial as lagrange_interpolate(), built instead via
///        Newton's divided differences (an O(n^2) triangular table, then expanded into
///        standard-basis coefficients). Included as the classical alternative construction:
///        useful when points are added incrementally, since the divided-difference table
///        extends by one row per new point rather than being rebuilt from scratch.
/// @throws std::invalid_argument if xs/ys sizes disagree, are empty, or xs has a duplicate.
[[nodiscard]] inline Polynomial newton_interpolate(const std::vector<double>& xs, const std::vector<double>& ys) {
    if (xs.size() != ys.size() || xs.empty()) throw std::invalid_argument("xs/ys must be the same nonzero size");

    const int n = static_cast<int>(xs.size());
    std::vector<double> coef = ys;
    for (int j = 1; j < n; ++j) {
        for (int i = n - 1; i >= j; --i) {
            const std::size_t si = static_cast<std::size_t>(i);
            if (xs[si] == xs[static_cast<std::size_t>(i - j)]) throw std::invalid_argument("xs must not contain duplicate values");
            coef[si] = (coef[si] - coef[si - 1]) / (xs[si] - xs[static_cast<std::size_t>(i - j)]);
        }
    }

    Polynomial result(coef[static_cast<std::size_t>(n - 1)]);
    for (int k = n - 2; k >= 0; --k) {
        result = result.multiply(Polynomial(std::vector<double>{-xs[static_cast<std::size_t>(k)], 1.0}))
                     .add(Polynomial(coef[static_cast<std::size_t>(k)]));
    }
    return result;
}

} // namespace datamunge::algebra
