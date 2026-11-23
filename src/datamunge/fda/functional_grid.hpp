#pragma once

/// @file functional_grid.hpp
/// @brief Shared quadrature for functional models on a common grid.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

/// @brief Internal numerical helpers shared by functional models.
namespace datamunge::fda::detail {

/// @brief Computes positive trapezoidal weights on a finite strictly increasing grid.
/// @param grid At least two finite grid points in strictly increasing order.
/// @return One positive integration weight per grid point.
/// @throws std::invalid_argument If the grid is malformed.
/// @throws std::overflow_error If a positive weight cannot be represented.
inline std::vector<double> trapezoidal_weights(const std::vector<double>& grid) {
    if (grid.size() < 2 || !std::isfinite(grid.front()))
        throw std::invalid_argument("functional grid requires at least two finite increasing points");
    std::vector<long double> weights(grid.size());
    for (std::size_t j = 1; j < grid.size(); ++j) {
        if (!std::isfinite(grid[j]) || !(grid[j] > grid[j-1]))
            throw std::invalid_argument("functional grid points must be finite and strictly increasing");
        const auto half_width = (static_cast<long double>(grid[j])-grid[j-1])/2;
        weights[j-1] += half_width;
        weights[j] += half_width;
    }
    std::vector<double> output;
    output.reserve(weights.size());
    for (const auto weight : weights) {
        if (weight > std::numeric_limits<double>::max())
            throw std::overflow_error("functional grid: a quadrature weight is outside the finite range");
        const auto cast = static_cast<double>(weight);
        if (!(cast > 0))
            throw std::overflow_error("functional grid: a quadrature weight rounds to zero");
        output.push_back(cast);
    }
    return output;
}

} // namespace datamunge::fda::detail
