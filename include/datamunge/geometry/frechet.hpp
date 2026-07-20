#pragma once

#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace datamunge::geometry {

/// @brief The discrete Fréchet distance between two point sequences ("curves") p and q
///        (Eiter & Mannila, 1994): the minimum, over every monotone coupling between the two
///        curves, of the MAXIMUM pairwise distance along that coupling -- informally, the
///        shortest leash length needed for a person walking along p and a dog walking along q
///        (both only moving forward, never backward) to stay connected the whole way.
///        Computed via the standard O(n*m) dynamic program (bottom-up, not the naive
///        exponential recursion). Differs from DTW in exactly one respect: DTW sums
///        (cumulative cost) where Fréchet takes a max (bottleneck cost) -- Fréchet cares about
///        the worst single moment of divergence, DTW about total accumulated divergence.
[[nodiscard]] inline double discrete_frechet_distance(const std::vector<Point2D>& p, const std::vector<Point2D>& q) {
    const std::size_t n = p.size();
    const std::size_t m = q.size();
    if (n == 0 || m == 0) {
        throw std::invalid_argument("discrete_frechet_distance: both curves must be non-empty");
    }

    std::vector<std::vector<double>> ca(n, std::vector<double>(m, -1.0));
    ca[0][0] = distance(p[0], q[0]);
    for (std::size_t i = 1; i < n; ++i) ca[i][0] = std::max(ca[i - 1][0], distance(p[i], q[0]));
    for (std::size_t j = 1; j < m; ++j) ca[0][j] = std::max(ca[0][j - 1], distance(p[0], q[j]));
    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t j = 1; j < m; ++j) {
            const double prev_min = std::min({ca[i - 1][j], ca[i][j - 1], ca[i - 1][j - 1]});
            ca[i][j] = std::max(prev_min, distance(p[i], q[j]));
        }
    }

    return ca[n - 1][m - 1];
}

} // namespace datamunge::geometry
