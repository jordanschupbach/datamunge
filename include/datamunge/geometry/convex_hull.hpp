#pragma once

#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace datamunge::geometry {

/// @brief The 2D convex hull of @p points (Andrew's monotone chain, O(n log n)), returned as
///        hull vertices in counterclockwise order with no repeated first/last point. Points
///        exactly on a hull edge (collinear with two hull vertices) are excluded, matching
///        the "strict" convex hull convention. Fewer than 3 distinct input points are
///        returned as-is (there's no well-defined polygon to build).
[[nodiscard]] inline std::vector<Point2D> convex_hull(std::vector<Point2D> points) {
    std::sort(points.begin(), points.end(),
              [](const Point2D& a, const Point2D& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
    points.erase(std::unique(points.begin(), points.end()), points.end());

    const std::size_t n = points.size();
    if (n < 3) {
        return points;
    }

    std::vector<Point2D> hull(2 * n);
    std::size_t k = 0;

    // Lower hull: left to right, popping any point that would make a non-left (clockwise or
    // straight) turn.
    for (std::size_t i = 0; i < n; ++i) {
        while (k >= 2 && cross(hull[k - 2], hull[k - 1], points[i]) <= 0.0) --k;
        hull[k++] = points[i];
    }

    // Upper hull: right to left, same rule. lower_size protects the already-built lower hull
    // from being popped into.
    const std::size_t lower_size = k + 1;
    for (std::size_t step = 1; step < n; ++step) {
        const std::size_t i = n - 1 - step; // n-2 downto 0
        while (k >= lower_size && cross(hull[k - 2], hull[k - 1], points[i]) <= 0.0) --k;
        hull[k++] = points[i];
    }

    hull.resize(k - 1); // drop the final point, which duplicates the first
    return hull;
}

} // namespace datamunge::geometry
