#pragma once

#include <datamunge/geometry/point2d.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::geometry {

namespace detail {
inline double point_to_line_distance(const Point2D& p, const Point2D& a, const Point2D& b) {
    const double len = distance(a, b);
    if (len == 0.0) return distance(p, a); // a == b: "line" degenerates to a point
    return std::abs(cross(a, b, p)) / len;
}
} // namespace detail

/// @brief Simplifies a polyline (Douglas-Peucker): recursively keeps only the point(s)
///        farthest from the current baseline segment whenever that distance exceeds
///        @p epsilon, discarding everything closer. Fewer than 3 points are returned
///        unchanged (nothing to simplify).
[[nodiscard]] inline std::vector<Point2D> simplify_polyline(const std::vector<Point2D>& points, double epsilon) {
    const std::size_t n = points.size();
    if (n < 3) {
        return points;
    }

    double max_dist = 0.0;
    std::size_t index = 0;
    for (std::size_t i = 1; i + 1 < n; ++i) {
        const double d = detail::point_to_line_distance(points[i], points.front(), points.back());
        if (d > max_dist) {
            max_dist = d;
            index = i;
        }
    }

    if (max_dist > epsilon) {
        const std::vector<Point2D> left_input(points.begin(), points.begin() + static_cast<std::ptrdiff_t>(index) + 1);
        const std::vector<Point2D> right_input(points.begin() + static_cast<std::ptrdiff_t>(index), points.end());
        auto left = simplify_polyline(left_input, epsilon);
        const auto right = simplify_polyline(right_input, epsilon);
        left.pop_back(); // drop the point shared by both halves before joining them
        left.insert(left.end(), right.begin(), right.end());
        return left;
    }

    return {points.front(), points.back()};
}

} // namespace datamunge::geometry
