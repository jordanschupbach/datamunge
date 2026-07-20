#pragma once

#include <datamunge/geometry/point2d.hpp>
#include <datamunge/geometry/segment_intersection.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::geometry {

namespace detail {
inline bool inside_clip_edge(const Point2D& p, const Point2D& edge_a, const Point2D& edge_b) {
    return cross(edge_a, edge_b, p) >= 0.0;
}
} // namespace detail

/// @brief Clips @p subject (any simple polygon, convex or not) against @p clip (Sutherland-
///        Hodgman) and returns their intersection as a new polygon. @p clip MUST be convex
///        AND counterclockwise-ordered (exactly what convex_hull() produces) -- the algorithm
///        processes it one directed edge at a time, keeping only the subject-polygon portion
///        to the left of every edge, so a clockwise or non-convex clip polygon silently
///        produces a wrong (or empty) result rather than an error. Returns an empty polygon
///        if the two don't overlap at all.
[[nodiscard]] inline std::vector<Point2D> clip_polygon(const std::vector<Point2D>& subject, const std::vector<Point2D>& clip) {
    std::vector<Point2D> output = subject;
    const std::size_t clip_n = clip.size();

    for (std::size_t i = 0; i < clip_n && !output.empty(); ++i) {
        const Point2D& edge_a = clip[i];
        const Point2D& edge_b = clip[(i + 1) % clip_n];

        const std::vector<Point2D> input = output;
        output.clear();
        const std::size_t n = input.size();
        for (std::size_t j = 0; j < n; ++j) {
            const Point2D& current = input[j];
            const Point2D& prev = input[(j + n - 1) % n];
            const bool current_inside = detail::inside_clip_edge(current, edge_a, edge_b);
            const bool prev_inside = detail::inside_clip_edge(prev, edge_a, edge_b);

            if (current_inside) {
                if (!prev_inside) {
                    Point2D intersection;
                    if (line_intersection_point(prev, current, edge_a, edge_b, intersection)) output.push_back(intersection);
                }
                output.push_back(current);
            } else if (prev_inside) {
                Point2D intersection;
                if (line_intersection_point(prev, current, edge_a, edge_b, intersection)) output.push_back(intersection);
            }
        }
    }

    return output;
}

} // namespace datamunge::geometry
