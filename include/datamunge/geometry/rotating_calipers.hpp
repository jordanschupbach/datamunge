#pragma once

#include <datamunge/geometry/convex_hull.hpp>
#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::geometry {

struct DiameterResult {
    Point2D a;
    Point2D b;
    double distance{0.0};
};

/// @brief The diameter of a point set: the farthest pair of points, by Euclidean distance.
///        The farthest pair is always a pair of convex hull vertices, so this computes the
///        hull first (convex_hull()) and then searches only among its (typically far fewer)
///        vertices -- brute-force O(h^2) over the hull rather than the classical O(h)
///        two-pointer rotating-calipers walk, the same "correct and simple over asymptotically
///        optimal" tradeoff this module already makes for closest_pair(). Throws
///        std::invalid_argument for fewer than 2 distinct points.
[[nodiscard]] inline DiameterResult polygon_diameter(const std::vector<Point2D>& points) {
    const auto hull = convex_hull(points);
    if (hull.size() < 2) {
        throw std::invalid_argument("polygon_diameter: need at least 2 distinct points");
    }

    DiameterResult best;
    best.distance = -1.0;
    for (std::size_t i = 0; i < hull.size(); ++i) {
        for (std::size_t j = i + 1; j < hull.size(); ++j) {
            const double d = distance(hull[i], hull[j]);
            if (d > best.distance) best = DiameterResult{hull[i], hull[j], d};
        }
    }
    return best;
}

struct MinimumBoundingRectangle {
    /// @brief The four corners, in order (so consecutive corners are adjacent, matching this
    ///        module's polygon convention elsewhere).
    std::vector<Point2D> corners;
    double width{0.0};
    double height{0.0};
    double area{0.0};
};

/// @brief The minimum-AREA bounding rectangle of a point set (not necessarily axis-aligned).
///        By a classical theorem, the optimal rectangle always has one side collinear with a
///        convex hull edge -- so this computes the hull, then for each hull edge, measures the
///        bounding rectangle aligned with that edge's direction (projecting every hull point
///        onto the edge direction and its perpendicular), keeping the smallest-area result.
///        O(h^2) (h = hull size): the classical rotating-calipers formulation does this in
///        O(h) via incrementally-advanced support points, but re-projecting from scratch for
///        every edge is far simpler to get right and, again, this module's established
///        complexity/simplicity tradeoff. Throws std::invalid_argument if the hull has fewer
///        than 3 vertices (no well-defined minimum-area rectangle for a degenerate point set).
[[nodiscard]] inline MinimumBoundingRectangle minimum_bounding_rectangle(const std::vector<Point2D>& points) {
    const auto hull = convex_hull(points);
    if (hull.size() < 3) {
        throw std::invalid_argument("minimum_bounding_rectangle: need at least 3 non-collinear points");
    }

    MinimumBoundingRectangle best;
    best.area = std::numeric_limits<double>::infinity();

    const std::size_t h = hull.size();
    for (std::size_t i = 0; i < h; ++i) {
        const Point2D& a = hull[i];
        const Point2D& b = hull[(i + 1) % h];
        double dx = b.x - a.x, dy = b.y - a.y;
        const double edge_len = std::sqrt(dx * dx + dy * dy);
        if (edge_len == 0.0) continue; // degenerate (duplicate) hull edge
        dx /= edge_len;
        dy /= edge_len;
        const double perp_x = -dy, perp_y = dx;

        double min_u = std::numeric_limits<double>::infinity(), max_u = -std::numeric_limits<double>::infinity();
        double min_v = std::numeric_limits<double>::infinity(), max_v = -std::numeric_limits<double>::infinity();
        for (const auto& p : hull) {
            const double u = (p.x - a.x) * dx + (p.y - a.y) * dy;
            const double v = (p.x - a.x) * perp_x + (p.y - a.y) * perp_y;
            min_u = std::min(min_u, u);
            max_u = std::max(max_u, u);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }

        const double width = max_u - min_u;
        const double height = max_v - min_v;
        const double area = width * height;
        if (area < best.area) {
            const auto to_world = [&](double u, double v) { return Point2D{a.x + u * dx + v * perp_x, a.y + u * dy + v * perp_y}; };
            best.area = area;
            best.width = width;
            best.height = height;
            best.corners = {to_world(min_u, min_v), to_world(max_u, min_v), to_world(max_u, max_v), to_world(min_u, max_v)};
        }
    }
    return best;
}

} // namespace datamunge::geometry
