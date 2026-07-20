#pragma once

#include <datamunge/geometry/point2d.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace datamunge::geometry {

/// @brief True if @p point lies strictly inside the simple polygon @p vertices (given in
///        either winding order), via the standard even-odd ray-casting rule (Franklin's
///        PNPOLY). Behavior for a point exactly on an edge is unspecified (as it is for every
///        ray-casting implementation) -- it may be reported as inside or outside depending on
///        which edge and floating-point rounding.
[[nodiscard]] inline bool point_in_polygon(const Point2D& point, const std::vector<Point2D>& vertices) {
    bool inside = false;
    const std::size_t n = vertices.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
        const Point2D& pi = vertices[i];
        const Point2D& pj = vertices[j];
        const bool straddles = (pi.y > point.y) != (pj.y > point.y);
        if (straddles && point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x) {
            inside = !inside;
        }
    }
    return inside;
}

/// @brief The polygon's signed area via the shoelace formula: positive for
///        counterclockwise-ordered vertices, negative for clockwise.
[[nodiscard]] inline double signed_polygon_area(const std::vector<Point2D>& vertices) {
    double sum = 0.0;
    const std::size_t n = vertices.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
        sum += vertices[j].x * vertices[i].y - vertices[i].x * vertices[j].y;
    }
    return 0.5 * sum;
}

/// @brief The polygon's (unsigned) area, regardless of vertex winding order.
[[nodiscard]] inline double polygon_area(const std::vector<Point2D>& vertices) { return std::abs(signed_polygon_area(vertices)); }

/// @brief The centroid (center of mass, assuming uniform density) of the polygon's ENCLOSED
///        AREA -- not the average of its vertex coordinates, which is a different point for
///        any non-regular polygon. Throws std::invalid_argument for a degenerate
///        (zero-area, e.g. collinear or fewer-than-3-vertex) polygon, since the formula
///        divides by the signed area.
[[nodiscard]] inline Point2D polygon_centroid(const std::vector<Point2D>& vertices) {
    const double area = signed_polygon_area(vertices);
    if (area == 0.0) {
        throw std::invalid_argument("polygon_centroid: polygon has zero area");
    }

    double cx = 0.0, cy = 0.0;
    const std::size_t n = vertices.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
        const double cross_term = vertices[j].x * vertices[i].y - vertices[i].x * vertices[j].y;
        cx += (vertices[j].x + vertices[i].x) * cross_term;
        cy += (vertices[j].y + vertices[i].y) * cross_term;
    }
    const double factor = 1.0 / (6.0 * area);
    return Point2D{cx * factor, cy * factor};
}

/// @brief True if @p vertices form a convex polygon: every triple of consecutive vertices
///        turns the same way (all left turns or all right turns), with no reflex vertices.
///        Fewer than 3 vertices is never convex.
[[nodiscard]] inline bool is_convex_polygon(const std::vector<Point2D>& vertices) {
    const std::size_t n = vertices.size();
    if (n < 3) {
        return false;
    }

    bool saw_positive = false, saw_negative = false;
    for (std::size_t i = 0; i < n; ++i) {
        const double turn = cross(vertices[i], vertices[(i + 1) % n], vertices[(i + 2) % n]);
        if (turn > 0.0) saw_positive = true;
        if (turn < 0.0) saw_negative = true;
        if (saw_positive && saw_negative) return false;
    }
    return true;
}

} // namespace datamunge::geometry
