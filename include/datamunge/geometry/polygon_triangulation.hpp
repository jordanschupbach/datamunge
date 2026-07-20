#pragma once

#include <datamunge/geometry/delaunay.hpp> // reuses the Triangle{a,b,c} index type
#include <datamunge/geometry/point2d.hpp>
#include <datamunge/geometry/polygon.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace datamunge::geometry {

namespace detail {
inline bool point_in_triangle(const Point2D& p, const Point2D& a, const Point2D& b, const Point2D& c) {
    const double d1 = cross(a, b, p);
    const double d2 = cross(b, c, p);
    const double d3 = cross(c, a, p);
    const bool has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    const bool has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
    return !(has_neg && has_pos);
}
} // namespace detail

/// @brief Triangulates a SIMPLE polygon (convex or not, but not self-intersecting) by ear
///        clipping: repeatedly finds a "ear" -- a convex vertex whose neighbor-triangle
///        contains no other polygon vertex -- clips it off as one output triangle, and
///        repeats on the remaining (n-1)-vertex polygon. O(n^3) worst case (each of the O(n)
///        clipped ears requires an O(n) scan to verify, each vertex check O(1)) -- unlike
///        delaunay_triangulation() (which only sees an unordered point SET), this respects
///        the polygon's actual boundary, so it correctly handles concave (non-convex) shapes.
///        Every triangle's vertices are indices into @p polygon. Works for any winding order
///        (detects and internally corrects clockwise input). Returns empty for fewer than 3
///        vertices; a triangle input returns itself unchanged.
[[nodiscard]] inline std::vector<Triangle> triangulate_polygon(const std::vector<Point2D>& polygon) {
    const std::size_t n = polygon.size();
    if (n < 3) {
        return {};
    }
    if (n == 3) {
        return {Triangle{0, 1, 2}};
    }

    std::vector<std::size_t> indices(n);
    for (std::size_t i = 0; i < n; ++i) indices[i] = i;
    if (signed_polygon_area(polygon) < 0.0) {
        std::reverse(indices.begin(), indices.end()); // work in CCW order internally
    }

    std::vector<Triangle> result;
    std::size_t guard = 0;
    while (indices.size() > 3 && guard < n * n) {
        ++guard;
        const std::size_t m = indices.size();
        bool clipped_an_ear = false;

        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t i_prev = (i + m - 1) % m;
            const std::size_t i_next = (i + 1) % m;
            const Point2D& a = polygon[indices[i_prev]];
            const Point2D& b = polygon[indices[i]];
            const Point2D& c = polygon[indices[i_next]];

            if (cross(a, b, c) <= 0.0) continue; // reflex (or collinear) vertex: can't be an ear

            bool is_ear = true;
            for (std::size_t k = 0; k < m; ++k) {
                if (k == i_prev || k == i || k == i_next) continue;
                if (detail::point_in_triangle(polygon[indices[k]], a, b, c)) {
                    is_ear = false;
                    break;
                }
            }

            if (is_ear) {
                result.push_back(Triangle{indices[i_prev], indices[i], indices[i_next]});
                indices.erase(indices.begin() + static_cast<std::ptrdiff_t>(i));
                clipped_an_ear = true;
                break;
            }
        }

        if (!clipped_an_ear) break; // malformed (e.g. self-intersecting) input; stop rather than loop forever
    }

    if (indices.size() == 3) {
        result.push_back(Triangle{indices[0], indices[1], indices[2]});
    }

    return result;
}

} // namespace datamunge::geometry
