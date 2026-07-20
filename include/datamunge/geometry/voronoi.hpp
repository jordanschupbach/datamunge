#pragma once

#include <datamunge/geometry/delaunay.hpp>
#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::geometry {

namespace detail {
inline Point2D circumcenter(const Point2D& a, const Point2D& b, const Point2D& c) {
    const double d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    const double a2 = a.x * a.x + a.y * a.y;
    const double b2 = b.x * b.x + b.y * b.y;
    const double c2 = c.x * c.x + c.y * c.y;
    return Point2D{(a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d,
                   (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d};
}
} // namespace detail

struct VoronoiDiagram {
    /// @brief The circumcenter of each Delaunay triangle -- the Voronoi diagram's vertices
    ///        (may repeat a coordinate if it's shared by multiple cells; kept as plain
    ///        resolved points rather than a shared, index-referenced pool for a simpler,
    ///        binding-friendly shape).
    std::vector<Point2D> vertices;
    /// @brief cells[i] is points[i]'s Voronoi cell boundary, as circumcenters in angular
    ///        order around points[i]. For an INTERIOR point this is a closed polygon (as with
    ///        any polygon in this module, the closing edge back to cells[i].front() is
    ///        implicit). For a point on the convex hull, the true Voronoi cell is unbounded --
    ///        this only returns its FINITE portion (an open polyline, missing the two rays to
    ///        infinity), since robustly constructing those rays is out of scope here. Empty
    ///        for a point that ended up in no Delaunay triangle (possible for an
    ///        exactly-duplicated input point).
    std::vector<std::vector<Point2D>> cells;
};

/// @brief The Voronoi diagram of @p points, computed as the dual of their Delaunay
///        triangulation (delaunay_triangulation()) -- see VoronoiDiagram's cells field for
///        the important caveat about unbounded cells at the convex hull boundary. Fewer than
///        3 points produce an empty diagram (matching delaunay_triangulation()).
[[nodiscard]] inline VoronoiDiagram voronoi_diagram(const std::vector<Point2D>& points) {
    const auto triangles = delaunay_triangulation(points);

    VoronoiDiagram result;
    result.vertices.reserve(triangles.size());
    for (const auto& t : triangles) {
        result.vertices.push_back(detail::circumcenter(points[t.a], points[t.b], points[t.c]));
    }

    std::vector<std::vector<std::size_t>> cell_vertex_indices(points.size());
    for (std::size_t ti = 0; ti < triangles.size(); ++ti) {
        const auto& t = triangles[ti];
        for (const std::size_t v : {t.a, t.b, t.c}) cell_vertex_indices[v].push_back(ti);
    }

    result.cells.assign(points.size(), {});
    for (std::size_t pi = 0; pi < points.size(); ++pi) {
        auto& indices = cell_vertex_indices[pi];
        const Point2D& center = points[pi];
        std::sort(indices.begin(), indices.end(), [&](std::size_t i, std::size_t j) {
            const double angle_i = std::atan2(result.vertices[i].y - center.y, result.vertices[i].x - center.x);
            const double angle_j = std::atan2(result.vertices[j].y - center.y, result.vertices[j].x - center.x);
            return angle_i < angle_j;
        });
        for (const std::size_t idx : indices) result.cells[pi].push_back(result.vertices[idx]);
    }

    return result;
}

} // namespace datamunge::geometry
