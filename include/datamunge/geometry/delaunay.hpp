#pragma once

#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace datamunge::geometry {

/// @brief A Delaunay triangle, referenced by index into the point array passed to
///        delaunay_triangulation(). Vertices are always counterclockwise.
struct Triangle {
    std::size_t a{0};
    std::size_t b{0};
    std::size_t c{0};
};

namespace detail {

inline Triangle make_ccw_triangle(std::size_t a, std::size_t b, std::size_t c, const std::vector<Point2D>& pts) {
    if (cross(pts[a], pts[b], pts[c]) < 0.0) std::swap(b, c);
    return Triangle{a, b, c};
}

// The standard in-circumcircle determinant test (Shewchuk): positive iff d lies strictly
// inside the circumcircle of the counterclockwise-oriented triangle (a, b, c).
inline bool in_circumcircle(const Point2D& a, const Point2D& b, const Point2D& c, const Point2D& d) {
    const double ax = a.x - d.x, ay = a.y - d.y;
    const double bx = b.x - d.x, by = b.y - d.y;
    const double cx = c.x - d.x, cy = c.y - d.y;
    const double det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay) +
                        (cx * cx + cy * cy) * (ax * by - bx * ay);
    return det > 0.0;
}

struct CanonicalEdge {
    std::size_t u, v; // always u <= v, so the same undirected edge hashes/compares equal
    CanonicalEdge(std::size_t a, std::size_t b) : u(std::min(a, b)), v(std::max(a, b)) {}
    bool operator==(const CanonicalEdge& other) const { return u == other.u && v == other.v; }
};
struct CanonicalEdgeHash {
    std::size_t operator()(const CanonicalEdge& e) const {
        return std::hash<std::size_t>()(e.u) ^ (std::hash<std::size_t>()(e.v) << 1);
    }
};

} // namespace detail

/// @brief The Delaunay triangulation of @p points (Bowyer-Watson incremental algorithm,
///        O(n^2) worst case -- a deliberately simple, easy-to-verify-correct choice over the
///        faster O(n log n) sweep/divide-and-conquer algorithms, matching this module's
///        "basic" scope). Uses plain double-precision arithmetic for the in-circumcircle
///        predicate (not CGAL's exact/adaptive-precision arithmetic), so nearly-degenerate
///        inputs (many points exactly or near-exactly cocircular) may be triangulated
///        inconsistently at the boundary between two valid choices -- fine for typical inputs.
///        Returns fewer than 3 points unchanged as an empty triangle list (nothing to
///        triangulate); duplicate points are not deduplicated (the caller should do so if
///        that matters for their use case, e.g. via the same convex_hull() dedup logic).
[[nodiscard]] inline std::vector<Triangle> delaunay_triangulation(const std::vector<Point2D>& points) {
    const std::size_t n = points.size();
    if (n < 3) {
        return {};
    }

    std::vector<Point2D> pts = points;
    double min_x = pts[0].x, max_x = pts[0].x, min_y = pts[0].y, max_y = pts[0].y;
    for (const auto& p : pts) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    const double delta_max = std::max(max_x - min_x, max_y - min_y) + 1.0; // +1 guards degenerate all-collinear inputs
    const double mid_x = (min_x + max_x) / 2.0;
    const double mid_y = (min_y + max_y) / 2.0;

    // A triangle guaranteed to strictly contain every input point, appended as three extra
    // working vertices (indices n, n+1, n+2) removed again at the end.
    const std::size_t super_a = n, super_b = n + 1, super_c = n + 2;
    pts.push_back({mid_x - 20.0 * delta_max, mid_y - delta_max});
    pts.push_back({mid_x, mid_y + 20.0 * delta_max});
    pts.push_back({mid_x + 20.0 * delta_max, mid_y - delta_max});

    std::vector<Triangle> triangles{detail::make_ccw_triangle(super_a, super_b, super_c, pts)};

    for (std::size_t pi = 0; pi < n; ++pi) {
        std::vector<Triangle> bad;
        for (const auto& t : triangles) {
            if (detail::in_circumcircle(pts[t.a], pts[t.b], pts[t.c], pts[pi])) bad.push_back(t);
        }

        // The boundary of the polygonal hole left by removing the bad triangles is exactly
        // the set of edges that belong to only ONE bad triangle (edges shared between two bad
        // triangles are interior to the hole and get discarded).
        std::unordered_map<detail::CanonicalEdge, int, detail::CanonicalEdgeHash> edge_count;
        for (const auto& t : bad) {
            for (const auto& e : {detail::CanonicalEdge(t.a, t.b), detail::CanonicalEdge(t.b, t.c), detail::CanonicalEdge(t.c, t.a)})
                ++edge_count[e];
        }

        triangles.erase(std::remove_if(triangles.begin(), triangles.end(),
                                        [&](const Triangle& t) {
                                            return std::find_if(bad.begin(), bad.end(),
                                                                 [&](const Triangle& b) {
                                                                     return b.a == t.a && b.b == t.b && b.c == t.c;
                                                                 }) != bad.end();
                                        }),
                         triangles.end());

        for (const auto& [edge, count] : edge_count) {
            if (count == 1) triangles.push_back(detail::make_ccw_triangle(edge.u, edge.v, pi, pts));
        }
    }

    std::vector<Triangle> result;
    for (const auto& t : triangles) {
        if (t.a < n && t.b < n && t.c < n) result.push_back(t); // drop anything still touching the super triangle
    }
    return result;
}

} // namespace datamunge::geometry
