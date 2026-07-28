#pragma once

// Classic convex-hull algorithms, complementing the module's default
// Andrew's-monotone-chain `convex_hull`. All three return the strict convex hull
// (hull vertices counterclockwise, points on an edge excluded, no repeated
// first/last vertex), matching `convex_hull`'s convention, so they are
// interchangeable and cross-checkable.

#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <vector>

namespace datamunge::geometry {

namespace detail {

inline std::vector<Point2D> sorted_unique(std::vector<Point2D> pts) {
    std::sort(pts.begin(), pts.end(),
              [](const Point2D& a, const Point2D& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    return pts;
}

// Signed area * 2 (shoelace); positive when the polygon is counterclockwise.
inline double signed_area2(const std::vector<Point2D>& poly) {
    double a = 0;
    const std::size_t n = poly.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Point2D& p = poly[i];
        const Point2D& q = poly[(i + 1) % n];
        a += p.x * q.y - q.x * p.y;
    }
    return a;
}

inline void make_ccw(std::vector<Point2D>& poly) {
    if (poly.size() >= 3 && signed_area2(poly) < 0) std::reverse(poly.begin(), poly.end());
}

} // namespace detail

// Graham scan: sort by polar angle around the lowest point, then sweep keeping
// only left turns. O(n log n).
inline std::vector<Point2D> graham_scan(std::vector<Point2D> points) {
    std::vector<Point2D> pts = detail::sorted_unique(std::move(points));
    const int            n   = static_cast<int>(pts.size());
    if (n < 3) return pts;

    int piv = 0;
    for (int i = 1; i < n; ++i)
        if (pts[i].y < pts[piv].y || (pts[i].y == pts[piv].y && pts[i].x < pts[piv].x)) piv = i;
    std::swap(pts[0], pts[piv]);
    const Point2D p0 = pts[0];
    std::sort(pts.begin() + 1, pts.end(), [&](const Point2D& a, const Point2D& b) {
        const double c = cross(p0, a, b);
        if (c != 0) return c > 0; // a before b when p0->a->b turns left
        return squared_distance(p0, a) < squared_distance(p0, b);
    });

    std::vector<Point2D> hull;
    for (const Point2D& p : pts) {
        while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), p) <= 0) hull.pop_back();
        hull.push_back(p);
    }
    return hull; // counterclockwise
}

// Gift wrapping (Jarvis march): from the leftmost point, repeatedly pick the most
// clockwise next point, wrapping the hull. O(n * h) for h hull vertices.
inline std::vector<Point2D> jarvis_march(std::vector<Point2D> points) {
    std::vector<Point2D> pts = detail::sorted_unique(std::move(points));
    const int            n   = static_cast<int>(pts.size());
    if (n < 3) return pts;

    int start = 0;
    for (int i = 1; i < n; ++i)
        if (pts[i].x < pts[start].x || (pts[i].x == pts[start].x && pts[i].y < pts[start].y)) start = i;

    std::vector<Point2D> hull;
    int                  cur = start;
    do {
        hull.push_back(pts[cur]);
        int next = (cur + 1) % n;
        for (int q = 0; q < n; ++q) {
            if (q == cur) continue;
            const double o = cross(pts[cur], pts[next], pts[q]);
            if (o < 0 || (o == 0 && squared_distance(pts[cur], pts[q]) > squared_distance(pts[cur], pts[next])))
                next = q;
        }
        cur = next;
    } while (cur != start && static_cast<int>(hull.size()) <= n + 1);

    detail::make_ccw(hull);
    return hull;
}

// Chan's algorithm: partition the points into groups, take each group's hull
// (Graham scan), then gift-wrap over the group-hull vertices. Doubling the guess
// m for the number of hull vertices until the wrap completes gives the optimal
// O(n log h) when the per-group tangent is found by binary search; this version
// scans each group's hull for clarity, so it wraps over far fewer candidates
// than plain gift wrapping while keeping the same grouping structure.
inline std::vector<Point2D> chans_algorithm(std::vector<Point2D> points) {
    std::vector<Point2D> pts = detail::sorted_unique(std::move(points));
    const int            n   = static_cast<int>(pts.size());
    if (n < 3) return pts;

    for (long long m = 4;; m = std::min<long long>(n, m * m)) {
        const int groups = static_cast<int>((n + m - 1) / m);
        std::vector<std::vector<Point2D>> subhulls;
        subhulls.reserve(groups);
        for (int g = 0; g < groups; ++g) {
            const int lo = static_cast<int>(g * m);
            const int hi = static_cast<int>(std::min<long long>(n, (g + 1) * m));
            subhulls.push_back(graham_scan(std::vector<Point2D>(pts.begin() + lo, pts.begin() + hi)));
        }

        Point2D start = pts[0];
        for (const Point2D& p : pts)
            if (p.y < start.y || (p.y == start.y && p.x < start.x)) start = p;

        std::vector<Point2D> hull;
        Point2D              cur       = start;
        bool                 completed = false;
        for (long long step = 0; step <= m; ++step) {
            hull.push_back(cur);
            Point2D next;
            bool    have = false;
            for (const auto& sh : subhulls)
                for (const Point2D& q : sh) {
                    if (q == cur) continue;
                    if (!have) { next = q; have = true; continue; }
                    const double o = cross(cur, next, q);
                    if (o < 0 || (o == 0 && squared_distance(cur, q) > squared_distance(cur, next))) next = q;
                }
            if (!have || next == start) { completed = true; break; }
            cur = next;
        }
        if (completed) {
            detail::make_ccw(hull);
            return hull;
        }
        if (m >= n) { // safety: should have completed by now
            detail::make_ccw(hull);
            return hull;
        }
    }
}

namespace detail {

// Recursively collect the hull vertices strictly left of the directed edge a->b,
// appended between a and b (a and b themselves are added by the caller).
inline void quickhull_rec(const std::vector<Point2D>& pts, const Point2D& a, const Point2D& b,
                          std::vector<Point2D>& out) {
    int    far  = -1;
    double best = 0;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        const double c = cross(a, b, pts[i]);
        if (c > best) { best = c; far = static_cast<int>(i); }
    }
    if (far < 0) return; // no point strictly left: edge a->b is on the hull
    const Point2D c = pts[far];

    std::vector<Point2D> left_ac, left_cb;
    for (const Point2D& p : pts) {
        if (cross(a, c, p) > 0) left_ac.push_back(p);
        else if (cross(c, b, p) > 0) left_cb.push_back(p);
    }
    quickhull_rec(left_ac, a, c, out);
    out.push_back(c);
    quickhull_rec(left_cb, c, b, out);
}

} // namespace detail

// Quickhull: split the points by the line through the two extreme x points, then
// recursively find the farthest point of each side and divide. O(n log n) average.
inline std::vector<Point2D> quickhull(std::vector<Point2D> points) {
    std::vector<Point2D> pts = detail::sorted_unique(std::move(points));
    const int            n   = static_cast<int>(pts.size());
    if (n < 3) return pts;

    const Point2D a = pts.front(); // leftmost (sorted by x, then y)
    const Point2D b = pts.back();  // rightmost

    std::vector<Point2D> above, below;
    for (const Point2D& p : pts) {
        const double c = cross(a, b, p);
        if (c > 0) above.push_back(p);
        else if (c < 0) below.push_back(p);
    }

    std::vector<Point2D> hull;
    hull.push_back(a);
    detail::quickhull_rec(above, a, b, hull); // upper chain a -> b
    hull.push_back(b);
    detail::quickhull_rec(below, b, a, hull); // lower chain b -> a
    detail::make_ccw(hull);
    return hull;
}

} // namespace datamunge::geometry
