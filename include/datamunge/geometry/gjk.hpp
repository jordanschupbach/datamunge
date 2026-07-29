#pragma once

// Gilbert-Johnson-Keerthi (GJK) distance algorithm: the minimum distance between
// two convex shapes, computed from their support functions alone. It searches the
// Minkowski difference A - B for the point closest to the origin: the shapes
// intersect iff that point is the origin, and otherwise its distance from the
// origin is the distance between the shapes. Here the shapes are convex polygons
// given as vertex lists.

#include <datamunge/geometry/point2d.hpp>

#include <cmath>
#include <vector>

namespace datamunge::geometry {

namespace detail {

inline double gjk_dot(const Point2D& a, const Point2D& b) { return a.x * b.x + a.y * b.y; }
inline Point2D gjk_sub(const Point2D& a, const Point2D& b) { return {a.x - b.x, a.y - b.y}; }
inline Point2D gjk_add_scaled(const Point2D& a, const Point2D& d, double t) { return {a.x + t * d.x, a.y + t * d.y}; }

inline Point2D gjk_support(const std::vector<Point2D>& P, const Point2D& d) {
    int    bi = 0;
    double bd = gjk_dot(P[0], d);
    for (std::size_t i = 1; i < P.size(); ++i) {
        const double v = gjk_dot(P[i], d);
        if (v > bd) { bd = v; bi = static_cast<int>(i); }
    }
    return P[bi];
}

// Closest point of triangle (a,b,c) to the origin, plus the supporting feature
// (its 1-2 vertices), and whether the origin lies inside (Ericson, RTCD).
struct TriResult {
    Point2D              closest;
    std::vector<Point2D> feature;
    bool                 inside{false};
};
inline TriResult closest_triangle(const Point2D& a, const Point2D& b, const Point2D& c) {
    const Point2D O{0, 0};
    const Point2D ab = gjk_sub(b, a), ac = gjk_sub(c, a), ap = gjk_sub(O, a);
    const double  d1 = gjk_dot(ab, ap), d2 = gjk_dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return {a, {a}, false};
    const Point2D bp = gjk_sub(O, b);
    const double  d3 = gjk_dot(ab, bp), d4 = gjk_dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return {b, {b}, false};
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) { const double v = d1 / (d1 - d3); return {gjk_add_scaled(a, ab, v), {a, b}, false}; }
    const Point2D cp = gjk_sub(O, c);
    const double  d5 = gjk_dot(ab, cp), d6 = gjk_dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return {c, {c}, false};
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) { const double w = d2 / (d2 - d6); return {gjk_add_scaled(a, ac, w), {a, c}, false}; }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        const double  w  = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const Point2D bc = gjk_sub(c, b);
        return {gjk_add_scaled(b, bc, w), {b, c}, false};
    }
    return {O, {a, b, c}, true}; // origin strictly inside
}

inline Point2D closest_segment(const Point2D& a, const Point2D& b, std::vector<Point2D>& feature) {
    const Point2D O{0, 0};
    const Point2D ab  = gjk_sub(b, a);
    const double  den = gjk_dot(ab, ab);
    double        t   = den > 0 ? gjk_dot(gjk_sub(O, a), ab) / den : 0.0;
    if (t <= 0) { feature = {a}; return a; }
    if (t >= 1) { feature = {b}; return b; }
    feature = {a, b};
    return gjk_add_scaled(a, ab, t);
}

} // namespace detail

// Minimum distance between convex polygons A and B (0 if they overlap/touch).
inline double gjk_distance(const std::vector<Point2D>& A, const std::vector<Point2D>& B) {
    using namespace detail;
    if (A.empty() || B.empty()) return 0.0;
    auto mink = [&](const Point2D& d) { return gjk_sub(gjk_support(A, d), gjk_support(B, {-d.x, -d.y})); };
    auto len  = [&](const Point2D& p) { return std::sqrt(gjk_dot(p, p)); };

    Point2D dir = gjk_sub(A[0], B[0]);
    if (dir.x == 0 && dir.y == 0) dir = {1, 0};
    std::vector<Point2D> S{mink(dir)};
    Point2D              closest = S[0];

    for (int iter = 0; iter < 100; ++iter) {
        dir = {-closest.x, -closest.y};
        if (gjk_dot(dir, dir) < 1e-20) return 0.0; // origin on the current feature
        const Point2D a = mink(dir);
        // No progress toward the origin => converged; current closest is the answer.
        if (gjk_dot(a, dir) - gjk_dot(closest, dir) < 1e-10) return len(closest);
        S.push_back(a);
        if (S.size() == 2) {
            std::vector<Point2D> feat;
            closest = closest_segment(S[0], S[1], feat);
            S       = feat;
        } else { // size 3
            const TriResult r = closest_triangle(S[0], S[1], S[2]);
            if (r.inside) return 0.0;
            closest = r.closest;
            S       = r.feature;
        }
    }
    return len(closest);
}

} // namespace datamunge::geometry
