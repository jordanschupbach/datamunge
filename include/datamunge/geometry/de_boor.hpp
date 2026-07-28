#pragma once

// De Boor's algorithm: numerically stable evaluation of a B-spline curve by
// repeated knot-weighted linear interpolation (the B-spline analogue of de
// Casteljau's algorithm for Bezier curves).

#include <datamunge/geometry/point2d.hpp>

#include <vector>

namespace datamunge::geometry {

// Evaluate the B-spline of the given `degree` with knot vector `knots` and
// `control` points at parameter t. Requires knots.size() == control.size() +
// degree + 1 and knots[degree] <= t <= knots[control.size()].
inline Point2D de_boor(int degree, const std::vector<double>& knots,
                       const std::vector<Point2D>& control, double t) {
    const int n = static_cast<int>(control.size());
    // Find the knot span k with knots[k] <= t < knots[k+1] (clamped to a valid span).
    int k = degree;
    while (k < n - 1 && knots[k + 1] <= t) ++k;

    // Working control points d[0..degree] = control[k-degree .. k].
    std::vector<Point2D> d(degree + 1);
    for (int j = 0; j <= degree; ++j) d[j] = control[j + k - degree];

    for (int r = 1; r <= degree; ++r) {
        for (int j = degree; j >= r; --j) {
            const int    i     = j + k - degree;
            const double denom = knots[i + degree - r + 1] - knots[i];
            const double a     = denom > 0 ? (t - knots[i]) / denom : 0.0;
            d[j].x = (1 - a) * d[j - 1].x + a * d[j].x;
            d[j].y = (1 - a) * d[j - 1].y + a * d[j].y;
        }
    }
    return d[degree];
}

// Sample the curve at `samples` uniformly spaced parameter values over its valid
// domain [knots[degree], knots[control.size()]].
inline std::vector<Point2D> de_boor_curve(int degree, const std::vector<double>& knots,
                                          const std::vector<Point2D>& control, int samples) {
    const int    n   = static_cast<int>(control.size());
    const double t0  = knots[degree];
    const double t1  = knots[n];
    std::vector<Point2D> out;
    out.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        const double t = t0 + (t1 - t0) * static_cast<double>(i) / static_cast<double>(samples - 1);
        out.push_back(de_boor(degree, knots, control, t < t1 ? t : t1 - 1e-12));
    }
    return out;
}

} // namespace datamunge::geometry
