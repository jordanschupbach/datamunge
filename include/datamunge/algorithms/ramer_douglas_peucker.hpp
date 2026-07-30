#pragma once

/// \file ramer_douglas_peucker.hpp
/// \brief The Ramer-Douglas-Peucker algorithm: simplify a polyline by dropping points
///        that lie within a tolerance of the retained shape.
///
/// A curve captured as many points (a GPS track, a hand-drawn stroke, a coastline) often
/// carries far more vertices than its shape needs. *Ramer-Douglas-Peucker* (1972/1973)
/// simplifies it while bounding the error: keep the two endpoints, find the intermediate
/// point *farthest* from the straight segment between them, and if that distance exceeds
/// a tolerance \f$\varepsilon\f$, keep it and recurse on the two sub-polylines it defines;
/// otherwise discard every intermediate point. The result is a subset of the original
/// vertices whose polyline stays within \f$\varepsilon\f$ of the original everywhere. It
/// is the standard line-generalization algorithm in cartography and vector graphics.

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

using RDPPoint = std::pair<double, double>;

namespace detail {
/// Perpendicular distance from point p to the line through a and b (segment endpoints).
inline double perp_distance(const RDPPoint& p, const RDPPoint& a, const RDPPoint& b) {
    const double dx = b.first - a.first, dy = b.second - a.second;
    const double len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-12) return std::sqrt((p.first - a.first) * (p.first - a.first) + (p.second - a.second) * (p.second - a.second));
    return std::fabs(dy * p.first - dx * p.second + b.first * a.second - b.second * a.first) / len;
}

inline void rdp_recurse(const std::vector<RDPPoint>& pts, std::size_t lo, std::size_t hi, double eps,
                        std::vector<char>& keep) {
    if (hi <= lo + 1) return;
    double      dmax = 0.0;
    std::size_t idx  = lo;
    for (std::size_t i = lo + 1; i < hi; ++i) {
        const double d = perp_distance(pts[i], pts[lo], pts[hi]);
        if (d > dmax) { dmax = d; idx = i; }
    }
    if (dmax > eps) {
        keep[idx] = 1;
        rdp_recurse(pts, lo, idx, eps, keep);
        rdp_recurse(pts, idx, hi, eps, keep);
    }
}
}  // namespace detail

/// \brief Simplify a polyline, keeping a vertex subset within \p epsilon of the original.
inline std::vector<RDPPoint> ramer_douglas_peucker(const std::vector<RDPPoint>& points, double epsilon) {
    const std::size_t n = points.size();
    if (n < 3) return points;
    std::vector<char> keep(n, 0);
    keep[0]     = 1;
    keep[n - 1] = 1;
    detail::rdp_recurse(points, 0, n - 1, epsilon, keep);
    std::vector<RDPPoint> out;
    for (std::size_t i = 0; i < n; ++i)
        if (keep[i]) out.push_back(points[i]);
    return out;
}

}  // namespace datamunge::algorithms
