#pragma once

/// \file newell_normal.hpp
/// \brief Newell's method for the normal (and plane) of a 3D polygon.
///
/// A triangle has an unambiguous normal from one cross product, but a polygon with more
/// vertices may be slightly *non-planar* (from floating-point drift or genuine curvature),
/// and a naive cross product of two edges is then unreliable -- it depends on which vertices
/// you pick and misbehaves at near-collinear corners. *Newell's method* computes a robust,
/// area-weighted normal by summing over all edges:
/// \f[
///   n_x = \sum_i (y_i - y_{i+1})(z_i + z_{i+1}),\quad
///   n_y = \sum_i (z_i - z_{i+1})(x_i + x_{i+1}),\quad
///   n_z = \sum_i (x_i - x_{i+1})(y_i + y_{i+1}),
/// \f]
/// with indices taken cyclically. Each term is twice the signed area the edge projects onto
/// a coordinate plane, so the sum is the polygon's *projected-area vector*; its length is
/// twice the polygon area and its direction is the average surface normal. Newell's method is
/// the standard way to find a polygon's supporting plane -- used in hidden-surface removal
/// (the painter's algorithm's depth sort), back-face culling, and mesh processing.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// \brief A 3D point {x, y, z}.
using Point3D = std::array<double, 3>;

/// \brief Unnormalized Newell normal of a polygon; its length is twice the polygon's area.
inline Point3D newell_normal_raw(const std::vector<Point3D>& poly) {
    Point3D n = {0.0, 0.0, 0.0};
    const std::size_t m = poly.size();
    for (std::size_t i = 0; i < m; ++i) {
        const Point3D& cur = poly[i];
        const Point3D& nxt = poly[(i + 1) % m];
        n[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2]);
        n[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0]);
        n[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1]);
    }
    return n;
}

/// \brief Unit surface normal of a 3D polygon via Newell's method.
inline Point3D newell_normal(const std::vector<Point3D>& poly) {
    Point3D n = newell_normal_raw(poly);
    double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len == 0.0) return {0.0, 0.0, 0.0};
    return {n[0] / len, n[1] / len, n[2] / len};
}

/// \brief Area of a planar (or near-planar) polygon: half the Newell normal's length.
inline double newell_polygon_area(const std::vector<Point3D>& poly) {
    Point3D n = newell_normal_raw(poly);
    return 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
}

}  // namespace datamunge::algorithms
