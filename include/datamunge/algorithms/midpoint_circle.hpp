#pragma once

/// \file midpoint_circle.hpp
/// \brief The midpoint circle algorithm: rasterize a circle with integer arithmetic
///        and 8-way symmetry (Bresenham's circle).
///
/// A circle has eight-fold symmetry: computing the cells of one 45-degree octant gives
/// the other seven by reflection. The *midpoint circle algorithm* walks that octant
/// with an integer *decision variable* that tracks whether the ideal circle passes
/// above or below the midpoint between the two candidate next cells, choosing the
/// closer one and updating the decision variable with only additions -- no
/// trigonometry, no square roots, no floating point. It is the circle analogue of
/// Bresenham's line algorithm and the standard way hardware and libraries draw circles.

#include <cstdlib>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

using CirclePoint = std::pair<int, int>;

/// \brief Rasterize a circle of integer radius \p r centered at (\p cx, \p cy).
///
/// Computes one octant with the integer midpoint decision variable and mirrors it into
/// all eight octants; returns the (unordered) set of boundary cells.
inline std::vector<CirclePoint> midpoint_circle(int cx, int cy, int r) {
    std::vector<CirclePoint> pts;
    if (r < 0) return pts;
    int x = r, y = 0;
    int d = 1 - r;  // initial decision variable
    auto plot8 = [&](int px, int py) {
        pts.push_back({cx + px, cy + py});
        pts.push_back({cx - px, cy + py});
        pts.push_back({cx + px, cy - py});
        pts.push_back({cx - px, cy - py});
        pts.push_back({cx + py, cy + px});
        pts.push_back({cx - py, cy + px});
        pts.push_back({cx + py, cy - px});
        pts.push_back({cx - py, cy - px});
    };
    while (x >= y) {
        plot8(x, y);
        ++y;
        if (d < 0) {
            d += 2 * y + 1;
        } else {
            --x;
            d += 2 * (y - x) + 1;
        }
    }
    return pts;
}

}  // namespace datamunge::algorithms
