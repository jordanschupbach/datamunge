#pragma once

/// \file bresenham.hpp
/// \brief Bresenham's line algorithm: rasterize a line segment onto an integer grid
///        using only integer arithmetic (Bresenham 1965).
///
/// To draw a line on a pixel grid we must pick, for each column (or row) it crosses,
/// the single grid cell nearest the ideal line. Doing this with floating-point slopes
/// is slow and error-prone; *Bresenham's algorithm* does it with *integers only*. It
/// walks along the major axis one step at a time, maintaining an integer *error* term
/// that accumulates the line's slope; when the accumulated error would carry the line
/// past the half-pixel boundary, it steps along the minor axis and corrects the error.
/// The "integer midpoint" formulation here handles all octants uniformly via the sign
/// and magnitude of \f$dx\f$ and \f$dy\f$. Every operation is an add, subtract, or
/// shift -- no multiplication or division inside the loop -- which is why it was (and
/// remains) the workhorse of line rasterization.

#include <cstdlib>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A rasterized integer grid point.
using GridPoint = std::pair<int, int>;

/// \brief Rasterize the line segment from (x0,y0) to (x1,y1) into grid points.
///
/// Returns the sequence of cells the line passes through, inclusive of both endpoints,
/// using the all-octant integer "midpoint" Bresenham formulation.
inline std::vector<GridPoint> bresenham_line(int x0, int y0, int x1, int y1) {
    std::vector<GridPoint> points;
    int       dx  = std::abs(x1 - x0);
    int       dy  = -std::abs(y1 - y0);
    const int sx  = x0 < x1 ? 1 : -1;
    const int sy  = y0 < y1 ? 1 : -1;
    int       err = dx + dy;  // error term, = dx - |dy|

    for (;;) {
        points.push_back({x0, y0});
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) {  // step in x
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {  // step in y
            err += dx;
            y0 += sy;
        }
    }
    return points;
}

}  // namespace datamunge::algorithms
