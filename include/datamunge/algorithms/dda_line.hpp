#pragma once

/// \file dda_line.hpp
/// \brief The DDA (Digital Differential Analyzer) line algorithm: rasterize a line by
///        stepping the major axis in unit increments and accumulating the slope.
///
/// The DDA is the most direct line rasterizer: pick the axis with the larger extent as
/// the *driving* axis, take unit steps along it, and increment the other coordinate by
/// the *slope* (or its reciprocal) each step, rounding to the nearest cell. Unlike
/// Bresenham it uses floating-point arithmetic (one add per step and a round), which
/// makes it simpler to derive and to generalize (to interpolating colors, depth, or
/// texture coordinates along the line -- the reason DDA-style interpolation survives in
/// modern rasterizers) at the cost of Bresenham's pure-integer speed.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A rasterized integer grid point.
using DDAPoint = std::pair<int, int>;

/// \brief Rasterize the segment (x0,y0)-(x1,y1) with the DDA algorithm.
///
/// Steps \f$\max(|dx|,|dy|)+1\f$ points, incrementing by \f$dx/\text{steps}\f$ and
/// \f$dy/\text{steps}\f$ each time and rounding to the nearest cell.
inline std::vector<DDAPoint> dda_line(int x0, int y0, int x1, int y1) {
    const int    dx    = x1 - x0;
    const int    dy    = y1 - y0;
    const int    steps = std::max(std::abs(dx), std::abs(dy));
    std::vector<DDAPoint> pts;
    if (steps == 0) {
        pts.push_back({x0, y0});
        return pts;
    }
    const double xinc = static_cast<double>(dx) / steps;
    const double yinc = static_cast<double>(dy) / steps;
    double       x = x0, y = y0;
    for (int i = 0; i <= steps; ++i) {
        pts.push_back({static_cast<int>(std::lround(x)), static_cast<int>(std::lround(y))});
        x += xinc;
        y += yinc;
    }
    return pts;
}

}  // namespace datamunge::algorithms
