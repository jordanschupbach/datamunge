#pragma once

/// \file xiaolin_wu_line.hpp
/// \brief Xiaolin Wu's line algorithm: draw an antialiased line by spreading each
///        step's intensity across the two nearest pixels (Wu 1991).
///
/// Bresenham picks a single cell per step, so its lines have jagged "staircase" edges.
/// *Xiaolin Wu's algorithm* antialiases them: at each step along the major axis it colors
/// the *two* pixels straddling the ideal line, splitting the ink between them in
/// proportion to how close the line passes to each -- a pixel the line nearly touches
/// gets nearly full brightness, its neighbor the remainder. The two brightnesses always
/// sum to one, so the line has constant total intensity and smooth edges. It is the
/// classic fast antialiased line, still used where a full coverage-based rasterizer is
/// overkill.

#include <cmath>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A pixel with a fractional coverage/brightness in [0, 1].
struct WuPixel {
    int    x, y;
    double brightness;
};

/// \brief Rasterize an antialiased line from (x0,y0) to (x1,y1); returns pixels with coverage.
inline std::vector<WuPixel> xiaolin_wu_line(double x0, double y0, double x1, double y1) {
    std::vector<WuPixel> out;
    const bool           steep = std::fabs(y1 - y0) > std::fabs(x1 - x0);
    if (steep) { std::swap(x0, y0); std::swap(x1, y1); }
    if (x0 > x1) { std::swap(x0, x1); std::swap(y0, y1); }

    const double dx       = x1 - x0;
    const double dy       = y1 - y0;
    const double gradient = (dx == 0.0) ? 1.0 : dy / dx;

    auto emit = [&](int px, int py, double c) {
        if (steep) out.push_back({py, px, c});
        else out.push_back({px, py, c});
    };
    auto fpart  = [](double v) { return v - std::floor(v); };
    auto rfpart = [&](double v) { return 1.0 - fpart(v); };

    double intery = y0 + gradient * 0;
    // Endpoints handled simply (integer x range); interior gets the two-pixel split.
    const int xstart = static_cast<int>(std::lround(x0));
    const int xend   = static_cast<int>(std::lround(x1));
    intery           = y0 + gradient * (xstart - x0);
    for (int x = xstart; x <= xend; ++x) {
        const int    yint = static_cast<int>(std::floor(intery));
        const double f    = fpart(intery);
        emit(x, yint, rfpart(intery));      // nearer pixel: brighter
        emit(x, yint + 1, f);               // farther pixel: dimmer (the two sum to 1)
        intery += gradient;
    }
    return out;
}

}  // namespace datamunge::algorithms
