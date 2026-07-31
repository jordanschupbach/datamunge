#pragma once

/// \file scanline_fill.hpp
/// \brief Scanline polygon fill via the active-edge / even-odd rule.
///
/// Scanline rendering fills a polygon one horizontal line at a time. For each scanline
/// \f$y\f$ it finds where the polygon's edges cross that line, sorts the crossings by
/// \f$x\f$, and fills the spans *between consecutive pairs* -- the *even-odd rule*: a pixel
/// is inside the polygon iff a ray to infinity crosses the boundary an odd number of times.
/// The crossing \f$x\f$ for an edge from \f$(x_0,y_0)\f$ to \f$(x_1,y_1)\f$ at height
/// \f$y\f$ is the linear interpolation
/// \f[
///   x = x_0 + \frac{y - y_0}{y_1 - y_0}\,(x_1 - x_0).
/// \f]
/// Using the half-open convention \f$[\min y, \max y)\f$ per edge counts each vertex once, so
/// shared endpoints and horizontal edges do not double-fill. This span-based fill is the core
/// of classic polygon rasterizers and flood-free region filling.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// \brief A 2D point {x, y}.
using Point2D = std::array<double, 2>;

/// \brief An integer pixel {x, y}.
struct Pixel {
    int x, y;
    bool operator==(const Pixel& o) const { return x == o.x && y == o.y; }
};

/// \brief Fill the polygon with vertices \p poly (in order) and return the interior pixels.
/// Uses the even-odd rule with a half-open edge convention so vertices are counted once.
inline std::vector<Pixel> scanline_fill(const std::vector<Point2D>& poly) {
    std::vector<Pixel> pixels;
    const std::size_t n = poly.size();
    if (n < 3) return pixels;

    double ymin = poly[0][1], ymax = poly[0][1];
    for (const auto& p : poly) {
        ymin = std::min(ymin, p[1]);
        ymax = std::max(ymax, p[1]);
    }
    int y0 = static_cast<int>(std::ceil(ymin));
    int y1 = static_cast<int>(std::floor(ymax));

    for (int y = y0; y <= y1; ++y) {
        double ys = y + 0.5;  // sample at pixel centre
        std::vector<double> xs;
        for (std::size_t i = 0; i < n; ++i) {
            const Point2D& a = poly[i];
            const Point2D& b = poly[(i + 1) % n];
            double ya = a[1], yb = b[1];
            // half-open [min, max): include the lower vertex, exclude the upper
            if ((ya <= ys && yb > ys) || (yb <= ys && ya > ys)) {
                double t = (ys - ya) / (yb - ya);
                xs.push_back(a[0] + t * (b[0] - a[0]));
            }
        }
        std::sort(xs.begin(), xs.end());
        for (std::size_t k = 0; k + 1 < xs.size(); k += 2) {
            int xa = static_cast<int>(std::ceil(xs[k] - 0.5));
            int xb = static_cast<int>(std::floor(xs[k + 1] - 0.5));
            for (int x = xa; x <= xb; ++x) pixels.push_back({x, y});
        }
    }
    return pixels;
}

}  // namespace datamunge::algorithms
