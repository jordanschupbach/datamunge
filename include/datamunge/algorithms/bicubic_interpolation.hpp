#pragma once

// Bicubic interpolation on a regular 2-D grid via Keys' cubic convolution.
// Where bilinear interpolation blends a 2x2 neighborhood with linear weights,
// bicubic convolution blends a 4x4 neighborhood with a piecewise-cubic kernel,
//
//   w(t) = (a+2)|t|^3 - (a+3)|t|^2 + 1          for |t| <= 1
//        = a|t|^3 - 5a|t|^2 + 8a|t| - 4a         for 1 < |t| < 2
//        = 0                                      otherwise,
//
// with a = -1/2 (Keys, 1981), the choice that makes the interpolant third-order
// accurate. The kernel is separable, so the 2-D weight is w(tx) * w(ty). The
// result passes exactly through the grid samples, reproduces bilinear functions
// exactly, and is markedly smoother (C^1) than bilinear -- the standard quality
// image/texture upsampler. The grid must be uniformly spaced in each axis.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

// Keys cubic-convolution kernel with a = -1/2.
inline double bicubic_kernel(double t) {
    constexpr double a = -0.5;
    t                  = std::fabs(t);
    if (t <= 1.0) return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    if (t < 2.0) return (((t - 5.0) * t + 8.0) * t - 4.0) * a;
    return 0.0;
}

// Clamp a grid index into [0, n-1] so queries near the border stay in range.
inline std::size_t bicubic_clamp(long i, std::size_t n) {
    if (i < 0) return 0;
    if (i >= static_cast<long>(n)) return n - 1;
    return static_cast<std::size_t>(i);
}

} // namespace detail

// Interpolate the value at (x, y). `z[i*ny + j]` is the sample at (xs[i], ys[j]),
// row-major; `xs`/`ys` are uniformly spaced and strictly increasing.
inline double bicubic_interpolate(const std::vector<double>& xs, const std::vector<double>& ys,
                                  const std::vector<double>& z, double x, double y) {
    const std::size_t nx = xs.size(), ny = ys.size();
    const double      hx = xs[1] - xs[0], hy = ys[1] - ys[0];

    // Base cell and fractional offsets.
    const double fx = (x - xs[0]) / hx, fy = (y - ys[0]) / hy;
    const long   ix = static_cast<long>(std::floor(fx)), iy = static_cast<long>(std::floor(fy));
    const double tx = fx - static_cast<double>(ix), ty = fy - static_cast<double>(iy);

    // Separable 4x4 convolution over neighbors ix-1..ix+2, iy-1..iy+2.
    double result = 0.0;
    for (int m = -1; m <= 2; ++m) {
        const double      wx = detail::bicubic_kernel(tx - static_cast<double>(m));
        const std::size_t gi = detail::bicubic_clamp(ix + m, nx);
        double            row = 0.0;
        for (int n = -1; n <= 2; ++n) {
            const double      wy = detail::bicubic_kernel(ty - static_cast<double>(n));
            const std::size_t gj = detail::bicubic_clamp(iy + n, ny);
            row += wy * z[gi * ny + gj];
        }
        result += wx * row;
    }
    return result;
}

} // namespace datamunge::algorithms
