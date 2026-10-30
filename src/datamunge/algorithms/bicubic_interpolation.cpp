/// @file bicubic_interpolation.cpp
/// @brief Cubic-convolution kernel, bicubic interpolation and grid resampling.

#include <datamunge/algorithms/bicubic_interpolation.hpp>
#include <datamunge/algorithms/interpolation_extra.hpp>

#include <stdexcept>

namespace datamunge::algorithms {

/// Evaluates the two cubic pieces in Horner form.
double cubic_convolution_kernel(double t, double a) {
    t = std::fabs(t);
    if (t <= 1.0) return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    if (t < 2.0) return (((t - 5.0) * t + 8.0) * t - 4.0) * a;
    return 0.0;
}

/// Separable 4x4 convolution with edge clamping; the four y-weights are computed once.
double bicubic_interpolate(const std::vector<double>& xs, const std::vector<double>& ys,
                           const std::vector<double>& z, double x, double y, double a) {
    const std::size_t nx = xs.size(), ny = ys.size();
    const double      hx = xs[1] - xs[0], hy = ys[1] - ys[0];
    const double      fx = (x - xs[0]) / hx, fy = (y - ys[0]) / hy;
    const long        ix = static_cast<long>(std::floor(fx)), iy = static_cast<long>(std::floor(fy));
    const double      tx = fx - static_cast<double>(ix), ty = fy - static_cast<double>(iy);
    double            wy[4];
    std::size_t       gj[4];
    for (int n = -1; n <= 2; ++n) {
        wy[n + 1] = cubic_convolution_kernel(ty - n, a);
        gj[n + 1] = detail::bicubic_clamp(iy + n, ny);
    }
    double result = 0.0;
    for (int m = -1; m <= 2; ++m) {
        const double      wx = cubic_convolution_kernel(tx - m, a);
        const std::size_t gi = detail::bicubic_clamp(ix + m, nx);
        double            row = 0.0;
        for (int n = 0; n < 4; ++n) row += wy[n] * z[gi * ny + gj[n]];
        result += wx * row;
    }
    return result;
}

/// Evaluates the chosen rule at every target node; nearest rounds to the closest node index.
std::vector<double> resample_grid(const std::vector<double>& xs, const std::vector<double>& ys,
                                  const std::vector<double>& z, const std::vector<double>& new_xs,
                                  const std::vector<double>& new_ys, GridInterpolation method, double a) {
    if (xs.size() < 2 || ys.size() < 2) throw std::invalid_argument("resample_grid: each axis needs at least 2 lines");
    if (z.size() != xs.size() * ys.size()) throw std::invalid_argument("resample_grid: z must have xs.size() * ys.size() values");
    const std::size_t ny = ys.size();
    const double      hx = xs[1] - xs[0], hy = ys[1] - ys[0];
    std::vector<double> out;
    out.reserve(new_xs.size() * new_ys.size());
    for (const double x : new_xs) {
        for (const double y : new_ys) {
            switch (method) {
            case GridInterpolation::Nearest: {
                const auto i = detail::bicubic_clamp(std::lround((x - xs[0]) / hx), xs.size());
                const auto j = detail::bicubic_clamp(std::lround((y - ys[0]) / hy), ny);
                out.push_back(z[i * ny + j]);
                break;
            }
            case GridInterpolation::Bilinear: out.push_back(bilinear_interpolate(xs, ys, z, x, y)); break;
            case GridInterpolation::Bicubic: out.push_back(bicubic_interpolate(xs, ys, z, x, y, a)); break;
            }
        }
    }
    return out;
}

} // namespace datamunge::algorithms
