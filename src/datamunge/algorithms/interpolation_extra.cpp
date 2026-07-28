#include <datamunge/algorithms/interpolation_extra.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

namespace {

// Index k of the interval [xs[k], xs[k+1]] containing x (clamped to a valid interval).
std::size_t locate(const std::vector<double>& xs, double x) {
    if (x <= xs.front()) return 0;
    if (x >= xs[xs.size() - 2]) return xs.size() - 2;
    // first node strictly greater than x, then step back one
    const auto it = std::upper_bound(xs.begin(), xs.end(), x);
    return static_cast<std::size_t>(it - xs.begin()) - 1;
}

// Cubic Hermite value on [x0,x1] given endpoint values/slopes.
double hermite_segment(double x, double x0, double x1, double y0, double y1, double m0, double m1) {
    const double h  = x1 - x0;
    const double t  = (x - x0) / h;
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double h00 = 2 * t3 - 3 * t2 + 1;
    const double h10 = t3 - 2 * t2 + t;
    const double h01 = -2 * t3 + 3 * t2;
    const double h11 = t3 - t2;
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;
}

} // namespace

double linear_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
    const std::size_t k = locate(xs, x);
    const double      t = (x - xs[k]) / (xs[k + 1] - xs[k]); // may be <0 or >1 -> linear extrapolation
    return ys[k] + t * (ys[k + 1] - ys[k]);
}

double hermite_interpolate(const std::vector<double>& xs, const std::vector<double>& ys,
                           const std::vector<double>& slopes, double x) {
    const std::size_t k = locate(xs, x);
    return hermite_segment(x, xs[k], xs[k + 1], ys[k], ys[k + 1], slopes[k], slopes[k + 1]);
}

double monotone_cubic_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
    const std::size_t n = xs.size();
    // secant slopes
    std::vector<double> delta(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) delta[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);

    // initial tangents: average of neighbouring secants (one-sided at the ends)
    std::vector<double> m(n);
    m[0]     = delta[0];
    m[n - 1] = delta[n - 2];
    for (std::size_t i = 1; i + 1 < n; ++i) m[i] = 0.5 * (delta[i - 1] + delta[i]);

    // Fritsch-Carlson limiter: keep the interpolant monotone on each interval
    for (std::size_t i = 0; i + 1 < n; ++i) {
        if (delta[i] == 0.0) { // flat segment -> flat tangents, avoids overshoot
            m[i]     = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        const double alpha = m[i] / delta[i];
        const double beta  = m[i + 1] / delta[i];
        // opposite sign to the secant -> not monotone; clamp to zero
        if (alpha < 0.0) m[i] = 0.0;
        if (beta < 0.0) m[i + 1] = 0.0;
        const double s = alpha * alpha + beta * beta;
        if (s > 9.0) {
            const double tau = 3.0 / std::sqrt(s);
            m[i]             = tau * alpha * delta[i];
            m[i + 1]         = tau * beta * delta[i];
        }
    }

    const std::size_t k = locate(xs, x);
    return hermite_segment(x, xs[k], xs[k + 1], ys[k], ys[k + 1], m[k], m[k + 1]);
}

double bilinear_interpolate(const std::vector<double>& xs, const std::vector<double>& ys,
                            const std::vector<double>& z, double x, double y) {
    const std::size_t ny = ys.size();
    const std::size_t i  = locate(xs, x);
    const std::size_t j  = locate(ys, y);
    const double      tx = (x - xs[i]) / (xs[i + 1] - xs[i]);
    const double      ty = (y - ys[j]) / (ys[j + 1] - ys[j]);

    const double z00 = z[i * ny + j];
    const double z10 = z[(i + 1) * ny + j];
    const double z01 = z[i * ny + (j + 1)];
    const double z11 = z[(i + 1) * ny + (j + 1)];

    const double a = z00 + tx * (z10 - z00); // interpolate in x along the two rows
    const double b = z01 + tx * (z11 - z01);
    return a + ty * (b - a); // then in y
}

} // namespace datamunge::algorithms
