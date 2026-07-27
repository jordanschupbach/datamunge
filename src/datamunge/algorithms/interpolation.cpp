#include <datamunge/algorithms/interpolation.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

double lagrange_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
    const std::size_t n   = xs.size();
    double            sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        double term = ys[j];
        for (std::size_t k = 0; k < n; ++k)
            if (k != j) term *= (x - xs[k]) / (xs[j] - xs[k]);
        sum += term;
    }
    return sum;
}

double neville_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
    const std::size_t   n = xs.size();
    std::vector<double> p(ys); // p[j] holds the running interpolant estimate ending at node j
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = n - 1; j >= i; --j) {
            p[j] = ((x - xs[j - i]) * p[j] - (x - xs[j]) * p[j - 1]) / (xs[j] - xs[j - i]);
            if (j == i) break; // avoid unsigned underflow
        }
    return n == 0 ? 0.0 : p[n - 1];
}

CubicSpline natural_cubic_spline(const std::vector<double>& xs, const std::vector<double>& ys) {
    const std::size_t n = xs.size();
    CubicSpline       s;
    s.xs = xs;
    if (n < 2) {
        s.a = ys;
        return s;
    }

    // Interval widths.
    std::vector<double> h(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) h[i] = xs[i + 1] - xs[i];

    // Symmetric tridiagonal system for the interior second-derivative moments m[1..n-2];
    // natural boundary conditions fix m[0] = m[n-1] = 0.
    std::vector<double> m(n, 0.0);
    if (n >= 3) {
        const std::size_t   k = n - 2; // number of interior unknowns
        std::vector<double> sub(k), diag(k), sup(k), rhs(k);
        for (std::size_t i = 0; i < k; ++i) {
            const std::size_t idx = i + 1; // node index 1..n-2
            sub[i]                = h[idx - 1];
            diag[i]               = 2.0 * (h[idx - 1] + h[idx]);
            sup[i]                = h[idx];
            rhs[i] = 6.0 * ((ys[idx + 1] - ys[idx]) / h[idx] - (ys[idx] - ys[idx - 1]) / h[idx - 1]);
        }
        // Thomas algorithm (forward elimination, back substitution).
        for (std::size_t i = 1; i < k; ++i) {
            const double w = sub[i] / diag[i - 1];
            diag[i] -= w * sup[i - 1];
            rhs[i] -= w * rhs[i - 1];
        }
        std::vector<double> sol(k);
        sol[k - 1] = rhs[k - 1] / diag[k - 1];
        for (std::size_t i = k - 1; i-- > 0;) sol[i] = (rhs[i] - sup[i] * sol[i + 1]) / diag[i];
        for (std::size_t i = 0; i < k; ++i) m[i + 1] = sol[i];
    }

    // Per-interval coefficients from the moments.
    s.a.resize(n - 1);
    s.b.resize(n - 1);
    s.c.resize(n - 1);
    s.d.resize(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        s.a[i] = ys[i];
        s.b[i] = (ys[i + 1] - ys[i]) / h[i] - h[i] * (2.0 * m[i] + m[i + 1]) / 6.0;
        s.c[i] = m[i] / 2.0;
        s.d[i] = (m[i + 1] - m[i]) / (6.0 * h[i]);
    }
    return s;
}

double cubic_spline_eval(const CubicSpline& spline, double x) {
    const std::size_t n = spline.xs.size();
    if (n == 0) return 0.0;
    if (n == 1) return spline.a.empty() ? 0.0 : spline.a[0];

    // Locate the interval: largest i with xs[i] <= x, clamped to [0, n-2].
    std::size_t i = 0;
    {
        const auto it = std::upper_bound(spline.xs.begin(), spline.xs.end(), x);
        const auto idx = static_cast<std::size_t>(it - spline.xs.begin());
        i = idx == 0 ? 0 : std::min(idx - 1, n - 2);
    }
    const double t = x - spline.xs[i];
    return spline.a[i] + t * (spline.b[i] + t * (spline.c[i] + t * spline.d[i]));
}

PlanarPoint de_casteljau(const std::vector<PlanarPoint>& control, double t) {
    std::vector<PlanarPoint> p(control);
    const std::size_t        n = p.size();
    for (std::size_t r = 1; r < n; ++r)
        for (std::size_t i = 0; i + r < n; ++i) {
            p[i].x = (1.0 - t) * p[i].x + t * p[i + 1].x;
            p[i].y = (1.0 - t) * p[i].y + t * p[i + 1].y;
        }
    return p.empty() ? PlanarPoint{} : p[0];
}

} // namespace datamunge::algorithms
