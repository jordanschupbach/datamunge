#include <gtest/gtest.h>

#include <datamunge/algorithms/interpolation.hpp>

#include <cmath>
#include <random>
#include <vector>

using datamunge::algorithms::CubicSpline;
using datamunge::algorithms::cubic_spline_eval;
using datamunge::algorithms::de_casteljau;
using datamunge::algorithms::lagrange_interpolate;
using datamunge::algorithms::natural_cubic_spline;
using datamunge::algorithms::neville_interpolate;
using datamunge::algorithms::PlanarPoint;

namespace {

double poly(double x) { return 2.0 * x * x * x - x + 1.0; } // degree 3

double binom(int n, int k) {
    double r = 1.0;
    for (int i = 0; i < k; ++i) r = r * (n - i) / (i + 1);
    return r;
}

PlanarPoint bernstein(const std::vector<PlanarPoint>& p, double t) {
    const int   n = static_cast<int>(p.size()) - 1;
    PlanarPoint out{0.0, 0.0};
    for (int i = 0; i <= n; ++i) {
        const double w = binom(n, i) * std::pow(t, i) * std::pow(1.0 - t, n - i);
        out.x += w * p[static_cast<std::size_t>(i)].x;
        out.y += w * p[static_cast<std::size_t>(i)].y;
    }
    return out;
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Lagrange & Neville
// ------------------------------------------------------------------------------------------------

TEST(PolynomialInterpolation, PassesThroughNodesAndReproducesPolynomial) {
    const std::vector<double> xs = {-2.0, -0.5, 1.0, 2.5, 4.0};
    std::vector<double>       ys;
    for (double x : xs) ys.push_back(poly(x));

    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(lagrange_interpolate(xs, ys, xs[i]), ys[i], 1e-9);
        EXPECT_NEAR(neville_interpolate(xs, ys, xs[i]), ys[i], 1e-9);
    }
    // Five nodes exactly determine the degree-3 polynomial, so interpolation is exact everywhere.
    std::mt19937                          rng(1);
    std::uniform_real_distribution<double> u(-3.0, 5.0);
    for (int t = 0; t < 1000; ++t) {
        const double x = u(rng);
        EXPECT_NEAR(lagrange_interpolate(xs, ys, x), poly(x), 1e-7);
        EXPECT_NEAR(neville_interpolate(xs, ys, x), poly(x), 1e-7);
        EXPECT_NEAR(lagrange_interpolate(xs, ys, x), neville_interpolate(xs, ys, x), 1e-9);
    }
}

// ------------------------------------------------------------------------------------------------
// Natural cubic spline
// ------------------------------------------------------------------------------------------------

TEST(CubicSpline, InterpolatesContinuityAndNaturalBoundary) {
    const std::vector<double> xs = {0.0, 1.0, 2.0, 3.5, 5.0, 7.0};
    std::vector<double>       ys = {0.0, 0.8, 0.9, 0.1, -0.8, -1.0};
    const CubicSpline         s  = natural_cubic_spline(xs, ys);

    // Interpolation: passes through every knot.
    for (std::size_t i = 0; i < xs.size(); ++i) EXPECT_NEAR(cubic_spline_eval(s, xs[i]), ys[i], 1e-9);

    const std::size_t segs = xs.size() - 1;
    // C1 and C2 continuity at interior knots.
    for (std::size_t i = 0; i + 1 < segs; ++i) {
        const double h  = xs[i + 1] - xs[i];
        const double d1_left  = s.b[i] + 2.0 * s.c[i] * h + 3.0 * s.d[i] * h * h;
        const double d1_right = s.b[i + 1];
        EXPECT_NEAR(d1_left, d1_right, 1e-9);                    // first derivative continuous
        const double d2_left  = 2.0 * s.c[i] + 6.0 * s.d[i] * h;
        const double d2_right = 2.0 * s.c[i + 1];
        EXPECT_NEAR(d2_left, d2_right, 1e-9);                    // second derivative continuous
    }
    // Natural boundary: S'' = 0 at both ends.
    EXPECT_NEAR(2.0 * s.c[0], 0.0, 1e-9);
    const double hn = xs[segs] - xs[segs - 1];
    EXPECT_NEAR(2.0 * s.c[segs - 1] + 6.0 * s.d[segs - 1] * hn, 0.0, 1e-9);
}

TEST(CubicSpline, ReproducesALine) {
    const std::vector<double> xs = {0.0, 1.0, 2.0, 4.0, 7.0};
    std::vector<double>       ys;
    for (double x : xs) ys.push_back(3.0 * x - 2.0);
    const CubicSpline s = natural_cubic_spline(xs, ys);
    std::mt19937                          rng(2);
    std::uniform_real_distribution<double> u(0.0, 7.0);
    for (int t = 0; t < 500; ++t) {
        const double x = u(rng);
        EXPECT_NEAR(cubic_spline_eval(s, x), 3.0 * x - 2.0, 1e-9);
    }
}

// ------------------------------------------------------------------------------------------------
// De Casteljau / Bezier
// ------------------------------------------------------------------------------------------------

TEST(DeCasteljau, EndpointsAndBernsteinAndHull) {
    const std::vector<PlanarPoint> ctrl = {{0.0, 0.0}, {1.0, 2.0}, {3.0, 3.0}, {4.0, 0.0}};

    // Endpoints: the curve interpolates the first and last control points.
    EXPECT_NEAR(de_casteljau(ctrl, 0.0).x, ctrl.front().x, 1e-12);
    EXPECT_NEAR(de_casteljau(ctrl, 0.0).y, ctrl.front().y, 1e-12);
    EXPECT_NEAR(de_casteljau(ctrl, 1.0).x, ctrl.back().x, 1e-12);
    EXPECT_NEAR(de_casteljau(ctrl, 1.0).y, ctrl.back().y, 1e-12);

    double xmin = ctrl[0].x, xmax = ctrl[0].x, ymin = ctrl[0].y, ymax = ctrl[0].y;
    for (const auto& p : ctrl) {
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
    }
    for (int k = 0; k <= 100; ++k) {
        const double      t  = k / 100.0;
        const PlanarPoint dc = de_casteljau(ctrl, t);
        const PlanarPoint bn = bernstein(ctrl, t);
        EXPECT_NEAR(dc.x, bn.x, 1e-9);           // matches the Bernstein form
        EXPECT_NEAR(dc.y, bn.y, 1e-9);
        EXPECT_GE(dc.x, xmin - 1e-12);           // stays within the control hull's bounding box
        EXPECT_LE(dc.x, xmax + 1e-12);
        EXPECT_GE(dc.y, ymin - 1e-12);
        EXPECT_LE(dc.y, ymax + 1e-12);
    }
}
