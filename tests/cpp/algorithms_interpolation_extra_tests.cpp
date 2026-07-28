#include <gtest/gtest.h>

#include <datamunge/algorithms/interpolation_extra.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::algorithms;

TEST(InterpExtra, LinearIsExactOnAffineData) {
    // ys = 3x - 1 sampled at irregular nodes; linear interp must reproduce it everywhere.
    const std::vector<double> xs{0.0, 1.0, 2.5, 4.0, 7.0};
    std::vector<double>       ys;
    for (double x : xs) ys.push_back(3.0 * x - 1.0);
    for (double x : {0.3, 1.0, 1.7, 3.2, 6.5}) EXPECT_NEAR(linear_interpolate(xs, ys, x), 3.0 * x - 1.0, 1e-12);
    // reproduces nodes and midpoints
    EXPECT_NEAR(linear_interpolate(xs, ys, 2.5), 3.0 * 2.5 - 1.0, 1e-12);
    EXPECT_NEAR(linear_interpolate(xs, ys, 0.5), 0.5 * (ys[0] + ys[1]), 1e-12);
    // linear extrapolation beyond the ends
    EXPECT_NEAR(linear_interpolate(xs, ys, -1.0), 3.0 * -1.0 - 1.0, 1e-12);
    EXPECT_NEAR(linear_interpolate(xs, ys, 9.0), 3.0 * 9.0 - 1.0, 1e-12);
}

TEST(InterpExtra, HermiteReproducesNodesAndSlopes) {
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0};
    std::vector<double>       ys, ms;
    // Use an exact cubic g(x) = x^3 - 2x + 1, g'(x) = 3x^2 - 2.
    auto g  = [](double x) { return x * x * x - 2 * x + 1; };
    auto dg = [](double x) { return 3 * x * x - 2; };
    for (double x : xs) { ys.push_back(g(x)); ms.push_back(dg(x)); }
    // Hermite with exact nodal values+slopes reproduces the cubic exactly on every interval.
    for (double x : {0.25, 0.5, 1.3, 2.0, 2.9}) EXPECT_NEAR(hermite_interpolate(xs, ys, ms, x), g(x), 1e-10);
    // exact at nodes
    for (std::size_t i = 0; i < xs.size(); ++i) EXPECT_NEAR(hermite_interpolate(xs, ys, ms, xs[i]), ys[i], 1e-12);
    // numerical derivative near a node matches the specified slope
    const double x0 = 1.0, e = 1e-6;
    const double d  = (hermite_interpolate(xs, ys, ms, x0 + e) - hermite_interpolate(xs, ys, ms, x0 - e)) / (2 * e);
    EXPECT_NEAR(d, dg(x0), 1e-4);
}

TEST(InterpExtra, MonotoneCubicStaysMonotoneWithoutOvershoot) {
    // A monotone step-like dataset that makes a natural cubic spline overshoot.
    const std::vector<double> xs{0, 1, 2, 3, 4, 5, 6};
    const std::vector<double> ys{0, 0, 0, 1, 1, 1, 1};
    double prev = monotone_cubic_interpolate(xs, ys, 0.0);
    double gmin = prev, gmax = prev;
    for (double x = 0.0; x <= 6.0; x += 0.02) {
        const double v = monotone_cubic_interpolate(xs, ys, x);
        EXPECT_GE(v + 1e-12, prev) << "not monotone at x=" << x; // nondecreasing
        gmin = std::min(gmin, v);
        gmax = std::max(gmax, v);
        prev = v;
    }
    // no overshoot beyond the data range [0,1]
    EXPECT_GE(gmin, -1e-9);
    EXPECT_LE(gmax, 1.0 + 1e-9);
    // reproduces the nodes
    for (std::size_t i = 0; i < xs.size(); ++i) EXPECT_NEAR(monotone_cubic_interpolate(xs, ys, xs[i]), ys[i], 1e-12);
}

TEST(InterpExtra, BilinearExactOnBilinearFunctionAndCorners) {
    // Grid over x in {0,1,3}, y in {0,2,5}; z = a + b x + c y + d x y (exact for bilinear).
    const std::vector<double> xs{0.0, 1.0, 3.0}, ys{0.0, 2.0, 5.0};
    const double a = 1.0, b = -2.0, c = 0.5, d = 0.3;
    auto f = [&](double x, double y) { return a + b * x + c * y + d * x * y; };
    std::vector<double> z;
    for (double x : xs)
        for (double y : ys) z.push_back(f(x, y)); // row-major z[i*ny + j]

    // exact at grid corners
    for (std::size_t i = 0; i < xs.size(); ++i)
        for (std::size_t j = 0; j < ys.size(); ++j)
            EXPECT_NEAR(bilinear_interpolate(xs, ys, z, xs[i], ys[j]), f(xs[i], ys[j]), 1e-12);
    // exact everywhere for a bilinear function
    for (double x : {0.4, 1.5, 2.9})
        for (double y : {0.7, 3.1, 4.8}) EXPECT_NEAR(bilinear_interpolate(xs, ys, z, x, y), f(x, y), 1e-10);
    // reduces to linear interpolation along a grid line (fixed y = ys[0])
    EXPECT_NEAR(bilinear_interpolate(xs, ys, z, 0.5, 0.0), 0.5 * (f(0, 0) + f(1, 0)), 1e-12);
}
