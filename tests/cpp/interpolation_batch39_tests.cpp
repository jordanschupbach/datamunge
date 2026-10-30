#include <gtest/gtest.h>

#include <datamunge/algorithms/bicubic_interpolation.hpp>
#include <datamunge/algorithms/birkhoff_interpolation.hpp>
#include <datamunge/algorithms/interpolation_extra.hpp>
#include <datamunge/algorithms/lanczos_resampling.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::algorithms;

// ---------------- Bicubic interpolation ----------------

namespace {
std::vector<double> make_grid(const std::vector<double>& xs, const std::vector<double>& ys,
                              double (*f)(double, double)) {
    std::vector<double> z(xs.size() * ys.size());
    for (std::size_t i = 0; i < xs.size(); ++i)
        for (std::size_t j = 0; j < ys.size(); ++j) z[i * ys.size() + j] = f(xs[i], ys[j]);
    return z;
}
} // namespace

TEST(Bicubic, InterpolatesGridNodesExactly) {
    std::vector<double> xs, ys;
    for (int i = 0; i < 8; ++i) xs.push_back(i * 0.5);
    for (int j = 0; j < 7; ++j) ys.push_back(j * 0.5);
    auto f = [](double x, double y) { return std::sin(x) * std::cos(y) + 0.3 * x; };
    const auto z = make_grid(xs, ys, +f);
    for (std::size_t i = 0; i < xs.size(); ++i)
        for (std::size_t j = 0; j < ys.size(); ++j)
            EXPECT_NEAR(bicubic_interpolate(xs, ys, z, xs[i], ys[j]), z[i * ys.size() + j], 1e-9);
}

TEST(Bicubic, ExactForBilinearFunctions) {
    // Cubic convolution reproduces bilinear functions exactly wherever the full 4x4
    // stencil lies inside the grid (i.e. away from the clamped border).
    std::vector<double> xs, ys;
    for (int i = 0; i < 9; ++i) xs.push_back(i);
    for (int j = 0; j < 9; ++j) ys.push_back(j);
    auto f = [](double x, double y) { return 2.0 + 3.0 * x - 1.5 * y + 0.7 * x * y; };
    const auto z = make_grid(xs, ys, +f);
    for (double x = 1.2; x < 6.5; x += 0.31) // interior: base cell in [1, nx-3]
        for (double y = 1.2; y < 6.5; y += 0.29)
            EXPECT_NEAR(bicubic_interpolate(xs, ys, z, x, y), f(x, y), 1e-9) << "x=" << x << " y=" << y;
}

TEST(Bicubic, MoreAccurateThanBilinearOnSmoothData) {
    std::vector<double> xs, ys;
    for (int i = 0; i < 12; ++i) xs.push_back(i * 0.4);
    for (int j = 0; j < 12; ++j) ys.push_back(j * 0.4);
    auto f = [](double x, double y) { return std::exp(-0.1 * (x * x + y * y)) * std::cos(x + y); };
    const auto z = make_grid(xs, ys, +f);
    double bic_err = 0, bil_err = 0;
    for (double x = 0.5; x < 4.0; x += 0.13)
        for (double y = 0.5; y < 4.0; y += 0.13) {
            const double truth = f(x, y);
            bic_err = std::max(bic_err, std::fabs(bicubic_interpolate(xs, ys, z, x, y) - truth));
            // simple bilinear for comparison
            const std::size_t ny = ys.size();
            const double hx = 0.4, hy = 0.4;
            const long   ix = static_cast<long>((x - xs[0]) / hx), iy = static_cast<long>((y - ys[0]) / hy);
            const double tx = (x - xs[ix]) / hx, ty = (y - ys[iy]) / hy;
            const double v = (1 - tx) * (1 - ty) * z[ix * ny + iy] + tx * (1 - ty) * z[(ix + 1) * ny + iy] +
                             (1 - tx) * ty * z[ix * ny + (iy + 1)] + tx * ty * z[(ix + 1) * ny + (iy + 1)];
            bil_err = std::max(bil_err, std::fabs(v - truth));
        }
    EXPECT_LT(bic_err, bil_err) << "bicubic " << bic_err << " vs bilinear " << bil_err;
}

// ---------------- Lanczos resampling ----------------

TEST(Lanczos, PassesThroughIntegerSamples) {
    std::vector<double> s{0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0};
    for (int k = 0; k < static_cast<int>(s.size()); ++k)
        EXPECT_NEAR(lanczos_resample_at(s, static_cast<double>(k), 3), s[k], 1e-9) << "k=" << k;
}

TEST(Lanczos, MoreAccurateThanLinearOnSmoothSignal) {
    // Sample a smooth low-frequency signal on integers, resample between them.
    const int           n = 40;
    std::vector<double> s(n);
    auto f = [](double t) { return std::sin(0.35 * t) + 0.5 * std::cos(0.2 * t); };
    for (int i = 0; i < n; ++i) s[i] = f(i);
    double lanc_err = 0, lin_err = 0;
    for (double p = 5.0; p < n - 5.0; p += 0.137) {
        const double truth = f(p);
        lanc_err           = std::max(lanc_err, std::fabs(lanczos_resample_at(s, p, 3) - truth));
        const int    i     = static_cast<int>(std::floor(p));
        const double t     = p - i;
        const double lin   = (1 - t) * s[i] + t * s[i + 1];
        lin_err            = std::max(lin_err, std::fabs(lin - truth));
    }
    EXPECT_LT(lanc_err, lin_err) << "lanczos " << lanc_err << " vs linear " << lin_err;
}

// ---------------- Birkhoff interpolation ----------------

TEST(Birkhoff, ReconstructsKnownPolynomialFromHermiteData) {
    // p(x) = 2 - x + 0.5 x^2 - 0.25 x^3 ; give value+first-derivative at two nodes.
    auto p  = [](double x) { return 2 - x + 0.5 * x * x - 0.25 * x * x * x; };
    auto dp = [](double x) { return -1 + x - 0.75 * x * x; };
    std::vector<BirkhoffConstraint> cons{
        {0.0, 0, p(0.0)}, {0.0, 1, dp(0.0)}, {2.0, 0, p(2.0)}, {2.0, 1, dp(2.0)}};
    const auto c = birkhoff_polynomial(cons);
    ASSERT_EQ(c.size(), 4u);
    for (double x = -1.0; x <= 3.0; x += 0.25) EXPECT_NEAR(birkhoff_eval(c, x), p(x), 1e-9) << "x=" << x;
}

TEST(Birkhoff, LacunarySchemeValueAndSecondDerivative) {
    // Genuine Birkhoff (no first-derivative data): p(x0), p''(x0), p(x1), p''(x1).
    auto p   = [](double x) { return 1 + 2 * x + 3 * x * x + 4 * x * x * x; };
    auto d2p = [](double x) { return 6 + 24 * x; };
    std::vector<BirkhoffConstraint> cons{
        {-1.0, 0, p(-1.0)}, {-1.0, 2, d2p(-1.0)}, {1.0, 0, p(1.0)}, {1.0, 2, d2p(1.0)}};
    const auto c = birkhoff_polynomial(cons);
    ASSERT_EQ(c.size(), 4u);
    for (double x = -1.5; x <= 1.5; x += 0.3) EXPECT_NEAR(birkhoff_eval(c, x), p(x), 1e-9) << "x=" << x;
}

TEST(Birkhoff, DetectsNonPoisedScheme) {
    // p(x0)=.., p''(x0)=.., p''(x1)=.., p''(x2)=.. with a symmetric node layout that is
    // singular: too many second-derivative constraints for a low-degree fit. Here we use a
    // classic singular pattern -- second derivatives at symmetric nodes with no value spread.
    std::vector<BirkhoffConstraint> cons{{0.0, 1, 0.0}, {-1.0, 2, 0.0}, {1.0, 2, 0.0}};
    // Degree-2 polynomial from {p'(0), p''(-1), p''(1)}: p''(-1)=p''(1)=2c2 are the same equation
    // (identical rows) -> singular.
    const auto c = birkhoff_polynomial(cons);
    EXPECT_TRUE(c.empty());
}

// ---------------- Cubic-convolution kernel and grid resampling ----------------

TEST(CubicConvolution, KernelInterpolatesAndReproducesPolynomials) {
    for (double a : {-0.5, -0.75, -1.0}) {
        EXPECT_DOUBLE_EQ(cubic_convolution_kernel(0.0, a), 1.0);
        for (int k : {-2, -1, 1, 2, 3}) EXPECT_NEAR(cubic_convolution_kernel(k, a), 0.0, 1e-15);
        for (double t : {0.1, 0.37, 0.5, 0.9}) {
            double s0 = 0, s1 = 0, s2 = 0;
            for (int m = -1; m <= 2; ++m) {
                const double w = cubic_convolution_kernel(t - m, a);
                s0 += w, s1 += w * m, s2 += w * m * m;
            }
            EXPECT_NEAR(s0, 1.0, 1e-14);  // constants, for every a
            // Linear functions only for a = -1/2: the bias is -(2a+1) t(1-t)(1-2t).
            EXPECT_NEAR(s1 - t, -(2 * a + 1) * t * (1 - t) * (1 - 2 * t), 1e-14);
            if (a == -0.5) EXPECT_NEAR(s2, t * t, 1e-14);  // and quadratics
        }
    }
}

TEST(CubicConvolution, ConvergenceOrderIsThreeOnlyForKeysParameter) {
    auto max_error = [](double h, double a) {
        std::vector<double> xs, ys;
        for (double v = -1.0; v <= 3.0 + 1e-12; v += h) xs.push_back(v), ys.push_back(v);
        std::vector<double> z;
        for (double x : xs)
            for (double y : ys) z.push_back(std::sin(x) * std::cos(1.3 * y));
        double err = 0;
        for (double x = 0.0; x <= 2.0; x += 0.0137)
            for (double y = 0.0; y <= 2.0; y += 0.0291)
                err = std::max(err, std::fabs(bicubic_interpolate(xs, ys, z, x, y, a) - std::sin(x) * std::cos(1.3 * y)));
        return err;
    };
    EXPECT_NEAR(std::log2(max_error(0.1, -0.5) / max_error(0.05, -0.5)), 3.0, 0.3);
    EXPECT_NEAR(std::log2(max_error(0.1, -0.75) / max_error(0.05, -0.75)), 1.0, 0.3);
}

TEST(ResampleGrid, MethodsAgreeWithPointwiseRulesAndHitNodes) {
    std::vector<double> xs, ys;
    for (int i = 0; i < 6; ++i) xs.push_back(i * 2.0);
    for (int j = 0; j < 5; ++j) ys.push_back(1.0 + j);
    std::vector<double> z;
    for (double x : xs)
        for (double y : ys) z.push_back(std::exp(-0.1 * x) + y * y);
    const std::vector<double> nx = {0.0, 1.0, 3.3, 9.9}, ny = {1.0, 2.6, 4.9};
    const auto bic = resample_grid(xs, ys, z, nx, ny, GridInterpolation::Bicubic);
    const auto bil = resample_grid(xs, ys, z, nx, ny, GridInterpolation::Bilinear);
    const auto near = resample_grid(xs, ys, z, nx, ny, GridInterpolation::Nearest);
    ASSERT_EQ(bic.size(), 12u);
    for (std::size_t i = 0; i < nx.size(); ++i)
        for (std::size_t j = 0; j < ny.size(); ++j) {
            EXPECT_DOUBLE_EQ(bic[i * 3 + j], bicubic_interpolate(xs, ys, z, nx[i], ny[j]));
            EXPECT_DOUBLE_EQ(bil[i * 3 + j], bilinear_interpolate(xs, ys, z, nx[i], ny[j]));
        }
    EXPECT_DOUBLE_EQ(near[2 * 3 + 1], z[2 * 5 + 2]);  // (3.3, 2.6) -> node (4, 3)
    const auto same = resample_grid(xs, ys, z, xs, ys, GridInterpolation::Bicubic);
    for (std::size_t k = 0; k < z.size(); ++k) EXPECT_NEAR(same[k], z[k], 1e-12);
    EXPECT_THROW(resample_grid({1.0}, ys, z, nx, ny, GridInterpolation::Nearest), std::invalid_argument);
    EXPECT_THROW(resample_grid(xs, ys, {1.0}, nx, ny, GridInterpolation::Nearest), std::invalid_argument);
}
