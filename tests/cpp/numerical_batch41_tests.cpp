#include <gtest/gtest.h>

#include <datamunge/algorithms/frank_wolfe.hpp>
#include <datamunge/algorithms/tricubic_interpolation.hpp>
#include <datamunge/geometry/icp.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

using namespace datamunge::algorithms;
using namespace datamunge::geometry;

// ---------------- ICP ----------------

namespace {
Point2D rot_trans(const Point2D& p, double th, double tx, double ty) {
    return {std::cos(th) * p.x - std::sin(th) * p.y + tx, std::sin(th) * p.x + std::cos(th) * p.y + ty};
}
double ang_diff(double a, double b) { return std::fmod(a - b + 3 * M_PI, 2 * M_PI) - M_PI; }
} // namespace

TEST(Icp, RecoversTransformWhenCorrespondenceUnambiguous) {
    // Well-separated points (grid, spacing 3) with a small transform, so nearest-
    // neighbor correspondence is the true one and ICP recovers it exactly. Target
    // is shuffled to prove ICP does not rely on input order.
    std::mt19937_64                        rng(4);
    std::uniform_real_distribution<double> jit(-0.4, 0.4);
    for (int trial = 0; trial < 30; ++trial) {
        std::vector<Point2D> src;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) src.push_back({i * 3.0 + jit(rng), j * 3.0 + jit(rng)});
        const double th = std::uniform_real_distribution<double>(-0.12, 0.12)(rng);
        const double tx = std::uniform_real_distribution<double>(-0.6, 0.6)(rng), ty = -tx * 0.7;
        std::vector<Point2D> tgt;
        for (const auto& p : src) tgt.push_back(rot_trans(p, th, tx, ty));
        std::shuffle(tgt.begin(), tgt.end(), rng);

        const auto r = icp(src, tgt, 100, 1e-15);
        EXPECT_NEAR(ang_diff(r.angle, th), 0.0, 1e-6) << "trial " << trial;
        EXPECT_NEAR(r.translation.x, tx, 1e-6) << "trial " << trial;
        EXPECT_NEAR(r.translation.y, ty, 1e-6) << "trial " << trial;
        EXPECT_LT(r.rmse, 1e-7);
    }
}

TEST(Icp, RmseIsNonIncreasing) {
    // The alignment error can never grow across ICP iterations.
    std::mt19937_64                        rng(8);
    std::uniform_real_distribution<double> pos(-5, 5);
    std::vector<Point2D>                   src, tgt;
    for (int i = 0; i < 20; ++i) src.push_back({pos(rng), pos(rng)});
    for (const auto& p : src) tgt.push_back(rot_trans(p, 0.3, 1.0, -0.5));
    double prev = 1e300;
    for (int iters = 1; iters <= 12; ++iters) {
        const auto r = icp(src, tgt, iters, 0.0); // force exactly `iters` sweeps
        EXPECT_LE(r.rmse, prev + 1e-9) << "iters " << iters;
        prev = r.rmse;
    }
}

// ---------------- Frank-Wolfe ----------------

TEST(FrankWolfe, ConvergesToInteriorSimplexOptimum) {
    // minimize 0.5 |x - c|^2 over the simplex; c is on the simplex so optimum = c.
    const std::vector<double> c{0.5, 0.2, 0.3};
    auto f    = [&](const std::vector<double>& x) {
        double s = 0; for (std::size_t i = 0; i < x.size(); ++i) s += (x[i] - c[i]) * (x[i] - c[i]); return 0.5 * s;
    };
    auto grad = [&](const std::vector<double>& x) {
        std::vector<double> g(x.size()); for (std::size_t i = 0; i < x.size(); ++i) g[i] = x[i] - c[i]; return g;
    };
    const auto res = frank_wolfe(f, grad, simplex_lmo, std::vector<double>{1, 0, 0}, 8000);
    // objective drives to zero and the iterate approaches c
    EXPECT_LT(res.objective.back(), 1e-4);
    for (std::size_t i = 0; i < c.size(); ++i) EXPECT_NEAR(res.x[i], c[i], 2e-2) << "coord " << i;
    // feasibility: nonnegative and sums to one
    const double sum = std::accumulate(res.x.begin(), res.x.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-9);
    for (double xi : res.x) EXPECT_GE(xi, -1e-12);
}

TEST(FrankWolfe, ObjectiveDecreasesAtOneOverK) {
    const std::vector<double> c{0.6, 0.1, 0.1, 0.2};
    auto f    = [&](const std::vector<double>& x) {
        double s = 0; for (std::size_t i = 0; i < x.size(); ++i) s += (x[i] - c[i]) * (x[i] - c[i]); return 0.5 * s;
    };
    auto grad = [&](const std::vector<double>& x) {
        std::vector<double> g(x.size()); for (std::size_t i = 0; i < x.size(); ++i) g[i] = x[i] - c[i]; return g;
    };
    const auto res = frank_wolfe(f, grad, simplex_lmo, std::vector<double>{1, 0, 0, 0}, 2000);
    // gap(2000) should be well below gap(100) by roughly the ratio of iterations
    const double g100 = res.objective[100], g2000 = res.objective[2000];
    EXPECT_LT(g2000, g100);
    EXPECT_LT(g2000, g100 * 0.1); // ~O(1/k): 20x more iterations -> much smaller gap
}

// ---------------- Tricubic interpolation ----------------

namespace {
std::vector<double> make_vol(const std::vector<double>& xs, const std::vector<double>& ys,
                             const std::vector<double>& zs, double (*f)(double, double, double)) {
    std::vector<double> v(xs.size() * ys.size() * zs.size());
    for (std::size_t i = 0; i < xs.size(); ++i)
        for (std::size_t j = 0; j < ys.size(); ++j)
            for (std::size_t k = 0; k < zs.size(); ++k)
                v[(i * ys.size() + j) * zs.size() + k] = f(xs[i], ys[j], zs[k]);
    return v;
}
} // namespace

TEST(Tricubic, InterpolatesGridNodesExactly) {
    std::vector<double> xs, ys, zs;
    for (int i = 0; i < 7; ++i) xs.push_back(i * 0.5);
    for (int j = 0; j < 6; ++j) ys.push_back(j * 0.5);
    for (int k = 0; k < 5; ++k) zs.push_back(k * 0.5);
    auto f = [](double x, double y, double z) { return std::sin(x) * std::cos(y) + 0.4 * z - 0.1 * x * z; };
    const auto v = make_vol(xs, ys, zs, +f);
    for (std::size_t i = 0; i < xs.size(); ++i)
        for (std::size_t j = 0; j < ys.size(); ++j)
            for (std::size_t k = 0; k < zs.size(); ++k)
                EXPECT_NEAR(tricubic_interpolate(xs, ys, zs, v, xs[i], ys[j], zs[k]),
                            v[(i * ys.size() + j) * zs.size() + k], 1e-9);
}

TEST(Tricubic, ExactForTrilinearInInterior) {
    std::vector<double> xs, ys, zs;
    for (int i = 0; i < 9; ++i) { xs.push_back(i); ys.push_back(i); zs.push_back(i); }
    auto f = [](double x, double y, double z) {
        return 1.0 + 2 * x - 3 * y + 0.5 * z + 0.7 * x * y - 0.2 * y * z + 0.1 * x * z + 0.05 * x * y * z;
    };
    const auto v = make_vol(xs, ys, zs, +f);
    for (double x = 1.3; x < 6.5; x += 0.7)
        for (double y = 1.4; y < 6.5; y += 0.7)
            for (double z = 1.2; z < 6.5; z += 0.7)
                EXPECT_NEAR(tricubic_interpolate(xs, ys, zs, v, x, y, z), f(x, y, z), 1e-8)
                    << x << "," << y << "," << z;
}
