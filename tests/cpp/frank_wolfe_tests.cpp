#include <gtest/gtest.h>

#include <datamunge/algorithms/frank_wolfe.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

namespace {

// 0.5 |x - c|^2 over the triangle (0,0), (1,0), (0,1); c = (0.7, 0.7) projects to (0.5, 0.5) on
// the hypotenuse, so the optimum lies on a face and f* = 0.04.
const std::vector<std::vector<double>> kTriangle = {{0, 0}, {1, 0}, {0, 1}};
double tri_f(const std::vector<double>& x) { return 0.5 * ((x[0] - 0.7) * (x[0] - 0.7) + (x[1] - 0.7) * (x[1] - 0.7)); }
std::vector<double> tri_g(const std::vector<double>& x) { return {x[0] - 0.7, x[1] - 0.7}; }
std::vector<double> tri_lmo(const std::vector<double>& g) { return vertex_lmo(g, kTriangle); }

FrankWolfeResult run_triangle(FrankWolfeVariant v, FrankWolfeStep s, int iters) {
    FrankWolfeOptions o;
    o.variant   = v;
    o.step      = s;
    o.lipschitz = 1.0;
    o.max_iter  = iters;
    o.tolerance = 1e-12;
    return frank_wolfe(tri_f, tri_g, tri_lmo, {0.0, 0.0}, o);
}

} // namespace

TEST(FrankWolfeVariants, GapBoundsPrimalError) {
    for (const auto v : {FrankWolfeVariant::Vanilla, FrankWolfeVariant::AwayStep, FrankWolfeVariant::Pairwise}) {
        const auto r = run_triangle(v, FrankWolfeStep::LineSearch, 300);
        ASSERT_EQ(r.gap.size(), r.objective.size());
        for (std::size_t k = 0; k < r.gap.size(); ++k) EXPECT_GE(r.gap[k] + 1e-12, r.objective[k] - 0.04);
        double sum = 0;
        for (const double w : r.weights) {
            sum += w;
            EXPECT_GE(w, -1e-12);
        }
        EXPECT_NEAR(sum, 1.0, 1e-9);
    }
}

TEST(FrankWolfeVariants, AwayAndPairwiseConvergeFastOnAFace) {
    const auto vanilla = run_triangle(FrankWolfeVariant::Vanilla, FrankWolfeStep::LineSearch, 2000);
    EXPECT_FALSE(vanilla.converged);
    EXPECT_GT(vanilla.objective.back() - 0.04, 1e-9); // still zig-zagging
    for (const auto v : {FrankWolfeVariant::AwayStep, FrankWolfeVariant::Pairwise})
        for (const auto s : {FrankWolfeStep::LineSearch, FrankWolfeStep::ShortStep}) {
            const auto r = run_triangle(v, s, 2000);
            EXPECT_TRUE(r.converged) << int(v) << int(s);
            EXPECT_LT(r.iterations, 50);
            EXPECT_NEAR(r.x[0], 0.5, 1e-6);
            EXPECT_NEAR(r.x[1], 0.5, 1e-6);
            EXPECT_EQ(r.atoms.size(), 2u); // the start vertex (0,0) was dropped
        }
}

TEST(FrankWolfeVariants, StandardStepRateAndErrors) {
    const auto r = run_triangle(FrankWolfeVariant::Vanilla, FrankWolfeStep::Standard, 4000);
    // O(1/k): k (f - f*) stays bounded by 2 L diam^2 = 4.
    for (std::size_t k = 1; k < r.objective.size(); ++k) EXPECT_LE(double(k) * (r.objective[k] - 0.04), 4.0);
    FrankWolfeOptions bad;
    bad.variant = FrankWolfeVariant::AwayStep;
    EXPECT_THROW(frank_wolfe(tri_f, tri_g, tri_lmo, {0.0, 0.0}, bad), std::invalid_argument);
    bad.step = FrankWolfeStep::ShortStep;
    EXPECT_THROW(frank_wolfe(tri_f, tri_g, tri_lmo, {0.0, 0.0}, bad), std::invalid_argument);
}

TEST(FrankWolfeVariants, Oracles) {
    EXPECT_EQ(l1_ball_lmo({0.5, -2.0, 1.0}, 3.0), (std::vector<double>{0, 3.0, 0}));
    EXPECT_EQ(simplex_lmo({0.5, -2.0, 1.0}), (std::vector<double>{0, 1, 0}));
    EXPECT_EQ(vertex_lmo({1.0, 1.0}, kTriangle), (std::vector<double>{0, 0}));
    EXPECT_THROW(vertex_lmo({1.0}, {}), std::invalid_argument);
}

TEST(GaussianLocationNpmle, FrankWolfeMatchesEmAndIsSparse) {
    std::mt19937_64                  rng(3);
    std::normal_distribution<double> z(0.0, 1.0);
    std::vector<double>              data;
    for (int i = 0; i < 300; ++i) data.push_back((i % 3 == 0 ? -2.0 : 2.5) + z(rng));
    std::vector<double> grid;
    for (int j = 0; j <= 60; ++j) grid.push_back(-6.0 + 0.2 * j);

    FrankWolfeOptions o;
    o.variant   = FrankWolfeVariant::Pairwise;
    o.step      = FrankWolfeStep::LineSearch;
    o.max_iter  = 5000;
    o.tolerance = 1e-9;
    const auto fw = gaussian_location_npmle(data, grid, 1.0, o);
    const auto em = gaussian_location_em(data, grid, 1.0, 200000, 1e-7);
    EXPECT_TRUE(fw.converged);
    EXPECT_NEAR(fw.log_likelihood, em.log_likelihood, 1e-4);
    EXPECT_GE(fw.log_likelihood, em.log_likelihood - 1e-9); // FW's certificate is tighter
    EXPECT_LE(fw.support.back(), 8);
    EXPECT_GT(em.iterations, 10 * fw.iterations); // EM converges sublinearly
    EXPECT_EQ(fw.support.size(), fw.objective.size());
    EXPECT_THROW(gaussian_location_em({}, grid, 1.0, 10, 0), std::invalid_argument);
}
