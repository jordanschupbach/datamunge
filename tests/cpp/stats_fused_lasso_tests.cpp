#include <datamunge/stats/fused_lasso.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fused_lasso;

TEST(FusedLasso, RecoversWeightedOptimumWithKnownDualCertificate) {
    const DenseMatrix<double> x(12, 6, {
        1,2,3,4,1,0, 0,1,4,2,3,1, -1,3,2,2,2,0, 2,0,1,3,1,-1, -2,-2,1,4,-1,3,
        1,1,0,-2,0,2, 3,2,-1,0,2,4, -1,-3,0,2,1,1, 0,1,2,-1,3,3, 4,-2,1,1,-2,0,
        2,1,2,5,0,2, 1,-1,4,1,2,1});
    const std::vector<double> y{
        1.69702108608528546,-.96121032323874667,1.04007278865544928,-1.58092590352104745,
        -3.21276134817227188,6.30996392618570034,5.50997480762288561,-7.31374671291147394,
        1.70859839378528156,4.10581834293119208,2.47288840323651415,-.80964858892762981};
    datamunge::stats::FusedLassoOptions options{.lambda_l1 = .2, .lambda_difference = .3,
        .weights = {1,2,.4,1.4,1,1.5,3,1,2,.5,2,1.2}, .tolerance = 1e-10};
    const auto fit = fused_lasso(x, y, options);
    ASSERT_TRUE(fit.converged);
    // R constructed the responses using the weighted normal equations
    // and fixed subgradients for both penalties.
    const double expected[]{1,1,0,-.5,-.5,.3};
    for (std::size_t j = 0; j < 6; ++j)
        EXPECT_NEAR(fit.coefficients[j], expected[j], 1e-8);
    EXPECT_NEAR(fit.intercept, .6, 1e-8);
    EXPECT_NEAR(fit.objective, 2.0089008827836228, 1e-10);
    EXPECT_LT(fit.kkt_residual, 1e-8);
    ASSERT_EQ(fit.segments.size(), 4u);
    EXPECT_EQ(fit.segments[0].begin, 0u);
    EXPECT_EQ(fit.segments[0].end, 2u);
    EXPECT_DOUBLE_EQ(fit.segments[1].level, 0);
    EXPECT_EQ(fit.segments[2].begin, 3u);
    EXPECT_EQ(fit.segments[2].end, 5u);
    EXPECT_NEAR(fit.predict(DenseMatrix<double>(1, 6, {1,2,3,4,5,6}))[0], .9, 3e-8);
    options.max_iterations = 1;
    const auto partial = fused_lasso(x, y, options);
    EXPECT_FALSE(partial.converged);
    EXPECT_EQ(partial.iterations, 1u);
    EXPECT_GT(partial.kkt_residual, 1e-4);
}

TEST(FusedLasso, OrthogonalProximalFitsMatchExhaustiveSignedSegments) {
    constexpr std::size_t d = 7;
    constexpr unsigned partitions = 729; // 3^(d-1): fused, rising or falling at each edge.
    DenseMatrix<double> x(2*d, d);
    for (std::size_t j = 0; j < d; ++j) {
        x(2*j, j) = 1;
        x(2*j + 1, j) = -1;
    }
    std::mt19937 rng(1149);
    for (int case_index = 0; case_index < 64; ++case_index) {
        const double l1 = (rng() % 7) * .25, difference = (rng() % 20) * .2;
        std::vector<double> observation(d), y(2*d);
        for (std::size_t j = 0; j < d; ++j) {
            observation[j] = (static_cast<int>(rng() % 23) - 11) * .5;
            y[2*j] = 2 + observation[j];
            y[2*j + 1] = 2 - observation[j];
        }
        double best = std::numeric_limits<double>::infinity();
        for (unsigned configuration = 0; configuration < partitions; ++configuration) {
            auto code = configuration;
            int left = 0;
            double candidate = 0, total = 0, previous = 0;
            std::size_t first = 0;
            for (std::size_t j = 0; j < d; ++j) {
                total += observation[j];
                const auto right = j + 1 == d ? 0 : static_cast<int>(code % 3) - 1;
                code /= 3;
                if (j + 1 < d && right == 0)
                    continue;
                const auto width = j - first + 1;
                const auto mean = (total + difference * (right - left)) / width;
                const auto level = std::copysign(std::max(0., std::abs(mean) - l1), mean);
                if (left * (level - previous) < 0) {
                    candidate = std::numeric_limits<double>::infinity();
                    break;
                }
                for (std::size_t row = first; row <= j; ++row)
                    candidate += .5 * std::pow(observation[row] - level, 2) + l1 * std::abs(level);
                if (first > 0)
                    candidate += difference * std::abs(level - previous);
                previous = level;
                first = j + 1;
                left = right;
                total = 0;
            }
            best = std::min(best, candidate);
        }
        const auto fit = fused_lasso(x, y, {.lambda_l1 = l1/d, .lambda_difference = difference/d});
        SCOPED_TRACE(case_index);
        EXPECT_TRUE(fit.converged);
        EXPECT_NEAR(fit.objective, best / d, 1e-11);
        EXPECT_NEAR(fit.intercept, 2, 1e-13);
        EXPECT_LT(fit.kkt_residual, 1e-10);
    }
}

TEST(FusedLasso, HandlesFullFusionWithoutLosingTheCommonLevel) {
    const DenseMatrix<double> x(4, 2, {-1,-1, -1,1, 1,-1, 1,1});
    const std::vector<double> y{-3,1,-1,3};
    const auto fused = fused_lasso(x, y, {.lambda_l1 = .5, .lambda_difference = 1e300});
    ASSERT_TRUE(fused.converged);
    ASSERT_EQ(fused.segments.size(), 1u);
    EXPECT_EQ(fused.segments[0].end, 2u);
    EXPECT_NEAR(fused.segments[0].level, 1, 1e-12);
    const auto ordinary = fused_lasso(x, y);
    ASSERT_TRUE(ordinary.converged);
    EXPECT_NEAR(ordinary.coefficients[0], 1, 1e-12);
    EXPECT_NEAR(ordinary.coefficients[1], 2, 1e-12);
    EXPECT_THROW((void)fused_lasso(x, y, {.lambda_difference = -1}), std::invalid_argument);
    EXPECT_THROW((void)fused_lasso(x, y, {.weights = {1,1,1,0}}), std::invalid_argument);
}
}
