#include <gtest/gtest.h>

#include <datamunge/stats/bart.hpp>

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace st = datamunge::stats;
using datamunge::linalg::DenseMatrix;

TEST(BART, GrowPruneOccupancyMatchesTheExactTwoStructurePosterior) {
    const DenseMatrix<double> x(2, 1, {0, 1});
    const st::BART fit(x, {0, 1},
        {.num_trees = 1, .warmup = 300, .draws = 6000, .min_leaf = 1, .max_depth = 1,
         .split_base = 0.4, .fixed_noise_sd = 0.5, .seed = 21});
    const auto draws = fit.posterior_draws(x);
    std::size_t split = 0;
    for (std::size_t i = 0; i < draws.rows(); ++i) split += draws(i, 0) != draws(i, 1);

    // On scaled y = {-0.5, 0.5}, noise variance is .25 and leaf variance is .0625.
    const double log_stump = -0.5 * std::log(1.5);
    const double log_split = -std::log(1.25) + 0.2;
    const double expected = 1 / (1 + (0.6 / 0.4) * std::exp(log_stump - log_split));
    EXPECT_NEAR(double(split) / draws.rows(), expected, 0.035);
    EXPECT_GT(fit.structure_acceptance_rate(), 0.1);
}

TEST(BART, GaussianStumpDrawsMatchConjugateLeafMoments) {
    const DenseMatrix<double> x(3, 1, {1, 2, 3});
    const st::BART fit(x, {0, 0, 1},
        {.num_trees = 1, .warmup = 0, .draws = 6000, .max_depth = 0,
         .fixed_noise_sd = 0.5, .seed = 16});
    const auto draws = fit.posterior_draws(x);
    const double expected_mean = 0.5 - 0.5 / 7;
    const double expected_variance = 1. / 28;
    double sum = 0, squared = 0;
    for (std::size_t i = 0; i < draws.rows(); ++i) {
        EXPECT_DOUBLE_EQ(draws(i, 0), draws(i, 2));
        sum += draws(i, 0);
        squared += (draws(i, 0) - expected_mean) * (draws(i, 0) - expected_mean);
    }
    EXPECT_NEAR(sum / draws.rows(), expected_mean, 0.008);
    EXPECT_NEAR(squared / draws.rows(), expected_variance, 0.0025);
    EXPECT_DOUBLE_EQ(fit.structure_acceptance_rate(), 0);
    EXPECT_DOUBLE_EQ(fit.noise_sd_draws().front(), 0.5);
    const auto credible = fit.predict_interval(x, 0.9);
    const auto predictive = fit.predict_interval(x, 0.9, st::BARTPredictionTarget::NewObservation);
    EXPECT_EQ(credible.mean, fit.predict(x));
    EXPECT_EQ(predictive.mean, credible.mean);
    EXPECT_LT(predictive.lower[0], credible.lower[0]);
    EXPECT_GT(predictive.upper[0], credible.upper[0]);
}

TEST(BART, ReversibleMovesMatchTheExactDepthTwoPosteriorMasses) {
    const DenseMatrix<double> x(3, 1, {0, 1, 2});
    const st::BART fit(x, {0, 0.5, 1},
        {.num_trees = 1, .warmup = 500, .draws = 14000, .min_leaf = 1, .max_depth = 2,
         .split_base = 0.65, .split_power = 1, .fixed_noise_sd = 0.5, .seed = 19});
    // Independent enumeration. The two binary root splits have the same
    // likelihood by reflection, as do both trees with three terminal nodes.
    const auto integral = [](double sum, int count) {
        const double factor = 1 + count * 0.25;
        return std::exp(0.5 * sum * sum / factor) / std::sqrt(factor);
    };
    const double child_split = 0.65 / 2;
    const double stump = 0.35 * integral(0, 3);
    const double two = 0.65 * (1 - child_split) * integral(0.5, 1) * integral(0.5, 2);
    const double three = 0.65 * child_split * std::pow(integral(0.5, 1), 2) * integral(0, 1);
    const double normalizer = stump + two + three;
    const auto draws = fit.posterior_draws(x);
    std::size_t leaves[4]{};
    for (std::size_t i = 0; i < draws.rows(); ++i)
        ++leaves[1 + (draws(i, 0) != draws(i, 1)) + (draws(i, 1) != draws(i, 2))];
    EXPECT_NEAR(double(leaves[1]) / draws.rows(), stump / normalizer, 0.035);
    EXPECT_NEAR(double(leaves[2]) / draws.rows(), two / normalizer, 0.035);
    EXPECT_NEAR(double(leaves[3]) / draws.rows(), three / normalizer, 0.035);
}

TEST(BART, FitsAStepAndReproducesTheSeededDraws) {
    DenseMatrix<double> x(64, 2);
    std::vector<double> y(64);
    for (std::size_t i = 0; i < y.size(); ++i) {
        x(i, 0) = static_cast<double>(i) / y.size();
        x(i, 1) = 1; // Constant features never supply a split.
        y[i] = (i < 32 ? -1 : 1) + 0.15 * std::sin(3.1 * i);
    }
    const st::BARTOptions options{.num_trees = 14, .warmup = 160, .draws = 180, .max_cutpoints = 64};
    const st::BART fit(x, y, options), repeat(x, y, options);
    const DenseMatrix<double> query(2, 2, {0.2, 1, 0.8, 1});
    const auto mean = fit.predict(query);
    EXPECT_EQ(mean, repeat.predict(query));
    EXPECT_EQ(fit.noise_sd_draws(), repeat.noise_sd_draws());
    EXPECT_EQ(fit.noise_sd_draws().size(), options.draws);
    EXPECT_LT(mean[0], -0.8);
    EXPECT_GT(mean[1], 0.8);
    const auto noise = fit.noise_sd_draws();
    const auto noise_mean = std::accumulate(noise.begin(), noise.end(), 0.) / noise.size();
    EXPECT_GT(noise_mean, 0.08);
    EXPECT_LT(noise_mean, 0.27);
}

TEST(BART, ValidatesShapesAndAcceptsEmptyQueries) {
    const DenseMatrix<double> x(2, 1, {0, 1});
    EXPECT_THROW(st::BART(x, {1}), std::invalid_argument);
    EXPECT_THROW(st::BART(x, {1, 2}, {.min_leaf = 0}), std::invalid_argument);
    EXPECT_THROW(st::BART(x, {1, 2}, {.fixed_noise_sd = -1}), std::invalid_argument);
    const st::BART fit(x, {1, 1}, {.num_trees = 2, .warmup = 2, .draws = 4});
    EXPECT_TRUE(fit.predict(DenseMatrix<double>(0, 1)).empty());
    EXPECT_EQ(fit.posterior_draws(DenseMatrix<double>(0, 1)).rows(), 4U);
    EXPECT_THROW((void)fit.predict(DenseMatrix<double>(1, 2)), std::invalid_argument);
    EXPECT_THROW((void)fit.predict_interval(x, 1.), std::invalid_argument);
    EXPECT_THROW((void)fit.predict(DenseMatrix<double>(1, 1, {std::numeric_limits<double>::quiet_NaN()})),
                 std::invalid_argument);
}
