#include <datamunge/bayes/slice_sampling.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

using datamunge::bayes::EllipticalSliceSampler;
using datamunge::bayes::univariate_slice_step;
using datamunge::linalg::DenseMatrix;
using datamunge::random::SplitMix64;

TEST(SliceSampling, SteppingOutRecoversGaussianMomentsWithNarrowInitialWidth) {
    SplitMix64 rng(81);
    constexpr int draws = 18000;
    const auto log_density = [](double x) { return -0.5 * std::pow((x - 1.5) / 2, 2); };
    double x = 0;
    double sum = 0, squared = 0;
    for (int i = -200; i < draws; ++i) {
        x = univariate_slice_step(log_density, x, rng, {.width = 0.25, .max_intervals = 32});
        if (i >= 0) {
            sum += x;
            squared += x * x;
        }
    }
    const double mean = sum / draws;
    EXPECT_NEAR(mean, 1.5, 0.07);
    EXPECT_NEAR(squared / draws - mean * mean, 4.0, 0.17);
}

TEST(SliceSampling, ShrinkageHandlesZeroDensityAndLimitedIntervalExpansion) {
    SplitMix64 rng(51);
    double x = 0, sum = 0, squared = 0;
    constexpr int draws = 18000;
    const auto log_density = [](double value) {
        return value >= -2 && value <= 3 ? 0.0 : -std::numeric_limits<double>::infinity();
    };
    for (int i = -100; i < draws; ++i) {
        const auto next = univariate_slice_step(log_density, x, rng, {.width = 3.0, .max_intervals = 1});
        ASSERT_GE(next, -2);
        ASSERT_LE(next, 3);
        ASSERT_LE(std::abs(next - x), 3.0);
        x = next;
        if (i >= 0) {
            sum += x;
            squared += x * x;
        }
    }
    const double mean = sum / draws;
    EXPECT_NEAR(mean, 0.5, 0.09);
    EXPECT_NEAR(squared / draws - mean * mean, 25.0 / 12.0, 0.10);
}

TEST(SliceSampling, EllipticalUpdatesRecoverCorrelatedGaussianPosterior) {
    const EllipticalSliceSampler sampler({1, -1}, DenseMatrix<double>(2, 2, {2, 0.8, 0.8, 1.5}));
    SplitMix64 rng(314);
    const auto log_likelihood = [](const std::vector<double>& x) { return -std::pow(x[0] + x[1] - 1.2, 2); };
    constexpr int draws = 26000;
    std::vector<double> state{1, -1};
    std::array<double, 2> mean{}, second_moment{};
    double cross = 0;
    for (int i = -200; i < draws; ++i) {
        state = sampler.step(log_likelihood, state, rng);
        if (i >= 0) {
            for (std::size_t j = 0; j < 2; ++j) {
                mean[j] += state[j] / draws;
                second_moment[j] += state[j] * state[j] / draws;
            }
            cross += state[0] * state[1] / draws;
        }
    }
    EXPECT_NEAR(mean[0], 1.6, 0.04);
    EXPECT_NEAR(mean[1], -0.5071428571428571, 0.04);
    EXPECT_NEAR(second_moment[0] - mean[0] * mean[0], 0.6, 0.04);
    EXPECT_NEAR(second_moment[1] - mean[1] * mean[1], 0.555357142857143, 0.04);
    EXPECT_NEAR(cross - mean[0] * mean[1], -0.35, 0.04);
}

TEST(SliceSampling, RejectsInvalidTargetsAndReportsProposalExhaustion) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    SplitMix64 rng(42);
    const auto flat = [](double) { return 0.0; };
    EXPECT_THROW(univariate_slice_step({}, 0, rng), std::invalid_argument);
    EXPECT_THROW(univariate_slice_step(flat, inf, rng), std::invalid_argument);
    EXPECT_THROW(univariate_slice_step(flat, 0, rng, {.width = 0}), std::invalid_argument);
    EXPECT_THROW(univariate_slice_step(flat, 0, rng, {.max_intervals = 0}), std::invalid_argument);
    EXPECT_THROW(univariate_slice_step(flat, 0, rng, {.max_proposals = 0}), std::invalid_argument);
    EXPECT_THROW(univariate_slice_step([=](double) { return -inf; }, 0, rng), std::invalid_argument);
    EXPECT_THROW(univariate_slice_step([=](double) { return inf; }, 0, rng), std::domain_error);
    EXPECT_THROW(univariate_slice_step([=](double x) { return x == 0 ? 0.0 : -inf; }, 0, rng,
        {.max_intervals = 1, .max_proposals = 1}), std::runtime_error);

    EXPECT_THROW(EllipticalSliceSampler({}, DenseMatrix<double>()), std::invalid_argument);
    EXPECT_THROW(EllipticalSliceSampler({0}, DenseMatrix<double>(2, 2)), std::invalid_argument);
    EXPECT_THROW(EllipticalSliceSampler({0, 0}, DenseMatrix<double>(2, 2, {1, 2, 3, 1})),
                 std::invalid_argument);
    EXPECT_THROW(EllipticalSliceSampler({0, 0}, DenseMatrix<double>(2, 2, {1, 1, 1, 1})),
                 std::invalid_argument);
    EXPECT_THROW(EllipticalSliceSampler({0}, DenseMatrix<double>(1, 1, inf)), std::invalid_argument);
    const EllipticalSliceSampler sampler({0}, DenseMatrix<double>(1, 1, 1.0));
    const auto log_likelihood = [](const auto&) { return 0.0; };
    EXPECT_THROW(sampler.step(log_likelihood, {}, rng), std::invalid_argument);
    EXPECT_THROW(sampler.step(log_likelihood, {nan}, rng), std::invalid_argument);
    EXPECT_THROW(sampler.step(log_likelihood, {0}, rng, 0), std::invalid_argument);
    EXPECT_THROW(sampler.step([=](const auto&) { return -inf; }, {0}, rng), std::invalid_argument);
    EXPECT_THROW(sampler.step([=](const auto&) { return nan; }, {0}, rng), std::domain_error);
    EXPECT_THROW(sampler.step([=](const auto& x) { return x[0] == 0 ? 0.0 : -inf; }, {0}, rng, 1),
                 std::runtime_error);
}
