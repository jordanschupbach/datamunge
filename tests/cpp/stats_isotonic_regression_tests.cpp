#include <datamunge/stats/isotonic_regression.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::stats::fit_isotonic;
using datamunge::stats::IsotonicDirection;

TEST(IsotonicRegression, PoolsUnsortedTiesAndPredictsRightContinuousSteps) {
    const auto fit = fit_isotonic({4,2,1,6,2,5,3}, {.6,.25,.15,.65,.2,.8,.7},
        {.weights = {8,10,5,10,6,10,12}});
    ASSERT_EQ(fit.blocks().size(), 4u);
    EXPECT_DOUBLE_EQ(fit.blocks()[1].first_x, 2);
    EXPECT_DOUBLE_EQ(fit.blocks()[1].last_x, 2);
    EXPECT_EQ(fit.blocks()[1].observations, 2u);
    EXPECT_DOUBLE_EQ(fit.blocks()[2].first_x, 3);
    EXPECT_DOUBLE_EQ(fit.blocks()[2].last_x, 4);
    const std::vector<double> expected{.66,.23125,.15,.725,.23125,.725,.66};
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(fit.fitted()[i], expected[i], 1e-15);
    EXPECT_DOUBLE_EQ(fit.predict(-5), fit.fitted()[2]);
    EXPECT_DOUBLE_EQ(fit.predict(std::nextafter(3., 0.)), fit.fitted()[1]);
    EXPECT_DOUBLE_EQ(fit.predict(3), fit.fitted()[6]);
    EXPECT_DOUBLE_EQ(fit.predict(4.8), fit.fitted()[6]);
    EXPECT_DOUBLE_EQ(fit.predict(6.5), fit.fitted()[5]);
    EXPECT_TRUE(fit.predict(std::vector<double>{}).empty());
}

TEST(IsotonicRegression, MatchesExhaustiveWeightedPartitionsInBothDirections) {
    constexpr std::size_t n = 7;
    const std::vector<double> x{0,1,2,3,4,5,6};
    std::mt19937 generator(4039);
    for (const bool increasing : {true, false})
        for (int trial = 0; trial < 32; ++trial) {
            std::vector<double> y(n), w(n);
            for (std::size_t i = 0; i < n; ++i) {
                y[i] = static_cast<double>(generator() % 17) - 8;
                w[i] = 1. + generator() % 5;
            }
            const auto fit = fit_isotonic(x, y, {
                .direction = increasing ? IsotonicDirection::Increasing : IsotonicDirection::Decreasing,
                .weights = w,
            });
            double best = std::numeric_limits<double>::infinity(), loss = 0;
            for (unsigned mask = 0; mask < (1u << (n - 1)); ++mask) {
                double previous = increasing ? -std::numeric_limits<double>::infinity()
                                             : std::numeric_limits<double>::infinity();
                double candidate = 0;
                std::size_t start = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    if (i + 1 != n && !(mask & (1u << i)))
                        continue;
                    double total = 0, weight = 0;
                    for (std::size_t j = start; j <= i; ++j) {
                        total += w[j] * y[j];
                        weight += w[j];
                    }
                    const auto level = total / weight;
                    if (increasing ? level < previous : level > previous) {
                        candidate = std::numeric_limits<double>::infinity();
                        break;
                    }
                    for (std::size_t j = start; j <= i; ++j)
                        candidate += w[j] * (y[j] - level) * (y[j] - level);
                    previous = level;
                    start = i + 1;
                }
                best = std::min(best, candidate);
            }
            for (std::size_t i = 0; i < n; ++i) {
                const auto level = fit.fitted()[i];
                loss += w[i] * (level - y[i]) * (level - y[i]);
                if (i > 0) {
                    EXPECT_TRUE(increasing ? level >= fit.fitted()[i - 1] : level <= fit.fitted()[i - 1]);
                }
            }
            EXPECT_NEAR(loss, best, 1e-10);
        }
}

TEST(IsotonicRegression, RetainsFiniteLevelsWithExtremeWeightsAndValidatesInputs) {
    const auto extreme = fit_isotonic({1,2,3}, {-1e308, 2, 1e308}, {.weights = {1e-300,1e300,1e-300}});
    EXPECT_EQ(extreme.fitted(), (std::vector<double>{-1e308, 2, 1e308}));
    const auto balanced = fit_isotonic({1,1}, {1e308, -1e308}, {.weights = {1e308,1e308}});
    ASSERT_EQ(balanced.blocks().size(), 1u);
    EXPECT_DOUBLE_EQ(balanced.predict(1.), 0);
    EXPECT_THROW((void)fit_isotonic({}, {}), std::invalid_argument);
    EXPECT_THROW((void)fit_isotonic({1,2}, {2}), std::invalid_argument);
    EXPECT_THROW((void)fit_isotonic({1,2}, {0,1}, {.weights = {1,0}}), std::invalid_argument);
    EXPECT_THROW((void)extreme.predict(std::numeric_limits<double>::infinity()), std::invalid_argument);
}
}
