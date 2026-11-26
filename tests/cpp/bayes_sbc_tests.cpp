#include <datamunge/bayes/sbc.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::bayes;
using datamunge::random::SplitMix64;

// theta_j ~ N(0, 1), five observations y ~ N(theta_j, 1): posterior N(sum y / 6, 1 / 6).
SbcResult run(double sd_factor, double shift, std::size_t reps = 1000) {
    return simulation_based_calibration(
        [](SplitMix64& g) { return std::vector<double>{g.normal(), g.normal()}; },
        [](const std::vector<double>& t, SplitMix64& g) {
            std::vector<double> y;
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 5; ++i)
                    y.push_back(t[j] + g.normal());
            return y;
        },
        [=](const std::vector<double>& y, SplitMix64& g) {
            std::vector<std::vector<double>> d(99, std::vector<double>(2));
            for (int j = 0; j < 2; ++j) {
                const double m = (y[5 * j] + y[5 * j + 1] + y[5 * j + 2] + y[5 * j + 3] + y[5 * j + 4]) / 6;
                for (auto& row : d)
                    row[j] = m + shift + sd_factor * std::sqrt(1.0 / 6) * g.normal();
            }
            return d;
        },
        {.replications = reps, .seed = 21});
}
} // namespace

TEST(Sbc, ExactPosteriorGivesUniformRanks) {
    const auto r = run(1.0, 0.0);
    ASSERT_EQ(r.parameters.size(), 2u);
    EXPECT_EQ(r.posterior_draws, 99u);
    for (const auto& p : r.parameters) {
        EXPECT_EQ(p.ranks.size(), 1000u);
        EXPECT_EQ(std::accumulate(p.histogram.begin(), p.histogram.end(), std::size_t{0}), 1000u);
        EXPECT_GT(p.chi_squared_p_value, 0.01);
        EXPECT_GT(p.ecdf_p_value, 0.01);
        EXPECT_NEAR(p.mean_rank_fraction, 0.5, 0.03);
        for (const std::size_t rank : p.ranks)
            EXPECT_LE(rank, 99u);
    }
}

TEST(Sbc, DetectsUnderdispersionAndBias) {
    const auto narrow = run(0.5, 0.0);
    const auto& h = narrow.parameters[0].histogram;
    EXPECT_LT(narrow.parameters[0].chi_squared_p_value, 1e-6);
    EXPECT_GT(h.front() + h.back(), 3 * (h[9] + h[10]));
    const auto biased = run(1.0, 0.3);
    EXPECT_LT(biased.parameters[1].mean_rank_fraction, 0.4);
    EXPECT_LT(biased.parameters[1].ecdf_p_value, 0.01);
}

TEST(Sbc, ChiSquaredUsesEqualExpectedCounts) {
    const auto r = run(1.0, 0.0, 400);
    const auto& p = r.parameters[0];
    double stat = 0;
    for (const std::size_t c : p.histogram)
        stat += (c - 20.0) * (c - 20.0) / 20.0;
    EXPECT_NEAR(p.chi_squared, stat, 1e-9);
}

TEST(Sbc, CountsFailuresAndValidates) {
    int calls = 0;
    const auto r = simulation_based_calibration(
        [](SplitMix64& g) { return std::vector<double>{g.normal()}; },
        [](const std::vector<double>& t, SplitMix64&) { return t; },
        [&](const std::vector<double>& y, SplitMix64&) {
            if (++calls % 10 == 0)
                throw std::runtime_error("fit failed");
            return std::vector<std::vector<double>>(9, y);
        },
        {.replications = 100, .bins = 10});
    EXPECT_EQ(r.failures, 10u);
    EXPECT_EQ(r.parameters[0].ranks.size(), 90u);
    EXPECT_THROW(run(1.0, 0.0, 1), std::invalid_argument);
    EXPECT_THROW(simulation_based_calibration(
                     [](SplitMix64& g) { return std::vector<double>{g.normal()}; },
                     [](const std::vector<double>& t, SplitMix64&) { return t; },
                     [](const std::vector<double>&, SplitMix64&) { return std::vector<std::vector<double>>{{1, 2}}; }),
                 std::invalid_argument);
}
