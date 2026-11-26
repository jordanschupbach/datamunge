#include <datamunge/stats/hawkes.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_hawkes_data.inc"
const HawkesParameters truth{.mu = 0.5, .alpha = 0.8, .beta = 1.5};
} // namespace

TEST(Hawkes, LogLikelihoodMatchesReferences) {
    EXPECT_NEAR(hawkes_log_likelihood(events, truth), ll_true[0], 1e-10);
    EXPECT_NEAR(hawkes_log_likelihood(events, truth), emhawkes_ll_true[0], 1e-10);
    EXPECT_NEAR(hawkes_log_likelihood(events, truth, events.back() + 10), ll_true[1], 1e-10);
    EXPECT_NEAR(hawkes_log_likelihood(events, {.mu = 1, .alpha = 0.2, .beta = 3}), ll_true[2], 1e-10);
}

TEST(Hawkes, FitMatchesExactOptimumAndEmhawkes) {
    const auto f = hawkes_fit(events);
    EXPECT_TRUE(f.converged);
    EXPECT_NEAR(f.parameters.mu, mle_exact[0], 1e-6);
    EXPECT_NEAR(f.parameters.alpha, mle_exact[1], 1e-6);
    EXPECT_NEAR(f.parameters.beta, mle_exact[2], 1e-6);
    for (std::size_t k = 0; k < 3; ++k)
        EXPECT_NEAR(f.standard_errors[k], mle_exact[3 + k], 1e-4 * mle_exact[3 + k]) << k;
    EXPECT_NEAR(f.log_likelihood, mle_exact[6], 1e-9);
    EXPECT_GE(f.log_likelihood, mle_emhawkes[3] - 1e-9);
    EXPECT_NEAR(f.parameters.mu, mle_emhawkes[0], 1e-4);
    EXPECT_NEAR(f.parameters.alpha, mle_emhawkes[1], 1e-4);
    EXPECT_NEAR(f.parameters.beta, mle_emhawkes[2], 1e-4);
    EXPECT_NEAR(f.branching_ratio, f.parameters.alpha / f.parameters.beta, 1e-15);
    EXPECT_NEAR(f.aic, -2 * f.log_likelihood + 6, 1e-12);
}

TEST(Hawkes, ResidualsAndSimulation) {
    const auto r = hawkes_residuals(events, truth);
    ASSERT_EQ(r.size(), events.size());
    for (std::size_t k = 0; k < residuals_true.size(); ++k)
        EXPECT_NEAR(r[k], residuals_true[k], 1e-12) << k;
    // Simulated processes recover their parameters, and residuals are Exp(1) on average.
    const HawkesParameters p{.mu = 1, .alpha = 1.2, .beta = 2};
    const auto t = hawkes_simulate(p, 5000, 42);
    EXPECT_TRUE(std::is_sorted(t.begin(), t.end()));
    // Stationary mean rate mu / (1 - alpha / beta) = 2.5.
    EXPECT_NEAR(static_cast<double>(t.size()) / 5000, 2.5, 0.15);
    const auto f = hawkes_fit(t, {.end_time = 5000});
    EXPECT_NEAR(f.parameters.mu, 1, 4 * f.standard_errors[0]);
    EXPECT_NEAR(f.parameters.alpha, 1.2, 4 * f.standard_errors[1]);
    EXPECT_NEAR(f.parameters.beta, 2, 4 * f.standard_errors[2]);
    double mean = 0;
    for (const double v : hawkes_residuals(t, p))
        mean += v;
    EXPECT_NEAR(mean / static_cast<double>(t.size()), 1, 0.05);
}

TEST(Hawkes, Validates) {
    EXPECT_THROW(hawkes_log_likelihood({1, 1}, truth), std::invalid_argument);
    EXPECT_THROW(hawkes_log_likelihood({1, 2}, {.mu = -1}), std::invalid_argument);
    EXPECT_THROW(hawkes_log_likelihood({1, 2}, truth, 1.5), std::invalid_argument);
    EXPECT_THROW(hawkes_fit({1, 2}), std::invalid_argument);
    EXPECT_THROW(hawkes_simulate(truth, 0, 1), std::invalid_argument);
}
