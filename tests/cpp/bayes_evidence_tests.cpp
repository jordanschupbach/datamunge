#include <datamunge/bayes/evidence.hpp>
#include <datamunge/random/random.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

using namespace datamunge::bayes;

namespace {

double normal_log_density(double x, double mean, double variance) {
    const double delta = x - mean;
    return -0.5 * (std::log(2 * std::numbers::pi * variance) + delta * delta / variance);
}

} // namespace

TEST(BridgeSampling, IsExactForProportionalDensitiesAndStableAtLargeLogOffsets) {
    const auto estimate = bridge_sampling({{1000, 0}, {-3000, -4000}}, {{990, -10}, {999, -1}, {-2000, -3000}});
    EXPECT_TRUE(estimate.converged);
    EXPECT_DOUBLE_EQ(estimate.log_evidence, 1000);
    EXPECT_DOUBLE_EQ(estimate.overlap, 1);
    EXPECT_DOUBLE_EQ(estimate.posterior_weight_ess, 2);
    EXPECT_DOUBLE_EQ(estimate.proposal_weight_ess, 3);
    EXPECT_LE(estimate.final_log_change, 1e-10);
}

TEST(BridgeSampling, QuantifiesObservedPartialOverlapAndReportsIterationLimits) {
    constexpr double zero = -std::numeric_limits<double>::infinity();
    const auto estimate = bridge_sampling({{5, 0}, {5, 0}, {5, zero}, {5, zero}},
                                          {{5, 0}, {5, 0}, {zero, 0}, {zero, 0}});
    EXPECT_DOUBLE_EQ(estimate.log_evidence, 5);
    EXPECT_DOUBLE_EQ(estimate.overlap, 0.5);
    EXPECT_DOUBLE_EQ(estimate.posterior_weight_ess, 2);
    EXPECT_DOUBLE_EQ(estimate.proposal_weight_ess, 2);
    const auto limited = bridge_sampling({{3, 0}, {4, 0}}, {{0, 0}, {1, 0}}, 1e-12, 1);
    EXPECT_FALSE(limited.converged);
    EXPECT_EQ(limited.iterations, 1U);
    EXPECT_GT(limited.final_log_change, 1e-12);
    EXPECT_THROW(bridge_sampling({{0, zero}}, {{zero, 0}}), std::domain_error);
    EXPECT_THROW(bridge_sampling({{0, 0}}, {{zero, 0}}), std::domain_error);
    EXPECT_THROW(bridge_sampling({{zero, 0}}, {{0, 0}}), std::invalid_argument);
    EXPECT_THROW(bridge_sampling({{0, 0}}, {{0, zero}}), std::invalid_argument);
    EXPECT_THROW(bridge_sampling({}, {{0, 0}}), std::invalid_argument);
    EXPECT_THROW(bridge_sampling({{0, 0}}, {{0, 0}}, 0.0), std::invalid_argument);
}

TEST(BridgeSampling, RecoversTheEvidenceOfAConjugateNormalModel) {
    datamunge::random::SplitMix64 rng(372);
    constexpr double offset = -1200;
    const auto evaluations = [](double x) {
        return BridgeLogDensities{normal_log_density(x, 0, 1) + normal_log_density(2, x, 1) + offset,
                                  normal_log_density(x, 0.7, 1.21)};
    };
    std::vector<BridgeLogDensities> posterior, proposal;
    for (std::size_t i = 0; i < 16000; ++i) {
        posterior.push_back(evaluations(1 + rng.normal() / std::numbers::sqrt2));
        proposal.push_back(evaluations(0.7 + 1.1 * rng.normal()));
    }
    const auto estimate = bridge_sampling(posterior, proposal);
    EXPECT_TRUE(estimate.converged);
    EXPECT_NEAR(estimate.log_evidence, normal_log_density(2, 0, 2) + offset, 0.012);
    EXPECT_GT(estimate.overlap, 0.85);
    EXPECT_LT(estimate.overlap, 0.99);
    EXPECT_GT(estimate.posterior_weight_ess, 10000);
    EXPECT_GT(estimate.proposal_weight_ess, 10000);
}

TEST(ThermodynamicIntegration, CorrectedRuleIntegratesCubicCurvesExactlyOnIrregularLadders) {
    std::vector<PowerPosteriorRung> ladder;
    for (double t : {0.0, 0.125, 0.6, 1.0})
        ladder.push_back({t, -2 + 3 * t + t * t * t, 3 + 3 * t * t});
    const auto result = thermodynamic_integration(ladder);
    EXPECT_NEAR(result.log_evidence, -0.25, 1e-15);
    EXPECT_GT(result.trapezoidal, -0.25);
    EXPECT_NEAR(result.discretization_error, result.trapezoidal + 0.25, 1e-15);
}

TEST(ThermodynamicIntegration, UsesUnbiasedSampleVarianceAndAcceptsUnequalDrawCounts) {
    const auto reference = thermodynamic_integration({{0, -3, 2}, {0.5, -2, 1}, {1, -1, 4}});
    const auto from_draws = thermodynamic_integration({0, 0.5, 1}, {{-4, -2}, {-3, -2, -1}, {-3, -1, 1}});
    EXPECT_DOUBLE_EQ(reference.log_evidence, from_draws.log_evidence);
    EXPECT_DOUBLE_EQ(reference.trapezoidal, from_draws.trapezoidal);
    EXPECT_DOUBLE_EQ(reference.discretization_error, from_draws.discretization_error);
}

TEST(ThermodynamicIntegration, ResolvesCurvatureAndValidatesLadderInputs) {
    std::vector<PowerPosteriorRung> ladder;
    for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const double v = 1 / (1 + t);
        const double r2 = 4 * v * v;
        ladder.push_back({t, -0.5 * (std::log(2 * std::numbers::pi) + v + r2), 0.5 * v * v + r2 * v});
    }
    const auto result = thermodynamic_integration(ladder);
    const double exact = normal_log_density(2, 0, 2);
    EXPECT_NEAR(result.log_evidence, exact, 0.001);
    EXPECT_GT(std::abs(result.trapezoidal - exact), 10 * std::abs(result.log_evidence - exact));
    EXPECT_THROW(thermodynamic_integration({{0, 0, 0}}), std::invalid_argument);
    EXPECT_THROW(thermodynamic_integration({{0, 0, 0}, {0.5, 0, 0}}), std::invalid_argument);
    EXPECT_THROW(thermodynamic_integration({{0, 0, 0}, {0, 0, 0}, {1, 0, 0}}), std::invalid_argument);
    EXPECT_THROW(thermodynamic_integration({{0, 0, -1}, {1, 0, 0}}), std::invalid_argument);
    EXPECT_THROW(thermodynamic_integration({0, 1}, {{1, 2}}), std::invalid_argument);
    EXPECT_THROW(thermodynamic_integration({0, 1}, {{1}, {2, 3}}), std::invalid_argument);
    EXPECT_THROW(thermodynamic_integration({0, 1}, {{1, 1}, {2, std::numeric_limits<double>::infinity()}}),
                 std::invalid_argument);
}
