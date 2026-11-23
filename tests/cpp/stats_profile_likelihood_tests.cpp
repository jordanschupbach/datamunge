#include <datamunge/stats/profile_likelihood.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {

using datamunge::stats::ProfileEndpointStatus;
using datamunge::stats::profile_likelihood_interval;

TEST(ProfileLikelihood, InvertsAsymmetricLikelihoodWithARefittedNuisanceRate) {
    const auto profile = [](double ratio) {
        if (ratio == 0)
            return -std::numeric_limits<double>::infinity();
        const auto base = 23 / (1 + ratio);
        return 5 * std::log(base * ratio) + 18 * std::log(base) - 23;
    };
    const auto fit = profile_likelihood_interval(5.0 / 18, profile, {
        .lower_bound = 0, .initial_step = .05, .tolerance = 1e-12,
    });
    EXPECT_EQ(fit.lower.status, ProfileEndpointStatus::Crossing);
    EXPECT_EQ(fit.upper.status, ProfileEndpointStatus::Crossing);
    EXPECT_NEAR(fit.likelihood_ratio_cutoff, 3.84145882069412, 1e-13);
    // Base R inverts the dpois() log-likelihood using qchisq() and uniroot().
    EXPECT_NEAR(fit.lower.value, .09172630658043332, 1e-12);
    EXPECT_NEAR(fit.upper.value, .69614405458695394, 1e-12);
}

TEST(ProfileLikelihood, DistinguishesParameterBoundsFromSearchLimits) {
    const auto flat = [](double) { return -2.0; };
    const auto bounded = profile_likelihood_interval(.5, flat, {
        .lower_bound = 0, .upper_bound = 1,
    });
    EXPECT_EQ(bounded.lower.status, ProfileEndpointStatus::ParameterBound);
    EXPECT_EQ(bounded.upper.status, ProfileEndpointStatus::ParameterBound);
    EXPECT_DOUBLE_EQ(bounded.lower.value, 0);
    EXPECT_DOUBLE_EQ(bounded.upper.value, 1);

    const auto unlimited = profile_likelihood_interval(0, flat, {.max_expansions = 2});
    EXPECT_EQ(unlimited.lower.status, ProfileEndpointStatus::SearchLimit);
    EXPECT_EQ(unlimited.upper.status, ProfileEndpointStatus::SearchLimit);
    EXPECT_DOUBLE_EQ(unlimited.lower.value, -3);
    EXPECT_DOUBLE_EQ(unlimited.upper.value, 3);

    const auto limited = profile_likelihood_interval(0, [](double x) { return -x * x / 2; },
                                                     {.max_bisections = 1});
    EXPECT_EQ(limited.lower.status, ProfileEndpointStatus::SearchLimit);
    EXPECT_EQ(limited.upper.status, ProfileEndpointStatus::SearchLimit);
    EXPECT_DOUBLE_EQ(limited.upper.value, 1);
}

TEST(ProfileLikelihood, RejectsInvalidControlsAndAnInconsistentMaximum) {
    const auto normal = [](double x) { return -x * x / 2; };
    EXPECT_THROW((void)profile_likelihood_interval(1, normal, {.upper_bound = .5}), std::invalid_argument);
    EXPECT_THROW((void)profile_likelihood_interval(1, normal), std::domain_error);
    EXPECT_THROW((void)profile_likelihood_interval(0, [](double x) {
        return x == 0 ? 0 : std::numeric_limits<double>::quiet_NaN();
    }), std::domain_error);
}

} // namespace
