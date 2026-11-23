#include <datamunge/bayes/diagnostics.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using datamunge::bayes::waic;
using datamunge::bayes::bulk_tail_effective_sample_size;
using datamunge::bayes::rank_normalized_rhat;

namespace {

std::vector<std::vector<double>> test_chains(std::size_t n,
    const std::array<int, 3>& locations, const std::array<int, 3>& spreads) {
    std::vector<std::vector<double>> chains;
    constexpr std::array steps{3, 5, 7};
    for (std::size_t j = 0; j < steps.size(); ++j) {
        auto& draws = chains.emplace_back();
        for (std::size_t draw = 1; draw <= n; ++draw) {
            const int width = static_cast<int>((draw * steps[j] + ((j + 1) * draw) * ((j + 1) * draw)) % 47) - 23;
            draws.push_back(locations[j] + spreads[j] * width);
        }
    }
    return chains;
}

} // namespace

TEST(RankDiagnostics, MatchesRForTiesOddSplitsAndLocationOrScaleDifferences) {
    const auto tied = rank_normalized_rhat(test_chains(25, {0, 2, -1}, {1, 1, 1}));
    EXPECT_NEAR(tied.rank_normalized, 0.99167749297052488, 1e-12);
    EXPECT_NEAR(tied.folded, 0.99162052054766558, 1e-12);
    const auto scale = rank_normalized_rhat(test_chains(24, {0, 0, 0}, {1, 1, 6}));
    EXPECT_NEAR(scale.rank_normalized, 1.01953812254244136, 1e-12);
    EXPECT_NEAR(scale.maximum, 1.51196379383032586, 1e-12);
    const auto location = rank_normalized_rhat(test_chains(27, {0, 20, -32}, {1, 1, 1}));
    EXPECT_NEAR(location.maximum, 1.95034962395209166, 1e-12);
    EXPECT_NEAR(location.folded, 1.24108534487881417, 1e-12);
}

TEST(RankDiagnostics, MatchesRForBulkAndBothTailsWithDirectAndFftCovariances) {
    const auto short_chains = bulk_tail_effective_sample_size(test_chains(25, {0, 2, -1}, {1, 1, 1}));
    EXPECT_NEAR(short_chains.bulk, 85.52506616080134, 1e-10);
    EXPECT_NEAR(short_chains.tail, 84.13856812933024, 1e-10);
    EXPECT_NEAR(short_chains.upper_tail, 103.69767574018300, 1e-10);
    const auto long_chains = bulk_tail_effective_sample_size(test_chains(275, {0, 2, -1}, {1, 3, 1}));
    EXPECT_NEAR(long_chains.bulk, 647.5322940318164, 1e-8);
    EXPECT_NEAR(long_chains.tail, 110.7652758074737, 1e-8);
    EXPECT_NEAR(long_chains.upper_tail, 442.7566943668772, 1e-8);
}

TEST(RankDiagnostics, ReflectionSwapsTailsAtAUserSelectedQuantile) {
    auto chains = test_chains(25, {0, 2, -1}, {1, 1, 1});
    const auto before = bulk_tail_effective_sample_size(chains, 0.2);
    for (auto& chain : chains)
        for (auto& draw : chain)
            draw = -draw;
    const auto after = bulk_tail_effective_sample_size(chains, 0.2);
    EXPECT_NEAR(after.bulk, before.bulk, 1e-10);
    EXPECT_NEAR(after.lower_tail, before.upper_tail, 1e-10);
    EXPECT_NEAR(after.upper_tail, before.lower_tail, 1e-10);
}

TEST(RankDiagnostics, DetectsUndefinedComponentsAndRejectsInvalidInputs) {
    const auto constant = rank_normalized_rhat({{1, 1, 1, 1}});
    EXPECT_TRUE(std::isnan(constant.rank_normalized));
    EXPECT_TRUE(std::isnan(constant.folded));
    EXPECT_TRUE(std::isnan(constant.maximum));
    const auto separated = rank_normalized_rhat({
        std::vector<double>(100, 0), std::vector<double>(100, 1), std::vector<double>(100, 3)});
    EXPECT_TRUE(std::isinf(separated.maximum));
    const auto constant_ess = bulk_tail_effective_sample_size({{1, 1, 1, 1, 1, 1}});
    EXPECT_TRUE(std::isnan(constant_ess.bulk));
    EXPECT_TRUE(std::isnan(constant_ess.tail));
    constexpr double inf = std::numeric_limits<double>::infinity();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(rank_normalized_rhat({}), std::invalid_argument);
    EXPECT_THROW(rank_normalized_rhat({{0, 1, inf, 3, 4}}), std::invalid_argument);
    EXPECT_THROW(rank_normalized_rhat({{1, 1, 1, 1}, {1, 1, 1}}), std::invalid_argument);
    EXPECT_THROW(bulk_tail_effective_sample_size({{1, 2, 3, 4}}), std::invalid_argument);
    EXPECT_THROW(bulk_tail_effective_sample_size({{1, 2, 3, 4, 5, 6}}, 0.5), std::invalid_argument);
    EXPECT_THROW(bulk_tail_effective_sample_size({{1, 2, 3, 4, 5, 6}}, nan), std::invalid_argument);
}

TEST(Waic, MatchesRPointwiseReferenceWithAConstantAndTinyLikelihoodColumn) {
    const std::vector<std::vector<double>> draws{{-1, -2, -1000}, {-3, -2, -1002}, {-2, -2, -1004}};
    const auto result = waic(draws);
    ASSERT_EQ(result.pointwise.size(), 3U);
    // R reference: apply(draws, 2, function(x) max(x) + log(mean(exp(x - max(x)))))
    // and apply(draws, 2, var).
    EXPECT_NEAR(result.pointwise[0].log_predictive_density, -1.691006324223729, 1e-14);
    EXPECT_DOUBLE_EQ(result.pointwise[0].effective_parameters, 1.0);
    EXPECT_DOUBLE_EQ(result.pointwise[1].log_predictive_density, -2.0);
    EXPECT_DOUBLE_EQ(result.pointwise[1].effective_parameters, 0.0);
    EXPECT_NEAR(result.pointwise[2].log_predictive_density, -1000.955680660168, 1e-12);
    EXPECT_DOUBLE_EQ(result.pointwise[2].effective_parameters, 4.0);
    EXPECT_NEAR(result.pointwise[2].elpd, -1004.955680660168, 1e-12);
    EXPECT_NEAR(result.log_predictive_density, -1004.646686984392, 1e-12);
    EXPECT_DOUBLE_EQ(result.effective_parameters, 5.0);
    EXPECT_NEAR(result.elpd, -1009.646686984392, 1e-12);
    EXPECT_NEAR(result.waic, 2019.293373968784, 1e-12);
}

TEST(Waic, RespectsObservationOffsetsAndDrawOrder) {
    std::vector<std::vector<double>> draws{{-1, -8}, {-2, -4}, {-3, -2}, {-1, -6}};
    const auto baseline = waic(draws);
    for (auto& draw : draws) {
        draw[0] -= 1024;
        draw[1] += 128;
    }
    std::reverse(draws.begin(), draws.end());
    const auto shifted = waic(draws);
    EXPECT_NEAR(shifted.pointwise[0].elpd, baseline.pointwise[0].elpd - 1024, 2e-13);
    EXPECT_NEAR(shifted.pointwise[1].elpd, baseline.pointwise[1].elpd + 128, 2e-14);
    EXPECT_DOUBLE_EQ(shifted.effective_parameters, baseline.effective_parameters);
    EXPECT_NEAR(shifted.waic, baseline.waic + 1792, 3e-13);
}

TEST(Waic, RejectsInvalidInputsAndUnrepresentableScores) {
    constexpr double max = std::numeric_limits<double>::max();
    constexpr double inf = std::numeric_limits<double>::infinity();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(waic({}), std::invalid_argument);
    EXPECT_THROW(waic({{-1}}), std::invalid_argument);
    EXPECT_THROW(waic({{}, {}}), std::invalid_argument);
    EXPECT_THROW(waic({{-1}, {-2, -3}}), std::invalid_argument);
    EXPECT_THROW(waic({{-inf}, {-1}}), std::invalid_argument);
    EXPECT_THROW(waic({{inf}, {-1}}), std::invalid_argument);
    EXPECT_THROW(waic({{-1}, {nan}}), std::invalid_argument);
    EXPECT_THROW(waic({{-max}, {max}}), std::overflow_error);
    EXPECT_THROW(waic({{-max}, {-max}}), std::overflow_error);
}
