#include <datamunge/stats/bootstrap_designs.hpp>
#include <datamunge/random/simplex.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using namespace datamunge::stats;
using datamunge::random::SplitMix64;

constexpr auto high = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t low = 1ULL << 60;

class ScriptedRandom : public datamunge::random::RandomGenerator {
public:
  explicit ScriptedRandom(std::vector<std::uint64_t> sequence) : sequence_(std::move(sequence)) {}
  std::uint64_t next_u64() override { return sequence_.at(position_++); }
  std::size_t used() const { return position_; }

private:
  std::vector<std::uint64_t> sequence_;
  std::size_t position_{0};
};

TEST(BootstrapDesigns, StratifiedRowsRetainInterleavedMembership) {
  const std::vector<std::size_t> stratum{50, 8, 50, 777, 8};
  ScriptedRandom rng({high, low, high, low, low});
  const auto rows = stratified_bootstrap_indices(stratum, rng);
  EXPECT_EQ(rows, (std::vector<std::size_t>{2, 4, 0, 3, 1}));
  for (std::size_t i = 0; i < rows.size(); ++i)
    EXPECT_EQ(stratum[rows[i]], stratum[i]);
  EXPECT_EQ(rng.used(), 5u);
  EXPECT_THROW(stratified_bootstrap_indices({}, rng), std::invalid_argument);
  EXPECT_EQ(rng.used(), 5u);
}

TEST(BootstrapDesigns, ClusterCopiesPreserveEveryRowAndAllowUnequalSizes) {
  ScriptedRandom rng({low, low});
  const auto copy = cluster_bootstrap_indices({50, 10, 50, 50}, rng);
  EXPECT_EQ(copy.indices, (std::vector<std::size_t>{0, 2, 3, 0, 2, 3}));
  EXPECT_EQ(copy.cluster_offsets, (std::vector<std::size_t>{0, 3, 6}));
  EXPECT_EQ(rng.used(), 2u);
}

TEST(BootstrapDesigns, FixedBlocksWrapOnlyWhenRequestedAndTruncateLastBlock) {
  ScriptedRandom moving_rng({high, low});
  EXPECT_EQ(block_bootstrap_indices(5, 3, moving_rng), (std::vector<std::size_t>{2, 3, 4, 0, 1}));
  ScriptedRandom circular_rng({high, low});
  EXPECT_EQ(block_bootstrap_indices(5, 3, circular_rng, true), (std::vector<std::size_t>{4, 0, 1, 0, 1}));
  EXPECT_EQ(circular_rng.used(), 2u);
  EXPECT_THROW(block_bootstrap_indices(5, 6, circular_rng), std::invalid_argument);
}

TEST(BootstrapDesigns, StationaryBlocksRestartOrContinueCircularly) {
  ScriptedRandom rng({high, high, 0, 1ULL << 62, high, high});
  EXPECT_EQ(stationary_bootstrap_indices(5, 2, rng), (std::vector<std::size_t>{4, 0, 1, 2, 3}));
  EXPECT_EQ(rng.used(), 6u);
  ScriptedRandom always_restart({low, high, high, high, low});
  EXPECT_EQ(stationary_bootstrap_indices(3, 1, always_restart), (std::vector<std::size_t>{0, 2, 0}));
  EXPECT_THROW(stationary_bootstrap_indices(5, .9, rng), std::invalid_argument);
}

TEST(BootstrapDesigns, BalancedPoolReplaysAndHasExactAppearanceTotals) {
  SplitMix64 first(173), replay(173);
  const auto draws = balanced_bootstrap_indices(5, 17, first);
  ASSERT_EQ(draws, balanced_bootstrap_indices(5, 17, replay));
  ASSERT_EQ(draws.size(), 17u);
  std::vector<std::size_t> counts(5);
  for (const auto& rows : draws) {
    ASSERT_EQ(rows.size(), 5u);
    for (const auto row : rows) {
      ASSERT_LT(row, counts.size());
      ++counts[row];
    }
  }
  EXPECT_EQ(counts, std::vector<std::size_t>(5, 17));
  EXPECT_THROW(balanced_bootstrap_indices(0, 2, first), std::invalid_argument);
  EXPECT_THROW(balanced_bootstrap_indices(std::numeric_limits<std::size_t>::max(), 2, first), std::length_error);
}

TEST(BootstrapDesigns, ParametricCalibrationRefitsEveryReplicateWithInclusiveTails) {
  const std::vector<double> stats{-3, -2, 1, 4, 5};
  SplitMix64 rng(62);
  for (const auto& [alternative, expected] :
       {std::pair{Alternative::Greater, 2.0}, {Alternative::Less, 3.0}, {Alternative::TwoSided, 4.0}}) {
    std::size_t simulated = 0, refitted = 0;
    const auto result = parametric_bootstrap_test(2, stats.size(),
        [&](auto&) { return std::vector{stats[simulated++] + 7, 7.0}; },
        [&](const auto& sample) {
          ++refitted;
          return sample[0] - sample[1];
        }, rng, alternative);
    EXPECT_EQ(simulated, stats.size());
    EXPECT_EQ(refitted, stats.size());
    EXPECT_EQ(result.parameter1, 5);
    EXPECT_EQ(result.parameter2, expected);
    EXPECT_DOUBLE_EQ(result.p_value, (1 + expected) / 6);
  }
  EXPECT_THROW(parametric_bootstrap_test(0, 1, [](auto&) { return std::vector{1.0}; },
      [](const auto&) { return std::numeric_limits<double>::infinity(); }, rng), std::domain_error);
}

TEST(BootstrapDesigns, WildCalibrationReconstructsResponsesWithBothWeightLaws) {
  ScriptedRandom rademacher({low, high, high, low});
  std::vector<std::vector<double>> responses;
  const auto refit = [&](const std::vector<double>& sample) {
    responses.push_back(sample);
    return sample[0] - 10;
  };
  const auto symmetric = wild_bootstrap_test(2, {10, 20}, {2, -3}, 2, refit, rademacher);
  EXPECT_EQ(responses, (std::vector<std::vector<double>>{{8, 17}, {12, 23}}));
  EXPECT_DOUBLE_EQ(symmetric.p_value, 1);
  responses.clear();
  ScriptedRandom mammen({0xb333333333333333ULL, 0xc000000000000000ULL});
  const auto asymmetric = wild_bootstrap_test(2, {10, 20}, {2, -3}, 1, refit, mammen,
                                               WildBootstrapWeight::Mammen, Alternative::Greater);
  ASSERT_EQ(responses.size(), 1u);
  EXPECT_NEAR(responses.front()[0], 8.76393202250021, 1e-14);
  EXPECT_NEAR(responses.front()[1], 15.145898033750315, 1e-14);
  EXPECT_DOUBLE_EQ(asymmetric.p_value, .5);
  EXPECT_THROW(wild_bootstrap_test(2, {1}, {1, 2}, 3, refit, mammen), std::invalid_argument);
}

TEST(BootstrapDesigns, BayesianBootstrapReproducesTheDirichletPosterior) {
  SplitMix64 rng(42), replay(42);
  const std::vector<double> values{0, 0, 1};
  const auto posterior = bayesian_bootstrap(values.size(), 12000, [&](const std::vector<double>& weights) {
    EXPECT_EQ(weights.size(), values.size());
    EXPECT_NEAR(std::accumulate(weights.begin(), weights.end(), 0.), 1, 3e-16);
    return std::inner_product(weights.begin(), weights.end(), values.begin(), 0.);
  }, rng);
  for (std::size_t i = 0; i < 20; ++i)
    EXPECT_DOUBLE_EQ(posterior[i], datamunge::random::dirichlet_sample(replay, {1, 1, 1}).back());
  const auto mean = std::accumulate(posterior.begin(), posterior.end(), 0.) / posterior.size();
  const auto second = std::inner_product(posterior.begin(), posterior.end(), posterior.begin(), 0.) / posterior.size();
  EXPECT_NEAR(mean, 1. / 3, .006);
  EXPECT_NEAR(second - mean * mean, 1. / 18, .003);
}

TEST(BootstrapDesigns, BayesianBootstrapChecksInputsAndHandlesOneObservation) {
  ScriptedRandom unchanged({});
  const auto fixed = bayesian_bootstrap(1, 3, [](const auto& w) { return 2 * w.front(); }, unchanged);
  EXPECT_EQ(fixed, (std::vector<double>{2, 2, 2}));
  EXPECT_THROW(bayesian_bootstrap(0, 3, [](const auto&) { return 0.; }, unchanged), std::invalid_argument);
  EXPECT_THROW(bayesian_bootstrap(1, 0, [](const auto&) { return 0.; }, unchanged), std::invalid_argument);
  EXPECT_THROW(bayesian_bootstrap(1, 3, {}, unchanged), std::invalid_argument);
  EXPECT_THROW(bayesian_bootstrap(1, 1, [](const auto&) { return std::numeric_limits<double>::quiet_NaN(); },
                                  unchanged), std::domain_error);
  EXPECT_EQ(unchanged.used(), 0u);
}

} // namespace
