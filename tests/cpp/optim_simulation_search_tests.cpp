#include <datamunge/optim/simulation_search.hpp>

#include <gtest/gtest.h>

#include <set>
#include <stdexcept>

namespace o = datamunge::optim;

TEST(SimulationSearch, PilotMayStopMidBlockAndSharesSeeds) {
  std::vector<std::pair<std::size_t, std::uint64_t>> seen;
  const auto score = [&](std::size_t i, std::uint64_t seed) {
    seen.emplace_back(i, seed);
    return static_cast<double>(i);
  };
  const auto run = o::simulation_search(score, 3, {5, 7, o::SimulationSearchMethod::kim_nelson});
  EXPECT_EQ(run.evaluations, 5);
  EXPECT_EQ(run.selected, 0);
  EXPECT_EQ(run.estimates[0].evaluations, 2);
  EXPECT_EQ(run.estimates[1].evaluations, 2);
  EXPECT_EQ(run.estimates[2].evaluations, 1);
  EXPECT_FALSE(run.screened_to_one);
  ASSERT_EQ(seen.size(), 5);
  EXPECT_EQ(seen[0].second, seen[2].second);
  EXPECT_EQ(seen[3].second, seen[4].second);
  EXPECT_NE(seen[0].second, seen[3].second);
}

TEST(SimulationSearch, ScreeningSelectsKnownBestAndStopsEarly) {
  const auto score = [](std::size_t i, std::uint64_t) { return i == 1 ? 0.0 : i == 0 ? 0.5 : 1; };
  for (const auto method : {o::SimulationSearchMethod::kim_nelson, o::SimulationSearchMethod::confidence_race,
                            o::SimulationSearchMethod::friedman_race}) {
    const auto result = o::simulation_search(score, 3, {1000, 7, method});
    EXPECT_EQ(result.selected, 1);
    EXPECT_LT(result.evaluations, 1000);
    EXPECT_TRUE(result.screened_to_one);
    EXPECT_FALSE(result.estimates[0].active);
    EXPECT_TRUE(result.estimates[1].active);
    EXPECT_FALSE(result.estimates[2].active);
    EXPECT_EQ(result.estimates[0].mean, 0.5);
    EXPECT_EQ(result.estimates[1].mean, 0);
    EXPECT_EQ(result.estimates[2].mean, 1);
  }
}

TEST(SimulationSearch, OcbaUsesExactlyItsBudgetAndChargesAdaptiveReplications) {
  std::set<std::uint64_t> seeds;
  const auto score = [&](std::size_t i, std::uint64_t seed) {
    seeds.insert(seed);
    return static_cast<double>(i) + 0.1 * (seed % 10);
  };
  const auto result = o::simulation_search(score, 3, {77, 7, o::SimulationSearchMethod::ocba});
  EXPECT_EQ(result.selected, 0);
  EXPECT_EQ(result.evaluations, 77);
  EXPECT_EQ(seeds.size(), 77);
  EXPECT_GT(result.estimates[0].evaluations, result.estimates[2].evaluations);
}

TEST(SimulationSearch, ConfidenceBoundsRejectOutOfRangeObservations) {
  EXPECT_THROW(o::simulation_search([](std::size_t, std::uint64_t) { return 2.0; },
      2, {15, 7, o::SimulationSearchMethod::confidence_race}), std::domain_error);
}
