#include <datamunge/optim/large_neighborhood_search.hpp>

#include <gtest/gtest.h>

namespace o = datamunge::optim;

TEST(LargeNeighborhoodSearch, LearnsWeightsFromImprovingAndRejectedTrials) {
  const auto score = [](int x) { return x; };
  const auto destroy = [](int x, std::size_t, auto&) -> std::optional<int> { return x - 1; };
  const auto repair = [](int x, std::size_t, auto&) -> std::optional<int> { return std::max(x, 0); };
  o::LargeNeighborhoodOptions options;
  options.evaluations = 3;
  options.reaction = 0.5;
  options.reaction_interval = 1;
  const auto result = o::large_neighborhood_search(1, score, destroy, repair, options);
  EXPECT_EQ(result.search.evaluations, 3);
  EXPECT_EQ(result.search.accepted, 1);
  EXPECT_EQ(result.search.best, 0);
  EXPECT_EQ(result.attempts, 2);
  EXPECT_EQ(result.destroy_weights, std::vector<double>{2.75});
  EXPECT_EQ(result.repair_weights, result.destroy_weights);
}

TEST(LargeNeighborhoodSearch, LimitsEmptyRepairsAndCountsOnlyFeasibleScores) {
  o::LargeNeighborhoodOptions options;
  options.evaluations = 20;
  options.empty_attempt_limit = 3;
  const auto score = [](int x) { return x; };
  const auto destroy = [](int x, std::size_t, auto&) -> std::optional<int> { return x - 1; };
  const auto repair = [](int, std::size_t, auto&) -> std::optional<int> { return {}; };
  const auto result = o::large_neighborhood_search(5, score, destroy, repair, options);
  EXPECT_EQ(result.attempts, 3);
  EXPECT_EQ(result.search.evaluations, 1);
  EXPECT_EQ(result.search.accepted, 0);
  EXPECT_EQ(result.search.stop, o::NeighborhoodStop::no_move);
}
