#include <datamunge/optim/ant_colony.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>

namespace o = datamunge::optim;

TEST(AntColony, AllPoliciesReplayAndFindKnownSquareOptimum) {
  const std::vector<std::vector<double>> distances{
      {0, 1, 2, 1}, {1, 0, 1, 2}, {2, 1, 0, 1}, {1, 2, 1, 0}};
  for (const auto policy : {o::AntColonyPolicy::ant_system, o::AntColonyPolicy::ant_colony_system,
                            o::AntColonyPolicy::max_min, o::AntColonyPolicy::rank_based,
                            o::AntColonyPolicy::elitist}) {
    o::AntColonyOptions options;
    options.evaluations = 37;
    options.ants = 8;
    options.policy = policy;
    options.seed = 17;
    const auto result = o::ant_colony_tour(distances, options);
    const auto replay = o::ant_colony_tour(distances, options);
    EXPECT_EQ(result.evaluations, 37);
    EXPECT_EQ(result.length, 4);
    EXPECT_EQ(result.tour, replay.tour);
    EXPECT_EQ(result.restarts, replay.restarts);
    auto cities = result.tour;
    std::ranges::sort(cities);
    EXPECT_EQ(cities, (std::vector<std::size_t>{0, 1, 2, 3}));
    ASSERT_EQ(result.tour.size(), 4);
    double length = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      length += distances[result.tour[i]][result.tour[(i + 1) % 4]];
    }
    EXPECT_EQ(result.length, length);
  }
}

TEST(AntColony, MaxMinRestartsWhileAcsCountsInitialization) {
  const std::vector<std::vector<double>> triangle{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
  o::AntColonyOptions options;
  options.evaluations = 7;
  options.ants = 2;
  options.restart_after = 1;
  options.policy = o::AntColonyPolicy::max_min;
  const auto result = o::ant_colony_tour(triangle, options);
  EXPECT_EQ(result.evaluations, 7);
  EXPECT_EQ(result.restarts, 2);
  options.policy = o::AntColonyPolicy::ant_colony_system;
  options.evaluations = 1;
  const auto initialized = o::ant_colony_tour(triangle, options);
  EXPECT_EQ(initialized.evaluations, 1);
  EXPECT_EQ(initialized.tour, (std::vector<std::size_t>{0, 1, 2}));
}

TEST(AntColony, RejectsInvalidDistancesAndControls) {
  o::AntColonyOptions options;
  const std::vector<std::vector<double>> triangle{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
  EXPECT_THROW(o::ant_colony_tour({{0, 1, 1}, {1}, {1, 1, 0}}, options), std::invalid_argument);
  EXPECT_THROW(o::ant_colony_tour({{0, 1, 2}, {1, 0, 1}, {1, 1, 0}}, options), std::invalid_argument);
  options.policy = o::AntColonyPolicy::max_min;
  options.min_trail_ratio = 0;
  EXPECT_THROW(o::ant_colony_tour(triangle, options), std::invalid_argument);
}
