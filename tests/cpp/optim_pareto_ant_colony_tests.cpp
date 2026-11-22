#include <datamunge/optim/pareto_ant_colony.hpp>
#include <datamunge/optim/multi_objective.hpp>

#include <gtest/gtest.h>

#include <algorithm>

namespace o = datamunge::optim;

TEST(ParetoAntColony, FrontIsNondominatedAndEveryTourMatchesBothCosts) {
  const std::vector<std::vector<std::vector<double>>> costs{
      {{0,1,3,1},{1,0,1,3},{3,1,0,1},{1,3,1,0}},
      {{0,3,1,1},{3,0,1,1},{1,1,0,3},{1,1,3,0}}};
  const o::ParetoAntOptions options{53, 7, 8, 8};
  const auto result = o::pareto_ant_colony_tour(costs, options);
  const auto replay = o::pareto_ant_colony_tour(costs, options);
  EXPECT_EQ(result.evaluations, 53);
  ASSERT_EQ(result.front.size(), 2);
  ASSERT_EQ(replay.front.size(), result.front.size());
  for (std::size_t i = 0; i < result.front.size(); ++i) {
    const auto& point = result.front[i];
    EXPECT_EQ(point.tour, replay.front[i].tour);
    auto sorted = point.tour;
    std::ranges::sort(sorted);
    EXPECT_EQ(sorted, (std::vector<std::size_t>{0,1,2,3}));
    ASSERT_EQ(point.objectives.size(), 2);
    for (std::size_t j = 0; j < point.objectives.size(); ++j) {
      double length = 0;
      for (std::size_t k = 0; k < point.tour.size(); ++k) {
        length += costs[j][point.tour[k]][point.tour[(k + 1) % point.tour.size()]];
      }
      EXPECT_EQ(length, point.objectives[j]);
    }
    EXPECT_FALSE(o::dominates(point.objectives, result.front[1-i].objectives));
  }
}
