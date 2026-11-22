#include <datamunge/optim/iterated_lin_kernighan.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>

namespace o = datamunge::optim;

TEST(IteratedLinKernighan, ImprovesAnExplicitCrossedTourAndCountsKicks) {
  const std::vector<std::vector<double>> square{
      {0, 1, 2, 1}, {1, 0, 1, 2}, {2, 1, 0, 1}, {1, 2, 1, 0}};
  const std::vector<std::size_t> crossed{0, 2, 1, 3};
  const auto initial = o::iterated_lin_kernighan(square, crossed, {1, 7});
  EXPECT_EQ(initial.evaluations, 1);
  EXPECT_EQ(initial.length, 6);
  EXPECT_EQ(initial.kicks, 0);
  const auto run = o::iterated_lin_kernighan(square, crossed, {37, 7});
  EXPECT_EQ(run.evaluations, 37);
  EXPECT_EQ(run.length, 4);
  EXPECT_GT(run.kicks, 0);
  EXPECT_GE(run.deepest_improvement, 2);
  const auto replay = o::iterated_lin_kernighan(square, crossed, {37, 7});
  EXPECT_EQ(run.tour, replay.tour);
  EXPECT_THROW(o::iterated_lin_kernighan(square, {0, 0, 1, 2}), std::invalid_argument);
}

TEST(IteratedLinKernighan, DoubleBridgeKeepsAValidPermutation) {
  std::vector<std::vector<double>> graph(8, std::vector<double>(8, 1));
  for (std::size_t j = 0; j < 8; ++j) graph[j][j] = 0;
  const auto run = o::iterated_lin_kernighan(graph, {}, {11, 12});
  EXPECT_EQ(run.evaluations, 11);
  EXPECT_EQ(run.kicks, 10);
  EXPECT_EQ(run.length, 8);
  auto tour = run.tour;
  std::ranges::sort(tour);
  EXPECT_EQ(tour, (std::vector<std::size_t>{0,1,2,3,4,5,6,7}));
}
