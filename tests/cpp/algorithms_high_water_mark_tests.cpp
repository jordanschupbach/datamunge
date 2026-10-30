#include <datamunge/algorithms/high_water_mark.hpp>
#include <datamunge/algorithms/search_batch.hpp>

#include <gtest/gtest.h>

#include <random>
#include <vector>

namespace alg = datamunge::algorithms;

TEST(HighWaterMark, BracketsGreedyBestFirstOnRandomGraphs) {
  std::mt19937 rng(17);
  for (int trial = 0; trial < 40; ++trial) {
    const std::size_t n = 120;
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    for (std::size_t v = 1; v < n; ++v) edges.emplace_back(rng() % v, v);  // a random tree keeps it connected
    for (int e = 0; e < 150; ++e) edges.emplace_back(rng() % n, rng() % n);
    std::vector<double> h(n);
    std::uniform_real_distribution<double> u(0.0, 10.0);
    for (auto& v : h) v = u(rng);
    const std::size_t s = rng() % n, t = rng() % n;
    const auto mark = alg::high_water_mark(n, edges, s, t, h, false);
    const auto run = alg::best_first_search(n, edges, s, t, h, false);
    ASSERT_TRUE(run.found);
    EXPECT_LE(mark.below, run.order.size());
    EXPECT_LE(run.order.size(), mark.at_most + 1);
    for (const auto v : run.order) EXPECT_LE(h[v], mark.level);
    // H is at least h(s) and h(t), and some expanded vertex reaches it unless t is the start.
    EXPECT_GE(mark.level, std::max(h[s], h[t]));
  }
  // A depression: a pocket of low h off the route must be flooded first.
  // Path 0-1-2-3 (goal 3) with a wall h=5 at 1; pocket 4,5,6 hangs off 0 with h=1.
  const std::vector<std::pair<std::size_t, std::size_t>> e = {{0, 1}, {1, 2}, {2, 3}, {0, 4}, {4, 5}, {5, 6}};
  const std::vector<double> h = {3, 5, 2, 0, 1, 1, 1};
  const auto mark = alg::high_water_mark(7, e, 0, 3, h, false);
  EXPECT_EQ(mark.level, 5.0);
  EXPECT_EQ(mark.below, 4u);  // 0, 4, 5, 6
  const auto run = alg::best_first_search(7, e, 0, 3, h, false);
  EXPECT_EQ(run.order, (std::vector<std::size_t>{0, 4, 5, 6, 1, 2, 3}));
  EXPECT_THROW((void)alg::high_water_mark(3, {}, 0, 5, {0, 0, 0}), std::invalid_argument);
}
