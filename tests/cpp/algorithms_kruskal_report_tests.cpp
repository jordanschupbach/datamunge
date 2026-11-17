#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <datamunge/algorithms/kruskal.hpp>
#include <limits>
#include <queue>
using namespace datamunge::algorithms;
TEST(KruskalReport, OriginalGraphAgainstCompleteSpanningSubsetEnumeration) {
  const std::vector<WeightedEdge> edges = {
      {0, 1, 7},
      {0, 3, 5},
      {1, 2, 8},
      {1, 3, 9},
      {1, 4, 6},
      {2, 4, 10},
      {2, 5, 4},
      {3, 4, 3},
      {4, 5, 2}};
  double best       = std::numeric_limits<double>::infinity();
  int    minimizers = 0;
  for (int mask = 0; mask < (1 << 9); ++mask) {
    if (std::popcount(unsigned(mask)) != 5)
      continue;
    std::vector<std::vector<int>> a(6);
    double                        sum = 0;
    for (int i = 0; i < 9; ++i)
      if (mask & (1 << i)) {
        auto [u, v, w] = edges[i];
        a[u].push_back(v);
        a[v].push_back(u);
        sum += w;
      }
    std::vector<bool> seen(6);
    seen[0] = true;
    std::queue<int> q;
    q.push(0);
    while (!q.empty()) {
      auto u = q.front();
      q.pop();
      for (int v : a[u])
        if (!seen[v]) {
          seen[v] = true;
          q.push(v);
        }
    }
    if (!std::all_of(seen.begin(), seen.end(), [](bool b) { return b; }))
      continue;
    if (sum < best) {
      best       = sum;
      minimizers = 1;
    } else
      minimizers += sum == best;
  }
  const auto t = kruskal_trace(6, edges);
  EXPECT_EQ(best, 20);
  EXPECT_EQ(minimizers, 1);
  EXPECT_EQ(t.outcome.result.total_weight, best);
  EXPECT_TRUE(t.outcome.result.is_connected);
  EXPECT_EQ(t.steps.size(), 5U);
  EXPECT_EQ(t.outcome.work.find_calls, 10U);
  EXPECT_EQ(t.outcome.work.cycle_rejections, 0U);
  EXPECT_EQ(
      t.outcome.result.edges,
      (std::vector<WeightedEdge>{{4, 5, 2}, {3, 4, 3}, {2, 5, 4}, {0, 3, 5}, {1, 4, 6}}));
}
TEST(KruskalReport, DisconnectedParallelLoopAndStableTies) {
  const auto t = kruskal_trace(4, {{0, 0, -10}, {0, 1, -2}, {1, 0, -2}, {2, 3, 3}});
  EXPECT_FALSE(t.outcome.result.is_connected);
  EXPECT_EQ(t.outcome.result.total_weight, 1);
  EXPECT_EQ(t.steps.size(), 4U);
  EXPECT_FALSE(t.steps[0].accepted);
  EXPECT_TRUE(t.steps[1].accepted);
  EXPECT_FALSE(t.steps[2].accepted);
  EXPECT_TRUE(t.steps[3].accepted);
  EXPECT_EQ(t.steps.back().components_after, 2U);
  const auto a = kruskal(3, {{0, 1, 1}, {1, 2, 1}, {0, 2, 1}});
  const auto b = kruskal(3, {{0, 2, 1}, {1, 2, 1}, {0, 1, 1}});
  EXPECT_EQ(a.total_weight, b.total_weight);
  EXPECT_NE(a.edges, b.edges);
  EXPECT_EQ(a.edges, (std::vector<WeightedEdge>{{0, 1, 1}, {1, 2, 1}}));
}
TEST(KruskalReport, ActualScansAndUnscannedHeavyTail) {
  for (std::size_t n : {2, 16, 64, 256, 1024}) {
    std::vector<WeightedEdge> e;
    for (std::size_t i = 0; i < n; ++i)
      e.emplace_back(i, i, -1);
    for (std::size_t i = 1; i < n; ++i)
      e.emplace_back(i - 1, i, double(i));
    e.emplace_back(0, n - 1, 1e6);
    const auto t = kruskal_trace(n, e);
    EXPECT_EQ(t.outcome.work.input_edges, 2 * n);
    EXPECT_EQ(t.outcome.work.scanned_edges, 2 * n - 1);
    EXPECT_EQ(t.outcome.work.cycle_rejections, n);
    EXPECT_EQ(t.outcome.work.accepted_edges, n - 1);
    EXPECT_EQ(t.outcome.work.find_calls, 2 * (2 * n - 1));
    EXPECT_EQ(t.steps.back().components_after, 1U);
    EXPECT_EQ(t.outcome.result.total_weight, double(n * (n - 1) / 2));
  }
}
