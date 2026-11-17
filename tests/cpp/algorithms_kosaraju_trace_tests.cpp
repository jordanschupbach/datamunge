#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/graph_connectivity.hpp>
#include <map>
#include <numeric>
using namespace datamunge::algorithms;
namespace {
void verify(const std::vector<std::vector<int>>& a) {
  const int  n = a.size();
  const auto t = kosaraju_trace(n, a);
  ASSERT_EQ(t.component.size(), a.size());
  EXPECT_EQ(t.component, kosaraju_scc(n, a));
  EXPECT_EQ(t.component, kosaraju_analyze(n, a).component);
  std::vector<std::vector<char>> reach(n, std::vector<char>(n));
  std::vector<std::vector<int>>  transpose(n);
  std::size_t                    E = 0;
  for (int u = 0; u < n; ++u) {
    reach[u][u] = 1;
    for (int v : a[u]) {
      reach[u][v] = 1;
      transpose[v].push_back(u);
      ++E;
    }
  }
  for (int k = 0; k < n; ++k)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        reach[i][j] |= reach[i][k] && reach[k][j];
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      EXPECT_EQ(t.component[i] == t.component[j], reach[i][j] && reach[j][i]);
  for (int u = 0; u < n; ++u)
    for (int v : a[u])
      EXPECT_LE(t.component[u], t.component[v]);
  EXPECT_EQ(t.transpose, transpose);
  EXPECT_EQ(t.work.first_edge_tests, E);
  EXPECT_EQ(t.work.second_edge_tests, E);
  EXPECT_EQ(t.work.finishes, a.size());
  EXPECT_EQ(t.work.assignments, a.size());
  EXPECT_EQ(t.events.size(), 3 * a.size() + 2 * E);
  EXPECT_EQ(t.captured_entries, 4 * a.size() + 3 * E);
  std::vector<int>                   disc(n), fin(n), assigned(n), finish_order;
  std::map<std::pair<int, int>, int> first, second, want;
  for (int u = 0; u < n; ++u)
    for (int v : a[u])
      ++want[{u, v}];
  for (const auto& e : t.events) {
    ASSERT_GE(e.u, 0);
    ASSERT_LT(e.u, n);
    switch (e.kind) {
      case SCCEventKind::Discover:
        EXPECT_EQ(disc[e.u]++, 0);
        break;
      case SCCEventKind::FirstEdge:
        EXPECT_EQ(disc[e.u], 1);
        EXPECT_EQ(fin[e.u], 0);
        ++first[{e.u, e.v}];
        break;
      case SCCEventKind::Finish:
        EXPECT_EQ(fin[e.u]++, 0);
        finish_order.push_back(e.u);
        break;
      case SCCEventKind::Assign:
        EXPECT_EQ(assigned[e.u]++, 0);
        EXPECT_EQ(e.component, t.component[e.u]);
        break;
      case SCCEventKind::TransposeEdge:
        EXPECT_EQ(assigned[e.u], 1);
        ++second[{e.v, e.u}];
        break;
    }
  }
  EXPECT_EQ(finish_order, t.finish_order);
  EXPECT_EQ(first, want);
  EXPECT_EQ(second, want);
  EXPECT_EQ(disc, std::vector<int>(n, 1));
  EXPECT_EQ(fin, disc);
  EXPECT_EQ(assigned, disc);
}
} // namespace
TEST(KosarajuTrace, EveryLoopFreeFourVertexGraphAgainstTransitiveClosure) {
  for (int mask = 0; mask < (1 << 12); ++mask) {
    std::vector<std::vector<int>> a(4);
    int                           bit = 0;
    for (int u = 0; u < 4; ++u)
      for (int v = 0; v < 4; ++v)
        if (u != v) {
          if (mask & (1 << bit))
            a[u].push_back(v);
          ++bit;
        }
    verify(a);
  }
}
TEST(KosarajuTrace, OriginalGraphLoopsParallelAndEmpty) {
  verify({{1}, {2}, {0, 3}, {4}, {5}, {3, 6}, {7}, {}});
  verify({{0, 1, 1}, {0, 1}, {2}, {}});
  verify({});
  verify({{}});
  EXPECT_EQ(
      kosaraju_scc(8, {{1}, {2}, {0, 3}, {4}, {5}, {3, 6}, {7}, {}}),
      (std::vector<int>{0, 0, 0, 1, 1, 1, 2, 3}));
}
TEST(KosarajuTrace, CompleteShapeEndpointAndCaptureGuards) {
  EXPECT_THROW((void)kosaraju_scc(-1, {}), std::invalid_argument);
  EXPECT_THROW((void)kosaraju_scc(2, {{}}), std::invalid_argument);
  EXPECT_THROW((void)kosaraju_scc(1, {{1}}), std::invalid_argument);
  EXPECT_THROW((void)kosaraju_scc(1, {{-1}}), std::invalid_argument);
  EXPECT_THROW((void)kosaraju_trace(0, {{}}, 1), std::invalid_argument);
  const auto t = kosaraju_trace(3, {{1}, {0, 2}, {}});
  EXPECT_NO_THROW((void)kosaraju_trace(3, t.adjacency, t.captured_entries));
  EXPECT_THROW((void)kosaraju_trace(3, t.adjacency, t.captured_entries - 1), std::length_error);
  EXPECT_THROW((void)kosaraju_trace(3, t.adjacency, 1), std::length_error);
  EXPECT_NO_THROW((void)kosaraju_trace(3, t.adjacency, 0));
  auto c = t;
  c.adjacency[0].clear();
  c.transpose[0].clear();
  c.finish_order.clear();
  c.component.clear();
  c.events[0].u = 99;
  EXPECT_EQ(t.adjacency[0], std::vector<int>{1});
  EXPECT_FALSE(t.finish_order.empty());
  EXPECT_NE(t.events[0].u, 99);
}
TEST(KosarajuTrace, LongIterativeChainAndSingleCycle) {
  const int                     n = 100000;
  std::vector<std::vector<int>> a(n);
  for (int i = 0; i < n - 1; ++i)
    a[i].push_back(i + 1);
  auto c = kosaraju_scc(n, a);
  ASSERT_EQ(c.size(), n);
  for (int i = 0; i < n; ++i)
    EXPECT_EQ(c[i], i);
  a.back().push_back(0);
  c = kosaraju_scc(n, a);
  EXPECT_EQ(c, std::vector<int>(n, 0));
}
