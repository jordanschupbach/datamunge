#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/johnson_analysis.hpp>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
// Independent integer Floyd-Warshall oracle, including global cycle detection.
void compare(std::size_t n, const std::vector<WeightedEdge>& edges) {
  const long long                     inf = 1000000000;
  std::vector<std::vector<long long>> d(n, std::vector<long long>(n, inf));
  for (std::size_t i = 0; i < n; ++i)
    d[i][i] = 0;
  for (auto [u, v, w] : edges)
    d[u][v] = std::min(d[u][v], static_cast<long long>(w));
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        if (d[i][k] != inf && d[k][j] != inf)
          d[i][j] = std::min(d[i][j], d[i][k] + d[k][j]);
  bool cycle = false;
  for (std::size_t i = 0; i < n; ++i)
    cycle |= d[i][i] < 0;
  const auto t = johnson_trace(n, edges), copy = t;
  const auto a = johnson_analyze(n, edges);
  const auto r = johnson_all_pairs_shortest_paths(n, edges);
  ASSERT_EQ(t.analysis.result.has_negative_cycle, cycle);
  EXPECT_EQ(r.has_negative_cycle, cycle);
  EXPECT_EQ(t.analysis.result.distance, a.result.distance);
  EXPECT_EQ(r.distance, a.result.distance);
  EXPECT_EQ(copy.analysis.result.distance, r.distance);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      if (cycle || d[i][j] == inf)
        EXPECT_TRUE(std::isinf(r.distance[i][j]));
      else
        EXPECT_EQ(r.distance[i][j], d[i][j]);
    }
  std::vector<long double> h(n), labels(n, std::numeric_limits<long double>::infinity());
  std::size_t checks = 0, updates = 0, reweight = 0, pushes = 0, pops = 0, stales = 0, relax = 0,
              changes = 0, restore = 0;
  for (const auto& e : t.events) {
    switch (e.action) {
      case JohnsonAction::Potential: {
        const auto [u, v, w] = edges[e.edge];
        EXPECT_EQ(e.u, u);
        EXPECT_EQ(e.v, v);
        EXPECT_EQ(e.before, h[v]);
        EXPECT_EQ(e.candidate, h[u] + w);
        h[v] = std::min(h[v], e.candidate);
        EXPECT_EQ(e.after, h[v]);
        ++checks;
        updates += e.after < e.before;
        break;
      }
      case JohnsonAction::Reweight: {
        const auto [u, v, w] = edges[e.edge];
        EXPECT_EQ(e.after, w + h[u] - h[v]);
        EXPECT_GE(e.after, 0);
        ++reweight;
        break;
      }
      case JohnsonAction::Seed:
        labels.assign(n, std::numeric_limits<long double>::infinity());
        labels[e.source] = 0;
        ++pushes;
        break;
      case JohnsonAction::Pop:
        EXPECT_EQ(e.before, labels[e.u]);
        EXPECT_EQ(e.stale, e.candidate != labels[e.u]);
        ++pops;
        stales += e.stale;
        break;
      case JohnsonAction::Relax:
        EXPECT_EQ(e.before, labels[e.v]);
        EXPECT_EQ(e.candidate, labels[e.u] + t.analysis.reduced_weights[e.edge]);
        labels[e.v] = std::min(labels[e.v], e.candidate);
        EXPECT_EQ(e.after, labels[e.v]);
        ++relax;
        if (e.after < e.before) {
          ++changes;
          ++pushes;
        }
        break;
      case JohnsonAction::Restore:
        EXPECT_EQ(e.before, labels[e.v]);
        EXPECT_EQ(e.after, labels[e.v] - h[e.source] + h[e.v]);
        EXPECT_EQ(r.distance[e.source][e.v], e.after);
        ++restore;
        break;
    }
  }
  const auto& w = t.analysis.work;
  EXPECT_EQ(w.matrix_cells, n * n);
  EXPECT_EQ(w.potential_checks, checks);
  EXPECT_EQ(w.potential_updates, updates);
  EXPECT_EQ(w.reweighted_edges, reweight);
  EXPECT_EQ(w.heap_pushes, pushes);
  EXPECT_EQ(w.heap_pops, pops);
  EXPECT_EQ(w.stale_pops, stales);
  EXPECT_EQ(w.dijkstra_checks, relax);
  EXPECT_EQ(w.dijkstra_updates, changes);
  EXPECT_EQ(w.restored_cells, restore);
  EXPECT_EQ(h, t.analysis.potentials);
}
} // namespace
TEST(JohnsonTrace, ExhaustiveDirectedTwoVertexGraphs) {
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b)
      for (int c = 0; c < 4; ++c)
        for (int d = 0; d < 4; ++d) {
          std::vector<WeightedEdge> e;
          const int                 code[] = {a, b, c, d};
          for (std::size_t i = 0; i < 4; ++i)
            if (code[i])
              e.emplace_back(i / 2, i % 2, int(code[i]) - 2);
          compare(2, e);
        }
}
TEST(JohnsonTrace, RandomIntegerMultigraphsAgainstFloyd) {
  std::mt19937 rng(61);
  for (int z = 0; z < 1500; ++z) {
    const std::size_t         n = 1 + rng() % 8;
    std::vector<WeightedEdge> e;
    for (std::size_t j = 0, m = rng() % 40; j < m; ++j)
      e.emplace_back(rng() % n, rng() % n, int(rng() % 15) - 5);
    compare(n, e);
  }
}
TEST(JohnsonTrace, PotentialGeneratedCycleSafeGraphs) {
  std::mt19937 rng(13);
  for (int z = 0; z < 500; ++z) {
    const std::size_t n = 1 + rng() % 10;
    std::vector<int>  h(n);
    for (auto& v : h)
      v = int(rng() % 21) - 10;
    std::vector<WeightedEdge> e;
    for (std::size_t j = 0, m = rng() % 50; j < m; ++j) {
      const auto u = rng() % n, v = rng() % n;
      e.emplace_back(u, v, int(rng() % 8) - h[u] + h[v]);
    }
    compare(n, e);
  }
}
TEST(JohnsonTrace, OriginalAndStaleHeap) {
  compare(3, johnson_demo_edges());
  const std::vector<WeightedEdge> e = {
      {0, 1, 10}, {0, 2, 1}, {2, 1, 1}, {1, 3, -2}, {2, 3, 5}, {3, 4, 2}, {0, 4, 20}, {4, 1, 1}};
  compare(5, e);
  EXPECT_GT(johnson_analyze(5, e).work.stale_pops, 0U);
}
TEST(JohnsonTrace, EmptyDisconnectedLoopsAndParallelEdges) {
  compare(0, {});
  compare(5, {});
  compare(4, {{2, 3, -1}, {3, 2, -1}});
  compare(2, {{0, 1, 0}, {1, 0, 0}, {0, 0, 0}});
  compare(2, {{0, 1, 8}, {0, 1, 3}, {0, 1, 3}});
  compare(1, {{0, 0, -1}});
}
TEST(JohnsonTrace, PathCountsAndFixtures) {
  for (std::size_t n = 2; n <= 64; ++n) {
    const auto a = johnson_analyze(n, johnson_path_edges(n));
    const auto b = johnson_analyze(n, johnson_path_edges(n, true));
    EXPECT_EQ(a.result.distance, b.result.distance);
    EXPECT_EQ(a.work.potential_passes, 2U);
    EXPECT_EQ(b.work.potential_passes, n);
    EXPECT_EQ(a.work.potential_checks, 2 * (n - 1));
    EXPECT_EQ(b.work.potential_checks, n * (n - 1));
    EXPECT_EQ(a.work.dijkstra_checks, n * (n - 1) / 2);
    EXPECT_EQ(a.work.restored_cells, n * (n + 1) / 2);
  }
  EXPECT_THROW((void)johnson_path_edges(6, false, 4), std::length_error);
  EXPECT_EQ(johnson_path_edges(6, false, 0).size(), 5U);
  EXPECT_TRUE(johnson_path_edges(0).empty());
}
TEST(JohnsonTrace, CompleteAndOnlineAdmissionOwnership) {
  auto       edges = johnson_demo_edges();
  auto       t     = johnson_trace(3, edges);
  const auto size  = t.events.size();
  EXPECT_NO_THROW((void)johnson_trace(3, edges, {6, 9, size}));
  EXPECT_THROW((void)johnson_trace(3, edges, {6, 9, size - 1}), std::length_error);
  EXPECT_THROW((void)johnson_analyze(3, edges, {5, 9, 0}), std::length_error);
  EXPECT_THROW((void)johnson_analyze(3, edges, {6, 8, 0}), std::length_error);
  EXPECT_NO_THROW((void)johnson_analyze(3, edges, {0, 0, 1}));
  edges[0] = {1, 0, 99};
  EXPECT_EQ(t.edges[0], johnson_demo_edges()[0]);
  EXPECT_EQ(t.analysis.result.distance[0][2], -1);
  EXPECT_THROW(
      (void)johnson_analyze(std::numeric_limits<std::size_t>::max(), {}, {0, 0, 0}),
      std::length_error);
}
TEST(JohnsonTrace, InvalidAndFiniteOverflow) {
  const auto inf = std::numeric_limits<double>::infinity(),
             nan = std::numeric_limits<double>::quiet_NaN(), m = std::numeric_limits<double>::max();
  EXPECT_THROW((void)johnson_analyze(2, {{0, 2, 1}}, {1, 1, 1}), std::invalid_argument);
  EXPECT_THROW((void)johnson_analyze(2, {{0, 1, inf}}), std::invalid_argument);
  EXPECT_THROW((void)johnson_analyze(2, {{0, 1, nan}}), std::invalid_argument);
  EXPECT_THROW((void)johnson_analyze(3, {{0, 1, m}, {1, 2, m}}), std::overflow_error);
  EXPECT_TRUE(johnson_analyze(3, {{0, 1, -m}, {1, 0, -m}}).result.has_negative_cycle);
  const double d = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(johnson_analyze(2, {{0, 1, -d}}).result.distance[0][1], -d);
}
TEST(JohnsonTrace, ActualDiagramAndEditedShapeGuards) {
  auto t = johnson_trace(3, johnson_demo_edges());
  EXPECT_FALSE(johnson_reweight_diagram(t, 1).to_svg().empty());
  EXPECT_THROW((void)johnson_reweight_diagram(t, 3), std::out_of_range);
  t.events.clear();
  EXPECT_THROW((void)johnson_reweight_diagram(t), std::invalid_argument);
}
