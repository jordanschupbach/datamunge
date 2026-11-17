#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <datamunge/algorithms/graph_coloring_analysis.hpp>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
using Edges = std::vector<std::pair<std::size_t, std::size_t>>;
// Independent exact k-color decision search in vertex-ID order, with explicit
// edge conflicts. Used only on all six-vertex simple graphs.
bool feasible(
    std::size_t       n,
    const Edges&      edges,
    std::size_t       k,
    std::vector<int>& colors,
    std::size_t       v) {
  if (v == n)
    return true;
  for (std::size_t c = 0; c < k; ++c) {
    bool okay = true;
    for (auto [a, b] : edges) {
      if (a == v && colors[b] == int(c))
        okay = false;
      if (b == v && colors[a] == int(c))
        okay = false;
    }
    if (okay) {
      colors[v] = int(c);
      if (feasible(n, edges, k, colors, v + 1))
        return true;
      colors[v] = -1;
    }
  }
  return false;
}
std::size_t chromatic(std::size_t n, const Edges& e) {
  if (!n)
    return 0;
  for (std::size_t k = 1; k <= n; ++k) {
    std::vector<int> c(n, -1);
    if (feasible(n, e, k, c, 0))
      return k;
  }
  return n;
}
void certificate(std::size_t n, const Edges& e) {
  const auto t = greedy_coloring_trace(n, e);
  const auto a = greedy_coloring_analyze(n, e);
  const auto r = greedy_coloring(n, e);
  EXPECT_EQ(r.color, a.result.color);
  EXPECT_EQ(r.color, t.analysis.result.color);
  EXPECT_EQ(r.num_colors, a.result.num_colors);
  std::vector<std::vector<std::size_t>> adj(n);
  std::size_t                           loops = 0;
  for (auto [u, v] : e) {
    if (u == v) {
      ++loops;
      continue;
    }
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(a.degree[i], adj[i].size());
  for (std::size_t j = 1; j < n; ++j) {
    auto u = a.order[j - 1], v = a.order[j];
    EXPECT_TRUE(a.degree[u] > a.degree[v] || (a.degree[u] == a.degree[v] && u < v));
  }
  std::vector<std::size_t> colors(n, SIZE_MAX);
  std::size_t              cursor = 0, slots = 0, visits = 0, writes = 0, probes = 0;
  for (auto v : a.order) {
    std::vector<bool> used(adj[v].size() + 1, false);
    slots += used.size();
    for (auto w : adj[v]) {
      ASSERT_LT(cursor, t.events.size());
      const auto x = t.events[cursor++];
      EXPECT_EQ(x.action, ColoringAction::neighbor);
      EXPECT_EQ(x.vertex, v);
      EXPECT_EQ(x.neighbor, w);
      const bool colored = colors[w] != SIZE_MAX, marked = colored && colors[w] < used.size();
      EXPECT_EQ(x.colored, colored);
      EXPECT_EQ(x.flag, marked);
      EXPECT_EQ(x.color, colored ? colors[w] : 0);
      if (marked) {
        used[colors[w]] = true;
        ++writes;
      }
      ++visits;
    }
    std::size_t c = 0;
    do {
      ASSERT_LT(cursor, t.events.size());
      const auto x = t.events[cursor++];
      EXPECT_EQ(x.action, ColoringAction::probe);
      EXPECT_EQ(x.vertex, v);
      EXPECT_EQ(x.color, c);
      EXPECT_EQ(x.flag, used[c]);
      ++probes;
      if (!used[c])
        break;
      ++c;
    } while (c < used.size());
    ASSERT_LT(cursor, t.events.size());
    const auto x = t.events[cursor++];
    EXPECT_EQ(x.action, ColoringAction::assign);
    EXPECT_EQ(x.vertex, v);
    EXPECT_EQ(x.color, c);
    colors[v] = c;
  }
  EXPECT_EQ(cursor, t.events.size());
  EXPECT_EQ(colors, r.color);
  EXPECT_EQ(a.work.marker_slots, slots);
  EXPECT_EQ(a.work.neighbor_visits, visits);
  EXPECT_EQ(a.work.marker_writes, writes);
  EXPECT_EQ(a.work.color_probes, probes);
  EXPECT_EQ(a.work.ignored_loops, loops);
  EXPECT_EQ(a.work.adjacency_entries, 2 * (e.size() - loops));
  EXPECT_EQ(visits, a.work.adjacency_entries);
  EXPECT_EQ(slots, n + visits);
  EXPECT_EQ(t.events.size(), visits + probes + n);
  EXPECT_LE(probes, slots);
  EXPECT_LE(r.num_colors, n ? a.work.max_degree + 1 : 0);
  EXPECT_TRUE(is_proper_coloring(n, e, r.color));
}
} // namespace
TEST(ColoringTrace, AllSixVertexSimpleGraphsAndExactChromaticOracle) {
  const auto all = coloring_complete(6).edges;
  EXPECT_EQ(chromatic(6, all), 6);
  EXPECT_EQ(chromatic(5, coloring_cycle(5).edges), 3);
  EXPECT_EQ(chromatic(0, {}), 0);
  std::size_t optimal = 0, suboptimal = 0;
  for (unsigned mask = 0; mask < (1u << all.size()); ++mask) {
    Edges e;
    for (std::size_t i = 0; i < all.size(); ++i)
      if (mask & (1u << i))
        e.push_back(all[i]);
    certificate(6, e);
    const auto result  = greedy_coloring(6, e);
    const auto optimum = chromatic(6, e);
    EXPECT_GE(result.num_colors, optimum);
    if (result.num_colors == optimum)
      ++optimal;
    else
      ++suboptimal;
  }
  EXPECT_GT(suboptimal, 0);
  EXPECT_EQ(optimal + suboptimal, 32768);
  std::printf(
      "Exact six-vertex oracle: %zu optimal, %zu suboptimal greedy results\n", optimal, suboptimal);
}
TEST(ColoringTrace, MultigraphLoopOrderAndCrownWitnesses) {
  certificate(0, {});
  certificate(1, {{0, 0}});
  certificate(5, {});
  std::mt19937 rng(661);
  for (int t = 0; t < 500; ++t) {
    Edges e;
    for (int i = 0; i < 30; ++i)
      e.emplace_back(rng() % 8, rng() % 8);
    certificate(8, e);
  }
  for (std::size_t k = 2; k <= 20; ++k)
    for (bool paired : {false, true}) {
      const auto g = coloring_crown(k, paired);
      const auto r = greedy_coloring(g.vertices, g.edges), w = coloring_crown_witness(k, paired);
      EXPECT_EQ(r.num_colors, paired ? k : 2);
      EXPECT_TRUE(is_proper_coloring(g.vertices, g.edges, w.color));
      EXPECT_EQ(g.edges.size(), k * (k - 1));
    }
}
TEST(ColoringTrace, CompleteCapsDomainOwnershipAndVerifierScope) {
  const auto g   = coloring_cycle(5);
  auto       t   = greedy_coloring_trace(g.vertices, g.edges);
  auto       cap = t.events.size();
  EXPECT_NO_THROW(greedy_coloring_trace(5, g.edges, {5, 5, cap}));
  EXPECT_THROW(greedy_coloring_trace(5, g.edges, {5, 5, cap - 1}), std::length_error);
  EXPECT_NO_THROW(greedy_coloring_trace(5, g.edges, {0, 0, 0}));
  EXPECT_NO_THROW(greedy_coloring_analyze(5, g.edges, {5, 5, 1}));
  EXPECT_THROW(greedy_coloring_trace(5, g.edges, {4, 5, 0}), std::length_error);
  EXPECT_THROW(greedy_coloring_trace(5, g.edges, {5, 4, 0}), std::length_error);
  EXPECT_THROW(greedy_coloring_trace(SIZE_MAX, {{SIZE_MAX, 0}}, {1, 1, 1}), std::invalid_argument);
  EXPECT_THROW(greedy_coloring(0, {{0, 0}}), std::invalid_argument);
  auto copy                     = t;
  copy.events[0].vertex         = 99;
  copy.analysis.order[0]        = 99;
  copy.analysis.result.color[0] = 99;
  EXPECT_NE(t.events[0].vertex, 99);
  EXPECT_NE(t.analysis.order[0], 99);
  EXPECT_NE(t.analysis.result.color[0], 99);
  EXPECT_TRUE(is_proper_coloring(1, {{0, 0}}, {SIZE_MAX}));
  EXPECT_TRUE(is_proper_coloring(2, {{0, 1}}, {7, 99}));
  EXPECT_FALSE(is_proper_coloring(2, {{0, 1}}, {7, 7}));
  EXPECT_FALSE(is_proper_coloring(1, {{0, 1}}, {7}));
  EXPECT_FALSE(is_proper_coloring(2, {}, {}));
}
TEST(ColoringTrace, FamilyAndDiagramBoundaries) {
  EXPECT_THROW(coloring_cycle(2), std::invalid_argument);
  EXPECT_THROW(coloring_cycle(5, 4), std::length_error);
  EXPECT_THROW(coloring_complete(SIZE_MAX, 0), std::length_error);
  EXPECT_THROW(coloring_crown(1), std::invalid_argument);
  EXPECT_THROW(coloring_crown(SIZE_MAX, true, 0), std::length_error);
  EXPECT_THROW(coloring_crown(4, true, 11), std::length_error);
  EXPECT_NO_THROW(coloring_crown(4, true, 12));
  EXPECT_THROW(coloring_diagram({2, {{0, 2}}}, {{0, 1}, 2}), std::invalid_argument);
  EXPECT_THROW(coloring_diagram({33, {}}, {}), std::invalid_argument);
  EXPECT_NO_THROW(coloring_diagram({0, {}}, {}));
  EXPECT_EQ(coloring_complete(0).edges.size(), 0);
}
