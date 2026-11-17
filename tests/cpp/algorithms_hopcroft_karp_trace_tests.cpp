#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <datamunge/algorithms/hopcroft_karp_analysis.hpp>
#include <queue>
using namespace datamunge::algorithms;
namespace {
std::size_t optimum(std::size_t nl, std::size_t nr, const std::vector<HopcroftKarpEdge>& edges) {
  std::vector<bool> masks(std::size_t{1} << nr);
  masks[0] = true;
  for (std::size_t u = 0; u < nl; ++u) {
    auto next = masks;
    for (std::size_t mask = 0; mask < masks.size(); ++mask)
      if (masks[mask])
        for (auto [l, r] : edges)
          if (l == u && !(mask & (std::size_t{1} << r)))
            next[mask | (std::size_t{1} << r)] = true;
    masks = std::move(next);
  }
  std::size_t best = 0;
  for (std::size_t mask = 0; mask < masks.size(); ++mask)
    if (masks[mask])
      best = std::max(best, std::size_t(std::popcount(mask)));
  return best;
}
void verify(std::size_t nl, std::size_t nr, const std::vector<HopcroftKarpEdge>& edges) {
  const auto  t   = hopcroft_karp_trace(nl, nr, edges, {0, 0, 0});
  const auto& run = t.run;
  EXPECT_EQ(run.result.size, optimum(nl, nr, edges));
  std::vector<std::size_t> ml(nl, kBipartiteUnmatched), mr(nr, kBipartiteUnmatched);
  std::size_t              units = 0, pathpairs = 0, paths = 0, maxdepth = 0, prior = 0;
  ASSERT_FALSE(t.phases.empty());
  for (const auto& phase : t.phases) {
    std::vector<std::size_t> distance(nl, kBipartiteUnmatched);
    std::queue<std::size_t>  q;
    for (std::size_t u = 0; u < nl; ++u)
      if (ml[u] == kBipartiteUnmatched) {
        distance[u] = 0;
        q.push(u);
      }
    std::size_t shortest = kBipartiteUnmatched;
    while (!q.empty()) {
      auto u = q.front();
      q.pop();
      for (auto [l, r] : edges)
        if (l == u) {
          auto w = mr[r];
          if (w == kBipartiteUnmatched)
            shortest = std::min(shortest, distance[u] + 1);
          else if (distance[w] == kBipartiteUnmatched) {
            distance[w] = distance[u] + 1;
            q.push(w);
          }
        }
    }
    EXPECT_EQ(phase.shortest_pairs, shortest == kBipartiteUnmatched ? 0 : shortest);
    ASSERT_EQ(phase.left_distance.size(), nl);
    for (std::size_t u = 0; u < nl; ++u) {
      if (phase.left_distance[u] != kBipartiteUnmatched)
        EXPECT_EQ(phase.left_distance[u], distance[u]);
      if (shortest == kBipartiteUnmatched || distance[u] < shortest)
        EXPECT_EQ(phase.left_distance[u], distance[u]);
    }
    units += 1 + nl;
    std::vector<bool> usedl(nl), usedr(nr);
    if (phase.shortest_pairs) {
      EXPECT_GT(phase.shortest_pairs, prior);
      prior = phase.shortest_pairs;
      EXPECT_FALSE(phase.augmentations.empty());
    } else
      EXPECT_TRUE(phase.augmentations.empty());
    for (const auto& p : phase.augmentations) {
      const auto k = p.left.size();
      ASSERT_EQ(k, phase.shortest_pairs);
      ASSERT_EQ(k, p.right.size());
      ASSERT_GT(k, 0);
      EXPECT_EQ(ml[p.left.front()], kBipartiteUnmatched);
      EXPECT_EQ(mr[p.right.back()], kBipartiteUnmatched);
      for (std::size_t i = 0; i < k; ++i) {
        auto u = p.left[i], v = p.right[i];
        ASSERT_LT(u, nl);
        ASSERT_LT(v, nr);
        EXPECT_FALSE(usedl[u]);
        EXPECT_FALSE(usedr[v]);
        usedl[u] = true;
        usedr[v] = true;
        EXPECT_NE(std::find(edges.begin(), edges.end(), HopcroftKarpEdge{u, v}), edges.end());
        EXPECT_EQ(phase.left_distance[u], i);
        if (i + 1 < k)
          EXPECT_EQ(mr[v], p.left[i + 1]);
        if (i)
          EXPECT_EQ(ml[u], p.right[i - 1]);
      }
      for (std::size_t i = 0; i < k; ++i) {
        ml[p.left[i]]  = p.right[i];
        mr[p.right[i]] = p.left[i];
      }
      units += 1 + 2 * k;
      pathpairs += k;
      ++paths;
      maxdepth = std::max(maxdepth, k);
    }
  }
  EXPECT_EQ(ml, run.result.match_left);
  EXPECT_EQ(mr, run.result.match_right);
  EXPECT_EQ(paths, run.result.size);
  EXPECT_EQ(units, t.captured_entries);
  EXPECT_EQ(pathpairs, run.work.path_pairs_written);
  EXPECT_EQ(maxdepth, run.work.max_path_left_vertices);
  EXPECT_EQ(run.work.bfs_calls, t.phases.size());
  EXPECT_EQ(run.augmenting_phases + 1, t.phases.size());
  EXPECT_EQ(run.work.initialized_left, nl * t.phases.size());
  EXPECT_EQ(run.work.adjacency_edges, edges.size());
  EXPECT_LE(run.work.bfs_edges, edges.size() * t.phases.size());
  EXPECT_LE(run.work.dfs_edges, edges.size() * run.augmenting_phases);
  std::vector<bool> coverl(nl), coverr(nr);
  for (auto u : run.cover_left) {
    ASSERT_LT(u, nl);
    coverl[u] = true;
  }
  for (auto v : run.cover_right) {
    ASSERT_LT(v, nr);
    coverr[v] = true;
  }
  EXPECT_EQ(run.cover_left.size() + run.cover_right.size(), run.result.size);
  for (auto [u, v] : edges)
    EXPECT_TRUE(coverl[u] || coverr[v]);
}
} // namespace
TEST(HopcroftKarpTrace, ExhaustiveFourByFourGraphs) {
  for (unsigned mask = 0; mask < 65536; ++mask) {
    std::vector<HopcroftKarpEdge> edges;
    for (unsigned u = 0; u < 4; ++u)
      for (unsigned v = 0; v < 4; ++v)
        if (mask & (1u << (4 * u + v)))
          edges.emplace_back(u, v);
    verify(4, 4, edges);
  }
}
TEST(HopcroftKarpTrace, ShortestCutoffRegressionAndOrderedDuplicates) {
  const std::vector<HopcroftKarpEdge> edges = {
      {0, 4},
      {1, 0},
      {1, 1},
      {1, 4},
      {1, 5},
      {2, 0},
      {2, 2},
      {2, 3},
      {3, 2},
      {3, 4},
      {4, 3},
      {5, 0},
      {5, 1},
      {5, 3},
      {5, 5}};
  verify(6, 6, edges);
  auto t = hopcroft_karp_trace(6, 6, edges);
  ASSERT_EQ(t.phases.size(), 3);
  EXPECT_EQ(t.phases[1].shortest_pairs, 3);
  ASSERT_EQ(t.phases[1].augmentations.size(), 1);
  EXPECT_EQ(t.phases[1].augmentations[0].left.size(), 3);
  verify(2, 2, {{0, 0}, {0, 0}, {1, 0}, {0, 1}, {1, 0}});
  verify(0, 3, {});
  verify(3, 0, {});
  verify(0, 0, {});
}
TEST(HopcroftKarpTrace, CompletePoliciesAndHeapStack) {
  const std::vector<HopcroftKarpEdge> edges = {{0, 0}, {0, 1}, {1, 0}};
  auto                                t     = hopcroft_karp_trace(2, 2, edges);
  EXPECT_EQ(t.captured_entries, 17);
  EXPECT_EQ(hopcroft_karp_trace(2, 2, edges, {4, 3, 17}).captured_entries, 17);
  EXPECT_THROW(hopcroft_karp_trace(2, 2, edges, HopcroftKarpOptions{4, 3, 16}), std::length_error);
  EXPECT_THROW(hopcroft_karp_run(2, 2, edges, HopcroftKarpOptions{3, 3, 0}), std::length_error);
  EXPECT_THROW(hopcroft_karp_run(2, 2, edges, HopcroftKarpOptions{4, 2, 0}), std::length_error);
  EXPECT_THROW(
      hopcroft_karp_run(2, 2, {{2, 0}}, HopcroftKarpOptions{1, 1, 1}), std::invalid_argument);
  EXPECT_EQ(hopcroft_karp_run(2, 2, edges, {4, 3, 1}).result.size, 2);
  auto copy                       = t;
  copy.phases[0].left_distance[0] = 42;
  EXPECT_EQ(t.phases[0].left_distance[0], 0);
  const std::size_t             n = 20000;
  std::vector<HopcroftKarpEdge> chain;
  for (std::size_t u = 0; u + 1 < n; ++u) {
    chain.emplace_back(u, u);
    chain.emplace_back(u, u + 1);
  }
  chain.emplace_back(n - 1, 0);
  auto run = hopcroft_karp_run(n, n, chain);
  EXPECT_EQ(run.result.size, n);
  EXPECT_EQ(run.augmenting_phases, 2);
  EXPECT_EQ(run.work.max_path_left_vertices, n);
  EXPECT_EQ(run.cover_left.size() + run.cover_right.size(), n);
}

TEST(HopcroftKarpTrace, PublicFixturesAndNativeView) {
  const auto g = hopcroft_karp_example_graph();
  const auto r = hopcroft_karp_run(g.n_left, g.n_right, g.edges);
  EXPECT_EQ(r.result.size, 4);
  EXPECT_EQ(g.edges.size(), 8);
  EXPECT_NO_THROW(hopcroft_karp_graph_plot(g, r.result));
  EXPECT_THROW(hopcroft_karp_graph_plot(g, r.result, "x", 8), std::length_error);
  EXPECT_THROW(hopcroft_karp_graph_plot(g, r.result, "x", 9, 7), std::length_error);
  auto bad = r.result;
  bad.size = 1;
  EXPECT_THROW(hopcroft_karp_graph_plot(g, bad, "x", 1), std::invalid_argument);
  auto cutoff = hopcroft_karp_cutoff_graph();
  EXPECT_EQ(
      hopcroft_karp_trace(cutoff.n_left, cutoff.n_right, cutoff.edges).phases[1].shortest_pairs, 3);
  for (std::size_t n = 0; n < 12; ++n) {
    auto c = hopcroft_karp_chain_graph(n, 2 * n);
    auto t = hopcroft_karp_trace(c.n_left, c.n_right, c.edges);
    EXPECT_EQ(t.run.result.size, n);
    EXPECT_EQ(c.edges.size(), n ? 2 * n - 1 : 0);
    if (n >= 2) {
      EXPECT_EQ(t.run.augmenting_phases, 2);
      EXPECT_EQ(t.phases[1].shortest_pairs, n);
      EXPECT_EQ(t.captured_entries, 8 * n + 1);
    }
  }
  EXPECT_THROW(hopcroft_karp_chain_graph(3, 5), std::length_error);
  EXPECT_THROW(
      hopcroft_karp_chain_graph(std::numeric_limits<std::size_t>::max(), 0), std::length_error);
  EXPECT_NO_THROW(hopcroft_karp_graph_plot(HopcroftKarpGraph{}, BipartiteMatchingResult{}));
}
