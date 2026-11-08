#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <datamunge/algorithms/gabow_scc.hpp>
#include <map>
#include <numeric>
#include <random>
#include <set>
using namespace datamunge::algorithms;
namespace {
using Reach = std::vector<std::vector<bool>>;
Reach closure(const SccAdjacency& a) {
  const auto n = a.size();
  Reach      reach(n, std::vector<bool>(n));
  for (std::size_t u = 0; u < n; ++u) {
    reach[u][u] = true;
    for (int v : a[u])
      reach[u][v] = true;
  }
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        reach[i][j] = reach[i][j] || (reach[i][k] && reach[k][j]);
  return reach;
}
void certificate(const GabowTrace& t) {
  const auto& a     = t.adjacency;
  const auto  n     = a.size();
  std::size_t edges = 0;
  for (const auto& row : a)
    edges += row.size();
  GabowEvent previous;
  previous.comp.assign(n, -1);
  previous.preorder.assign(n, 0);
  std::map<std::string, std::size_t> counts;
  std::size_t tree = 0, open = 0, closed = 0, pdfs = 0, popen = 0, proots = 0, entries = n + edges;
  for (const auto& e : t.events) {
    SCOPED_TRACE(e.kind + " " + std::to_string(e.u));
    ASSERT_GE(e.u, 0);
    ASSERT_LT(std::size_t(e.u), n);
    ASSERT_EQ(e.comp.size(), n);
    ASSERT_EQ(e.preorder.size(), n);
    ++counts[e.kind];
    pdfs   = std::max(pdfs, e.stack.size());
    popen  = std::max(popen, e.path.size());
    proots = std::max(proots, e.roots.size());
    entries += 1 + 2 * n + e.path.size() + e.roots.size() + e.stack.size();
    ASSERT_GE(e.ncomp, 0);
    ASSERT_LE(e.ncomp, e.counter);
    ASSERT_LE(e.counter, int(n));
    std::set<int> seen, discovery;
    for (std::size_t i = 0; i < n; ++i) {
      ASSERT_GE(e.preorder[i], 0);
      ASSERT_LE(e.preorder[i], e.counter);
      if (e.preorder[i])
        ASSERT_TRUE(discovery.insert(e.preorder[i]).second);
      ASSERT_GE(e.comp[i], -1);
      ASSERT_LE(e.comp[i], e.ncomp);
      if (e.comp[i] >= 0)
        ASSERT_GT(e.preorder[i], 0);
    }
    ASSERT_EQ(discovery.size(), std::size_t(e.counter));
    int pre = 0;
    for (int v : e.path) {
      ASSERT_GE(v, 0);
      ASSERT_LT(std::size_t(v), n);
      ASSERT_TRUE(seen.insert(v).second);
      ASSERT_GT(e.preorder[v], pre);
      pre = e.preorder[v];
      ASSERT_EQ(e.comp[v], -1);
    }
    for (std::size_t i = 0; i < n; ++i)
      ASSERT_EQ(seen.contains(int(i)), e.preorder[i] > 0 && e.comp[i] == -1);
    pre = 0;
    for (int v : e.roots) {
      ASSERT_TRUE(seen.contains(v));
      ASSERT_GT(e.preorder[v], pre);
      pre = e.preorder[v];
    }
    std::set<int> active;
    pre = 0;
    for (std::size_t j = 0; j < e.stack.size(); ++j) {
      const auto [u, next] = e.stack[j];
      ASSERT_TRUE(active.insert(u).second);
      ASSERT_GT(e.preorder[u], pre);
      pre = e.preorder[u];
      ASSERT_LE(next, a[u].size());
      if (j) {
        const auto [parent, index] = e.stack[j - 1];
        ASSERT_GT(index, 0u);
        ASSERT_EQ(a[parent][index - 1], u);
      }
    }
    ASSERT_EQ(e.counter, previous.counter + (e.kind == "enter"));
    const int increment =
        e.kind == "finish" && previous.kind == "assign" && previous.u == e.u && previous.v == e.u
            ? 1
            : 0;
    ASSERT_EQ(e.ncomp, previous.ncomp + increment);
    if (e.kind == "enter") {
      auto p = previous.preorder;
      ASSERT_EQ(p[e.u], 0);
      p[e.u] = e.counter;
      EXPECT_EQ(e.preorder, p);
      auto s = previous.path;
      s.push_back(e.u);
      EXPECT_EQ(e.path, s);
      auto b = previous.roots;
      b.push_back(e.u);
      EXPECT_EQ(e.roots, b);
      auto f = previous.stack;
      f.push_back({e.u, 0});
      EXPECT_EQ(e.stack, f);
      EXPECT_EQ(e.comp, previous.comp);
    } else {
      EXPECT_EQ(e.preorder, previous.preorder);
      if (e.kind == "edge") {
        ASSERT_FALSE(previous.stack.empty());
        auto f = previous.stack;
        ASSERT_EQ(f.back().first, e.u);
        ++f.back().second;
        EXPECT_EQ(e.stack, f);
        ASSERT_GT(f.back().second, 0u);
        EXPECT_EQ(e.v, a[e.u][f.back().second - 1]);
        EXPECT_EQ(e.comp, previous.comp);
        EXPECT_EQ(e.path, previous.path);
        EXPECT_EQ(e.roots, previous.roots);
        if (!e.preorder[e.v])
          ++tree;
        else if (e.comp[e.v] == -1)
          ++open;
        else
          ++closed;
      } else if (e.kind == "collapse" || e.kind == "close-root") {
        auto b = previous.roots;
        ASSERT_FALSE(b.empty());
        const auto root = b.back();
        b.pop_back();
        EXPECT_EQ(e.roots, b);
        if (e.kind == "collapse")
          EXPECT_GT(e.preorder[root], e.preorder[e.v]);
        else
          EXPECT_EQ(root, e.u);
        EXPECT_EQ(e.comp, previous.comp);
        EXPECT_EQ(e.path, previous.path);
        EXPECT_EQ(e.stack, previous.stack);
      } else if (e.kind == "assign") {
        auto s = previous.path;
        ASSERT_FALSE(s.empty());
        EXPECT_EQ(s.back(), e.v);
        s.pop_back();
        EXPECT_EQ(e.path, s);
        auto c = previous.comp;
        ASSERT_EQ(c[e.v], -1);
        c[e.v] = e.ncomp;
        EXPECT_EQ(e.comp, c);
        EXPECT_EQ(e.roots, previous.roots);
        EXPECT_EQ(e.stack, previous.stack);
      } else if (e.kind == "finish") {
        auto f = previous.stack;
        ASSERT_FALSE(f.empty());
        EXPECT_EQ(f.back().first, e.u);
        f.pop_back();
        EXPECT_EQ(e.stack, f);
        EXPECT_EQ(e.comp, previous.comp);
        EXPECT_EQ(e.path, previous.path);
        EXPECT_EQ(e.roots, previous.roots);
      } else
        FAIL() << "unknown event " << e.kind;
    }
    previous = e;
  }
  const auto& w = t.work;
  EXPECT_EQ(t.components, previous.comp);
  EXPECT_EQ(t.captured_entries, entries);
  EXPECT_EQ(w.event_count, edges + 4 * n);
  EXPECT_EQ(t.events.size(), w.event_count);
  EXPECT_EQ(w.visited_vertices, n);
  EXPECT_EQ(w.edges_examined, edges);
  EXPECT_EQ(w.tree_edges, tree);
  EXPECT_EQ(w.open_edges, open);
  EXPECT_EQ(w.closed_edges, closed);
  EXPECT_EQ(w.root_pushes, n);
  EXPECT_EQ(w.root_pops, n);
  EXPECT_EQ(w.root_collapses, counts["collapse"]);
  EXPECT_EQ(w.component_closures, counts["close-root"]);
  EXPECT_EQ(w.assignments, n);
  EXPECT_EQ(w.dfs_pushes, n);
  EXPECT_EQ(w.dfs_pops, n);
  EXPECT_EQ(w.open_pushes, n);
  EXPECT_EQ(w.open_pops, n);
  EXPECT_EQ(w.peak_dfs, pdfs);
  EXPECT_EQ(w.peak_open, popen);
  EXPECT_EQ(w.peak_roots, proots);
  EXPECT_LE(t.captured_entries, gabow_scc_trace_entries(int(n), edges));
}
} // namespace
TEST(GabowChecked, EveryFourVertexDigraphMatchesIndependentReachability) {
  std::size_t tested = 0;
  for (int n = 0; n <= 4; ++n)
    for (std::uint64_t mask = 0; mask < (UINT64_C(1) << (n * n)); ++mask) {
      SccAdjacency a(n);
      for (int u = 0; u < n; ++u)
        for (int v = 0; v < n; ++v)
          if ((mask >> (u * n + v)) & 1)
            a[u].push_back(v);
      const auto reach = closure(a);
      const auto r     = gabow_scc_work(n, a);
      ASSERT_EQ(r.components.size(), std::size_t(n));
      std::set<int> ids(r.components.begin(), r.components.end());
      EXPECT_EQ(ids.size(), r.work.component_closures);
      int next = 0;
      for (int id : ids)
        EXPECT_EQ(id, next++);
      for (int u = 0; u < n; ++u)
        for (int v = 0; v < n; ++v)
          ASSERT_EQ(r.components[u] == r.components[v], reach[u][v] && reach[v][u]);
      for (int u = 0; u < n; ++u)
        for (int v : a[u])
          if (r.components[u] != r.components[v])
            ASSERT_GT(r.components[u], r.components[v]);
      if (n <= 3)
        certificate(gabow_scc_trace(n, a));
      ++tested;
    }
  EXPECT_EQ(tested, 66067u);
}
TEST(GabowChecked, ActualPrimitiveCertificatesOnRandomDuplicateGraphs) {
  std::mt19937_64 rng(124);
  for (int trial = 0; trial < 300; ++trial) {
    const auto   n = int(rng() % 12);
    SccAdjacency a(n);
    for (int u = 0; u < n; ++u)
      for (int v = 0; v < n; ++v)
        if (rng() % 3 == 0) {
          a[u].push_back(v);
          if (trial % 3 == 0)
            a[u].push_back(v);
        }
    for (auto& row : a)
      std::shuffle(row.begin(), row.end(), rng);
    const auto t = gabow_scc_trace(n, a);
    certificate(t);
    const auto reach = closure(a);
    for (int u = 0; u < n; ++u)
      for (int v = 0; v < n; ++v)
        EXPECT_EQ(t.components[u] == t.components[v], reach[u][v] && reach[v][u]);
  }
}
TEST(GabowChecked, FinishedDFSVertexRemainsOpenForACrossEdge) {
  const auto t = gabow_scc_trace(3, {{1, 2}, {0}, {1}});
  certificate(t);
  EXPECT_EQ(t.components, (std::vector<int>{0, 0, 0}));
  bool finished_open = false, cross = false;
  for (const auto& e : t.events) {
    if (e.kind == "finish" && e.u == 1) {
      EXPECT_EQ(e.path, (std::vector<int>{0, 1}));
      EXPECT_EQ(e.stack, (std::vector<GabowDfsFrame>{{0, 1}}));
      EXPECT_EQ(e.comp[1], -1);
      finished_open = true;
    }
    if (e.kind == "edge" && e.u == 2 && e.v == 1) {
      EXPECT_EQ(e.path, (std::vector<int>{0, 1, 2}));
      EXPECT_EQ(e.roots, (std::vector<int>{0, 2}));
      EXPECT_EQ(e.stack, (std::vector<GabowDfsFrame>{{0, 2}, {2, 1}}));
      cross = true;
    }
  }
  EXPECT_TRUE(finished_open);
  EXPECT_TRUE(cross);
  EXPECT_EQ(t.work.peak_dfs, 2u);
  EXPECT_EQ(t.work.peak_open, 3u);
  EXPECT_EQ(t.work.peak_roots, 2u);
  EXPECT_EQ(t.work.tree_edges, 2u);
  EXPECT_EQ(t.work.open_edges, 2u);
  EXPECT_EQ(t.work.closed_edges, 0u);
  EXPECT_EQ(t.work.root_collapses, 2u);
  EXPECT_EQ(t.work.component_closures, 1u);
  EXPECT_EQ(t.captured_entries, 185u);
  EXPECT_EQ(t.events.size(), 16u);
}
TEST(GabowChecked, SingleEdgeMayCollapseManyRootsButTotalIsLinear) {
  constexpr int n = 80;
  SccAdjacency  cycle(n);
  for (int u = 0; u < n - 1; ++u)
    cycle[u].push_back(u + 1);
  cycle.back().push_back(0);
  const auto t = gabow_scc_trace(n, cycle, 0);
  certificate(t);
  std::size_t run = 0, longest = 0;
  for (const auto& e : t.events) {
    if (e.kind == "collapse") {
      ++run;
      longest = std::max(longest, run);
      EXPECT_EQ(e.u, n - 1);
      EXPECT_EQ(e.v, 0);
    } else
      run = 0;
  }
  EXPECT_EQ(longest, std::size_t(n - 1));
  EXPECT_EQ(t.work.root_pops, std::size_t(n));
  EXPECT_EQ(t.work.component_closures, 1u);
  EXPECT_EQ(t.work.edges_examined, std::size_t(n));
}
TEST(GabowChecked, DenseFamiliesAndDeepHeapDFSCounts) {
  constexpr int n = 80;
  SccAdjacency  complete(n), dag(n);
  for (int u = 0; u < n; ++u)
    for (int v = 0; v < n; ++v) {
      complete[u].push_back(v);
      if (u < v)
        dag[u].push_back(v);
    }
  const auto a = gabow_scc_work(n, complete), b = gabow_scc_work(n, dag);
  EXPECT_EQ(a.work.edges_examined, n * n);
  EXPECT_EQ(a.work.tree_edges, n - 1);
  EXPECT_EQ(a.work.open_edges, n * n - n + 1);
  EXPECT_EQ(a.work.closed_edges, 0u);
  EXPECT_EQ(a.work.root_collapses, n - 1);
  EXPECT_EQ(a.work.component_closures, 1u);
  EXPECT_EQ(b.work.edges_examined, n * (n - 1) / 2);
  EXPECT_EQ(b.work.tree_edges, n - 1);
  EXPECT_EQ(b.work.open_edges, 0u);
  EXPECT_EQ(b.work.closed_edges, n * (n - 1) / 2 - n + 1);
  EXPECT_EQ(b.work.root_collapses, 0u);
  EXPECT_EQ(b.work.component_closures, n);
  constexpr int deep = 200000;
  SccAdjacency  chain(deep);
  for (int u = 0; u < deep - 1; ++u)
    chain[u].push_back(u + 1);
  const auto c = gabow_scc_work(deep, chain);
  EXPECT_EQ(c.work.peak_dfs, deep);
  EXPECT_EQ(c.work.component_closures, deep);
  EXPECT_EQ(c.work.root_collapses, 0u);
  EXPECT_EQ(c.work.event_count, 5 * deep - 1);
  chain.back().push_back(0);
  const auto d = gabow_scc_work(deep, chain);
  EXPECT_EQ(d.work.peak_dfs, deep);
  EXPECT_EQ(d.work.component_closures, 1u);
  EXPECT_EQ(d.work.root_collapses, deep - 1);
  EXPECT_EQ(d.work.event_count, 5 * deep);
  EXPECT_THROW(gabow_scc_trace(deep, chain), std::length_error);
}
TEST(GabowChecked, LoopsDuplicatesAndInputPermutationPreserveThePartition) {
  std::mt19937 rng(8);
  for (int trial = 0; trial < 200; ++trial) {
    constexpr int n = 9;
    SccAdjacency  a(n);
    for (int u = 0; u < n; ++u)
      for (int v = 0; v < n; ++v)
        if (rng() % 5 == 0)
          a[u].push_back(v);
    auto changed = a;
    for (int u = 0; u < n; ++u) {
      changed[u].push_back(u);
      const auto copy = changed[u];
      changed[u].insert(changed[u].end(), copy.begin(), copy.end());
      std::shuffle(changed[u].begin(), changed[u].end(), rng);
    }
    EXPECT_TRUE(scc_same_partition(gabow_scc(n, a), gabow_scc(n, changed)));
    std::vector<int> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), rng);
    SccAdjacency renamed(n);
    for (int u = 0; u < n; ++u)
      for (int v : a[u])
        renamed[permutation[u]].push_back(permutation[v]);
    const auto old = gabow_scc(n, a), now = gabow_scc(n, renamed);
    for (int u = 0; u < n; ++u)
      for (int v = 0; v < n; ++v)
        EXPECT_EQ(old[u] == old[v], now[permutation[u]] == now[permutation[v]]);
  }
  const SccAdjacency a{{1, 2}, {}, {}};
  auto               reversed = a;
  std::reverse(reversed[0].begin(), reversed[0].end());
  EXPECT_NE(gabow_scc(3, a), gabow_scc(3, reversed));
  EXPECT_TRUE(scc_same_partition(gabow_scc(3, a), gabow_scc(3, reversed)));
}
TEST(GabowChecked, FullValidationPrecedesEmptyAndQuotaShortcuts) {
  EXPECT_THROW(gabow_scc(-1, {}), std::invalid_argument);
  EXPECT_THROW(gabow_scc(2, {{}}), std::invalid_argument);
  EXPECT_THROW(gabow_scc(0, {{-1}}), std::invalid_argument);
  for (const SccAdjacency a :
       std::vector<SccAdjacency>{{{1}, {0, -1}}, {{1}, {0, 2}}, {{1}, {INT_MAX}}})
    EXPECT_THROW(gabow_scc_trace(2, a, 1), std::invalid_argument);
  EXPECT_TRUE(gabow_scc(0, {}).empty());
  const auto empty = gabow_scc_trace(0, {}, 1);
  EXPECT_TRUE(empty.events.empty());
  EXPECT_EQ(empty.captured_entries, 0u);
  EXPECT_THROW(gabow_scc_trace_entries(-1, SIZE_MAX), std::invalid_argument);
  EXPECT_THROW(gabow_scc_trace_entries(0, 1), std::invalid_argument);
  EXPECT_THROW(gabow_scc_trace_entries(1, SIZE_MAX), std::length_error);
  EXPECT_THROW(gabow_scc_trace_entries(INT_MAX, SIZE_MAX), std::length_error);
  EXPECT_THROW(scc_summary({{1}, {}}, {0}), std::invalid_argument);
  EXPECT_THROW(scc_summary({{2}, {}}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(scc_summary({{}, {}}, {0, -1}), std::invalid_argument);
  EXPECT_THROW(scc_same_partition({0}, {-1, 0}), std::invalid_argument);
  EXPECT_THROW(scc_same_partition({-1}, {0}), std::invalid_argument);
}
TEST(GabowChecked, WholeConservativeAdmissionAndOwnedSnapshots) {
  SccAdjacency a{{1, 2}, {0}, {1}};
  EXPECT_EQ(gabow_scc_trace_entries(3, 4), 263u);
  EXPECT_EQ(gabow_scc_trace_entries(8, 10), 1740u);
  EXPECT_THROW(gabow_scc_trace(3, a, 262), std::length_error);
  EXPECT_THROW(gabow_scc_trace(3, a, 185), std::length_error);
  auto t = gabow_scc_trace(3, a, 263);
  EXPECT_EQ(t.captured_entries, 185u);
  certificate(t);
  a[0].clear();
  EXPECT_EQ(t.adjacency[0], (std::vector<int>{1, 2}));
  auto copy               = t;
  t.events[0].path[0]     = 99;
  t.events[0].preorder[0] = 99;
  EXPECT_EQ(copy.events[0].path[0], 0);
  EXPECT_EQ(t.events[1].path[0], 0);
  EXPECT_EQ(t.events[1].preorder[0], 1);
  EXPECT_EQ(t.components[0], 0);
  EXPECT_NO_THROW(gabow_scc_trace(3, a, 0));
  EXPECT_EQ(gabow_scc_trace_entries(0, 0), 0u);
  EXPECT_EQ(gabow_scc_trace(1, {{}}).captured_entries, 19u);
  EXPECT_EQ(gabow_scc_trace_entries(1, 0), 25u);
}
TEST(GabowChecked, PartitionBijectionAndQuotientAreIndependentOfLabelNames) {
  EXPECT_TRUE(scc_same_partition({INT_MAX, 7, INT_MAX}, {0, 3, 0}));
  EXPECT_FALSE(scc_same_partition({0, 0}, {0, 1}));
  EXPECT_FALSE(scc_same_partition({0, 1}, {0, 0}));
  EXPECT_TRUE(scc_same_partition({}, {}));
  EXPECT_FALSE(scc_same_partition({0}, {}));
  const auto q = scc_summary({{0, 1, 2, 2}, {0, 2}, {}}, {INT_MAX, INT_MAX, 7});
  ASSERT_EQ(q.groups.size(), 2u);
  EXPECT_EQ(q.groups[0].label, 7);
  EXPECT_EQ(q.groups[0].vertices, (std::vector<int>{2}));
  EXPECT_EQ(q.groups[1].label, INT_MAX);
  EXPECT_EQ(q.groups[1].vertices, (std::vector<int>{0, 1}));
  EXPECT_EQ(q.adjacency, (SccAdjacency{{}, {0}}));
  EXPECT_EQ(q.internal_edges, 3u);
  EXPECT_EQ(q.cross_edges, 3u);
  EXPECT_EQ(q.unique_cross_edges, 1u);
  EXPECT_TRUE(q.reverse_topological);
  const auto not_scc = scc_summary({{1}, {0}}, {0, 1});
  EXPECT_EQ(not_scc.adjacency, (SccAdjacency{{1}, {0}}));
  EXPECT_FALSE(not_scc.reverse_topological);
  const auto empty = scc_summary({}, {});
  EXPECT_TRUE(empty.groups.empty());
  EXPECT_TRUE(empty.adjacency.empty());
  EXPECT_TRUE(empty.reverse_topological);
}
TEST(GabowChecked, OriginalFixtureAndStrictlyDecreasingQuotientLabels) {
  const auto a = gabow_scc_example_graph();
  EXPECT_EQ(a, (SccAdjacency{{1}, {0, 2}, {3}, {4}, {2, 5}, {6}, {5, 7}, {}}));
  const auto t = gabow_scc_trace(8, a);
  certificate(t);
  EXPECT_EQ(t.components, (std::vector<int>{3, 3, 2, 2, 2, 1, 1, 0}));
  EXPECT_EQ(t.work.component_closures, 4u);
  EXPECT_EQ(t.work.root_collapses, 4u);
  EXPECT_EQ(t.work.tree_edges, 7u);
  EXPECT_EQ(t.work.open_edges, 3u);
  EXPECT_EQ(t.work.closed_edges, 0u);
  EXPECT_EQ(t.work.peak_dfs, 8u);
  EXPECT_EQ(t.work.peak_open, 8u);
  EXPECT_EQ(t.work.peak_roots, 4u);
  EXPECT_EQ(t.work.event_count, 42u);
  const auto q = scc_summary(a, t.components);
  EXPECT_EQ(q.adjacency, (SccAdjacency{{}, {0}, {1}, {2}}));
  EXPECT_EQ(q.internal_edges, 7u);
  EXPECT_EQ(q.cross_edges, 3u);
  EXPECT_EQ(q.unique_cross_edges, 3u);
  EXPECT_TRUE(q.reverse_topological);
  const auto                    reach = closure(a);
  std::set<std::pair<int, int>> expected;
  for (std::size_t u = 0; u < a.size(); ++u)
    for (int v : a[u])
      if (t.components[u] != t.components[v])
        expected.emplace(t.components[u], t.components[v]);
  for (std::size_t i = 0; i < q.groups.size(); ++i) {
    for (int u : q.groups[i].vertices) {
      EXPECT_EQ(t.components[u], q.groups[i].label);
      for (int v : q.groups[i].vertices)
        EXPECT_TRUE(reach[u][v] && reach[v][u]);
    }
    for (int j : q.adjacency[i])
      EXPECT_EQ(expected.erase({q.groups[i].label, q.groups[j].label}), 1u);
  }
  EXPECT_TRUE(expected.empty());
}
