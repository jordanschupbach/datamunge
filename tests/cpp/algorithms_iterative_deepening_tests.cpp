#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/iterative_deepening.hpp>
#include <functional>
#include <limits>
#include <queue>
using namespace datamunge::algorithms;
namespace {
using Edge = std::pair<std::size_t, std::size_t>;
BasicPathResult recursive(
    std::size_t              n,
    const std::vector<Edge>& edges,
    std::size_t              start,
    std::size_t              goal,
    std::size_t              maximum) {
  std::vector<std::vector<std::size_t>> a(n);
  for (auto [u, v] : edges)
    a[u].push_back(v);
  BasicPathResult out;
  for (std::size_t limit = 0; limit <= maximum; ++limit) {
    std::vector<bool>                             on(n);
    std::vector<std::size_t>                      path;
    std::function<bool(std::size_t, std::size_t)> walk = [&](auto u, auto depth) {
      ++out.expanded;
      path.push_back(u);
      if (u == goal)
        return true;
      if (depth < limit) {
        on[u] = true;
        for (auto v : a[u])
          if (!on[v] && walk(v, depth + 1))
            return true;
        on[u] = false;
      }
      path.pop_back();
      return false;
    };
    if (walk(start, 0)) {
      out.path  = path;
      out.found = true;
      break;
    }
  }
  return out;
}
std::size_t
bfs_depth(std::size_t n, const std::vector<Edge>& edges, std::size_t start, std::size_t goal) {
  std::vector<std::size_t> d(n, n);
  d[start] = 0;
  std::queue<std::size_t> q;
  q.push(start);
  while (!q.empty()) {
    auto u = q.front();
    q.pop();
    for (auto [a, b] : edges)
      if (a == u && d[b] == n) {
        d[b] = d[u] + 1;
        q.push(b);
      }
  }
  return d[goal];
}
void check(
    std::size_t              n,
    const std::vector<Edge>& edges,
    std::size_t              start,
    std::size_t              goal,
    std::size_t              depth) {
  auto t   = iterative_deepening_dfs_trace(n, edges, start, goal, depth);
  auto old = recursive(n, edges, start, goal, std::min(depth, n - 1));
  EXPECT_EQ(t.run.result.path, old.path);
  EXPECT_EQ(t.run.result.expanded, old.expanded);
  EXPECT_EQ(t.run.result.found, old.found);
  auto bd = bfs_depth(n, edges, start, goal);
  EXPECT_EQ(t.run.result.found, bd <= std::min(depth, n - 1));
  if (t.run.result.found)
    EXPECT_EQ(t.run.result.path.size(), bd + 1);
  std::size_t              total = 0, pass = 0;
  std::vector<std::size_t> active;
  DeepeningPass            counts;
  for (const auto& e : t.events) {
    ASSERT_EQ(e.limit, pass);
    switch (e.kind) {
      case DeepeningEventKind::pass_start:
        EXPECT_TRUE(active.empty());
        counts       = {};
        counts.limit = pass;
        break;
      case DeepeningEventKind::enter:
        EXPECT_EQ(e.depth, active.size());
        EXPECT_EQ(std::find(active.begin(), active.end(), e.vertex), active.end());
        active.push_back(e.vertex);
        ++counts.entered;
        counts.maximum_stack = std::max(counts.maximum_stack, active.size());
        break;
      case DeepeningEventKind::examine:
        ASSERT_FALSE(active.empty());
        EXPECT_EQ(active.back(), e.vertex);
        ASSERT_LT(e.edge_index, edges.size());
        EXPECT_EQ(edges[e.edge_index], (Edge{e.vertex, e.neighbor}));
        EXPECT_EQ(
            e.cycle_blocked, std::find(active.begin(), active.end(), e.neighbor) != active.end());
        ++counts.edges_examined;
        counts.cycle_skips += e.cycle_blocked;
        break;
      case DeepeningEventKind::cutoff:
        EXPECT_EQ(e.depth, e.limit);
        EXPECT_NE(e.vertex, goal);
        ++counts.cutoffs;
        break;
      case DeepeningEventKind::leave:
        ASSERT_FALSE(active.empty());
        EXPECT_EQ(active.back(), e.vertex);
        active.pop_back();
        break;
      case DeepeningEventKind::goal:
        EXPECT_EQ(e.vertex, goal);
        counts.found = true;
        EXPECT_EQ(active, t.run.result.path);
        break;
      case DeepeningEventKind::pass_end: {
        const auto& p = t.run.passes[pass];
        EXPECT_EQ(p.entered, counts.entered);
        EXPECT_EQ(p.edges_examined, counts.edges_examined);
        EXPECT_EQ(p.cutoffs, counts.cutoffs);
        EXPECT_EQ(p.cycle_skips, counts.cycle_skips);
        EXPECT_EQ(p.maximum_stack, counts.maximum_stack);
        EXPECT_EQ(p.found, counts.found);
        total += p.entered;
        ++pass;
        break;
      }
    }
  }
  EXPECT_EQ(total, t.run.result.expanded);
  EXPECT_EQ(pass, t.run.passes.size());
  EXPECT_EQ(iterative_deepening_dfs(n, edges, start, goal, depth).path, t.run.result.path);
}
} // namespace
TEST(IterativeDeepening, ExhaustiveOrderedGraphsMatchLegacyAndBfs) {
  for (unsigned mask = 0; mask < 4096; ++mask) {
    std::vector<Edge> e;
    unsigned          bit = 0;
    for (unsigned u = 0; u < 4; ++u)
      for (unsigned v = 0; v < 4; ++v)
        if (u != v) {
          if (mask & (1u << bit))
            e.push_back({u, v});
          ++bit;
        }
    for (unsigned s = 0; s < 4; ++s)
      for (unsigned g = 0; g < 4; ++g)
        check(4, e, s, g, 3);
  }
  for (unsigned mask = 0; mask < 512; ++mask) {
    std::vector<Edge> e;
    for (unsigned bit = 0; bit < 9; ++bit)
      if (mask & (1u << bit))
        e.push_back({bit / 3, bit % 3});
    check(3, e, 0, 2, 2);
  }
}
TEST(IterativeDeepening, PathLocalCyclesDuplicatesAndShallowest) {
  check(6, {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {3, 4}, {4, 5}}, 0, 5, 3);
  EXPECT_EQ(
      iterative_deepening_dfs(6, {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {3, 4}, {4, 5}}, 0, 5, 3).path,
      (std::vector<std::size_t>{0, 3, 4, 5}));
  check(4, {{0, 0}, {0, 1}, {0, 1}, {1, 0}, {1, 2}, {2, 1}}, 0, 3, 3);
  check(2, {{0, 1}}, 0, 1, 0);
  check(2, {{0, 1}}, 0, 1, 1);
  check(1, {{0, 0}}, 0, 0, 0);
}
TEST(IterativeDeepening, SafeMaximumDepthAndExplicitFrames) {
  auto bound = iterative_deepening_dfs_run(
      3, {{0, 1}, {1, 0}}, 0, 2, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(bound.effective_max_depth, 2);
  EXPECT_EQ(bound.passes.size(), 3);
  EXPECT_FALSE(bound.result.found);
  const std::size_t n = 2001;
  std::vector<Edge> e;
  for (std::size_t i = 1; i < n; ++i)
    e.push_back({i - 1, i});
  auto r = iterative_deepening_dfs_run(n, e, 0, n - 1, n - 1);
  EXPECT_EQ(r.result.path.size(), n);
  EXPECT_EQ(r.passes.back().maximum_stack, n);
  EXPECT_EQ(r.result.expanded, n * (n + 1) / 2);
}
TEST(IterativeDeepening, CompleteAdmissionOwnershipAndInvalidUnusedArcs) {
  auto t   = iterative_deepening_dfs_trace(3, {{0, 1}, {1, 2}}, 0, 2, 2);
  auto cap = t.events.size();
  EXPECT_NO_THROW(iterative_deepening_dfs_trace(3, t.edges, 0, 2, 2, {3, 2, cap}));
  EXPECT_THROW(
      iterative_deepening_dfs_trace(3, t.edges, 0, 2, 2, {3, 2, cap - 1}), std::length_error);
  EXPECT_THROW(iterative_deepening_dfs_trace(3, t.edges, 0, 2, 2, {2, 2, 0}), std::length_error);
  EXPECT_THROW(iterative_deepening_dfs_trace(3, t.edges, 0, 2, 2, {3, 1, 0}), std::length_error);
  EXPECT_NO_THROW(iterative_deepening_dfs_run(3, t.edges, 0, 2, 2, {3, 2, 1}));
  EXPECT_NO_THROW(iterative_deepening_dfs_trace(3, t.edges, 0, 2, 2, {0, 0, 0}));
  EXPECT_THROW(iterative_deepening_dfs(0, {}, 0, 0, 0), std::invalid_argument);
  EXPECT_THROW(
      iterative_deepening_dfs_trace(3, {{2, 3}}, 0, 0, 0, {1, 0, 0}), std::invalid_argument);
  std::vector<Edge> e{{0, 1}};
  auto              h = iterative_deepening_dfs_trace(2, e, 0, 1, 1);
  e[0]                = {1, 0};
  EXPECT_EQ(h.edges[0], (Edge{0, 1}));
}

#include <datamunge/algorithms/iterative_deepening_analysis.hpp>
TEST(IterativeDeepening, NativeCompleteTreeWorkAndPassDiagram) {
  for (std::size_t b = 1; b <= 3; ++b)
    for (std::size_t depth = 0; depth <= 7; ++depth) {
      auto        t        = deepening_tree(b, depth);
      auto        r        = iterative_deepening_dfs_run(t.n, t.edges, 0, t.goal, depth);
      std::size_t expected = 0, level = 1;
      for (std::size_t k = 0; k <= depth; ++k) {
        expected += (depth - k + 1) * level;
        level *= b;
      }
      EXPECT_EQ(r.result.expanded, expected);
      EXPECT_EQ(r.result.path.size(), depth + 1);
      EXPECT_EQ(r.passes.back().maximum_stack, depth + 1);
    }
  auto t = deepening_tree(2, 2);
  EXPECT_EQ(t.n, 7);
  EXPECT_EQ(t.edges.back(), (Edge{2, 6}));
  auto r = iterative_deepening_dfs_run(t.n, t.edges, 0, t.goal, t.depth);
  EXPECT_FALSE(deepening_pass_diagram(r).to_svg().empty());
  EXPECT_THROW(deepening_pass_diagram(r, 2), std::length_error);
  auto bad            = r;
  bad.result.expanded = 0;
  EXPECT_THROW(deepening_pass_diagram(bad), std::invalid_argument);
  bad                 = r;
  bad.passes[1].found = true;
  EXPECT_THROW(deepening_pass_diagram(bad), std::invalid_argument);
  EXPECT_THROW(deepening_tree(0, 3), std::invalid_argument);
  EXPECT_THROW(deepening_tree(2, 3, 14), std::length_error);
  EXPECT_NO_THROW(deepening_tree(2, 3, 15));
  EXPECT_THROW(deepening_tree(1, std::numeric_limits<std::size_t>::max(), 0), std::length_error);
  EXPECT_THROW(deepening_tree(2, std::numeric_limits<std::size_t>::max(), 0), std::length_error);
}
