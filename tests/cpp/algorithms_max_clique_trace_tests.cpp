#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/max_clique.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
std::size_t brute(const std::vector<std::vector<int>>& g) {
  std::size_t best = 0;
  for (unsigned mask = 0; mask < (1u << g.size()); ++mask) {
    bool        valid = true;
    std::size_t size  = 0;
    for (std::size_t i = 0; i < g.size(); ++i)
      if ((mask >> i) & 1) {
        ++size;
        for (std::size_t j = 0; j < i; ++j)
          if ((mask >> j) & 1)
            valid &= std::find(g[i].begin(), g[i].end(), j) != g[i].end() ||
                     std::find(g[j].begin(), g[j].end(), i) != g[j].end();
      }
    if (valid)
      best = std::max(best, size);
  }
  return best;
}
void verify(const MaxCliqueTrace& t) {
  auto graph    = max_clique_normalize(t.n, t.input);
  auto adjacent = [&](int a, int b) {
    return std::find(graph[a].begin(), graph[a].end(), b) != graph[a].end();
  };
  std::size_t entries = t.n + t.colour_checks.size() + t.decisions.size(), checks = 0;
  for (auto& row : t.input)
    entries += row.size();
  for (auto& node : t.nodes) {
    entries +=
        1 + node.clique.size() + node.candidates.size() + node.order.size() + node.colours.size();
    for (std::size_t i = 0; i < node.clique.size(); ++i)
      for (std::size_t j = 0; j < i; ++j)
        EXPECT_TRUE(adjacent(node.clique[i], node.clique[j]));
    for (int v : node.candidates)
      for (int r : node.clique)
        EXPECT_TRUE(adjacent(v, r));
    std::vector<std::vector<int>> classes;
    for (int v : node.candidates) {
      std::size_t c = 0;
      for (; c < classes.size(); ++c) {
        bool conflict = false;
        for (int u : classes[c]) {
          ASSERT_LT(checks, t.colour_checks.size());
          auto& event = t.colour_checks[checks++];
          EXPECT_EQ(event.node, node.id);
          EXPECT_EQ(event.vertex, v);
          EXPECT_EQ(event.against, u);
          EXPECT_EQ(event.colour, c + 1);
          conflict = adjacent(v, u);
          EXPECT_EQ(event.conflict, conflict);
          if (conflict)
            break;
        }
        if (!conflict)
          break;
      }
      if (c == classes.size())
        classes.emplace_back();
      classes[c].push_back(v);
    }
    std::vector<int> order, colours;
    for (std::size_t c = 0; c < classes.size(); ++c)
      for (int v : classes[c]) {
        order.push_back(v);
        colours.push_back(c + 1);
      }
    EXPECT_EQ(node.order, order);
    EXPECT_EQ(node.colours, colours);
  }
  EXPECT_EQ(checks, t.colour_checks.size());
  EXPECT_EQ(entries, t.captured_entries);
  EXPECT_EQ(t.analysis.work.nodes, t.nodes.size());
  EXPECT_EQ(t.analysis.work.colour_checks, t.colour_checks.size());
  std::size_t      branches = 0, prunes = 0, best = 0, improvements = 0, candidatechecks = 0;
  std::vector<int> witness;
  for (auto& event : t.decisions) {
    ASSERT_LT(event.node, t.nodes.size());
    auto& node = t.nodes[event.node];
    EXPECT_EQ(event.active_size, node.clique.size());
    EXPECT_EQ(event.best_before, best);
    if (event.kind == 0) {
      EXPECT_TRUE(node.candidates.empty());
      if (node.clique.size() > best) {
        best    = node.clique.size();
        witness = node.clique;
        ++improvements;
      }
      EXPECT_EQ(event.bound, node.clique.size());
    } else {
      auto it = std::find(node.order.begin(), node.order.end(), event.vertex);
      ASSERT_NE(it, node.order.end());
      std::size_t index = it - node.order.begin();
      EXPECT_EQ(event.bound, node.clique.size() + node.colours[index]);
      if (event.kind == 2) {
        ++prunes;
        EXPECT_LE(event.bound, best);
      } else {
        EXPECT_EQ(event.kind, 1);
        EXPECT_GT(event.bound, best);
        ++branches;
        candidatechecks += index;
        if (branches < t.nodes.size()) {
          auto expected = node.clique;
          expected.push_back(event.vertex);
          EXPECT_EQ(t.nodes[branches].clique, expected);
          std::vector<int> candidates;
          for (std::size_t j = 0; j < index; ++j)
            if (adjacent(event.vertex, node.order[j]))
              candidates.push_back(node.order[j]);
          EXPECT_EQ(t.nodes[branches].candidates, candidates);
        }
      }
    }
    EXPECT_EQ(event.best_after, best);
  }
  std::sort(witness.begin(), witness.end());
  EXPECT_EQ(witness, t.analysis.clique);
  EXPECT_EQ(branches, t.analysis.work.branches);
  EXPECT_EQ(prunes, t.analysis.work.prunes);
  EXPECT_EQ(improvements, t.analysis.work.improvements);
  EXPECT_EQ(candidatechecks, t.analysis.work.candidate_checks);
}
} // namespace
TEST(MaxCliqueTrace, EverySimpleGraphThroughFiveVerticesExactBruteForceAndActualColourSearch) {
  for (int n = 0; n <= 5; ++n) {
    unsigned edges = n * (n - 1) / 2;
    for (unsigned mask = 0; mask < (1u << edges); ++mask) {
      std::vector<std::vector<int>> g(n);
      unsigned                      bit = 0;
      for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j, ++bit)
          if ((mask >> bit) & 1) {
            g[i].push_back(j);
            g[j].push_back(i);
          }
      auto t = max_clique_trace(n, g);
      verify(t);
      EXPECT_TRUE(t.analysis.complete);
      EXPECT_EQ(t.analysis.clique.size(), brute(g));
      EXPECT_EQ(t.analysis.upper_bound, t.analysis.clique.size());
      EXPECT_EQ(max_clique_dyn(n, g), t.analysis.clique);
    }
  }
}
TEST(MaxCliqueTrace, OriginalAsymmetricFixtureWorkBudgetsGuardsAndCaptureOwnership) {
  auto g = max_clique_demo_graph();
  auto t = max_clique_trace(8, g);
  EXPECT_EQ(t.analysis.clique, (std::vector<int>{0, 1, 2, 3}));
  auto normalized = max_clique_normalize(8, g);
  EXPECT_EQ(normalized[7], (std::vector<int>{6}));
  EXPECT_NE(std::find(normalized[6].begin(), normalized[6].end(), 7), normalized[6].end());
  verify(t);
  for (std::size_t cap = 1; cap <= t.analysis.work.nodes + 1; ++cap) {
    auto limited = max_clique_trace(8, g, cap);
    verify(limited);
    EXPECT_LE(limited.analysis.work.nodes, cap);
    EXPECT_LE(limited.analysis.clique.size(), 4);
    EXPECT_GE(limited.analysis.upper_bound, 4);
    if (limited.analysis.complete)
      EXPECT_EQ(limited.analysis.clique.size(), 4);
  }
  EXPECT_NO_THROW(max_clique_trace(8, g, 0, t.captured_entries));
  EXPECT_THROW(max_clique_trace(8, g, 0, t.captured_entries - 1), std::length_error);
  EXPECT_NO_THROW(max_clique_trace(8, g, 0, 0));
  EXPECT_EQ(max_clique_trace(0, {}).captured_entries, 2);
  EXPECT_NO_THROW(max_clique_trace(0, {}, 0, 2));
  EXPECT_THROW(max_clique_trace(0, {}, 0, 1), std::length_error);
  EXPECT_THROW(max_clique_dyn(-1, {}), std::invalid_argument);
  EXPECT_THROW(max_clique_dyn(2, {{}}), std::invalid_argument);
  EXPECT_THROW(max_clique_trace(2, {{-1}, {}}, 0, 1), std::invalid_argument);
  EXPECT_THROW(max_clique_dyn(2, {{}, {2}}), std::invalid_argument);
  EXPECT_EQ(max_clique_dyn(2, {{0, 1, 1}, {1}}).size(), 2);
  g[0].clear();
  EXPECT_FALSE(t.input[0].empty());
}
TEST(MaxCliqueTrace, OriginalDensityDrawStreamAndExactReturnedWitnesses) {
  auto rows = max_clique_density_study();
  ASSERT_EQ(rows.size(), 9);
  std::mt19937_64 rng(31);
  for (std::size_t k = 0; k < rows.size(); ++k) {
    auto& row = rows[k];
    EXPECT_EQ(row.percent, 10 * (k + 1));
    std::vector<std::vector<int>> expected(40);
    for (int u = 0; u < 40; ++u)
      for (int v = u + 1; v < 40; ++v)
        if (static_cast<int>(rng() % 100) < row.percent) {
          expected[u].push_back(v);
          expected[v].push_back(u);
        }
    EXPECT_EQ(row.graph, expected);
    EXPECT_TRUE(row.analysis.complete);
    for (int u : row.analysis.clique)
      for (int v : row.analysis.clique)
        if (u != v)
          EXPECT_NE(std::find(expected[u].begin(), expected[u].end(), v), expected[u].end());
  }
}
