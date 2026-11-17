#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/lex_bfs.hpp>
#include <numeric>
using namespace datamunge::algorithms;
namespace {
// Independent explicit lexicographic labels; no ordered partition refinement.
std::vector<int> label_order(const std::vector<std::vector<int>>& g) {
  std::size_t                   n = g.size();
  std::vector<std::vector<int>> label(n);
  std::vector<char>             seen(n);
  std::vector<int>              out;
  for (std::size_t k = 0; k < n; ++k) {
    int best = -1;
    for (std::size_t i = 0; i < n; ++i)
      if (!seen[i] &&
          (best < 0 || label[i] > label[best] || (label[i] == label[best] && int(i) > best)))
        best = i;
    out.push_back(best);
    seen[best] = 1;
    for (int v : g[best])
      if (!seen[v])
        label[v].push_back(int(n - k));
  }
  return out;
}
bool brute_peo(const std::vector<std::vector<int>>& g, const std::vector<int>& p) {
  std::vector<std::vector<bool>> a(g.size(), std::vector<bool>(g.size()));
  for (std::size_t u = 0; u < g.size(); ++u)
    for (int v : g[u])
      a[u][v] = a[v][u] = true;
  for (std::size_t i = 0; i < p.size(); ++i)
    for (std::size_t j = i + 1; j < p.size(); ++j)
      for (std::size_t k = j + 1; k < p.size(); ++k)
        if (a[p[i]][p[j]] && a[p[i]][p[k]] && !a[p[j]][p[k]])
          return false;
  return true;
}
} // namespace
TEST(LexBFSTrace, EveryFiveVertexGraphLabelsAndChordality) {
  for (unsigned mask = 0; mask < 1024; ++mask) {
    std::vector<std::vector<int>> g(5);
    int                           edge = 0;
    for (int i = 0; i < 5; ++i)
      for (int j = i + 1; j < 5; ++j, ++edge)
        if (mask >> edge & 1) {
          g[i].push_back(j);
          g[j].push_back(i);
        }
    auto t = lex_bfs_trace(5, g);
    EXPECT_EQ(t.analysis.order, label_order(g));
    EXPECT_EQ(t.analysis.work.neighbor_tests, 10);
    EXPECT_EQ(t.classifications.size(), 10);
    EXPECT_EQ(t.stages.size(), 6);
    EXPECT_EQ(t.analysis.order, lex_bfs(5, g));
    for (auto& e : t.classifications) {
      EXPECT_EQ(e.selected, t.analysis.order[e.step]);
      EXPECT_EQ(e.neighbor, bool(t.matrix[e.selected][e.vertex]));
    }
    for (std::size_t k = 0; k < t.stages.size(); ++k) {
      std::vector<int> members;
      for (auto& b : t.stages[k].blocks)
        members.insert(members.end(), b.begin(), b.end());
      EXPECT_EQ(members.size(), 5 - k);
      for (int v : members)
        EXPECT_TRUE(
            std::find(t.analysis.order.begin(), t.analysis.order.begin() + k, v) ==
            t.analysis.order.begin() + k);
    }
    auto reverse = t.analysis.order;
    std::reverse(reverse.begin(), reverse.end());
    auto check = check_perfect_elimination_order(5, g, reverse);
    EXPECT_EQ(check.valid, brute_peo(g, reverse));
    std::vector<int> candidate{0, 1, 2, 3, 4};
    bool             any = false;
    do {
      if (brute_peo(g, candidate)) {
        any = true;
        break;
      }
    } while (std::next_permutation(candidate.begin(), candidate.end()));
    EXPECT_EQ(check.valid, any);
    if (!check.valid) {
      EXPECT_TRUE(t.matrix[check.vertex][check.left]);
      EXPECT_TRUE(t.matrix[check.vertex][check.right]);
      EXPECT_FALSE(t.matrix[check.left][check.right]);
    }
  }
}
TEST(LexBFSTrace, OriginalAndNormalization) {
  std::vector<std::vector<int>> g{
      {1, 2}, {0, 2, 3}, {0, 1, 3}, {1, 2, 4, 5}, {3, 5}, {3, 4, 6}, {5}};
  EXPECT_EQ(lex_bfs(7, g), (std::vector<int>{6, 5, 4, 3, 2, 1, 0}));
  auto t = lex_bfs_trace(3, {{0, 1, 1}, {2}, {}});
  EXPECT_EQ(t.matrix, (std::vector<std::vector<char>>{{0, 1, 0}, {1, 0, 1}, {0, 1, 0}}));
  EXPECT_EQ(t.analysis.order, (std::vector<int>{2, 1, 0}));
  EXPECT_EQ(t.analysis.work.adjacency_entries, 4);
  EXPECT_EQ(t.captured_entries, 26);
}
TEST(LexBFSTrace, GuardsQuotaEmptyAndOwnership) {
  EXPECT_THROW(lex_bfs(-1, {}), std::invalid_argument);
  EXPECT_THROW(lex_bfs(2, {{1}}), std::invalid_argument);
  EXPECT_THROW(lex_bfs(2, {{2}, {}}), std::invalid_argument);
  EXPECT_THROW(lex_bfs(2, {{-1}, {}}), std::invalid_argument);
  EXPECT_THROW(check_perfect_elimination_order(2, {{1}, {0}}, {1, 1}), std::invalid_argument);
  EXPECT_THROW(check_perfect_elimination_order(2, {{1}, {0}}, {1}), std::invalid_argument);
  EXPECT_EQ(lex_bfs_trace(0, {}).captured_entries, 1);
  EXPECT_TRUE(lex_bfs(0, {}).empty());
  EXPECT_THROW(lex_bfs_trace(2, {{1}, {0}}, 12), std::length_error);
  EXPECT_EQ(lex_bfs_trace(2, {{1}, {0}}, 13).captured_entries, 13);
  std::vector<std::vector<int>> g{{1}, {0}};
  auto                          t = lex_bfs_trace(2, g, 0);
  g[0].clear();
  EXPECT_EQ(t.input_adjacency[0], (std::vector<int>{1}));
  EXPECT_TRUE(check_perfect_elimination_order(0, {}, {}).valid);
}
