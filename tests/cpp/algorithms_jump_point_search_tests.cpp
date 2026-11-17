#include <gtest/gtest.h>

#include <datamunge/algorithms/jump_point_search.hpp>
#include <queue>
#include <random>
using namespace datamunge::algorithms;
namespace {
int bfs(const std::vector<std::vector<bool>>& g, std::pair<int, int> s, std::pair<int, int> goal) {
  const int rows = g.size(), cols = g[0].size();
  if (g[s.first][s.second] || g[goal.first][goal.second])
    return -1;
  std::vector<std::vector<int>>   d(rows, std::vector<int>(cols, -1));
  std::queue<std::pair<int, int>> q;
  d[s.first][s.second] = 0;
  q.push(s);
  while (!q.empty()) {
    auto [r, c] = q.front();
    q.pop();
    if (std::pair{r, c} == goal)
      return d[r][c];
    for (auto [dr, dc] : std::vector<std::pair<int, int>>{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
      const int rr = r + dr, cc = c + dc;
      if (rr >= 0 && cc >= 0 && rr < rows && cc < cols && !g[rr][cc] && d[rr][cc] < 0) {
        d[rr][cc] = d[r][c] + 1;
        q.push({rr, cc});
      }
    }
  }
  return -1;
}
void compare(
    const std::vector<std::vector<bool>>& g,
    std::pair<int, int>                   s,
    std::pair<int, int>                   goal) {
  const auto r        = jump_point_search(g, s, goal);
  const int  expected = bfs(g, s, goal);
  ASSERT_EQ(r.found, expected >= 0);
  if (!r.found) {
    EXPECT_TRUE(r.path.empty());
    EXPECT_EQ(r.cost, 0);
    return;
  }
  EXPECT_EQ(r.cost, expected);
  ASSERT_EQ(r.path.size(), std::size_t(expected + 1));
  EXPECT_EQ(r.path.front(), s);
  EXPECT_EQ(r.path.back(), goal);
  for (std::size_t i = 0; i < r.path.size(); ++i) {
    auto [x, y] = r.path[i];
    ASSERT_FALSE(g[x][y]);
    if (i) {
      auto [a, b] = r.path[i - 1];
      EXPECT_EQ(std::abs(a - x) + std::abs(b - y), 1);
    }
  }
}
} // namespace
TEST(JumpPointSearch, JunctionCannotBeSkipped) {
  compare({{true, false, true}, {false, false, false}, {true, true, true}}, {1, 0}, {0, 1});
}
TEST(JumpPointSearch, AllThreeByThreeOccupanciesAndTerminals) {
  for (unsigned mask = 0; mask < 512; ++mask) {
    std::vector<std::vector<bool>> g(3, std::vector<bool>(3));
    for (int i = 0; i < 9; ++i)
      g[i / 3][i % 3] = mask & (1U << i);
    for (int s = 0; s < 9; ++s)
      for (int t = 0; t < 9; ++t)
        compare(g, {s / 3, s % 3}, {t / 3, t % 3});
  }
}
TEST(JumpPointSearch, RandomLargerMazes) {
  std::mt19937 rng(73);
  for (int trial = 0; trial < 1000; ++trial) {
    const int                      rows = 1 + rng() % 15, cols = 1 + rng() % 15;
    std::vector<std::vector<bool>> g(rows, std::vector<bool>(cols));
    for (auto& row : g)
      for (std::size_t j = 0; j < row.size(); ++j)
        row[j] = rng() % 4 == 0;
    compare(g, {int(rng() % rows), int(rng() % cols)}, {int(rng() % rows), int(rng() % cols)});
  }
}
TEST(JumpPointSearch, EmptyRaggedAndOutsideTerminals) {
  EXPECT_THROW((void)jump_point_search({}, {0, 0}, {0, 0}), std::invalid_argument);
  EXPECT_THROW((void)jump_point_search({{}}, {0, 0}, {0, 0}), std::invalid_argument);
  EXPECT_THROW(
      (void)jump_point_search({{false}, {false, false}}, {0, 0}, {1, 0}), std::invalid_argument);
  EXPECT_FALSE(jump_point_search({{false}}, {-1, 0}, {0, 0}).found);
  compare({{false}}, {0, 0}, {0, 0});
  compare({{false, false, false, false, false}}, {0, 0}, {0, 4});
}
