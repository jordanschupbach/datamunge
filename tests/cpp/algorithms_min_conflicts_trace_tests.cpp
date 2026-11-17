#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/min_conflicts.hpp>
#include <limits>
#include <random>
#include <stdexcept>
namespace {
long long pairs(const std::vector<int>& q) {
  long long total = 0;
  for (std::size_t i = 0; i < q.size(); ++i)
    for (std::size_t j = i + 1; j < q.size(); ++j)
      if (q[i] >= 0 && q[j] >= 0 && (q[i] == q[j] || std::abs(q[i] - q[j]) == int(j - i)))
        ++total;
  return total;
}
} // namespace
TEST(MinConflictsTrace, IndependentPairReplayAndActualRNGChoices) {
  using namespace datamunge::algorithms;
  for (int n = 0; n <= 12; ++n)
    for (unsigned seed = 0; seed < 15; ++seed) {
      auto t        = min_conflicts_nqueens_trace(n, 60, seed);
      auto ordinary = min_conflicts_nqueens(n, 60, seed);
      EXPECT_EQ(ordinary.queens, t.result.queens);
      EXPECT_EQ(ordinary.steps, t.result.steps);
      EXPECT_EQ(ordinary.solved, t.result.solved);
      std::mt19937_64  rng(seed);
      std::vector<int> q(n, -1);
      std::size_t      repairs = 0, units = 3 + n;
      for (auto& m : t.moves) {
        EXPECT_EQ(m.conflicts_before, pairs(q));
        if (m.initialization) {
          EXPECT_EQ(m.row, int(std::count_if(q.begin(), q.end(), [](int x) { return x >= 0; })));
          EXPECT_TRUE(m.conflicted.empty());
        } else {
          ++repairs;
          std::vector<int> conflict;
          for (int r = 0; r < n; ++r) {
            bool yes = false;
            for (int s = 0; s < n; ++s)
              if (r != s && (q[r] == q[s] || std::abs(q[r] - q[s]) == std::abs(r - s)))
                yes = true;
            if (yes)
              conflict.push_back(r);
          }
          EXPECT_EQ(m.conflicted, conflict);
          auto draw = rng();
          EXPECT_EQ(m.row_draw, draw);
          ASSERT_FALSE(conflict.empty());
          EXPECT_EQ(m.row, conflict[draw % conflict.size()]);
          EXPECT_EQ(m.column_before, q[m.row]);
        }
        q[m.row] = -1;
        std::vector<int> costs(n), ties;
        for (int c = 0; c < n; ++c)
          for (int r = 0; r < n; ++r)
            if (q[r] >= 0 && (q[r] == c || std::abs(q[r] - c) == std::abs(r - m.row)))
              ++costs[c];
        EXPECT_EQ(m.costs, costs);
        int best = *std::min_element(costs.begin(), costs.end());
        for (int c = 0; c < n; ++c)
          if (costs[c] == best)
            ties.push_back(c);
        EXPECT_EQ(m.candidates, ties);
        auto draw = rng();
        EXPECT_EQ(m.column_draw, draw);
        EXPECT_EQ(m.column_after, ties[draw % ties.size()]);
        q[m.row] = m.column_after;
        EXPECT_EQ(m.conflicts_after, pairs(q));
        units += 1 + m.costs.size() + m.candidates.size() + m.conflicted.size();
      }
      EXPECT_EQ(q, t.result.queens);
      EXPECT_EQ(t.result.steps, int(repairs));
      EXPECT_EQ(t.result.solved, pairs(q) == 0);
      EXPECT_EQ(t.captured_entries, units);
      EXPECT_NO_THROW(min_conflicts_nqueens_trace(n, 60, seed, units));
      if (units > 1)
        EXPECT_THROW(min_conflicts_nqueens_trace(n, 60, seed, units - 1), std::length_error);
    }
}
TEST(MinConflictsTrace, GuardsAndLastMoveTerminalCheck) {
  using namespace datamunge::algorithms;
  EXPECT_THROW(min_conflicts_nqueens(-1, 0, 1), std::invalid_argument);
  EXPECT_THROW(min_conflicts_nqueens(1, -1, 1), std::invalid_argument);
  EXPECT_THROW(min_conflicts_nqueens(std::numeric_limits<int>::max(), 0, 1), std::length_error);
  EXPECT_TRUE(min_conflicts_nqueens(0, 0, 0).solved);
  EXPECT_TRUE(min_conflicts_nqueens(1, 0, 0).solved);
  EXPECT_FALSE(min_conflicts_nqueens(2, 20, 0).solved);
  EXPECT_FALSE(min_conflicts_nqueens(3, 20, 0).solved);
  bool found = false;
  for (unsigned seed = 0; seed < 100; ++seed) {
    auto t = min_conflicts_nqueens_trace(4, 100, seed);
    if (t.result.solved && t.result.steps > 0) {
      auto exact = min_conflicts_nqueens(4, t.result.steps, seed);
      EXPECT_TRUE(exact.solved);
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}
