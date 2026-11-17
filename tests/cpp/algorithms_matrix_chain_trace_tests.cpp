#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/matrix_chain.hpp>
#include <functional>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
std::vector<long long> costs(const std::vector<int>& d, std::size_t i, std::size_t j) {
  if (i == j)
    return {0};
  std::vector<long long> out;
  for (auto k = i; k < j; ++k)
    for (auto l : costs(d, i, k))
      for (auto r : costs(d, k + 1, j))
        out.push_back(l + r + static_cast<long long>(d[i - 1]) * d[k] * d[j]);
  return out;
}
} // namespace
TEST(MatrixChainTrace, EverySmallDimensionSequenceAndProducingHistory) {
  for (std::size_t n = 1; n <= 6; ++n) {
    std::size_t count = 1;
    for (std::size_t q = 0; q <= n; ++q)
      count *= 3;
    for (std::size_t bits = 0; bits < count; ++bits) {
      auto             b = bits;
      std::vector<int> d(n + 1);
      for (auto& x : d) {
        x = 1 + b % 3;
        b /= 3;
      }
      auto t   = matrix_chain_trace(d);
      auto all = costs(d, 1, n);
      EXPECT_EQ(t.analysis.result.cost, *std::min_element(all.begin(), all.end()));
      std::size_t                             step = 0, commit = 0, improved = 0;
      std::vector<std::vector<std::uint64_t>> table(n + 1, std::vector<std::uint64_t>(n + 1));
      for (std::size_t len = 2; len <= n; ++len)
        for (std::size_t i = 1; i + len <= n + 1; ++i) {
          auto          j     = i + len - 1;
          std::uint64_t best  = UINT64_MAX;
          std::size_t   split = 0;
          for (std::size_t k = i; k < j; ++k) {
            ASSERT_LT(step, t.trials.size());
            auto& r      = t.trials[step++];
            auto  p      = static_cast<std::uint64_t>(d[i - 1]) * d[k] * d[j];
            auto  value  = table[i][k] + table[k + 1][j] + p;
            bool  select = value < best;
            EXPECT_EQ(r.i, i);
            EXPECT_EQ(r.j, j);
            EXPECT_EQ(r.k, k);
            EXPECT_EQ(r.left, table[i][k]);
            EXPECT_EQ(r.right, table[k + 1][j]);
            EXPECT_EQ(r.product, p);
            EXPECT_EQ(r.cost, value);
            EXPECT_EQ(r.selected, select);
            if (select) {
              best  = value;
              split = k;
              ++improved;
            }
          }
          table[i][j] = best;
          auto& r     = t.commits[commit++];
          EXPECT_EQ(r.i, i);
          EXPECT_EQ(r.j, j);
          EXPECT_EQ(r.split, split);
          EXPECT_EQ(r.cost, best);
          EXPECT_EQ(t.analysis.splits[i][j], split);
        }
      EXPECT_EQ(table, t.analysis.costs);
      EXPECT_EQ(step, n * (n - 1) * (n + 1) / 6);
      EXPECT_EQ(commit, n * (n - 1) / 2);
      EXPECT_EQ(improved, t.analysis.work.improvements);
      EXPECT_EQ(t.analysis.work.trials, step);
      EXPECT_EQ(t.analysis.work.cells, commit);
      EXPECT_EQ(t.captured_entries, d.size() + 2 * (n + 1) * (n + 1) + step + commit);
      EXPECT_EQ(matrix_chain_order(d).parenthesization, t.analysis.result.parenthesization);
    }
  }
}
TEST(MatrixChainTrace, OriginalTiesEmptyQuotaAndOwnership) {
  auto d = matrix_chain_demo_dimensions();
  auto t = matrix_chain_trace(d);
  EXPECT_EQ(t.analysis.result.cost, 15125);
  EXPECT_EQ(t.analysis.result.parenthesization, "((A1(A2A3))((A4A5)A6))");
  EXPECT_EQ(matrix_chain_left_cost(d), 40500);
  EXPECT_EQ(matrix_chain_right_cost(d), 47500);
  EXPECT_EQ(matrix_chain_order({2, 2, 2, 2}).parenthesization, "(A1(A2A3))");
  EXPECT_EQ(matrix_chain_order({}).cost, 0);
  EXPECT_TRUE(matrix_chain_order({1}).parenthesization.empty());
  EXPECT_EQ(matrix_chain_trace({}).captured_entries, 2);
  EXPECT_NO_THROW(matrix_chain_trace({2, 3, 4}, 23));
  EXPECT_THROW(matrix_chain_trace({2, 3, 4}, 22), std::length_error);
  EXPECT_NO_THROW(matrix_chain_trace({2, 3, 4}, 0));
  d[0] = 1;
  EXPECT_EQ(t.dimensions[0], 30);
  for (auto bad : std::vector<std::vector<int>>{{0}, {-1}, {1, 0}, {2, -3, 4}}) {
    EXPECT_THROW(matrix_chain_order(bad), std::invalid_argument);
    EXPECT_THROW(matrix_chain_trace(bad, 1), std::invalid_argument);
    EXPECT_THROW(matrix_chain_left_cost(bad), std::invalid_argument);
    EXPECT_THROW(matrix_chain_right_cost(bad), std::invalid_argument);
  }
}
TEST(MatrixChainTrace, UnrepresentableSubproblemDoesNotInvalidateRepresentableFullOptimum) {
  const long long  m = INT_MAX;
  std::vector<int> d{1, INT_MAX, INT_MAX, INT_MAX, 1};
  auto             t = matrix_chain_trace(d);
  EXPECT_EQ(t.analysis.result.cost, 2 * m * m + m);
  EXPECT_EQ(t.analysis.costs[2][3], std::uint64_t{1} << 63);
  EXPECT_THROW(matrix_chain_order({INT_MAX, INT_MAX, INT_MAX}), std::overflow_error);
  EXPECT_THROW(matrix_chain_left_cost({INT_MAX, INT_MAX, INT_MAX}), std::overflow_error);
  EXPECT_THROW(matrix_chain_right_cost({INT_MAX, INT_MAX, INT_MAX}), std::overflow_error);
  EXPECT_EQ(matrix_chain_order({INT_MAX, INT_MAX, 1}).cost, m * m);
}
