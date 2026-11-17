#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/subsequences.hpp>
using namespace datamunge::algorithms;
namespace {
// Independent enumeration of all first-sequence subsequences, checking second by a forward scan.
std::size_t brute(const std::vector<int>& a, const std::vector<int>& b) {
  std::size_t best = 0;
  for (unsigned mask = 0; mask < (1u << a.size()); ++mask) {
    std::size_t j = 0, k = 0;
    bool        ok = true;
    for (std::size_t i = 0; i < a.size(); ++i)
      if (mask >> i & 1u) {
        while (j < b.size() && b[j] != a[i])
          ++j;
        if (j == b.size()) {
          ok = false;
          break;
        }
        ++j;
        ++k;
      }
    if (ok)
      best = std::max(best, k);
  }
  return best;
}
} // namespace
TEST(LCSTrace, ExhaustiveBinaryWordsAndActualRecurrence) {
  for (unsigned n = 0; n <= 5; ++n)
    for (unsigned m = 0; m <= 5; ++m)
      for (unsigned av = 0; av < (1u << n); ++av)
        for (unsigned bv = 0; bv < (1u << m); ++bv) {
          std::vector<int> a(n), b(m);
          for (unsigned i = 0; i < n; ++i)
            a[i] = av >> i & 1;
          for (unsigned i = 0; i < m; ++i)
            b[i] = bv >> i & 1;
          auto t = lcs_trace(a, b);
          EXPECT_EQ(t.analysis.sequence.size(), brute(a, b));
          EXPECT_EQ(t.analysis.sequence, longest_common_subsequence(a, b));
          EXPECT_EQ(t.analysis.work.cells, n * m);
          for (std::size_t i = 1; i <= n; ++i)
            for (std::size_t j = 1; j <= m; ++j)
              EXPECT_EQ(
                  t.table[i][j],
                  a[i - 1] == b[j - 1] ? t.table[i - 1][j - 1] + 1
                                       : std::max(t.table[i - 1][j], t.table[i][j - 1]));
          std::size_t i = n, j = m;
          for (auto& s : t.steps) {
            EXPECT_EQ(s.i, i);
            EXPECT_EQ(s.j, j);
            if (s.move == LCSMove::Match) {
              EXPECT_EQ(a[i - 1], b[j - 1]);
              --i;
              --j;
            } else if (s.move == LCSMove::Up) {
              EXPECT_GE(t.table[i - 1][j], t.table[i][j - 1]);
              --i;
            } else {
              EXPECT_LT(t.table[i - 1][j], t.table[i][j - 1]);
              --j;
            }
          }
          EXPECT_TRUE(i == 0 || j == 0);
          EXPECT_EQ(t.captured_entries, n + m + (n + 1) * (m + 1) + t.steps.size());
        }
}
TEST(LCSTrace, OriginalTieEmptyIntegersQuotaOwnership) {
  std::string      as = "ABCBDAB", bs = "BDCAB";
  std::vector<int> a(as.begin(), as.end()), b(bs.begin(), bs.end());
  auto             t = lcs_trace(a, b);
  EXPECT_EQ(std::string(t.analysis.sequence.begin(), t.analysis.sequence.end()), "BCAB");
  a[0] = 999;
  EXPECT_EQ(t.a[0], 'A');
  EXPECT_EQ(longest_common_subsequence({1, 2}, {2, 1}), (std::vector<int>{1}));
  EXPECT_EQ(
      longest_common_subsequence({INT_MIN, 0, INT_MAX}, {INT_MIN, INT_MAX}),
      (std::vector<int>{INT_MIN, INT_MAX}));
  EXPECT_TRUE(lcs_trace({}, {}).analysis.sequence.empty());
  EXPECT_EQ(lcs_trace({}, {}).captured_entries, 1);
  EXPECT_THROW(lcs_trace({1}, {1}, 7), std::length_error);
  EXPECT_EQ(lcs_trace({1}, {1}, 8).captured_entries, 7);
  EXPECT_EQ(lcs_trace({1}, {1}, 0).table[1][1], 1);
}
