#include <gtest/gtest.h>

#include <datamunge/algorithms/longest_common_substring.hpp>
#include <limits>
using namespace datamunge::algorithms;
namespace {
std::vector<int> word(int bits, int n) {
  std::vector<int> a(n);
  for (int i = 0; i < n; ++i)
    a[i] = (bits >> i) & 1;
  return a;
}
} // namespace
TEST(LCSubstringTrace, EveryShortBinaryPairAndActualSuffixCell) {
  for (int n = 0; n <= 5; ++n)
    for (int x = 0; x < (1 << n); ++x)
      for (int m = 0; m <= 5; ++m)
        for (int y = 0; y < (1 << m); ++y) {
          auto        a = word(x, n), b = word(y, m);
          auto        t      = lcsubstring_trace(a, b);
          std::size_t length = 0, as = 0, bs = 0, records = 0, matches = 0;
          ASSERT_EQ(t.rows.size(), n + 1);
          ASSERT_EQ(t.cells.size(), n * m);
          for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= m; ++j) {
              std::size_t suffix = 0;
              while (suffix < std::size_t(std::min(i, j)) && a[i - 1 - suffix] == b[j - 1 - suffix])
                ++suffix;
              auto& c = t.cells[(i - 1) * m + j - 1];
              EXPECT_EQ(c.i, i);
              EXPECT_EQ(c.j, j);
              EXPECT_EQ(c.value, suffix);
              EXPECT_EQ(t.rows[i][j], suffix);
              EXPECT_EQ(c.equal, suffix > 0);
              EXPECT_EQ(c.new_best, suffix > length);
              matches += c.equal;
              if (suffix > length) {
                length = suffix;
                as     = i - length;
                bs     = j - length;
                ++records;
              }
            }
          EXPECT_EQ(t.analysis.length, length);
          EXPECT_EQ(t.analysis.a_start, as);
          EXPECT_EQ(t.analysis.b_start, bs);
          EXPECT_EQ(
              t.analysis.sequence, (std::vector<int>(a.begin() + as, a.begin() + as + length)));
          EXPECT_EQ(t.analysis.sequence, longest_common_substring(a, b));
          EXPECT_EQ(t.analysis.work.cells, n * m);
          EXPECT_EQ(t.analysis.work.matches, matches);
          EXPECT_EQ(t.analysis.work.resets, n * m - matches);
          EXPECT_EQ(t.analysis.work.records, records);
          EXPECT_EQ(t.captured_entries, n + m + (n + 1) * (m + 1) + n * m);
        }
}
TEST(LCSubstringTrace, OriginalIdentityTiesAndContiguity) {
  auto t = lcsubstring_trace(lcsubstring_bytes("GeeksforGeeks"), lcsubstring_bytes("GeeksQuiz"));
  EXPECT_EQ(t.analysis.sequence, lcsubstring_bytes("Geeks"));
  EXPECT_EQ(t.analysis.a_start, 0);
  EXPECT_EQ(t.analysis.b_start, 0);
  EXPECT_EQ(t.analysis.length, 5);
  auto a = lcsubstring_analyze({1, 9, 2}, {1, 2});
  EXPECT_EQ(a.sequence, (std::vector<int>{1}));
  EXPECT_EQ(a.a_start, 0);
  EXPECT_EQ(a.b_start, 0);
  EXPECT_EQ(
      longest_common_substring(
          {std::numeric_limits<int>::min(), -1}, {-1, std::numeric_limits<int>::min()}),
      (std::vector<int>{std::numeric_limits<int>::min()}));
}
TEST(LCSubstringTrace, BytesQuotaEmptyAndOwnership) {
  auto bytes = lcsubstring_bytes(std::string("\0\xff", 2));
  EXPECT_EQ(bytes, (std::vector<int>{0, 255}));
  EXPECT_EQ(lcsubstring_trace({}, {}).captured_entries, 1);
  EXPECT_TRUE(lcsubstring_analyze({}, {}).sequence.empty());
  EXPECT_EQ(lcsubstring_trace({1, 2}, {2}, 11).captured_entries, 11);
  EXPECT_THROW(lcsubstring_trace({1, 2}, {2}, 10), std::length_error);
  std::vector<int> a{1, 2};
  auto             t = lcsubstring_trace(a, {2}, 0);
  a[1]               = 7;
  EXPECT_EQ(t.a[1], 2);
  EXPECT_EQ(t.analysis.sequence, (std::vector<int>{2}));
}
