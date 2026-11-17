#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <datamunge/algorithms/hirschberg.hpp>
#include <datamunge/algorithms/sequence_alignment.hpp>
#include <limits>
using namespace datamunge::algorithms;
namespace {
std::vector<std::string> words(int maxn) {
  std::vector<std::string> out{""};
  for (int n = 1; n <= maxn; ++n)
    for (int k = 0; k < (1 << n); ++k) {
      std::string s(n, 'A');
      for (int j = 0; j < n; ++j)
        if (k & (1 << j))
          s[j] = 'B';
      out.push_back(s);
    }
  return out;
}
void certificate(
    const Alignment&   r,
    const std::string& a,
    const std::string& b,
    AlignmentScoring   s) {
  ASSERT_EQ(r.a_aligned.size(), r.b_aligned.size());
  std::string aa, bb;
  int         score = 0;
  for (std::size_t j = 0; j < r.a_aligned.size(); ++j) {
    auto x = r.a_aligned[j], y = r.b_aligned[j];
    EXPECT_FALSE(x == '-' && y == '-');
    if (x != '-')
      aa.push_back(x);
    if (y != '-')
      bb.push_back(y);
    score += (x == '-' || y == '-') ? s.gap : x == y ? s.match : s.mismatch;
  }
  EXPECT_EQ(aa, a);
  EXPECT_EQ(bb, b);
  EXPECT_EQ(score, r.score);
}
} // namespace
TEST(Hirschberg, AllShortBinaryPairsAndScoringSigns) {
  const auto all = words(5);
  for (const auto& a : all)
    for (const auto& b : all)
      for (int match : {-2, 0, 2})
        for (int mismatch : {-2, 0, 2})
          for (int gap : {-2, 0, 2}) {
            const AlignmentScoring s{match, mismatch, gap};
            const auto             expected = needleman_wunsch(a, b, s);
            const auto             legacy   = hirschberg(a, b, s);
            const auto             run      = hirschberg_run(a, b, s);
            EXPECT_EQ(expected.score, legacy.score);
            EXPECT_EQ(expected.score, run.alignment.score);
            certificate(legacy, a, b, s);
            certificate(run.alignment, a, b, s);
            EXPECT_LE(run.work.peak_score_slots, 3 * (std::min(a.size(), b.size()) + 1));
          }
}
TEST(Hirschberg, ActualSplitProfilesAndExactOnlinePolicies) {
  const auto all = words(3);
  for (const auto& a : all)
    for (const auto& b : all) {
      const auto t = hirschberg_trace(a, b);
      certificate(t.run.alignment, a, b, {});
      const auto& x     = t.run.transposed ? b : a;
      const auto& y     = t.run.transposed ? a : b;
      std::size_t units = 0;
      for (const auto& r : t.splits) {
        const auto n = r.b_end - r.b_begin;
        ASSERT_EQ(r.forward.size(), n + 1);
        ASSERT_EQ(r.reverse.size(), n + 1);
        ASSERT_EQ(r.total.size(), n + 1);
        units += 1 + 3 * (n + 1);
        std::size_t best = 0;
        for (std::size_t j = 0; j <= n; ++j) {
          EXPECT_EQ(
              r.forward[j],
              needleman_wunsch(x.substr(r.a_begin, r.a_mid - r.a_begin), y.substr(r.b_begin, j))
                  .score);
          EXPECT_EQ(
              r.reverse[j],
              needleman_wunsch(x.substr(r.a_mid, r.a_end - r.a_mid), y.substr(r.b_begin + j, n - j))
                  .score);
          EXPECT_EQ(r.total[j], r.forward[j] + r.reverse[j]);
          if (r.total[j] > r.total[best])
            best = j;
        }
        EXPECT_EQ(r.b_mid, r.b_begin + best);
      }
      EXPECT_EQ(t.captured_entries, units);
      HirschbergOptions o;
      o.max_trace_entries = units;
      EXPECT_NO_THROW(hirschberg_trace(a, b, {}, o));
      if (units) {
        o.max_trace_entries = units - 1;
        EXPECT_THROW(hirschberg_trace(a, b, {}, o), std::length_error);
      }
      const auto work  = t.run.work.rolling_cells + t.run.work.base_pairs;
      o                = {};
      o.max_work_cells = work;
      EXPECT_NO_THROW(hirschberg_run(a, b, {}, o));
      if (work > 1) {
        o.max_work_cells = work - 1;
        EXPECT_THROW(hirschberg_run(a, b, {}, o), std::length_error);
      }
    }
  auto t                    = hirschberg_trace("AGTACGCA", "TATGC");
  auto copy                 = t;
  copy.splits[0].forward[0] = 999;
  EXPECT_NE(t.splits[0].forward[0], 999);
  EXPECT_EQ(t.run.work.peak_score_slots, 18);
}
TEST(Hirschberg, DomainsBytesOrientationAndUnbalancedSpace) {
  EXPECT_THROW(hirschberg("-", "A"), std::invalid_argument);
  EXPECT_THROW(
      hirschberg("A", "A", {std::numeric_limits<int>::max(), -1, -2}), std::overflow_error);
  EXPECT_THROW(hirschberg("A", "", {2, -1, std::numeric_limits<int>::min()}), std::overflow_error);
  EXPECT_EQ(hirschberg("", "", {std::numeric_limits<int>::min(), 0, 0}).score, 0);
  HirschbergOptions o;
  o.max_input_bytes = 1;
  EXPECT_THROW(hirschberg_run("A", "B", {}, o), std::length_error);
  const std::string a("A\0B", 3), b("A\0C", 3);
  auto              r = hirschberg_run(a, b);
  certificate(r.alignment, a, b, {});
  EXPECT_EQ(r.alignment.score, 3);
  auto small = hirschberg_run("AB", std::string(1024, 'A'));
  EXPECT_TRUE(small.transposed);
  EXPECT_LE(small.work.peak_score_slots, 9);
  o          = {false, 0, 0, 0};
  auto large = hirschberg_run("AB", std::string(1024, 'A'), {}, o);
  EXPECT_FALSE(large.transposed);
  EXPECT_EQ(large.work.peak_score_slots, 3075);
  EXPECT_EQ(small.alignment.score, large.alignment.score);
}
