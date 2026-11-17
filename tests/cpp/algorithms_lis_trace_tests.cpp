#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/subsequences.hpp>
#include <numeric>
using namespace datamunge::algorithms;
TEST(LISTrace, ExhaustiveTernaryWordsIndependentDPEndpoints) {
  for (int n = 0; n <= 8; ++n) {
    int total = 1;
    for (int i = 0; i < n; ++i)
      total *= 3;
    for (int code = 0; code < total; ++code) {
      std::vector<int> a(n);
      int              v = code;
      for (auto& x : a) {
        x = v % 3 - 1;
        v /= 3;
      }
      auto             t = lis_trace(a);
      std::vector<int> length(n, 1);
      int              best = 0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j)
          if (a[j] < a[i])
            length[i] = std::max(length[i], length[j] + 1);
        best       = std::max(best, length[i]);
        auto& step = t.insertions[i];
        for (std::size_t k = 0; k < step.after.size(); ++k) {
          int minimum = INT_MAX;
          for (int j = 0; j <= i; ++j)
            if (length[j] >= int(k + 1))
              minimum = std::min(minimum, a[j]);
          EXPECT_EQ(a[step.after[k]], minimum);
        }
        if (step.length_class) {
          EXPECT_EQ(step.predecessor, step.before[step.length_class - 1]);
          EXPECT_LT(step.predecessor, step.input_index);
          EXPECT_LT(a[step.predecessor], a[step.input_index]);
        }
      }
      EXPECT_EQ(t.analysis.sequence.size(), best);
      EXPECT_EQ(t.analysis.indices.size(), best);
      EXPECT_EQ(t.analysis.sequence, longest_increasing_subsequence(a));
      for (std::size_t i = 0; i < t.analysis.indices.size(); ++i) {
        EXPECT_EQ(t.analysis.sequence[i], a[t.analysis.indices[i]]);
        if (i) {
          EXPECT_LT(t.analysis.indices[i - 1], t.analysis.indices[i]);
          EXPECT_LT(t.analysis.sequence[i - 1], t.analysis.sequence[i]);
        }
      }
      EXPECT_EQ(t.probes.size(), t.analysis.work.comparisons);
      for (auto& p : t.probes) {
        EXPECT_EQ(p.tail_value, a[p.tail_index]);
        EXPECT_EQ(p.less, p.tail_value < a[p.input_index]);
      }
      EXPECT_EQ(t.analysis.work.appends + t.analysis.work.replacements, a.size());
    }
  }
}
TEST(LISTrace, OriginalSourceIndicesAndFrontierIsNotWitness) {
  std::vector<int> a{0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
  auto             t = lis_trace(a);
  EXPECT_EQ(t.analysis.sequence, (std::vector<int>{0, 2, 6, 9, 11, 15}));
  EXPECT_EQ(t.analysis.indices, (std::vector<std::size_t>{0, 4, 6, 9, 13, 15}));
  auto tiny = lis_trace({2, 3, 1});
  EXPECT_EQ(tiny.final_tails, (std::vector<std::size_t>{2, 1}));
  EXPECT_EQ(tiny.analysis.indices, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(lis_analyze({1, 1}).indices, (std::vector<std::size_t>{1}));
  EXPECT_EQ(lis_analyze({INT_MIN, 0, INT_MAX}).sequence, (std::vector<int>{INT_MIN, 0, INT_MAX}));
}
TEST(LISTrace, ActualQuotaEmptyAndOwnership) {
  auto t = lis_trace({2, 3, 1});
  EXPECT_THROW(lis_trace({2, 3, 1}, t.captured_entries - 1), std::length_error);
  EXPECT_EQ(lis_trace({2, 3, 1}, t.captured_entries).captured_entries, t.captured_entries);
  EXPECT_EQ(lis_trace({2, 3, 1}, 0).analysis.sequence, t.analysis.sequence);
  EXPECT_EQ(lis_trace({}).captured_entries, 0);
  EXPECT_TRUE(lis_analyze({}).sequence.empty());
  std::vector<int> a{2, 3, 1};
  auto             own = lis_trace(a);
  a[0]                 = 9;
  EXPECT_EQ(own.input[0], 2);
}
