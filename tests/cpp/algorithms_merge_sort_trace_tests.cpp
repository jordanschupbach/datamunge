#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <datamunge/algorithms/merge_sort.hpp>
#include <numeric>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
void verify(const MergeSortTrace& t) {
  auto        data = t.input, buffer = data;
  std::size_t comparison = 0, write = 0;
  for (auto& m : t.merges) {
    EXPECT_EQ(m.first_write, write);
    std::size_t i = m.lo, j = m.mid;
    for (std::size_t k = m.lo; k < m.hi; ++k) {
      bool right = false;
      if (i < m.mid && j < m.hi) {
        ASSERT_LT(comparison, t.comparisons.size());
        auto& c = t.comparisons[comparison++];
        right   = data[j].key < data[i].key;
        EXPECT_EQ(c.lo, m.lo);
        EXPECT_EQ(c.mid, m.mid);
        EXPECT_EQ(c.hi, m.hi);
        EXPECT_EQ(c.left, i);
        EXPECT_EQ(c.right, j);
        EXPECT_EQ(c.left_key, data[i].key);
        EXPECT_EQ(c.right_key, data[j].key);
        EXPECT_EQ(c.take_right, right);
      } else
        right = i == m.mid;
      auto  source = right ? j++ : i++;
      auto& w      = t.writes[write++];
      EXPECT_FALSE(w.copyback);
      EXPECT_EQ(w.source, source);
      EXPECT_EQ(w.destination, k);
      EXPECT_EQ(w.item.key, data[source].key);
      EXPECT_EQ(w.item.tag, data[source].tag);
      buffer[k] = w.item;
    }
    for (std::size_t k = m.lo; k < m.hi; ++k) {
      auto& w = t.writes[write++];
      EXPECT_TRUE(w.copyback);
      EXPECT_EQ(w.source, k);
      EXPECT_EQ(w.destination, k);
      EXPECT_EQ(w.item.key, buffer[k].key);
      EXPECT_EQ(w.item.tag, buffer[k].tag);
      data[k] = w.item;
    }
    EXPECT_EQ(m.past_write, write);
  }
  EXPECT_EQ(comparison, t.comparisons.size());
  EXPECT_EQ(write, t.writes.size());
  EXPECT_EQ(
      t.captured_entries,
      2 * t.input.size() + t.comparisons.size() + t.writes.size() + t.merges.size());
  ASSERT_EQ(data.size(), t.output.size());
  for (std::size_t k = 0; k < data.size(); ++k) {
    EXPECT_EQ(data[k].key, t.output[k].key);
    EXPECT_EQ(data[k].tag, t.output[k].tag);
  }
  EXPECT_EQ(t.work.comparisons, comparison);
  EXPECT_EQ(t.work.scratch_writes * 2, write);
  EXPECT_EQ(t.work.scratch_writes, t.work.copy_writes);
  EXPECT_EQ(t.work.merges, t.merges.size());
}
} // namespace
TEST(MergeSortTrace, AllSmallDuplicateKeyWordsActualHistoryAndStableOriginalIdentities) {
  for (std::size_t n = 0; n <= 7; ++n) {
    std::size_t count = 1;
    for (std::size_t k = 0; k < n; ++k)
      count *= 3;
    for (std::size_t encoded = 0; encoded < count; ++encoded) {
      auto                             bits = encoded;
      std::vector<std::int64_t>        data(n);
      std::vector<std::pair<int, int>> pairs;
      for (std::size_t k = 0; k < n; ++k) {
        data[k] = bits % 3;
        bits /= 3;
        pairs.push_back({static_cast<int>(data[k]), 100 - static_cast<int>(k)});
      }
      auto t = merge_sort_trace(data);
      verify(t);
      auto expected = data;
      std::sort(expected.begin(), expected.end());
      auto ordinary = data;
      merge_sort(ordinary);
      EXPECT_EQ(ordinary, expected);
      EXPECT_EQ(merge_sort_analyze(data).values, expected);
      for (std::size_t k = 0; k < n; ++k) {
        EXPECT_EQ(t.output[k].key, expected[k]);
        if (k && t.output[k - 1].key == t.output[k].key)
          EXPECT_LT(t.output[k - 1].tag, t.output[k].tag);
      }
      auto stable = pairs;
      std::stable_sort(
          stable.begin(), stable.end(), [](auto a, auto b) { return a.first < b.first; });
      auto pt = merge_sort_pairs_trace(pairs);
      verify(pt);
      merge_sort_pairs(pairs);
      EXPECT_EQ(pairs, stable);
      for (std::size_t k = 0; k < n; ++k) {
        EXPECT_EQ(pt.output[k].key, stable[k].first);
        EXPECT_EQ(pt.output[k].tag, stable[k].second);
      }
      EXPECT_LE(t.work.comparisons, merge_sort_comparison_bound(n));
      EXPECT_EQ(t.work.merges, n ? n - 1 : 0);
      EXPECT_EQ(t.work.max_depth, n > 1 ? std::bit_width(n - 1) + 1 : 1);
      if (n)
        EXPECT_EQ(t.work.scratch_writes, merge_sort_comparison_bound(n) + n - 1);
    }
  }
}
TEST(MergeSortTrace, WorstBoundAchievedAcrossEveryDistinctPermutationThroughSeven) {
  for (std::size_t n = 0; n <= 7; ++n) {
    std::vector<std::int64_t> data(n);
    std::iota(data.begin(), data.end(), 0);
    std::size_t maximum = 0;
    do {
      auto a  = merge_sort_analyze(data);
      maximum = std::max(maximum, a.work.comparisons);
    } while (std::next_permutation(data.begin(), data.end()));
    EXPECT_EQ(maximum, merge_sort_comparison_bound(n));
  }
  EXPECT_EQ(merge_sort_comparison_bound(8), 17);
  EXPECT_THROW(merge_sort_comparison_bound(SIZE_MAX), std::overflow_error);
}
TEST(MergeSortTrace, OriginalCountsFullWidthKeysEmptyQuotasAndOwnership) {
  auto data = merge_sort_demo_values();
  auto t    = merge_sort_trace(data);
  EXPECT_EQ(t.work.comparisons, 15);
  EXPECT_EQ(t.work.scratch_writes, 24);
  EXPECT_EQ(t.work.copy_writes, 24);
  EXPECT_EQ(t.work.merges, 7);
  verify(t);
  data[0] = 999;
  EXPECT_EQ(t.input[0].key, 4);
  EXPECT_NO_THROW(merge_sort_trace({2, 1}, 10));
  EXPECT_THROW(merge_sort_trace({2, 1}, 9), std::length_error);
  EXPECT_THROW(merge_sort_trace({2, 1}, 3), std::length_error);
  EXPECT_NO_THROW(merge_sort_trace({2, 1}, 0));
  EXPECT_TRUE(merge_sort_trace({}, 1).output.empty());
  EXPECT_NO_THROW(merge_sort_trace({7}, 2));
  EXPECT_THROW(merge_sort_trace({7}, 1), std::length_error);
  auto a = merge_sort_analyze({INT64_MAX, INT64_MIN, 0, INT64_MAX});
  EXPECT_EQ(a.values, (std::vector<std::int64_t>{INT64_MIN, 0, INT64_MAX, INT64_MAX}));
  auto pairs = merge_sort_pairs_trace({{INT_MAX, INT_MIN}, {INT_MIN, INT_MAX}});
  EXPECT_EQ(pairs.output[0].key, INT_MIN);
  EXPECT_EQ(pairs.output[0].tag, INT_MAX);
}
