#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/merge.hpp>
#include <limits>
#include <random>
#include <set>
#include <tuple>
using namespace datamunge::algorithms;
namespace {
void verify(const std::vector<std::vector<int>>& lists, bool unique) {
  std::vector<int>                                       want;
  std::vector<std::tuple<int, std::size_t, std::size_t>> identities;
  std::size_t                                            R = 0;
  for (std::size_t l = 0; l < lists.size(); ++l) {
    R += !lists[l].empty();
    for (std::size_t e = 0; e < lists[l].size(); ++e) {
      want.push_back(lists[l][e]);
      identities.emplace_back(lists[l][e], l, e);
    }
  }
  const auto N = want.size();
  std::sort(want.begin(), want.end());
  std::sort(identities.begin(), identities.end());
  if (unique)
    want.erase(std::unique(want.begin(), want.end()), want.end());
  const auto t = k_way_merge_trace(lists, unique);
  EXPECT_EQ(t.values, want);
  EXPECT_EQ(t.values, unique ? union_merge(lists) : k_way_merge(lists));
  const auto a = k_way_merge_analyze(lists, unique);
  EXPECT_EQ(a.values, want);
  EXPECT_EQ(a.work.node_comparisons, t.work.node_comparisons);
  EXPECT_EQ(t.work.input_values, N);
  EXPECT_EQ(t.work.pushes, N);
  EXPECT_EQ(t.work.pops, N);
  EXPECT_EQ(t.work.emitted, want.size());
  EXPECT_EQ(t.work.discarded_duplicates, N - want.size());
  EXPECT_EQ(t.work.peak_heap, R);
  EXPECT_EQ(t.steps.size(), 2 * N);
  EXPECT_EQ(t.captured_entries, lists.size() + 3 * N);
  std::multiset<std::tuple<int, std::size_t, std::size_t>> heap;
  std::vector<int>                                         out;
  std::size_t                                              popped = 0;
  for (const auto& e : t.steps) {
    ASSERT_LT(e.list, lists.size());
    ASSERT_LT(e.index, lists[e.list].size());
    EXPECT_EQ(e.value, lists[e.list][e.index]);
    EXPECT_EQ(heap.size(), e.heap_before);
    const auto key = std::tuple{e.value, e.list, e.index};
    if (e.kind == MergeHeapKind::Push) {
      EXPECT_FALSE(e.emitted);
      heap.insert(key);
    } else {
      ASSERT_FALSE(heap.empty());
      EXPECT_EQ(key, *heap.begin());
      EXPECT_EQ(key, identities[popped++]);
      heap.erase(heap.begin());
      const bool emit = !unique || out.empty() || out.back() != e.value;
      EXPECT_EQ(e.emitted, emit);
      if (emit)
        out.push_back(e.value);
    }
    EXPECT_EQ(heap.size(), e.heap_after);
  }
  EXPECT_TRUE(heap.empty());
  EXPECT_EQ(out, want);
}
} // namespace
TEST(MergeTrace, SortedMultisetAndActualHeapIdentityOracle) {
  std::mt19937 rng(189);
  for (int r = 0; r < 1000; ++r) {
    std::vector<std::vector<int>> lists(r % 9);
    for (auto& row : lists) {
      for (int i = 0, n = rng() % 20; i < n; ++i)
        row.push_back(int(rng() % 15) - 7);
      std::sort(row.begin(), row.end());
    }
    verify(lists, false);
    verify(lists, true);
  }
}
TEST(MergeTrace, OriginalRunsDuplicatesEmptyAndIntegerExtremes) {
  verify({{1, 4, 7, 10}, {2, 3, 11}, {5, 6, 8, 9, 12}}, false);
  verify({{1, 1, 4}, {1, 2}, {}}, false);
  verify({{1, 1, 4}, {1, 2}, {}}, true);
  verify({}, false);
  verify({{}, {}}, true);
  verify(
      {{std::numeric_limits<int>::min(), 0, std::numeric_limits<int>::max()},
       {std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}},
      false);
  EXPECT_EQ(simple_merge({1, 4, 7, 10}, {2, 3, 11}), (std::vector<int>{1, 2, 3, 4, 7, 10, 11}));
}
TEST(MergeTrace, CompleteSortedValidationCapsAndOwnedData) {
  EXPECT_THROW((void)k_way_merge({{1, 0}}), std::invalid_argument);
  EXPECT_THROW((void)union_merge({{0}, {2, 1}}), std::invalid_argument);
  EXPECT_THROW((void)simple_merge({2, 1}, {}), std::invalid_argument);
  EXPECT_THROW((void)simple_merge({}, {2, 1}), std::invalid_argument);
  EXPECT_THROW((void)k_way_merge_trace({{2, 1}}, false, 1), std::invalid_argument);
  const auto t = k_way_merge_trace({{1, 1}, {1, 2}}, true);
  EXPECT_NO_THROW((void)k_way_merge_trace(t.lists, true, t.captured_entries));
  EXPECT_THROW((void)k_way_merge_trace(t.lists, true, t.captured_entries - 1), std::length_error);
  EXPECT_NO_THROW((void)k_way_merge_trace(t.lists, true, 0));
  auto c = t;
  c.lists[0].clear();
  c.values.clear();
  c.steps[0].value = 99;
  EXPECT_EQ(t.lists[0].size(), 2U);
  EXPECT_FALSE(t.values.empty());
  EXPECT_EQ(t.steps[0].value, 1);
}
TEST(MergeTrace, FixedOriginalProfilePopulationAndHeapOccupancy) {
  for (std::size_t k = 2; k <= 64; k += 2) {
    std::vector<std::vector<int>> lists(k);
    for (int i = 0; i < 3000; ++i)
      lists[i % k].push_back(i);
    const auto a = k_way_merge_analyze(lists);
    ASSERT_EQ(a.values.size(), 3000U);
    for (int i = 0; i < 3000; ++i)
      EXPECT_EQ(a.values[i], i);
    EXPECT_EQ(a.work.pushes, 3000U);
    EXPECT_EQ(a.work.pops, 3000U);
    EXPECT_EQ(a.work.peak_heap, k);
    EXPECT_GT(a.work.node_comparisons, 0U);
  }
}
