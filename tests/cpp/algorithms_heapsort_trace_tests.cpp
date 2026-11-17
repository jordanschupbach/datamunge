#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <datamunge/algorithms/heapsort.hpp>
#include <limits>
#include <numeric>
using namespace datamunge::algorithms;
namespace {
void certify(const std::vector<std::int64_t>& input) {
  auto t      = heapsort_trace(input);
  auto r      = heapsort_run(input);
  auto legacy = input;
  heapsort(legacy);
  auto expected = input;
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(t.run.data, expected);
  EXPECT_EQ(r.data, expected);
  EXPECT_EQ(legacy, expected);
  auto heap = input;
  build_max_heap(heap);
  ASSERT_GE(t.states.size(), 2);
  EXPECT_EQ(t.states[1].data, heap);
  EXPECT_TRUE(is_max_heap(heap));
  const auto n = input.size();
  EXPECT_EQ(t.states.size(), 2 + (n ? n - 1 : 0));
  auto        state = input;
  std::size_t seen  = 0;
  for (const auto& s : t.states) {
    while (seen < s.swaps_done) {
      const auto& e = t.swaps[seen++];
      ASSERT_LT(e.left, n);
      ASSERT_LT(e.right, n);
      EXPECT_EQ(state[e.left], e.left_before);
      EXPECT_EQ(state[e.right], e.right_before);
      EXPECT_LT(e.left, e.heap_size);
      EXPECT_LT(e.right, e.heap_size);
      if (e.root_parking) {
        EXPECT_EQ(e.left, 0);
        EXPECT_EQ(e.right, e.heap_size - 1);
        EXPECT_EQ(e.phase, HeapSortPhase::Extract);
      } else {
        EXPECT_TRUE(e.right == 2 * e.left + 1 || e.right == 2 * e.left + 2);
        EXPECT_GT(e.right_before, e.left_before);
      }
      std::swap(state[e.left], state[e.right]);
    }
    EXPECT_EQ(state, s.data);
    auto sorted = s.data;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted, expected);
    if (s.phase != HeapSortPhase::Input) {
      EXPECT_TRUE(std::is_heap(s.data.begin(), s.data.begin() + s.heap_size));
      EXPECT_TRUE(std::is_sorted(s.data.begin() + s.heap_size, s.data.end()));
      if (s.heap_size < n)
        for (std::size_t i = 0; i < s.heap_size; ++i)
          EXPECT_LE(s.data[i], s.data[s.heap_size]);
    }
  }
  EXPECT_EQ(seen, t.swaps.size());
  EXPECT_EQ(t.captured_entries, t.states.size() * (1 + n) + t.swaps.size());
  EXPECT_EQ(t.run.build.sift_calls, n / 2);
  EXPECT_EQ(t.run.extraction.sift_calls, n ? n - 1 : 0);
  EXPECT_EQ(t.run.extraction.root_swaps, n ? n - 1 : 0);
  for (const auto& w : {t.run.build, t.run.extraction}) {
    EXPECT_EQ(w.sift_visits, w.sift_calls + w.sift_swaps);
    EXPECT_LE(w.sift_swaps, w.key_comparisons);
    EXPECT_LE(w.key_comparisons, 2 * w.sift_visits);
  }
  EXPECT_LE(t.run.build.key_comparisons, 2 * (n - std::popcount(n)));
  EXPECT_EQ(
      t.swaps.size(),
      t.run.build.sift_swaps + t.run.extraction.sift_swaps + t.run.extraction.root_swaps);
  auto copy = t;
  copy.states[0].data.push_back(999);
  EXPECT_EQ(t.states[0].data, input);
  EXPECT_NO_THROW(heapsort_trace(input, {0, t.captured_entries}));
  if (t.captured_entries > 1)
    EXPECT_THROW(heapsort_trace(input, {0, t.captured_entries - 1}), std::length_error);
}
} // namespace
TEST(HeapSortTrace, ExhaustiveTernaryInputsThroughEightPositions) {
  std::size_t total = 1;
  for (int n = 0; n <= 8; ++n) {
    if (n)
      total *= 3;
    for (std::size_t mask = 0; mask < total; ++mask) {
      auto                      v = mask;
      std::vector<std::int64_t> input(n);
      for (auto& x : input) {
        x = std::int64_t(v % 3) - 1;
        v /= 3;
      }
      certify(input);
    }
  }
}
TEST(HeapSortTrace, CompleteAndOnlineCapsExtremesAndEqualSwaps) {
  certify(
      {std::numeric_limits<std::int64_t>::min(),
       0,
       std::numeric_limits<std::int64_t>::max(),
       -1,
       1});
  const std::vector<std::int64_t> input{2, 1};
  EXPECT_NO_THROW(heapsort_trace(input, {2, 10}));
  EXPECT_THROW(heapsort_trace(input, {1, 100}), std::length_error);
  EXPECT_THROW(heapsort_trace(input, {2, 8}), std::length_error);
  EXPECT_THROW(heapsort_trace(input, {2, 9}), std::length_error);
  EXPECT_EQ(input, (std::vector<std::int64_t>{2, 1}));
  EXPECT_NO_THROW(heapsort_run(input, {2, 1}));
  EXPECT_NO_THROW(heapsort_trace(input, {0, 0}));
  auto equal = heapsort_trace({2, 2});
  ASSERT_EQ(equal.swaps.size(), 1);
  EXPECT_TRUE(equal.swaps[0].root_parking);
  EXPECT_EQ(equal.swaps[0].left_before, equal.swaps[0].right_before);
  EXPECT_EQ(heapsort_trace({}).captured_entries, 2);
  EXPECT_EQ(heapsort_trace({7}).captured_entries, 4);
}
TEST(HeapSortTrace, LinearBuildBoundOnLargeFamilies) {
  for (std::size_t n : {1023, 1024, 1025, 4096})
    for (int type = 0; type < 3; ++type) {
      std::vector<std::int64_t> a(n);
      for (std::size_t i = 0; i < n; ++i)
        a[i] = type == 0 ? i : type == 1 ? n - i : 7;
      const auto r = heapsort_run(a);
      EXPECT_TRUE(std::is_sorted(r.data.begin(), r.data.end()));
      EXPECT_LE(r.build.key_comparisons, 2 * (n - std::popcount(n)));
      EXPECT_EQ(r.extraction.root_swaps, n - 1);
      if (type == 2)
        EXPECT_EQ(r.extraction.sift_swaps, 0);
    }
}
