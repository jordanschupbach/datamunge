#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/library_sort.hpp>
#include <numeric>
using namespace datamunge::algorithms;
namespace {
void verify(const std::vector<int>& input) {
  auto t      = library_sort_trace(input);
  auto sorted = input;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(t.analysis.values, sorted);
  EXPECT_EQ(library_sort(input), sorted);
  if (input.size() <= 1) {
    EXPECT_TRUE(t.shelves.empty());
    EXPECT_EQ(t.analysis.work.placements, 0);
    return;
  }
  EXPECT_EQ(t.shelves.size(), input.size() + 1);
  EXPECT_EQ(t.insertions.size(), input.size());
  std::size_t moved = 0;
  for (std::size_t k = 0; k < t.shelves.size(); ++k) {
    auto& s = t.shelves[k];
    EXPECT_EQ(s.inserted, k);
    std::vector<int> occupied;
    for (std::size_t j = 0; j < s.slots.size(); ++j)
      if (s.occupied[j])
        occupied.push_back(s.slots[j]);
    auto expected = std::vector<int>(input.begin(), input.begin() + k);
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(occupied, expected);
    if (k) {
      auto& e = t.insertions[k - 1];
      EXPECT_EQ(e.input_index, k - 1);
      EXPECT_EQ(e.value, input[k - 1]);
      EXPECT_FALSE(t.shelves[k - 1].occupied[e.gap]);
      EXPECT_TRUE(s.occupied[e.target]);
      EXPECT_EQ(s.slots[e.target], e.value);
      moved += e.moved;
      if (e.kind == LibrarySortPlacement::ShiftRight)
        EXPECT_EQ(e.moved, e.gap - e.target);
      else
        EXPECT_EQ(e.moved, 0);
    }
  }
  EXPECT_EQ(moved, t.analysis.work.shift_writes);
  EXPECT_EQ(t.analysis.work.output_slots, 2 * input.size() + 2);
  EXPECT_EQ(t.analysis.work.placements, input.size());
}
} // namespace
TEST(LibrarySortTrace, AllSmallPermutationsAndTernaryWords) {
  for (int n = 0; n <= 7; ++n) {
    std::vector<int> a(n);
    std::iota(a.begin(), a.end(), 0);
    do {
      verify(a);
    } while (std::next_permutation(a.begin(), a.end()));
  }
  for (int n = 0; n <= 6; ++n) {
    int count = 1;
    for (int i = 0; i < n; ++i)
      count *= 3;
    for (int k = 0; k < count; ++k) {
      std::vector<int> a(n);
      int              word = k;
      for (auto& x : a) {
        x = word % 3 - 1;
        word /= 3;
      }
      verify(a);
    }
  }
}
TEST(LibrarySortTrace, ExactDescendingQuadraticShiftAndOriginal) {
  for (int n : {4, 8, 16, 32, 64, 128}) {
    std::vector<int> a(n);
    std::iota(a.rbegin(), a.rend(), 0);
    auto r = library_sort_analyze(a);
    EXPECT_EQ(r.work.shift_writes, std::size_t(n - 1) * (n - 2) / 2);
    std::reverse(a.begin(), a.end());
    EXPECT_EQ(library_sort_analyze(a).work.shift_writes, 0);
  }
  std::vector<int> data;
  unsigned         s = 20240501;
  for (int i = 0; i < 40; ++i) {
    s = s * 1103515245u + 12345u;
    data.push_back((s >> 16) % 100);
  }
  EXPECT_EQ(data, library_sort_demo_input());
  verify(data);
  verify({INT_MAX, 0, INT_MIN, 0, INT_MAX});
}
TEST(LibrarySortTrace, CompleteQuotaTrivialCopyAndOwnership) {
  auto t = library_sort_trace({2, 1}, 43);
  EXPECT_EQ(t.captured_entries, 43);
  EXPECT_THROW(library_sort_trace({2, 1}, 42), std::length_error);
  EXPECT_EQ(library_sort_trace({2, 1}, 0).analysis.values, (std::vector<int>{1, 2}));
  EXPECT_EQ(library_sort_trace({}).captured_entries, 0);
  EXPECT_EQ(library_sort_trace({9}).captured_entries, 1);
  std::vector<int> a{2, 1};
  auto             u = library_sort_trace(a);
  a[0]               = 8;
  EXPECT_EQ(u.input[0], 2);
}
