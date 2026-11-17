#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/heap_permutations.hpp>
#include <limits>
#include <map>
#include <numeric>
#include <set>
using namespace datamunge::algorithms;
TEST(HeapTrace, CompleteDistinctEnumerationsAndActualEventCertificates) {
  for (int n = 0; n <= 8; ++n) {
    std::vector<int> input(n);
    std::iota(input.begin(), input.end(), 1);
    auto       t    = heap_permutations_trace(input);
    const auto size = heap_permutation_size(n);
    EXPECT_EQ(t.run.permutations, heap_permutations(input));
    EXPECT_EQ(t.run.permutations, heap_permutations_run(input).permutations);
    EXPECT_EQ(t.run.permutations.size(), size.permutations);
    EXPECT_EQ(t.run.work.copied_values, size.copied_values);
    EXPECT_EQ(t.run.work.recursive_calls, size.recursive_calls);
    EXPECT_EQ(t.run.work.emissions, size.permutations);
    EXPECT_EQ(t.run.work.swaps, size.permutations - 1);
    EXPECT_EQ(t.swaps.size(), size.permutations - 1);
    std::set<std::vector<int>> actual(t.run.permutations.begin(), t.run.permutations.end()), oracle;
    auto                       a = input;
    do {
      oracle.insert(a);
    } while (std::next_permutation(a.begin(), a.end()));
    EXPECT_EQ(actual, oracle);
    EXPECT_EQ(actual.size(), size.permutations);
    std::size_t even = 0, odd = 0;
    auto        state = input;
    for (std::size_t j = 0; j < t.swaps.size(); ++j) {
      const auto& e = t.swaps[j];
      EXPECT_EQ(e.output_index, j + 1);
      EXPECT_GE(e.prefix_size, 2);
      EXPECT_LT(e.block, e.prefix_size - 1);
      EXPECT_EQ(e.left, e.prefix_size % 2 == 0 ? e.block : 0);
      EXPECT_EQ(e.right, e.prefix_size - 1);
      EXPECT_EQ(e.left_value, state[e.left]);
      EXPECT_EQ(e.right_value, state[e.right]);
      EXPECT_NE(e.left, e.right);
      std::swap(state[e.left], state[e.right]);
      EXPECT_EQ(state, t.run.permutations[j + 1]);
      if (e.prefix_size % 2 == 0)
        ++even;
      else
        ++odd;
    }
    EXPECT_EQ(even, t.run.work.even_swaps);
    EXPECT_EQ(odd, t.run.work.odd_swaps);
    EXPECT_EQ(input, t.run.permutations[0]);
    auto copy = t;
    copy.run.permutations[0].push_back(999);
    EXPECT_EQ(t.run.permutations[0], input);
    if (!copy.swaps.empty()) {
      copy.swaps[0].left_value = 999;
      EXPECT_NE(copy.swaps[0].left_value, t.swaps[0].left_value);
    }
  }
}
TEST(HeapTrace, RepeatedPositionsRetainMultiplicityAndInvisibleSwaps) {
  for (int mask = 0; mask < 243; ++mask) {
    int              v = mask;
    std::vector<int> input(5);
    for (auto& x : input) {
      x = v % 3 - 1;
      v /= 3;
    }
    auto t = heap_permutations_trace(input);
    EXPECT_EQ(t.run.permutations.size(), 120);
    EXPECT_EQ(t.swaps.size(), 119);
    auto state = input;
    for (const auto& e : t.swaps) {
      EXPECT_EQ(e.left_value, state[e.left]);
      EXPECT_EQ(e.right_value, state[e.right]);
      std::swap(state[e.left], state[e.right]);
      EXPECT_EQ(state, t.run.permutations[e.output_index]);
    }
    auto sorted = input;
    std::sort(sorted.begin(), sorted.end());
    std::set<std::vector<int>> oracle;
    do {
      oracle.insert(sorted);
    } while (std::next_permutation(sorted.begin(), sorted.end()));
    std::set<std::vector<int>> actual(t.run.permutations.begin(), t.run.permutations.end());
    EXPECT_EQ(actual, oracle);
    std::map<std::vector<int>, std::size_t> frequencies;
    for (const auto& row : t.run.permutations)
      ++frequencies[row];
    for (const auto& [row, count] : frequencies)
      EXPECT_EQ(count, 120 / actual.size());
  }
  auto t = heap_permutations_trace({1, 1, 2});
  EXPECT_EQ(t.run.permutations[0], t.run.permutations[1]);
  EXPECT_EQ(t.swaps[0].left_value, t.swaps[0].right_value);
}
TEST(HeapTrace, ExactCapsEmptyAndCheckedCountOverflow) {
  EXPECT_NO_THROW(heap_permutations_trace({1, 2, 3}, {6, 18, 5}));
  EXPECT_THROW(heap_permutations_trace({1, 2, 3}, {5, 18, 5}), std::length_error);
  EXPECT_THROW(heap_permutations_trace({1, 2, 3}, {6, 17, 5}), std::length_error);
  EXPECT_THROW(heap_permutations_trace({1, 2, 3}, {6, 18, 4}), std::length_error);
  EXPECT_NO_THROW(heap_permutations_run({1, 2, 3}, {6, 18, 1}));
  EXPECT_NO_THROW(heap_permutations_trace({1, 2, 3}, {0, 0, 0}));
  EXPECT_EQ(
      heap_permutations_trace({}, {1, 1, 1}).run.permutations, (std::vector<std::vector<int>>{{}}));
  EXPECT_THROW(heap_permutation_size(std::numeric_limits<std::size_t>::max()), std::length_error);
  EXPECT_THROW(heap_permutations(std::vector<int>(100)), std::length_error);
  EXPECT_THROW(heap_permutations_run(std::vector<int>(9)), std::length_error);
}
