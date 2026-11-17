#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/odd_even_sort.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(OddEvenSortTrace, ExhaustivePermutationOrderStabilityActualPairsAndPhases) {
  std::mt19937_64 rng(264273);
  for (std::size_t n = 0; n <= 30; ++n)
    for (int repeat = 0; repeat < 20; ++repeat) {
      std::vector<int> a(n);
      for (auto& x : a)
        x = int(rng() % 9) - 4;
      if (n > 1 && repeat == 0) {
        a[0] = INT_MAX;
        a[1] = INT_MIN;
      }
      auto t        = odd_even_sort_trace(a);
      auto expected = a;
      std::stable_sort(expected.begin(), expected.end());
      EXPECT_EQ(t.run.output, expected);
      EXPECT_EQ(odd_even_sort(a), expected);
      std::vector<std::size_t> ids(n);
      std::iota(ids.begin(), ids.end(), 0);
      std::stable_sort(ids.begin(), ids.end(), [&](auto i, auto j) { return a[i] < a[j]; });
      EXPECT_EQ(t.run.original_indices, ids);
      auto values = a;
      std::iota(ids.begin(), ids.end(), 0);
      std::size_t k = 0, swaps = 0;
      for (auto& p : t.phases) {
        EXPECT_EQ(p.ordinal, &p - t.phases.data());
        EXPECT_EQ(p.odd, p.ordinal % 2 == 0);
        std::size_t local = 0, comparisons = 0;
        for (std::size_t i = p.odd ? 1 : 0; i < n && n - i > 1; i += 2) {
          ASSERT_LT(k, t.comparisons.size());
          auto& c = t.comparisons[k++];
          EXPECT_EQ(c.phase, p.ordinal);
          EXPECT_EQ(c.left, i);
          EXPECT_EQ(c.left_value, values[i]);
          EXPECT_EQ(c.right_value, values[i + 1]);
          EXPECT_EQ(c.left_id, ids[i]);
          EXPECT_EQ(c.right_id, ids[i + 1]);
          EXPECT_EQ(c.swapped, values[i] > values[i + 1]);
          if (c.swapped) {
            std::swap(values[i], values[i + 1]);
            std::swap(ids[i], ids[i + 1]);
            ++local;
          }
          ++comparisons;
        }
        EXPECT_EQ(p.values, values);
        EXPECT_EQ(p.original_indices, ids);
        EXPECT_EQ(p.swaps, local);
        EXPECT_EQ(p.comparisons, comparisons);
        swaps += local;
      }
      EXPECT_EQ(k, t.comparisons.size());
      EXPECT_EQ(t.run.work.comparisons, k);
      EXPECT_EQ(t.run.work.swaps, swaps);
      std::size_t inversions = 0;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
          inversions += a[i] > a[j];
      EXPECT_EQ(swaps, inversions);
      EXPECT_EQ(t.run.work.phases, t.phases.size());
      EXPECT_EQ(t.run.work.phases, 2 * t.run.work.rounds);
      EXPECT_EQ(t.phases.back().swaps, 0);
      EXPECT_EQ(t.phases[t.phases.size() - 2].swaps, 0);
      EXPECT_EQ(t.captured_entries, 3 * n + k + t.phases.size() * (1 + 2 * n));
      EXPECT_NO_THROW(odd_even_sort_trace(a, t.captured_entries));
      EXPECT_THROW(odd_even_sort_trace(a, t.captured_entries - 1), std::length_error);
    }
}
TEST(OddEvenSortTrace, OriginalSevenValuesAndFinalCertificationPhases) {
  auto t = odd_even_sort_trace(odd_even_sort_demo_input());
  EXPECT_EQ(t.run.output, (std::vector<int>{1, 2, 3, 5, 7, 8, 9}));
  auto empty = odd_even_sort_trace({});
  EXPECT_EQ(empty.run.work.rounds, 1);
  EXPECT_EQ(empty.phases.size(), 2);
  EXPECT_EQ(empty.captured_entries, 2);
  auto descending = odd_even_sort_trace({4, 3, 2, 1});
  EXPECT_EQ(descending.run.work.swaps, 6);
  EXPECT_GT(descending.run.work.phases, 4);
  EXPECT_EQ(descending.run.output, odd_even_sort_run({4, 3, 2, 1}).output);
}
