#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/sorted_search.hpp>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
namespace {
using namespace datamunge::algorithms;
const std::vector<int> original{1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
void                   same(const SearchResult& a, const SearchResult& b) {
  EXPECT_EQ(a.found, b.found);
  EXPECT_EQ(a.index, b.index);
  EXPECT_EQ(a.probes, b.probes);
}
void check(const std::vector<int>& a, int key) {
  const auto t = fibonacci_search_trace(a, key, 0);
  const auto w = fibonacci_search_work(a, key);
  same(t.outcome.result, w.result);
  same(w.result, fibonacci_search(a, key));
  EXPECT_EQ(w.comparisons, t.outcome.comparisons);
  EXPECT_EQ(w.transitions, t.outcome.transitions);
  EXPECT_EQ(t.key, key);
  EXPECT_EQ(t.outcome.schedule.length, a.size());
  EXPECT_EQ(t.captured_entries, t.probes.size());
  EXPECT_EQ(t.probes.size(), w.result.probes);
  EXPECT_EQ(w.result.found, std::binary_search(a.begin(), a.end(), key));
  if (w.result.found) {
    ASSERT_GE(w.result.index, 0);
    ASSERT_LT(std::size_t(w.result.index), a.size());
    EXPECT_EQ(a[w.result.index], key);
  } else
    EXPECT_EQ(w.result.index, -1);
  std::size_t comparisons = 0, transitions = 0;
  for (std::size_t j = 0; j < t.probes.size(); ++j) {
    const auto& r = t.probes[j];
    ASSERT_LT(r.index, a.size());
    EXPECT_EQ(r.value, a[r.index]);
    EXPECT_EQ(r.small_before + r.middle_before, r.large_before);
    EXPECT_EQ(r.small_after + r.middle_after, r.large_after);
    comparisons += r.comparisons;
    const bool match = r.decision == 0;
    EXPECT_EQ(match, r.value == key);
    if (r.final_check) {
      EXPECT_EQ(j + 1, t.probes.size());
      EXPECT_EQ(r.comparisons, 1);
      EXPECT_EQ(r.decision, match ? 0 : 2);
      EXPECT_EQ(r.index, std::size_t(r.offset_before + 1));
      EXPECT_EQ(r.middle_before, 1);
      EXPECT_EQ(r.large_before, 1);
    } else {
      EXPECT_GT(r.large_before, 1);
      const auto logical = static_cast<__int128>(r.offset_before) + r.small_before;
      EXPECT_EQ(r.index, std::size_t(std::min<__int128>(logical, a.size() - 1)));
      EXPECT_EQ(r.comparisons, r.value < key ? 1 : 2);
      if (!match) {
        ++transitions;
        EXPECT_LT(r.large_after, r.large_before);
        EXPECT_EQ(r.decision, r.value < key ? 1 : -1);
        EXPECT_EQ(r.offset_after, r.value < key ? std::ptrdiff_t(r.index) : r.offset_before);
      }
    }
    if (r.final_check || match) {
      EXPECT_EQ(r.small_before, r.small_after);
      EXPECT_EQ(r.middle_before, r.middle_after);
      EXPECT_EQ(r.large_before, r.large_after);
      EXPECT_EQ(r.offset_before, r.offset_after);
    }
    if (j) {
      const auto& p = t.probes[j - 1];
      EXPECT_EQ(p.small_after, r.small_before);
      EXPECT_EQ(p.middle_after, r.middle_before);
      EXPECT_EQ(p.large_after, r.large_before);
      EXPECT_EQ(p.offset_after, r.offset_before);
    } else {
      EXPECT_EQ(r.small_before, t.outcome.schedule.small);
      EXPECT_EQ(r.middle_before, t.outcome.schedule.middle);
      EXPECT_EQ(r.large_before, t.outcome.schedule.large);
      EXPECT_EQ(r.offset_before, -1);
    }
  }
  EXPECT_EQ(comparisons, w.comparisons);
  EXPECT_EQ(transitions, w.transitions);
  EXPECT_LE(w.result.probes, t.outcome.schedule.growth_steps + 2);
}
TEST(FibonacciSearchTrace, OriginalActualPath) {
  const auto        t       = fibonacci_search_trace(original, 13);
  const std::size_t index[] = {4, 7, 5, 6}, small[] = {5, 3, 1, 1}, mid[] = {8, 5, 2, 1},
                    large[] = {13, 8, 3, 2};
  const int decision[]      = {1, -1, 1, 0};
  ASSERT_EQ(t.probes.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(t.probes[i].index, index[i]);
    EXPECT_EQ(t.probes[i].small_before, small[i]);
    EXPECT_EQ(t.probes[i].middle_before, mid[i]);
    EXPECT_EQ(t.probes[i].large_before, large[i]);
    EXPECT_EQ(t.probes[i].decision, decision[i]);
    EXPECT_FALSE(t.probes[i].final_check);
  }
  EXPECT_EQ(t.outcome.result.index, 6);
  EXPECT_EQ(t.outcome.comparisons, 6);
  EXPECT_EQ(t.outcome.transitions, 3);
  for (int key : {13, 1, 19, 8, 20})
    check(original, key);
}
TEST(FibonacciSearchTrace, ExhaustiveSmallDistinct) {
  for (std::size_t n = 0; n <= 128; ++n) {
    std::vector<int> a(n);
    for (std::size_t i = 0; i < n; ++i)
      a[i] = 2 * i;
    for (int key = -1; key <= int(2 * n) + 1; ++key)
      check(a, key);
  }
}
TEST(FibonacciSearchTrace, AllLengthsAndEdgeQueries) {
  for (std::size_t n = 0; n <= 4096; ++n) {
    std::vector<int> a(n);
    std::iota(a.begin(), a.end(), 0);
    for (int key : {INT_MIN, -1, 0, int(n / 3), int(n / 2), int(n) - 1, int(n), INT_MAX})
      check(a, key);
  }
}
TEST(FibonacciSearchTrace, AllSmallNondecreasingArrays) {
  std::size_t total = 0;
  for (std::size_t n = 0; n <= 8; ++n) {
    std::vector<int>                      a(n);
    std::function<void(std::size_t, int)> visit = [&](std::size_t i, int lo) {
      if (i == n) {
        ++total;
        for (int key = -2; key <= 2; ++key)
          check(a, key);
        return;
      }
      for (int x = lo; x <= 1; ++x) {
        a[i] = x;
        visit(i + 1, x);
      }
    };
    visit(0, -1);
  }
  EXPECT_EQ(total, 165);
  const auto r = fibonacci_search(std::vector<int>(8, 4), 4);
  EXPECT_EQ(r.index, 2);
}
TEST(FibonacciSearchTrace, SeededWideDuplicates) {
  std::mt19937 rng(109);
  for (int fixture = 0; fixture < 600; ++fixture) {
    std::vector<int> a(rng() % 1025);
    for (auto& x : a)
      x = int(rng() % 101) - 50;
    if (!a.empty()) {
      a.front() = INT_MIN;
      a.back()  = INT_MAX;
    }
    std::sort(a.begin(), a.end());
    const auto before = a;
    for (int j = 0; j < 20; ++j)
      check(a, j == 0 ? INT_MIN : j == 1 ? INT_MAX : int(rng() % 105) - 52);
    EXPECT_EQ(a, before);
  }
}
TEST(FibonacciSearchTrace, FullDomainSchedulesWithoutArrays) {
  using U = unsigned __int128;
  std::vector<U> f{0, 1, 1};
  while (f.back() <= std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
    f.push_back(f[f.size() - 1] + f[f.size() - 2]);
  std::vector<std::size_t> lengths{
      0, 1, 2, 10, std::size_t(std::numeric_limits<std::ptrdiff_t>::max())};
  for (auto x : f)
    for (int d : {-1, 0, 1})
      if (x + d > 0 && x + d <= std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
        lengths.push_back(std::size_t(x + d));
  std::mt19937_64 rng(109);
  for (int i = 0; i < 10000; ++i)
    lengths.push_back(rng() & std::size_t(std::numeric_limits<std::ptrdiff_t>::max()));
  for (auto n : lengths) {
    const auto s = fibonacci_search_schedule(n);
    EXPECT_EQ(s.length, n);
    if (!n) {
      EXPECT_EQ(s.small, 0);
      EXPECT_EQ(s.middle, 0);
      EXPECT_EQ(s.large, 0);
      EXPECT_EQ(s.growth_steps, 0);
      continue;
    }
    auto i = std::lower_bound(f.begin() + 2, f.end(), U(n)) - f.begin();
    EXPECT_EQ(s.small, f[i - 2]);
    EXPECT_EQ(s.middle, f[i - 1]);
    EXPECT_EQ(s.large, f[i]);
    EXPECT_EQ(s.growth_steps, i - 2);
    EXPECT_GE(s.large, n);
    EXPECT_EQ(s.small + s.middle, s.large);
  }
  EXPECT_THROW(
      fibonacci_search_schedule(std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) + 1),
      std::length_error);
  EXPECT_THROW(
      fibonacci_search_schedule(std::numeric_limits<std::size_t>::max()), std::length_error);
}
TEST(FibonacciSearchTrace, RepeatedClampsAndEqualityFallback) {
  const auto high = fibonacci_search_trace(original, 20);
  ASSERT_EQ(high.probes.size(), 5);
  const std::size_t indices[] = {4, 7, 9, 9, 9};
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(high.probes[i].index, indices[i]);
    EXPECT_FALSE(high.probes[i].final_check);
  }
  EXPECT_EQ(high.outcome.comparisons, 5);
  EXPECT_EQ(high.outcome.transitions, 5);
  const auto tail = fibonacci_search_trace(original, 8);
  ASSERT_EQ(tail.probes.size(), 5);
  EXPECT_EQ(tail.probes.front().index, 4);
  EXPECT_EQ(tail.probes.back().index, 4);
  EXPECT_TRUE(tail.probes.back().final_check);
  EXPECT_EQ(tail.probes.back().decision, 2);
  EXPECT_EQ(tail.outcome.comparisons, 6);
  EXPECT_EQ(tail.outcome.transitions, 4);
  const auto only = fibonacci_search_trace({7}, 7);
  ASSERT_EQ(only.probes.size(), 1);
  EXPECT_TRUE(only.probes[0].final_check);
  EXPECT_EQ(only.outcome.comparisons, 1);
}
TEST(FibonacciSearchTrace, CaptureLimitsAndOwnership) {
  EXPECT_THROW(fibonacci_search_trace(original, 13, 3), std::length_error);
  EXPECT_NO_THROW(fibonacci_search_trace(original, 13, 4));
  EXPECT_NO_THROW(fibonacci_search_trace(original, 13, 0));
  EXPECT_THROW(fibonacci_search_trace(original, 8, 4), std::length_error);
  EXPECT_THROW(fibonacci_search_trace(original, 20, 4), std::length_error);
  EXPECT_NO_THROW(fibonacci_search_trace({}, 13, 1));
  EXPECT_NO_THROW(fibonacci_search_work(original, 13));
  auto       a      = original;
  auto       t      = fibonacci_search_trace(a, 13);
  const auto copy   = t;
  a[4]              = -100;
  t.probes[0].value = -200;
  EXPECT_EQ(copy.probes[0].value, 9);
  EXPECT_EQ(copy.outcome.result.index, 6);
}
TEST(FibonacciSearchTrace, MillionElementOrdinaryAndTrace) {
  std::vector<int> a(1000001);
  std::iota(a.begin(), a.end(), 0);
  for (int key : {-1, 0, 500000, 1000000, 1000001})
    check(a, key);
  EXPECT_EQ(fibonacci_search(a, 1000000).index, 1000000);
}
TEST(FibonacciSearchTrace, EmptyAndCallableSignatures) {
  SearchResult (*legacy)(const std::vector<int>&, int) = &fibonacci_search;
  same(legacy(original, 13), fibonacci_search_work(original, 13).result);
  const auto t = fibonacci_search_trace({}, INT_MIN);
  EXPECT_FALSE(t.outcome.result.found);
  EXPECT_EQ(t.outcome.result.index, -1);
  EXPECT_EQ(t.outcome.result.probes, 0);
  EXPECT_TRUE(t.probes.empty());
  EXPECT_EQ(t.captured_entries, 0);
  EXPECT_EQ(t.outcome.comparisons, 0);
  EXPECT_EQ(t.outcome.transitions, 0);
  check({INT_MIN}, INT_MIN);
  check({INT_MAX}, INT_MAX);
  check({INT_MIN, INT_MAX}, 0);
}
} // namespace
